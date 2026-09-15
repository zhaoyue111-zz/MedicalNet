"""Train a 2D ImageNet-ResNet18 with independent fake/Composition heads."""

import argparse
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18
from sklearn.metrics import roc_auc_score

from datasets.luna25 import CLASS_NAMES, patient_split
from datasets.luna25_2d import (DualHeadBatchSampler, Luna25Dataset2D,
                                limit_normal_patients_one_series,
                                scan_luna25_2d)


NUM_SLICES = 11
IMAGE_SIZE = 224


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='luna25_organized')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='fixed at 4: 2 normal + 1 fake + 1 Composition')
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--normal_size', type=int, default=0,
                        help='normal patients retained in train; 0 keeps all')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--save_path', default='train/models/luna25_2d_dualhead_best.pth')
    parser.add_argument('--latest_path', default='',
                        help='latest checkpoint; default: latest.pth beside save_path')
    parser.add_argument('--resume_path', default='',
                        help='resume model and optimizer from a checkpoint')
    parser.add_argument('--no_pretrained', action='store_true',
                        help='do not initialize the ResNet18 with ImageNet weights')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--smoke_sampler', action='store_true',
                        help='inspect split and exact batch counts without training')
    return parser.parse_args()


def counts(records):
    values = Counter(item[1] for item in records)
    return {CLASS_NAMES[label]: values[label] for label in range(3)}


class DualHeadResNet18(nn.Module):
    """Shared 2D ResNet18 backbone plus two independent CT-level heads."""

    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = resnet18(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.fake_head = nn.Linear(feature_dim, 1)
        self.comp_head = nn.Linear(feature_dim, 1)
        self.feature_dim = feature_dim

    def forward(self, x):
        if x.ndim != 5 or x.shape[1] != NUM_SLICES or x.shape[2] != 3:
            raise ValueError('expected input [B, 11, 3, 224, 224], got {}'.format(tuple(x.shape)))
        batch_size, num_slices = x.shape[:2]
        features = self.backbone(x.reshape(batch_size * num_slices, 3,
                                           x.shape[-2], x.shape[-1]))
        features = features.reshape(batch_size, num_slices, self.feature_dim)
        ct_features = features.mean(dim=1)
        fake_logits = self.fake_head(ct_features).squeeze(1)
        comp_logits = self.comp_head(ct_features).squeeze(1)
        return fake_logits, comp_logits


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def masked_losses(fake_logits, comp_logits, labels, criterion):
    fake_mask = (labels == 0) | (labels == 1)
    comp_mask = (labels == 0) | (labels == 2)
    fake_targets = (labels == 1).float()
    comp_targets = (labels == 2).float()
    # Composition is never a fake negative, and fake is never a Composition
    # negative: each head only sees its explicitly defined binary task.
    # A very small validation split can contain no example for one task.  Keep
    # that task's loss differentiable and zero instead of passing an empty
    # tensor to BCEWithLogitsLoss (which would produce NaN).
    loss_fake = (criterion(fake_logits[fake_mask], fake_targets[fake_mask])
                 if fake_mask.any() else fake_logits.sum() * 0.0)
    loss_comp = (criterion(comp_logits[comp_mask], comp_targets[comp_mask])
                 if comp_mask.any() else comp_logits.sum() * 0.0)
    return loss_fake, loss_comp, fake_mask, comp_mask, fake_targets, comp_targets


def binary_metrics(targets, probabilities):
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = probabilities >= 0.5
    positives = targets == 1
    negatives = targets == 0
    tp = int(np.logical_and(predictions, positives).sum())
    fp = int(np.logical_and(predictions, negatives).sum())
    tn = int(np.logical_and(~predictions, negatives).sum())
    fn = int(np.logical_and(~predictions, positives).sum())
    total = tp + fp + tn + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    auc = float('nan')
    if positives.any() and negatives.any():
        auc = float(roc_auc_score(targets, probabilities))
    return {
        'accuracy': (tp + tn) / total if total else 0.0,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'auc': auc,
        'tp': tp,
        'fp': fp,
        'tn': tn,
        'fn': fn,
    }


def run_epoch(loader, model, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_total_sum = 0.0
    loss_fake_sum = 0.0
    loss_comp_sum = 0.0
    fake_count = 0
    comp_count = 0
    num_batches = 0
    task_targets = {'fake': [], 'comp': []}
    task_probs = {'fake': [], 'comp': []}
    label_counts = Counter()

    for images, labels, _patient_ids, _series_ids in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.set_grad_enabled(training):
            fake_logits, comp_logits = model(images)
            loss_fake, loss_comp, fake_mask, comp_mask, fake_targets, comp_targets = masked_losses(
                fake_logits, comp_logits, labels, criterion)
            loss_total = loss_fake + loss_comp
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss_total.backward()
                optimizer.step()

        num_batches += 1
        loss_total_sum += float(loss_total.item())
        loss_fake_sum += float(loss_fake.item()) * int(fake_mask.sum().item())
        loss_comp_sum += float(loss_comp.item()) * int(comp_mask.sum().item())
        fake_count += int(fake_mask.sum().item())
        comp_count += int(comp_mask.sum().item())
        label_counts.update(labels.detach().cpu().tolist())
        task_targets['fake'].extend(fake_targets[fake_mask].detach().cpu().numpy())
        task_probs['fake'].extend(torch.sigmoid(fake_logits[fake_mask]).detach().cpu().numpy())
        task_targets['comp'].extend(comp_targets[comp_mask].detach().cpu().numpy())
        task_probs['comp'].extend(torch.sigmoid(comp_logits[comp_mask]).detach().cpu().numpy())

    result = {
        'loss_total': loss_total_sum / max(1, num_batches),
        'loss_fake': loss_fake_sum / max(1, fake_count),
        'loss_comp': loss_comp_sum / max(1, comp_count),
        'fake': binary_metrics(task_targets['fake'], task_probs['fake']),
        'comp': binary_metrics(task_targets['comp'], task_probs['comp']),
        'counts': {CLASS_NAMES[label]: label_counts[label] for label in range(3)},
        'batches': num_batches,
    }
    return result


def metric_text(metric, complete=False):
    names = ('accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc')
    values = []
    for name in names if complete else ('accuracy', 'precision', 'recall', 'f1', 'auc'):
        value = metric[name]
        rendered = 'nan' if np.isnan(value) else '{:.4f}'.format(value)
        values.append('{}={}'.format(name, rendered))
    if complete:
        values.append('TP={} FP={} TN={} FN={}'.format(
            metric['tp'], metric['fp'], metric['tn'], metric['fn']))
    return ' '.join(values)


def print_epoch_result(epoch, name, result, complete=False):
    print('epoch {} {} loss_total={:.4f} loss_fake={:.4f} loss_comp={:.4f}'.format(
        epoch, name, result['loss_total'], result['loss_fake'], result['loss_comp']))
    print('{} fake: {}'.format(name, metric_text(result['fake'], complete)))
    print('{} comp: {}'.format(name, metric_text(result['comp'], complete)))
    if name == 'train':
        print('{} sampled: normal={} fake={} Composition={} batches={}'.format(
            name, result['counts']['normal'], result['counts']['fake'],
            result['counts']['Composition'], result['batches']))


def checkpoint_state(args, epoch, model, optimizer, best_score):
    return {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_score': best_score,
        'model_config': {
            'backbone': 'torchvision.resnet18',
            'weights': None if args.no_pretrained else 'IMAGENET1K_V1',
            'feature_dim': model.feature_dim,
            'num_slices': NUM_SLICES,
            'image_size': IMAGE_SIZE,
            'pooling': 'mean',
            'fake_head': 'Linear(512, 1)',
            'comp_head': 'Linear(512, 1)',
        },
    }


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def main():
    args = options()
    if args.batch_size != 4:
        raise ValueError('--batch_size must be 4 for the required 2N+1F+1C batches')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    records = scan_luna25_2d(args.data_root)
    train_records, val_records = patient_split(records, args.val_ratio, args.seed)
    train_before_limit = train_records
    train_records = limit_normal_patients_one_series(
        train_records, args.normal_size, args.seed, epoch=0)
    print('all:', counts(records))
    print('train before normal limit:', counts(train_before_limit))
    print('train:', counts(train_records))
    print('val:', counts(val_records))

    batch_sampler = DualHeadBatchSampler(
        [item[1] for item in train_records], args.steps_per_epoch, args.seed, args.batch_size)
    if args.smoke_sampler:
        first_batches = []
        sampled = Counter()
        for batch in batch_sampler:
            labels = [train_records[index][1] for index in batch]
            first_batches.append(labels)
            sampled.update(labels)
        print('first batches:', first_batches[:6])
        print('sampled:', dict(sampled))
        return

    val_set = Luna25Dataset2D(val_records, train=False)
    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        'num_workers': args.num_workers,
        'pin_memory': not args.no_cuda,
        'worker_init_fn': seed_worker,
        'generator': generator,
    }
    # Validation is natural-distribution, deterministic and CT-level.
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = DualHeadResNet18(pretrained=not args.no_pretrained and not args.resume_path).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay)

    best_score = -1.0
    start_epoch = 0
    if args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint.get('epoch', 0))
        best_score = float(checkpoint.get('best_score', -1.0))
        print('Resumed {} at epoch {}, best_score={:.4f}'.format(
            args.resume_path, start_epoch, best_score))

    save_path = Path(args.save_path)
    latest_path = Path(args.latest_path) if args.latest_path else save_path.with_name('latest.pth')
    for epoch in range(start_epoch, args.epochs):
        # Keep the selected patient set fixed, but randomly choose one of that
        # patient's series for this epoch.
        epoch_train_records = limit_normal_patients_one_series(
            train_before_limit, args.normal_size, args.seed, epoch=epoch)
        train_set = Luna25Dataset2D(epoch_train_records, train=True)
        batch_sampler = DualHeadBatchSampler(
            train_set.labels, args.steps_per_epoch, args.seed, args.batch_size)
        batch_sampler.set_epoch(epoch)
        train_loader = DataLoader(train_set, batch_sampler=batch_sampler, **loader_kwargs)
        train_result = run_epoch(train_loader, model, criterion, device, optimizer)
        print_epoch_result(epoch + 1, 'train', train_result)
        val_result = run_epoch(val_loader, model, criterion, device)
        print_epoch_result(epoch + 1, 'val', val_result, complete=True)

        score = (val_result['fake']['f1'] + val_result['comp']['f1']) / 2.0
        if score > best_score:
            best_score = score
            save_checkpoint(checkpoint_state(args, epoch + 1, model, optimizer, best_score), save_path)
            print('Saved best checkpoint to {} (score={:.4f})'.format(save_path, best_score))
        save_checkpoint(checkpoint_state(args, epoch + 1, model, optimizer, best_score), latest_path)
        print('Saved latest checkpoint to {}'.format(latest_path))


if __name__ == '__main__':
    main()
