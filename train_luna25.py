"""Minimal three-class training entry point for luna25_organized."""

import argparse
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datasets.luna25 import (BalancedBatchSampler, CLASS_NAMES, Luna25Dataset,
                             limit_normal_patients, patient_split, scan_luna25)
from models import resnet


def options():
    p = argparse.ArgumentParser()
    p.add_argument('--data_root', default='luna25_organized')
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--class_ratio', nargs=3, type=int, default=[2, 1, 1])
    p.add_argument('--normal_size', type=int, default=0,
                   help='number of normal patients retained in train; 0 keeps all')
    p.add_argument('--input_D', type=int, default=56)
    p.add_argument('--input_H', type=int, default=448)
    p.add_argument('--input_W', type=int, default=448)
    p.add_argument('--model_depth', type=int, default=18, choices=[10, 18, 34, 50, 101, 152, 200])
    p.add_argument('--resnet_shortcut', default='A')
    p.add_argument('--pretrain_path', default='pretrained/resnet_18.pth')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--learning_rate', type=float, default=1e-3)
    p.add_argument('--val_ratio', type=float, default=0.2)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--save_path', default='trails/models/luna25_resnet18_best.pth')
    p.add_argument('--no_cuda', action='store_true')
    p.add_argument('--smoke_sampler', action='store_true', help='only inspect counts and sampler batches')
    return p.parse_args()


def counts(records):
    c = Counter(item[1] for item in records)
    return {CLASS_NAMES[i]: c[i] for i in range(3)}


def make_model(args):
    factory = getattr(resnet, 'resnet{}'.format(args.model_depth))
    model = factory(sample_input_W=args.input_W, sample_input_H=args.input_H,
                    sample_input_D=args.input_D, shortcut_type=args.resnet_shortcut,
                    no_cuda=args.no_cuda, num_seg_classes=2,
                    task='classification', num_classes=3)
    if args.pretrain_path:
        checkpoint = torch.load(args.pretrain_path, map_location='cpu', weights_only=False)
        source = checkpoint.get('state_dict', checkpoint)
        target = model.state_dict()
        compatible = {}
        for key, value in source.items():
            key = key.removeprefix('module.')
            if key in target and target[key].shape == value.shape and not key.startswith('conv_seg.'):
                compatible[key] = value
        model.load_state_dict(compatible, strict=False)
        print('Loaded {} pretrained backbone tensors'.format(len(compatible)))
    return model


def metrics(confusion):
    total = int(confusion.sum())
    accuracy = confusion.diag().sum().item() / total if total else 0.0
    recalls = []
    for i in range(3):
        denom = confusion[i].sum().item()
        recalls.append(confusion[i, i].item() / denom if denom else 0.0)
    return accuracy, recalls


def run_epoch(loader, model, criterion, device, optimizer=None):
    model.train(optimizer is not None)
    confusion = torch.zeros(3, 3, dtype=torch.long)
    loss_sum = 0.0
    for volumes, labels in loader:
        volumes, labels = volumes.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(volumes)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item() * labels.numel()
        for truth, pred in zip(labels.cpu(), logits.argmax(1).cpu()):
            confusion[truth.long(), pred.long()] += 1
    accuracy, recalls = metrics(confusion)
    return loss_sum / max(1, int(confusion.sum())), accuracy, recalls, confusion


def report(name, result):
    loss, accuracy, recalls, confusion = result
    print('{} loss={:.4f} overall_accuracy={:.4f}'.format(name, loss, accuracy))
    print('{} recall: {}'.format(name, ', '.join('{}={:.4f}'.format(n, r) for n, r in zip(CLASS_NAMES, recalls))))
    print('{} confusion_matrix (rows=true, cols=pred):\n{}'.format(name, confusion.numpy()))


def main():
    args = options()
    torch.manual_seed(args.seed)
    records = scan_luna25(args.data_root)
    train_records, val_records = patient_split(records, args.val_ratio, args.seed)
    train_before_limit = train_records
    train_records = limit_normal_patients(train_records, args.normal_size, args.seed)
    print('all:', counts(records))
    print('train before normal limit:', counts(train_before_limit))
    print('train:', counts(train_records)); print('val:', counts(val_records))
    if args.smoke_sampler:
        for batch_size in (1, 2, 4):
            sampler = BalancedBatchSampler([x[1] for x in train_records], batch_size, args.class_ratio, args.seed)
            labels = [x[1] for x in train_records]
            batches = [[labels[i] for i in batch] for batch in sampler]
            flat = [y for batch in batches for y in batch]
            print('batch_size={}: labels={} totals={}'.format(batch_size, batches[:6], dict(Counter(flat))))
        return
    train_set = Luna25Dataset(train_records, (args.input_D, args.input_H, args.input_W))
    val_set = Luna25Dataset(val_records, (args.input_D, args.input_H, args.input_W))
    batch_sampler = BalancedBatchSampler(train_set.labels, args.batch_size, args.class_ratio, args.seed)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler, num_workers=args.num_workers,
                              pin_memory=not args.no_cuda)
    # Validation preserves the natural distribution and order.
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=not args.no_cuda)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model = make_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    best = -1.0
    for epoch in range(args.epochs):
        batch_sampler.set_epoch(epoch)
        report('epoch {} train'.format(epoch + 1), run_epoch(train_loader, model, criterion, device, optimizer))
        val_result = run_epoch(val_loader, model, criterion, device)
        report('epoch {} val'.format(epoch + 1), val_result)
        if val_result[1] > best:
            best = val_result[1]
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_config': {
                    'model_depth': args.model_depth,
                    'resnet_shortcut': args.resnet_shortcut,
                    'input_D': args.input_D,
                    'input_H': args.input_H,
                    'input_W': args.input_W,
                    'num_classes': 3,
                },
            }, args.save_path)


if __name__ == '__main__':
    main()
