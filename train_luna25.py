"""Minimal three-class training entry point for luna25_organized."""

import argparse
import os
from collections import Counter
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
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
    p.add_argument('--save_path', default='train/models/best.pth')
    p.add_argument('--latest_path', default='',
                   help='latest checkpoint path; default: lasted.pth beside save_path')
    p.add_argument('--resume_path', default='',
                   help='resume model and optimizer from a training checkpoint')
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
    if args.pretrain_path and not args.resume_path:
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


def checkpoint_state(args, epoch, model, optimizer, best_balanced_accuracy):
    model = model.module if isinstance(model, DDP) else model
    return {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_balanced_accuracy': best_balanced_accuracy,
        'model_config': {
            'model_depth': args.model_depth,
            'resnet_shortcut': args.resnet_shortcut,
            'input_D': args.input_D,
            'input_H': args.input_H,
            'input_W': args.input_W,
            'num_classes': 3,
        },
    }


def save_checkpoint(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def metrics(confusion):
    total = int(confusion.sum())
    accuracy = confusion.diag().sum().item() / total if total else 0.0
    recalls = []
    for i in range(3):
        denom = confusion[i].sum().item()
        recalls.append(confusion[i, i].item() / denom if denom else 0.0)
    balanced_accuracy = sum(recalls) / len(recalls)
    return accuracy, recalls, balanced_accuracy


def run_epoch(loader, model, criterion, device, optimizer=None, distributed=False):
    model.train(optimizer is not None)
    confusion = torch.zeros(3, 3, dtype=torch.long, device=device)
    loss_sum = 0.0
    for volumes, labels in loader:
        volumes, labels = volumes.to(device), labels.to(device)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(volumes)
            loss = criterion(logits, labels)
            if optimizer is not None:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
        loss_sum += loss.item() * labels.numel()
        encoded = labels * 3 + logits.argmax(1)
        confusion += torch.bincount(encoded, minlength=9).reshape(3, 3)
    stats = torch.tensor([loss_sum, confusion.sum().item()], dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.all_reduce(confusion, op=dist.ReduceOp.SUM)
    confusion = confusion.cpu()
    accuracy, recalls, balanced_accuracy = metrics(confusion)
    return (stats[0].item() / max(1, int(stats[1].item())), accuracy, recalls,
            balanced_accuracy, confusion)


def report(name, result):
    loss, accuracy, recalls, balanced_accuracy, confusion = result
    print('{} loss={:.4f} overall_accuracy={:.4f} balanced_accuracy={:.4f}'.format(
        name, loss, accuracy, balanced_accuracy))
    print('{} recall: {}'.format(name, ', '.join('{}={:.4f}'.format(n, r) for n, r in zip(CLASS_NAMES, recalls))))
    print('{} confusion_matrix (rows=true, cols=pred):\n{}'.format(name, confusion.numpy()))


def main():
    args = options()
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    distributed = world_size > 1
    if distributed:
        backend = 'gloo' if args.no_cuda else 'nccl'
        dist.init_process_group(backend=backend, init_method='env://')
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if args.resume_path:
        resume_metadata = torch.load(args.resume_path, map_location='cpu', weights_only=False)
        resume_config = resume_metadata.get('model_config', {})
        for name in ('model_depth', 'resnet_shortcut', 'input_D', 'input_H', 'input_W'):
            if name in resume_config:
                setattr(args, name, resume_config[name])
    torch.manual_seed(args.seed)
    records = scan_luna25(args.data_root)
    train_records, val_records = patient_split(records, args.val_ratio, args.seed)
    train_before_limit = train_records
    train_records = limit_normal_patients(train_records, args.normal_size, args.seed)
    if rank == 0:
        print('DDP world_size={}, batch_size_per_gpu={}, global_batch_size={}'.format(
            world_size, args.batch_size, args.batch_size * world_size))
        print('all:', counts(records))
        print('train before normal limit:', counts(train_before_limit))
        print('train:', counts(train_records)); print('val:', counts(val_records))
    if args.smoke_sampler:
        if rank == 0:
            for batch_size in (1, 2, 4):
                sampler = BalancedBatchSampler([x[1] for x in train_records], batch_size, args.class_ratio, args.seed)
                labels = [x[1] for x in train_records]
                batches = [[labels[i] for i in batch] for batch in sampler]
                flat = [y for batch in batches for y in batch]
                print('batch_size={}: labels={} totals={}'.format(batch_size, batches[:6], dict(Counter(flat))))
        if distributed:
            dist.destroy_process_group()
        return
    train_set = Luna25Dataset(train_records, (args.input_D, args.input_H, args.input_W))
    val_set = Luna25Dataset(val_records, (args.input_D, args.input_H, args.input_W))
    batch_sampler = BalancedBatchSampler(
        train_set.labels, args.batch_size, args.class_ratio, args.seed,
        num_replicas=world_size, rank=rank)
    train_loader = DataLoader(train_set, batch_sampler=batch_sampler, num_workers=args.num_workers,
                              pin_memory=not args.no_cuda)
    # Validation preserves the natural distribution and order.
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=not args.no_cuda)
    model = make_model(args).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=1e-3)
    best = -1.0
    start_epoch = 0
    if args.resume_path:
        checkpoint = torch.load(args.resume_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = int(checkpoint.get('epoch', 0))
        best = float(checkpoint.get('best_balanced_accuracy', -1.0))
        print('Resumed {} at epoch {}, best_balanced_accuracy={:.4f}'.format(
            args.resume_path, start_epoch, best))
    if distributed:
        model = DDP(model, device_ids=None if device.type == 'cpu' else [local_rank],
                    output_device=None if device.type == 'cpu' else local_rank)
    save_path = Path(args.save_path)
    latest_path = (Path(args.latest_path) if args.latest_path else
                   save_path.with_name('lasted.pth'))
    for epoch in range(start_epoch, args.epochs):
        batch_sampler.set_epoch(epoch)
        train_result = run_epoch(train_loader, model, criterion, device, optimizer, distributed)
        if rank == 0:
            report('epoch {} train'.format(epoch + 1), train_result)
            eval_model = model.module if isinstance(model, DDP) else model
            val_result = run_epoch(val_loader, eval_model, criterion, device)
            report('epoch {} val'.format(epoch + 1), val_result)
            if val_result[3] > best:
                best = val_result[3]
                save_checkpoint(checkpoint_state(
                    args, epoch + 1, model, optimizer, best), save_path)
                print('Saved best checkpoint to {}'.format(save_path))
            save_checkpoint(checkpoint_state(
                args, epoch + 1, model, optimizer, best), latest_path)
            print('Saved latest checkpoint to {}'.format(latest_path))
        if distributed:
            value = torch.tensor(best, dtype=torch.float64, device=device)
            dist.broadcast(value, src=0)
            best = value.item()
            dist.barrier()
    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
