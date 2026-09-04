"""Run LUNA25 three-class inference and export challenge probabilities."""

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.luna25 import Luna25Dataset
from models import resnet


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='test')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output_csv', default='predictions.csv')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--model_depth', type=int, default=None,
                        choices=[10, 18, 34, 50, 101, 152, 200])
    parser.add_argument('--resnet_shortcut', default=None)
    parser.add_argument('--input_D', type=int, default=None)
    parser.add_argument('--input_H', type=int, default=None)
    parser.add_argument('--input_W', type=int, default=None)
    parser.add_argument('--no_cuda', action='store_true')
    return parser.parse_args()


def scan_test(root):
    root = Path(root)
    records = []
    for path in sorted(root.glob('*/*/*/*.nii.gz')):
        patient_id = path.relative_to(root).parts[0]
        # Luna25Dataset expects (path, label, patient_id); label is unused here.
        records.append((str(path), 0, patient_id))
    if not records:
        raise RuntimeError('no .nii.gz files found under {}'.format(root))
    return records


def setting(cli_value, config, name, default):
    return cli_value if cli_value is not None else config.get(name, default)


def main():
    args = options()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = checkpoint.get('model_config', {})
    depth = setting(args.model_depth, config, 'model_depth', 18)
    shortcut = setting(args.resnet_shortcut, config, 'resnet_shortcut', 'A')
    input_d = setting(args.input_D, config, 'input_D', 56)
    input_h = setting(args.input_H, config, 'input_H', 448)
    input_w = setting(args.input_W, config, 'input_W', 448)

    records = scan_test(args.data_root)
    dataset = Luna25Dataset(records, (input_d, input_h, input_w))
    # Evaluation deliberately uses ordinary sequential batching, never balancing.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=not args.no_cuda)
    factory = getattr(resnet, 'resnet{}'.format(depth))
    model = factory(sample_input_W=input_w, sample_input_H=input_h,
                    sample_input_D=input_d, shortcut_type=shortcut,
                    no_cuda=args.no_cuda, num_seg_classes=2,
                    task='classification', num_classes=3)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {key.removeprefix('module.'): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    model.to(device).eval()

    rows = []
    offset = 0
    with torch.no_grad():
        for volumes, _ in loader:
            probabilities = torch.softmax(model(volumes.to(device)), dim=1).cpu()
            for probability in probabilities:
                rows.append((records[offset][2], float(probability[1]), float(probability[2])))
                offset += 1

    output = Path(args.output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['patient_id', 'IsNotHumanBodyProb', 'IsStitchedProb'])
        writer.writerows(rows)
    print('Wrote {} predictions to {}'.format(len(rows), output))


if __name__ == '__main__':
    main()
