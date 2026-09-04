"""LUNA25 three-class dataset and balanced batch sampler."""

import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from scipy import ndimage
from torch.utils.data import Dataset, Sampler


CLASS_NAMES = ("normal", "fake", "Composition")


def scan_luna25(root):
    """Return (path, label, patient_id) records; special folders aren't normal."""
    root = Path(root)
    records = []
    for label, folder in ((1, root / "fake"), (2, root / "Composition")):
        if folder.exists():
            for path in sorted(folder.glob("*/*/*/*.nii.gz")):
                records.append((str(path), label, path.parts[-4]))
    excluded = {"fake", "Composition"}
    for path in sorted(root.glob("*/*/*/*.nii.gz")):
        if path.parts[-5] not in excluded:
            records.append((str(path), 0, path.parts[-4]))
    return records


def patient_split(records, val_ratio=0.2, seed=1):
    """Stratified patient-level split, with no patient shared across splits."""
    by_label = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_label[record[1]][record[2]].append(record)
    rng = random.Random(seed)
    train, val = [], []
    for label in range(3):
        patients = sorted(by_label[label])
        rng.shuffle(patients)
        n_val = 0 if val_ratio <= 0 or len(patients) < 2 else min(
            len(patients) - 1, max(1, round(len(patients) * val_ratio)))
        val_patients = set(patients[:n_val])
        for patient, items in by_label[label].items():
            (val if patient in val_patients else train).extend(items)
    # A patient may exceptionally occur under more than one class directory.
    # Prefer training it wholesale over allowing cross-split leakage.
    train_patients = {item[2] for item in train}
    leaked = train_patients.intersection(item[2] for item in val)
    if leaked:
        train.extend(item for item in val if item[2] in leaked)
        val = [item for item in val if item[2] not in leaked]
    return train, val


def limit_normal_patients(records, normal_size=0, seed=1):
    """Randomly retain at most ``normal_size`` class-0 patients.

    Non-normal records are always retained.  A value of zero keeps every normal
    patient.  Selection is patient based, so all series belonging to a selected
    patient stay together.
    """
    if normal_size < 0:
        raise ValueError("normal_size must be >= 0")
    normal_patients = sorted({item[2] for item in records if item[1] == 0})
    if normal_size == 0 or normal_size >= len(normal_patients):
        return list(records)
    rng = random.Random(seed)
    selected = set(rng.sample(normal_patients, normal_size))
    return [item for item in records if item[1] != 0 or item[2] in selected]


class Luna25Dataset(Dataset):
    def __init__(self, records, input_size=(56, 448, 448)):
        self.records = list(records)
        self.labels = [item[1] for item in self.records]
        self.input_size = tuple(input_size)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path, label, _ = self.records[index]
        volume = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        # Reuse MedicalNet's invalid-border crop, resize and non-zero z-score.
        valid = np.where(volume != volume[0, 0, 0])
        if valid[0].size:
            lo = [int(axis.min()) for axis in valid]
            hi = [int(axis.max()) + 1 for axis in valid]
            volume = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        scale = [target / current for target, current in zip(self.input_size, volume.shape)]
        volume = ndimage.zoom(volume, scale, order=1)
        pixels = volume[volume != 0]
        if not pixels.size:
            pixels = volume.reshape(-1)
        mean, std = float(pixels.mean()), float(pixels.std())
        volume = (volume - mean) / max(std, 1e-6)
        return torch.from_numpy(volume[None].astype(np.float32)), label


class BalancedBatchSampler(Sampler):
    """Batch sampler with epoch length driven by class 0 (normal) samples."""
    def __init__(self, labels, batch_size, class_ratio=(2, 1, 1), seed=1,
                 num_replicas=1, rank=0):
        if batch_size < 1 or len(class_ratio) != 3 or any(x <= 0 for x in class_ratio):
            raise ValueError("batch_size and all three class_ratio values must be positive")
        self.batch_size = batch_size
        self.ratio = tuple(class_ratio)
        self.seed = seed
        self.num_replicas = num_replicas
        self.rank = rank
        if num_replicas < 1 or rank < 0 or rank >= num_replicas:
            raise ValueError("invalid num_replicas or rank")
        self.epoch = 0
        self.indices = [[i for i, y in enumerate(labels) if y == c] for c in range(3)]
        if any(not values for values in self.indices):
            raise ValueError("training split must contain all three classes")
        self.num_samples = math.ceil(len(self.indices[0]) * sum(self.ratio) / self.ratio[0])

    def __len__(self):
        return math.ceil(self.num_samples / (self.batch_size * self.num_replicas))

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = [values[:] for values in self.indices]
        for pool in pools:
            rng.shuffle(pool)
        positions = [0, 0, 0]
        scores = [0, 0, 0]
        tie_order = list(range(3))
        rng.shuffle(tie_order)
        tie_rank = {label: rank for rank, label in enumerate(tie_order)}
        stream = []
        for _ in range(self.num_samples):
            for c in range(3):
                scores[c] += self.ratio[c]
            label = max(range(3), key=lambda c: (scores[c], -tie_rank[c]))
            scores[label] -= sum(self.ratio)
            if positions[label] == len(pools[label]):
                positions[label] = 0
                rng.shuffle(pools[label])
            stream.append(pools[label][positions[label]])
            positions[label] += 1
        # DDP requires the same number of full steps on every rank.  Pad only
        # with minority classes so normal remains an effectively no-replacement
        # traversal even when the final global step is incomplete.
        total = len(self) * self.batch_size * self.num_replicas
        minority = pools[1] + pools[2]
        rng.shuffle(minority)
        while len(stream) < total:
            stream.append(minority[(len(stream) - self.num_samples) % len(minority)])
        global_batches = [stream[start:start + self.batch_size]
                          for start in range(0, total, self.batch_size)]
        yield from global_batches[self.rank::self.num_replicas]
