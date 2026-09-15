"""LUNA25 2D coronal-slice dataset and fixed-ratio CT batch sampler.

Each record is one CT/series.  The dataset returns eleven coronal slices as one
sample; the model is responsible for aggregating those slices to CT level.
"""

import random
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage
import torch
from torch.utils.data import Dataset, Sampler

from datasets.luna25 import scan_luna25


NUM_SLICES = 11
SLICE_SIZE = (224, 224)


def scan_luna25_2d(root):
    """Scan records and add a series id to the existing LUNA25 records.

    ``scan_luna25`` already handles the different normal/special directory
    depths and class labels.  Its patient id is the fourth path component from
    the end; the series id is the next component toward the file.
    """
    records = []
    for path, label, patient_id in scan_luna25(root):
        path_parts = Path(path).parts
        if len(path_parts) < 3:
            raise ValueError("Cannot infer series_id from path: {}".format(path))
        series_id = path_parts[-3]
        records.append((path, label, patient_id, series_id))
    return records


def limit_normal_patients_one_series(records, normal_size=0, seed=1):
    """Limit training normal data by patient and retain one series per patient.

    ``normal_size=0`` means all normal patients.  Fake and Composition records
    are always retained unchanged.  The first series in sorted order is kept
    for each selected normal patient, making the choice deterministic.
    """
    if normal_size < 0:
        raise ValueError("normal_size must be >= 0")

    normal_patients = sorted({item[2] for item in records if item[1] == 0})
    if normal_size and normal_size < len(normal_patients):
        rng = random.Random(seed)
        selected_patients = set(rng.sample(normal_patients, normal_size))
    else:
        selected_patients = set(normal_patients)

    selected_normal = []
    seen_patients = set()
    for item in sorted(records, key=lambda value: (value[2], value[3], value[0])):
        if item[1] == 0 and item[2] in selected_patients and item[2] not in seen_patients:
            selected_normal.append(item)
            seen_patients.add(item[2])

    return [item for item in records if item[1] != 0] + selected_normal


class Luna25Dataset2D(Dataset):
    """Return ``[11, 3, 224, 224]`` coronal input for one CT/series."""

    def __init__(self, records, train=False, num_slices=NUM_SLICES,
                 slice_size=SLICE_SIZE):
        if num_slices != 11 or num_slices % 2 == 0:
            raise ValueError("This first version requires exactly 11 slices")
        self.records = list(records)
        self.labels = [item[1] for item in self.records]
        self.train = train
        self.num_slices = num_slices
        self.slice_size = tuple(slice_size)

    def __len__(self):
        return len(self.records)

    @staticmethod
    def _invalid_border_crop(volume):
        # Keep the same invalid-border rule as the original MedicalNet dataset.
        valid = np.where(volume != volume[0, 0, 0])
        if valid[0].size:
            lo = [int(axis.min()) for axis in valid]
            hi = [int(axis.max()) + 1 for axis in valid]
            volume = volume[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        return volume

    def _center_y(self, y_size):
        center = y_size // 2
        if self.train:
            center += random.randint(-2, 2)
        return min(max(center, 0), y_size - 1)

    def __getitem__(self, index):
        path, label, patient_id, series_id = self.records[index]
        volume = sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(np.float32)
        if volume.ndim != 3 or any(size == 0 for size in volume.shape):
            raise ValueError("Expected a non-empty [Z,Y,X] CT volume: {}".format(path))

        # SimpleITK returns [Z,Y,X].  Coronal planes therefore index the Y axis:
        # volume[:, y, :] has shape [Z, X] and is one coronal slice.
        volume = self._invalid_border_crop(volume)
        y_size = volume.shape[1]
        center = self._center_y(y_size)
        y_indices = np.clip(
            np.arange(center - self.num_slices // 2,
                      center + self.num_slices // 2 + 1),
            0, y_size - 1)

        slices = []
        for y in y_indices:
            coronal = volume[:, int(y), :]
            resized = ndimage.zoom(
                coronal,
                (self.slice_size[0] / coronal.shape[0],
                 self.slice_size[1] / coronal.shape[1]),
                order=1)
            # zoom normally gives the requested shape, but explicitly correct
            # rounding differences so the dataset contract is unconditional.
            if resized.shape != self.slice_size:
                resized = torch.nn.functional.interpolate(
                    torch.from_numpy(resized).float()[None, None],
                    size=self.slice_size, mode='bilinear', align_corners=False
                )[0, 0].numpy()
            slices.append(resized.astype(np.float32, copy=False))
        slices = np.stack(slices, axis=0)

        # Match the original dataset's non-zero z-score convention.
        pixels = slices[slices != 0]
        if not pixels.size:
            pixels = slices.reshape(-1)
        mean, std = float(pixels.mean()), float(pixels.std())
        slices = (slices - mean) / max(std, 1e-6)
        slices = np.repeat(slices[:, None], 3, axis=1).astype(np.float32)
        return torch.from_numpy(slices), int(label), patient_id, series_id


class DualHeadBatchSampler(Sampler):
    """Yield exactly 2 normal, 1 fake and 1 Composition CT per batch.

    The number of batches is fixed by ``steps_per_epoch``.  Every class pool is
    shuffled at the beginning of each epoch and reshuffled whenever it wraps,
    so minority classes may be repeated without determining epoch length.
    """

    BATCH_SIZE = 4
    CLASS_COUNTS = (2, 1, 1)

    def __init__(self, labels, steps_per_epoch=100, seed=1, batch_size=BATCH_SIZE):
        if batch_size != self.BATCH_SIZE:
            raise ValueError("DualHeadBatchSampler requires batch_size=4")
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1")
        self.steps_per_epoch = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.batch_size = self.BATCH_SIZE
        self.indices = {
            label: [i for i, value in enumerate(labels) if value == label]
            for label in range(3)
        }
        if any(not self.indices[label] for label in range(3)):
            raise ValueError("training split must contain normal, fake and Composition")

    def __len__(self):
        return self.steps_per_epoch

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        pools = {label: values[:] for label, values in self.indices.items()}
        positions = {label: 0 for label in range(3)}
        for pool in pools.values():
            rng.shuffle(pool)

        def take(label):
            if positions[label] == len(pools[label]):
                positions[label] = 0
                rng.shuffle(pools[label])
            value = pools[label][positions[label]]
            positions[label] += 1
            return value

        for _ in range(self.steps_per_epoch):
            batch = [take(0), take(0), take(1), take(2)]
            rng.shuffle(batch)
            yield batch
