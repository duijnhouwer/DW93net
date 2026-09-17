# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:22:29 2025

@author: jduij
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, RandomSampler
from typing import Dict
import re
import json
from pathlib import Path
from features.dog_motion import DoGFilter


class DoGDataset(Dataset):
    """Load memory-mapped float32 signed DoG contrast bundles."""

    def __init__(self, folder, expected_metadata=None):
        self.folder = Path(folder)
        self._mapped = {}
        index_path = self.folder / 'bundles.json'
        if not index_path.exists():
            raise ValueError(f'DoG cache requires bundled storage: missing {index_path}')
        self.bundle_index = json.loads(index_path.read_text())
        self.files = [entry[0] for entry in self.bundle_index]
        metadata = json.loads((self.folder.parent / 'dog.json').read_text())
        if metadata.get('storage_format') != 'dog-contrast-bundles-v1':
            raise ValueError('Unsupported DoG cache storage format')
        if (metadata.get('storage_dtype') != 'torch.float32'
                or metadata.get('finite_validated') is not True
                or not isinstance(metadata.get('contrast_shape'), list)
                or len(metadata['contrast_shape']) != 2
                or metadata['contrast_shape'][0] != 2):
            raise ValueError('Invalid DoG cache storage metadata')
        dog_filter = DoGFilter.from_metadata(metadata)
        self.metadata = dog_filter.metadata
        if expected_metadata is not None and self.metadata != expected_metadata:
            raise ValueError('DoG filter configuration does not match')

    def __len__(self):
        return len(self.bundle_index)

    def __getitem__(self, index):
        filename, bundle, row = self.bundle_index[index]
        if bundle not in self._mapped:
            if len(self._mapped) >= 4:
                self._mapped.pop(next(iter(self._mapped)))
            self._mapped[bundle] = np.load(self.folder / bundle, mmap_mode='r')
        sample = self._mapped[bundle][row]
        contrast = torch.from_numpy(sample['dog_contrast'].copy())
        target = torch.from_numpy(sample['target'].copy())
        if (contrast.ndim != 2 or contrast.shape[0] != 2
                or contrast.dtype != torch.float32):
            raise ValueError(f'Invalid DoG contrast tensor: {filename}')
        if target.shape != (3,) or target.dtype != torch.float32 or not torch.isfinite(target).all():
            raise ValueError(f'Invalid motion target in bundle: {filename}')
        return contrast, target

    def __getstate__(self):
        return dict(self.__dict__, _mapped={})


class MirroredDoGDataset(Dataset):
    """In-memory horizontal reflections of signed DoG contrasts."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        contrast, target = self.dataset[index]
        if contrast.ndim != 2 or contrast.shape[0] != 2:
            raise ValueError('Expected signed DoG contrast shaped [2, X]')
        mirrored_target = target.clone()
        mirrored_target[0] = -mirrored_target[0]
        mirrored_target[2] = -mirrored_target[2]
        return contrast.flip(-1), mirrored_target


def with_mirrored_dog_training(dataset):
    return ConcatDataset([dataset, MirroredDoGDataset(dataset)])


def make_flow_loader(dataset, batch_size, device, *, shuffle=False, num_workers=4,
                     shuffle_generator=None, worker_generator=None):
    """Prepare batches in the background, pinning host memory for CUDA copies."""
    # DataLoader consumes its generator when creating worker seeds, but not
    # when resetting persistent workers. Give the sampler a separate generator
    # so restarting the loader cannot change an epoch's shuffled trial order.
    # Loading and mirroring are deterministic; workers do not choose trials.
    sampler = RandomSampler(dataset, generator=shuffle_generator) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=num_workers,
                      generator=worker_generator,
                      pin_memory=torch.device(device).type == "cuda",
                      persistent_workers=num_workers > 0)


def extract_ranges_from_foldername(foldername: str) -> Dict[str, float]:
    """
    Derive Cartesian component ranges from polar ranges in the folder name.

    Expected format: nspeed=<min>,<max>_nheading=<min>,<max>_nrot=<min>,<max>.
    Speed uses nano scene-units per step and heading/rotation use nano-degrees.
    Component extrema include the range endpoints and any sine/cosine extrema
    inside the heading interval.

    Returns:
        Dictionary with keys: 'dx', 'dz', 'rot_deg' (scene units and degrees)
    """
    def get_bounds(key):
        match = re.search(rf"{key}=(-?\d*\.?\d*),(-?\d*\.?\d*)", foldername)
        if not match:
            raise ValueError(f"Dataset folder is missing {key}=min,max: {foldername}")
        low, high = float(match.group(1)), float(match.group(2))
        if low > high:
            raise ValueError(f"Dataset folder has reversed {key} bounds: {foldername}")
        return low, high

    speed_low, speed_high = (value / 1e9 for value in get_bounds('nspeed'))
    heading_low, heading_high = (value / 1e9 for value in get_bounds('nheading'))
    rot_low, rot_high = (value / 1e9 for value in get_bounds('nrot'))
    if speed_low <= 0 or heading_low < -180 or heading_high > 180:
        raise ValueError(f"Invalid polar motion bounds in dataset folder: {foldername}")

    def component_range(function, critical_angles):
        angles = [heading_low, heading_high]
        angles.extend(angle for angle in critical_angles
                      if heading_low <= angle <= heading_high)
        values = [speed * function(np.deg2rad(angle))
                  for speed in (speed_low, speed_high) for angle in angles]
        return max(values) - min(values)

    return {
        'dx': component_range(np.sin, (-90, 90)),
        'dz': component_range(np.cos, (-180, 0, 180)),
        'rot_deg': rot_high - rot_low,
    }


def stimulus_set_name(root_folder):
    """Parameter-set name containing a DoG cache directory."""
    root = Path(root_folder)
    return root.parent.name if root.name == 'dog' else root.name


def require_pure_translation_root(root_folder):
    """Fail before any file I/O unless folder bounds explicitly specify zero yaw."""
    name = stimulus_set_name(root_folder)
    match = re.search(r"(?:^|_)nrot=(-?\d+),(-?\d+)(?=_|$)", name)
    if match is None or any(int(bound) != 0 for bound in match.groups()):
        raise ValueError(
            f"Pure translation requires rot_deg=0 for the entire dataset "
            f"(nrot=0,0); stopping before loading files: {root_folder}"
        )
