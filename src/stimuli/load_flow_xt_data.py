# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:22:29 2025

@author: jduij
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset, RandomSampler
import json
from pathlib import Path
from features.dog_motion import DoGFilter
from stimuli.metadata import read_metadata, DATA_SCHEMA


class DoGDataset(Dataset):
    """Load memory-mapped float32 signed DoG contrast bundles."""

    def __init__(self, folder, expected_metadata=None):
        self.folder = Path(folder)
        self._mapped = {}
        index_path = self.folder / 'bundles.json'
        if not index_path.exists():
            raise ValueError(f'DoG cache requires bundled storage: missing {index_path}')
        self.bundle_index = json.loads(index_path.read_text())
        self.sample_ids = [entry[0] for entry in self.bundle_index]
        metadata = read_metadata(self.folder.parent)
        self.storage_metadata = metadata
        if metadata.get('storage_format') != DATA_SCHEMA:
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
        sample_id, bundle, row = self.bundle_index[index]
        if bundle not in self._mapped:
            if len(self._mapped) >= 4:
                self._mapped.pop(next(iter(self._mapped)))
            self._mapped[bundle] = np.load(self.folder / bundle, mmap_mode='r')
        sample = self._mapped[bundle][row]
        contrast = torch.from_numpy(sample['dog_contrast'].copy())
        target = torch.from_numpy(sample['target'].copy())
        if (contrast.ndim != 2 or contrast.shape[0] != 2
                or contrast.dtype != torch.float32):
            raise ValueError(f'Invalid DoG contrast tensor: {sample_id}')
        if target.shape != (3,) or target.dtype != torch.float32 or not torch.isfinite(target).all():
            raise ValueError(f'Invalid motion target in bundle: {sample_id}')
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
