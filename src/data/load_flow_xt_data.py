# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:22:29 2025

@author: jduij
"""

import os
import torch
from torch.utils.data import Dataset, Subset
from typing import Dict
import re
import random
import json
from pathlib import Path
from features.reichardt import ReichardtBank


class FlowXtDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(f for f in os.listdir(folder) if f.endswith('.pt'))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.folder, filename)

        input_tensor = torch.load(filepath, map_location="cpu", weights_only=True)

        cammot = parse_camera_motion_from_filename(filename)
        label_tensor = torch.tensor([
            cammot['dx'],
            cammot['dz'],
            cammot['rot_deg']
        ], dtype=torch.float32)

        return input_tensor, label_tensor


class ReichardtDataset(FlowXtDataset):
    """Load saved maps, requiring identical detector semantics across datasets."""

    def __init__(self, folder, expected_metadata=None):
        super().__init__(folder)
        metadata = json.loads((Path(folder).parent / "reichardt.json").read_text())
        bank = ReichardtBank(metadata["dx_values"])
        self.metadata = bank.metadata
        if any(metadata.get(key) != value for key, value in self.metadata.items()):
            raise ValueError("Unsupported Reichardt cache format")
        if expected_metadata is not None and self.metadata != expected_metadata:
            raise ValueError("Reichardt detector configuration does not match")
        self.n_channels = bank.n_output_channels

    def __getitem__(self, idx):
        responses, label = super().__getitem__(idx)
        if (not isinstance(responses, torch.Tensor) or responses.ndim != 3
                or responses.shape[0] != self.n_channels
                or responses.shape[1] != 1 or responses.shape[2] < 1 or not responses.is_floating_point()
                or not torch.isfinite(responses).all()):
            raise ValueError(f"Invalid Reichardt response tensor: {self.files[idx]}")
        return responses.float(), label


def parse_camera_motion_from_filename(filename: str) -> Dict[str, float]:
    """
    Extract dx, dz, and rot_deg (in physical units) from filename.
    Also detect if the filename contains '_superimposed'.
    Expects the format: flow_nux=<int>_nuz=<int>_nrot=<int>[_superimposed]_id=<int>.pt
    where nux/nuz are nano-unit translations (meters * 1e9) and nrot is the
    rotation in nano-degrees (degrees * 1e9).

    Example:
        flow_nux=1234000000_nuz=0_nrot=-1234500000000_superimposed_id=42.pt

    Returns:
        Dictionary with keys:
            - 'dx': translation in x (meters)
            - 'dz': translation in z (meters)
            - 'rot_deg': rotation in degrees
            - 'superimposed': True if '_superimposed' is in the filename, else False
    """
    pattern = r"nux=(-?\d+)_nuz=(-?\d+)_nrot=(-?\d+)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")

    nux, nuz, nrot = match.groups()
    return {
        'dx': int(nux) / 1e9,
        'dz': int(nuz) / 1e9,
        'rot_deg': int(nrot) / 1e9,
        'superimposed': '_superimposed' in filename
    }



def extract_ranges_from_foldername(foldername: str) -> Dict[str, float]:
    """
    Extract dx, dz, and rot ranges from folder name in physical units.
    Expects the format: nux=<min>,<max>_nuz=<min>,<max>_nrot=<min>,<max>
    where the values are nano-units (meters * 1e9) for dx/dz and
    nano-degrees (degrees * 1e9) for rotation.
    Example foldername: 'nux=0,20000000_nuz=0,0_nrot=-500000000,500000000'

    Returns:
        Dictionary with keys: 'dx', 'dz', 'rot_deg' (in meters and degrees)
    """
    def get_range(key):
        match = re.search(rf"{key}=(-?\d*\.?\d*),(-?\d*\.?\d*)", foldername)
        if not match:
            return 0.0
        low, high = float(match.group(1)), float(match.group(2))
        return abs(high - low)

    return {
        'dx': get_range('nux') / 1e9,
        'dz': get_range('nuz') / 1e9,
        'rot_deg': get_range('nrot') / 1e9
    }


def subset_dataset(dataset, subset_cfg: dict):
    """
    Return a subset of a dataset using the given configuration.

    Args:
        dataset: a PyTorch Dataset instance
        subset_cfg: dictionary with keys:
            - 'n' (int): number of samples
            - 'method' (str): 'first' or 'random'

    Returns:
        Subset or original dataset
    """
    if subset_cfg is None:
        return dataset

    n = subset_cfg.get("n", 0)
    method = subset_cfg.get("method", "random")

    if n <= 0 or n >= len(dataset):
        return dataset

    if method == "first":
        indices = list(range(n))
    elif method == "random":
        indices = random.sample(range(len(dataset)), n)
    else:
        raise ValueError(f"Unknown subset method: {method}")

    return Subset(dataset, indices)
