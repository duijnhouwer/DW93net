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


class FlowXtDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = [f for f in os.listdir(folder) if f.endswith('.pt')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.folder, filename)

        input_tensor = torch.load(filepath)

        cammot = parse_camera_motion_from_filename(filename)
        label_tensor = torch.tensor([
            cammot['dx'],
            cammot['dz'],
            cammot['rot_deg']
        ], dtype=torch.float32)

        return input_tensor, label_tensor


def parse_camera_motion_from_filename(filename: str) -> Dict[str, float]:
    """
    Extract dx, dz, and rot_deg (in physical units) from filename.

    Example:
        flow_udx=1234_udz=0_rot=-1.2345_id=42.pt

    Returns:
        Dictionary with keys: 'dx', 'dz', 'rot_deg' (values in meters and degrees)
    """
    pattern = r"udx=(-?\d+)_udz=(-?\d+)_rot=(-?\d+(?:\.\d+)?)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")

    udx, udz, rot = match.groups()
    return {
        'dx': int(udx) / 1e6,
        'dz': int(udz) / 1e6,
        'rot_deg': float(rot)
    }


def extract_range_from_foldername(foldername: str) -> Dict[str, float]:
    """
    Extract dx, dz, and rot ranges from folder name in physical units.
    Example foldername: 'udx=0,200000_udz=0,0_rot=-0.5,0.5'

    Returns:
        Dictionary with keys: 'dx', 'dz', 'rot_deg' (in meters and degrees)
    """
    def get_range(key):
        match = re.search(f"{key}=(-?\d*\.?\d*),(-?\d*\.?\d*)", foldername)
        if not match:
            return 0.0
        low, high = float(match.group(1)), float(match.group(2))
        return abs(high - low)

    return {
        'dx': get_range('udx') / 1e6,
        'dz': get_range('udz') / 1e6,
        'rot_deg': get_range('rot')
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
