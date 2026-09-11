# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:22:29 2025

@author: jduij
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from typing import Dict
import re
import random
import json
from pathlib import Path
from features.dog_motion import DoGMotionBank
from stimuli.stimulus_io import decode_stimulus


class FlowXtDataset(Dataset):
    def __init__(self, folder):
        self.folder = folder
        self.files = sorted(f for f in os.listdir(folder) if f.endswith('.pt'))

    def __len__(self):
        return len(self.files)

    def _decode_input(self, tensor, filename):
        return decode_stimulus(tensor)

    def __getitem__(self, idx):
        filename = self.files[idx]
        filepath = os.path.join(self.folder, filename)

        input_tensor = torch.load(filepath, map_location="cpu", weights_only=True)
        input_tensor = self._decode_input(input_tensor, filename)

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
        bank = DoGMotionBank.from_metadata(metadata)
        self.metadata = bank.metadata
        if any(metadata.get(key) != value for key, value in self.metadata.items()):
            raise ValueError("Unsupported Reichardt cache format")
        if expected_metadata is not None and self.metadata != expected_metadata:
            raise ValueError("Reichardt detector configuration does not match")
        self.n_channels = bank.n_output_channels

    def _decode_input(self, responses, filename):
        if (not isinstance(responses, torch.Tensor) or responses.ndim != 3
                or responses.shape[0] != self.n_channels
                or responses.shape[1] != 1 or responses.shape[2] < 1 or responses.dtype != torch.float16
                or not torch.isfinite(responses).all()):
            raise ValueError(f"Invalid Reichardt response tensor: {filename}")
        return responses.float()


def make_flow_loader(dataset, batch_size, device, *, shuffle=False, num_workers=4):
    """Prepare batches in the background, pinning host memory for CUDA copies."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=torch.device(device).type == "cuda",
                      persistent_workers=num_workers > 0)


def parse_camera_motion_from_filename(filename: str) -> Dict[str, float]:
    """
    Extract dx, dz, and rot_deg (in physical units) from filename.
    Also detect if the filename contains '_superimposed'.
    Expects the format: ndx=<int>_ndz=<int>_nrot=<int>_id=<int>[_superimposed].pt
    where ndx/ndz are nano-unit translation deltas (scene units * 1e9) and nrot is the
    rotation in nano-degrees (degrees * 1e9).

    Example:
        ndx=1234000000_ndz=0_nrot=-1234500000000_id=42_superimposed.pt

    Returns:
        Dictionary with keys:
            - 'dx': translation in x (scene units)
            - 'dz': translation in z (scene units)
            - 'rot_deg': rotation in degrees
            - 'superimposed': True if '_superimposed' is in the filename, else False
    """
    pattern = r"^ndx=(-?\d+)_ndz=(-?\d+)_nrot=(-?\d+)_id=\d+(?:_superimposed)?\.pt$"
    match = re.fullmatch(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")

    ndx, ndz, nrot = match.groups()
    return {
        'dx': int(ndx) / 1e9,
        'dz': int(ndz) / 1e9,
        'rot_deg': int(nrot) / 1e9,
        'superimposed': '_superimposed' in filename
    }



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


def require_pure_translation(dataset):
    """Validate actual labels, including constant nonzero yaw folder ranges."""
    for filename in dataset.files:
        if parse_camera_motion_from_filename(filename)["rot_deg"] != 0:
            raise ValueError(f"Pure translation requires zero yaw: {filename}")


def stimulus_set_name(root_folder):
    """Parameter-set name for a raw/reichardt directory (or a legacy root)."""
    root = Path(root_folder)
    return root.parent.name if root.name in ('raw', 'reichardt') else root.name


def require_pure_translation_root(root_folder):
    """Fail before any file I/O unless folder bounds explicitly specify zero yaw."""
    name = stimulus_set_name(root_folder)
    match = re.search(r"(?:^|_)nrot=(-?\d+),(-?\d+)(?=_|$)", name)
    if match is None or any(int(bound) != 0 for bound in match.groups()):
        raise ValueError(
            f"Pure translation requires rot_deg=0 for the entire dataset "
            f"(nrot=0,0); stopping before loading files: {root_folder}"
        )
