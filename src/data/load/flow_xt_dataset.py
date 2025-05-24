# -*- coding: utf-8 -*-
"""
Created on Fri May 23 14:22:29 2025

@author: jduij
"""

import os
import torch
from torch.utils.data import Dataset
from flow_xt import parse_camera_motion_from_filename


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
