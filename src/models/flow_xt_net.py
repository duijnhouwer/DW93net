# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:48:51 2025

@author: jduij
"""

"""
FlowXTNet: Motion Estimation Network for XT Stimuli

This model uses the BixBitKernelBank as a frozen preprocessing layer to extract all 2×2 binary
spatiotemporal features across a range of spatial and temporal gaps. These features
are processed through a stack of convolutional layers with batch normalization and dropout,
and pooled globally to regress camera motion parameters.

Architecture:
    Input: [B, 1, T, X] XT slice of optic flow
    → BixBitKernelBank (frozen)
    → Conv2D + BatchNorm + ReLU + Dropout
    → Conv2D + BatchNorm + ReLU
    → AdaptiveAvgPool2d
    → Fully connected layer
    Output: [B, 3] representing dx, dz, and rot_deg

Args:
    x_gap_range (iterable): Horizontal gaps between kernel elements (default: range(0, 5))
    t_gap_range (iterable): Temporal gaps between kernel elements (default: range(0, 5))
    dropout_p (float): Dropout probability for regularization (default: 0.5)

Returns:
    Tensor of shape [B, 3], where each column represents dx, dz, and rot_deg respectively
"""

import torch.nn as nn
from features.bix_bit_kernel_bank import BixBitKernelBank

class FlowXTNet(nn.Module):
    def __init__(self, x_gap_range=range(0, 10), t_gap_range=range(0, 2), dropout_p=0.5):
        super().__init__()

        # Frozen preprocessing
        self.pre = BixBitKernelBank(x_gap_range=x_gap_range, t_gap_range=t_gap_range)
        for p in self.pre.parameters():
            p.requires_grad = False

        # Compute number of kernel outputs dynamically
        n_kernels = self.pre.n_output_channels

        # Learnable feature processing
        self.net = nn.Sequential(
            nn.Conv2d(n_kernels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # [B, 64, 1, 1]
            nn.Flatten(),
            nn.Linear(64, 3)  # [dx, dz, rot_deg]
        )

    def forward(self, x):  # x: [B, 1, T, X]
        x = self.pre(x)     # [B, N_kernels, T_out, X_out]
        return self.net(x)  # [B, 3]
