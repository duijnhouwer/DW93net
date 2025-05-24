# -*- coding: utf-8 -*-
"""
Created on Fri May 23 17:48:51 2025

@author: jduij
"""

"""
FlowXTNet: Motion Estimation Network for XT Stimuli

This model uses the BixBitKernelBank as a frozen preprocessing layer to extract all 2x2 binary
spatiotemporal features across a range of spatial steps and temporal delays. These features
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
    step_range (iterable): Spatial dilation values to apply to kernels (default: range(0, 5))
    delay_range (iterable): Temporal dilation values (default: range(1, 5))
    dropout_p (float): Dropout probability for regularization (default: 0.5)

Returns:
    Tensor of shape [B, 3], where each column represents dx, dz, and rot_deg respectively
"""

import torch.nn as nn
from features.bix_bit_kernel_bank import BixBitKernelBank

class FlowXTNet(nn.Module):
    def __init__(self, step_range=range(0, 5), delay_range=range(1, 5), dropout_p=0.5):
        super().__init__()

        # Frozen preprocessing
        self.pre = BixBitKernelBank(step_range=step_range, delay_range=delay_range)
        for p in self.pre.parameters():
            p.requires_grad = False

        # Compute number of kernel outputs
        n_kernels = 16 * len(step_range) * len(delay_range)

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
        x = self.pre(x)    # [B, N_kernels, T_out, X_out]
        return self.net(x) # [B, 3]
