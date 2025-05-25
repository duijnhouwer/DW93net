"""
BixBitKernelBank: Bilocal-Bitemporal Kernel Bank for XT Stimulus Processing

This module defines a PyTorch layer that applies all possible 2×2 binary spatiotemporal
base kernels (16 in total) over an input tensor representing XT slices of optic flow data.

Each kernel is applied using specified temporal and spatial **gaps** between its elements,
implemented as dilation in PyTorch. A gap of 0 means the kernel is applied contiguously
(base kernel), while a gap of 1 inserts one empty row or column between elements, and so on.

Inputs:
    x: Tensor of shape [B, 1, T, X] representing a batch of XT data

Parameters:
    x_gap_range: Iterable of non-negative integers (default: range(0, 5))
        The number of horizontal (X) gaps inserted between kernel elements.

    t_gap_range: Iterable of non-negative integers (default: range(0, 5))
        The number of vertical (T) gaps inserted between kernel elements.

Output:
    A tensor of shape [B, N_kernels, T_out, X_out] where
    N_kernels = 16 × len(x_gap_range) × len(t_gap_range).
    Each channel corresponds to one binary kernel at one gap configuration.
"""

import torch
import torch.nn as nn
import itertools
import numpy as np

class BixBitKernelBank(nn.Module):
    def __init__(self, x_gap_range=range(0, 10), t_gap_range=range(0, 2)):
        super().__init__()

        self.base_kernels = self._generate_all_base_kernels()
        self.n_base_kernels = len(self.base_kernels)
        self.x_gaps = list(x_gap_range)
        self.t_gaps = list(t_gap_range)

        self.detectors = nn.ModuleList()

        for kernel in self.base_kernels:
            for x_gap in self.x_gaps:
                for t_gap in self.t_gaps:
                    dilation = (t_gap + 1, x_gap + 1)  # dilation = gap + 1
                    conv = nn.Conv2d(
                        in_channels=1,
                        out_channels=1,
                        kernel_size=(2, 2),
                        dilation=dilation,
                        padding=0,
                        bias=False
                    )
                    conv.weight.data[:] = torch.tensor(kernel, dtype=torch.float32).view(1, 1, 2, 2)
                    conv.requires_grad_(False)
                    self.detectors.append(conv)

    def forward(self, x):  # x shape: [B, 1, T, X]
        responses = [detector(x) for detector in self.detectors]
        return torch.cat(responses, dim=1)  # [B, N_kernels, T_out, X_out]

    @property
    def n_output_channels(self):
        return self.n_base_kernels * len(self.x_gaps) * len(self.t_gaps)

    @staticmethod
    def _generate_all_base_kernels():
        return [np.array(p).reshape(2, 2) for p in itertools.product([0, 1], repeat=4)]
