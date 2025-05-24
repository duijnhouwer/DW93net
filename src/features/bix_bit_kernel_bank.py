"""
BixBitKernelBank: Bilocal-Bitemporal Kernel Bank for XT Stimulus Processing

This module defines a PyTorch layer that applies all possible 2x2 binary spatiotemporal
kernels (16 in total) over an input tensor representing XT slices of optic flow data.

Each kernel is convolved over the input with a range of spatial steps (dilation in X)
and temporal delays (dilation in T), enabling the detection of local motion, flicker,
and other spatiotemporal patterns.

The layer is non-trainable and intended to serve as a biologically inspired preprocessing
stage before learnable layers. It provides a complete and interpretable basis for early
spatiotemporal pattern recognition.

Inputs:
    x: Tensor of shape [B, 1, T, X] representing a batch of XT data

Parameters:
    step_range: Iterable of non-negative integers specifying spatial dilations (step sizes)
    delay_range: Iterable of positive integers specifying temporal dilations (delay lengths)

Output:
    A tensor of shape [B, N_kernels, T_out, X_out] where N_kernels = 16 * len(step_range) * len(delay_range)
    Each channel corresponds to the response of one binary kernel at one step/delay configuration.
"""

import torch
import torch.nn as nn
import itertools
import numpy as np

class BixBitKernelBank(nn.Module):
    def __init__(self, step_range=range(0, 5), delay_range=range(1, 5)):
        super().__init__()

        self.kernels = self._generate_all_binary_kernels()
        self.steps = list(step_range)
        self.delays = list(delay_range)

        self.detectors = nn.ModuleList()

        for kernel in self.kernels:
            for step in self.steps:
                for delay in self.delays:
                    dilation = (delay, step if step > 0 else 1)  # use step=1 for step=0
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

    @staticmethod
    def _generate_all_binary_kernels():
        return [np.array(p).reshape(2, 2) for p in itertools.product([0, 1], repeat=4)]
