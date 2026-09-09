"""Learnable processing of precomputed Reichardt maps; predicts dx, dz, yaw."""

import torch.nn as nn

class FlowXTNet(nn.Module):
    def __init__(self, n_input_channels, dropout_p=0.5):
        super().__init__()
        if isinstance(n_input_channels, bool) or not isinstance(n_input_channels, int) or n_input_channels < 1:
            raise ValueError("n_input_channels must be a positive integer")
        self.n_input_channels = n_input_channels
        self.net = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, padding=1),
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

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.n_input_channels:
            raise ValueError("Expected precomputed [B,C,T,X] Reichardt responses")
        return self.net(x)
