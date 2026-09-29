"""Shared spatial boundary policy and periodic indexing for the flow pipeline."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def validate_boundary(boundary):
    if boundary not in ('valid', 'circular'):
        raise ValueError("boundary must be 'valid' or 'circular'")
    return boundary


def resolve_boundary(fov_deg):
    if (isinstance(fov_deg, bool) or not isinstance(fov_deg, (int, float))
            or not math.isfinite(fov_deg) or not 0 < fov_deg <= 360):
        raise ValueError('fov_deg must be finite and satisfy 0 < value <= 360')
    return 'circular' if fov_deg == 360 else 'valid'


def modular_indices(indices, width):
    """Works for NumPy and Torch arrays, including offsets larger than width."""
    if width < 1:
        raise ValueError('Spatial width must be positive')
    return indices % width


def circular_pad(x, left, right):
    indices = torch.arange(-left, x.shape[-1] + right, device=x.device)
    return x.index_select(-1, modular_indices(indices, x.shape[-1]))


class CircularAvgPool1d(nn.AvgPool1d):
    def forward(self, x):
        padding = self.padding[0] if isinstance(self.padding, tuple) else self.padding
        return F.avg_pool1d(circular_pad(x, padding, padding), self.kernel_size,
                            self.stride, 0, self.ceil_mode, self.count_include_pad)


class CircularConv1d(nn.Conv1d):
    def forward(self, x):
        padding = self.padding[0]
        return F.conv1d(circular_pad(x, padding, padding), self.weight, self.bias,
                        self.stride, 0, self.dilation, self.groups)


def spatial_pool(kernel_size, *, stride, padding, boundary):
    cls = CircularAvgPool1d if validate_boundary(boundary) == 'circular' else nn.AvgPool1d
    return cls(kernel_size, stride=stride, padding=padding, count_include_pad=False)


def spatial_conv(*args, boundary, **kwargs):
    cls = CircularConv1d if validate_boundary(boundary) == 'circular' else nn.Conv1d
    return cls(*args, **kwargs)
