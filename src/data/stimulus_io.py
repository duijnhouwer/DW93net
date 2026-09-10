"""Decode saved raw XT intensities before display or detector computation."""

import torch


def decode_stimulus(tensor):
    """Return float32 intensities; uint8 uses 0..255, legacy floats use 0..1."""
    if not isinstance(tensor, torch.Tensor):
        raise ValueError("Expected a raw XT tensor")
    if tensor.dtype == torch.uint8:
        return tensor.float().div_(255)
    if tensor.is_floating_point():
        return tensor.float()
    raise ValueError("Raw XT storage must be uint8 grayscale or floating point")
