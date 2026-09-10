"""Decode saved raw XT intensities before display or detector computation."""

import torch


def decode_stimulus(tensor):
    """Decode uint8 grayscale (0..255) to float32 intensities (0..1)."""
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.uint8:
        raise ValueError("Raw XT storage must be uint8 grayscale")
    return tensor.float().div_(255)
