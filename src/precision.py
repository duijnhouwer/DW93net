"""Explicit TF32 policy for matrix multiplication and convolution."""
import torch


def configure_tf32(allow_tp32):
    # The mission plan calls this allow_tp32; the hardware format is named TF32.
    mode = 'tf32' if allow_tp32 else 'ieee'
    torch.backends.cuda.matmul.fp32_precision = mode
    torch.backends.cudnn.conv.fp32_precision = mode
