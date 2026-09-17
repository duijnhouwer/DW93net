"""Cacheable signed luminance difference-of-Gaussians filtering."""
import math

import torch
import torch.nn.functional as F


class DoGFilter:
    """Fixed luminance DoG stage whose signed output is safe to cache."""

    def __init__(self, *, sigma_center=1., sigma_surround=3.):
        if not (math.isfinite(sigma_center) and math.isfinite(sigma_surround)
                and 0 < sigma_center < sigma_surround):
            raise ValueError("DoG requires finite 0 < sigma_center < sigma_surround")
        self.sigma_center = float(sigma_center)
        self.sigma_surround = float(sigma_surround)
        self.radius = math.ceil(3 * self.sigma_surround)
        self._kernels_by_device = {}

    @property
    def metadata(self):
        return dict(
            format="dog-signed-contrast-v1",
            dog=dict(sigma_center_px=self.sigma_center, sigma_surround_px=self.sigma_surround,
                     radius_px=self.radius, gaussian_normalization="separate_unit_sum",
                     boundary="valid", compute_dtype="float32"),
            frames_order=["first", "second"],
            signed_contrast="center_gaussian_minus_surround_gaussian",
        )

    def filter_frames(self, xt):
        """Return signed [B,1,2,X-2r] contrasts; no temporal filtering/padding."""
        if xt.ndim == 2:
            xt = xt[None, None]
        elif xt.ndim == 3 and xt.shape[0] == 1:
            xt = xt[None]
        if (xt.ndim != 4 or xt.shape[1:3] != (1, 2)
                or not xt.is_floating_point() or not torch.isfinite(xt).all()):
            raise ValueError("Expected finite floating-point [2,X], [1,2,X], or [B,1,2,X]")
        if xt.shape[-1] <= 2 * self.radius:
            raise ValueError("Stimulus too narrow for valid DoG filtering")
        # Construct normalized weights in float64, then explicitly compute in
        # float32. Separate Gaussian convolutions cancel uniform fields to float32
        # precision, without per-frame rescaling.
        # A dataset uses fixed sigmas. Construct once per device, retaining the
        # original float64 normalization and float32 convolution arithmetic.
        if xt.device not in self._kernels_by_device:
            positions = torch.arange(-self.radius, self.radius + 1,
                                     device=xt.device, dtype=torch.float64)
            kernels = []
            for sigma in (self.sigma_center, self.sigma_surround):
                kernel = torch.exp(-0.5 * (positions / sigma).square())
                kernels.append((kernel / kernel.sum()).float())
            self._kernels_by_device[xt.device] = torch.stack(kernels)[:, None]
        frames = xt.float().reshape(-1, 1, xt.shape[-1])
        filtered = F.conv1d(frames, self._kernels_by_device[xt.device])
        difference = filtered[:, 0] - filtered[:, 1]
        return difference.reshape(xt.shape[0], 1, 2, -1)

    __call__ = filter_frames

    @classmethod
    def from_metadata(cls, metadata):
        dog = metadata.get("dog", {})
        if metadata.get("format") != "dog-signed-contrast-v1":
            raise ValueError("Expected signed DoG contrast metadata")
        result = cls(sigma_center=dog.get("sigma_center_px"),
                     sigma_surround=dog.get("sigma_surround_px"))
        if any(metadata.get(key) != value for key, value in result.metadata.items()):
            raise ValueError("Inconsistent DoG contrast metadata")
        return result
