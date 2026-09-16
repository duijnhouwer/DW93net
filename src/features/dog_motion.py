"""Cacheable signed luminance DoG and legacy four-polarity product reference."""
import math

import torch
import torch.nn.functional as F

from features.reichardt import ReichardtBank


class DoGFilter:
    """Fixed luminance DoG stage whose signed output is safe to cache."""

    def __init__(self, *, sigma_center=1., sigma_surround=3.):
        if not (math.isfinite(sigma_center) and math.isfinite(sigma_surround)
                and 0 < sigma_center < sigma_surround):
            raise ValueError("DoG requires finite 0 < sigma_center < sigma_surround")
        self.sigma_center = float(sigma_center)
        self.sigma_surround = float(sigma_surround)
        self.radius = math.ceil(3 * self.sigma_surround)

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
        positions = torch.arange(-self.radius, self.radius + 1, device=xt.device, dtype=torch.float64)
        kernels = []
        for sigma in (self.sigma_center, self.sigma_surround):
            kernel = torch.exp(-0.5 * (positions / sigma).square())
            kernels.append((kernel / kernel.sum()).float())
        frames = xt.float().reshape(-1, 1, xt.shape[-1])
        filtered = F.conv1d(frames, torch.stack(kernels)[:, None])
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


class DoGMotionBank:
    def __init__(self, dx_values=tuple(range(-60, 61)), *, sigma_center=1., sigma_surround=3.):
        self.bank = ReichardtBank(dx_values)
        self.filter = DoGFilter(sigma_center=sigma_center, sigma_surround=sigma_surround)
        self.sigma_center = self.filter.sigma_center
        self.sigma_surround = self.filter.sigma_surround
        self.radius = self.filter.radius
        self.dx_values = self.bank.dx_values

    @property
    def n_output_channels(self):
        return 4 * self.bank.n_output_channels

    @property
    def metadata(self):
        return dict(
            format="dog-onoff-products-v2", dx_values=list(self.dx_values),
            displacement_units="pixels_per_frame_step", temporal_delay_steps=1,
            dog=self.filter.metadata["dog"],
            polarity_order=["ON_ON", "OFF_OFF", "ON_OFF", "OFF_ON"],
            polarity_time_order="first_frame_then_second_frame",
            channel_order="polarity_then_displacement",
            rectification="ON=relu(DoG);OFF=relu(-DoG)",
            crop=dict(left=self.radius + max(0, max(self.dx_values)),
                      right=self.radius + max(0, -min(self.dx_values)),
                      anchor="present_frame_original_pixel"),
        )

    def filter_frames(self, xt):
        return self.filter(xt)

    def __call__(self, xt):
        batched = xt.ndim == 4
        contrast = self.filter_frames(xt)
        on, off = contrast.relu(), (-contrast).relu()
        # Keep four nonnegative blocks for future interaction/normalization tests.
        # Pair names mean first-frame polarity followed by second-frame polarity.
        # Subtraction belongs in the model, not in the recoverable cache.
        on_off = torch.cat((on[:, :, :1], off[:, :, 1:]), dim=2)
        off_on = torch.cat((off[:, :, :1], on[:, :, 1:]), dim=2)
        result = torch.cat([self.bank(pair) for pair in (on, off, on_off, off_on)], dim=1)
        return result if batched else result[0]

    @classmethod
    def from_metadata(cls, metadata):
        dog = metadata.get("dog", {})
        if metadata.get("format") != "dog-onoff-products-v2":
            raise ValueError("Expected DoG ON/OFF product cache; regenerate preprocessing")
        bank = cls(metadata["dx_values"], sigma_center=dog["sigma_center_px"],
                   sigma_surround=dog["sigma_surround_px"])
        if any(metadata.get(key) != value for key, value in bank.metadata.items()):
            raise ValueError("Inconsistent DoG ON/OFF metadata")
        return bank
