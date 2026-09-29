"""Cacheable signed luminance difference-of-Gaussians filtering."""
import math
from features.spatial import validate_boundary, circular_pad

import torch
import torch.nn.functional as F


class SupersampledDoG:
    """CPU float32 FFT DoG followed by box downsampling, one stimulus at a time."""

    def __init__(self, dog_filter, width, factor):
        import numpy as np
        from scipy import fft
        if dog_filter.boundary != 'valid':
            raise ValueError('Circular DoG requires filtering after downsampling')
        self.width, self.factor = width, factor
        self.radius = dog_filter.radius * factor
        self.high_width = width * factor
        if width <= 2 * dog_filter.radius:
            raise ValueError('Stimulus too narrow for valid DoG filtering')
        self.length = fft.next_fast_len(self.high_width + 2 * self.radius, real=True)
        self.workers = torch.get_num_threads()
        x = np.arange(-self.radius, self.radius + 1, dtype=np.float64)
        weights = []
        for sigma in (dog_filter.sigma_center, dog_filter.sigma_surround):
            w = np.exp(-.5 * (x / (sigma * factor)) ** 2)
            weights.append((w / w.sum()).astype(np.float32))
        self.spectrum = fft.rfft(weights[0] - weights[1], n=self.length, workers=self.workers)

    def __call__(self, frames):
        import numpy as np
        from scipy import fft
        if frames.shape != (1, 2, self.high_width) or not torch.isfinite(frames).all():
            raise ValueError('Expected finite supersampled [1,2,X] frames')
        # Honor the background generator's current thread throttle.
        workers = torch.get_num_threads()
        spectrum = fft.rfft(frames[0].numpy(), n=self.length, axis=-1, workers=workers)
        spectrum *= self.spectrum
        filtered = fft.irfft(spectrum, n=self.length, axis=-1, workers=workers,
                            overwrite_x=True)[:, 2*self.radius:self.high_width]
        low_width = self.width - 2 * (self.radius // self.factor)
        return torch.from_numpy(filtered.reshape(2, low_width, self.factor).mean(
            axis=2, dtype=np.float32))


class DoGFilter:
    """Fixed luminance DoG stage whose signed output is safe to cache."""

    def __init__(self, *, sigma_center=1., sigma_surround=3., boundary='valid'):
        if not (math.isfinite(sigma_center) and math.isfinite(sigma_surround)
                and 0 < sigma_center < sigma_surround):
            raise ValueError("DoG requires finite 0 < sigma_center < sigma_surround")
        self.boundary = validate_boundary(boundary)
        self.sigma_center = float(sigma_center)
        self.sigma_surround = float(sigma_surround)
        self.radius = math.ceil(3 * self.sigma_surround)
        self._kernels_by_device = {}

    @property
    def metadata(self):
        return dict(
            format="dog-signed-contrast-v2",
            dog=dict(sigma_center_px=self.sigma_center, sigma_surround_px=self.sigma_surround,
                     radius_px=self.radius, gaussian_normalization="separate_unit_sum",
                     boundary=self.boundary, compute_dtype="float32"),
            frames_order=["first", "second"],
            signed_contrast="center_gaussian_minus_surround_gaussian",
        )

    def filter_frames(self, xt):
        """Return signed [B,1,2,X] circular or [B,1,2,X-2r] valid contrasts."""
        if xt.ndim == 2:
            xt = xt[None, None]
        elif xt.ndim == 3 and xt.shape[0] == 1:
            xt = xt[None]
        if (xt.ndim != 4 or xt.shape[1:3] != (1, 2)
                or not xt.is_floating_point() or not torch.isfinite(xt).all()):
            raise ValueError("Expected finite floating-point [2,X], [1,2,X], or [B,1,2,X]")
        if xt.shape[-1] < 1 or (self.boundary == 'valid' and xt.shape[-1] <= 2 * self.radius):
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
        if self.boundary == 'circular':
            frames = circular_pad(frames, self.radius, self.radius)
        filtered = F.conv1d(frames, self._kernels_by_device[xt.device])
        difference = filtered[:, 0] - filtered[:, 1]
        return difference.reshape(xt.shape[0], 1, 2, -1)

    __call__ = filter_frames

    @classmethod
    def from_metadata(cls, metadata):
        dog = metadata.get("dog", {})
        if metadata.get("format") != "dog-signed-contrast-v2":
            raise ValueError("Unsupported DoG contrast metadata; regenerate this dataset")
        result = cls(sigma_center=dog.get("sigma_center_px"),
                     sigma_surround=dog.get("sigma_surround_px"), boundary=dog.get("boundary"))
        if any(metadata.get(key) != value for key, value in result.metadata.items()):
            raise ValueError("Inconsistent DoG contrast metadata")
        return result
