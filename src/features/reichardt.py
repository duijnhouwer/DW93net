"""Deterministic products R(x) = I(0,x-dX) * I(1,x)."""

from numbers import Integral
import torch


class ReichardtBank:
    """Signed dX in pixels; one-bin delay and supplied span order.

    All channels share present-time anchors, cropped to the intersection of
    valid coordinates. Positive dX detects motion toward increasing X.
    Opposing directions remain separate products (no subtraction).
    """

    def __init__(self, dx_values=tuple(range(-10, 11))):
        self.dx_values = tuple(dx_values)
        if (not self.dx_values
                or any(isinstance(v, bool) or not isinstance(v, Integral)
                       for v in self.dx_values)
                or len(set(self.dx_values)) != len(self.dx_values)):
            raise ValueError("dX must contain distinct integers")
        self.dx_values = tuple(int(v) for v in self.dx_values)

    @property
    def n_output_channels(self):
        return len(self.dx_values)

    @property
    def metadata(self):
        return dict(format="reichardt-products-v2", dx_values=list(self.dx_values),
                    crop="common-valid-present-anchor")

    def __call__(self, xt):
        """Accept [2,X], [1,2,X], or [B,1,2,X]; return [C,1,X'] or batched."""
        batched = xt.ndim == 4
        if xt.ndim == 2:
            xt = xt[None, None]
        elif xt.ndim == 3 and xt.shape[0] == 1:
            xt = xt[None]
        if xt.ndim != 4 or xt.shape[1] != 1 or xt.shape[2] != 2:
            raise ValueError("Expected [2,X], [1,2,X], or [B,1,2,X]")
        if not xt.is_floating_point() or not torch.isfinite(xt).all():
            raise ValueError("XT input must contain finite floating-point intensities")
        x0 = max(0, max(self.dx_values))
        x1 = xt.shape[-1] + min(0, min(self.dx_values))
        if x0 >= x1:
            raise ValueError("Offsets leave no common valid XT region")
        now = xt[:, :, 1:2, x0:x1]
        responses = torch.cat([
            xt[:, :, 0:1, x0-dx:x1-dx] * now
            for dx in self.dx_values
        ], dim=1)
        if not torch.isfinite(responses).all():
            raise ValueError("Products overflowed; use a wider floating-point dtype")
        return responses if batched else responses[0]
