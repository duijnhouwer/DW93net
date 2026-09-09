"""Deterministic products R(t,x) = I(t-dT,x-dX) * I(t,x)."""

from numbers import Integral
import torch


class ReichardtBank:
    """Signed dX in pixels, positive dT in frames; dT-major channel order.

    All channels share present-time anchors, cropped to the intersection of
    valid coordinates. Positive dX detects motion toward increasing X.
    Opposing directions remain separate products (no subtraction).
    """

    def __init__(self, dx_values=tuple(range(-10, 0)) + tuple(range(1, 11)),
                 dt_values=(1, 2)):
        self.dx_values = self._offsets(dx_values, "dX", positive=False)
        self.dt_values = self._offsets(dt_values, "dT", positive=True)
        self.channels = tuple((dx, dt) for dt in self.dt_values for dx in self.dx_values)

    @staticmethod
    def _offsets(values, name, positive):
        values = tuple(values)
        if (not values or any(isinstance(v, bool) or not isinstance(v, Integral)
                              or (positive and v <= 0) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError(f"{name} must contain distinct {'positive ' if positive else ''}integers")
        return tuple(int(v) for v in values)

    @property
    def n_output_channels(self):
        return len(self.channels)

    @property
    def metadata(self):
        return dict(format="reichardt-products-v1", dx_values=list(self.dx_values),
                    dt_values=list(self.dt_values), crop="common-valid-present-anchor")

    def __call__(self, xt):
        """Accept [T,X], [1,T,X], or [B,1,T,X]; return [C,T',X'] or batched."""
        batched = xt.ndim == 4
        if xt.ndim == 2:
            xt = xt[None, None]
        elif xt.ndim == 3 and xt.shape[0] == 1:
            xt = xt[None]
        if xt.ndim != 4 or xt.shape[1] != 1:
            raise ValueError("Expected [T,X], [1,T,X], or [B,1,T,X]")
        if not xt.is_floating_point() or not torch.isfinite(xt).all():
            raise ValueError("XT input must contain finite floating-point intensities")
        t0 = max(self.dt_values)
        x0 = max(0, max(self.dx_values))
        x1 = xt.shape[-1] + min(0, min(self.dx_values))
        if t0 >= xt.shape[-2] or x0 >= x1:
            raise ValueError("Offsets leave no common valid XT region")
        now = xt[:, :, t0:, x0:x1]
        responses = torch.cat([
            xt[:, :, t0-dt:xt.shape[-2]-dt, x0-dx:x1-dx] * now
            for dx, dt in self.channels
        ], dim=1)
        if not torch.isfinite(responses).all():
            raise ValueError("Products overflowed; use a wider floating-point dtype")
        return responses if batched else responses[0]
