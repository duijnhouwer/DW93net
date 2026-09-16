"""Translation-only V1 -> MT -> MST inspired rate-network baseline.

Input [B,C,1,X] contains cached DoG ON_ON, OFF_OFF, ON_OFF, OFF_ON products, not biological
motion-energy measurements. At the default 7200 pixels / 120 degrees and spans
-60..60, C=484 and X=7062. Displacements are pixels per step, NOT deg/s.
Output [dx,dz,rot] uses scene distance per step for translation and degrees per
step for yaw. Translation-only training targets zero on the rotation output.
Heading=atan2(dx,dz), with 0 forward, -90 left and +90 right; speed per
step=hypot(dx,dz). Absolute scale depends on the training distance distribution.

Qian, Andersen & Adelson (1994), doi:10.1523/JNEUROSCI.14-12-07381.1994,
motivate local opponent suppression BEFORE larger MT pooling. Their suppression
stage is primarily MT, not V1. Matching +/- displacement magnitudes here is an
engineering approximation to their frequency/disparity-specific interactions.
Duffy & Wurtz (1991), doi:10.1152/jn.1991.65.6.1329, motivate a broad-field MST
population. Our learned units are not guaranteed to acquire biological tuning.

This is a 1D, two-frame functional abstraction: no disparity, orientation,
spatial-frequency bank, cortical magnification, recurrent circuit, or spiking.
RF sizes, population counts and inhibition constants are unvalidated starting
choices. Keep stimulus geometry and channel semantics fixed between splits.
"""

import torch
import torch.nn as nn
from features.reichardt import ReichardtBank


class LocalMotionOpponency(nn.Module):
    """Half-wave rectified, equal-speed opposite-direction suppression.

    Uses physical span labels, not channel index reversal. Zero displacement is
    passed through as stationary evidence, never subtracted from itself.
    """

    def __init__(self, dx_values):
        super().__init__()
        spans = ReichardtBank(dx_values).dx_values
        lookup = {span: i for i, span in enumerate(spans)}
        if any(-span not in lookup for span in spans):
            raise ValueError("Motion opponency requires paired +dX and -dX spans")
        self.register_buffer('opposite', torch.tensor([lookup[-s] for s in spans]))
        self.register_buffer('moving', torch.tensor([s != 0 for s in spans]).view(1, -1, 1))

    def forward(self, x):
        # Unit inhibition makes balanced opposite responses cancel locally.
        # Rectify BEFORE MT pooling so separated opposing motions can survive.
        # This deliberately tests Qian-style subtraction, not a fitted constant.
        return torch.relu(x - x.index_select(1, self.opposite) * self.moving)


class V1Motion(nn.Module):
    """Local motion populations with a same-channel suppressive surround."""

    def __init__(self):
        super().__init__()
        # ~0.25-degree anchor neighborhood; keep stride 1 until opponency has
        # acted. The actual detector also reaches back by up to 60 pixels.
        self.center = nn.AvgPool1d(15, stride=1, padding=7, count_include_pad=False)
        # ~1-degree surround scale, larger than center. This is a broad pool
        # including the center, not an anatomically fitted annular surround.
        self.surround = nn.AvgPool1d(61, stride=1, padding=30, count_include_pad=False)
        # Fixed semisaturation on normalized product responses: prevents division
        # by zero and preserves some amplitude dependence at low activity.
        self.semisaturation = 0.03

    def forward(self, x):
        center = self.center(x)
        # Two signed populations: local correlation and surround-suppressed correlation.
        # Keep ordinary responses too: spatially uniform motion is valid evidence,
        # and speed recovery must not depend exclusively on motion boundaries.
        contrast = center / (self.semisaturation + self.surround(x.abs()))
        # Absolute correlation supplies nonnegative suppressive drive; a negative
        # correlation must not turn the denominator into excitation or zero.
        # Preserve signs until direction opponency, where rates are rectified.
        return center, contrast


class PureTranslationNet(nn.Module):
    def __init__(self, dx_values=tuple(range(-60, 61))):
        super().__init__()
        # Explicit signed labels are required: channel count cannot identify
        # opponent pairs or distinguish custom/reordered detector banks.
        self.dx_values = ReichardtBank(dx_values).dx_values
        self.n_input_channels = 4 * len(self.dx_values)
        self.v1 = V1Motion()
        self.mt_opponency = LocalMotionOpponency(self.dx_values)
        self.mt = nn.Sequential(
            # Integrate rectified V1-sized subunits over 31 pixels, stride 8.
            # Opposite directions at different subunits remain represented.
            # No averaging across displacement channels: preserve speed diversity.
            nn.AvgPool1d(31, stride=8, padding=15, count_include_pad=False),
            # 96 local rate units combine direction/speed evidence from both V1
            # populations. A 1x1 kernel mixes channels without mixing positions.
            nn.Conv1d(2 * len(self.dx_values), 96, kernel_size=1),
            nn.ReLU(),  # Nonnegative rate abstraction; not a spiking-neuron model.
            # Nine taps spaced by 4 (~4.3 degrees between first/last centers)
            # compare broader neighborhoods with modest parameter/memory cost.
            nn.Conv1d(96, 96, kernel_size=9, dilation=4, padding=16),
            nn.ReLU(),  # Rectified MT population responses after learned integration.
            # Wider MT context: another 8.5 degrees of receptive-field support.
            # Stride 1 retains the positional map for subsequent MST integration.
            nn.Conv1d(96, 96, kernel_size=9, dilation=8, padding=32),
            nn.ReLU(),  # Different units can retain different local motion patterns.
        )
        self.mst = nn.Sequential(
            # 120 ordered bins (~1 degree at this FOV), NOT global average pooling.
            # This retains where opposite directions and speed gradients occur.
            nn.AdaptiveAvgPool1d(120),
            nn.Flatten(),  # Bin identity becomes a fixed visual-field input address.
            # 256 broad-field units can learn spatial motion templates, inspired
            # by MST's large RFs. Count is a capacity choice, not a neuron estimate.
            nn.Linear(96 * 120, 256),
            nn.ReLU(),  # Nonnegative MST-like population; tuning must be measured.
        )
        # A downstream linear decoder reads dx, dz, and yaw. Translation-only
        # training teaches the yaw output to remain zero.
        # No dropout/batch normalization: begin with deterministic rate responses;
        # regularization can be studied separately if held-out performance demands it.
        self.decoder = nn.Linear(256, 3)

    def forward(self, x):
        if (x.ndim != 4 or x.shape[1] != self.n_input_channels
                or x.shape[2] != 1 or x.shape[3] < 1):
            raise ValueError("Expected precomputed [B,C,1,X] Reichardt products")
        signed = self.signed_correlation(x.squeeze(2))
        local, contrast = self.v1(signed)
        # Negative correlation at +s is positive evidence at -s after subtraction.
        # Do not rectify cross-polarity subtraction before direction opponency.
        mt_input = torch.cat([self.mt_opponency(population)
                              for population in (local, contrast)], dim=1)
        return self.decoder(self.mst(self.mt(mt_input)))

    @staticmethod
    def signed_correlation(products):
        """Reconstruct L0*L1 from the four nonnegative polarity-pair products."""
        on_on, off_off, on_off, off_on = products.chunk(4, dim=1)
        return on_on + off_off - on_off - off_on
