# Handoff: 360° Panoramic Flow

## Current state

The exact-360° plan in `PLAN-PANORAMIQUE.md` is implemented on
`panoramica-flow360`.

- `src/features/spatial.py` owns the boundary resolver, modular indexing,
  circular padding, and circular local pooling/convolution helpers.
- `StimConfig` derives `circular` only for exactly 360°. Other FOVs use `valid`.
  Features and model layers receive this mode explicitly instead of inspecting FOV.
- The renderer uses a half-open `[-180°, 180°)` grid scaled by W, canonicalizes
  both seam endpoints, and wraps dot footprints. Nonpanoramic rendering is unchanged.
- Circular DoG and Reichardt preserve all W columns. V1/MT local spatial
  operators wrap; adaptive pooling and the decoder are unchanged.
- Boundary metadata is recorded in stimulus geometry, dataset identity and
  contract, DoG, and correlations. Training/resume and evaluation validate it.
- Schema versions: dataset `dog-contrast-bundles-v3`, DoG
  `dog-signed-contrast-v2`, correlation `signed-reichardt-correlation-v2`,
  checkpoint `dw93net-checkpoint-v4`, managed dataset identity version 6.
  Obsolete caches/checkpoints require regeneration. Existing obsolete managed
  caches are rejected without deleting their artifacts.
- The renamed setting is `apply_dog_filter_after_downsample`. Panoramas require
  `true`; smaller FOVs retain both filtering orders.

## Active experiment

`recipe/trans_plus_rot_panoramic.toml` is the sole active recipe. The original
is preserved at `recipe/archive/rot_only.toml`; the runner still requires exactly
one directly contained `recipe/*.toml` file.

The panoramic recipe uses 360° at the existing 60 px/degree (21,600 columns),
speed `[0.001, 0.006]`, heading `[-180, 180]`, and rotation `[-0.5, 0.5]`.
Detector spans remain -120..120. Training settings, including batch size 64,
are retained. Production generation/training has not been started.

## Verification

`tests/test_panorama.py` covers seam bins and wrapped footprints, roll
consistency, periodic reference calculations, narrow circular padding,
metadata rejection, production recipe parsing, and a reduced 128-column
360° generation/training/resume/validation/test smoke run.

The full test suite passes. Multiprocessing tests need permission to create
local sockets outside the desktop sandbox.

Both one-batch GPU preflights passed on the 12 GB RTX 3080 Ti at width 21,600,
241 channels, batch size 64: AMP and float32 fallback each peaked at 9,242 MiB
allocated / 9,274 MiB reserved. Free VRAM before each run was about 9,500 MiB,
so headroom is narrow. See `benchmarks/panorama-20260929/REPORT.md` and the
repeatable `scripts/preflight_panorama.py`. Reduce batch size explicitly if a
longer run is memory-bound; do not reduce scientific resolution.

## Known limitation

Near-panoramic FOVs such as 359° retain disconnected edges even when motion
crosses the omitted angular gap. Subsequent nonpanoramic experiments will use
FOVs below 350°. Circular high-resolution/pre-downsample DoG is unsupported.
