# Exact-360° Circular Flow Plan

## Summary

Implement end-to-end circular spatial processing only when `fov_deg == 360`.
Smaller FOVs retain the existing nonperiodic behavior.

This intentionally does not model seam-crossing motion for near-panoramic FOVs
such as 359°. Subsequent nonpanoramic experiments will use FOVs below 350° so
that limitation is not practically relevant.

## Organization and implementation

- Introduce one boundary policy with `valid` and `circular` modes.
  - Derive `circular` only for exactly 360°.
  - Record the derived mode in dataset, DoG, correlation, and checkpoint
    metadata.
  - Pass it explicitly through every spatial processing stage. Individual
    feature and model layers must not independently inspect the FOV.
- Correct the 360° renderer:
  - Represent the panorama as the half-open interval `[-180°, 180°)` with
    exactly `W` bins.
  - Map angles with modulo arithmetic and a `W` scale, not `W - 1`.
  - Wrap dot footprints across the seam without duplicating an endpoint column.
  - Leave rendering below 360° unchanged.
- Put reusable circular padding and modular indexing in one small shared helper
  rather than scattering seam arithmetic throughout the project.
- Add circular behavior behind the existing processing interfaces:
  - DoG filtering uses circular padding and preserves all `W` columns.
  - Reichardt offsets use modular indexing and preserve all `W` columns.
  - V1/MT spatial pooling and convolution use circular padding.
  - Existing valid/cropped implementations remain unchanged.
- Reject 360° configurations with
  `apply_dog_filter_after_downsample = false`. Circular supersampled DoG
  filtering is deliberately out of scope.
- Make the changed data semantics explicit:
  - Dataset schema: `dog-contrast-bundles-v3`
  - DoG format: `dog-signed-contrast-v2`
  - Reichardt format: `signed-reichardt-correlation-v2`
  - Reject old caches with a clear instruction to regenerate them; do not add
    compatibility fallbacks.
- Add `recipe/trans_plus_rot_panoramic.toml` with:
  - `fov_deg = 360`
  - Effective width of 21,600 pixels at the existing 60 px/degree
  - `speed_range = [0.001, 0.006]`
  - `heading_deg_range = [-180, 180]`
  - `rot_deg_range = [-0.5, 0.5]`
  - Post-downsample DoG filtering enabled
  - Existing detector spans and training settings retained for comparison
  - Network name derived from the filename: `trans_plus_rot_panoramic`

## Test plan

- Renderer:
  - Verify `-180°` and `+180°` resolve to one seam bin.
  - Verify there is no duplicated endpoint column.
  - Verify dot footprints wrap without losing intensity.
  - Verify motion crossing the seam produces a short local displacement rather
    than a frame-wide jump.
- DoG:
  - Verify circular mode preserves width and is equivariant to horizontal
    rolls.
  - Verify a seam impulse has the same response as an interior impulse.
  - Verify valid mode retains its current shape and numerical behavior.
- Reichardt:
  - Verify positive and negative spans correlate across the seam.
  - Verify circular mode preserves width and is roll-equivariant.
  - Retain the existing no-wraparound assertion as a valid-mode test.
- Model:
  - Verify circular V1/MT local operators treat seam and interior neighborhoods
    equivalently.
  - Verify expected tensor shapes.
  - Verify nonpanoramic behavior remains unchanged.
- Integration:
  - Verify the boundary mode survives dataset loading, identity validation,
    checkpointing, resume, and evaluation.
  - Verify near-panoramic FOVs are not accidentally treated as circular.
  - Verify unsupported 360° high-resolution DoG configurations fail clearly.
  - Run a reduced end-to-end 360° generation/training/evaluation smoke test
    with an explicitly small test width.
  - Parse the production recipe and verify its ranges and derived 21,600-pixel
    width.
  - Run the full test suite.
  - Perform a one-batch GPU memory preflight at 21,600 pixels. If necessary,
    reduce batch size without changing the stimulus resolution.

## Assumptions and known limitation

- Circular behavior is selected only by exact `fov_deg == 360`.
- Near-panoramic views can physically cross the omitted angular gap when
  frame-to-frame motion exceeds that gap, but the implementation continues to
  treat their image edges as disconnected.
- Experiments following the panoramic test will use FOVs below 350° to avoid
  relying on that unsupported case.
- Adaptive pooling and the final decoder remain unchanged; only local spatial
  operators become circular.
- Old generated stimuli and checkpoints are regenerated rather than migrated.

