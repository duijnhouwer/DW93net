# Handoff: 360° Panoramic Flow

## Current state

- Repository: `/home/jacob/Documents/dw93net`
- Branch: `panoramica-flow360`
- The working tree was clean before these two documentation files were added.
- No panorama implementation has been performed yet.
- The accepted specification is in `PLAN-PANORAMIQUE.md`.

The user is concerned about panorama support introducing scattered,
bug-prone code. The main architectural requirement is therefore to derive one
boundary mode and propagate it through the pipeline, with shared circular
operations and no repeated `fov_deg == 360` checks in individual layers.

## Decisions already made

- Support circular processing end-to-end: renderer, DoG, Reichardt, and V1/MT
  spatial operations.
- Select circular mode only for exact 360° FOV. All smaller FOVs retain current
  valid/nonperiodic behavior.
- Accept the known limitation for near-panoramic FOVs: a 359° view can still
  have physical seam crossings when a frame step exceeds the missing 1°.
  Testing after the panoramic experiment will use FOVs below 350°.
- Use a half-open angular grid `[-180°, 180°)` containing exactly `W` columns.
  `-180°` and `+180°` must address the same bin; there must be no duplicated
  0°/360° column.
- Reject panoramic use of the pre-downsample/high-resolution DoG path
  (`apply_dog_filter_after_downsample = false`) instead of implementing a second
  circular filtering path.
- Breaking cache metadata changes are acceptable during this exploratory phase;
  regenerate old data instead of adding compatibility code.
- The experiment recipe is named `trans_plus_rot_panoramic` and uses:
  - FOV 360° and the existing 60 px/degree, yielding width 21,600
  - Speed `[0.001, 0.006]`
  - Heading `[-180, 180]`
  - Rotation `[-0.5, 0.5]`
  - Existing detector span and training settings

## Repository findings

- `src/stimuli/generate_flow_xt_data.py`
  - Uses `atan2` and accepts FOV up to 360°.
  - At 360°, it maps exact `theta == -pi` to `+pi` and maps normalized position
    with `(render_wid_px - 1)`.
  - Dot footprints are clipped at the image edges rather than wrapped.
  - A controlled one-dot check at width 360 showed `-180°`, `+179.9°`, and
    `+180°` landing at pixel 358, while `-179.9°` disappeared. Thus Flow-XT is
    not currently seam-correct despite using world angles.
- `src/features/dog_motion.py`
  - `DoGFilter` performs valid convolution and shrinks width by twice the
    kernel radius.
  - `SupersampledDoG` also implements a nonperiodic valid-style crop.
  - Metadata currently uses `dog-signed-contrast-v1` and
    `boundary="valid"`.
- `src/features/reichardt.py`
  - Crops signals to common valid present-frame anchors, so output width shrinks
    with detector displacement.
  - Metadata currently uses `signed-reichardt-correlation-v1` and
    `crop="common-valid-present-anchor"`.
  - `tests/test_reichardt.py::test_no_wraparound` explicitly asserts current
    nonwrapping behavior; preserve this test for valid mode and add circular
    tests separately.
- `src/models/flatland_flow.py`
  - `V1Motion` uses padded `AvgPool1d` layers.
  - The MT stack contains another padded average pool and spatial `Conv1d`
    layers with ordinary zero padding.
  - These local operators need circular padding in panoramic mode. Adaptive
    pooling and the decoder should remain unchanged.
- `src/stimuli/metadata.py` currently defines
  `DATA_SCHEMA = 'dog-contrast-bundles-v2'`.
- Training and evaluation currently construct `ReichardtBank(dx_values)` and
  `FlatlandFlowNet(dx_values)` without a boundary argument. Dataset metadata is
  the natural source from which to pass the resolved mode.
- `DoGDataset` reconstructs and validates the DoG filter from metadata, so the
  new boundary field must participate in this validation.
- The default stimulus width is `round(fov_deg * 60)`, making a 360° production
  stimulus 21,600 pixels wide.
- The current recipe is `recipe/rot_only.toml`; its commented values preserve
  the desired original joint speed and heading ranges. Its batch size is 64,
  which should be measured rather than assumed safe at panoramic width.

## Recommended implementation sequence

1. Add the boundary-policy resolver and shared circular padding/index helpers,
   with focused unit tests.
2. Fix renderer binning and wrapped dot rasterization, then lock the seam
   convention with tests before touching later stages.
3. Implement circular DoG and Reichardt modes while leaving valid paths intact.
4. Propagate boundary metadata through dataset identity, loading, training,
   evaluation, and checkpoints; bump the formats named in the plan.
5. Add circular local pooling/convolution behavior to the model.
6. Add the panoramic recipe and configuration rejection for unsupported
   high-resolution DoG.
7. Run focused tests, the full suite, a tiny end-to-end smoke test, and finally
   a one-batch GPU memory preflight at production width.

Do not silently lower the 60 px/degree scientific resolution if the preflight
is memory-bound. Reduce batch size explicitly and report the measured reason.

