# Geometry-Aware Architecture with Learnable Motion Processing and Rotational Weight Sharing

Date: 2026-09-30

Branch: `Geometry-Aware-Architecture-Learnable-Motion-Processing-Rotational-Weight-Sharing`

## Status and purpose

This document preserves the architecture plan discussed with Jacob. It records
the selected direction, proposed starting settings, and the remaining decisions
needed before implementation. It is not a claim that the upgrade is implemented
or that every numerical and mathematical detail has been approved.

The goal is to estimate translation and rotation from motion evidence while:

- Making V1 processing and MT direction opponency learnable.
- Letting MT units learn different balances of spatial integration and suppression.
- Deriving spatial geometry from configurable angular resolution and FOV.
- Sharing computations across visual directions, including in MST.
- Allowing MST effective receptive fields to emerge without prescribing their size.

Accurate translation and rotation estimates do not require separate internal
populations for the two components. Distributed, mixed representations are valid.

## Current baseline

The implementation is in `src/models/flatland_flow.py`.

- Fixed signed DoG filtering precedes fixed Reichardt product computation.
- V1 produces a local average and a surround-normalized version of that average.
  Both operations currently have fixed parameters.
- Local opposite-direction subtraction and rectification occur at MT input,
  before broader spatial pooling. This is an MT-like mechanism, not V1 opponency.
- MT uses three learned convolutions producing 96 channels per spatial position.
  The same weights are used across positions. Channels are learned response
  functions, not predefined direction/speed categories.
- MST pools into 120 ordered bins, flattens them, and applies a dense layer with
  256 outputs. These dense weights are not shared across preferred directions.
- A linear decoder predicts sideways translation, forward translation, and yaw.

At the current 360-degree FOV and one-arcminute sampling, the input has 21,600
columns and 241 displacement channels (-120 through +120). The two V1 populations
produce 482 channels entering MT. MT has 2,700 spatial positions after stride-8
pooling. The fixed 120 MST bins cover about 3 degrees each, rather than the
approximately 1 degree implied by comments inherited from the 120-degree setup.

The original model is a useful trained baseline. Preserve its artifacts and
source revision for comparison. Saving this plan does not stop, restart, or
otherwise change a training process.

## 1. Explicit angular geometry and automatic sizing

### Selected behavior

- Support both full panoramas and partial FOVs.
- Make angular sampling configurable through `arcmin_per_pixel`, initially `1.0`.
- Derive stimulus width from `fov_deg * 60 / arcmin_per_pixel`.
- Require an integer width, allowing only numerical tolerance for floating-point
  arithmetic; do not silently round a materially incompatible configuration.
- Preserve physical angular sizes when resolution changes: dot width, DoG
  scales, detector motion range, and model spatial neighborhoods.
- Record actual sampling geometry in dataset metadata, including angular spacing,
  width, boundary policy, and the angle of the first retained column after cropping.
- Construct training and evaluation models from that metadata. A Reichardt map's
  width alone cannot determine its angular spacing or original FOV.
- Fix geometry for each training run. Evaluation and resume must match the run's
  geometry; mixed-FOV or mixed-resolution training in one model is out of scope.

### Implementation work

Replace the implicit `round(fov_deg * 60)` default in stimulus generation with
explicit resolution configuration. Resolve the current difference between
panoramic width-based projection and partial-view width-minus-one projection
using an explicitly documented angular grid convention.

Convert physical sizes to sampled quantities in one geometry utility, rather
than duplicating conversions in individual layers. Cropping changes the visible
angular interval and its origin, not the spacing between retained columns.

Use circular boundaries only for a complete 360-degree field. Partial views
have genuine boundaries; their visible edges must not wrap together. Missing
directions must not be interpreted as observations of zero motion.

Pixel displacement hypotheses remain discrete and explicitly labeled. Specify
and report the actual angular range and spacing obtained after discretization.
Do not introduce interpolation or fractional-displacement detectors implicitly.

The existing `x_supersampling` is an internal rendering factor, separate from
the angular sampling of the output columns.

## 2. Learnable V1 with identifiable displacement channels

### Selected design

Retain two populations: local signed responses and surround-normalized signed
responses. Use depthwise spatial convolutions so each displacement channel has
its own learned filter, shared across visual positions, without mixing its
displacement identity with other channels.

Constrain center and surround filter weights to be nonnegative and normalized.
Learn their spatial profiles, along with per-channel normalization strength and
positive semisaturation. Preserve signs until MT direction opponency.

A proposed functional form is:

\[
L_d(x) = (K^c_d * R_d)(x), \qquad
N_d(x) = \frac{L_d(x)}{\epsilon_d + \beta_d(K^s_d * |R_d|)(x)}.
\]

Here, both kernels are normalized nonnegative averages, `epsilon` is strictly
positive, and `beta` is nonnegative. This is an interpretable engineering model,
not a claim that the two branches are established biological cell classes.

### Proposed initialization

Start near the current uniform averages, with center and surround supports of
approximately 0.25 and 1 degree. Initialize semisaturation near `0.03` and surround
strength near the current value. Spatial profiles learn within explicit maximum
supports; this does not make the support size itself continuously trainable.

Keep full spatial sampling until local MT opponency has acted. Learnable V1
responses cannot be cached as fixed training inputs: they change with the weights.
Continue using the fixed DoG cache and online Reichardt computation initially.

## 3. Learnable local direction opponency within MT

Place opponency at the beginning of the MT-like stage, after V1 measurements and
before broader spatial pooling:

\[
o_d(x) = \operatorname{ReLU}\bigl(v_d(x) - \alpha_d v_{-d}(x)\bigr).
\]

Learn nonnegative inhibition strengths, initialized near 1. Pair channels using
their explicit displacement labels, and treat zero displacement separately.
The proposed implementation uses separate parameters for the two V1 populations.

Because inputs are signed correlations, this is an engineering approximation to
opponent motion processing, not literal subtraction between biological firing
rates. Tests must cover both positive and negative correlation inputs.

Do not include the earlier proposed unsuppressed bypass in the initial baseline.
It remains a possible ablation, not a requirement attributed to Qian and Andersen.

Local direction opponency and spatial motion contrast are different mechanisms:
the former compares directions within a local region; the latter compares motion
features across regions.

## 4. One MT population spanning suppression and integration

### Selected direction

Use one population with a learnable surround contribution per feature channel,
rather than preassigning separate integrative and suppressive branches:

\[
m_i(x) = \operatorname{ReLU}\bigl(C_i(x) + g_i S_i(x) + b_i\bigr).
\]

- `g_i < 0`: suppressive surround.
- `g_i > 0`: integrative surround.
- `g_i` near zero: predominantly center-driven response.

Learn center/surround spatial profiles and direction/speed tuning. Surround
responses and their aggregation weights must be nonnegative so the sign of
`g_i` remains interpretable. Each channel's parameters are shared across visual
positions; they are not independent coefficients for every column.

This coefficient is learned during training and fixed during inference.
Input-dependent modulation is a separate extension, not part of this initial plan.

### Proposed starting settings

Retain 96 MT feature channels as an explicit configurable capacity choice.
Increasing FOV increases the number of spatial responses, not automatically the
number of feature types. Initialize surround coefficients with small positive
and negative values rather than making every coefficient exactly zero, which
would initially block gradients to the corresponding surround filters.

Perform broader comparisons after spatial reduction for efficiency, while keeping
local direction opponency before that reduction. Specify spatial support and
sampling in angular units. Actual feature tuning must be measured after training.

## 5. Shared whole-field MST and a geometry-aware decoder

### Selected design

Replace the position-specific dense MST stage with learned broad-field templates
evaluated across angular positions using shared weights. For a full panorama,
assemblies at different preferred directions use rotated copies of the same
template. Multiple template types can learn different motion patterns.

Each MST unit may access the entire available FOV. Do not impose a smaller RF,
a localization penalty, or a prescribed effective RF width. Learned weights may
produce localized, distributed, or whole-field sensitivity. Template support is
the maximum available support, not a claim about the effective RF after training.

For partial views, use nonwrapping boundaries and explicit visibility information.
Full rotational symmetry is a property of the panoramic case, not of a fixed
partial-view window.

### Decoder requirements

Preserve the output convention `[dx, dz, yaw]`. Shifting a full panorama should
rotate the estimated translation vector consistently with the coordinate
convention, while leaving its magnitude and yaw unchanged.

Use angularly indexed MST responses with a symmetry-respecting readout. Sine/cosine
projections for translation and a rotation-invariant aggregation for yaw are the
proposed starting approach. Do not append an unrestricted position-specific dense
decoder that silently removes the intended symmetry.

Specify the discrete shifts for which equivariance is guaranteed. Striding and
angular binning generally prevent exact equivariance to every original column
shift; test and document this limitation rather than claiming continuous symmetry.

## 6. Configuration, checkpoints, and implementation sequence

Expose an architecture configuration containing angular scales, population sizes,
and sampling choices. Save both requested settings and resolved discrete geometry
with checkpoints. Reconstruct the exact same architecture in training, evaluation,
and resume; never create or resize trainable layers inside `forward()`.

Implement in this order:

1. Angular geometry utilities, generation configuration, metadata, and contracts.
2. Learnable V1 and MT input opponency.
3. MT spatial integration/suppression population.
4. Shared full-support MST templates and the decoder.
5. Checkpoint reconstruction, evaluation, and reporting.
6. Mechanism tests, production-size GPU preflight, and controlled training comparisons.

Treat this as a new architecture trained from scratch. Clearly reject incompatible
checkpoints. Preserve all baseline datasets and training artifacts; do not add
silent legacy guessing or overwrite prior results. Changes to sampling conventions
or preprocessing require explicit dataset compatibility decisions and potentially
new caches. Reusing artifacts must be based on verified contracts, not filenames.

## 7. Validation and acceptance

- Verify width, angular spacing, origin, cropping, and boundary calculations for
  full and partial FOVs and more than one angular resolution.
- Confirm requested physical sizes remain consistent across resolutions within
  declared discretization error, including dots, DoG, and detector spans.
- Test circular seams and partial-view edges; missing data must not be treated as
  observed zero motion or connected across a partial-view gap.
- Check gradients reach V1 kernels, normalization parameters, opponent strengths,
  MT surround profiles and coefficients, and MST templates.
- Test positive normalization denominators, finite outputs/gradients, signed
  correlation behavior, zero displacement, and supported reordered detector banks.
- Use controlled constant-motion and locally varying-motion inputs to probe
  integration, suppression, and sensitivity to small speed differences.
- Verify full-panorama shift behavior for intermediate maps and final outputs at
  the supported shift increments; test the translation sign convention explicitly.
- Verify save/load and resume reconstruction, and useful errors for incompatible
  geometry or architecture.
- Run production-width forward/backward GPU preflights in AMP and the required
  float32 fallback, measuring memory and throughput before choosing batch size.
- Compare baseline and upgraded models on the same scientific stimulus
  distributions, accounting explicitly for any rendering changes.
- Measure rotation-only, translation-only, and mixed-motion errors; test cross-talk
  by holding one component fixed while varying the other.
- Probe learned direction/speed tuning and effective RFs with localized stimuli.
  Inspect surround coefficients, but do not infer necessity from weights alone.
- Distinguish post-training ablations (reliance) from retrained comparisons
  (ability to compensate), using multiple seeds and comparable capacity where feasible.

No separate internal translation/rotation populations are required for success.

## Remaining decisions before implementation

The high-level choices above are settled or labeled as proposals. The following
details were not finalized in the discussion and must not be presented as already
approved:

- Exact angular grid origin/convention, discretization rules for kernel support,
  detector spans, dot footprints, and changes to pixel-based recipe fields.
- V1 parameter transforms, numerical floors, exact initial kernel widths, and
  whether opposite directions share any reflected parameters.
- Exact MT center/surround construction, number of spatial stages, support sizes,
  stride/anti-aliasing behavior, and whether unconstrained signed spatial paths
  could undermine the interpretation of the explicit surround coefficient.
- MST angular sampling density, number of template types, template operator and
  efficient implementation, and precise partial-view visibility handling.
- Decoder equations, normalization, and the exact discrete symmetry guarantees.
- Final architecture/cache/checkpoint schema versions, experiment configuration,
  capacity comparisons, and acceptance thresholds for accuracy and runtime.

Resolve these choices into a concrete implementation specification before coding;
do not revert to the superseded fixed-120-bin dense MST or mandatory two-branch MT
design without discussing the change.
