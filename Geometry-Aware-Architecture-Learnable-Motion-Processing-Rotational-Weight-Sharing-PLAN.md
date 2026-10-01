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
- Learning each directional V1, MT, and MST template once and using it in left-handed and
  right-handed variants with tied parameters.
- Allowing MST effective receptive fields to emerge without prescribing their size.

Accurate translation and rotation estimates do not require separate internal
populations for the two components. Distributed, mixed representations are valid.
The decoder learns the decomposition; do not prescribe sine/cosine translation
projections or a dedicated invariant yaw aggregation.

Jacob presents training stimuli in both left-handed and right-handed versions.
Use this symmetry as an architectural prior: learn one receptive-field template
per handed pair and reuse it in mirrored form, so examples of either handedness
train the same parameters. This reduces independent paired-template parameters
by half relative to separately parameterized variants and encourages consistent
responses to the two stimulus versions. Symmetric training data alone would not
guarantee that independently learned receptive fields become exact mirrors.

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
- Resolve each model's spatial architecture automatically from configured FOV,
  sampling, and preprocessing crops at construction time. Keep 48 MT and 60 MST
  template pairs per position; derive spatial-position counts, MST support, and
  position-dependent decoder dimensions from the resolved geometry. Do not embed
  partial views into a padded 360-degree architecture or add mask input channels.

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

### Selected grid and resampling specification

Use cell-centered angular grids over the nominal interval `[-F/2, F/2)`:
`theta[i] = -F/2 + (i + 0.5) * delta`, with `N = F/delta` samples.
Apply this convention to the stimulus, 0.25-degree MT grid, and 1-degree MST grid.
A panorama has no duplicated endpoint. Reflection about zero reverses the sample
indices exactly. Cropping records the surviving original indices and origin;
it must not recenter the grid. For partial views, retain only output-grid cells
whose full angular intervals lie within the retained input interval. Trim the
others rather than restoring cropped columns or padding to the original FOV.
Record the retained indices, new first sample center, and actual count at every
stage. Generate positions from the original FOV-aligned grid, so cropping does
not introduce a new sampling phase. Symmetric preprocessing crops must produce
symmetrically trimmed grids. Reject geometries with no retained output cells.

Require integer sample counts and integer reduction factors for this initial
implementation, within `abs(value - round(value)) <= 1e-8` in sample units.
Reject incompatible geometry with the requested and nearest supported values;
do not silently round or upsample. This means the selected architecture initially
requires FOVs divisible by 1 degree and input spacing that divides 0.25 degree.
These are explicit implementation constraints, not claims about visual geometry.

For reduction factor `q`, output sample `j` lies at input index
`q*j + (q-1)/2`. Thus the 15-fold input-to-MT reduction uses indices `15*j+7`,
and the 4-fold MT-to-MST reduction is centered at `4*j+1.5`. Evaluate the filter
at these exact centers; do not approximate the latter by selecting every fourth
input sample. This fixes the half-sample phase and preserves reflection symmetry.

Use a fixed nonnegative Gaussian anti-alias filter with
`sigma_filter_deg = 0.75 * output_spacing_deg`, truncated at radius `4*sigma`.
At each output center, weight contributing input samples by
`h = exp(-0.5 * (angular_distance/sigma_filter_deg)**2)` and normalize the sampled
weights to sum to one. Construct coefficients in float64 and store float32
buffers. At the selected resolutions, the input-to-MT filter has sigma 0.1875
degrees and 91 taps; MT-to-MST has sigma 0.75 degrees and 24 half-phase taps.
Use circular distance/wrapping only for panoramas. Precompute the phases and
indices; this filters feature maps, not fractional Reichardt displacement channels.

This is a deliberately simple smoothing/alias-reduction tradeoff: the sampled
filters have about 0.06235 amplitude gain (-24.1 dB) at their respective output
Nyquist frequencies, not ideal brick-wall rejection. It preserves nonnegative
motion evidence and avoids sinc ringing, but attenuates fine structure. The filter
is separate from the Gaussian MT initialization envelope and does not constrain
learned MT spatial profiles. Record it as part of the model geometry contract.

### Selected FOV-dependent architecture and partial-view boundaries

Construct a separate resolved architecture for each configured FOV and retained
geometry. Before preprocessing crops, 120 degrees gives 480 MT positions at
0.25 degrees and 120 MST positions at 1 degree, versus 1,440 and 360 for a panorama.
Use the actual retained counts after cropping/resampling to allocate MST templates
and the position-aware decoder. V1 and MT have fewer spatial responses for smaller
fields, while feature-channel counts and local template parameters stay unchanged.
No visibility/coverage tensors or learned mask channels are part of the design.

At partial-view boundaries, omit samples outside the retained array. For positive
V1 pooling and resampling kernels, divide the weighted sum over available samples
by the sum of their available kernel weights. Reject an empty support rather than
creating a fictitious observation. Preserve constant signals at the edges. Never
reflect, repeat, or wrap content into the missing region.

For learned signed MT/MST/decoder correlations, omit unavailable terms without
renormalizing the signed weights. Ordinary zero padding may implement this finite
sum; the padding is an implementation of absent terms at a known fixed boundary,
not additional observed positions or an assertion of measured zero motion. Keep
only the selected in-field output positions. Do not pad the feature map or decoder
to a full-circle shape. Edge behavior is fixed by each model's geometry.

The initial channel counts remain 482 V1 and 96 MT for the stated detector bank
at every FOV. Edge normalization changes positive filters near boundaries;
frequency-response tests apply to full-support locations, with separate edge tests.

## 2. Learnable V1 with identifiable displacement channels

### Selected design

Retain two populations: local signed responses and surround-normalized signed
responses. Use depthwise spatial convolutions to preserve each displacement
channel's identity, with filters shared across visual positions. Opposite
displacement channels use left-handed and right-handed versions of the same
learned base template, rather than independently learned filters.

Constrain center and surround filter weights to be nonnegative and normalized.
Learn their spatial profiles, normalization strength, and positive semisaturation.
Share normalization strength and semisaturation within each handed pair. Preserve
signs until MT direction opponency.

For each nonzero displacement magnitude, store one learned center kernel and one
learned surround kernel. Generate their opposite-handed versions by reversing
spatial offsets: `K^c_{-d}(u) = K^c_d(-u)` and likewise for `K^s`. Both variants
contribute gradients to the same underlying parameters. The base kernels need
not themselves be symmetric. Pair channels using explicit displacement labels;
retain both signed displacement responses and both V1 populations.

Zero displacement has no distinct opposite-displacement partner. Learn half of
each zero-displacement kernel plus its center, mirror its logits, then normalize
the full kernel. This enforces symmetry for that channel only; directional base
kernels remain unrestricted apart from nonnegativity and normalization.

The selected functional form is:

\[
L_d(x) = (K^c_d * R_d)(x), \qquad
N_d(x) = \frac{L_d(x)}{\epsilon_d + \beta_d(K^s_d * |R_d|)(x)}.
\]

Here, both kernels are normalized nonnegative averages, `epsilon` is strictly
positive, and `beta` is nonnegative. This is an interpretable engineering model,
not a claim that the two branches are established biological cell classes.

### Selected initialization

Start near the current uniform averages, with center and surround supports of
approximately 0.25 and 1 degree. Initialize semisaturation near `0.03` and surround
strength near the current value. Spatial profiles learn within explicit maximum
supports; this does not make the support size itself continuously trainable.

Resolve each requested support D to odd kernel size `2*floor(D/(2*delta))+1`,
allowing `1e-8` tolerance before flooring a mathematically integral radius. The
0.25/1-degree requests give 15/61 taps at one arcminute. Record the actual span
between sample centers (14/60 and 60/60 degrees here). Initialize kernel logits
to zero for uniform profiles and use softmax over the full spatial kernel.
Use `epsilon = 1e-4 + softplus(raw_epsilon)` initialized to 0.03 and
`beta = softplus(raw_beta)` initialized to 1. Compute normalization in float32,
including under AMP; these are engineering stability defaults.

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
For each nonzero displacement magnitude, learn one inhibition strength shared by
both directions: `alpha(+d) = alpha(-d)`. Both directions contribute gradients to
that parameter; do not learn independent left/right inhibition strengths.
Use separate opponent parameters for the two V1 populations.
Use `alpha = softplus(raw_alpha)` initialized to 1. Zero displacement passes
through ReLU without self-subtraction and has no learned opponent strength.

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

Use learned signed weights jointly over spatial offsets and opponent V1 channels,
followed by rectification. Do not prescribe Gaussian profiles or designated
center and surround pathways:

\[
m_i(x) = \operatorname{ReLU}\left(\sum_{j,k} W_{i,j,k}o_j(x+u_k)+b_i\right).
\]

Here, `j` indexes the incoming opponent-motion channels from both V1 populations,
and `k` indexes spatial subfield samples at offsets `u_k`. Each template can learn
different displacement preferences at different offsets, asymmetric profiles,
multiple sensitivity peaks, and excitatory or suppressive contributions. Share
the same template across evaluation positions.

Integration and suppression emerge from learned spatial/channel weights; there
is no explicit surround coefficient `g_i`. The three-Gaussian bank and its
2/8/18-degree FWHM initializations are superseded. Richert, Albright, and Krekelberg
(2013), listed in REFERENCES.md, motivate allowing spatially heterogeneous MT
receptive fields; this flexibility does not guarantee biological tuning.

These operations provide additive/subtractive interactions, not divisive
normalization. Start without additional MT divisive normalization; retain the V1
normalized branch. MT normalization is a possible later controlled experiment.

### Selected left/right template pairing

Each MT template occurs in a left-handed and a right-handed variant. Store and
optimize one base template; derive its mirrored partner from those same parameters
on every use. Gradients from both variants update the base template. Do not train
two independent copies, periodically synchronize them, or require the base
template itself to be mirror-symmetric.

For displacement-labeled inputs, mirroring reverses spatial kernel offsets and
swaps input channels labeled `+d` and `-d`, keeping the two V1 populations distinct.
For subsequent feature inputs, it reverses spatial offsets and swaps the paired
left/right feature channels. Zero displacement maps to itself. Use explicit
channel-pair mappings rather than assuming channel order.

Apply this pairing to the complete spatial/channel template, reversing spatial
offsets and permuting opposite-direction input channels, with a shared bias.
Both response channels remain present. This halves independent template parameters relative to
learning the same number of response channels independently; it does not halve
activation storage or necessarily halve computation.

These are handed variants of learned response functions, not predefined pure
translation or rotation units. Pairing MT/MST weights alone does not guarantee
reflection equivariance of the entire network: V1, input opponency, sampling,
boundaries, and the decoder also affect that guarantee. V1 handedness pairing and
tied opposite-direction opponent strengths are selected in Sections 2 and 3.

### Proposed starting settings

Retain 96 MT feature channels as an explicit configurable capacity choice.
With paired variants, this means 48 independently learned base templates producing
96 response channels, not 96 base templates producing 192 channels.
Increasing FOV increases the number of spatial responses, not automatically the
number of feature types.

Selected spatial settings:

- `max_mt_rf_diam_deg = 10`
- `mt_rf_nr_subfields = 41`

Use uniformly spaced sample centers including both endpoints of the support:
`u_k = -max_mt_rf_diam_deg/2 + k * max_mt_rf_diam_deg/(mt_rf_nr_subfields - 1)`.
The selected settings give offsets from -5 to +5 degrees inclusive, 0.25-degree
spacing, and a sample at zero. There is one learned weight per input channel per
sample per base template: `48 * 482 * 41 = 948,576` kernel weights for the stated
detector bank, plus 48 shared biases. Subfields here mean spatial sampling sites,
not additional independent output units.

### Selected MT initialization

Initialize each base template with independent signed random weights across
input channels and all 41 spatial subfields, multiplied by a centered Gaussian
envelope:

`W_initial[i,j,k] = scale * random_signed[i,j,k] * exp(-u_k**2 / (2*sigma**2))`

Draw each `random_signed[i,j,k]` independently from `Uniform(-1, 1)` for the base
templates. Account for input count and the actual sampled envelope in the overall
scale. For the selected ReLU, use the starting rule
`scale = sqrt(6 / (C * sum_k envelope[k]**2))`, where `C` is the number of incoming
opponent V1 channels (482 for the stated detector bank). Since a uniform draw has
variance 1/3, this gives an expected sum of squared incoming weights of 2 per
base template. It is a ReLU-oriented initialization heuristic, not a guarantee of
equal activation variance for correlated V1 inputs; check activation and gradient
scales during preflight. It does not impose a fixed absolute integral or normalize
each realized template to an exact norm.

Use the requested configuration name
`mt_rf_initial_gaussian_envelople_sigma = max_mt_rf_diam_deg / 4`, in degrees.
At the selected 10-degree support, sigma is 2.5 degrees. This is Gaussian sigma,
not FWHM. The envelope has peak one at the center and reduces the expected
magnitude of weights toward the edges; it does not guarantee that each central
random weight exceeds each edge weight or that every template's magnitude
centroid is exactly zero.

Do not mirror 21 random values to initialize 41 spatial weights. The individual
base template may be spatially asymmetric from initialization. Generate its
opposite-handed variant using the existing tied spatial reflection and
input-channel permutation, rather than drawing another random initialization.
This guarantees the mirror relationship between template variants, not symmetry
of each individual template or of the unconstrained decoder.

The Gaussian is an initialization envelope only. Store and train the resulting
spatial/channel weights freely; do not keep multiplying by the envelope during
training or parameterize the learned RF as a Gaussian. Initialize all biases to zero.

### Sampling and effective support

The diameter specifies the span between extreme input sample centers. Upstream
V1/detector support and any resampling filter can make the receptive field with
respect to the original stimulus larger. Effective sensitivity within the selected
support is learned, not prescribed. Alignment to the incoming feature grid and
any required filtering/resampling must be explicit.

Selected MT evaluation spacing is 0.25 degrees, giving 1,440 positions on a full
360-degree panorama. After local opponency at full input resolution, apply
anti-alias filtering and resample to a 0.25-degree feature grid. The 41-subfield
MT kernel then uses adjacent grid samples and is evaluated with stride 1 on that
grid. At one arcminute input sampling, the reduction factor is 15. Derive the
sampling relationship from geometry at other input resolutions; do not hardcode
15 or silently round incompatible grids. Use the filter, phase, and boundary
rules specified in Section 1 (without mask channels).

Perform broader comparisons after spatial reduction for efficiency, while keeping
local direction opponency before that reduction. Specify spatial support and
sampling in angular units. Actual feature tuning must be measured after training.

## 5. Shared whole-field MST and a geometry-aware decoder

### Selected design

Replace the position-specific dense MST stage with learned broad-field templates
evaluated across angular positions using shared weights. For a full panorama,
assemblies at different preferred directions use rotated copies of the same
template. Multiple template types can learn different motion patterns.

Start with 60 learned MST base templates, each supplying a left/right pair:
120 response channels per evaluated angular position. Keep this count configurable.
For a full-support template with independent weights over 96 MT input channels
and M angular samples, the bank has `60 * 96 * M` kernel parameters plus 60 shared
biases. Evaluating shifted or mirrored copies does not add template parameters.
Selected MST template sampling and bank evaluation spacing are both 1 degree.
Apply anti-alias filtering to the 0.25-degree MT output map and downsample by 4
onto an aligned 1-degree grid before MST. For a full panorama, each template has
M=360 spatial samples and the bank is evaluated at 360 angular positions, using
stride 1 on that grid. Thus the MST bank has 2,073,600 kernel weights plus 60 shared
biases, and produces a `[batch, 120, 360]` response map. The template sample count
and evaluation-position count are distinct concepts even though both are 360 here.
Use angular spacing and retained geometry to resolve counts for partial views;
do not hardcode 360 for every FOV.
Decoder parameter count and evaluation compute must be budgeted separately.
Use the filtering/alignment rules in Section 1.

### Selected MST operator and initialization

Use shared full-support spatial cross-correlation, a shared bias per handed pair,
then ReLU. For a panorama of N samples, define
`z_i[n] = sum_{c,k=0..N-1} W_i[c,k] * x_c[(n+k) mod N] + b_i`.
Here k is a relative cyclic offset, not an absolute viewing angle. Its mirror is
`W_mirror[c,k] = W_base[paired_channel(c), (-k) mod N]`.
For even N=360 this modular permutation is not plain array reversal; zero offset
and the antipodal offset map to themselves. Verify the indexing directly.

Use an FFT implementation of circular cross-correlation in float32 for panoramic
production runs (also under AMP), with a direct implementation as the correctness
reference on small grids. Store spatial weights and differentiate through the FFT;
do not store independent spectral or mirrored trainable copies. Benchmark against
direct correlation before finalizing the runtime backend.
For the stated real-valued correlation convention, use the inverse transform of
`sum_c FFT(x_c) * conjugate(FFT(W_i,c))`, with matching transform normalization.

For a partial window with N retained samples, use nonwrapping correlation with
relative offsets `-(N-1)..+(N-1)`. This larger support
lets every evaluated unit access all observed positions even near the window edge.
A centered N-tap kernel would not provide that access at every position. Thus the
partial-view kernel has `2*N-1` taps and a different resolved parameter count from
the N-tap circular operator. Omit out-of-field terms using ordinary padded linear
correlation, returning only N in-field positions and using no mask channels.
Reversal of this odd offset array implements spatial mirroring. Reject windows
with no surviving valid evidence.

Initialize MST base motion-feature weights independently with uniform
`[-sqrt(6/fan_in), +sqrt(6/fan_in)]`, where `fan_in` counts motion input channels
times spatial taps. Use no Gaussian envelope or localization prior for MST.
Derive handed variants from the base weights and initialize shared biases to zero.

Each MST base template also has left-handed and right-handed variants sharing
one parameter set. Generate its mirror by reversing spatial offsets and swapping
the paired MT input channels, with a shared bias. Both variants train the same
base template and may represent mixed translation/rotation evidence.

Handedness pairing and sharing across angular positions are distinct: the former
reflects a template, while the latter shifts it around the visual field. Retain
the existing angular-position sharing alongside the selected handedness pairing.
Report base-template counts separately from the number of handed response channels
and angular positions.

Each MST unit may access the entire available FOV. Do not impose a smaller RF,
a localization penalty, or a prescribed effective RF width. Learned weights may
produce localized, distributed, or whole-field sensitivity. Template support is
the maximum available support, not a claim about the effective RF after training.

For partial views, construct templates for the retained field with nonwrapping boundaries.
Full rotational symmetry is a property of the panoramic case, not of a fixed
partial-view window.

### Decoder requirements

Preserve the output convention `[dx, dz, yaw]`. The selected decoder is:

`angularly indexed mixed MST responses -> circular convolutions + nonlinearities -> position-aware joint readout -> [dx, dz, yaw]`

Preserve the MST channel-by-angle map through shared learned spatial convolutions
and nonlinearities. For a full panorama, these convolutions use circular
boundaries so the first and last angular positions are neighbors. They mix both
MST feature channels and angular neighborhoods. Hidden channel counts are feature
capacities, not angular positions.

After this circular nonlinear processing, use a learned joint readout ending in
a three-output linear layer. Retain angular position information in the readout,
for example by flattening only at this final stage; do not globally average the
angular axis and thereby discard the reference needed for translation direction.
The exact readout arrangement and capacity remain to be specified.

Translation and rotation evidence can interact throughout the shared decoder,
allowing it to learn rotation compensation when estimating translation.
Separate output coordinates do not imply independent estimation processes. Do not
require MST alone to resolve this interaction before a purely linear readout.
The earlier prohibition on a position-specific dense readout is withdrawn.
The number of convolutional layers, feature counts, angular kernel supports,
strides, activations, and any final dense hidden layers remain to be specified.
The earlier suggested dense-only `256 -> 128 -> 3` decoder is not selected.

Initialize decoder convolutional/dense layers followed by ReLU with uniform
`[-sqrt(6/fan_in), +sqrt(6/fan_in)]` and all biases zero. Initialize the final
linear output layer with Xavier uniform bounds `sqrt(6/(fan_in+fan_out))`.
If a different hidden activation is later selected, revise its initialization
gain explicitly.

For partial views, use nonwrapping processing on the resolved smaller map instead of circular
padding; the decoder must not connect the two visible edges across missing space.

If this joint decoder does not meet the evaluation criteria, consider explicit
translation and rotation decoder stages as a later experiment. Such a design could
feed a rotation estimate together with MST evidence into the translation stage.
This is a fallback to investigate, not part of the initial architecture or a
requirement to supervise separate internal translation/rotation representations.

Do not hardcode sine/cosine projections for translation, an invariant aggregation
specifically for yaw, or a sum/difference output rule as part of the selected
design. Left/right template pairing constrains MT/MST parameter sharing, not the
translation/rotation decomposition in the readout.

As a desired behavior to measure, shifting a full panorama should
rotate the estimated translation vector consistently with the coordinate
convention, while leaving its magnitude and yaw unchanged.
Reflecting the stimulus should change `[dx, dz, yaw]` to `[-dx, dz, -yaw]`.
These output behaviors are not guaranteed by paired/shared MT and MST templates
or by circular nonlinear decoder layers when the final readout is unconstrained.
Circular processing preserves angular adjacency and can preserve intermediate
shift equivariance, but does not alone guarantee rotation-consistent outputs.
Symmetry augmentation with transformed targets
is a proposed training approach, with explicit evaluation of residual errors.

Specify the discrete shifts for which intermediate-map equivariance is guaranteed
and the conditions for reflected maps to exchange handed channels. Striding and
angular binning generally prevent exact equivariance to every original column
shift; test and document this limitation rather than claiming continuous symmetry.

## 6. Configuration, checkpoints, and implementation sequence

Expose an architecture configuration containing angular scales, population sizes,
and sampling choices. Save both requested settings and resolved discrete geometry
with checkpoints. Reconstruct the exact same architecture in training, evaluation,
and resume; never create or resize trainable layers inside `forward()`.
Here, dynamic FOV-dependent architecture means construction from configuration
and metadata, not changing FOV or trainable shapes within a run. A different FOV
builds a new model with its own compatible checkpoint; evaluate/resume an existing
checkpoint only at its saved geometry.

Use architecture identifier `geometry-aware-handed-v1` and a new
`dw93net-checkpoint-v5` schema for this implementation; bump dataset identity from
version 6 to 7 for the cell-centered rendering contract. Preserve existing files
and make new datasets/caches use new identities. Reject older checkpoints rather
than loading them by inferred shape or adding a silent fallback.

Save requested configuration plus a canonical resolved contract: FOV, every grid's
spacing/origin/count, retained indices and intervals, crop rules, boundary mode,
ordered displacement labels and temporal units, DoG parameters, V1 support sizes,
MT/MST offsets and strides, filter coefficients and phases, handed-channel maps,
partial-edge rules, every layer's shape/activation, and initialization settings.
Register fixed tensor geometry/filter data as buffers. Compare schema/architecture,
categorical fields, ordered labels, shapes, and coefficient fingerprints exactly;
use the declared geometry tolerance only when resolving user floating-point input
into the canonical contract. Include target units and loss scaling in the contract.
Show field-level mismatches on resume/evaluation instead of resizing or guessing.

Persist model, optimizer, scheduler, AMP scaler, CPU/device RNG, sampler/training
progress, dataset identities, source revision, and numerical-mode settings for
resume. Restore these before the next training step. Validate state health before
atomic checkpoint publication, preserving the existing float32-retry and prior
healthy-checkpoint behavior. A retry must restore optimizer and RNG state as well
as weights. Operational items such as measured timing are reports, not geometry.

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
- Verify the selected full-panorama MT and MST grids have 1,440 and 360 positions,
  respectively, and that filtered MT-to-MST reduction uses a factor of 4 with
  documented angular alignment. Check MT's 41 adjacent subfields span 10 degrees.
- Construct full and partial FOV models and verify layer shapes are derived from
  retained geometry, no mask channels/full-circle placeholder maps are created,
  and trainable parameter shapes remain constant across forward calls.
- Confirm requested physical sizes remain consistent across resolutions within
  declared discretization error, including dots, DoG, and detector spans.
- Test circular seams and partial-view edges; missing data must not be treated as
  observed zero motion or connected across a partial-view gap.
- Check gradients reach V1 kernels, normalization parameters, opponent strengths,
  MT spatial/channel weights, and MST templates.
- Verify each directional V1/MT/MST handed pair derives from one stored base template and both
  variants contribute gradients to it. Check reflected spatial offsets, explicit
  input-channel permutations, shared scalar parameters, and parameter counts.
- Verify MT initialization applies the configured centered Gaussian envelope to
  signed random base weights once, derives handed partners from the same weights,
  and leaves individual spatial weights free to change independently afterward.
- Test layer-level reflection behavior on paired inputs at the supported grid
  alignment. Do not claim end-to-end reflection equivariance unless preprocessing
  and the decoder satisfy the corresponding contracts.
- Test positive normalization denominators, finite outputs/gradients, signed
  correlation behavior, zero displacement, and supported reordered detector banks.
- Verify opposite directions use one learned opponent strength per displacement
  magnitude within each V1 population, with gradients from both directions.
- Use controlled constant-motion and locally varying-motion inputs to probe
  integration, suppression, and sensitivity to small speed differences.
- Verify guaranteed full-panorama shift behavior for intermediate maps at the
  supported shift increments. Measure final-output rotation and reflection errors
  separately; test the translation and yaw sign conventions explicitly.
- Verify circular decoder layers process neighborhoods across the panoramic seam
  consistently with other positions, while partial-view decoder layers do not wrap.
  Check that the final joint readout retains angular position information.
- Verify save/load and resume reconstruction, and useful errors for incompatible
  geometry or architecture.
- Run production-width forward/backward GPU preflights in AMP and the required
  float32 fallback, measuring memory and throughput before choosing batch size.
- Compare baseline and upgraded models on the same scientific stimulus
  distributions, accounting explicitly for any rendering changes.
- Measure rotation-only, translation-only, and mixed-motion errors; test cross-talk
  by holding one component fixed while varying the other.
- Probe learned direction/speed tuning and effective RFs with localized stimuli.
  Inspect learned spatial/channel profiles, but do not infer necessity
  from weights alone.
- Distinguish post-training ablations (reliance) from retrained comparisons
  (ability to compensate), using multiple seeds and comparable capacity where feasible.

No separate internal translation/rotation populations are required for success.

### Numerical tolerances and production preflight

Jacob reduced `[optimization].batch_size` to 16 in
`scripts/mission_control/mission_plan/trans_plus_rot_panoramic.toml`, reporting
approximately 99% VRAM usage at batch 32 with the existing model. Preserve that
setting; do not restore 32 or assume 16 will fit the new architecture. Budget for
additional trainable layers, gradients, optimizer state, activations, and temporary
buffers, and measure the complete new pipeline rather than estimating total VRAM
from parameter count alone. Treat 16 as the current requested training batch and
the upper limit of the initial batch-size preflight, not a verified safe size.

- Geometry construction uses float64; accept integer ratios within `1e-8` sample
  units and check reconstructed angular coordinates within `1e-9` degrees.
  Reject NaN, infinity, nonpositive spacing, and mismatched explicit counts.
- Small deterministic operator/reference and symmetry tests use float64 with
  `atol=1e-10, rtol=1e-8`, and float32 with `atol=1e-5, rtol=1e-4`. Disable TF32
  for strict float32 reference tests. Use `atol=1e-3, rtol=1e-2` for AMP comparisons
  on bounded, documented fixtures. These are numerical tolerances, not predictive
  accuracy thresholds; investigate failures rather than silently loosening them.
- Assert finite outputs, loss, gradients, and optimizer state; V1 denominators
  are at least `1e-4` (within rounding), and normalized positive kernels sum to
  one within `1e-6` in float32. Check constant-signal preservation at partial edges
  and rejection of empty retained geometry. Use float64 gradient checks away from ReLU
  kinks on small fixtures, including both contributions to tied parameters.
- Verify filter DC gain is one, coefficients are nonnegative, and at full-support
  positions output-Nyquist amplitude gain is at most 0.065 for the selected two
  reductions. Test sinusoids across frequency, impulses at the seam, exact
  half-phase alignment, reflection, and nonwrapping partial edges. Quantify attenuation
  and residual aliasing rather than claiming an ideal low-pass response.
- On the selected grids, guarantee input-to-MT shift equivariance for shifts of
  0.25 degrees and input-to-MST for shifts of 1 degree on full panoramas. For any
  decoder downsampling later introduced, restrict the corresponding guarantee
  to its compatible shift subgroup. Test V1/MT/MST reflection with channel-pair
  permutations; do not apply an exact reflection/output-rotation assertion to an
  unconstrained decoder. Record output symmetry errors as scientific metrics.
- Save/load must reproduce outputs within the float32 tolerance. A same-device,
  deterministic small training fixture must resume the next update consistently
  with an uninterrupted run, including optimizer and RNG state. Cross-device
  bitwise equality is not required. Production/scripts/shared helpers must not
  import test files; every test file creates its own temporary data.
- Before long training, benchmark complete forward, backward, and Adam steps at
  production geometry (21,600 input columns and the stated 241-channel bank).
  Run both AMP and full float32, testing candidate batches 1, 2, 4, 8, and at most
  16 only while memory permits, with 3 warmup steps and
  10 measured steps per candidate batch. Include online correlation, resampling,
  decoder, optimizer state, and synchronized GPU timing. Report peak allocated
  and reserved memory, examples/second, per-stage activation shapes, and hardware.
  Do not test batch 32 as a default candidate. Require at least 20% device-memory
  headroom for the chosen batch and successful float32 fallback, using a smaller
  fallback microbatch with equivalent accumulation if needed. If batch 1 fails,
  revise implementation/memory use rather than silently reducing the geometry.
- Compare a baseline run on the same hardware/data pipeline. Set scientific
  accuracy and acceptable runtime thresholds after measuring the baseline;
  initialization health and the above correctness checks are the initial gate.
  Do not report production preflight or training success until actually executed.

## Remaining decisions before implementation

The high-level choices above are settled or labeled as proposals. The following
details were not finalized in the discussion and must not be presented as already
approved:

- Physical-unit recipe fields and discretization rules for rendered dot footprints,
  DoG scales, and detector spans at resolutions other than the initial one arcminute.
  Grid origins, selected layer grids, V1 kernel sizes, filtering, and boundary
  behavior are specified above.
- Circular nonlinear decoder depth, feature counts, angular kernel supports,
  strides, activations, and position-aware final joint readout architecture;
  rotation augmentation policy beyond the existing handed stimulus presentations.
- Experiment configuration, capacity comparisons, and baseline-derived scientific
  accuracy, output symmetry error, and runtime thresholds. These need measurement,
  not invented numerical performance claims. Schemas, initialization, numerical
  correctness tolerances, and the production preflight protocol are specified above.

Resolve these choices into a concrete implementation specification before coding;
do not revert to the superseded fixed-120-bin dense MST or mandatory two-branch MT
design without discussing the change.
