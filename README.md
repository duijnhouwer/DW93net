# dw93net

## Repository layout

Generated input examples and precomputed detector maps are local experiment
data and are ignored by Git. Their absolute locations are configured by the
committed recipe.
`src/stimuli/` contains the code that generates, preprocesses, loads, and
displays those inputs. A committed `recipe/<network-name>.toml` generates the
datasets, trains one network, selects its final weights, and evaluates them.

Each parameter set groups its original images and precomputed responses:

```text
stimuli/<parameter-set>/
  raw/          # generation.json and train/, validate/, test/ images
  reichardt/    # detector metadata, provenance, and matching split responses
```

Generation returns the `raw/` path; combined generation/preprocessing writes
its sibling `reichardt/`. The orchestrator supplies this cache path to the trainer; evaluation
`--data` points to `stimuli/<parameter-set>/reichardt/`.

## Unattended training recipes

Exactly one TOML file may exist directly under `recipe/` in a commit. Its
filename stem is the network name. The recipe gives the common stimulus ranges,
absolute training/validation/test paths, a sequence of training seeds, epoch
counts, dynamic learning-rate settings, and an absolute output root. The runner
takes no arguments and discovers that one file.

The shared topology always outputs `[dx, dz, rot]`. Rotation training is
currently disabled, so the recipe must specify
`rot_deg_range = [0, 0]`; this trains the rotation output toward zero. The same
topology can be retained when rotation experiments are enabled later.

Edit the recipe, commit the recipe and all code changes, then run:

```bash
PYTHONPATH=src /home/jacob/venvs/nvidia/bin/python scripts/run_training_recipe.py
```

The runner requires a clean commit and CUDA by default. For a recipe named
`translation_only.toml`, it creates
`<output_path>/translation_only/{checkpoints,final}` plus an empty file named
`commit_id=<full-40-character-Git-ID>`. A mismatching marker prevents resume.
`latest.pt` is a full epoch-boundary resume checkpoint; `previous.pt` is its
optional predecessor. Model-only candidates are saved at the configured epoch
interval.

Each training seed creates a new dataset at `training_sets.path`. Raw stimuli
are deleted as soon as their Reichardt bundles are complete, and the entire
training cache is deleted after its epochs. The learning-rate schedule and
optimizer continue across replacements. After all training, validation is
generated once and evaluates every candidate without shuffling. The best
normalized-RMSE candidate is promoted to `final/model.pt`; validation is then
deleted unless requested otherwise. The persistent test set is generated or
reused at `test_set.path` and evaluates only the final model. A recipe mismatch
never silently overwrites that test set.

`final/validation-results.json` records candidate selection and
`final/test-results.json` records the final test measurement. Rerunning restarts
only an interrupted epoch and otherwise resumes from the recorded training-set
index and seed.

## Training behavior

Adam and the plateau scheduler are configured in the recipe's `[learning_rate]`
section. The scheduler monitors training loss; its history continues across
training sets and survives restarts. Validation happens after training finishes.
Run only one orchestrator per network output directory at a time.

Training also presents an in-memory horizontal reflection of every Reichardt
response by default (`optimization.mirror_train_stimuli = true`). It reverses the
spatial and signed-displacement axes within each of the four polarity groups,
and changes the motion target from `[dx, dz, rot]` to `[-dx, dz, -rot]`. Validation
and test examples remain unchanged.

GPU feeding defaults to batches of 64 and CUDA float16 mixed precision (with
float32 loss and gradient scaling). Cached products remain float16 through
loading, augmentation, pinned memory, and transfer; conversion happens on the
device. CPU training and evaluation continue to use float32 model computation.
Set `optimization.mixed_precision = false` to compare numerical behavior. Larger
batches change optimizer updates per epoch, so compare validation quality as
well as throughput; reduce `batch_size` if your run exceeds available VRAM.
Scaler state is saved for resume.

Each epoch reports samples/second and host time waiting for the loader. On CUDA,
`optimization.profile_batches` (default 20; 0 disables it) samples transfer/cast and
model/optimizer durations using GPU events. The first samples include warmup;
use later epochs for comparisons. Loader wait can overlap queued GPU work and
is not itself GPU idle time. Loss totals synchronize once per phase instead of
once per batch, and the allocator cache is retained between epochs.

Reichardt caches use memory-mapped NumPy stimulus bundles. The bundle size defaults
to 64 and must be at least 1; a size of 1 stores one stimulus per bundle.
`ReichardtDataset` reads only this bundled format. Each bundle stores
`reichardt_responses` and their float32
`[dx, dz, rot]` targets as binary fields, so the loader does not parse targets
from filenames. Bundles preserve all four polarity groups and exact float16
response values, avoiding a change to the model's input representation. They
reduce file-opening overhead but do not compress the cache. The writer checks
finite values after float16 conversion before publishing `reichardt.json`;
loading checks shape/dtype without repeating a finite scan every epoch.

The orchestrator generates raw stimuli and bundles automatically.

## Reusable training evaluation

From the repository root, evaluate saved checkpoints with:

```bash
/home/jacob/venvs/nvidia/bin/python scripts/evaluate_progress.py \
  --checkpoints /absolute/output/translation_only/checkpoints \
  --data /absolute/validation/cache/parameter-set/reichardt
```

The current experiment recovers translation vectors from one-step flow fields
with recipe-defined heading and speed ranges. Heading angular error is the primary
metric; speed and vector errors are tracked as well. Fixed-speed accuracy does
not demonstrate recovery across different speeds. Rotation is deferred until
simple translation recovery works.

The command uses 256 seeded validation examples, CPU inference, and no worker
processes by default. It saves a sample manifest, per-checkpoint JSON metrics,
and a `progress.md` report ordered by global epoch under `evaluations/`. Repeating the
command reuses results for unchanged checkpoints, data, and evaluation code.
New or overwritten checkpoints are evaluated once; old cached records remain.
Use `--samples 0` for the whole split, `--split test` for a final held-out
assessment, or `--help` for explicit dataset/checkpoint paths and other options.
Changing sample size or seed creates a separate comparison set. Reports include
heading mean/median error and within-15-degree accuracy, speed MAE/RMSE, and
mean Euclidean vector-tip error (`||[dx,dz]-[dx_hat,dz_hat]||`); JSON also
includes vector-tip median/90th percentile, heading bias/90th percentile/
undefined-heading count, and speed bias. Speed and vector errors are in scene
units per step.

By default, reports also stratify results by true heading (global and four
cardinal bins: front/right/back/left) and true speed (global plus slow/medium/
fast thirds). Use `--heading-bins 1 4 8 12` or `--speed-bins 1 3 5` to request
any positive number of equal-width bins. Heading bins are circular and centred
on front (0 degrees); speed-bin edges span the fixed evaluation sample's true
speed range.

The evaluator saves PNG plots by default under its report directory: global
metrics plus one plot for every requested non-global heading or speed bin count.
Pass `--no-plots` to skip them, or `--show-plots` to additionally open local
interactive plot windows when a graphical desktop is available. All plot axes
start at zero and mean-error points have normal-approximation 95% confidence
interval error bars. By default, heading is plotted in degrees while speed and
vector-tip error are plotted as per-sample percentages. Add
`--plot-heading-percent` to also save
heading percentage plots. The percentage definitions are, respectively,
`100 * heading_error_deg / 180`, `100 * |speed_hat - speed| / speed`, and
`100 * ||[dx_hat,dz_hat]-[dx,dz]|| / ||[dx,dz]||`.

The report reads model-only candidates and requires their Git commit to match
this checkout. Supply an existing validation cache; use `keep_after_run = true`
in the recipe if you want it retained for these retrospective reports.

**dw93net** stands for **Duffy & Wurtz (1993) network**. It is an artificial
neural network experiment designed to test the *"compensation for eye rotation"*
hypothesis proposed to explain a perceptual illusion in optic-flow perception.

## The phenomenon

Duffy, C. J., & Wurtz, R. H. (1993) showed the following effect. When an
observer views an **expanding optic flow field** (for example, a "starfield"
style screensaver in which the flow radiates outward from a single point) with
a **uniform flow superimposed** on top of it, the **focus of expansion (FOE)**
of the expanding field appears to shift in the direction of the superimposed
uniform flow.

The focus of expansion is the point in the visual field from which the
expanding flow appears to radiate. For a purely translating observer, the FOE
marks the current **heading direction** — the point in the scene toward which
the observer is moving.

## The hypothesis: compensation for eye rotation

The authors' explanation rests on how the visual system interprets each
component of the flow:

1. The **expanding flow** is interpreted as evidence of **forward translation**
   of the observer. The location of the FOE indicates the heading direction.
2. The **uniform (rotational) flow** is interpreted as evidence of **eye
   rotation**. Eye rotation adds a motion component to the entire visual field,
   which shifts the FOE away from the true heading direction.
3. To still be able to determine heading, the brain **compensates for the eye
   rotation**. This compensation distorts the interpretation of the expanding
   flow, which accounts for the apparent shift of the FOE in the direction of
   the uniform flow.

## The idea of dw93net

dw93net tests this *"compensation for eye rotation"* hypothesis using
artificial neural networks. Two convolutional deep vision networks are built
with **exactly the same layout** but **trained on different stimuli**.

The stimuli are **optic flow fields generated by moving a virtual camera in any
direction, with or without rotation about the vertical axis (yaw)**.

| Network | Training stimuli |
| ------- | ---------------- |
| **NOYAW** | Pure camera translation only (no yaw). |
| **YESYAW** | Camera translation **and** rotation (yaw). |

The experiment is carried out in two steps.

### Step 1 — parameter retrieval

- How well does **NOYAW** recover the heading parameters from translation-only
  stimuli?
- How well does **YESYAW** recover the heading parameters *and* the yaw rate from
  stimuli in which the camera both translates and rotates?

If **YESYAW** can accurately recover the heading translation even on stimuli
where the camera is rotating, it must have learned **how to compensate for
yaw**. **NOYAW** is expected to produce **biased** heading estimates in the
presence of yaw, because it was never trained on any yaw stimuli.

### Step 2 — the Duffy & Wurtz test

Both networks are then shown **Duffy & Wurtz style stimuli**: a pure-translation
optic flow field (yaw = 0) with a uniform flow superimposed on it.

If the Duffy & Wurtz illusion is indeed caused by compensation for eye
rotation, then:

- **YESYAW** is expected to **shift** its heading estimate in the direction of
  the superimposed uniform flow.
- **NOYAW** is expected to **not** show this shift.

## Terminology / variable mapping

The scientific quantities above are represented in the code as follows:

| Scientific quantity | Code variable |
| ------------------- | ------------- |
| Camera translation along x (heading) | `dx` |
| Camera translation along z / depth (heading) | `dz` |
| Camera rotation about the vertical axis (yaw) | `rot_deg` |

## Reference

- Duffy, C. J., & Wurtz, R. H. (1993). An illusory transformation of optic flow
  fields. *Vision Research*, 33(11), 1481–1490.
  https://doi.org/10.1016/0042-6989(93)90141-I




## Reichardt product preprocessing

Each stimulus is a pair of time bins: the initial dots and the same dots after
one translation-and-rotation step. There are no dot lifetimes, replacements, or
boundary wrapping; X-only supersampling preserves fractional intensities.
By default, `stim_wid_px = round(fov_deg * 60)` provides nominal one-arcminute
angular sampling: a 120° FOV uses 7,200 pixels. An explicit `stim_wid_px`
overrides this default, for example in small tests.
`generate_flow_xt_dataset(..., n_samples=M)` saves M `.pt` files of shape
`[1, 2, X]`, one stimulus per file, as **uint8 grayscale (0–255)**. Default
X supersampling is **255**, so the rendered intensity steps align with the byte
levels. Values are rounded to the nearest byte after summing and clipping any
superimposed layers. Explicit supersampling overrides remain supported, with
quantization to the same byte grid at save time. The renderer itself still
returns float32 intensities in [0, 1]. Each file has newly sampled motion
parameters and dot positions.

Translation is sampled in polar coordinates. The defaults draw speed uniformly
from the half-open interval `[0.001, 0.015)` scene units per frame step and
heading uniformly from `[-180, 180)` degrees. Heading 0 is forward (+Z), -90 is
left (-X), and +90 is right (+X). Cartesian motion is then calculated as
`dx = speed*sin(heading)` and `dz = speed*cos(heading)`. The strictly positive
minimum avoids samples whose translation is zero or too close to zero to define
a useful heading. Derived `dx` and `dz` are rounded to nano-units before both
rendering and filename labeling.

The generated folder name records speed, heading, conservative Cartesian
component bounds, and rotation bounds. Its `generation.json` records the exact
half-open interval semantics, seed, split, coordinate conversion, and stimulus
configuration. When present, this document is copied into the preprocessed
cache's `reichardt.json` as `stimulus_generation`.

`FlowXtDataset`, preprocessing, and the sample viewer decode uint8 by dividing
by 255 before using the intensities. Raw files must contain uint8 tensors;
product files must contain float16 tensors.

Preprocessing applies a **1D luminance DoG independently to each frame**:
center sigma 1 pixel, surround sigma 3 pixels, kernel radius 9 pixels (19 taps).
Both discrete Gaussians are separately normalized to unit sum. Filtering uses
valid support, no padding, no temporal filtering, and no per-image rescaling.
The signed contrast is split into `ON=relu(DoG)` and `OFF=relu(-DoG)`.

`DoGMotionBank(dx_values)` then computes `ON(0,x-s)*ON(1,x)` and
`OFF(0,x-s)*OFF(1,x)`, plus `ON(0,x-s)*OFF(1,x)` and
`OFF(0,x-s)*ON(1,x)`, for each signed displacement `s` and a fixed one-frame
delay. Four nonnegative blocks are saved in order: ON–ON, OFF–OFF, ON–OFF,
OFF–ON, each in the supplied span order. The first polarity is the first frame;
the second polarity is the second frame.

The model reconstructs signed correlation as `ON_ON + OFF_OFF - ON_OFF - OFF_ON`
before local pooling and directional opponency. Negative responses are retained
until opposite-direction subtraction and rectification. The suppressive surround
uses absolute signed correlation to keep its denominator positive. These fixed
interactions are a starting hypothesis; the separate cached blocks remain
available for future divisive or learned cross-polarity interactions.

For radius `r`, `left=r+max(0,max(dx_values))` and
`right=r+max(0,-min(dx_values))`, a `[2,X]` or `[1,2,X]` stimulus produces
`[4*len(dx_values),1,X-left-right]`. The defaults give `[484,1,7062]` at X=7200.
Positions are anchored in the present frame's original pixel coordinates.
Filtering and products use float32; cached products use float16. The lower-level
`ReichardtBank` remains the product primitive; cache generation always uses
`DoGMotionBank`.

For standalone preprocessing experiments, `python -m stimuli.precompute_reichardt`
accepts separate raw-source and cache-destination paths.

Preprocessing preserves splits and label filenames, and writes `reichardt.json`
with full preprocessing metadata only after completion. Existing destinations are
refused. Products are computed from normalized intensities in float32 and stored
as **float16**, with rounding and an overflow check. Regenerate caches when
stimuli or detector settings change. The fixed-delay cache format is
`dog-onoff-products-v2`.

`ReichardtDataset` reads cached maps as float32. `PureTranslationNet` learns from these
maps downstream of detection. Checkpoints retain detector metadata, and
validation/evaluation reject mismatched configurations. Use the cached dataset
root for evaluation; start fresh when detector configurations change.

Training uses range-normalized MSE on each example's motion prediction.

The recipe configures training and evaluation workers separately (eight by default).
Training workers persist across epochs; CUDA batches use pinned RAM and
non-blocking transfers. Training loads and transfers float16 products, then
converts them on the device. Set `optimization.num_workers = 0` for in-process
loading. Programs using workers need an `if __name__ == '__main__':` entry point.

The earlier [pipeline benchmark](benchmarks/storage-20260910-final/REPORT.md)
found four workers competitive with individual files on this machine. Current
Reichardt caches use memory-mapped stimulus bundles; recheck throughput on a
full dataset when tuning bundle and worker counts.

Run the correctness and pipeline tests with the project's Python environment:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Translation-only model baseline

On `toward_translation`, `src/models/pure_translation.py` replaces the original
regressor. It consumes `[B,C,1,X]` product maps and predicts `[dx,dz,rot]` in
scene units per frame step and degrees per step. Current translation-only
datasets target zero on `rot`. Heading is `atan2(dx,dz)` (zero toward +Z,
positive toward +X); translation distance per step is `hypot(dx,dz)`. Heading
is undefined at zero translation. Speed in units per second additionally
requires the step duration.

The default detector bank now includes every integer displacement from **-60
through +60 pixels per frame step**. At 7200 pixels over 120 degrees, that is
approximately +/-1 degree per step; degrees/second additionally requires the
frame duration. With DoG and four polarity pairs it produces `[484,1,7062]` maps: 6,836,016
bytes of float16 payload per sample, excluding file overhead.
Existing caches do not acquire new channels automatically: precompute from raw
stimuli into a new destination using the default CLI settings. Raw stimuli do
not need regeneration solely to change the detector bank. This extends the
measurement range, not the camera-translation label ranges, and still cannot
represent displacements outside +/-60.

The model has explicit V1, MT, and MST inspired stages:

- **V1:** combine the four products into signed correlations; form local responses over 15 pixels
  and a second population with a broader, same-channel absolute-correlation divisive surround (61
  pixels, semisaturation 0.03). These are functional rate approximations using
  cached products, not a physiological motion-energy model.
- **MT input:** subtract equal-magnitude opposite-direction responses and
  rectify locally, before larger spatial pooling. Zero-span evidence passes
  through. Pairing uses actual signed metadata, so both signs must be supplied.
  This adapts Qian, Andersen & Adelson's local opponency principle; their
  primary suppression stage is MT, not V1. Our speed-channel matching does
  not reproduce their spatial-frequency/disparity channels.
- **MT:** pool the surviving subunits and learn 96-channel local combinations
  with progressively larger receptive fields. Retain multiple speed signals
  instead of explicitly reducing them to one local velocity estimate.
- **MST:** retain 120 ordered spatial bins, then learn 256 broad-field units
  and decode their activity to two translation components. Broad spatial
  integration is motivated by MST physiology; biological tuning is not imposed
  or guaranteed by this learned readout.

Layer comments explain all starting hyperparameters. No dropout or batch
normalization is used. This simplified feedforward model has no binocular
signals, orientation channels, temporal dynamics, recurrence, or eccentricity-
dependent receptive fields. Its mechanisms and estimation accuracy need tests
and ablations; speed scale still relies on the training distance distribution.

Training and evaluation stop before loading files unless the dataset root
explicitly specifies `nrot=0,0`.
Checkpoints store detector metadata and the full Git commit ID. Resume requires
the exact committed code and recipe used to create the network.
Keep FOV, width, crop, and detector semantics fixed across splits. Pipeline
tests verify implementation, not translation accuracy.

Biological references: [Qian, Andersen & Adelson (1994)](https://doi.org/10.1523/JNEUROSCI.14-12-07381.1994),
[Tsui et al. (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2888240/),
and [Duffy & Wurtz (1991)](https://doi.org/10.1152/jn.1991.65.6.1329).

## Preprocessing provenance and recovery

Every new cache root contains `reichardt.json`, recording DoG sigmas/support,
normalization and boundary rules, polarity/displacement ordering, crop offsets,
input/output shapes and dtypes, sample count, UTC start/end times, source path,
and the full Git HEAD commit ID. `inputs.json` records each input's relative
path and SHA-256 of the exact bytes loaded. The metadata records its hash too.

The `provenance/` folder contains:

- `working-tree.patch`: combined staged and unstaged tracked changes from HEAD,
  including binary differences and deletions.
- `source.tar.gz`: a standalone snapshot of the working repository files,
  including nonignored untracked source. It does not require the base commit
  to reconstruct the working source.
- `source-manifest.json`: archived file hashes and permissions.
- `environment.json`: Python/platform details and installed package versions.
- `RESTORE.md`: recovery instructions and archive scope.

Extract the source archive into an **empty directory** to recover the code.
The archive contains tracked and nonignored untracked files, excluding the
current output directory; Git-ignored datasets/environments and Git history
are not bundled. Dependencies are recorded, not packaged. Artifact hashes are
stored in `reichardt.json`. Git must be available with an existing HEAD commit.
If repository files change during preprocessing, the run fails without publishing
`reichardt.json`, so training cannot mistake the partial cache for a complete one.

Use `--sigma-center` and `--sigma-surround` to override the pixel sigmas in
standalone preprocessing; the kernel radius is always `ceil(3*sigma_surround)`.
