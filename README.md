# dw93net

## Repository layout

Generated signed DoG contrast bundles are local experiment
data and are ignored by Git. Their absolute locations are configured by the
committed recipe.
`src/stimuli/` contains the code that generates, preprocesses, loads, and
displays those inputs. A committed `recipe/<network-name>.toml` generates the
datasets, trains one network, selects its final weights, and evaluates them.

Each parameter set contains the compact detector stage used by training:

```text
stimuli/<parameter-set>/
  dog/          # dog.json, provenance, and one train/validate/test role
```

The orchestrator renders Flow-XT stimuli and applies the DoG filter in memory,
then writes only float32 signed DoG bundles. Training and evaluation calculate
the requested shifted correlations on the model device. Evaluation `--data`
points to `stimuli/<parameter-set>/dog/`.

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

Only one recipe run per user may be active on this machine, regardless of
checkout or dataset paths. A second run fails immediately. The lock covers
background generation and cleanup, and releases automatically when all owning
processes exit, including after crashes. The file
`/tmp/dw93net-run-locks-<uid>/run.lock` intentionally remains; do not delete it
to unlock a run. This advisory lock does not coordinate different hosts or
users, or standalone tools that do not acquire it.

The runner requires a clean commit and CUDA by default. For a recipe named
`translation_only.toml`, it creates
`<output_path>/translation_only/{checkpoints,final}` plus an empty file named
`commit_id=<full-40-character-Git-ID>`. A mismatching marker prevents resume.
`latest.pt` is a full epoch-boundary resume checkpoint; `previous.pt` is its
optional predecessor. Model-only candidates are saved at the configured epoch
interval.

Each training seed creates a new DoG dataset at `training_sets.path` without
writing intermediate Flow-XT files. The entire training cache is deleted after
its epochs. The learning-rate schedule and
optimizer continue across replacements. After all training, validation is
generated once and evaluates every candidate without shuffling. The best
normalized-RMSE candidate is promoted to `final/model.pt`; validation is then
deleted unless requested otherwise. The persistent test set is generated or
reused at `test_set.path` and evaluates only the final model. A recipe mismatch
never silently overwrites that test set.

Validation progress is atomically saved after every candidate in
`final/validation-progress.json`. If the runner is interrupted, it verifies each
checkpoint fingerprint and resumes at the first candidate that still needs
evaluation. `final/validation-results.json` records completed candidate selection and
`final/test-results.json` records the final test measurement. Rerunning restarts
only an interrupted epoch and otherwise resumes from the recorded training-set
index and seed.

## Training behavior

The committed recipe enables `[parallel_generation]`. While training runs on
one set, a CPU-only subprocess generates the next in a sibling `training.next`
directory. A successful worker exit and published dataset state are required
before an atomic directory rename makes that set current. Set
`enabled = false` for sequential generation. Omitted settings default to
sequential operation for older recipes.

`min_threads` (default 1) and `max_threads` (default 8) bound the generator's
PyTorch threads and Linux CPU allocation. The generator runs at niceness 10
and requests best-effort low I/O priority. Its CPU pool excludes a portion of
the available CPUs for training; training itself keeps its existing affinity.
BLAS and inter-op thread pools are limited to one. CPU affinity uses logical
CPUs, not topology-aware physical-core allocation.

Every 32 training batches, two consecutive windows with loader wait above 15%
halve generator threads; four below 3% add one thread. Intermediate windows
reset these streaks. The worker applies changes every 64 generated examples.
At a set boundary, unfinished generation receives its maximum allocation.
Allocation changes and boundary wait time are logged. Loader wait is a host
timing proxy, not a measurement of GPU idle time; throughput should be compared
against sequential operation on the target machine before tuning these limits.

Completed staged data can be reused on restart when its recipe specification
matches. Incomplete managed staging is removed on orderly failure or rebuilt
on restart; a worker also stops if its training parent disappears. Existing
unmanaged staging directories are never overwritten. As with the sequential
runner, run only one orchestrator per output/dataset path. Allow disk space for
two training sets during overlap.

Adam and the plateau scheduler are configured in the recipe's `[learning_rate]`
section. The scheduler monitors training loss; its history continues across
training sets and survives restarts. Validation happens after training finishes.
Run only one orchestrator per network output directory at a time.

Training also presents an in-memory horizontal reflection of every signed DoG
contrast by default (`optimization.mirror_train_stimuli = true`). It reverses
the spatial axis and changes the motion target from `[dx, dz, rot]` to
`[-dx, dz, -rot]`; online correlation then produces the reversed displacement
responses. Validation
and test examples remain unchanged.

GPU feeding defaults to batches of 64 and CUDA float16 mixed precision (with
float32 loss and gradient scaling). Cached signed DoG contrast remains float32
through loading and transfer. Correlations are calculated in VRAM immediately
before the model. CPU training and evaluation use the same float32 path.
Set `optimization.mixed_precision = false` to compare numerical behavior. Larger
batches change optimizer updates per epoch, so compare validation quality as
well as throughput; reduce `batch_size` if your run exceeds available VRAM.
Scaler state is saved for resume.

Each epoch reports samples/second and host time waiting for the loader. On CUDA,
`optimization.profile_batches` (default 20; 0 disables it) samples transfer/correlation and
model/optimizer durations using GPU events. The first samples include warmup;
use later epochs for comparisons. Loader wait can overlap queued GPU work and
is not itself GPU idle time. Loss totals synchronize once per phase instead of
once per batch, and the allocator cache is retained between epochs.

DoG caches use memory-mapped NumPy bundles. `stimuli.dog_bundle_size` defaults
to 64 and must be at least 1. `DoGDataset` reads this format. Each row stores a
float32 signed contrast `[2,X]` and its float32 `[dx,dz,rot]` target; targets are
never parsed from filenames. The writer validates finite values before
publishing `dog.json`, and the loader validates shape and dtype.

## Reusable training evaluation

From the repository root, evaluate saved checkpoints with:

```bash
/home/jacob/venvs/nvidia/bin/python scripts/evaluate_progress.py \
  --checkpoints /absolute/output/translation_only/checkpoints \
  --data /absolute/validation/cache/parameter-set/dog
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




## DoG caching and online correlation

Each stimulus is a pair of time bins: the initial dots and the same dots after
one translation-and-rotation step. There are no dot lifetimes, replacements, or
boundary wrapping; X-only supersampling preserves fractional intensities.
By default, `stim_wid_px = round(fov_deg * 60)` provides nominal one-arcminute
angular sampling: a 120° FOV uses 7,200 pixels. An explicit `stim_wid_px`
overrides this default, for example in small tests.
The recipe pipeline renders each `[2,X]` Flow-XT stimulus in memory. Default
X supersampling is **255**, so the rendered intensity steps align with the byte
levels. Values are rounded to the nearest byte after summing and clipping any
superimposed layers. Explicit supersampling overrides remain supported, with
quantization to the same byte grid before DoG filtering. The byte tensor is
decoded in memory and never written. There is no raw Flow-XT dataset format or
disk-backed compatibility path.

Translation is sampled in polar coordinates. The defaults draw speed uniformly
from the half-open interval `[0.001, 0.015)` scene units per frame step and
heading uniformly from `[-180, 180)` degrees. Heading 0 is forward (+Z), -90 is
left (-X), and +90 is right (+X). Cartesian motion is then calculated as
`dx = speed*sin(heading)` and `dz = speed*cos(heading)`. The strictly positive
minimum avoids samples whose translation is zero or too close to zero to define
a useful heading. Derived `dx` and `dz` are rounded to nano-units before both
rendering and filename labeling.

The generated folder name records speed, heading, and rotation bounds.
`dog.json` records the exact half-open interval semantics, seed, dataset role,
coordinate conversion, and stimulus configuration.

Preprocessing applies a **1D luminance DoG independently to each frame**:
center sigma 1 pixel, surround sigma 3 pixels, kernel radius 9 pixels (19 taps).
Both discrete Gaussians are separately normalized to unit sum. Filtering uses
valid support, no padding, no temporal filtering, and no per-image rescaling.
The float32 signed contrast `[2,X-2r]` is stored directly. ON and OFF remain
recoverable as `relu(contrast)` and `relu(-contrast)`.

For each training or evaluation batch, `ReichardtBank(dx_values)` computes the
signed product `contrast(0,x-s)*contrast(1,x)` on the model device for every
configured displacement `s`. This equals
`ON_ON + OFF_OFF - ON_OFF - OFF_ON` algebraically, without allocating four
redundant polarity blocks. Negative responses remain available until local
opposite-direction suppression and rectification.

For radius `r`, `left=r+max(0,max(dx_values))` and
`right=r+max(0,-min(dx_values))`, a `[2,X]` or `[1,2,X]` stimulus produces
`[len(dx_values),1,X-left-right]`. The defaults give `[121,1,7062]` at X=7200.
Positions are anchored in the present frame's original pixel coordinates.
Filtering, cached contrast, and online correlations use float32.

Generation preserves the requested role and stable sample identifiers, and publishes
`dog.json` only after completion. Existing destinations are refused. The cache
format is `dog-contrast-bundles-v1`. `DoGDataset` reads the contrast bundles;
checkpoints retain both DoG and displacement metadata, and evaluation rejects
incompatible caches.

Training uses range-normalized MSE on each example's motion prediction.

The recipe configures training and evaluation workers separately (eight by default).
Training workers persist across epochs; CUDA batches use pinned RAM and
non-blocking transfers. Training loads and transfers float32 DoG contrast, then
computes correlations on the device. Set `optimization.num_workers = 0` for in-process
loading. Programs using workers need an `if __name__ == '__main__':` entry point.

The earlier [pipeline benchmark](benchmarks/storage-20260910-final/REPORT.md)
found four workers competitive with individual files on this machine. Current
DoG caches use memory-mapped stimulus bundles; recheck throughput on a
full dataset when tuning bundle and worker counts.

Run the correctness and pipeline tests with the project's Python environment:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

## Translation-only model baseline

On `toward_translation`, `src/models/pure_translation.py` replaces the original
regressor. It consumes `[B,D,1,X]` signed correlations and predicts `[dx,dz,rot]` in
scene units per frame step and degrees per step. Current translation-only
datasets target zero on `rot`. Heading is `atan2(dx,dz)` (zero toward +Z,
positive toward +X); translation distance per step is `hypot(dx,dz)`. Heading
is undefined at zero translation. Speed in units per second additionally
requires the step duration.

The default detector bank now includes every integer displacement from **-60
through +60 pixels per frame step**. At 7200 pixels over 120 degrees, that is
approximately +/-1 degree per step; degrees/second additionally requires the
frame duration. Online correlation produces `[121,1,7062]` maps and discards
them after the batch. Changing the displacement bank does not require new DoG
stimuli, but it does define a different committed detector and checkpoint run.
The configured range extends measurement capacity, not the camera-translation
label ranges, and cannot represent displacements outside +/-60.

The model has explicit V1, MT, and MST inspired stages:

- **V1:** form local signed-correlation responses over 15 pixels
  and a second population with a broader, same-channel absolute-correlation divisive surround (61
  pixels, semisaturation 0.03). These are functional rate approximations using
  online correlations, not a physiological motion-energy model.
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
Keep FOV, width, crop, and detector semantics fixed across dataset roles. Pipeline
tests verify implementation, not translation accuracy.

Biological references: [Qian, Andersen & Adelson (1994)](https://doi.org/10.1523/JNEUROSCI.14-12-07381.1994),
[Tsui et al. (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2888240/),
and [Duffy & Wurtz (1991)](https://doi.org/10.1152/jn.1991.65.6.1329).

## Preprocessing provenance and recovery

Every new cache root contains `dog.json`, recording DoG sigmas/support,
normalization and boundary rules, input/output shapes and dtypes, sample count,
UTC start/end times, generation settings, and the full Git HEAD commit ID.

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
are not bundled. Dependencies are recorded, not packaged. Artifact metadata is
stored in `dog.json`. Git must be available with an existing HEAD commit.
If repository files change during preprocessing, the run fails without publishing
`dog.json`, so training cannot mistake the partial cache for a complete one.

Use `--sigma-center` and `--sigma-surround` to override the pixel sigmas in
standalone preprocessing; the kernel radius is always `ceil(3*sigma_surround)`.
