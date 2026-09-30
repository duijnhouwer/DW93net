# dw93net

## Repository layout

Generated signed DoG contrast bundles are local experiment
data and are ignored by Git. Their absolute locations are configured by the
committed recipe.
`src/stimuli/` contains the code that generates, preprocesses, loads, and
displays those inputs. A committed `recipe/<network-name>.toml` generates the
datasets, trains one network, selects its final weights, and evaluates them.

Each managed dataset contains the compact detector stage used by training:

```text
<dataset-path>/
  dog/          # dog.json and one train/validate/test/duffy_wurtz_1993 role
```

The orchestrator renders Flow-XT stimuli and applies the DoG filter in memory,
then writes only float32 signed DoG bundles. Training and evaluation calculate
the requested shifted correlations on the model device. Evaluation `--data`
points to `<dataset-path>/dog/`. Directory names have no scientific meaning.

## Unattended training recipes

Exactly one TOML file may exist directly under `recipe/` in a commit. Its
filename stem is the network name. The recipe gives the common stimulus ranges,
absolute training/validation/test/Duffy-Wurtz paths, a sequence of training seeds, epoch
counts, dynamic learning-rate settings, and an absolute output root. The runner
discovers that one file; mission control chooses training or evaluation mode.

The shared topology outputs `[dx, dz, rot]`. `rot_deg_range` controls yaw in
degrees per frame step; `[0, 0]` trains a translation-only dataset, while a
nonzero range trains translation and yaw estimation together.

The required `[duffy_wurtz_1993_set]` section specifies a separate persistent
transparent-motion dataset:

```toml
[duffy_wurtz_1993_set]
path = "/home/jacob/Documents/dw93net_stimuli/duffy_wurtz_1993_set"
seed = 1000
n_samples = 100000
```

It is generated immediately after the validation set, before training begins.
It inherits the recipe's stimulus ranges and preprocessing, with `superimpose`
forced to `true`: half the dots undergo translation only and half rotation only.
The existing generator uses integer division for each half, so an odd configured
dot count produces one fewer dot. Neither this dataset nor the test dataset is
used for training, learning-rate scheduling, or checkpoint selection. All three
held-out seeds must differ, and training seeds must start above all three.

### Mission control

From the repository root, use the training environment:

```bash
/home/jacob/venvs/nvidia/bin/python scripts/mission_control/mission_control.py
```

The menu shows the active recipe and phase (preparing datasets, training,
validation, final_test, or duffy_wurtz_1993). It also shows the saved run’s branch
and commit, the current checkout’s branch and commit, and clean/dirty status.
Dirty includes staged, unstaged, and nonignored untracked files and blocks
starting/resuming. A saved-run commit mismatch is flagged separately. Older runs
without recorded branch metadata show “not recorded”; branch names are saved
when new runs are created.


- **1 — Report training log:** run `report_training_log.py` for the displayed run,
  with `--dashboard-gui` to open a dedicated dashboard window. Closing the window
  deletes its temporary files and returns to the menu.
- **2 — Start / Stop / Resume training:** start a new recipe, stop the active worker, or
  resume training from the latest completed checkpoint.
- **3 — Evaluate now:** interrupt training and evaluate the **best validation
  checkpoint**, first on the final test set and then the Duffy-Wurtz set. Requires
  a completed checkpoint. When idle, launch evaluation without training.
- **4 — Refresh:** reread worker status and display the menu again.
- **Q — Exit mission control:** exit the menu; the background worker continues.
  Ctrl+C has the same effect in the menu.

Each action returns to the menu. Stop takes effect at the next safe poll (normally
one training/evaluation batch or dataset-generation bundle), followed by worker
cleanup. An unfinished epoch is discarded. Once stopped, **Resume training** continues
training immediately, even if an evaluation was interrupted. Optimizer, scheduler,
and precision state come from `checkpoints/latest.pt`; training does not rewind to
the best checkpoint selected for evaluation. An unfinished training set is reused.
The configured set limit and minimum-learning-rate stop remain terminal.

The detached worker writes console output to
`<output_path>/<run-directory-name>/mission-control.log`. The menu itself does not need
Torch to refresh status. Launching requires the training environment and the same
clean, committed recipe/source identity as the runner. Existing run artifacts are
never replaced to bypass a revision mismatch.

For Python callers, `scripts.mission_control.mission_control.request_finish_training()` requests
evaluation and `request_stop()` requests a stop. The helper discovers the active
runner, validates its process identity and signal handlers, and uses a Linux PID
file descriptor to prevent signaling a reused PID. Commands return after delivery;
refresh to check progress. The menu controls one runner per user on this machine.
It cannot retrofit old, already-running Python processes without these handlers.

The updated runner uses SIGUSR1 for evaluation and SIGUSR2 for stopping; its signal
handlers only set flags. Requests are not persisted across restarts: starting the
runner normally resumes training; `--evaluate` explicitly selects evaluation.

The final results are cached separately as `final/test-results.json` and
`final/duffy_wurtz_1993-results.json`, identified by exact model/dataset hashes,
evaluation settings, and revision. Both are also recorded in `training_log.jsonl`.
Duffy-Wurtz results include circular signed heading bias grouped by overlay retinal
direction (positive `rot_deg` moves toward increasing columns in this renderer). Positive aligned bias
means toward the overlay motion. These are deviations from the separate component
labels, not errors against a single rigid-camera motion. They are not a paired
baseline subtraction and by themselves do not establish the perceptual illusion.

Ordinary evaluation still requires an exact dataset contract. Only the named
Duffy-Wurtz protocol permits the intended `superimpose=true` difference; all other
geometry, DoG, detector, and revision checks remain in force.

This adds a required recipe section and does not retrofit already-running Python
processes. Existing runs retain their strict recipe/revision identity: changing
the recipe or committing this code does not authorize resuming old checkpoints
under the new revision. Preserve existing run artifacts and use matching source
for them; historical archived recipes must be updated explicitly before use with
the new runner.

Edit the recipe, commit the recipe and all code changes, then run:

```bash
/home/jacob/venvs/nvidia/bin/python scripts/mission_control/mission_control.py
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
`<output_path>/translation_only-<YYYYmmDDHHMMSS>-<full-commit-id>/`, containing
`checkpoints/`, `final/`, `provenance/`, `run.json`, and `training_log.jsonl`. The timestamp is the Git
commit's **committer time in UTC**, not the training start time. The commit ID is
all 40 lowercase hexadecimal characters. For example:
`trans_rot-20260930131523-0123456789abcdef0123456789abcdef01234567`.
The same recipe and commit resolve to the same run directory; a new commit gets
a separate directory. Mission control uses this identity for start/resume,
evaluation, reports, and console-log naming. Each run owns its `training_log.jsonl`, inside that commit-specific directory.
Training, background generation, and evaluation all write to this file; its run ID
is the absolute checkpoint-directory path.

Existing recipe-only directories are preserved but are not automatically resumed
or migrated by this naming scheme. The standalone log reporter can still report
their recorded measurements by their original run IDs. Existing shared logs at
the output root are preserved; new runs do not append to them.

The manifest records the exact Git commit and recipe. Resume requires that same
commit and a clean working tree. Do not edit the repository while a run is active.
Before each dataset generation (including background generation), training-set
boundary, validation, and final testing, the runner checks for staged, unstaged,
and nonignored untracked changes and verifies that HEAD has not moved. Any change
stops the run with ERROR. Even documentation-only commits require a new run.
`latest.pt` is a full epoch-boundary resume checkpoint; `previous.pt` is its
optional predecessor. Model-only candidates are saved at the configured epoch
interval and at the last scheduled epoch.
Before publishing checkpoints, training checks model and optimizer tensor health
even when the epoch loss is finite. On a nonfinite loss or state, it discards the
failed epoch, restores the last healthy model/optimizer/scheduler state, and
retries the same epoch once with AMP and TF32 disabled. Before the first checkpoint,
the retry restores the original initialization. The data and shuffle order are
unchanged. If full float32 also fails (or was already in use), training stops and
leaves previous checkpoint files intact. Other errors do not trigger this retry.
Loading checks nested tensor state too,
including optimizer, gradient-scaler, and scheduler tensors when present.

The fallback stays active for all later epochs, datasets, and restarts, even if
the recipe permits AMP/TF32. Actual precision flags are saved in checkpoints;
`<output_path>/<run-directory-name>/training_log.jsonl` records each numerical failure and the precision
switch before retrying. On restart it supplies the persistent fallback policy;
keep this file alongside the checkpoints. The recipe and learning-rate policy
are not changed by recovery.

This is the single training event history, stored as JSON Lines: one complete
JSON object per line, each with `version: 1`. Each event has a UTC `timestamp`,
`event` type, `run_id` (the absolute
checkpoint directory), and process ID. Structured records include epoch loss,
learning rate, precision settings, numerical failures/fallbacks, validation
metrics/selection, test metrics, and fatal-error tracebacks. Console output,
including background generation, is also captured as `console` events while
remaining visible in the terminal. Restarts retain the history; writes are
serialized across processes, appended without rewriting history, and synced to
disk. Readers ignore an unfinished final line left by an interrupted write; the
next writer truncates only that tail and records a `log_tail_recovered` event.
Malformed complete records cause an explicit error when read, not silent recovery.
Do not replace or rotate the file during a run: its file lock coordinates writers.
The obsolete `training_log.json` format is not supported or migrated. If present,
it causes an explicit error; use a fresh output folder or archive the old run.
No separate failure, fallback, validation-progress/results or test-results logs
are produced. Dataset indexes/metadata, `run.json` identity, and source/environment
provenance remain separate because they describe artifacts rather than events.
Standalone analysis and historical benchmark reports are not modified.
Recipe parsing errors before the output path is known, and abrupt process kills
or power loss, cannot produce a final failure entry.

The log is also the central progress/evaluation record. `epoch_started` identifies
each attempt; `epoch_completed` records successful-attempt training seconds
(including loading and health checks), total epoch wall seconds (including retries
and checkpoint saving), checkpoint-save seconds, loader wait, presentations,
batches, throughput, loss, learning rate, and precision. Numerical failures record
their attempt number and elapsed seconds. Timings use a monotonic clock; timestamps
are UTC. An interrupted attempt without completion is not a completed epoch.

Validation and test records contain checkpoint epoch, dataset path/split/generation
seed, sample count, evaluation duration, precision, and explicit metric definitions.
Besides absolute errors and RMSE, they include:

- Heading MAE in degrees and as `100 * MAE / 180` (not percent of true heading).
- Mean normalized dot product (cosine similarity), and `50 * (1 - cosine)` error:
  aligned = 0%, perpendicular = 50%, opposite = 100%. These measure direction only.
- Percentage of samples within 15 degrees; undefined directions count as failures.
- Mean per-sample speed error and Euclidean vector-tip error, each divided by that
  sample's true speed and multiplied by 100. These can exceed 100%.
- Counts excluded from direction or relative metrics because the relevant vector
  length is effectively zero (`<= 1e-12`); undefined aggregates are JSON `null`.

Print a readable terminal report without loading checkpoints or rerunning inference:

```bash
/home/jacob/venvs/nvidia/bin/python scripts/mission_control/backend/report_training_log.py \
  /home/jacob/Documents/dw93net_output/RUN_DIRECTORY/training_log.jsonl
```

Text is the default and creates no files or windows. The report includes completed
and latest epochs, duration in seconds, the best checkpoint and its validation
errors for each dataset, epoch timings/losses, failures/fallbacks, and evaluation
tables. `--run-id /absolute/path/to/checkpoints` selects one run.

Add `--dashboard-gui` to open the graphical dashboard in a dedicated window. Its
`dw92net_trainging_dashboard.html` file lives in a unique system temporary directory,
which is deleted when the window closes, on viewer failure, or on Ctrl+C. The
command waits for the window to close; nothing is saved beside training artifacts.
The GUI requires Linux GJS, GTK 3, and WebKit2 4.1 (available on this machine);
terminal reporting needs only Python's standard library. No network connection is
needed. The old `--text`, `--output`, and `--no-open` options are removed.

The dashboard shows training and validation loss, validation heading/speed/yaw/
translation-vector errors, and epoch durations in seconds. Choose epoch or elapsed
hours, linear or logarithmic loss, absolute or relative speed/vector errors, and
an epoch range. Hover for exact values. Best validation-loss checkpoints are marked.
Separate validation datasets have separate selections; test and sampled progress
evaluations are not mixed into these curves. Only loss is recorded for training;
the error charts show validation measurements. The report is a snapshot: rerun it
to refresh.

Compare measurements only on matching datasets/splits and sample selections.
Validation runs after every epoch; final testing evaluates only the selected model.
Use validation, rather than repeated test-set feedback, when making model-selection decisions.

Validation and test seeds must differ. `training_sets.start_seed` must be strictly
greater than both; the required positive seed increment keeps every training set
outside the held-out seeds, regardless of the number of sets.

Each training seed creates a new DoG dataset at `training_sets.path` without
writing intermediate Flow-XT files. The entire training cache is deleted after
its epochs. The learning-rate schedule and
optimizer continue across replacements. Validation is generated before any
training set, retained permanently, and reused unchanged after every epoch and
across restarts. Validation loss is the mean summed squared motion error normalized
by the fixed motion scales (`normalized_mse`). It drives both the plateau scheduler
and best-checkpoint selection. The best checkpoint is saved even between periodic
candidate saves; ties keep the earlier best. Training stops at the configured
set/epoch limit or immediately when the scheduler reaches its minimum learning
rate, whichever comes first. No additional epoch runs at the minimum rate.
The best-validation checkpoint is promoted to `final/model.pt`. Only then is
the persistent test set generated or
reused at `test_set.path` and evaluates only the final model. A recipe mismatch
never silently overwrites that test set.

Validation results are logged after every successfully completed epoch in
`training_log.jsonl`. Scheduler history, validation metrics, best-candidate identity,
and the minimum-rate stop condition are saved in `latest.pt`. If interrupted,
the runner verifies the Git commit and resumes at the next incomplete epoch;
reaching the minimum learning rate remains terminal on restart. A checkpoint
published just before a logging interruption restores its missing log records;
unrecorded checkpoint-save/wall timings remain explicitly unavailable rather than
being invented. Completed candidates belong to this run and must not be manually
replaced. `validation_completed` records candidate selection and
`test_result` and `duffy_wurtz_1993_result` record the final measurements. Rerunning restarts
only an interrupted epoch and otherwise resumes from the recorded training-set
index and seed unless finalization was requested. Final evaluations reuse results
only when the complete cache identity matches; otherwise they are recomputed.

## Training behavior

The committed recipe enables `[parallel_generation]`. While training runs on
one set, a CPU-only subprocess generates the next in a sibling `training.next`
directory. A successful worker exit and published dataset state are required
before an atomic directory rename makes that set current. Set
`enabled = false` for sequential generation. Omitted settings default to
sequential operation for older recipes.

`min_threads` (default 1) and `max_threads` (default 8) apply to every
stimulus-generation phase. Foreground validation, training-set, and test-set
generation use the maximum while training is idle, restoring the previous
PyTorch thread count afterward. This also applies when `enabled = false`;
that switch controls lookahead, not the thread limits. The obsolete
`full_force_first_set` option is rejected and must be removed from recipes.

The background generator starts at the minimum and adjusts within the same
bounds. Both limits are capped by the available generation CPU pool. The
background worker runs at niceness 10 and requests best-effort low I/O priority.
Its CPU pool excludes a portion of the available CPUs for training; training
itself keeps its existing affinity. Background BLAS and inter-op thread pools
are limited to one. CPU affinity uses logical CPUs, not topology-aware
physical-core allocation. These settings control numerical-operation threads,
not independent stimulus workers.

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
section. `monitor = "validation_loss"` is required; training loss does not control
the learning rate. Scheduler history continues across training sets and restarts.
The initial learning rate must exceed the minimum. Validation and final testing
use full float32 with AMP and TF32 off, keeping measurements comparable when
training falls back from mixed precision. Training retains its configured precision.
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
`optimization.allow_amp` enables training AMP. `optimization.allow_tp32` controls
TF32 matrix multiplication and convolution during training and evaluation.
Both are currently true. Set both false for full float32 arithmetic. Larger
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
/home/jacob/venvs/nvidia/bin/python scripts/archive/evaluate_progress.py \
  --checkpoints /absolute/output/translation_only-YYYYmmDDHHMMSS-FULL_COMMIT_ID/checkpoints \
  --data /absolute/validation/cache/parameter-set/dog
```

The current experiment recovers translation vectors from one-step flow fields
with recipe-defined heading and speed ranges. Heading angular error is the primary
metric; speed and vector errors are tracked as well. Fixed-speed accuracy does
not demonstrate recovery across different speeds. Rotation is deferred until
simple translation recovery works.

The command uses 256 seeded validation examples, CPU inference, and no worker
processes by default. It appends a sample manifest and per-checkpoint metrics,
timings, and bin breakdowns to the run’s `training_log.jsonl` in its commit-specific output
folder (`--log` overrides the path). Each evaluation has a unique session ID, so
different subsets or splits are not silently mixed. Reports and plots are derived
from those logged results; no separate metric JSON files are written.
A `progress.md` report ordered by global epoch is saved under `evaluations/`. Each
invocation recomputes the requested checkpoints in a new dated report directory.
Use `--samples 0` for the whole split, `--split test` for a final held-out
assessment, or `--help` for explicit dataset/checkpoint paths and other options.
Changing sample size or seed creates a separate comparison set. Reports include
heading mean/median error and within-15-degree accuracy, speed MAE/RMSE, and
mean Euclidean vector-tip error (`||[dx,dz]-[dx_hat,dz_hat]||`); logged metrics also
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

The report reads model-only candidates and validates the current checkpoint
schema, matching Git commit, DoG settings, and geometry. Run the report from a
clean checkout of the checkpoint's exact commit. Supply the permanent validation
cache for retrospective progress reports.

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
overrides this default, for example in small tests. `fov_deg` may be greater
than 180° and at most 360°; visibility is selected from the full-circle
`atan2(x, z)` angle, so rear-hemisphere dots are represented correctly.
The recipe pipeline renders each `[2,X]` Flow-XT stimulus in memory. Default
X supersampling is **255**. Patterns remain float32 after antialiasing and after
summing and clipping any superimposed layers, and are passed directly to DoG
filtering without byte quantization. Higher supersampling factors retain their
finer intensity levels to float32 precision. Raw patterns are held in memory
and never written; saved DoG contrast is also float32. There is no raw Flow-XT dataset format or
disk-backed compatibility path.

Exact 360° FOV selects a shared `circular` boundary policy. The renderer uses
`[-180°, 180°)` with no duplicated endpoint and wraps dot footprints across the
seam. DoG and Reichardt preserve all columns; V1/MT local pools and convolutions
also wrap. Adaptive pooling and the decoder are unchanged. Smaller FOVs keep
nonperiodic behavior, including 359°; subsequent nonpanoramic experiments should
use FOVs below 350° because motion across an omitted angular gap is not modeled.

The sole active recipe is `recipe/trans_plus_rot_panoramic.toml`: 360° at 60
pixels/degree (21,600 columns), speed `[0.001, 0.006]`, heading `[-180, 180]`,
and rotation `[-0.5, 0.5]`. Previous recipes live in `recipe/archive/`, which the
single-recipe runner ignores.

`stimuli.apply_dog_filter_after_downsample = true` keeps the usual order:
downsample, then DoG. Set it to `false` to apply DoG at supersampled resolution
before downsampling for FOVs below 360°. Panoramas require `true`; the
high-resolution circular path is unsupported. The high-resolution path scales Gaussian sigmas/support
to preserve their angular size, uses CPU float32 FFT convolution, and processes
one high-resolution stimulus at a time to bound memory. Output width and dtype
remain unchanged. The choice is recorded in dataset geometry and must match
across training, validation, and test. High-resolution filtering is substantially
slower; see the 1023x comparison under `evaluations/dog-order-1023-seed42-1000/`.

Translation is sampled in polar coordinates. The defaults draw speed uniformly
from the half-open interval `[0.001, 0.015)` scene units per frame step and
heading uniformly from `[-180, 180)` degrees. Heading 0 is forward (+Z), -90 is
left (-X), and +90 is right (+X). Cartesian motion is then calculated as
`dx = speed*sin(heading)` and `dz = speed*cos(heading)`. A zero fixed speed is
allowed for rotation-only experiments; positive speeds are required when heading
or translation metrics are meaningful. Motion is sampled once, then `dx`, `dz`,
and yaw are converted to their stored float32 values before rendering. There is
no nano-unit grid or rejection loop. Speed bounds must remain finite and
nonnegative in float32. Half-open ranges describe sampling
before float32 rounding, which can move a stored value slightly across an endpoint.
Samples have sequential numeric IDs; replay uses the same sampling/conversion code.

`dog.json`, not the directory name, records speed, heading, and rotation bounds,
loss-normalization inputs, geometry, the exact half-open interval semantics, seed, dataset role,
coordinate conversion, and stimulus configuration.

Preprocessing applies a **1D luminance DoG independently to each frame**:
center sigma 1 pixel, surround sigma 3 pixels, kernel radius 9 pixels (19 taps).
Both discrete Gaussians are separately normalized to unit sum. Filtering uses
valid support for nonpanoramic FOVs and circular padding for 360°, with no
temporal filtering or per-image rescaling.
The float32 signed contrast is stored as `[2,X-2r]` in valid mode or `[2,X]`
in circular mode. ON and OFF remain
recoverable as `relu(contrast)` and `relu(-contrast)`.

For each training or evaluation batch, `ReichardtBank(dx_values)` computes the
signed product `contrast(0,x-s)*contrast(1,x)` on the model device for every
configured displacement `s`. This equals
`ON_ON + OFF_OFF - ON_OFF - OFF_ON` algebraically, without allocating four
redundant polarity blocks. Negative responses remain available until local
opposite-direction suppression and rectification.

In valid mode, for radius `r`, `left=r+max(0,max(dx_values))` and
`right=r+max(0,-min(dx_values))`, a `[2,X]` or `[1,2,X]` stimulus produces
`[len(dx_values),1,X-left-right]`. The defaults give `[121,1,7062]` at X=7200.
Circular correlations instead have shape `[len(dx_values),1,X]` and use modular
offsets. Positions are anchored in the present frame's original pixel coordinates.
Filtering, cached contrast, and online correlations use float32.

Generation preserves the requested role and stable sample identifiers, and publishes
`dog.json` only after completion. Existing destinations are refused. The cache
format is `dog-contrast-bundles-v3`. `DoGDataset` reads the contrast bundles;
checkpoints retain both DoG and displacement metadata, and evaluation rejects
incompatible caches. Boundary metadata is checked against stimulus geometry.
DoG format `dog-signed-contrast-v2`, correlation format
`signed-reichardt-correlation-v2`, and checkpoint schema `dw93net-checkpoint-v4`
reject older formats; regenerate data and checkpoints rather than migrating.

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

Each `test_*.py` file is independent and may be removed without breaking the
application or remaining tests. Shared recipe fixtures live in
`tests/recipe_helpers.py`; training-engine helpers live in `tests/training_helpers.py`.
Test files must not import each other or leave global settings changed for later
tests. These requirements are recorded in `AGENTS.md`.

Run an individual file using discovery, for example:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_evaluation.py -v
```

## Motion-estimation model baseline

`src/models/flatland_flow.py` defines `FlatlandFlowNet`, which consumes
`[B,D,1,X]` signed correlations and
predicts `[dx,dz,rot]` in scene units per frame step and degrees per frame step.
A `[0,0]` rotation range targets zero on `rot`; nonzero ranges train yaw
estimation. Heading is `atan2(dx,dz)` (zero toward +Z,
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
  and decode their activity to two translation components and yaw. Broad spatial
  integration is motivated by MST physiology; biological tuning is not imposed
  or guaranteed by this learned readout.

Layer comments explain all starting hyperparameters. No dropout or batch
normalization is used. This simplified feedforward model has no binocular
signals, orientation channels, temporal dynamics, recurrence, or eccentricity-
dependent receptive fields. Its mechanisms and estimation accuracy need tests
and ablations; speed scale still relies on the training distance distribution.

Training and evaluation accept structured metadata with fixed or varying yaw.
Checkpoints use one current schema with mandatory fields (including scaler state
for resume); older schemas are rejected. `src/checkpoints.py` checks structure
and the exact Git commit. There are no per-file source fingerprints or selective
code-compatibility rules. Existing runs from the previous schema must be used
with their original code; new runs use the rigid commit policy.
Keep FOV, width, crop, and detector semantics fixed across dataset roles. Pipeline
tests verify implementation, not translation accuracy.

Biological references: [Qian, Andersen & Adelson (1994)](https://doi.org/10.1523/JNEUROSCI.14-12-07381.1994),
[Tsui et al. (2010)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2888240/),
and [Duffy & Wurtz (1991)](https://doi.org/10.1152/jn.1991.65.6.1329).

## Preprocessing provenance and recovery

Every new cache root contains `dog.json`, recording DoG sigmas/support,
normalization and boundary rules, input/output shapes and dtypes, sample count,
UTC start/end times, generation settings, Git commit, and a reference to
the run's source snapshot. A recipe run captures source and environment once in
`<output_path>/<network>/provenance/`; all datasets, including background-generated
sets, reference it. Standalone generation captures once at its output root.
Preserve that provenance directory when archiving datasets separately.

The run's `provenance/` folder contains:

- `working-tree.patch`: combined staged and unstaged tracked changes from HEAD,
  including binary differences and deletions.
- `source.tar.gz`: a standalone snapshot of the working repository files,
  including nonignored untracked source. It does not require the base commit
  to reconstruct the working source.
- `environment.json`: Python/platform details and installed package versions.
- `RESTORE.md`: recovery instructions and archive scope.

Extract the source archive into an **empty directory** to recover the code.
The archive contains tracked and nonignored untracked files, excluding the
current output directory; Git-ignored datasets/environments and Git history
are not bundled. Dependencies are recorded, not packaged. Artifact metadata is
stored in `dog.json`. Git must be available with an existing HEAD commit.
Recipe runs check Git at set boundaries and before generation, training, validation,
and testing. Keep the repository untouched throughout the run. Dataset reuse
requires the same commit, seed, role, sample count, stimulus and DoG settings.
Structural manifest validation checks every bundle
header, row count, shape, dtype, and file size without scanning the full payload.

DoG kernels are cached per device and computed in float32 after float64 kernel
normalization. Filtering uses batches of at most 16, independent of bundle size.
The kernel radius is always `ceil(3*sigma_surround)`.

Runtime dependencies are declared only in `requirements.txt`; `setup.py` reads it.
