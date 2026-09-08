# dw93net Forensic Review

**Date:** 2026-09-07
**Branch / HEAD:** `codex/setup-cline-qwen` @ `e900064`
**Scope:** full-text + Git-history audit of the forward stimulus pipeline, inverse estimator, data/label contract, environment, and the training-path import mismatch.
**Conventions:** *Observed* facts are cited to `file:line` or commit; *Inference* is labeled as such.

---

## 1. Confirmed defect: import/function-name mismatch in the training path

`src/train/train_flow_xt_net.py` imports a function that no longer exists in the data module.

**Observed (current tree):**
- `src/train/train_flow_xt_net.py:7`
  `from data.load_flow_xt_data import FlowXtDataset, extract_range_from_foldername, subset_dataset` (singular)
- `src/train/train_flow_xt_net.py:30`
  `ranges = extract_range_from_foldername(os.path.basename(cfg.root_folder))`
- `src/data/load_flow_xt_data.py:70`
  defines only `def extract_ranges_from_foldername(foldername: str)` (plural). No singular name exists anywhere in the module.
- Consequence: `import train_flow_xt_net` fails with `ImportError` at line 7; the training script is fully blocked.
- `src/train/test_flow_xt_net.py` does not import this function, so only the train script is broken.

**Origin, traced through history (observed, via `git log -S` and per-commit `grep`):**

| Commit | Train script import | Data module definition | Consistent? |
|---|---|---|---|
| `17a57bc` | (no train script yet) | (no load module yet) | — |
| `b1df1a5` | no range import | no load module | — |
| `211903f` | no range import | — | — |
| `f63a649` | `extract_range_from_foldername` (line 7, line 31) | `extract_range_from_foldername` (line 63) | ✅ matched |
| `3fe55ff` | `extract_range_from_foldername` (unchanged) | **renamed to** `extract_ranges_from_foldername` (line 70) | ❌ **broken here** |
| `e9b0830` → HEAD | still singular | still plural | ❌ still broken |

**Inference:** commit `3fe55ff` ("work in progress", the superimpose commit) renamed the data-module function from singular to plural but did not update the train script. `git log -S 'extract_ranges_from_foldername' -- src/train/train_flow_xt_net.py` returns nothing — the plural name was never written into the train script. The bug persists in every commit since, on both `main` and this branch.

### 1.1 Correct minimal fix

Change the **train script** to use the existing plural function (2 lines: import + call):

```python
# src/train/train_flow_xt_net.py
# line 7:
from data.load_flow_xt_data import FlowXtDataset, extract_ranges_from_foldername, subset_dataset
# line 30:
ranges = extract_ranges_from_foldername(os.path.basename(cfg.root_folder))
```

Rationale:
- The data module's plural name is the current canonical API; its return keys (`dx`, `dz`, `rot_deg` in metres/degrees) already match what the train script consumes — no interface change needed.
- Renaming the data module function back would be the larger, riskier change and regresses the (slightly better) naming.
- No other file imports the singular name, so nothing else is coupled.

### 1.2 Validation plan for the fix

A bare `import train_flow_xt_net` from `src/` is insufficient as validation because importing executes the whole script, including dataset construction from `F:/...` paths that do not exist on this machine (see §2B). Use:

1. `python -m py_compile src/train/train_flow_xt_net.py` (syntax).
2. Targeted check from `src/`: import `data.load_flow_xt_data` and assert `hasattr(..., 'extract_ranges_from_foldername')`; AST-parse the train script and assert the singular name no longer appears.
3. Full end-to-end validation requires the environment work in §2B/§4 (Linux paths, missing deps, smoke test).

---

## 2. Additional audit findings

### 2A. `r2_score` call in the test script — **verified correct, no action**

`src/train/test_flow_xt_net.py:79-81`:

```python
reg  = LinearRegression().fit(true_vals, pred_vals)   # fits true -> pred
fit  = reg.predict(true_vals)
r2   = r2_score(pred_vals, fit_vals)                  # y_true=preds, y_pred=fit
```

`r2_score(y_true, y_pred)` uses the network predictions as actuals and the regression line as predictor — the correct convention (R² of how well the fit line explains prediction variance). **No bug; no change required.** (An earlier draft of this review flagged this as possibly swapped; it was re-verified character-by-character and cleared.)

### 2B. Windows paths and interactive I/O block any local run (observed)

- `src/train/train_flow_xt_net.py:12` `root_folder = "F:/dw93net/data/udx=-10000,10000_udz=-10000,10000_rot=0,0"`; line 13 `checkpoints_dir = "F:/dw93net/checkpoints"`.
- `src/train/test_flow_xt_net.py:248-260` — `data_root = "F:/dw93net/data"`; `select_subfolder(...)` (interactive prompt) for both data and checkpoint folders.
- `src/data/generate_flow_xt_data.py:207` — `output_root="F:/dw93net/data"`.
- On this Linux box, only `data/udx=-10000,10000_udz=-10000,10000_rot=0,0/` exists (tracked in git, 100k samples; folder name has no `_superimposed`). No `checkpoints/` directory exists locally.
- **Inference:** the evaluation plots committed in `e9b0830` (`src/train/*.png`) were produced on the Windows machine with untracked data/checkpoints; their underlying data is not in this repo.
- `src/train/test_flow_xt_net.py:42` — test subset uses `method="random"` **without a seed** → the 1000-sample evaluation subset is not reproducible across runs.

### 2C. Generator mutable default argument (latent bug)

`src/data/generate_flow_xt_data.py:15` — `def generate_flow_xt_stim(..., cam_trans_xz=[0, 1e-3], ...)`, mutated in place at line 37 (`cam_trans_xz[0] += dx`). Safe today only because every real caller passes `cam_trans_xz` explicitly; a future bare call would accumulate state. Fix: default `None`, build the list inside.

### 2D. Environment gaps (observed, verified live 2026-09-07)

- `/home/jacob/venvs/nvidia/bin/python`: `torch 2.11.0+cu128` (CUDA 12.8), `torch.version.hip = None`, single visible device **NVIDIA GeForce RTX 3080 Ti** — preflight passes.
- **Missing in that venv:** `matplotlib` and `scikit-learn`, both imported by `src/train/test_flow_xt_net.py:3-6` → the evaluation script cannot run until they are installed.
- `requirements.txt` pins `torch==2.2.2` (stale vs. installed 2.11.0).
- `setup.py` uses `find_packages(where='src')`, which finds zero packages (all `__init__.py` files live *under* `src/`, e.g. `src/data/__init__.py`), so `pip install -e .` installs nothing. The codebase actually runs by executing from inside `src/` with implicit cwd imports; the editable install is a no-op. Not worth fixing now; just don't rely on it.
- Qwen local server is up (container on 127.0.0.1:8080); per the GPU boundary rule, dw93net work stays on the 3080 Ti. Confirmed it is the only visible CUDA device.

### 2E. Stale documentation (cosmetic)

- `src/features/bix_bit_kernel_bank.py` docstring: "16 base kernels", "x_gap_range (0-2)", "t_gap_range (0-2)", "256 output channels" — actual defaults are `range(0, 10)` × `range(0, 2)` → **320** output channels.
- `src/models/flow_xt_net.py` docstring: defaults `range(0, 5)` for both gap ranges — actual `range(0, 10)`, `range(0, 2)`.

### 2F. Training-loop characteristics (flagged, not defects per se)

- `num_epochs = 1_000_000` with no early stopping; saves `epoch_XXXXXXXX.pt` **every epoch** → unbounded disk growth (each checkpoint = model + optimizer state). Recommend a checkpoint-interval setting.
- Val metric printed as "Val RMSE" is `sqrt(sum of range-normalized MSEs)` — a normalized error, not a physical RMSE. Well-defined and consistent between train/val/best-selection, but the name is slightly misleading.
- Checkpoint layout changed in `3fe55ff` from `checkpoints/flow_xt_net/<data>/` to `checkpoints/<data>/`. Any Windows-era checkpoints in the old layout won't be found by current code.
- `src/train/test_flow_xt_net.py:36` rebuilds `FlowXTNet()` with default gap ranges; gap ranges are not stored in the checkpoint `settings`, so a checkpoint trained with non-default gaps would load into the wrong architecture silently.

### 2G. Data/label contract (verified consistent)

- Forward: generator → `tensor_image` `[T=128, X=1028]` float {0,1} → `torch.tensor(..., dtype=torch.float32).unsqueeze(0)` → `[1, 128, 1028]` saved as `flow_udx=<µm>_udz=<µm>_rot=<deg>[_superimposed]_id=<i>.pt`.
- Inverse: `FlowXtDataset` loads `[1, T, X]`; label from filename via `parse_camera_motion_from_filename` (µm → m by `/1e6`; rot in degrees). Units consistent with `Settings` and the loss normalization (ranges from `extract_ranges_from_foldername` are in metres/degrees).
- Minor: `src/data/visualize_flow_xt_data.py:24-26` names the variable `xt_tensor_uint8` though tensors are float32 — cosmetic.
- The tracked data folder predates superimpose (no `_superimposed` in folder name; committed at `f63a649`). Current generator defaults (`superimpose=True`, `rot_deg_range=(-0.5, 0.5)`) would produce a *different* folder. Regenerating data is a scientific decision.
- Stray Windows artifact: `data/udx=-10000,10000_udz=-10000,10000_rot=0,0/Jacob - Personal - Shortcut.lnk` — harmless (dataset filters to `.pt`), deletable.
- **Inference (superimpose subtlety):** in `generate_flow_xt_stim` with `superimpose=True`, the rotation layer has zero translation, so its dots are never re-randomized/respawned. Over 128 frames at up to 0.5°/frame (up to 64° total), some dots rotate out of the visible z-window, slightly reducing the rotation signal over the sequence, while the translation layer respawns per frame. If equal per-frame signal strength is desired in both superimposed layers, the rotation layer should re-randomize dots each frame too.

---

## 3. Three paths compared

| | A. Continue as-is | B. Refactor incrementally (**recommended**) | C. Restart, preserving ideas |
|---|---|---|---|
| **Evidence** | A 2-line fix unblocks the core loop; pipeline shapes/units verified consistent | Same fix + small robustness fixes keep the validated design (kernel bank, range-normalized loss, filename label contract) intact | No evidence the design is wrong; kernel bank + XT-slice regression are the validated ideas worth keeping. A rewrite costs the existing 100k dataset contract and all Windows-era results for no scientific gain |
| **Risk** | Latent bugs (§2B–§2F) will bite at run time; Windows paths make local runs impossible without edits anyway | Low; each fix is isolated and individually verifiable | High cost, low upside; loses continuity of the research record |
| **Smallest distinguishing experiment** | 200-sample, fixed-seed generation → 3-epoch training → 100-sample eval, end-to-end on the 3080 Ti. If loss decreases and eval runs, A/B is viable; if shapes/labels mismatch, it surfaces immediately | Same | Same, but from a new skeleton |

**Recommendation: B (incremental refactor).** The design is sound and verified; the defects are localized and cheap. Do not restart.

---

## 4. Proposed plan (pending Jacob's approval)

**Scope — branch `codex/setup-cline-qwen` (or a new `fix/train-import` off it; Jacob's choice):**

1. **Fix the import mismatch** in `src/train/train_flow_xt_net.py` (lines 7 and 30 → plural `extract_ranges_from_foldername`). Validate per §1.2.
2. **No change** to the `r2_score` call (verified correct, §2A).
3. **Make the scripts locally runnable** (Linux paths, no interactive prompts):
   - `train_flow_xt_net.py`: `Settings.root_folder` / `checkpoints_dir` default to `/home/jacob/Documents/dw93net/data/...` and `/home/jacob/Documents/dw93net/checkpoints`, overridable via argv.
   - `test_flow_xt_net.py`: accept `data_folder` and `checkpoint_path` as argv (keep `select_subfolder` as fallback only when no args).
   - `generate_flow_xt_data.py`: `output_root` default → `/home/jacob/Documents/dw93net/data`.
   - Also fix the mutable default `cam_trans_xz` (§2C).
   - Portability changes only — no scientific meaning, units, or conventions change.
4. **Environment:** `pip install matplotlib scikit-learn` into `/home/jacob/venvs/nvidia`; note `requirements.txt` torch pin is stale (update to `>=2.11` or leave — Jacob's call). Re-run the GPU preflight before any CUDA work (currently passing: 3080 Ti only, CUDA build, Qwen up on the R9700).
5. **Smoke test (smallest distinguishing experiment):** generate a 200-sample dataset with a fixed seed into a scratch folder (superimpose per current defaults), train 3 epochs (batch 12, on 3080 Ti), run evaluation on a 100-sample seeded subset; report exact commands, seed, paths, and results. **No** full 100k regeneration and **no** million-epoch run without Jacob's go-ahead.
6. **Deferred (list only, no action without approval):** checkpoint save-interval; storing gap ranges in checkpoint `settings`; docstring corrections (§2E); reproducible test-subset seed; superimpose rotation-layer respawn (§2G); `requirements.txt` / `setup.py` cleanup; deleting the `.lnk` artifact.

**Success metrics:** import fix validated statically (§1.2); smoke test shows train loss decreasing over 3 epochs and the eval script producing finite per-label errors; all commands, seeds, and paths recorded as evidence.

---

## 5. Detailed per-file findings

### 5.1 Model — `src/models/flow_xt_net.py`

- Architecture (lines 37-65): frozen `BixBitKernelBank` → `Conv2d(320→64, k=3, pad=1)` + `BatchNorm2d` + `ReLU` + `Dropout(0.5)` → `Conv2d(64→64, k=3, pad=1)` + `BatchNorm2d` + `ReLU` → `AdaptiveAvgPool2d(1)` → `Flatten` → `Linear(64→3)`.
- Kernel bank frozen in `__init__` (`p.requires_grad = False`, lines 43-44).
- Learnable parameter count ≈ 222k (320·64·9 ≈ 184k + 64·64·9 ≈ 37k + 64·3 ≈ 200). Small by design.
- **Inference:** global average pooling collapses both T and X, so the network has no access to *where in the sequence* motion occurs (e.g., a rotation onset frame vs. a steady flow). Acceptable for a 3-output regression, but worth knowing if accuracy plateaus.
- Docstring stale (see §2E): claims gap ranges `range(0, 5)` × `range(0, 5)`; actual defaults `range(0, 10)` × `range(0, 2)` → 320 kernel channels.
- `forward()` assumes input `[B, 1, T, X]`; no shape validation — a `[B, T, X]` sample would fail deep inside the kernel bank, not at the boundary.

### 5.2 Kernel bank — `src/features/bix_bit_kernel_bank.py`

- 16 binary 2×2 base kernels (all bit patterns, `itertools.product([0,1], repeat=4)`) × `len(x_gaps)=10` × `len(t_gaps)=2` → **320** frozen dilated `Conv2d` layers (`bias=False`, `padding=dilation`, lines 46-58).
- With kernel size 2, dilation d, padding d: `out = in + d`. Each detector therefore outputs `T+d_t × X+d_x`; `forward` trims all responses to the minimum size (lines 66-68) before concatenating, restoring exactly `128 × 1028`. **Verified consistent** with the model's expected input.
- `forward` runs 320 separate conv calls in a Python list comprehension (line 64) — this is the compute hotspot of the whole pipeline. Since the bank is frozen and deterministic, its output could be (a) precomputed and cached once per sample, or (b) collapsed into a single grouped/dilated convolution. Performance optimization only, not a correctness issue.
- Docstring claims "256 output channels" and gap ranges (0-2) — stale (see §2E); actual is 320 with x gaps 0-9, t gaps 0-1.

### 5.3 Forward stimulus generator — `src/data/generate_flow_xt_data.py`

- World model: `n_world_dots` dots in the unit square `[-1, 1]²`. Every frame the camera applies the **full** `cam_trans_xz` translation and the **full** `cam_rot_rad` rotation (lines 160-178). Consequently the labels are **per-frame** quantities: `dx`, `dz` are m/frame and `rot_deg` is deg/frame. Total displacement over 128 frames can reach `128·dx` (≈ 1.28 of the 2-unit-wide square at the 0.01 m/frame bound) and total rotation `128·rot_deg` (up to 64° at the 0.5 deg/frame bound).
- **Respawn behavior (refines §2G):** a dot is re-randomized **only when it exits the square on an axis** — it wraps by ±2 on that axis and the *other* axis coordinate is redrawn uniformly (lines 163-171). It is not a per-frame respawn. The rotation-only layer (zero translation) never wraps, so a dot rotated out of the `z ∈ (near_clip, far_clip)` band or the frustum (lines 181-183) stays culled for the remainder of the sequence — the asymmetry flagged in §2G.
- **Reproducibility gap (new finding):** `generate_flow_xt_dataset(seed=...)` seeds only label sampling and split permutation through `np.random.default_rng(seed)` (line 63), but `generate_flow_xt_stim` draws dot positions from the **global** `np.random` (line 157), which is never seeded from `rng`. Same `seed` ⇒ same labels and splits, but *different* dot fields. A seeded smoke test must therefore also call `np.random.seed(...)` before generation (or pass the rng into the stim function — a change requiring Jacob's approval).
- `far_clip = cos(half_fov_rad)` (line 155): at fov 90° the far plane is 0.707; with `near_clip = 0.05` only the band `z ∈ (0.05, 0.707)` of the unit square contributes — the effective sampling volume is a thin slice.
- Projection is angular (`theta = arctan2(x, z)`, line 188) with no depth scaling; multiple dots landing on the same pixel merge via binarization (lines 113-114).
- Superimpose mode: layer 0 = rotation only (500 dots), layer 1 = translation only (500 dots); the two tensors are summed and re-binarized (lines 88-114). The label encodes the full `(dx, dz, rot_deg)` of the combined stimulus.
- `__main__` (lines 205-208) uses the Windows path `F:/dw93net/data` and `superimpose=True`; `n_samples`/`dx_range`/etc. fall back to the function defaults (100k samples, ±0.01 m, ±0.5 deg).
- Mutable default `cam_trans_xz=[0, 1e-3]` (line 138) — see §2C.

### 5.4 Data loader — `src/data/load_flow_xt_data.py`

- `FlowXtDataset.files` comes from `os.listdir` **without sorting** (line 19) — order is arbitrary and platform-dependent. Consequences:
  - `subset_dataset(method="first")`, used for the 10k validation set (train line 18), selects a **non-deterministic** set of files across runs/machines.
  - `method="random"` uses the global `random` module, unseeded (line 117).
- `parse_camera_motion_from_filename` (lines 40-66) cross-checks correctly against the generator's filename format (`generate_flow_xt_data.py` line 119): `udx`/`udz` are micro-units → /1e6, `rot` is float degrees. The returned `'superimposed'` flag is computed but never used by the dataset.
- `extract_ranges_from_foldername` returns `0.0` for a missing key (line 81), and the train script **skips any loss term whose range is 0** (train lines 78-83, 100-106). Consequence for the tracked data folder `rot=0,0`: the rotation output is **never trained** on that data — rotation evaluation plots for it would reflect an untrained head. Not a bug, but must be kept in mind when interpreting results.
- `torch.load(filepath)` with no `weights_only` argument — the files are plain tensors, so this is safe under PyTorch 2.x defaults.

### 5.5 Train script — `src/train/train_flow_xt_net.py` (beyond the §1 defect)

- The entire script executes at import (module-level code, no `main()` guard) — data paths, checkpoint dirs, and `os.makedirs` (line 37) happen as a side effect of importing.
- Loss: per-term MSE normalized by the folder-encoded range squared, terms with range 0 skipped (lines 77-83, 100-106). For the tracked `rot=0,0` data this means the loss covers only `dx` and `dz` (§5.4).
- Resume logic (lines 55-63): loads `_latest.pt` (model + optimizer + epoch) unless `from_scratch`; `best_val_loss` is restored from the latest checkpoint's `val_rmse` (line 60) — consistent with the best-checkpoint selection at line 128.
- Checkpointing (lines 116-131): writes `epoch_NNNNNN.pt` **and** `_latest.pt` every epoch, plus `_best.pt` on improvement — unbounded disk growth at 1M epochs (see §2F).
- Checkpoint `settings` = `asdict(cfg)` (line 122) — this is how the test script recovers `batch_size` (test line 33). The normalization gap ranges are *not* stored (see §2F).
- Validation subset = "first" 10k — non-deterministic because of the unsorted `os.listdir` (§5.4).
- `torch.cuda.empty_cache()` every epoch (line 133) — minor overhead, harmless.
- Device selection (line 22) is plain `torch.cuda.is_available()` — no device-name preflight as the collaboration rules require for local runs.

### 5.6 Test/eval script — `src/train/test_flow_xt_net.py` (beyond §2A/§2B)

- Analysis plots: per-label regressions with R² (`r2_score` usage verified correct, §2A), heading-error-vs-heading with signed error via 2D cross product (lines 122-128), speed error, rotation error wrapped to ±180° (line 188), and heading bias vs. rotation.
- **Unit mismatch in plot labels (new finding):** `analyze_speed_error_vs_speed` labels its axes "m/s" (lines 167-168), but `dx`/`dz` are m/frame per the generator (§5.3). The rotation plot correctly says "deg/frame". Speed-axis labels should read m/frame.
- `heading = arctan2(dz_true, dx_true)` (line 112); the plot's axis caption "90 = left" (line 134) depends on the camera-frame convention — worth an explicit confirmation from Jacob rather than an assumption.
- `main()` hard-codes `F:/dw93net/data` and `F:/dw93net/checkpoints` with interactive `select_subfolder` (lines 248-259); the checkpoint used is always `_best.pt` (line 260).
- Rebuilds `FlowXTNet()` with default gap ranges (line 36) and never verifies the checkpoint's settings — see §2F.
- Plots are saved to the CWD as `*.png` (e.g., line 139) — do **not** send these files to the local Qwen endpoint (text-only).

### 5.7 Other files

- `src/data/show_flow_xt_data.py:5` — imports `from load_flow_xt_data import ...` with **no `data.` prefix**, so it only works when CWD is `src/data`; inconsistent with the `data.load_flow_xt_data` style used by the train/test scripts. Also the `xt_tensor_uint8` misnomer (see §2G).
- `src/data/utils.py` — `select_subfolder` interactive helper is functionally fine; the `str | None` return annotation (line 3) requires Python ≥ 3.10.
- `setup.py` — `find_packages(where='src')` with `package_dir={'': 'src'}`; `install_requires` lists numpy/matplotlib/torch but **not** `scikit-learn` (imported by the test script) — see §2D.
- `requirements.txt` — contains `-e .` (no-op per §2D), the stale `torch==2.2.2` pin, and a Windows-era `spyder` entry; `scikit-learn` is missing.

### 5.8 Label and unit conventions (consolidated — to be confirmed with Jacob)

- `dx`, `dz`: per-frame camera translation, **m/frame**. `rot_deg`: per-frame rotation, **deg/frame**.
- Folder name encodes the sampling ranges in micro-units (`udx`/`udz`) and degrees; filenames encode per-sample `udx`, `udz` (int) and `rot` (float degrees). Generator and loader agree on this encoding (verified line-by-line, §5.3/§5.4).
- Zero-range terms are excluded from the loss, so on `rot=0,0` data the network's `rot_deg` output is untrained (§5.4).
- The test script's "m/s" plot labels contradict the m/frame convention (§5.6).

---

*Document generated by Cline from a full read of all source files, `git log` / `git diff` / `git show` across all 13 commits, and live environment checks. No binary artifacts (PNGs) were opened or interpreted.*
