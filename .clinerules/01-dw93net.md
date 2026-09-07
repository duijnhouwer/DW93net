# dw93net collaboration rules

Jacob owns the scientific question, architecture, priorities, and success criteria. Cline investigates and implements approved work.

## Begin with discovery

- Read `CLINE_START_HERE.md`, the full Git history, and all source files before recommending a direction.
- Treat the current repository as a research record. Do not delete, rewrite, or broadly refactor it during discovery.
- Separate observed facts from inference. Cite file paths and relevant symbols or commits.
- Identify the intended experiment, data flow, tensor shapes, training objective, evaluation method, dependencies, hard-coded assumptions, reproducibility gaps, and likely defects.
- Preserve the architectural distinction between the forward stimulus generator (x-z translation and rotation phi to XT-flow plots) and the inverse estimator (XT stimuli to `dx`, `dz`, and `rot_deg`). Review each separately, then review the saved tensor/filename-label contract connecting them.
- Compare three paths: continue as-is, refactor incrementally, or restart while preserving validated ideas. Give evidence, risks, and the smallest experiment that would distinguish them.

## Working agreement

- Start substantive tasks in Plan mode. State the goal, scope, proposed architecture, success metrics, affected files, and verification plan. Wait for Jacob to approve those decisions before editing.
- Once a plan is approved, carry out routine implementation and verification without repeatedly asking about minor details.
- Never change scientific meaning, label units, coordinate conventions, data generation assumptions, architecture, dependencies, or acceptance criteria silently.
- Do not claim success from training loss alone. Preserve raw measurements and report the exact command, configuration, data subset, seed, hardware, and result used as evidence.
- Work on a task branch. Keep commits small and meaningful. Inspect the diff and run relevant checks before each commit. Do not push or merge unless Jacob asks.
- Preserve unrelated changes. Never discard work or rewrite Git history without explicit instruction.

## Current environment

- This is an older Python/PyTorch prototype originally run with Windows `F:/dw93net/...` paths.
- The local model endpoint is OpenAI-compatible at `http://127.0.0.1:8080/v1`; model ID `qwen-local`.
- Start or stop it with `bash /home/jacob/Documents/MyLCA/assistant start` and `bash /home/jacob/Documents/MyLCA/assistant stop`.

## GPU and Python environment boundary

- During development, the AMD Radeon AI PRO R9700 is reserved for the local Qwen server and its context. Do not use it for dw93net computation while Qwen is loaded.
- During development, dw93net must use the NVIDIA GeForce RTX 3080 Ti through CUDA and `/home/jacob/venvs/nvidia/bin/python`.
- Never activate `/home/jacob/venvs/amd` or install a ROCm PyTorch build into the dw93net development environment.
- Before any GPU-backed dw93net run, verify `torch.version.cuda` is set, `torch.version.hip` is `None`, CUDA is available, and the selected device name contains `RTX 3080 Ti`. Stop and report the mismatch if any check fails.
- Do not change GPU visibility variables or rely on automatic device selection to work around a failed preflight.
- A later full training/evaluation phase may move dw93net to the R9700 only after Jacob approves the phase change and the Qwen service is stopped. That phase requires a separate ROCm execution plan and fresh device verification; do not reuse the development environment in place.
