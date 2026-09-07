# Cline: start here

This repository is an early PyTorch research prototype from May 2025. It generates binary XT optic-flow stimuli, applies a frozen bank of bilocal/bitemporal 2×2 kernels, and trains a convolutional regressor to estimate camera `dx`, `dz`, and rotation.

The immediate goal is understanding and decision support. Jacob has not decided whether to preserve and refactor this implementation or restart from validated ideas.

During development, hardware responsibilities are fixed: the Radeon AI PRO R9700 runs local Qwen, and the GeForce RTX 3080 Ti runs dw93net through the CUDA-only `~/venvs/nvidia` environment. A later, explicitly approved training phase may stop Qwen and move training/evaluation to the R9700 using the separate ROCm environment.

## First Cline task

Use Plan mode and paste:

> Perform a read-only forensic review of this repository. Read `CLINE_START_HERE.md`, `.clinerules/01-dw93net.md`, every source file, and the Git history. Explain what experiment the code appears to implement and trace data from generation through evaluation, including tensor shapes and units. Identify what is demonstrated versus merely intended, current blockers on this Linux system, likely correctness or reproducibility risks, and questions only I can answer. Respect the GPU boundary: Qwen uses the R9700 and dw93net development uses the RTX 3080 Ti with `~/venvs/nvidia`. Then compare continuing as-is, incremental refactoring, and a clean restart. Recommend the smallest experiments needed before choosing. Do not edit files, install packages, run training, or commit anything.

## Known observations to verify independently

- `README.md` is empty and there are no automated tests.
- `src/data/generate_flow_xt_data.py` generates train/validate/test `.pt` samples whose motion labels are encoded in filenames.
- `src/features/bix_bit_kernel_bank.py` constructs 320 frozen convolutions by default.
- `src/models/flow_xt_net.py` regresses three values after global average pooling.
- Training and evaluation scripts contain Windows-specific paths.
- `requirements.txt` pins Torch 2.2.2 but omits scikit-learn, which the evaluation script imports.
- Generated plots and some Python bytecode are present in Git history.

These are orientation notes, not conclusions. Inspect the code and history directly and flag disagreements.

## Local Qwen connection

In Cline select **OpenAI Compatible** and set:

- Base URL: `http://127.0.0.1:8080/v1`
- Model ID: `qwen-local`
- API key, if the UI requires one: `local`
- Context window: `65536`
- Maximum output: `4096` initially

Use the same model for Plan and Act. Keep broad auto-approval disabled during discovery.

## dw93net development interpreter

VS Code is configured to use `/home/jacob/venvs/nvidia/bin/python`. Before running dw93net on a GPU, run the **dw93net: verify NVIDIA environment** task. A failed check is a blocker to diagnose, not permission to switch to the AMD environment.
