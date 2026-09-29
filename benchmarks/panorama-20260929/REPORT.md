# Panoramic GPU memory preflight — 2026-09-29

The production geometry (21,600 columns at 60 px/degree), 241 detector channels
(-120 through +120), and batch size 64 passed one training batch on the NVIDIA
GeForce RTX 3080 Ti. Resolution and recipe batch sizes remain unchanged.

| Mode | Peak allocated | Peak reserved | Free before run | Result |
| --- | ---: | ---: | ---: | --- |
| AMP / TF32 enabled | 9,242.06 MiB | 9,274 MiB | 9,497.88 MiB | Passed |
| Float32 fallback / TF32 disabled | 9,242.06 MiB | 9,274 MiB | 9,506.00 MiB | Passed |

Each fresh process constructed the circular model and bank, computed online
float32 correlations from synthetic signed DoG contrasts, then ran forward,
backward, and an Adam optimizer step. The production pipeline computes DoG on
CPU during generation; this probe starts at the cached contrast input. Peak
memory therefore measures GPU training, not renderer throughput or host RAM.

These are one-batch feasibility checks, not throughput or accuracy benchmarks.
Only about 224–232 MiB separated peak reserved memory from initially free memory.
Other GPU applications, persistent training state, or allocator behavior may
consume that margin during a longer run. If that occurs, explicitly reduce the
batch size (for example to 32) and repeat the preflight; keep 21,600 columns.

The probe was run against the implementation working tree based on commit
`3c21b01`. CUDA required execution outside the desktop filesystem sandbox.
Raw results are in `amp.json` and `float32.json`.

Reproduce from the repository root:

```bash
PYTHONPATH=src /home/jacob/venvs/nvidia/bin/python scripts/preflight_panorama.py --batch-size 64
PYTHONPATH=src /home/jacob/venvs/nvidia/bin/python scripts/preflight_panorama.py --batch-size 64 --float32
```
