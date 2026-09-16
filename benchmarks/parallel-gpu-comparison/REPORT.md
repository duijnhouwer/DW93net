# Parallel stimulus generation benchmark

The RTX 3080 Ti run used the motherboard's PCIe Gen5 x16 slot (slot placement
reported by the user). The GPU itself supports PCIe Gen4 x16; a Gen5 slot does
not make this card operate at Gen5 speed.

The requested follow-up uses the Radeon AI PRO R9700 through the PCIe Gen4 x4
slot. Linux confirms a 16 GT/s, four-lane upstream link at `0000:00:1c.4` and
`0000:07:00.0`. Its downstream GPU link reports Gen5 x16 behind a bridge;
that downstream reading does not remove the upstream Gen4 x4 bottleneck.

The NVIDIA result indicates a GPU-bound workload, with very little loader
wait and ample spare VRAM. More VRAM capacity alone is therefore unlikely to
improve this configuration. GPU utilization measures kernel-active time,
not arithmetic-unit occupancy, so these measurements do not distinguish
compute, memory-bandwidth, and power/thermal limits.

The R9700 comparison completed successfully. Its installed runtime is PyTorch
`2.13.0+rocm10.0.0` (HIP `7.15.26333`), versus PyTorch `2.11.0+cu128` for the
3080 Ti. Results compare these complete hardware/software configurations,
not GPU architecture alone.

Completed successfully on 2026-09-16, commit `fcf1661546ff208990df14e035f833def2077fb3`.
Hardware: 28 logical CPUs, approximately 30.6 GiB system RAM, NVIDIA RTX 3080 Ti
with 12 GiB VRAM. Total elapsed time including monitoring shutdown: **736.49 s
(12 min 16 s)**; the training runner reported completion at 735.69 s.

## Configuration

- Two training sets of 20,000 original stimuli, seeds 1000 and 1001.
- 25 epochs per set, 50 global epochs; mirroring gives 40,000 presentations per
  epoch and 2,000,000 presentations overall.
- 2,000 validation examples, evaluating all 50 candidate checkpoints.
- 10,000 test examples, evaluated only with the selected checkpoint.
- Batch size 64, mixed precision, eight loader workers.
- Background generation enabled, dynamically bounded to 1–8 threads.

## Timing and overlap

| Operation | Time |
| --- | ---: |
| First 20,000-example generation | 47.69 s |
| Background second-set generation, including process startup | 43.52 s |
| Training period, including set transition and checkpoint work | about 617 s |
| Validation-set generation | 3.82 s |
| All 50 validation evaluations and selection | 32.41 s |
| Test-set generation | 30.89 s |
| Final test evaluation | 2.94 s |
| Generation wait at training-set switch | **0.00 s** |

Background generation started at 48.383 s and published its complete dataset
at 91.899 s, during epoch 4. Training switched sets at 356.406 s, so the next
set was ready approximately 264.5 s before it was needed. The generation time
was fully overlapped with training; overlapping times must not be added to
the total duration.

The controller increased allocation from one to eight threads between
48.928 s and 69.744 s. Reported loader-wait fractions at these increases were
0.3–0.4%. No pressure-triggered decreases occurred in this workload, so this
benchmark validates the increase path but does not exercise throttling under
CPU contention.

Mean reported training throughput was **3,255.2 presentations/s** (3,263.9
for set 1 and 3,246.5 for set 2). Mean loader wait was **0.0654 s per epoch**;
maximum was 0.09 s. Overlap epochs 2 and 3 ran at 3,339.2 and 3,332.4
presentations/s; immediately subsequent epochs 5–7 ran at 3,315–3,321.
There is no visible contention penalty in this run. This is a within-run
comparison, not a controlled sequential-versus-parallel speedup measurement;
warmup, temperature, and clock changes can affect throughput.

## Resource measurements

719 samples were recorded approximately once per second. All values below
are system-wide, including other applications. CPU percentage is normalized
across all 28 logical CPUs. RAM used is MemTotal minus MemAvailable, so
reclaimable cache is not counted as unavailable memory.

| Phase | Mean CPU busy | Mean GPU utilization | Peak RAM used | Peak VRAM |
| --- | ---: | ---: | ---: | ---: |
| Startup and initial generation | 71.9% | 30.5% | 6.62 GiB | 1,559 MiB |
| Training plus background generation | 30.8% | 95.2% | 8.65 GiB | 3,823 MiB |
| Set 1 after background generation | 7.3% | 98.1% | 8.55 GiB | 4,163 MiB |
| Set 2 training | 7.0% | 98.6% | 8.58 GiB | 4,245 MiB |

Overall peaks: CPU busy 73.8%, RAM 8.65 GiB, VRAM 4,245 MiB (4.15 GiB),
GPU power 349.65 W, GPU temperature 83°C. Swap usage remained zero.
Mean system I/O wait was 0.009%, maximum 0.277%.

The active NVMe device (`nvme1n1`) recorded approximately 24.69 MiB read
and 6,960.24 MiB written between the first and last samples. These are
physical-device counters, not dataset byte counts: filesystem caching,
checkpoint writes, and other applications contribute. Pending writes after
the last sample are not included. GPU activity during CPU-only generation
also reflects other system activity; it is not attributed to the generator.

## Model results

Selected checkpoint: **epoch 47**, validation normalized RMSE 0.0198722.

| Test metric (10,000 examples) | Value |
| --- | ---: |
| Heading MAE | 1.8479° |
| Normalized RMSE | 0.0194365 |
| Speed MAE | 0.000133656 scene units/step |
| Vector-tip MAE | 0.000188402 scene units/step |

Training and validation caches were removed by the runner. The new test cache
and model outputs remain available. Previous output and test-cache directories
were preserved with the suffix `-7c13dd4`.

## Artifacts

- [Timestamped training log](/home/jacob/Documents/dw93net_output/benchmark-fcf1661/training.log)
- [Resource samples (JSONL)](/home/jacob/Documents/dw93net_output/benchmark-fcf1661/resources.jsonl)
- [Configuration and commit metadata](/home/jacob/Documents/dw93net_output/benchmark-fcf1661/metadata.json)
- [Completion status](/home/jacob/Documents/dw93net_output/benchmark-fcf1661/completion.json)
- [NVIDIA validation results](/home/jacob/Documents/dw93net_output/translation_only-fcf1661/final/validation-results.json)
- [NVIDIA test results](/home/jacob/Documents/dw93net_output/translation_only-fcf1661/final/test-results.json)

## R9700 repeat and comparison

The repeat completed on 2026-09-16 using commit
`78cc55be1dac4c6bee8a842bfad19c9478d9d980`. All **50 automated tests passed**
in the AMD environment immediately before the benchmark (15.72 s). ROCm
emitted compiler/MIOpen warnings during the tests, but there were no test
failures. The benchmark exited successfully with code zero.

The recipe, seeds, batch sizes, mirroring, mixed precision, epochs, candidate
selection, and sample counts were unchanged. Fresh training, validation, and
test data were generated. NVIDIA artifacts and its persistent test set were
archived with the suffix `-fcf1661` before starting AMD. The training code was
unchanged between runs; the intervening commit added this report and AMD
telemetry support.

| Measurement | RTX 3080 Ti | Radeon AI PRO R9700 |
| --- | ---: | ---: |
| Motherboard slot / upstream connection | Gen5 x16 slot; GPU supports Gen4 x16 | Gen4 x4 upstream, bridge to GPU |
| GPU memory capacity | 12 GiB | 32 GiB nominal |
| End-to-end time, including logger shutdown | 736.49 s (12m 16s) | 990.04 s (16m 30s) |
| Training period including checkpoints/set transition | 617.25 s | 746.41 s |
| Mean reported epoch throughput | 3,255.2 presentations/s | 2,693.0 presentations/s |
| Mean loader wait per epoch | 0.0654 s | 0.0516 s |
| Next-set generation including worker startup | 43.52 s | 40.25 s |
| Generation wait at set switch | 0.00 s | 0.00 s |
| Validation of 50 candidates and selection | 32.41 s | 165.71 s |
| Final 10k-example test inference | 2.94 s | 15.90 s |
| Set 2 mean GPU utilization | 98.6% | 98.6% |
| Set 2 mean system CPU busy | 7.0% | 9.5% |
| Peak system RAM used | 8.65 GiB | 10.00 GiB |
| Peak GPU memory used | 4,245 MiB | 2,902.82 MiB |
| Peak reported GPU power | 349.65 W | 354 W |
| Peak reported GPU / AMD edge temperature | 83°C | 79°C |
| Swap use | None | None |
| Selected epoch | 47 | 49 |
| Validation normalized RMSE | 0.0198722 | 0.0215562 |
| Test normalized RMSE | 0.0194365 | 0.0216333 |
| Test heading MAE | 1.8479° | 2.4063° |
| Test speed MAE, scene units/step | 0.000133656 | 0.000151602 |
| Test vector-tip MAE, scene units/step | 0.000188402 | 0.000217117 |

The R9700 delivered **17.3% lower training throughput** in this configuration
and took **34.4% longer end to end**. Float32 evaluation, unlike mixed-precision
training, took about five times as long, accounting for much of the additional
end-to-end gap. This identifies a performance difference in the current
evaluation path, not its underlying cause.

AMD next-set generation completed at 78.296 s; the set switch was at 411.694 s.
The next set was ready 333.4 s before use. Allocation rose from one to eight
threads with 0.3–0.4% loader wait; no downward-throttling events occurred.
Peak epoch loader wait was 0.08 s. The initial 20k-example generation took
37.23 s, validation generation 3.65 s, and test generation 19.19 s.

The AMD logger captured 989 samples via Linux amdgpu sysfs/hwmon, approximately
once per second. During overlap, mean CPU busy was 15.2%, GPU utilization 91.0%,
and peak RAM 10.00 GiB. After generation, first-set GPU utilization averaged
98.7%. Overall mean CPU busy was 9.8%; maximum was 26.7%. I/O wait averaged
0.009%, with a 0.394% peak. These remain system-wide measurements. NVIDIA
and AMD driver memory accounting and temperature/power sensor definitions
are not necessarily identical.

AMD additional temperature peaks were **102°C junction** and **93°C memory**.
Those sensors were not recorded for NVIDIA, so only the available readings
are reported. Clock/throttle-reason counters were not logged; the data do not
establish whether either card hit a thermal or power limit.

Late-run sampled batch times were approximately 2.5 ms transfer-plus-correlation
and 21.5–21.6 ms model/backward/optimizer on AMD, versus 1.8–1.9 ms and
17.6–17.7 ms on NVIDIA. The transfer measurement includes on-device
correlation; it cannot quantify the PCIe x4 penalty independently. A matched
R9700 run in the x16 slot, with the same software, would be needed to isolate
the slot effect. Different backend/kernel implementations and PyTorch versions
also prevent attributing this result solely to GPU hardware. Accuracy differences
are from single training runs and do not establish a systematic accuracy ranking.

Both devices had substantial spare VRAM and negligible loader wait. The
R9700's larger VRAM capacity did not help this particular configuration.
The results support retaining the 3080 Ti for the measured workflow on these
installed configurations; they do not predict performance of larger models,
larger batches, or a differently tuned ROCm implementation.

### AMD artifacts

- [Timestamped training log](/home/jacob/Documents/dw93net_output/benchmark-r9700-78cc55b/training.log)
- [Resource samples, including junction/memory temperatures](/home/jacob/Documents/dw93net_output/benchmark-r9700-78cc55b/resources.jsonl)
- [Configuration metadata](/home/jacob/Documents/dw93net_output/benchmark-r9700-78cc55b/metadata.json)
- [Completion status](/home/jacob/Documents/dw93net_output/benchmark-r9700-78cc55b/completion.json)
- [AMD validation results](/home/jacob/Documents/dw93net_output/translation_only/final/validation-results.json)
- [AMD test results](/home/jacob/Documents/dw93net_output/translation_only/final/test-results.json)
