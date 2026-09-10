# Storage and training pipeline benchmark — 2026-09-10

Recommendation: keep one file per sample initially. Use float16 product storage
and four background DataLoader workers with pinned memory. Packing improved
serial loading, but provided no material throughput advantage over individual
files with four workers in this bounded test. Recheck on the full dataset.

The requested generator change (255 supersampling and uint8 raw storage) and
float16 default have not been applied: this benchmark addresses the user's
request to decide the file layout first. Existing data and checkpoints were
untouched. Temporary benchmark data was removed automatically.

## Hardware and method

- NVIDIA GeForce RTX 3080 Ti, PyTorch 2.11.0+cu128; CUDA preflight passed.
- Intel Core i7-14700KF; data stored on the mounted Crucial T705 NVMe filesystem.
- 384 synthetic samples with the current default product shape [21, 1, 7180].
  Batch size 12; existing FlowXTNet, float32 training, Adam, forward/backward
  and optimizer update. Synthetic values match dimensions/dtypes, not actual
  stimulus statistics. Zero labels; all three normalized loss terms exercised.
- Individual files use the existing ReichardtDataset, including finite-value
  validation, label parsing, float32 conversion and default batch collation.
- Packed comparison: one float16 .npy array, independently memory-mapped in each
  worker; per-sample copy, finite-value validation and float32 conversion retained.
- CUDA warmup precedes timing. Each measured step finishes with loss.item(), as
  in current training. Transfer timing adds synchronization for attribution.
  No mixed precision, model changes, or overlapping CUDA copy streams.
- Cold tests fsync and advise eviction of only benchmark files using
  POSIX_FADV_DONTNEED. Process and worker /proc I/O counters confirm actual
  storage reads. Filesystem metadata can remain cached; this does not simulate
  a cold boot or a 500,000-entry directory.
- Serial warm comparisons: three runs of 24 batches, two CPU intra-op threads.
  Cold serial comparisons: one run of 24 batches, two threads. Current default
  baseline also tested with 20 threads. Worker cold comparisons: three runs of
  24 batches, default 20 parent threads, pin_memory=True; worker startup and the
  first prefetched batch excluded. Default worker prefetch behavior retained.
- Canonical results are results.json in this directory. Earlier sibling outputs
  are exploratory: the confirmation run's packed worker case accidentally
  serialized a memmap into worker RAM; its profiler total also double-counted
  parent CPU attribution. Both issues are corrected in this final run.

## Results

All values are mean milliseconds per batch. Worker rows average three runs.
Loading means foreground wait, not the total background CPU work.

| Configuration | Loading/wait | Transfer | Training step | Total |
| --- | ---: | ---: | ---: | ---: |
| Current float32, cached, default threads | 5.12 | 0.81 | 4.60 | 10.53 |
| Individual float32, uncached, no workers | 10.33 | 0.80 | 4.17 | 15.30 |
| Individual float16, uncached, no workers | 8.45 | 0.80 | 4.28 | 13.53 |
| Packed float16, uncached, no workers | 5.18 | 0.78 | 4.22 | 10.18 |
| Individual float16, uncached, 2 workers | 1.69 | 0.56 | 4.18 | 6.44 |
| Packed float16, uncached, 2 workers | 0.37 | 0.56 | 4.34 | 5.26 |
| Individual float16, uncached, 4 workers | 0.20 | 0.62 | 4.40 | 5.22 |

Repeated GPU-resident training took 4.41 ms per batch. CUDA-only profiler events
sum to 4.20 ms per batch, with convolution work the largest component. This
supports GPU work becoming the dominant stage once background loading is enabled.

The existing loader converts float16 products to float32 before transfer, so
current transfer volume is 7,237,440 bytes of inputs per batch, excluding the
tiny labels. Isolated transfer tests using an already allocated CPU source:

| Input transfer | Time, including GPU float32 cast when needed |
| --- | ---: |
| Pageable float32 | 0.434 ms |
| Pinned float32 | 0.344 ms |
| Pageable float16 | 0.274 ms |
| Pinned float16 | 0.218 ms |

These isolated numbers exclude loading, pinning, collation and label transfer;
they should not replace the end-to-end transfer column above. PCIe transfer is
a smaller cost than either serial loading or GPU compute. Keeping float16 until
the GPU would halve input traffic, but the measured absolute saving is small.

## Space and next steps

For the 384 float16 products, individual files allocated 117,964,800 bytes versus
115,799,168 bytes for the packed array's logical size: approximately 1.8% saving,
excluding directory/inode overhead. Most storage savings come from float16
itself, not packing. The earlier conversation's 1.5% allocation estimate sampled
older pilot files and is not the full-resolution product measurement here.

Implement 255 supersampling with normalized uint8 loading for raw stimuli, and
float16 product storage. Initially retain individual files and use four loading
workers plus pin_memory; persistent workers should avoid repeated epoch startup,
although startup savings were not measured here. Validate numerical scaling and
cache compatibility separately from these performance measurements. Rebenchmark
the real large dataset before deciding on shards; packing may still be useful
for directory management, copying, or if large-scale loading stops keeping up.

Reproduce from repository root:

```bash
PYTHONPATH=src /home/jacob/venvs/nvidia/bin/python scripts/benchmark_pipeline.py --output /tmp/dw93net-benchmark
```

Choose an output directory on the storage device being evaluated. The benchmark
requires the NVIDIA environment and permission to access CUDA devices.
