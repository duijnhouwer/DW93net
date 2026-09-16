# Parallel Stimulus Generation

## Implementation status

Implemented as an optional `[parallel_generation]` recipe section, enabled in
the current recipe. See README's Training behavior section for the exact
controller thresholds, resource controls, restart behavior, and limitations.
The controller adjusts numerical-library threads and CPU affinity in one
background process; it does not create a pool of independent stimulus workers.
Loader wait is the feedback signal in this implementation. Direct GPU
utilization sampling and completion-time prediction remain future refinements.

## Objective

Generate the next training stimulus set while the GPU trains on the current
set. The overlap should hide stimulus-generation latency without reducing
training throughput or exposing training to a partially generated dataset.

The system should dynamically balance CPU resources between data loading and
background generation. If loader wait rises, it should reduce the generator's
worker or thread allocation. If the GPU remains fully fed and loader wait stays
low, it may give the generator more concurrency so the next set is ready before
the current set finishes training.

## Proposed pipeline

1. Generate the first training set before training starts.
2. Train on the current set while a separate process generates the next set in
   a distinct staging directory.
3. Publish a generated set atomically only after all bundles and metadata have
   been validated.
4. At the set boundary, wait for generation only if the staged set is not yet
   ready, switch to it, delete the consumed set, and begin generating the next
   one.
5. Preserve the existing epoch-boundary resume behavior and deterministic seed
   sequence.

## Priority and throttling

Training has priority over background generation. Initial controls should
include:

- separate CPU affinity pools for loader workers and stimulus generation;
- lower CPU priority (`nice`) and disk-I/O priority (`ionice`) for generation;
- explicit limits on PyTorch, OpenMP, MKL, and OpenBLAS thread counts in the
  generator process; and
- a configurable minimum and maximum generator concurrency.

Dynamic throttling should use measured training signals rather than maximizing
generator throughput in isolation. The primary feedback signal is loader wait
per epoch or over a rolling batch window. GPU utilization and the amount of
unfinished staged work can be secondary signals.

A conservative controller can reduce generator concurrency when loader wait
exceeds a configured threshold for multiple observations. It can increase
concurrency slowly after loader wait remains low and the next set is at risk of
not completing before the training-set boundary. Hysteresis and cooldown
periods should prevent rapid oscillation.

## Success criteria

- Training samples per second do not regress materially relative to sequential
  generation.
- Loader wait remains within a configured budget and the GPU remains fed.
- The next set is normally ready by the time the current set finishes.
- Interrupted runs never consume incomplete data and resume deterministically.
- Consumed and abandoned staging data are removed safely.
- Logs report generation progress, allocated concurrency, loader wait, training
  throughput, and time spent waiting at each set boundary.

Peak disk usage must be planned for two sets during overlap: the current set
and the staged next set. For the measured representation, one million stimuli
are estimated at roughly 55 GB, so double buffering may temporarily require
about 110 GB.
