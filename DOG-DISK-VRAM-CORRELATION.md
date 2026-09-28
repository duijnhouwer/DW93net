# DoG Disk Cache and VRAM Correlation Refactor

## Purpose

The current dataset pipeline renders each two-frame Flow-XT stimulus, writes it
as an individual `.pt` file, reloads every file, computes all DoG/Reichardt
products, and finally writes those products into NumPy bundles. At the default
geometry, one raw stimulus is small while its full four-polarity correlation
map is approximately 6.84 MB in float16. Increasing the spatial width or the
number of tested displacements makes this cached representation grow linearly
and eventually makes storage and input bandwidth impractical.

This branch changes the persistent representation to compact float32 signed
DoG contrast bundles. Flow-XT stimuli are rendered and filtered entirely in
memory. During training and evaluation, the signed correlations for the
configured displacements are calculated on the same GPU that runs the model and
are discarded after the batch has been consumed.

The intended pipeline is:

```text
Dataset generation, once:
motion parameters
  -> render Flow-XT in memory
  -> quantize to uint8 and decode in memory
  -> float32 DoG filtering
  -> float32 signed DoG bundles on disk

Training and evaluation, every batch:
memory-mapped signed DoG bundle
  -> pinned host batch
  -> transfer to the model GPU
  -> shifted multiplication in VRAM
  -> model
  -> discard correlations
```

## Decisions

- Persist signed DoG contrast rather than separate ON and OFF maps. ON and OFF
  remain exactly recoverable as `relu(contrast)` and `relu(-contrast)` while the
  signed representation uses half the storage.
- Store DoG contrast as float32. This preserves the present DoG computation
  precision and avoids quantizing before correlation.
- Do not persist Flow-XT tensors in the recipe pipeline.
- Do not persist full Reichardt product maps.
- Use one GPU for online correlation and model training.
- Do not add resumable stimulus generation. An interrupted managed dataset is
  incomplete and is regenerated from its seed, as in the current orchestrator.
- Retain atomic publication: completion metadata is written only after every
  expected bundle and the role index have been written successfully.
- Keep validation progress resumption independent of this refactor.

## Cache format

The new cache format will be named `dog-contrast-bundles-v1`. A cache contains:

```text
dog/
  dog.json
  provenance/
  <role>/
    bundle-000000000.npy
    ...
    bundles.json
```

Each generation call creates exactly one role (`train`, `validate`, or `test`).
The recipe orchestrator invokes those calls independently at the appropriate
stage of the run.

Each structured NumPy bundle has two fields:

- `dog_contrast`: float32 with shape `[2, X_filtered]`
- `target`: float32 with shape `[3]` for `[dx, dz, rot_deg]`

`bundles.json` maps a stable sample identifier to a bundle filename and row.
Targets live in the bundle and are never reconstructed from filenames.

`dog.json` records the storage schema, tensor shapes, sample count, generation
configuration, input byte-quantization semantics, DoG parameters, software
provenance, and creation/completion timestamps. It is published last so a
partial cache cannot be used for training.

At the default width, signed float32 DoG contrast is roughly 57 KB per sample,
compared with approximately 6.84 MB per sample for the current float16
four-polarity products.

## Detector separation

The present `DoGMotionBank` combines two distinct stages. It will be separated
conceptually into:

1. A DoG filter that converts `[B, 1, 2, X]` luminance stimuli to signed
   float32 contrast `[B, 1, 2, X_dog]` and owns the persisted cache metadata.
2. A correlation bank that converts signed contrast to one signed product
   channel per configured displacement and owns the displacement metadata.

The former combined cache path is removed rather than retained as a
compatibility interface. Tests cover the separated DoG and correlation stages
directly.

The model currently reconstructs signed correlation from four nonnegative
polarity products:

```text
ON_ON + OFF_OFF - ON_OFF - OFF_ON
```

For signed contrast `c = ON - OFF`, this is algebraically equal to:

```text
c(first frame, shifted) * c(second frame, present)
```

The online path will therefore create one signed channel per displacement
directly. It will not allocate the four redundant polarity blocks. This reduces
the ephemeral correlation activation and transfer volume by a factor of four.

## Direct dataset generation

The recipe generator will sample targets and render stimuli in the same order
with the same seeded NumPy generator used today. For every sample it will:

1. Quantize `dx`, `dz`, and yaw to their existing nano-unit representation.
2. Render the two Flow-XT frames in memory.
3. Apply the existing byte-grid operation,
   `(stimulus * 255).round().to(torch.uint8)`.
4. Decode back to float32 by dividing by 255, preserving the existing detector
   input semantics.
5. Apply the DoG filter in float32.
6. Validate shape and finiteness.
7. Add the contrast and target to the current in-memory bundle.
8. Write the bundle when it reaches the configured bundle size or dataset end.

No per-sample file is created, and the former raw Flow-XT generator and loader
are removed. Focused experiments use the in-memory renderer directly.

## Loading and augmentation

A memory-mapped `DoGDataset` will replace `ReichardtDataset` in production. It
will validate `dog-contrast-bundles-v1`, expose DoG metadata, and return
float32 contrast and float32 targets.

Horizontal training augmentation will operate before correlation:

- flip signed contrast along the spatial axis;
- negate target `dx`;
- preserve target `dz`;
- negate target yaw.

Computing correlations after this reflection naturally reverses displacement
direction, so the augmentation no longer needs to reorder already-materialized
product channels.

## Online correlation and model input

The DataLoader will load compact float32 contrast batches, use pinned memory on
CUDA, and transfer them to the model device. The correlation bank will perform
the shifted multiplications in VRAM and pass `[B, D, X_product]` signed
correlations to the model.

`FlatlandFlowNet` will accept signed correlation directly. Its trainable V1,
MT, MST, and decoder layers retain their existing displacement semantics and
parameter shapes. The four-block reconstruction at the start of `forward` will
be removed from the production path.

The initial implementation may materialize all configured displacement
channels for the current `-60..60` bank. The correlation API will keep the
operation isolated so displacement tiling or fusion with the model's first
reduction stage can be added later if much larger banks make the activation too
large for VRAM.

## Recipe and checkpoint semantics

Detector geometry will become explicit recipe data rather than an implicit
code default. The strict recipe schema will gain a detector section containing
the DoG sigmas and integer displacement range/step.

Checkpoints will record both:

- the persisted DoG cache semantics;
- the online correlation displacement semantics.

Training resume, candidate validation, final evaluation, and reusable progress
evaluation will reject a cache or checkpoint whose detector semantics do not
match. A DoG cache can be reused with a different displacement bank only in a
new committed recipe/run whose checkpoints record that new bank.

## Consumers to update

The new cache and online correlation path must be used consistently by:

- the recipe orchestrator;
- the training engine;
- candidate selection validation;
- final test evaluation;
- `scripts/evaluate_progress.py`;
- product/sample preview tooling;
- checkpoint compatibility validation;
- documentation and recipe examples.

## Migration

There is no backward-compatibility or automatic-conversion path for existing
`npy-stimulus-bundles-v2` product caches. New recipe runs create only
`dog-contrast-bundles-v1` caches. Existing persistent managed test data must be
explicitly regenerated when moving a recipe to the new cache format. The old
raw stimulus and precomputed-product loaders are removed.

## Verification

The refactor is complete when tests demonstrate that:

- recipe generation writes no raw stimulus `.pt` files;
- recipe generation writes no cached Reichardt product maps;
- fixed seeds reproduce identical targets and float32 DoG bundles;
- changing bundle size does not change sample values or order;
- DoG bundles match the existing byte-decoded float32 DoG stage;
- direct signed correlation matches an independent scalar product reference;
- CPU and CUDA correlation agree within an explicit numerical tolerance;
- mirrored contrast produces the expected spatial and target reflection;
- cache and checkpoint metadata mismatches are rejected;
- epoch-boundary training resume remains correct;
- interrupted candidate validation still resumes from saved progress;
- the tiny end-to-end recipe test trains, selects, and evaluates successfully;
- the complete test suite passes in the documented environment.

## Performance checks

After correctness is established, measure:

- generation samples per second;
- final and temporary dataset size;
- DataLoader wait time;
- host-to-device transfer time;
- correlation time per batch;
- forward/backward time per batch;
- end-to-end training samples per second;
- peak host RAM and GPU memory.

The expected result is a large reduction in cache size and filesystem work,
with correlation cost absorbed into the GPU training step. The measured timing
will determine whether later work should vectorize displacement construction,
tile large banks, or overlap transfers using an additional CUDA stream.
