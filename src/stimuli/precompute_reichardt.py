"""Precompute an XT dataset tree once, preserving split paths and label filenames."""

import argparse
import json
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
import torch
import numpy as np
from features.dog_motion import DoGMotionBank
from stimuli.load_flow_xt_data import parse_camera_motion_from_filename
from stimuli.preprocessing_provenance import write_provenance, repository_state
from stimuli.stimulus_io import decode_stimulus


def precompute_dataset(source, destination, bank=None, *, stimulus_bundle_size=64):
    if type(stimulus_bundle_size) is not int or stimulus_bundle_size < 1:
        raise ValueError('stimulus_bundle_size must be a positive integer')
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-nested directories")
    files = sorted(source.rglob("*.pt"))
    if not files:
        raise ValueError("Source contains no XT .pt files")
    bank = bank or DoGMotionBank()
    if not isinstance(bank, DoGMotionBank):
        raise ValueError("Preprocessing requires a DoGMotionBank")
    destination.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    repo = Path(__file__).resolve().parents[2]
    provenance, initial_state = write_provenance(destination, repo)
    samples = []
    raw_shape = None
    product_shape = None
    pending = []
    indexes = {}
    bundle_number = 0

    def flush():
        nonlocal bundle_number
        if not pending:
            return
        relative_parent = pending[0][0].parent
        folder = destination / relative_parent
        folder.mkdir(parents=True, exist_ok=True)
        name = f'bundle-{bundle_number:09d}.npy'
        response_shape = pending[0][1].shape
        bundle = np.empty(len(pending), dtype=np.dtype([
            ('reichardt_responses', np.float16, response_shape),
            ('target', np.float32, (3,)),
        ]))
        bundle['reichardt_responses'] = np.stack([tensor.numpy() for _, tensor, _ in pending])
        bundle['target'] = np.asarray([target for _, _, target in pending], dtype=np.float32)
        np.save(folder / name, bundle)
        indexes.setdefault(relative_parent, []).extend(
            [path.name, name, row] for row, (path, _, _) in enumerate(pending))
        pending.clear()
        bundle_number += 1
    # Publish metadata only after all samples succeed; partial caches cannot train.
    with torch.no_grad():
        for path in files:
            raw_bytes = path.read_bytes()
            xt = torch.load(io.BytesIO(raw_bytes), map_location="cpu", weights_only=True)
            if not isinstance(xt, torch.Tensor) or xt.ndim not in (2, 3):
                raise ValueError(f"Expected one raw XT tensor in {path}")
            decoded = decode_stimulus(xt)
            if raw_shape is None:
                raw_shape = list(decoded.shape)
            elif list(decoded.shape) != raw_shape:
                raise ValueError(f"Inconsistent stimulus shape in {path}")
            responses = bank(decoded).half()
            product_shape = list(responses.shape)
            if not torch.isfinite(responses).all():
                raise ValueError(f"Storage dtype overflow in {path}")
            relative = path.relative_to(source)
            if pending and pending[0][0].parent != relative.parent:
                flush()
            motion = parse_camera_motion_from_filename(path.name)
            pending.append((relative, responses,
                            (motion['dx'], motion['dz'], motion['rot_deg'])))
            if len(pending) >= stimulus_bundle_size:
                flush()
            samples.append(dict(path=str(path.relative_to(source)),
                                input_sha256=hashlib.sha256(raw_bytes).hexdigest()))
    flush()
    for folder, entries in indexes.items():
        (destination / folder / 'bundles.json').write_text(json.dumps(entries) + '\n')
    if repository_state(repo, destination) != initial_state:
        raise RuntimeError("Source changed during preprocessing; cache left incomplete, rerun in a new folder")
    (destination / 'inputs.json').write_text(json.dumps(samples, indent=2) + "\n")
    generation_path = source / "generation.json"
    stimulus_generation = json.loads(generation_path.read_text()) if generation_path.exists() else None
    metadata = dict(bank.metadata, storage_dtype="torch.float16", finite_validated=True,
                    storage_format='npy-stimulus-bundles-v2', source=str(source),
                    created_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                    n_samples=len(samples), raw_shape=raw_shape, product_shape=product_shape,
                    input_encoding="uint8 / 255", input_manifest="inputs.json",
                    input_manifest_sha256=hashlib.sha256((destination / 'inputs.json').read_bytes()).hexdigest(),
                    stimulus_generation=stimulus_generation, software=provenance)
    (destination / "reichardt.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument('--stimulus-bundle-size', type=int, default=64,
                        help='Samples per memory-mapped bundle (minimum: 1; default: 64)')
    parser.add_argument("--dx", nargs="+", type=int, default=list(DoGMotionBank().dx_values))
    parser.add_argument("--sigma-center", type=float, default=1., help="DoG center sigma in pixels")
    parser.add_argument("--sigma-surround", type=float, default=3., help="DoG surround sigma in pixels")
    args = parser.parse_args()
    if args.stimulus_bundle_size < 1:
        parser.error('--stimulus-bundle-size must be at least 1')
    precompute_dataset(args.source, args.destination, DoGMotionBank(args.dx, sigma_center=args.sigma_center, sigma_surround=args.sigma_surround), stimulus_bundle_size=args.stimulus_bundle_size)


if __name__ == "__main__":
    main()
