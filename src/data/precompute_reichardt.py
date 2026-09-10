"""Precompute an XT dataset tree once, preserving split paths and label filenames."""

import argparse
import json
import hashlib
import io
from datetime import datetime, timezone
from pathlib import Path
import torch
from features.dog_motion import DoGMotionBank
from data.preprocessing_provenance import write_provenance, repository_state
from data.stimulus_io import decode_stimulus


def precompute_dataset(source, destination, bank=None):
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
            output = destination / path.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(responses, output)
            samples.append(dict(path=str(path.relative_to(source)),
                                input_sha256=hashlib.sha256(raw_bytes).hexdigest()))
    if repository_state(repo, destination) != initial_state:
        raise RuntimeError("Source changed during preprocessing; cache left incomplete, rerun in a new folder")
    (destination / 'inputs.json').write_text(json.dumps(samples, indent=2) + "\n")
    metadata = dict(bank.metadata, storage_dtype="torch.float16", source=str(source),
                    created_utc=started, completed_utc=datetime.now(timezone.utc).isoformat(),
                    n_samples=len(samples), raw_shape=raw_shape, product_shape=product_shape,
                    input_encoding="uint8 / 255", input_manifest="inputs.json",
                    input_manifest_sha256=hashlib.sha256((destination / 'inputs.json').read_bytes()).hexdigest(),
                    software=provenance)
    (destination / "reichardt.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--dx", nargs="+", type=int, default=list(DoGMotionBank().dx_values))
    parser.add_argument("--sigma-center", type=float, default=1., help="DoG center sigma in pixels")
    parser.add_argument("--sigma-surround", type=float, default=3., help="DoG surround sigma in pixels")
    args = parser.parse_args()
    precompute_dataset(args.source, args.destination, DoGMotionBank(args.dx, sigma_center=args.sigma_center, sigma_surround=args.sigma_surround))


if __name__ == "__main__":
    main()
