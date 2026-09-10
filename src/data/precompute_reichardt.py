"""Precompute an XT dataset tree once, preserving split paths and label filenames."""

import argparse
import json
from pathlib import Path
import torch
from features.reichardt import ReichardtBank
from data.stimulus_io import decode_stimulus


def precompute_dataset(source, destination, bank=None, dtype=torch.float16):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError("Source and destination must be separate, non-nested directories")
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("Storage dtype must be float16 or float32")
    files = sorted(source.rglob("*.pt"))
    if not files:
        raise ValueError("Source contains no XT .pt files")
    bank = bank or ReichardtBank()
    destination.mkdir(parents=True, exist_ok=False)
    # Publish metadata only after all samples succeed; partial caches cannot train.
    with torch.no_grad():
        for path in files:
            xt = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(xt, torch.Tensor) or xt.ndim not in (2, 3):
                raise ValueError(f"Expected one raw XT tensor in {path}")
            responses = bank(decode_stimulus(xt)).to(dtype)
            if not torch.isfinite(responses).all():
                raise ValueError(f"Storage dtype overflow in {path}")
            output = destination / path.relative_to(source)
            output.parent.mkdir(parents=True, exist_ok=True)
            torch.save(responses, output)
    metadata = dict(bank.metadata, storage_dtype=str(dtype), source=str(source))
    (destination / "reichardt.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument("--dx", nargs="+", type=int, default=list(ReichardtBank().dx_values))
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    args = parser.parse_args()
    precompute_dataset(args.source, args.destination, ReichardtBank(args.dx),
                       getattr(torch, args.dtype))


if __name__ == "__main__":
    main()
