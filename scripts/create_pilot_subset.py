"""Create a deterministic hard-linked subset of an existing stimulus dataset."""

import argparse
import json
import os
from pathlib import Path
import random


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--train", type=int, default=2000)
    parser.add_argument("--validate", type=int, default=250)
    parser.add_argument("--test", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    source_metadata = json.loads((source / "generation.json").read_text())
    rng = random.Random(args.seed)
    selected = {}
    for split, count in (("train", args.train), ("validate", args.validate),
                         ("test", args.test)):
        candidates = sorted((source / split).glob("*.pt"))
        if count > len(candidates):
            raise ValueError(f"Requested {count} {split} samples from {len(candidates)}")
        selected[split] = rng.sample(candidates, count)

    for split, paths in selected.items():
        output = destination / split
        output.mkdir(parents=True)
        for path in paths:
            os.link(path, output / path.name)

    metadata = dict(source_metadata)
    metadata.update({
        "format": "polar-translation-stimulus-subset-v1",
        "n_samples": sum(map(len, selected.values())),
        "split_counts": {split: len(paths) for split, paths in selected.items()},
        "subset_seed": args.seed,
        "subset_source": str(source),
        "storage": "hard_links_to_subset_source",
    })
    (destination / "generation.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"destination": str(destination), **metadata["split_counts"]}, indent=2))


if __name__ == "__main__":
    main()
