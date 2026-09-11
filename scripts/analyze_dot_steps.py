"""Replay a generated flow dataset and plot true projected dot displacements.

The saved XT tensors no longer contain dot identities.  This script uses the
generation seed to replay the exact stimuli, checks the replayed camera labels
against the filenames, and measures dots whose centers are rendered in both
frames.  Horizontal displacement is reported in output pixels per frame step.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABEL_RE = re.compile(
    r"^ndx=(-?\d+)_ndz=(-?\d+)_nrot=(-?\d+)_id=(\d+)(?:_superimposed)?\.pt$"
)


def saved_labels(dataset_root: Path) -> dict[int, tuple[int, int, int]]:
    labels = {}
    for split in ("train", "validate", "test"):
        for path in (dataset_root / split).glob("*.pt"):
            match = LABEL_RE.search(path.name)
            if match is None:
                raise ValueError(f"Unrecognized stimulus filename: {path.name}")
            ndx, ndz, nrot, sample_id = map(int, match.groups())
            labels[sample_id] = (ndx, ndz, nrot)
    return labels


def rendered_x(positions: np.ndarray, config: dict) -> tuple[np.ndarray, np.ndarray]:
    radius_squared = np.sum(positions**2, axis=1)
    x, z = positions[:, 0], positions[:, 1]
    half_fov_rad = np.deg2rad(config["fov_deg"] / 2)

    theta = np.arctan2(x, z)
    x_pixel = ((theta / half_fov_rad + 1) / 2) * (config["stim_wid_px"] - 1)

    # A center inside these bounds contributes to an output pixel. Requiring
    # both frames excludes appearance/disappearance events with no defined step.
    rendered = (
        (radius_squared < 1)
        & (radius_squared > config["near_exclusion_radius"] ** 2)
        & (x_pixel >= 0)
        & (x_pixel <= config["stim_wid_px"] - 1)
    )
    return x_pixel, rendered


def analyze(dataset_root: Path, output_png: Path, output_json: Path) -> None:
    metadata = json.loads((dataset_root / "generation.json").read_text())
    if metadata["stimulus_config"]["superimpose"]:
        raise ValueError("This replay currently supports non-superimposed stimuli only")

    labels = saved_labels(dataset_root)
    n_samples = metadata["n_samples"]
    if len(labels) != n_samples or set(labels) != set(range(n_samples)):
        raise ValueError(f"Expected IDs 0..{n_samples - 1}, found {len(labels)} files")

    def decode_spec(spec):
        if spec.get("sampling") == "fixed":
            return float(spec["value"]), float(spec["value"]), True
        return float(spec["low_inclusive"]), float(spec["high_exclusive"]), False

    speed_low, speed_high, speed_fixed = decode_spec(metadata["speed_range"])
    heading_low, heading_high, heading_fixed = decode_spec(metadata["heading_deg_range"])
    rot_low, rot_high, rot_fixed = decode_spec(metadata["rot_deg_range"])
    if speed_fixed:
        raise ValueError("Five speed bins require a ranged-speed dataset")
    config = metadata["stimulus_config"]
    bin_edges = np.linspace(speed_low, speed_high, 6)
    steps_by_bin: list[list[np.ndarray]] = [[] for _ in range(5)]
    stimuli_by_bin = np.zeros(5, dtype=int)

    rng = np.random.default_rng(metadata["seed"])
    rng.permutation(n_samples)  # The generator assigns splits before drawing stimuli.

    for sample_id in range(n_samples):
        while True:
            speed = speed_low if speed_fixed else rng.uniform(speed_low, speed_high)
            heading_deg = (heading_low if heading_fixed
                           else rng.uniform(heading_low, heading_high))
            heading_rad = np.deg2rad(heading_deg)
            ndx = int(round(speed * np.sin(heading_rad) * 1e9))
            ndz = int(round(speed * np.cos(heading_rad) * 1e9))
            quantized_speed = np.hypot(ndx, ndz) / 1e9
            if speed_low <= quantized_speed < speed_high:
                break
        rot_deg = rot_low if rot_fixed else rng.uniform(rot_low, rot_high)
        nrot = int(round(rot_deg * 1e9))
        if labels[sample_id] != (ndx, ndz, nrot):
            raise RuntimeError(
                f"Replay diverged at ID {sample_id}: generated {(ndx, ndz, nrot)}, "
                f"saved {labels[sample_id]}"
            )

        positions = rng.uniform(-1, 1, size=(config["n_world_dots"], 2))
        moved = positions - np.array([ndx / 1e9, ndz / 1e9])
        # Rotation is zero in the current dataset; retain the general transform
        # so the replay remains identical if used for another diagnostic set.
        angle = np.deg2rad(-(nrot / 1e9))
        c, s = np.cos(angle), np.sin(angle)
        moved = moved @ np.array([[c, -s], [s, c]]).T

        x0, visible0 = rendered_x(positions, config)
        x1, visible1 = rendered_x(moved, config)
        jointly_visible = visible0 & visible1
        bin_index = min(np.searchsorted(bin_edges, quantized_speed, side="right") - 1, 4)
        steps_by_bin[bin_index].append((x1[jointly_visible] - x0[jointly_visible]).astype(np.float32))
        stimuli_by_bin[bin_index] += 1

    steps = [np.concatenate(parts) for parts in steps_by_bin]
    all_steps = np.concatenate(steps)
    # Keep the common plot scale informative while recording the exact extrema.
    central_limit = float(np.quantile(np.abs(all_steps), 0.999))
    plot_limit = np.ceil(central_limit / 10) * 10
    hist_edges = np.linspace(-plot_limit, plot_limit, 181)

    stats = {
        "dataset": str(dataset_root.resolve()),
        "definition": "x_pixel(frame_1)-x_pixel(frame_0), dot center rendered in both frames",
        "units": "output pixels per frame step",
        "speed_bin_edges_scene_units_per_step": bin_edges.tolist(),
        "plot_central_abs_quantile": 0.999,
        "plot_limit_pixels": plot_limit,
        "bins": [],
    }

    fig, axes = plt.subplots(5, 1, figsize=(12, 13), sharex=True, constrained_layout=True)
    for i, (ax, values) in enumerate(zip(axes, steps)):
        low, high = bin_edges[i : i + 2]
        outside = np.mean(np.abs(values) > 60) * 100
        bin_stats = {
            "speed_low_inclusive": float(low),
            "speed_high_exclusive": float(high),
            "n_stimuli": int(stimuli_by_bin[i]),
            "n_jointly_visible_dots": int(values.size),
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "median_absolute": float(np.median(np.abs(values))),
            "p95_absolute": float(np.quantile(np.abs(values), 0.95)),
            "p99_absolute": float(np.quantile(np.abs(values), 0.99)),
            "percent_abs_over_60": float(outside),
        }
        stats["bins"].append(bin_stats)

        ax.hist(values, bins=hist_edges, density=True, color="#31688e", alpha=0.9)
        ax.axvline(-60, color="#b2182b", linestyle="--", linewidth=1)
        ax.axvline(60, color="#b2182b", linestyle="--", linewidth=1)
        ax.set_ylabel("Density")
        ax.set_title(
            f"Camera speed [{low:.4f}, {high:.4f}) — {stimuli_by_bin[i]:,} stimuli, "
            f"{values.size:,} tracked dots"
        )
        ax.text(
            0.99,
            0.93,
            f"median |step| {bin_stats['median_absolute']:.1f} px; "
            f"95% {bin_stats['p95_absolute']:.1f} px; "
            f"|step|>60: {outside:.1f}%\n"
            f"full range {values.min():.1f} to {values.max():.1f} px",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
        )
        ax.grid(alpha=0.2)

    axes[-1].set_xlabel("Signed horizontal dot step (pixels per frame step)")
    fig.suptitle(
        "True projected dot steps in the 50,000-stimulus translation dataset\n"
        f"Central 99.9% shown (±{plot_limit:.0f} px); red lines mark current detector limit ±60 px",
        fontsize=14,
    )
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    output_json.write_text(json.dumps(stats, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--output", type=Path, default=Path("analysis/dot_steps_5_speed_bins.png"))
    parser.add_argument("--stats", type=Path, default=Path("analysis/dot_steps_5_speed_bins.json"))
    args = parser.parse_args()
    analyze(args.dataset_root, args.output, args.stats)


if __name__ == "__main__":
    main()
