"""Replay a generated DoG cache and plot true projected dot displacements.

This script uses the generation seed to replay the exact stimuli, checks the
replayed camera labels against stored bundle targets, and measures dots
whose centers are rendered in both frames. Horizontal displacement is reported
in output pixels per frame step.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from stimuli.generate_flow_xt_data import sample_motion
from stimuli.metadata import bounds


def saved_labels(dataset_root: Path):
    labels = {}
    indexes = list(dataset_root.glob('*/bundles.json'))
    if len(indexes) != 1:
        raise ValueError(f"Expected exactly one role index under {dataset_root}")
    previous, bundle = None, None
    for sample_id, name, row in json.loads(indexes[0].read_text()):
        if type(sample_id) is not int or sample_id in labels:
            raise ValueError('Invalid numeric sample ID')
        if name != previous:
            bundle = np.load(indexes[0].parent / name, mmap_mode='r', allow_pickle=False)
            previous = name
        labels[sample_id] = bundle[row]['target'].copy()
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
    metadata = json.loads((dataset_root / "dog.json").read_text())["stimulus_generation"]
    if metadata["stimulus_config"]["superimpose"]:
        raise ValueError("This replay currently supports non-superimposed stimuli only")

    labels = saved_labels(dataset_root)
    n_samples = metadata["n_samples"]
    if len(labels) != n_samples or set(labels) != set(range(n_samples)):
        raise ValueError(f"Expected IDs 0..{n_samples - 1}, found {len(labels)} files")

    speed_low, speed_high = bounds(metadata["speed_range"])
    heading_low, heading_high = bounds(metadata["heading_deg_range"])
    rot_low, rot_high = bounds(metadata["rot_deg_range"])
    if speed_low == speed_high:
        raise ValueError("Five speed bins require a ranged-speed dataset")
    config = metadata["stimulus_config"]
    bin_edges = np.linspace(speed_low, speed_high, 6)
    steps_by_bin: list[list[np.ndarray]] = [[] for _ in range(5)]
    stimuli_by_bin = np.zeros(5, dtype=int)

    rng = np.random.default_rng(metadata["seed"])

    for sample_id in range(n_samples):
        dx, dz, rot_deg = sample_motion(rng, (speed_low, speed_high),
                                        (heading_low, heading_high), (rot_low, rot_high))
        if not np.array_equal(labels[sample_id], np.asarray((dx, dz, rot_deg), dtype=np.float32)):
            raise RuntimeError(
                f"Replay diverged at ID {sample_id}: generated {(dx, dz, rot_deg)}, "
                f"saved {labels[sample_id]}"
            )

        positions = rng.uniform(-1, 1, size=(config["n_world_dots"], 2))
        moved = positions - np.array([dx, dz])
        # Rotation is zero in the current dataset; retain the general transform
        # so the replay remains identical if used for another diagnostic set.
        angle = np.deg2rad(-rot_deg)
        c, s = np.cos(angle), np.sin(angle)
        moved = moved @ np.array([[c, -s], [s, c]]).T

        x0, visible0 = rendered_x(positions, config)
        x1, visible1 = rendered_x(moved, config)
        jointly_visible = visible0 & visible1
        # Float32 conversion can round just outside a sampled interval endpoint.
        bin_index = int(np.clip(np.searchsorted(bin_edges, np.hypot(dx, dz), side="right") - 1, 0, 4))
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
