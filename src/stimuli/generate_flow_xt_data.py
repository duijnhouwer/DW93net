import os
import json
import sys

# Support launching this file directly from an IDE's "Run" button.  In that
# case Python puts src/stimuli (rather than src) on sys.path, so the top-level
# ``data`` package would otherwise be unavailable.
_SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import numpy as np
import torch
from dataclasses import dataclass, replace
from pathlib import Path
from stimuli.precompute_reichardt import precompute_dataset

# Repo root is two levels up from this file (src/stimuli/ -> repo root), so the
# data folder tracks wherever the dw93net repo is checked out.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "stimuli")


@dataclass
class StimConfig:
    stim_wid_px: int | None = None
    fov_deg: float = 120
    near_exclusion_radius: float = 0.04
    n_world_dots: int = 1000
    superimpose: bool = False
    x_supersampling: int = 255

    def __post_init__(self):
        if self.stim_wid_px is None:
            self.stim_wid_px = round(self.fov_deg * 60)


def generate_flow_xt_dataset(
    output_root: str = None,
    n_samples: int = 100, # 100_000
    speed_range=(0.01, 0.01),
    heading_deg_range=(-180, 180),
    rot_deg_range=(-0.5, 0.5),
    split=(0.8, 0.1, 0.1),
    seed: int = None,
    config: StimConfig = StimConfig(),
) -> str:
    """
    Generates a dataset of two-frame, one-step XT flow tensors and saves them into train/validate/test subfolders.
    Speed, heading, and rotation are each specified as a two-value range. Equal
    endpoints mean one fixed value; increasing endpoints define a half-open
    uniform distribution. Heading 0 degrees is forward (+Z), -90 is left (-X),
    and +90 is right (+X). The sampled polar translation is converted with
    dx=speed*sin(heading) and dz=speed*cos(heading).

    Each sample's resulting dx, dz and rot_deg are rounded to
    nano-units/nano-degrees FIRST. Those exact Cartesian values are rendered and
    written to the filename, so the label always matches the stimulus. If
    `superimpose` is True,
    generates separate rotational and translational flows and combines them by summing and
    clipping to [0, 1], preserving fractional antialiased intensities.
    Each file contains one uint8 pair with shape [1, 2, X], scaled to 0..255.
    """

    if output_root is None:
        output_root = DEFAULT_DATA_ROOT

    def validate_range(name, value):
        values = np.asarray(value, dtype=np.float64)
        valid_order = values.shape == (2,) and np.isfinite(values).all()
        if valid_order:
            valid_order = values[0] <= values[1]
        if not valid_order:
            raise ValueError(
                f"{name} must contain two finite values with low <= high. Got: {value}"
            )
        return values, bool(values[0] == values[1])

    def validate_split(split):
        if not np.isclose(sum(split), 1.0):
            raise ValueError(f"split must sum to 1. Got: {split}")

    speed_range, speed_is_fixed = validate_range('speed_range', speed_range)
    heading_deg_range, heading_is_fixed = validate_range(
        'heading_deg_range', heading_deg_range)
    rot_deg_range, rot_is_fixed = validate_range('rot_deg_range', rot_deg_range)
    if speed_range[0] <= 0:
        raise ValueError("speed_range must have a strictly positive lower bound")
    if heading_deg_range[0] < -180 or heading_deg_range[1] > 180:
        raise ValueError("heading_deg_range must lie within [-180, 180]")
    validate_split(split)

    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)

    # Dataset identity is defined by the sampled polar variables. Cartesian
    # deltas belong to individual files because they are derived labels.
    nspeed_min = int(round(speed_range[0] * 1e9))
    nspeed_max = int(round(speed_range[1] * 1e9))
    nheading_min = int(round(heading_deg_range[0] * 1e9))
    nheading_max = int(round(heading_deg_range[1] * 1e9))
    nrot_min = int(round(rot_deg_range[0] * 1e9))
    nrot_max = int(round(rot_deg_range[1] * 1e9))

    folder_name = (
        f"nspeed={nspeed_min},{nspeed_max}_"
        f"nheading={nheading_min},{nheading_max}_"
        f"nrot={nrot_min},{nrot_max}"
    )
    if config.superimpose:
        folder_name += "_superimposed"
    output_folder = os.path.join(output_root, folder_name, "raw")

    # Create subfolders
    for sub in ["train", "validate", "test"]:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

    def range_metadata(values, is_fixed, **extra):
        if is_fixed:
            result = {"value": float(values[0]), "sampling": "fixed"}
        else:
            result = {"low_inclusive": float(values[0]),
                      "high_exclusive": float(values[1]),
                      "sampling": "uniform_half_open"}
        return result | extra

    generation_metadata = {
        "format": "polar-translation-stimuli-v2",
        "n_samples": n_samples,
        "speed_range": range_metadata(
            speed_range, speed_is_fixed, units="scene_units_per_frame_step"),
        "heading_deg_range": range_metadata(
            heading_deg_range, heading_is_fixed,
            convention="0=+Z_forward,-90=-X_left,+90=+X_right"),
        "conversion": {"dx": "speed*sin(heading)", "dz": "speed*cos(heading)"},
        "rot_deg_range": range_metadata(
            rot_deg_range, rot_is_fixed, units="degrees_per_frame_step"),
        "split": list(split), "seed": seed, "stimulus_config": vars(config),
    }
    with open(os.path.join(output_folder, "generation.json"), "w") as metadata_file:
        json.dump(generation_metadata, metadata_file, indent=2)
        metadata_file.write("\n")

    rng = np.random.default_rng(seed)

    # Compute number of samples per split
    n_train = int(n_samples * split[0])
    n_validate = int(n_samples * split[1])
    n_test = n_samples - n_train - n_validate
    split_counts = [n_train, n_validate, n_test]
    split_names = ["train", "validate", "test"]

    # Prepare shuffled list of indices and assign split
    indices = rng.permutation(n_samples)
    split_map = np.empty(n_samples, dtype=object)
    current = 0
    for name, count in zip(split_names, split_counts):
        split_map[indices[current:current + count]] = name
        current += count

    for i in range(n_samples):
        # Component labels are quantized below. Resample the vanishingly thin
        # boundary cases where Cartesian nano-unit rounding would move the
        # actual rendered magnitude outside the requested half-open interval.
        while True:
            speed = speed_range[0] if speed_is_fixed else rng.uniform(*speed_range)
            heading_deg = (heading_deg_range[0] if heading_is_fixed
                           else rng.uniform(*heading_deg_range))
            heading_rad = np.deg2rad(heading_deg)
            ndx = int(round(speed * np.sin(heading_rad) * 1e9))
            ndz = int(round(speed * np.cos(heading_rad) * 1e9))
            quantized_speed = np.hypot(ndx, ndz) / 1e9
            if speed_is_fixed or speed_range[0] <= quantized_speed < speed_range[1]:
                break
        rot_deg = rot_deg_range[0] if rot_is_fixed else rng.uniform(*rot_deg_range)

        # Round to nano-units/nano-degrees FIRST, then use those exact values for BOTH
        # stimulus generation and the filename, so the label always matches the stimulus.
        nrot = int(round(rot_deg * 1e9))
        dx, dz, rot_deg = ndx / 1e9, ndz / 1e9, nrot / 1e9

        # Generate one or two tensors depending on `superimpose`
        xt_tensor_list = []

        for layer in range(2 if config.superimpose else 1):
            layer_config = config
            cam_trans = [dx, dz]
            cam_rot = rot_deg
            if config.superimpose:
                layer_config = replace(config, n_world_dots=config.n_world_dots // 2)
                if layer == 0:
                    cam_trans = [0, 0]
                else:
                    cam_rot = 0

            xt = generate_flow_xt_stim(
                cam_trans_xz=cam_trans,
                cam_rot_deg=cam_rot,
                config=layer_config,
                rng=rng
            )
            xt_tensor_list.append(xt)

        # Preserve antialiased intensities; saturate overlapping flow layers.
        xt_tensor = sum(xt_tensor_list).clamp(0, 1).unsqueeze(0)  # [1, 2, X]

        filename = f"ndx={ndx}_ndz={ndz}_nrot={nrot}_id={i}"
        if config.superimpose:
            filename += "_superimposed"
        filename += ".pt"

        split_name = split_map[i]
        filepath = os.path.join(output_folder, split_name, filename)

        # Round rather than truncate fractional intensities. At 255x
        # supersampling the rendered levels align with the byte grayscale grid.
        torch.save((xt_tensor * 255).round().to(torch.uint8), filepath)

        if i % 100 == 0:
            print(f"Saved {split_name}/{filename}")

    print(f"\n✅ Done: {n_samples} samples saved to {output_folder}")
    return output_folder


def generate_and_precompute(*, precomputed_root=None, **generation_kwargs):
    """Generate raw XT stimuli, then cache their default DoG product maps."""
    raw_folder = Path(generate_flow_xt_dataset(**generation_kwargs)).resolve()
    product_folder = (raw_folder.parent / "reichardt" if precomputed_root is None
                      else Path(precomputed_root).resolve() / raw_folder.parent.name / "reichardt")
    precompute_dataset(raw_folder, product_folder)
    print(f"\n✅ Precomputed DoG/Reichardt maps saved to {product_folder}")
    return str(raw_folder), str(product_folder)



def generate_flow_xt_stim(
    cam_trans_xz=[0, 1e-3],
    cam_rot_deg=0,
    config: StimConfig = StimConfig(),
    rng=None
) -> torch.Tensor:

    """Render [2, X]: initial dots and the same dots after one motion step.

    Translation is expressed in the initial camera frame; rotation is in degrees
    per step. Dots are neither replaced nor wrapped between the two frames.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Validate X supersampling.
    if (
        isinstance(config.x_supersampling, bool)
        or not isinstance(config.x_supersampling, (int, np.integer))
        or config.x_supersampling < 1
    ):
        raise ValueError(
            f"x_supersampling must be a positive integer. "
            f"Got: {config.x_supersampling}"
        )

    x_supersampling = int(config.x_supersampling)
    render_wid_px = config.stim_wid_px * x_supersampling

    cam_trans_xz = np.array(cam_trans_xz, dtype=np.float64)

    half_fov_rad = np.deg2rad(config.fov_deg / 2)
    cam_rot_rad = np.deg2rad(cam_rot_deg)
    tan_half_fov = np.tan(half_fov_rad)

    # Initial dot positions in the 2x2 XZ plane.
    dot_positions = rng.uniform(
        -1, 1,
        size=(config.n_world_dots, 2)
    )

    # Render at supersampled X resolution.
    image = np.zeros(
        (2, render_wid_px),
        dtype=np.float32
    )

    # Rotation applied between successive rendered frames.
    c = np.cos(-cam_rot_rad)
    s = np.sin(-cam_rot_rad)
    rot = np.array([[c, -s], [s, c]])

    # Apply one rigid transform to every dot, then render both states.
    moved_positions = (dot_positions - cam_trans_xz) @ rot.T
    for t, positions in enumerate((dot_positions, moved_positions)):

        # --------------------------------------------------------------
        # 1. RENDER CURRENT DOT POSITIONS
        # --------------------------------------------------------------
        radius_squared = np.sum(positions ** 2, axis=1)
        inside_unit_circle = radius_squared < 1
        outside_near_exclusion = radius_squared > config.near_exclusion_radius**2

        x = positions[:, 0]
        z = positions[:, 1]

        x_in_fov = np.abs(x / z) < tan_half_fov

        visible = (
            inside_unit_circle &
            outside_near_exclusion &
            x_in_fov
        )

        if np.any(visible):
            x_visible = x[visible]
            z_visible = z[visible]

            theta = np.arctan2(x_visible, z_visible)
            x_proj = theta / half_fov_rad

            # Continuous projected position in supersampled X coordinates.
            x_scaled = (
                (x_proj + 1) / 2 *
                (render_wid_px - 1)
            )

            # A dot should remain one output pixel wide.
            # Therefore at supersampled resolution it occupies
            # x_supersampling pixels.
            dot_width = x_supersampling

            for xc in x_scaled:
                left = int(np.floor(xc - dot_width / 2))
                right = left + dot_width

                left = max(left, 0)
                right = min(right, render_wid_px)

                if left < right:
                    image[t, left:right] = 1.0

    # --------------------------------------------------------------
    # DOWNSAMPLE X ONLY
    # --------------------------------------------------------------
    if x_supersampling > 1:
        image = image.reshape(
            2,
            config.stim_wid_px,
            x_supersampling
        ).mean(axis=2, dtype=np.float32)

    tensor_image = torch.from_numpy(image)

    return tensor_image



if __name__ == "__main__":
    generate_and_precompute(
        n_samples=100_000,
        speed_range=(0.001, 0.006),
        rot_deg_range=(0, 0),
        split=(0.8, 0.1, 0.1),
        config=StimConfig(superimpose=False),
    )
