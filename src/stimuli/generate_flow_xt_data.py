import os
import json
import sys
from datetime import datetime, timezone

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
from features.dog_motion import DoGFilter
from stimuli.preprocessing_provenance import write_provenance, repository_state
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
    role: str = "train",
    seed: int = None,
    config: StimConfig = StimConfig(),
    _sample_consumer=None,
    _save_raw: bool = True,
) -> str:
    """
    Generate one role-specific dataset of two-frame, one-step XT flow tensors.
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
    When raw saving is enabled, each file contains one uint8 pair with shape
    [1, 2, X], scaled to 0..255, under the requested role directory.
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

    speed_range, speed_is_fixed = validate_range('speed_range', speed_range)
    heading_deg_range, heading_is_fixed = validate_range(
        'heading_deg_range', heading_deg_range)
    rot_deg_range, rot_is_fixed = validate_range('rot_deg_range', rot_deg_range)
    if speed_range[0] <= 0:
        raise ValueError("speed_range must have a strictly positive lower bound")
    if heading_deg_range[0] < -180 or heading_deg_range[1] > 180:
        raise ValueError("heading_deg_range must lie within [-180, 180]")
    if role not in {"train", "validate", "test"}:
        raise ValueError(f"role must be train, validate, or test. Got: {role}")

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

    if _save_raw:
        os.makedirs(os.path.join(output_folder, role), exist_ok=True)

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
        "role": role, "seed": seed, "stimulus_config": vars(config),
    }
    if _save_raw:
        with open(os.path.join(output_folder, "generation.json"), "w") as metadata_file:
            json.dump(generation_metadata, metadata_file, indent=2)
            metadata_file.write("\n")

    rng = np.random.default_rng(seed)

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

        # Round rather than truncate fractional intensities. At 255x
        # supersampling the rendered levels align with the byte grayscale grid.
        encoded = (xt_tensor * 255).round().to(torch.uint8)
        if _sample_consumer is not None:
            _sample_consumer(Path(output_folder), generation_metadata, role,
                             filename, encoded, (dx, dz, rot_deg))
        if _save_raw:
            filepath = os.path.join(output_folder, role, filename)
            torch.save(encoded, filepath)

        if i % 100 == 0:
            action = "Saved" if _save_raw else "Generated"
            print(f"{action} {role}/{filename}")

    if _save_raw:
        print(f"\n✅ Done: {n_samples} samples saved to {output_folder}")
    return output_folder


class _DogBundleWriter:
    """Collect generated stimuli into float32 signed DoG cache bundles."""

    def __init__(self, destination, generation_metadata, bundle_size, dog_filter, role):
        self.destination = Path(destination).resolve()
        self.generation_metadata = generation_metadata
        self.bundle_size = bundle_size
        self.dog_filter = dog_filter
        self.role = role
        self.destination.mkdir(parents=True, exist_ok=False)
        self.started = datetime.now(timezone.utc).isoformat()
        self.repo = Path(__file__).resolve().parents[2]
        self.provenance, self.initial_state = write_provenance(self.destination, self.repo)
        self.pending = []
        self.index = []
        self.bundle_number = 0
        self.raw_shape = None
        self.contrast_shape = None
        self.n_samples = 0

    def add(self, filename, encoded, target):
        if (encoded.dtype != torch.uint8 or encoded.ndim != 3
                or encoded.shape[:2] != (1, 2)):
            raise ValueError(f"Invalid generated XT tensor: {filename}")
        raw_shape = list(encoded.shape[1:])
        if self.raw_shape is None:
            self.raw_shape = raw_shape
        elif raw_shape != self.raw_shape:
            raise ValueError(f"Inconsistent generated XT shape: {filename}")
        contrast = self.dog_filter(encoded.float() / 255)[0, 0]
        if not torch.isfinite(contrast).all():
            raise ValueError(f"Nonfinite DoG contrast: {filename}")
        if self.contrast_shape is None:
            self.contrast_shape = list(contrast.shape)
        elif list(contrast.shape) != self.contrast_shape:
            raise ValueError(f"Inconsistent DoG contrast shape: {filename}")
        self.pending.append((filename, contrast, target))
        self.n_samples += 1
        if len(self.pending) >= self.bundle_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        folder = self.destination / self.role
        folder.mkdir(parents=True, exist_ok=True)
        name = f'bundle-{self.bundle_number:09d}.npy'
        bundle = np.empty(len(self.pending), dtype=np.dtype([
            ('dog_contrast', np.float32, tuple(self.contrast_shape)),
            ('target', np.float32, (3,)),
        ]))
        bundle['dog_contrast'] = np.stack([value.numpy() for _, value, _ in self.pending])
        bundle['target'] = np.asarray([target for _, _, target in self.pending], dtype=np.float32)
        np.save(folder / name, bundle)
        self.index.extend(
            [filename, name, row] for row, (filename, _, _) in enumerate(self.pending))
        self.pending.clear()
        self.bundle_number += 1

    def finish(self):
        self.flush()
        folder = self.destination / self.role
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'bundles.json').write_text(json.dumps(self.index) + '\n')
        if repository_state(self.repo, self.destination) != self.initial_state:
            raise RuntimeError("Source changed during generation; cache left incomplete")
        metadata = {
            **self.dog_filter.metadata,
            'storage_format': 'dog-contrast-bundles-v1',
            'storage_dtype': 'torch.float32',
            'finite_validated': True,
            'created_utc': self.started,
            'completed_utc': datetime.now(timezone.utc).isoformat(),
            'n_samples': self.n_samples,
            'raw_shape': self.raw_shape,
            'contrast_shape': self.contrast_shape,
            'input_encoding': 'uint8 / 255',
            'stimulus_generation': self.generation_metadata,
            'software': self.provenance,
        }
        (self.destination / 'dog.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata


def generate_dog_dataset(*, dog_bundle_size=64, sigma_center=1., sigma_surround=3.,
                         **generation_kwargs):
    """Render stimuli in memory and persist only float32 signed DoG bundles."""
    if type(dog_bundle_size) is not int or dog_bundle_size < 1:
        raise ValueError('dog_bundle_size must be a positive integer')
    writer = None

    def consume(raw_folder, metadata, role, filename, encoded, target):
        nonlocal writer
        if writer is None:
            writer = _DogBundleWriter(raw_folder.parent / 'dog', metadata, dog_bundle_size,
                                      DoGFilter(sigma_center=sigma_center,
                                                sigma_surround=sigma_surround), role)
        # Keep a readable stable identifier without implying that a per-sample
        # .pt file exists in this cache.
        writer.add(Path(filename).stem, encoded, target)

    raw_folder = Path(generate_flow_xt_dataset(
        **generation_kwargs, _sample_consumer=consume, _save_raw=False)).resolve()
    if writer is None:
        raise ValueError('Cannot create an empty DoG dataset')
    writer.finish()
    print(f"\n✅ Float32 signed DoG bundles saved to {writer.destination}")
    return str(writer.destination)


def generate_and_precompute(*, precomputed_root=None, stimulus_bundle_size=64,
                            **generation_kwargs):
    """Generate raw XT stimuli, then cache their default DoG product maps."""
    raw_folder = Path(generate_flow_xt_dataset(**generation_kwargs)).resolve()
    product_folder = (raw_folder.parent / "reichardt" if precomputed_root is None
                      else Path(precomputed_root).resolve() / raw_folder.parent.name / "reichardt")
    precompute_dataset(raw_folder, product_folder,
                       stimulus_bundle_size=stimulus_bundle_size)
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
    generate_dog_dataset(
        n_samples=100_000,
        speed_range=(0.001, 0.006),
        rot_deg_range=(0, 0),
        role="train",
        config=StimConfig(superimpose=False),
    )
