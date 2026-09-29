import os
import json
import math
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
from features.dog_motion import DoGFilter, SupersampledDoG
from stimuli.preprocessing_provenance import capture_provenance
from stimuli.metadata import DATA_SCHEMA

# Repo root is two levels up from this file (src/stimuli/ -> repo root), so the
# data folder tracks wherever the dw93net repo is checked out.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "stimuli")


def validate_speed_float32(values):
    """Reject nonnegative speeds that cannot be stored as finite float32."""
    source = np.asarray(values, dtype=np.float64)
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        stored = source.astype(np.float32)
    if (not np.isfinite(stored).all() or np.any(stored < 0)
            or np.any((source > 0) & (stored == 0))):
        raise ValueError('speed_range must remain finite and nonnegative in float32')


def sample_motion(rng, speed_range, heading_deg_range, rot_deg_range):
    """Sample once, then use the exact stored float32 labels for rendering/replay."""
    def sample(bounds):
        if not all(math.isfinite(value) for value in bounds):
            raise ValueError('Motion bounds must be finite for float32 targets')
        return bounds[0] if bounds[0] == bounds[1] else rng.uniform(*bounds)

    speed, heading, rotation = (sample(bounds) for bounds in
                                 (speed_range, heading_deg_range, rot_deg_range))
    validate_speed_float32(speed)
    if not np.isfinite([heading, rotation]).all():
        raise ValueError('Motion angles must be finite')
    # Cardinal directions have an exactly zero component, not a trig residual.
    cardinal = {0: (0., 1.), 90: (1., 0.), -90: (-1., 0.),
                180: (0., -1.), -180: (0., -1.)}
    sine, cosine = cardinal[heading] if heading in cardinal else (
        np.sin(np.deg2rad(heading)), np.cos(np.deg2rad(heading)))
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        target = np.asarray([speed * sine, speed * cosine, rotation], dtype=np.float32)
    if not np.isfinite(target).all():
        raise ValueError('Motion must have finite float32 targets')
    # Python floats preserve these float32 values exactly while keeping renderer
    # geometry calculations in float64, as before.
    return tuple(target.tolist())


@dataclass
class StimConfig:
    stim_wid_px: int | None = None
    fov_deg: float = 120
    near_exclusion_radius: float = 0.04
    n_world_dots: int = 1000
    superimpose: bool = False
    x_supersampling: int = 255
    apply_dog_filter_post_downsample: bool = True

    def __post_init__(self):
        if type(self.apply_dog_filter_post_downsample) is not bool:
            raise ValueError('apply_dog_filter_post_downsample must be true or false')
        fov = self.fov_deg
        if (isinstance(fov, bool) or not isinstance(fov, (int, float))
                or not math.isfinite(fov) or not 0 < fov <= 360):
            raise ValueError('fov_deg must be finite and satisfy 0 < value <= 360')
        radius = self.near_exclusion_radius
        if (isinstance(radius, bool) or not isinstance(radius, (int, float))
                or not math.isfinite(radius) or not 0 <= radius < 1):
            raise ValueError('near_exclusion_radius must be finite and satisfy 0 <= value < 1')
        if self.stim_wid_px is None:
            self.stim_wid_px = round(self.fov_deg * 60)


def _generate_flow_xt_samples(
    output_root: str = None,
    n_samples: int = 100, # 100_000
    speed_range=(0.01, 0.01),
    heading_deg_range=(-180, 180),
    rot_deg_range=(-0.5, 0.5),
    role: str = "train",
    seed: int = None,
    config: StimConfig = StimConfig(),
    sample_consumer=None,
) -> Path:
    """
    Generate one role-specific stream of two-frame, one-step XT flow tensors.
    Speed, heading, and rotation are each specified as a two-value range. Equal
    endpoints mean one fixed value; increasing endpoints define a half-open
    uniform distribution. Heading 0 degrees is forward (+Z), -90 is left (-X),
    and +90 is right (+X). The sampled polar translation is converted with
    dx=speed*sin(heading) and dz=speed*cos(heading).

    Each sample's resulting dx, dz and rot_deg are converted to float32 FIRST.
    Those exact Cartesian values are rendered and
    stored as targets, so the label always matches the stimulus. If
    `superimpose` is True,
    generates separate rotational and translational flows and combines them by summing and
    clipping to [0, 1], preserving fractional antialiased intensities. Float32
    pairs are passed directly to ``sample_consumer`` and never saved.
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
    if speed_range[0] < 0:
        raise ValueError("speed_range must have a nonnegative lower bound")
    validate_speed_float32(speed_range)
    if heading_deg_range[0] < -180 or heading_deg_range[1] > 180:
        raise ValueError("heading_deg_range must lie within [-180, 180]")
    if role not in {"train", "validate", "test"}:
        raise ValueError(f"role must be train, validate, or test. Got: {role}")

    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)

    dataset_root = Path(output_root)

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
        "target_encoding": "float32 before rendering; no nano-unit quantization",
        "rot_deg_range": range_metadata(
            rot_deg_range, rot_is_fixed, units="degrees_per_frame_step"),
        "role": role, "seed": seed, "stimulus_config": vars(config),
    }
    rng = np.random.default_rng(seed)

    for i in range(n_samples):
        dx, dz, rot_deg = sample_motion(rng, speed_range, heading_deg_range, rot_deg_range)

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
                rng=rng,
                downsample=config.apply_dog_filter_post_downsample,
            )
            xt_tensor_list.append(xt)

        # Preserve antialiased intensities; saturate overlapping flow layers.
        xt_tensor = sum(xt_tensor_list).clamp(0, 1).unsqueeze(0)  # [1, 2, X]

        sample_id = i

        sample_consumer(dataset_root, generation_metadata, role, sample_id,
                        xt_tensor, (dx, dz, rot_deg))

        if i % 100 == 0:
            print(f"Generated {role}/{sample_id}")

    return dataset_root


class _DogBundleWriter:
    """Collect generated stimuli into float32 signed DoG cache bundles."""

    def __init__(self, destination, generation_metadata, bundle_size, dog_filter, role, provenance):
        self.destination = Path(destination).resolve()
        self.generation_metadata = generation_metadata
        self.bundle_size = bundle_size
        self.dog_filter = dog_filter
        config = generation_metadata['stimulus_config']
        self.high_filter = None
        if not config['apply_dog_filter_post_downsample']:
            self.high_filter = SupersampledDoG(dog_filter, config['stim_wid_px'],
                                               config['x_supersampling'])
        self.role = role
        self.destination.mkdir(parents=True, exist_ok=False)
        self.started = datetime.now(timezone.utc).isoformat()
        self.provenance = provenance
        self.pending = []
        self.index = []
        self.bundle_number = 0
        self.raw_shape = None
        self.contrast_shape = None
        self.n_samples = 0

    def add(self, sample_id, frames, target):
        if (frames.dtype != torch.float32 or frames.ndim != 3
                or frames.shape[:2] != (1, 2)):
            raise ValueError(f"Invalid generated XT tensor: {sample_id}")
        raw_shape = list(frames.shape[1:])
        if self.raw_shape is None:
            self.raw_shape = raw_shape
        elif raw_shape != self.raw_shape:
            raise ValueError(f"Inconsistent generated XT shape: {sample_id}")
        # Filter high-resolution rasters immediately: never buffer a whole
        # bundle of supersampled images (several GB at large factors).
        if self.high_filter is not None:
            frames = self.high_filter(frames)
        # Buffer float32 frames or final low-resolution contrast.
        self.pending.append((sample_id, frames, target))
        self.n_samples += 1
        if len(self.pending) >= self.bundle_size:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        # Bound convolution workspace independently of the on-disk bundle size.
        # Small batches amortize dispatch without a full-bundle temporary.
        if self.high_filter is not None:
            contrast = torch.stack([frames for _, frames, _ in self.pending])
        else:
            filtered = []
            for start in range(0, len(self.pending), 16):
                frames = torch.stack([frames for _, frames, _ in self.pending[start:start + 16]])
                filtered.append(self.dog_filter(frames)[:, 0])
            contrast = torch.cat(filtered)
        if not torch.isfinite(contrast).all():
            raise ValueError(f'Nonfinite DoG contrast in bundle {self.bundle_number}')
        self.contrast_shape = list(contrast.shape[1:])
        folder = self.destination / self.role
        folder.mkdir(parents=True, exist_ok=True)
        name = f'bundle-{self.bundle_number:09d}.npy'
        bundle = np.empty(len(self.pending), dtype=np.dtype([
            ('dog_contrast', np.float32, tuple(self.contrast_shape)),
            ('target', np.float32, (3,)),
        ]))
        bundle['dog_contrast'] = contrast.numpy()
        bundle['target'] = np.asarray([target for _, _, target in self.pending], dtype=np.float32)
        np.save(folder / name, bundle)
        self.index.extend(
            [sample_id, name, row] for row, (sample_id, _, _) in enumerate(self.pending))
        self.pending.clear()
        self.bundle_number += 1

    def finish(self):
        self.flush()
        folder = self.destination / self.role
        folder.mkdir(parents=True, exist_ok=True)
        (folder / 'bundles.json').write_text(json.dumps(self.index) + '\n')
        metadata = {
            **self.dog_filter.metadata,
            'storage_format': DATA_SCHEMA,
            'git_commit': self.provenance['git_commit'],
            'storage_dtype': 'torch.float32',
            'finite_validated': True,
            'created_utc': self.started,
            'completed_utc': datetime.now(timezone.utc).isoformat(),
            'n_samples': self.n_samples,
            'raw_shape': self.raw_shape,
            'contrast_shape': self.contrast_shape,
            'input_encoding': 'float32 [0, 1]; no byte quantization',
            'stimulus_generation': self.generation_metadata,
            'software': self.provenance,
        }
        (self.destination / 'dog.json').write_text(json.dumps(metadata, indent=2) + '\n')
        return metadata


def generate_dog_dataset(*, dog_bundle_size=64, sigma_center=1., sigma_surround=3.,
                         progress_callback=None, provenance=None, **generation_kwargs):
    """Render stimuli in memory and persist only float32 signed DoG bundles."""
    if type(dog_bundle_size) is not int or dog_bundle_size < 1:
        raise ValueError('dog_bundle_size must be a positive integer')
    if provenance is None:
        provenance = capture_provenance(Path(generation_kwargs.get("output_root") or DEFAULT_DATA_ROOT),
                                        Path(__file__).resolve().parents[2])
    writer = None

    def consume(dataset_root, metadata, role, sample_id, frames, target):
        nonlocal writer
        if writer is None:
            writer = _DogBundleWriter(dataset_root / 'dog', metadata, dog_bundle_size,
                                      DoGFilter(sigma_center=sigma_center,
                                                sigma_surround=sigma_surround), role, provenance)
        writer.add(sample_id, frames, target)
        if progress_callback is not None:
            progress_callback(writer.n_samples)

    _generate_flow_xt_samples(**generation_kwargs, sample_consumer=consume)
    if writer is None:
        raise ValueError('Cannot create an empty DoG dataset')
    writer.finish()
    print(f"\n✅ Float32 signed DoG bundles saved to {writer.destination}")
    return str(writer.destination)



def generate_flow_xt_stim(
    cam_trans_xz=[0, 1e-3],
    cam_rot_deg=0,
    config: StimConfig = StimConfig(),
    rng=None,
    *,
    downsample=True,
) -> torch.Tensor:

    """Render [2, X]: initial dots and the same dots after one motion step.

    Translation is expressed in the initial camera frame; rotation is in degrees
    per step. Dots are neither replaced nor wrapped between the two frames.
    Set downsample=False to inspect/filter the supersampled float32 raster.
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

        # atan2 spans the entire camera-centred circle, unlike x/z and
        # tan(half_fov), which become ambiguous behind the camera and invert
        # for fields wider than 180 degrees. The ordinary boundary remains
        # open, matching the previous renderer; 360 degrees includes the
        # +/-pi seam because it has no outside region.
        theta = np.arctan2(x, z)
        if config.fov_deg == 360:
            # Canonicalize the two numeric representations of the rearward
            # direction onto the drawable right edge of the projection.
            theta = np.where(theta == -np.pi, np.pi, theta)
            angle_in_fov = np.ones(theta.shape, dtype=bool)
        else:
            angle_in_fov = np.abs(theta) < half_fov_rad

        visible = (
            inside_unit_circle &
            outside_near_exclusion &
            angle_in_fov
        )

        if np.any(visible):
            x_proj = theta[visible] / half_fov_rad

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
    if downsample and x_supersampling > 1:
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
