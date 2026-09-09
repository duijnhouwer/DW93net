import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from dataclasses import dataclass, replace

# Repo root is two levels up from this file (src/data/ -> repo root), so the
# data folder tracks wherever the dw93net repo is checked out.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_DATA_ROOT = os.path.join(_REPO_ROOT, "data")


@dataclass
class StimConfig:
    stim_wid_px: int = 1028
    n_time_bins: int = 128
    fov_deg: float = 90
    near_clip: float = 0.05
    n_world_dots: int = 1000
    superimpose: bool = False
    max_dot_lifetime: int = 10 # max lifetime of a dot in frames; if <=0, dots never die


def generate_flow_xt_dataset(
    output_root: str = None,
    n_samples: int = 100, # 100_000
    dx_range=(-0.01, 0.01),
    dz_range=(-0.01, 0.01),
    rot_deg_range=(-0.5, 0.5),
    split=(0.8, 0.1, 0.1),
    seed: int = None,
    config: StimConfig = StimConfig()
) -> str:
    """
    Generates a dataset of XT flow tensors and saves them into train/validate/test subfolders.
    Folder name encodes parameter ranges in nano-units (nux, nuz) and nano-degrees (nrot).
    Each sample's dx, dz and rot_deg are rounded to nano-units/nano-degrees FIRST, and those
    exact values are used to render the stimulus and written to the filename (nux, nuz, nrot),
    so the label always matches the stimulus that was generated. If `superimpose` is True,
    generates separate rotational and translational flows and combines them by summing, then
    binarizing the result.
    """

    if output_root is None:
        output_root = DEFAULT_DATA_ROOT

    def validate_range(name, r):
        r = np.asarray(r)
        if r.shape != (2,) or r[0] > r[1]:
            raise ValueError(f"{name} must be a tuple/list with two values where r[0] <= r[1]. Got: {r}")

    def validate_split(split):
        if not np.isclose(sum(split), 1.0):
            raise ValueError(f"split must sum to 1. Got: {split}")

    validate_range('dx_range', dx_range)
    validate_range('dz_range', dz_range)
    validate_range('rot_deg_range', rot_deg_range)
    validate_split(split)

    if seed is None:
        import time
        seed = int(time.time() * 1000) % (2**32)

    # Generate folder name using nano-units for dx and dz and nano-degrees for rotation
    nux_min = int(round(dx_range[0] * 1e9))
    nux_max = int(round(dx_range[1] * 1e9))
    nuz_min = int(round(dz_range[0] * 1e9))
    nuz_max = int(round(dz_range[1] * 1e9))
    nrot_min = int(round(rot_deg_range[0] * 1e9))
    nrot_max = int(round(rot_deg_range[1] * 1e9))

    folder_name = f"nux={nux_min},{nux_max}_nuz={nuz_min},{nuz_max}_nrot={nrot_min},{nrot_max}"
    if config.superimpose:
        folder_name += "_superimposed"
    output_folder = os.path.join(output_root, folder_name)

    # Create subfolders
    for sub in ["train", "validate", "test"]:
        os.makedirs(os.path.join(output_folder, sub), exist_ok=True)

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
        dx = rng.uniform(*dx_range)
        dz = rng.uniform(*dz_range)
        rot_deg = rng.uniform(*rot_deg_range)

        # Round to nano-units/nano-degrees FIRST, then use those exact values for BOTH
        # stimulus generation and the filename, so the label always matches the stimulus.
        nux = int(round(dx * 1e9))
        nuz = int(round(dz * 1e9))
        nrot = int(round(rot_deg * 1e9))
        dx, dz, rot_deg = nux / 1e9, nuz / 1e9, nrot / 1e9

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
                filename='',
                rng=rng
            )
            xt_tensor_list.append(xt)

        # Sum tensors and binarize
        xt_tensor = sum(xt_tensor_list)
        xt_tensor = (xt_tensor > 0).float()
        xt_tensor = xt_tensor.unsqueeze(0)  # shape [1, T, X]

        filename = f"flow_nux={nux}_nuz={nuz}_nrot={nrot}_id={i}"
        if config.superimpose:
            filename += "_superimposed"
        filename += ".pt"

        split_name = split_map[i]
        filepath = os.path.join(output_folder, split_name, filename)

        torch.save(xt_tensor, filepath)

        if i % 100 == 0:
            print(f"Saved {split_name}/{filename}")

    print(f"\n✅ Done: {n_samples} samples saved to {output_folder}")
    return output_folder



def generate_flow_xt_stim(
    cam_trans_xz=[0, 1e-3],
    cam_rot_deg=0,
    config: StimConfig = StimConfig(),
    filename='',
    rng=None
) -> torch.Tensor:
    """
    Simulates optic flow and returns a space-time plot as a PyTorch tensor
    of shape (n_time_bins, stim_wid_px).

    Dot lifetime:
        If config.max_dot_lifetime > 0, each dot is rendered exactly
        max_dot_lifetime times before being replaced by a new dot at a
        random position in the [-1, 1] x [-1, 1] XZ plane.

        Thus, a dot with lifetime L undergoes exactly L-1 motion
        displacements between its L rendered appearances.

        If max_dot_lifetime <= 0, dots never expire.

    rng: optional numpy Generator (or legacy RandomState) used for dot placement.
    Defaults to the global np.random for backward compatibility.
    """
    if rng is None:
        rng = np.random

    cam_trans_xz = np.array(cam_trans_xz, dtype=np.float64)

    half_fov_rad = np.deg2rad(config.fov_deg / 2)
    cam_rot_rad = np.deg2rad(cam_rot_deg)
    tan_half_fov = np.tan(half_fov_rad)
    far_clip = np.cos(half_fov_rad)

    # Initial dot positions in the 2x2 XZ plane.
    dot_positions = rng.uniform(
        -1, 1,
        size=(config.n_world_dots, 2)
    )

    # Number of times the current incarnation of each dot has been rendered.
    dot_ages = np.zeros(config.n_world_dots, dtype=np.int32)

    image = np.zeros(
        (config.n_time_bins, config.stim_wid_px),
        dtype=np.float32
    )

    # Rotation applied between successive rendered frames.
    c = np.cos(-cam_rot_rad)
    s = np.sin(-cam_rot_rad)
    rot = np.array([[c, -s], [s, c]])

    for t in range(config.n_time_bins):

        # --------------------------------------------------------------
        # 1. RENDER CURRENT DOT POSITIONS
        # --------------------------------------------------------------
        radius_squared = np.sum(dot_positions ** 2, axis=1)
        in_unit_circle = radius_squared < 1

        x = dot_positions[:, 0]
        z = dot_positions[:, 1]

        z_in_bounds = (
            (z > config.near_clip) &
            (z < far_clip)
        )

        x_in_fov = np.abs(x / z) < tan_half_fov

        visible = (
            in_unit_circle &
            z_in_bounds &
            x_in_fov
        )

        if np.any(visible):
            x_visible = x[visible]
            z_visible = z[visible]

            theta = np.arctan2(x_visible, z_visible)
            x_proj = theta / half_fov_rad

            x_scaled = (
                (x_proj + 1) / 2 *
                (config.stim_wid_px - 1)
            ).astype(int)

            x_scaled = np.clip(
                x_scaled,
                0,
                config.stim_wid_px - 1
            )

            image[t, x_scaled] = 1.0

        # Current positions have now been rendered once more.
        dot_ages += 1

        # There is no need to update positions after the final frame.
        if t == config.n_time_bins - 1:
            continue

        # --------------------------------------------------------------
        # 2. EXPIRE DOTS THAT HAVE REACHED THEIR MAXIMUM LIFETIME
        # --------------------------------------------------------------
        if config.max_dot_lifetime > 0:
            expired = dot_ages >= config.max_dot_lifetime
        else:
            expired = np.zeros(
                config.n_world_dots,
                dtype=bool
            )

        # Expired dots are reborn at completely new random XZ positions.
        if np.any(expired):
            dot_positions[expired] = rng.uniform(
                -1, 1,
                size=(expired.sum(), 2)
            )
            dot_ages[expired] = 0

        # --------------------------------------------------------------
        # 3. MOVE ONLY THE DOTS THAT DID NOT EXPIRE
        #
        # This is what guarantees:
        #
        #   lifetime L -> L renders and L-1 displacements
        # --------------------------------------------------------------
        alive = ~expired

        if np.any(alive):

            # Camera translation.
            dot_positions[alive] -= cam_trans_xz

            # Preserve the existing boundary/recycling behavior.
            for axis in [0, 1]:

                if cam_trans_xz[axis] > 0:
                    out_of_bounds = (
                        alive &
                        (dot_positions[:, axis] < -1)
                    )
                else:
                    out_of_bounds = (
                        alive &
                        (dot_positions[:, axis] > 1)
                    )

                dot_positions[out_of_bounds, axis] += (
                    2 if cam_trans_xz[axis] > 0 else -2
                )

                other_axis = 1 - axis

                dot_positions[out_of_bounds, other_axis] = rng.uniform(
                    -1,
                    1,
                    size=out_of_bounds.sum()
                )

            # Camera rotation.
            radius_squared = np.sum(
                dot_positions ** 2,
                axis=1
            )

            rotate_dots = alive & (radius_squared < 1)

            if np.any(rotate_dots):
                dot_positions[rotate_dots] = (
                    dot_positions[rotate_dots] @ rot.T
                )

    tensor_image = torch.from_numpy(image)

    if filename:
        if not filename.lower().endswith('.png'):
            raise ValueError("filename must end with '.png'")

        plt.imsave(
            filename,
            image,
            cmap='gray',
            vmin=0,
            vmax=1
        )

    return tensor_image



if __name__ == "__main__":
    generate_flow_xt_dataset(config=StimConfig(superimpose=True))
