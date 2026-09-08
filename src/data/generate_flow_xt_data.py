import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def generate_flow_xt_dataset(
    output_root: str,
    n_samples: int = 100_000, # 100_000
    dx_range=(-0.01, 0.01),
    dz_range=(-0.01, 0.01),
    rot_deg_range=(-0.5, 0.5),
    split=(0.8, 0.1, 0.1),
    seed: int = None,
    stim_wid_px: int = 1028,
    n_time_steps: int = 128,
    fov_deg: float = 90,
    near_clip: float = 0.05,
    superimpose: bool = False,
    n_world_dots: int = 1000
) -> str:
    """
    Generates a dataset of XT flow tensors and saves them into train/validate/test subfolders.
    Folder name encodes parameter ranges using micro-units (udx, udz). Filenames include dx, dz
    in micro-units and rot in float degrees. If `superimpose` is True, generates separate
    rotational and translational flows and combines them by summing, then binarizing the result.
    """

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

    # Generate folder name using µ-units for dx and dz only
    udx_min = int(round(dx_range[0] * 1e6))
    udx_max = int(round(dx_range[1] * 1e6))
    udz_min = int(round(dz_range[0] * 1e6))
    udz_max = int(round(dz_range[1] * 1e6))
    rot_min = round(rot_deg_range[0], 3)
    rot_max = round(rot_deg_range[1], 3)

    folder_name = f"udx={udx_min},{udx_max}_udz={udz_min},{udz_max}_rot={rot_min},{rot_max}"
    if superimpose:
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

        # Generate one or two tensors depending on `superimpose`
        xt_tensor_list = []

        for layer in range(2 if superimpose else 1):
            n_layer_dots = n_world_dots
            cam_trans = [dx, dz]
            cam_rot = rot_deg
            if superimpose:
                n_layer_dots = n_layer_dots//2
                if layer == 0:
                    cam_trans = [0, 0]
                else:
                    cam_rot = 0


            xt = generate_flow_xt_stim(
                cam_trans_xz=cam_trans,
                cam_rot_deg=cam_rot,
                stim_wid_px=stim_wid_px,
                n_time_steps=n_time_steps,
                fov_deg=fov_deg,
                n_world_dots = n_layer_dots,
                near_clip=near_clip,
                filename='',
                rng=rng
            )
            xt_tensor_list.append(xt)

        # Sum tensors and binarize
        xt_tensor = sum(xt_tensor_list)
        xt_tensor = (xt_tensor > 0).float()
        xt_tensor = xt_tensor.unsqueeze(0)  # shape [1, T, X]

        udx = int(round(dx * 1e6))
        udz = int(round(dz * 1e6))
        filename = f"flow_udx={udx}_udz={udz}_rot={rot_deg:.6f}_id={i}"
        if superimpose:
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
    n_world_dots=1000,
    stim_wid_px=1028,
    n_time_steps=128,
    fov_deg=90,
    near_clip=0.05,
    filename='',
    rng=None
) -> torch.Tensor:
    """
    Simulates optic flow and returns a space-time plot as a PyTorch tensor of shape (n_time_steps, stim_wid_px).
    The projection uses a 2D trapezoidal frustum defined by fov_deg and near_clip.

    rng: optional numpy Generator (or legacy RandomState) used for dot placement.
    Defaults to the global np.random for backward compatibility. Pass the dataset's
    seeded rng from generate_flow_xt_dataset for reproducible generation.
    """
    if rng is None:
        rng = np.random
    cam_trans_xz = np.array(cam_trans_xz, dtype=np.float64)
    half_fov_rad = np.deg2rad(fov_deg / 2)
    cam_rot_rad = np.deg2rad(cam_rot_deg)
    tan_half_fov = np.tan(half_fov_rad)
    far_clip = np.cos(half_fov_rad)

    dot_positions = rng.uniform(-1, 1, size=(n_world_dots, 2))
    image = np.zeros((n_time_steps, stim_wid_px), dtype=np.float32)

    for t in range(n_time_steps):
        dot_positions -= cam_trans_xz

        for axis in [0, 1]:
            if cam_trans_xz[axis] > 0:
                out_of_bounds = dot_positions[:, axis] < -1
            else:
                out_of_bounds = dot_positions[:, axis] > 1

            dot_positions[out_of_bounds, axis] += 2 if cam_trans_xz[axis] > 0 else -2
            other_axis = 1 - axis
            dot_positions[out_of_bounds, other_axis] = rng.uniform(-1, 1, size=out_of_bounds.sum())

        radius_squared = np.sum(dot_positions**2, axis=1)
        in_unit_circle = radius_squared < 1
        if np.any(in_unit_circle):
            c, s = np.cos(-cam_rot_rad), np.sin(-cam_rot_rad)
            rot = np.array([[c, -s], [s, c]])
            dot_positions[in_unit_circle] = dot_positions[in_unit_circle] @ rot.T

        x, z = dot_positions[:, 0], dot_positions[:, 1]
        z_in_bounds = (z > near_clip) & (z < far_clip)
        x_in_fov = np.abs(x / z) < tan_half_fov
        visible = in_unit_circle & z_in_bounds & x_in_fov

        if np.any(visible):
            x_visible = x[visible]
            z_visible = z[visible]
            theta = np.arctan2(x_visible, z_visible)
            x_proj = theta / half_fov_rad
            x_scaled = ((x_proj + 1) / 2 * (stim_wid_px - 1)).astype(int)
            x_scaled = np.clip(x_scaled, 0, stim_wid_px - 1)
            image[t, x_scaled] = 1.0

    tensor_image = torch.from_numpy(image)

    if filename:
        if not filename.lower().endswith('.png'):
            raise ValueError("filename must end with '.png'")
        plt.imsave(filename, image, cmap='gray', vmin=0, vmax=1)

    return tensor_image



if __name__ == "__main__":
    generate_flow_xt_dataset(
        output_root="F:/dw93net/data", # ← update path as needed)
        superimpose= True
        );
