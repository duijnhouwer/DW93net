# -*- coding: utf-8 -*-

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from typing import Dict


def generate_flow_xt_stim(
    cam_trans_xz=[0, 1e-3],
    cam_rot_deg=0,
    n_dots=1000,
    stim_wid_px=1028,
    n_time_steps=128,
    fov_deg=90,
    near_clip=0.05,
    filename=''
) -> torch.Tensor:
    """
    Simulates optic flow and returns a space-time plot as a PyTorch tensor of shape (n_time_steps, stim_wid_px).
    The projection uses a 2D trapezoidal frustum defined by fov_deg and near_clip.
    """
    cam_trans_xz = np.array(cam_trans_xz, dtype=np.float64)
    half_fov_rad = np.deg2rad(fov_deg / 2)
    cam_rot_rad = np.deg2rad(cam_rot_deg)
    tan_half_fov = np.tan(half_fov_rad)
    far_clip = np.cos(half_fov_rad)

    dot_positions = np.random.uniform(-1, 1, size=(n_dots, 2))
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
            dot_positions[out_of_bounds, other_axis] = np.random.uniform(-1, 1, size=out_of_bounds.sum())

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


def generate_flow_xt_dataset(
    output_root: str,
    n_samples: int = 1000,
    dx_range=(0.0, 0.0),
    dz_range=(0.0, 0.0),
    rot_deg_range=(-0.5, 0.5),
    seed: int = 42,
    stim_wid_px: int = 1028,
    n_time_steps: int = 128,
    fov_deg: float = 90,
    near_clip: float = 0.05
) -> str:
    """
    Generates a dataset of XT flow tensors and saves them in a subfolder whose name encodes
    the parameter ranges using micro-units (udx, udz, udeg). Filenames also contain per-sample values.
    Returns the path to the output folder.
    """

    def validate_range(name, r):
        r = np.asarray(r)
        if r.shape != (2,) or r[0] > r[1]:
            raise ValueError(f"{name} must be a tuple/list with two values where r[0] <= r[1]. Got: {r}")

    validate_range('dx_range', dx_range)
    validate_range('dz_range', dz_range)
    validate_range('rot_deg_range', rot_deg_range)

    udx_min = int(round(dx_range[0] * 1e6))
    udx_max = int(round(dx_range[1] * 1e6))
    udz_min = int(round(dz_range[0] * 1e6))
    udz_max = int(round(dz_range[1] * 1e6))
    udeg_min = int(round(rot_deg_range[0] * 1e6))
    udeg_max = int(round(rot_deg_range[1] * 1e6))

    folder_name = f"udx={udx_min},{udx_max}_udz={udz_min},{udz_max}_udeg={udeg_min},{udeg_max}"
    output_folder = os.path.join(output_root, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    rng = np.random.default_rng(seed)

    for i in range(n_samples):
        dx = rng.uniform(*dx_range)
        dz = rng.uniform(*dz_range)
        rot_deg = rng.uniform(*rot_deg_range)

        xt_tensor = generate_flow_xt_stim(
            cam_trans_xz=[dx, dz],
            cam_rot_deg=rot_deg,
            stim_wid_px=stim_wid_px,
            n_time_steps=n_time_steps,
            fov_deg=fov_deg,
            near_clip=near_clip,
            filename=''
        )

        udx = int(round(dx * 1e6))
        udz = int(round(dz * 1e6))
        udeg = int(round(rot_deg * 1e6))

        filename = f"flow_udx={udx}_udz={udz}_udeg={udeg}_id={i}.pt"
        filepath = os.path.join(output_folder, filename)

        torch.save((xt_tensor.to(torch.uint8), [dx, dz, rot_deg]), filepath)

        if i % 100 == 0:
            print(f"Saved {filename}")

    print(f"\n✅ Done: {n_samples} samples saved to {output_folder}")
    return output_folder


def parse_camera_motion_from_filename(filename: str) -> Dict[str, float]:
    """
    Extract dx, dz, and rot_deg (in physical units) from filename using micro-units.

    Example:
        flow_udx=1234_udz=0_udeg=-1800000_id=42.pt

    Returns:
        Dictionary with keys: 'dx', 'dz', 'rot_deg' (values in meters and degrees)
    """
    pattern = r"udx=(-?\d+)_udz=(-?\d+)_udeg=(-?\d+)"
    match = re.search(pattern, filename)
    if not match:
        raise ValueError(f"Filename '{filename}' does not match expected pattern.")

    udx, udz, udeg = map(int, match.groups())
    return {
        'dx': udx / 1e6,
        'dz': udz / 1e6,
        'rot_deg': udeg / 1e6
    }


def visualize_xt_sample(folder: str, id):
    """
    Visualize a saved XT flow sample.

    Parameters:
    - folder: path to the dataset folder
    - id:
        - an integer (e.g., 17 → finds file containing _id=17.pt)
        - a full filename (must end in '.pt')
        - the string 'random' → picks a random .pt file from folder
    """
    import matplotlib.pyplot as plt
    import random

    if id == 'random':
        pt_files = [f for f in os.listdir(folder) if f.endswith('.pt')]
        if not pt_files:
            raise FileNotFoundError(f"No .pt files found in {folder}")
        filename = random.choice(pt_files)

    elif isinstance(id, int):
        match = None
        for f in os.listdir(folder):
            if f.endswith('.pt') and f"_id={id}.pt" in f:
                match = f
                break
        if match is None:
            raise FileNotFoundError(f"No file with _id={id}.pt found in {folder}")
        filename = match

    elif isinstance(id, str) and id.endswith('.pt'):
        filename = id

    else:
        raise ValueError("id must be an integer, 'random', or a filename ending in '.pt'")

    path = os.path.join(folder, filename)
    xt_tensor, motion = torch.load(path)
    dx, dz, rot_deg = motion

    plt.figure(figsize=(10, 4))
    plt.imshow(xt_tensor.numpy(), cmap='gray', vmin=0, vmax=1, aspect='auto')
    plt.title(f'dx={dx:.6f} m, dz={dz:.6f} m, rot={rot_deg:.6f} deg\n{filename}')
    plt.xlabel('X position (pixels)')
    plt.ylabel('Time (frames)')
    plt.tight_layout()
    plt.show()
