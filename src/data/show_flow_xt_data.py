import os
import random
import torch
import matplotlib.pyplot as plt
from load_flow_xt_data import parse_camera_motion_from_filename  # adjust if needed

def show_xt_sample(folder: str, id):
    """
    Visualize a saved XT flow sample.

    Parameters:
    - folder: path to the dataset folder
    - id:
        - an integer (e.g., 17 → finds file containing _id=17.pt)
        - a full filename (must end in '.pt')
        - the string 'random' → picks a random .pt file from folder
    """
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
    xt_tensor_uint8 = torch.load(path)

    # Fix shape if it's (1, H, W)
    if xt_tensor_uint8.ndim == 3 and xt_tensor_uint8.shape[0] == 1:
        xt_tensor_uint8 = xt_tensor_uint8.squeeze(0)

    xt_tensor = xt_tensor_uint8.float()

    motion = parse_camera_motion_from_filename(filename)
    dx, dz, rot_deg = motion['dx'], motion['dz'], motion['rot_deg']

    plt.figure(figsize=(10, 4))
    plt.imshow(xt_tensor.numpy(), cmap='gray', vmin=0, vmax=1, aspect='auto')
    plt.title(f'dx={dx:.6f} m, dz={dz:.6f} m, rot={rot_deg:.6f} deg\n{filename}')
    plt.xlabel('X position (pixels)')
    plt.ylabel('Time (frames)')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Display a saved XT flow stimulus.")
    parser.add_argument("folder", type=str, help="Path to folder with .pt files")
    parser.add_argument("id", help="Sample ID: integer (e.g., 42), 'random', or full .pt filename")

    args = parser.parse_args()

    try:
        id_val = int(args.id)
    except ValueError:
        id_val = args.id

    show_xt_sample(args.folder, id_val)
