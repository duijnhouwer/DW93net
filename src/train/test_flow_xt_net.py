import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from models.flow_xt_net import FlowXTNet
from data.load_flow_xt_data import FlowXtDataset
from data.load_flow_xt_data import extract_ranges_from_foldername, subset_dataset
from data.utils import select_subfolder
import sys

# --- Data selection
data_root = "F:/dw93net/data"
data_folder = select_subfolder(data_root,header_str="\nSelect a data folder")
if data_folder is None:
    print("No data folder selected. Exiting.")
    sys.exit()
data_folder_name = os.path.basename(data_folder)

# --- Model selection
checkpoints_root = "F:/dw93net/checkpoints"
checkpoints_folder = select_subfolder(checkpoints_root,header_str="\nSelect a checkpoints folder")
if checkpoints_folder is None:
    print("No checpoints folder selected. Exiting.")
    sys.exit()
checkpoint_path = os.path.join(checkpoints_root, checkpoints_folder, "_best.pt")

checkpoint = torch.load(checkpoint_path, map_location="cpu")
settings = checkpoint.get("settings", {})  # a plain dict
root_folder = settings.get("root_folder", data_folder)
batch_size = settings.get("batch_size", 64)

# --- Parameter ranges
ranges = extract_ranges_from_foldername(os.path.basename(root_folder))
dx_range = ranges['dx']
dz_range = ranges['dz']
rot_range = ranges['rot_deg']

# --- Device and model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FlowXTNet().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# --- Load test data
test_folder = os.path.join(root_folder, "test")
test_dataset = FlowXtDataset(folder=test_folder)
test_dataset = subset_dataset(test_dataset, {"n": 1000, "method": "random"})
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Gather predictions
all_targets = []
all_preds = []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        pred = model(x).cpu().numpy()
        all_preds.append(pred)
        all_targets.append(y.numpy())

all_preds = np.vstack(all_preds)
all_targets = np.vstack(all_targets)

# --- Plotting
labels = ['dx', 'dz', 'rot_deg']
for i, label in enumerate(labels):
    true_vals = all_targets[:, i].reshape(-1, 1)
    pred_vals = all_preds[:, i].reshape(-1, 1)
    N = true_vals.shape[0]

    reg = LinearRegression().fit(true_vals, pred_vals)
    fit_vals = reg.predict(true_vals)
    r2 = r2_score(pred_vals, fit_vals)

    plt.figure(figsize=(6, 6))
    plt.scatter(true_vals, pred_vals, alpha=0.3, label='Predictions')
    plt.plot(true_vals, fit_vals, color='red',
             label=f'Regression (R²={r2:.3f}, N={N})')
    plt.plot(true_vals, true_vals, color='gray', linestyle='--', label='Ideal')
    plt.xlabel(f"True {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"Prediction vs. Ground Truth: {label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"scatter_{label}.png", dpi=150)
    plt.show()
