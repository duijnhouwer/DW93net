import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from models.pure_translation import PureTranslationNet
from data.load_flow_xt_data import ReichardtDataset
from data.load_flow_xt_data import subset_dataset, make_flow_loader, require_pure_translation, require_pure_translation_root
from data.utils import select_subfolder
import sys

def run_flow_xt_model_on_test_data(
    data_folder: str,
    checkpoint_path: str,
    subset_size: int = 1000,
    num_workers: int = 4,
) -> dict:
    """
    Loads a trained PureTranslationNet model and applies it to test data.

    Args:
        data_folder: Path to the root of the data folder.
        checkpoint_path: Path to the model checkpoint (.pt file).
        subset_size: Number of random test samples to evaluate.
        num_workers: Background loader workers; use 0 for in-process loading.

    Returns:
        Dictionary with keys:
            'preds'   → ndarray of predicted values [N x 2]
            'targets' → ndarray of ground truth values [N x 2]
    """

    require_pure_translation_root(data_folder)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    batch_size = checkpoint.get("settings", {}).get("batch_size", 64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if "reichardt" not in checkpoint or "n_input_channels" not in checkpoint:
        raise ValueError("Expected a checkpoint trained on Reichardt response maps")
    if checkpoint.get("architecture") != PureTranslationNet.architecture:
        raise ValueError("Expected a pure-translation checkpoint")
    model = PureTranslationNet(checkpoint["reichardt"]["dx_values"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_folder = os.path.join(data_folder, "test")
    test_dataset = ReichardtDataset(folder=test_folder, expected_metadata=checkpoint["reichardt"])
    require_pure_translation(test_dataset)
    test_dataset = subset_dataset(test_dataset, {"n": subset_size, "method": "random"})
    test_loader = make_flow_loader(test_dataset, batch_size, device, num_workers=num_workers)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            pred = model(x).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y[:, :2].numpy())

    return {
        'preds': np.vstack(all_preds),
        'targets': np.vstack(all_targets)
    }


def regress_labels_and_estimates(results: dict):
    """
    Perform linear regression and generate scatter plots comparing
    predicted and ground truth values for each label.

    Args:
        results: Dictionary with keys 'preds' and 'targets',
                 each a numpy array of shape [N x 2] corresponding to dx and dz.
    """
    labels = ['dx', 'dz']
    preds = results['preds']
    targets = results['targets']

    for i, label in enumerate(labels):
        true_vals = targets[:, i].reshape(-1, 1)
        pred_vals = preds[:, i].reshape(-1, 1)
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


def analyze_heading_error_vs_heading(results: dict):
    """
    Analyzes heading direction estimation error vs. true heading direction.

    Args:
        results: Dictionary with keys 'preds' and 'targets',
                 each an [N x 2] numpy array for dx, dz.
    """
    dx_true = results['targets'][:, 0]
    dz_true = results['targets'][:, 1]
    dx_pred = results['preds'][:, 0]
    dz_pred = results['preds'][:, 1]

    # Compute true and predicted heading angles in degrees
    heading_true = np.degrees(np.arctan2(dx_true, dz_true))  # range [-180, 180]

    # Heading is undefined for zero true or predicted translation.
    valid = (np.hypot(dx_true, dz_true) > 0) & (np.hypot(dx_pred, dz_pred) > 0)
    heading_true = heading_true[valid]
    dx_true, dz_true = dx_true[valid], dz_true[valid]
    dx_pred, dz_pred = dx_pred[valid], dz_pred[valid]

    # Normalize direction vectors
    true_vecs = np.stack([dx_true, dz_true], axis=1)
    pred_vecs = np.stack([dx_pred, dz_pred], axis=1)

    true_norm = true_vecs / np.linalg.norm(true_vecs, axis=1, keepdims=True)
    pred_norm = pred_vecs / np.linalg.norm(pred_vecs, axis=1, keepdims=True)

    # Compute signed angular error using dot and 2D cross product
    dot = np.einsum('ij,ij->i', true_norm, pred_norm)
    dot = np.clip(dot, -1.0, 1.0)  # Ensure numerical safety
    angle_error = np.degrees(np.arccos(dot))

    # Compute sign using 2D cross product
    cross = true_norm[:, 1] * pred_norm[:, 0] - true_norm[:, 0] * pred_norm[:, 1]
    signed_error = np.where(cross >= 0, angle_error, -angle_error)

    # Plot error vs. true heading
    plt.figure(figsize=(8, 5))
    plt.scatter(heading_true, signed_error, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("True Heading (deg)\n0 = forward (+Z), +90 = right (+X), -90 = left (-X)")
    plt.ylabel("Heading Estimate Error (deg)\n(+ = toward increasing heading)")
    plt.title("Signed Heading Error vs. True Heading")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("heading_error_vs_heading.png", dpi=150)
    plt.show()


def analyze_speed_error_vs_speed(results: dict):
    """
    Analyzes translation speed estimation error vs. true translation speed.

    Args:
        results: Dictionary with keys 'preds' and 'targets',
                 each an [N x 2] numpy array for dx, dz.
    """
    dx_true = results['targets'][:, 0]
    dz_true = results['targets'][:, 1]
    dx_pred = results['preds'][:, 0]
    dz_pred = results['preds'][:, 1]

    # Compute true and predicted translation speeds
    speed_true = np.sqrt(dx_true**2 + dz_true**2)
    speed_pred = np.sqrt(dx_pred**2 + dz_pred**2)

    # Compute signed speed error (predicted - true)
    speed_error = speed_pred - speed_true

    # Plot error vs. true speed
    plt.figure(figsize=(8, 5))
    plt.scatter(speed_true, speed_error, alpha=0.3)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("True Translation Distance per Step (scene units)")
    plt.ylabel("Distance per Step Error (scene units)\n(+ = overestimate, – = underestimate)")
    plt.title("Translation Speed Error vs. True Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("speed_error_vs_speed.png", dpi=150)
    plt.show()


def main():
    # --- Data selection
    data_root = "F:/dw93net/data"
    data_folder = select_subfolder(data_root, header_str="\nSelect a data folder")
    if data_folder is None:
        print("No data folder selected. Exiting.")
        sys.exit()

    # --- Checkpoint selection
    checkpoints_root = "F:/dw93net/checkpoints"
    checkpoints_folder = select_subfolder(checkpoints_root, header_str="\nSelect a checkpoints folder")
    if checkpoints_folder is None:
        print("No checkpoints folder selected. Exiting.")
        sys.exit()
    checkpoint_path = os.path.join(checkpoints_root, checkpoints_folder, "_best.pt")

    # --- Run model on test data
    #from evaluate_flow_xt import run_flow_xt_model_on_test_data  # update this import path as needed
    results = run_flow_xt_model_on_test_data(data_folder, checkpoint_path)

    regress_labels_and_estimates(results)
    analyze_heading_error_vs_heading(results)
    analyze_speed_error_vs_speed(results)

if __name__ == "__main__":
    main()
