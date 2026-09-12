import os
import sys
import subprocess
from pathlib import Path
import torch
from dataclasses import dataclass, asdict, field

# VS Code's Play button executes this file directly, so Python starts with
# src/train on sys.path rather than the project-wide src package directory.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from models.pure_translation import PureTranslationNet
from stimuli.load_flow_xt_data import (ReichardtDataset, extract_ranges_from_foldername,
                                   subset_dataset, make_flow_loader, require_pure_translation,
                                   require_pure_translation_root, stimulus_set_name,
                                   with_mirrored_reichardt_training)


@dataclass
class Settings:
    root_folder: str = (
        "stimuli/"
        "nspeed=5000000,5000000_nheading=-180000000000,180000000000_nrot=0,0/reichardt"
    )
    checkpoints_dir: str = "checkpoints"
    batch_size: int = 36
    num_workers: int = 8
    num_epochs: int = 1_000_000
    checkpoint_save_interval: int = 25
    learning_rate: float = 1e-3
    learning_rate_factor: float = 0.5
    learning_rate_patience: int = 10
    learning_rate_minimum: float = 1e-6
    mirror_train_stimuli: bool = True
    train_subset: dict = field(default_factory=lambda: {"n": 0, "method": "random"})
    val_subset: dict = field(default_factory=lambda: {"n": 0, "method": "first"})
    from_scratch: bool = False

def train(cfg):
    require_pure_translation_root(cfg.root_folder)
    if cfg.checkpoint_save_interval < 1:
        raise ValueError('checkpoint_save_interval must be at least 1')
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data paths
    train_folder = os.path.join(cfg.root_folder, "train")
    val_folder = os.path.join(cfg.root_folder, "validate")
    data_folder_name = stimulus_set_name(cfg.root_folder)

    # --- Extract normalization factors from folder name
    ranges = extract_ranges_from_foldername(data_folder_name)
    dx_range = ranges['dx']
    dz_range = ranges['dz']

    # A fixed component (e.g. dx=0) still needs supervision. Use the varying
    # component's range as its scale rather than dropping that output's loss.
    translation_scale = max(dx_range, dz_range)
    if translation_scale <= 0:
        raise ValueError("Dataset must specify a nonzero translation range")
    dx_range = dx_range or translation_scale
    dz_range = dz_range or translation_scale

    # --- Checkpoint paths
    checkpoint_subdir = os.path.join(cfg.checkpoints_dir, data_folder_name)
    os.makedirs(checkpoint_subdir, exist_ok=True)
    latest_ckpt_path = os.path.join(checkpoint_subdir, "_latest.pt")
    best_ckpt_path = os.path.join(checkpoint_subdir, "_best.pt")

    # --- Dataset setup
    train_cache = ReichardtDataset(folder=train_folder)
    val_cache = ReichardtDataset(folder=val_folder, expected_metadata=train_cache.metadata)
    require_pure_translation(train_cache)
    require_pure_translation(val_cache)
    train_dataset = subset_dataset(train_cache, cfg.train_subset)
    val_dataset = subset_dataset(val_cache, cfg.val_subset)
    if cfg.mirror_train_stimuli:
        train_dataset = with_mirrored_reichardt_training(
            train_dataset, train_cache.metadata['dx_values'])
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Training and validation datasets must be nonempty")
    if not any(value > 0 for value in ranges.values()):
        raise ValueError("Dataset folder name must specify at least one nonzero motion range")
    if cfg.mirror_train_stimuli:
        print(f'Mirrored training augmentation: {len(train_dataset):,} presentations per epoch')

    train_loader = make_flow_loader(train_dataset, cfg.batch_size, device,
                                   shuffle=True, num_workers=cfg.num_workers)
    val_loader = make_flow_loader(val_dataset, cfg.batch_size, device,
                                 num_workers=cfg.num_workers)

    # --- Model and optimizer
    model = PureTranslationNet(train_cache.metadata["dx_values"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=cfg.learning_rate_factor,
        patience=cfg.learning_rate_patience, min_lr=cfg.learning_rate_minimum)

    start_epoch = 0
    best_val_rmse = float("inf")

    if os.path.exists(latest_ckpt_path) and not cfg.from_scratch:
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        if checkpoint.get("architecture") != PureTranslationNet.architecture:
            raise ValueError("Checkpoint architecture mismatch; start with an empty checkpoint directory")
        if checkpoint.get("reichardt") != train_cache.metadata:
            raise ValueError("Checkpoint detector configuration does not match cache")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print('Checkpoint has no scheduler history; starting a new plateau schedule.')
        start_epoch = checkpoint["epoch"]
        best_val_rmse = checkpoint.get("best_val_rmse", checkpoint.get("val_rmse", float("inf")))
        print(f"🔁 Resumed from epoch {start_epoch} with val RMSE {best_val_rmse:.6f}")
    elif os.path.exists(latest_ckpt_path) and cfg.from_scratch:
        print("⚠️ Found _latest.pt but starting from scratch")

    # --- Training loop
    torch.cuda.empty_cache()
    for epoch in range(start_epoch, cfg.num_epochs):
        model.train()
        train_loss_total = 0.0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            pred = model(x)

            loss = 0.0
            if dx_range > 0:
                loss += torch.nn.functional.mse_loss(pred[:, 0], y[:, 0]) / dx_range**2
            if dz_range > 0:
                loss += torch.nn.functional.mse_loss(pred[:, 1], y[:, 1]) / dz_range**2

            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * x.size(0)

        epoch_train_loss = train_loss_total / len(train_dataset)

        # --- Validation
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                pred = model(x)

                loss = 0.0
                if dx_range > 0:
                    loss += torch.nn.functional.mse_loss(pred[:, 0], y[:, 0]) / dx_range**2
                if dz_range > 0:
                    loss += torch.nn.functional.mse_loss(pred[:, 1], y[:, 1]) / dz_range**2

                val_loss_total += loss.item() * x.size(0)

        epoch_val_loss = val_loss_total / len(val_dataset)
        val_rmse = epoch_val_loss ** 0.5
        previous_learning_rate = optimizer.param_groups[0]['lr']
        scheduler.step(val_rmse)
        next_learning_rate = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:,} - Train Loss: {epoch_train_loss:.6f} - Val RMSE: {val_rmse:.6f} - Learning rate: {previous_learning_rate:.6g}")
        if next_learning_rate < previous_learning_rate:
            print(f'Validation plateau: learning rate reduced to {next_learning_rate:.6g} for the next epoch')
        sys.stdout.flush()

        # --- Save
        improved = val_rmse < best_val_rmse
        best_val_rmse = min(best_val_rmse, val_rmse)
        checkpoint_data = {
            "architecture": PureTranslationNet.architecture,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": epoch_train_loss,
            "val_rmse": val_rmse,
            "best_val_rmse": best_val_rmse,
            "settings": asdict(cfg),
            "reichardt": train_cache.metadata,
            "n_input_channels": train_cache.n_channels,
        }

        if (epoch + 1) % cfg.checkpoint_save_interval == 0:
            torch.save(checkpoint_data, os.path.join(checkpoint_subdir, f"epoch_{epoch+1:06d}.pt"))
        torch.save(checkpoint_data, latest_ckpt_path)

        if improved:
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"🎯 New best model saved (Val RMSE: {val_rmse:.6f})")

        torch.cuda.empty_cache()


def main():
    # CLI parameter overrides are disabled: edit Settings above and commit.
    # parser.add_argument("root_folder")
    # parser.add_argument("--checkpoints-dir", default="checkpoints")
    # parser.add_argument("--epochs", type=int, default=Settings.num_epochs)
    # parser.add_argument("--batch-size", type=int, default=Settings.batch_size)
    # parser.add_argument("--num-workers", type=int, default=Settings.num_workers)
    # parser.add_argument("--from-scratch", action="store_true")
    if len(sys.argv) != 1:
        raise SystemExit('Training takes no arguments. Edit Settings and commit before running.')
    root = Path(__file__).resolve().parents[2]
    commit = committed_revision(root)
    run_id = commit_run_id(root, commit)
    os.chdir(root)
    cfg = Settings()
    cfg.checkpoints_dir = str(Path(cfg.checkpoints_dir) / run_id)
    print(f'Training commit {commit}\nCheckpoints: {cfg.checkpoints_dir}', flush=True)
    train(cfg)


def committed_revision(root):
    """Reject staged, unstaged, and untracked changes before terminal training."""
    status = subprocess.check_output(
        ['git', '-C', str(root), 'status', '--porcelain', '--untracked-files=all'], text=True)
    if status.strip():
        raise SystemExit('Commit your repository changes before training:\n' + status)
    return subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()


def commit_run_id(root, commit):
    """Stable, human-readable checkpoint parent: YYYYMMDD-<full commit ID>."""
    date = subprocess.check_output(
        ['git', '-C', str(root), 'show', '-s', '--format=%cd', '--date=format:%Y%m%d', commit],
        text=True).strip()
    return f'{date}-{commit}'


if __name__ == "__main__":
    main()
