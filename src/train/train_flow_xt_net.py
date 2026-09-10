import os
import sys
import torch
from dataclasses import dataclass, asdict, field
from models.pure_translation import PureTranslationNet
from data.load_flow_xt_data import (ReichardtDataset, extract_ranges_from_foldername,
                                   subset_dataset, make_flow_loader, require_pure_translation,
                                   require_pure_translation_root)


@dataclass
class Settings:
    root_folder: str = "data/reichardt/nux=-10000000,10000000_nuz=-10000000,10000000_nrot=0,0"
    checkpoints_dir: str = "checkpoints/reichardt"
    batch_size: int = 12
    num_workers: int = 4
    num_epochs: int = 1_000_000
    lr: float = 1e-3
    train_subset: dict = field(default_factory=lambda: {"n": 80_000, "method": "random"})
    val_subset: dict = field(default_factory=lambda: {"n": 10_000, "method": "first"})
    from_scratch: bool = False

def train(cfg):
    require_pure_translation_root(cfg.root_folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data paths
    train_folder = os.path.join(cfg.root_folder, "train")
    val_folder = os.path.join(cfg.root_folder, "validate")
    data_folder_name = os.path.basename(cfg.root_folder)

    # --- Extract normalization factors from folder name
    ranges = extract_ranges_from_foldername(os.path.basename(cfg.root_folder))
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
    if not len(train_dataset) or not len(val_dataset):
        raise ValueError("Training and validation datasets must be nonempty")
    if not any(value > 0 for value in ranges.values()):
        raise ValueError("Dataset folder name must specify at least one nonzero motion range")

    train_loader = make_flow_loader(train_dataset, cfg.batch_size, device,
                                   shuffle=True, num_workers=cfg.num_workers)
    val_loader = make_flow_loader(val_dataset, cfg.batch_size, device,
                                 num_workers=cfg.num_workers)

    # --- Model and optimizer
    model = PureTranslationNet(train_cache.metadata["dx_values"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    start_epoch = 0
    best_val_rmse = float("inf")

    if os.path.exists(latest_ckpt_path) and not cfg.from_scratch:
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        if checkpoint.get("architecture") != PureTranslationNet.architecture:
            raise ValueError("Checkpoint architecture mismatch; use --from-scratch")
        if checkpoint.get("reichardt") != train_cache.metadata:
            raise ValueError("Checkpoint detector configuration does not match cache")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
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
        print(f"Epoch {epoch+1:,} - Train Loss: {epoch_train_loss:.6f} - Val RMSE: {val_rmse:.6f}")
        sys.stdout.flush()

        # --- Save
        improved = val_rmse < best_val_rmse
        best_val_rmse = min(best_val_rmse, val_rmse)
        checkpoint_data = {
            "architecture": PureTranslationNet.architecture,
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_train_loss,
            "val_rmse": val_rmse,
            "best_val_rmse": best_val_rmse,
            "settings": asdict(cfg),
            "reichardt": train_cache.metadata,
            "n_input_channels": train_cache.n_channels,
        }

        torch.save(checkpoint_data, os.path.join(checkpoint_subdir, f"epoch_{epoch+1:06d}.pt"))
        torch.save(checkpoint_data, latest_ckpt_path)

        if improved:
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"🎯 New best model saved (Val RMSE: {val_rmse:.6f})")

        torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train on precomputed Reichardt maps")
    parser.add_argument("root_folder")
    parser.add_argument("--checkpoints-dir", default="checkpoints/reichardt")
    parser.add_argument("--epochs", type=int, default=Settings.num_epochs)
    parser.add_argument("--batch-size", type=int, default=Settings.batch_size)
    parser.add_argument("--num-workers", type=int, default=Settings.num_workers,
                        help="Background data workers (0 loads in the main process)")
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    train(Settings(root_folder=args.root_folder, checkpoints_dir=args.checkpoints_dir,
                   num_epochs=args.epochs, batch_size=args.batch_size,
                   num_workers=args.num_workers,
                   from_scratch=args.from_scratch))


if __name__ == "__main__":
    main()
