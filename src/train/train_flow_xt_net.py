import os
import sys
import torch
from dataclasses import dataclass, asdict, field
from torch.utils.data import DataLoader
from models.flow_xt_net import FlowXTNet
from data.load_flow_xt_data import FlowXtDataset, extract_range_from_foldername, subset_dataset


@dataclass
class Settings:
    root_folder: str = "F:/dw93net/data/udx=-10000,10000_udz=-10000,10000_rot=0,0"
    checkpoints_dir: str = "F:/dw93net/checkpoints"
    model_name: str = "flow_xt_net"
    batch_size: int = 12
    num_epochs: int = 1_000_000
    lr: float = 1e-3
    train_subset: dict = field(default_factory=lambda: {"n": 80_000, "method": "random"})
    val_subset: dict = field(default_factory=lambda: {"n": 10_000, "method": "first"})
    from_scratch: bool = False

cfg = Settings()
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Data paths
train_folder = os.path.join(cfg.root_folder, "train")
val_folder = os.path.join(cfg.root_folder, "validate")
data_folder_name = os.path.basename(cfg.root_folder)

# --- Extract normalization factors from folder name
ranges = extract_range_from_foldername(os.path.basename(cfg.root_folder))
dx_range = ranges['dx']
dz_range = ranges['dz']
rot_range = ranges['rot_deg']

# --- Checkpoint paths
checkpoint_subdir = os.path.join(cfg.checkpoints_dir, cfg.model_name, data_folder_name)
os.makedirs(checkpoint_subdir, exist_ok=True)
latest_ckpt_path = os.path.join(checkpoint_subdir, "_latest.pt")
best_ckpt_path = os.path.join(checkpoint_subdir, "_best.pt")

# --- Dataset setup
train_dataset = subset_dataset(FlowXtDataset(folder=train_folder), cfg.train_subset)
val_dataset = subset_dataset(FlowXtDataset(folder=val_folder), cfg.val_subset)

train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

# --- Model and optimizer
model = FlowXTNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

start_epoch = 0
best_val_loss = float("inf")

if os.path.exists(latest_ckpt_path) and not cfg.from_scratch:
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint.get("val_rmse", float("inf"))
    print(f"🔁 Resumed from epoch {start_epoch} with val RMSE {best_val_loss:.6f}")
elif os.path.exists(latest_ckpt_path) and cfg.from_scratch:
    print("⚠️ Found _latest.pt but starting from scratch")

# --- Training loop
torch.cuda.empty_cache()
for epoch in range(start_epoch, cfg.num_epochs):
    model.train()
    train_loss_total = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)

        loss = 0.0
        if dx_range > 0:
            loss += torch.nn.functional.mse_loss(pred[:, 0], y[:, 0]) / dx_range**2
        if dz_range > 0:
            loss += torch.nn.functional.mse_loss(pred[:, 1], y[:, 1]) / dz_range**2
        if rot_range > 0:
            loss += torch.nn.functional.mse_loss(pred[:, 2], y[:, 2]) / rot_range**2

        loss.backward()
        optimizer.step()
        train_loss_total += loss.item() * x.size(0)

    epoch_train_loss = train_loss_total / len(train_dataset)

    # --- Validation
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            loss = 0.0
            if dx_range > 0:
                loss += torch.nn.functional.mse_loss(pred[:, 0], y[:, 0]) / dx_range**2
            if dz_range > 0:
                loss += torch.nn.functional.mse_loss(pred[:, 1], y[:, 1]) / dz_range**2
            if rot_range > 0:
                loss += torch.nn.functional.mse_loss(pred[:, 2], y[:, 2]) / rot_range**2

            val_loss_total += loss.item() * x.size(0)

    epoch_val_loss = val_loss_total / len(val_dataset)
    val_rmse = epoch_val_loss ** 0.5
    print(f"Epoch {epoch+1:,} - Train Loss: {epoch_train_loss:.6f} - Val RMSE: {val_rmse:.6f}")
    sys.stdout.flush()

    # --- Save
    checkpoint_data = {
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_train_loss,
        "val_rmse": val_rmse,
        "settings": asdict(cfg),
    }

    torch.save(checkpoint_data, os.path.join(checkpoint_subdir, f"epoch_{epoch+1:06d}.pt"))
    torch.save(checkpoint_data, latest_ckpt_path)

    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(checkpoint_data, best_ckpt_path)
        print(f"🎯 New best model saved (Val RMSE: {val_rmse:.6f})")

    torch.cuda.empty_cache()
