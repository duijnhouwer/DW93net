import os
import argparse
import torch
from torch.utils.data import DataLoader
from models.flow_xt_net import FlowXTNet
from datasets.flow_xt_dataset import FlowXtDataset

# --- CLI args ---
parser = argparse.ArgumentParser()
parser.add_argument("--fromscratch", action="store_true", help="Ignore checkpoint and start training from scratch.")
parser.add_argument("--test", metavar="CHECKPOINT", type=str,
    help="Evaluate model only. Options: 'best', 'latest', or an epoch number like '42'")
args = parser.parse_args()

# --- Config ---
root_folder = "F:/dw93net/data/udx=0,0_udz=0,0_rot=-0.5,0.5"
train_folder = os.path.join(root_folder, "train")
val_folder = os.path.join(root_folder, "validate")
test_folder = os.path.join(root_folder, "test")
checkpoints_dir = "F:/dw93net/checkpoints"
model_name = "flow_xt_net"
batch_size = 64
num_epochs = 1_000_000
lr = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Prepare checkpoint folder ---
data_folder_name = os.path.basename(root_folder)
checkpoint_subdir = os.path.join(checkpoints_dir, model_name, data_folder_name)
os.makedirs(checkpoint_subdir, exist_ok=True)

# Checkpoint file paths
latest_ckpt_path = os.path.join(checkpoint_subdir, "_latest.pt")
best_ckpt_path = os.path.join(checkpoint_subdir, "_best.pt")

# --- TEST MODE ONLY ---
if args.test:
    test_dataset = FlowXtDataset(folder=test_folder)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    if args.test == "best":
        ckpt_path = best_ckpt_path
    elif args.test == "latest":
        ckpt_path = latest_ckpt_path
    elif args.test.isdigit():
        ckpt_path = os.path.join(checkpoint_subdir, f"epoch_{int(args.test):06d}.pt")
    else:
        raise ValueError(f"Invalid value for --test: {args.test}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    # Load model and evaluate
    print(f"📦 Loading checkpoint: {os.path.basename(ckpt_path)}")
    model = FlowXTNet().to(device)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    total_loss = 0.0
    loss_fn = torch.nn.MSELoss()
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            total_loss += loss.item() * x.size(0)

    avg_loss = total_loss / len(test_dataset)
    rmse = avg_loss ** 0.5
    print(f"✅ Test RMSE: {rmse:.6f}")
    exit(0)

# --- Training setup ---
train_dataset = FlowXtDataset(folder=train_folder)
val_dataset = FlowXtDataset(folder=val_folder)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = FlowXTNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.MSELoss()

start_epoch = 0
best_val_loss = float("inf")

if os.path.exists(latest_ckpt_path) and not args.fromscratch:
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('val_rmse', float('inf'))
    print(f"✅ Resumed from _latest.pt (epoch {start_epoch}, val RMSE: {best_val_loss:.6f})")
elif os.path.exists(latest_ckpt_path) and args.fromscratch:
    print(f"⚠️ Found _latest.pt but starting from scratch (--fromscratch).")

# --- Training loop ---
for epoch in range(start_epoch, num_epochs):
    model.train()
    train_loss_total = 0.0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        train_loss_total += loss.item() * x.size(0)

    epoch_train_loss = train_loss_total / len(train_dataset)

    # --- Validation ---
    model.eval()
    val_loss_total = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = loss_fn(pred, y)
            val_loss_total += loss.item() * x.size(0)

    epoch_val_loss = val_loss_total / len(val_dataset)
    val_rmse = epoch_val_loss ** 0.5

    print(f"Epoch {epoch+1:,} - Train Loss: {epoch_train_loss:.6f} - Val RMSE: {val_rmse:.6f}")

    # Save per-epoch
    epoch_ckpt_path = os.path.join(checkpoint_subdir, f"epoch_{epoch+1:06d}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_train_loss,
        "val_rmse": val_rmse,
    }, epoch_ckpt_path)

    # Save _latest.pt
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": epoch_train_loss,
        "val_rmse": val_rmse,
    }, latest_ckpt_path)

    # Save _best.pt
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": epoch_train_loss,
            "val_rmse": val_rmse,
        }, best_ckpt_path)
        print(f"🎯 New best model saved as _best.pt (Val RMSE: {val_rmse:.6f})")
