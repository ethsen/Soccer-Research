import os
import math
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.soccermap import soccermap_model

# ----------------------------
# Small in-RAM dataset for one shard
# ----------------------------
class OneShardDataset(Dataset):
    """
    Holds one shard fully in memory.
    features: (C, N, W, H)   e.g., (8, 20000, 105, 68)
    targets:  (N, 3)         [x, y, outcome]
    Returns a single sample as (C, H, W), [x, y], outcome
    """
    def __init__(self, features_cxnwh: torch.Tensor, targets_n3: torch.Tensor):
        assert features_cxnwh.dim() == 4, "features must be C×N×W×H"
        assert targets_n3.dim() == 2 and targets_n3.shape[1] == 3, "targets must be N×3"
        self.x = features_cxnwh     # C×N×W×H
        self.t = targets_n3         # N×3
        self.N = self.x.shape[1]

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        # C×N×W×H -> C×W×H for sample i
        xi = self.x[:, i, :, :]          # (C, W, H)
        # If your saved order were (C,H,W), this still works; we standardize to (C,H,W) below in collate
        ti = self.t[i]                   # (3,)
        dst_xy = ti[:2]                  # [x, y]
        y = ti[2]                        # outcome
        return xi, dst_xy, y


def set_seed(seed=1337):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loaders_for_shard(features_path, targets_path, batch_size=16, num_workers=4):
    # load to CPU
    feats_obj = torch.load(features_path, map_location="cpu")
    feats = feats_obj["X"] if isinstance(feats_obj, dict) else feats_obj  # C×N×H×W (preferred)

    tobj = torch.load(targets_path, map_location="cpu")
    targs = tobj["targets"] if isinstance(tobj, dict) else tobj
    assert feats.dim() == 4, f"Expected 4D features, got {feats.shape}"

    # Enforce (C, N, H, W)
    C, N, A, B = feats.shape
    if (A, B) == (105, 68):
        # already (C,N,H,W)
        pass
    elif (A, B) == (68, 105):
        feats = feats.permute(0, 1, 3, 2).contiguous()
    else:
        raise ValueError(f"Unexpected spatial size {A}x{B}; expected 105x68 (or 68x105).")

    assert feats.shape[0] == 14, f"Expected 14 channels, got {feats.shape[0]}"

    if targs.dim() == 2 and targs.shape[0] == 3 and targs.shape[1] != 3:
        targs = targs.t().contiguous()  # normalize to N×3

    dataset = OneShardDataset(feats, targs)

    # 80/20 split (shuffle indices once per shard)
    idx = torch.randperm(len(dataset))
    split = int(0.8 * len(dataset))
    train_idx, val_idx = idx[:split], idx[split:]

    train_ds = Subset(dataset, train_idx)
    val_ds   = Subset(dataset, val_idx)

    def collate(batch):
        xs, dsts, ys = zip(*batch)

        X = torch.stack(xs, dim=0).to(torch.float32)          # (B,C,H,W)
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        dst_xy = torch.stack(dsts, dim=0).to(torch.float32)   # (B,2)
        dst_xy = torch.nan_to_num(dst_xy, nan=0.0, posinf=0.0, neginf=0.0)

        y = torch.as_tensor(ys, dtype=torch.float32)          # (B,)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return X, dst_xy, y

    
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate
    )
    return train_loader, val_loader, len(train_ds), len(val_ds)


def evaluate(model, loader, device):
    model.eval()
    total_loss, total = 0.0, 0
    with torch.no_grad():
        for X, dst_xy, y in loader:
            X = X.to(device)
            dst_xy = dst_xy.to(device)
            y = y.to(device)
            _, fused_logits, _ = model(X)
            loss = model.target_location_loss(fused_logits, y, dst_xy)
            total_loss += loss.item() * X.size(0)
            total += X.size(0)
    return total_loss / max(total, 1)


def train_sequential_shards(
    feature_dir="data/soccer_shards",
    target_dir="data/soccer_shards_targets",
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-5,
    num_workers=4,
    total_epochs=30,
    grad_clip=1.0,
    seed=1337,
    save_every_epoch=True,
    ckpt_path="soccermap_checkpoint.pt",
    plot_path="training_losses.png",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Discover shards ---
    feat_paths = sorted([os.path.join(feature_dir, f) for f in os.listdir(feature_dir) if f.endswith(".pt")])
    targ_paths = sorted([os.path.join(target_dir,  f) for f in os.listdir(target_dir)  if f.endswith(".pt")])
    assert len(feat_paths) == len(targ_paths) and len(feat_paths) > 0, "Shard mismatch / not found."
    n_shards = len(feat_paths)

    # --- Model + optimizer ---
    model = soccermap_model(in_channels=14, base=32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler(enabled=False)

    # --- Compute total batches per epoch (for tqdm) ---
    all_train_lens = []
    for fpath, tpath in zip(feat_paths, targ_paths):
        _, _, ntr, _ = make_loaders_for_shard(fpath, tpath, batch_size=batch_size, num_workers=0)
        all_train_lens.append(ntr)
    total_train_samples = sum(all_train_lens)
    total_batches_per_epoch = math.ceil(total_train_samples / batch_size)

    # --- Tracking lists ---
    epoch_train_losses = []
    epoch_val_losses = []

    # ---- Main training loop ----
    for epoch in range(1, total_epochs + 1):
        model.train()
        running_loss, seen = 0.0, 0
        val_loss_epoch = 0.0
        val_seen = 0

        with tqdm(total=total_batches_per_epoch, desc=f"Epoch {epoch}/{total_epochs}", leave=True) as pbar:
            for s, (fpath, tpath) in enumerate(zip(feat_paths, targ_paths), start=1):
                train_loader, val_loader, ntr, nva = make_loaders_for_shard(
                    fpath, tpath, batch_size=batch_size, num_workers=num_workers
                )

                # --- train on this shard ---
                for X, dst_xy, y in train_loader:
                    optimizer.zero_grad(set_to_none=True)
                    X = X.to(device, dtype=torch.float32)
                    dst_xy = dst_xy.to(device, dtype=torch.float32)
                    y = y.to(device, dtype=torch.float32)

                    #with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type=="cuda")):
                    _, fused_logits, _ = model(X)
                    loss = model.target_location_loss(fused_logits, y, dst_xy)
                    if not torch.isfinite(loss):
                        print("Loss is NaN/Inf.")
                        print("X finite:", torch.isfinite(X).all().item())
                        print("X min/max:", X.min().item(), X.max().item())
                        raise RuntimeError("Non-finite loss")



                    scaler.scale(loss).backward()
                    if grad_clip is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    bs = X.size(0)
                    running_loss += loss.item() * bs
                    seen += bs
                    train_running_avg = running_loss / max(seen, 1)
                    pbar.set_postfix(train_loss=f"{train_running_avg:.4f}")
                    pbar.update(1)

                # --- evaluate shard ---
                val_loss = evaluate(model, val_loader, device)
                val_loss_epoch += val_loss * nva
                val_seen += nva
                pbar.set_postfix(train_loss=f"{train_running_avg:.4f}", val_loss=f"{val_loss:.4f}")

        # ---- end of epoch ----
        final_train_loss = running_loss / max(seen, 1)
        final_val_loss = val_loss_epoch / max(val_seen, 1)
        epoch_train_losses.append(final_train_loss)
        epoch_val_losses.append(final_val_loss)

        print(f"Epoch {epoch}/{total_epochs} | Train: {final_train_loss:.6f} | Val: {final_val_loss:.6f}")

        if save_every_epoch:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": final_train_loss,
                    "val_loss": final_val_loss,
                },
                ckpt_path,
            )

    print("\nTraining complete")

    # --- Plot losses ---
    epochs = range(1, total_epochs + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_train_losses, label="Train Loss", linewidth=2)
    plt.plot(epochs, epoch_val_losses, label="Validation Loss", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved training loss plot → {plot_path}")

    return model, epoch_train_losses, epoch_val_losses


if __name__ == "__main__":
    # Adjust epochs_per_shard if you want more than one pass per shard
    train_sequential_shards(
    feature_dir="data/soccer_shards",
    target_dir="data/soccer_shards_targets",
    batch_size=16,
    lr=1e-3,
    weight_decay=1e-5,
    num_workers=4,
    total_epochs=30,
    grad_clip=1.0,
    seed=1337,
    save_every_epoch=True,
    ckpt_path="soccermap_checkpoint_v2.pt",
    )

    
