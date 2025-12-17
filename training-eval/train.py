#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler


from utils.static_maps import PitchStaticChannels, PitchDims
from utils.train_utils import log_fixed_val_grid_to_tensorboard, pick_fixed_val_indices
from models.bettermap import BetterSoccerMap2Head
from models.footballmap import PassMap
from models.pitchvision import PitchVisionNet 


# -------------------------
# Repro
# -------------------------

def set_all_seeds(seed: int):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# -------------------------
# Memmap dataset
# -------------------------

class MemmapShard:
    def __init__(self, root_dir: str, x_name: str, t_name: str, n: int, C: int, H: int, W: int):
        self.n = int(n)
        self.C, self.H, self.W = int(C), int(H), int(W)
        x_path = os.path.join(root_dir, x_name)
        t_path = os.path.join(root_dir, t_name)
        self.X = np.memmap(x_path, mode="r", dtype=np.float16, shape=(self.n, self.C, self.H, self.W))
        self.T = np.memmap(t_path, mode="r", dtype=np.float32, shape=(self.n, 3))


class MemmapManifestDataset(Dataset):
    """
    Returns:
      x: (C,H,W) float32
      dst_xy: (2,) long  (x_idx, y_idx)
      y: scalar float32  (0/1)
    """
    def __init__(self, manifest_path: str, root_dir: str, swap_xy: bool = False, cache_size: int = 2):
        self.root_dir = root_dir
        self.swap_xy = bool(swap_xy)
        self.cache_size = int(cache_size)

        with open(manifest_path, "r") as f:
            man = json.load(f)

        if man.get("format") != "memmap_v1":
            raise RuntimeError(f"Expected memmap_v1, got {man.get('format')}")

        self.C = int(man["C"])
        self.H = int(man["H"])
        self.W = int(man["W"])
        self.channels = list(man.get("channels", []))
        self.shards = list(man["shards"])

        self.starts = []
        cur = 0
        for s in self.shards:
            self.starts.append(cur)
            cur += int(s["n"])
        self.total = cur

        self._cache: "OrderedDict[int, MemmapShard]" = OrderedDict()

    def __len__(self):
        return self.total

    def _open_shard(self, shard_id: int) -> MemmapShard:
        shard_id = int(shard_id)
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]

        s = self.shards[shard_id]
        mm = MemmapShard(
            root_dir=self.root_dir,
            x_name=s["x_path"],
            t_name=s["t_path"],
            n=int(s["n"]),
            C=self.C, H=self.H, W=self.W,
        )
        self._cache[shard_id] = mm
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return mm

    def _locate(self, idx: int) -> Tuple[int, int]:
        lo, hi = 0, len(self.starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.starts[mid]
            end = self.starts[mid + 1] if mid + 1 < len(self.starts) else self.total
            if start <= idx < end:
                return mid, idx - start
            if idx < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise IndexError(idx)

    def __getitem__(self, idx: int):
        sid, j = self._locate(int(idx))
        shard = self._open_shard(sid)

        x_np = np.array(shard.X[j], copy=True)          # (C,H,W) float16
        t_np = np.array(shard.T[j], copy=True)          # (3,) float32

        x = torch.from_numpy(x_np).to(torch.float32)    # (C,H,W)
        dst_xy = torch.from_numpy(t_np[:2]).to(torch.long)  # (2,)
        if self.swap_xy:
            dst_xy = dst_xy[[1, 0]]
        y = torch.tensor(float(t_np[2]), dtype=torch.float32)  # scalar

        return x, dst_xy, y


def collate_batch(batch):
    xs, xys, ys = zip(*batch)
    X = torch.stack(xs, dim=0)                   # (B,C,H,W)
    dst_xy = torch.stack(xys, dim=0).long()      # (B,2)
    y = torch.stack(ys, dim=0).float().view(-1)  # (B,)

    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, dst_xy, y


"""# -------------------------
# Loss (dest CE + succ BCE@dest)
# -------------------------

@dataclass
class TwoHeadLossCfg:
    w_dest: float = 1.0
    w_succ: float = 1.0


def twohead_loss(out: Dict[str, torch.Tensor], y: torch.Tensor, dst_xy: torch.Tensor, cfg: TwoHeadLossCfg):
    dest_logits = out["dest_logits"]   # (B,1,H,W)
    succ_logits = out["succ_logits"]   # (B,1,H,W)

    if dest_logits.shape != succ_logits.shape:
        raise RuntimeError(f"dest/succ logits mismatch: {tuple(dest_logits.shape)} vs {tuple(succ_logits.shape)}")

    B, _, H, W = dest_logits.shape
    device = dest_logits.device

    y = y.to(device=device, dtype=dest_logits.dtype).view(B).clamp(0.0, 1.0)
    x_idx = dst_xy[:, 0].to(device=device).long().clamp(0, H - 1)
    y_idx = dst_xy[:, 1].to(device=device).long().clamp(0, W - 1)

    logs: Dict[str, float] = {}
    total = dest_logits.new_tensor(0.0)

    if cfg.w_dest > 0:
        dest_flat = dest_logits.view(B, -1)                 # (B, H*W)
        dest_index = (x_idx * W + y_idx).long()             # (B,)
        l_dest = F.cross_entropy(dest_flat, dest_index, reduction="mean")

        l_dest_w = cfg.w_dest * l_dest
        total = total + l_dest_w

        logs["loss_dest"] = float(l_dest_w.detach().cpu())

    if cfg.w_succ > 0:
        succ_at = succ_logits[torch.arange(B, device=device), 0, x_idx, y_idx]
        l_succ = F.binary_cross_entropy_with_logits(succ_at, y, reduction="mean")

        l_succ_w = cfg.w_succ * l_succ
        total = total + l_succ_w

        logs["loss_succ"] = float(l_succ_w.detach().cpu())


    logs["loss_total"] = float(total.detach().cpu())
    return total, logs"""

@dataclass
class TwoHeadLossCfg:
    w_dest: float = 1.0
    w_succ: float = 1.0
    dest_sigma: float = 2.5   # meters/cells
    dest_clip: float = 50.0   # numeric stability on logits (optional)


def _dense_dest_target(
    x_idx: torch.Tensor,  # (B,) long
    y_idx: torch.Tensor,  # (B,) long
    H: int,
    W: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # grid (H,W)
    xs = torch.arange(H, device=device, dtype=dtype).view(H, 1).expand(H, W)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, W).expand(H, W)

    # (B,H,W)
    dx = xs.unsqueeze(0) - x_idx.to(device=device, dtype=dtype).view(-1, 1, 1)
    dy = ys.unsqueeze(0) - y_idx.to(device=device, dtype=dtype).view(-1, 1, 1)

    inv2s2 = 1.0 / (2.0 * (sigma * sigma))
    logp = -(dx * dx + dy * dy) * inv2s2

    # normalize per-example to a prob distribution over HW
    p = torch.softmax(logp.view(-1, H * W), dim=1)
    return p



def twohead_loss(out: Dict[str, torch.Tensor], y: torch.Tensor, dst_xy: torch.Tensor, cfg: TwoHeadLossCfg):
    dest_logits = out["dest_logits"]   # (B,1,H,W)
    succ_logits = out["succ_logits"]   # (B,1,H,W)

    if dest_logits.shape != succ_logits.shape:
        raise RuntimeError(f"dest/succ logits mismatch: {tuple(dest_logits.shape)} vs {tuple(succ_logits.shape)}")

    B, _, H, W = dest_logits.shape
    device = dest_logits.device
    dtype = dest_logits.dtype

    y = y.to(device=device, dtype=dtype).view(B).clamp(0.0, 1.0)
    x_idx = dst_xy[:, 0].to(device=device).long().clamp(0, H - 1)
    y_idx = dst_xy[:, 1].to(device=device).long().clamp(0, W - 1)

    logs: Dict[str, float] = {}
    total = dest_logits.new_tensor(0.0)

    # ---- dense destination loss (KL to Gaussian target) ----
    if cfg.w_dest > 0:
        d = dest_logits[:, 0]  # (B,H,W)
        if cfg.dest_clip and cfg.dest_clip > 0:
            d = d.clamp(min=-cfg.dest_clip, max=cfg.dest_clip)

        log_probs = F.log_softmax(d.view(B, -1), dim=1)  # (B,HW)
        target = _dense_dest_target(
            x_idx=x_idx, y_idx=y_idx,
            H=H, W=W,
            sigma=float(cfg.dest_sigma),
            device=device, dtype=log_probs.dtype,
        )  # (B,HW)

        # KL(target || probs) implemented as KLDiv(log_probs, target)
        l_dest = F.kl_div(log_probs, target, reduction="batchmean")

        l_dest_w = cfg.w_dest * l_dest
        total = total + l_dest_w
        logs["loss_dest"] = float(l_dest_w.detach().cpu())

    # ---- success-at-destination loss (localized BCE) ----
    if cfg.w_succ > 0:
        succ_at = succ_logits[torch.arange(B, device=device), 0, x_idx, y_idx]
        l_succ = F.binary_cross_entropy_with_logits(succ_at, y, reduction="mean")

        l_succ_w = cfg.w_succ * l_succ
        total = total + l_succ_w
        logs["loss_succ"] = float(l_succ_w.detach().cpu())

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs

# -------------------------
# Model hook (you fill this in)
# -------------------------

def build_model(in_channels: int,
                arch: str) -> nn.Module:
    """
    Must return dict with:
      {"dest_logits": (B,1,H,W), "succ_logits": (B,1,H,W)}
    """
    if arch == 'passmap':
        model = PassMap(in_channels= in_channels, base=64, blocks_per_stage= 4)
    
    elif arch == 'pitchvision':
        model = PitchVisionNet(in_channels=in_channels,base=64, blocks_per_stage = 3)
    
    return model

# -------------------------
# Train / Eval loop
# -------------------------

def run_epoch(
    model: nn.Module,
    dl: DataLoader,
    device: torch.device,
    static: PitchStaticChannels,
    loss_cfg: TwoHeadLossCfg,
    opt: torch.optim.Optimizer | None,
    grad_clip: float,
    writer: SummaryWriter,
    global_step: int,
    epoch: int,
    split: str,
    scaler: GradScaler
):
    train_mode = opt is not None
    model.train(train_mode)

    running_total = 0.0
    running_dest = 0.0
    running_succ = 0.0
    n_seen = 0

    it = tqdm(dl, desc=f"{split}", leave=False)
    t0 = time.time()

    for xb, dst_xy, y in it:
        xb = xb.to(device, non_blocking=True)
        dst_xy = dst_xy.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        xb = static.concat_to(xb, dim=1)  # (B, C_dyn + C_static, H, W)

        if train_mode:
            opt.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train_mode):
            with autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type=="cuda")):
                out = model(xb)
                loss, logs = twohead_loss(out, y, dst_xy, loss_cfg)

            if train_mode:
                scaler.scale(loss).backward()
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()


        b = xb.size(0)
        n_seen += b
        running_total += logs["loss_total"] * b
        running_dest += logs.get("loss_dest", 0.0) * b
        running_succ += logs.get("loss_succ", 0.0) * b

        it.set_postfix({
            "loss": f"{running_total/max(1,n_seen):.4f}",
            "dest": f"{running_dest/max(1,n_seen):.4f}",
            "succ": f"{running_succ/max(1,n_seen):.4f}",
        })

        if train_mode:
            writer.add_scalar(f"{split}/loss_total_step", logs["loss_total"], global_step)
            if "loss_dest" in logs: writer.add_scalar(f"{split}/loss_dest_step", logs["loss_dest"], global_step)
            if "loss_succ" in logs: writer.add_scalar(f"{split}/loss_succ_step", logs["loss_succ"], global_step)
            global_step += 1

    dt = time.time() - t0
    avg_total = running_total / max(1, n_seen)
    avg_dest = running_dest / max(1, n_seen)
    avg_succ = running_succ / max(1, n_seen)

    writer.add_scalar(f"{split}/loss_total", avg_total, epoch)
    writer.add_scalar(f"{split}/loss_dest", avg_dest, epoch)
    writer.add_scalar(f"{split}/loss_succ", avg_succ, epoch)
    writer.add_scalar(f"{split}/epoch_time_s", dt, epoch)

    return avg_total, avg_dest, avg_succ, global_step


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--cache_size", type=int, default=2)

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--num_workers", type=int, default=16)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--swap_xy", action="store_true")

    p.add_argument("--w_dest", type=float, default=1.0)
    p.add_argument("--w_succ", type=float, default=1.0)

    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--save_path", type=str, default="best_ckpt.pt")

    p.add_argument("--arch", type=str, default="pitchvision", choices=["passmap", "pitchvision"],required= True)


    # Scheduler (ReduceLROnPlateau)
    p.add_argument("--sched", type=str, default="plateau", choices=["none", "plateau"])
    p.add_argument("--plateau_patience", type=int, default=2)
    p.add_argument("--plateau_factor", type=float, default=0.5)
    p.add_argument("--plateau_min_lr", type=float, default=1e-6)
    p.add_argument("--plateau_threshold", type=float, default=1e-4)

    p.add_argument("--viz_n", type=int, default=5)
    p.add_argument("--viz_seed", type=int, default=42)
    p.add_argument("--viz_every", type=int, default=1)  # log every N epochs


    args = p.parse_args()

    set_all_seeds(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"Device: {device}")

    train_root = os.path.join(args.data_root, "train")
    val_root = os.path.join(args.data_root, "val")

    train_ds = MemmapManifestDataset(
        manifest_path=os.path.join(train_root, "manifest.json"),
        root_dir=train_root,
        swap_xy=args.swap_xy,
        cache_size=args.cache_size,
    )
    val_ds = MemmapManifestDataset(
        manifest_path=os.path.join(val_root, "manifest.json"),
        root_dir=val_root,
        swap_xy=args.swap_xy,
        cache_size=args.cache_size,
    )

    fixed_viz_idxs = pick_fixed_val_indices(val_ds, n=args.viz_n, seed=args.viz_seed)
    print(f"[viz] fixed val indices: {fixed_viz_idxs}")


    print(f"[data] train={len(train_ds)} val={len(val_ds)}")
    print(f"[data] C,H,W={train_ds.C},{train_ds.H},{train_ds.W}")

    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=4,
        collate_fn=collate_batch,
        drop_last=True,
    )

    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
        collate_fn=collate_batch,
        drop_last=False,
    )

    static = PitchStaticChannels(dims=PitchDims(H=train_ds.H, W=train_ds.W))
    static = static.to(device=device)

    C_static = int(static.forward().shape[0])
    in_channels = int(train_ds.C + C_static)
    print(f"[static] C_static={C_static} -> model in_channels={in_channels}")


    model = build_model(in_channels=in_channels, arch= args.arch).to(device).float()
    arch_name = model.__class__.__name__

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.runs_dir) / f"{stamp}_{arch_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(run_dir))
    writer.add_text("hparams",
                    json.dumps(vars(args), indent=2),
                    global_step=0)
    writer.add_text("viz/fixed_indices", str(fixed_viz_idxs), 0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    writer.add_scalar("opt/lr", args.lr, 0)

    scheduler = None
    if args.sched == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            threshold=args.plateau_threshold,
            min_lr=args.plateau_min_lr,
        )

    scaler = GradScaler(enabled=(device.type == "cuda"))


    loss_cfg = TwoHeadLossCfg(w_dest=args.w_dest, w_succ=args.w_succ)

    best_val = float("inf")
    history = {
        "train_total": [], "train_dest": [], "train_succ": [],
        "val_total": [], "val_dest": [], "val_succ": [],
    }
    global_step = 0

    for epoch in range(args.epochs):
        t0 = time.time()

        train_total, train_dest, train_succ, global_step = run_epoch(
            model=model, dl=train_dl, device=device,
            static=static, loss_cfg=loss_cfg,
            opt=opt, grad_clip=args.grad_clip,
            writer=writer, global_step=global_step,
            epoch=epoch, split="train", scaler= scaler
        )
        val_total, val_dest, val_succ, _ = run_epoch(
            model=model, dl=val_dl, device=device,
            static=static, loss_cfg=loss_cfg,
            opt=None, grad_clip=0.0,
            writer=writer, global_step=global_step,
            epoch=epoch, split="val", scaler= scaler
        )

        if (epoch % args.viz_every) == 0:
            try:
                log_fixed_val_grid_to_tensorboard(
                    model=model,
                    val_ds=val_ds,
                    static=static,
                    device=device,
                    writer=writer,
                    epoch=epoch,
                    fixed_idxs=fixed_viz_idxs,
                    tag="viz/fixed_val_grid",
                    coords_are_centers=False,
                    ch_dist2ball=0,
                    ch_in_pos=3,
                    ch_out_pos=4,
                )
            except Exception as e:
                print(f"[warn] fixed viz logging failed: {e}")


        if scheduler is not None:
            scheduler.step(val_total)

        # log current LR (after scheduler step)
        cur_lr = opt.param_groups[0]["lr"]
        writer.add_scalar("opt/lr", cur_lr, epoch)


        dt = time.time() - t0
        print(
            f"Epoch {epoch:03d} | "
            f"train={train_total:.6f} (dest={train_dest:.6f}, succ={train_succ:.6f}) | "
            f"val={val_total:.6f} (dest={val_dest:.6f}, succ={val_succ:.6f}) | "
            f"time={dt:.1f}s"
        )

        history["train_total"].append(train_total)
        history["train_dest"].append(train_dest)
        history["train_succ"].append(train_succ)
        history["val_total"].append(val_total)
        history["val_dest"].append(val_dest)
        history["val_succ"].append(val_succ)

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        if val_total < best_val:
            best_val = val_total
            ckpt = {
                "epoch": epoch,
                "best_val": best_val,
                "model_state": model.state_dict(),
                "opt_state": opt.state_dict(),
                "args": vars(args),
                "C_dyn": train_ds.C,
                "C_static": C_static,
            }
            torch.save(ckpt, run_dir / args.save_path)
            writer.add_text("status", f"saved best ckpt at epoch {epoch} val_total={best_val:.6f}", epoch)

    # Plot losses
    fig = plt.figure(figsize=(10, 6))
    plt.plot(history["train_total"], label="train_total")
    plt.plot(history["val_total"], label="val_total")
    plt.plot(history["train_dest"], label="train_dest", linestyle="--")
    plt.plot(history["val_dest"], label="val_dest", linestyle="--")
    plt.plot(history["train_succ"], label="train_succ", linestyle=":")
    plt.plot(history["val_succ"], label="val_succ", linestyle=":")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    fig.savefig(run_dir / "loss_curves.png", dpi=200)
    plt.close(fig)

    writer.close()
    print(f"[done] run_dir={run_dir}")
    print(f"[done] best_val={best_val:.6f}")


if __name__ == "__main__":
    main()
