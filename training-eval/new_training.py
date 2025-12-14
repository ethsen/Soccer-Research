#!/usr/bin/env python3
"""
Minimal two-head trainer for pass destination + completion.

Model:
  BetterSoccerMap2Head outputs:
    dest_logits: (B,1,H,W)  -> softmax over HW for p(dest|state)
    succ_logits: (B,1,H,W)  -> sigmoid per-cell for p(success | state, dest=cell)
See models/passmap.py :contentReference[oaicite:3]{index=3}

Data (torch-converter_clean.py):
  Feature shard: {"X": (C,N,H,W), "channels":[...], "H":H, "W":W, ...}
  Target shard : {"targets": (N,3), "schema":["end_x_m","end_y_m","success"], ...}
Targets are end_x_m/end_y_m in meters on a 1m grid (H=105, W=68), and success in {0,1}.
:contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

Key requirement:
  Fixed shard-level train/val split across all epochs (saved in split.json).
This matches the intent of your current shard split logic. :contentReference[oaicite:6]{index=6}
"""

from __future__ import annotations
from datetime import datetime

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


import os
import json
import hashlib

from collections import OrderedDict
from torch.utils.data import Dataset


import numpy as np
from typing import Any, Optional

class MemmapShard:
    """
    Lightweight wrapper holding one shard's memmaps:
      X: (n,C,H,W) float16
      T: (n,3) float32
    """
    def __init__(self, root_dir: str, x_name: str, t_name: str, n: int, C: int, H: int, W: int):
        self.root_dir = root_dir
        self.n = int(n)
        self.C, self.H, self.W = int(C), int(H), int(W)

        x_path = os.path.join(root_dir, x_name)
        t_path = os.path.join(root_dir, t_name)

        self.X = np.memmap(x_path, mode="r", dtype=np.float16, shape=(self.n, self.C, self.H, self.W))
        self.T = np.memmap(t_path, mode="r", dtype=np.float32, shape=(self.n, 3))


class MemmapManifestDataset(Dataset):
    """
    Dataset over memmap shards produced by torch_converter_stratified_memmap.py.
    Uses small LRU cache of open memmaps to avoid too many open files.

    Returns:
      x: (C,H,W) float32
      xy: (2,) long
      y: scalar float32
    """
    def __init__(
        self,
        manifest_path: str,    # e.g. <data_root>/train/manifest.json
        root_dir: str,         # e.g. <data_root>/train
        swap_xy: bool = False,
        cache_size: int = 2,
    ):
        self.root_dir = root_dir
        self.swap_xy = swap_xy
        self.cache_size = int(cache_size)

        with open(manifest_path, "r") as f:
            man = json.load(f)

        assert man.get("format") == "memmap_v1", f"Expected memmap_v1 manifest, got {man.get('format')}"
        self.C = int(man["C"])
        self.H = int(man["H"])
        self.W = int(man["W"])
        self.channels = man.get("channels", [])
        self.shards = man["shards"]

        # prefix sums to map global index -> (shard_id, local_idx)
        self.starts = []
        cur = 0
        for s in self.shards:
            self.starts.append(cur)
            cur += int(s["n"])
        self.total = cur

        # LRU cache: shard_id -> MemmapShard
        self._cache = OrderedDict()

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

    def _locate(self, idx: int):
        # binary search in starts
        # starts[k] <= idx < starts[k+1]
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

        # safe copy from read-only memmap -> writable numpy array
        x_np = np.array(shard.X[j], copy=True)  # (C,H,W) float16
        x = torch.from_numpy(x_np).to(torch.float32)

        t = shard.T[j]  # (3,) float32
        xy = torch.tensor(t[:2], dtype=torch.long)
        if self.swap_xy:
            xy = xy[[1, 0]]
        y = torch.tensor(t[2], dtype=torch.float32)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
        return x, xy, y



# -------------------------
# Utilities
# -------------------------
def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def save_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def resolve_manifest_paths(manifest: dict, root_dir: str) -> List[str]:
    # manifest["shards"] entries look like {"path": "...", "frames": n}
    shards = manifest["shards"]
    paths = []
    for s in shards:
        p = s["path"]
        # allow either absolute or relative
        if not os.path.isabs(p):
            p = os.path.join(root_dir, p)
        paths.append(p)
    return paths

def load_tensor_or_dict(path: str, keys: Tuple[str, ...]) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
    raise KeyError(f"Could not find any of keys={keys} in {path}")

def ensure_cnhw(X: torch.Tensor, H: int, W: int) -> torch.Tensor:
    # expected (C,N,H,W)
    if X.dim() != 4:
        raise RuntimeError(f"Expected X dim=4 (C,N,H,W), got {tuple(X.shape)}")
    if X.shape[-2] != H or X.shape[-1] != W:
        raise RuntimeError(f"Expected X HxW={H}x{W}, got {X.shape[-2]}x{X.shape[-1]}")
    return X.contiguous()

def ensure_n3(T: torch.Tensor) -> torch.Tensor:
    # expected (N,3)
    if T.dim() != 2 or T.shape[1] != 3:
        raise RuntimeError(f"Expected targets (N,3), got {tuple(T.shape)}")
    return T.contiguous()

def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Dataset: one shard at a time
# -------------------------
class OneShardDataset(Dataset):
    def __init__(self, X_cnhw: torch.Tensor, T_n3: torch.Tensor, swap_xy: bool = False):
        self.X = X_cnhw  # (C,N,H,W)
        self.T = T_n3    # (N,3) -> [end_x, end_y, success]
        self.swap_xy = swap_xy

    def __len__(self) -> int:
        return self.T.shape[0]

    def __getitem__(self, i: int):
        x = self.X[:, i].to(torch.float32)  # (C,H,W)
        xy = self.T[i, :2].to(torch.long)   # (2,) integer meters -> indices
        if self.swap_xy:
            xy = xy[[1, 0]]
        y = self.T[i, 2].to(torch.float32)  # scalar 0/1
        return x, xy, y


def collate_batch(batch):
    xs, xys, ys = zip(*batch)
    X = torch.stack(xs, dim=0)                            # (B,C,H,W)
    dst_xy = torch.stack(xys, dim=0).to(torch.long)       # (B,2)
    y = torch.as_tensor(ys, dtype=torch.float32)          # (B,)

    # extra NaN/inf protection
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, dst_xy, y


# -------------------------
# Two-head loss (same structure as your current implementation)
# -------------------------
@dataclass
class TwoHeadLossCfg:
    w_dest: float = 1.0
    w_succ: float = 1.0

def twohead_loss(out: Dict[str, torch.Tensor],
                 y: torch.Tensor,
                 dst_xy: torch.Tensor,
                 cfg: TwoHeadLossCfg) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    out: {"dest_logits": (B,1,H,W), "succ_logits": (B,1,H,W)}
    Destination: softmax over HW with cross-entropy to the true cell.
    Success: BCEWithLogits at the true destination cell.
    Mirrors the intent of your existing two-head loss. :contentReference[oaicite:7]{index=7}
    """
    dest_logits = out["dest_logits"]
    succ_logits = out["succ_logits"]
    if dest_logits.shape != succ_logits.shape:
        raise RuntimeError(f"dest/succ logits mismatch: {tuple(dest_logits.shape)} vs {tuple(succ_logits.shape)}")

    B, _, H, W = dest_logits.shape
    device = dest_logits.device

    # sanitize targets
    y = y.to(device=device, dtype=dest_logits.dtype).view(B)
    y = torch.clamp(y, 0.0, 1.0)

    x_idx = dst_xy[:, 0].long().clamp(0, H - 1).to(device)
    y_idx = dst_xy[:, 1].long().clamp(0, W - 1).to(device)

    logs: Dict[str, float] = {}
    total = dest_logits.new_tensor(0.0)

    # dest loss
    if cfg.w_dest > 0:
        dest_flat = dest_logits.view(B, -1)               # (B, H*W)
        dest_index = (x_idx * W + y_idx).long()           # (B,)
        l_dest = F.cross_entropy(dest_flat, dest_index, reduction="mean")
        total = total + cfg.w_dest * l_dest
        logs["loss_dest"] = float(l_dest.detach().cpu())

    # success loss at the sampled cell
    if cfg.w_succ > 0:
        succ_at = succ_logits[torch.arange(B, device=device), 0, x_idx, y_idx]
        l_succ = F.binary_cross_entropy_with_logits(succ_at, y, reduction="mean")
        total = total + cfg.w_succ * l_succ
        logs["loss_succ"] = float(l_succ.detach().cpu())

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def eval_epoch(model: nn.Module,
               feat_paths: List[str],
               targ_paths: List[str],
               device: torch.device,
               batch_size: int,
               num_workers: int,
               swap_xy: bool,
               loss_cfg: TwoHeadLossCfg) -> Tuple[float, float, float]:
    model.eval()
    total, total_dest, total_succ, n = 0.0, 0.0, 0.0, 0

    for fp, tp in zip(feat_paths, targ_paths):
        X = ensure_cnhw(load_tensor_or_dict(fp, ("X",)), H=105, W=68)
        T = ensure_n3(load_tensor_or_dict(tp, ("targets", "y", "Y")))

        ds = OneShardDataset(X, T, swap_xy=swap_xy)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)

        for xb, dst_xy, y in dl:
            xb = xb.to(device, non_blocking=True)
            dst_xy = dst_xy.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(xb)
            loss, logs = twohead_loss(out, y, dst_xy, loss_cfg)
            b = xb.size(0)
            total += float(loss.detach().cpu()) * b
            total_dest += logs.get("loss_dest", 0.0) * b
            total_succ += logs.get("loss_succ", 0.0) * b
            n += b

    n = max(1, n)
    return total / n, total_dest / n, total_succ / n


def train():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True,
               help="Root directory containing train/ and val/ folders with memmap manifests.")
    p.add_argument("--cache_size", type=int, default=2,
                help="How many memmap shards to keep open (LRU).")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--swap_xy", action="store_true")

    # loss weights
    p.add_argument("--w_dest", type=float, default=1.0)
    p.add_argument("--w_succ", type=float, default=1.0)

    # output
    p.add_argument("--runs_dir", type=str, default="runs")
    p.add_argument("--save_path", type=str, default="best_ckpt.pt")
    args = p.parse_args()

    # final runs directory

    arch_name = model.__class__.__name__

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    args.run_dir = os.path.join(args.runs_dir, f"{stamp}_{arch_name}")
    os.makedirs(args.run_dir, exist_ok=True)



    set_all_seeds(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"Device: {device}")

    train_root = os.path.join(args.data_root, "train")
    val_root   = os.path.join(args.data_root, "val")

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

    print(f"[data] train samples={len(train_ds)} val samples={len(val_ds)}")
    print(f"[data] C,H,W = {train_ds.C},{train_ds.H},{train_ds.W}")


    train_dl = DataLoader(
    train_ds,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=(device.type == "cuda"),
    collate_fn=collate_batch,
    drop_last=True,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_batch,
        drop_last=False,
    )


    # model
    from models.passmap import BetterSoccerMap2Head
    model = BetterSoccerMap2Head(in_channels=14, base=64, blocks_per_stage=2, dropout=0.0).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_cfg = TwoHeadLossCfg(w_dest=args.w_dest, w_succ=args.w_succ)

    best_val = float("inf")
    history = {"train_total": [], "train_dest": [], "train_succ": [],
               "val_total": [], "val_dest": [], "val_succ": []}

    def _run_epoch(dl, train_mode: bool):
        if train_mode:
            model.train()
        else:
            model.eval()

        running_total, running_dest, running_succ, n_seen = 0.0, 0.0, 0.0, 0

        it = tqdm(dl, desc=("train" if train_mode else "val"), leave=False)
        for xb, dst_xy, y in it:
            xb = xb.to(device, non_blocking=True)
            dst_xy = dst_xy.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # hard stop if NaNs sneak in
            if not torch.isfinite(xb).all():
                raise RuntimeError("Non-finite xb detected in dataloader batch")

            if train_mode:
                opt.zero_grad(set_to_none=True)

            with torch.set_grad_enabled(train_mode):
                out = model(xb)
                loss, logs = twohead_loss(out, y, dst_xy, loss_cfg)

                if not torch.isfinite(loss):
                    print("\nNon-finite loss encountered!")
                    print("  xb min/max:", float(xb.min()), float(xb.max()))
                    print("  dest_logits min/max:",
                          float(out["dest_logits"].min().detach().cpu()),
                          float(out["dest_logits"].max().detach().cpu()))
                    print("  succ_logits min/max:",
                          float(out["succ_logits"].min().detach().cpu()),
                          float(out["succ_logits"].max().detach().cpu()))
                    raise RuntimeError("Stopping due to non-finite loss")

                if train_mode:
                    loss.backward()
                    if args.grad_clip and args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    opt.step()

            b = xb.size(0)
            running_total += float(logs["loss_total"]) * b
            running_dest  += float(logs.get("loss_dest", 0.0)) * b
            running_succ  += float(logs.get("loss_succ", 0.0)) * b
            n_seen += b

            it.set_postfix({
                "loss": f"{running_total/max(1,n_seen):.4f}",
                "dest": f"{running_dest/max(1,n_seen):.4f}",
                "succ": f"{running_succ/max(1,n_seen):.4f}",
            })

        return (
            running_total / max(1, n_seen),
            running_dest  / max(1, n_seen),
            running_succ  / max(1, n_seen),
        )

    for epoch in range(args.epochs):
        t0 = time.time()

        train_total, train_dest, train_succ = _run_epoch(train_dl, train_mode=True)
        val_total, val_dest, val_succ = _run_epoch(val_dl,   train_mode=False)

        dt = time.time() - t0
        print(f"Epoch {epoch:03d} | "
              f"train={train_total:.6f} (dest={train_dest:.6f}, succ={train_succ:.6f}) | "
              f"val={val_total:.6f} (dest={val_dest:.6f}, succ={val_succ:.6f}) | "
              f"time={dt:.1f}s")

        history["train_total"].append(train_total)
        history["train_dest"].append(train_dest)
        history["train_succ"].append(train_succ)
        history["val_total"].append(val_total)
        history["val_dest"].append(val_dest)
        history["val_succ"].append(val_succ)

        if val_total < best_val:
            best_val = val_total
            ckpt_path = os.path.join(args.run_dir, args.save_path)
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "history": history,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ✓ saved best checkpoint -> {ckpt_path}")

    # plot losses
    fig_path = os.path.join(args.run_dir, "loss_curve.png")
    xs = list(range(args.epochs))
    plt.figure()
    plt.plot(xs, history["train_total"], label="train_total")
    plt.plot(xs, history["val_total"], label="val_total")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Epoch vs Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    print(f"Saved loss plot -> {fig_path}")

    save_json(os.path.join(args.run_dir, "history.json"), history)



if __name__ == "__main__":
    train()
