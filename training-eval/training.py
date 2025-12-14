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


class ManifestDataset(Dataset):
    """
    Dataset over (shard_id, sample_idx) pairs.
    Uses a small LRU cache of loaded shards to avoid thrashing disk.
    """
    def __init__(
        self,
        ids: torch.Tensor,          # (N,2) long: (shard_id, idx)
        feat_paths: list,
        targ_paths: list,
        swap_xy: bool = False,
        cache_size: int = 4,
        H: int = 105,
        W: int = 68,
    ):
        assert ids.ndim == 2 and ids.shape[1] == 2
        self.ids = ids.long().cpu()
        self.feat_paths = feat_paths
        self.targ_paths = targ_paths
        self.swap_xy = swap_xy
        self.cache_size = int(cache_size)
        self.H, self.W = int(H), int(W)

        # shard_id -> (X_cnhw, T_n3)
        self._cache = OrderedDict()

    def __len__(self) -> int:
        return self.ids.shape[0]

    def _get_shard(self, shard_id: int):
        shard_id = int(shard_id)
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]

        fp = self.feat_paths[shard_id]
        tp = self.targ_paths[shard_id]

        X = ensure_cnhw(load_tensor_or_dict(fp, ("X",)), H=self.H, W=self.W)
        T = ensure_n3(load_tensor_or_dict(tp, ("targets", "y", "Y")))

        # keep on CPU, float32 conversion happens per-sample
        X = X.cpu()
        T = T.cpu()

        self._cache[shard_id] = (X, T)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)  # LRU eviction
        return X, T

    def __getitem__(self, j: int):
        sid = int(self.ids[j, 0].item())
        i = int(self.ids[j, 1].item())

        X, T = self._get_shard(sid)

        x = X[:, i].to(torch.float32)  # (C,H,W)
        xy = T[i, :2].to(torch.long)   # (2,)
        if self.swap_xy:
            xy = xy[[1, 0]]
        y = T[i, 2].to(torch.float32)  # scalar 0/1

        return x, xy, y

def _fingerprint_paths(paths: List[str]) -> str:
    """
    Stable-ish fingerprint so you can detect if shards changed order/content.
    Uses: basename + size + mtime_ns.
    """
    h = hashlib.sha1()
    for p in paths:
        st = os.stat(p)
        key = f"{os.path.basename(p)}|{st.st_size}|{st.st_mtime_ns}\n"
        h.update(key.encode("utf-8"))
    return h.hexdigest()


def make_or_load_sample_split(
    split_path: str,
    feat_paths: List[str],
    targ_paths: List[str],
    val_frac: float,
    seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """
    Returns:
      train_ids: (N_train, 2) long tensor of (shard_id, sample_idx)
      val_ids:   (N_val, 2)   long tensor of (shard_id, sample_idx)
      meta: dict with useful info + fingerprints

    Stratification is on success label T[:,2] (0/1).
    """
    os.makedirs(os.path.dirname(split_path), exist_ok=True)

    # if exists -> load and validate
    if os.path.exists(split_path):
        obj = torch.load(split_path, map_location="cpu")
        train_ids = obj["train_ids"].long()
        val_ids = obj["val_ids"].long()
        meta = obj.get("meta", {})

        # sanity: no overlap
        train_set = set(map(tuple, train_ids.tolist()))
        val_set = set(map(tuple, val_ids.tolist()))
        inter = train_set.intersection(val_set)
        if len(inter) != 0:
            raise RuntimeError(f"Split file has leakage: {len(inter)} overlapping samples.")

        # optional: warn if data changed
        cur_feat_fp = _fingerprint_paths(feat_paths)
        cur_targ_fp = _fingerprint_paths(targ_paths)
        if meta.get("feat_fingerprint") != cur_feat_fp or meta.get("targ_fingerprint") != cur_targ_fp:
            print(
                "[WARN] Split fingerprints do not match current shard manifests.\n"
                "       You may have regenerated/reordered shards. Consider deleting the split file and rebuilding."
            )

        return train_ids, val_ids, meta

    # build new split
    rng = torch.Generator().manual_seed(seed)

    ones: List[Tuple[int, int]] = []
    zeros: List[Tuple[int, int]] = []

    for sid, tp in enumerate(targ_paths):
        T = ensure_n3(load_tensor_or_dict(tp, ("targets", "y", "Y"))).cpu()
        if T.numel() == 0:
            continue
        y = T[:, 2]
        # robust to float 0/1
        y = (y > 0.5).to(torch.int64)

        for i in range(y.shape[0]):
            if y[i].item() == 1:
                ones.append((sid, i))
            else:
                zeros.append((sid, i))

    if len(ones) == 0 or len(zeros) == 0:
        raise RuntimeError(
            f"Cannot stratify: ones={len(ones)} zeros={len(zeros)}. "
            "Check your targets tensor format."
        )

    # shuffle within each class deterministically
    ones_t = torch.tensor(ones, dtype=torch.long)
    zeros_t = torch.tensor(zeros, dtype=torch.long)

    ones_perm = torch.randperm(ones_t.shape[0], generator=rng)
    zeros_perm = torch.randperm(zeros_t.shape[0], generator=rng)
    ones_t = ones_t[ones_perm]
    zeros_t = zeros_t[zeros_perm]

    n_val_ones = int(round(val_frac * ones_t.shape[0]))
    n_val_zeros = int(round(val_frac * zeros_t.shape[0]))

    val_ids = torch.cat([ones_t[:n_val_ones], zeros_t[:n_val_zeros]], dim=0)
    train_ids = torch.cat([ones_t[n_val_ones:], zeros_t[n_val_zeros:]], dim=0)

    # final shuffle (still deterministic)
    val_ids = val_ids[torch.randperm(val_ids.shape(meta=val_ids.shape[0])[0] if False else val_ids.shape[0], generator=rng)]
    train_ids = train_ids[torch.randperm(train_ids.shape[0], generator=rng)]

    meta = {
        "seed": seed,
        "val_frac": val_frac,
        "n_train": int(train_ids.shape[0]),
        "n_val": int(val_ids.shape[0]),
        "n_ones": int(len(ones)),
        "n_zeros": int(len(zeros)),
        "feat_fingerprint": _fingerprint_paths(feat_paths),
        "targ_fingerprint": _fingerprint_paths(targ_paths),
    }

    torch.save({"train_ids": train_ids, "val_ids": val_ids, "meta": meta}, split_path)
    print(f"[split] wrote {split_path} | train={meta['n_train']} val={meta['n_val']}")
    return train_ids, val_ids, meta


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
# Fixed shard split (saved once)
# -------------------------
def make_or_load_split(split_path: str, n_shards: int, val_frac: float, seed: int) -> Tuple[List[int], List[int]]:
    """
    Fixed split of shard indices saved to split_path, so val never leaks into training.
    This is the same *idea* as your current shard-level split logic. :contentReference[oaicite:8]{index=8}
    """
    if os.path.exists(split_path):
        s = load_json(split_path)
        return s["train_idx"], s["val_idx"]

    idx = list(range(n_shards))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(round(val_frac * n_shards))
    val_idx = sorted(idx[:n_val])
    train_idx = sorted(idx[n_val:])

    save_json(split_path, {"train_idx": train_idx, "val_idx": val_idx, "seed": seed, "val_frac": val_frac})
    return train_idx, val_idx


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
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--targets_dir", type=str, required=True)

    # persistent split (shared across runs)
    p.add_argument("--split_path", type=str, default="data/splits/split_v1_seed19_val0.10.pt")
    p.add_argument("--split_level", type=str, choices=["shard", "sample"], default="sample")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=19)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--swap_xy", action="store_true")

    # loss weights
    p.add_argument("--w_dest", type=float, default=1.0)
    p.add_argument("--w_succ", type=float, default=1.0)

    # output
    p.add_argument("--run_dir", type=str, default="runs/better2head")
    p.add_argument("--save_path", type=str, default="best_ckpt.pt")

    args = p.parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.split_path), exist_ok=True)

    set_all_seeds(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    print(f"Device: {device}")

    # manifests
    fman = load_json(os.path.join(args.features_dir, "manifest.json"))
    tman = load_json(os.path.join(args.targets_dir, "targets_manifest.json"))
    feat_paths = resolve_manifest_paths(fman, args.features_dir)
    targ_paths = resolve_manifest_paths(tman, args.targets_dir)
    if len(feat_paths) != len(targ_paths):
        raise RuntimeError(f"Shard mismatch: feats={len(feat_paths)} targets={len(targ_paths)}")

    # split (sample-level stratified)
    if args.split_level != "sample":
        raise NotImplementedError("This train() rewrite expects --split_level sample")

    train_ids, val_ids, split_meta = make_or_load_sample_split(
        split_path=args.split_path,
        feat_paths=feat_paths,
        targ_paths=targ_paths,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    print(
        f"[split] train={train_ids.shape[0]} val={val_ids.shape[0]} "
        f"(saved at {args.split_path})"
    )

    # datasets + loaders
    train_ds = ManifestDataset(train_ids, feat_paths, targ_paths, swap_xy=args.swap_xy, cache_size=4)
    val_ds   = ManifestDataset(val_ids,   feat_paths, targ_paths, swap_xy=args.swap_xy, cache_size=4)

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
                "split_meta": split_meta,
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
