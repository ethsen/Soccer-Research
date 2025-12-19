#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from utils.static_maps import PitchStaticChannels, PitchDims
from utils.train_utils import *
from models.footballmap import PassMap
from models.pitchvision import PitchVisionNet 

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

# -----------------------------
# Metrics helpers
# -----------------------------
def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_score = y_score.astype(np.float64)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(y_score), dtype=np.float64) + 1.0

    # average ranks for ties
    sorted_scores = y_score[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i : j + 1]] = avg
        i = j + 1

    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    y_true = y_true.astype(np.float64)
    y_prob = np.clip(y_prob.astype(np.float64), 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N = len(y_true)

    for b0, b1 in zip(bins[:-1], bins[1:]):
        if b1 < 1.0:
            mask = (y_prob >= b0) & (y_prob < b1)
        else:
            mask = (y_prob >= b0) & (y_prob <= b1)
        nb = int(mask.sum())
        if nb == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (nb / N) * abs(acc - conf)

    return float(ece)


def mean_patch(prob_map: torch.Tensor, x: torch.Tensor, y: torch.Tensor, r: int) -> torch.Tensor:
    """prob_map: (B,H,W), x/y: (B,), returns (B,) mean in (2r+1)^2 patch"""
    B, H, W = prob_map.shape
    out = []
    for i in range(B):
        x0 = max(0, int(x[i]) - r)
        x1 = min(H, int(x[i]) + r + 1)
        y0 = max(0, int(y[i]) - r)
        y1 = min(W, int(y[i]) + r + 1)
        out.append(prob_map[i, x0:x1, y0:y1].mean())
    return torch.stack(out, dim=0)


def clip_xy(dst_xy: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x_idx = dst_xy[:, 0].long().clamp(0, H - 1)
    y_idx = dst_xy[:, 1].long().clamp(0, W - 1)
    return x_idx, y_idx


# -----------------------------
# Ablation application (match train.py)
# -----------------------------
def apply_ablation(
    xb: torch.Tensor,                      # (B,C_dyn,H,W)
    static: PitchStaticChannels,           # produces (C_static,H,W)
    dyn_keep_idxs: List[int],
    static_keep_idxs: List[int],
    zero_ablation: bool,
) -> torch.Tensor:
    """
    Returns concatenated input (B, C_dyn_eff + C_static_eff, H, W)
    following same logic as in train.run_epoch().
    """
    device = xb.device
    dtype = xb.dtype

    # dyn
    if zero_ablation:
        m = torch.zeros(xb.shape[1], device=device, dtype=dtype)
        m[dyn_keep_idxs] = 1
        xb = xb * m.view(1, -1, 1, 1)
    else:
        xb = xb[:, dyn_keep_idxs]

    # static
    st = static.expand_to_batch(xb.size(0)).to(device=device, dtype=dtype)  # (B,C_static,H,W)
    if zero_ablation:
        sm = torch.zeros(st.shape[1], device=device, dtype=dtype)
        sm[static_keep_idxs] = 1
        st = st * sm.view(1, -1, 1, 1)
    else:
        st = st[:, static_keep_idxs]

    return torch.cat([xb, st], dim=1)


# -----------------------------
# Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    val_dl: DataLoader,
    device: torch.device,
    static: PitchStaticChannels,
    dyn_keep_idxs: List[int],
    static_keep_idxs: List[int],
    zero_ablation: bool,
    patch_r: int = 0,
    ece_bins: int = 15,
    topk: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:

    model.eval()

    all_p = []
    all_y = []
    all_p_patch = []

    all_dest_nll = []
    topk_hits = {k: 0 for k in topk}
    n_total = 0

    for xb, dst_xy, y in val_dl:
        xb = xb.to(device, non_blocking=True)          # (B,C_dyn,H,W)
        dst_xy = dst_xy.to(device, non_blocking=True)  # (B,2)
        y = y.to(device, non_blocking=True).float().view(-1).clamp(0.0, 1.0)

        x_in = apply_ablation(xb, static, dyn_keep_idxs, static_keep_idxs, zero_ablation)
        out = model(x_in)

        dest_logits = out["dest_logits"][:, 0]  # (B,H,W)
        succ_logits = out["succ_logits"][:, 0]  # (B,H,W)

        B, H, W = dest_logits.shape
        x_idx, y_idx = clip_xy(dst_xy, H, W)

        # ---- Success@Dest ----
        succ_at = succ_logits[torch.arange(B, device=device), x_idx, y_idx]
        p = torch.sigmoid(succ_at)  # (B,)
        all_p.append(p.detach().cpu().numpy())
        all_y.append(y.detach().cpu().numpy())

        if patch_r and patch_r > 0:
            succ_map = torch.sigmoid(succ_logits)  # (B,H,W)
            p_patch = mean_patch(succ_map, x_idx, y_idx, r=patch_r)
            all_p_patch.append(p_patch.detach().cpu().numpy())

        # ---- Destination metrics ----
        log_probs = F.log_softmax(dest_logits.view(B, -1), dim=1)  # (B,HW)
        gt_index = (x_idx * W + y_idx).long()
        nll = -log_probs[torch.arange(B, device=device), gt_index]
        all_dest_nll.append(nll.detach().cpu().numpy())

        for k in topk:
            topk_idx = torch.topk(log_probs, k=k, dim=1).indices
            hit = (topk_idx == gt_index.view(B, 1)).any(dim=1).sum().item()
            topk_hits[k] += int(hit)

        n_total += B

    p = np.concatenate(all_p, axis=0)
    y = np.concatenate(all_y, axis=0)
    dest_nll = np.concatenate(all_dest_nll, axis=0)

    # Success@Dest metrics
    succ_nll = float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())
    succ_auc = auc_roc(y, p)
    mean_pos = float(p[y == 1].mean()) if (y == 1).any() else float("nan")
    mean_neg = float(p[y == 0].mean()) if (y == 0).any() else float("nan")
    delta = float(mean_pos - mean_neg) if np.isfinite(mean_pos) and np.isfinite(mean_neg) else float("nan")
    ece = expected_calibration_error(y, p, n_bins=ece_bins)

    # Destination metrics
    dest_nll_mean = float(dest_nll.mean())
    dest_topk = {f"dest_top{k}": float(topk_hits[k] / max(1, n_total)) for k in topk}

    metrics: Dict[str, float] = {
        "N": float(n_total),
        "succ_at_dest_nll": succ_nll,
        "succ_at_dest_auc": float(succ_auc),
        "succ_at_dest_mean_pos": mean_pos,
        "succ_at_dest_mean_neg": mean_neg,
        "succ_at_dest_delta": delta,
        "succ_at_dest_ece": float(ece),
        "dest_nll": dest_nll_mean,
        **dest_topk,
    }

    if patch_r and patch_r > 0:
        p_patch = np.concatenate(all_p_patch, axis=0)
        succ_patch_nll = float(-(y * np.log(p_patch + 1e-9) + (1 - y) * np.log(1 - p_patch + 1e-9)).mean())
        succ_patch_auc = auc_roc(y, p_patch)
        metrics.update({
            f"succ_patch_r{patch_r}_nll": succ_patch_nll,
            f"succ_patch_r{patch_r}_auc": float(succ_patch_auc),
        })

    return metrics


# -----------------------------
# Main
# -----------------------------
def parse_csv(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--arch", type=str, required=True, choices=["passmap", "pitchvision"])

    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--swap_xy", action="store_true")
    ap.add_argument("--cache_size", type=int, default=2)

    # Ablation spec
    ap.add_argument("--dyn_used", type=str, required=True,
                    help="Comma-separated dynamic channel names used by this model.")
    ap.add_argument("--use_static", action="store_true",
                    help="If set, include static channels (default: off).")
    ap.add_argument("--zero_ablation", action="store_true",
                    help="If set, zero-out dropped channels instead of slicing.")
    ap.add_argument("--patch_r", type=int, default=0)
    ap.add_argument("--ece_bins", type=int, default=15)

    ap.add_argument("--out_json", type=str, default="val_metrics.json")

    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")

    # ---- dataset ----
    val_root = os.path.join(args.data_root, "val")
    val_ds = MemmapManifestDataset(
        manifest_path=os.path.join(val_root, "manifest.json"),
        root_dir=val_root,
        swap_xy=args.swap_xy,
        cache_size=args.cache_size,
    )

    dyn_names = val_ds.channels
    dyn_name_to_idx = {n: i for i, n in enumerate(dyn_names)}

    # Static channels must match PitchStaticChannels stacking order
    static_names = ["boundary_dist_norm", "centerline_dist_norm", "goal_sin", "goal_cos", "goal_dist_norm"]
    static_name_to_idx = {n: i for i, n in enumerate(static_names)}

    dyn_used = parse_csv(args.dyn_used)
    dyn_keep_idxs = compute_keep_indices(dyn_names, dyn_name_to_idx, ",".join(dyn_used), "", allow_empty=False)

    if args.use_static:
        static_keep_idxs = list(range(len(static_names)))
    else:
        static_keep_idxs = []  # drop all statics

    # ---- static maps module ----
    static = PitchStaticChannels(dims=PitchDims(H=val_ds.H, W=val_ds.W)).to(device=device)

    C_static = int(static.forward().shape[0])
    C_dyn_eff = val_ds.C if args.zero_ablation else len(dyn_keep_idxs)
    C_static_eff = C_static if args.zero_ablation else len(static_keep_idxs)
    in_channels = C_dyn_eff + C_static_eff

    # ---- model ----
    model = build_model(in_channels=in_channels, arch=args.arch).to(device).float()

    ckpt = torch.load(args.ckpt, map_location="cpu",weights_only=False)
    state = ckpt.get("model_state", ckpt)  # train.py saves under "model_state" :contentReference[oaicite:1]{index=1}
    model.load_state_dict({k.replace("module.", ""): v for k, v in state.items()}, strict=True)

    # ---- loader ----
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

    metrics = evaluate_val(
        model=model,
        val_dl=val_dl,
        device=device,
        static=static,
        dyn_keep_idxs=dyn_keep_idxs,
        static_keep_idxs=static_keep_idxs,
        zero_ablation=args.zero_ablation,
        patch_r=args.patch_r,
        ece_bins=args.ece_bins,
    )

    print(json.dumps(metrics, indent=2))

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[saved] {out_path.resolve()}")


if __name__ == "__main__":
    main()
