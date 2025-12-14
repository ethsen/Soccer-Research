#!/usr/bin/env python3
import os, json, time, random, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ------------------------------------------------------------
# Helpers: robust loading
# ------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def load_tensor_or_dict(path: str, key_candidates: Tuple[str, ...]) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in key_candidates:
            if k in obj and torch.is_tensor(obj[k]):
                return obj[k]
        # fallback: first tensor value
        for v in obj.values():
            if torch.is_tensor(v):
                return v
        raise KeyError(f"No tensor under keys {key_candidates} in {path}. Keys={list(obj.keys())}")
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Unsupported object in {path}: {type(obj)}")

def ensure_cnhw(feats: torch.Tensor, H: int, W: int) -> torch.Tensor:
    assert feats.dim() == 4, f"features must be 4D (C,N,*,*), got {feats.shape}"
    C, N, A, B = feats.shape
    if (A, B) == (H, W):
        return feats
    if (A, B) == (W, H):
        return feats.permute(0, 1, 3, 2).contiguous()
    raise ValueError(f"Unexpected feature spatial size {(A,B)}; expected {(H,W)} or {(W,H)}")

def ensure_n3(targs: torch.Tensor) -> torch.Tensor:
    assert targs.dim() == 2, f"targets must be 2D, got {targs.shape}"
    if targs.shape[1] == 3:
        return targs
    if targs.shape[0] == 3:
        return targs.t().contiguous()
    raise ValueError(f"Unexpected target shape {targs.shape}; expected (N,3) or (3,N)")

def resolve_manifest_paths(man: dict, root: str) -> List[str]:
    """
    Robustly resolves shard paths regardless of whether manifest stores:
      - "features_shard_0001.pt"
      - "soccer_shards/features_shard_0001.pt"
      - absolute paths
    Strategy:
      - If absolute -> keep
      - Else try join(root, p); if exists, use it
      - Else try join(dirname(root), p); if exists, use it
      - Else raise
    """
    out = []
    root = os.path.normpath(root)
    root_parent = os.path.dirname(root)

    for sh in man["shards"]:
        p = sh["path"]
        if os.path.isabs(p):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            out.append(p)
            continue

        p1 = os.path.normpath(os.path.join(root, p))
        if os.path.exists(p1):
            out.append(p1)
            continue

        p2 = os.path.normpath(os.path.join(root_parent, p))
        if os.path.exists(p2):
            out.append(p2)
            continue

        raise FileNotFoundError(f"Could not resolve shard path '{p}' from root='{root}'")
    return out

# ------------------------------------------------------------
# Dataset (single shard in memory)
# ------------------------------------------------------------

class OneShardDataset(Dataset):
    """
    features: (C,N,H,W)
    targets:  (N,3) [x_idx, y_idx, outcome]
    """
    def __init__(self, feats: torch.Tensor, targs: torch.Tensor, swap_xy: bool = False):
        super().__init__()
        self.feats = feats
        self.targs = targs
        self.swap_xy = swap_xy
        assert feats.shape[1] == targs.shape[0], f"N mismatch feats N={feats.shape[1]} vs targs N={targs.shape[0]}"

    def __len__(self) -> int:
        return self.targs.shape[0]

    def __getitem__(self, i: int):
        x = self.feats[:, i]                 # (C,H,W)
        xy = self.targs[i, :2].long()        # (2,)
        if self.swap_xy:
            xy = xy[[1, 0]]
        y = self.targs[i, 2].float()         # scalar
        return x, xy, y

def collate_batch(batch):
    xs, xys, ys = zip(*batch)
    X = torch.stack(xs, dim=0).to(torch.float32)      # (B,C,H,W)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    dst_xy = torch.stack(xys, dim=0).to(torch.long)   # (B,2)
    y = torch.as_tensor(ys, dtype=torch.float32)      # (B,)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, dst_xy, y

# ------------------------------------------------------------
# Losses
# ------------------------------------------------------------

@dataclass
class OneHeadLossCfg:
    w_single: float = 1.0          # pixel BCE at dst cell
    w_heatmap: float = 0.0         # optional dense aux
    sigma: float = 2.0
    clip: float = 1e-4

def gaussian_heatmap(dst_xy: torch.Tensor, H: int, W: int, sigma: float, dtype, device) -> torch.Tensor:
    B = dst_xy.shape[0]
    xs = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    x0 = dst_xy[:, 0].to(dtype).view(B, 1, 1)
    y0 = dst_xy[:, 1].to(dtype).view(B, 1, 1)
    inv = 1.0 / (2.0 * (sigma ** 2))
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    g = torch.exp(-d2 * inv)  # (B,H,W)
    return g.unsqueeze(1)     # (B,1,H,W)

def onehead_loss_from_logits(logits: torch.Tensor, y: torch.Tensor, dst_xy: torch.Tensor, cfg: OneHeadLossCfg) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    logits: (B,1,H,W)
    y: (B,) {0,1}
    dst_xy: (B,2) (x_idx,y_idx)
    """
    B, _, H, W = logits.shape
    device = logits.device
    y = y.to(device=device, dtype=logits.dtype).view(B)
    x_idx = dst_xy[:, 0].long().clamp(0, H - 1).to(device)
    y_idx = dst_xy[:, 1].long().clamp(0, W - 1).to(device)

    total = logits.new_tensor(0.0)
    logs: Dict[str, float] = {}

    if cfg.w_single > 0:
        pix = logits[torch.arange(B, device=device), 0, x_idx, y_idx]
        l_single = F.binary_cross_entropy_with_logits(pix, y, reduction="mean")
        total = total + cfg.w_single * l_single
        logs["loss_single"] = float(l_single.detach().cpu())

    if cfg.w_heatmap > 0:
        g = gaussian_heatmap(dst_xy.to(device), H, W, cfg.sigma, logits.dtype, device)
        g = g * y.view(B, 1, 1, 1)  # gate by success if you want; keep as-is from prior approach
        if cfg.clip and cfg.clip > 0:
            g = g.clamp(cfg.clip, 1.0 - cfg.clip)
        l_heat = F.binary_cross_entropy_with_logits(logits, g, reduction="mean")
        total = total + cfg.w_heatmap * l_heat
        logs["loss_heatmap"] = float(l_heat.detach().cpu())

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs


@dataclass
class TwoHeadLossCfg:
    w_dest: float = 1.0
    w_succ: float = 1.0
    # optional dense destination supervision
    dense_dest: bool = False
    dense_dest_sigma: float = 2.0
    dense_dest_eps: float = 1e-6
    # optional dense succ auxiliary
    dense_succ: bool = False
    dense_succ_sigma: float = 2.0
    dense_succ_clip: float = 1e-4
    dense_succ_aux_scale: float = 0.25  # small extra weight

def gaussian_unnormalized(dst_xy: torch.Tensor, H: int, W: int, sigma: float, dtype, device) -> torch.Tensor:
    B = dst_xy.shape[0]
    xs = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    x0 = dst_xy[:, 0].to(dtype).view(B, 1, 1)
    y0 = dst_xy[:, 1].to(dtype).view(B, 1, 1)
    inv = 1.0 / (2.0 * (sigma ** 2))
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    g = torch.exp(-d2 * inv)  # (B,H,W)
    return g.unsqueeze(1)

def twohead_loss(out: Dict[str, torch.Tensor], y: torch.Tensor, dst_xy: torch.Tensor, cfg: TwoHeadLossCfg) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    out: {"dest_logits": (B,1,H,W), "succ_logits": (B,1,H,W)}
    """
    dest_logits = out["dest_logits"]
    succ_logits = out["succ_logits"]
    assert dest_logits.shape == succ_logits.shape, "dest/succ logits shape mismatch"
    B, _, H, W = dest_logits.shape
    device = dest_logits.device

    y = y.to(device=device, dtype=dest_logits.dtype).view(B)
    x_idx = dst_xy[:, 0].long().clamp(0, H - 1).to(device)
    y_idx = dst_xy[:, 1].long().clamp(0, W - 1).to(device)

    total = dest_logits.new_tensor(0.0)
    logs: Dict[str, float] = {}

    # Destination: p(dest|s) softmax over HW
    if cfg.w_dest > 0:
        if not cfg.dense_dest:
            # hard CE
            dest_flat = dest_logits.view(B, -1)
            dest_index = (x_idx * W + y_idx).long()
            l_dest = F.cross_entropy(dest_flat, dest_index, reduction="mean")
        else:
            g = gaussian_unnormalized(torch.stack([x_idx, y_idx], dim=1), H, W,
                                      cfg.dense_dest_sigma, dest_logits.dtype, device).view(B, -1)
            g = g / (g.sum(dim=1, keepdim=True) + cfg.dense_dest_eps)  # target distribution
            logp = F.log_softmax(dest_logits.view(B, -1), dim=1)
            l_dest = (g * (torch.log(g + cfg.dense_dest_eps) - logp)).sum(dim=1).mean()
        total = total + cfg.w_dest * l_dest
        logs["loss_dest"] = float(l_dest.detach().cpu())

    # Success: p(success|s,dest) sigmoid at sampled dest cell
    if cfg.w_succ > 0:
        succ_at = succ_logits[torch.arange(B, device=device), 0, x_idx, y_idx]
        l_succ = F.binary_cross_entropy_with_logits(succ_at, y, reduction="mean")
        total = total + cfg.w_succ * l_succ
        logs["loss_succ"] = float(l_succ.detach().cpu())

    # Optional dense succ auxiliary
    if cfg.dense_succ and cfg.w_succ > 0:
        g = gaussian_unnormalized(torch.stack([x_idx, y_idx], dim=1), H, W,
                                  cfg.dense_succ_sigma, succ_logits.dtype, device)
        g = g * y.view(B, 1, 1, 1)
        if cfg.dense_succ_clip and cfg.dense_succ_clip > 0:
            g = g.clamp(cfg.dense_succ_clip, 1.0 - cfg.dense_succ_clip)
        l_succ_dense = F.binary_cross_entropy_with_logits(succ_logits, g, reduction="mean")
        total = total + cfg.w_succ * cfg.dense_succ_aux_scale * l_succ_dense
        logs["loss_succ_dense"] = float(l_succ_dense.detach().cpu())

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs

# ------------------------------------------------------------
# Model builders (plug & play)
# ------------------------------------------------------------

def build_model(model_name: str, in_channels: int) -> nn.Module:
    """
    Supported:
      --model soccermap       (your original file soccermap.py, factory soccermap_model)
      --model better          (single-head BetterSoccerMap in better_soccermap.py)
      --model better2head     (two-head BetterSoccerMap2Head in better_soccermap_twohead.py)
      --model module:ClassOrFactory
    """
    if model_name == "soccermap":
        from models.soccermap import soccermap_model
        return soccermap_model(in_channels=in_channels, base=32)

    #if model_name == "better":
        #from models.passmap import BetterSoccerMap
        #return BetterSoccerMap(in_channels=in_channels, base=64, blocks_per_stage=2, dropout=0.0)

    if model_name == "better2head":
        from models.passmap import BetterSoccerMap2Head
        return BetterSoccerMap2Head(in_channels=in_channels, base=64, blocks_per_stage=2, dropout=0.0)

    if ":" in model_name:
        mod, sym = model_name.split(":", 1)
        m = __import__(mod, fromlist=[sym])
        obj = getattr(m, sym)
        if callable(obj):
            try:
                return obj(in_channels=in_channels)
            except TypeError:
                return obj()
        raise TypeError(f"{model_name} is not callable")

    raise ValueError(f"Unknown model '{model_name}'")

def extract_onehead_logits(model_out) -> torch.Tensor:
    """
    For one-head training, accept:
      - logits tensor (B,1,H,W)
      - (probs, logits, aux)
      - dict {"logits": ...}
    """
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, dict):
        if "logits" in model_out and torch.is_tensor(model_out["logits"]):
            return model_out["logits"]
        # try common keys
        for k in ("fused_logits", "out", "pred"):
            if k in model_out and torch.is_tensor(model_out[k]):
                return model_out[k]
    if isinstance(model_out, (tuple, list)):
        for t in model_out:
            if torch.is_tensor(t) and t.dim() == 4 and t.size(1) == 1:
                return t
        if len(model_out) >= 2 and torch.is_tensor(model_out[1]):
            return model_out[1]
    raise TypeError("Could not extract logits for onehead from model output")

# ------------------------------------------------------------
# Eval
# ------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module,
             feat_paths: List[str],
             targ_paths: List[str],
             device: torch.device,
             *,
             mode: str,
             H: int, W: int,
             swap_xy: bool,
             batch_size: int,
             num_workers: int,
             one_cfg: OneHeadLossCfg,
             two_cfg: TwoHeadLossCfg) -> float:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for fp, tp in zip(feat_paths, targ_paths):
        feats = load_tensor_or_dict(fp, ("X",))
        targs = load_tensor_or_dict(tp, ("targets", "y", "Y"))
        feats = ensure_cnhw(feats, H=H, W=W)
        targs = ensure_n3(targs)

        ds = OneShardDataset(feats, targs, swap_xy=swap_xy)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)

        for X, dst_xy, y in dl:
            X = X.to(device, non_blocking=True)
            dst_xy = dst_xy.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(X)
            if mode == "onehead":
                logits = extract_onehead_logits(out)
                loss, _ = onehead_loss_from_logits(logits, y, dst_xy, one_cfg)
            else:
                loss, _ = twohead_loss(out, y, dst_xy, two_cfg)

            b = X.size(0)
            total_loss += float(loss.detach().cpu()) * b
            total_n += b

    return total_loss / max(1, total_n)

# ------------------------------------------------------------
# Main train
# ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--targets_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="better")
    p.add_argument("--mode", type=str, choices=["onehead", "twohead"], default="onehead")
    p.add_argument("--in_channels", type=int, default=None)
    p.add_argument("--swap_xy", action="store_true")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=19)

    p.add_argument("--amp", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--save_path", type=str, default="best_ckpt.pt")

    # onehead loss knobs
    p.add_argument("--w_single", type=float, default=1.0)
    p.add_argument("--w_heatmap", type=float, default=0.0)
    p.add_argument("--sigma", type=float, default=2.0)

    # twohead loss knobs
    p.add_argument("--w_dest", type=float, default=1.0)
    p.add_argument("--w_succ", type=float, default=1.0)
    p.add_argument("--dense_dest", action="store_true")
    p.add_argument("--dense_dest_sigma", type=float, default=2.0)
    p.add_argument("--dense_succ", action="store_true")
    p.add_argument("--dense_succ_sigma", type=float, default=2.0)

    args = p.parse_args()

    from tqdm import tqdm

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # manifests
    fman = load_json(os.path.join(args.features_dir, "manifest.json"))
    tman = load_json(os.path.join(args.targets_dir, "targets_manifest.json"))

    H = int(fman.get("H", 105))
    W = int(fman.get("W", 68))
    channels = fman.get("channels", [])
    in_channels = args.in_channels if args.in_channels is not None else (len(channels) if channels else 14)

    feat_paths = resolve_manifest_paths(fman, args.features_dir)
    targ_paths = resolve_manifest_paths(tman, args.targets_dir)
    if len(feat_paths) != len(targ_paths):
        raise RuntimeError(f"Shard count mismatch: feats={len(feat_paths)} targets={len(targ_paths)}")

    # split by indices to preserve alignment
    idx = list(range(len(feat_paths)))
    rng = random.Random(args.seed)
    rng.shuffle(idx)
    n_val = int(round(args.val_frac * len(idx)))
    val_set = set(idx[:n_val])
    train_idx = [i for i in range(len(idx)) if i not in val_set]
    val_idx   = [i for i in range(len(idx)) if i in val_set]

    train_feat = [feat_paths[i] for i in train_idx]
    train_targ = [targ_paths[i] for i in train_idx]
    val_feat   = [feat_paths[i] for i in val_idx]
    val_targ   = [targ_paths[i] for i in val_idx]

    # model
    model = build_model(args.model, in_channels=in_channels).to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    one_cfg = OneHeadLossCfg(w_single=args.w_single, w_heatmap=args.w_heatmap, sigma=args.sigma)
    two_cfg = TwoHeadLossCfg(
        w_dest=args.w_dest,
        w_succ=args.w_succ,
        dense_dest=args.dense_dest,
        dense_dest_sigma=args.dense_dest_sigma,
        dense_succ=args.dense_succ,
        dense_succ_sigma=args.dense_succ_sigma,
    )

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Mode: {args.mode} | Model: {args.model} | swap_xy={args.swap_xy}")
    print(f"Data: HxW={H}x{W} | in_channels={in_channels} | train_shards={len(train_feat)} val_shards={len(val_feat)}")
    if channels:
        print(f"Channels({len(channels)}): {channels}")

    best_val = float("inf")
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_sum = 0.0
        epoch_n = 0
        epoch_t0 = time.time()

        # shuffle shard order each epoch
        order = list(range(len(train_feat)))
        random.Random(args.seed + epoch).shuffle(order)

        shard_pbar = tqdm(order, desc=f"Epoch {epoch:03d} [shards]", leave=False)
        for si in shard_pbar:
            fp = train_feat[si]
            tp = train_targ[si]

            feats = load_tensor_or_dict(fp, ("X",))
            targs = load_tensor_or_dict(tp, ("targets", "y", "Y"))
            feats = ensure_cnhw(feats, H=H, W=W)
            targs = ensure_n3(targs)

            if feats.shape[0] != in_channels:
                raise RuntimeError(f"C mismatch in {fp}: got {feats.shape[0]} expected {in_channels}")
            if feats.shape[1] != targs.shape[0]:
                raise RuntimeError(f"N mismatch shard {fp}: feats N={feats.shape[1]} targets N={targs.shape[0]}")

            ds = OneShardDataset(feats, targs, swap_xy=args.swap_xy)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=True, collate_fn=collate_batch)

            shard_loss_sum = 0.0
            shard_n = 0
            shard_t0 = time.time()

            batch_pbar = tqdm(dl, desc=f"Epoch {epoch:03d} [batches]", leave=False)
            for X, dst_xy, y in batch_pbar:
                X = X.to(device, non_blocking=True)
                dst_xy = dst_xy.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    out = model(X)

                    if args.mode == "onehead":
                        logits = extract_onehead_logits(out)
                        loss, logs = onehead_loss_from_logits(logits, y, dst_xy, one_cfg)
                    else:
                        # expects dict with dest_logits/succ_logits
                        loss, logs = twohead_loss(out, y, dst_xy, two_cfg)

                if not torch.isfinite(loss):
                    print("\nNon-finite loss encountered.")
                    print("  shard:", fp)
                    print("  X finite:", torch.isfinite(X).all().item())
                    print("  X min/max:", float(X.min()), float(X.max()))
                    raise RuntimeError("Non-finite loss")

                scaler.scale(loss).backward()

                if args.grad_clip and args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                scaler.step(opt)
                scaler.update()

                b = X.size(0)
                shard_loss_sum += float(loss.detach().cpu()) * b
                shard_n += b
                epoch_loss_sum += float(loss.detach().cpu()) * b
                epoch_n += b
                global_step += 1

                shard_loss = shard_loss_sum / max(1, shard_n)
                postfix = {"loss": f"{shard_loss:.4f}", "step": global_step}
                # show key components if present
                for k in ("loss_single", "loss_heatmap", "loss_dest", "loss_succ"):
                    if k in logs:
                        postfix[k.replace("loss_", "")] = f"{logs[k]:.4f}"
                batch_pbar.set_postfix(postfix)

            shard_time = time.time() - shard_t0
            shard_ips = shard_n / max(1e-9, shard_time)
            shard_pbar.set_postfix({"sh_loss": f"{(shard_loss_sum/max(1,shard_n)):.4f}", "ips": f"{shard_ips:.1f}"})

        train_loss = epoch_loss_sum / max(1, epoch_n)
        epoch_time = time.time() - epoch_t0
        train_ips = epoch_n / max(1e-9, epoch_time)

        val_loss = float("nan")
        if len(val_feat) > 0:
            val_loss = evaluate(model, val_feat, val_targ, device,
                                mode=args.mode, H=H, W=W, swap_xy=args.swap_xy,
                                batch_size=args.batch_size, num_workers=args.num_workers,
                                one_cfg=one_cfg, two_cfg=two_cfg)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | items={epoch_n} | ips={train_ips:.1f} | time={epoch_time:.1f}s")

        # Save best (based on val if exists, else train)
        score = val_loss if len(val_feat) > 0 else train_loss
        if score < best_val:
            best_val = score
            if args.save_path:
                ckpt = {
                    "epoch": epoch,
                    "model": args.model,
                    "mode": args.mode,
                    "in_channels": in_channels,
                    "H": H, "W": W,
                    "channels": channels,
                    "state_dict": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "one_cfg": one_cfg.__dict__,
                    "two_cfg": two_cfg.__dict__,
                }
                torch.save(ckpt, args.save_path)
                print(f"  saved best -> {args.save_path} (score={best_val:.6f})")

    print("Done.")


if __name__ == "__main__":
    main()
