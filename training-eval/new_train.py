import os, json, math, random, argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ----------------------------
# Utilities: loading + shapes
# ----------------------------

def _load_tensor_or_dict(path: str, key_candidates: Tuple[str, ...]) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in key_candidates:
            if k in obj:
                return obj[k]
        # fallback: if dict has only one tensor value, take it
        for v in obj.values():
            if torch.is_tensor(v):
                return v
        raise KeyError(f"No keys {key_candidates} found in dict at {path}. Keys={list(obj.keys())}")
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Unsupported object in {path}: {type(obj)}")


def _ensure_cnhw(feats: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    feats: (C,N,H,W) or (C,N,W,H) -> return (C,N,H,W)
    """
    assert feats.dim() == 4, f"feats must be 4D, got {feats.shape}"
    C, N, A, B = feats.shape
    if (A, B) == (H, W):
        return feats
    if (A, B) == (W, H):
        return feats.permute(0, 1, 3, 2).contiguous()
    raise ValueError(f"Unexpected feature spatial size {(A, B)}; expected {(H, W)} or {(W, H)}")


def _ensure_n3(targs: torch.Tensor) -> torch.Tensor:
    """
    targs: (N,3) or (3,N) -> return (N,3)
    """
    assert targs.dim() == 2, f"targets must be 2D, got {targs.shape}"
    if targs.shape[1] == 3:
        return targs
    if targs.shape[0] == 3:
        return targs.t().contiguous()
    raise ValueError(f"Unexpected target shape {targs.shape}; expected (N,3) or (3,N)")


# ----------------------------
# Dataset for a single shard
# ----------------------------

class OneShardDataset(Dataset):
    """
    Holds a single shard in memory:
      feats: (C,N,H,W)
      targs: (N,3) with [x_idx, y_idx, outcome]
    """
    def __init__(self, feats: torch.Tensor, targs: torch.Tensor, *, swap_xy: bool = False):
        super().__init__()
        self.feats = feats
        self.targs = targs
        self.swap_xy = swap_xy
        assert feats.shape[1] == targs.shape[0], f"N mismatch feats N={feats.shape[1]} vs targs N={targs.shape[0]}"

    def __len__(self) -> int:
        return self.targs.shape[0]

    def __getitem__(self, i: int):
        x = self.feats[:, i]               # (C,H,W)
        xy = self.targs[i, :2].long()      # (2,)
        y  = self.targs[i, 2].float()      # scalar
        if self.swap_xy:
            xy = xy[[1, 0]]
        return x, xy, y


def collate_batch(batch):
    xs, xys, ys = zip(*batch)
    X = torch.stack(xs, dim=0).to(torch.float32)    # (B,C,H,W)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    dst_xy = torch.stack(xys, dim=0).to(torch.long) # (B,2)
    y = torch.as_tensor(ys, dtype=torch.float32)    # (B,)
    y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return X, dst_xy, y


# ----------------------------
# Loss: single pixel + dense heatmap
# ----------------------------

@dataclass
class LossCfg:
    single_pixel_weight: float = 1.0
    heatmap_weight: float = 0.0
    heatmap_sigma: float = 2.0
    heatmap_clip: float = 1e-4


def gaussian_targets(dst_xy: torch.Tensor, H: int, W: int, sigma: float, dtype, device):
    # dst_xy: (B,2) in (x_idx, y_idx)
    B = dst_xy.shape[0]
    xs = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    x0 = dst_xy[:, 0].to(dtype).view(B, 1, 1)
    y0 = dst_xy[:, 1].to(dtype).view(B, 1, 1)
    inv = 1.0 / (2.0 * (sigma ** 2))
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    g = torch.exp(-d2 * inv)   # (B,H,W)
    return g.unsqueeze(1)      # (B,1,H,W)


def compute_loss(logits: torch.Tensor, y: torch.Tensor, dst_xy: torch.Tensor, cfg: LossCfg) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    logits: (B,1,H,W)
    y: (B,) in {0,1}
    dst_xy: (B,2) (x_idx, y_idx)
    """
    assert logits.dim() == 4 and logits.size(1) == 1, f"logits must be (B,1,H,W), got {logits.shape}"
    B, _, H, W = logits.shape

    x_idx = dst_xy[:, 0].clamp(0, H - 1)
    y_idx = dst_xy[:, 1].clamp(0, W - 1)

    total = logits.new_tensor(0.0)
    logs: Dict[str, float] = {}

    if cfg.single_pixel_weight > 0:
        pix = logits[torch.arange(B, device=logits.device), 0, x_idx, y_idx]
        l_single = F.binary_cross_entropy_with_logits(pix, y, reduction="mean")
        total = total + cfg.single_pixel_weight * l_single
        logs["loss_single"] = float(l_single.detach().cpu())

    if cfg.heatmap_weight > 0:
        g = gaussian_targets(dst_xy, H, W, cfg.heatmap_sigma, logits.dtype, logits.device)
        g = g * y.view(B, 1, 1, 1)
        if cfg.heatmap_clip and cfg.heatmap_clip > 0:
            g = g.clamp(cfg.heatmap_clip, 1.0 - cfg.heatmap_clip)
        l_heat = F.binary_cross_entropy_with_logits(logits, g, reduction="mean")
        total = total + cfg.heatmap_weight * l_heat
        logs["loss_heatmap"] = float(l_heat.detach().cpu())

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs


# ----------------------------
# Model loader (plug & play)
# ----------------------------

def build_model(model_name: str, in_channels: int) -> nn.Module:
    """
    Options:
      --model better      -> expects better_soccermap.py with BetterSoccerMap
      --model soccermap   -> expects soccermap.py with soccermap_model(in_channels=...)
      --model module:ClassName -> importable path
    """
    if model_name == "better":
        from models.passmap import BetterSoccerMap
        return BetterSoccerMap(in_channels=in_channels, base=64, blocks_per_stage=2, dropout=0.0)

    if model_name == "soccermap":
        from models.soccermap import soccermap_model
        return soccermap_model(in_channels=in_channels, base=32)

    # import path: "pkg.module:ClassOrFactory"
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


def extract_logits(model_out) -> torch.Tensor:
    """
    Make this robust to different model returns:
      - logits tensor
      - (probs, logits, aux)
      - dict with "logits"
    """
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, dict) and "logits" in model_out:
        return model_out["logits"]
    if isinstance(model_out, (tuple, list)):
        # common pattern: (probs, logits, aux)
        for item in model_out:
            if torch.is_tensor(item) and item.dim() == 4 and item.size(1) == 1:
                return item
        # fallback: second element
        if len(model_out) >= 2 and torch.is_tensor(model_out[1]):
            return model_out[1]
    raise TypeError("Could not extract logits from model output")


# ----------------------------
# Shard iteration / manifests
# ----------------------------

def load_manifests(features_dir: str, targets_dir: str):
    with open(os.path.join(features_dir, "manifest.json"), "r") as f:
        fman = json.load(f)
    with open(os.path.join(targets_dir, "targets_manifest.json"), "r") as f:
        tman = json.load(f)
    return fman, tman


def shard_paths_from_manifest(man: dict, root: str) -> List[str]:
    paths = []
    for sh in man["shards"]:
        p = sh["path"]

        # If path already contains the root directory name, don't re-join
        if not os.path.isabs(p):
            if os.path.basename(root) in p.split(os.sep):
                p = os.path.join(os.path.dirname(root), p)
            else:
                p = os.path.join(root, p)

        paths.append(p)
    return paths



def split_shards(paths: List[str], val_frac: float, seed: int):
    rng = random.Random(seed)
    idx = list(range(len(paths)))
    rng.shuffle(idx)
    n_val = int(round(val_frac * len(paths)))
    val_idx = set(idx[:n_val])
    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    return train, val


# ----------------------------
# Train / Eval
# ----------------------------

@torch.no_grad()
def run_eval(model: nn.Module,
             feat_paths: List[str],
             targ_paths: List[str],
             device: torch.device,
             *,
             H: int, W: int,
             swap_xy: bool,
             batch_size: int,
             num_workers: int,
             loss_cfg: LossCfg) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_n = 0

    for fp, tp in zip(feat_paths, targ_paths):
        feats = _load_tensor_or_dict(fp, ("X",))
        targs = _load_tensor_or_dict(tp, ("targets", "y", "Y"))

        feats = _ensure_cnhw(feats, H=H, W=W)
        targs = _ensure_n3(targs)

        ds = OneShardDataset(feats, targs, swap_xy=swap_xy)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_batch)

        for X, dst_xy, y in dl:
            X = X.to(device, non_blocking=True)
            dst_xy = dst_xy.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            out = model(X)
            logits = extract_logits(out)
            loss, _ = compute_loss(logits, y, dst_xy, loss_cfg)
            b = X.size(0)
            total_loss += float(loss.detach().cpu()) * b
            total_n += b

    return {"val_loss": total_loss / max(1, total_n)}


def train(args):
    from tqdm import tqdm
    import time

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    fman, tman = load_manifests(args.features_dir, args.targets_dir)
    H = int(fman.get("H", 105))
    W = int(fman.get("W", 68))
    channels = fman.get("channels", [])
    in_channels = len(channels) if channels else (args.in_channels or 14)
    if args.in_channels is not None:
        in_channels = args.in_channels

    feat_paths = shard_paths_from_manifest(fman, args.features_dir)
    targ_paths = shard_paths_from_manifest(tman, args.targets_dir)
    assert len(feat_paths) == len(targ_paths), "Feature and target shard counts differ—ensure they were built aligned."

    # Split by indices ONCE to preserve alignment
    idx = list(range(len(feat_paths)))
    random.Random(args.seed).shuffle(idx)
    n_val = int(round(args.val_frac * len(idx)))
    val_set = set(idx[:n_val])
    train_idx = [i for i in range(len(feat_paths)) if i not in val_set]
    val_idx   = [i for i in range(len(feat_paths)) if i in val_set]

    train_feat = [feat_paths[i] for i in train_idx]
    train_targ = [targ_paths[i] for i in train_idx]
    val_feat   = [feat_paths[i] for i in val_idx]
    val_targ   = [targ_paths[i] for i in val_idx]

    model = build_model(args.model, in_channels=in_channels).to(device)

    loss_cfg = LossCfg(
        single_pixel_weight=args.w_single,
        heatmap_weight=args.w_heatmap,
        heatmap_sigma=args.sigma,
        heatmap_clip=1e-4,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    use_amp = (device.type == "cuda") and args.amp
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val = float("inf")
    global_step = 0

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Shards: train={len(train_feat)} val={len(val_feat)} | in_channels={in_channels} | HxW={H}x{W}")
    print(f"Loss: w_single={args.w_single} w_heatmap={args.w_heatmap} sigma={args.sigma} | swap_xy={args.swap_xy}")
    if channels:
        print(f"Channels({len(channels)}): {channels}")

    for epoch in range(args.epochs):
        model.train()

        # Shuffle shard order per epoch
        order = list(range(len(train_feat)))
        random.Random(args.seed + epoch).shuffle(order)

        epoch_loss_sum = 0.0
        epoch_n = 0
        epoch_t0 = time.time()

        shard_pbar = tqdm(order, desc=f"Epoch {epoch:03d} [shards]", leave=False)

        for s_i in shard_pbar:
            fp = train_feat[s_i]
            tp = train_targ[s_i]

            # Load shard
            feats = _load_tensor_or_dict(fp, ("X",))
            targs = _load_tensor_or_dict(tp, ("targets", "y", "Y"))

            feats = _ensure_cnhw(feats, H=H, W=W)
            targs = _ensure_n3(targs)

            if feats.shape[0] != in_channels:
                raise RuntimeError(f"Shard {fp} has C={feats.shape[0]} but expected {in_channels}")
            if feats.shape[1] != targs.shape[0]:
                raise RuntimeError(f"N mismatch: feats N={feats.shape[1]} vs targs N={targs.shape[0]}")

            ds = OneShardDataset(feats, targs, swap_xy=args.swap_xy)
            dl = DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True,
                collate_fn=collate_batch,
                drop_last=False,
            )

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
                    logits = extract_logits(out)
                    loss, loss_logs = compute_loss(logits, y, dst_xy, loss_cfg)

                if not torch.isfinite(loss):
                    print("\nNon-finite loss encountered.")
                    print("  shard:", fp)
                    print("  X finite:", torch.isfinite(X).all().item())
                    print("  X min/max:", float(X.min()), float(X.max()))
                    raise RuntimeError("Non-finite loss")

                scaler.scale(loss).backward()

                if args.grad_clip > 0:
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

                # Live batch stats
                shard_loss = shard_loss_sum / max(1, shard_n)
                batch_pbar.set_postfix({
                    "loss": f"{shard_loss:.4f}",
                    "single": f"{loss_logs.get('loss_single', float('nan')):.4f}" if "loss_single" in loss_logs else "—",
                    "heat": f"{loss_logs.get('loss_heatmap', float('nan')):.4f}" if "loss_heatmap" in loss_logs else "—",
                    "step": global_step,
                })

            # End shard stats
            shard_time = time.time() - shard_t0
            shard_loss = shard_loss_sum / max(1, shard_n)
            shard_ips = shard_n / max(1e-9, shard_time)  # items/sec

            shard_pbar.set_postfix({
                "sh_loss": f"{shard_loss:.4f}",
                "ips": f"{shard_ips:.1f}",
            })

        train_loss = epoch_loss_sum / max(1, epoch_n)
        epoch_time = time.time() - epoch_t0
        train_ips = epoch_n / max(1e-9, epoch_time)

        # Validation
        val_loss = float("nan")
        if len(val_feat) > 0:
            val_metrics = run_eval(
                model, val_feat, val_targ, device,
                H=H, W=W,
                swap_xy=args.swap_xy,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                loss_cfg=loss_cfg
            )
            val_loss = float(val_metrics["val_loss"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
            f"items={epoch_n} | ips={train_ips:.1f} | time={epoch_time:.1f}s"
        )

        # Save best
        if len(val_feat) > 0 and val_loss < best_val:
            best_val = val_loss
            if args.save_path:
                ckpt = {
                    "epoch": epoch,
                    "model": args.model,
                    "in_channels": in_channels,
                    "state_dict": model.state_dict(),
                    "opt_state": opt.state_dict(),
                    "loss_cfg": vars(loss_cfg),
                    "H": H, "W": W,
                    "channels": channels,
                }
                torch.save(ckpt, args.save_path)
                print(f"  saved best -> {args.save_path} (val_loss={best_val:.6f})")

    print("Done.")



def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", type=str, required=True)
    p.add_argument("--targets_dir", type=str, required=True)
    p.add_argument("--model", type=str, default="better",
                   help="better | soccermap | module:ClassOrFactory")
    p.add_argument("--in_channels", type=int, default=14, help="override channels count")
    p.add_argument("--swap_xy", action="store_true", help="swap target indices (use if your saved targets are flipped)")

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

    # loss knobs
    p.add_argument("--w_single", type=float, default=1.0)
    p.add_argument("--w_heatmap", type=float, default=1.0)
    p.add_argument("--sigma", type=float, default=2.0)

    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
