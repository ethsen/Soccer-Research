# new_test.py (memmap split aware)
import os
import json
import argparse
import torch
import numpy as np
from collections import OrderedDict

from utils.visualizer import SoccerVisualizer  # your class


# ---------------------------
# Memmap dataset utilities
# ---------------------------

class MemmapShard:
    def __init__(self, root_dir: str, x_name: str, t_name: str, n: int, C: int, H: int, W: int):
        self.root_dir = root_dir
        self.n = int(n)
        self.C, self.H, self.W = int(C), int(H), int(W)

        x_path = os.path.join(root_dir, x_name)
        t_path = os.path.join(root_dir, t_name)

        self.X = np.memmap(x_path, mode="r", dtype=np.float16, shape=(self.n, self.C, self.H, self.W))
        self.T = np.memmap(t_path, mode="r", dtype=np.float32, shape=(self.n, 3))


class MemmapManifest:
    """
    Provides:
      - open shard by id (LRU)
      - map global index -> (shard_id, local_i)
      - load one sample (x_chw, dst_xy, y)
    """
    def __init__(self, root_dir: str, cache_size: int = 2):
        self.root_dir = root_dir
        self.cache_size = int(cache_size)

        man_path = os.path.join(root_dir, "manifest.json")
        with open(man_path, "r") as f:
            man = json.load(f)

        assert man.get("format") == "memmap_v1", f"Expected memmap_v1 manifest, got {man.get('format')}"
        self.C = int(man["C"])
        self.H = int(man["H"])
        self.W = int(man["W"])
        self.channels = man.get("channels", [])
        self.shards = man["shards"]

        # prefix sums: starts[k] is global start index of shard k
        self.starts = []
        cur = 0
        for s in self.shards:
            self.starts.append(cur)
            cur += int(s["n"])
        self.total = cur

        self._cache = OrderedDict()  # shard_id -> MemmapShard

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

    def locate(self, k: int):
        k = int(k)
        if k < 0 or k >= self.total:
            raise IndexError(f"Global index {k} out of range (N={self.total}).")

        lo, hi = 0, len(self.starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.starts[mid]
            end = self.starts[mid + 1] if mid + 1 < len(self.starts) else self.total
            if start <= k < end:
                return mid, k - start
            if k < start:
                hi = mid - 1
            else:
                lo = mid + 1
        raise RuntimeError("locate() failed unexpectedly")

    def load_by_shard_local(self, shard_id: int, local_i: int, swap_xy: bool = False):
        shard_id = int(shard_id)
        local_i = int(local_i)

        shard = self._open_shard(shard_id)
        if local_i < 0 or local_i >= shard.n:
            raise IndexError(f"local_i {local_i} out of range for shard {shard_id} (n={shard.n}).")

        # IMPORTANT: memmap is read-only -> make a writable copy to avoid PyTorch warning
        x_np = np.array(shard.X[local_i], copy=True)  # (C,H,W) float16
        x = torch.from_numpy(x_np).to(torch.float32)

        t = shard.T[local_i]  # (3,)
        dst_xy = torch.tensor(t[:2], dtype=torch.long)
        if swap_xy:
            dst_xy = dst_xy[[1, 0]]
        y = torch.tensor(t[2], dtype=torch.float32)

        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        return x, dst_xy, y


# ---------------------------
# Model loading (two-head)
# ---------------------------

def build_model(model_name: str, in_channels: int):
    if model_name == "better2head":
        from models.passmap import BetterSoccerMap2Head
        return BetterSoccerMap2Head(in_channels=in_channels, base=64, blocks_per_stage=2, dropout=0.0)
    raise ValueError(f"Unknown model_name={model_name}")


def load_weights(model: torch.nn.Module, ckpt_path: str, device):
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded ckpt: {ckpt_path}")
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")


def compute_twohead_maps(out: dict):
    dest = out["dest_logits"]
    succ = out["succ_logits"]

    if dest.dim() == 4:
        dest = dest[:, 0]
    if succ.dim() == 4:
        succ = succ[:, 0]

    B, H, W = dest.shape
    dest_flat = dest.view(B, -1)
    dest_probs = torch.softmax(dest_flat, dim=1).view(B, H, W)
    succ_probs = torch.sigmoid(succ)
    completion = dest_probs * succ_probs

    return (
        dest_probs[0].detach().cpu().numpy(),
        succ_probs[0].detach().cpu().numpy(),
        completion[0].detach().cpu().numpy(),
    )


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_root", type=str, required=True,
                    help="Root containing train/ and val/ folders with memmap manifest.json")
    ap.add_argument("--split_set", type=str, choices=["train", "val"], default="val")
    ap.add_argument("--cache_size", type=int, default=2)

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="better2head")

    # choose sample
    ap.add_argument("--example_k", type=int, default=0, help="Global index within split_set (ignored if --random or shard/local provided)")
    ap.add_argument("--random", action="store_true")
    ap.add_argument("--shard_id", type=int, default=None)
    ap.add_argument("--local_i", type=int, default=None)

    ap.add_argument("--coords_are_centers", action="store_true")
    ap.add_argument("--swap_xy", action="store_true")

    ap.add_argument("--out_dir", type=str, default="viz")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    split_root = os.path.join(args.data_root, args.split_set)
    mm = MemmapManifest(split_root, cache_size=args.cache_size)
    print(f"[data] split={args.split_set} N={mm.total} | C,H,W={mm.C},{mm.H},{mm.W}")

    # choose sample
    if args.shard_id is not None or args.local_i is not None:
        if args.shard_id is None or args.local_i is None:
            raise ValueError("If using --shard_id or --local_i, you must provide BOTH.")
        shard_id = int(args.shard_id)
        local_i = int(args.local_i)
        k = None
    else:
        if mm.total == 0:
            raise RuntimeError(f"No samples found in {split_root}")
        if args.random:
            k = int(np.random.randint(0, mm.total))
        else:
            k = int(np.clip(args.example_k, 0, mm.total - 1))
        shard_id, local_i = mm.locate(k)

    x_c_hw, dst_xy, y = mm.load_by_shard_local(shard_id, local_i, swap_xy=args.swap_xy)

    if k is not None:
        print(f"[sample] split={args.split_set} k={k} -> shard={shard_id} local={local_i}")
    else:
        print(f"[sample] split={args.split_set} shard={shard_id} local={local_i}")

    # model
    model = build_model(args.model, in_channels=x_c_hw.shape[0]).to(device).float()
    load_weights(model, args.ckpt, device)
    model.eval()

    with torch.no_grad():
        X = x_c_hw.unsqueeze(0).to(device, dtype=torch.float32)
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        out = model(X)
        if not isinstance(out, dict) or ("dest_logits" not in out) or ("succ_logits" not in out):
            raise TypeError("Expected output dict with keys: dest_logits, succ_logits")
        dest_map, succ_map, comp_map = compute_twohead_maps(out)

    # coords
    dst_x = float(dst_xy[0].item())
    dst_y = float(dst_xy[1].item())
    if not args.coords_are_centers:
        dst_x += 0.5
        dst_y += 0.5

    # start location from dist_to_ball argmin
    ball_dist = x_c_hw[0].to(torch.float32)
    flat_idx = torch.argmin(ball_dist)
    x_idx = (flat_idx // ball_dist.shape[1]).item()
    y_idx = (flat_idx %  ball_dist.shape[1]).item()
    bx, by = float(x_idx) + 0.5, float(y_idx) + 0.5

    # occupancy maps (your channel indices)
    in_pos  = (x_c_hw[3] > 0).to(torch.float32)
    out_pos = (x_c_hw[4] > 0).to(torch.float32)

    vis = SoccerVisualizer(pitch_length=mm.H, pitch_width=mm.W, layout="x_rows")

    def _plot(name: str, heat_np: np.ndarray, title: str):
        heat_t = torch.as_tensor(heat_np, dtype=torch.float32)
        fig, ax, _ = vis.plot_state(
            in_possession=in_pos,
            out_possession=out_pos,
            heatmap=heat_t,
            cmap="Blues",
            heatmap_kwargs=dict(alpha=0.9),
            add_colorbar=True,
        )
        ax.scatter([bx], [by], c="black", s=30, marker="o", zorder=6, linewidths=0.5, label="Start")
        ax.scatter([dst_x], [dst_y], c="red",   s=30, marker="o", zorder=6, linewidths=0.5, label="End")
        ax.set_title(title)
        fig.tight_layout()
        fig.legend()

        tag = f"split{args.split_set}_sh{shard_id}_i{local_i}"
        if k is not None:
            tag = f"split{args.split_set}_k{k}_sh{shard_id}_i{local_i}"
        out_path = os.path.join(args.out_dir, f"{name}_{tag}.png")
        fig.savefig(out_path, dpi=150)
        print(f"saved → {out_path}")

    ok = "✓" if int(y.item()) == 1 else "✗"
    _plot("dest", dest_map, f"Destination P(cell)  | pass {ok}  | end=({dst_x:.1f},{dst_y:.1f})")
    _plot("succ", succ_map, f"Success P(complete | cell)  | pass {ok}")
    _plot("complete", comp_map, f"Completion surface P(cell & complete)  | pass {ok}")


if __name__ == "__main__":
    main()
