import os
import json
import argparse
import torch
import numpy as np

from utils.visualizer import SoccerVisualizer  # your class


# ---------------------------
# Basic IO helpers
# ---------------------------

def load_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def resolve_manifest_paths(man: dict, root: str):
    paths = []
    for sh in man["shards"]:
        p = sh["path"]
        if not os.path.isabs(p):
            # assume manifest paths are relative to root
            p = os.path.join(root, p)
        paths.append(p)
    return paths

def load_tensor_or_dict(path: str, keys=("X", "targets")):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        # fallback: first tensor value
        for v in obj.values():
            if torch.is_tensor(v):
                return v
        raise KeyError(f"No tensor found in dict at {path}. Keys={list(obj.keys())}")
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Unsupported object in {path}: {type(obj)}")

def ensure_cnhw(feats: torch.Tensor, H=105, W=68) -> torch.Tensor:
    assert feats.ndim == 4, f"feats must be 4D, got {feats.shape}"
    C, N, A, B = feats.shape
    if (A, B) == (H, W):
        return feats
    if (A, B) == (W, H):
        return feats.permute(0, 1, 3, 2).contiguous()
    raise ValueError(f"Unexpected spatial size {(A,B)} expected {(H,W)}")

def ensure_n3(targs: torch.Tensor) -> torch.Tensor:
    assert targs.ndim == 2, f"targets must be 2D, got {targs.shape}"
    if targs.shape[1] == 3:
        return targs
    if targs.shape[0] == 3:
        return targs.t().contiguous()
    raise ValueError(f"Unexpected target shape {targs.shape} expected (N,3) or (3,N)")


# ---------------------------
# Model helpers
# ---------------------------

def build_model(model_name: str, in_channels: int):
    if model_name == "better2head":
        from models.bettermap import BetterSoccerMap2Head
        return BetterSoccerMap2Head(in_channels=in_channels, base=64, blocks_per_stage=2, dropout=0.0)
    raise ValueError(f"Unknown model: {model_name}")

def load_weights(model: torch.nn.Module, ckpt_path: str, device):
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

    if dest.ndim == 4:
        dest = dest[:, 0]  # (B,H,W)
    if succ.ndim == 4:
        succ = succ[:, 0]  # (B,H,W)

    B, H, W = dest.shape

    dest_probs = torch.softmax(dest.view(B, -1), dim=1).view(B, H, W)
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

    ap.add_argument("--features_dir", type=str, required=True)
    ap.add_argument("--targets_dir", type=str, required=True)

    ap.add_argument("--shard_id", type=int, required=True, help="Which shard to load: 0,1,2,...")
    ap.add_argument("--local_i", type=int, default=0, help="Which sample within shard (ignored if --random)")
    ap.add_argument("--random", action="store_true", help="Pick a random sample within the shard")

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--model", type=str, default="better2head")

    ap.add_argument("--swap_xy", action="store_true")
    ap.add_argument("--coords_are_centers", action="store_true")

    ap.add_argument("--out_dir", type=str, default="viz_by_shard")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load manifests & resolve shard paths
    fman = load_json(os.path.join(args.features_dir, "manifest.json"))
    tman = load_json(os.path.join(args.targets_dir, "targets_manifest.json"))
    feat_paths = resolve_manifest_paths(fman, args.features_dir)
    targ_paths = resolve_manifest_paths(tman, args.targets_dir)

    if len(feat_paths) != len(targ_paths):
        raise RuntimeError(f"Shard mismatch feats={len(feat_paths)} targets={len(targ_paths)}")

    H = int(fman.get("H", 105))
    W = int(fman.get("W", 68))

    sid = int(args.shard_id)
    if sid < 0 or sid >= len(feat_paths):
        raise IndexError(f"--shard_id {sid} out of range [0, {len(feat_paths)-1}]")

    fp = feat_paths[sid]
    tp = targ_paths[sid]

    X = ensure_cnhw(load_tensor_or_dict(fp, keys=("X",)), H=H, W=W)
    T = ensure_n3(load_tensor_or_dict(tp, keys=("targets", "y", "Y")))

    if X.shape[1] != T.shape[0]:
        raise RuntimeError(f"N mismatch in shard {sid}: X N={X.shape[1]} vs T N={T.shape[0]}")

    N = X.shape[1]
    if args.random:
        local_i = int(torch.randint(low=0, high=N, size=(1,)).item())
    else:
        local_i = int(np.clip(args.local_i, 0, N - 1))

    x_c_hw = X[:, local_i].to(torch.float32)
    dst_xy = T[local_i, :2].clone()
    y = T[local_i, 2].clone()

    print(f"[shard] id={sid} sample={local_i}/{N-1}")
    print(f"  feats: {fp}")
    print(f"  targs: {tp}")
    print(f"  label(success)={float(y):.0f}, dst_xy={dst_xy.tolist()}")

    # Build/load model
    model = build_model(args.model, in_channels=x_c_hw.shape[0]).to(device).float()
    load_weights(model, args.ckpt, device)
    model.eval()

    with torch.no_grad():
        xb = x_c_hw.unsqueeze(0).to(device)
        xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)

        out = model(xb)
        if not isinstance(out, dict) or "dest_logits" not in out or "succ_logits" not in out:
            raise TypeError("Expected output dict with keys: dest_logits, succ_logits")

        dest_map, succ_map, comp_map = compute_twohead_maps(out)

    # coords
    dst_x = float(dst_xy[0].item())
    dst_y = float(dst_xy[1].item())
    if args.swap_xy:
        dst_x, dst_y = dst_y, dst_x
    if not args.coords_are_centers:
        dst_x += 0.5
        dst_y += 0.5

    # infer ball start from channel 0 distance map (same assumption as before)
    ball_dist = x_c_hw[0]
    flat_idx = torch.argmin(ball_dist)
    bx = float((flat_idx // ball_dist.shape[1]).item()) + 0.5
    by = float((flat_idx %  ball_dist.shape[1]).item()) + 0.5

    # NOTE: adjust these if your channels differ
    in_pos  = (x_c_hw[3] > 0).to(torch.float32)
    out_pos = (x_c_hw[4] > 0).to(torch.float32)

    vis = SoccerVisualizer(pitch_length=H, pitch_width=W, layout="x_rows")

    def plot(name: str, heat_np: np.ndarray, title: str):
        heat_t = torch.as_tensor(heat_np, dtype=torch.float32)

        fig, ax, _ = vis.plot_state(
            in_possession=in_pos,
            out_possession=out_pos,
            heatmap=heat_t,
            cmap="Blues",
            heatmap_kwargs=dict(alpha=0.9),
            add_colorbar=True,
        )

        ax.scatter([bx], [by], c="black", s=30, marker="o", zorder=6, label="Start")
        ax.scatter([dst_x], [dst_y], c="red",   s=30, marker="o", zorder=6, label="End")

        ok = "✓" if float(y) > 0.5 else "✗"
        ax.set_title(f"{title} | pass {ok} | shard={sid} i={local_i}")
        fig.tight_layout()
        fig.legend()

        out_path = os.path.join(args.out_dir, f"{name}_sh{sid:04d}_i{local_i:06d}.png")
        fig.savefig(out_path, dpi=150)
        print(f"saved → {out_path}")

    plot("dest", dest_map, "Destination P(cell)")
    plot("succ", succ_map, "Success P(complete | cell)")
    plot("complete", comp_map, "Completion surface P(cell & complete)")


if __name__ == "__main__":
    main()
