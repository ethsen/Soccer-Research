# visualize_pass_map_new.py
import os
import json
import torch
import numpy as np

from utils.visualizer import SoccerVisualizer       # your class above

# --- Choose model here ---
MODEL_NAME = "better"   # "better" or "soccermap"

# If MODEL_NAME == "better", this file must exist in your repo:
#   from better_soccermap import BetterSoccerMap
# If MODEL_NAME == "soccermap":
#   from soccermap import soccermap_model

FEATURE_DIR = "data/soccer_shards"
TARGET_DIR  = "data/soccer_shards_targets"
CKPT_PATH   = "best_ckpt.pt"                  # can be empty/None to run random weights
EXAMPLE_IDX = np.random.randint(0,18000)                           # global index across ALL shards
COORDS_ARE_CENTERS = True                     # False if dst_xy are integer cell indices

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------
# Helpers: load by manifest
# ---------------------------

def _load_tensor_or_dict(path: str, keys=("X", "targets")):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        for k in keys:
            if k in obj:
                return obj[k]
        # fallback: first tensor-like value
        for v in obj.values():
            if torch.is_tensor(v):
                return v
        raise KeyError(f"No tensor found in dict at {path}. Keys={list(obj.keys())}")
    if torch.is_tensor(obj):
        return obj
    raise TypeError(f"Unsupported object in {path}: {type(obj)}")


def _ensure_cnhw(feats: torch.Tensor, H=105, W=68) -> torch.Tensor:
    # feats: (C,N,H,W) or (C,N,W,H) -> (C,N,H,W)
    assert feats.dim() == 4, f"feats must be 4D, got {feats.shape}"
    C, N, A, B = feats.shape
    if (A, B) == (H, W):
        return feats
    if (A, B) == (W, H):
        return feats.permute(0, 1, 3, 2).contiguous()
    raise ValueError(f"Unexpected feature spatial size {(A,B)}; expected {(H,W)} or {(W,H)}")


def _ensure_n3(targs: torch.Tensor) -> torch.Tensor:
    # targs: (N,3) or (3,N) -> (N,3)
    assert targs.dim() == 2, f"targets must be 2D, got {targs.shape}"
    if targs.shape[1] == 3:
        return targs
    if targs.shape[0] == 3:
        return targs.t().contiguous()
    raise ValueError(f"Unexpected target shape {targs.shape}; expected (N,3) or (3,N)")


def load_manifests():
    fpath = os.path.join(FEATURE_DIR, "manifest.json")
    tpath = os.path.join(TARGET_DIR,  "targets_manifest.json")
    if not os.path.exists(fpath) or not os.path.exists(tpath):
        raise FileNotFoundError("Missing manifest.json or targets_manifest.json")
    with open(fpath, "r") as f:
        fman = json.load(f)
    with open(tpath, "r") as f:
        tman = json.load(f)
    return fman, tman


def _resolve_paths(man: dict, root: str):
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


def find_shard_for_global_index(fman: dict, feat_paths, targ_paths, global_idx: int):
    # global_idx in [0, total_frames)
    running = 0
    for si, sh in enumerate(fman["shards"]):
        n = int(sh["frames"])
        if global_idx < running + n:
            local = global_idx - running
            return si, local, feat_paths[si], targ_paths[si]
        running += n
    raise IndexError(f"EXAMPLE_IDX={global_idx} out of range total={running}")


def load_example_by_global_index(global_idx: int):
    fman, tman = load_manifests()
    feat_paths = _resolve_paths(fman, FEATURE_DIR)
    targ_paths = _resolve_paths(tman, TARGET_DIR)
    assert len(feat_paths) == len(targ_paths), "Feature/target shard count mismatch."

    si, local_i, fp, tp = find_shard_for_global_index(fman, feat_paths, targ_paths, global_idx)

    feats = _load_tensor_or_dict(fp, keys=("X",))
    targs = _load_tensor_or_dict(tp, keys=("targets", "y", "Y"))

    feats = _ensure_cnhw(feats, H=int(fman.get("H", 105)), W=int(fman.get("W", 68)))
    targs = _ensure_n3(targs)

    C, N, H, W = feats.shape
    if local_i < 0 or local_i >= N:
        raise IndexError(f"Local idx {local_i} out of range for shard {si} (N={N}).")

    x_c_hw = feats[:, local_i, :, :]   # (C,H,W)
    dst_xy = targs[local_i, :2].clone()
    y      = targs[local_i, 2].clone()

    return x_c_hw, dst_xy, y, (si, local_i, fp, tp)


# ---------------------------
# Model loading
# ---------------------------

def build_model(in_channels: int):
    if MODEL_NAME == "better":
        from models.passmap import BetterSoccerMap
        return BetterSoccerMap(in_channels=in_channels, base=64, blocks_per_stage=2, dropout=0.0)
    elif MODEL_NAME == "soccermap":
        from models.soccermap import soccermap_model
        return soccermap_model(in_channels=in_channels, base=32)
    else:
        raise ValueError(f"Unknown MODEL_NAME={MODEL_NAME}")


def load_weights(model: torch.nn.Module, ckpt_path: str):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[warn] ckpt not found: {ckpt_path} (running random weights)")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    # Support common key names
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            sd = ckpt["model_state_dict"]
        else:
            # maybe ckpt itself is a state_dict
            sd = ckpt
    else:
        sd = ckpt

    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Loaded ckpt: {ckpt_path}")
    if missing:
        print(f"[warn] missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] unexpected keys: {len(unexpected)}")


def extract_logits(model_out):
    # BetterSoccerMap returns logits tensor; soccermap_model returns (probs, logits, aux)
    if torch.is_tensor(model_out):
        return model_out
    if isinstance(model_out, (tuple, list)):
        # prefer a (B,1,H,W) tensor
        for t in model_out:
            if torch.is_tensor(t) and t.dim() == 4 and t.size(1) == 1:
                return t
        # fallback second item
        if len(model_out) >= 2 and torch.is_tensor(model_out[1]):
            return model_out[1]
    if isinstance(model_out, dict) and "logits" in model_out:
        return model_out["logits"]
    raise TypeError("Could not extract logits from model output")


# ---------------------------
# Main
# ---------------------------

def main():
    x_c_hw, dst_xy, y, meta = load_example_by_global_index(EXAMPLE_IDX)
    shard_i, local_i, fp, tp = meta
    print(f"Example global={EXAMPLE_IDX} -> shard={shard_i} local={local_i}")
    print(f"  feats: {fp}")
    print(f"  targs: {tp}")

    # Build model & (optional) load weights
    model = build_model(in_channels=x_c_hw.shape[0]).to(device).float()
    load_weights(model, CKPT_PATH)

    model.eval()
    with torch.no_grad():
        X = x_c_hw.unsqueeze(0).to(device, dtype=torch.float32)  # (1,C,105,68)
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        out = model(X)
        logits = extract_logits(out)
        probs = torch.sigmoid(logits)  # (1,1,H,W) for both models
        heat = probs[0, 0].detach().cpu().numpy()                # (105,68)

    # Destination coordinates: use as-is if already centers
    dst_x = float(dst_xy[0].item())
    dst_y = float(dst_xy[1].item())
    if not COORDS_ARE_CENTERS:
        dst_x += 0.5
        dst_y += 0.5

    # ------------------------------
    # PLOTTING (USING YOUR SoccerVisualizer)
    # ------------------------------
    vis = SoccerVisualizer(pitch_length=105, pitch_width=68, layout="x_rows")

    # occupancy maps
    in_pos  = (x_c_hw[3] > 0).to(torch.float32)   # (105,68)
    out_pos = (x_c_hw[4] > 0).to(torch.float32)   # (105,68)

    # heatmap from model probs: expect (105,68) in x_rows
    heat_t = torch.as_tensor(heat, dtype=torch.float32)

    fig, ax, _ = vis.plot_state(
        in_possession=in_pos,
        out_possession=out_pos,
        heatmap=heat_t,
        cmap="hot",
        heatmap_kwargs=dict(alpha=0.9),
        add_colorbar=True,
    )

    # --- start location from dist_to_ball map: argmin gives (x_idx, y_idx) in x_rows layout
    ball_dist = x_c_hw[0].to(torch.float32)
    flat_idx = torch.argmin(ball_dist)
    x_idx = (flat_idx // ball_dist.shape[1]).item()
    y_idx = (flat_idx %  ball_dist.shape[1]).item()
    bx, by = float(x_idx) + 0.5, float(y_idx) + 0.5  # cell centers

    # --- end location (dst) already in grid coords; optionally convert to center
    dst_x = float(dst_xy[0].item())
    dst_y = float(dst_xy[1].item())
    if not COORDS_ARE_CENTERS:
        dst_x += 0.5
        dst_y += 0.5

    # markers
    ax.scatter([bx], [by], c="black", s=30, marker="o", zorder=6, linewidths=0.5, label="Start Location")
    ax.scatter([dst_x], [dst_y], c="red",   s=30, marker="o", zorder=6, linewidths=0.5, label="End Location")

    ax.set_title(f"Pass {'✓ success' if int(y.item())==1 else '✗ fail'}  •  dest=({dst_x:.1f}, {dst_y:.1f})")

    out_path = f"viz_shard{shard_i}_idx{EXAMPLE_IDX}_{MODEL_NAME}.png"
    fig.tight_layout()
    fig.legend()
    fig.savefig(out_path, dpi=150)
    print(f"saved → {out_path}")



if __name__ == "__main__":
    main()
