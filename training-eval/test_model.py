# visualize_pass_map.py
import os
import torch
import numpy as np

from models.soccermap import soccermap_model          # your model module
from models.passmap import BetterSoccerMap2Head
from utils.visualizer import SoccerVisualizer       # your class above

FEATURE_DIR = "data/soccer_shards"
TARGET_DIR  = "data/soccer_shards_targets"
CKPT_PATH   = "runs/simpe_twohead/best_ckpt.pt"             # optional
EXAMPLE_IDX = 18828 #np.random.randint(0,20000-1)
COORDS_ARE_CENTERS = True    # <-- set False if your dst_xy are integer cell indices (0..104, 0..67)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_first_shard():
    fpaths = sorted([os.path.join(FEATURE_DIR, f) for f in os.listdir(FEATURE_DIR) if f.endswith(".pt")])
    tpaths = sorted([os.path.join(TARGET_DIR,  f) for f in os.listdir(TARGET_DIR)  if f.endswith(".pt")])
    if not fpaths or not tpaths:
        raise FileNotFoundError("No shard .pt files found.")

    fobj = torch.load(fpaths[1], map_location="cpu")
    tobj = torch.load(tpaths[1], map_location="cpu")

    feats = fobj["X"] if isinstance(fobj, dict) else fobj
    targs = tobj["targets"] if isinstance(tobj, dict) else tobj

    if targs.dim() == 2 and targs.shape[0] == 3 and targs.shape[1] != 3:
        targs = targs.t().contiguous()
    assert feats.dim() == 4 and targs.dim() == 2 and targs.shape[1] == 3
    return feats, targs

def get_example(feats_cnhw, targs_n3, i):
    C, N, H, W = feats_cnhw.shape
    if i < 0 or i >= N:
        raise IndexError(f"Index {i} out of range (N={N}).")

    x = feats_cnhw[:, i, :, :]        # (C, H, W) == (C, 105, 68)

    if targs_n3.dim() == 2 and targs_n3.shape[0] == 3 and targs_n3.shape[1] != 3:
        targs_n3 = targs_n3.t().contiguous()
    assert targs_n3.shape[1] == 3, f"targets should be (N,3), got {targs_n3.shape}"

    dst_xy = targs_n3[i, :2].clone()  # stored indices/coords
    y      = targs_n3[i, 2].clone()   # 0/1
    return x, dst_xy, y


def main():
    feats, targs = load_first_shard()
    x_c_hw, dst_xy, y = get_example(feats, targs, EXAMPLE_IDX)  # x: (C, 105, 68)

    # Build model & (optional) load weights
    #model = soccermap_model(in_channels=x_c_hw.shape[0], base=32).to(device).float()
    model = BetterSoccerMap2Head().to(device).float()
    if os.path.exists(CKPT_PATH):
        ckpt = torch.load(CKPT_PATH, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    with torch.no_grad():
        X = x_c_hw.unsqueeze(0).to(device, dtype=torch.float32)  # (1, C, 105, 68)
        probs, fused_logits, _ = model(X)                        # (1,1,105,68)
        heat = probs[0, 0].detach().cpu().numpy()                # (105, 68), rows=x, cols=y

    # Destination coordinates: use as-is if already centers
    dst_x = float(dst_xy[0].item())
    dst_y = float(dst_xy[1].item())
    if not COORDS_ARE_CENTERS:
        # if you stored integer cell indices (0..104, 0..67), convert to centers
        dst_x += 0.5
        dst_y += 0.5

    # Plot heatmap + destination marker
    vis = SoccerVisualizer(pitch_length=105, pitch_width=68, layout="x_rows")
    # no player scatters here; just the map
    in_pos  = (x_c_hw[3] > 0).to(torch.float32)   # (105, 68)
    out_pos = (x_c_hw[4] > 0).to(torch.float32)   # (105, 68)
    ball_dist = (x_c_hw[0]).to(torch.float32)   # note: >0, not ==0

    fig, ax, _ = vis.plot_state(
        in_possession=in_pos,   # empty overlays
        out_possession=out_pos,
        heatmap=heat,
        cmap="hot",
        heatmap_kwargs=dict(alpha=0.9),
        add_colorbar=True,
        draw=False,
    )

    # Overlay actual pass destination and outcome
    import matplotlib.pyplot as plt

    color = "lime" if int(y.item()) == 1 else "red"
    marker = "X"
    flat_idx = torch.argmin(ball_dist)
    yy = (flat_idx // ball_dist.shape[1]).item()
    x = (flat_idx %  ball_dist.shape[1]).item()
    bx, by = float(yy), float(x)
    ax.scatter([bx], [by], c="black", s=30, marker="o", zorder=5, linewidths=0.5,label = 'Start Location')
    ax.scatter([dst_x], [dst_y], c="red", s=30, marker="o", zorder=5, linewidths=0.5,label = 'End Location')


    ax.set_title(f"Pass {'✓ success' if int(y.item())==1 else '✗ fail'}  •  dest=({dst_x:.1f}, {dst_y:.1f})")
    out_path = f"viz_shard0_idx{EXAMPLE_IDX}_v2.png"
    fig.tight_layout()
    fig.legend()
    fig.savefig(out_path, dpi=150)
    print(f"saved → {out_path}")

if __name__ == "__main__":
    main()
