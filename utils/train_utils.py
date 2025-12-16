import matplotlib.pyplot as plt
import numpy as np
import torch


@torch.no_grad()
def compute_twohead_maps_single(dest_logits_1hw: torch.Tensor, succ_logits_1hw: torch.Tensor):
    """
    Inputs:
      dest_logits_1hw: (1,H,W)
      succ_logits_1hw: (1,H,W)
    Returns:
      dest_probs(H,W), succ_probs(H,W), comp_map(H,W)
    """
    H, W = dest_logits_1hw.shape[-2], dest_logits_1hw.shape[-1]
    d = dest_logits_1hw.view(1, -1)
    dest_probs = torch.softmax(d, dim=1).view(H, W)
    succ_probs = torch.sigmoid(succ_logits_1hw[0])
    comp_map = dest_probs * succ_probs
    return dest_probs, succ_probs, comp_map


def pick_fixed_val_indices(val_ds, n: int, seed: int):
    n = int(n)
    rng = np.random.RandomState(int(seed))
    idxs = rng.choice(len(val_ds), size=min(n, len(val_ds)), replace=False).tolist()
    return [int(i) for i in idxs]


@torch.no_grad()
def log_fixed_val_grid_to_tensorboard(
    *,
    model: torch.nn.Module,
    val_ds,
    static,
    device: torch.device,
    writer,
    epoch: int,
    fixed_idxs: list[int],
    tag: str = "viz/fixed_val_grid",
    coords_are_centers: bool = False,
    ch_dist2ball: int = 0,
    ch_in_pos: int = 3,
    ch_out_pos: int = 4,
):
    """
    Logs a (N rows x 3 cols) figure:
      col0: P(dest|s)
      col1: P(complete|s,cell)
      col2: P(dest & complete|s)
    using the SAME fixed validation examples every epoch.
    """
    model.eval()

    N = len(fixed_idxs)
    if N == 0:
        return

    # Create grid figure: N rows, 3 cols
    fig, axes = plt.subplots(N, 3, figsize=(18, 4 * N), constrained_layout=True)
    if N == 1:
        axes = np.expand_dims(axes, axis=0)  # make it (1,3)

    from utils.visualizer import SoccerVisualizer

    for r, ds_idx in enumerate(fixed_idxs):
        x_chw, dst_xy, y = val_ds[ds_idx]  # x: (C,H,W) on CPU
        # move to device + add batch dim
        X = x_chw.unsqueeze(0).to(device, non_blocking=True)
        dst_xy = dst_xy.unsqueeze(0).to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # append static
        X = static.concat_to(X, dim=1)
        X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        out = model(X)
        dest_logits = out["dest_logits"][0, 0]  # (H,W)
        succ_logits = out["succ_logits"][0, 0]  # (H,W)

        dest_map, succ_map, comp_map = compute_twohead_maps_single(
            dest_logits.unsqueeze(0), succ_logits.unsqueeze(0)
        )

        # markers
        dst_xy_cpu = dst_xy[0].detach().cpu()
        dst_x = float(dst_xy_cpu[0].item())
        dst_y = float(dst_xy_cpu[1].item())
        if not coords_are_centers:
            dst_x += 0.5
            dst_y += 0.5

        x_cpu = x_chw.detach().cpu()
        ball_dist = x_cpu[ch_dist2ball]
        flat_idx = torch.argmin(ball_dist)
        bx = float((flat_idx // ball_dist.shape[1]).item()) + 0.5
        by = float((flat_idx %  ball_dist.shape[1]).item()) + 0.5

        in_pos = (x_cpu[ch_in_pos] > 0).float()
        out_pos = (x_cpu[ch_out_pos] > 0).float()

        H, W = ball_dist.shape
        vis = SoccerVisualizer(pitch_length=H, pitch_width=W, layout="x_rows")
        ok = "✓" if int(float(y.item()) > 0.5) else "✗"

        panels = [
            ("P(dest | s)", dest_map.detach().cpu(), "Blues"),
            ("P(complete | s, cell)", succ_map.detach().cpu(), "Blues"),
            ("P(dest & complete | s)", comp_map.detach().cpu(), "Blues"),
        ]

        for c, (title, heat, cmap) in enumerate(panels):
            ax = axes[r, c]
            vis.plot_state(
                in_possession=in_pos,
                out_possession=out_pos,
                heatmap=heat,
                cmap=cmap,
                heatmap_kwargs=dict(alpha=0.9),
                add_colorbar=False,
                ax=ax,
            )
            ax.scatter([bx], [by], c="black", s=25, marker="o", zorder=6)
            ax.scatter([dst_x], [dst_y], c="red", s=25, marker="o", zorder=6)

            if r == 0:
                ax.set_title(title, fontsize=12)
            if c == 0:
                ax.text(
                    0.01, 0.98,
                    f"idx={ds_idx} pass {ok}",
                    transform=ax.transAxes,
                    va="top", ha="left",
                    fontsize=10,
                    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
                )

    fig.suptitle(f"Fixed validation examples (N={N}) | epoch={epoch:03d}", fontsize=16)

    writer.add_figure(tag, fig, global_step=epoch, close=True)
    plt.close(fig)
