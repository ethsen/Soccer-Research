import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
# -------------------------
# Blocks (same style as before)
# -------------------------

def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, num_channels)
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ResBlock(nn.Module):
    def __init__(self, c: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _gn(c)
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.norm2 = _gn(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation, bias=False)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return x + h


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.conv(x)

# -------------------------
# Two-head model
# -------------------------

class BetterSoccerMap2Head(nn.Module):
    """
    Outputs two logit maps:
      - dest_logits: (B,1,H,W)  -> softmax over HW for p(dest|state)
      - succ_logits: (B,1,H,W)  -> sigmoid per-cell for p(success|state,dest=cell)
    """
    def __init__(
        self,
        in_channels: int = 14,
        base: int = 64,
        blocks_per_stage: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Backbone U-Net (additive skips)
        self.stem = nn.Conv2d(in_channels, base, 3, padding=1, bias=False)

        self.enc1 = nn.Sequential(*[ResBlock(base, dropout=dropout) for _ in range(blocks_per_stage)])
        self.down1 = Down(base, base * 2)

        self.enc2 = nn.Sequential(*[ResBlock(base * 2, dropout=dropout) for _ in range(blocks_per_stage)])
        self.down2 = Down(base * 2, base * 4)

        self.enc3 = nn.Sequential(*[ResBlock(base * 4, dropout=dropout) for _ in range(blocks_per_stage)])

        self.bot = nn.Sequential(
            ResBlock(base * 4, dilation=2, dropout=dropout),
            ResBlock(base * 4, dilation=2, dropout=dropout),
        )

        self.up2 = Up(base * 4, base * 2)
        self.dec2 = nn.Sequential(*[ResBlock(base * 2, dropout=dropout) for _ in range(blocks_per_stage)])

        self.up1 = Up(base * 2, base)
        self.dec1 = nn.Sequential(*[ResBlock(base, dropout=dropout) for _ in range(blocks_per_stage)])

        # Shared trunk head features
        self.trunk = nn.Sequential(
            _gn(base),
            nn.SiLU(),
            nn.Conv2d(base, base, 3, padding=1, bias=False),
            _gn(base),
            nn.SiLU(),
        )

        # Two heads
        self.dest_head = nn.Conv2d(base, 1, kernel_size=1, bias=True)
        self.succ_head = nn.Conv2d(base, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # x: (B,C,H,W)
        x = self.stem(x)

        f1 = self.enc1(x)         # (B,base,H,W)
        x = self.down1(f1)        # (B,2b,H/2,W/2)

        f2 = self.enc2(x)         # (B,2b,H/2,W/2)
        x = self.down2(f2)        # (B,4b,H/4,W/4)

        x = self.enc3(x)
        x = self.bot(x)

        x = self.up2(x, (f2.shape[-2], f2.shape[-1]))
        x = x + f2
        x = self.dec2(x)

        x = self.up1(x, (f1.shape[-2], f1.shape[-1]))
        x = x + f1
        x = self.dec1(x)

        h = self.trunk(x)
        dest_logits = self.dest_head(h)  # (B,1,H,W)
        succ_logits = self.succ_head(h)  # (B,1,H,W)

        return {"dest_logits": dest_logits, "succ_logits": succ_logits}

    @torch.no_grad()
    def predict_maps(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns:
          p_dest: (B,1,H,W) softmax distribution
          p_succ: (B,1,H,W) sigmoid per-cell
          p_joint_success: (B,1,H,W) = p_dest * p_succ
          p_success: (B,) marginal success = sum_c p_dest(c) p_succ(c)
        """
        out = self.forward(x)
        dest_logits = out["dest_logits"]
        succ_logits = out["succ_logits"]
        B, _, H, W = dest_logits.shape

        p_dest = torch.softmax(dest_logits.view(B, -1), dim=1).view(B, 1, H, W)
        p_succ = torch.sigmoid(succ_logits)
        p_joint = p_dest * p_succ
        p_success = p_joint.sum(dim=(1, 2, 3))
        return {"p_dest": p_dest, "p_succ": p_succ, "p_joint_success": p_joint, "p_success": p_success}


# -------------------------
# Loss (CE for dest + BCE at sampled dest for success)
# -------------------------

@dataclass
class TwoHeadLossCfg:
    w_dest: float = 1.0
    w_succ: float = 1.0

    # Optional dense supervision for destination (Gaussian target distribution)
    use_dense_dest: bool = False
    dense_dest_sigma: float = 2.0
    dense_dest_eps: float = 1e-6  # for normalization stability

    # Optional dense auxiliary for success (Gaussian mask gated by outcome)
    use_dense_succ: bool = False
    dense_succ_sigma: float = 2.0
    dense_succ_clip: float = 1e-4


def _gaussian_unnormalized(dst_xy: torch.Tensor, H: int, W: int, sigma: float, dtype, device) -> torch.Tensor:
    """
    dst_xy: (B,2) with (x_idx,y_idx)
    returns g: (B,1,H,W) unnormalized exp(-d^2/(2sigma^2))
    """
    B = dst_xy.shape[0]
    xs = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)
    x0 = dst_xy[:, 0].to(dtype).view(B, 1, 1)
    y0 = dst_xy[:, 1].to(dtype).view(B, 1, 1)
    inv = 1.0 / (2.0 * (sigma ** 2))
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    g = torch.exp(-d2 * inv)  # (B,H,W)
    return g.unsqueeze(1)


def twohead_loss(
    out: Dict[str, torch.Tensor],
    y: torch.Tensor,
    dst_xy: torch.Tensor,
    cfg: TwoHeadLossCfg,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    out: {"dest_logits": (B,1,H,W), "succ_logits": (B,1,H,W)}
    y: (B,) in {0,1}
    dst_xy: (B,2) integer indices (x_idx,y_idx)
    """
    dest_logits = out["dest_logits"]
    succ_logits = out["succ_logits"]
    assert dest_logits.shape == succ_logits.shape
    B, _, H, W = dest_logits.shape

    device = dest_logits.device
    y = y.to(device=device, dtype=dest_logits.dtype).view(B)
    dst_xy = dst_xy.to(device=device)
    x_idx = dst_xy[:, 0].long().clamp(0, H - 1)
    y_idx = dst_xy[:, 1].long().clamp(0, W - 1)

    logs: Dict[str, float] = {}
    total = dest_logits.new_tensor(0.0)

    # ---- Destination: categorical distribution over HW ----
    if cfg.w_dest > 0:
        if not cfg.use_dense_dest:
            # hard CE
            dest_logits_flat = dest_logits.view(B, -1)  # (B,HW)
            dest_index = (x_idx * W + y_idx).long()     # (B,)
            l_dest = F.cross_entropy(dest_logits_flat, dest_index, reduction="mean")
        else:
            # dense target distribution: KL(p_target || p_pred) where p_pred = softmax(dest_logits)
            # build p_target from gaussian, normalize
            g = _gaussian_unnormalized(
                torch.stack([x_idx, y_idx], dim=1), H, W,
                cfg.dense_dest_sigma, dest_logits.dtype, device
            )  # (B,1,H,W)
            g = g.view(B, -1)
            g = g / (g.sum(dim=1, keepdim=True) + cfg.dense_dest_eps)  # (B,HW)

            logp = F.log_softmax(dest_logits.view(B, -1), dim=1)        # (B,HW)
            # KL(target || pred) = sum target * (log target - log pred)
            l_dest = (g * (torch.log(g + cfg.dense_dest_eps) - logp)).sum(dim=1).mean()

        total = total + cfg.w_dest * l_dest
        logs["loss_dest"] = float(l_dest.detach().cpu())

    # ---- Success: Bernoulli at the sampled destination cell ----
    if cfg.w_succ > 0:
        succ_at_dest = succ_logits[torch.arange(B, device=device), 0, x_idx, y_idx]  # (B,)
        l_succ = F.binary_cross_entropy_with_logits(succ_at_dest, y, reduction="mean")
        total = total + cfg.w_succ * l_succ
        logs["loss_succ"] = float(l_succ.detach().cpu())

    # ---- Optional dense auxiliary for success map (gated by outcome) ----
    if cfg.use_dense_succ and cfg.w_succ > 0:
        g = _gaussian_unnormalized(
            torch.stack([x_idx, y_idx], dim=1), H, W,
            cfg.dense_succ_sigma, succ_logits.dtype, device
        )  # (B,1,H,W)
        g = g * y.view(B, 1, 1, 1)  # unsuccessful -> zeros

        if cfg.dense_succ_clip and cfg.dense_succ_clip > 0:
            g = g.clamp(cfg.dense_succ_clip, 1.0 - cfg.dense_succ_clip)

        l_succ_dense = F.binary_cross_entropy_with_logits(succ_logits, g, reduction="mean")
        total = total + cfg.w_succ * 0.25 * l_succ_dense  # small auxiliary weight
        logs["loss_succ_dense"] = float(l_succ_dense.detach().cpu())

    logs["loss_total"] = float(total.detach().cpu())
    return total, logs
