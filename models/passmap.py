import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Building blocks
# -------------------------

def _gn(num_channels: int, num_groups: int = 8) -> nn.GroupNorm:
    g = min(num_groups, num_channels)
    # ensure divisible
    while num_channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, num_channels)


class ResBlock(nn.Module):
    """
    Residual block with GroupNorm + SiLU.
    Optionally uses dilation in the 2nd conv.
    """
    def __init__(self, c: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.norm1 = _gn(c)
        self.conv1 = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.norm2 = _gn(c)
        self.conv2 = nn.Conv2d(
            c, c, kernel_size=3,
            padding=dilation, dilation=dilation,
            bias=False
        )
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return x + h


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        # after upsample, reduce channels with 3x3
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        return self.conv(x)


# -------------------------
# BetterSoccerMap (U-Net)
# -------------------------

class BetterSoccerMap(nn.Module):
    """
    U-Net style dense map predictor.

    Input:  (B, in_channels, H=105, W=68)
    Output: logits (B, 1, H, W)
    """
    def __init__(
        self,
        in_channels: int = 14,
        base: int = 64,
        blocks_per_stage: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Conv2d(in_channels, base, kernel_size=3, padding=1, bias=False)

        # Encoder
        self.enc1 = nn.Sequential(*[ResBlock(base, dropout=dropout) for _ in range(blocks_per_stage)])
        self.down1 = Down(base, base * 2)

        self.enc2 = nn.Sequential(*[ResBlock(base * 2, dropout=dropout) for _ in range(blocks_per_stage)])
        self.down2 = Down(base * 2, base * 4)

        self.enc3 = nn.Sequential(*[ResBlock(base * 4, dropout=dropout) for _ in range(blocks_per_stage)])

        # Bottleneck (dilated to expand receptive field without more pooling)
        self.bot = nn.Sequential(
            ResBlock(base * 4, dilation=2, dropout=dropout),
            ResBlock(base * 4, dilation=2, dropout=dropout),
        )

        # Decoder
        self.up2 = Up(base * 4, base * 2)               # up to enc2 spatial size
        self.dec2 = nn.Sequential(*[ResBlock(base * 2, dropout=dropout) for _ in range(blocks_per_stage)])

        self.up1 = Up(base * 2, base)                   # up to enc1 spatial size
        self.dec1 = nn.Sequential(*[ResBlock(base, dropout=dropout) for _ in range(blocks_per_stage)])

        # Head
        self.head = nn.Sequential(
            _gn(base),
            nn.SiLU(),
            nn.Conv2d(base, base, kernel_size=3, padding=1, bias=False),
            _gn(base),
            nn.SiLU(),
            nn.Conv2d(base, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        x = self.stem(x)

        f1 = self.enc1(x)               # (B, base, H, W)
        x = self.down1(f1)              # (B, 2base, H/2, W/2)

        f2 = self.enc2(x)               # (B, 2base, H/2, W/2)
        x = self.down2(f2)              # (B, 4base, H/4, W/4)

        x = self.enc3(x)                # (B, 4base, H/4, W/4)
        x = self.bot(x)                 # (B, 4base, H/4, W/4)

        # decode
        x = self.up2(x, size_hw=(f2.shape[-2], f2.shape[-1]))   # (B, 2base, H/2, W/2)
        x = x + f2                                              # additive skip (stable vs concat)
        x = self.dec2(x)

        x = self.up1(x, size_hw=(f1.shape[-2], f1.shape[-1]))   # (B, base, H, W)
        x = x + f1
        x = self.dec1(x)

        logits = self.head(x)                                   # (B,1,H,W)
        return logits


# -------------------------
# Loss: single-pixel BCE + dense heatmap BCE
# -------------------------

@dataclass
class PassMapLossConfig:
    single_pixel_weight: float = 1.0
    heatmap_weight: float = 1.0
    heatmap_sigma: float = 2.0         # in grid cells
    heatmap_clip: float = 1e-4         # avoid exact 0/1 targets everywhere


def _make_gaussian_targets(
    dst_xy: torch.Tensor,
    H: int,
    W: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    dst_xy: (B,2) with (x_idx, y_idx) in grid coordinates.
    Returns: (B,1,H,W) Gaussian heatmaps in [0,1].
    """
    # grid: x in [0..H-1], y in [0..W-1]
    xs = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)

    x0 = dst_xy[:, 0].to(dtype).view(-1, 1, 1)
    y0 = dst_xy[:, 1].to(dtype).view(-1, 1, 1)

    # exp(-((x-x0)^2 + (y-y0)^2)/(2*sigma^2))
    inv = 1.0 / (2.0 * (sigma ** 2))
    d2 = (xs - x0) ** 2 + (ys - y0) ** 2
    g = torch.exp(-d2 * inv)  # (B,H,W)
    return g.unsqueeze(1)     # (B,1,H,W)


class PassMapLoss(nn.Module):
    """
    Combines:
      - single-pixel BCE on logits at destination cell
      - dense heatmap BCE (Gaussian around destination)

    labels y: (B,) with 0/1 for pass success (or your outcome)
    dst_xy: (B,2) integer grid indices (x_idx, y_idx)
    """
    def __init__(self, cfg: PassMapLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, logits: torch.Tensor, y: torch.Tensor, dst_xy: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        logits: (B,1,H,W)
        y:      (B,) float or int in {0,1}
        dst_xy: (B,2) long/int (x_idx, y_idx)

        Returns: (loss, metrics_dict)
        """
        assert logits.dim() == 4 and logits.size(1) == 1, f"logits must be (B,1,H,W), got {logits.shape}"
        B, _, H, W = logits.shape

        y = y.to(dtype=logits.dtype, device=logits.device).view(B)
        dst_xy = dst_xy.to(device=logits.device)

        # clamp indices to be safe
        x_idx = dst_xy[:, 0].long().clamp(0, H - 1)
        y_idx = dst_xy[:, 1].long().clamp(0, W - 1)

        losses = {}
        total = logits.new_tensor(0.0)

        # --- 1) single-pixel BCE at destination cell ---
        if self.cfg.single_pixel_weight > 0:
            # gather logits at [b, 0, x_idx[b], y_idx[b]]
            pix_logits = logits[torch.arange(B, device=logits.device), 0, x_idx, y_idx]
            single = F.binary_cross_entropy_with_logits(pix_logits, y, reduction="mean")
            total = total + self.cfg.single_pixel_weight * single
            losses["loss_single"] = float(single.detach().cpu())

        # --- 2) dense heatmap BCE (Gaussian target, gated by outcome y) ---
        if self.cfg.heatmap_weight > 0:
            # if y==0, target should be near-zero everywhere; if y==1, gaussian peak at dst
            g = _make_gaussian_targets(
                torch.stack([x_idx, y_idx], dim=1),
                H=H, W=W,
                sigma=self.cfg.heatmap_sigma,
                device=logits.device,
                dtype=logits.dtype,
            )
            # gate by outcome: unsuccessful -> all zeros
            g = g * y.view(B, 1, 1, 1)

            # avoid exact zeros/ones (numerical stability)
            if self.cfg.heatmap_clip is not None and self.cfg.heatmap_clip > 0:
                eps = self.cfg.heatmap_clip
                g = g.clamp(eps, 1.0 - eps)

            dense = F.binary_cross_entropy_with_logits(logits, g, reduction="mean")
            total = total + self.cfg.heatmap_weight * dense
            losses["loss_heatmap"] = float(dense.detach().cpu())

        losses["loss_total"] = float(total.detach().cpu())
        return total, losses
