### passmap
from __future__ import annotations

# passmap.py

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    # stable for small batches; choose groups that divide c
    g = 8
    while c % g != 0 and g > 1:
        g //= 2
    return nn.GroupNorm(g, c)


class ResBlock(nn.Module):
    def __init__(self, c: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = _gn(c)
        self.n2 = _gn(c)
        self.a = nn.SiLU()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.do = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(self.a(self.n1(x)))
        h = self.do(h)
        h = self.c2(self.a(self.n2(h)))
        return x + h


class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PassMap(nn.Module):
    """
    Fully-conv two-head surface model.

    Returns:
      {
        "dest_logits": (B,1,H,W),
        "succ_logits": (B,1,H,W),
      }
    """
    def __init__(
        self,
        in_channels: int,
        base: int = 32,
        blocks_per_stage: int = 2,
        succ_dropout: float = 0.10,
    ):
        super().__init__()

        C0 = base
        C1 = base * 2
        C2 = base * 4

        self.stem = nn.Conv2d(in_channels, C0, 3, padding=1)

        self.enc0 = nn.Sequential(*[ResBlock(C0) for _ in range(blocks_per_stage)])
        self.down1 = Down(C0, C1)
        self.enc1 = nn.Sequential(*[ResBlock(C1) for _ in range(blocks_per_stage)])
        self.down2 = Down(C1, C2)
        self.enc2 = nn.Sequential(*[ResBlock(C2) for _ in range(blocks_per_stage)])

        self.up1 = Up(C2 + C1, C1)
        self.dec1 = nn.Sequential(*[ResBlock(C1) for _ in range(blocks_per_stage)])
        self.up0 = Up(C1 + C0, C0)
        self.dec0 = nn.Sequential(*[ResBlock(C0) for _ in range(blocks_per_stage)])

        # shared trunk output features
        self.trunk_norm = _gn(C0)
        self.trunk_act = nn.SiLU()

        # --- head trunks (decoupled) ---
        # Destination head: light trunk
        self.dest_trunk = nn.Sequential(
            nn.Conv2d(C0, C0, 3, padding=1),
            _gn(C0),
            nn.SiLU(),
            nn.Conv2d(C0, C0, 3, padding=1),
            _gn(C0),
            nn.SiLU(),
        )
        self.dest_head = nn.Conv2d(C0, 1, 1)

        # Success head: slightly stronger + regularized
        self.succ_trunk = nn.Sequential(
            nn.Conv2d(C0, C0, 3, padding=1),
            _gn(C0),
            nn.SiLU(),
            nn.Dropout2d(succ_dropout),
            nn.Conv2d(C0, C0, 3, padding=1),
            _gn(C0),
            nn.SiLU(),
            nn.Dropout2d(succ_dropout),
        )
        self.succ_head = nn.Conv2d(C0, 1, 1)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x0 = self.stem(x)

        s0 = self.enc0(x0)
        x1 = self.down1(s0)
        s1 = self.enc1(x1)
        x2 = self.down2(s1)
        x2 = self.enc2(x2)

        x = self.up1(x2, s1)
        x = self.dec1(x)
        x = self.up0(x, s0)
        x = self.dec0(x)

        f = self.trunk_act(self.trunk_norm(x))

        dest_logits = self.dest_head(self.dest_trunk(f))
        succ_logits = self.succ_head(self.succ_trunk(f))

        return {"dest_logits": dest_logits, "succ_logits": succ_logits}