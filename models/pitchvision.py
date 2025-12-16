import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvGNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: int = 1, groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False)
        self.gn = nn.GroupNorm(num_groups=min(groups, out_ch), num_channels=out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))


class _ResBlock(nn.Module):
    def __init__(self, ch: int, groups: int = 8):
        super().__init__()
        self.b1 = _ConvGNAct(ch, ch, 3, 1, groups)
        self.b2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, ch), num_channels=ch),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        y = self.b1(x)
        y = self.b2(y)
        return self.act(x + y)


class _Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, groups: int = 8):
        super().__init__()
        self.proj = _ConvGNAct(in_ch, out_ch, 1, 0, groups)
        self.block = _ResBlock(out_ch, groups)

    def forward(self, x, size_hw):
        x = F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)
        x = self.proj(x)
        return self.block(x)


class PitchVisionNet(nn.Module):
    """
    Simple soccer-oriented two-head net:
      - Shared encoder (downsample x2 then x2)
      - Separate decoders for destination + success
      - Heads output logits at full resolution (H,W)

    forward(x):
      x: (B, C_in, H, W)
      returns dict with:
        dest_logits: (B,1,H,W)
        succ_logits: (B,1,H,W)
    """
    def __init__(self, in_channels: int, base: int = 64, groups: int = 8, succ_dropout: float = 0.2):
        super().__init__()

        # ----- shared encoder -----
        self.stem = nn.Sequential(
            _ConvGNAct(in_channels, base, 3, 1, groups),
            _ResBlock(base, groups),
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, base * 2), num_channels=base * 2),
            nn.SiLU(inplace=True),
            _ResBlock(base * 2, groups),
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, base * 4), num_channels=base * 4),
            nn.SiLU(inplace=True),
            _ResBlock(base * 4, groups),
        )

        bott_ch = base * 4

        # ----- specialized decoders (separate) -----
        self.dest_up1 = _Up(bott_ch, base * 2, groups)
        self.dest_up2 = _Up(base * 2, base, groups)
        self.dest_refine = nn.Sequential(_ResBlock(base, groups), _ConvGNAct(base, base, 3, 1, groups))
        self.dest_head = nn.Conv2d(base, 1, kernel_size=1)

        self.succ_up1 = _Up(bott_ch, base * 2, groups)
        self.succ_up2 = _Up(base * 2, base, groups)
        self.succ_refine = nn.Sequential(_ResBlock(base, groups), _ConvGNAct(base, base, 3, 1, groups))
        self.succ_head = nn.Sequential(
            nn.Dropout2d(p=succ_dropout),
            nn.Conv2d(base, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        x0 = self.stem(x)          # (B, base, H, W)
        x1 = self.down1(x0)        # (B, 2*base, H/2, W/2)
        x2 = self.down2(x1)        # (B, 4*base, H/4, W/4)

        # destination decoder
        d = self.dest_up1(x2, size_hw=x1.shape[-2:])
        d = self.dest_up2(d,  size_hw=x0.shape[-2:])
        d = self.dest_refine(d)
        dest_logits = self.dest_head(d)  # (B,1,H,W)

        # success decoder
        s = self.succ_up1(x2, size_hw=x1.shape[-2:])
        s = self.succ_up2(s,  size_hw=x0.shape[-2:])
        s = self.succ_refine(s)
        succ_logits = self.succ_head(s)  # (B,1,H,W)

        return {"dest_logits": dest_logits, "succ_logits": succ_logits}
