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
    Scalable soccer-oriented two-head net:
      - Shared encoder (optionally 3 downsamples)
      - Separate decoders for destination + success
      - blocks_per_stage controls depth like PassMap
    """
    def __init__(
        self,
        in_channels: int,
        base: int = 64,
        groups: int = 8,
        succ_dropout: float = 0.2,
        blocks_per_stage: int = 2,   # NEW: like PassMap
        decoder_blocks: int | None = None,  # NEW: if None, = blocks_per_stage
        num_down: int = 2,           # NEW: 2 (current) or 3 (bigger receptive field)
    ):
        super().__init__()
        assert num_down in (2, 3)
        if decoder_blocks is None:
            decoder_blocks = blocks_per_stage

        def _stack(ch: int, n: int):
            return nn.Sequential(*[_ResBlock(ch, groups) for _ in range(n)])

        # ----- shared encoder -----
        self.stem = nn.Sequential(
            _ConvGNAct(in_channels, base, 3, 1, groups),
            _stack(base, blocks_per_stage),  # was 1 block before :contentReference[oaicite:3]{index=3}
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(base, base * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, base * 2), num_channels=base * 2),
            nn.SiLU(inplace=True),
            _stack(base * 2, blocks_per_stage),  # was 1 block before :contentReference[oaicite:4]{index=4}
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(groups, base * 4), num_channels=base * 4),
            nn.SiLU(inplace=True),
            _stack(base * 4, blocks_per_stage),  # was 1 block before :contentReference[oaicite:5]{index=5}
        )

        self.num_down = num_down
        if num_down == 3:
            self.down3 = nn.Sequential(
                nn.Conv2d(base * 4, base * 8, kernel_size=3, stride=2, padding=1, bias=False),
                nn.GroupNorm(num_groups=min(groups, base * 8), num_channels=base * 8),
                nn.SiLU(inplace=True),
                _stack(base * 8, blocks_per_stage),
            )
            bott_ch = base * 8
        else:
            bott_ch = base * 4

        # ----- specialized decoders (separate) -----
        # We’ll do: bott -> (2*base) -> base, and if num_down==3 add an extra up.
        # Also add configurable block stacks at each up-resolution.
        if num_down == 3:
            # up from 1/8 -> 1/4
            self.dest_up2 = _Up(bott_ch, base * 4, groups)
            self.succ_up2 = _Up(bott_ch, base * 4, groups)

            self.dest_dec2 = _stack(base * 4, decoder_blocks)
            self.succ_dec2 = _stack(base * 4, decoder_blocks)

            up1_in = base * 4
        else:
            up1_in = bott_ch

        # up from 1/4 -> 1/2
        self.dest_up1 = _Up(up1_in, base * 2, groups)
        self.succ_up1 = _Up(up1_in, base * 2, groups)

        self.dest_dec1 = _stack(base * 2, decoder_blocks)
        self.succ_dec1 = _stack(base * 2, decoder_blocks)

        # up from 1/2 -> 1/1
        self.dest_up0 = _Up(base * 2, base, groups)
        self.succ_up0 = _Up(base * 2, base, groups)

        self.dest_dec0 = _stack(base, decoder_blocks)
        self.succ_dec0 = _stack(base, decoder_blocks)

        # refine + heads (you already had these, but we’ll allow them to be a bit beefier)
        self.dest_refine = nn.Sequential(
            _stack(base, max(1, decoder_blocks // 2)),
            _ConvGNAct(base, base, 3, 1, groups),
        )
        self.dest_head = nn.Conv2d(base, 1, kernel_size=1)

        self.succ_refine = nn.Sequential(
            _stack(base, max(1, decoder_blocks // 2)),
            _ConvGNAct(base, base, 3, 1, groups),
        )
        self.succ_head = nn.Sequential(
            nn.Dropout2d(p=succ_dropout),
            nn.Conv2d(base, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        x0 = self.stem(x)      # (B, base, H, W)
        x1 = self.down1(x0)    # (B, 2*base, H/2, W/2)
        x2 = self.down2(x1)    # (B, 4*base, H/4, W/4)

        if self.num_down == 3:
            x3 = self.down3(x2)  # (B, 8*base, H/8, W/8)

            # destination
            d = self.dest_up2(x3, size_hw=x2.shape[-2:])
            d = self.dest_dec2(d)
            d = self.dest_up1(d,  size_hw=x1.shape[-2:])
        else:
            # destination
            d = self.dest_up1(x2, size_hw=x1.shape[-2:])

        d = self.dest_dec1(d)
        d = self.dest_up0(d,  size_hw=x0.shape[-2:])
        d = self.dest_dec0(d)
        d = self.dest_refine(d)
        dest_logits = self.dest_head(d)

        if self.num_down == 3:
            # success
            s = self.succ_up2(x3, size_hw=x2.shape[-2:])
            s = self.succ_dec2(s)
            s = self.succ_up1(s,  size_hw=x1.shape[-2:])
        else:
            # success
            s = self.succ_up1(x2, size_hw=x1.shape[-2:])

        s = self.succ_dec1(s)
        s = self.succ_up0(s,  size_hw=x0.shape[-2:])
        s = self.succ_dec0(s)
        s = self.succ_refine(s)
        succ_logits = self.succ_head(s)

        return {"dest_logits": dest_logits, "succ_logits": succ_logits}
