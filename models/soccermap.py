import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Two 5x5 convs with ReLU, same padding."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=5, stride=1, padding=2, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PredHead(nn.Module):
    """1x1 conv -> ReLU -> 1x1 conv (linear). Produces a single-channel logit map."""
    def __init__(self, in_ch):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, bias=True),
        )

    def forward(self, x):
        return self.head(x)


class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=3, padding=1, bias=True),
        )

    def forward(self, x, size=None):
        x = F.interpolate(x, size=size, mode="nearest") if size is not None \
            else F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)




class Fusion1x1(nn.Module):
    """Concatenate two maps along channels, fuse with a single 1x1 conv."""
    def __init__(self, in_ch, out_ch=1):
        super().__init__()
        self.fuse = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=True)

    def forward(self, *xs):
        x = torch.cat(xs, dim=1)
        return self.fuse(x)


class soccermap_model(nn.Module):
    """
    SoccerMap-like fully-convolutional architecture.

    - Input:  (B, 8, H, W)
    - Output: (B, 1, H, W) probability surface (sigmoid)
    - Also returns:
        * logits at each scale (pre-sigmoid)
        * final fused logit map (pre-sigmoid)
    """
    def __init__(self, in_channels=8, base=32):
        super().__init__()

        # Encoder: three scales (1x, 1/2x, 1/4x)
        self.enc1 = ConvBlock(in_channels, base)         # -> (B, base, H, W)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 1/2x

        self.enc2 = ConvBlock(base, base * 2)            # -> (B, 2*base, H/2, W/2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # -> 1/4x

        self.enc3 = ConvBlock(base * 2, base * 4)        # -> (B, 4*base, H/4, W/4)

        # Prediction heads at each scale
        self.pred_1x4 = PredHead(base * 4)   # at 1/4x
        self.pred_1x2 = PredHead(base * 2)   # at 1/2x
        self.pred_1x  = PredHead(base)       # at 1x

        # Upsampling blocks to bring coarser preds up to 1x
        self.up_from_1x4 = UpBlock(1, 1)     # 1/4x -> 1/2x (on logits)
        self.up_from_1x2 = UpBlock(1, 1)     # 1/2x -> 1x    (on logits)

        # Final fusion of {pred_1x (1ch), up(pred_1x2), up2(pred_1x4)} via 1x1 conv
        self.fuse = Fusion1x1(in_ch=3, out_ch=1)

        # Sigmoid at the very end for probability surface
        self.act = nn.Sigmoid()

    def forward(self, x):
        # ----- encoder -----
        f1 = self.enc1(x)      # 1×   -> (B, C1, 105, 68)
        p1 = self.pool1(f1)    # 1/2× -> (B, C1, 52, 34)
        f2 = self.enc2(p1)     # 1/2× -> (B, C2, 52, 34)
        p2 = self.pool2(f2)    # 1/4× -> (B, C2, 26, 17)
        f3 = self.enc3(p2)     # 1/4× -> (B, C3, 26, 17)

        # ----- per-scale logits -----
        logit_1x4 = self.pred_1x4(f3)   # (B,1,26,17)
        logit_1x2 = self.pred_1x2(f2)   # (B,1,52,34)
        logit_1x  = self.pred_1x(f1)    # (B,1,105,68)

        H1W1 = f1.shape[-2:]  # (105, 68)
        H2W2 = f2.shape[-2:]  # (52, 34)

        logit_1x4_to_1x2 = self.up_from_1x4(logit_1x4, size=H2W2)  # (B,1,52,34)
        logit_1x2_to_1x  = self.up_from_1x2(logit_1x2,  size=H1W1) # (B,1,105,68)
        logit_1x4_to_1x  = self.up_from_1x2(logit_1x4_to_1x2, size=H1W1)  # (B,1,105,68)

        # ----- fuse -----
        fused_logits = self.fuse(logit_1x, logit_1x2_to_1x, logit_1x4_to_1x)  # (B,1,105,68)
        probs = self.act(fused_logits)

        aux = {
            "logit_1x": logit_1x,
            "logit_1x2": logit_1x2,
            "logit_1x4": logit_1x4,
            "logit_1x4_to_1x2": logit_1x4_to_1x2,
            "logit_1x2_to_1x": logit_1x2_to_1x,
        }
        return probs, fused_logits, aux

    @staticmethod
    def target_location_loss(logits, y, dst_xy):
        """
        Target-location (single-pixel) loss from SoccerMap:
          - logits: (B,1,H,W) pre-sigmoid fused logits
          - y:      (B,)       binary 0/1 pass-success labels
          - dst_xy: (B,2)      integer destination coordinates (x=col, y=row) in 1x grid
        Returns BCE-with-logits on the selected pixels.
        """
        # Select the logit at the destination pixel of each example
        # dst_xy: (x, y) -> indices are (batch, channel, y, x)
        bsz, _, H, W = logits.shape
        x_idx = dst_xy[:, 0].clamp_(0, W - 1).long()
        y_idx = dst_xy[:, 1].clamp_(0, H - 1).long()

        # Gather: build linear indices
        lin = y_idx * W + x_idx                      # (B,)
        flat = logits.view(bsz, -1)                  # (B, H*W) across the single channel
        selected = flat.gather(1, lin.view(-1, 1)).squeeze(1)  # (B,)

        return F.binary_cross_entropy_with_logits(selected, y.float(), reduction="mean")
