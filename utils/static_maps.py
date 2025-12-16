import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch


@dataclass(frozen=True)
class PitchDims:
    H: int = 105
    W: int = 68


class PitchStaticChannels(torch.nn.Module):
    """
    Standalone generator for static pitch-geometry channels.

    Buffers (float32):
      boundary_dist_norm    : (H,W) in [0,1]
      centerline_dist_norm  : (H,W) in [0,1]
      midfield_dist_norm    : (H,W) in [0,1]
      goal_sin              : (H,W) in [-1,1]
      goal_cos              : (H,W) in [-1,1]
      goal_dist_norm        : (H,W) in [0,1]
    """
    def __init__(
        self,
        dims: PitchDims = PitchDims(),
        goal_xy: Tuple[float, float] = (105.0, 34.0),
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.H = int(dims.H)
        self.W = int(dims.W)
        self.goal_x = float(goal_xy[0])
        self.goal_y = float(goal_xy[1])
        self.eps = float(eps)

        device = device if device is not None else torch.device("cpu")

        yy, xx = torch.meshgrid(
            torch.arange(self.H, device=device, dtype=dtype),  # x (0..H-1)
            torch.arange(self.W, device=device, dtype=dtype),  # y (0..W-1)
            indexing="ij",
        )

        max_dist = math.hypot(self.H - 1, self.W - 1)
        max_center_y = self.W / 2.0

        # --- boundary dist (unchanged)
        dist_top = yy
        dist_bottom = (self.H - 1) - yy
        dist_left = xx
        dist_right = (self.W - 1) - xx
        boundary = torch.minimum(torch.minimum(dist_top, dist_bottom),
                                 torch.minimum(dist_left, dist_right))
        boundary_norm = (boundary / max_dist).clamp_(0.0, 1.0)

        # --- centerline dist (y = W/2) (unchanged)
        center_y = self.W / 2.0
        centerline = (xx - center_y).abs()
        centerline_norm = (centerline / max_center_y).clamp_(0.0, 1.0)

        # --- goal direction + dist (unchanged)
        dx = self.goal_x - yy
        dy = self.goal_y - xx
        r = torch.sqrt(dx * dx + dy * dy).clamp_min(self.eps)

        goal_cos = (dx / r).clamp_(-1.0, 1.0)
        goal_sin = (dy / r).clamp_(-1.0, 1.0)
        goal_dist_norm = (r / max_dist).clamp_(0.0, 1.0)

        self.register_buffer("boundary_dist_norm", boundary_norm.contiguous())
        self.register_buffer("centerline_dist_norm", centerline_norm.contiguous())
        self.register_buffer("goal_sin", goal_sin.contiguous())
        self.register_buffer("goal_cos", goal_cos.contiguous())
        self.register_buffer("goal_dist_norm", goal_dist_norm.contiguous())

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        """
        Returns: (C_static,H,W) in channel order:
          [boundary_dist_norm, centerline_dist_norm, midfield_dist_norm, goal_sin, goal_cos, goal_dist_norm]
        """
        return torch.stack(
            [
                self.boundary_dist_norm,
                self.centerline_dist_norm,
                self.goal_sin,
                self.goal_cos,
                self.goal_dist_norm
            ],
            dim=0,
        )


    @torch.no_grad()
    def expand_to_batch(self, B: int) -> torch.Tensor:
        """
        Returns: (B,C_static,H,W) float32, broadcast-expanded without new computation.
        """
        x = self.forward()  # (C,H,W)
        return x.unsqueeze(0).expand(int(B), -1, -1, -1)

    @torch.no_grad()
    def concat_to(self, X_dyn: torch.Tensor, dim: int = 1) -> torch.Tensor:
        """
        Concatenate static channels to an existing tensor.

        Inputs:
          X_dyn: either (C_dyn,H,W) or (B,C_dyn,H,W)
        Output:
          if X_dyn is (C_dyn,H,W): (C_dyn+C_static,H,W)
          if X_dyn is (B,C_dyn,H,W): (B,C_dyn+C_static,H,W)

        Summary: Appends precomputed static geometry channels to dynamic features for a unified model input.
        """
        if X_dyn.dim() == 3:
            # (C,H,W)
            return torch.cat([X_dyn, self.forward().to(X_dyn.device, dtype=X_dyn.dtype)], dim=0)
        elif X_dyn.dim() == 4:
            # (B,C,H,W)
            B = X_dyn.shape[0]
            static_b = self.expand_to_batch(B).to(X_dyn.device, dtype=X_dyn.dtype)
            return torch.cat([X_dyn, static_b], dim=dim)
        else:
            raise ValueError(f"X_dyn must be (C,H,W) or (B,C,H,W), got shape {tuple(X_dyn.shape)}")
