#!/usr/bin/env python3
"""
Memmap shard writer for SoccerIQ dynamic state tensors.

Writes (per split):
  X_shard_XXXX.mmap : (N,C,H,W) float16
  T_shard_XXXX.mmap : (N,3)     float32  [end_x_m, end_y_m, success]
  manifest.json     : memmap_v1 metadata

Dynamic channels stored (C=13):
  0 dist_to_ball_norm
  1 sin_to_ball
  2 cos_to_ball
  3 atk_team_1hot
  4 def_team_1hot
  5 ball_vx_norm
  6 ball_vy_norm
  7 att_glob_vx_norm
  8 att_glob_vy_norm
  9 def_glob_vx_norm
 10 def_glob_vy_norm
 11 gaussian_control
 12 nearest_def_dist_norm
"""
from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader


# -------------------- Pitch constants --------------------
FL_M, FW_M = 105.0, 68.0
H, W = 105, 68


# -------------------- Channel schema (dynamic only) --------------------
CHANNELS_DYNAMIC = [
    "dist_to_ball_norm",
    "sin_to_ball",
    "cos_to_ball",
    "atk_team_1hot",
    "def_team_1hot",
    "ball_vx_norm",
    "ball_vy_norm",
    "att_glob_vx_norm",
    "att_glob_vy_norm",
    "def_glob_vx_norm",
    "def_glob_vy_norm",
    "gaussian_control",
    "nearest_def_dist_norm",
]

DTYPE_X = np.float16
DTYPE_T = np.float32

# Normalizers
MAX_DIST = float(np.hypot(H - 1, W - 1))  # ~125.0 grid-meters at 1m resolution


# -------------------- Memmap writer --------------------
class MemmapShardWriter:
    """
    Writes memmap_v1 shards:
      X: (shard_size,C,H,W) float16
      T: (shard_size,3)     float32
    """
    def __init__(self, out_dir: Path, shard_size: int, C: int, H: int, W: int, channels: list, split_name: str):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = int(shard_size)
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.channels = list(channels)
        self.split_name = split_name

        self.shard_id = 0
        self.pos = 0
        self.X = None
        self.T = None

        self.manifest = {
            "format": "memmap_v1",
            "split": split_name,
            "dtype_X": str(DTYPE_X),
            "dtype_T": str(DTYPE_T),
            "C": self.C, "H": self.H, "W": self.W,
            "shard_size": self.shard_size,
            "channels": self.channels,
            "shards": []
        }
        self._open_new()

    def _open_new(self):
        if self.X is not None:
            self._close()

        x_path = self.out_dir / f"X_shard_{self.shard_id:04d}.mmap"
        t_path = self.out_dir / f"T_shard_{self.shard_id:04d}.mmap"

        self.X = np.memmap(x_path, mode="w+", dtype=DTYPE_X, shape=(self.shard_size, self.C, self.H, self.W))
        self.T = np.memmap(t_path, mode="w+", dtype=DTYPE_T, shape=(self.shard_size, 3))
        self.pos = 0

        self.manifest["shards"].append({"x_path": x_path.name, "t_path": t_path.name, "n": 0})
        self.shard_id += 1

    def _close(self):
        self.manifest["shards"][-1]["n"] = int(self.pos)
        self.X.flush()
        self.T.flush()
        self.X = None
        self.T = None

    def add(self, x_chw: torch.Tensor, t_3: torch.Tensor):
        if self.pos >= self.shard_size:
            self._open_new()

        x_chw = torch.nan_to_num(x_chw, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32).cpu()
        t_3 = torch.nan_to_num(t_3, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32).cpu()

        self.X[self.pos] = x_chw.numpy().astype(DTYPE_X, copy=False)
        self.T[self.pos] = t_3.numpy().astype(DTYPE_T, copy=False)
        self.pos += 1

    def finalize(self):
        if self.X is not None:
            self._close()
        self.manifest["shards"] = [s for s in self.manifest["shards"] if s["n"] > 0]


# -------------------- Coordinate helpers --------------------
def sb_to_spadl_xy(x_sb: float, y_sb: float) -> Tuple[float, float]:
    x_m = np.clip(x_sb / 120.0 * FL_M, 0.0, FL_M)
    y_m = np.clip(FW_M - (y_sb / 80.0 * FW_M), 0.0, FW_M)
    return float(x_m), float(y_m)


def bucket_xy_to_idx(x_m: float, y_m: float, method: str = "nearest") -> Tuple[int, int, bool]:
    if not np.isfinite(x_m) or not np.isfinite(y_m):
        return 0, 0, False
    if method == "nearest":
        xi = int(np.rint(x_m))
        yi = int(np.rint(y_m))
    else:
        xi = int(np.floor(x_m))
        yi = int(np.floor(y_m))
    xi = max(0, min(H - 1, xi))
    yi = max(0, min(W - 1, yi))
    return xi, yi, True


# -------------------- Map builders --------------------
def point_distance_map_batch(points_xy: np.ndarray, device=None) -> torch.Tensor:
    """
    points_xy: (2,N) meters -> dist: (N,H,W) float32 in meters (grid units).
    """
    locs = torch.as_tensor(points_xy, dtype=torch.float32, device=device)
    x, y = locs[0], locs[1]
    N = x.shape[0]
    xs = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    ys = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    dx = xs - x.view(N, 1, 1)
    dy = ys - y.view(N, 1, 1)
    return torch.hypot(dx, dy)


def angle_sin_cos_map_batch(points_xy: np.ndarray, device=None, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    points_xy: (2,N) meters -> sin_map, cos_map each (N,H,W) float32.
    """
    P = torch.as_tensor(points_xy, dtype=torch.float32, device=device)
    x, y = P[0], P[1]
    N = x.shape[0]
    xs = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    ys = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    dx = xs - x.view(N, 1, 1)
    dy = ys - y.view(N, 1, 1)
    r = torch.hypot(dx, dy).clamp_min(eps)
    cos_map = dx / r
    sin_map = dy / r
    return sin_map, cos_map


def angle_sin_cos_map_single(x_m: float, y_m: float, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    sin_map, cos_map = angle_sin_cos_map_batch(np.array([[x_m], [y_m]], dtype=np.float32), device=device)
    return sin_map[0], cos_map[0]


def transform_freeze_frame_to_team_maps(frames: List[list], device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    frames: list length N, each item is StatsBomb 360 freeze_frame list.
    Returns:
      atk_1hot: (N,H,W) uint8
      def_1hot: (N,H,W) uint8
    """
    atk = torch.zeros((len(frames), H, W), dtype=torch.uint8, device=device)
    deff = torch.zeros((len(frames), H, W), dtype=torch.uint8, device=device)

    for i, ff in enumerate(frames):
        if not isinstance(ff, list) or len(ff) == 0:
            continue
        for p in ff:
            loc = p.get("location", None)
            if not loc or len(loc) < 2:
                continue
            xm, ym = sb_to_spadl_xy(loc[0], loc[1])
            xi, yi, ok = bucket_xy_to_idx(xm, ym, method="nearest")
            if not ok:
                continue
            if bool(p.get("teammate", False)):
                atk[i, xi, yi] = 1
            else:
                deff[i, xi, yi] = 1
    return atk, deff


# -------------------- Labels & filters --------------------
def pass_success_mask(spadl_passes) -> np.ndarray:
    if "result_name" in spadl_passes.columns:
        return (spadl_passes.result_name.values == "success")
    return (spadl_passes.result_id.values == 1)


def successful_pass_has_teammate_within_radius(
    spadl_passes,
    freeze_frames: List[list],
    home_team_id: int,
    radius_m: float = 5.0,
) -> np.ndarray:
    """
    For each pass: True iff any teammate in freeze-frame is within radius_m of pass end (SPADL meters).
    Output: (N,) bool aligned with spadl_passes rows.
    """
    r2 = float(radius_m) * float(radius_m)
    mask = np.zeros(len(spadl_passes), dtype=bool)

    is_away = (spadl_passes.team_id.values.astype(int) != int(home_team_id))

    for i, (row, ff) in enumerate(zip(spadl_passes.itertuples(index=False), freeze_frames)):
        if not isinstance(ff, list) or len(ff) == 0:
            continue
        ex, ey = float(row.end_x), float(row.end_y)
        if not np.isfinite(ex) or not np.isfinite(ey):
            continue

        flip_lr = bool(is_away[i])

        close = False
        for p in ff:
            if not p.get("teammate", False):
                continue
            loc = p.get("location", None)
            if not loc or len(loc) < 2:
                continue
            xm, ym = sb_to_spadl_xy(loc[0], loc[1])
            if flip_lr:
                xm, ym = (FL_M - xm, FW_M - ym)
            dx, dy = xm - ex, ym - ey
            if dx * dx + dy * dy <= r2:
                close = True
                break
        mask[i] = close

    return mask


# -------------------- Velocity helpers --------------------
def compute_ball_velocity(
    spadl_passes,
    dt_min: float = 0.20,
    dt_max: float = 8.00,
    vmax: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns vx, vy arrays (N,) in normalized units (m/s divided by vmax), clipped to [-1,1].
    dt is clamped to [dt_min, dt_max] to prevent explosions from tiny durations.
    """
    sx = spadl_passes.start_x.values.astype(np.float32)
    sy = spadl_passes.start_y.values.astype(np.float32)
    ex = spadl_passes.end_x.values.astype(np.float32)
    ey = spadl_passes.end_y.values.astype(np.float32)

    dx = ex - sx
    dy = ey - sy

    if "duration" in spadl_passes.columns:
        dt = spadl_passes.duration.values.astype(np.float32)
    else:
        ts = spadl_passes.time_seconds.values.astype(np.float32)
        dt = np.diff(ts, append=ts[-1] + 1e-3).astype(np.float32)

    dt = np.clip(dt, float(dt_min), float(dt_max))
    vx = dx / dt
    vy = dy / dt

    vx = np.clip(vx, -vmax, vmax) / float(vmax)
    vy = np.clip(vy, -vmax, vmax) / float(vmax)
    return vx.astype(np.float32), vy.astype(np.float32)


def _centroids_from_360_frame(ff: list, flip_lr: bool = False) -> Optional[Tuple[float, float, float, float]]:
    if not isinstance(ff, list) or len(ff) == 0:
        return None
    ax = ay = dx = dy = 0.0
    ac = dc = 0
    for p in ff:
        loc = p.get("location", None)
        if not loc or len(loc) < 2:
            continue
        xm, ym = sb_to_spadl_xy(loc[0], loc[1])
        if flip_lr:
            xm, ym = (FL_M - xm, FW_M - ym)
        if bool(p.get("teammate", False)):
            ax += xm; ay += ym; ac += 1
        else:
            dx += xm; dy += ym; dc += 1
    if ac == 0 or dc == 0:
        return None
    return (ax / ac, ay / ac, dx / dc, dy / dc)


def compute_glob_velocity_next_action_360(
    spadl_actions_l2r,
    freeze_frames_all: List[Optional[list]],
    pass_event_ids: np.ndarray,
    pass_team_ids: np.ndarray,
    home_team_id: int,
    dt_min: float = 0.20,
    dt_max_lookahead: float = 8.0,
    vmax: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pass: centroid velocity from current 360 frame to next available 360 frame within dt_max_lookahead.
    Outputs normalized velocities (N,) in [-1,1] after clipping to ±vmax and dividing by vmax.
    """
    event_ids_all = spadl_actions_l2r.original_event_id.tolist()
    eid_to_idx = {eid: i for i, eid in enumerate(event_ids_all)}
    t_all = spadl_actions_l2r.time_seconds.values.astype(np.float32)

    N = len(pass_event_ids)
    att_vx = np.zeros(N, dtype=np.float32)
    att_vy = np.zeros(N, dtype=np.float32)
    def_vx = np.zeros(N, dtype=np.float32)
    def_vy = np.zeros(N, dtype=np.float32)

    is_away_pass = (pass_team_ids.astype(int) != int(home_team_id))

    for n in range(N):
        k = eid_to_idx.get(pass_event_ids[n], None)
        if k is None:
            continue

        t_k = float(t_all[k])
        if not np.isfinite(t_k):
            continue

        flip_lr = bool(is_away_pass[n])
        ff_k = freeze_frames_all[k]
        c_k = _centroids_from_360_frame(ff_k, flip_lr=flip_lr)
        if c_k is None:
            continue
        cx_att_k, cy_att_k, cx_def_k, cy_def_k = c_k

        next_j = None
        c_j = None
        for j in range(k + 1, len(event_ids_all)):
            ff_j = freeze_frames_all[j]
            if not isinstance(ff_j, list) or len(ff_j) == 0:
                continue
            t_j = float(t_all[j])
            if not np.isfinite(t_j) or t_j <= t_k:
                continue
            dt = t_j - t_k
            if dt > float(dt_max_lookahead):
                break
            if dt < float(dt_min):
                continue
            c_try = _centroids_from_360_frame(ff_j, flip_lr=flip_lr)
            if c_try is None:
                continue
            next_j = j
            c_j = c_try
            break

        if next_j is None or c_j is None:
            continue

        dt = float(t_all[next_j]) - t_k
        if dt < float(dt_min) or dt > float(dt_max_lookahead):
            continue

        cx_att_j, cy_att_j, cx_def_j, cy_def_j = c_j

        vx_a = (cx_att_j - cx_att_k) / dt
        vy_a = (cy_att_j - cy_att_k) / dt
        vx_d = (cx_def_j - cx_def_k) / dt
        vy_d = (cy_def_j - cy_def_k) / dt

        vx_a = np.clip(vx_a, -vmax, vmax) / float(vmax)
        vy_a = np.clip(vy_a, -vmax, vmax) / float(vmax)
        vx_d = np.clip(vx_d, -vmax, vmax) / float(vmax)
        vy_d = np.clip(vy_d, -vmax, vmax) / float(vmax)

        att_vx[n] = float(vx_a)
        att_vy[n] = float(vy_a)
        def_vx[n] = float(vx_d)
        def_vy[n] = float(vy_d)

    return att_vx, att_vy, def_vx, def_vy


def expand_global_scalar_to_map(vals: np.ndarray, device=None) -> torch.Tensor:
    """
    vals: (N,) -> maps: (N,H,W) float32 with broadcast fill.
    """
    v = torch.as_tensor(vals, dtype=torch.float32, device=device)
    return v.view(-1, 1, 1).expand(-1, H, W)


# -------------------- New dynamic channels --------------------
def gaussian_control(
    atk_1hot: torch.Tensor,
    def_1hot: torch.Tensor,
    H: int = 105,
    W: int = 68,
    sigma: float = 6.0,
    alpha: float = 1e-2,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Inputs:
      atk_1hot, def_1hot: (H,W) or (B,H,W) or (B,1,H,W) with {0,1}
    Output:
      control: (H,W) or (B,H,W) in [0,1]
    """
    if atk_1hot.dim() == 2:
        atk = atk_1hot.unsqueeze(0)
        deff = def_1hot.unsqueeze(0)
        squeeze_out = True
    elif atk_1hot.dim() == 4:
        atk = atk_1hot.squeeze(1)
        deff = def_1hot.squeeze(1)
        squeeze_out = False
    else:
        atk = atk_1hot
        deff = def_1hot
        squeeze_out = False

    B, HH, WW = atk.shape
    device = atk.device
    assert (HH, WW) == (H, W)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    grid = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1).float()  # (HW,2)

    inv_two_sigma2 = 1.0 / (2.0 * float(sigma) * float(sigma))

    out = []
    for b in range(B):
        atk_pts = torch.nonzero(atk[b] > 0, as_tuple=False).float()
        def_pts = torch.nonzero(deff[b] > 0, as_tuple=False).float()

        if atk_pts.numel() == 0:
            Ap = torch.zeros((H * W,), device=device)
        else:
            dA2 = torch.cdist(grid, atk_pts) ** 2
            Ap = torch.exp(-dA2 * inv_two_sigma2).sum(dim=1)

        if def_pts.numel() == 0:
            Dp = torch.zeros((H * W,), device=device)
        else:
            dD2 = torch.cdist(grid, def_pts) ** 2
            Dp = torch.exp(-dD2 * inv_two_sigma2).sum(dim=1)

        A = Ap + float(alpha)
        D = Dp + float(alpha)

        control = A / (A + D + float(eps))

        evidence = Ap + Dp
        ev = evidence / (evidence.max() + float(eps))
        control = ev * control + (1.0 - ev) * 0.5

        out.append(control.view(H, W))

    control = torch.stack(out, dim=0)
    if squeeze_out:
        control = control[0]
    return control


def nearest_defender_distance_map(def_1hot: torch.Tensor, H: int = 105, W: int = 68, eps: float = 1e-8) -> torch.Tensor:
    """
    Input:
      def_1hot: (H,W) or (B,H,W) or (B,1,H,W) with {0,1}
    Output:
      dist: (H,W) or (B,H,W) in meters (grid units), 0 if no defenders.
    """
    if def_1hot.dim() == 2:
        deff = def_1hot.unsqueeze(0)
        squeeze_out = True
    elif def_1hot.dim() == 4:
        deff = def_1hot.squeeze(1)
        squeeze_out = False
    else:
        deff = def_1hot
        squeeze_out = False

    B, HH, WW = deff.shape
    device = deff.device
    assert (HH, WW) == (H, W)

    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    grid = torch.stack([yy.reshape(-1), xx.reshape(-1)], dim=-1).float()  # (HW,2)

    out = []
    for b in range(B):
        pts = torch.nonzero(deff[b] > 0, as_tuple=False).float()
        if pts.numel() == 0:
            out.append(torch.zeros((H, W), device=device))
            continue
        d = torch.cdist(grid, pts)  # (HW, P)
        out.append(d.min(dim=1).values.view(H, W))
    dist = torch.stack(out, dim=0)
    if squeeze_out:
        dist = dist[0]
    return dist


# -------------------- Feature packer (dynamic only) --------------------
def build_feature_tensor_dynamic(
    pass_start_xy: np.ndarray,
    atk_team_maps: torch.Tensor,
    def_team_maps: torch.Tensor,
    vx_ball: np.ndarray,
    vy_ball: np.ndarray,
    att_vx: np.ndarray,
    att_vy: np.ndarray,
    def_vx: np.ndarray,
    def_vy: np.ndarray,
    sigma_pc: float = 6.0,
    device=None,
) -> torch.Tensor:
    """
    Returns:
      X: (C,N,H,W) float32, C=len(CHANNELS_DYNAMIC), NaN-safe.
    """
    dist_to_ball = point_distance_map_batch(pass_start_xy, device=device) / MAX_DIST
    sin_ball, cos_ball = angle_sin_cos_map_batch(pass_start_xy, device=device)

    ball_vx = expand_global_scalar_to_map(vx_ball, device=device)
    ball_vy = expand_global_scalar_to_map(vy_ball, device=device)
    att_gvx = expand_global_scalar_to_map(att_vx, device=device)
    att_gvy = expand_global_scalar_to_map(att_vy, device=device)
    def_gvx = expand_global_scalar_to_map(def_vx, device=device)
    def_gvy = expand_global_scalar_to_map(def_vy, device=device)

    pc = gaussian_control(atk_team_maps, def_team_maps, H=H, W=W, sigma=sigma_pc)  # (N,H,W)
    ndd = nearest_defender_distance_map(def_team_maps, H=H, W=W) / MAX_DIST          # (N,H,W)

    chs = [
        dist_to_ball,
        sin_ball,
        cos_ball,
        atk_team_maps.float(),
        def_team_maps.float(),
        ball_vx,
        ball_vy,
        att_gvx,
        att_gvy,
        def_gvx,
        def_gvy,
        pc,
        ndd,
    ]
    X = torch.stack(chs, dim=0)  # (C,N,H,W)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.contiguous()


# -------------------- Main build routine --------------------
def build_paired_shards_stratified(
    game_ids: np.ndarray,
    api: StatsBombLoader,
    out_root: Path,
    shard_size: int = 10000,
    val_frac: float = 0.10,
    seed: int = 19,
    device: str = "cpu",
    receiver_radius_m: float = 5.0,
    dt_min: float = 0.20,
    dt_max_ball: float = 8.0,
    dt_max_lookahead: float = 8.0,
    vmax: float = 5.0,
    sigma_pc: float = 6.0,
):
    device_t = torch.device(device)
    rng = np.random.RandomState(seed)

    # -------- PASS 1: count kept succ/fail for stratification --------
    total_succ = 0
    total_fail = 0

    for gid in game_ids.tolist():
        try:
            events = api.events(game_id=int(gid), load_360=True)
        except Exception:
            continue

        home_team_id = int(events.iloc[0].team_id)
        spadl_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
        spadl_actions = spadl.add_names(spadl_actions)
        spadl_actions_l2r = spadl.play_left_to_right(spadl_actions, home_team_id=home_team_id)

        passes_all = spadl_actions_l2r[spadl_actions_l2r.type_id == 0]
        if len(passes_all) == 0:
            continue

        ff_passes = []
        for eid in passes_all.original_event_id.tolist():
            idx = events.index[events.event_id == eid]
            ff_passes.append(events.freeze_frame_360.iloc[idx[0]] if len(idx) else None)

        has_ff = np.array([isinstance(ff, list) and len(ff) > 0 for ff in ff_passes], dtype=bool)
        finite_start = np.isfinite(passes_all.start_x.values) & np.isfinite(passes_all.start_y.values)
        finite_end   = np.isfinite(passes_all.end_x.values)   & np.isfinite(passes_all.end_y.values)

        success = pass_success_mask(passes_all)
        closeR  = successful_pass_has_teammate_within_radius(
            passes_all, ff_passes, home_team_id=home_team_id, radius_m=float(receiver_radius_m)
        )

        keep = has_ff & finite_start & finite_end & ((~success) | (success & closeR))
        if keep.sum() == 0:
            continue

        succ_kept = int(success[keep].sum())
        n_kept = int(keep.sum())
        total_succ += succ_kept
        total_fail += (n_kept - succ_kept)

    if total_succ == 0 or total_fail == 0:
        raise RuntimeError(f"Cannot stratify: succ={total_succ} fail={total_fail}")

    need_val_succ = int(round(val_frac * total_succ))
    need_val_fail = int(round(val_frac * total_fail))

    # -------- writers --------
    out_root = Path(out_root)
    train_dir = out_root / "train"
    val_dir   = out_root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    C = len(CHANNELS_DYNAMIC)
    train_writer = MemmapShardWriter(train_dir, shard_size, C=C, H=H, W=W, channels=CHANNELS_DYNAMIC, split_name="train")
    val_writer   = MemmapShardWriter(val_dir,   shard_size, C=C, H=H, W=W, channels=CHANNELS_DYNAMIC, split_name="val")

    kept = 0
    val_succ = val_fail = 0
    train_succ = train_fail = 0

    # -------- PASS 2: replay and write --------
    for gid in game_ids.tolist():
        try:
            events = api.events(game_id=int(gid), load_360=True)
        except Exception:
            continue

        home_team_id = int(events.iloc[0].team_id)
        spadl_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
        spadl_actions = spadl.add_names(spadl_actions)
        spadl_actions_l2r = spadl.play_left_to_right(spadl_actions, home_team_id=home_team_id)

        passes_all = spadl_actions_l2r[spadl_actions_l2r.type_id == 0]
        if len(passes_all) == 0:
            continue

        ff_passes = []
        for eid in passes_all.original_event_id.tolist():
            idx = events.index[events.event_id == eid]
            ff_passes.append(events.freeze_frame_360.iloc[idx[0]] if len(idx) else None)

        has_ff = np.array([isinstance(ff, list) and len(ff) > 0 for ff in ff_passes], dtype=bool)
        finite_start = np.isfinite(passes_all.start_x.values) & np.isfinite(passes_all.start_y.values)
        finite_end   = np.isfinite(passes_all.end_x.values)   & np.isfinite(passes_all.end_y.values)

        success_all = pass_success_mask(passes_all)
        closeR = successful_pass_has_teammate_within_radius(
            passes_all, ff_passes, home_team_id=home_team_id, radius_m=float(receiver_radius_m)
        )

        keep = has_ff & finite_start & finite_end & ((~success_all) | (success_all & closeR))
        if keep.sum() == 0:
            continue

        passes = passes_all[keep].reset_index(drop=True)
        ff_kept = [ff for ff, k in zip(ff_passes, keep) if k]
        success = pass_success_mask(passes).astype(np.uint8)

        atk_maps, def_maps = transform_freeze_frame_to_team_maps(ff_kept, device=device_t)

        start_xy = np.vstack([
            passes.start_x.values.astype(np.float32),
            passes.start_y.values.astype(np.float32)
        ])

        vx_ball, vy_ball = compute_ball_velocity(
            passes, dt_min=float(dt_min), dt_max=float(dt_max_ball), vmax=float(vmax)
        )

        ff_all = []
        for eid in spadl_actions_l2r.original_event_id.tolist():
            idx = events.index[events.event_id == eid]
            ff_all.append(events.freeze_frame_360.iloc[idx[0]] if len(idx) else None)

        att_vx, att_vy, def_vx, def_vy = compute_glob_velocity_next_action_360(
            spadl_actions_l2r=spadl_actions_l2r,
            freeze_frames_all=ff_all,
            pass_event_ids=passes.original_event_id.values,
            pass_team_ids=passes.team_id.values,
            home_team_id=home_team_id,
            dt_min=float(dt_min),
            dt_max_lookahead=float(dt_max_lookahead),
            vmax=float(vmax),
        )

        X = build_feature_tensor_dynamic(
            pass_start_xy=start_xy,
            atk_team_maps=atk_maps,
            def_team_maps=def_maps,
            vx_ball=vx_ball, vy_ball=vy_ball,
            att_vx=att_vx, att_vy=att_vy,
            def_vx=def_vx, def_vy=def_vy,
            sigma_pc=float(sigma_pc),
            device=device_t,
        )  # (C,N,H,W)

        end_x = torch.as_tensor(passes.end_x.values.astype(np.float32), device=device_t)
        end_y = torch.as_tensor(passes.end_y.values.astype(np.float32), device=device_t)
        succ  = torch.as_tensor(success.astype(np.float32), device=device_t)

        N = X.shape[1]
        for i in range(N):
            x_chw = X[:, i]
            t_3 = torch.stack([end_x[i], end_y[i], succ[i]], dim=0)

            is_succ = int(success[i]) == 1

            send_to_val = False
            if is_succ and val_succ < need_val_succ:
                send_to_val = True
            elif (not is_succ) and val_fail < need_val_fail:
                send_to_val = True

            if send_to_val:
                val_writer.add(x_chw, t_3)
                val_succ += int(is_succ)
                val_fail += int(not is_succ)
            else:
                train_writer.add(x_chw, t_3)
                train_succ += int(is_succ)
                train_fail += int(not is_succ)

            kept += 1

        if kept % 5000 == 0:
            print(f"[write] kept={kept} | val(s={val_succ}/{need_val_succ}, f={val_fail}/{need_val_fail})")

        del X, atk_maps, def_maps

    train_writer.finalize()
    val_writer.finalize()

    (train_dir / "manifest.json").write_text(json.dumps(train_writer.manifest, indent=2))
    (val_dir / "manifest.json").write_text(json.dumps(val_writer.manifest, indent=2))

    meta = {
        "seed": int(seed),
        "val_frac": float(val_frac),
        "shard_size": int(shard_size),
        "receiver_radius_m": float(receiver_radius_m),
        "dt_min": float(dt_min),
        "dt_max_ball": float(dt_max_ball),
        "dt_max_lookahead": float(dt_max_lookahead),
        "vmax": float(vmax),
        "sigma_pc": float(sigma_pc),
        "total_succ": int(total_succ),
        "total_fail": int(total_fail),
        "need_val_succ": int(need_val_succ),
        "need_val_fail": int(need_val_fail),
        "final_val_succ": int(val_succ),
        "final_val_fail": int(val_fail),
        "final_train_succ": int(train_succ),
        "final_train_fail": int(train_fail),
    }
    (out_root / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print("[done]")
    print(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open_data_root", type=str, required=True, help=".../open-data-master/data")
    ap.add_argument("--game_ids_path", type=str, required=True, help="path to game_ids.npy")

    ap.add_argument("--out_root", type=str, default="data/soccer_splits_memmap_v2_dynamic")
    ap.add_argument("--shard_size", type=int, default=10000)

    ap.add_argument("--val_frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=19)

    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--receiver_radius_m", type=float, default=5.0)

    ap.add_argument("--dt_min", type=float, default=0.20)
    ap.add_argument("--dt_max_ball", type=float, default=8.0)
    ap.add_argument("--dt_max_lookahead", type=float, default=8.0)

    ap.add_argument("--vmax", type=float, default=5.0)
    ap.add_argument("--sigma_pc", type=float, default=7.0)

    args = ap.parse_args()

    api = StatsBombLoader(getter="local", root=args.open_data_root)
    game_ids = np.load(args.game_ids_path)

    build_paired_shards_stratified(
        game_ids=game_ids,
        api=api,
        out_root=Path(args.out_root),
        shard_size=args.shard_size,
        val_frac=args.val_frac,
        seed=args.seed,
        device=args.device,
        receiver_radius_m=args.receiver_radius_m,
        dt_min=args.dt_min,
        dt_max_ball=args.dt_max_ball,
        dt_max_lookahead=args.dt_max_lookahead,
        vmax=args.vmax,
        sigma_pc=args.sigma_pc,
    )


if __name__ == "__main__":
    main()
