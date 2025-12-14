#!/usr/bin/env python3
"""
Clean, NaN-safe StatsBomb->SPADL torch shard writer for SoccerIQ.

What it writes
--------------
1) Feature shards:
   data/soccer_shards/features_shard_XXXX.pt containing:
     {"X": (C,N,H,W) float32, "channels": [...], "H":H, "W":W, "meta": {...}}

2) Target shards:
   data/soccer_shards_targets/targets_shard_XXXX.pt containing:
     {"targets": (N,3) float32/uint8, "schema": ["end_x_m","end_y_m","success"], "meta": {...}}

Critical guarantees
-------------------
- No NaNs or infs are ever written (we filter bad samples + torch.nan_to_num).
- Pass success filter:
    * Unsuccessful passes: always kept (if valid freeze frame + finite coords).
    * Successful passes: kept ONLY if pass end is within 3m of ANY teammate in the 360 freeze-frame.
- dt=0 protection for any velocity computation (duration or next-action dt clamped to eps).

Run
---
python torch-converter_clean.py \
  --open_data_root ../data/open-data-master/data \
  --game_ids_path  ../data/game_ids.npy \
  --max_per_shard  20000
"""
from __future__ import annotations
import warnings

warnings.filterwarnings("ignore")
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader

# -------------------- Constants --------------------
FL_M, FW_M = 105.0, 68.0           # meters (SPADL pitch)
H, W = 105, 68                     # 1m grid (x rows, y cols)
DT_EPS = 1e-3

FEATURE_OUT_DIR = Path("data/soccer_shards")
TARGET_OUT_DIR  = Path("data/soccer_shards_targets")

FEATURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
TARGET_OUT_DIR.mkdir(parents=True, exist_ok=True)

CHANNELS = [
    "dist_to_ball", "sin_to_ball", "cos_to_ball",
    "pos_team_onehot", "def_team_onehot",
    "sin_to_goal(105,34)", "cos_to_goal(105,34)", "dist_to_goal(105,34)",
    "ball_vx", "ball_vy",
    "att_glob_vx", "att_glob_vy",
    "def_glob_vx", "def_glob_vy",
]

# -------------------- Geometry helpers --------------------
def sb_to_spadl_xy(x_sb: float, y_sb: float, fidelity_version=None) -> Tuple[float, float]:
    """StatsBomb (0..120, 0..80) -> SPADL meters (0..105, 0..68)."""
    x_m = np.clip(x_sb / 120.0 * FL_M, 0.0, FL_M)
    y_m = np.clip(FW_M - (y_sb / 80.0 * FW_M), 0.0, FW_M)
    return float(x_m), float(y_m)

def bucket_xy_to_idx(x_m: float, y_m: float, method: str = "floor") -> Tuple[int, int, bool]:
    """Meters -> integer cell index on (H,W) grid. Returns ok=False if non-finite."""
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

def point_distance_map_batch(ball_locs_xy: np.ndarray, device=None) -> torch.Tensor:
    """ball_locs_xy: (2,N) meters -> (N,H,W) distances."""
    locs = torch.as_tensor(ball_locs_xy, dtype=torch.float32, device=device)
    x, y = locs[0], locs[1]  # (N,)
    N = x.shape[0]
    xs = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1)
    ys = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W)
    dx = xs - x.view(N, 1, 1)
    dy = ys - y.view(N, 1, 1)
    return torch.hypot(dx, dy)  # (N,H,W)

def angle_sin_cos_map_batch(points_xy: np.ndarray, device=None, eps: float = 1e-8) -> Tuple[torch.Tensor, torch.Tensor]:
    """points_xy: (2,N) meters -> sin_map, cos_map each (N,H,W), angle from point to each cell."""
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
    sin_map, cos_map = angle_sin_cos_map_batch(np.array([[x_m],[y_m]], dtype=np.float32), device=device)
    return sin_map[0], cos_map[0]

# -------------------- 360 helpers --------------------
def transform_freeze_frame_to_team_maps(frames: List[list], device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    frames: list of freeze-frame lists (already aligned to kept passes)
    Returns:
      pos_team, def_team : (N,H,W) uint8 occupancy maps.
    """
    pos = torch.zeros((len(frames), H, W), dtype=torch.uint8, device=device)
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
                pos[i, xi, yi] = 1
            else:
                deff[i, xi, yi] = 1
    return pos, deff

def pass_success_mask(spadl_passes) -> np.ndarray:
    if "result_name" in spadl_passes.columns:
        return (spadl_passes.result_name.values == "success")
    return (spadl_passes.result_id.values == 1)

def successful_pass_has_teammate_within_radius(
    spadl_passes, freeze_frames: List[list], home_team_id: int, fidelity_version=None, radius_m: float = 3.0
) -> np.ndarray:
    """
    For each pass, check if ANY teammate in the freeze-frame is within radius_m of pass end.
    Computed in SPADL meters.
    Note: we do not L2R flip here because your SPADL actions are already L2R, and the
    360 freeze-frame coordinates are in the same global orientation the events use.
    """
    r2 = radius_m * radius_m
    mask = np.zeros(len(spadl_passes), dtype=bool)

    # Determine if pass belongs to away team (different from home_team_id) to L2R flip 360 coords.
    # Your SPADL is left-to-right, so we must flip 360 coordinates for away-team actions.
    is_away = (spadl_passes.team_id.values.astype(int) != int(home_team_id))

    for i, (row, ff) in enumerate(zip(spadl_passes.itertuples(index=False), freeze_frames)):
        if not isinstance(ff, list) or len(ff) == 0:
            continue
        end_x = float(row.end_x)
        end_y = float(row.end_y)
        if not np.isfinite(end_x) or not np.isfinite(end_y):
            continue

        close = False
        for p in ff:
            if not p.get("teammate", False):
                continue
            loc = p.get("location", None)
            if not loc or len(loc) < 2:
                continue
            xm, ym = sb_to_spadl_xy(loc[0], loc[1], fidelity_version=fidelity_version)
            if bool(is_away[i]):
                xm, ym = (FL_M - xm, FW_M - ym)
            dx = xm - end_x
            dy = ym - end_y
            if dx*dx + dy*dy <= r2:
                close = True
                break
        mask[i] = close
    return mask

# -------------------- Velocity helpers --------------------
def compute_ball_velocity(spadl_passes) -> Tuple[np.ndarray, np.ndarray]:
    """Returns vx,vy in m/s for each pass. Protects dt=0."""
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
        dt = np.diff(ts, append=ts[-1] + DT_EPS)
    dt = np.clip(dt, DT_EPS, None)
    return dx / dt, dy / dt

def _centroids_from_360_frame(ff: list, fidelity_version=None, flip_lr: bool = False) -> Optional[Tuple[float,float,float,float]]:
    """Return (cx_att, cy_att, cx_def, cy_def) in SPADL meters. None if insufficient."""
    if not isinstance(ff, list) or len(ff) == 0:
        return None
    ax = ay = dx = dy = 0.0
    ac = dc = 0
    for p in ff:
        loc = p.get("location", None)
        if not loc or len(loc) < 2:
            continue
        xm, ym = sb_to_spadl_xy(loc[0], loc[1], fidelity_version=fidelity_version)
        if flip_lr:
            xm, ym = (FL_M - xm, FW_M - ym)
        if bool(p.get("teammate", False)):
            ax += xm; ay += ym; ac += 1
        else:
            dx += xm; dy += ym; dc += 1
    if ac == 0 or dc == 0:
        return None
    return (ax/ac, ay/ac, dx/dc, dy/dc)

def compute_glob_velocity_next_action_360(
    spadl_actions_l2r,
    freeze_frames_all: List[Optional[list]],
    pass_event_ids: np.ndarray,
    pass_team_ids: np.ndarray,
    home_team_id: int,
    fidelity_version=None,
    dt_max: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For each pass (identified by original_event_id), find the next SPADL action with a valid 360 frame
    within dt_max seconds and compute centroid velocity (att/def) in m/s.
    If no valid next action, velocity is 0.
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
        ff_k = freeze_frames_all[k]
        c_k = _centroids_from_360_frame(ff_k, fidelity_version=fidelity_version, flip_lr=bool(is_away_pass[n]))
        if c_k is None:
            continue
        cx_att_k, cy_att_k, cx_def_k, cy_def_k = c_k
        t_k = float(t_all[k])
        if not np.isfinite(t_k):
            continue

        next_j = None
        for j in range(k + 1, len(event_ids_all)):
            ff_j = freeze_frames_all[j]
            if not isinstance(ff_j, list) or len(ff_j) == 0:
                continue
            t_j = float(t_all[j])
            if not np.isfinite(t_j) or t_j <= t_k:
                continue
            dt = t_j - t_k
            if dt > dt_max:
                break
            c_j = _centroids_from_360_frame(ff_j, fidelity_version=fidelity_version, flip_lr=bool(is_away_pass[n]))
            if c_j is None:
                continue
            next_j = j
            break

        if next_j is None:
            continue

        dt = float(t_all[next_j]) - t_k
        if dt <= DT_EPS:
            continue

        cx_att_j, cy_att_j, cx_def_j, cy_def_j = c_j
        att_vx[n] = (cx_att_j - cx_att_k) / dt
        att_vy[n] = (cy_att_j - cy_att_k) / dt
        def_vx[n] = (cx_def_j - cx_def_k) / dt
        def_vy[n] = (cy_def_j - cy_def_k) / dt

    return att_vx, att_vy, def_vx, def_vy

def expand_global_scalar_to_map(vals: np.ndarray, device=None) -> torch.Tensor:
    """(N,) -> (N,H,W) filled maps."""
    v = torch.as_tensor(vals, dtype=torch.float32, device=device)
    return v.view(-1, 1, 1).expand(-1, H, W)

# -------------------- Feature packer --------------------
def build_feature_tensor(
    pass_start_xy: np.ndarray,
    pos_team_maps: torch.Tensor,
    def_team_maps: torch.Tensor,
    vx_ball: np.ndarray,
    vy_ball: np.ndarray,
    att_vx: np.ndarray,
    att_vy: np.ndarray,
    def_vx: np.ndarray,
    def_vy: np.ndarray,
    device=None,
) -> torch.Tensor:
    """
    Returns X: (C,N,H,W) float32, NaN-safe.
    """
    # geometry
    dist_to_ball = point_distance_map_batch(pass_start_xy, device=device)
    sin_ball, cos_ball = angle_sin_cos_map_batch(pass_start_xy, device=device)

    # goal (broadcast)
    sg, cg = angle_sin_cos_map_single(105.0, 34.0, device=device)
    gd = point_distance_map_batch(np.array([[105.0],[34.0]], dtype=np.float32), device=device)[0]
    N = dist_to_ball.shape[0]
    sin_goal = sg.unsqueeze(0).expand(N, -1, -1)
    cos_goal = cg.unsqueeze(0).expand(N, -1, -1)
    dist_goal = gd.unsqueeze(0).expand(N, -1, -1)

    # motion
    ball_vx = expand_global_scalar_to_map(vx_ball, device=device)
    ball_vy = expand_global_scalar_to_map(vy_ball, device=device)
    att_gvx = expand_global_scalar_to_map(att_vx, device=device)
    att_gvy = expand_global_scalar_to_map(att_vy, device=device)
    def_gvx = expand_global_scalar_to_map(def_vx, device=device)
    def_gvy = expand_global_scalar_to_map(def_vy, device=device)

    chs = [
        dist_to_ball, sin_ball, cos_ball,
        pos_team_maps.float(), def_team_maps.float(),
        sin_goal, cos_goal, dist_goal,
        ball_vx, ball_vy,
        att_gvx, att_gvy, def_gvx, def_gvy,
    ]
    X = torch.stack(chs, dim=0)  # (C,N,H,W)

    # Absolute safety: scrub any NaN/inf
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X.contiguous()

# -------------------- Shard writer --------------------
def flush_shard(
    shard_id: int,
    X_parts: List[torch.Tensor],
    T_parts: List[torch.Tensor],
    meta_rows: List[Dict],
) -> Tuple[Optional[str], Optional[str], int]:
    if len(X_parts) == 0:
        return None, None, 0
    X = torch.cat(X_parts, dim=1).contiguous()       # (C,N,H,W)
    T = torch.cat(T_parts, dim=0).contiguous()       # (N,3)

    # final scrub (paranoia)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)

    fpath = FEATURE_OUT_DIR / f"features_shard_{shard_id:04d}.pt"
    tpath = TARGET_OUT_DIR  / f"targets_shard_{shard_id:04d}.pt"

    torch.save(
        {"X": X.cpu(), "channels": CHANNELS, "H": H, "W": W, "meta": {"rows": meta_rows}},
        fpath,
    )
    torch.save(
        {"targets": T.cpu(), "schema": ["end_x_m", "end_y_m", "success"], "meta": {"rows": meta_rows}},
        tpath,
    )
    return str(fpath), str(tpath), int(X.shape[1])

def build_paired_shards(game_ids: np.ndarray, api: StatsBombLoader, max_per_shard: int = 20000, device: str = "cpu"):
    device_t = torch.device(device)

    feature_manifest = {"shards": [], "channels": CHANNELS, "H": H, "W": W, "dtype": "float32"}
    target_manifest  = {"shards": [], "schema": ["end_x_m", "end_y_m", "success"], "dtype": "float32/uint8"}

    shard_id = 0
    n_in_shard = 0
    X_parts: List[torch.Tensor] = []
    T_parts: List[torch.Tensor] = []
    meta_rows: List[Dict] = []

    for gid in game_ids.tolist():
        try:
            events = api.events(game_id=int(gid), load_360=True)
        except Exception:
            continue

        fidelity = spadl.statsbomb._infer_xy_fidelity_versions(events)
        home_team_id = int(events.iloc[0].team_id)

        spadl_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
        spadl_actions = spadl.add_names(spadl_actions)
        spadl_actions_l2r = spadl.play_left_to_right(spadl_actions, home_team_id=home_team_id)

        # all passes
        passes_all = spadl_actions_l2r[spadl_actions_l2r.type_id == 0]
        if len(passes_all) == 0:
            continue

        # freeze-frame list for passes
        ff_passes: List[Optional[list]] = []
        for eid in passes_all.original_event_id.tolist():
            idx = events.index[events.event_id == eid]
            ff_passes.append(events.freeze_frame_360.iloc[idx[0]] if len(idx) else None)

        # require valid freeze frame + finite coords
        has_ff = np.array([isinstance(ff, list) and len(ff) > 0 for ff in ff_passes], dtype=bool)
        finite_start = np.isfinite(passes_all.start_x.values) & np.isfinite(passes_all.start_y.values)
        finite_end   = np.isfinite(passes_all.end_x.values)   & np.isfinite(passes_all.end_y.values)

        success = pass_success_mask(passes_all)
        close3  = successful_pass_has_teammate_within_radius(
            passes_all, ff_passes, home_team_id=home_team_id, fidelity_version=fidelity, radius_m=3.0
        )

        keep = has_ff & finite_start & finite_end & ( (~success) | (success & close3) )
        if keep.sum() == 0:
            continue

        passes = passes_all[keep].reset_index(drop=True)
        ff_kept = [ff for ff, k in zip(ff_passes, keep) if k]

        # team maps
        pos_maps, def_maps = transform_freeze_frame_to_team_maps(ff_kept, device=device_t)

        # ball geometry at pass start
        start_xy = np.vstack([passes.start_x.values.astype(np.float32), passes.start_y.values.astype(np.float32)])  # (2,N)

        # velocities
        vx_ball, vy_ball = compute_ball_velocity(passes)

        # freeze-frames for ALL actions (for global velocity)
        ff_all: List[Optional[list]] = []
        for eid in spadl_actions_l2r.original_event_id.tolist():
            idx = events.index[events.event_id == eid]
            ff_all.append(events.freeze_frame_360.iloc[idx[0]] if len(idx) else None)

        att_vx, att_vy, def_vx, def_vy = compute_glob_velocity_next_action_360(
            spadl_actions_l2r=spadl_actions_l2r,
            freeze_frames_all=ff_all,
            pass_event_ids=passes.original_event_id.values,
            pass_team_ids=passes.team_id.values,
            home_team_id=home_team_id,
            fidelity_version=fidelity,
            dt_max=8.0,
        )

        # features
        X = build_feature_tensor(
            pass_start_xy=start_xy,
            pos_team_maps=pos_maps,
            def_team_maps=def_maps,
            vx_ball=vx_ball, vy_ball=vy_ball,
            att_vx=att_vx, att_vy=att_vy,
            def_vx=def_vx, def_vy=def_vy,
            device=device_t,
        )  # (C,N,H,W)

        # targets: (end_x_m, end_y_m, success)
        y_success = pass_success_mask(passes).astype(np.uint8)
        T = torch.empty((len(passes), 3), dtype=torch.float32, device=device_t)
        T[:, 0] = torch.as_tensor(passes.end_x.values.astype(np.float32), device=device_t)
        T[:, 1] = torch.as_tensor(passes.end_y.values.astype(np.float32), device=device_t)
        T[:, 2] = torch.as_tensor(y_success.astype(np.float32), device=device_t)

        # scrub targets too
        T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0).contiguous()

        # append
        X_parts.append(X)
        T_parts.append(T)
        meta_rows.append({"game_id": int(gid), "n": int(X.shape[1])})
        n_in_shard += int(X.shape[1])

        # flush shard
        if n_in_shard >= max_per_shard:
            fpath, tpath, n = flush_shard(shard_id, X_parts, T_parts, meta_rows)
            if fpath is not None:
                feature_manifest["shards"].append({"path": fpath, "frames": n})
                target_manifest["shards"].append({"path": tpath, "frames": n})
            shard_id += 1
            n_in_shard = 0
            X_parts, T_parts, meta_rows = [], [], []

    # final flush
    if n_in_shard > 0:
        fpath, tpath, n = flush_shard(shard_id, X_parts, T_parts, meta_rows)
        if fpath is not None:
            feature_manifest["shards"].append({"path": fpath, "frames": n})
            target_manifest["shards"].append({"path": tpath, "frames": n})

    (FEATURE_OUT_DIR / "manifest.json").write_text(json.dumps(feature_manifest, indent=2))
    (TARGET_OUT_DIR  / "targets_manifest.json").write_text(json.dumps(target_manifest, indent=2))
    print(f"Wrote {len(feature_manifest['shards'])} paired shards.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--open_data_root", type=str, required=True, help=".../open-data-master/data")
    ap.add_argument("--game_ids_path", type=str, required=True, help="path to game_ids.npy")
    ap.add_argument("--max_per_shard", type=int, default=20000)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    api = StatsBombLoader(getter="local", root=args.open_data_root)
    game_ids = np.load(args.game_ids_path)
    build_paired_shards(game_ids, api, max_per_shard=args.max_per_shard, device=args.device)

if __name__ == "__main__":
    main()
