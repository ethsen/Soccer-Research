
import socceraction.spadl as spadl
from socceraction.data.statsbomb import StatsBombLoader
import warnings
import numpy as np
import json
from tqdm import tqdm
warnings.filterwarnings('ignore')
import os

import torch

SAVE_PATH = "dataset_features.pt"
OUT_DIR = "data/soccer_shards"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_OUT_DIR = "data/soccer_shards_targets"
os.makedirs(TARGET_OUT_DIR, exist_ok=True)

DTYPE = torch.float32        # inputs only (targets separate later)
FL, FW = 105.0, 68.0  # meters
H, W = 105, 68

def _save_target_shard(target_rows, shard_id, schema, index_slices):
    """
    target_rows: list of torch.Tensors shaped (N_i, 3) [x,y,outcome]
    index_slices: list of tuples (game_id, start, end)
    """
    if len(target_rows) == 0:
        return None, 0
    T = torch.cat(target_rows, dim=0).contiguous()  # (N_total, 3)
    path = os.path.join(TARGET_OUT_DIR, f"targets_shard_{shard_id:04d}.pt")
    torch.save({"targets": T.cpu(), "schema": schema, "index": index_slices}, path)
    return path, T.shape[0]

# ---------- 1) Quantize float meters to integer cell indices ----------
def bucket_xy_to_idx(x_m: float, y_m: float, H: int = FL, W: int = FW, method: str = "floor"):
    """
    Quantize (x_m, y_m) in meters to integer cell indices (xi, yi) on an H×W grid.
    - method='floor' (default): 55.63 -> 55, safe for "in this 1m cell"
    - method='nearest':         55.63 -> 56 (round to nearest)
    Returns (xi, yi, valid) where valid=False if inputs are NaN/inf or far OOB.
    """
    # invalid?
    if not np.isfinite(x_m) or not np.isfinite(y_m):
        return 0, 0, False

    if method == "nearest":
        xi = int(np.rint(x_m))
        yi = int(np.rint(y_m))
    else:  # 'floor'
        xi = int(np.floor(x_m))
        yi = int(np.floor(y_m))

    # clamp into grid (e.g., x=105.0 -> 104)
    xi = max(0, min(H - 1, xi))
    yi = max(0, min(W - 1, yi))
    return xi, yi, True


# ---------- 2) Write a 1 into a grid at (xi, yi) ----------
def mark_onehot(grid: torch.Tensor, xi: int, yi: int, value: int | float = 1):
    """
    Sets grid[xi, yi] = value. Assumes grid is (H, W) with row=x, col=y.
    """
    # ensure dtype is write-compatible
    if grid.dtype.is_floating_point:
        v = float(value)
    else:
        v = int(value)
    grid[xi, yi] = v


# ---------- 3) Vectorized bucketing for many points ----------
def bucket_many_xy_to_grid(xs: list[float] | np.ndarray,
                           ys: list[float] | np.ndarray,
                           H: int = FL, W: int = FW,
                           method: str = "floor",
                           device: torch.device | None = None,
                           dtype: torch.dtype = torch.int16) -> torch.Tensor:
    """
    Rasterize many (x_m, y_m) into a (H, W) one-hot grid.
    Duplicates are fine (remain 1). NaNs are skipped.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    grid = torch.zeros((H, W), dtype=dtype, device=device)

    # valid mask
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not valid.any():
        return grid

    xv = xs[valid]
    yv = ys[valid]

    if method == "nearest":
        xi = np.rint(xv).astype(int)
        yi = np.rint(yv).astype(int)
    else:
        xi = np.floor(xv).astype(int)
        yi = np.floor(yv).astype(int)

    # clamp
    xi = np.clip(xi, 0, H - 1)
    yi = np.clip(yi, 0, W - 1)

    # write ones (vectorized via advanced indexing)
    grid[torch.as_tensor(xi, device=device), torch.as_tensor(yi, device=device)] = 1
    return grid

def sb_to_spadl_xy(x, y, fidelity_version=None, assume_cell_center=False):
    """
    Convert a single StatsBomb (x,y) to SPADL meters.
    - If `assume_cell_center=True`, subtract half-cell. Use for old, integer-ish event coords.
    - For 360 freeze-frames, pass assume_cell_center=False (no center shift).
    """
    if assume_cell_center:
        cell_side = 0.1 if fidelity_version == 2 else 1.0
        x = x - cell_side/2.0
        y = y - cell_side/2.0

    x_m = np.clip(x / 120.0 * FL, 0, FL)
    y_m = np.clip(FW - (y / 80.0 * FW), 0, FW)
    return x_m, y_m

def ltr_flip_if_away(x_m, y_m, is_away):
    if is_away:
        return (FL - x_m, FW - y_m)
    return (x_m, y_m)

import numpy as np
import torch
        
FL, FW = 105, 68  # cells

def transform_freeze_frame(frames, is_away, fidelity_version=None, method="nearest"):
    """
    Always returns pos_stack and def_stack with EXACTLY len(frames) entries.
    Missing frames produce zero grids.
    """
    pos_list = []
    def_list = []

    for three_sixty in frames:

        # default empty grids
        pos_team = torch.zeros((FL, FW), dtype=torch.uint8)
        def_team = torch.zeros((FL, FW), dtype=torch.uint8)

        for p in three_sixty:
            loc = p.get("location")
            if not loc or len(loc) < 2:
                continue

            x, y = loc
            xm, ym = sb_to_spadl_xy(x, y, fidelity_version, assume_cell_center=False)
            xm, ym = ltr_flip_if_away(xm, ym, is_away)

            xi, yi, ok = bucket_xy_to_idx(xm, ym, H=FL, W=FW, method=method)
            if not ok:
                continue

            if bool(p.get("teammate", False)):
                pos_team[xi, yi] = 1
            else:
                def_team[xi, yi] = 1

        pos_list.append(pos_team)
        def_list.append(def_team)

    # Now ALWAYS matches len(frames)
    pos_stack = torch.stack(pos_list, dim=0)
    def_stack = torch.stack(def_list, dim=0)

    return pos_stack, def_stack



def save_ids(api):
    count = 0
    game_ids = []
    for entry in os.scandir("data/open-data-master/data/three-sixty"):
        if entry.is_file():
            id = int(entry.name.split('.')[0])
            try:
                events = api.events(game_id = id, load_360 = True)
                game_ids.append(id)
            except:
                count += 1
                continue


    print(f'Example Game IDs: {game_ids[3:6]}')
    print(f'Total Games: {len(game_ids)}')
    print(f"Faulty Game Count: {count}/{len(game_ids)}")

    np.save('game_ids.npy',np.array(game_ids))

def one_hot_from_xy(pass_end_loc: np.ndarray | torch.Tensor,
                    H: int = FL, W: int = FW,
                    method: str = "floor",
                    binary: bool = True,
                    dtype: torch.dtype = torch.uint8,
                    device: torch.device | None = None) -> torch.Tensor:
    """
    Build a (H, W) map with 1s at given (x,y) coords.
    - pass_end_loc: array/tensor of shape (2, N) with rows [xs; ys] in SPADL meters.
    - method: 'floor' (default) keeps point in its 1m cell; 'nearest' rounds to nearest cell.
    - binary: if True, cells are clamped to {0,1}; if False, cells accumulate counts.
    - layout: row=x, col=y (matches earlier convention).

    Returns: torch.Tensor (H, W), dtype `dtype` on `device`.
    """
    # to numpy first (robust to lists/torch), then to torch on chosen device
    if isinstance(pass_end_loc, torch.Tensor):
        arr = pass_end_loc.detach().cpu().numpy()
    else:
        arr = np.asarray(pass_end_loc)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"pass_end_loc must be shape (2, N); got {arr.shape}")

    xs, ys = arr[0], arr[1]
    valid = np.isfinite(xs) & np.isfinite(ys)
    if not valid.any():
        return torch.zeros((H, W), dtype=dtype, device=device)

    xv = xs[valid]
    yv = ys[valid]

    if method == "nearest":
        xi = np.rint(xv).astype(np.int64)
        yi = np.rint(yv).astype(np.int64)
    else:  # 'floor'
        xi = np.floor(xv).astype(np.int64)
        yi = np.floor(yv).astype(np.int64)

    # clamp to grid bounds (e.g., x=105 -> 104)
    np.clip(xi, 0, H - 1, out=xi)
    np.clip(yi, 0, W - 1, out=yi)

    xi_t = torch.from_numpy(xi).to(device=device)
    yi_t = torch.from_numpy(yi).to(device=device)

    grid = torch.zeros((H, W), dtype=dtype, device=device)

    if binary:
        grid[xi_t, yi_t] = 1
    else:
        # accumulate counts (use scatter_add)
        vals = torch.ones_like(xi_t, dtype=torch.int64, device=device)
        idx_flat = xi_t * W + yi_t
        flat = torch.zeros(H * W, dtype=torch.int64, device=device)
        flat.scatter_add_(0, idx_flat, vals)
        grid = flat.view(H, W).to(dtype=dtype)

    return grid

def one_hot_from_xy_batch(pass_end_loc,
                          H: int = FL, W: int = FW,
                          method: str = "floor",
                          dtype: torch.dtype = torch.uint8,
                          device: torch.device | None = None) -> torch.Tensor:
    """
    pass_end_loc: (2, N) array/tensor with rows [xs; ys] in SPADL meters.
    Returns: (N, H, W) one-hot maps with row=x, col=y (cell centers at integers).
    """
    arr = torch.as_tensor(pass_end_loc, dtype=torch.float32, device=device)
    if arr.ndim != 2 or arr.shape[0] != 2:
        raise ValueError(f"pass_end_loc must be shape (2, N); got {tuple(arr.shape)}")
    xs, ys = arr[0], arr[1]
    N = xs.shape[0]

    valid = torch.isfinite(xs) & torch.isfinite(ys)
    xi_f = xs.clone(); yi_f = ys.clone()

    if method == "nearest":
        xi = torch.round(xi_f[valid]).long()
        yi = torch.round(yi_f[valid]).long()
    else:  # 'floor'
        xi = torch.floor(xi_f[valid]).long()
        yi = torch.floor(yi_f[valid]).long()

    xi.clamp_(0, H - 1)
    yi.clamp_(0, W - 1)

    grid = torch.zeros((N, H, W), dtype=dtype, device=device)
    b = torch.arange(N, device=device)[valid]
    grid[b, xi, yi] = 1  # one-hot per pass

    return grid


def point_distance_map(x, y, H=105, W=68, device=None, dtype=torch.float32, normalized=False):
    """
    Returns a (H, W) tensor where entry [i, j] is the distance (in meters) from cell (x=i, y=j) to (x, y).

    x, y: ball location in meters (floats, 0<=x<=H, 0<=y<=W). Values outside will still work.
    H, W: grid size; we use row=x in [0..H-1], col=y in [0..W-1] to match (105,68).
    squared: if True, returns squared Euclidean distance.
    normalized: if True, divide by the max possible distance on the grid (corner-to-corner).

    Notes:
    - Fully differentiable; no loops.
    - If you prefer conventional row=y, col=x indexing, swap the dx/dy grids below.
    """
    xs = torch.arange(H, device=device, dtype=dtype).unsqueeze(1)  # shape (H, 1)
    ys = torch.arange(W, device=device, dtype=dtype).unsqueeze(0)  # shape (1, W)

    dx = xs - torch.as_tensor(x, device=device, dtype=dtype)  # (H, 1)
    dy = ys - torch.as_tensor(y, device=device, dtype=dtype)  # (1, W)


    dist = torch.hypot(dx, dy)

    """if normalized: Should I normalize or will batch norm take care of this for me? 
        max_d = torch.hypot(torch.tensor(H-1, dtype=dtype, device=device),
                            torch.tensor(W-1, dtype=dtype, device=device))
        dist = dist / max_d"""

    return dist  # shape (H, W)

def passes_within_radius_mask(spadl_passes, three_sixty, home_team_id, fidelity_version,
                              radius_m: float = 5.0):
    """
    Returns a boolean mask of shape (N_passes,) where True iff:
      - freeze-frame is valid
      - there is at least one *teammate* in the freeze-frame within `radius_m` of pass end location.
    Distances are computed in SPADL meters, in your left-to-right frame.
    """
    N = len(spadl_passes)
    mask = np.zeros(N, dtype=bool)
    r2 = radius_m * radius_m

    # We'll iterate in row order; three_sixty must be aligned to spadl_passes (as you already assume)
    for i, (row, ff) in enumerate(zip(spadl_passes.itertuples(index=False), three_sixty)):
        # freeze-frame sanity
        if ff is None or (isinstance(ff, float) and not np.isfinite(ff)) or not isinstance(ff, list):
            continue  # leave mask[i] = False

        end_x = float(row.end_x)   # SPADL meters, already left-to-right
        end_y = float(row.end_y)
        team_id = int(row.team_id)
        is_away = (team_id != home_team_id)

        close = False
        for p in ff:
            if not p.get("teammate", False):
                continue
            loc = p.get("location", None)
            if not loc or len(loc) < 2:
                continue

            x_sb, y_sb = loc
            # Convert StatsBomb (0–120, 0–80) → SPADL meters
            xm, ym = sb_to_spadl_xy(x_sb, y_sb, fidelity_version, assume_cell_center=False)
            # Flip into your left-to-right frame if this is an away-team pass
            xm, ym = ltr_flip_if_away(xm, ym, is_away=is_away)

            dx = xm - end_x
            dy = ym - end_y
            if dx * dx + dy * dy <= r2:
                close = True
                break

        mask[i] = close

    return mask

def pass_success_mask(spadl_passes):
    if "result_name" in spadl_passes.columns:
        return (spadl_passes.result_name.values == "success")
    else:
        return (spadl_passes.result_id.values == 1)

def compute_ball_velocity(spadl_passes):
    """
    Returns vx, vy arrays of shape (N,) in meters/second (approx).
    Uses 'duration' if present, otherwise finite-difference on 'time_seconds'.
    """
    start_x = spadl_passes.start_x.values.astype(np.float32)
    start_y = spadl_passes.start_y.values.astype(np.float32)
    end_x   = spadl_passes.end_x.values.astype(np.float32)
    end_y   = spadl_passes.end_y.values.astype(np.float32)

    dx = end_x - start_x
    dy = end_y - start_y

    if "duration" in spadl_passes.columns:
        dt = spadl_passes.duration.values.astype(np.float32)
    else:
        ts = spadl_passes.time_seconds.values.astype(np.float32)
        dt = np.diff(ts, append=ts[-1] + 1e-3)  # last one: small dummy +1e-3

    dt = np.clip(dt, 1e-3, None)  # avoid division by 0
    vx = dx / dt
    vy = dy / dt
    return vx, vy

def expand_global_scalar_to_map(vals: np.ndarray, H: int = H, W: int = W,
                                device=None, dtype=torch.float32):
    """
    vals: (N,) array → (N, H, W) tensor where each frame is filled with vals[n].
    """
    v = torch.as_tensor(vals, dtype=dtype, device=device)  # (N,)
    return v.view(-1, 1, 1).expand(-1, H, W)               # (N,H,W)

def _centroids_from_360_frame(three_sixty, fidelity_version=None):
    """
    Given a single StatsBomb 360 freeze-frame (list of player dicts),
    return (cx_att, cy_att, cx_def, cy_def) in SPADL meters, WITHOUT left/right flipping.

    This matches transform_freeze_frame's coordinate convention when called with is_away=False.
    """
    if three_sixty is None or (isinstance(three_sixty, float) and not np.isfinite(three_sixty)):
        return None, None, None, None
    if not isinstance(three_sixty, list) or len(three_sixty) == 0:
        return None, None, None, None

    sum_x_att = 0.0
    sum_y_att = 0.0
    cnt_att = 0
    sum_x_def = 0.0
    sum_y_def = 0.0
    cnt_def = 0

    for p in three_sixty:
        loc = p.get("location", None)
        if not loc or len(loc) < 2:
            continue
        x_sb, y_sb = loc

        # Convert StatsBomb (0–120, 0–80) → SPADL meters (0–105, 0–68)
        xm, ym = sb_to_spadl_xy(x_sb, y_sb, fidelity_version, assume_cell_center=False)
        # IMPORTANT: we do NOT flip by team here -> same global orientation as transform_freeze_frame(..., is_away=False)
        xm, ym = ltr_flip_if_away(xm, ym, is_away=False)

        if bool(p.get("teammate", False)):
            sum_x_att += xm
            sum_y_att += ym
            cnt_att += 1
        else:
            sum_x_def += xm
            sum_y_def += ym
            cnt_def += 1

    if cnt_att == 0 or cnt_def == 0:
        return None, None, None, None

    cx_att = sum_x_att / cnt_att
    cy_att = sum_y_att / cnt_att
    cx_def = sum_x_def / cnt_def
    cy_def = sum_y_def / cnt_def
    return cx_att, cy_att, cx_def, cy_def


def compute_glob_velocity_next_action_360(
    spadl_actions_l2r,
    spadl_passes,
    freeze_frames_all,   # freeze_frame_360 for ALL actions in spadl_actions order
    fidelity_version=None,
    dt_max=8.0,
):
    """
    Computes glob velocity for each pass using the next SPADL action that has
    valid 360 freeze-frame data, regardless of team, as long as it's within dt_max seconds.

    Inputs:
        spadl_actions_l2r: full SPADL action table for a game (left-to-right)
        spadl_passes: filtered passes used for this shard (subset)
        freeze_frames_all: list of length len(spadl_actions_l2r),
                           where each element is the 360 freeze frame or None.

    Returns:
        att_vx, att_vy, def_vx, def_vy (each N-pass-length np arrays)
        using centroid(vx,vy) differences between pass frame k and next action j.
    """

    # Map original_event_id → index in spadl_actions_l2r
    event_ids = spadl_actions_l2r.original_event_id.tolist()
    eid_to_index = {eid: i for i, eid in enumerate(event_ids)}

    # Extract the *times of all actions*
    times_all = spadl_actions_l2r.time_seconds.values.astype(np.float32)

    N = len(spadl_passes)
    att_vx = np.zeros(N, dtype=np.float32)
    att_vy = np.zeros(N, dtype=np.float32)
    def_vx = np.zeros(N, dtype=np.float32)
    def_vy = np.zeros(N, dtype=np.float32)

    # For each pass, find its row index in spadl_actions
    pass_event_ids = spadl_passes.original_event_id.values

    for n in range(N):
        eid_k = pass_event_ids[n]
        k = eid_to_index.get(eid_k, None)
        if k is None:
            continue

        ff_k = freeze_frames_all[k]
        cx_att_k, cy_att_k, cx_def_k, cy_def_k = _centroids_from_360_frame(
            ff_k, fidelity_version=fidelity_version
        )
        if cx_att_k is None:
            continue

        t_k = times_all[k]

        # Search forward for next SPADL action with 360 data
        next_j = None
        for j in range(k + 1, len(spadl_actions_l2r)):
            ff_j = freeze_frames_all[j]
            if not isinstance(ff_j, list):
                continue

            t_j = times_all[j]
            if not np.isfinite(t_j) or t_j <= t_k:
                continue

            dt = t_j - t_k
            if dt > dt_max:
                break

            cx_att_j, cy_att_j, cx_def_j, cy_def_j = _centroids_from_360_frame(
                ff_j, fidelity_version=fidelity_version
            )
            if cx_att_j is None:
                continue

            next_j = j
            break

        if next_j is None:
            # No future 360 frame → leave velocity = 0
            continue

        dt = times_all[next_j] - t_k
        if dt <= 1e-3:
            continue

        att_vx[n] = (cx_att_j - cx_att_k) / dt
        att_vy[n] = (cy_att_j - cy_att_k) / dt
        def_vx[n] = (cx_def_j - cx_def_k) / dt
        def_vy[n] = (cy_def_j - cy_def_k) / dt

    return att_vx, att_vy, def_vx, def_vy


def compute_team_centroids(team_maps: torch.Tensor):
    """
    team_maps: (N, H, W) tensor with 0/1 occupancy.
    Returns:
        cx, cy: tensors of shape (N,) giving centroid in cell coordinates.
    """
    N, H_, W_ = team_maps.shape
    device = team_maps.device
    dtype = team_maps.dtype

    xs = torch.arange(H_, device=device, dtype=torch.float32).view(1, H_, 1)
    ys = torch.arange(W_, device=device, dtype=torch.float32).view(1, 1, W_)

    mass = team_maps.sum(dim=(1, 2)).clamp_min(1e-6)       # (N,)
    cx = (team_maps * xs).sum(dim=(1, 2)) / mass           # (N,)
    cy = (team_maps * ys).sum(dim=(1, 2)) / mass           # (N,)
    return cx, cy


def point_distance_map_batch(ball_locs,
                             H: int = 105, W: int = 68,
                             device=None, dtype=torch.float32,
                             normalized: bool = False) -> torch.Tensor:
    """
    ball_locs: (2, N) with rows [xs; ys] in meters.
    Returns: (N, H, W) where entry [n, i, j] = dist from cell (x=i, y=j) to ball_locs[:, n].
    Layout: row=x, col=y.
    """
    locs = torch.as_tensor(ball_locs, dtype=dtype, device=device)
    if locs.ndim != 2 or locs.shape[0] != 2:
        raise ValueError(f"ball_locs must be shape (2, N); got {tuple(locs.shape)}")
    x, y = locs[0], locs[1]            # (N,), (N,)
    N = x.shape[0]

    xs = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)   # (1,H,1)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)   # (1,1,W)

    dx = xs - x.view(N, 1, 1)          # (N,H,1)
    dy = ys - y.view(N, 1, 1)          # (N,1,W)
    dist = torch.hypot(dx, dy)         # (N,H,W)

    if normalized:
        max_d = torch.hypot(torch.tensor(H-1, dtype=dtype, device=device),
                            torch.tensor(W-1, dtype=dtype, device=device))
        dist = dist / max_d

    return dist

def angle_sin_cos_map_batch(points,
                            H: int = 105, W: int = 68,
                            device=None, dtype=torch.float32,
                            eps: float = 1e-8):
    """
    points: (2, N) or (N, 2) of [x, y] in meters.
    Returns:
      sin_map: (N, H, W)
      cos_map: (N, H, W)
    Layout: row=x in [0..H-1], col=y in [0..W-1].
    """
    P = torch.as_tensor(points, dtype=dtype, device=device)
    if P.ndim != 2 or (P.shape[0] not in (2,) and P.shape[1] not in (2,)):
        raise ValueError(f"points must be (2,N) or (N,2); got {tuple(P.shape)}")
    if P.shape[0] == 2:
        x, y = P[0], P[1]                   # (N,)
    else:
        x, y = P[:, 0], P[:, 1]             # (N,)
    N = x.shape[0]

    xs = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)  # (1,H,1)
    ys = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)  # (1,1,W)

    dx = xs - x.view(N, 1, 1)               # (N,H,1)
    dy = ys - y.view(N, 1, 1)               # (N,1,W)
    r  = torch.hypot(dx, dy).clamp_min(eps) # (N,H,W)

    cos_map = dx / r                         # (N,H,W)
    sin_map = dy / r                         # (N,H,W)
    return sin_map, cos_map


# Convenience single-point wrapper (returns HxW):
def angle_sin_cos_map(x, y, H=105, W=68, device=None, dtype=torch.float32, eps: float = 1e-8):
    sin_map, cos_map = angle_sin_cos_map_batch(
        torch.tensor([[x, y]], dtype=dtype, device=device), H, W, device, dtype, eps
    )
    return sin_map[0], cos_map[0]


def build_feature_tensor(
    distance_maps,
    sin_ball_map,
    cos_ball_map,
    pos_team_states=None,
    def_team_states=None,
    sin_goal_map=None,
    cos_goal_map=None,
    goal_dist_map=None,
    pass_end_onehot=None,
    include_target=False,
    ball_vx_map=None,      # NEW
    ball_vy_map=None,      # NEW
    att_glob_vx_map=None,  # NEW
    att_glob_vy_map=None,  # NEW
    def_glob_vx_map=None,  # NEW
    def_glob_vy_map=None,  # NEW
    dtype=torch.float32
    ):

    """
    Returns:
      X: (C, N, H, W) features (float32 by default)
      y: (N, H, W)      target (if include_target=False and pass_end_onehot provided, returned separately)
    """
    N = distance_maps.shape[0]

    def to_f32(t):
        return t.to(dtype) if t is not None else None

    # Required channels
    chs = [
        to_f32(distance_maps),   # c0
        to_f32(sin_ball_map),    # c1
        to_f32(cos_ball_map),    # c2
    ]

    # Optional team state channels
    if pos_team_states is not None:
        chs.append(to_f32(pos_team_states))
    if def_team_states is not None:
        chs.append(to_f32(def_team_states))

    # Optional goal geometry
    if sin_goal_map is not None:
        chs.append(to_f32(sin_goal_map))
    if cos_goal_map is not None:
        chs.append(to_f32(cos_goal_map))
    if goal_dist_map is not None:
        chs.append(to_f32(goal_dist_map))

    # Optional motion channels
    if ball_vx_map is not None:
        chs.append(to_f32(ball_vx_map))
    if ball_vy_map is not None:
        chs.append(to_f32(ball_vy_map))
    if att_glob_vx_map is not None:
        chs.append(to_f32(att_glob_vx_map))
    if att_glob_vy_map is not None:
        chs.append(to_f32(att_glob_vy_map))
    if def_glob_vx_map is not None:
        chs.append(to_f32(def_glob_vx_map))
    if def_glob_vy_map is not None:
        chs.append(to_f32(def_glob_vy_map))

    # Optional target as a channel (usually better to keep separate)
    y = None
    if pass_end_onehot is not None:
        y = pass_end_onehot.to(torch.uint8)  # keep binary as uint8 target
        if include_target:
            chs.append(y.to(dtype))  # also as feature if you want (not typical)

    # Stack to (N, C, H, W) then permute -> (C, N, H, W)
    X = torch.stack(chs, dim=1).permute(1, 0, 2, 3)  # (C, N, H, W)
    return (X, y) if pass_end_onehot is not None else (X, None)


def save_shard(X_parts, shard_id, channels):
    # X_parts: list of (C, N_i, H, W) float tensors
    X = torch.cat(X_parts, dim=1).contiguous()   # (C, N_total, H, W)
    path = os.path.join(OUT_DIR, f"features_shard_{shard_id:04d}.pt")
    torch.save({"X": X.cpu(), "channels": channels, "H": H, "W": W}, path)
    return path, X.shape[1]

def build_all_shards(game_ids, api, max_frames_per_shard=20000):
    manifest = {
        "shards": [],
        "channels": [
            "dist_to_ball", "sin_to_ball", "cos_to_ball",
            "pos_team_onehot", "def_team_onehot",
            "sin_to_goal(105,34)", "cos_to_goal(105,34)", "dist_to_goal(105,34)",
            "ball_vx", "ball_vy",
            "att_glob_vx", "att_glob_vy",
            "def_glob_vx", "def_glob_vy",
        ],
        "H": H,
        "W": W,
        "dtype": "float32",
    }

    shard_id, frames_in_shard = 0, 0
    X_parts = []  # accumulate (C, N_i, H, W)

    for gid in tqdm(game_ids, desc="Writing shards"):
        events = api.events(game_id=gid, load_360=True)
        fidelity = spadl.statsbomb._infer_xy_fidelity_versions(events)
        home_team_id = events[0:1].team_id.item()

        spadl_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
        spadl_actions = spadl.add_names(spadl_actions)
        spadl_actions_l2r = spadl.play_left_to_right(spadl_actions, home_team_id=home_team_id)

        # Build freeze-frame list for *all* actions (not just passes)
        freeze_frames_all = []
        for eid in spadl_actions_l2r.original_event_id.tolist():
            idx = events.index[events.event_id == eid]
            if len(idx) == 0:
                freeze_frames_all.append(None)
            else:
                freeze_frames_all.append(events.freeze_frame_360.iloc[idx[0]])

        # All passes for this game
        spadl_passes_all = spadl_actions_l2r[(spadl_actions_l2r.type_id == 0).tolist()]
        if len(spadl_passes_all) == 0:
            continue

        three_sixty_passes_all = []
        for eid in spadl_passes_all.original_event_id.tolist():
            idx = events.index[events.event_id == eid]
            if len(idx) == 0:
                three_sixty_passes_all.append(None)
            else:
                three_sixty_passes_all.append(events.freeze_frame_360.iloc[idx[0]])      

        # Identify which passes have valid 360 frames
        has_360_mask = np.array([
            isinstance(ff, list) and ff is not None and len(ff) > 0
            for ff in three_sixty_passes_all
        ], dtype=bool)

        # --- FILTER: keep all misses, and only those successes with a teammate within 10m ---
        success_mask = pass_success_mask(spadl_passes_all)          # True if pass is successful
        within10_mask = passes_within_radius_mask(                  # True if teammate within 10m of pass end
            spadl_passes_all, three_sixty_passes_all, home_team_id,
            fidelity_version=fidelity, radius_m=3.0
        )

        # keep all unsuccessful passes, OR successful passes that are within 10m radius
        
        keep_mask = (~success_mask) | (success_mask & within10_mask)
        if keep_mask.sum() == 0:
            continue

        # Final mask: keep_mask AND valid 360 frame
        final_mask = keep_mask & has_360_mask

        if final_mask.sum() == 0:
            continue

        # Apply final mask consistently
        spadl_passes = spadl_passes_all[final_mask].reset_index(drop=True)
        three_sixty = [ff for ff, keep in zip(three_sixty_passes_all, final_mask) if keep]

        # freeze-frames (players) for only kept passes: static team occupancy maps
        pos_team_states, def_team_states = transform_freeze_frame(
            three_sixty, is_away=False, fidelity_version=fidelity, method="nearest"
        )
        N = pos_team_states.shape[0]
        if N == 0:
            continue

        # --- Basic geometry: ball location at pass start ---
        pass_start_loc = np.array(
            [spadl_passes.start_x.values, spadl_passes.start_y.values]
        )  # (2, N) in SPADL meters
        distance_maps = point_distance_map_batch(pass_start_loc, normalized=False)
        sin_ball_map, cos_ball_map = angle_sin_cos_map_batch(pass_start_loc)

        # --- Goal maps (broadcast over N frames) ---
        sg, cg = angle_sin_cos_map(105, 34)
        gd = point_distance_map(105, 34)
        sin_goal_map = sg.unsqueeze(0).repeat(N, 1, 1)
        cos_goal_map = cg.unsqueeze(0).repeat(N, 1, 1)
        goal_dist_map = gd.unsqueeze(0).repeat(N, 1, 1)

        # --- Ball velocity (displacement / duration or Δt between actions) ---
        vx_ball, vy_ball = compute_ball_velocity(spadl_passes)
        ball_vx_map = expand_global_scalar_to_map(vx_ball, H=H, W=W, dtype=torch.float32)
        ball_vy_map = expand_global_scalar_to_map(vy_ball, H=H, W=W, dtype=torch.float32)

        

        # --- Glob velocity from current pass frame -> next same-team 360 event ---
        att_vx, att_vy, def_vx, def_vy = compute_glob_velocity_next_action_360(
            spadl_actions_l2r,
            spadl_passes,
            freeze_frames_all,
            fidelity_version=fidelity,
            dt_max=8.0,
        )

        att_glob_vx_map = expand_global_scalar_to_map(att_vx, H=H, W=W, dtype=torch.float32)
        att_glob_vy_map = expand_global_scalar_to_map(att_vy, H=H, W=W, dtype=torch.float32)
        def_glob_vx_map = expand_global_scalar_to_map(def_vx, H=H, W=W, dtype=torch.float32)
        def_glob_vy_map = expand_global_scalar_to_map(def_vy, H=H, W=W, dtype=torch.float32)

        # --- Pack features into (C, N_i, H, W) ---
        Xg, _ = build_feature_tensor(
            distance_maps=distance_maps,
            sin_ball_map=sin_ball_map,
            cos_ball_map=cos_ball_map,
            pos_team_states=pos_team_states,
            def_team_states=def_team_states,
            sin_goal_map=sin_goal_map,
            cos_goal_map=cos_goal_map,
            goal_dist_map=goal_dist_map,
            pass_end_onehot=None,
            include_target=False,
            ball_vx_map=ball_vx_map,
            ball_vy_map=ball_vy_map,
            att_glob_vx_map=att_glob_vx_map,
            att_glob_vy_map=att_glob_vy_map,
            def_glob_vx_map=def_glob_vx_map,
            def_glob_vy_map=def_glob_vy_map,
            dtype=torch.float32,
        )
        Xg = Xg.to(DTYPE)  # (C, N_i, H, W)

        # --- shard control ---
        C, N_i, _, _ = Xg.shape
        X_parts.append(Xg)
        frames_in_shard += N_i

        if frames_in_shard >= max_frames_per_shard:
            path, n = save_shard(X_parts, shard_id, manifest["channels"])
            manifest["shards"].append({"path": path, "frames": int(n)})
            shard_id += 1
            frames_in_shard = 0
            X_parts = []

    # flush remainder
    if X_parts:
        path, n = save_shard(X_parts, shard_id, manifest["channels"])
        manifest["shards"].append({"path": path, "frames": int(n)})

    # write manifest
    with open(os.path.join(OUT_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest['shards'])} shards to {OUT_DIR}")


def build_target_shards(game_ids, api, max_frames_per_shard=20000, method="floor"):
    """
    Saves shards of targets as int tensors:
      targets[n] = [x_idx, y_idx, outcome], where outcome ∈ {0,1}
    Indexing/bucketing matches your features pipeline.
    """
    schema = ["x", "y", "outcome"]
    manifest = {"shards": [], "schema": schema, "H": H, "W": W, "quant_method": method}

    shard_id, frames_in_shard = 0, 0
    target_rows = []   # list of (N_i, 3) tensors per game
    shard_index = []   # (game_id, start, end) inside this shard
    n_so_far_in_shard = 0

    for gid in tqdm(game_ids, desc="Writing target shards"):
        events = api.events(game_id=gid, load_360=True)
        fidelity = spadl.statsbomb._infer_xy_fidelity_versions(events)
        home_team_id = events[0:1].team_id.item()

        spadl_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
        spadl_actions = spadl.add_names(spadl_actions)
        spadl_actions_l2r = spadl.play_left_to_right(spadl_actions, home_team_id=home_team_id)

        spadl_passes_all = spadl_actions_l2r[(spadl_actions_l2r.type_id == 0).tolist()]
        if len(spadl_passes_all) == 0:
            continue

        three_sixty_all = []
        for eid in spadl_passes_all.original_event_id.tolist():
            idx = events.index[events.event_id == eid]
            if len(idx) == 0:
                three_sixty_all.append(None)
            else:
                three_sixty_all.append(events.freeze_frame_360.iloc[idx[0]])

        # Same keep logic as features: all failures + successes within 7m
        success_mask = pass_success_mask(spadl_passes_all)
        within10_mask = passes_within_radius_mask(
            spadl_passes_all, three_sixty_all, home_team_id,
            fidelity_version=fidelity, radius_m=3.0,
        )
        keep_mask = (~success_mask) | (success_mask & within10_mask)

        # New: require valid 360 frame, exactly like feature shards
        has_360_mask = np.array([
            isinstance(ff, list) and ff is not None and len(ff) > 0
            for ff in three_sixty_all
        ], dtype=bool)

        final_mask = keep_mask & has_360_mask
        if final_mask.sum() == 0:
            continue

        spadl_passes = spadl_passes_all[final_mask].reset_index(drop=True)


        # end locations
        end_xy = np.array(
            [spadl_passes.end_x.values, spadl_passes.end_y.values]
        )  # (2, N_i)

        # outcomes: success -> 1, everything else -> 0
        if "result_name" in spadl_passes.columns:
            outc = (spadl_passes.result_name.values == "success").astype(np.uint8)
        else:
            outc = (spadl_passes.result_id.values == 1).astype(np.uint8)

        xs, ys = end_xy[0], end_xy[1]
        valid = np.isfinite(xs) & np.isfinite(ys)
        if not valid.any():
            continue

        xs_v = xs[valid]
        ys_v = ys[valid]
        outc_v = outc[valid]

        if method == "nearest":
            xi = np.rint(xs_v).astype(np.int64)
            yi = np.rint(ys_v).astype(np.int64)
        else:
            xi = np.floor(xs_v).astype(np.int64)
            yi = np.floor(ys_v).astype(np.int64)
        xi = np.clip(xi, 0, H - 1)
        yi = np.clip(yi, 0, W - 1)

        Ti = torch.empty((xi.shape[0], 3), dtype=torch.int16)
        Ti[:, 0] = torch.from_numpy(xi.astype(np.int16))
        Ti[:, 1] = torch.from_numpy(yi.astype(np.int16))
        Ti[:, 2] = torch.from_numpy(outc_v.astype(np.int16))

        # append to shard accumulators
        target_rows.append(Ti)
        start = n_so_far_in_shard
        end = start + Ti.shape[0]
        shard_index.append((int(gid), int(start), int(end)))
        n_so_far_in_shard = end
        frames_in_shard += Ti.shape[0]

        # flush shard if needed
        if frames_in_shard >= max_frames_per_shard:
            path, n = _save_target_shard(target_rows, shard_id, schema, shard_index)
            if path is not None:
                manifest["shards"].append({"path": path, "frames": int(n)})
            shard_id += 1
            frames_in_shard = 0
            n_so_far_in_shard = 0
            target_rows = []
            shard_index = []

    # flush remainder
    if target_rows:
        path, n = _save_target_shard(target_rows, shard_id, schema, shard_index)
        if path is not None:
            manifest["shards"].append({"path": path, "frames": int(n)})

    # manifest
    with open(os.path.join(TARGET_OUT_DIR, "targets_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved {len(manifest['shards'])} target shards to {TARGET_OUT_DIR}")


# ------------ run & save ------------
if __name__ == "__main__":
    api = StatsBombLoader(getter="local", root="../data/open-data-master/data")
    game_ids = np.load("../data/game_ids.npy")

    """payload = accumulate_dataset(game_ids, api)
    torch.save(payload, SAVE_PATH)
    print(f"Saved {payload['X'].shape} to {OUT_DIR}")"""
    #build_all_shards(game_ids,api)
    build_target_shards(game_ids, api, max_frames_per_shard=20000, method="floor")

    
        
