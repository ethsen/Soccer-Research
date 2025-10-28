
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
OUT_DIR = "soccer_shards"
os.makedirs(OUT_DIR, exist_ok=True)
DTYPE = torch.float16        # inputs only (targets separate later)
FL, FW = 105.0, 68.0  # meters
H, W = 105, 68

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
    Convert a list of 360 freeze-frames into stacked one-hot maps.
    Each frame -> two grids (possession / defending) of shape (105, 68) with 0/1.
    Returns:
        pos_stack: (N, 105, 68) uint8
        def_stack: (N, 105, 68) uint8
    """
    out_pos, out_def = [], []

    nan_track = []

    for three_sixty in frames:
        # Skip NaN / None frames
        if three_sixty is None or (isinstance(three_sixty, float) and not np.isfinite(three_sixty)):
            nan_track.append(0)
            continue
        if not isinstance(three_sixty, list):
            nan_track.append(0)
            continue
        nan_track.append(1)
        pos_team = torch.zeros((FL, FW), dtype=torch.uint8)  # CPU on purpose (dataset)
        def_team = torch.zeros((FL, FW), dtype=torch.uint8)

        for p in three_sixty:
            loc = p.get("location", None)
            if not loc or len(loc) < 2:
                continue
            x, y = loc

            # For 360 floats: no half-cell shift
            xm, ym = sb_to_spadl_xy(x, y, fidelity_version, assume_cell_center=False)
            xm, ym = ltr_flip_if_away(xm, ym, is_away)

            # bucket to integer cell
            xi, yi, ok = bucket_xy_to_idx(xm, ym, H=FL, W=FW, method=method)
            if not ok:
                continue

            # write into the right grid
            if bool(p.get("teammate", False)):
                pos_team[xi, yi] = 1
            else:
                def_team[xi, yi] = 1

        # append this frame’s tensors
        out_pos.append(pos_team)
        out_def.append(def_team)

    if len(out_pos) == 0:
        # no valid frames
        return (torch.empty((0, FL, FW), dtype=torch.uint8),
                torch.empty((0, FL, FW), dtype=torch.uint8))

    # stack to (N, 105, 68)
    pos_stack = torch.stack(out_pos, dim=0)
    def_stack = torch.stack(out_def, dim=0)
    return pos_stack, def_stack,nan_track


def save_ids(api):
    count = 0
    game_ids = []
    for entry in os.scandir("open-data-master/data/three-sixty"):
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
    distance_maps,          # (N, H, W) float
    sin_ball_map,           # (N, H, W) float
    cos_ball_map,           # (N, H, W) float
    pos_team_states=None,   # (N, H, W) uint8 or float, optional
    def_team_states=None,   # (N, H, W) uint8 or float, optional
    sin_goal_map=None,      # (N, H, W) float, optional
    cos_goal_map=None,      # (N, H, W) float, optional
    goal_dist_map=None,     # (N, H, W) float, optional
    pass_end_onehot=None,   # (N, H, W) uint8 or float, optional (target can be kept separate)
    include_target=False,   # if True, append pass_end_onehot as a channel
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

    # Optional target as a channel (usually better to keep separate)
    y = None
    if pass_end_onehot is not None:
        y = pass_end_onehot.to(torch.uint8)  # keep binary as uint8 target
        if include_target:
            chs.append(y.to(dtype))  # also as feature if you want (not typical)

    # Stack to (N, C, H, W) then permute -> (C, N, H, W)
    X = torch.stack(chs, dim=1).permute(1, 0, 2, 3)  # (C, N, H, W)
    return (X, y) if pass_end_onehot is not None else (X, None)

def accumulate_dataset(game_ids, api):
    X_parts = []               # list of (C, N_i, H, W)
    index = []                 # (game_id, start, end) on the concatenated N axis
    n_so_far = 0

    for gid in tqdm(game_ids, desc="Building dataset"):
        # ---- per-game pipeline (your code block) ----
        events = api.events(game_id=gid, load_360=True)
        fidelity = spadl.statsbomb._infer_xy_fidelity_versions(events)
        home_team_id = events[0:1].team_id.item()

        spadl_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
        spadl_actions = spadl.add_names(spadl_actions)
        spadl_actions_l2r = spadl.play_left_to_right(spadl_actions, home_team_id=home_team_id)

        spadl_passes = spadl_actions_l2r[(spadl_actions_l2r.type_id == 0).tolist()]
        frame_ids = spadl_passes.original_event_id.tolist()

        three_sixty = events.freeze_frame_360[events["event_id"].isin(frame_ids)].tolist()

        # NOTE: make sure your transform returns (pos, def, nan_track) with shapes (N,H,W)
        pos_team_states, def_team_states, nan_track = transform_freeze_frame(
            three_sixty, is_away=False, fidelity_version=fidelity
        )

        # Filter pass lists to frames we actually converted
        mask = np.array(nan_track, dtype=bool)
        if mask.sum() == 0:
            continue

        # Build per-frame features (all are (N,H,W))
        pass_start_loc = np.array([spadl_passes.start_x.tolist(), spadl_passes.start_y.tolist()])[:, mask]
        distance_maps = point_distance_map_batch(pass_start_loc, normalized=False)
        sin_ball_map, cos_ball_map = angle_sin_cos_map_batch(pass_start_loc)

        # Goal geometry (broadcast once per game)
        sg, cg = angle_sin_cos_map(105, 34)     # (H,W)
        gd = point_distance_map(105, 34)        # (H,W)
        N = pos_team_states.shape[0]
        sin_goal_map = sg.unsqueeze(0).repeat(N, 1, 1)
        cos_goal_map = cg.unsqueeze(0).repeat(N, 1, 1)
        goal_dist_map = gd.unsqueeze(0).repeat(N, 1, 1)

        # Target kept for later (don’t include in X now)
        pass_end_loc = np.array([spadl_passes.end_x.tolist(), spadl_passes.end_y.tolist()])[:, mask]
        y_onehot = one_hot_from_xy_batch(pass_end_loc, method="floor")   # (N,H,W)
        # (OPTIONAL) save y separately elsewhere

        # Stack channels -> (C, N, H, W)
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
            dtype=torch.float32
        )
        # cast to save space
        Xg = Xg.to(DTYPE)   # (C,N,H,W)

        # record slice + append
        C, N_i, _, _ = Xg.shape
        X_parts.append(Xg)
        index.append((int(gid), int(n_so_far), int(n_so_far + N_i)))
        n_so_far += N_i

    if len(X_parts) == 0:
        raise RuntimeError("No valid frames found across provided game_ids.")

    # concat along N (dim=1) → (C, N_total, H, W)
    X = torch.cat(X_parts, dim=1)

    # channel names (keep in same order as build_feature_tensor)
    channels = [
        "dist_to_ball",
        "sin_to_ball",
        "cos_to_ball",
        "pos_team_onehot",
        "def_team_onehot",
        "sin_to_goal(105,34)",
        "cos_to_goal(105,34)",
        "dist_to_goal(105,34)",
    ]

    payload = {
        "X": X.cpu(),            # (C, N_total, H, W) float16
        "index": index,          # list of (game_id, start, end)
        "channels": channels,
        "H": H, "W": W, "dtype": str(DTYPE)
    }
    return payload


def save_shard(X_parts, shard_id, channels):
    # X_parts: list of (C, N_i, H, W) float tensors
    X = torch.cat(X_parts, dim=1).contiguous()   # (C, N_total, H, W)
    path = os.path.join(OUT_DIR, f"features_shard_{shard_id:04d}.pt")
    torch.save({"X": X.cpu(), "channels": channels, "H": H, "W": W}, path)
    return path, X.shape[1]

def build_all_shards(game_ids, api, max_frames_per_shard=20000):
    manifest = {"shards": [], "channels": [
        "dist_to_ball","sin_to_ball","cos_to_ball",
        "pos_team_onehot","def_team_onehot",
        "sin_to_goal(105,34)","cos_to_goal(105,34)","dist_to_goal(105,34)"
    ], "H": H, "W": W, "dtype": "float16"}

    shard_id, frames_in_shard = 0, 0
    X_parts = []  # accumulate (C, N_i, H, W)

    for gid in tqdm(game_ids, desc="Writing shards"):
        # --- your per-game block (trimmed) ---
        events = api.events(game_id=gid, load_360=True)
        fidelity = spadl.statsbomb._infer_xy_fidelity_versions(events)
        home_team_id = events[0:1].team_id.item()

        spadl_actions = spadl.statsbomb.convert_to_actions(events, home_team_id=home_team_id)
        spadl_actions = spadl.add_names(spadl_actions)
        spadl_actions_l2r = spadl.play_left_to_right(spadl_actions, home_team_id=home_team_id)

        spadl_passes = spadl_actions_l2r[(spadl_actions_l2r.type_id == 0).tolist()]
        frame_ids = spadl_passes.original_event_id.tolist()
        three_sixty = events.freeze_frame_360[events["event_id"].isin(frame_ids)].tolist()

        pos_team_states, def_team_states, nan_track = transform_freeze_frame(
            three_sixty, is_away=False, fidelity_version=fidelity
        )

        mask = np.array(nan_track, dtype=bool)
        if mask.sum() == 0:
            continue

        pass_start_loc = np.array([spadl_passes.start_x.tolist(), spadl_passes.start_y.tolist()])[:, mask]
        distance_maps = point_distance_map_batch(pass_start_loc, normalized=False)
        sin_ball_map, cos_ball_map = angle_sin_cos_map_batch(pass_start_loc)

        # goal maps (broadcast to N)
        sg, cg = angle_sin_cos_map(105, 34)       # (H,W)
        gd = point_distance_map(105, 34)          # (H,W)
        N = pos_team_states.shape[0]
        sin_goal_map = sg.unsqueeze(0).repeat(N,1,1)
        cos_goal_map = cg.unsqueeze(0).repeat(N,1,1)
        goal_dist_map = gd.unsqueeze(0).repeat(N,1,1)

        # pack features (no targets yet)
        Xg, _ = build_feature_tensor(
            distance_maps, sin_ball_map, cos_ball_map,
            pos_team_states, def_team_states,
            sin_goal_map, cos_goal_map, goal_dist_map,
            pass_end_onehot=None, include_target=False, dtype=torch.float32
        )
        Xg = Xg.to(DTYPE)  # (C, N_i, H, W)

        # shard control
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

# ------------ run & save ------------
if __name__ == "__main__":
    api = StatsBombLoader(getter="local", root="open-data-master/data")
    game_ids = np.load("game_ids.npy")

    """payload = accumulate_dataset(game_ids, api)
    torch.save(payload, SAVE_PATH)
    print(f"Saved {payload['X'].shape} to {OUT_DIR}")"""
    build_all_shards(game_ids,api)

    
        
