# avg_players_captured.py
import os, json
from collections import OrderedDict
import numpy as np
import torch
from tqdm import tqdm

class MemmapShard:
    def __init__(self, root_dir: str, x_name: str, t_name: str, n: int, C: int, H: int, W: int):
        self.n = int(n)
        self.C, self.H, self.W = int(C), int(H), int(W)
        self.X = np.memmap(os.path.join(root_dir, x_name), mode="r", dtype=np.float16,
                           shape=(self.n, self.C, self.H, self.W))
        self.T = np.memmap(os.path.join(root_dir, t_name), mode="r", dtype=np.float32,
                           shape=(self.n, 3))

class MemmapManifest:
    def __init__(self, root_dir: str, cache_size: int = 2):
        self.root_dir = root_dir
        self.cache_size = int(cache_size)

        with open(os.path.join(root_dir, "manifest.json"), "r") as f:
            man = json.load(f)
        assert man.get("format") == "memmap_v1"

        self.C = int(man["C"]); self.H = int(man["H"]); self.W = int(man["W"])
        self.channels = list(man.get("channels", []))
        self.shards = list(man["shards"])

        self.starts = []
        cur = 0
        for s in self.shards:
            self.starts.append(cur)
            cur += int(s["n"])
        self.total = cur

        self._cache = OrderedDict()

    def _open_shard(self, shard_id: int) -> MemmapShard:
        shard_id = int(shard_id)
        if shard_id in self._cache:
            self._cache.move_to_end(shard_id)
            return self._cache[shard_id]
        s = self.shards[shard_id]
        mm = MemmapShard(self.root_dir, s["x_path"], s["t_path"], int(s["n"]), self.C, self.H, self.W)
        self._cache[shard_id] = mm
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return mm

    def locate(self, k: int):
        k = int(k)
        lo, hi = 0, len(self.starts) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start = self.starts[mid]
            end = self.starts[mid + 1] if mid + 1 < len(self.starts) else self.total
            if start <= k < end:
                return mid, k - start
            if k < start: hi = mid - 1
            else: lo = mid + 1
        raise RuntimeError("locate failed")

    def load_by_shard_local(self, shard_id: int, local_i: int):
        shard = self._open_shard(shard_id)
        x = torch.from_numpy(np.array(shard.X[local_i], copy=True)).float()  # (C,H,W)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x  # only need x for this analysis


# -------------------------
# Counting logic
# -------------------------
def _find_channel_idx(channels, candidates):
    # exact match, then contains match
    for c in candidates:
        if c in channels:
            return channels.index(c)
    for i, name in enumerate(channels):
        low = name.lower()
        if any(c.lower() in low for c in candidates):
            return i
    return None

def avg_players_for_split(split_root: str, cache_size: int = 2, thresh: float = 0.5, stride: int = 1):
    mm = MemmapManifest(split_root, cache_size=cache_size)

    # try to locate channels by name; fall back to (3,4) since you often used those
    att_idx = _find_channel_idx(mm.channels, ["att_team", "attacking", "in_possession", "att_onehot"])
    def_idx = _find_channel_idx(mm.channels, ["def_team", "defending", "out_possession", "def_onehot"])

    if att_idx is None or def_idx is None:
        # common fallback in your earlier schema
        att_idx = 3 if att_idx is None else att_idx
        def_idx = 4 if def_idx is None else def_idx
        print(f"[warn] channel names not found; using fallback att_idx={att_idx}, def_idx={def_idx}")
    else:
        print(f"[info] using channels: att_idx={att_idx} ({mm.channels[att_idx]}), def_idx={def_idx} ({mm.channels[def_idx]})")

    total_att = 0.0
    total_def = 0.0
    n_used = 0

    # iterate shard-by-shard to keep it fast
    for shard_id, s in enumerate(mm.shards):
        shard = mm._open_shard(shard_id)
        n = int(s["n"])

        for j in range(0, n, stride):
            # read only the two channels from memmap to avoid copying full C,H,W
            # shard.X[j] is (C,H,W) float16; slice channels then convert to torch
            att = torch.from_numpy(np.array(shard.X[j, att_idx], copy=False)).float()
            deff = torch.from_numpy(np.array(shard.X[j, def_idx], copy=False)).float()

            # threshold to convert gaussian/heat into "presence"
            att_count = int((att > thresh).sum().item())
            def_count = int((deff > thresh).sum().item())

            total_att += att_count
            total_def += def_count
            n_used += 1

    avg_att = total_att / max(1, n_used)
    avg_def = total_def / max(1, n_used)
    avg_tot = (total_att + total_def) / max(1, n_used)

    return {
        "split_root": split_root,
        "N_used": n_used,
        "avg_att": avg_att,
        "avg_def": avg_def,
        "avg_total": avg_tot,
        "att_idx": att_idx,
        "def_idx": def_idx,
        "thresh": thresh,
        "stride": stride,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True, help="path containing train/ and val/")
    p.add_argument("--cache_size", type=int, default=2)
    p.add_argument("--thresh", type=float, default=0.5, help="presence threshold for player cells")
    p.add_argument("--stride", type=int, default=1, help="use every stride-th sample (speed)")
    args = p.parse_args()

    train_root = os.path.join(args.data_root, "train")
    val_root   = os.path.join(args.data_root, "val")

    train_stats = avg_players_for_split(train_root, cache_size=args.cache_size, thresh=args.thresh, stride=args.stride)
    val_stats   = avg_players_for_split(val_root,   cache_size=args.cache_size, thresh=args.thresh, stride=args.stride)

    def _print(stats, name):
        print(f"\n[{name}] root={stats['split_root']}")
        print(f"  N_used      : {stats['N_used']}")
        print(f"  att_idx/def : {stats['att_idx']}/{stats['def_idx']}  (thresh={stats['thresh']}, stride={stats['stride']})")
        print(f"  avg_att     : {stats['avg_att']:.2f}")
        print(f"  avg_def     : {stats['avg_def']:.2f}")
        print(f"  avg_total   : {stats['avg_total']:.2f}")

    _print(train_stats, "train")
    _print(val_stats, "val")
