import os, json, shutil
import numpy as np

def export_memmap_range(
    src_split_root: str,      # e.g. "test_data/val"
    dst_split_root: str,      # e.g. "test_data_subset/val"
    shard_id: int,
    start: int,               # inclusive local index within shard
    end: int,                 # exclusive local index within shard
    *,
    out_prefix: str = "subset_shard0000",
):
    os.makedirs(dst_split_root, exist_ok=True)

    man_path = os.path.join(src_split_root, "manifest.json")
    with open(man_path, "r") as f:
        man = json.load(f)
    assert man.get("format") == "memmap_v1", f"Unexpected format: {man.get('format')}"

    C, H, W = int(man["C"]), int(man["H"]), int(man["W"])
    channels = list(man.get("channels", []))
    shards = list(man["shards"])
    assert 0 <= shard_id < len(shards), "shard_id out of range"

    s = shards[shard_id]
    n_src = int(s["n"])
    assert 0 <= start < end <= n_src, f"range must satisfy 0 <= start < end <= {n_src}"

    # Source memmaps
    src_x_path = os.path.join(src_split_root, s["x_path"])
    src_t_path = os.path.join(src_split_root, s["t_path"])
    X_src = np.memmap(src_x_path, mode="r", dtype=np.float16, shape=(n_src, C, H, W))
    T_src = np.memmap(src_t_path, mode="r", dtype=np.float32, shape=(n_src, 3))

    # Destination memmaps (one new shard)
    n_out = end - start
    out_x_name = f"{out_prefix}_X.f16.mmap"
    out_t_name = f"{out_prefix}_T.f32.mmap"
    dst_x_path = os.path.join(dst_split_root, out_x_name)
    dst_t_path = os.path.join(dst_split_root, out_t_name)

    X_dst = np.memmap(dst_x_path, mode="w+", dtype=np.float16, shape=(n_out, C, H, W))
    T_dst = np.memmap(dst_t_path, mode="w+", dtype=np.float32, shape=(n_out, 3))

    # Copy slice (reads/writes only that slice)
    X_dst[:] = X_src[start:end]
    T_dst[:] = T_src[start:end]
    X_dst.flush()
    T_dst.flush()

    # New manifest for the mini dataset
    out_manifest = {
        "format": "memmap_v1",
        "C": C, "H": H, "W": W,
        "channels": channels,
        "shards": [
            {
                "x_path": out_x_name,
                "t_path": out_t_name,
                "n": n_out,
                "source": {
                    "src_split_root": os.path.abspath(src_split_root),
                    "shard_id": int(shard_id),
                    "range": [int(start), int(end)],
                },
            }
        ],
    }

    with open(os.path.join(dst_split_root, "manifest.json"), "w") as f:
        json.dump(out_manifest, f, indent=2)

    print(f"[ok] wrote subset split to: {dst_split_root}")
    print(f"     examples: {n_out} (from shard {shard_id} [{start}:{end}])")
    print(f"     files: {out_x_name}, {out_t_name}, manifest.json")


# ---- Example usage: take local indices 200..300 from a single shard in val/ ----
export_memmap_range(
    src_split_root="test_data/val",
    dst_split_root="test_data_subset/val",
    shard_id=0,     # pick the shard you want
    start=450,
    end=500,
)
