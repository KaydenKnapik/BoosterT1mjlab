"""Trim an NPZ motion file to a frame range and save as a new file.

Usage:
    uv run python src/booster_t1_mjlab/scripts/trim_motion.py \
        src/booster_t1_mjlab/tasks/kick/motions/kick_06.npz \
        --start 38 --end 90 \
        --out src/booster_t1_mjlab/tasks/kick/motions/kick_06_close.npz
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npz", help="Input NPZ file")
    parser.add_argument("--start", type=int, default=0, help="Start frame (inclusive)")
    parser.add_argument("--end", type=int, default=-1, help="End frame (exclusive, -1 = all)")
    parser.add_argument("--out", type=str, default=None, help="Output path (default: <name>_trimmed.npz)")
    args = parser.parse_args()

    src = Path(args.npz)
    data = np.load(src)
    fps = int(np.asarray(data["fps"]).flat[0])
    T = data["joint_pos"].shape[0]

    start = max(0, args.start)
    end = T if args.end < 0 else min(args.end, T)
    print(f"Trimming {src.name}: frames {start}–{end-1} ({end-start} frames, {(end-start)/fps:.2f}s)")

    out_path = Path(args.out) if args.out else src.with_name(src.stem + "_trimmed.npz")
    arrays = {k: v for k, v in data.items()}
    for key in ["joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]:
        if key in arrays and arrays[key].ndim >= 2:
            arrays[key] = arrays[key][start:end]

    np.savez(out_path, **arrays)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
