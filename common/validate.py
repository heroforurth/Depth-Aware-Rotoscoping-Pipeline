"""Check that a shot folder follows the interface contract.

    python -m common.validate data/dummy

Checks: 4-digit names, same frame numbers in every folder, same resolution
as frames/, alpha in [0, 1], depth float32 H x W with no NaN/Inf.
Run this after every stage; it catches most integration bugs early.
"""
import re
import sys
from pathlib import Path

import numpy as np

from common.io import ShotPaths, list_indices, read_alpha, read_depth, read_mask, read_rgb

NAME = re.compile(r"^\d{4}\.(png|npy|exr)$")


def validate(root) -> list[str]:
    p = ShotPaths(Path(root))
    errors = []
    frames = list_indices(p.frames, "png")
    if not frames:
        return [f"no frames in {p.frames}"]
    h, w = read_rgb(p.frames / f"{frames[0]:04d}.png").shape[:2]
    print(f"frames: {len(frames)} ({w}x{h}), {frames[0]:04d}..{frames[-1]:04d}")

    checks = [
        ("mask", "png", read_mask, None),
        ("alpha", "png", read_alpha, (0.0, 1.0)),
        ("depth_raw", "npy", read_depth, None),
        ("depth", "npy", read_depth, None),
        ("depth_final", "npy", read_depth, None),
    ]
    for sub, ext, reader, rng in checks:
        folder = getattr(p, sub)
        if not folder.exists() or not any(folder.iterdir()):
            print(f"{sub:12s} (empty, skipped)")
            continue
        bad_names = [f.name for f in folder.iterdir() if not NAME.match(f.name)]
        if bad_names:
            errors.append(f"{sub}: bad file names {bad_names[:3]}")
        idx = list_indices(folder, ext)
        if idx != frames:
            errors.append(f"{sub}: frame numbers differ from frames/ ({len(idx)} vs {len(frames)})")
        # full check on first/middle/last frame keeps it fast
        for i in sorted({idx[0], idx[len(idx) // 2], idx[-1]}) if idx else []:
            a = reader(folder / f"{i:04d}.{ext}")
            if a.shape[:2] != (h, w):
                errors.append(f"{sub}/{i:04d}: shape {a.shape} != frames {(h, w)}")
            if ext == "npy":
                if a.dtype != np.float32 or a.ndim != 2:
                    errors.append(f"{sub}/{i:04d}: need float32 HxW, got {a.dtype} {a.shape}")
                if not np.isfinite(a).all():
                    errors.append(f"{sub}/{i:04d}: NaN or Inf values")
            if rng and (a.min() < rng[0] or a.max() > rng[1]):
                errors.append(f"{sub}/{i:04d}: values outside {rng}")
        print(f"{sub:12s} {len(idx)} files")

    exr = list_indices(p.exr, "exr")
    if exr:
        print(f"{'exr':12s} {len(exr)} files")
        if exr != frames:
            errors.append("exr: frame numbers differ from frames/")
    return errors


if __name__ == "__main__":
    errs = validate(sys.argv[1])
    for e in errs:
        print("ERROR:", e)
    print("OK" if not errs else f"{len(errs)} problem(s)")
    sys.exit(1 if errs else 0)
