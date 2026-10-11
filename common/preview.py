"""Quick visual check of a stage: first / middle / last frame side by side.

    python -m common.preview data/tos_08_3a mask
    python -m common.preview data/tos_08_3a alpha
    python -m common.preview data/tos_08_3a depth_raw     (also depth, depth_final)

Writes data/<shot>/_preview/<what>.jpg
"""
import sys
from pathlib import Path

import cv2
import numpy as np

from common.io import ShotPaths, frame_name, list_indices, read_alpha, read_depth, read_mask, read_rgb


def colorize_depth(d, lo, hi):
    x = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return cv2.cvtColor(cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def render(paths: ShotPaths, what: str, i: int, depth_range=None) -> np.ndarray:
    rgb = read_rgb(paths.frames / frame_name(i, "png")).astype(np.float32)
    if what == "mask":
        m = read_mask(paths.mask / frame_name(i, "png"))[..., None]
        out = rgb * (1 - 0.5 * m) + np.array([0, 255, 0]) * 0.5 * m   # green overlay
    elif what == "alpha":
        a = read_alpha(paths.alpha / frame_name(i, "png"))[..., None]
        out = rgb * a + np.array([255, 0, 255]) * (1 - a)              # object over magenta
    else:
        out = colorize_depth(read_depth(getattr(paths, what) / frame_name(i, "npy")), *depth_range)
    return np.clip(out, 0, 255).astype(np.uint8)


def preview(root, what, width=640):
    paths = ShotPaths(Path(root))
    ext = "npy" if what.startswith("depth") else "png"
    idx = list_indices(getattr(paths, what), ext)
    picks = sorted({idx[0], idx[len(idx) // 2], idx[-1]})
    depth_range = None
    if ext == "npy":
        # one colour range for all picked frames, so flicker is visible
        ds = [read_depth(getattr(paths, what) / frame_name(i, "npy")) for i in picks]
        depth_range = (min(np.percentile(d, 1) for d in ds), max(np.percentile(d, 99) for d in ds))
    tiles = []
    for i in picks:
        t = render(paths, what, i, depth_range)
        t = cv2.resize(t, (width, int(t.shape[0] * width / t.shape[1])))
        cv2.putText(t, f"{what} {i:04d}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        tiles.append(t)
    out = paths.root / "_preview" / f"{what}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(np.hstack(tiles), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out)
    return out


if __name__ == "__main__":
    preview(sys.argv[1], sys.argv[2])
