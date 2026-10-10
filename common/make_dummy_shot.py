"""Make a synthetic test shot with known ground truth, no GPU needed.

A soft-edged ball moves in front of a tilted background plane. We write:
    frames/     RGB images
    mask/       binary mask (what SAM 2 would give)
    alpha/      soft alpha (what MatAnyone would give)
    depth_raw/  "relative" depth (larger = closer) with a DIFFERENT random
                scale and shift on every frame, to imitate per-frame
                normalisation flicker. The alignment step must undo this.
    gt_depth/   the true, flicker-free depth (for checking alignment)

This lets us test alignment, fusion, refinement and the EXR writer before
the real models run.

    python -m common.make_dummy_shot --shot dummy --frames 30 --height 270
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from common.io import ShotPaths, frame_name, write_alpha, write_depth, write_mask, write_rgb


def make_dummy_shot(root: Path, n_frames=30, height=270, width=None, seed=0):
    rng = np.random.default_rng(seed)
    width = width or height * 16 // 9
    paths = ShotPaths(Path(root))
    paths.make_dirs()
    gt_dir = paths.root / "gt_depth"
    gt_dir.mkdir(exist_ok=True)

    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    # background: plane, closer at the bottom (disparity 0.2 at top -> 0.5 at bottom)
    bg_disp = 0.2 + 0.3 * yy / height
    # texture so the RGB frame has edges for the guided filter
    bg_rgb = np.stack([
        110 + 60 * np.sin(xx / 23.0),
        120 + 50 * np.cos(yy / 17.0),
        140 + 40 * np.sin((xx + yy) / 31.0),
    ], -1)

    r = height * 0.18
    for i in range(1, n_frames + 1):
        t = (i - 1) / max(n_frames - 1, 1)
        cx = width * (0.25 + 0.5 * t)
        cy = height * (0.5 + 0.1 * np.sin(2 * np.pi * t))
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # soft alpha: 1 inside, linear ramp over 3 px at the border
        alpha = np.clip((r - dist) / 3.0 + 0.5, 0, 1).astype(np.float32)
        # ball is a sphere, near (disparity ~0.8-0.9)
        ball_disp = 0.8 + 0.1 * np.sqrt(np.clip(1 - (dist / r) ** 2, 0, 1))
        gt = np.where(alpha > 0.5, ball_disp, bg_disp).astype(np.float32)

        ball_rgb = np.array([220, 70, 60], np.float32)
        rgb = alpha[..., None] * ball_rgb + (1 - alpha[..., None]) * bg_rgb
        rgb = np.clip(rgb + rng.normal(0, 2, rgb.shape), 0, 255).astype(np.uint8)

        # imitate per-frame normalised model output: random scale/shift + blurry edges + noise
        s, b = rng.uniform(0.7, 1.4), rng.uniform(-0.2, 0.2)
        raw = cv2.GaussianBlur(gt, (0, 0), 2.0) * s + b
        raw = raw + rng.normal(0, 0.005, raw.shape).astype(np.float32)

        write_rgb(paths.frames / frame_name(i, "png"), rgb)
        write_mask(paths.mask / frame_name(i, "png"), alpha > 0.5)
        write_alpha(paths.alpha / frame_name(i, "png"), alpha)
        write_depth(paths.depth_raw / frame_name(i, "npy"), raw)
        write_depth(gt_dir / frame_name(i, "npy"), gt)
    print(f"[dummy] wrote {n_frames} frames ({width}x{height}) to {paths.root}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="dummy")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--frames", type=int, default=30)
    ap.add_argument("--height", type=int, default=270)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    make_dummy_shot(Path(a.data_root) / a.shot, a.frames, a.height, seed=a.seed)
