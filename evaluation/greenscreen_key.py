"""Reference alpha for a green-screen shot (tos_08_3a) by classic colour-difference keying.

Why: to score SAM 2 and MatAnyone on REAL film footage we need a "true"
alpha. On a green screen, a keyer can compute it from colour alone:
  green excess  e = G - max(R, B)     (big on the screen, ~0 or negative on skin/cloth)
  alpha         = 1 - clip((e - lo) / (hi - lo), 0, 1)
lo/hi are measured on the shot itself: hi = typical green excess of the screen,
lo = a small value (pure foreground). Done on the LINEAR camera EXR (no
display curve), which is how compositors key.

Limits (say so in the report): a keyer is not perfect ground truth. Green
spill on the hair and the dark green board on the right are the weak spots,
so we only score pixels inside a generous region around the actor (the SAM 2
mask grown by `grow` px); outside it, reference alpha = 0.

    python -m evaluation.greenscreen_key data/tos_08_3a      -> data/tos_08_3a/alpha_key/
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from common.get_tears_of_steel import read_exr_rgb
from common.io import frame_name, list_indices, read_mask, write_alpha


def key_frame(rgb_lin, lo, hi):
    e = rgb_lin[..., 1] - np.maximum(rgb_lin[..., 0], rgb_lin[..., 2])
    return 1.0 - np.clip((e - lo) / (hi - lo), 0, 1)


def make_reference(shot_root, grow=60):
    root = Path(shot_root)
    src = root / "source_exr"
    idx = list_indices(src, "exr")
    out = root / "alpha_key"
    out.mkdir(exist_ok=True)

    # measure the screen's green excess on frame 1, away from the actor
    rgb0 = read_exr_rgb(src / frame_name(idx[0], "exr"))
    m0 = read_mask(root / "mask" / frame_name(idx[0], "png"))
    far = cv2.dilate(m0.astype(np.uint8), np.ones((151, 151), np.uint8)) == 0
    e0 = rgb0[..., 1] - np.maximum(rgb0[..., 0], rgb0[..., 2])
    hi = float(np.percentile(e0[far], 25))       # most of the screen is at least this green
    lo = 0.15 * hi                                # below this = fully foreground
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    for i in idx:
        a = key_frame(read_exr_rgb(src / frame_name(i, "exr")), lo, hi)
        region = cv2.dilate(read_mask(root / "mask" / frame_name(i, "png")).astype(np.uint8), k) > 0
        a[~region] = 0.0
        write_alpha(out / frame_name(i, "png"), a)
    print(f"[key] screen green excess hi={hi:.4f}, lo={lo:.4f}; wrote {len(idx)} reference mattes to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shot_root")
    ap.add_argument("--grow", type=int, default=60)
    a = ap.parse_args()
    make_reference(a.shot_root, a.grow)
