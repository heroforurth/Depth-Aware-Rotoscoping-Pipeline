"""Demo video from the final EXRs: what a compositor can do with R, G, B, A, Z.

    python -m demo.make_demo data/tos_01_2a          -> data/tos_01_2a/demo.mp4

Four panels:
  1. original plate
  2. the cut-out (A channel) over a checkerboard
  3. the Z channel (bright = near)
  4. DEPTH-AWARE INSERT: a CG banner placed BETWEEN the actor and the
     background. It is hidden by the actor (thanks to A) but covers the
     background behind it (thanks to Z), plus a little depth fog.
     The banner depth comes from the actor's Z stats in the EXR header:
     a bit farther than the actor's 98th percentile Z.

Everything is read back from the EXR files, so the demo also proves that
the deliverable works.
"""
import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import OpenEXR
from tqdm import tqdm

from common.io import list_indices

NAMES = [("R", "G", "B", "A", "Z"),
         ("ViewLayer.Combined.R", "ViewLayer.Combined.G", "ViewLayer.Combined.B",
          "ViewLayer.Combined.A", "ViewLayer.Depth.Z")]


def read_exr(path):
    with OpenEXR.File(str(path), separate_channels=True) as f:
        ch, hdr = f.channels(), f.header()
        names = next(n for n in NAMES if all(c in ch for c in n))
        r, g, b, a, z = (ch[c].pixels.astype(np.float32) for c in names)
        return np.dstack([r, g, b]), a, z, dict(hdr)


def to_srgb8(lin):
    x = np.clip(lin, 0, 1)
    return (np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055) * 255).astype(np.uint8)


def checker(h, w, s=32):
    yy, xx = np.mgrid[0:h, 0:w]
    return np.where(((yy // s) + (xx // s)) % 2 == 0, 0.35, 0.6).astype(np.float32)[..., None]


def banner(h, w, t):
    """A glowing text strip that slides across the frame; returns (rgb, coverage)."""
    layer = np.zeros((h, w), np.uint8)
    x = int((-0.6 + 1.6 * t) * w)
    cv2.putText(layer, "DEPTH-AWARE ROTO", (x, int(h * 0.5)), cv2.FONT_HERSHEY_DUPLEX,
                h / 220, 255, max(2, h // 120), cv2.LINE_AA)
    cv2.rectangle(layer, (x - h // 40, int(h * 0.38)), (x + int(w * 0.75), int(h * 0.56)), 120, max(2, h // 200))
    cov = cv2.GaussianBlur(layer.astype(np.float32) / 255, (0, 0), 1.0)
    glow = cv2.GaussianBlur(cov, (0, 0), h / 100)
    cov = np.clip(cov + 0.6 * glow, 0, 1)
    color = np.array([0.2, 0.9, 1.6], np.float32)       # linear cyan, slightly over 1 = "glow"
    return color * np.ones((h, w, 3), np.float32), cov


def label(img, text):
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 2, cv2.LINE_AA)
    return img


def make_demo(shot_root, out=None, fps=24, width=1920):
    root = Path(shot_root)
    idx = list_indices(root / "exr", "exr")
    if not idx:
        raise FileNotFoundError(f"no EXRs in {root / 'exr'}; run the pipeline first")
    out = Path(out or root / "demo.mp4")

    # one banner depth for the whole shot: behind the actor, in front of the background
    _, _, z0, hdr0 = read_exr(root / "exr" / f"{idx[len(idx) // 2]:04d}.exr")
    actor_far = float(hdr0.get("Z_p98", np.percentile(z0, 30)))
    z_banner = actor_far * 1.15
    z_lo, z_hi = np.percentile(z0, [1, 99])

    writer = imageio.get_writer(str(out), fps=fps, codec="libx264", quality=8, macro_block_size=8)
    for n, i in enumerate(tqdm(idx, desc="demo")):
        rgb, a, z, _ = read_exr(root / "exr" / f"{i:04d}.exr")
        h, w = a.shape
        A = a[..., None]
        p1 = to_srgb8(rgb)
        p2 = to_srgb8(rgb * A + checker(h, w) * (1 - A))
        zn = 1 - np.clip((z - z_lo) / (z_hi - z_lo), 0, 1)                    # near = bright
        p3 = cv2.cvtColor(cv2.applyColorMap((zn * 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

        # depth-aware insert: banner over the background only where the background is farther,
        # then the actor (A) on top. Z is never multiplied by A: soft edges stay correct.
        b_rgb, b_cov = banner(h, w, n / max(len(idx) - 1, 1))
        b_cov = b_cov * (z > z_banner)
        fog = np.clip((z - actor_far) / (z_hi - actor_far), 0, 1)[..., None] * 0.35
        bg = rgb * (1 - fog) + np.array([0.55, 0.6, 0.7], np.float32) * fog
        bg = b_rgb * b_cov[..., None] + bg * (1 - b_cov[..., None])
        p4 = to_srgb8(rgb * A + bg * (1 - A))

        grid = np.vstack([np.hstack([label(p1, "plate"), label(p2, "A: matte")]),
                          np.hstack([label(p3, "Z: depth"), label(p4, "insert behind actor (A + Z)")])])
        s = width / grid.shape[1]
        grid = cv2.resize(grid, (width, int(round(grid.shape[0] * s / 2)) * 2), interpolation=cv2.INTER_AREA)
        writer.append_data(grid)
    writer.close()
    print(f"[demo] wrote {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("shot_root")
    ap.add_argument("--out")
    a = ap.parse_args()
    make_demo(a.shot_root, a.out)
