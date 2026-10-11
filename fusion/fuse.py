"""[C] Fusion: alpha/ + depth/ + frames/ -> depth_final/ + stats.json

Two jobs:

1. EDGE-AWARE REFINEMENT (design decision 5)
   Depth models work at low resolution (518 px), so after upsampling the
   depth edge around the actor is blurry and often a few pixels off the
   real edge. A compositor then gets a "halo": background pixels with
   foreground depth (or the opposite).
   Fix: a JOINT BILATERAL FILTER guided by the RGB frame (plus the alpha).
   Each pixel's depth becomes a weighted average of its neighbours, and a
   neighbour only gets a big weight if it is (a) close by and (b) has a
   similar COLOUR (and alpha). So depth is smoothed along the actor and
   along the background, but not across the edge between them, and the
   depth edge snaps to the colour/matte edge.
   We only change pixels in a band of `band_px` around the matte edge; the
   rest of the depth map is left exactly as the model produced it.

2. ALPHA-AWARE Z STATS (design decision 2)
   Per frame: Z min / max / mean, plus robust 2% / 98% percentiles, using
   ONLY pixels where alpha > 0.5 (the object). Given both in pipeline
   units (disparity, larger = closer) and in the EXR's Z units, so a
   compositor can read e.g. the object's depth range straight from
   stats.json or the EXR header.
   Note: Z is never multiplied by alpha (design decision 1).

    python run_pipeline.py --config configs/dummy.yaml --stages fuse
"""
import json

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from common.io import frame_name, list_indices, read_alpha, read_depth, read_rgb, write_depth
from common.timing import timed
from exr_writer.write_exr import disparity_to_z


def joint_bilateral(depth, guide, radius, sigma_space, sigma_color, device):
    """depth: H x W. guide: H x W x C (0-255 scale). Returns filtered H x W.
    Done with torch so it runs on the GPU when there is one (CPU also works)."""
    d = torch.from_numpy(depth).to(device)[None, None]                     # 1x1xHxW
    g = torch.from_numpy(guide.astype(np.float32)).to(device).permute(2, 0, 1)[None]  # 1xCxHxW
    pad = (radius, radius, radius, radius)
    dp, gp = F.pad(d, pad, mode="replicate"), F.pad(g, pad, mode="replicate")
    H, W = depth.shape
    num = torch.zeros_like(d)
    den = torch.zeros_like(d)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ws = np.exp(-(dx * dx + dy * dy) / (2 * sigma_space ** 2))
            gq = gp[:, :, radius + dy:radius + dy + H, radius + dx:radius + dx + W]
            dq = dp[:, :, radius + dy:radius + dy + H, radius + dx:radius + dx + W]
            wc = torch.exp(-((gq - g) ** 2).sum(1, keepdim=True) / (2 * sigma_color ** 2))
            w = ws * wc
            num += w * dq
            den += w
    return (num / den.clamp_min(1e-8))[0, 0].cpu().numpy()


def edge_band_weight(alpha, band_px):
    """1.0 near the matte edge, fading to 0 at band_px away from it."""
    fg = (alpha > 0.5).astype(np.uint8)
    # distance to the edge: inside, distance to the nearest background pixel; outside, to the
    # nearest object pixel. Each pixel is non-zero in only one of the two maps, so we add them.
    dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3) + cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3)
    soft = (alpha > 0.02) & (alpha < 0.98)          # soft alpha (hair) is always "edge"
    w = np.clip((band_px - dist) / (0.25 * band_px), 0, 1)
    w[soft] = 1.0
    return w.astype(np.float32)


def refine(depth, rgb, alpha, rc, device):
    guide = np.concatenate([rgb.astype(np.float32), (alpha * 255 * rc["alpha_weight"])[..., None]], -1)
    filt = depth
    for _ in range(rc["iterations"]):
        filt = joint_bilateral(filt, guide, rc["diameter"] // 2, rc["sigma_space"], rc["sigma_color"], device)
    w = edge_band_weight(alpha, rc["band_px"])
    return (w * filt + (1 - w) * depth).astype(np.float32)


def z_stats(values: np.ndarray, percentiles):
    if values.size == 0:
        return {"min": None, "max": None, "mean": None, **{f"p{p:02d}": None for p in percentiles}, "pixels": 0}
    out = {"min": float(values.min()), "max": float(values.max()), "mean": float(values.mean())}
    for p, v in zip(percentiles, np.percentile(values, percentiles)):
        out[f"p{p:02d}"] = float(v)
    out["pixels"] = int(values.size)
    return out


def run(cfg, paths):
    fc = cfg["fusion"]
    rc = fc["refine"]
    indices = list_indices(paths.depth, "npy")
    if not indices:
        raise FileNotFoundError(f"no depth in {paths.depth}; run the align stage first")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    thr, pcts = fc["alpha_threshold"], fc["percentiles"]

    frames = {}
    with timed("fuse", paths.root, len(indices), {"refine": rc["enabled"]}):
        for i in tqdm(indices, desc="fuse"):
            depth = read_depth(paths.depth / frame_name(i, "npy"))
            alpha = read_alpha(paths.alpha / frame_name(i, "png"))
            if rc["enabled"]:
                rgb = read_rgb(paths.frames / frame_name(i, "png"))
                depth = refine(depth, rgb, alpha, rc, device)
            write_depth(paths.depth_final / frame_name(i, "npy"), depth)

            obj = alpha > thr
            frames[f"{i:04d}"] = {
                "disparity": z_stats(depth[obj], pcts),                       # pipeline units, larger = closer
                "z": z_stats(disparity_to_z(depth[obj], cfg["exr"]), pcts),   # EXR Z units
            }

    lost = [k for k, v in frames.items() if v["z"]["pixels"] == 0]
    paths.stats_json.write_text(json.dumps({
        "alpha_threshold": thr,
        "z_mode": cfg["exr"]["z_mode"],
        "note": "stats only where alpha > threshold; 'disparity' = pipeline units (larger = closer), "
                "'z' = EXR Z channel units; relative depth, not metric",
        "frames_without_object": lost,
        "frames": frames,
    }, indent=1), encoding="utf-8")
    if lost:
        print(f"[fuse] WARNING: no alpha > {thr} on {len(lost)} frames, stats are null there")
    print(f"[fuse] wrote {len(indices)} frames to {paths.depth_final} and {paths.stats_json}")
