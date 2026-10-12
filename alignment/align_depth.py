"""[B2] Depth alignment: depth_raw/ -> depth/  (design decision 3, REQUIRED step)

The problem: Depth Anything gives RELATIVE depth (disparity, larger = closer)
with an unknown scale and shift, and a per-frame model can pick a different
scale/shift on every frame. Played as video, that looks like flicker:
the whole scene "breathes" closer and farther.

The fix, in two steps:

1. (modes `per_frame` and `window`) make frames agree with each other.
   For each frame i we fit two numbers s_i, b_i so that
        s_i * raw_i + b_i  ~=  reference
   in the least-squares sense. The reference is the median of the
   (already aligned) neighbouring frames, so it changes slowly.
   We use only BACKGROUND pixels (alpha < 0.5) when alpha exists,
   because the actor moves and should not pull the fit.
   - per_frame: reference = median of the whole shot
   - window:    reference = median of the frames within +-window/2
   For Video Depth Anything this is usually not needed (it is already
   consistent over time), so its default is `global`.

2. (all modes) ONE global scale and shift for the whole shot, mapping
   the shot's robust 2% / 98% percentiles to 0 / 1. Every frame gets the
   SAME numbers, so depth relations between frames are kept.

Output stays in the native direction: larger = closer (normalised disparity).
The EXR writer converts it to Z once (design decision 4).

    python run_pipeline.py --config configs/dummy.yaml --stages align
"""
import json

import numpy as np
from tqdm import tqdm

from common.io import frame_name, list_indices, read_alpha, read_depth, read_rgb, write_depth
from common.timing import timed

SAMPLE_STEP = 4   # use every 4th pixel in x and y for fitting (and flow): 16x faster, same answer


def fit_scale_shift(src, dst, weight=None):
    """Least squares for s, b in: s * src + b ~= dst. Returns (s, b)."""
    if weight is not None:
        src, dst = src[weight], dst[weight]
    A = np.stack([src, np.ones_like(src)], 1).astype(np.float64)
    (s, b), *_ = np.linalg.lstsq(A, dst.astype(np.float64), rcond=None)
    return float(s), float(b)


def robust_fit(src, dst, weight=None, iters=3, keep=0.8):
    """Fit, drop the 20% worst pixels (moving things, sky noise), fit again."""
    w = np.ones_like(src, bool) if weight is None else weight.copy()
    s, b = fit_scale_shift(src, dst, w)
    for _ in range(iters - 1):
        err = np.abs(s * src + b - dst)
        thr = np.quantile(err[w], keep)
        w2 = w & (err <= thr)
        if w2.sum() < 100:
            break
        s, b = fit_scale_shift(src, dst, w2)
    return s, b


def flow_chain_fit(paths, indices, small):
    """`flow` mode: works with a moving camera and moving people.
    Frame t is fitted to the ALREADY ALIGNED frame t-1, moved onto frame t's
    pixels with optical flow. Flow follows the camera and the actors, so we
    compare the same surface point in both frames. Only pixels where the flow
    passes the forward/backward check are used."""
    import cv2

    from common.flow import reliable_flow, warp

    h, w = small[indices[0]].shape
    gray = {}
    for i in indices:
        g = cv2.cvtColor(read_rgb(paths.frames / frame_name(i, "png")), cv2.COLOR_RGB2GRAY)
        gray[i] = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
    sb = {indices[0]: (1.0, 0.0)}
    prev = small[indices[0]]
    for a, b in tqdm(list(zip(indices[:-1], indices[1:])), desc="align (flow)", leave=False):
        flow_ba, ok = reliable_flow(gray[b], gray[a])        # where each pixel of b was in a
        prev_on_b = warp(prev, flow_ba)
        ok &= np.isfinite(prev_on_b)
        if ok.sum() < 200:
            print(f"[align] frame {b}: too few reliable flow pixels, keeping previous scale")
            sb[b] = sb[a]
        else:
            s, sh = robust_fit(small[b][ok], prev_on_b[ok])
            sb[b] = (s, sh) if s > 0 else sb[a]
        prev = sb[b][0] * small[b] + sb[b][1]
    return sb


def run(cfg, paths):
    ac = cfg["align"]
    mode = ac["mode"]
    src_dir = paths.root / cfg["depth"]["out_subdir"]
    indices = list_indices(src_dir, "npy")
    if not indices:
        raise FileNotFoundError(f"no depth in {src_dir}; run the depth stage first")
    has_alpha = len(list_indices(paths.alpha, "png")) == len(indices)

    with timed("align", paths.root, len(indices), {"mode": mode}):
        # small copies for fitting (memory: 100 frames x 1080p x 4 bytes = 800 MB at full size)
        raw = {i: read_depth(src_dir / frame_name(i, "npy")) for i in indices}
        small = {i: d[::SAMPLE_STEP, ::SAMPLE_STEP] for i, d in raw.items()}
        bg = {}
        for i in indices:
            if has_alpha:
                bg[i] = read_alpha(paths.alpha / frame_name(i, "png"))[::SAMPLE_STEP, ::SAMPLE_STEP] < 0.5
            else:
                bg[i] = np.ones_like(small[i], bool)

        # ---- step 1: per-frame scale/shift so frames agree ----
        sb = {i: (1.0, 0.0) for i in indices}
        if mode in ("per_frame", "window"):
            # first pass: normalise each frame by its own percentiles to get a rough common scale
            for i in indices:
                lo, hi = np.percentile(small[i], [2, 98])
                sb[i] = (1.0 / max(hi - lo, 1e-6), -lo / max(hi - lo, 1e-6))
            half = ac["window"] // 2
            for _ in range(2):   # two rounds: the reference improves after the first
                cur = np.stack([sb[i][0] * small[i] + sb[i][1] for i in indices])
                shot_median = np.median(cur, axis=0) if mode == "per_frame" else None
                for n, i in enumerate(tqdm(indices, desc="align", leave=False)):
                    if mode == "window":
                        ref = np.median(cur[max(0, n - half):n + half + 1], axis=0)
                    else:
                        ref = shot_median
                    s, b = robust_fit(small[i].ravel(), ref.ravel(), bg[i].ravel())
                    if s <= 0:   # a negative scale would flip near/far: keep rough normalisation
                        print(f"[align] frame {i}: bad fit (s={s:.3g}), kept rough normalisation")
                        continue
                    sb[i] = (s, b)
        elif mode == "flow":
            sb = flow_chain_fit(paths, indices, small)
        elif mode != "global":
            raise ValueError(f"align.mode must be global, per_frame, window or flow, not {mode!r}")

        # ---- step 2: ONE global scale/shift for the whole shot -> robust [0, 1] ----
        allv = np.concatenate([(sb[i][0] * small[i] + sb[i][1]).ravel() for i in indices])
        lo, hi = np.percentile(allv, [2, 98])
        g_s, g_b = 1.0 / max(hi - lo, 1e-6), -lo / max(hi - lo, 1e-6)

        per_frame = {}
        for i in indices:
            s, b = sb[i]
            write_depth(paths.depth / frame_name(i, "npy"), (g_s * (s * raw[i] + b) + g_b).astype(np.float32))
            per_frame[f"{i:04d}"] = {"scale": g_s * s, "shift": g_s * b + g_b}

    # keep the numbers: useful for debugging and for the report
    (paths.root / "align.json").write_text(json.dumps({
        "mode": mode, "source": cfg["depth"]["out_subdir"], "used_alpha_as_background": has_alpha,
        "global": {"scale": g_s, "shift": g_b, "p2": float(lo), "p98": float(hi)},
        "per_frame": per_frame,
    }, indent=1), encoding="utf-8")
    print(f"[align] mode={mode}, wrote {len(indices)} frames to {paths.depth}")
