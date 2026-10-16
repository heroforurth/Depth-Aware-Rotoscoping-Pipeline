"""Evaluation metrics. Plain numpy/OpenCV so anyone can read them.

DEPTH ACCURACY (needs ground truth, e.g. Sintel, Bonn)
  Models give relative disparity. Standard protocol (Depth Anything / Video Depth
  Anything / DepthCrafter papers): fit ONE scale and shift per SEQUENCE in
  disparity space against the ground truth, turn it into depth, then
    AbsRel = mean(|pred - gt| / gt)                    lower is better
    delta1 = share of pixels with max(pred/gt, gt/pred) < 1.25   higher is better

TEMPORAL CONSISTENCY (no ground truth needed)
  OPW, optical-flow warping error: estimate optical flow between frame t and
  t+1 from the RGB frames, warp depth(t+1) back onto frame t, and compare it
  with depth(t). Pixels where the flow is unreliable (forward/backward check
  fails) are skipped. A stable depth model gives a small error; flicker gives
  a big one. Depth is first normalised to [0, 1] for the whole sequence so
  models are compared on the same scale.

ALPHA (needs ground-truth alpha, e.g. YouTubeMatte)
  MAD   = mean |alpha - gt| x 1000
  MSE   = mean (alpha - gt)^2 x 1000
  dtSSD = temporal coherence: compares the CHANGE of alpha between frames with
          the change of the ground truth, x 100. Low = no flicker.

EDGE SHARPNESS (raw vs refined depth, no ground truth needed)
  Depth step across the matte edge measured 3 px from the edge, divided by the
  step measured 12 px from the edge. 1.0 = the depth edge sits exactly on the
  matte edge; lower = the depth edge is blurred or offset.
"""
import cv2
import numpy as np

from common.flow import reliable_flow, warp


# ---------------- depth accuracy ----------------
def fit_disparity_to_gt(pred_disp, gt_depth, valid):
    """ONE scale/shift for the whole sequence, in disparity space."""
    x = pred_disp[valid].astype(np.float64)
    y = 1.0 / gt_depth[valid].astype(np.float64)
    step = max(1, x.size // 2_000_000)                     # sub-sample for speed
    A = np.stack([x[::step], np.ones_like(x[::step])], 1)
    s, b = np.linalg.lstsq(A, y[::step], rcond=None)[0]
    return float(s), float(b)


def depth_accuracy(pred_disp, gt_depth, max_depth, min_depth=1e-3):
    """pred_disp, gt_depth: T x H x W. Returns dict with abs_rel and delta1."""
    valid = (gt_depth > min_depth) & (gt_depth < max_depth) & np.isfinite(gt_depth)
    s, b = fit_disparity_to_gt(pred_disp, gt_depth, valid)
    disp = np.clip(s * pred_disp + b, 1.0 / max_depth, 1.0 / min_depth)
    pred = 1.0 / disp
    p, g = pred[valid], gt_depth[valid]
    ratio = np.maximum(p / g, g / p)
    return {"abs_rel": float(np.mean(np.abs(p - g) / g)), "delta1": float(np.mean(ratio < 1.25)),
            "scale": s, "shift": b, "valid_px": int(valid.sum())}


# ---------------- temporal consistency ----------------
def flow_pairs(rgb_frames, max_side=640):
    """Flow + reliable-pixel mask for each pair (t, t+1), computed once and reused for all models.
    Frames are shrunk to max_side for speed; returns (scale, list of (flow_fw, ok))."""
    h, w = rgb_frames[0].shape[:2]
    sc = min(1.0, max_side / max(h, w))
    small = [cv2.cvtColor(cv2.resize(f, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA),
                          cv2.COLOR_RGB2GRAY) for f in rgb_frames]
    pairs = []
    for t in range(len(small) - 1):
        pairs.append(reliable_flow(small[t], small[t + 1]))
    return sc, pairs


def opw_error(depth_frames, pairs, sc):
    """Mean |d_t - warp(d_t+1)| on reliable pixels, depth normalised to [0,1] per sequence."""
    d = np.stack(depth_frames).astype(np.float32)
    lo, hi = np.percentile(d[:, ::4, ::4], [2, 98])
    d = (d - lo) / max(hi - lo, 1e-6)
    h, w = pairs[0][0].shape[:2]
    errs = []
    for t, (fw, ok) in enumerate(pairs):
        a = cv2.resize(d[t], (w, h), interpolation=cv2.INTER_AREA)
        b = cv2.resize(d[t + 1], (w, h), interpolation=cv2.INTER_AREA)
        bw = warp(b, fw)
        m = ok & np.isfinite(bw)
        if m.any():
            errs.append(np.abs(a[m] - bw[m]).mean())
    return float(np.mean(errs))


# ---------------- alpha ----------------
def alpha_metrics(pred, gt):
    """pred, gt: T x H x W in [0, 1]."""
    pred, gt = np.asarray(pred, np.float32), np.asarray(gt, np.float32)
    mad = np.abs(pred - gt).mean() * 1e3
    mse = ((pred - gt) ** 2).mean() * 1e3
    dt = np.sqrt((((pred[1:] - pred[:-1]) - (gt[1:] - gt[:-1])) ** 2).sum((1, 2))).mean() * 1e2 / \
        np.sqrt(pred.shape[1] * pred.shape[2]) if len(pred) > 1 else float("nan")
    return {"MAD": float(mad), "MSE": float(mse), "dtSSD": float(dt)}


# ---------------- segmentation (binary masks) ----------------
def iou(pred, gt):
    """Jaccard / IoU = |A and B| / |A or B|. Both empty = 1 (correctly found nothing)."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    union = np.logical_or(pred, gt).sum()
    return 1.0 if union == 0 else float(np.logical_and(pred, gt).sum() / union)


def dice(pred, gt):
    """Dice = 2 |A and B| / (|A| + |B|)."""
    pred, gt = np.asarray(pred, bool), np.asarray(gt, bool)
    total = pred.sum() + gt.sum()
    return 1.0 if total == 0 else float(2 * np.logical_and(pred, gt).sum() / total)


def _boundary(mask):
    m = np.asarray(mask, np.uint8)
    return (m - cv2.erode(m, np.ones((3, 3), np.uint8))) > 0


def boundary_f(pred, gt, tol_frac=0.008):
    """DAVIS boundary F-measure: how well the predicted OUTLINE matches the true outline.
    A boundary pixel counts as correct if it lies within tol = 0.8% of the image diagonal
    of the other outline. F = 2PR / (P + R)."""
    bp, bg = _boundary(pred), _boundary(gt)
    if bp.sum() == 0 and bg.sum() == 0:
        return 1.0
    if bp.sum() == 0 or bg.sum() == 0:
        return 0.0
    r = max(1, int(np.ceil(tol_frac * np.hypot(*bp.shape))))
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    gt_near = cv2.dilate(bg.astype(np.uint8), disk) > 0
    pred_near = cv2.dilate(bp.astype(np.uint8), disk) > 0
    precision = (bp & gt_near).sum() / bp.sum()
    recall = (bg & pred_near).sum() / bg.sum()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


# ---------------- edge sharpness ----------------
def edge_sharpness(depth, alpha, near=3, far=12):
    """Depth step across the matte edge at `near` px divided by the step at `far` px."""
    fg = (alpha > 0.5).astype(np.uint8)
    din = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
    dout = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3)
    ring = lambda dist, r: (dist > r - 1) & (dist <= r)

    def step(r):
        a, b = ring(din, r), ring(dout, r)
        if a.sum() < 50 or b.sum() < 50:
            return None
        return abs(float(np.median(depth[a]) - np.median(depth[b])))

    s_near, s_far = step(near), step(far)
    if not s_near or not s_far:
        return None
    return s_near / s_far
