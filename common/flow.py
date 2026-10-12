"""Optical flow helpers (OpenCV DIS: built in, fast, no extra model).

Optical flow = for every pixel of frame A, where did it move to in frame B.
Used by the `flow` alignment mode (flicker removal with a moving camera)
and by the OPW temporal metric.
"""
import cv2
import numpy as np

_dis = None


def optical_flow(a_gray, b_gray):
    """Dense flow a -> b. Inputs: uint8 grayscale images of the same size."""
    global _dis
    if _dis is None:
        _dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dis.calc(a_gray, b_gray, None)


def warp(img, flow):
    """Sample `img` at (x + flow): brings frame B's content onto frame A's pixel grid.
    Pixels that land outside the image become NaN."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(img.astype(np.float32), gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=np.nan)


def reliable_flow(a_gray, b_gray):
    """Flow a -> b plus a mask of pixels where the flow can be trusted:
    going a -> b -> a must come back (forward/backward check), so occluded
    pixels and bad matches are dropped."""
    fw = optical_flow(a_gray, b_gray)
    bw = optical_flow(b_gray, a_gray)
    back = warp(bw, fw)
    err = np.linalg.norm(fw + back, axis=-1)
    ok = np.isfinite(err) & (err < 0.5 + 0.05 * np.linalg.norm(fw, axis=-1))
    return fw, ok
