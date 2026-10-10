"""Shared file I/O that implements the interface contract (see CLAUDE.md).

Every module reads and writes through these helpers, so the folder layout,
4-digit file names and data formats are defined in exactly one place.

Formats on disk:
    frames/      0001.png   8-bit RGB
    mask/        0001.png   8-bit, 0 or 255 (binary SAM 2 mask)
    alpha/       0001.png   16-bit grayscale, loaded as float32 in [0, 1]
    depth_raw/   0001.npy   float32, H x W, native model output (larger = closer)
    depth/       0001.npy   float32, after global scale/shift alignment
    depth_final/ 0001.npy   float32, after edge-aware refinement
    exr/         0001.exr   R,G,B,A (half) + Z (float32)
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SUBDIRS = ["frames", "mask", "alpha", "depth_raw", "depth", "depth_final", "exr"]


def frame_name(index: int, ext: str) -> str:
    """frame_name(1, 'png') -> '0001.png'. Frames are numbered from 1."""
    return f"{index:04d}.{ext}"


@dataclass
class ShotPaths:
    """All folders of one shot: data/<shot>/frames, data/<shot>/mask, ..."""

    root: Path

    @classmethod
    def from_config(cls, cfg: dict) -> "ShotPaths":
        return cls(Path(cfg["data_root"]) / cfg["shot"])

    def __getattr__(self, name):
        # paths.frames, paths.alpha, ... -> data/<shot>/<name>
        if name in SUBDIRS:
            return self.root / name
        raise AttributeError(name)

    @property
    def stats_json(self) -> Path:
        return self.root / "stats.json"

    def make_dirs(self):
        for d in SUBDIRS:
            (self.root / d).mkdir(parents=True, exist_ok=True)


def list_indices(folder: Path, ext: str) -> list[int]:
    """Sorted frame numbers found in a folder, e.g. [1, 2, 3, ...]."""
    return sorted(int(p.stem) for p in Path(folder).glob(f"*.{ext}") if p.stem.isdigit())


# ---------- RGB frames ----------
def read_rgb(path) -> np.ndarray:
    """Returns uint8 H x W x 3 in RGB order (OpenCV loads BGR, so we flip)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_rgb(path, rgb: np.ndarray):
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


# ---------- binary mask ----------
def read_mask(path) -> np.ndarray:
    """Returns bool H x W."""
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(path)
    return m > 127


def write_mask(path, mask: np.ndarray):
    cv2.imwrite(str(path), (np.asarray(mask) > 0).astype(np.uint8) * 255)


# ---------- soft alpha ----------
def read_alpha(path) -> np.ndarray:
    """Returns float32 H x W in [0, 1]. Accepts 8-bit or 16-bit PNG."""
    a = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise FileNotFoundError(path)
    if a.ndim == 3:
        a = a[..., 0]
    return a.astype(np.float32) / np.iinfo(a.dtype).max


def write_alpha(path, alpha: np.ndarray):
    """Saves float alpha as 16-bit PNG (precision 1/65535, viewable in any image tool)."""
    a = np.clip(np.asarray(alpha, dtype=np.float32), 0.0, 1.0)
    cv2.imwrite(str(path), np.round(a * 65535).astype(np.uint16))


# ---------- depth ----------
def read_depth(path) -> np.ndarray:
    return np.load(path).astype(np.float32)


def write_depth(path, depth: np.ndarray):
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be H x W, got {depth.shape}")
    np.save(path, depth)
