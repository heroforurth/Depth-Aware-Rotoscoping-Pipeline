"""Log seconds-per-frame and GPU memory for every stage (needed for the speed table)."""
import json
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


def _gpu_peak_mb():
    try:
        import torch

        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 2**20, 1)
    except (ImportError, OSError):
        pass
    return None


def _reset_gpu_peak():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except (ImportError, OSError):
        pass


@contextmanager
def timed(stage: str, shot_root: Path, n_frames: int, extra: dict | None = None):
    """Usage:
        with timed("depth", paths.root, n_frames=100):
            ...
    Appends one JSON line to data/<shot>/timings.jsonl.
    """
    _reset_gpu_peak()
    t0 = time.perf_counter()
    yield
    sec = time.perf_counter() - t0
    rec = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "stage": stage,
        "frames": n_frames,
        "seconds": round(sec, 3),
        "sec_per_frame": round(sec / max(n_frames, 1), 4),
        "gpu_peak_mb": _gpu_peak_mb(),
        **(extra or {}),
    }
    Path(shot_root).mkdir(parents=True, exist_ok=True)
    with open(Path(shot_root) / "timings.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[{stage}] {n_frames} frames in {sec:.1f}s ({rec['sec_per_frame']} s/frame)")
