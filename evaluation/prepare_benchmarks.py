"""Turn depth benchmarks into normal pipeline shots, so the SAME depth code runs on them.

    python -m evaluation.prepare_benchmarks sintel     -> data/bench_sintel_<seq>/frames + gt_depth
    python -m evaluation.prepare_benchmarks bonn       -> data/bench_bonn_<seq>/frames + gt_depth

Ground truth goes to gt_depth/0001.npy (float32, metres, 0 = no measurement).

Sintel (MPI Sintel, "final" render pass, 23 sequences, 20-50 frames each, max depth 70 m)
  datasets/sintel/depth.zip  (MPI-Sintel-depth-training-20150305.zip)
  datasets/sintel/images.zip (MPI-Sintel-training_images.zip)
Bonn RGB-D Dynamic (real Kinect footage of people moving; 5 sequences, 110-frame clips
  starting at frame 30, max depth 10 m). Depth PNGs are 16-bit, value / 5000 = metres (TUM format).
  datasets/bonn/rgbd_bonn_<seq>.zip
"""
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from common.io import ShotPaths, frame_name, write_rgb

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DS = ROOT / "datasets"
BONN_SEQS = ["balloon2", "crowd2", "crowd3", "person_tracking2", "synchronous"]
BONN_START, BONN_LEN = 30, 110


def read_dpt(buf: bytes) -> np.ndarray:
    """Sintel .dpt: 'PIEH' tag (float 202021.25), width, height (int32), then float32 depth."""
    tag = np.frombuffer(buf[:4], np.float32)[0]
    if abs(tag - 202021.25) > 1e-3:
        raise ValueError("not a Sintel .dpt file")
    w, h = np.frombuffer(buf[4:12], np.int32)
    return np.frombuffer(buf[12:12 + 4 * w * h], np.float32).reshape(h, w).copy()


def new_shot(name):
    paths = ShotPaths(DATA / name)
    paths.make_dirs()
    (paths.root / "gt_depth").mkdir(exist_ok=True)
    return paths


def prepare_sintel():
    zd = zipfile.ZipFile(DS / "sintel" / "depth.zip")
    zi = zipfile.ZipFile(DS / "sintel" / "images.zip")
    dpts = sorted(n for n in zd.namelist() if n.endswith(".dpt"))
    imgs = {n.split("training/final/")[1]: n for n in zi.namelist() if "training/final/" in n and n.endswith(".png")}
    seqs = sorted({n.split("/")[-2] for n in dpts})
    for seq in seqs:
        paths = new_shot(f"bench_sintel_{seq}")
        files = sorted(n for n in dpts if n.split("/")[-2] == seq)
        for k, n in enumerate(files, 1):
            stem = Path(n).stem                                       # frame_0001
            img = cv2.imdecode(np.frombuffer(zi.read(imgs[f"{seq}/{stem}.png"]), np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(str(paths.frames / frame_name(k, "png")), img)
            np.save(paths.root / "gt_depth" / frame_name(k, "npy"), read_dpt(zd.read(n)))
        print(f"sintel {seq}: {len(files)} frames")


def prepare_bonn():
    for seq in BONN_SEQS:
        z = zipfile.ZipFile(DS / "bonn" / f"rgbd_bonn_{seq}.zip")
        base = f"rgbd_bonn_{seq}/"

        def listing(txt):
            rows = [l.split() for l in z.read(base + txt).decode().splitlines() if l and not l.startswith("#")]
            return [(float(t), f) for t, f in rows]

        rgb, dep = listing("rgb.txt"), listing("depth.txt")
        dts = np.array([t for t, _ in dep])
        paths = new_shot(f"bench_bonn_{seq}")
        for k, (t, f) in enumerate(rgb[BONN_START:BONN_START + BONN_LEN], 1):
            j = int(np.abs(dts - t).argmin())                        # nearest depth frame in time
            img = cv2.imdecode(np.frombuffer(z.read(base + f), np.uint8), cv2.IMREAD_COLOR)
            d16 = cv2.imdecode(np.frombuffer(z.read(base + dep[j][1]), np.uint8), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(str(paths.frames / frame_name(k, "png")), img)
            np.save(paths.root / "gt_depth" / frame_name(k, "npy"), d16.astype(np.float32) / 5000.0)
        print(f"bonn {seq}: {BONN_LEN} frames")


if __name__ == "__main__":
    {"sintel": prepare_sintel, "bonn": prepare_bonn}[sys.argv[1]]()
