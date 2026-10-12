"""Download Tears of Steel raw footage and turn it into a pipeline shot.

Tears of Steel (CC-BY 3.0) is a Blender open movie whose ORIGINAL camera
footage is public: the same kind of plate a VFX studio receives.
Attribution required: "(CC) Blender Foundation | mango.blender.org"
The actors keep their portrait rights: OK for tech demos, not for ads.

Source frames are scene-linear 16-bit EXR (1920x1012 in 'linear_hd').
Our models expect normal display images, so we convert:
    linear EXR --(exposure, sRGB curve)--> 8-bit PNG in frames/
The original linear EXRs are kept in data/<shot>/source_exr/ so the EXR
writer can later put the untouched linear RGB into the final file.

    python -m common.get_tears_of_steel --tos_shot 04_2b --start 300 --count 100
    python -m common.get_tears_of_steel --tos_shot 08_3a --start 100 --count 30 --res 4k --shot_name tos_08_3a_4k
    python -m common.get_tears_of_steel --preview 01_2a 02_3a 04_2b   # one frame per shot, to choose shots
"""
import argparse
import concurrent.futures as cf
import urllib.request
from pathlib import Path

import numpy as np
import OpenEXR

from common.io import ShotPaths, frame_name, write_rgb

BASE = "https://ftp.lysator.liu.se/pub/xiph/media/tearsofsteel/tearsofsteel-footage-exr"


def read_exr_rgb(path) -> np.ndarray:
    """Returns float32 H x W x 3 (linear)."""
    with OpenEXR.File(str(path)) as f:
        ch = f.channels()
        if "RGB" in ch:
            rgb = ch["RGB"].pixels
        else:
            rgb = np.stack([ch[c].pixels for c in "RGB"], -1)
    return np.asarray(rgb, dtype=np.float32)


def linear_to_srgb8(rgb: np.ndarray, exposure: float) -> np.ndarray:
    """Simple display transform: exposure gain, clip, sRGB curve, 8-bit."""
    x = np.clip(rgb * exposure, 0.0, 1.0)
    srgb = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return np.round(srgb * 255).astype(np.uint8)


def auto_exposure(rgb: np.ndarray, target=0.18) -> float:
    """Gain so that the median luminance lands on mid-grey (0.18)."""
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    return float(target / max(np.median(lum), 1e-4))


def download(url: str, dst: Path, retries=3):
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(".part")
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(dst)
            return dst
        except Exception:
            if attempt == retries - 1:
                raise


def src_url(tos_shot: str, frame: int, res: str = "hd") -> str:
    folder = "linear_hd" if res == "hd" else "linear"     # linear = original 4096x2160
    return f"{BASE}/{tos_shot}/{folder}/{tos_shot}_{frame:05d}.exr"


def make_shot(tos_shot, start, count, data_root="data", shot_name=None, exposure=None, workers=6, res="hd"):
    shot_name = shot_name or f"tos_{tos_shot}"
    paths = ShotPaths(Path(data_root) / shot_name)
    paths.make_dirs()
    src_dir = paths.root / "source_exr"
    src_dir.mkdir(exist_ok=True)

    # 1) download in parallel (the server is slow per file, fast in parallel)
    jobs = {i: src_dir / frame_name(i, "exr") for i in range(1, count + 1)}
    with cf.ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(download, src_url(tos_shot, start + i - 1, res), dst) for i, dst in jobs.items()]
        for n, fut in enumerate(cf.as_completed(futs), 1):
            fut.result()
            if n % 10 == 0 or n == count:
                print(f"[tos] downloaded {n}/{count}")

    # 2) ONE exposure for the whole shot (per-frame exposure would flicker)
    if exposure is None:
        exposure = auto_exposure(read_exr_rgb(jobs[(count + 1) // 2]))
    for i, src in jobs.items():
        write_rgb(paths.frames / frame_name(i, "png"), linear_to_srgb8(read_exr_rgb(src), exposure))

    (paths.root / "SOURCE.txt").write_text(
        f"Tears of Steel shot {tos_shot}, source frames {start}..{start + count - 1} (linear_hd)\n"
        f"Display exposure gain: {exposure:.4f}\n"
        "(CC) Blender Foundation | mango.blender.org  -- CC-BY 3.0\n",
        encoding="utf-8",
    )
    print(f"[tos] {shot_name}: {count} frames, exposure x{exposure:.3f} -> {paths.frames}")


def preview(shots, out_dir, frame=None):
    """Download one frame per shot to choose good shots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in shots:
        f = frame if frame is not None else 100
        exr = out_dir / f"{s}.exr"
        try:
            download(src_url(s, f), exr)
        except Exception as e:
            print(f"{s}: failed ({e})")
            continue
        rgb = read_exr_rgb(exr)
        write_rgb(out_dir / f"{s}.png", linear_to_srgb8(rgb, auto_exposure(rgb)))
        print(f"{s}: ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tos_shot", help="e.g. 04_2b")
    ap.add_argument("--start", type=int, default=0, help="first source frame number")
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--shot_name", help="default: tos_<tos_shot>")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--exposure", type=float, help="fixed gain; default = auto from middle frame")
    ap.add_argument("--res", choices=["hd", "4k"], default="hd", help="hd = 1920x1012, 4k = 4096x2160")
    ap.add_argument("--preview", nargs="*", help="shots to preview (one frame each)")
    ap.add_argument("--preview_dir", default="data/_tos_preview")
    a = ap.parse_args()
    if a.preview:
        preview(a.preview, a.preview_dir)
    else:
        make_shot(a.tos_shot, a.start, a.count, a.data_root, a.shot_name, a.exposure, res=a.res)
