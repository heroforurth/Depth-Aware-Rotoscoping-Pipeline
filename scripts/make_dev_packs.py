"""Zip each shot's GPU-made outputs into a "dev pack" for teammates without a big GPU.

    python scripts/make_dev_packs.py                 -> release_packs/<shot>_devpack.zip

A pack holds data/<shot>/ with frames/, mask/ (SAM 2), alpha/ (MatAnyone),
depth_raw/ (Video Depth Anything) and SOURCE.txt. Unzip it in the repo root
and the CPU stages run at once:

    python run_pipeline.py --config configs/<shot>.yaml --stages align fuse exr

Left out on purpose (rebuilt in seconds on any machine): depth/ (align),
depth_final/ + stats.json (fuse), exr/ (exr). Also left out: source_exr/ (the
16-bit camera files, 0.7 GB a shot); without them the EXR writer converts the
PNG frames to linear instead. Packs are about 1 GB, under GitHub's 2 GB limit
for a release file.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ["tos_01_2a", "tos_04_2d", "tos_08_3a"]
FOLDERS = ["frames", "mask", "alpha", "depth_raw"]
FILES = ["SOURCE.txt"]


def make_pack(shot, out_dir):
    src = ROOT / "data" / shot
    out = out_dir / f"{shot}_devpack.zip"
    n = 0
    with zipfile.ZipFile(out, "w") as z:
        for folder in FOLDERS:
            for f in sorted((src / folder).iterdir()):
                # PNGs are already compressed; .npy depth shrinks a little with deflate
                comp = zipfile.ZIP_STORED if f.suffix == ".png" else zipfile.ZIP_DEFLATED
                z.write(f, f"data/{shot}/{folder}/{f.name}", compress_type=comp)
                n += 1
        for name in FILES:
            if (src / name).exists():
                z.write(src / name, f"data/{shot}/{name}", compress_type=zipfile.ZIP_DEFLATED)
    size = out.stat().st_size / 2**30
    print(f"{out.name}: {n} files, {size:.2f} GB")
    if size > 2.0:
        print("  WARNING: over 2 GB, GitHub releases will refuse it")


if __name__ == "__main__":
    out_dir = ROOT / "release_packs"
    out_dir.mkdir(exist_ok=True)
    for shot in sys.argv[1:] or SHOTS:
        make_pack(shot, out_dir)
