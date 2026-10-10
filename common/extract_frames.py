"""Stage 0: video -> data/<shot>/frames/0001.png, 0002.png, ...

Uses the ffmpeg binary shipped with the `imageio-ffmpeg` pip package,
so nobody has to install ffmpeg by hand.

Run alone:
    python -m common.extract_frames --config configs/demo.yaml
"""
import argparse
import subprocess

import imageio_ffmpeg

from common.config import load_config
from common.io import ShotPaths, list_indices


def extract(cfg: dict):
    paths = ShotPaths.from_config(cfg)
    paths.make_dirs()
    fc = cfg["frames"]
    if list_indices(paths.frames, "png") and not fc.get("overwrite", False):
        print(f"[extract] {paths.frames} already has frames, skipping (set frames.overwrite: true to redo)")
        return

    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y"]
    if fc.get("start_time"):
        cmd += ["-ss", str(fc["start_time"])]
    cmd += ["-i", str(cfg["input_video"])]
    if fc.get("max_frames"):
        cmd += ["-frames:v", str(fc["max_frames"])]
    if fc.get("resize_height"):
        # keep aspect ratio, width rounded to an even number
        cmd += ["-vf", f"scale=-2:{fc['resize_height']}:flags=lanczos"]
    # -start_number 1 gives 0001.png as the first frame (contract: 1-based, 4 digits)
    cmd += ["-start_number", "1", str(paths.frames / "%04d.png")]
    print("[extract]", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"[extract] wrote {len(list_indices(paths.frames, 'png'))} frames to {paths.frames}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    extract(load_config(ap.parse_args().config))
