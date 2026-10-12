"""Open an output EXR in Blender (headless) and check Blender sees our Z values.

    python -m exr_writer.verify_blender data/tos_01_2a/exr/0050.exr

Needs portable Blender in tools/ (see README). Prints PASS/FAIL and the max
difference between our Z channel and what Blender read.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import OpenEXR

ROOT = Path(__file__).resolve().parent.parent
BLENDER = next(iter(sorted((ROOT / "tools").glob("blender-*/blender.exe"))), None)


def depth_channel_name(channels):
    for name in ("Z", "ViewLayer.Depth.Z", "Depth.Z"):
        if name in channels:
            return name
    raise KeyError(f"no depth channel in {list(channels)}")


def verify(exr_path) -> dict:
    exr_path = Path(exr_path).resolve()
    with OpenEXR.File(str(exr_path), separate_channels=True) as f:
        ch = f.channels()
        z = ch[depth_channel_name(ch)].pixels.astype(np.float32).copy()
    H, W = z.shape
    if BLENDER is None:
        raise FileNotFoundError("portable Blender not found in tools/ (see exr_writer/README.md)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "blender_depth.exr"
        log = subprocess.run([str(BLENDER), "-b", "--factory-startup", "--python",
                              str(ROOT / "exr_writer" / "blender_check.py"), "--",
                              str(exr_path), str(out), str(W), str(H)], capture_output=True, text=True)
        line = next((l for l in log.stdout.splitlines() if l.startswith("BLENDER_CHECK")), None)
        info = json.loads(line.split(" ", 1)[1]) if line else {"error": log.stdout[-2000:] + log.stderr[-2000:]}
        if out.exists():
            with OpenEXR.File(str(out), separate_channels=True) as f:
                bz = f.channels()["R"].pixels.astype(np.float32).copy()
            info["max_abs_diff"] = float(np.abs(bz - z).max())
            info["blender_z_range"] = [float(bz.min()), float(bz.max())]
    info["our_z_range"] = [float(z.min()), float(z.max())]
    info["pass"] = info.get("max_abs_diff", 1e9) < 1e-3
    return info


if __name__ == "__main__":
    res = verify(sys.argv[1])
    print(json.dumps(res, indent=1))
    print("PASS: Blender reads the same depth" if res["pass"] else "FAIL")
