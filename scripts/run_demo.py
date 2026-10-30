"""What run_demo.bat runs: prove the pipeline works on this computer, then open the result.

With an NVIDIA GPU: the full pipeline on samples/videos/tos_01_2a.mp4 (video -> EXR, ~3 min),
then the demo video (plate | matte | depth | CG banner inserted behind the actor).
Without a GPU: the CPU stages on the synthetic shot, and on a dev pack if you unzipped one.
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def run(*args):
    print(f"\n>> {' '.join(args)}", flush=True)
    r = subprocess.run([PY, *args], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"\nFAILED: {' '.join(args)}\nSee the error above, and 'When something goes wrong' in README.md.")


def has_gpu():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception as e:
        if "4551" in str(e):
            raise SystemExit("Windows Smart App Control is blocking PyTorch (WinError 4551). See README.md.")
        return False


def open_file(path):
    print(f"\nOpening {path}")
    try:
        os.startfile(path)            # Windows: open with the default app
    except (AttributeError, OSError):
        pass


def main():
    if has_gpu():
        print("NVIDIA GPU found: running the full pipeline on a sample film shot (about 3 minutes).")
        run("run_pipeline.py", "--config", "configs/tos_01_2a_from_mp4.yaml")
        run("-m", "common.validate", "data/tos_01_2a_from_mp4")
        run("-m", "demo.make_demo", "data/tos_01_2a_from_mp4")
        out = ROOT / "data" / "tos_01_2a_from_mp4"
        print(f"\nDONE. EXR files: {out / 'exr'}\n      Demo video: {out / 'demo.mp4'}")
        open_file(out / "demo.mp4")
    else:
        print("No NVIDIA GPU: running the CPU stages on a synthetic shot.")
        run("-m", "common.make_dummy_shot", "--shot", "dummy")
        run("run_pipeline.py", "--config", "configs/dummy.yaml")
        run("-m", "common.validate", "data/dummy")
        print(f"\nDONE. EXR files: {ROOT / 'data' / 'dummy' / 'exr'}")
        pack = ROOT / "data" / "tos_08_3a" / "depth_raw"
        if pack.exists():
            print("Dev pack tos_08_3a found: running the CPU stages on real film footage too.")
            run("run_pipeline.py", "--config", "configs/tos_08_3a.yaml", "--stages", "align", "fuse", "exr")
            run("-m", "demo.make_demo", "data/tos_08_3a")
            open_file(ROOT / "data" / "tos_08_3a" / "demo.mp4")
        else:
            print("Tip: download a dev pack from the GitHub Releases page (see README) to try real footage.")
            open_file(ROOT / "data" / "dummy" / "exr")


if __name__ == "__main__":
    main()
