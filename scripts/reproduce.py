"""Re-run the evaluation from scratch and compare with the reference numbers.

    python scripts/reproduce.py quick      # Bonn depth + DAVIS segmentation   (~3.3 GB download, ~1 h on a laptop GPU)
    python scripts/reproduce.py full       # + Sintel depth + YouTubeMatte     (~13 GB download, ~3 h)

Steps: download datasets -> prepare -> run the models and metrics -> error-analysis
pictures -> compare results/ with reference_results/. Each step runs on its own, so
one failure does not stop the others; a summary at the end says what worked.
Needs an NVIDIA GPU (the models are far too slow on CPU).
"""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable

PLANS = {
    "quick": [
        ("download Bonn + DAVIS", [PY, "scripts/get_datasets.py", "bonn", "davis"]),
        ("prepare Bonn (test)", [PY, "-m", "evaluation.prepare_benchmarks", "bonn"]),
        ("prepare Bonn (validation)", [PY, "-m", "evaluation.prepare_benchmarks", "bonnval"]),
        ("depth: Bonn test", [PY, "-m", "evaluation.run_depth_eval", "bonn"]),
        ("depth: Bonn validation", [PY, "-m", "evaluation.run_depth_eval", "bonnval"]),
        ("segmentation: DAVIS 2017 val", [PY, "-m", "evaluation.run_seg_benchmark", "davis"]),
    ],
    "full": [
        ("download Sintel + YouTubeMatte", [PY, "scripts/get_datasets.py", "sintel", "youtubematte"]),
        ("prepare Sintel", [PY, "-m", "evaluation.prepare_benchmarks", "sintel"]),
        ("depth: Sintel", [PY, "-m", "evaluation.run_depth_eval", "sintel"]),
        ("segmentation + matting: YouTubeMatte", [PY, "-m", "evaluation.run_seg_benchmark", "youtubematte"]),
    ],
}


def gpu_ok():
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def main(level):
    if level not in PLANS:
        raise SystemExit("usage: python scripts/reproduce.py quick|full")
    if not gpu_ok():
        print("WARNING: no NVIDIA GPU found. This will take many hours on CPU. Press Ctrl+C to stop.")
    steps = PLANS["quick"] + (PLANS["full"] if level == "full" else [])
    steps += [("error-analysis pictures", [PY, "-m", "evaluation.make_gallery"]),
              ("compare with reference_results/", [PY, "scripts/compare_results.py"])]
    report = []
    for name, cmd in steps:
        print(f"\n{'=' * 70}\n>> {name}\n{'=' * 70}", flush=True)
        t0 = time.time()
        code = subprocess.run(cmd, cwd=ROOT).returncode
        mins = (time.time() - t0) / 60
        status = "ok" if code == 0 else ("numbers differ (see above)" if "compare" in name else f"FAILED (exit {code})")
        report.append((name, status, mins))
    print(f"\n{'=' * 70}\nSUMMARY ({level})")
    for name, status, mins in report:
        print(f"  {name:42s} {status:28s} {mins:6.1f} min")
    print("Your tables are in results/, the reference in reference_results/.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "quick")
