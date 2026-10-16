"""Quick check after you edit code: did anything break?  (~30 s, no GPU, no downloads)

    python scripts/self_test.py

Checks: every config loads; every stage module imports; the metrics give known answers on
hand-made masks; the CPU pipeline (align -> fuse -> exr) runs on a synthetic shot with true
depth, passes the interface contract, writes valid R,G,B,A,Z EXRs and alignment beats raw depth;
the reference result tables can be read. Uses a temporary folder, never touches data/.
Run it before every commit; it exits with code 1 if anything fails.
"""
import importlib
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
results = []


def check(name):
    def wrap(fn):
        try:
            fn()
            results.append((name, True, ""))
        except Exception as e:
            results.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        return fn
    return wrap


@check("all configs load")
def _():
    from common.config import load_config

    for p in sorted((ROOT / "configs").glob("*.yaml")):
        if p.name == "defaults.yaml":       # shared settings, not a shot
            continue
        cfg = load_config(p)
        assert cfg["shot"], p.name


@check("all stage modules import")
def _():
    import run_pipeline

    for spec in run_pipeline.STAGES.values():
        importlib.import_module(spec.split(":")[0])
    for m in ["evaluation.metrics", "evaluation.run_seg_benchmark", "evaluation.run_depth_eval",
              "evaluation.make_gallery", "demo.make_demo", "common.pick_points"]:
        importlib.import_module(m)


@check("metrics give known answers")
def _():
    from evaluation.metrics import boundary_f, depth_accuracy, dice, iou

    a = np.zeros((100, 100), bool); a[20:60, 20:60] = True
    b = np.zeros((100, 100), bool); b[30:70, 20:60] = True
    assert abs(iou(a, b) - 0.6) < 1e-9 and abs(dice(a, b) - 0.75) < 1e-9
    assert iou(a, a) == 1.0 and boundary_f(a, a) == 1.0
    gt = np.random.default_rng(0).uniform(1, 10, (3, 20, 20)).astype(np.float32)
    acc = depth_accuracy(2.0 / gt + 0.1, gt, 70.0)        # any scale/shift of disparity = perfect
    assert acc["abs_rel"] < 1e-4 and acc["delta1"] == 1.0, acc


@check("CPU pipeline on a synthetic shot (align -> fuse -> exr)")
def _():
    import OpenEXR

    from common.make_dummy_shot import make_dummy_shot
    from common.validate import validate

    with tempfile.TemporaryDirectory() as tmp:
        make_dummy_shot(Path(tmp) / "selftest", n_frames=8, height=90)
        r = subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--config", str(ROOT / "configs" / "dummy.yaml"),
                            "--set", f"data_root={tmp}", "shot=selftest", "align.mode=per_frame",
                            "--stages", "align", "fuse", "exr"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-800:] + r.stderr[-1500:]
        errs = validate(Path(tmp) / "selftest")
        assert not errs, errs
        with OpenEXR.File(str(Path(tmp) / "selftest" / "exr" / "0004.exr"), separate_channels=True) as f:
            ch = f.channels()
            assert set(ch) == set("RGBAZ"), list(ch)
            assert ch["Z"].pixels.dtype == np.float32 and ch["A"].pixels.dtype == np.float16
        # alignment must undo the random per-frame scale of the fake model depth
        d = Path(tmp) / "selftest"
        gt = np.stack([np.load(d / "gt_depth" / f"{i:04d}.npy") for i in range(1, 9)])
        def err(sub):
            x = np.stack([np.load(d / sub / f"{i:04d}.npy") for i in range(1, 9)])
            A = np.stack([x.ravel(), np.ones(x.size)], 1)
            s, b = np.linalg.lstsq(A, gt.ravel(), rcond=None)[0]
            return np.abs(s * x + b - gt).mean()
        assert err("depth") < 0.5 * err("depth_raw"), (err("depth"), err("depth_raw"))


@check("reference result tables are readable")
def _():
    import csv

    names = sorted(p.name for p in (ROOT / "reference_results").glob("*.csv"))
    assert len(names) >= 8, names
    for n in names:
        rows = list(csv.DictReader(open(ROOT / "reference_results" / n, encoding="utf-8")))
        assert rows, n


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("\nSELF-TEST")
    for name, ok, msg in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n        {msg}" if msg else ""))
    bad = sum(not ok for _, ok, _ in results)
    print("All good." if not bad else f"{bad} check(s) failed: see the error above.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
