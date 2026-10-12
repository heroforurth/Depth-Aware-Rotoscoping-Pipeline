"""Depth accuracy + temporal consistency on Sintel / Bonn (required comparison:
per-frame Depth Anything V2 vs Video Depth Anything).

    python -m evaluation.prepare_benchmarks bonn          # once
    python -m evaluation.run_depth_eval bonn              # -> results/depth_bonn.csv + .md

For every benchmark sequence (data/bench_<dataset>_<seq>/) it runs:
  vda          Video Depth Anything Small                 (main model)
  dav2         Depth Anything V2 Small, one frame at a time (baseline)
  +flow-align  each model after OUR flow-based alignment (align.mode=flow)
and scores each with AbsRel / delta1 (one scale+shift per sequence vs ground truth)
and OPW (optical-flow warping error, lower = less flicker).
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from common.io import frame_name, list_indices, read_rgb
from evaluation.metrics import depth_accuracy, flow_pairs, opw_error

ROOT = Path(__file__).resolve().parent.parent
MAX_DEPTH = {"sintel": 70.0, "bonn": 10.0}


def pipeline(shot, *sets):
    cmd = [sys.executable, str(ROOT / "run_pipeline.py"), "--config", str(ROOT / "configs" / "bench.yaml"),
           "--set", f"shot={shot}", *sets]
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{shot}: {' '.join(sets)}\n{r.stdout[-1500:]}\n{r.stderr[-3000:]}")


def load_stack(folder, idx):
    return np.stack([np.load(folder / frame_name(i, "npy")) for i in idx]).astype(np.float32)


def evaluate(dataset, rerun=False):
    shots = sorted(p for p in (ROOT / "data").glob(f"bench_{dataset}_*") if (p / "gt_depth").exists())
    if not shots:
        raise SystemExit(f"no data/bench_{dataset}_* found; run: python -m evaluation.prepare_benchmarks {dataset}")
    rows = []
    for root in shots:
        shot = root.name
        idx = list_indices(root / "frames", "png")
        # 1) run the models (skipped if outputs exist)
        if rerun or len(list_indices(root / "depth_raw", "npy")) != len(idx):
            pipeline(shot, "depth.model=vda", "depth.out_subdir=depth_raw")
        if rerun or len(list_indices(root / "depth_raw_dav2", "npy")) != len(idx):
            pipeline(shot, "depth.model=dav2", "depth.out_subdir=depth_raw_dav2")
        gt = load_stack(root / "gt_depth", idx)
        sc, pairs = flow_pairs([read_rgb(root / "frames" / frame_name(i, "png")) for i in idx])
        # (name, raw folder, our alignment mode or None). The align stage writes to depth/ by
        # contract, so each aligned variant is run and read before the next one.
        variants = [("vda", "depth_raw", None), ("vda+flow-align", "depth_raw", "flow"),
                    ("dav2", "depth_raw_dav2", None), ("dav2+flow-align", "depth_raw_dav2", "flow")]
        for name, sub, mode in variants:
            if mode:
                pipeline(shot, f"depth.out_subdir={sub}", f"align.mode={mode}", "--stages", "align")
                pred = load_stack(root / "depth", idx)
            else:
                pred = load_stack(root / sub, idx)
            acc = depth_accuracy(pred, gt, MAX_DEPTH[dataset])
            rows.append({"dataset": dataset, "sequence": shot.replace(f"bench_{dataset}_", ""), "model": name,
                         "frames": len(idx), "abs_rel": acc["abs_rel"], "delta1": acc["delta1"],
                         "opw": opw_error(list(pred), pairs, sc)})
            print(f"{shot:32s} {name:16s} AbsRel {acc['abs_rel']:.4f}  d1 {acc['delta1']:.4f}  OPW {rows[-1]['opw']:.4f}")
    return rows


def write_results(dataset, rows):
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    with open(out / f"depth_{dataset}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    models = list(dict.fromkeys(r["model"] for r in rows))
    lines = [f"## Depth on {dataset} ({len({r['sequence'] for r in rows})} sequences, "
             f"{sum(r['frames'] for r in rows if r['model'] == models[0])} frames)", "",
             "| Model | AbsRel ↓ | δ1 ↑ | OPW (flicker) ↓ |", "|---|---|---|---|"]
    for m in models:
        rs = [r for r in rows if r["model"] == m]
        # frame-weighted mean over sequences
        wts = np.array([r["frames"] for r in rs], float)
        mean = lambda k: float(np.average([r[k] for r in rs], weights=wts))
        lines.append(f"| {m} | {mean('abs_rel'):.4f} | {mean('delta1'):.4f} | {mean('opw'):.4f} |")
    lines += ["", "AbsRel/δ1: one scale + shift per sequence fitted in disparity space against ground truth.",
              "OPW: optical-flow warping error on depth normalised to [0, 1] per sequence (lower = less flicker).",
              "+flow-align: our alignment module in `flow` mode applied to that model's output.", ""]
    (out / f"depth_{dataset}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console cannot print arrows otherwise
    ds = sys.argv[1]
    write_results(ds, evaluate(ds, rerun="--rerun" in sys.argv))
