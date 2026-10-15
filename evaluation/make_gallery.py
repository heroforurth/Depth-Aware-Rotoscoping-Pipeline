"""Error analysis: pictures of correct results and failures, next to the ground truth.

    python -m evaluation.make_gallery          -> results/figures/*.jpg + results/error_analysis.md

Segmentation (DAVIS 2017 val, YouTubeMatte): for the pipeline approach, the 3 best and 3 worst
clips by J (IoU). Each row = input frame | ground truth | prediction | error map
(red = predicted but wrong, i.e. false positive; blue = missed, i.e. false negative), shown on the
clip's WORST frame so the failure is visible.
Depth (Sintel): the 3 worst and 2 best sequences for Video Depth Anything, with an AbsRel error map.
"""
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from common.io import frame_name, list_indices
from evaluation.metrics import depth_accuracy, iou
from evaluation.run_seg_benchmark import CACHE, DATASETS, load_preds

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
FIG = OUT / "figures"
TILE_W = 360


def label(img, text):
    img = img.copy()
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def tile(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (TILE_W, round(h * TILE_W / w)), interpolation=cv2.INTER_AREA)


def overlay(rgb, mask, color):
    out = rgb.astype(np.float32)
    out[mask] = out[mask] * 0.45 + np.array(color, np.float32) * 0.55
    return out.astype(np.uint8)


def error_map(rgb, pred, gt):
    out = (rgb.astype(np.float32) * 0.35).astype(np.uint8)
    out[pred & ~gt] = (0, 0, 255)      # BGR red: false positive
    out[gt & ~pred] = (255, 80, 0)     # BGR blue: missed
    out[pred & gt] = (90, 170, 90)     # green: correct object
    return out


def seg_rows(dataset, approach="pipeline", n=3):
    rows = [r for r in csv.DictReader(open(OUT / f"seg_{dataset}.csv", encoding="utf-8"))
            if r["approach"] == approach]
    rows.sort(key=lambda r: float(r["J"]))
    return rows[:n], rows[-n:][::-1]


def seg_figure(dataset, approach="pipeline"):
    worst, best = seg_rows(dataset, approach)
    wanted = {r["clip"]: r for r in worst + best}
    clips = {c["name"]: c for c in DATASETS[dataset]() if c["name"] in wanted}
    lines, md = [], []
    for kind, rows in [("BEST", best), ("WORST", worst)]:
        for r in rows:
            c = clips[r["clip"]]
            preds = load_preds(CACHE / dataset / approach / c["name"], len(c["jpgs"]))
            js = [iou(p > 0.5, g) for p, g in zip(preds[1:], c["gt"][1:])]
            k = int(np.argmin(js)) + 1                       # the clip's worst frame
            rgb = cv2.imread(str(c["jpgs"][k]))
            pred, gt = preds[k] > 0.5, c["gt"][k]
            lines.append(np.hstack([tile(label(rgb, f"{kind} {c['name']} f{k}")),
                                    tile(label(overlay(rgb, gt, (0, 200, 0)), "ground truth")),
                                    tile(label(overlay(rgb, pred, (0, 200, 255)), f"{approach} J={js[k - 1]:.2f}")),
                                    tile(label(error_map(rgb, pred, gt), "red=false pos  blue=missed"))]))
            md.append(f"| {kind.lower()} | {c['name']} | {float(r['J']):.3f} | {float(r['F']):.3f} | "
                      f"frame {k}: J {js[k - 1]:.2f} |")
    h = max(l.shape[0] for l in lines)
    fig = np.vstack([cv2.copyMakeBorder(l, 0, h - l.shape[0] + 6, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                     for l in lines])
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"seg_{dataset}_{approach}.jpg"
    cv2.imwrite(str(path), fig, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path, md


def depth_figure(dataset="sintel", model_dir="depth_raw", n_worst=3, n_best=2):
    rows = [r for r in csv.DictReader(open(OUT / f"depth_{dataset}.csv", encoding="utf-8")) if r["model"] == "vda"]
    rows.sort(key=lambda r: float(r["abs_rel"]))
    picks = [("WORST", r) for r in rows[::-1][:n_worst]] + [("BEST", r) for r in rows[:n_best]]
    lines, md = [], []
    for kind, r in picks:
        root = ROOT / "data" / f"bench_{dataset}_{r['sequence']}"
        idx = list_indices(root / "frames", "png")
        gt = np.stack([np.load(root / "gt_depth" / frame_name(i, "npy")) for i in idx])
        pred = np.stack([np.load(root / model_dir / frame_name(i, "npy")) for i in idx])
        acc = depth_accuracy(pred, gt, 70.0)
        k = len(idx) // 2
        d = 1.0 / np.clip(acc["scale"] * pred[k] + acc["shift"], 1 / 70.0, 1e3)
        lo, hi = np.percentile(gt[k][gt[k] < 70], [2, 98])
        col = lambda x: cv2.applyColorMap((255 - np.clip((x - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8),
                                          cv2.COLORMAP_INFERNO)
        rel = np.clip(np.abs(d - gt[k]) / np.maximum(gt[k], 1e-3), 0, 0.5) / 0.5
        rgb = cv2.imread(str(root / "frames" / frame_name(idx[k], "png")))
        lines.append(np.hstack([tile(label(rgb, f"{kind} {r['sequence']}")), tile(label(col(gt[k]), "true depth")),
                                tile(label(col(d), f"VDA AbsRel {float(r['abs_rel']):.2f}")),
                                tile(label(cv2.applyColorMap((rel * 255).astype(np.uint8), cv2.COLORMAP_MAGMA),
                                           "error (bright = 50%+)"))]))
        md.append(f"| {kind.lower()} | {r['sequence']} | {float(r['abs_rel']):.3f} | {float(r['delta1']):.3f} |")
    fig = np.vstack(lines)
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / f"depth_{dataset}_vda.jpg"
    cv2.imwrite(str(path), fig, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return path, md


def per_sequence_gain(dataset="sintel"):
    """Where does the video model help most vs per-frame? AbsRel and flicker per sequence."""
    by = defaultdict(dict)
    for r in csv.DictReader(open(OUT / f"depth_{dataset}.csv", encoding="utf-8")):
        by[r["sequence"]][r["model"]] = r
    out = []
    for s, m in by.items():
        if "vda" in m and "dav2" in m:
            out.append((s, float(m["dav2"]["abs_rel"]) - float(m["vda"]["abs_rel"]),
                        float(m["dav2"]["opw"]) / max(float(m["vda"]["opw"]), 1e-9)))
    out.sort(key=lambda x: x[1])
    return out


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    md = ["# Error analysis", ""]
    for ds in ["youtubematte", "davis"]:
        if (OUT / f"seg_{ds}.csv").exists():
            path, rows = seg_figure(ds)
            md += [f"## Segmentation, {ds} (approach: pipeline = clicks → SAM 2 → MatAnyone)", "",
                   f"![{ds}](figures/{path.name})", "",
                   "| | clip | clip J | clip F | worst frame |", "|---|---|---|---|---|", *rows, ""]
    if (OUT / "depth_sintel.csv").exists():
        path, rows = depth_figure()
        md += ["## Depth, Sintel (Video Depth Anything)", "", f"![sintel](figures/{path.name})", "",
               "| | sequence | AbsRel | δ1 |", "|---|---|---|---|", *rows, ""]
        gains = per_sequence_gain()
        worse = [g for g in gains if g[1] < 0]
        md += [f"VDA beats per-frame DA V2 on AbsRel in {len(gains) - len(worse)} of {len(gains)} Sintel sequences "
               f"and flickers less in {sum(g[2] > 1 for g in gains)} of {len(gains)}.",
               "Sequences where VDA is less accurate: " + (", ".join(f"{s} ({d:+.3f})" for s, d, _ in worse) or "none") + "."]
    md += ["", "## What the failures show (written from the figures above)", "",
           "| Failure | Where | Cause | Fix / mitigation |", "|---|---|---|---|",
           "| object made of separate parts | YouTubeMatte 0023, 0012 (two people = one object) | the 3 automatic clicks "
           "all landed on one person, so SAM 2 tracked only that one; with the full first mask the same clips score "
           "J 0.99 | click on every separate part (`common/pick_points.py`), or start from a mask |",
           "| object tiny in the first frame | DAVIS lab-coat objects 1–2 (14–18 px in frame 1, ~400–500 px later) | "
           "the given first mask is a speck, so the model learns almost nothing about the object's look | "
           "prompt again on a later frame where the object is bigger |",
           "| thin structures | DAVIS paragliding-launch (lines) | 1–2 px wide, below the models' patch size | "
           "out of scope for this pipeline; manual roto |",
           "| smoke / fog | Sintel ambush_2 | no clear surface to measure depth on | flag low-confidence frames |",
           "| fast motion, bright sky | Sintel temple_2 | depth of the sky and the motion blur are ambiguous | "
           "temporal model helps flicker, not this error |",
           "| many thin overlapping stems | Sintel bamboo_2 | depth edges everywhere at 518 px input | edge refinement "
           "only near the matte, so not here |",
           "| non-human objects | DAVIS (animals, bikes) | MatAnyone is trained to matte people | use SAM 2's mask "
           "for objects, MatAnyone for people |"]
    (OUT / "error_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    main()
