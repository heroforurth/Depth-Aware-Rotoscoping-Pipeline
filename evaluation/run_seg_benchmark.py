"""Segmentation + matting benchmark with ground truth (held-out test sets).

    python -m evaluation.run_seg_benchmark davis          # DAVIS 2017 val, 30 videos, objects scored separately
    python -m evaluation.run_seg_benchmark youtubematte   # YouTubeMatte 512x288, 64 clips, true alpha

Every approach gets the SAME information: the true mask of the FIRST frame only
(standard "semi-supervised video object segmentation" protocol). Frame 1 is
therefore not scored. Five approaches:

  copy_first     no model: repeat the first-frame mask on every frame
                 (lower bound: shows what using NO temporal information gives)
  sam2_clicks    SAM 2 video, prompted with 3 clicks taken automatically from the
                 first mask (what our pipeline does with a user's clicks)
  sam2_mask      SAM 2 video, prompted with the whole first mask
  matanyone      MatAnyone from the first mask (authors' low-res settings:
                 erode = dilate = 4 px, warm-up 1; fixed in advance, not tuned)
  pipeline       our real pipeline: clicks -> SAM 2 -> its frame-1 mask -> MatAnyone

Metrics per clip (frames 2..N), then the mean over clips:
  J (IoU), Dice, F (boundary F-measure), J&F = (J + F) / 2         segmentation
  MAD, MSE (x1000), dtSSD (x100) on the soft alpha                   YouTubeMatte only

Predictions are cached in data/bench_seg/<dataset>/<approach>/<clip>/ so a
stopped run resumes, and the error-analysis gallery can reuse them.
"""
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from evaluation.metrics import alpha_metrics, boundary_f, dice, iou

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "datasets"
CACHE = ROOT / "data" / "bench_seg"
OUT = ROOT / "results"
APPROACHES = ["copy_first", "sam2_clicks", "sam2_mask", "matanyone", "pipeline"]
MATANYONE_SETTINGS = dict(erode=4, dilate=4, warmup=1, max_size=-1)


# ---------------- datasets: each clip = frames folder + true masks/alpha ----------------
def davis_clips():
    base = DS / "davis" / "DAVIS"
    for seq in (base / "ImageSets" / "2017" / "val.txt").read_text().split():
        jpgs = sorted((base / "JPEGImages" / "480p" / seq).glob("*.jpg"))
        ann = [np.array(Image.open(p)) for p in sorted((base / "Annotations" / "480p" / seq).glob("*.png"))]
        for obj in sorted(set(np.unique(ann[0])) - {0, 255}):
            gt = [a == obj for a in ann]
            yield {"name": f"{seq}_{obj}", "jpg_dir": jpgs[0].parent, "jpgs": jpgs, "gt": gt,
                   "gt_alpha": None, "first_mask": gt[0]}


def youtubematte_clips():
    base = DS / "youtubematte" / "YouTubeMatte" / "youtubematte_512x288"
    masks = DS / "youtubematte" / "YouTubeMatte_first_frame_seg_mask" / "youtubematte_512x288"
    for kind in ["youtubematte_motion", "youtubematte_static"]:
        for clip in sorted((base / kind).iterdir()):
            frames = sorted((clip / "har").glob("*.png"))
            pha = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255
                   for p in sorted((clip / "pha").glob("*.png"))]
            first = cv2.imread(str(masks / f"{clip.name}.png"), cv2.IMREAD_GRAYSCALE) > 127
            name = f"{kind.split('_')[1]}_{clip.name}"
            yield {"name": name, "jpg_dir": jpeg_copy(frames, CACHE / "_jpg" / "youtubematte" / name), "jpgs": frames,
                   "gt": [p > 0.5 for p in pha], "gt_alpha": pha, "first_mask": first}


def jpeg_copy(pngs, folder):
    """SAM 2 only reads a folder of JPEGs: convert once (quality 95), reuse afterwards.
    MatAnyone and the scoring still read the original PNGs."""
    if len(list(folder.glob("*.jpg"))) != len(pngs):
        folder.mkdir(parents=True, exist_ok=True)
        for k, p in enumerate(pngs):
            cv2.imwrite(str(folder / f"{k:05d}.jpg"), cv2.imread(str(p)), [cv2.IMWRITE_JPEG_QUALITY, 95])
    return folder


DATASETS = {"davis": davis_clips, "youtubematte": youtubematte_clips}


# ---------------- automatic "clicks" from a mask (deterministic) ----------------
def mask_to_clicks(mask, k=3):
    """k points deep inside the object, spread out: the first at the most interior pixel
    (distance transform maximum), each next one as far as possible from the ones chosen."""
    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    if dist.max() == 0:
        return []
    ys, xs = np.nonzero(dist >= 0.3 * dist.max())
    cand = np.stack([xs, ys], 1)[:: max(1, len(xs) // 4000)]
    pts = [np.array(np.unravel_index(dist.argmax(), dist.shape)[::-1])]
    for _ in range(k - 1):
        d = np.min([np.linalg.norm(cand - p, axis=1) for p in pts], axis=0)
        pts.append(cand[d.argmax()])
    return [[int(x), int(y)] for x, y in pts]


# ---------------- run one approach on one clip (cached) ----------------
def save_preds(folder, preds):
    folder.mkdir(parents=True, exist_ok=True)
    for k, p in enumerate(preds):
        cv2.imwrite(str(folder / f"{k:05d}.png"), np.round(np.clip(p, 0, 1) * 255).astype(np.uint8))


def load_preds(folder, n):
    files = sorted(folder.glob("*.png"))
    if len(files) != n:
        return None
    return [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255 for f in files]


class Runner:
    def __init__(self, device):
        from matting.matanyone_alpha import load_matanyone
        from segmentation.sam2_mask import load_sam2

        self.device = device
        self.sam2_model = load_sam2(device)
        self.matanyone = load_matanyone(device)

    def sam2(self, clip, prompt):
        from segmentation.sam2_mask import track

        return [m.astype(np.float32) for m in track(self.sam2_model, clip["jpg_dir"], [prompt], self.device)]

    def matte(self, clip, first_mask):
        from matting.matanyone_alpha import matte_video

        frames = (cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in clip["jpgs"])
        return list(matte_video(self.matanyone, frames, first_mask, MATANYONE_SETTINGS["erode"],
                                MATANYONE_SETTINGS["dilate"], MATANYONE_SETTINGS["warmup"],
                                MATANYONE_SETTINGS["max_size"], self.device))

    def run(self, approach, clip, cache):
        n = len(clip["jpgs"])
        if approach == "copy_first":
            return [clip["first_mask"].astype(np.float32)] * n
        if approach == "sam2_clicks":
            pts = mask_to_clicks(clip["first_mask"])
            return self.sam2(clip, {"frame_idx": 0, "points": pts, "labels": [1] * len(pts)})
        if approach == "sam2_mask":
            return self.sam2(clip, {"frame_idx": 0, "mask": clip["first_mask"]})
        if approach == "matanyone":
            return self.matte(clip, clip["first_mask"])
        if approach == "pipeline":
            clicks = load_preds(cache / "sam2_clicks" / clip["name"], n)
            return self.matte(clip, clicks[0] > 0.5)
        raise ValueError(approach)


# ---------------- scoring ----------------
def score(preds, clip):
    """Frames 2..N only (frame 1 is the given input)."""
    p_bin = [p > 0.5 for p in preds[1:]]
    gt = clip["gt"][1:]
    row = {"J": np.mean([iou(a, b) for a, b in zip(p_bin, gt)]),
           "Dice": np.mean([dice(a, b) for a, b in zip(p_bin, gt)]),
           "F": np.mean([boundary_f(a, b) for a, b in zip(p_bin, gt)])}
    row["J&F"] = (row["J"] + row["F"]) / 2
    if clip["gt_alpha"] is not None:
        row.update(alpha_metrics(np.stack(preds[1:]), np.stack(clip["gt_alpha"][1:])))
    return {k: float(v) for k, v in row.items()}


def main(dataset, limit=None):
    sys.stdout.reconfigure(encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runner = Runner(device)
    cache = CACHE / dataset
    rows, seconds, frames = [], {a: 0.0 for a in APPROACHES}, {a: 0 for a in APPROACHES}
    clips = list(DATASETS[dataset]())[:limit]
    for clip in tqdm(clips, desc=dataset):
        for a in APPROACHES:
            folder = cache / a / clip["name"]
            preds = load_preds(folder, len(clip["jpgs"]))
            if preds is None:
                t0 = time.perf_counter()
                preds = runner.run(a, clip, cache)
                seconds[a] += time.perf_counter() - t0
                frames[a] += len(preds)
                save_preds(folder, preds)
            rows.append({"dataset": dataset, "clip": clip["name"], "approach": a, "frames": len(clip["jpgs"]),
                         **score(preds, clip)})

    OUT.mkdir(exist_ok=True)
    with open(OUT / f"seg_{dataset}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    matting = clips[0]["gt_alpha"] is not None
    head = "| Approach | uses frames together? | J (IoU) ↑ | Dice ↑ | F ↑ | J&F ↑ |" + \
           (" MAD ↓ | MSE ↓ | dtSSD ↓ |" if matting else "") + " s/frame |"
    md = [f"## {dataset}: {len(clips)} clips, true masks, first-frame mask given", "", head,
          "|" + "---|" * (head.count("|") - 1)]
    temporal = {"copy_first": "no", "sam2_clicks": "yes (memory)", "sam2_mask": "yes (memory)",
                "matanyone": "yes (memory)", "pipeline": "yes (memory)"}
    for a in APPROACHES:
        rs = [r for r in rows if r["approach"] == a]
        m = lambda k: np.mean([r[k] for r in rs])
        spf = f"{seconds[a] / frames[a]:.3f}" if frames[a] else "cached"
        line = f"| {a} | {temporal[a]} | {m('J'):.3f} | {m('Dice'):.3f} | {m('F'):.3f} | {m('J&F'):.3f} |"
        if matting:
            line += f" {m('MAD'):.2f} | {m('MSE'):.2f} | {m('dtSSD'):.2f} |"
        md.append(line + f" {spf} |")
    md += ["", "Mean over clips; frame 1 (the given mask) is not scored. MAD and MSE are ×1000, dtSSD ×100.",
           "Settings were fixed before testing (SAM 2 defaults; MatAnyone: the authors' low-res settings)."]
    (OUT / f"seg_{dataset}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    lim = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    main(sys.argv[1], lim)
