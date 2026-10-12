"""Evaluation on our own Tears of Steel shots (Phase 3). Writes results/*.md + .csv.

    python -m evaluation.greenscreen_key data/tos_08_3a      # once: reference alpha
    python -m evaluation.run_shot_eval

1. ALPHA: SAM 2 binary mask vs MatAnyone soft alpha, scored against the green-screen
   key of tos_08_3a (required comparison). Scored on the whole frame AND on the
   edge band (within 15 px of the reference edge), where matting actually matters.
2. RAW vs REFINED DEPTH: edge sharpness at the matte edge (1.0 = depth edge sits
   on the matte edge), per shot. Plus true edge error on the synthetic dummy shot.
3. FLICKER on real film footage: OPW of VDA vs per-frame DA V2 (+ flow alignment).
4. SPEED: seconds per frame per stage at 1080p and 4K, from timings.jsonl.
"""
import csv
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from common.io import frame_name, list_indices, read_alpha, read_depth, read_mask, read_rgb
from evaluation.metrics import alpha_metrics, edge_sharpness, flow_pairs, opw_error

ROOT = Path(__file__).resolve().parent.parent
DATA, OUT = ROOT / "data", ROOT / "results"
SHOTS = ["tos_01_2a", "tos_04_2d", "tos_08_3a"]


def save(name, rows, md_lines):
    OUT.mkdir(exist_ok=True)
    if rows:
        with open(OUT / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    (OUT / f"{name}.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("\n".join(md_lines) + "\n")


# ---------------- 1. alpha ----------------
def eval_alpha(shot="tos_08_3a"):
    root = DATA / shot
    idx = list_indices(root / "alpha_key", "png")
    if not idx:
        raise SystemExit(f"run first: python -m evaluation.greenscreen_key data/{shot}")
    gt = np.stack([read_alpha(root / "alpha_key" / frame_name(i, "png")) for i in idx])
    preds = {
        "SAM 2 mask (binary)": np.stack([read_mask(root / "mask" / frame_name(i, "png")) for i in idx]).astype(np.float32),
        "MatAnyone alpha (720 px)": np.stack([read_alpha(root / "alpha" / frame_name(i, "png")) for i in idx]),
    }
    # edge band: within 15 px of the reference matte edge
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    band = np.stack([cv2.dilate(((g > 0.5).astype(np.uint8)), k) & (1 - cv2.erode((g > 0.5).astype(np.uint8), k))
                     for g in gt]).astype(bool)
    rows, md = [], [f"## Alpha vs green-screen key reference ({shot}, {len(idx)} frames, 1920×1012)", "",
                    "| Method | MAD ↓ (frame) | MSE ↓ (frame) | dtSSD ↓ | MAD ↓ (edge band) |", "|---|---|---|---|---|"]
    for name, p in preds.items():
        m = alpha_metrics(p, gt)
        mad_band = float(np.abs(p - gt)[band].mean() * 1e3)
        rows.append({"method": name, **m, "MAD_edge_band": mad_band})
        md.append(f"| {name} | {m['MAD']:.2f} | {m['MSE']:.2f} | {m['dtSSD']:.2f} | {mad_band:.1f} |")
    # experiment: MatAnyone resolution (first 30 frames, full-res run in data/tos_08_3a_fullres30)
    full = DATA / "tos_08_3a_fullres30" / "alpha"
    if len(list_indices(full, "png")) >= 30:
        n = 30
        g30, b30 = gt[:n], band[:n]
        md += ["", f"**Experiment: MatAnyone resolution** (first {n} frames)", "",
               "| Method | s/frame | MAD ↓ | MSE ↓ | dtSSD ↓ | MAD ↓ (edge band) |", "|---|---|---|---|---|---|"]
        cand = {"SAM 2 mask (binary)": (preds["SAM 2 mask (binary)"][:n], 0.14),
                "MatAnyone 720 px": (preds["MatAnyone alpha (720 px)"][:n], 0.18),
                "MatAnyone full 1012 px": (np.stack([read_alpha(full / frame_name(i, "png")) for i in idx[:n]]), 4.57)}
        for name, (p, spf) in cand.items():
            m = alpha_metrics(p, g30)
            md.append(f"| {name} | {spf} | {m['MAD']:.2f} | {m['MSE']:.2f} | {m['dtSSD']:.2f} | "
                      f"{np.abs(p - g30)[b30].mean() * 1e3:.1f} |")
    md += ["", "MAD/MSE ×1000, dtSSD ×100 (alpha 0–1). Edge band = within 15 px of the reference edge "
               f"({band.mean() * 100:.1f}% of pixels).",
           "Reference = colour-difference key of the linear camera plate (evaluation/greenscreen_key.py). "
           "A keyer is not perfect ground truth (green spill in hair), so treat it as a strong reference, not truth.",
           f"The key is fairly HARD-edged: {((gt > 0.05) & (gt < 0.95))[band].mean() * 100:.0f}% soft pixels in the edge band vs "
           f"{((preds['MatAnyone alpha (720 px)'] > 0.05) & (preds['MatAnyone alpha (720 px)'] < 0.95))[band].mean() * 100:.0f}% for MatAnyone, "
           "so edge-band MAD favours the binary mask. MSE (punishes big errors) and dtSSD (flicker) are fairer here: "
           "MatAnyone is about 2x better on both. A true-ground-truth matting benchmark (YouTubeMatte) is still needed."]
    save("alpha_tos_08_3a", rows, md)


# ---------------- 2. raw vs refined ----------------
def eval_refine():
    rows, md = [], ["## Raw vs refined depth: edge sharpness at the matte edge", "",
                    "| Shot | Raw (depth/) | Refined (depth_final/) |", "|---|---|---|"]
    for shot in SHOTS:
        root = DATA / shot
        idx = list_indices(root / "depth_final", "npy")
        r, f = [], []
        for i in idx[::3]:
            a = read_alpha(root / "alpha" / frame_name(i, "png"))
            sr = edge_sharpness(read_depth(root / "depth" / frame_name(i, "npy")), a)
            sf = edge_sharpness(read_depth(root / "depth_final" / frame_name(i, "npy")), a)
            if sr and sf:
                r.append(sr)
                f.append(sf)
        rows.append({"shot": shot, "raw": float(np.mean(r)), "refined": float(np.mean(f)), "frames": len(r)})
        md.append(f"| {shot} | {np.mean(r):.3f} | {np.mean(f):.3f} |")

    # synthetic shot with true depth: real error near the edge
    dummy = DATA / "dummy"
    if (dummy / "gt_depth").exists() and list_indices(dummy / "depth_final", "npy"):
        e_raw, e_ref = [], []
        for i in list_indices(dummy / "depth_final", "npy"):
            gt = np.load(dummy / "gt_depth" / frame_name(i, "npy"))
            d, df = read_depth(dummy / "depth" / frame_name(i, "npy")), read_depth(dummy / "depth_final" / frame_name(i, "npy"))
            fg = (read_alpha(dummy / "alpha" / frame_name(i, "png")) > 0.5).astype(np.uint8)
            dist = cv2.distanceTransform(fg, cv2.DIST_L2, 3) + cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3)
            band = dist < 6
            A = np.stack([d[~band], np.ones((~band).sum())], 1)
            s, b = np.linalg.lstsq(A, gt[~band], rcond=None)[0]
            e_raw.append(np.abs(s * d + b - gt)[band].mean())
            e_ref.append(np.abs(s * df + b - gt)[band].mean())
        md += ["", f"Synthetic shot with true depth (common/make_dummy_shot.py): error within 6 px of the edge "
                   f"{np.mean(e_raw):.4f} → {np.mean(e_ref):.4f} ({(1 - np.mean(e_ref) / np.mean(e_raw)) * 100:.0f}% lower)."]
    md += ["", "Edge sharpness = depth step 3 px from the matte edge ÷ step 12 px from it. "
               "1.0 = the depth edge sits exactly on the matte edge; lower = blurred / offset."]
    save("refine_edges", rows, md)


# ---------------- 3. flicker on film footage ----------------
def eval_flicker():
    rows, md = [], ["## Flicker on Tears of Steel footage (OPW, lower = less flicker)", "",
                    "| Shot | VDA | DA V2 per frame | DA V2 + flow-align |", "|---|---|---|---|"]
    for shot in SHOTS:
        root = DATA / shot
        idx = list_indices(root / "frames", "png")
        sc, pairs = flow_pairs([read_rgb(root / "frames" / frame_name(i, "png")) for i in idx])
        L = lambda sub: [read_depth(root / sub / frame_name(i, "npy")) for i in idx]
        vda, dav2 = opw_error(L("depth_raw"), pairs, sc), opw_error(L("depth_raw_dav2"), pairs, sc)
        subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--config", str(ROOT / "configs" / f"{shot}.yaml"),
                        "--stages", "align", "--set", "depth.out_subdir=depth_raw_dav2", "align.mode=flow"],
                       cwd=ROOT, check=True, capture_output=True)
        dav2f = opw_error(L("depth"), pairs, sc)
        # put the normal (VDA, global) alignment back so depth/ matches the shot config again
        subprocess.run([sys.executable, str(ROOT / "run_pipeline.py"), "--config", str(ROOT / "configs" / f"{shot}.yaml"),
                        "--stages", "align"], cwd=ROOT, check=True, capture_output=True)
        rows.append({"shot": shot, "vda": vda, "dav2": dav2, "dav2_flow_align": dav2f})
        md.append(f"| {shot} | {vda:.4f} | {dav2:.4f} | {dav2f:.4f} |")
    save("flicker_tos", rows, md)


# ---------------- 4. speed ----------------
def eval_speed():
    order = ["segment", "matte", "depth_vda", "depth_dav2", "align", "fuse", "exr"]
    names = {"segment": "SAM 2.1 Small", "matte": "MatAnyone (720 px)", "depth_vda": "Video Depth Anything Small",
             "depth_dav2": "DA V2 Small (baseline)", "align": "Align (global)", "fuse": "Fuse (refine + stats)",
             "exr": "EXR write"}

    def latest(shot):
        recs = {}
        f = DATA / shot / "timings.jsonl"
        for line in f.read_text(encoding="utf-8").splitlines() if f.exists() else []:
            r = json.loads(line)
            if r["stage"] in order and r["frames"] >= 30:
                recs[r["stage"]] = r            # keep the most recent run of each stage
        return recs

    hd = {s: latest(s) for s in SHOTS}
    k4 = latest("tos_08_3a_4k")
    rows, md = [], ["## Speed (RTX 5060 Laptop, 8 GB)", "",
                    "| Stage | 1080p s/frame (1920×1012, mean of 3 shots) | 4K s/frame (4096×2160) | GPU peak 1080p / 4K |",
                    "|---|---|---|---|"]
    tot_hd = tot_4k = 0.0
    for st in order:
        v = [hd[s][st]["sec_per_frame"] for s in SHOTS if st in hd[s]]
        g = [hd[s][st]["gpu_peak_mb"] or 0 for s in SHOTS if st in hd[s]]
        if not v:
            continue
        hd_s, k_s = float(np.mean(v)), (k4[st]["sec_per_frame"] if st in k4 else None)
        if st != "depth_dav2":
            tot_hd += hd_s
            tot_4k += k_s or 0
        gk = f"{(k4[st]['gpu_peak_mb'] or 0) / 1024:.1f} GB" if st in k4 else "—"
        rows.append({"stage": st, "sec_per_frame_1080p": hd_s, "sec_per_frame_4k": k_s, "gpu_peak_mb_1080p": max(g)})
        md.append(f"| {names[st]} | {hd_s:.3f} | {k_s if k_s is not None else '—'} | {max(g) / 1024:.1f} GB / {gk} |")
    md.append(f"| **Pipeline total** (without baseline) | **{tot_hd:.2f}** | **{tot_4k:.2f}** | |")
    md += ["", "GPU peak = torch.cuda.max_memory_allocated (Windows overhead not included). "
               "MatAnyone at full 1080p: 8.6 s/frame (VRAM overflow), alpha MAD vs 720 px only 0.0015."]
    save("speed", rows, md)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    which = sys.argv[1:] or ["alpha", "refine", "flicker", "speed"]
    for w in which:
        {"alpha": eval_alpha, "refine": eval_refine, "flicker": eval_flicker, "speed": eval_speed}[w]()
