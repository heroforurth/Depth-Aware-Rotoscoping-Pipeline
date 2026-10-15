# Evaluation (Phase 3)

All result tables are written to [results/](../results/) as `.md` (for the report) and `.csv` (for plots).

## How to reproduce
```powershell
# depth benchmarks (download into datasets/ first, see below)
python -m evaluation.prepare_benchmarks bonn
python -m evaluation.prepare_benchmarks sintel
python -m evaluation.run_depth_eval bonn          # -> results/depth_bonn.md        (test)
python -m evaluation.run_depth_eval sintel        # -> results/depth_sintel.md      (test)
python -m evaluation.prepare_benchmarks bonnval
python -m evaluation.run_depth_eval bonnval       # -> results/depth_bonnval.md     (validation)

# segmentation + matting with ground truth (extract datasets first, see docs/DATA_AND_SPLITS.md)
python -m evaluation.run_seg_benchmark davis          # -> results/seg_davis.md
python -m evaluation.run_seg_benchmark youtubematte   # -> results/seg_youtubematte.md

# error analysis: best/worst examples vs ground truth
python -m evaluation.make_gallery                 # -> results/error_analysis.md, results/figures/

# our Tears of Steel shots (run the pipeline on them first)
python -m evaluation.greenscreen_key data/tos_08_3a
python -m evaluation.run_shot_eval                # -> results/alpha_tos_08_3a.md, refine_edges.md, flicker_tos.md, speed.md
```

## Metrics ([metrics.py](metrics.py))
| Metric | Measures | Better |
|---|---|---|
| AbsRel | depth error relative to the true depth (one scale + shift per sequence) | lower |
| δ1 | share of pixels within 25% of the true depth | higher |
| OPW | optical-flow warping error: depth change along motion = flicker | lower |
| MAD, MSE | alpha error (×1000) | lower |
| dtSSD | alpha flicker over time (×100) | lower |
| Edge sharpness | depth step at 3 px ÷ step at 12 px from the matte edge | closer to 1 |
| J (IoU) | overlap of predicted and true mask: intersection ÷ union | higher |
| Dice | 2 × intersection ÷ (predicted area + true area) | higher |
| F (boundary) | precision/recall of the outline, tolerance 0.8% of the diagonal (DAVIS standard) | higher |

## Datasets
| Dataset | Used for | Where | Notes |
|---|---|---|---|
| Bonn RGB-D Dynamic | depth accuracy + flicker, real people, moving camera | `datasets/bonn/rgbd_bonn_<seq>.zip` from ipb.uni-bonn.de | 5 sequences (balloon2, crowd2, crowd3, person_tracking2, synchronous), 110-frame clips from frame 30, max depth 10 m. Cite ReFusion (Palazzolo et al., IROS 2019) |
| MPI Sintel | depth accuracy, animated film | `datasets/sintel/depth.zip`, `images.zip` from sintel.is.tue.mpg.de | 23 sequences, final pass, max depth 70 m |
| Tears of Steel (green-screen shot 08_3a) | alpha, against a keyed reference | our shot | key = reference, not perfect ground truth |
| YouTubeMatte | segmentation + alpha, true ground truth | Google Drive (MatAnyone README) | 64 clips at 512×288, the authors' first-frame masks. The Drive limit sometimes blocks the download; retry later |
| DAVIS 2017 val | video object segmentation, true masks | davischallenge.org (`DAVIS-2017-trainval-480p.zip`) | 30 videos, 61 objects, each object scored on its own |
| KITTI, ScanNet | — | — | **cut** (CLAUDE.md scope cut "extra datasets"): ScanNet needs a signed licence request, and KITTI depth is a very large download for driving scenes that don't fit rotoscoping |

## Known limits
- AbsRel/δ1 use a scale + shift fitted against ground truth. That's the standard protocol for relative depth, but the pipeline itself never sees ground truth.
- OPW uses OpenCV DIS optical flow. Errors in the flow add the same noise floor to every model, so compare models with each other, not with zero.
- `per_frame` alignment assumes a static camera; on Bonn (moving camera) it made accuracy worse. Use `flow` or `window`.
