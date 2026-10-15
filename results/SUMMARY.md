# Results summary (in grading-rubric order)

Hardware: RTX 5060 Laptop GPU (8 GB), Python 3.11, torch 2.14 + CUDA 13.0. All models are Small variants, pretrained, used zero-shot.
Full tables: [seg_davis](seg_davis.md) · [seg_youtubematte](seg_youtubematte.md) · [depth_bonn](depth_bonn.md) ·
[depth_sintel](depth_sintel.md) · [depth_bonnval](depth_bonnval.md) · [flicker_tos](flicker_tos.md) ·
[refine_edges](refine_edges.md) · [alpha_tos_08_3a](alpha_tos_08_3a.md) · [speed](speed.md) · [error_analysis](error_analysis.md)

## 1. Task type: video analysis
Every model links frames (SAM 2 memory, MatAnyone memory propagation, Video Depth Anything temporal
attention). Each sub-task is scored separately, against a baseline that does NOT use other frames:

| Sub-task | No-temporal baseline | With frames linked | Test set |
|---|---|---|---|
| Segmentation (J&F ↑) | copy first mask: **0.229** | SAM 2: **0.883** | DAVIS 2017 val, 61 objects |
| Segmentation (J&F ↑) | copy first mask: **0.688** | MatAnyone: **0.990** | YouTubeMatte, 64 clips |
| Depth flicker (OPW ↓) | per-frame DA V2: 0.0155 / 0.0250 | VDA: **0.0068 / 0.0103**, less flicker on **28 of 28** sequences | Bonn / Sintel |

## 2. Problem, data and splits
See [docs/DATA_AND_SPLITS.md](../docs/DATA_AND_SPLITS.md): dev set (our film shots, synthetic shot) → settings;
validation (2 Bonn sequences) → confirms design choices; test (Sintel, 5 Bonn, DAVIS 2017 val, YouTubeMatte)
→ reported numbers only. Whole sequences are the split unit. No network trained; leakage check per model.

## 3. Experiments: approaches compared on the same test data
**Segmentation and matting** (everyone gets only the true first-frame mask; frame 1 not scored):

| Approach | DAVIS J ↑ | DAVIS Dice ↑ | DAVIS F ↑ | YTM J ↑ | YTM MAD ↓ | YTM dtSSD ↓ | s/frame |
|---|---|---|---|---|---|---|---|
| copy first mask (no model) | 0.219 | 0.302 | 0.239 | 0.798 | 51.54 | 3.24 | 0 |
| SAM 2, 3 automatic clicks | 0.774 | 0.836 | 0.847 | 0.922 | 30.67 | 4.73 | 0.17 |
| SAM 2, first mask | **0.854** | **0.899** | **0.912** | 0.979 | 5.56 | 4.10 | 0.17 |
| MatAnyone, first mask | 0.778 | 0.842 | 0.834 | **0.990** | **2.72** | **1.61** | 0.06 |
| pipeline: clicks → SAM 2 → MatAnyone | 0.711 | 0.790 | 0.779 | 0.940 | 25.75 | 2.05 | 0.06 |

- Our MatAnyone run reproduces the published YouTubeMatte 512×288 result (MAD 2.72), so the evaluation code is trustworthy.
- On people (YouTubeMatte), soft matting halves the error of the best hard mask (MAD 2.72 vs 5.56) and has the lowest flicker.
- On general objects (DAVIS: animals, bikes) SAM 2's mask is best: MatAnyone is a people-matting model.
- The pipeline's weak point is the FIRST frame: 3 automatic clicks sometimes miss part of the object, and that
  error stays for the whole clip. Careful clicks (or a mask) close most of the gap.

**Depth** (one scale + shift per sequence fitted to ground truth):

| Approach | Bonn AbsRel ↓ (median) | Bonn δ1 ↑ | Bonn OPW ↓ | Sintel AbsRel ↓ (median) | Sintel δ1 ↑ | Sintel OPW ↓ |
|---|---|---|---|---|---|---|
| DA V2 Small, per frame | 0.104 (0.101) | 0.929 | 0.0155 | 0.390 (0.273) | 0.571 | 0.0250 |
| DA V2 + our `flow` alignment | 0.106 (0.088) | 0.912 | 0.0119 | 0.374 (0.236) | 0.588 | 0.0163 |
| **VDA Small (pipeline default)** | **0.055 (0.049)** | **0.983** | 0.0068 | **0.325** (0.281) | 0.589 | 0.0103 |
| VDA + our `flow` alignment | 0.076 (0.060) | 0.964 | **0.0063** | 0.338 (0.276) | **0.592** | **0.0078** |

- Bonn: VDA halves the error (mean and median) and the flicker.
- Sintel: VDA has a lower mean, but the median is about the same as per-frame; VDA is more accurate on 12 of 23 sequences. Its consistent win is flicker (23 of 23).
- Validation (2 separate Bonn sequences) gives the same picture: `flow` cuts flicker (DA V2 −30%), VDA + `global` stays most accurate.

**Edge refinement** (raw vs refined depth): edge sharpness 0.59 → 0.95 (outdoor shot), 0.67 → 0.79 (indoor),
0.90 → 0.90 (green screen, already sharp); −24% edge error on the synthetic shot with true depth.

## 4. Error analysis
[error_analysis.md](error_analysis.md) shows best and worst cases next to ground truth. Main failure types:
objects made of separate parts with clicks on only one part; objects that are tiny in the first frame;
thin structures (lines); smoke and fog (depth); non-human objects for MatAnyone.

## 5. Speed
1.35 s/frame at 1080p, 3.2 s/frame at 4K for the whole pipeline on the 8 GB laptop GPU ([speed](speed.md)).

## 6. Deliverable checks
EXR with R, G, B, A (half) + Z (float32) and Z statistics in the header; Blender 4.2 reads the depth exactly
(max difference 0.0); one command from `.mp4` to EXR.

## Limits
- No network was trained; results show what pretrained models plus our alignment, refinement and packaging achieve.
- SAM 2 may have seen DAVIS data in training (see the data document); YouTubeMatte is the clean segmentation test.
- Bonn validation and test sequences share the same lab and camera.
- Depth is relative (scale + shift fitted to ground truth for the metrics); OPW depends on optical flow quality.
- The green-screen alpha test on our film shot uses a keyer as reference; YouTubeMatte is the true-alpha test.
