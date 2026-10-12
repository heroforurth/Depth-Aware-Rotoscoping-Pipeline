# Results summary (Phase 3)

Hardware: RTX 5060 Laptop GPU (8 GB), Python 3.11, torch 2.14 + CUDA 13.0. All models are the Small variants.
Full tables: [depth_bonn](depth_bonn.md) · [depth_sintel](depth_sintel.md) · [flicker_tos](flicker_tos.md) ·
[alpha_tos_08_3a](alpha_tos_08_3a.md) · [refine_edges](refine_edges.md) · [speed](speed.md)

## 1. Per-frame Depth Anything V2 vs Video Depth Anything (required comparison)
| Dataset | Model | AbsRel ↓ | δ1 ↑ | Flicker (OPW) ↓ |
|---|---|---|---|---|
| Bonn (real, 5 seq, 550 frames) | DA V2 Small | 0.104 | 0.929 | 0.0155 |
| | **VDA Small** | **0.055** | **0.983** | **0.0068** |
| Sintel (film, 23 seq, 1064 frames) | DA V2 Small | 0.390 | 0.571 | 0.0250 |
| | **VDA Small** | **0.325** | **0.589** | **0.0103** |

On our Tears of Steel shots (no ground truth), VDA flickers **1.7–6.3× less** than DA V2 ([flicker_tos](flicker_tos.md)).
**Result:** the video model is more accurate and 2–2.5× more stable over time, so it's the right main model.

## 2. Depth alignment / flicker removal (design decision 3)
| | Bonn AbsRel | Bonn flicker | Sintel AbsRel | Sintel flicker |
|---|---|---|---|---|
| DA V2 | 0.104 | 0.0155 | 0.390 | 0.0250 |
| DA V2 + our `flow` alignment | 0.106 | **0.0119** (−23%) | **0.374** | **0.0163** (−35%) |
| VDA + our `flow` alignment | 0.076 | 0.0063 | 0.338 | 0.0078 |

- `per_frame` alignment works on a static camera (13× lower error on the synthetic shot) but **fails on a moving camera**
  (Bonn crowd2 AbsRel 0.16 → 0.32). That's why we added the optical-flow based `flow` mode.
- For VDA, `flow` alignment lowers flicker further but costs accuracy (drift), so the default stays `global`.

## 3. SAM 2 binary mask vs MatAnyone alpha (required comparison)
Reference: green-screen key of `tos_08_3a` (100 frames, real film plate, hair and beard).
| | MSE ↓ | dtSSD (alpha flicker) ↓ | MAD ↓ | MAD edge band ↓ |
|---|---|---|---|---|
| SAM 2 mask | 1.21 | 3.05 | **1.76** | **41.8** |
| MatAnyone 720 px | **0.68** | **1.58** | 2.22 | 51.9 |
| MatAnyone full res (30 frames) | 0.77 | 1.88 | 2.14 | 43.4 |

- MatAnyone **halves big errors (MSE) and alpha flicker (dtSSD)**. On average error at the edge (MAD), it does *not* beat
  the binary mask against this reference. Partly that's the reference: the key is hard-edged (9% soft pixels vs 27%
  for MatAnyone). Partly it's real: MatAnyone misses a patch by the ear/collar, and at 720 px its edges are blurred
  by upsampling (full resolution: edge MAD 51.9 → 43.4, but 4.6 vs 0.18 s/frame).
- **Still open:** a true-ground-truth matting benchmark (YouTubeMatte). The download is blocked by Google Drive's limit.

## 4. Raw vs refined depth (edge-aware refinement, design decision 5)
| Shot | Edge sharpness raw → refined (1.0 = depth edge on the matte edge) |
|---|---|
| tos_01_2a (outdoor) | 0.59 → **0.95** |
| tos_04_2d (indoor) | 0.67 → **0.79** |
| tos_08_3a (green screen) | 0.90 → 0.90 (already sharp) |
| synthetic (true depth) | edge error **−24%** |

## 5. Speed (s/frame, see [speed](speed.md))
| | 1080p (1920×1012) | 4K (4096×2160) |
|---|---|---|
| Whole pipeline | **1.35** | **3.16** |
| Slowest stage | VDA 0.47 | EXR write 1.08 |

VDA peaks at 7.3 GB of GPU memory at both resolutions, because frames over 1080p are shrunk for the model (design decision 7).

## 6. Deliverable checks
- EXR: R, G, B, A half + Z float32; header carries Z_min/Z_max/Z_mean/Z_p02/Z_p98 (alpha > 0.5).
- Blender 4.2.23: with `exr.layout=blender`, Blender's Depth output equals our Z exactly (max diff 0.0).
- One command from `.mp4` to EXR: 100 frames in 115 s (`configs/tos_01_2a_from_mp4.yaml`).

## Limits to state in the report
- Depth is relative (scale + shift fitted to ground truth for the metrics), not metric.
- OPW uses DIS optical flow, so it has a noise floor. Compare models with each other, not with 0.
- The alpha reference is a keyer output, not hand-made ground truth.
- KITTI and ScanNet were cut (CLAUDE.md scope cut "extra datasets"). ScanNet needs a licence request.
