# Depth alignment (`depth_raw/` → `depth/`)

**Run**
```powershell
python run_pipeline.py --config configs/tos_08_3a.yaml --stages align
python run_pipeline.py --config configs/tos_08_3a.yaml --stages align --set align.mode=per_frame
```

**Why:** depth models give relative depth with an unknown scale and shift, and a per-frame model
can pick different ones on every frame, which shows up as flicker. CLAUDE.md decision 3 makes this step required.

**Modes (`align.mode`)**
| Mode | What it does | Use for |
|---|---|---|
| `global` (default) | ONE scale/shift for the whole shot, mapping the 2%–98% depth range to 0–1 | Video Depth Anything (already consistent) |
| `per_frame` | fit `s_i·raw_i + b_i ≈ shot median` per frame on background pixels (robust least squares), then global | per-frame models (DA V2 baseline), static camera |
| `window` | same, but the reference is the median of ±`window/2` frames | slow camera moves |
| `flow` | fit each frame to the previous aligned frame, carried over by optical flow (only pixels that pass a forward/backward check) | moving camera, moving people |

Background = alpha < 0.5, so the moving actor doesn't pull the fit.
The numbers are saved to `data/<shot>/align.json`.

**Test on the fake shot with known ground truth** (`python -m common.make_dummy_shot`):
| | Error vs truth | Background flicker |
|---|---|---|
| raw / global | 0.078 | 0.066 |
| per_frame | **0.0058** | **0.0052** |

**Speed (CPU):** `global` 0.03 s/frame; `flow` 0.07 s/frame at 1920×1012 and 0.01 s/frame at 640×480 (flow computed at 1/4 size).

**Real benchmark (Bonn, moving camera, 5 sequences)**, flicker = OPW (lower is better):
| | AbsRel | Flicker |
|---|---|---|
| DA V2 raw | 0.104 | 0.0155 |
| DA V2 + `per_frame` | worse (up to 0.32 on crowd2): the median of a moving shot is meaningless | — |
| DA V2 + `flow` | 0.106 | **0.0119** (−23%) |
| VDA (global only, default) | **0.055** | **0.0068** |

So `flow` is the right flicker remover for a per-frame model. Video Depth Anything doesn't need it:
`flow` cuts its flicker slightly (0.0068 → 0.0063) but chained fits drift and cost accuracy (0.055 → 0.076).

**Known failures:** `per_frame` assumes a static camera. `flow` chains frame to frame, so small errors
add up over long shots (drift). If a fit gives a negative scale, that frame keeps its own rough normalisation (logged).
