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
| `window` | same, but the reference is the median of ±`window/2` frames | moving camera |

Background = alpha < 0.5, so the moving actor doesn't pull the fit.
The numbers are saved to `data/<shot>/align.json`.

**Test on the fake shot with known ground truth** (`python -m common.make_dummy_shot`):
| | Error vs truth | Background flicker |
|---|---|---|
| raw / global | 0.078 | 0.066 |
| per_frame | **0.0058** | **0.0052** |

**Speed:** 0.03 s/frame (CPU).

**Known failures:** `per_frame` on a moving camera with big depth changes can cancel real motion,
so use `window`. If a fit gives a negative scale, that frame keeps its own rough normalisation (logged).
