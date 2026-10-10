# Project: Depth-Aware Rotoscoping Pipeline (2D video to 3D depth, EXR output)

## Goal
Take a video (1080p, 4K later) and produce multi-channel OpenEXR files with channels R, G, B, A, Z:
alpha matte of a chosen object plus a temporally consistent depth map, for VFX compositing (Nuke, Fusion, Blender).
Deadline: 2026-10-23. Team of 8 people in 4 pairs, one pair per module. Develop at 1080p, test 4K only at the end.

## Pipeline
```
frames/ -> [A] SAM 2 mask -> MatAnyone alpha -> alpha/
frames/ -> [B] Video Depth Anything (+ Depth Anything V2-Small baseline) -> depth_raw/ -> global scale/shift alignment -> depth/
alpha/ + depth/ + frames/ -> [C] alpha-aware stats + edge-aware refinement -> depth_final/, stats.json
-> [D] EXR writer (R,G,B,A,Z) -> exr/
```

## Models
- Segmentation: SAM 2, checkpoint `facebook/sam2.1-hiera-small` (Apache 2.0). Video predictor with point or box prompts; add a second prompt on a later frame if the object is lost.
- Matting: MatAnyone (takes frames + first-frame mask, returns soft alpha). SAM 2 masks are binary, so they are NOT a final alpha.
- Depth main model: Video Depth Anything (CVPR 2025, built on Depth Anything V2, temporal head, handles long videos).
- Depth baseline: Depth Anything V2-Small (Apache 2.0). Base/Large are CC-BY-NC-4.0, so avoid them for commercial work. It flickers when run per frame; keep it only as the comparison baseline.

## Design decisions (do not change without asking the team leader)
1. Do NOT compute `Z = alpha * depth`. It zeroes the background and gives wrong depth on soft edges.
   Store the full depth map as the Z channel and alpha as the separate A channel.
2. Z_min, Z_max, Z_mean are computed only where alpha > 0.5, as per-frame metadata (`stats.json` or EXR header). Also provide robust 2% / 98% percentile versions.
3. Depth is relative, not metric, and each frame may be normalized differently. Fit ONE global scale and shift per shot (or sliding window) so frames share a scale. This is a required step, not an option.
4. Depth direction: Depth Anything outputs larger = closer. Keep native direction through the pipeline and convert once in the EXR writer. Document the convention in the README.
5. Depth is soft at object borders. Apply edge-aware refinement near the matte boundary (e.g. joint bilateral filter guided by the RGB frame).
6. EXR: RGB and A as half float, Z as 32-bit float, channel named `Z`. Verify it opens in Blender or Nuke with a visible Z.
7. 4K: process at reduced depth resolution and upsample guided by the image, or tile. Never feed raw 4K frames to the models without checking GPU memory.

## Interface contract
- Layout: `data/<shot>/frames|mask|alpha|depth_raw|depth|depth_final|exr/`
- Frame names: 4-digit numbering everywhere (`0001.png`, `0001.npy`, `0001.exr`).
- Alpha: float 0-1, same resolution as frames. Depth: float32 `.npy`, same resolution as frames.
- Prompts (SAM 2 points/boxes) live in a per-shot config file, not hard-coded.
- One entry point: `run_pipeline.py --config configs/<shot>.yaml`.

## Roles (8 people)
| # | Role | Pair | Output |
|---|---|---|---|
| 1 | Data, infra, pipeline script | with 8 | `data/<shot>/frames/`, `run_pipeline.py` |
| 2 | SAM 2 segmentation | with 3 | `mask/` |
| 3 | MatAnyone matting | with 2 | `alpha/` |
| 4 | Depth models | with 5 | `depth_raw/` |
| 5 | Depth alignment, flicker removal | with 4 | `depth/` |
| 6 | Fusion stats, edge refinement | with 7 | `depth_final/`, `stats.json` |
| 7 | EXR writer, compositing check | with 6 | `exr/` |
| 8 | Evaluation, report, demo | with 1 | results tables, report |

## Phases and checkpoints
- Phase 0 (Oct 9-11): environment, data, interface contract. Checkpoint Oct 11.
- Phase 1 (Oct 12-16): each module runs alone on all test clips. Checkpoints Oct 12, 14, 16.
- Phase 2 (Oct 16-19): integration, one command video to EXR. Checkpoint Oct 19.
- Phase 3 (Oct 19-21): evaluation and experiments. Results frozen Oct 21.
- Phase 4 (Oct 21-23): report and demo; Oct 23 is buffer.
Scope cuts if behind (drop in this order): 4K, edge refinement, temporal smoothing, extra datasets.
Must survive: SAM 2 mask, Video Depth Anything depth, alpha-aware Z stats, EXR with R,G,B,A,Z, one clear demo.

## Evaluation
- Depth accuracy: AbsRel and delta1 on Sintel, Bonn, ScanNet, KITTI (global scale alignment per sequence).
- Temporal consistency: Temporal Alignment Error (TAE) or flow-warping error; also std of depth over time on a static region.
- Alpha: MAD on YouTubeMatte and VideoMatte240K test.
- Speed: seconds per frame per module at 1080p and 4K, plus GPU memory.
- Required comparisons: per-frame Depth Anything V2 vs Video Depth Anything; SAM 2 binary mask vs MatAnyone alpha; raw vs refined depth.

## Datasets
DAVIS and SA-V (segmentation), VideoMatte240K and YouTubeMatte (matting; check VideoMatte240K licence),
Sintel, Bonn, ScanNet, KITTI (video depth), Dynamic Replica (temporal consistency), plus 3-5 own clips
(moving person, hair, cluttered background).

## Papers
SAM 2 (arXiv 2408.00714); Depth Anything V2 (arXiv 2406.09414); Video Depth Anything (arXiv 2501.12375);
Online Video Depth Anything (arXiv 2510.09182); Real-Time High-Resolution Background Matting (arXiv 2012.07810);
MatAnyone (CVPR 2025); NVDS+ (arXiv 2307.08695); DepthCrafter (arXiv 2409.02095);
Learning Temporally Consistent Video Depth from Video Diffusion Priors (arXiv 2406.01493).
Details above came from search snippets; verify dataset sizes, licences and checkpoint names on official pages before citing.

## Working rules for Claude Code
- Read the interface contract and design decisions above before writing any code.
- The user is building the whole pipeline (all roles); teammates may contribute pieces later. Keep each module in its own folder.
- Write a README per module: run command, expected time per frame, known failures.
- Pin versions in `requirements.txt`; do not upgrade PyTorch or CUDA without asking.
- Test on a short clip (about 100 frames) at 1080p before full runs.
- Use dummy alpha and depth arrays to test code that depends on another module.
- Keep results reproducible: seeds, config files, and logged timings.
- Explain code simply; the team has mixed skill levels.

## Environment notes (dev laptop)
- GPU: RTX 5060 Laptop, 8 GB VRAM (Blackwell, needs CUDA 12.8+ builds). Python 3.11.9, torch 2.14.0+cu130.
- Windows Smart App Control blocks unsigned DLLs (torch, OpenEXR). Must be off, or run in WSL2 / Colab.
- ffmpeg comes from the `imageio-ffmpeg` pip package (no system ffmpeg).
