# Depth models (`frames/` → `depth_raw/`)

**Run**
```powershell
# main model: Video Depth Anything Small
python run_pipeline.py --config configs/tos_08_3a.yaml --stages depth
# baseline: Depth Anything V2 Small, per frame, into its own folder
python run_pipeline.py --config configs/tos_08_3a.yaml --stages depth --set depth.model=dav2 depth.out_subdir=depth_raw_dav2
python -m common.preview data/tos_08_3a depth_raw
```

**Output:** float32 `.npy`, same size as the frame, relative disparity: **larger = closer**,
with an unknown scale and shift. The align stage fixes the scale; the EXR writer converts it to Z.

| Model | Licence | Speed (1920×1012) | GPU peak | Temporal |
|---|---|---|---|---|
| Video Depth Anything Small (`vda`) | Apache 2.0 | 0.40 s/frame | 7.5 GB | consistent (32-frame windows, 10 overlap) |
| Depth Anything V2 Small (`dav2`) | Apache 2.0 | 0.10 s/frame | small | flickers (one frame at a time) |

Only the Small (`vits`) models are allowed. Base/Large are CC-BY-NC.

**Flicker check** (std over time of static background, lower = better; Phase 1 quick test):
| Shot | DA V2 raw | DA V2 + per-frame align | VDA |
|---|---|---|---|
| tos_01_2a | 0.0059 | 0.0054 | **0.0045** |
| tos_08_3a | 0.0483 | **0.0093** | 0.0105 |

The proper TAE / flow-warping evaluation is in Phase 3. `tos_04_2d` has a moving camera, so it needs flow-based TAE.

**Known failures / notes**
- VDA peaks at 7.5 GB, close to the 8 GB limit. Close other GPU apps. For 4K, don't raise `input_size`.
  The model input stays about 518 px, and only the upsampling cost grows.
- "xFormers not available" is harmless: plain PyTorch attention is used.
- Frames wider than 16:9 (1920×1012 is 1.9:1) get a slightly smaller input (490 px) inside VDA to save memory.
- **Long shots:** VDA runs in chunks of `depth.chunk` (120) new frames with `depth.overlap` (16) shared frames.
  Each chunk is scale-matched to the previous one on the shared frames, then **cross-faded**.
  A hard switch gave a visible pop (5× normal frame-to-frame change). With the cross-fade, the biggest jump
  equals normal motion, and the result differs from one-piece processing by only 0.4%.
- **4K:** frames taller than `depth.max_height` (1080) are shrunk before the model and the depth is upsampled.
  4096×2160: 0.51 s/frame, the same 7.5 GB peak as 1080p.
