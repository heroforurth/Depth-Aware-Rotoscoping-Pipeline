# Depth-Aware Rotoscoping Pipeline

Video in, multi-channel OpenEXR out: **R, G, B, A, Z** per frame
(alpha matte of a chosen object + temporally consistent depth) for Nuke, Fusion and Blender.

```
frames/ -> [A] SAM 2 mask -> MatAnyone alpha            -> alpha/
frames/ -> [B] Video Depth Anything -> scale/shift align -> depth/
alpha + depth + frames -> [C] stats + edge refinement    -> depth_final/, stats.json
                       -> [D] EXR writer                 -> exr/
```

## Setup
```powershell
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
powershell -ExecutionPolicy Bypass -File scripts/setup_models.ps1   # model code (pinned) + weights, ~520 MB
```
On Windows 11, **Smart App Control must be off**, or it blocks the PyTorch and OpenEXR DLLs
(`WinError 4551`). The alternative is WSL2 or Colab.

## Run
```powershell
python run_pipeline.py --config configs/<shot>.yaml
python run_pipeline.py --config configs/<shot>.yaml --stages align fuse exr   # only some stages
python -m common.validate data/<shot>                                         # check the interface contract
```

Test without a GPU on a synthetic shot:
```powershell
python -m common.make_dummy_shot --shot dummy
python run_pipeline.py --config configs/dummy.yaml
```

## Folder layout (interface contract)
| Folder | Content | Format |
|---|---|---|
| `data/<shot>/frames/` | input frames | `0001.png`, 8-bit RGB |
| `mask/` | SAM 2 binary mask | `0001.png`, 0/255 |
| `alpha/` | MatAnyone soft alpha | `0001.png`, 16-bit, read as float 0-1 |
| `depth_raw/` | model output | `0001.npy`, float32 H x W |
| `depth/` | after global scale/shift | `0001.npy`, float32 |
| `depth_final/` | after edge refinement | `0001.npy`, float32 |
| `exr/` | final output | `0001.exr`, RGBA half + Z float32 |
| `stats.json` | per-frame Z min/max/mean where alpha > 0.5 | JSON |
| `timings.jsonl` | seconds/frame and GPU memory per stage | JSON lines |

All file I/O goes through [common/io.py](common/io.py). Frames are numbered from 1 with 4 digits.

## Depth convention
- Inside the pipeline (`depth_raw/`, `depth/`, `depth_final/`): **native Depth Anything direction, larger = closer** (relative disparity).
- In the EXR `Z` channel: converted **once** by the EXR writer, set by `exr.z_mode`:
  `depth` (default, larger = farther) or `disparity` (larger = closer).
- Depth is **relative, not metric**. One scale/shift is fitted per shot so all frames share a scale.
- `Z` holds the full depth map; it is **not** multiplied by alpha. Use the A channel to cut out the object.

## Modules
| Folder | Stage | README |
|---|---|---|
| `common/` | config, I/O, frame extraction, validator, dummy data, previews, Tears of Steel download | this file |
| `segmentation/` | SAM 2 | [README](segmentation/README.md) |
| `matting/` | MatAnyone | [README](matting/README.md) |
| `depth/` | Video Depth Anything, DA V2-Small baseline | [README](depth/README.md) |
| `alignment/` | global / per-frame scale and shift | [README](alignment/README.md) |
| `fusion/` | alpha-aware stats, edge-aware refinement | [README](fusion/README.md) |
| `exr_writer/` | EXR output | [README](exr_writer/README.md) |
| `demo/` | demo video from the EXRs | [make_demo.py](demo/make_demo.py) |
| `evaluation/` | metrics, benchmarks, result tables | [README](evaluation/README.md) |

## Speed (RTX 5060 Laptop 8 GB, 1920×1012, 100 frames)
| Stage | s/frame | GPU peak |
|---|---|---|
| SAM 2.1 Small | 0.13 | — |
| MatAnyone (720 px) | 0.17 | 1.5 GB |
| Video Depth Anything Small | 0.40 | 7.5 GB |
| DA V2 Small (baseline) | 0.10 | — |
| Align | 0.03 | CPU |
| Fuse (refine + stats) | 0.28 | GPU |
| EXR write | 0.26 | CPU |
| **Total** | **≈ 1.3** | |

## Test footage
`python -m common.get_tears_of_steel --tos_shot 01_2a --start 100 --count 100` downloads original
camera frames from *Tears of Steel* (CC-BY 3.0, **(CC) Blender Foundation | mango.blender.org**).
The actors keep their portrait rights, so use it for tech demos only, not ads.
Shots in `configs/`: `tos_01_2a` (outdoor bridge), `tos_04_2d` (indoor, several actors), `tos_08_3a` (green screen, hair).

## Licences of the models
SAM 2.1, Video Depth Anything Small and Depth Anything V2 Small are Apache 2.0.
**MatAnyone is NTU S-Lab License 1.0 (non-commercial).**

## 4K (30 frames of `tos_08_3a_4k`, 4096×2160)
| Stage | s/frame | GPU peak |
|---|---|---|
| SAM 2.1 Small | 0.22 | 0.7 GB |
| MatAnyone (720 px) | 0.38 | 1.5 GB |
| Video Depth Anything (model at ≤1080p, upsampled) | 0.51 | 7.5 GB |
| Align | 0.17 | CPU |
| Fuse (crop around the matte edge only) | 0.81 | 0.8 GB |
| EXR write | 1.08 | CPU |
| **Total** | **≈ 3.2** | |

## Demo video
```powershell
python -m demo.make_demo data/tos_01_2a        # -> data/tos_01_2a/demo.mp4
```
2×2 panels: plate | matte | Z | a CG banner inserted *between* the actor and the background using only A + Z.

## Results
See [results/SUMMARY.md](results/SUMMARY.md). In short:
- Video Depth Anything vs per-frame DA V2: **AbsRel 0.055 vs 0.104 on Bonn, 0.325 vs 0.390 on Sintel, 2–2.5× less flicker.**
- Our `flow` alignment cuts the per-frame model's flicker by 23–35%.
- Edge refinement moves the depth edge onto the matte edge (sharpness 0.59 → 0.95 on the outdoor shot).
- Whole pipeline: 1.35 s/frame at 1080p, 3.2 s/frame at 4K on an 8 GB laptop GPU.
