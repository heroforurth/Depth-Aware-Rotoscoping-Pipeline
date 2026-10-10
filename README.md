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
| `common/` | config, I/O, frame extraction, validator, dummy data | this file |
| `segmentation/` | SAM 2 | Phase 1 |
| `matting/` | MatAnyone | Phase 1 |
| `depth/` | Video Depth Anything, DA V2-Small baseline | Phase 1 |
| `alignment/` | global scale/shift | Phase 1 |
| `fusion/` | alpha-aware stats, edge-aware refinement | Phase 1 |
| `exr_writer/` | EXR output | Phase 1 |
| `evaluation/` | metrics, report | Phase 3 |
