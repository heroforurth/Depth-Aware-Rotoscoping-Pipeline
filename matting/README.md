# Matting: MatAnyone (`frames/` + `mask/0001.png` → `alpha/`)

**Run**
```powershell
python run_pipeline.py --config configs/tos_08_3a.yaml --stages matte
python -m common.preview data/tos_08_3a alpha     # object over magenta
```

**What it does:** turns the first binary SAM 2 mask into a soft alpha (0–1) for every frame:
hair, motion blur and soft edges. The mask is eroded and dilated by 10 px; the band
between the two is "unknown", and MatAnyone solves it. The first frame gets 10 warm-up passes.
Alpha is saved as 16-bit PNG and read back as float 0–1.

**Speed (8 GB laptop GPU)**
| `matting.max_size` | Speed | GPU peak | Alpha difference vs full res |
|---|---|---|---|
| 720 (default) | 0.17 s/frame | 1.5 GB | MAD 0.0015 (only fine hair edges) |
| -1 (full 1012 px) | 8.6 s/frame | spills out of VRAM | — |

Use full resolution only for a final render, ideally on a GPU with more than 8 GB.

**Licence:** MatAnyone is under the **NTU S-Lab License 1.0 (non-commercial)**. That's fine for
this university project, but not for commercial use.

**Known failures**
- Only runs forward from `matting.mask_frame`, which must be frame 1.
- An object that is cut by a foreground object (the bridge railing in `tos_01_2a`) is handled correctly:
  the railing becomes alpha 0.
- Objects that enter later (a second actor) are not added. Only what is in the first mask is tracked.
