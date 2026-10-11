# EXR writer (`frames/` or `source_exr/` + `alpha/` + `depth_final/` → `exr/`)

**Run**
```powershell
python run_pipeline.py --config configs/tos_08_3a.yaml --stages exr
```

**Channels (decisions 1, 4, 6)**
| Channel | Type | Content |
|---|---|---|
| R, G, B | half | linear colour. Tears of Steel: copied from the original camera EXR; other footage: sRGB PNG → linear |
| A | half | the object matte from MatAnyone (straight, not premultiplied by default) |
| Z | float32 | the **full** depth map. It is **not** multiplied by alpha |

**Depth convention.** This is the ONLY place where depth is converted:
- `exr.z_mode: depth` (default): Z = distance, **larger = farther**. The shot's 0–1 disparity
  maps to Z from `z_near` (1.0) to `z_far` (10.0) via Z = 1/disparity. Relative units, not metres.
- `exr.z_mode: disparity`: Z = pipeline value, larger = closer.

**Header:** `z_mode`, `depth_note`, and the object stats `Z_min`, `Z_max`, `Z_mean`, `Z_p02`, `Z_p98`.

**In Nuke:** Read node → the `depth.Z` / `Z` channel. Add a Premult node to cut out the object.
**In Blender:** Image node → the Z output. Set alpha to "Straight".

**Speed:** 0.25–0.28 s/frame (ZIP compression), about 14 MB/frame at 1920×1012.

**Status:** files verified with the OpenEXR library (5 channels, correct types, header present).
Opening in Blender/Nuke is still to do (Phase 2 checkpoint).
