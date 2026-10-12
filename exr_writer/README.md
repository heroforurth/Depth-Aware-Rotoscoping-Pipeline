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

**Channel layout (`exr.layout`)**
| Layout | Channel names | For |
|---|---|---|
| `standard` (default) | `R, G, B, A, Z` | Nuke (Read → `depth.Z`), Fusion, Resolve. The EXR convention |
| `blender` | `ViewLayer.Combined.R/G/B/A`, `ViewLayer.Depth.Z` | Blender 4.x: Image node → **Depth** output |

Tested with Blender 4.2.23 LTS: with plain `R,G,B,A,Z`, Blender shows only Image + Alpha (no Depth).
With the `blender` layout, it shows a Depth output, and the values Blender reads are **identical**
to what we wrote (max difference 0.0, `tos_01_2a` frame 50).

```powershell
python run_pipeline.py --config configs/tos_01_2a.yaml --stages exr --set exr.layout=blender
python -m exr_writer.verify_blender data/tos_01_2a/exr/0050.exr     # needs tools/blender-4.2.23-windows-x64
```
Get portable Blender (no install): unzip `https://download.blender.org/release/Blender4.2/blender-4.2.23-windows-x64.zip` into `tools/`.

In Nuke, add a Premult node to cut out the object. In Blender, set alpha to "Straight".

**Speed:** 0.25–0.28 s/frame (ZIP compression), about 14 MB/frame at 1920×1012.

**Status:** verified with the OpenEXR library (5 channels, correct types, header) and in Blender 4.2
(`blender` layout, exact depth values). Nuke is not installed here, so it hasn't been tested, but `standard` uses
the plain `Z` channel name that Nuke maps to `depth.Z`.

**4K (4096×2160):** 1.08 s/frame, about 66 MB/frame.
