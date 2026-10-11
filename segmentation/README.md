# Segmentation: SAM 2 (`frames/` → `mask/`)

**Run**
```powershell
python run_pipeline.py --config configs/tos_08_3a.yaml --stages segment
python -m common.preview data/tos_08_3a mask      # check: data/tos_08_3a/_preview/mask.jpg
```

**What it does:** you click points or draw a box on one frame in the shot config
(`segmentation.prompts`). SAM 2.1 Small (`facebook/sam2.1-hiera-small`, Apache 2.0)
makes a mask on that frame and tracks it through the video with its memory.
The output is binary (0/255). It feeds MatAnyone and is not the final alpha.

**Speed (RTX 5060 Laptop 8 GB, 1920×1012):** 0.13–0.14 s/frame.

**Prompt tips**
- Put points in the middle of large areas: face, torso, legs.
- If a part is missed (e.g. a dark patterned sleeve), add a point on it.
- Use label `0` on a nearby person who must NOT be included.
- If the object is lost later, add a second prompt entry on that frame.
- Pixel coordinates are in the original frame (x right, y down).

**Known failures**
- Dark, busy textures (plaid sleeves) can be missed: needs an extra point.
- The `cannot import name '_C'` warning is harmless: SAM 2's optional CUDA extension
  is not built, so its hole-filling post-process is skipped. MatAnyone's dilate step closes small holes anyway.
- Frames are copied to `data/<shot>/_cache/sam2_jpg/` because SAM 2 only reads JPEG folders.
