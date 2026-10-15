# Getting data to develop with

`data/` is not in git: one shot with all its outputs is about 5 GB. Pick the option that fits your module.

| You work on | You need | Get it with |
|---|---|---|
| anything, with an NVIDIA GPU | the input videos | **already here**: `samples/videos/*.mp4` (option 1) |
| alignment, fusion, EXR writer, evaluation (no big GPU) | masks, alpha, raw depth already computed | the **dev packs** on the GitHub Release (option 2) |
| quick code tests, any computer | small fake shot with known true depth | `python -m common.make_dummy_shot` (option 3) |
| the original 16-bit camera frames, more shots, 4K | Tears of Steel raw footage | `python -m common.get_tears_of_steel` (option 4) |

## 1. Sample videos (in git)
Three 100-frame shots from *Tears of Steel*, 1920×1012, 24 fps, H.264:

| File | Shot | Target object |
|---|---|---|
| `tos_01_2a.mp4` | outdoor bridge, two actors | the woman (right) |
| `tos_04_2d.mp4` | indoor set, overlapping actors | the grey-haired man (centre) |
| `tos_08_3a.mp4` | green screen, close-up, hair and beard | the man |

Run the whole pipeline from a fresh clone (needs the setup in the main README and a GPU):
```powershell
python run_pipeline.py --config configs/tos_01_2a_from_mp4.yaml
python run_pipeline.py --config configs/tos_04_2d_from_mp4.yaml
python run_pipeline.py --config configs/tos_08_3a_from_mp4.yaml
```
The SAM 2 click points are already in those configs.

## 2. Dev packs (GitHub Release, about 1 GB each)
`tos_01_2a_devpack.zip`, `tos_04_2d_devpack.zip`, `tos_08_3a_devpack.zip`: the frames plus the GPU outputs
(`mask/` from SAM 2, `alpha/` from MatAnyone, `depth_raw/` from Video Depth Anything).

1. Download a zip from the repo's **Releases** page.
2. Unzip it in the repo root. It creates `data/<shot>/`.
3. Run the CPU stages (about 1 minute for 100 frames):
   ```powershell
   python run_pipeline.py --config configs/tos_08_3a.yaml --stages align fuse exr
   python -m common.validate data/tos_08_3a
   ```
Rebuild the packs with `python scripts/make_dev_packs.py`.

## 3. Fake shot (no download, no GPU)
```powershell
python -m common.make_dummy_shot --shot dummy
python run_pipeline.py --config configs/dummy.yaml
```
A moving ball over a sloped background, with flickering "model" depth and the true depth in `data/dummy/gt_depth/`.

## 4. Original camera footage
```powershell
python -m common.get_tears_of_steel --tos_shot 01_2a --start 100 --count 100            # HD, ~0.9 GB
python -m common.get_tears_of_steel --tos_shot 08_3a --start 100 --count 30 --res 4k --shot_name tos_08_3a_4k
python -m common.get_tears_of_steel --preview 01_2a 04_2d 08_3a                          # one frame per shot
```

## Licence
*Tears of Steel* footage: Creative Commons Attribution 3.0, **(CC) Blender Foundation | mango.blender.org**.
The actors keep their portrait rights: use it for technical work and demos, not for advertising.
