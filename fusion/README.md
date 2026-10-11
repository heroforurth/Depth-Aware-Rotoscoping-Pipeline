# Fusion: Z stats + edge refinement (`depth/` + `alpha/` + `frames/` → `depth_final/`, `stats.json`)

**Run**
```powershell
python run_pipeline.py --config configs/tos_08_3a.yaml --stages fuse
python run_pipeline.py --config configs/tos_08_3a.yaml --stages fuse --set fusion.refine.enabled=false   # raw vs refined comparison
```

**1. Edge-aware refinement (decision 5).** A joint bilateral filter guided by RGB + alpha:
depth is averaged only with neighbours of similar colour and matte value, so the depth edge
snaps to the actor's real edge. Only pixels within `band_px` (12 px) of the matte edge,
plus all soft-alpha pixels, are changed. The rest of the depth map is untouched.
On the fake shot, edge error drops by 24% (0.0655 → 0.0500).

**2. Alpha-aware Z stats (decision 2).** For each frame, `min`, `max`, `mean`, `p02` and `p98`
are computed only where alpha > 0.5. They're given in two units:
- `disparity`: pipeline units, larger = closer
- `z`: the units of the EXR Z channel

The `z` stats are also written into each EXR header (`Z_min`, `Z_p98`, …).
Use `p02`/`p98`: `min`/`max` are sensitive to a few stray edge pixels
(e.g. `tos_01_2a` frame 50: Z_max 7.6 but Z_p98 1.42).

**Speed:** 0.26–0.31 s/frame (GPU, 9×9 window, 2 iterations).

**Known failures:** if alpha has no pixels > 0.5 on a frame, its stats are `null`; those frames
are listed in `frames_without_object`.
