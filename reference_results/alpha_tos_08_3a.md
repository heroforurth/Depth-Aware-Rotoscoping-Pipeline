## Alpha vs green-screen key reference (tos_08_3a, 100 frames, 1920×1012)

| Method | MAD ↓ (frame) | MSE ↓ (frame) | dtSSD ↓ | MAD ↓ (edge band) |
|---|---|---|---|---|
| SAM 2 mask (binary) | 1.76 | 1.21 | 3.05 | 41.8 |
| MatAnyone alpha (720 px) | 2.22 | 0.68 | 1.58 | 51.9 |

**Experiment: MatAnyone resolution** (first 30 frames)

| Method | s/frame | MAD ↓ | MSE ↓ | dtSSD ↓ | MAD ↓ (edge band) |
|---|---|---|---|---|---|
| SAM 2 mask (binary) | 0.14 | 1.90 | 1.35 | 3.73 | 40.9 |
| MatAnyone 720 px | 0.18 | 2.20 | 0.67 | 2.02 | 51.6 |
| MatAnyone full 1012 px | 4.57 | 2.14 | 0.77 | 1.88 | 43.4 |

MAD/MSE ×1000, dtSSD ×100 (alpha 0–1). Edge band = within 15 px of the reference edge (3.7% of pixels).
Reference = colour-difference key of the linear camera plate (evaluation/greenscreen_key.py). A keyer is not perfect ground truth (green spill in hair), so treat it as a strong reference, not truth.
The key is fairly HARD-edged: 9% soft pixels in the edge band vs 27% for MatAnyone, so edge-band MAD favours the binary mask. MSE (punishes big errors) and dtSSD (flicker) are fairer here: MatAnyone is about 2x better on both. A true-ground-truth matting benchmark (YouTubeMatte) is still needed.
