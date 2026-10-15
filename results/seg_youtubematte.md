## youtubematte: 64 clips, true masks, first-frame mask given

| Approach | uses frames together? | J (IoU) ↑ | Dice ↑ | F ↑ | J&F ↑ | MAD ↓ | MSE ↓ | dtSSD ↓ | s/frame |
|---|---|---|---|---|---|---|---|---|---|
| copy_first | no | 0.798 | 0.859 | 0.577 | 0.688 | 51.54 | 50.02 | 3.24 | 0.000 |
| sam2_clicks | yes (memory) | 0.922 | 0.954 | 0.889 | 0.906 | 30.67 | 29.15 | 4.73 | 0.170 |
| sam2_mask | yes (memory) | 0.979 | 0.989 | 0.984 | 0.982 | 5.56 | 4.04 | 4.10 | 0.165 |
| matanyone | yes (memory) | 0.990 | 0.995 | 0.990 | 0.990 | 2.72 | 1.01 | 1.61 | 0.059 |
| pipeline | yes (memory) | 0.940 | 0.963 | 0.918 | 0.929 | 25.75 | 23.74 | 2.05 | 0.055 |

Mean over clips; frame 1 (the given mask) is not scored. MAD and MSE are ×1000, dtSSD ×100.
Settings were fixed before testing (SAM 2 defaults; MatAnyone: the authors' low-res settings).
