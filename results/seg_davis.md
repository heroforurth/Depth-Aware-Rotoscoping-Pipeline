## davis: 61 clips, true masks, first-frame mask given

| Approach | uses frames together? | J (IoU) ↑ | Dice ↑ | F ↑ | J&F ↑ | s/frame |
|---|---|---|---|---|---|---|
| copy_first | no | 0.219 | 0.302 | 0.239 | 0.229 | 0.000 |
| sam2_clicks | yes (memory) | 0.774 | 0.836 | 0.847 | 0.811 | 0.184 |
| sam2_mask | yes (memory) | 0.854 | 0.899 | 0.912 | 0.883 | 0.180 |
| matanyone | yes (memory) | 0.778 | 0.842 | 0.834 | 0.806 | 0.091 |
| pipeline | yes (memory) | 0.711 | 0.790 | 0.779 | 0.745 | 0.094 |

Mean over clips; frame 1 (the given mask) is not scored. MAD and MSE are ×1000, dtSSD ×100.
Settings were fixed before testing (SAM 2 defaults; MatAnyone: the authors' low-res settings).
