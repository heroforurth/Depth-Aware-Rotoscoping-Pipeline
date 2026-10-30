## Raw vs refined depth: edge sharpness at the matte edge

| Shot | Raw (depth/) | Refined (depth_final/) |
|---|---|---|
| tos_01_2a | 0.588 | 0.945 |
| tos_04_2d | 0.670 | 0.789 |
| tos_08_3a | 0.904 | 0.903 |

Synthetic shot with true depth (common/make_dummy_shot.py): error within 6 px of the edge 0.0655 → 0.0500 (24% lower).

Edge sharpness = depth step 3 px from the matte edge ÷ step 12 px from it. 1.0 = the depth edge sits exactly on the matte edge; lower = blurred / offset.
