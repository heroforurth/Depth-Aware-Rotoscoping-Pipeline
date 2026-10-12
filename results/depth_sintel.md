## Depth on sintel (23 sequences, 1064 frames)

| Model | AbsRel ↓ | δ1 ↑ | OPW (flicker) ↓ |
|---|---|---|---|
| vda | 0.3251 | 0.5886 | 0.0103 |
| vda+flow-align | 0.3384 | 0.5917 | 0.0078 |
| dav2 | 0.3896 | 0.5714 | 0.0250 |
| dav2+flow-align | 0.3738 | 0.5876 | 0.0163 |

AbsRel/δ1: one scale + shift per sequence fitted in disparity space against ground truth.
OPW: optical-flow warping error on depth normalised to [0, 1] per sequence (lower = less flicker).
+flow-align: our alignment module in `flow` mode applied to that model's output.
