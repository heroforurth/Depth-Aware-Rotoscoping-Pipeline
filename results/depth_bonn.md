## Depth on bonn (5 sequences, 550 frames)

| Model | AbsRel ↓ | δ1 ↑ | OPW (flicker) ↓ |
|---|---|---|---|
| vda | 0.0554 | 0.9826 | 0.0068 |
| vda+flow-align | 0.0762 | 0.9638 | 0.0063 |
| dav2 | 0.1037 | 0.9288 | 0.0155 |
| dav2+flow-align | 0.1059 | 0.9123 | 0.0119 |

AbsRel/δ1: one scale + shift per sequence fitted in disparity space against ground truth.
OPW: optical-flow warping error on depth normalised to [0, 1] per sequence (lower = less flicker).
+flow-align: our alignment module in `flow` mode applied to that model's output.
