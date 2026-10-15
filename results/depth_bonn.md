## Depth on bonn (5 sequences, 550 frames)

| Model | AbsRel ↓ | median AbsRel ↓ | δ1 ↑ | OPW (flicker) ↓ | sequences with less flicker than dav2 |
|---|---|---|---|---|---|
| vda | 0.0554 | 0.0489 | 0.9826 | 0.0068 | 5 / 5 |
| vda+flow-align | 0.0762 | 0.0595 | 0.9638 | 0.0063 | 5 / 5 |
| dav2 | 0.1037 | 0.1011 | 0.9288 | 0.0155 | — |
| dav2+flow-align | 0.1059 | 0.0882 | 0.9123 | 0.0119 | 5 / 5 |

AbsRel/δ1: one scale + shift per sequence fitted in disparity space against ground truth.
OPW: optical-flow warping error on depth normalised to [0, 1] per sequence (lower = less flicker).
+flow-align: our alignment module in `flow` mode applied to that model's output.
AbsRel mean is frame-weighted over sequences; the median over sequences shows how much a few hard
sequences pull the mean (see results/error_analysis.md).
