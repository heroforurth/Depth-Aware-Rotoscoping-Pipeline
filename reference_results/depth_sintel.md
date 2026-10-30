## Depth on sintel (23 sequences, 1064 frames)

| Model | AbsRel ↓ | median AbsRel ↓ | δ1 ↑ | OPW (flicker) ↓ | sequences with less flicker than dav2 |
|---|---|---|---|---|---|
| vda | 0.3251 | 0.2811 | 0.5886 | 0.0103 | 23 / 23 |
| vda+flow-align | 0.3384 | 0.2763 | 0.5917 | 0.0078 | 23 / 23 |
| dav2 | 0.3896 | 0.2731 | 0.5714 | 0.0250 | — |
| dav2+flow-align | 0.3738 | 0.2364 | 0.5876 | 0.0163 | 23 / 23 |

AbsRel/δ1: one scale + shift per sequence fitted in disparity space against ground truth.
OPW: optical-flow warping error on depth normalised to [0, 1] per sequence (lower = less flicker).
+flow-align: our alignment module in `flow` mode applied to that model's output.
AbsRel mean is frame-weighted over sequences; the median over sequences shows how much a few hard
sequences pull the mean (see results/error_analysis.md).
