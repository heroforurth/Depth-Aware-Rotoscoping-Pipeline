## Depth on bonnval (2 sequences, 220 frames)

| Model | AbsRel ↓ | median AbsRel ↓ | δ1 ↑ | OPW (flicker) ↓ | sequences with less flicker than dav2 |
|---|---|---|---|---|---|
| vda | 0.0642 | 0.0642 | 0.9892 | 0.0073 | 2 / 2 |
| vda+flow-align | 0.0672 | 0.0672 | 0.9894 | 0.0063 | 2 / 2 |
| dav2 | 0.0791 | 0.0791 | 0.9870 | 0.0187 | — |
| dav2+flow-align | 0.0711 | 0.0711 | 0.9887 | 0.0130 | 2 / 2 |

AbsRel/δ1: one scale + shift per sequence fitted in disparity space against ground truth.
OPW: optical-flow warping error on depth normalised to [0, 1] per sequence (lower = less flicker).
+flow-align: our alignment module in `flow` mode applied to that model's output.
AbsRel mean is frame-weighted over sequences; the median over sequences shows how much a few hard
sequences pull the mean (see results/error_analysis.md).
## Validation check of the alignment design choice (DA V2 output)
| Sequence | per_frame AbsRel | flow AbsRel |
|---|---|---|
| moving_nonobstructing_box | 0.0787 | **0.0687** |
| balloon_tracking | **0.0718** | 0.0734 |

On these two validation sequences `per_frame` does not fail; accuracy is similar for both modes. What the
validation set confirms is that `flow` lowers flicker (DA V2: 0.0187 → 0.0130, VDA: 0.0073 → 0.0063) and
that VDA with plain `global` alignment stays the most accurate choice. The large `per_frame` failure was
specific to Bonn crowd2 (many people, big camera motion), so `flow` is preferred because it never failed
that way, not because it always wins on accuracy.
