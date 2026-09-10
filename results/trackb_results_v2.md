# Track B Results Table v2

Full corpus: 65,238 articles; 42,467 never appear in labelled impressions; 5,369 are shown on the test day. Absolute Recall@K is structurally small.

R0b-prior has access to Nov 14 labels that retrieval model training does not. Empty-history fallback uses that same prior baseline for every new retrieval model.

Windows selected on validation: train-only 12h (`r0b_031`); prior 12h (`r0b-prior_051`). Test-best windows are exploratory only.

| Rung | Run IDs | Seeds | Recall@10 | Recall@50 | Delta R0b | x R0b | Delta R0b-prior | x R0b-prior | Recall@100 | nDCG@10 | nDCG@50 | Coverage@100 | Gini@100 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0a | r0a_029 | single seed | 0.000000 | 0.002426 | 0.000000 | 1.000000 | -0.094012 | 0.025155 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 |
| R0a-prior | r0a-prior_049 | single seed | 0.000415 | 0.005512 | 0.003086 | 2.272099 | -0.090926 | 0.057154 | 0.008935 | 0.000149 | 0.001278 | 0.001533 | 0.998467 |
| R0b | r0b_031 | single seed | 0.000000 | 0.002426 | 0.000000 | 1.000000 | -0.094012 | 0.025155 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 |
| R0b-prior | r0b-prior_051 | single seed | 0.035168 | 0.096438 | 0.094012 | 39.753765 | 0.000000 | 1.000000 | 0.173786 | 0.013493 | 0.027804 | 0.005012 | 0.997109 |
| R1 | r1_016 | single seed | 0.031350 | 0.086134 | 0.083708 | 35.506106 | -0.010304 | 0.893151 | 0.155023 | 0.012115 | 0.024917 | 0.034045 | 0.995668 |
| R3 | r3_061 | single seed | 0.002393 | 0.008469 | 0.006043 | 3.490912 | -0.087969 | 0.087813 | 0.014015 | 0.001111 | 0.002536 | 0.641574 | 0.841328 |
| R4 | r4_066, r4_067, r4_068 | 3 seeds | 0.002775 +/- 0.000201 | 0.007843 +/- 0.000771 | 0.005417 +/- 0.000771 | 3.233115 +/- 0.317784 | -0.088595 +/- 0.000771 | 0.081329 +/- 0.007994 | 0.013071 +/- 0.001166 | 0.001535 +/- 0.000126 | 0.002742 +/- 0.000270 | 0.056419 +/- 0.008111 | 0.994693 +/- 0.000386 |
| C1 | c1_054 | single seed | 0.001225 | 0.003832 | 0.001407 | 1.579813 | -0.092605 | 0.039740 | 0.006763 | 0.000567 | 0.001161 | 0.026304 | 0.991912 |
| C2 | c2_056 | single seed | 0.003064 | 0.011781 | 0.009355 | 4.856326 | -0.084657 | 0.122160 | 0.021557 | 0.001835 | 0.003834 | 0.935712 | 0.802693 |
| C3 | c3_057 | single seed | 0.003088 | 0.011897 | 0.009472 | 4.904370 | -0.084540 | 0.123369 | 0.021117 | 0.001479 | 0.003512 | 0.878491 | 0.858255 |
| C4 | c4_069 | single seed | 0.002393 | 0.008469 | 0.006043 | 3.490912 | -0.087969 | 0.087813 | 0.014015 | 0.001111 | 0.002536 | 0.641574 | 0.841328 |

Spread is population standard deviation across the stated seeds, not a confidence interval. R1 is the historical ALS applicability diagnostic; its fallback was selected on test in Week 2, so its end-to-end score is not a selection-clean comparison. Superseded R3/R4 runs are excluded. Track A AUC is an OFF-OBJECTIVE DIAGNOSTIC and is intentionally absent from this retrieval table.
