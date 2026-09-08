# Track B Results Table v1

Framing note: Track B retrieves from the full 65,238-article corpus. 42,467 articles never appear in any labelled impression, and only 5,369 articles are shown on the test day, so absolute Recall@K is structurally small.

Denominator: R0b-prior `r0b-prior_012` (prior, 12h), Recall@50 = 0.096437830.

| Rung | Runs | Seeds | Recall@10 | Recall@50 | x Denom | Recall@100 | nDCG@10 | nDCG@50 | Coverage | Gini | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R0a | `r0a_002` | 1 | 0.000000 | 0.002426 | 0.025155 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 | naive all-time popularity over train split clicks |
| R0b | `r0b_006` | 1 | 0.000224 | 0.003265 | 0.033857 | 0.007997 | 0.000099 | 0.000840 | 0.005855 | 0.998117 | train-only, 48h |
| R0b-prior | `r0b-prior_012` | 1 | 0.035168 | 0.096438 | 1.000000 | 0.173786 | 0.013493 | 0.027804 | 0.005012 | 0.997109 | corrected denominator, prior-window 12h |
| R1 | `r1_016` | 1 | 0.031350 | 0.086134 | 0.893151 | 0.155023 | 0.012115 | 0.024917 | 0.034045 | 0.995668 | ALS diagnostic; full-test fallback included |
| R3 | `r3_018` | 1 | 0.001928 | 0.006773 | 0.070228 | 0.012588 | 0.000840 | 0.001970 | 0.958460 | 0.581140 | single seed; Track A AUC logged as off-objective diagnostic |
| R4 | `r4_019`, `r4_020`, `r4_021` | 3 | 0.001323 +/- 0.000333 | 0.004141 +/- 0.000957 | 0.042943 +/- 0.009927 | 0.007215 +/- 0.000901 | 0.000508 +/- 0.000153 | 0.001173 +/- 0.000296 | 0.010291 +/- 0.002233 | 0.997559 +/- 0.000511 | 3 seeds; logQ correction; report Recall@50, coverage, and Gini together |
