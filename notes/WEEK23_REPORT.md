# Week 2-3 Report

> Correction dated 2026-09-09: R3/R4 interpretations below are SUPERSEDED.
> Normalized logits used temperature 1.0, and the baseline window was selected
> on test. These historical rows are retained; they do not establish an
> architecture or logQ effect. See notes/WEEK34_REPORT.md and the dated
> D-021/D-022 corrections plus D-023. Week 2 data/embedding work remains accepted.

## 1. Task Status

| Task | Status | Acceptance criteria and evidence |
|---|---|---|
| W2-T0 | DONE | Added train-only and prior-window R0b variants, swept 6h/12h/24h/48h for both, marked old empty-window rows as degenerate, added regression test `tests/test_popularity.py`, and updated `data/stats/dataset_stats.json` with both cold definitions. Strongest denominator is `R0b-prior` 12h, run `r0b-prior_012`. |
| W2-T1 | DONE | Implemented ALS as a bounded diagnostic in `src/newsrec/baselines/als.py`. Run `r1_016` logs full-test fallback metrics plus addressability. One earlier diagnostic row, `r1_015`, is marked failed/superseded because its addressable-slice filter had mismatched query-id dtypes. |
| W2-T2 | DONE | Embedded all 65,238 corpus articles with `sentence-transformers/all-MiniLM-L6-v2`, title + abstract, max 32 tokens. `data/embeddings/minilm_frozen/` contains ignored `emb.npy`, `ids.json`, `meta.json`, and `spotcheck.json`. MiniLM loaded successfully on CUDA, so S1 did not fire. |
| W2-T3 | DONE | Implemented history-only attention user tower and projection item tower. Batch 4096 OOMed and is logged as failed run `r3_017`; batch 2048 trained successfully as `r3_018`. Loss decreased from 7.5267 to 7.4168, so S2 did not fire. |
| W2-T4 | DONE | Built exact `IndexFlatIP` over 128-d projected item vectors and evaluated full-corpus Track B. R3 run `r3_018` includes Track B metrics and Track A AUC explicitly labeled off-objective. Track B evaluation completed in 161.1s, so S3 did not fire. |
| W3-T1 | DONE | Implemented precomputed train-click logQ correction and ran R4 for seeds 0, 1, and 2: runs `r4_019`, `r4_020`, `r4_021`. |
| W3-T2 | DONE | Implemented `src/newsrec/report.py`; generated `results/trackb_results_v1.md` directly from `results/runs.csv` and tracked dataset stats. |

No stop condition halted a task. S2 did not fire because batch 2048 trained and loss decreased. S3 did not fire because full-corpus FAISS evaluation completed with batching.

## 2. Baseline Table

| Variant | Window | Run | Recall@10 | Recall@50 | Recall@100 | nDCG@10 | nDCG@50 | Coverage | Gini | Empty-window note |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| R0b train-only | 6h | `r0b_007` | 0.000000 | 0.002426 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 | 73,152 empty windows |
| R0b train-only | 12h | `r0b_008` | 0.000000 | 0.002426 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 | 73,152 empty windows |
| R0b train-only | 24h | `r0b_009` | 0.000000 | 0.002426 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 | 73,152 empty windows |
| R0b train-only | 48h | `r0b_010` | 0.000224 | 0.003265 | 0.007997 | 0.000099 | 0.000840 | 0.005855 | 0.998117 | 0 empty windows |
| R0b-prior | 6h | `r0b-prior_011` | 0.025439 | 0.051451 | 0.071616 | 0.013100 | 0.019389 | 0.004369 | 0.998096 | 61,325 empty windows |
| R0b-prior | 12h | `r0b-prior_012` | 0.035168 | 0.096438 | 0.173786 | 0.013493 | 0.027804 | 0.005012 | 0.997109 | 23,073 empty windows |
| R0b-prior | 24h | `r0b-prior_013` | 0.003881 | 0.040080 | 0.077597 | 0.001931 | 0.010085 | 0.005442 | 0.998150 | 0 empty windows |
| R0b-prior | 48h | `r0b-prior_014` | 0.002105 | 0.012158 | 0.033188 | 0.001044 | 0.003369 | 0.002345 | 0.998374 | 0 empty windows |

The denominator for all later Track B multipliers is `R0b-prior` 12h, run `r0b-prior_012`, Recall@50 = 0.096437830. This is the strongest R0b variant and the conservative denominator under invariant I4.

## 3. ALS Applicability

R1 is structurally inapplicable as a competitive rung under `split_v1`; it is useful as a diagnostic.

| Measurement | Value |
|---|---:|
| Raw train users | 50,000 |
| Raw dev users | 50,000 |
| Raw train/dev user overlap | 5,943 |
| Split train/test user overlap | 5,530 |
| Test queries with learned user factor | 8,388 / 73,152 = 11.4665% |
| Unique test candidate articles with learned item factor | 1,063 / 5,369 = 19.7988% |
| Test clicks reachable by ALS | 1,317 / 111,383 = 1.1824% |
| Addressable queries | 1,216 |
| Warm test clicked items | 363 |

Full-test R1 with strongest-R0b fallback: Recall@50 = 0.086133521, nDCG@10 = 0.012114968, coverage = 0.034044575, Gini = 0.995668045. Addressable-slice ALS: Recall@50 = 0.037554825 over 1,216 queries.

This is presentable as-is: the result explains why collaborative filtering is not a fair or useful baseline on MIND-small under this temporal/cold-start protocol.

## 4. R3

Architecture as built: frozen MiniLM-L6 article vectors, shared 384-to-128 projection head, history-only one-layer self-attention user tower with attention pooling, no user-id embedding table, dot product over normalized vectors, in-batch negatives.

Training: batch size 4096 OOMed during backward and is logged as failed run `r3_017`. Batch size 2048 trained successfully with fp16 AMP and GradScaler. Peak VRAM for the successful run was 1253.7 MB. Training took 13.60 minutes and stopped after 11 epochs with best validation Recall@50 at epoch 6.

Convergence: loss decreased from 7.526745 at epoch 1 to 7.416767 at epoch 11. Validation Recall@50 peaked at 0.007105.

Track B, run `r3_018`: Recall@10 = 0.001928, Recall@50 = 0.006773, Recall@100 = 0.012588, nDCG@10 = 0.000840, nDCG@50 = 0.001970, coverage = 0.958460, Gini = 0.581140.

Delta over denominator `r0b-prior_012`: Recall@50 delta = -0.089665195, multiplier = 0.070228x.

Track A: AUC = 0.621167, explicitly OFF-OBJECTIVE DIAGNOSTIC. It is not a benchmark result and should not be compared to published MIND impression-ranking AUCs.

## 5. R4

LogQ method: precomputed empirical train-window click frequencies. Three seeds, batch size 2048, fp16 AMP.

| Seed | Run | Recall@50 | Coverage | Gini | Best epoch | Train minutes |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `r4_019` | 0.003351 | 0.007204 | 0.998267 | 1 | 7.04 |
| 1 | `r4_020` | 0.005489 | 0.011251 | 0.997331 | 2 | 8.76 |
| 2 | `r4_021` | 0.003585 | 0.012416 | 0.997079 | 2 | 13.43 |

Aggregate: Recall@50 = 0.004141 +/- 0.000957, coverage = 0.010291 +/- 0.002233, Gini = 0.997559 +/- 0.000511.

The intended trade-off did not appear. R4 lowered Recall@50 relative to R3 and also made recommendations much more concentrated: R3 Gini was 0.581140, while R4 mean Gini was 0.997559. Report Recall@50, coverage, and Gini together; any single-metric summary is misleading here.

## 6. Decisions Logged

- D-018: R0b has train-only and prior-window variants; headline denominator is strongest.
- D-019: Cold pool has training and genuine-novelty definitions.
- D-020: ALS is a bounded applicability diagnostic, not a tuned rung.
- D-021: R3/R4 batch size is 2048 on GTX 1650 Ti.
- D-022: R4 logQ correction uses precomputed train-click frequencies.

## 7. Embedding Spot Checks

MiniLM frozen artifact: `n_items` = 65,238, `D` = 384, dtype = float32, normalized = true, device = cuda, batch size = 128, max tokens = 32. `emb_sha256` = `cd3d44e958630b14f1bfc3bca61d14639b42207025f99b822868b8bea30ae5f9`.

Five nearest-neighbour spot checks:

| Query title | Top neighbours |
|---|---|
| Chip and Joanna Gaines' First Show on Their New Magnolia Network will Feature Johnnyswim | Joanna Gaines's Second Magnolia Cookbook Is Officially Available For Preorder (0.476); Chip Gaines Says He Could See Himself Giving Son Crew Another 'Little Sibling' with Wife Joanna (0.454); What to Eat and Drink Near 'Fixer Upper' Stars Chip and Joanna Gaineses' Magnolia Market (0.447) |
| He was a member of a boy band in the 90s. Now he's an ER doctor in Dallas | No quit in Maryland safety Antoine Brooks Jr. four years after suffering career-threatening injury (0.474); Texas Medical Center Orchestra heals through music (0.459); Airman Missing In Gulf Of Mexico Id'ed As Dallas Native (0.459) |
| Kendall Brown Takes An Official Visit To MU | Dawson Garcia Will Take A 2nd Official Visit To MU (0.523); Gopher Volleyball Round: National Signing Day and Conference Race (0.422); West Virginia QB Kendall to start against former Oklahoma team (0.412) |
| 3 things we heard from Bears coordinators, including what Chuck Pagano has 'to do a better job of' and the 'huge' loss of Danny Trevathan | 4 things we heard from the Bears on Thursday, including a potential Cody Whitehair-James Daniels swap and Eddy Pineiro's newest challenge (0.622); 3 things we heard from Bears quarterback Mitch Trubisky, including building on the win over the Lions and facing the Rams' Aaron Donald (0.603); Bears DC Chuck Pagano Made $100 Bills with His Face on Them and That's Hilarious (0.602) |
| What they're saying nationally about the Saints loss to Atlanta | Was the New Orleans Saints' loss to Atlanta Falcons an anomaly? (0.744); Streaking Saints intent on keeping momentum off bye week (0.701); Week 10 winners, losers: Saints implode in stunning loss; Eagles win without playing (0.649) |

The spot checks are mixed but useful: several clusters are clearly topical, while the Dallas ER doctor example shows that frozen MiniLM over short news text is not uniformly semantic enough to trust without downstream evaluation.

## 8. Surprises

- The corrected R0b-prior baseline is not a small repair. The 12h prior-window Recall@50 is 0.096438, about 29.5x the Week 1 train-only 48h R0b and 39.8x R0a.
- The strongest prior window is 12h, not 24h or 48h. News popularity in this one-day test span is sharply local in time.
- ALS full-test metrics look near the corrected popularity baseline only because fallback dominates. The actual reachable-click fraction is 1.18%.
- R3 has very high coverage and much lower Gini, but the labelled-click recall is far below the corrected popularity baseline. The model explores broadly but rarely hits the observed clicked articles.
- LogQ correction collapsed coverage and worsened Gini in this implementation. The expected diversity trade-off did not appear.
- Batch 4096 OOMed despite Option C being comfortable for frozen embeddings. The binding memory path is the trainable history tower/backward pass, not the MiniLM embedding artifact.
- The implicit ALS library emitted an OpenBLAS threadpool warning. It did not block the run, but Week 4+ CPU baselines may be faster with `OPENBLAS_NUM_THREADS=1`.
- PyTorch emitted a transformer nested-tensor warning because `norm_first=True`. This affected an optimization path, not correctness.

## 9. Open Questions For Week 4

- R5 should report cold-start under both definitions from D-019: `unseen_in_train` for model training and `unseen_in_anything_prior` for genuine novelty.
- R4 should not be presented as a successful debiasing/diversity result yet. Week 4 needs either a corrected logQ formulation, a temperature check, or a decision to keep it as a negative result.
- R3's high coverage but low recall suggests the projection/tower is not learning the short-horizon popularity signal. Week 4 controls C3/C4/R5 need to separate "semantic retrieval is weak here" from "fine-tuned embeddings help."
- Because R0b-prior is so strong, all resume-facing multipliers should be expected to be modest unless R5 learns a real next-day popularity/cold-start signal.
