# 02 — ENGINEERING

**Status:** Starting design. Written before any code was run or any data downloaded. **Expected to change.**
Change it freely when reality disagrees — log the change in `04_DECISIONS.md`.

Its purpose is to remove a hundred small "how should I structure this" decisions so time goes into the parts that matter. It is not a contract.

---

## 1. Environment

**Python 3.11.** PyTorch with CUDA — the exact install command depends on the driver/CUDA version from W1-T1, so get that first.

```bash
python -m venv .venv && source .venv/bin/activate
# torch install command determined AFTER nvidia-smi (W1-T1)
pip install -r requirements.txt
```

Verify:

```python
import torch
assert torch.cuda.is_available(), "CPU wheel installed — reinstall from the PyTorch index"
print(torch.__version__, torch.cuda.get_device_name(0))
```

Expected dependencies (confirm and pin):
`torch`, `sentence-transformers`, `transformers`, `faiss-cpu`, `lightgbm`, `implicit` (ALS), `scikit-learn`, `pandas`, `pyarrow`, `numpy`, `scipy`, `pyyaml`, `tqdm`, `matplotlib`, `fastapi`, `uvicorn`.

⚠ `faiss-cpu` via pip is usually fine; if it fights the environment, conda-forge is the reliable path. Don't burn a day on faiss-gpu — the corpus is small enough that CPU exact search is instant.

⚠ `implicit` (for ALS) can be awkward to install on some systems. If it resists, `scipy` + a hand-rolled ALS or `sklearn`'s `TruncatedSVD` as an MF stand-in is an acceptable substitute for R1 — log the substitution.

Pin with `pip freeze > requirements.txt` once it works.

---

## 2. Repo layout

```
two-stage-news-recommender/
├── README.md
├── requirements.txt
├── .gitignore
│
├── configs/
│   ├── base.yaml
│   ├── r3_twotower.yaml
│   ├── r4_logq.yaml
│   ├── r5_textencoder.yaml
│   └── k1_ranker.yaml
│
├── data/                          # GITIGNORED except splits/ and stats/
│   ├── raw/
│   │   ├── MINDsmall_train/       # behaviors.tsv, news.tsv, entity_embedding.vec
│   │   ├── MINDsmall_dev/
│   │   └── MINDdemo_train/        # smoke-test subset
│   ├── processed/
│   │   ├── news.parquet
│   │   ├── impressions.parquet    # exploded, one row per (impression × shown item)
│   │   └── history.parquet
│   ├── splits/
│   │   └── split_v1.parquet       # COMMITTED (small)
│   ├── stats/
│   │   └── dataset_stats.json     # COMMITTED — the numbers that go on the resume
│   ├── embeddings/<variant>/      # item vectors + ids + meta
│   └── index/<variant>.faiss
│
├── src/newsrec/
│   ├── __init__.py
│   ├── config.py                  # YAML load/merge/validate
│   ├── ingest.py                  # tsv → parquet, explode impressions
│   ├── stats.py                   # dataset_stats.json, cold-pool measurement
│   ├── splits.py                  # temporal split construction + assertions
│   ├── baselines/
│   │   ├── popularity.py          # R0a, R0b
│   │   ├── als.py                 # R1
│   │   ├── category_pop.py        # C1
│   │   └── tfidf.py               # C2
│   ├── encoders.py                # sentence-transformer load, embed, Phase-1 fine-tune
│   ├── towers.py                  # user tower (attention), item projection head
│   ├── losses.py                  # sampled softmax / InfoNCE, logQ correction
│   ├── train_retrieval.py         # R3/R4/R5
│   ├── index.py                   # FAISS build + search
│   ├── candidates.py              # generate retrieval candidates for the ranker
│   ├── features.py                # ranker feature engineering
│   ├── train_ranker.py            # K1 LightGBM (K2 DCNv2 optional)
│   ├── calibration.py             # K3 Platt / isotonic, ECE, Brier, reliability
│   ├── metrics.py                 # Track A + Track B — see §5
│   ├── runlog.py                  # append to results/runs.csv
│   └── report.py                  # runs.csv → README tables + figures
│
├── service/
│   ├── app.py
│   ├── Dockerfile
│   └── requirements-service.txt
│
├── scripts/
│   ├── download_mind.sh
│   └── smoke_test.sh              # full pipeline on MINDdemo
│
├── tests/
│   ├── test_leakage.py            # THE important one
│   ├── test_splits.py
│   ├── test_metrics.py
│   └── test_runlog.py
│
├── results/
│   ├── runs.csv                   # COMMITTED, append-only
│   ├── figures/
│   └── tables/
│
├── notes/
│   └── JOURNAL.md
└── 04_DECISIONS.md              # append-only decision log
```

### .gitignore essentials

```
data/raw/
data/processed/
data/embeddings/
data/index/
*.pth
*.faiss
*.npy
.venv/
__pycache__/
```

`results/runs.csv`, `data/splits/split_v1.parquet`, and `data/stats/dataset_stats.json` **are committed** — the reproducibility backbone.

---

## 3. Data contracts

Four schemas are the interfaces between modules. Changing them is fine; changing them silently is not.

### 3.1 `data/processed/news.parquet`

| Column | Type | Notes |
|---|---|---|
| `news_id` | str | `N12345` |
| `category`, `subcategory` | str | |
| `title`, `abstract` | str | Abstract may be empty — handle it |
| `url` | str | |
| `title_entities`, `abstract_entities` | str (JSON) | Kept raw; unused by default |
| `source_split` | str | `train` \| `dev` \| `both` |
| `first_impression_ts` | timestamp | Earliest appearance in any impression. **Derived, and the basis of cold-item definition** |
| `is_cold` | bool | True if it never appears in the training window |

⚠ `first_impression_ts` is a *proxy for publication time*, not publication time — MIND doesn't ship publish dates. Say so in the README. An article could have existed before first appearing in a logged impression.

### 3.2 `data/processed/impressions.parquet` — exploded, one row per shown item

| Column | Type | Notes |
|---|---|---|
| `impression_id` | int | |
| `user_id` | str | |
| `ts` | timestamp | Parsed from `MM/DD/YYYY HH:MM:SS AM/PM` |
| `news_id` | str | |
| `label` | int8 | 1 clicked, 0 not |
| `position` | int16 | **0-based rank within the impression as listed.** Preserve it — needed for position-bias analysis and destroyed by a careless explode |
| `n_shown` | int16 | Items in this impression |

Estimated scale: MIND-small train has on the order of 10⁵ impressions with a few dozen candidates each, so expect a few million exploded rows. Comfortable in pandas; use parquet, not CSV. **Verify actual scale in W1-T4** — if it's much larger than expected, switch to chunked processing or polars.

### 3.3 `data/processed/history.parquet`

| Column | Type |
|---|---|
| `impression_id` | int |
| `user_id` | str |
| `hist_news_id` | str |
| `hist_position` | int16 (0 = oldest) |

Long format rather than a list column — easier to join and assert on. If it's too large, a list column on the impression table is a fine alternative.

⚠ MIND's history is **not timestamped per item** — it's an ordered list of prior clicks. The leakage assertion (§6) is therefore about *construction*, not per-item timestamps: verify that history items are drawn from the pre-window period and that any custom sequence feature you build doesn't reach forward. Be precise about this in the README rather than claiming a timestamp check you can't actually perform.

### 3.4 `data/splits/split_v1.parquet` — COMMITTED, frozen

| Column | Type |
|---|---|
| `impression_id` | int |
| `split` | `train` \| `val` \| `test` |

Plus `split_v1_meta.json`:

```json
{
  "created": "...",
  "construction": "train=MINDsmall_train before cutoff, val=MINDsmall_train after cutoff, test=MINDsmall_dev",
  "val_cutoff_ts": "...",
  "train_ts_range": ["...", "..."],
  "val_ts_range": ["...", "..."],
  "test_ts_range": ["...", "..."],
  "dev_starts_after_train_ends": true,
  "n_impressions": {"train": 0, "val": 0, "test": 0},
  "n_users": {"train": 0, "val": 0, "test": 0},
  "n_cold_items_in_test": 0,
  "pct_test_impressions_with_cold_item": 0.0
}
```

Drawn once. A change makes it `split_v2` and **invalidates every prior number**.

### 3.5 `data/stats/dataset_stats.json` — the resume numbers

Written by `stats build`. Everything in `01_SPEC.md` §5.4 plus the §11.4 cold-pool measurements. **This file is the single source of truth for any dataset figure quoted anywhere** — README, resume, interview prep.

### 3.6 `data/embeddings/<variant>/`

`emb.npy` (float32, `[N_items, D]`), `ids.json` (row index → news_id), `meta.json`:

```json
{
  "variant": "minilm_finetuned_phase1",
  "base_model": "sentence-transformers/all-MiniLM-L6-v2",
  "finetuned": true,
  "finetune_data": "co-click pairs, training window only",
  "max_tokens": 32,
  "fields": "title + abstract",
  "D": 384,
  "n_items": 0,
  "created": "..."
}
```

Variants to expect: `minilm_frozen` (C3/C4 input), `minilm_finetuned_phase1` (R5 input).

**Footprint:** even at 100k articles × 384 dims × 4 bytes ≈ 150 MB. Trivial. Keep float32; no need for fp16 here.

### 3.7 `results/runs.csv` — append-only, COMMITTED

Columns per `01_SPEC.md` §16. Written **at completion including on failure** (`status` field). Invariant I8.

`report.py` reads only this file to generate every README table.

---

## 4. Config

**Week 3 repair update:** `configs/base.yaml` documents project settings. The
current experiment scripts pass explicit arguments (rather than loading YAML)
and serialize them into `config_json`. Retrieval uses batch 2048 and
validation-selected temperature 0.05. The runnable Week 3/4 commands are
`scripts/select_baselines.py`, `scripts/week34.py {sweep,r3,r4,r5}`,
`scripts/evaluate_week34.py {r3,r4,controls,c4,r5}`, and
`python -m newsrec.finetune --temperature 0.05`. Training and test evaluation
are separate commands and rows. The prospective YAML/CLI design below is not
fully implemented; do not assume its `--config` examples are current APIs.
Generate the consolidated report with `scripts/report_week34.py`.

YAML with `base.yaml` defaults and experiment overrides. Resolve into a validated dataclass so typos fail loudly.

```yaml
# configs/base.yaml
data:
  processed_dir: data/processed
  split_file: data/splits/split_v1.parquet
  min_history_len: 1               # policy for empty-history users — see spec §9.1
  empty_history_policy: fallback_r0b   # fallback_r0b | learned_default | exclude

item_tower:
  mode: frozen_finetuned           # frozen_offshelf | frozen_finetuned | end_to_end
  base_model: sentence-transformers/all-MiniLM-L6-v2
  max_tokens: 32
  fields: [title, abstract]
  projection_dim: 128

user_tower:
  type: attention                  # attention | meanpool | gru
  heads: 4
  max_history: 50
  embed_dim: 128
  dropout: 0.2

train:
  batch_size: 4096                 # LOG THIS — spec §7, a real variable
  epochs: 20
  lr: 1.0e-3
  weight_decay: 1.0e-5
  amp: true
  logq_correction: false           # true at R4
  early_stop_metric: val_recall50
  early_stop_patience: 5

retrieval:
  top_k: 200
  index_type: IndexFlatIP

ranker:
  model: lightgbm
  num_leaves: 63
  learning_rate: 0.05
  n_estimators: 500
  mix_retrieval_negatives: true    # spec §10 mitigation

eval:
  recall_k: [10, 50, 100]
  ndcg_k: [10, 50]
  popularity_window_hours: 48      # R0b — tune, may need shortening (spec §6.3)

run:
  seed: 0
  rung: R3
  machine: laptop
  out_dir: results
```

**Model selection:** select on **validation** metric, report **test**. Never select on test. Never report best-of-many test numbers.

---

## 5. Metrics — define once, use everywhere

`metrics.py` is where the Track A / Track B distinction becomes code. Keep the two families in separate namespaces so they can't be accidentally mixed.

### 5.1 Track A — within-impression ranking

Input: for each impression, the shown items with scores and labels.

- `auc` — per-impression AUC, averaged over impressions (this is the MIND convention, **not** a global pooled AUC — they differ, and the published numbers are the per-impression version)
- `mrr`, `ndcg@5`, `ndcg@10`

Skip impressions with all-positive or all-negative labels; **report how many were skipped.**

### 5.2 Track B — full-corpus retrieval

Input: for each impression (or user-time point), a top-K list from the full corpus plus the set of items actually clicked.

- `recall@K` — fraction of clicked items appearing in top-K
- `ndcg@K`
- `coverage` — distinct items appearing in any top-K, over corpus size
- `gini` — over recommended-item frequency across all queries

⚠ **Define the retrieval candidate universe explicitly and identically across all rungs.** Options: all articles, or articles seen in the training window, or articles "live" at query time. They give different numbers and are not comparable. Whatever you choose, apply it to R0a, R0b, R1, R3, R4, R5 alike, and record it in `runs.csv` — otherwise the "X× over baseline" headline compares different tasks.

### 5.3 Cold-start

- `cold_recall@K_restricted` — cold items retrieved from the cold pool only
- `cold_recall@K_fullcorpus` — cold items retrieved from the full corpus
- `cold_share_topk` — fraction of top-K that is cold

Never conflate the first two (invariant I7).

### 5.4 Calibration

- `ece` (specify bin count — 10 or 15, and keep it fixed)
- `brier`
- reliability diagram → `results/figures/`

---

## 6. Tests worth writing

Three tests that catch the failures that would actually ruin the project.

**`test_leakage.py`** (the important one)
- No impression appears in more than one split
- `max(train.ts) <= min(val.ts)` and `max(val.ts) <= min(test.ts)`
- No article in the encoder's Phase-1 fine-tuning set appears only in the test period
- Popularity features computed for a given timestamp use no data after it — test with a synthetic fixture
- History items for a training impression are drawn only from the pre-window click record

**`test_metrics.py`**
- Per-impression AUC on a hand-built toy example matches a hand-computed value
- `recall@K` on a toy case with known overlap is exact
- Gini of a uniform distribution ≈ 0; Gini of a single-item-dominant distribution ≈ 1
- Track A and Track B functions raise if handed each other's input shape

**`test_runlog.py`**
- Appending preserves prior rows
- A crashed run still writes `status=failed`

`scripts/smoke_test.sh` runs the whole pipeline on **MINDdemo** and must pass before MIND-small is processed.

---

## 7. CLI surface

Plain `argparse`, invoked as modules.

```bash
# Ingest
python -m newsrec.ingest --raw data/raw --out data/processed --dataset small
python -m newsrec.stats build --processed data/processed --out data/stats/dataset_stats.json
python -m newsrec.stats cold-pool     # spec §11.4 — gates week 4

# Splits (run ONCE)
python -m newsrec.splits build --out data/splits/split_v1.parquet --val-frac 0.12
python -m newsrec.splits verify --split data/splits/split_v1.parquet

# Baselines
python -m newsrec.baselines.popularity --mode naive        # R0a
python -m newsrec.baselines.popularity --mode recency --window-hours 48   # R0b
python -m newsrec.baselines.als --factors 128              # R1

# Embeddings
python -m newsrec.encoders embed --variant minilm_frozen
python -m newsrec.encoders finetune --out-variant minilm_finetuned_phase1   # Phase 1
python -m newsrec.encoders embed --variant minilm_finetuned_phase1

# Retrieval
python -m newsrec.train_retrieval --config configs/r3_twotower.yaml --seed 0
python -m newsrec.train_retrieval --config configs/r4_logq.yaml --seed 0,1,2

# Index + candidates
python -m newsrec.index build --emb data/embeddings/<variant> --out data/index/<variant>.faiss
python -m newsrec.candidates generate --run-id <retrieval_run_id> --top-k 200

# Ranker
python -m newsrec.train_ranker --config configs/k1_ranker.yaml
python -m newsrec.calibration fit --ranker-run-id <id> --method isotonic

# Report
python -m newsrec.report --runs results/runs.csv --out results/tables/
```

`--override key.subkey=value` for one-offs; recorded in `config_json`.

---

## 8. Two implementation notes worth getting right early

### 8.1 logQ needs a frequency estimate, and where it comes from matters

With normalized towers, training logits are `(user @ item.T) / temperature`.
The selected temperature is explicit in every training call and run config.
Subtract precomputed train target log-probabilities only after this scaling.
Record the logQ/scaled-logit range ratio over sampled classes at training start;
an assertion exceeding 2x is a logged diagnostic per D-022, not a silent crash.
Sampling correction does not guarantee a reduction in exposure Gini (D-026).

`Q(i)` is the probability item `i` appears as an in-batch negative. Two implementations:

- **Streaming counter** (as in Yi et al.) — updated during training, adapts to the sampling distribution actually seen. Closer to the paper.
- **Precomputed empirical frequency** from the training set — simpler, static, easier to reason about.

Either is defensible. **Pick one, implement it, and be able to explain the difference** — "why not just use the global frequency?" is a natural follow-up question. Log which one you used.

### 8.2 Ranker training candidates

The ranker trains on `(impression, shown item, label)` rows. The §10 mitigation adds retrieval-generated candidates as extra negatives. Two things to keep straight:

- A retrieval candidate that *was* shown in the impression already has a real label — use it, don't duplicate it
- A retrieval candidate that was never shown has **no label**; treating it as a negative is an assumption, not a fact. It's a common and defensible one, but say so in the README rather than silently labelling unshown items as 0

Track the overlap rate between retrieved and shown candidates — it's the number that quantifies the selection-bias gap (§10).

---

## 9. Reproducibility hygiene

- One `set_seed()` covering `random`, `numpy`, `torch` at run start
- Record the git commit in every run row; commit before any run whose numbers get reported
- Log the resolved config, not the file path
- Record `split_version` in every run row so stale comparisons are detectable
- Generate README tables with `report.py`; never hand-type a number

---

## 10. Estimated run count

| Group | Runs |
|---|---|
| R0a, R0b (window variants) | ~4 |
| R1 (ALS, a couple of factor settings) | ~3 |
| R3 | ~2 |
| R4 × 3 seeds (+ logQ off/on comparison) | ~6 |
| R5 × 3 seeds | ~3 |
| C1, C2, C3, C4 (C4 × 3 seeds) | ~7 |
| K1 (+ feature iterations) | ~4 |
| K3 calibration variants | ~3 |

**~30–35 logged runs.** Modest compared to a grid search, but well past the point where hand-tracking is safe. `runlog.py` from the first experiment.
