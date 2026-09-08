# Week 1 Report — Access, Data, Guardrails, R0

## 1. Task-by-task status

| Task | Acceptance criteria | Status | Evidence |
|---|---|---|---|
| W1-T2 | Verified MIND access; extracted train/dev/demo; behavior/news files parse; no body-text field assumed without checking | Met | Recommenders/MIND mirror downloaded and extracted. All four archives contain `behaviors.tsv`, `news.tsv`, `entity_embedding.vec`, `relation_embedding.vec`. All behavior rows have 5 fields; all news rows have 8 fields. No body-text field is present. Raw data remains ignored and untracked. |
| W1-T3 | Repo skeleton and environment | Met | Module skeleton exists for all planned `src/newsrec` files. Imports are clean. `requirements.txt` is pinned. `faiss-cpu`, `implicit`, `lightgbm`, `pyarrow`, `sentence_transformers`, and `torch` import successfully. |
| W1-T4 | Ingest and real dataset stats | Met | Ingest ran on MINDdemo first, then full MIND-small. `data/processed/{news,impressions,history}.parquet` and `data/stats/dataset_stats.json` were produced. |
| W1-T5 | Temporal split constructed and verified | Met | `split_v1` is frozen. Train = Nov 9-13, val = Nov 14, test = all MINDsmall_dev Nov 15. All three split assertions passed. |
| W1-T6 | Cold pool measured | Met | Cold-pool fields were added to `dataset_stats.json` and split metadata. |
| W1-T7 | Leakage and split assertions are code, not prose | Met | `pytest` passed 11 tests. The history test verifies our processed history table exactly matches MIND's raw History field; it does not claim per-click timestamps exist. |
| W1-T8 | Run log and metrics modules | Met | `results/runs.csv` exists, a dummy row was written before real baselines, and Track A/Track B metric namespaces reject each other's shapes. |
| W1-T9 | R0a and R0b evaluated on Track B | Met | R0a and four R0b window rows are in `results/runs.csv`, evaluated over the full-corpus candidate universe. |

## 2. THE NUMBERS

### Date range first

| Source | Min timestamp | Max timestamp | Span days |
|---|---:|---:|---:|
| MINDsmall_train | 2019-11-09 00:00:19 | 2019-11-14 23:59:13 | 5.999236 |
| MINDsmall_dev | 2019-11-15 00:00:01 | 2019-11-15 23:58:03 | 0.998634 |
| Combined labelled impressions | 2019-11-09 00:00:19 | 2019-11-15 23:58:03 | 6.998426 |

A3 held: `min(raw_dev.ts) >= max(raw_train.ts)` is true. The dev split starts 48 seconds after the train file ends.

This confirms the spec's warning: MIND-small gives roughly one week of labelled impressions, not six. Consequences:

- R0b 6h/12h/24h windows cannot see train-split clicks on the Nov 15 test day because Nov 14 is held out as validation; they fall back to all-time popularity.
- The 48h R0b window is the only swept window with useful train-click mass for test-time recency.
- The original six-week temporal decay experiment is not runnable on MIND-small. A short day-by-day or cold-share turnover plot remains the viable Week 7 replacement.
- The cold pool is large despite the short train-to-dev gap.

### Full dataset_stats.json

```json
{
  "ARTICLE_COUNTS_BY_NEWS_FILE_PRESENCE": {
    "both": 28460,
    "dev_only": 13956,
    "train_only": 22822
  },
  "AVG_HISTORY": 32.462412598808434,
  "CANDIDATES_PER_IMPRESSION_DISTRIBUTION": {
    "histogram": {
      "10": 6736,
      "11": 5749,
      "12": 6004,
      "13": 4107,
      "14": 4404,
      "15": 2928,
      "16": 3416,
      "17": 5344,
      "18": 2830,
      "19": 3282,
      "2": 14434,
      "20": 3671,
      "21": 3093,
      "22": 4737,
      "23": 3262,
      "24": 2647,
      "25": 2847,
      "26": 3584,
      "27": 3661,
      "28": 3124,
      "29": 2413,
      "3": 3989,
      "30": 3139,
      "31": 2078,
      "32": 2640,
      "33": 2042,
      "34": 2008,
      "35": 2032,
      "36": 1838,
      "37": 2375,
      "38": 1973,
      "39": 1608,
      "4": 8898,
      "40": 1617,
      "41": 1793,
      "42": 1907,
      "43": 2020,
      "44": 1669,
      "45": 1401,
      "46": 1415,
      "47": 1463,
      "48": 1674,
      "49": 1742,
      "5": 5080,
      "50": 1567,
      "6": 4036,
      "7": 10468,
      "8": 3489,
      "9": 3851,
      ">50": 58032
    },
    "summary": {
      "max": 299.0,
      "mean": 37.30468413893802,
      "min": 2.0,
      "p25": 10.0,
      "p50": 24.0,
      "p75": 51.0,
      "p90": 92.0,
      "p95": 118.0,
      "p99": 171.0
    }
  },
  "COLD_POOL": {
    "fraction_test_clicks_on_cold_items": 0.8682653546771051,
    "fraction_test_impressions_with_at_least_one_cold_item": 0.999876968503937,
    "n_test_clicks": 111383,
    "n_test_clicks_on_cold_items": 96710,
    "n_test_period_articles_absent_from_training_window": 3810
  },
  "CTR": 0.04050665145154455,
  "DATE_RANGE": {
    "max": "2019-11-15 23:58:03",
    "min": "2019-11-09 00:00:19",
    "span_days": 6.998425925925926
  },
  "DATE_RANGE_BY_SOURCE": {
    "dev": {
      "max": "2019-11-15 23:58:03",
      "min": "2019-11-15 00:00:01",
      "span_days": 0.9986342592592593
    },
    "train": {
      "max": "2019-11-14 23:59:13",
      "min": "2019-11-09 00:00:19",
      "span_days": 5.999236111111111
    }
  },
  "FRACTION_EMPTY_HISTORY": 0.023692295658295563,
  "FRACTION_VERY_SHORT_HISTORY_LT5": 0.13537026816793196,
  "HISTORY_LENGTH_DISTRIBUTION": {
    "histogram": {
      "0": 5452,
      "1": 3239,
      "10": 6046,
      "11": 5963,
      "12": 5333,
      "13": 5168,
      "14": 4894,
      "15": 4600,
      "16": 4361,
      "17": 4146,
      "18": 3926,
      "19": 3826,
      "2": 5442,
      "20": 3479,
      "21": 3370,
      "22": 3359,
      "23": 3288,
      "24": 3112,
      "25": 3196,
      "26": 2866,
      "27": 2728,
      "28": 2519,
      "29": 2401,
      "3": 7933,
      "30": 2626,
      "31": 2292,
      "32": 2226,
      "33": 2184,
      "34": 2087,
      "35": 1923,
      "36": 1922,
      "37": 1958,
      "38": 1721,
      "39": 1903,
      "4": 9085,
      "40": 1698,
      "41": 1898,
      "42": 1612,
      "43": 1630,
      "44": 1351,
      "45": 1839,
      "46": 1547,
      "47": 1309,
      "48": 1190,
      "49": 1198,
      "5": 8777,
      "50": 1219,
      "6": 8191,
      "7": 7685,
      "8": 6898,
      "9": 6390,
      ">50": 45111
    },
    "summary": {
      "max": 558.0,
      "mean": 32.462412598808434,
      "min": 0.0,
      "p25": 8.0,
      "p50": 19.0,
      "p75": 42.0,
      "p90": 78.0,
      "p95": 108.0,
      "p99": 190.0
    }
  },
  "INGEST_MANIFEST": {
    "behavior_field_counts": {
      "dev": {
        "all_rows_match_expected": true,
        "bad_rows": 0,
        "first_row_field_count": 5,
        "rows_with_5_fields": 73152
      },
      "train": {
        "all_rows_match_expected": true,
        "bad_rows": 0,
        "first_row_field_count": 5,
        "rows_with_5_fields": 156965
      }
    },
    "body_text_field_present": false,
    "dataset": "small",
    "news_field_counts": {
      "dev": {
        "all_rows_match_expected": true,
        "bad_rows": 0,
        "first_row_field_count": 8,
        "rows_with_8_fields": 42416
      },
      "train": {
        "all_rows_match_expected": true,
        "bad_rows": 0,
        "first_row_field_count": 8,
        "rows_with_8_fields": 51282
      }
    },
    "news_union": {
      "both": 28460,
      "conflicting_shared_news_rows": 0,
      "dev_only": 13956,
      "duplicates_dropped_dev": 0,
      "duplicates_dropped_train": 0,
      "train_only": 22822
    },
    "outputs": {
      "history": "data/processed/history.parquet",
      "impressions": "data/processed/impressions.parquet",
      "news": "data/processed/news.parquet"
    },
    "raw_dev_dir": "data/raw/MINDsmall_dev",
    "raw_train_dir": "data/raw/MINDsmall_train"
  },
  "NEWS_WITHOUT_FIRST_IMPRESSION_TS": 42467,
  "N_ARTICLES": 65238,
  "N_CANDIDATES": 8584442,
  "N_CANDIDATES_BY_SOURCE": {
    "dev": 2740998,
    "train": 5843444
  },
  "N_CLICKS": 347727,
  "N_CLICKS_BY_SOURCE": {
    "dev": 111383,
    "train": 236344
  },
  "N_IMPRESSIONS": 230117,
  "N_IMPRESSIONS_BY_SOURCE": {
    "dev": 73152,
    "train": 156965
  },
  "N_USERS": 94057,
  "N_USERS_BY_SOURCE": {
    "dev": 50000,
    "train": 50000
  }
}
```

Pandas verdict: keep pandas. The exploded table has 8,584,442 rows, below the ~20M threshold, and no memory pressure appeared during ingest or stats.

MIND-small verification note: each official split has 50,000 users, which matches the MIND-small shape. The combined train ∪ dev labelled user union is 94,057, because the two 50,000-user files are not identical user sets.

## 3. Split

Construction: train = `MINDsmall_train` before the validation cutoff; val = `MINDsmall_train` at/after the cutoff; test = all `MINDsmall_dev`.

Cutoff: `2019-11-14 00:00:00` via `last_calendar_day`.

| Split | Impressions | Users | Time range |
|---|---:|---:|---|
| train | 126,695 | 46,012 | 2019-11-09 00:00:19 to 2019-11-13 23:59:21 |
| val | 30,270 | 20,179 | 2019-11-14 00:00:00 to 2019-11-14 23:59:13 |
| test | 73,152 | 50,000 | 2019-11-15 00:00:01 to 2019-11-15 23:58:03 |

User overlap:

| Pair | Overlap users |
|---|---:|
| train/val | 16,191 |
| train/test | 5,530 |
| val/test | 3,236 |

Assertions quoted from code:

```text
max(train.ts) <= min(val.ts)
max(val.ts) <= min(test.ts)
min(raw_dev.ts) >= max(raw_train.ts)
```

All three passed.

## 4. Cold pool

| Measure | Value |
|---|---:|
| Test-period articles absent from training window | 3,810 |
| Fraction of test impressions with >=1 cold item | 99.9877% |
| Fraction of test clicks landing on cold items | 86.8265% |
| Cold test clicks | 96,710 / 111,383 |

Read: R5 is viable under the strict absent-from-training-window definition. Cold items are not a tiny edge case here; they dominate clicked items in the test period.

## 5. R0a vs R0b

| Baseline | Window | Recall@10 | Recall@50 | Recall@100 | nDCG@10 | nDCG@50 | Coverage | Gini |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R0a | all-time | 0.000000 | 0.002426 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 |
| R0b | 6 | 0.000000 | 0.002426 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 |
| R0b | 12 | 0.000000 | 0.002426 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 |
| R0b | 24 | 0.000000 | 0.002426 | 0.002937 | 0.000000 | 0.000551 | 0.001533 | 0.998467 |
| R0b | 48 | 0.000224 | 0.003265 | 0.007997 | 0.000099 | 0.000840 | 0.005855 | 0.998117 |

Strongest R0b window: 48h.

R0b 48h vs R0a Recall@50:

- R0a Recall@50: 0.002425879
- R0b 48h Recall@50: 0.003265130
- Absolute gap: 0.000839250
- Multiplier: 1.346x

Later Track B numbers should be reported against R0b 48h, because it is the strongest recency-weighted popularity baseline in the sweep. The 6h/12h/24h rows matching R0a is a consequence of holding out Nov 14 as validation and using only train-split clicks for popularity counts.

## 6. Decisions logged

- D-010 — MIND source is the Recommenders/MIND HuggingFace mirror.
- D-011 — MINDdemo restored as the smoke subset; supersedes D-009.
- D-012 — Rejected mirror: loicmagne/mind_small.
- D-013 — Environment pinned at Week 1 checkpoint.
- D-014 — Keep pandas for Week 1 processing.
- D-015 — Validation split is the last calendar day of MINDsmall_train.
- D-016 — Cold pool is large under split_v1.
- D-017 — R0b headline window is 48 hours under split_v1.

## 7. SURPRISES

- The first HuggingFace mirror tried earlier had valid checksums but was not raw MIND; it was the MTEB reranking derivative. This is the clearest reminder that checksums prove integrity, not provenance.
- The correct Recommenders/MIND archives are much smaller than the loicmagne derivative archives and contain the expected original TSV files.
- `news.tsv` has 8 fields in all four extracted files. The 7-vs-8 documentation inconsistency did not affect implementation.
- The train and dev behavior files each have exactly 50,000 users, but the combined labelled union has 94,057 users. User overlap across train/test is only 5,530 users, so cold-user handling will matter too.
- 42,467 articles in the train ∪ dev news files have no `first_impression_ts`, meaning they appear in `news.tsv` but not in any logged impression candidate list. They stay in the full-corpus candidate universe by D-007, but this makes full-corpus retrieval harder.
- 6h/12h/24h R0b windows collapse to R0a because validation consumes Nov 14 and popularity counts use only the training split. This is correct for leakage control, but it makes 48h the only useful recency window in the W1 sweep.
- The initial R0b implementation was inefficient when recency windows were empty because it repeatedly materialized full fallback rankings. It was fixed before any partial R0b run row was written.
- The baseline did compute nDCG@10 but initially omitted it from `runs.csv`. The uncommitted rows were backfilled from the command outputs, and the code now writes nDCG@10 for future baseline runs.
- Editable package install needed `--no-build-isolation` because build isolation tried to fetch setuptools in the restricted network, despite the venv already having a sufficient setuptools version.
- `02_ENGINEERING.md` still had two stale documentation bugs: the broken `data/` plus `!data/splits/` negation pattern, and a tree entry claiming the decision log lived at `notes/DECISIONS.md`. Both were corrected; the real log is root `04_DECISIONS.md`, and the repo ignores only specific heavy data subdirectories.

## 8. Open questions for Week 2

- W2-T1: Whether `implicit` ALS is strong enough and fast enough on this split, or whether we need the documented TruncatedSVD/hand-rolled fallback.
- W2-T2: Whether MiniLM embedding downloads from HuggingFace work smoothly under approval, and where to cache model files.
- W2-T3: Whether batch size 4096 is comfortable for the frozen-vector two-tower on this laptop once the attention user tower is active.
- W2-T4: Whether R1 beats R3 on warm items. The story is already decided in the spec, but the result changes the README framing.
- Week 4 carryover: cold-start appears viable, but the high cold-click share means the R5/C4 comparison must be especially careful not to credit fine-tuning without isolation.
