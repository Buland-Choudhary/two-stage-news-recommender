# 03 — RUNBOOK

Ordered tasks with explicit acceptance criteria. Work one at a time. State which task before starting and whether the criteria were met after.

**Task IDs are stable.** Reference them in decision-log entries and commit messages.

**Legend:** 🔴 gate (blocks everything after it) · 🟡 important · ⚪ nice to have

---

## Week 0 — before any of this

### 🔴 W0-T1 — The two-hour resume fixes
**Buland, not the assistant.** Per `01_SPEC.md` §1.

Planet Hunter AI gets a named model and a real metric. JabbyAI gets the classifier accuracy. Cut SnapAuction, TrackIt, Capstone, Employee Self-Service Portal, Automated Testing.

**Done when:** the updated resume is saved. This project finishes in November; your October interviews need the resume fixed now.

---

## Week 1 — access, data, guardrails

The week's goal is **R0a and R0b producing real numbers on a split you trust**. Everything else exists to make those numbers mean something.

---

### 🔴 W1-T1 — Confirm hardware
**Buland runs this.**

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
df -h .
```

**Done when:** GPU, VRAM, driver, free disk recorded in `04_DECISIONS.md`, and `torch.cuda.is_available()` is `True`.

**This number decides §7 of the spec.** At 4GB, Option C is close to mandatory. At 6GB, Option B becomes at least arguable. Don't pick the item-tower mode before this task completes.

---

### 🔴 W1-T2 — Verify MIND access, day one
**Buland runs this.** Download `MINDsmall_train.zip`, `MINDsmall_dev.zip`, and `MINDdemo_train.zip` from Microsoft's public release blob.

**Done when:** all three extract, `behaviors.tsv` and `news.tsv` are present in each, and the first few lines of each parse into the expected number of tab-separated fields.

**If access has changed or links are dead: STOP and escalate.** Six weeks of planning depends on this dataset. Alternatives (Adressa, Globo.com news dataset) exist but change the spec substantially — that's a decision, not a workaround.

**Also confirm:** whether an article body field exists in `news.tsv`. The spec assumes it does not (title + abstract only). If it's there, that's a pleasant surprise worth logging.

---

### 🟡 W1-T3 — Repo skeleton + environment
Layout from `02_ENGINEERING.md` §2, `.gitignore` first, initial commit as Buland Choudhary.

**Done when:** `.venv` activates, torch sees the GPU, faiss imports, `pytest` runs, first commit pushed.

⚠ Budget time for `faiss-cpu` and `implicit`. If either fights the environment for more than an hour, use the fallbacks in `02_ENGINEERING.md` §1 and log the substitution.

---

### 🔴 W1-T4 — Ingest, and record the real numbers
Run the pipeline on **MINDdemo first** (smoke test), then MIND-small.

**Done when** `data/stats/dataset_stats.json` exists and contains:
- `N_ARTICLES`, `N_USERS`, `N_IMPRESSIONS`, `N_CANDIDATES`, `N_CLICKS`, `CTR`
- **`DATE_RANGE` — the actual min/max impression timestamp, and the span in days**
- `AVG_HISTORY`, and the distribution of history lengths
- **Fraction of impressions with empty or very short history** (feeds the §9.1 policy)
- Candidates-per-impression distribution
- Article counts present in train only / dev only / both

**⚠ The date span is the load-bearing number here.** `01_SPEC.md` §5.3 predicts roughly one week of labelled impressions, not six. Confirm or refute it. The answer determines:
- whether the R0b trailing window needs shortening (§6.3)
- whether the temporal-decay experiment is runnable at all (§13.2)
- how large the cold pool can possibly be (§11.4)

Log the finding either way. If the span turns out longer than expected, that's good news that unlocks the original decay experiment.

**Also:** verify the exploded row count is in the expected range. Much larger than a few million means switching to chunked processing or polars before continuing.

---

### 🔴 W1-T5 — Construct and verify the split
Per `01_SPEC.md` §8. This is where leakage enters.

**Done when:**
- `data/splits/split_v1.parquet` and `split_v1_meta.json` exist and are **committed**
- `max(train.ts) <= min(val.ts) <= max(val.ts) <= min(test.ts)` — asserted in code
- **`min(dev.ts) >= max(train.ts)` verified.** If dev overlaps train in time, the default construction leaks — **escalate**, don't patch around it
- Split sizes and user overlap between splits recorded
- `tests/test_leakage.py` and `tests/test_splits.py` pass

**After this task the split is frozen.** Any change makes it `split_v2` and invalidates everything collected before.

---

### 🔴 W1-T6 — Measure the cold pool
**This gates an entire week of work (week 4). Do it now, not then.** Per `01_SPEC.md` §11.4.

**Done when** `dataset_stats.json` records:
- Count of test-period articles absent from the training window
- Fraction of test impressions containing at least one cold item
- Fraction of test-period clicks landing on cold items

**Interpretation:**
- **Healthy pool** (cold items are a meaningful share of test clicks) → R5 proceeds as specced
- **Thin pool** → escalate. Options: define "cold" more loosely (e.g. items with fewer than N training interactions rather than zero), reframe R5's claim around warm-tail items instead of strictly cold ones, or drop R5 and reallocate week 4 to the ranker
- **Near-empty pool** → R5 cannot carry a headline claim. Better to know now.

Report these three numbers in the README regardless — they establish that cold-start matters in practice rather than being a contrived slice.

---

### 🟡 W1-T7 — Leakage assertions as tests
Write `tests/test_leakage.py` fully, per `02_ENGINEERING.md` §6.

**Done when:** all leakage tests pass and are wired into `pytest`.

⚠ Be precise about what you can actually assert. MIND's history field is an ordered list without per-item timestamps, so the assertion is about *construction*, not per-item time comparison. Write the test you can honestly write, and describe it accurately in the README rather than claiming a check you didn't perform.

---

### 🟡 W1-T8 — Run log + metrics module
`runlog.py` and `metrics.py` with Track A and Track B in separate namespaces (`02_ENGINEERING.md` §5). `tests/test_metrics.py` passing on hand-computed toy cases.

**Done when:** a dummy run writes a complete row to `results/runs.csv`, and the toy-case metric tests pass.

**Do this before the first real experiment.** Retrofitting a run log after fifteen runs means reconstructing rows from memory.

---

### 🔴 W1-T9 — R0a and R0b
Naive popularity and recency-weighted popularity, evaluated on Track B.

**Done when:** both have rows in `runs.csv` with Recall@10/50/100, nDCG, coverage, Gini — and the **R0b / R0a gap is reported**.

⚠ **Check the trailing window against the actual span** (W1-T4). If the labelled window is ~5 days, a 48h window covers a large fraction of it and R0a and R0b may nearly coincide. Try 6h / 12h / 24h / 48h and report which is strongest and why. **If R0b barely beats R0a, that itself is a finding** — it means popularity in this window isn't decaying as fast as assumed, and the headline multiplier should be stated against whichever baseline is genuinely strongest.

**This is the week's deliverable**, and R0b is the scale every later Track B number is reported against.

---

## Week 2 — baselines and first tower

### 🟡 W2-T1 — R1: ALS
Matrix factorisation over the user×item click matrix from the training window.

**Done when:** R1 has a `runs.csv` row on the same Track B candidate universe as R0 (`02_ENGINEERING.md` §5.2 — this must be identical across rungs or the comparison is meaningless).

---

### 🟡 W2-T2 — Item embeddings, frozen baseline
Embed all articles with off-the-shelf MiniLM (title + abstract, ~32 tokens). This variant is the input for C3/C4 later.

**Done when:** `data/embeddings/minilm_frozen/` has `emb.npy`, `ids.json`, `meta.json`, and a spot-check confirms semantically similar articles have high cosine similarity.

---

### 🟡 W2-T3 — R3: two-tower with in-batch negatives
Attention user tower, projection item tower over frozen embeddings, sampled-softmax loss. **Decide and log the §7 option now** (Option C recommended, subject to W1-T1's VRAM number).

**Done when:** R3 trains to convergence, has a `runs.csv` row, and `batch_size` is recorded.

**Empty-history policy** (`01_SPEC.md` §9.1) must be implemented and its affected fraction reported.

---

### 🟡 W2-T4 — FAISS index + Track B evaluation wired end to end
`IndexFlatIP` over item embeddings; retrieve top-K; compute Track B metrics.

**Done when:** R3's Track B numbers exist and the **delta over R0b** is computed. Also compute R3's Track A impression AUC — **labelled as an off-objective diagnostic** in both the run log and any reporting (invariant I3).

**If R1 beats R3:** don't panic and don't hide it. `01_SPEC.md` §12.1 has the story. Log it as a finding and continue.

---

## Week 3 — the logQ claim

### 🟡 W3-T1 — R4: logQ correction, 3 seeds
Implement the correction (`02_ENGINEERING.md` §8.1 — pick streaming or precomputed, log which).

**Done when:** R4 has 3 seeded rows, and the R3→R4 comparison reports **Gini, coverage, AND Recall@50** together.

⚠ **If Gini improves but Recall drops, that is the result.** Report both (invariant in spirit — `01_SPEC.md` §9.3). It's a better talking point than a one-sided claim, and the resume bullet has a rewrite prepared (§18).

---

### 🟡 W3-T2 — Track B results table v1
Every rung so far, reported as a multiple of R0b, with R0a shown alongside.

**Done when:** `report.py` generates the table from `runs.csv` and it renders correctly in markdown.

---

## Week 4 — text tower and cold-start

**Gated by W1-T6.** If the cold pool came back thin, revisit scope before starting.

### 🟡 W4-T1 — Phase 1: fine-tune the encoder
Contrastive fine-tuning on co-click pairs, **training-window articles only** (leakage trap 2).

**Done when:** `data/embeddings/minilm_finetuned_phase1/` exists, the leakage test confirms no test-only articles were in the fine-tuning set, and training loss curves are saved.

---

### 🟡 W4-T2 — R5 + cold-start baselines
R5 (retrieval over fine-tuned frozen embeddings, 3 seeds) plus C1 (category popularity), C2 (TF-IDF NN), C3 (frozen, no projection), **C4 (frozen + projection — required)**.

**Done when:** all five have `runs.csv` rows with **both** `cold_recall50_restricted` and `cold_recall50_fullcorpus` and `cold_share_topk` populated.

**The R5 − C4 delta is the fine-tuning claim.** Without C4 the resume bullet is unsupported (invariant I6). If the delta is near zero, that's the honest finding: the projection did the work, not the fine-tuning — and the bullet must be rewritten to credit the projection instead.

---

## Week 5 — ranking, and the completion point

### 🟡 W5-T1 — K1: LightGBM ranker
Features per `01_SPEC.md` §9.4: user×item category crosses, retrieval score, item recency, popularity, history stats.

**Done when:** K1 has a `runs.csv` row with Track A metrics (AUC, MRR, nDCG@5, nDCG@10) — **this is the defensible benchmark comparison against published MIND numbers**, with a note on whether your split construction matches theirs.

---

### 🟡 W5-T2 — Sample-selection-bias measurement
Per `01_SPEC.md` §10 and `02_ENGINEERING.md` §8.2.

**Done when:** the overlap rate between retrieved and logged candidates is reported, along with ranker performance on the intersection. Be explicit about which population each number describes, and about the fact that unshown retrieved candidates have no ground-truth labels.

---

### 🔴 W5-T3 — README v1
Generate all tables via `report.py`. Write System architecture, Dataset (your parsed counts, with the MIND-small vs -large distinction stated), Results, Design decisions, Limitations, Reproducing.

**Done when:** someone who has never seen the project can read it and understand what was measured, how, and what the numbers mean.

**Do this even if weeks 6–7 will happen.** Week 5 is the realistic completion point, and weeks 6–7 are the most likely to be eaten by RA/TA/interview load. An undocumented finished project is worth much less than a documented partial one.

---

## Weeks 6–7 — upside

### ⚪ W6-T1 — K3: calibration
Uncalibrated / Platt / isotonic, fit on the **validation** slice only. ECE, Brier, reliability diagram.

### ⚪ W6-T2 — FastAPI + Docker + latency breakdown
Relative per-stage percentages, with the "laptop, not production" caveat stated.

### ⚪ W7-T1 — Temporal decay
Per `01_SPEC.md` §13.2 — day-by-day within the available span, or the cold-share turnover plot, or dropped with a Limitations note. **Decided by W1-T4's date-span finding**, not improvised here.

### ⚪ W7-T2 — Monitoring design section
Written, not built.

### 🔴 W7-T3 — README final
Update every table. Set the one-line pitch per the claim ladder (`01_SPEC.md` §2.2) according to what actually ran. Update the three resume bullets with real numbers; delete any bullet whose work didn't get done.

---

## Stop conditions

Stop and reassess — don't push through — if:

- **MIND access is broken.** Escalate. Everything depends on it.
- **The dev split overlaps train in time.** The split construction leaks and needs redesign, not a patch.
- **The cold pool is near-empty.** R5's headline claim isn't supportable; reallocate week 4 rather than producing a claim the data can't back.
- **Week 3 ends without R3 working.** Cut R5 and the text tower entirely, go straight to the ranker in week 4, README in week 5. "Two-tower retrieval + LightGBM reranker with logQ correction" is a complete, strong project without the cold-start arm.
- **Week 5 ends without a README.** Stop building and write it. A finished undocumented system converts to nothing in an interview.
- **RA or interview load spikes.** Week 5's end state is a complete project. Ship it and stop.
