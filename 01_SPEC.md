# 01 — SPEC: Two-Stage News Recommender

**Owner:** Buland Choudhary
**Version:** v3 (final) · September 7, 2026
**Timeline:** 5–7 weeks part-time. See §2 — this is not a 3-week project.
**Hardware:** HP Pavilion Gaming laptop, NVIDIA GPU — model/VRAM confirmed in W1-T1
**Budget:** $0

---

## 1. Do the cheap resume fixes first

**This project will not be on your resume before your October interviews.** Even at a realistic pace it finishes in November, and you cannot put `XX` on a resume.

Before starting, spend two hours on fixes that pay off immediately:

- **Planet Hunter AI** names no model and no metric. Open the notebook, find what you trained, get the AUC or F1, rewrite the bullets.
- **JabbyAI** — add the classifier's accuracy.
- **Cut** SnapAuction, TrackIt, Capstone, Employee Self-Service Portal, Automated Testing. They bury the ML work.

Those close real gaps this month. This project closes bigger gaps in two months. Sequence accordingly.

---

## 2. What this project is

A **production-shaped multi-stage recommender**: two-tower retrieval over a full item corpus, then a neural ranker over retrieved candidates, with calibrated outputs and a text-based item tower handling cold-start articles.

Three things separate it from the standard portfolio two-tower project:

1. **Both stages actually built.** Most portfolio recsys projects stop at retrieval.
2. **Ranker scores calibrated and evaluated as probabilities**, not just ranked.
3. **Text-based item tower**, making cold-start tractable.

### 2.1 Why this shape

The multi-stage retrieval→ranking pipeline is the canonical answer to "design a feed ranker." Building one means describing your own system under whiteboard pressure instead of reciting a blog post. §14 lists the questions it prepares you for.

### 2.2 Claim ladder — state only what you reached

| Stopped at | Honest pitch |
|---|---|
| **Retrieval (R0–R4)** | "Two-tower retrieval with in-batch negatives and sampling-bias correction." |
| **+ Ranker** | "Two-stage retrieval-and-ranking recommender pipeline." |
| **+ Cold-start (R5)** | "…with a text-based item tower for unseen items." |
| **+ Calibration** | "…and calibrated ranking scores validated by reliability diagrams." |

Decide at the end based on what ran. Do not pre-commit.

### 2.3 Timeline honesty

The first draft claimed 3–4 weeks. That was wrong. Counting honestly — 6 retrieval rungs × 3 seeds, a user-tower ablation, 4 ranking rungs, calibration variants, FastAPI, Docker, ONNX, FAISS index comparison, monitoring, and a drift experiment — is a **2–3 month part-time build**.

Against an RA position, a Course Assistant job, coursework, and interview prep, that pace means hitting week 4 with retrieval half-working and no ranker — **below the interesting claim threshold**, which is the worst place to stop.

**This version is scoped to 5–7 weeks** by cutting roughly 40% of the original. Treat the **ranker (week 5) as the completion target**; calibration and serving are upside.

**What was cut and why:**

| Cut | Reason |
|---|---|
| User-tower ablation (mean-pool/GRU/attention) | Three configs × seeds for a secondary question. Pick attention, justify in prose. |
| R2 (uniform random negatives) | The in-batch-vs-random tradeoff can be discussed without running it. |
| MIND-large | MIND-small answers every question. |
| ONNX / quantization | No latency claim worth making from a laptop. |
| IndexFlat vs IndexIVF comparison | Interesting, not load-bearing. Discuss in README. |
| Built monitoring dashboard | **Write monitoring as a design section instead.** Interviewers ask what you'd monitor far more often than they ask to see a dashboard. |

---

## 3. INVARIANTS — do not break these

If the assistant believes one should be broken, it must stop and make the case explicitly rather than working around it.

**I1. Splits are temporal, never random.**
Random interaction splits let the model see a user's future while predicting their past, and are known to inflate recsys metrics badly. See §8 — MIND-small requires you to *construct* the split, which makes this easier to get wrong, not harder.

**I2. All reported dataset statistics come from parsing your own files.**
Never quote MIND-large's figures (160k articles, 1M users, 15M impressions, six weeks) for MIND-small work. "How many articles were in your corpus?" is an utterly ordinary interview question.

**I3. Track A and Track B are never conflated.**
The ranker on impression AUC is a defensible benchmark comparison. The retrieval model on impression AUC is an **off-objective diagnostic** — labelled as such wherever it appears, never presented as a benchmark result. See §6.

**I4. Every Track B headline is a delta over R0b (recency-weighted popularity).**
R0a (naive all-time popularity) is reported alongside so the gap is visible. Absolute Recall@K on a corpus with no published retrieval baseline is uninterpretable. See §6.3.

**I5. Leakage guards are assertions in code, not intentions in prose.**
History strictly precedes impression timestamp. No test-period article appears in the encoder's fine-tuning set. Popularity features computed on the training window only. Written in week 1, before any model runs.

**I6. Baseline C4 is required before any claim crediting the fine-tuning.**
Your system differs from frozen off-the-shelf embeddings in two ways — fine-tuned encoder *and* learned projection. Without C4 you cannot attribute any of the gap to fine-tuning. See §11.

**I7. Cold-start is reported both restricted and full-corpus.**
Restricted-only evaluation removes the part that makes cold-start hard. See §11.3.

**I8. Every run appends a row to `results/runs.csv`, including failed runs.**
README tables are generated from that file programmatically, never typed by hand.

**I9. Seeds: 3 on R4 and R5. Single-seed everything else, stated explicitly in the results table.**
Do not claim improvements the error bars don't support.

**I10. Calibration is fit on a held-out slice, never on training data.**
And the sample-selection-bias limitation (§10) appears in the README as a stated limitation, not a footnote.

**I11. Every number in the README traces to an actual run.** Estimates labelled in-line as estimates.

**I12. Every deviation from this spec is logged in `04_DECISIONS.md`.**

---

## 4. FLEXIBLE ZONE — assistant's call, log the choice

- Embedding dimension, encoder choice, tokenizer truncation length
- Batch size (within the §7 constraint), optimizer, LR schedule, epochs
- FAISS index type and parameters
- LightGBM hyperparameters and feature engineering specifics
- Whether DCNv2 (K2) happens at all
- Config system, CLI shape, module boundaries
- Parquet vs. feather vs. anything else for intermediates
- Which mitigation for sample-selection bias, and how far to take it
- How far up the rung ladder to go

**Rule of thumb:** if changing it would alter what a number *means*, it's an invariant question — ask. If it only alters what a number *is*, decide and log.

---

## 5. Dataset — MIND-small

### 5.1 Why MIND, not MovieLens

Two of the three differentiators are **fake** on MovieLens:

| Requirement | MovieLens | MIND |
|---|---|---|
| Real negatives (for calibration) | **No** — only ratings. Sampled "negatives" mean calibrated probabilities aren't probabilities of anything real. | **Yes** — impressions record clicked *and* non-clicked items actually shown |
| Rich item text | Thin (title, genres, tags) | **Rich** (title, abstract, category, entities) |
| Genuine cold-start | Simulated | **Intrinsic** — news turns over daily |
| Temporal structure | Timestamps | **Timestamped impression logs** |

### 5.2 Access

MIND-small is distributed as `MINDsmall_train.zip` and `MINDsmall_dev.zip` from Microsoft's public release blob, free for research use under Microsoft Research License Terms. A smaller `MINDdemo` release (5k users, identical format) also exists and is useful as a smoke-test subset — build the whole pipeline on demo before touching small.

**Verify the download works on day one** (W1-T2). If access terms or links have changed, you need to know before planning six weeks around it.

### 5.3 ⚠ What MIND-small actually is — this differs from the paper's headline description

**The widely-quoted figures — ~160k articles, ~1M users, 15M+ impressions, six weeks (Oct 12 – Nov 22, 2019) — describe MIND-large.** MIND-small is a 50,000-user subsample with a substantially smaller article corpus.

Two structural facts about the small release that change the project design:

**(a) There is no test set.** Only training and validation sets are included in MIND-small. You get `train` and `dev`, nothing else. **You must construct train/val/test yourself** — see §8.

**(b) The impression logs span roughly one week, not six.** In the MIND construction, the fifth week's logs form the training set and the last day of that week is the validation set; the first four weeks appear only as *user click history* used for user modelling, not as labelled impressions. The small release inherits this structure.

This matters in three places:

1. **The temporal-decay experiment as originally written is not runnable.** "Train on early weeks, evaluate week-by-week forward" needs six weeks of labelled impressions. See §13.2 for the replacement.
2. **The cold-start pool is smaller than a week-long gap would give.** If train and dev are separated by roughly a day, far fewer articles are genuinely new. **This must be measured before R5 is planned** — see §11.4.
3. **Absolute numbers will differ from published MIND-small results** if those used a different train/test construction than yours. State your construction explicitly.

**Also: article bodies are not in the download.** Full MSN bodies are withheld for licensing reasons; a URL-scraping utility exists but is a dependency and a time sink. Plan on **title + abstract only**. Do not write "including body text" in the README.

### 5.4 Week 1, day 1: record the real numbers

```python
# fill these in from your own files — do not copy from papers
N_ARTICLES     = ...   # unique news IDs in news.tsv (train ∪ dev)
N_USERS        = ...   # unique user IDs in behaviors.tsv
N_IMPRESSIONS  = ...   # rows in behaviors.tsv
N_CANDIDATES   = ...   # exploded (impression × shown item) rows
N_CLICKS       = ...   # positive labels
CTR            = ...   # N_CLICKS / N_CANDIDATES
DATE_RANGE     = ...   # min/max impression timestamp — CHECK THE ACTUAL SPAN
AVG_HISTORY    = ...   # mean history length per impression
```

Use those numbers everywhere — README, resume, interviews.

### 5.5 File formats

`behaviors.tsv`: Impression ID · User ID · Time (`MM/DD/YYYY HH:MM:SS AM/PM`) · History (space-separated news IDs clicked before this impression) · Impressions (`N12345-1` clicked, `N12345-0` not).

`news.tsv`: news ID · category · subcategory · title · abstract · URL · title entities · abstract entities.

**Preserve the within-impression order when exploding** the Impressions field. Position is needed for the position-bias discussion (§10) and is destroyed by a careless explode.

---

## 6. ⚠ Evaluation: two tracks, and only one is comparable

**MIND is published as a *ranking* benchmark, not a retrieval benchmark.** Published results (NAML 66.12 AUC, NRMS 65.63, LSTUR 65.87, UNBERT 67.62 on MIND-small) rank the ~20 articles shown *within an impression*.

### 6.1 Track A — impression ranking

**Only your ranker is fairly comparable to those numbers.**

NAML and NRMS are trained end-to-end to discriminate within impressions. Your two-tower model is trained with a sampled-softmax objective over the whole corpus. Scoring retrieval on impression AUC evaluates it **off-objective** — if it lands at 0.61, that is not "slightly below NRMS," it's a model doing a task it wasn't trained for.

- **Ranker on impression AUC → a defensible benchmark comparison.**
- **Retrieval model on impression AUC → an off-objective diagnostic.** Report it, label it, never present it as a benchmark result.

**One more caveat:** published numbers use the official MIND-small train/dev protocol. If your split construction (§8) differs, say so when comparing. A half-point gap could be protocol, not model.

Metrics: AUC, MRR, nDCG@5, nDCG@10 (the standard MIND four).

### 6.2 Track B — full-corpus retrieval

Retrieve top-K from the entire article corpus. Metrics: Recall@K, nDCG@K, catalogue coverage, Gini.

**No published baseline exists, so absolute numbers are uninterpretable to a reader.** "Recall@50 = 0.11" tells nobody anything.

**Always lead with the delta over the popularity baseline.**

### 6.3 ⚠ R0 must be recency-weighted, and it is the most important number in the project

Because every Track B headline is "X× over R0," **R0 determines whether your headline is honest.** Pick a weak baseline and the multiplier is inflated — and an interviewer who knows news recsys will spot it instantly.

**All-time popularity is a strawman for news.** Article popularity decays over hours. A real system's non-personalised fallback is *"most-clicked articles in the last 24–48 hours,"* which is dramatically stronger.

| Baseline | Definition |
|---|---|
| **R0a — naive popularity** | Most-clicked articles over the entire training window |
| **R0b — recency-weighted popularity** *(the real baseline)* | Most-clicked articles within a trailing 24–48h window relative to each impression timestamp |

**Headline against R0b.** Report R0a alongside — the gap between them is itself an informative result about how fast news popularity decays, and reporting it shows you understood the trap rather than fell into it.

Tune the trailing window (24h vs. 48h) as a small experiment and report which is stronger.

⚠ **Given §5.3, check that the training window is long enough for a trailing 24–48h window to be meaningful.** If the whole labelled span is ~5 days, a 48h window covers a large fraction of it and R0a and R0b may nearly coincide. If so, shorten the window (6h, 12h) and report what you used. This is a W1-T7 finding, not a week-3 surprise.

---

## 7. ⚠ The VRAM conflict — resolve before writing code

**In-batch negatives and a fine-tuned text encoder compete for the same memory.**

In-batch negatives work *because* batch size is large — the benefit is B−1 free negatives per example. Production two-tower systems use batches in the thousands. At B=128 you get a materially weaker training signal than at B=4096.

Put a sentence transformer in the item tower and backprop through it, and on 4–6GB you're capped in the low hundreds. **Your headline technique quietly cripples your other headline technique.**

Three ways out. Pick one deliberately, in week 1.

### Option A — Frozen off-the-shelf embeddings + learned projection
Precompute embeddings for all articles once; train only a projection head.
- Batch size unconstrained (the encoder isn't in the graph). B=8192 is easy.
- **Honest claim: "trained a projection over frozen pretrained embeddings."**
- **Does NOT close the "trained a transformer" gap.** An interviewer asking "walk me through your fine-tuning loop" finds out in thirty seconds.

### Option B — End-to-end fine-tuning
Backprop through the encoder during retrieval training.
- **Closes the transformer gap honestly.**
- Batch capped in the low hundreds → weakened in-batch negatives, and the logQ analysis becomes less meaningful.

### Option C — Two-phase (recommended)
1. **Phase 1:** fine-tune the sentence transformer separately on a cheaper contrastive objective — articles co-clicked by the same user in a session as positives. Small batch is fine; this phase doesn't need in-batch negatives at scale.
2. **Phase 2:** freeze the fine-tuned encoder, precompute all article embeddings, train two-tower retrieval with large batches over those frozen vectors.

- **You genuinely fine-tune a transformer** (Phase 1) **and** keep large-batch in-batch negatives (Phase 2).
- Costs one extra training phase. **Phase 1's evaluation is baseline C4** (§11) — the R5-vs-C4 delta is what shows the fine-tuning did something.
- **Claim: "fine-tuned a sentence transformer on click co-occurrence, then trained two-tower retrieval over the resulting frozen embeddings."**

### 7.1 ⚠ Be precise: these embeddings are not content-only

Phase 1 fine-tunes on click co-occurrence, so the embeddings **carry collaborative signal**, not purely textual signal. Don't call this a "content-based item tower" — someone will ask whether your cold-start advantage comes from text understanding or from co-click structure a brand-new article doesn't have.

**Have the answer ready:** the *supervision* was collaborative, but the learned function maps **text → embedding**. At inference a new article is embedded from title and abstract alone, with no interaction data. So the capability is genuinely content-based; the training signal that taught it was behavioural. Good answer — but only if thought through beforehand.

Describe it as a **"text-encoder item tower trained on collaborative signal,"** not a content-based tower.

### 7.2 Defaults

MiniLM-L6 (22M params, 384-dim), truncate to title + abstract at ~32 tokens, AMP on. **Record batch size in the run log** — it's a real experimental variable here, not incidental config.

---

## 8. ⚠ Splitting — you must construct it, and that's where leakage enters

MIND-large ships train/dev/test. **MIND-small ships only train and dev** (§5.3). You define the third split.

**Default construction (verify the timestamps support it, then log the decision):**

| Split | Source | Purpose |
|---|---|---|
| **train** | `MINDsmall_train` impressions, excluding the final portion by timestamp | Model training |
| **val** | The final ~10–15% of `MINDsmall_train` by timestamp | Model selection, calibration fitting |
| **test** | All of `MINDsmall_dev` | Reported numbers |

**Verify before adopting:** that `min(dev timestamp) >= max(train timestamp)`. If dev overlaps train in time, this construction leaks and needs rethinking — escalate.

**Never re-split randomly.** Invariant I1.

### 8.1 Three leakage traps

1. **The `History` field** must be strictly before the impression timestamp. The dataset is built this way; custom sequence features can break it.
2. **Text encoder fitted on test-period articles = leakage.** Fine-tune (Phase 1) on training-window articles only, then embed test articles with the frozen model.
3. **Popularity features** computed globally instead of on the training window.

**Write these as assertions, not intentions.** Assert history-timestamp ordering for every training example, and that no test-period article ID appears in the encoder's fine-tuning set. Week 1, before any model runs. Invariant I5.

---

## 9. Architecture

```
   OFFLINE:  news.tsv → item tower → embeddings → FAISS index

   ONLINE:   (user_id, history)
                  │
             ┌────▼─────┐   user emb   ┌──────────┐  top-K (K≈200)
             │ User     │─────────────►│  FAISS   │──────────┐
             │ tower    │              │  search  │          │
             └──────────┘              └──────────┘          ▼
                                                      ┌─────────────┐
                                                      │   Ranker    │
                                                      └─────────────┘
                                                             │
                                                             ▼
                                                      ┌─────────────┐
                                                      │ Calibration │
                                                      └─────────────┘
                                                             │
                                                             ▼
                                                   top-N, calibrated p(click)
```

### 9.1 User tower

Input: click history → article embeddings. **Self-attention over the history** (essentially what NRMS does). Mean-pool and GRU alternatives are worth *discussing* in the README as a design tradeoff, not worth three ablation runs.

Output: single user embedding, L2-normalised. **Embedding dim 128.**

⚠ **Cold-start users exist too.** Some impressions will have empty or very short histories. Decide the policy — fall back to R0b, use a learned default embedding, or exclude from Track B evaluation — and apply it consistently. Report what fraction of impressions this affects. It's a legitimate design question interviewers ask.

### 9.2 Item tower

Per §7. Keep the towers **independent — no cross-features.** That independence is exactly what allows item embeddings to be precomputed and ANN-indexed, and it's the central design tradeoff interviewers probe.

**Scoring:** dot product between normalised user and item embeddings.

### 9.3 Negative sampling — the part interviewers probe

**In-batch negatives.** For a batch of B (user, clicked-item) pairs, the other B−1 items serve as negatives for each user. One forward pass yields B positives and B(B−1) negatives. Loss: sampled softmax / InfoNCE.

**Why in-batch rather than uniform random?**
- Near-free — embeddings already computed
- Far more negatives per step
- Negatives drawn from the *interaction* distribution — items users actually engage with, not uniformly random obscure ones

**The bias it creates:** in-batch negatives are sampled proportional to popularity, so popular items get pushed down as negatives disproportionately and are systematically under-scored.

**The correction — logQ:**

```
corrected_logit(u, i) = s(u, i) − log Q(i)
```

`Q(i)` = estimated probability item `i` appears as an in-batch negative, from a streaming frequency counter. Yi et al. (2019), *Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations* — the two-tower paper industry actually implements.

**Measure the effect.** Report recommendation popularity concentration with and without logQ: Gini coefficient over recommended-item frequency, plus catalogue coverage. **Seeded claim 1.**

⚠ **Report Recall alongside Gini.** logQ trades some head accuracy for tail coverage. If Recall@50 drops while Gini improves, that is the honest finding and a better talking point than pretending both moved the right way. Do not report only the flattering metric.

### 9.4 The ranker

**Why a second model?** The towers can't see user×item interactions. Retrieval optimises recall over the full corpus under tight latency; ranking optimises precision over a few hundred candidates with rich cross-features. Different objectives, constraints, models.

**Build one ranker, not two.** Start with **LightGBM** — fast, and gradient boosting is frequently competitive with neural rankers on tabular ranking features. Add **DCNv2** only if time permits. **If LightGBM wins, report that honestly**; the system design story is unaffected.

**Features the ranker has that the towers don't:** explicit user×item crosses (preferred category × item category), the retrieval score as a feature, item recency (enormous for news), popularity, user history statistics.

**Label:** click / no-click from the impression log — real observed negatives. This is the payoff for choosing MIND.

---

## 10. ⚠ Sample-selection bias and calibration are one problem

**The bias:** the ranker trains on impressions — items a *previous production system* chose to show. At serving time it scores items *your retrieval stage* selected. Different distributions.

**The consequence for calibration:** your calibration map is fit on held-out impressions, so your "calibrated probabilities" are calibrated for **the production system's candidate distribution, not yours**. The reliability diagram can look excellent while being measured on the wrong distribution.

**What to do:**

1. **State it plainly in the README as a known limitation**, not a footnote. A clearly articulated limitation is a far better interview talking point than a hidden one.
2. **Mitigate partially:** mix retrieval-generated candidates into ranker training as additional negatives.
3. **Measure the gap:** report ranker performance on logged-impression candidates vs. retrieval-generated candidates separately.

⚠ **Note the asymmetry when you do (3):** retrieval-generated candidates have **no ground-truth labels** except where they coincide with logged impressions. You cannot compute true AUC on candidates that were never shown. What you *can* report is the overlap rate between retrieved and logged candidates, and performance on the intersection. Be precise about which population each number describes — this is exactly the kind of thing that looks like sloppiness if left vague and looks like rigour if stated cleanly.

**Calibration methods to compare:** uncalibrated, Platt scaling, isotonic regression. Fit on a held-out slice, never the training set (invariant I10).

**Metrics:** ECE, Brier score, reliability diagram.

**Also state:** position bias confounds the target — items shown higher get clicked more regardless of relevance. You preserved position when exploding impressions (§5.5), so you can at least quantify it.

---

## 11. ⚠ Cold-start needs a real baseline

An early draft's headline was "ID-embedding tower: Recall ≈ 0 by construction." **That's tautological** — it demonstrates the definition of an ID embedding, not a property of your model. A sharp interviewer will say: *you compared against a method that structurally cannot do the task — what did that measure?*

### 11.1 Compare against baselines that CAN do cold-start but do it worse

| Baseline | What it isolates |
|---|---|
| **C1** Category-level popularity | Can you beat "recommend the popular thing in this category"? |
| **C2** TF-IDF nearest-neighbour over article text | Do learned embeddings beat lexical matching? |
| **C3** Frozen off-the-shelf embeddings, **no learned projection** | Value of everything you added (fine-tuning + projection combined) |
| **C4** Frozen off-the-shelf embeddings **+ learned projection** | ⚠ **Isolates Phase 1 fine-tuning specifically** |

### 11.2 ⚠ C4 is required, not optional

Without C4 your headline claim is unsupported by your own experiments.

Your system differs from C3 in **two** ways: the encoder was fine-tuned (Phase 1) *and* a projection head was learned (Phase 2). The C3 gap measures both together, so **you cannot attribute any of it to fine-tuning.** The resume bullet credits the fine-tuning — C4 earns that credit.

C4 is cheap: same projection head, same training loop, different frozen input vectors.

**C4 also serves as Phase 1's evaluation.** The R5-vs-C4 delta *is* the answer to "did the fine-tuning do anything."

### 11.3 ⚠ Evaluate two ways — restricted is the easy version

Ranking cold articles *among other cold articles* removes exactly the competition that makes cold-start hard. In production, a new article competes against the **entire warm corpus**, where every popularity and engagement signal disadvantages it.

| Evaluation | What it answers |
|---|---|
| **Restricted** — retrieve cold items from the cold-item pool only | Controlled comparison across C1–C4 and R5. Clean attribution. |
| **Full-corpus** — recall of cold items when retrieving over everything, **plus what share of top-K is cold** | ⚠ The production question. What an interviewer means by *"how does a brand-new article get recommended?"* |

Report both. The second pair answers the interview question honestly.

### 11.4 ⚠ Verify the cold pool is big enough before planning R5

Per §5.3, train and test may be separated by only about a day rather than a week. Before building R5, measure:

- **How many test-period articles never appear in the training window?**
- **What fraction of test impressions contain at least one cold item?**
- **What fraction of test-period clicks land on cold items?**

If cold items are a negligible slice, the cold-start experiment can't carry a headline claim. **Report these three numbers regardless** — they establish that cold-start matters in practice rather than being a contrived slice. If they come back small, escalate and reconsider whether R5 earns its week.

**This is a W1 measurement, not a W4 discovery.** It gates a full week of work.

**This is seeded claim 2.**

---

## 12. Experiment ladder

| Rung | Configuration | Question |
|---|---|---|
| **R0a** | Naive all-time popularity | Strawman reference point |
| **R0b** | **Recency-weighted popularity** | **The real baseline** — the interpretable scale for Track B |
| **R1** | Matrix factorisation / ALS | Does deep learning earn its complexity? |
| **R3** | Two-tower, in-batch negatives | Baseline neural retrieval |
| **R4** | R3 + logQ correction | **Seeded claim 1** — popularity bias effect |
| **R5** | Text-encoder item tower (§7 Option C) | **Seeded claim 2** — cold-start vs. C1–C4 |
| **K1** | + LightGBM ranker | Does stage 2 earn its cost? |
| **K2** | + DCNv2 | *Optional.* Does neural ranking beat GBDT? |
| **K3** | + calibration | ECE / Brier / reliability, before vs. after |

*(R2 — uniform random negatives — cut. Discuss the tradeoff in prose. Cold-start baselines C1–C4 are in §11.)*

**R0a, R0b, R1, R3, R4 is a complete honest project.** R5 and the K-rungs are progressive upside.

### 12.1 If R3 loses to R1 — decide the story now

On MIND-small with short user histories, **ALS beating the two-tower is a live possibility.** Don't improvise.

The honest answer: ALS is a warm-item-only method. It cannot embed an article with no interaction history, cannot scale the item tower to a text encoder, and cannot serve cold-start at all. So the comparison is valid **on warm items only**, and the two-tower's value is the capability ALS structurally lacks — which is exactly what R5 demonstrates.

If this happens, report it plainly and make that argument. A candidate who says *"the classical baseline won on warm items, and here's why I built the neural one anyway"* is more credible than one whose deep model conveniently wins everything.

### 12.2 Seed allocation

**3 seeds on R4 and R5 only** — the two rungs where you make an actual claim. **Single-seed everything else and say so explicitly.**

Running 3 seeds on everything is how you spend week 2 re-running experiments instead of building the ranker. Published MIND baselines cluster within ~2 AUC points; a single-seed 0.5-point gap is noise.

---

## 13. Serving and monitoring

### 13.1 Serving — build it

Locally hosted. FastAPI: `POST /recommend`, `GET /health`. Docker, non-root. FAISS `IndexFlatIP` (exact is fine at this corpus size).

**Report latency as a relative per-stage breakdown, not an SLA.** Numbers from a gaming laptop are not production numbers — say so. What matters is knowing *which stage dominates*:

| Stage | Relative cost |
|---|---|
| User tower forward | ~x% |
| FAISS search (top-200) | ~y% |
| Ranker (200 candidates) | ~z% |

That breakdown is the operational awareness that distinguishes an ML engineer from someone who trained a model.

### 13.2 ⚠ The temporal decay experiment — rescoped

The original plan — "MIND spans six weeks, train on early weeks and evaluate week-by-week forward" — **cannot run on MIND-small** (§5.3). The labelled impression span is roughly a week.

Two viable replacements, in order of preference:

1. **Day-by-day decay within the available span.** If the labelled window covers ~5–7 days, train on the first N days and evaluate on each subsequent day. A shorter curve, but real, and the news-turnover effect is fast enough that daily granularity should still show movement. **Check the actual day count in W1-T4 before committing.**
2. **Cold-share decay.** Plot what fraction of each evaluation day's clicked articles were absent from the training window. This directly shows corpus turnover — the underlying mechanism — even if the metric curve is too short to be convincing.

If neither is viable, **drop it and say so in Limitations.** A three-point curve presented as a decay trend is worse than no curve.

### 13.3 Monitoring — write it, don't build it

**A README design section, not instrumentation.** Interviewers ask what you'd monitor far more often than they ask to see a dashboard.

Cover: rolling ECE (calibration drift — the strongest signal here, since news shifts fast), item distribution drift (category mix, recency, cold-item share), coverage/Gini collapse onto popular items, per-stage latency, and failure modes (unknown user, empty history, unseen item, stale index).

---

## 14. Interview preparation — what this lets you answer

**System design**
- Design a news feed / video recommender. *(You built one.)*
- Why multi-stage instead of one model?
- What changes at 100M items instead of this corpus?
- Where does latency go, what would you optimise first?

**Two-tower**
- Why can't the towers interact? What does that independence buy?
- Why in-batch negatives over uniform random?
- What bias does that introduce, how do you correct it? *(logQ — with the Gini plot.)*
- How do you serve it? *(Precompute items, ANN index, online user tower.)*
- How does batch size interact with in-batch negative quality? *(You hit this constraint for real — §7.)*

**Ranking**
- Why a separate ranker instead of retrieving more?
- What features can it use that retrieval can't?
- What's sample selection bias between stages? *(Most candidates can't answer this.)*

**Evaluation**
- Why not a random split?
- Difference between impression AUC and full-corpus Recall@K? *(You measured both and know why they're not comparable.)*
- Is AUC 0.67 good?

**Cold start**
- New article published — how does it get recommended? *(Answer with the full-corpus numbers, not the restricted ones.)*
- Does your cold-start advantage come from text understanding or co-click structure? *(§7.1.)*

**Questions to expect and not be surprised by**
- *"So your item tower is one matrix multiply?"* — Under Option C, yes in Phase 2. The representational work happens in Phase 1; Phase 2 learns the alignment into retrieval space. A legitimate production pattern (precompute-then-project), and it's what makes large-batch in-batch negatives affordable.
- *"Your classical baseline beat your neural model?"* — If R1 beats R3, say so and explain why you built the neural one anyway (§12.1).
- *"How many articles were in your corpus?"* — Know the real MIND-small number.
- *"Did logQ hurt your recall?"* — Know the answer; you measured it (§9.3).

---

## 15. Week-by-week

**Week 1 — access, data, guardrails**
- Environment + **verify MIND download day one**
- Parse both files; record real counts and **the actual date span**
- **Construct and verify the train/val/test split** (§8)
- Write leakage assertions as tests
- **Measure the cold-item pool** (§11.4) — gates week 4
- Set up run log
- **R0a and R0b working → real numbers by end of week**

**Week 2 — baselines and first tower**
- R1 (ALS)
- R3: two-tower with in-batch negatives, attention user tower
- FAISS index, Track A + Track B metrics wired up

**Week 3 — the logQ claim**
- R4 + popularity-bias analysis (Gini, coverage, **and recall**), 3 seeds
- Track B deltas over R0b computed and reported

**Week 4 — text tower**
- §7 Phase 1: fine-tune encoder
- R5: retrieval over frozen fine-tuned embeddings
- Cold-start: C1–C4 (**C4 required**), restricted **and** full-corpus, 3 seeds

**Week 5 — ranking**
- K1: LightGBM ranker
- Sample-selection-bias measurement
- **README v1** — do this even if weeks 6–7 will happen

**This is the realistic completion point.**

**Weeks 6–7 — upside**
- K3: calibration + reliability diagrams
- FastAPI + Docker + latency breakdown
- Temporal decay experiment (§13.2 — whichever version is viable)
- README final, figures, monitoring design section

---

## 16. Run logging

Even after cuts you'll exceed 20 runs. One row per run, from the first experiment:

```
run_id, timestamp, status, rung, stage, item_tower_mode, batch_size, negatives,
logq, embed_dim, seed, split_version,
auc_impression, mrr, ndcg5, ndcg10,
recall50, ndcg50, coverage, gini,
cold_recall50_restricted, cold_recall50_fullcorpus, cold_share_topk,
ece, brier, train_minutes, peak_vram_mb, machine, git_commit, config_json, notes
```

`batch_size` and `item_tower_mode` matter here — they're the §7 tradeoff, not incidental config. The two cold-start columns are separate on purpose (§11.3): restricted and full-corpus answer different questions and must never be conflated. Generate README tables from this file programmatically.

---

## 17. README structure

```
# Two-Stage News Recommender: Retrieval + Ranking with Cold-Start Support

## System architecture      ← diagram first
## Dataset                  ← YOUR parsed counts, and the MIND-small vs -large distinction
## Results
   ### Full-corpus retrieval (lead with delta over R0b)
   ### Impression ranking (ranker vs. published MIND baselines)
   ### Cold-start (vs. C1–C4, restricted AND full-corpus)
   ### Calibration
## Design decisions         ← in-batch negatives, logQ, batch-size/encoder tradeoff,
                              two-stage rationale, split construction
## Serving                  ← relative per-stage latency breakdown
## What I would monitor     ← design section, not a dashboard
## Limitations              ← specific
## Reproducing
## References               ← MIND paper, Yi et al. 2019, DCNv2
```

The **Limitations** section is a strength. Name plainly: rungs not reached, single-seed runs, MIND-small not large, ~one week of labelled impressions, self-constructed test split, position bias unmodelled, calibration fit on the production candidate distribution, offline evaluation only with no online A/B, laptop latency figures, title+abstract only (no body text).

---

## 18. Resume bullets — three, not five

Use your parsed MIND-small counts (§5.4), not MIND-large's.

⚠ **Do not combine the retrieval delta and impression AUC in one sentence.** §6 exists because they're different tasks on different candidate sets; a resume has no room for that caveat, and one sentence containing both reads as a single system achieving both.

- Built a two-stage news recommender over [N] news articles and [M] impression logs — two-tower ANN retrieval (FAISS) feeding a LightGBM reranker — achieving XX× the recency-weighted popularity baseline's Recall@50
- Implemented in-batch negative sampling with logQ sampling-bias correction, reducing recommendation popularity concentration (Gini XX → XX) while improving Recall@50 by XX%
- Fine-tuned a sentence transformer on click co-occurrence for article embedding, improving cold-start Recall@50 by XX% over an identical projection trained on off-the-shelf embeddings

Bullet 3 is attributable: it credits the fine-tuning against **C4**, the baseline that isolates it. Crediting it against TF-IDF or a no-projection baseline would claim more than the experiment shows.

⚠ **If logQ improved Gini but hurt Recall**, rewrite bullet 2 rather than dropping the recall clause and hoping nobody asks. Something like "…reducing popularity concentration (Gini XX → XX) at a XX% Recall@50 cost, quantifying the coverage/accuracy tradeoff."

*(Hold in reserve: the calibration bullet; the ranker-vs-published-MIND-AUC bullet as its own line; the self-hosted FastAPI/latency bullet.)*

---

## 19. Risks

| Risk | Mitigation |
|---|---|
| **MIND-small structure differs from the paper's description** | §5.3. Verify span, splits, and cold pool in week 1. |
| **Cold-item pool too small to support R5's claim** | §11.4. Measure in week 1, before committing week 4. |
| **Weak R0 inflates every headline** | §6.3. R0b is the real baseline; report R0a alongside. |
| **Trailing window too long for the available span** | §6.3. Shorten it; report what you used. |
| **Cold-start gain not attributable to fine-tuning** | §11.2. C4 is required, not optional. |
| **MIND-large numbers on a MIND-small project** | §5.4. Parse actual counts week 1. |
| **Restricted-only cold-start evaluation** | §11.3. Report full-corpus recall and cold share too. |
| **Timeline overrun** | Scope cut ~40%. Week 5 is the completion target. |
| **Batch-size/encoder conflict silently weakens results** | §7. Decide before writing code; log `batch_size`. |
| **Overstating Track A comparability** | §6.1. Ranker = benchmark; retrieval = off-objective diagnostic. |
| **logQ helps Gini but hurts Recall** | §9.3. Report both. Rewrite the bullet honestly. |
| **Calibration measured on wrong distribution** | §10. State as a limitation; measure the overlap. |
| **Temporal decay experiment not runnable** | §13.2. Two fallbacks; drop it if neither works. |
| **Seed budget consumes week 2** | 3 seeds on R4 and R5 only. |
| **ALS beats the two-tower** | §12.1. Story decided in advance. |
| **LightGBM beats DCNv2** | Report it. System design story unaffected. |

---

## 20. Positioning

**Distinctiveness:** two-tower on MovieLens is a tutorial path. This differs by the second stage, the calibration work, the cold-start tower, and a dataset choice that makes all three genuine rather than decorative. **Claim completeness, not novelty.**

**Framing:** lead with "multi-stage recommender system," not "news classifier." The system design story is the asset — the same architecture runs at Meta, Google, TikTok, and Pinterest, and that transferability is the point.
