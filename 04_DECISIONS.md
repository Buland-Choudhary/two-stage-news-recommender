# 04 — DECISION LOG

**Append-only.** Never edit or delete a past entry — if a decision is reversed, write a new entry that supersedes it. The history of what was tried and rejected is as valuable as the final choice, both for the README's Limitations section and for answering "why did you do it that way?" in an interview.

Keep a copy in the repo at `notes/DECISIONS.md`.

---

## Entry format

```
### D-NNN — <short title>
**Date:** YYYY-MM-DD
**Task:** <Runbook task ID, e.g. W1-T5>
**Type:** discovery | choice | reversal | escalation
**Spec section affected:** <e.g. 01_SPEC §8, 02_ENGINEERING §3.2>

**What we found / decided:**
<one or two sentences>

**Why:**
<the reasoning — this is the part that matters later>

**Alternatives considered:**
<what else was on the table and why it lost>

**Invalidates:**
<any prior runs, numbers, or decisions this makes stale. "Nothing" is valid.>

**Goes in the README?** yes/no — <if yes, which section>
```

---

## Open questions — resolve these, then log the answers

Every item is an assumption that hasn't been checked against reality.

| # | Question | Resolves in | Why it matters |
|---|---|---|---|
| Q1 | GPU model and VRAM? | W1-T1 | **Decides the §7 item-tower option.** 4GB vs 6GB changes what's feasible |
| Q2 | Does the MIND-small download still work? | W1-T2 | Six weeks depend on it |
| Q3 | Is there a body-text field in `news.tsv`? | W1-T2 | Spec assumes title + abstract only |
| Q4 | **What is the actual impression date span?** | W1-T4 | Load-bearing. Decides the R0b window, the decay experiment, and the cold-pool ceiling |
| Q5 | Real article / user / impression / click counts? | W1-T4 | **These go on the resume.** Never quote MIND-large's |
| Q6 | Exploded row count — does it fit comfortably in pandas? | W1-T4 | Determines whether to switch to chunked processing or polars |
| Q7 | What fraction of impressions have empty or very short history? | W1-T4 | Sets the §9.1 empty-history policy |
| Q8 | Does `MINDsmall_dev` start strictly after `MINDsmall_train` ends? | W1-T5 | If not, the default split construction leaks — escalate |
| Q9 | **How big is the cold-item pool?** | W1-T6 | **Gates all of week 4.** Three numbers per §11.4 |
| Q10 | What trailing window makes R0b strongest? | W1-T9 | If R0b ≈ R0a, the recency assumption doesn't hold on this span |
| Q11 | Does ALS (R1) beat the two-tower (R3)? | W2-T4 | Story is pre-decided (§12.1) but the answer changes the README's framing |
| Q12 | Does logQ improve Gini at a Recall cost? | W3-T1 | Determines which version of resume bullet 2 is honest |
| Q13 | Is the R5 − C4 delta meaningful? | W4-T2 | If near zero, the fine-tuning claim must be rewritten to credit the projection |
| Q14 | Streaming or precomputed Q(i) for logQ? | W3-T1 | Both defensible; must be able to explain the choice |
| Q15 | Does `faiss-cpu` / `implicit` install cleanly? | W1-T3 | Fallbacks exist; don't burn a day |
| Q16 | Which temporal-decay variant is viable? | W7-T1, decided by Q4 | Original six-week version is not runnable on MIND-small |

---

## Log

### D-001 — Spec frozen at v3, handed to assistant
**Date:** 2026-09-07
**Task:** —
**Type:** choice
**Spec section affected:** all

**What we decided:**
The spec is final in its *framing, invariants, and evaluation design*. Its *engineering details and task estimates* are explicitly provisional and expected to change during the build.

**Why:**
Written before any data was downloaded or code run. Treating engineering details as fixed forces the build to work around wrong assumptions. Treating invariants as negotiable lets results quietly become dishonest under deadline pressure — and under deadline pressure the first things cut are exactly the methodological controls (temporal splits, the C4 baseline, full-corpus cold-start evaluation, Track A/B separation) that make the project worth doing.

**Alternatives considered:**
A single flat spec — rejected for the reason above.

**Invalidates:** Nothing. Supersedes the v1 and v2 drafts.

**Goes in the README?** No.

---

### D-002 — MIND-small has no test set and spans roughly one week, not six
**Date:** 2026-09-07
**Task:** — (pre-build finding, confirm in W1-T4)
**Type:** discovery
**Spec section affected:** 01_SPEC §5.3, §8, §11.4, §13.2

**What we found:**
The official MIND documentation states that only training and validation sets are included in MIND-small. The impression logs come from the fifth week of the collection period; the first four weeks appear only as user click history, not as labelled impressions.

**Why it matters:**
Three downstream consequences. (1) The train/val/test split must be constructed manually rather than taken from the release, which is where leakage enters. (2) The "train on early weeks, evaluate week-by-week forward" temporal-decay experiment is not runnable as originally written. (3) The cold-item pool is bounded by a much shorter train→test gap than a six-week dataset would give, which is why measuring it moved to week 1.

**Alternatives considered:**
Using MIND-large instead — rejected, it doesn't fit the timeline or the laptop, and MIND-small answers every question the project asks. Ignoring the distinction — rejected, it would produce a false claim about a headline dataset statistic on a resume.

**Invalidates:** The v2 spec's §8 ("MIND ships an official temporal split... first five weeks train/dev, final week test") and §13's temporal decay experiment as written.

**Goes in the README?** Yes — Dataset section and Limitations.

**Still to confirm:** the exact span in days, in W1-T4. This finding is from documentation, not from parsing the files. Verify before quoting.

---

### D-003 — Article bodies are not available
**Date:** 2026-09-07
**Task:** — (confirm in W1-T2)
**Type:** discovery
**Spec section affected:** 01_SPEC §5.1, §5.3

**What we found:**
Full MSN article bodies are withheld from the MIND download for licensing reasons. A utility script for scraping bodies from the article URLs is offered, but that's an external dependency and a time sink.

**Why:**
Plan on title + abstract only. This costs nothing — the spec already truncates to ~32 tokens for the encoder — but the "rich text including body" framing must not appear in the README or in interview answers.

**Alternatives considered:**
Scraping bodies via the provided utility — rejected. 2019 MSN URLs will have substantial link rot, and it buys nothing the 32-token truncation would use.

**Invalidates:** Nothing.

**Goes in the README?** Yes — Dataset section, one line.

---

### D-004 — Framing: multi-stage system, claim completeness not novelty
**Date:** 2026-09-07
**Task:** —
**Type:** choice
**Spec section affected:** 01_SPEC §2, §20

**What we decided:**
The project is presented as a complete production-shaped multi-stage recommender, with the claim ladder in §2.2 determining the final pitch based on what actually ran.

**Why:**
Two-tower-on-MovieLens is a tutorial path and reads as one. What differentiates this is the second stage, the calibration work, the cold-start tower, and a dataset choice that makes all three genuine rather than decorative. None of that is novel research, and claiming novelty invites a comparison the project would lose. Claiming completeness is both true and the thing that actually transfers to a system-design interview.

**Alternatives considered:**
Framing around a single headline metric — rejected; with no published full-corpus retrieval baseline on MIND, an absolute number is uninterpretable, which is why every Track B claim is a delta over R0b.

**Invalidates:** Nothing.

**Goes in the README?** Yes — shapes the title line and the Results ordering.

---

### D-005 — Timeline rescoped from 3–4 weeks to 5–7, with ~40% of scope cut
**Date:** 2026-09-07
**Task:** —
**Type:** reversal
**Spec section affected:** 01_SPEC §2.3

**What we decided:**
The original 3–4 week estimate was wrong. Scope cut per the §2.3 table (user-tower ablation, R2, MIND-large, ONNX, FAISS index comparison, built monitoring dashboard), and week 5 designated the completion target rather than week 7.

**Why:**
The full original scope is a 2–3 month part-time build. Against an RA position, a CA job, coursework, and interview prep, that pace lands week 4 with retrieval half-working and no ranker — below the interesting claim threshold, which is the worst possible stopping point. Cutting scope to guarantee a complete two-stage system beats attempting everything and shipping a fragment.

**Alternatives considered:**
Keeping full scope and accepting a longer timeline — rejected; it pushes completion past the point where the project is useful for the current recruiting cycle, and the marginal items cut are all secondary questions.

**Invalidates:** The v1 timeline.

**Goes in the README?** No — but the cut items belong in Limitations as "not attempted," not silently omitted.

---

<!-- New entries below. Newest at the bottom. -->

### D-006 — Hardware locks item tower to Option C
**Date:** 2026-09-07
**Task:** W1-T1
**Type:** choice
**Spec section affected:** 01_SPEC §7, 02_ENGINEERING §1, 02_ENGINEERING §4

**What we found / decided:**
The machine is an HP Pavilion Gaming laptop with an NVIDIA GeForce GTX 1650 Ti Mobile, 4096 MiB total VRAM, about 3.4 GiB practically usable after desktop processes, driver 580.173.02, driver-reported CUDA capability 13.0, Python 3.12.3, and 254 GiB free disk. The §7 item-tower mode is Option C: fine-tune MiniLM separately, then freeze it and train retrieval over precomputed embeddings.

**Why:**
End-to-end retrieval-loop fine-tuning on about 3.4 GiB usable VRAM would cap batch size in the low hundreds and weaken the in-batch-negative signal that R3/R4 depend on. Option C is comfortable on this hardware rather than a weak compromise: Phase 1 fine-tunes MiniLM-L6 at short sequence length, and Phase 2 trains only the projection head plus attention user tower over frozen 384-dimensional vectors, where batch embedding memory is small. The GTX 1650 Ti is Turing / compute capability 7.5, so AMP must use fp16 rather than bf16.

**Alternatives considered:**
Option A, frozen off-the-shelf embeddings plus projection, was rejected because it does not honestly close the transformer fine-tuning gap. Option B, end-to-end fine-tuning inside the retrieval loop, was rejected because the VRAM cost would gut the large-batch in-batch-negative setup. bf16 AMP was rejected because Turing does not support bfloat16.

**Invalidates:**
Any plan that treats end-to-end retrieval-loop transformer fine-tuning as feasible on this laptop.

**Goes in the README?** yes — Design decisions and Reproducing.

### D-007 — Retrieval candidate universe is the full corpus
**Date:** 2026-09-07
**Task:** W1-T9
**Type:** choice
**Spec section affected:** 01_SPEC §6.2, 01_SPEC §6.3, 01_SPEC §11.3, 02_ENGINEERING §5.2

**What we found / decided:**
The Track B retrieval candidate universe is every distinct `news_id` appearing in `news.tsv` across both MINDsmall train and dev. This universe applies identically to R0a, R0b, R1, R3, R4, and R5, and must be recorded in every `results/runs.csv` row.

**Why:**
Restricting to training-window articles would structurally exclude cold items and break the full-corpus cold-start evaluation required by 01_SPEC §11.3. A query-time live-articles universe would be more realistic, but MIND does not provide publication timestamps, and `first_impression_ts` is only a proxy. Including articles that had not yet appeared makes retrieval harder by enlarging the haystack, but it does not expose labels or interactions.

**Alternatives considered:**
Training-window-only articles were rejected because they define away cold-start. Query-time live articles were rejected because publication time is unavailable and using first impression time as a load-bearing proxy would add an avoidable assumption. Random or sampled candidate universes were rejected because Track B is explicitly full-corpus retrieval.

**Invalidates:**
Nothing.

**Goes in the README?** yes — Design decisions and Results methodology.

### D-008 — Local environment uses Python 3.12
**Date:** 2026-09-07
**Task:** W1-T3
**Type:** discovery
**Spec section affected:** 02_ENGINEERING §1

**What we found / decided:**
The local shell has Python 3.12.3 available and does not have `python3.11` installed. The project venv was created with Python 3.12.3, and PyTorch was installed as the CUDA 12.8 wheel for CPython 3.12.

**Why:**
Using the installed system Python keeps W1-T3 moving and matches the machine facts reported for W1-T1. If Python 3.11 is later required by a dependency, that will need a new environment decision.

**Alternatives considered:**
Installing Python 3.11 before starting was deferred because no current dependency failure requires it, and W1-T3's immediate goal is to verify the environment rather than optimize the interpreter version.

**Invalidates:**
The assumption in 02_ENGINEERING §1 that the initial local environment is Python 3.11.

**Goes in the README?** yes — Reproducing.

### D-009 — SUPERSEDED by D-011: replace MINDdemo with a user-sampled smoke subset
**Date:** 2026-09-07
**Task:** W1-T2, W1-T4
**Type:** choice
**Spec section affected:** 01_SPEC §5.2, 03_RUNBOOK W1-T2, 03_RUNBOOK W1-T4

**What we found / decided:**
Do not depend on the separate MINDdemo download for smoke testing. After downloading `MINDsmall_train`, create `data/raw/DEMO/` by sampling 2,000 users and keeping all impressions for those users.

**Superseded:**
D-011 restores MINDdemo from the verified Recommenders/MIND HuggingFace mirror. The premise of this decision was wrong.

**Why:**
The MINDdemo URL is less certain than the confirmed MINDsmall train/dev URLs. Sampling by user preserves the user/history structure that the pipeline is meant to exercise, while a random row sample would shred that structure and test the wrong thing.

**Alternatives considered:**
Downloading MINDdemo was rejected because it adds a smoke-test dependency on a less certain URL. Sampling random behavior rows was rejected because it breaks the user-level sequence structure. Skipping a smoke subset was rejected because the full MINDsmall path should not be the first place ingest assumptions are tested.

**Invalidates:**
The W1-T2 expectation that `MINDdemo_train.zip` must be downloaded and extracted.

**Goes in the README?** no — this is a development workflow decision.

### D-010 — MIND source is the Recommenders/MIND HuggingFace mirror
**Date:** 2026-09-07
**Task:** W1-T2, W1-T4
**Type:** discovery
**Spec section affected:** 01_SPEC §5.2, 01_SPEC §5.3, 03_RUNBOOK W1-T2

**What we found / decided:**
The official Microsoft release blob is not usable: `mind201910small.blob.core.windows.net` returned `409 PublicAccessNotPermitted`, and `recodatasets.z20.web.core.windows.net` did not resolve in DNS from this machine. The data source for this project is the `Recommenders/MIND` HuggingFace mirror, using `MINDsmall_train.zip`, `MINDsmall_dev.zip`, `MINDdemo_train.zip`, and `MINDdemo_dev.zip`.

**Why:**
The mirror carries the original MIND TSV layout and is the source the Microsoft Recommenders project migrated to after the public blob stopped working. Verification was done on the files themselves, not just on URL names: each archive contained `behaviors.tsv`, `news.tsv`, `entity_embedding.vec`, and `relation_embedding.vec`; all behavior rows parsed as 5 tab-separated fields; all news rows parsed as 8 tab-separated fields; `MINDsmall_train` and `MINDsmall_dev` each had 50,000 unique users; the train impression range was 2019-11-09 00:00:19 to 2019-11-14 23:59:13; the dev impression range was 2019-11-15 00:00:01 to 2019-11-15 23:58:03.

**Alternatives considered:**
The official Microsoft blob was rejected because public access is blocked. The recodatasets storage account was rejected because DNS did not resolve. The loicmagne HuggingFace dataset was rejected in D-012 because it is a reranking derivative rather than raw MIND. Kaggle was not attempted because the verified Recommenders/MIND mirror succeeded and Kaggle would require credentials.

**Invalidates:**
The assumption that the official Microsoft release blob can be used for W1-T2 downloads.

**Goes in the README?** yes — Reproducing.

**Archive SHA-256 sums:**
`MINDsmall_train.zip`: `6ef97a271580b98ccfc4301ada55cc639423cb0576a78b8dcfcf74a4dbcc3194`
`MINDsmall_dev.zip`: `d6ce515dcaa6b6d47ddf0a326eebc8a31b84735ae410285c9882ca2a06eec669`
`MINDdemo_train.zip`: `b1031e2ffcc6d802fc13f60bb959add02ce0d80bb20532e576b7cc8716f43b44`
`MINDdemo_dev.zip`: `4e330564ab66591ed19ab4421de52a594d9140e3a7bf41f37d1c54b55381f898`

### D-011 — MINDdemo restored as the smoke subset
**Date:** 2026-09-07
**Task:** W1-T2, W1-T4
**Type:** reversal
**Spec section affected:** 01_SPEC §5.2, 03_RUNBOOK W1-T4

**What we found / decided:**
Use the real `MINDdemo_train` and `MINDdemo_dev` archives from the verified `Recommenders/MIND` HuggingFace mirror as the smoke-test subset.

**Why:**
D-009 assumed MINDdemo was less certain than the small split URLs. That premise was wrong: MINDdemo is available from the same verified mirror as MINDsmall, while the previously attempted small URLs were either dead or the wrong dataset. A real MINDdemo smoke test is better than a synthetic user sample because it preserves the release format and keeps smoke-test provenance simple.

**Alternatives considered:**
The 2,000-user sampled subset from D-009 was kept as a fallback only if MINDdemo failed. It was not needed.

**Invalidates:**
D-009.

**Goes in the README?** no — this is a development workflow decision.

### D-012 — Rejected mirror: loicmagne/mind_small
**Date:** 2026-09-07
**Task:** W1-T2
**Type:** discovery
**Spec section affected:** 01_SPEC §5.2, 03_RUNBOOK W1-T2

**What we found / decided:**
The `loicmagne/mind_small` HuggingFace files downloaded successfully and matched their published SHA-256 sums, but they were not raw MIND-small. Each archive contained a single JSONL file for the MTEB reranking derivative, with `{id, query, positive, negative}` records over raw article titles only.

**Why:**
Those files have no user IDs, no news IDs, no timestamps, no abstracts, and no categories, so they cannot support temporal splitting, recency-weighted popularity, two-stage recommendation, or cold-start evaluation as specified. Checksums prove file integrity, not dataset provenance; the acceptance test was unsatisfiable, and that was the signal to reject the mirror.

**Alternatives considered:**
Adapting the project to the reranking derivative was rejected because it would be a different project and would break the MIND-specific invariants. Keeping the files under MIND-shaped names was rejected because a later task could accidentally ingest them.

**Invalidates:**
Nothing. The files were deleted before the verified Recommenders/MIND archives were downloaded.

**Goes in the README?** yes — Reproducing or Limitations.

**Rejected archive SHA-256 sums:**
`train.zip`: `5572e63a8ff4079de93e7a00621b3c590a5f07c421892912ba8408a3a40617d5`
`test.zip`: `9329676593513e0824f3378022da664b79997f174263956e7d082f44ad42b868`

### D-013 — Environment pinned at Week 1 checkpoint
**Date:** 2026-09-07
**Task:** W1-T3, W1-T4
**Type:** choice
**Spec section affected:** 02_ENGINEERING §1, 02_ENGINEERING §9

**What we found / decided:**
The environment was pinned with `pip freeze` at the first Week 1 checkpoint rather than waiting until the end of Week 1. `requirements.txt` now records the installed package versions, including `torch==2.11.0+cu128` and `pandas==3.0.5`, with a top comment preserving the CUDA 12.8 torch install command.

**Why:**
Run comparability matters as soon as rows start entering `results/runs.csv`. Pinning before the first real baselines prevents a mid-week environment drift from making R0/R1/R3 comparisons ambiguous.

**Alternatives considered:**
Leaving requirements unpinned until the end of Week 1 was rejected because W1-T9 writes real baseline rows. Downgrading pandas was rejected because pandas 3.0.5 handled the ingest path cleanly; no pandas 3.x behavior workaround was required.

**Invalidates:**
The earlier note in `notes/JOURNAL.md` saying to wait until the end of Week 1 to pin.

**Goes in the README?** yes — Reproducing.

### D-014 — Keep pandas for Week 1 processing
**Date:** 2026-09-07
**Task:** W1-T4
**Type:** choice
**Spec section affected:** 02_ENGINEERING §3.2

**What we found / decided:**
Use pandas for Week 1 ingest and stats. The exploded MIND-small impression table has 8,584,442 rows, below the pre-authorized ~20M-row threshold for switching to polars or chunked processing.

**Why:**
The data fits comfortably on this machine, ingest completed without memory pressure, and keeping pandas avoids adding a second dataframe dependency before the project needs it.

**Alternatives considered:**
Switching to polars or chunked processing was deferred because the measured row count and observed runtime did not justify the added implementation complexity.

**Invalidates:**
Nothing.

**Goes in the README?** no — unless reproducibility notes need to explain local processing choices.

### D-015 — Validation split is the last calendar day of MINDsmall_train
**Date:** 2026-09-07
**Task:** W1-T5
**Type:** choice
**Spec section affected:** 01_SPEC §8, 03_RUNBOOK W1-T5

**What we found / decided:**
Use the last calendar day of `MINDsmall_train` as validation. The cutoff is 2019-11-14 00:00:00, yielding 126,695 train impressions, 30,270 validation impressions, and 73,152 test impressions.

**Why:**
The pre-authorized default was last calendar day unless that produced an implausibly small validation set. It produced 19.3% of MINDsmall_train impressions, comfortably above the 5% threshold, and preserves a simple temporal construction: train on Nov 9-13, validate on Nov 14, test on MINDsmall_dev Nov 15.

**Alternatives considered:**
Using the last 10% by timestamp was rejected because the default last-day validation set was not too small. Random splitting remains rejected by invariant I1.

**Invalidates:**
Nothing. This freezes `split_v1`; any later split change must become `split_v2` and invalidates prior run rows.

**Goes in the README?** yes — Dataset and Reproducing.

### D-016 — Cold pool is large under split_v1
**Date:** 2026-09-07
**Task:** W1-T6
**Type:** discovery
**Spec section affected:** 01_SPEC §11.4

**What we found / decided:**
Under `split_v1`, 3,810 test-period articles are absent from the training window. 99.9877% of test impressions contain at least one cold item, and 86.8265% of test clicks land on cold items.

**Why:**
The cold-start arm is viable on this split because cold items are not a contrived slice; they dominate the test-period clicked items. This does not decide Week 4 scope by itself, but it removes the near-empty-pool concern.

**Alternatives considered:**
No cold definition change was considered in Week 1 because the strict absent-from-training-window pool is large enough to measure.

**Invalidates:**
Nothing.

**Goes in the README?** yes — Dataset and Cold-start results methodology.

### D-017 — R0b headline window is 48 hours under split_v1
**Status:** SUPERSEDED by D-018.
**Date:** 2026-09-07
**Task:** W1-T9
**Type:** choice
**Spec section affected:** 01_SPEC §6.3, 03_RUNBOOK W1-T9

**What we found / decided:**
Use the 48-hour recency-weighted popularity baseline as R0b for Track B headline deltas under `split_v1`. It achieved the strongest Recall@50 in the 6h/12h/24h/48h sweep.

**Why:**
The 6h, 12h, and 24h windows had no train-split clicks available for the Nov 15 test day because Nov 14 is held out as validation, so they fell back to all-time popularity and matched R0a. The 48h window reaches back into Nov 13 training clicks for part of the test day and improved Recall@50 from 0.002425879 to 0.003265130.

**Alternatives considered:**
Using 6h, 12h, or 24h was rejected because those windows were indistinguishable from R0a under the frozen split. Reporting against R0a was rejected by invariant I4.

**Invalidates:**
Nothing.

**Goes in the README?** yes — Results methodology and Design decisions.

### D-018 — R0b has train-only and prior-window variants; headline denominator is strongest
**Date:** 2026-09-07
**Task:** W2-T0
**Type:** correction
**Spec section affected:** 01_SPEC §6.3, 01_SPEC invariant I4, 03_RUNBOOK W2-T0

**What we found / decided:**
Keep the Week 1 train-only R0b variant, but add `R0b-prior`, which counts clicks from `train ∪ val` strictly before each test impression timestamp. The Track B denominator for later multipliers is the strongest R0b variant by Recall@50, which is `R0b-prior` with a 12-hour trailing window: Recall@50 = 0.096437830.

**Why:**
The original 6h, 12h, and 24h train-only rows were empty-window cases because train ends on 2019-11-13 and test starts on 2019-11-15. They silently fell back to the R0a ranking, making them bit-identical to R0a. That fallback is not a valid recency measurement. Using Nov 14 validation clicks for Nov 15 test queries is not leakage because those clicks occur strictly before the query timestamp. The stronger denominator is the conservative choice because a weak denominator would inflate every later Track B multiplier.

**Alternatives considered:**
Using train-only 48h as the headline was rejected because it unnecessarily handicaps the baseline. Deleting the degenerate Week 1 rows was rejected because the run log is append-only; those rows are retained with notes marking the empty-window fallback. Using R0a as the denominator remains rejected by invariant I4.

**Invalidates:**
D-017 as the headline denominator. Existing rows remain valid historical rows, but the three short-window Week 1 R0b rows are marked degenerate.

**Goes in the README?** yes — Results methodology and Reproducing.

**W2-T0 sweep results:**
Train-only R0b Recall@50: 6h = 0.002425879, 12h = 0.002425879, 24h = 0.002425879, 48h = 0.003265130.
Prior-window R0b-prior Recall@50: 6h = 0.051450836, 12h = 0.096437830, 24h = 0.040079731, 48h = 0.012157856.

### D-019 — Cold pool has training and genuine-novelty definitions
**Date:** 2026-09-07
**Task:** W2-T0
**Type:** correction
**Spec section affected:** 01_SPEC §11.4, 03_RUNBOOK W2-T0

**What we found / decided:**
Report two cold-pool definitions. The primary model-training definition remains `unseen_in_train`: 3,810 test-period articles are absent from the train window; 99.9877% of test impressions contain at least one such item; 86.8265% of test clicks land on such items. Add `unseen_in_anything_prior` as the genuine-novelty definition: 2,483 test-period articles are absent from train and validation impressions before the test day; 92.8956% of test impressions contain at least one such item; 24.8952% of test clicks land on such items.

**Why:**
The two questions are different. `unseen_in_train` is the model constraint because R3/R4 train only on the training window. `unseen_in_anything_prior` is the novelty constraint because 1,327 of the train-cold items appeared during validation before the test day. Keeping both prevents Week 4 from overstating novelty while preserving the training-time cold-start measurement.

**Alternatives considered:**
Replacing the Week 1 cold definition outright was rejected because it describes the model training problem correctly. Ignoring prior validation exposure was rejected because it would overstate genuine novelty on Nov 15.

**Invalidates:**
Nothing in `split_v1`. It refines the interpretation of D-016.

**Goes in the README?** yes — Dataset and Cold-start methodology.
