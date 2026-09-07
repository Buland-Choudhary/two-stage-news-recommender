# 00 — START HERE

**Owner:** Buland Choudhary
**Project:** Two-Stage News Recommender (retrieval → ranking, MIND-small)
**Purpose:** This file explains the document set and contains the exact prompt to paste into ChatGPT to begin.

---

## The document set

Upload all five files. They serve different roles — do not merge them.

| File | Role | Changes during the build? |
|---|---|---|
| `00_START_HERE.md` | Kickoff prompt + how the docs relate | No |
| `01_SPEC.md` | What to build and why. Contains the **invariants** (non-negotiable) and the **flexible zone** (assistant's call). | Rarely — only if an invariant is proven wrong |
| `02_ENGINEERING.md` | Repo layout, data contracts, config schema, CLI surface, metric definitions | Yes — a starting design, expected to evolve |
| `03_RUNBOOK.md` | Ordered tasks with explicit acceptance criteria per week | Yes — tasks get checked off, re-ordered, re-scoped |
| `04_DECISIONS.md` | Append-only log of every deviation + open questions | Yes — written to constantly |

**Reading order:** 01 → 02 → 03. Consult 04 before every session.

---

## The core working principle

This spec was written **before any data was downloaded or any code was run**. Parts of it are guesses. Some will be wrong.

> **Deviate freely where the spec is wrong. Log every deviation in `04_DECISIONS.md`. Never silently drop an invariant.**

The invariants in `01_SPEC.md` §3 are the small set of things that, if broken, make the project's numbers meaningless or dishonest. Everything else is a suggestion.

---

## Two corrections already found — read these first

Both were discovered while preparing this package and are folded into `01_SPEC.md`. They're called out here because they change the shape of the project, and an assistant working from an older draft would build the wrong thing.

**1. MIND-small has no test set, and spans roughly one week of impressions — not six.**
The official MIND documentation states that only training and validation sets are included in MIND-small. The impression logs in the small release come from the fifth week; the first four weeks appear only as *user click history*, not as labelled impressions. Consequences: you must define your own train/val/test from the two files you get, and the "train on early weeks, evaluate week-by-week forward" temporal-decay experiment as originally written is **not runnable**. See `01_SPEC.md` §5.3 and §13.

**2. Article bodies are not in the download.**
Full MSN article bodies are withheld for licensing reasons; a scraping utility is offered but is a dependency and a time sink. Plan on **title + abstract only**. This doesn't hurt — the spec already truncates to ~32 tokens — but the "rich text including body" framing must not appear in the README.

---

## The kickoff prompt

Copy everything between the lines into the first message, with all five files attached.

---

```
You are my engineering partner on a machine learning project running 5–7 weeks
part-time. I am a first-year MS in Machine Learning student building this as my
flagship portfolio project for Summer 2027 internship applications. I work on it
alongside a research assistantship, a course assistantship, coursework, and
interview prep.

I've attached five documents. Read them in this order: 01_SPEC.md (what and why),
02_ENGINEERING.md (how it's structured), 03_RUNBOOK.md (ordered tasks),
04_DECISIONS.md (the running log). 00_START_HERE.md explains how they relate.

Critical context: these documents were written before any data was downloaded or
any code was run. Several assumptions in them — about dataset scale, file
contents, library behaviour, and how long things take — are educated guesses that
may turn out wrong. You have full authority to change anything in
02_ENGINEERING.md and 03_RUNBOOK.md when reality disagrees.

The exception is the INVARIANTS section of 01_SPEC.md. Those are the constraints
that keep the results honest under deadline pressure, which is exactly when
they're most likely to get quietly dropped. If you ever believe an invariant
should be broken, stop and make the case to me explicitly.

How I want you to work with me:

1. Work one RUNBOOK task at a time. Before starting, tell me which one and what
   "done" looks like. After finishing, state whether the acceptance criteria were
   met.

2. When an assumption turns out wrong, say so plainly, propose the fix, and write
   a dated entry for 04_DECISIONS.md that I can paste in.

3. Give me complete, runnable files, not fragments — I'll be copying them into a
   local repo. Python 3.11, PyTorch. Include the imports.

4. When you need information only I can get (my GPU, whether a download
   succeeded, what the actual row counts are), ask for exactly that and wait.
   Don't invent plausible values and continue.

5. This project has a specific failure mode I care about: numbers that look good
   but measure the wrong thing. Retrieval scored on a ranking benchmark. A
   headline multiplier over a strawman baseline. Cold-start evaluated in a way
   that removes the hard part. A cold-start gain credited to fine-tuning when no
   experiment isolates fine-tuning. The spec has guards against each of these.
   Treat them as load-bearing, not as pedantry.

6. Don't fabricate results. Every number in the README must trace to a row in
   results/runs.csv from an actual run. Label estimates as estimates.

7. Prefer boring, readable code. This repo gets read by interviewers.

Start by reading all five documents and giving me:
  (a) a short summary of what you understand the project to be, in your own words
  (b) anything internally inconsistent, underspecified, or likely to break on
      contact with reality
  (c) the first RUNBOOK task and exactly what you need from me to start it

Do not write any code yet.
```

---

## Before week 1 — the two-hour resume fixes

`01_SPEC.md` §1 covers this and it is not optional. **This project will not be on your resume before your October interviews.** Even on schedule it finishes in November, and you cannot put `XX` on a resume.

Spend two hours first on: adding a model and metric to Planet Hunter AI, adding the classifier accuracy to JabbyAI, and cutting SnapAuction / TrackIt / Capstone / Employee Self-Service Portal / Automated Testing. Those pay off this month.

Don't let the ambitious project crowd out the cheap wins.

---

## What Buland does outside the assistant

- **Run `nvidia-smi`** and report GPU + VRAM (W1-T1)
- **Download MIND-small** and confirm the files parse (W1-T2)
- **Report the actual parsed counts** — these go on the resume (W1-T4)
- **Run training jobs** and paste back logs and errors
- **Make the call** on any escalation the assistant raises

The assistant writes code and makes design calls. Ground truth about the machine and the data comes from Buland.

---

## Git

Repo name suggestion: `two-stage-news-recommender`.
Commits and pushes under **Buland Choudhary**.

`.gitignore` before the first commit — `data/`, `*.pth`, `*.faiss`, `*.npy`, `__pycache__/`, `.venv/`. MIND is licensed for non-commercial research use and must never be committed.
