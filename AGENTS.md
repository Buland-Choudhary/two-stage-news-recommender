# Project Instructions

The authoritative project facts live in the five root documents:

- `00_START_HERE.md`: orientation and document relationships.
- `01_SPEC.md`: objectives and immutable invariants.
- `02_ENGINEERING.md`: data contracts and implementation conventions.
- `03_RUNBOOK.md`: tasks and acceptance criteria.
- `04_DECISIONS.md`: dated decisions, corrections, and measured deviations.

Read relevant tasks, decisions, and the latest report before work. Later
corrections take precedence over historical report interpretations. Do not
duplicate changing project facts here. CLAUDE.md is legacy director context;
use the five documents and the active user request for current authority.

## Working Agreement

Buland Choudhary owns this project. Author requested commits in his name.
Follow the role assigned in the active request. Autonomous implementation blocks
authorize direct implementation, experiments, verification, and checkpoints.
Director review requests call for an audit and a reusable implementation prompt.

State task acceptance criteria, measure outcomes, log decisions, and report
surprises. A stop condition blocks only affected tasks and their dependencies.
Complete independent work and identify anything blocked precisely.

Verify actual code and artifacts. Preserve failed and superseded run rows.
Select hyperparameters on validation, never test. Do not weaken invariants or
change frozen splits without Buland's ruling. Limit leakage test names to what
the assertions actually check.

Use the project virtual environment. Verify CUDA access under approved execution
on the laptop; hardware and precision decisions live in the log and config.
Before each commit verify no dataset files, weights, embedding matrices, or
indexes are tracked. Preserve unrelated working-tree files. Generate experiment
tables from results/runs.csv and label validation/test/diagnostic results.
