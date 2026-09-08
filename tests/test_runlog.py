from __future__ import annotations

import csv

from newsrec.runlog import append_run


def test_append_run_preserves_prior_rows(tmp_path) -> None:
    path = tmp_path / "runs.csv"
    append_run(path, {"run_id": "first", "status": "success", "rung": "R0a"})
    append_run(path, {"run_id": "second", "status": "failed", "rung": "R0b"})

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert [row["run_id"] for row in rows] == ["first", "second"]
    assert rows[1]["status"] == "failed"
