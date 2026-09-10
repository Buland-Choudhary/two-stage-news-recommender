"""Append-only run log for experiments."""

from __future__ import annotations

import argparse
import csv
import json
import fcntl
import socket
from datetime import datetime
from pathlib import Path
from typing import Iterable


RUN_COLUMNS = [
    "run_id",
    "timestamp",
    "status",
    "rung",
    "stage",
    "item_tower_mode",
    "batch_size",
    "negatives",
    "logq",
    "embed_dim",
    "seed",
    "split_version",
    "candidate_universe",
    "auc_impression",
    "mrr",
    "ndcg5",
    "ndcg10",
    "recall10",
    "recall50",
    "recall100",
    "ndcg50",
    "coverage",
    "gini",
    "cold_recall50_restricted",
    "cold_recall50_fullcorpus",
    "cold_share_topk",
    "ece",
    "brier",
    "train_minutes",
    "peak_vram_mb",
    "machine",
    "git_commit",
    "config_json",
    "notes",
]


def append_run(path: Path, row: dict[str, object]) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = {column: row.get(column, "") for column in RUN_COLUMNS}
    output['machine'] = output['machine'] or socket.gethostname()
    if not output["timestamp"]:
        output["timestamp"] = datetime.now().isoformat(timespec="seconds")
    config = output["config_json"]
    if not config:
        config = {}
    if isinstance(config, dict):
        config.setdefault('temperature', None)
        output["config_json"] = json.dumps(config, sort_keys=True)

    with path.open("a+", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        if not output["run_id"]:
            output["run_id"] = next_run_id(path, str(output.get("rung") or "RUN"))
        fh.seek(0, 2)
        file_exists = fh.tell() > 0
        writer = csv.DictWriter(fh, fieldnames=RUN_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(output)
        fh.flush()
    return output


def next_run_id(path: Path, rung: str) -> str:
    prefix = rung.lower().replace(" ", "_").replace("/", "_")
    if not path.exists():
        return f"{prefix}_001"
    with path.open("r", newline="", encoding="utf-8") as fh:
        count = sum(1 for _ in csv.DictReader(fh))
    return f"{prefix}_{count + 1:03d}"


def write_dummy_run(path: Path) -> dict[str, object]:
    return append_run(
        path,
        {
            "run_id": "smoke_runlog_001",
            "status": "success",
            "rung": "SMOKE",
            "stage": "runlog",
            "candidate_universe": "full_corpus",
            "machine": "buland-HP-Pavilion-Gaming-Laptop-15-dk1xxx",
            "config_json": {"purpose": "verify append-only run log schema"},
            "notes": "dummy row written before first real experiment",
        },
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage the experiment run log.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    dummy = subparsers.add_parser("dummy")
    dummy.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "dummy":
        row = write_dummy_run(args.runs)
    else:
        raise ValueError(args.command)
    print(json.dumps(row, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
