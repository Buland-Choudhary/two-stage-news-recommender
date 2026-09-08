"""Generate reproducible markdown result tables from the append-only run log."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable


METRIC_COLUMNS = ["recall10", "recall50", "recall100", "ndcg10", "ndcg50", "coverage", "gini"]


def generate_trackb_table(runs_file: Path, stats_file: Path, out_path: Path) -> str:
    rows = load_runs(runs_file)
    stats = json.loads(stats_file.read_text(encoding="utf-8"))
    denominator = strongest_r0b(rows)
    denominator_config = denominator.get("config", {})
    table_rows = select_table_rows(rows, denominator)

    lines = [
        "# Track B Results Table v1",
        "",
        (
            f"Framing note: Track B retrieves from the full {stats['N_ARTICLES']:,}-article corpus. "
            f"{stats['NEWS_WITHOUT_FIRST_IMPRESSION_TS']:,} articles never appear in any labelled impression, "
            "and only 5,369 articles are shown on the test day, so absolute Recall@K is structurally small."
        ),
        "",
        (
            f"Denominator: {denominator['rung']} `{denominator['run_id']}` "
            f"({denominator_config.get('history_source')}, {denominator_config.get('window_hours')}h), "
            f"Recall@50 = {float(denominator['recall50']):.9f}."
        ),
        "",
        "| Rung | Runs | Seeds | Recall@10 | Recall@50 | x Denom | Recall@100 | nDCG@10 | nDCG@50 | Coverage | Gini | Notes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in table_rows:
        lines.append(
            "| {rung} | {runs} | {seeds} | {recall10} | {recall50} | {multiple} | "
            "{recall100} | {ndcg10} | {ndcg50} | {coverage} | {gini} | {notes} |".format(**row)
        )
    text = "\n".join(lines) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    return text


def load_runs(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        try:
            row["config"] = json.loads(row.get("config_json") or "{}")
        except json.JSONDecodeError:
            row["config"] = {}
    return rows


def strongest_r0b(rows: list[dict[str, object]]) -> dict[str, object]:
    candidates = []
    for row in rows:
        if row.get("status") != "success" or row.get("rung") not in {"R0b", "R0b-prior"}:
            continue
        config = row.get("config", {})
        if not isinstance(config, dict):
            continue
        if config.get("history_source") not in {"train", "prior"}:
            continue
        if config.get("window_hours") is None:
            continue
        candidates.append(row)
    if not candidates:
        raise ValueError("no successful corrected R0b rows found")
    return max(candidates, key=lambda row: float(row["recall50"]))


def select_table_rows(rows: list[dict[str, object]], denominator: dict[str, object]) -> list[dict[str, str]]:
    selected: list[dict[str, object]] = []
    selected.extend(best_single(rows, "R0a"))
    selected.extend(best_single(rows, "R0b"))
    selected.append(denominator)
    selected.extend(best_single(rows, "R1"))
    selected.extend(best_single(rows, "R3"))
    selected.extend([row for row in rows if row.get("status") == "success" and row.get("rung") == "R4"])

    grouped: dict[str, list[dict[str, object]]] = {}
    for row in selected:
        grouped.setdefault(str(row["rung"]), []).append(row)

    output: list[dict[str, str]] = []
    for rung in ["R0a", "R0b", "R0b-prior", "R1", "R3", "R4"]:
        group = grouped.get(rung)
        if not group:
            continue
        if rung == "R4":
            output.append(format_aggregate(rung, group, denominator))
        else:
            output.append(format_single(group[0], denominator))
    return output


def best_single(rows: list[dict[str, object]], rung: str) -> list[dict[str, object]]:
    candidates = [
        row
        for row in rows
        if row.get("status") == "success" and row.get("rung") == rung and row.get("stage") == "retrieval"
    ]
    if not candidates:
        return []
    return [max(candidates, key=lambda row: float_or_nan(row.get("recall50")))]


def format_single(row: dict[str, object], denominator: dict[str, object]) -> dict[str, str]:
    config = row.get("config", {})
    notes = str(row.get("notes") or "")
    if row["rung"] == "R0b":
        notes = f"train-only, {config.get('window_hours')}h"
    elif row["rung"] == "R0b-prior":
        notes = f"corrected denominator, prior-window {config.get('window_hours')}h"
    elif row["rung"] == "R1":
        notes = "ALS diagnostic; full-test fallback included"
    elif row["rung"] == "R3":
        notes = "single seed; Track A AUC logged as off-objective diagnostic"
    return {
        "rung": str(row["rung"]),
        "runs": f"`{row['run_id']}`",
        "seeds": "1",
        "recall10": fmt(row.get("recall10")),
        "recall50": fmt(row.get("recall50")),
        "multiple": fmt(float_or_nan(row.get("recall50")) / float(denominator["recall50"])),
        "recall100": fmt(row.get("recall100")),
        "ndcg10": fmt(row.get("ndcg10")),
        "ndcg50": fmt(row.get("ndcg50")),
        "coverage": fmt(row.get("coverage")),
        "gini": fmt(row.get("gini")),
        "notes": notes,
    }


def format_aggregate(rung: str, rows: list[dict[str, object]], denominator: dict[str, object]) -> dict[str, str]:
    return {
        "rung": rung,
        "runs": ", ".join(f"`{row['run_id']}`" for row in rows),
        "seeds": str(len(rows)),
        "recall10": fmt_mean_std(rows, "recall10"),
        "recall50": fmt_mean_std(rows, "recall50"),
        "multiple": fmt_mean_std_values(
            [float_or_nan(row.get("recall50")) / float(denominator["recall50"]) for row in rows]
        ),
        "recall100": fmt_mean_std(rows, "recall100"),
        "ndcg10": fmt_mean_std(rows, "ndcg10"),
        "ndcg50": fmt_mean_std(rows, "ndcg50"),
        "coverage": fmt_mean_std(rows, "coverage"),
        "gini": fmt_mean_std(rows, "gini"),
        "notes": "3 seeds; logQ correction; report Recall@50, coverage, and Gini together",
    }


def fmt_mean_std(rows: list[dict[str, object]], key: str) -> str:
    return fmt_mean_std_values([float_or_nan(row.get(key)) for row in rows])


def fmt_mean_std_values(values: list[float]) -> str:
    clean = [value for value in values if value == value]
    if not clean:
        return "nan"
    if len(clean) == 1:
        return fmt(clean[0])
    return f"{mean(clean):.6f} +/- {pstdev(clean):.6f}"


def fmt(value: object) -> str:
    numeric = float_or_nan(value)
    if numeric != numeric:
        return "nan"
    return f"{numeric:.6f}"


def float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate markdown reports from results/runs.csv.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    trackb = subparsers.add_parser("trackb")
    trackb.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    trackb.add_argument("--stats", type=Path, default=Path("data/stats/dataset_stats.json"))
    trackb.add_argument("--out", type=Path, default=Path("results/trackb_results_v1.md"))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "trackb":
        generate_trackb_table(args.runs, args.stats, args.out)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
