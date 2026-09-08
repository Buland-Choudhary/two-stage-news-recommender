"""Build dataset statistics from parsed MIND parquet files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def timestamp_summary(series: pd.Series) -> dict[str, object]:
    min_ts = series.min()
    max_ts = series.max()
    return {
        "min": min_ts.isoformat(sep=" "),
        "max": max_ts.isoformat(sep=" "),
        "span_days": (max_ts - min_ts).total_seconds() / 86400.0,
    }


def numeric_summary(series: pd.Series) -> dict[str, float | int]:
    quantiles = series.quantile([0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0])
    return {
        "min": float(quantiles.loc[0.0]),
        "p25": float(quantiles.loc[0.25]),
        "p50": float(quantiles.loc[0.5]),
        "p75": float(quantiles.loc[0.75]),
        "p90": float(quantiles.loc[0.9]),
        "p95": float(quantiles.loc[0.95]),
        "p99": float(quantiles.loc[0.99]),
        "max": float(quantiles.loc[1.0]),
        "mean": float(series.mean()),
    }


def small_int_histogram(series: pd.Series, max_exact: int = 50) -> dict[str, int]:
    counts = series.value_counts().sort_index()
    hist: dict[str, int] = {}
    overflow = 0
    for value, count in counts.items():
        key = int(value)
        if key <= max_exact:
            hist[str(key)] = int(count)
        else:
            overflow += int(count)
    if overflow:
        hist[f">{max_exact}"] = overflow
    return hist


def load_processed(processed_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    news = pd.read_parquet(processed_dir / "news.parquet")
    impressions = pd.read_parquet(processed_dir / "impressions.parquet")
    history = pd.read_parquet(processed_dir / "history.parquet")
    return news, impressions, history


def build_dataset_stats(processed_dir: Path, out_path: Path | None = None) -> dict[str, object]:
    news, impressions, history = load_processed(processed_dir)

    impression_level = impressions[
        ["impression_id", "raw_impression_id", "user_id", "source_split", "ts", "n_shown"]
    ].drop_duplicates("impression_id")
    history_lengths = (
        history.groupby("impression_id", observed=True).size().rename("history_len").reset_index()
        if not history.empty
        else pd.DataFrame({"impression_id": pd.Series(dtype="int64"), "history_len": pd.Series(dtype="int64")})
    )
    impression_level = impression_level.merge(history_lengths, on="impression_id", how="left")
    impression_level["history_len"] = impression_level["history_len"].fillna(0).astype("int64")

    clicks = int(impressions["label"].sum())
    candidates = int(len(impressions))
    date_range_by_source = {
        split: timestamp_summary(group["ts"])
        for split, group in impression_level.groupby("source_split", observed=True)
    }
    users_by_source = {
        split: int(group["user_id"].nunique())
        for split, group in impression_level.groupby("source_split", observed=True)
    }
    impressions_by_source = {
        split: int(len(group))
        for split, group in impression_level.groupby("source_split", observed=True)
    }
    candidates_by_source = {
        split: int(len(group))
        for split, group in impressions.groupby("source_split", observed=True)
    }
    clicks_by_source = {
        split: int(group["label"].sum())
        for split, group in impressions.groupby("source_split", observed=True)
    }

    article_source_counts = news["source_split"].value_counts().to_dict()
    stats: dict[str, object] = {
        "N_ARTICLES": int(news["news_id"].nunique()),
        "N_USERS": int(impression_level["user_id"].nunique()),
        "N_USERS_BY_SOURCE": users_by_source,
        "N_IMPRESSIONS": int(impression_level["impression_id"].nunique()),
        "N_IMPRESSIONS_BY_SOURCE": impressions_by_source,
        "N_CANDIDATES": candidates,
        "N_CANDIDATES_BY_SOURCE": candidates_by_source,
        "N_CLICKS": clicks,
        "N_CLICKS_BY_SOURCE": clicks_by_source,
        "CTR": clicks / candidates,
        "DATE_RANGE": timestamp_summary(impression_level["ts"]),
        "DATE_RANGE_BY_SOURCE": date_range_by_source,
        "AVG_HISTORY": float(impression_level["history_len"].mean()),
        "HISTORY_LENGTH_DISTRIBUTION": {
            "summary": numeric_summary(impression_level["history_len"]),
            "histogram": small_int_histogram(impression_level["history_len"]),
        },
        "FRACTION_EMPTY_HISTORY": float((impression_level["history_len"] == 0).mean()),
        "FRACTION_VERY_SHORT_HISTORY_LT5": float((impression_level["history_len"] < 5).mean()),
        "CANDIDATES_PER_IMPRESSION_DISTRIBUTION": {
            "summary": numeric_summary(impression_level["n_shown"]),
            "histogram": small_int_histogram(impression_level["n_shown"]),
        },
        "ARTICLE_COUNTS_BY_NEWS_FILE_PRESENCE": {
            "train_only": int(article_source_counts.get("train", 0)),
            "dev_only": int(article_source_counts.get("dev", 0)),
            "both": int(article_source_counts.get("both", 0)),
        },
        "NEWS_WITHOUT_FIRST_IMPRESSION_TS": int(news["first_impression_ts"].isna().sum()),
    }
    manifest_path = processed_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stats["INGEST_MANIFEST"] = manifest

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return stats


def update_cold_pool_stats(processed_dir: Path, split_file: Path, stats_path: Path) -> dict[str, object]:
    news, impressions, _history = load_processed(processed_dir)
    split = pd.read_parquet(split_file)
    impressions = impressions.merge(split, on="impression_id", how="left", validate="many_to_one")
    missing = int(impressions["split"].isna().sum())
    if missing:
        raise ValueError(f"{missing} impression candidate rows are missing split labels")

    train_items = set(impressions.loc[impressions["split"] == "train", "news_id"].astype(str))
    test = impressions.loc[impressions["split"] == "test"].copy()
    test_items = set(test["news_id"].astype(str))
    cold_items = test_items - train_items
    test["is_cold"] = test["news_id"].astype(str).isin(cold_items)

    test_impression_has_cold = test.groupby("impression_id", observed=True)["is_cold"].any()
    test_clicks = test.loc[test["label"] == 1]
    cold_clicks = int(test_clicks["is_cold"].sum())
    n_test_clicks = int(len(test_clicks))

    cold_pool = {
        "n_test_period_articles_absent_from_training_window": int(len(cold_items)),
        "fraction_test_impressions_with_at_least_one_cold_item": float(test_impression_has_cold.mean()),
        "fraction_test_clicks_on_cold_items": cold_clicks / n_test_clicks if n_test_clicks else 0.0,
        "n_test_clicks_on_cold_items": cold_clicks,
        "n_test_clicks": n_test_clicks,
    }

    news = news.copy()
    news["is_cold"] = news["news_id"].astype(str).isin(cold_items)
    news.to_parquet(processed_dir / "news.parquet", index=False)

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    stats["COLD_POOL"] = cold_pool
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")
    return cold_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or update MIND dataset statistics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build")
    build.add_argument("--processed", type=Path, default=Path("data/processed"))
    build.add_argument("--out", type=Path, default=Path("data/stats/dataset_stats.json"))

    cold = subparsers.add_parser("cold-pool")
    cold.add_argument("--processed", type=Path, default=Path("data/processed"))
    cold.add_argument("--split", type=Path, default=Path("data/splits/split_v1.parquet"))
    cold.add_argument("--stats", type=Path, default=Path("data/stats/dataset_stats.json"))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        stats = build_dataset_stats(args.processed, args.out)
    elif args.command == "cold-pool":
        stats = update_cold_pool_stats(args.processed, args.split, args.stats)
    else:
        raise ValueError(args.command)
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
