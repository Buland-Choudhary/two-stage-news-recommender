"""Popularity baselines for Track B full-corpus retrieval."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from newsrec.metrics import TrackB
from newsrec.runlog import append_run


@dataclass(frozen=True)
class RecencyDiagnostics:
    n_queries: int
    empty_window_queries: int
    empty_window_fraction: float
    history_source: str
    history_splits: tuple[str, ...]


def build_truth(impressions: pd.DataFrame, split_name: str = "test") -> pd.DataFrame:
    truth = impressions.loc[
        (impressions["split"] == split_name) & (impressions["label"] == 1),
        ["impression_id", "news_id"],
    ].copy()
    truth.rename(columns={"impression_id": "query_id"}, inplace=True)
    return truth


def naive_popularity_recommendations(
    impressions: pd.DataFrame,
    corpus_ids: list[str],
    split_name: str = "test",
    top_k: int = 100,
) -> pd.DataFrame:
    train_clicks = impressions.loc[(impressions["split"] == "train") & (impressions["label"] == 1), "news_id"]
    ranked_ids = rank_from_counter(Counter(train_clicks.astype(str)), corpus_ids)[:top_k]
    query_ids = impressions.loc[impressions["split"] == split_name, "impression_id"].drop_duplicates().astype(str)
    return _constant_recommendations(query_ids, ranked_ids)


def recency_popularity_recommendations(
    impressions: pd.DataFrame,
    corpus_ids: list[str],
    window_hours: int,
    history_source: str = "train",
    split_name: str = "test",
    top_k: int = 100,
) -> tuple[pd.DataFrame, RecencyDiagnostics]:
    if history_source == "train":
        history_splits = ("train",)
    elif history_source == "prior":
        history_splits = ("train", "val")
    else:
        raise ValueError(f"unknown recency history_source: {history_source}")

    source_clicks = impressions.loc[
        (impressions["split"].isin(history_splits)) & (impressions["label"] == 1),
        ["ts", "news_id"],
    ].sort_values("ts")
    global_rank = rank_from_counter(Counter(source_clicks["news_id"].astype(str)), corpus_ids)

    test_queries = (
        impressions.loc[impressions["split"] == split_name, ["impression_id", "ts"]]
        .drop_duplicates("impression_id")
        .sort_values("ts")
    )
    window = pd.Timedelta(hours=window_hours)
    active: Counter[str] = Counter()
    active_queue: deque[tuple[pd.Timestamp, str]] = deque()
    click_iter = source_clicks.itertuples(index=False)
    current_click = next(click_iter, None)

    rows: list[tuple[str, str, int]] = []
    empty_window_queries = 0
    for query in test_queries.itertuples(index=False):
        query_ts = query.ts
        while current_click is not None and current_click.ts < query_ts:
            news_id = str(current_click.news_id)
            active[news_id] += 1
            active_queue.append((current_click.ts, news_id))
            current_click = next(click_iter, None)
        lower = query_ts - window
        while active_queue and active_queue[0][0] < lower:
            _old_ts, old_news_id = active_queue.popleft()
            active[old_news_id] -= 1
            if active[old_news_id] <= 0:
                del active[old_news_id]
        if not active:
            empty_window_queries += 1
        ranked = rank_from_counter_head(active, global_rank, top_k)
        if len(ranked) < top_k:
            seen = set(ranked)
            ranked.extend(news_id for news_id in global_rank if news_id not in seen)
        for rank, news_id in enumerate(ranked[:top_k], start=1):
            rows.append((str(query.impression_id), news_id, rank))

    n_queries = int(test_queries["impression_id"].nunique())
    diagnostics = RecencyDiagnostics(
        n_queries=n_queries,
        empty_window_queries=empty_window_queries,
        empty_window_fraction=empty_window_queries / n_queries if n_queries else 0.0,
        history_source=history_source,
        history_splits=history_splits,
    )
    return pd.DataFrame(rows, columns=["query_id", "news_id", "rank"]), diagnostics


def rank_from_counter(counter: Counter[str], corpus_ids: list[str]) -> list[str]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    seen = {news_id for news_id, _count in ranked}
    tail = sorted(news_id for news_id in corpus_ids if news_id not in seen)
    return [news_id for news_id, _count in ranked] + tail


def rank_from_counter_head(counter: Counter[str], fallback_rank: list[str], top_k: int) -> list[str]:
    ranked_active = [news_id for news_id, _count in sorted(counter.items(), key=lambda item: (-item[1], item[0]))]
    if len(ranked_active) >= top_k:
        return ranked_active[:top_k]
    seen = set(ranked_active)
    for news_id in fallback_rank:
        if news_id in seen:
            continue
        ranked_active.append(news_id)
        if len(ranked_active) >= top_k:
            break
    return ranked_active[:top_k]


def evaluate_and_log(
    processed_dir: Path,
    split_file: Path,
    runs_file: Path,
    mode: str,
    window_hours: int | None,
    top_k: int,
    history_source: str = "train",
) -> dict[str, object]:
    news = pd.read_parquet(processed_dir / "news.parquet")
    impressions = pd.read_parquet(processed_dir / "impressions.parquet")
    split = pd.read_parquet(split_file)
    impressions = impressions.merge(split, on="impression_id", how="left", validate="many_to_one")
    missing_split = int(impressions["split"].isna().sum())
    if missing_split:
        raise ValueError(f"{missing_split} rows missing split")

    corpus_ids = sorted(news["news_id"].astype(str).unique())
    if mode == "naive":
        recommendations = naive_popularity_recommendations(impressions, corpus_ids, top_k=top_k)
        rung = "R0a"
        notes = "naive all-time popularity over train split clicks"
        config = {"mode": mode, "top_k": top_k, "candidate_universe": "full_corpus"}
    elif mode == "recency":
        if window_hours is None:
            raise ValueError("window_hours is required for recency mode")
        recommendations, diagnostics = recency_popularity_recommendations(
            impressions,
            corpus_ids,
            window_hours=window_hours,
            history_source=history_source,
            top_k=top_k,
        )
        rung = "R0b-prior" if history_source == "prior" else "R0b"
        source_label = "train+val clicks strictly before each query" if history_source == "prior" else "train split clicks only"
        notes = f"recency-weighted popularity, trailing {window_hours}h, {source_label}"
        config = {
            "mode": mode,
            "window_hours": window_hours,
            "history_source": history_source,
            "top_k": top_k,
            "candidate_universe": "full_corpus",
            **asdict(diagnostics),
        }
        if diagnostics.empty_window_queries:
            notes += (
                f"; {diagnostics.empty_window_queries} empty recency-window queries "
                f"used the {history_source} global popularity fallback"
            )
    else:
        raise ValueError(mode)

    truth = build_truth(impressions)
    result = TrackB.evaluate(
        recommendations,
        truth,
        corpus_size=len(corpus_ids),
        recall_k=(10, 50, 100),
        ndcg_k=(10, 50),
    )
    row = append_run(
        runs_file,
        {
            "status": "success",
            "rung": rung,
            "stage": "retrieval",
            "item_tower_mode": "none",
            "batch_size": "",
            "negatives": "none",
            "logq": False,
            "embed_dim": "",
            "seed": 0,
            "split_version": "split_v1",
            "candidate_universe": "full_corpus",
            "recall10": result.recall[10],
            "recall50": result.recall[50],
            "recall100": result.recall[100],
            "ndcg10": result.ndcg[10],
            "ndcg50": result.ndcg[50],
            "coverage": result.coverage,
            "gini": result.gini,
            "machine": "buland-HP-Pavilion-Gaming-Laptop-15-dk1xxx",
            "git_commit": _git_commit(),
            "config_json": config,
            "notes": notes,
        },
    )
    output = {
        "run": row,
            "metrics": {
            "recall": result.recall,
            "ndcg": result.ndcg,
            "coverage": result.coverage,
            "gini": result.gini,
            "evaluated_queries": result.evaluated_queries,
        },
        "diagnostics": asdict(diagnostics) if mode == "recency" else {},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def _constant_recommendations(query_ids: Iterable[str], ranked_ids: list[str]) -> pd.DataFrame:
    rows: list[tuple[str, str, int]] = []
    for query_id in query_ids:
        for rank, news_id in enumerate(ranked_ids, start=1):
            rows.append((str(query_id), news_id, rank))
    return pd.DataFrame(rows, columns=["query_id", "news_id", "rank"])


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run popularity baselines.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/split_v1.parquet"))
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--mode", choices=["naive", "recency"], required=True)
    parser.add_argument("--window-hours", type=int)
    parser.add_argument("--history-source", choices=["train", "prior"], default="train")
    parser.add_argument("--top-k", type=int, default=100)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate_and_log(
        args.processed,
        args.split,
        args.runs,
        args.mode,
        args.window_hours,
        args.top_k,
        history_source=args.history_source,
    )


if __name__ == "__main__":
    main()
