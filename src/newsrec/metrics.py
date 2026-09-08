"""Metric implementations with Track A and Track B kept separate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class TrackAResult:
    auc: float
    mrr: float
    ndcg5: float
    ndcg10: float
    skipped_impressions: int
    evaluated_impressions: int


@dataclass(frozen=True)
class TrackBResult:
    recall: dict[int, float]
    ndcg: dict[int, float]
    coverage: float
    gini: float
    evaluated_queries: int


class TrackA:
    """Within-impression ranking metrics."""

    REQUIRED_COLUMNS = {"impression_id", "score", "label"}

    @staticmethod
    def evaluate(scores: pd.DataFrame) -> TrackAResult:
        _require_columns(scores, TrackA.REQUIRED_COLUMNS, "Track A")
        if {"query_id", "rank"}.issubset(scores.columns):
            raise ValueError("Track A expects impression candidates, not full-corpus retrieval rows")

        aucs: list[float] = []
        mrrs: list[float] = []
        ndcg5s: list[float] = []
        ndcg10s: list[float] = []
        skipped = 0

        for _impression_id, group in scores.groupby("impression_id", observed=True):
            labels = group["label"].to_numpy(dtype=np.int8)
            if labels.min() == labels.max():
                skipped += 1
                continue
            values = group["score"].to_numpy(dtype=np.float64)
            aucs.append(float(roc_auc_score(labels, values)))
            order = np.argsort(-values)
            ranked_labels = labels[order]
            mrrs.append(_mrr(ranked_labels))
            ndcg5s.append(_ndcg(ranked_labels, k=5))
            ndcg10s.append(_ndcg(ranked_labels, k=10))

        evaluated = len(aucs)
        return TrackAResult(
            auc=float(np.mean(aucs)) if aucs else float("nan"),
            mrr=float(np.mean(mrrs)) if mrrs else float("nan"),
            ndcg5=float(np.mean(ndcg5s)) if ndcg5s else float("nan"),
            ndcg10=float(np.mean(ndcg10s)) if ndcg10s else float("nan"),
            skipped_impressions=skipped,
            evaluated_impressions=evaluated,
        )


class TrackB:
    """Full-corpus retrieval metrics."""

    REC_COLUMNS = {"query_id", "news_id", "rank"}
    TRUTH_COLUMNS = {"query_id", "news_id"}

    @staticmethod
    def evaluate(
        recommendations: pd.DataFrame,
        truth: pd.DataFrame,
        corpus_size: int,
        recall_k: Iterable[int] = (10, 50, 100),
        ndcg_k: Iterable[int] = (10, 50),
    ) -> TrackBResult:
        _require_columns(recommendations, TrackB.REC_COLUMNS, "Track B recommendations")
        _require_columns(truth, TrackB.TRUTH_COLUMNS, "Track B truth")
        if {"impression_id", "score", "label"}.issubset(recommendations.columns):
            raise ValueError("Track B expects full-corpus retrieval rows, not impression-ranking scores")
        if corpus_size <= 0:
            raise ValueError("corpus_size must be positive")

        max_k = max([*recall_k, *ndcg_k])
        recs = recommendations.loc[recommendations["rank"] <= max_k].copy()
        recs["query_id"] = recs["query_id"].astype(str)
        recs["news_id"] = recs["news_id"].astype(str)
        truth = truth.copy()
        truth["query_id"] = truth["query_id"].astype(str)
        truth["news_id"] = truth["news_id"].astype(str)

        truth_by_query = {
            query_id: set(group["news_id"])
            for query_id, group in truth.groupby("query_id", observed=True)
        }
        rec_by_query = {
            query_id: list(group.sort_values("rank")["news_id"])
            for query_id, group in recs.groupby("query_id", observed=True)
        }

        recall_values: dict[int, list[float]] = {int(k): [] for k in recall_k}
        ndcg_values: dict[int, list[float]] = {int(k): [] for k in ndcg_k}
        for query_id, relevant in truth_by_query.items():
            if not relevant:
                continue
            ranked = rec_by_query.get(query_id, [])
            for k in recall_values:
                top_k = ranked[:k]
                recall_values[k].append(len(set(top_k) & relevant) / len(relevant))
            for k in ndcg_values:
                gains = np.array([1.0 if news_id in relevant else 0.0 for news_id in ranked[:k]])
                ndcg_values[k].append(_ndcg_from_gains(gains, min(len(relevant), k)))

        counts = recs["news_id"].value_counts()
        frequency = np.zeros(corpus_size, dtype=np.float64)
        frequency[: len(counts)] = counts.to_numpy(dtype=np.float64)

        return TrackBResult(
            recall={k: float(np.mean(values)) if values else float("nan") for k, values in recall_values.items()},
            ndcg={k: float(np.mean(values)) if values else float("nan") for k, values in ndcg_values.items()},
            coverage=float(recs["news_id"].nunique() / corpus_size),
            gini=float(gini(frequency)),
            evaluated_queries=len(truth_by_query),
        )


def gini(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("gini requires at least one value")
    if np.any(arr < 0):
        raise ValueError("gini values must be non-negative")
    total = arr.sum()
    if total == 0:
        return 0.0
    sorted_arr = np.sort(arr)
    n = arr.size
    cumulative = np.cumsum(sorted_arr)
    return float((n + 1 - 2 * cumulative.sum() / total) / n)


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _mrr(ranked_labels: np.ndarray) -> float:
    positives = np.flatnonzero(ranked_labels > 0)
    return 0.0 if len(positives) == 0 else float(1.0 / (positives[0] + 1))


def _ndcg(ranked_labels: np.ndarray, k: int) -> float:
    gains = ranked_labels[:k].astype(np.float64)
    ideal_positive_count = int(min(ranked_labels.sum(), k))
    return _ndcg_from_gains(gains, ideal_positive_count)


def _ndcg_from_gains(gains: np.ndarray, ideal_positive_count: int) -> float:
    if ideal_positive_count == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * discounts).sum())
    ideal_discounts = 1.0 / np.log2(np.arange(2, ideal_positive_count + 2))
    idcg = float(ideal_discounts.sum())
    return dcg / idcg if idcg else 0.0
