"""ALS diagnostic baseline for the structurally sparse MIND-small split.

This is intentionally a bounded applicability measurement, not a tuned
collaborative-filtering rung. MIND-small train and dev are independent user
samples, so most test users and many clicked test items have no train-window
factor.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import sparse

from newsrec.baselines.popularity import build_truth, recency_popularity_recommendations
from newsrec.metrics import TrackB
from newsrec.runlog import append_run


def run_als_diagnostic(
    processed_dir: Path,
    split_file: Path,
    runs_file: Path,
    factors: int = 64,
    regularization: float = 0.01,
    iterations: int = 10,
    alpha: float = 40.0,
    top_k: int = 100,
    seed: int = 0,
) -> dict[str, object]:
    started = time.perf_counter()
    news, impressions = load_inputs(processed_dir, split_file)
    corpus_ids = sorted(news["news_id"].astype(str).unique())
    fallback = strongest_r0b_config(runs_file)

    train_clicks = impressions.loc[
        (impressions["split"] == "train") & (impressions["label"] == 1),
        ["user_id", "news_id"],
    ].copy()
    train_clicks["user_id"] = train_clicks["user_id"].astype(str)
    train_clicks["news_id"] = train_clicks["news_id"].astype(str)

    factor_model = fit_factor_model(train_clicks, factors, regularization, iterations, alpha, seed)
    fallback_recs, fallback_diag = recency_popularity_recommendations(
        impressions,
        corpus_ids,
        window_hours=int(fallback["window_hours"]),
        history_source=str(fallback["history_source"]),
        top_k=top_k,
    )
    fallback_by_query = {
        query_id: list(group.sort_values("rank")["news_id"].astype(str))
        for query_id, group in fallback_recs.groupby("query_id", observed=True)
    }

    query_level = (
        impressions.loc[impressions["split"] == "test", ["impression_id", "user_id"]]
        .drop_duplicates("impression_id")
        .copy()
    )
    query_level["query_id"] = query_level["impression_id"].astype(str)
    query_level["user_id"] = query_level["user_id"].astype(str)

    full_recs, als_only_recs = build_recommendations(
        query_level,
        factor_model,
        fallback_by_query,
        top_k=top_k,
    )
    truth = build_truth(impressions)
    full_metrics = TrackB.evaluate(
        full_recs,
        truth,
        corpus_size=len(corpus_ids),
        recall_k=(10, 50, 100),
        ndcg_k=(10, 50),
    )

    addressability = measure_addressability(impressions, factor_model)
    truth_for_addressable = truth.copy()
    truth_for_addressable["query_id"] = truth_for_addressable["query_id"].astype(str)
    truth_for_addressable["news_id"] = truth_for_addressable["news_id"].astype(str)
    addressable_truth = truth_for_addressable.loc[
        truth_for_addressable["query_id"].isin(addressability["addressable_query_ids"])
        & truth_for_addressable["news_id"].isin(factor_model.item_id_to_idx)
    ].copy()
    addressable_metrics = TrackB.evaluate(
        als_only_recs.loc[als_only_recs["query_id"].isin(addressability["addressable_query_ids"])],
        addressable_truth,
        corpus_size=len(corpus_ids),
        recall_k=(10, 50, 100),
        ndcg_k=(10, 50),
    )

    train_minutes = (time.perf_counter() - started) / 60.0
    config = {
        "algorithm": factor_model.algorithm,
        "factors": factors,
        "regularization": regularization,
        "iterations": iterations,
        "alpha": alpha,
        "top_k": top_k,
        "seed": seed,
        "candidate_universe": "full_corpus",
        "fallback": {
            **fallback,
            "empty_window_queries": fallback_diag.empty_window_queries,
            "empty_window_fraction": fallback_diag.empty_window_fraction,
        },
        "addressability": public_addressability(addressability),
        "addressable_metrics": trackb_to_dict(addressable_metrics),
    }
    row = append_run(
        runs_file,
        {
            "status": "success",
            "rung": "R1",
            "stage": "retrieval",
            "item_tower_mode": "als",
            "batch_size": "",
            "negatives": "implicit_click_matrix",
            "logq": False,
            "embed_dim": factors,
            "seed": seed,
            "split_version": "split_v1",
            "candidate_universe": "full_corpus",
            "recall10": full_metrics.recall[10],
            "recall50": full_metrics.recall[50],
            "recall100": full_metrics.recall[100],
            "ndcg10": full_metrics.ndcg[10],
            "ndcg50": full_metrics.ndcg[50],
            "coverage": full_metrics.coverage,
            "gini": full_metrics.gini,
            "train_minutes": train_minutes,
            "machine": "buland-HP-Pavilion-Gaming-Laptop-15-dk1xxx",
            "git_commit": git_commit(),
            "config_json": config,
            "notes": (
                "ALS applicability diagnostic; full test uses strongest R0b fallback for "
                "unknown users and items outside the learned train-window factor set"
            ),
        },
    )
    output = {
        "run": row,
        "full_metrics": trackb_to_dict(full_metrics),
        "addressable_metrics": trackb_to_dict(addressable_metrics),
        "addressability": public_addressability(addressability),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return output


def load_inputs(processed_dir: Path, split_file: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    news = pd.read_parquet(processed_dir / "news.parquet")
    impressions = pd.read_parquet(processed_dir / "impressions.parquet")
    split = pd.read_parquet(split_file)
    impressions = impressions.merge(split, on="impression_id", how="left", validate="many_to_one")
    missing = int(impressions["split"].isna().sum())
    if missing:
        raise ValueError(f"{missing} rows missing split")
    return news, impressions


class FactorModel:
    def __init__(
        self,
        algorithm: str,
        user_factors: np.ndarray,
        item_factors: np.ndarray,
        user_id_to_idx: dict[str, int],
        item_id_to_idx: dict[str, int],
        idx_to_item_id: list[str],
    ) -> None:
        self.algorithm = algorithm
        self.user_factors = user_factors.astype(np.float32, copy=False)
        self.item_factors = item_factors.astype(np.float32, copy=False)
        self.user_id_to_idx = user_id_to_idx
        self.item_id_to_idx = item_id_to_idx
        self.idx_to_item_id = idx_to_item_id


def fit_factor_model(
    train_clicks: pd.DataFrame,
    factors: int,
    regularization: float,
    iterations: int,
    alpha: float,
    seed: int,
) -> FactorModel:
    grouped = train_clicks.groupby(["user_id", "news_id"], observed=True).size().reset_index(name="count")
    user_ids = sorted(grouped["user_id"].unique())
    item_ids = sorted(grouped["news_id"].unique())
    user_id_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_id_to_idx = {item_id: idx for idx, item_id in enumerate(item_ids)}

    rows = grouped["news_id"].map(item_id_to_idx).to_numpy(dtype=np.int32)
    cols = grouped["user_id"].map(user_id_to_idx).to_numpy(dtype=np.int32)
    values = grouped["count"].to_numpy(dtype=np.float32) * alpha
    item_users = sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(len(item_ids), len(user_ids)),
        dtype=np.float32,
    ).tocsr()

    try:
        from implicit.als import AlternatingLeastSquares

        model = AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            iterations=iterations,
            random_state=seed,
        )
        model.fit(item_users, show_progress=True)
        # With an item-user matrix, implicit stores row factors in user_factors
        # and column factors in item_factors.
        item_factors = np.asarray(model.user_factors, dtype=np.float32)
        user_factors = np.asarray(model.item_factors, dtype=np.float32)
        algorithm = "implicit_als"
    except Exception as exc:
        from sklearn.decomposition import TruncatedSVD

        user_items = item_users.T.tocsr()
        svd = TruncatedSVD(n_components=factors, random_state=seed)
        user_factors = svd.fit_transform(user_items).astype(np.float32)
        item_factors = svd.components_.T.astype(np.float32)
        algorithm = f"truncated_svd_fallback_after_{type(exc).__name__}"

    return FactorModel(
        algorithm=algorithm,
        user_factors=user_factors,
        item_factors=item_factors,
        user_id_to_idx=user_id_to_idx,
        item_id_to_idx=item_id_to_idx,
        idx_to_item_id=item_ids,
    )


def build_recommendations(
    query_level: pd.DataFrame,
    factor_model: FactorModel,
    fallback_by_query: dict[str, list[str]],
    top_k: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fallback_default = next(iter(fallback_by_query.values()))
    user_groups = query_level.groupby("user_id", observed=True)["query_id"].apply(list)
    item_factors = factor_model.item_factors
    full_rows: list[tuple[str, str, int]] = []
    als_rows: list[tuple[str, str, int]] = []

    for user_id, query_ids in user_groups.items():
        user_idx = factor_model.user_id_to_idx.get(str(user_id))
        als_ids: list[str] = []
        if user_idx is not None:
            scores = item_factors @ factor_model.user_factors[user_idx]
            candidate_count = min(top_k, len(scores))
            head = np.argpartition(-scores, np.arange(candidate_count))[:candidate_count]
            head = head[np.argsort(-scores[head])]
            als_ids = [factor_model.idx_to_item_id[int(idx)] for idx in head]

        for query_id in query_ids:
            query_id = str(query_id)
            fallback = fallback_by_query.get(query_id, fallback_default)
            if user_idx is None:
                ranked = fallback[:top_k]
            else:
                ranked = fill_from_fallback(als_ids, fallback, top_k)
                for rank, news_id in enumerate(als_ids[:top_k], start=1):
                    als_rows.append((query_id, news_id, rank))
            for rank, news_id in enumerate(ranked[:top_k], start=1):
                full_rows.append((query_id, news_id, rank))

    return (
        pd.DataFrame(full_rows, columns=["query_id", "news_id", "rank"]),
        pd.DataFrame(als_rows, columns=["query_id", "news_id", "rank"]),
    )


def fill_from_fallback(primary: list[str], fallback: list[str], top_k: int) -> list[str]:
    ranked = list(primary[:top_k])
    seen = set(ranked)
    for news_id in fallback:
        if news_id in seen:
            continue
        ranked.append(news_id)
        seen.add(news_id)
        if len(ranked) >= top_k:
            break
    return ranked[:top_k]


def measure_addressability(impressions: pd.DataFrame, factor_model: FactorModel) -> dict[str, object]:
    test = impressions.loc[impressions["split"] == "test"].copy()
    test["user_id"] = test["user_id"].astype(str)
    test["news_id"] = test["news_id"].astype(str)

    query_level = test[["impression_id", "user_id"]].drop_duplicates("impression_id").copy()
    query_level["query_id"] = query_level["impression_id"].astype(str)
    has_user_factor = query_level["user_id"].isin(factor_model.user_id_to_idx)
    test_unique_items = set(test["news_id"].unique())
    known_test_items = test_unique_items & set(factor_model.item_id_to_idx)
    test_clicks = test.loc[test["label"] == 1].copy()
    reachable_click_mask = test_clicks["user_id"].isin(factor_model.user_id_to_idx) & test_clicks["news_id"].isin(
        factor_model.item_id_to_idx
    )
    addressable_query_ids = set(
        test_clicks.loc[reachable_click_mask, "impression_id"].astype(str).unique()
    )
    warm_clicked_items = set(test_clicks.loc[test_clicks["news_id"].isin(factor_model.item_id_to_idx), "news_id"])

    raw_by_source = {
        split: set(group["user_id"].astype(str))
        for split, group in impressions[["source_split", "user_id"]].drop_duplicates().groupby(
            "source_split",
            observed=True,
        )
    }
    train_window_users = set(impressions.loc[impressions["split"] == "train", "user_id"].astype(str))
    test_users = set(query_level["user_id"])

    return {
        "n_test_queries": int(len(query_level)),
        "n_test_queries_with_user_factor": int(has_user_factor.sum()),
        "fraction_test_queries_with_user_factor": float(has_user_factor.mean()),
        "n_unique_test_candidate_items": int(len(test_unique_items)),
        "n_unique_test_candidate_items_with_item_factor": int(len(known_test_items)),
        "fraction_unique_test_candidate_items_with_item_factor": len(known_test_items) / len(test_unique_items),
        "n_test_clicks": int(len(test_clicks)),
        "n_test_clicks_reachable_by_als": int(reachable_click_mask.sum()),
        "fraction_test_clicks_reachable_by_als": float(reachable_click_mask.mean()),
        "n_addressable_queries": int(len(addressable_query_ids)),
        "addressable_query_ids": addressable_query_ids,
        "n_test_warm_items_receiving_any_click": int(len(warm_clicked_items)),
        "n_source_train_users": int(len(raw_by_source.get("train", set()))),
        "n_source_dev_users": int(len(raw_by_source.get("dev", set()))),
        "n_source_train_dev_user_overlap": int(
            len(raw_by_source.get("train", set()) & raw_by_source.get("dev", set()))
        ),
        "n_split_train_test_user_overlap": int(len(train_window_users & test_users)),
    }


def public_addressability(addressability: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in addressability.items() if key != "addressable_query_ids"}


def strongest_r0b_config(runs_file: Path) -> dict[str, object]:
    best: dict[str, object] | None = None
    with runs_file.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "success" or row.get("rung") not in {"R0b", "R0b-prior"}:
                continue
            try:
                recall50 = float(row.get("recall50") or "nan")
                config = json.loads(row.get("config_json") or "{}")
            except (TypeError, ValueError, json.JSONDecodeError):
                continue
            if config.get("history_source") not in {"train", "prior"} or config.get("window_hours") is None:
                continue
            if best is None or recall50 > float(best["recall50"]):
                best = {
                    "run_id": row["run_id"],
                    "rung": row["rung"],
                    "recall50": recall50,
                    "history_source": config["history_source"],
                    "window_hours": int(config["window_hours"]),
                }
    if best is None:
        raise ValueError("no successful R0b/R0b-prior run with history_source and window_hours found")
    return best


def trackb_to_dict(result: object) -> dict[str, object]:
    return {
        "recall": result.recall,
        "ndcg": result.ndcg,
        "coverage": result.coverage,
        "gini": result.gini,
        "evaluated_queries": result.evaluated_queries,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run bounded ALS applicability diagnostic.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/split_v1.parquet"))
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_als_diagnostic(
        processed_dir=args.processed,
        split_file=args.split,
        runs_file=args.runs,
        factors=args.factors,
        regularization=args.regularization,
        iterations=args.iterations,
        alpha=args.alpha,
        top_k=args.top_k,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
