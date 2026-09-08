"""Train and evaluate the frozen-embedding two-tower retrieval model."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from newsrec.baselines.popularity import build_truth, recency_popularity_recommendations
from newsrec.index import build_index, save_index, search_batched
from newsrec.losses import in_batch_contrastive_loss
from newsrec.metrics import TrackA, TrackB
from newsrec.runlog import append_run
from newsrec.towers import TwoTowerRetrievalModel


@dataclass(frozen=True)
class ExampleArrays:
    history_idx: np.ndarray
    target_idx: np.ndarray
    stats: dict[str, object]


@dataclass(frozen=True)
class QueryHistories:
    queries: pd.DataFrame
    history_idx: np.ndarray
    nonempty_query_ids: list[str]
    stats: dict[str, object]


class RetrievalDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: ExampleArrays) -> None:
        self.history_idx = torch.from_numpy(examples.history_idx.astype(np.int64, copy=False))
        self.target_idx = torch.from_numpy(examples.target_idx.astype(np.int64, copy=False))

    def __len__(self) -> int:
        return int(len(self.target_idx))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.history_idx[idx], self.target_idx[idx]


def train_and_evaluate(
    processed_dir: Path,
    split_file: Path,
    embedding_dir: Path,
    runs_file: Path,
    out_dir: Path,
    index_path: Path,
    rung: str = "R3",
    seed: int = 0,
    batch_size: int = 4096,
    epochs: int = 20,
    patience: int = 5,
    lr: float = 1.0e-3,
    weight_decay: float = 1.0e-5,
    embed_dim: int = 128,
    heads: int = 4,
    max_history: int = 50,
    dropout: float = 0.2,
    amp: bool = True,
    logq_correction: bool = False,
    temperature: float = 1.0,
    eval_batch_size: int = 1024,
    top_k: int = 200,
) -> dict[str, object]:
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    news, impressions, history = load_inputs(processed_dir, split_file)
    ids, base_emb = load_embedding_artifact(embedding_dir)
    item_id_to_idx = {news_id: idx for idx, news_id in enumerate(ids)}
    base_tensor = torch.from_numpy(base_emb).to(device)

    train_examples = build_examples(impressions, history, item_id_to_idx, "train", max_history)
    val_queries = build_query_histories(impressions, history, item_id_to_idx, "val", max_history)
    test_queries = build_query_histories(impressions, history, item_id_to_idx, "test", max_history)

    model = TwoTowerRetrievalModel(
        input_dim=base_emb.shape[1],
        embed_dim=embed_dim,
        heads=heads,
        max_history=max_history,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    use_amp = bool(amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    logq = build_logq(train_examples.target_idx, len(ids)).to(device) if logq_correction else None

    loader = DataLoader(
        RetrievalDataset(train_examples),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    best_metric = -1.0
    best_epoch = -1
    bad_epochs = 0
    history_rows: list[dict[str, object]] = []
    train_started = time.perf_counter()
    best_state_path = out_dir / "model.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for hist_idx, target_idx in loader:
            hist_idx = hist_idx.to(device, non_blocking=True)
            target_idx = target_idx.to(device, non_blocking=True)
            hist_vectors, hist_mask = gather_history_vectors(base_tensor, hist_idx)
            target_vectors = base_tensor[target_idx]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=torch.float16):
                user_vec, item_vec = model(hist_vectors, hist_mask, target_vectors)
                loss = in_batch_contrastive_loss(
                    user_vec,
                    item_vec,
                    target_item_idx=target_idx,
                    logq=logq,
                    temperature=temperature,
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))

        val_started = time.perf_counter()
        item_projected = project_all_items(model, base_tensor, batch_size=8192)
        val_recall50 = evaluate_recall_at_k(
            model,
            base_tensor,
            item_projected,
            ids,
            impressions,
            val_queries,
            split_name="val",
            k=50,
            batch_size=eval_batch_size,
        )
        val_seconds = time.perf_counter() - val_started
        train_loss = float(np.mean(losses)) if losses else float("nan")
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_recall50": val_recall50,
            "val_eval_seconds": val_seconds,
        }
        history_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

        if val_recall50 > best_metric:
            best_metric = val_recall50
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "rung": rung,
                        "seed": seed,
                        "embed_dim": embed_dim,
                        "max_history": max_history,
                        "logq_correction": logq_correction,
                    },
                },
                best_state_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    train_minutes = (time.perf_counter() - train_started) / 60.0
    checkpoint = torch.load(best_state_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    item_projected = project_all_items(model, base_tensor, batch_size=8192)
    np.save(out_dir / "item_emb_128.npy", item_projected)
    (out_dir / "ids.json").write_text(json.dumps(ids, indent=2), encoding="utf-8")

    index = build_index(item_projected)
    save_index(index, index_path)

    eval_started = time.perf_counter()
    trackb, retrieval_diag = evaluate_track_b(
        model,
        base_tensor,
        item_projected,
        ids,
        impressions,
        test_queries,
        runs_file,
        top_k=top_k,
        batch_size=eval_batch_size,
    )
    trackb_seconds = time.perf_counter() - eval_started

    tracka_started = time.perf_counter()
    tracka, tracka_diag = evaluate_track_a(
        model,
        base_tensor,
        item_projected,
        impressions,
        test_queries,
        item_id_to_idx,
        batch_size=eval_batch_size,
    )
    tracka_seconds = time.perf_counter() - tracka_started
    peak_vram = peak_vram_mb(device)

    baseline = strongest_r0b_config(runs_file)
    recall50_delta = trackb.recall[50] - float(baseline["recall50"])
    recall50_multiplier = trackb.recall[50] / float(baseline["recall50"]) if float(baseline["recall50"]) else float("nan")
    meta = {
        "variant": rung.lower(),
        "base_embedding_dir": str(embedding_dir),
        "checkpoint": str(best_state_path),
        "item_embedding_path": str(out_dir / "item_emb_128.npy"),
        "index_path": str(index_path),
        "created": datetime.now().isoformat(timespec="seconds"),
        "best_epoch": best_epoch,
        "best_val_recall50": best_metric,
        "train_minutes": train_minutes,
        "peak_vram_mb": peak_vram,
        "train_examples": train_examples.stats,
        "val_queries": val_queries.stats,
        "test_queries": test_queries.stats,
        "training_history": history_rows,
        "retrieval_eval_seconds": trackb_seconds,
        "tracka_eval_seconds": tracka_seconds,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")

    config = {
        "architecture": {
            "item_projection": "384 -> 128 projection head",
            "user_tower": "history-only 1-layer self-attention with attention pooling",
            "user_id_embedding": False,
            "max_history": max_history,
            "heads": heads,
            "dropout": dropout,
        },
        "base_embeddings": str(embedding_dir),
        "batch_size": batch_size,
        "epochs_requested": epochs,
        "epochs_completed": len(history_rows),
        "patience": patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "amp": amp,
        "amp_dtype": "fp16" if use_amp else "disabled",
        "logq_correction": logq_correction,
        "logq_method": "precomputed_train_click_frequency" if logq_correction else None,
        "temperature": temperature,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_recall50": best_metric,
        "train_examples": train_examples.stats,
        "val_queries": val_queries.stats,
        "test_queries": test_queries.stats,
        "retrieval": retrieval_diag,
        "track_a_diagnostic": tracka_diag,
        "baseline_denominator": baseline,
        "recall50_delta_over_baseline": recall50_delta,
        "recall50_multiplier_over_baseline": recall50_multiplier,
    }
    row = append_run(
        runs_file,
        {
            "status": "success",
            "rung": rung,
            "stage": "retrieval",
            "item_tower_mode": "minilm_frozen_projection",
            "batch_size": batch_size,
            "negatives": "in_batch",
            "logq": logq_correction,
            "embed_dim": embed_dim,
            "seed": seed,
            "split_version": "split_v1",
            "candidate_universe": "full_corpus",
            "auc_impression": tracka.auc,
            "mrr": tracka.mrr,
            "ndcg5": tracka.ndcg5,
            "ndcg10": trackb.ndcg[10],
            "recall10": trackb.recall[10],
            "recall50": trackb.recall[50],
            "recall100": trackb.recall[100],
            "ndcg50": trackb.ndcg[50],
            "coverage": trackb.coverage,
            "gini": trackb.gini,
            "train_minutes": train_minutes,
            "peak_vram_mb": peak_vram,
            "machine": "buland-HP-Pavilion-Gaming-Laptop-15-dk1xxx",
            "git_commit": git_commit(),
            "config_json": config,
            "notes": (
                f"{rung} two-tower retrieval; Track A AUC is an off-objective diagnostic; "
                f"Recall@50 delta over {baseline['run_id']} is {recall50_delta:.9f}"
            ),
        },
    )
    output = {
        "run": row,
        "track_b": trackb_to_dict(trackb),
        "track_a_off_objective": tracka_to_dict(tracka),
        "train_history": history_rows,
        "meta": meta,
    }
    print(json.dumps(output, indent=2, sort_keys=True), flush=True)
    return output


def load_inputs(processed_dir: Path, split_file: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    news = pd.read_parquet(processed_dir / "news.parquet")
    impressions = pd.read_parquet(processed_dir / "impressions.parquet")
    history = pd.read_parquet(processed_dir / "history.parquet")
    split = pd.read_parquet(split_file)
    impressions = impressions.merge(split, on="impression_id", how="left", validate="many_to_one")
    missing = int(impressions["split"].isna().sum())
    if missing:
        raise ValueError(f"{missing} rows missing split")
    history = history.merge(split, on="impression_id", how="left", validate="many_to_one")
    missing_history = int(history["split"].isna().sum())
    if missing_history:
        raise ValueError(f"{missing_history} history rows missing split")
    return news, impressions, history


def load_embedding_artifact(embedding_dir: Path) -> tuple[list[str], np.ndarray]:
    ids = json.loads((embedding_dir / "ids.json").read_text(encoding="utf-8"))
    emb = np.load(embedding_dir / "emb.npy").astype(np.float32, copy=False)
    if len(ids) != emb.shape[0]:
        raise ValueError("ids.json length does not match emb.npy rows")
    return [str(news_id) for news_id in ids], emb


def build_examples(
    impressions: pd.DataFrame,
    history: pd.DataFrame,
    item_id_to_idx: dict[str, int],
    split_name: str,
    max_history: int,
) -> ExampleArrays:
    histories, history_stats = build_history_lookup(history, item_id_to_idx, split_name, max_history)
    positives = impressions.loc[
        (impressions["split"] == split_name) & (impressions["label"] == 1),
        ["impression_id", "news_id"],
    ].copy()
    positives["target_idx"] = positives["news_id"].astype(str).map(item_id_to_idx)
    missing_target = int(positives["target_idx"].isna().sum())
    positives = positives.dropna(subset=["target_idx"])

    rows: list[np.ndarray] = []
    targets: list[int] = []
    skipped_empty_history = 0
    for row in positives.itertuples(index=False):
        hist = histories.get(int(row.impression_id))
        if hist is None or len(hist) == 0:
            skipped_empty_history += 1
            continue
        arr = np.full(max_history, -1, dtype=np.int32)
        arr[: len(hist)] = hist
        rows.append(arr)
        targets.append(int(row.target_idx))

    if not rows:
        raise ValueError(f"no {split_name} training examples with usable history")
    stats = {
        "split": split_name,
        "n_positive_rows": int(len(positives) + missing_target),
        "n_missing_target": missing_target,
        "n_examples": int(len(rows)),
        "n_skipped_empty_history": skipped_empty_history,
        "fraction_skipped_empty_history": skipped_empty_history / max(1, len(positives)),
        **history_stats,
    }
    return ExampleArrays(np.stack(rows), np.asarray(targets, dtype=np.int64), stats)


def build_query_histories(
    impressions: pd.DataFrame,
    history: pd.DataFrame,
    item_id_to_idx: dict[str, int],
    split_name: str,
    max_history: int,
) -> QueryHistories:
    histories, history_stats = build_history_lookup(history, item_id_to_idx, split_name, max_history)
    queries = (
        impressions.loc[impressions["split"] == split_name, ["impression_id", "ts", "user_id"]]
        .drop_duplicates("impression_id")
        .sort_values(["ts", "impression_id"])
        .copy()
    )
    queries["query_id"] = queries["impression_id"].astype(str)

    rows: list[np.ndarray] = []
    nonempty_query_ids: list[str] = []
    empty = 0
    for row in queries.itertuples(index=False):
        hist = histories.get(int(row.impression_id))
        if hist is None or len(hist) == 0:
            empty += 1
            continue
        arr = np.full(max_history, -1, dtype=np.int32)
        arr[: len(hist)] = hist
        rows.append(arr)
        nonempty_query_ids.append(str(row.query_id))

    history_idx = np.stack(rows) if rows else np.empty((0, max_history), dtype=np.int32)
    stats = {
        "split": split_name,
        "n_queries": int(len(queries)),
        "n_nonempty_history_queries": int(len(nonempty_query_ids)),
        "n_empty_history_queries": int(empty),
        "fraction_empty_history_queries": empty / max(1, len(queries)),
        **history_stats,
    }
    return QueryHistories(queries=queries, history_idx=history_idx, nonempty_query_ids=nonempty_query_ids, stats=stats)


def build_history_lookup(
    history: pd.DataFrame,
    item_id_to_idx: dict[str, int],
    split_name: str,
    max_history: int,
) -> tuple[dict[int, np.ndarray], dict[str, object]]:
    hist = history.loc[history["split"] == split_name, ["impression_id", "hist_news_id", "hist_position"]].copy()
    original_rows = int(len(hist))
    hist["item_idx"] = hist["hist_news_id"].astype(str).map(item_id_to_idx)
    missing_rows = int(hist["item_idx"].isna().sum())
    hist = hist.dropna(subset=["item_idx"])
    hist["item_idx"] = hist["item_idx"].astype("int32")
    hist = hist.sort_values(["impression_id", "hist_position"])

    lookup: dict[int, np.ndarray] = {}
    truncated = 0
    lengths: list[int] = []
    for impression_id, group in hist.groupby("impression_id", observed=True):
        values = group["item_idx"].to_numpy(dtype=np.int32)
        if len(values) > max_history:
            values = values[-max_history:]
            truncated += 1
        lookup[int(impression_id)] = values
        lengths.append(int(len(values)))
    stats = {
        f"{split_name}_history_rows": original_rows,
        f"{split_name}_history_rows_missing_from_corpus": missing_rows,
        f"{split_name}_histories_with_usable_rows": int(len(lookup)),
        f"{split_name}_histories_truncated_to_max_history": truncated,
        f"{split_name}_mean_usable_history_len": float(np.mean(lengths)) if lengths else 0.0,
    }
    return lookup, stats


def gather_history_vectors(base_tensor: torch.Tensor, hist_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mask = hist_idx >= 0
    safe_idx = hist_idx.clamp_min(0)
    vectors = base_tensor[safe_idx]
    vectors = vectors.masked_fill(~mask[..., None], 0.0)
    return vectors, mask


def project_all_items(model: TwoTowerRetrievalModel, base_tensor: torch.Tensor, batch_size: int) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, base_tensor.shape[0], batch_size):
            projected = model.encode_items(base_tensor[start : start + batch_size])
            rows.append(projected.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(rows)


def encode_query_histories(
    model: TwoTowerRetrievalModel,
    base_tensor: torch.Tensor,
    history_idx: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    if len(history_idx) == 0:
        return np.empty((0, model.user_tower.out_norm.normalized_shape[0]), dtype=np.float32)
    rows: list[np.ndarray] = []
    device = base_tensor.device
    with torch.no_grad():
        for start in range(0, len(history_idx), batch_size):
            hist_idx = torch.from_numpy(history_idx[start : start + batch_size].astype(np.int64, copy=False)).to(device)
            hist_vectors, hist_mask = gather_history_vectors(base_tensor, hist_idx)
            user_vec = model.encode_user(hist_vectors, hist_mask)
            rows.append(user_vec.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(rows)


def evaluate_recall_at_k(
    model: TwoTowerRetrievalModel,
    base_tensor: torch.Tensor,
    item_projected: np.ndarray,
    ids: list[str],
    impressions: pd.DataFrame,
    queries: QueryHistories,
    split_name: str,
    k: int,
    batch_size: int,
) -> float:
    query_vectors = encode_query_histories(model, base_tensor, queries.history_idx, batch_size=batch_size)
    if len(query_vectors) == 0:
        return float("nan")
    index = build_index(item_projected)
    _scores, indices = search_batched(index, query_vectors, top_k=k, batch_size=batch_size)
    recs = recommendations_from_indices(queries.nonempty_query_ids, indices, ids, top_k=k)
    truth = build_truth(impressions, split_name=split_name)
    truth["query_id"] = truth["query_id"].astype(str)
    truth = truth.loc[truth["query_id"].isin(set(queries.nonempty_query_ids))]
    result = TrackB.evaluate(recs, truth, corpus_size=len(ids), recall_k=(k,), ndcg_k=(k,))
    return result.recall[k]


def evaluate_track_b(
    model: TwoTowerRetrievalModel,
    base_tensor: torch.Tensor,
    item_projected: np.ndarray,
    ids: list[str],
    impressions: pd.DataFrame,
    queries: QueryHistories,
    runs_file: Path,
    top_k: int,
    batch_size: int,
) -> tuple[object, dict[str, object]]:
    query_vectors = encode_query_histories(model, base_tensor, queries.history_idx, batch_size=batch_size)
    index = build_index(item_projected)
    _scores, indices = search_batched(index, query_vectors, top_k=top_k, batch_size=batch_size)
    model_recs = recommendations_from_indices(queries.nonempty_query_ids, indices, ids, top_k=top_k)

    fallback = strongest_r0b_config(runs_file)
    fallback_recs, fallback_diag = recency_popularity_recommendations(
        impressions,
        ids,
        window_hours=int(fallback["window_hours"]),
        history_source=str(fallback["history_source"]),
        top_k=top_k,
    )
    full_recs = combine_model_and_fallback(queries.queries, model_recs, fallback_recs, top_k=top_k)
    truth = build_truth(impressions, split_name="test")
    result = TrackB.evaluate(
        full_recs,
        truth,
        corpus_size=len(ids),
        recall_k=(10, 50, 100),
        ndcg_k=(10, 50),
    )
    diag = {
        "top_k": top_k,
        "index_type": "IndexFlatIP",
        "search_batch_size": batch_size,
        "n_model_queries": len(queries.nonempty_query_ids),
        "n_fallback_queries": int(queries.stats["n_empty_history_queries"]),
        "fallback": {
            **fallback,
            "empty_window_queries": fallback_diag.empty_window_queries,
            "empty_window_fraction": fallback_diag.empty_window_fraction,
        },
    }
    return result, diag


def combine_model_and_fallback(
    queries: pd.DataFrame,
    model_recs: pd.DataFrame,
    fallback_recs: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    model_by_query = {
        query_id: list(group.sort_values("rank")["news_id"].astype(str))
        for query_id, group in model_recs.groupby("query_id", observed=True)
    }
    fallback_by_query = {
        query_id: list(group.sort_values("rank")["news_id"].astype(str))
        for query_id, group in fallback_recs.groupby("query_id", observed=True)
    }
    fallback_default = next(iter(fallback_by_query.values()))
    rows: list[tuple[str, str, int]] = []
    for query_id in queries["query_id"].astype(str):
        ranked = model_by_query.get(query_id)
        if ranked is None:
            ranked = fallback_by_query.get(query_id, fallback_default)
        for rank, news_id in enumerate(ranked[:top_k], start=1):
            rows.append((query_id, news_id, rank))
    return pd.DataFrame(rows, columns=["query_id", "news_id", "rank"])


def evaluate_track_a(
    model: TwoTowerRetrievalModel,
    base_tensor: torch.Tensor,
    item_projected: np.ndarray,
    impressions: pd.DataFrame,
    queries: QueryHistories,
    item_id_to_idx: dict[str, int],
    batch_size: int,
) -> tuple[object, dict[str, object]]:
    query_vectors = encode_query_histories(model, base_tensor, queries.history_idx, batch_size=batch_size)
    query_pos = {query_id: idx for idx, query_id in enumerate(queries.nonempty_query_ids)}
    candidates = impressions.loc[
        (impressions["split"] == "test") & (impressions["impression_id"].astype(str).isin(query_pos)),
        ["impression_id", "news_id", "label"],
    ].copy()
    candidates["query_pos"] = candidates["impression_id"].astype(str).map(query_pos)
    candidates["item_idx"] = candidates["news_id"].astype(str).map(item_id_to_idx)
    candidates = candidates.dropna(subset=["query_pos", "item_idx"])
    qpos = candidates["query_pos"].to_numpy(dtype=np.int64)
    ipos = candidates["item_idx"].to_numpy(dtype=np.int64)
    scores = np.sum(query_vectors[qpos] * item_projected[ipos], axis=1)
    score_frame = pd.DataFrame(
        {
            "impression_id": candidates["impression_id"].to_numpy(),
            "score": scores,
            "label": candidates["label"].to_numpy(dtype=np.int8),
        }
    )
    result = TrackA.evaluate(score_frame)
    diag = {
        "label": "OFF-OBJECTIVE DIAGNOSTIC",
        "n_scored_candidate_rows": int(len(score_frame)),
        "n_empty_history_impressions_excluded": int(queries.stats["n_empty_history_queries"]),
        "skipped_degenerate_impressions": result.skipped_impressions,
        "evaluated_impressions": result.evaluated_impressions,
    }
    return result, diag


def recommendations_from_indices(query_ids: list[str], indices: np.ndarray, ids: list[str], top_k: int) -> pd.DataFrame:
    rows: list[tuple[str, str, int]] = []
    for query_id, row in zip(query_ids, indices, strict=True):
        for rank, item_idx in enumerate(row[:top_k], start=1):
            rows.append((str(query_id), ids[int(item_idx)], rank))
    return pd.DataFrame(rows, columns=["query_id", "news_id", "rank"])


def build_logq(target_idx: np.ndarray, n_items: int) -> torch.Tensor:
    counts = np.bincount(target_idx, minlength=n_items).astype(np.float64)
    probs = np.maximum(counts / max(1.0, counts.sum()), 1e-12)
    return torch.from_numpy(np.log(probs).astype(np.float32))


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
        raise ValueError("no successful R0b/R0b-prior denominator found")
    return best


def peak_vram_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))


def trackb_to_dict(result: object) -> dict[str, object]:
    return {
        "recall": result.recall,
        "ndcg": result.ndcg,
        "coverage": result.coverage,
        "gini": result.gini,
        "evaluated_queries": result.evaluated_queries,
    }


def tracka_to_dict(result: object) -> dict[str, object]:
    return {
        "auc": result.auc,
        "mrr": result.mrr,
        "ndcg5": result.ndcg5,
        "ndcg10": result.ndcg10,
        "skipped_impressions": result.skipped_impressions,
        "evaluated_impressions": result.evaluated_impressions,
    }


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return ""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train frozen-embedding two-tower retrieval.")
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/split_v1.parquet"))
    parser.add_argument("--embeddings", type=Path, default=Path("data/embeddings/minilm_frozen"))
    parser.add_argument("--runs", type=Path, default=Path("results/runs.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/embeddings/r3_seed0"))
    parser.add_argument("--index", type=Path, default=Path("data/index/r3_seed0.faiss"))
    parser.add_argument("--rung", default="R3")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-5)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--max-history", type=int, default=50)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--logq", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=200)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    train_and_evaluate(
        processed_dir=args.processed,
        split_file=args.split,
        embedding_dir=args.embeddings,
        runs_file=args.runs,
        out_dir=args.out,
        index_path=args.index,
        rung=args.rung,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        lr=args.lr,
        weight_decay=args.weight_decay,
        embed_dim=args.embed_dim,
        heads=args.heads,
        max_history=args.max_history,
        dropout=args.dropout,
        amp=not args.no_amp,
        logq_correction=args.logq,
        temperature=args.temperature,
        eval_batch_size=args.eval_batch_size,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
