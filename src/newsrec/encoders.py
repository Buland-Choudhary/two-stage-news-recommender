"""Article text embedding utilities.

The Week 2 frozen MiniLM artifact embeds every article from the full corpus
using title + abstract only. The resulting vectors are normalized float32 rows
and are treated as immutable inputs for later projection/tower experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def embed_articles(
    processed_dir: Path,
    out_dir: Path,
    variant: str = "minilm_frozen",
    base_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    max_tokens: int = 32,
    batch_size: int = 128,
    cache_dir: Path = Path("data/embeddings/model_cache"),
    seed: int = 0,
) -> dict[str, object]:
    from sentence_transformers import SentenceTransformer
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    news = pd.read_parquet(processed_dir / "news.parquet").copy()
    news["news_id"] = news["news_id"].astype(str)
    news = news.sort_values("news_id").reset_index(drop=True)
    texts = [article_text(row.title, row.abstract) for row in news.itertuples(index=False)]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(base_model, cache_folder=str(cache_dir), device=device)
    model.max_seq_length = max_tokens
    emb = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32, copy=False)

    ids = news["news_id"].tolist()
    emb_path = out_dir / "emb.npy"
    ids_path = out_dir / "ids.json"
    meta_path = out_dir / "meta.json"
    spotcheck_path = out_dir / "spotcheck.json"

    np.save(emb_path, emb)
    ids_path.write_text(json.dumps(ids, indent=2), encoding="utf-8")
    spotcheck = build_spotcheck(news, emb, seed=seed)
    spotcheck_path.write_text(json.dumps(spotcheck, indent=2, sort_keys=True), encoding="utf-8")

    meta = {
        "variant": variant,
        "base_model": base_model,
        "finetuned": False,
        "finetune_data": None,
        "max_tokens": max_tokens,
        "fields": "title + abstract",
        "D": int(emb.shape[1]),
        "n_items": int(emb.shape[0]),
        "normalized": True,
        "dtype": str(emb.dtype),
        "device": device,
        "batch_size": batch_size,
        "created": datetime.now().isoformat(timespec="seconds"),
        "emb_sha256": sha256_file(emb_path),
        "ids_sha256": sha256_file(ids_path),
        "spotcheck_path": str(spotcheck_path),
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"meta": meta, "spotcheck": spotcheck}, indent=2, sort_keys=True))
    return meta


def article_text(title: object, abstract: object) -> str:
    title_text = "" if pd.isna(title) else str(title).strip()
    abstract_text = "" if pd.isna(abstract) else str(abstract).strip()
    if abstract_text:
        return f"{title_text}. {abstract_text}" if title_text else abstract_text
    return title_text


def build_spotcheck(news: pd.DataFrame, emb: np.ndarray, seed: int, n_queries: int = 5, n_neighbors: int = 5) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    eligible = news.index[news["title"].astype(str).str.len() > 0].to_numpy()
    chosen = sorted(rng.choice(eligible, size=n_queries, replace=False).tolist())
    spotcheck: list[dict[str, object]] = []
    for idx in chosen:
        scores = emb @ emb[idx]
        neighbor_count = min(n_neighbors + 1, len(scores))
        head = np.argpartition(-scores, np.arange(neighbor_count))[:neighbor_count]
        head = head[np.argsort(-scores[head])]
        neighbors = []
        for neighbor_idx in head:
            if int(neighbor_idx) == int(idx):
                continue
            row = news.iloc[int(neighbor_idx)]
            neighbors.append(
                {
                    "news_id": str(row.news_id),
                    "title": str(row.title),
                    "category": str(row.category),
                    "cosine": float(scores[int(neighbor_idx)]),
                }
            )
            if len(neighbors) >= n_neighbors:
                break
        row = news.iloc[int(idx)]
        spotcheck.append(
            {
                "query_news_id": str(row.news_id),
                "query_title": str(row.title),
                "query_category": str(row.category),
                "neighbors": neighbors,
            }
        )
    return spotcheck


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build article embedding artifacts.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed")
    embed.add_argument("--processed", type=Path, default=Path("data/processed"))
    embed.add_argument("--out", type=Path, default=Path("data/embeddings/minilm_frozen"))
    embed.add_argument("--variant", default="minilm_frozen")
    embed.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    embed.add_argument("--max-tokens", type=int, default=32)
    embed.add_argument("--batch-size", type=int, default=128)
    embed.add_argument("--cache-dir", type=Path, default=Path("data/embeddings/model_cache"))
    embed.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "embed":
        embed_articles(
            processed_dir=args.processed,
            out_dir=args.out,
            variant=args.variant,
            base_model=args.base_model,
            max_tokens=args.max_tokens,
            batch_size=args.batch_size,
            cache_dir=args.cache_dir,
            seed=args.seed,
        )
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
