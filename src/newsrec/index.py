"""Exact FAISS index helpers for Track B retrieval."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np


def build_index(vectors: np.ndarray):
    import faiss

    if vectors.ndim != 2:
        raise ValueError("vectors must have shape [N, D]")
    vectors = np.ascontiguousarray(vectors.astype(np.float32, copy=False))
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save_index(index, path: Path) -> None:
    import faiss

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def search_batched(index, queries: np.ndarray, top_k: int, batch_size: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    queries = np.ascontiguousarray(queries.astype(np.float32, copy=False))
    all_scores: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    for start in range(0, len(queries), batch_size):
        scores, indices = index.search(queries[start : start + batch_size], top_k)
        all_scores.append(scores)
        all_indices.append(indices)
    return np.vstack(all_scores), np.vstack(all_indices)


def build_from_embedding_dir(embedding_dir: Path, out_path: Path) -> dict[str, object]:
    vectors = np.load(embedding_dir / "emb.npy").astype(np.float32, copy=False)
    index = build_index(vectors)
    save_index(index, out_path)
    meta = {
        "index_type": "IndexFlatIP",
        "source_embedding_dir": str(embedding_dir),
        "out_path": str(out_path),
        "n_items": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]),
    }
    print(json.dumps(meta, indent=2, sort_keys=True))
    return meta


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build exact FAISS indexes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--emb", type=Path, required=True)
    build.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        build_from_embedding_dir(args.emb, args.out)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
