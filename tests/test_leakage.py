from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from newsrec.baselines.popularity import rank_from_counter


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
SPLIT_FILE = Path("data/splits/split_v1.parquet")


def test_no_impression_appears_in_multiple_splits() -> None:
    if not SPLIT_FILE.exists():
        pytest.skip("split_v1 has not been built")
    split = pd.read_parquet(SPLIT_FILE)
    assert not split["impression_id"].duplicated().any()


def test_history_table_matches_raw_history_field_exactly() -> None:
    """Verify our construction used only MIND's provided History field."""
    history_path = PROCESSED_DIR / "history.parquet"
    if not history_path.exists():
        pytest.skip("history.parquet has not been built")

    history = pd.read_parquet(history_path)
    actual = {
        (row.source_split, int(row.raw_impression_id)): []
        for row in history[["source_split", "raw_impression_id"]].drop_duplicates().itertuples(index=False)
    }
    for row in history.sort_values(["source_split", "raw_impression_id", "hist_position"]).itertuples(index=False):
        actual[(row.source_split, int(row.raw_impression_id))].append(str(row.hist_news_id))

    for source_split, raw_file in [
        ("train", RAW_DIR / "MINDsmall_train" / "behaviors.tsv"),
        ("dev", RAW_DIR / "MINDsmall_dev" / "behaviors.tsv"),
    ]:
        if not raw_file.exists():
            pytest.skip("raw MIND-small files are required")
        raw = pd.read_csv(
            raw_file,
            sep="\t",
            header=None,
            names=["raw_impression_id", "user_id", "time", "history_raw", "impressions_raw"],
            dtype={"raw_impression_id": "int64", "history_raw": "string"},
            keep_default_na=False,
            na_filter=False,
        )
        for row in raw.itertuples(index=False):
            expected = [] if not str(row.history_raw).strip() else str(row.history_raw).split()
            got = actual.get((source_split, int(row.raw_impression_id)), [])
            assert got == expected


def test_popularity_ranking_uses_supplied_training_counts_only() -> None:
    from collections import Counter

    corpus = ["A", "B", "C"]
    ranked = rank_from_counter(Counter({"B": 2}), corpus)
    assert ranked == ["B", "A", "C"]
