from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from newsrec.splits import assert_split_temporal_order, load_impression_level


DATA_DIR = Path("data/processed")
SPLIT_FILE = Path("data/splits/split_v1.parquet")


def test_split_file_has_one_label_per_impression() -> None:
    if not SPLIT_FILE.exists():
        pytest.skip("split_v1 has not been built")
    split = pd.read_parquet(SPLIT_FILE)
    assert not split["impression_id"].duplicated().any()
    assert set(split["split"]) == {"train", "val", "test"}


def test_split_temporal_assertions_hold_on_real_split() -> None:
    if not SPLIT_FILE.exists() or not (DATA_DIR / "impressions.parquet").exists():
        pytest.skip("processed data and split_v1 are required")
    impression_level = load_impression_level(DATA_DIR)
    split = pd.read_parquet(SPLIT_FILE)
    assert_split_temporal_order(impression_level, split)
