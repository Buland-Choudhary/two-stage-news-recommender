"""Temporal split construction and split assertions for MIND-small."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


def load_impression_level(processed_dir: Path) -> pd.DataFrame:
    impressions = pd.read_parquet(processed_dir / "impressions.parquet")
    return impressions[["impression_id", "user_id", "source_split", "ts"]].drop_duplicates("impression_id")


def choose_val_cutoff(train_impressions: pd.DataFrame) -> tuple[pd.Timestamp, str]:
    max_train_ts = train_impressions["ts"].max()
    last_day_start = pd.Timestamp(max_train_ts.date())
    val_count = int((train_impressions["ts"] >= last_day_start).sum())
    frac = val_count / len(train_impressions)
    if frac >= 0.05:
        return last_day_start, "last_calendar_day"

    cutoff = train_impressions["ts"].quantile(0.9, interpolation="nearest")
    return pd.Timestamp(cutoff), "last_10_percent_by_timestamp"


def build_split(processed_dir: Path, out: Path, meta_out: Path) -> dict[str, object]:
    impression_level = load_impression_level(processed_dir)
    raw_train = impression_level.loc[impression_level["source_split"] == "train"].copy()
    raw_dev = impression_level.loc[impression_level["source_split"] == "dev"].copy()
    if raw_train.empty or raw_dev.empty:
        raise ValueError("both train and dev impressions are required")

    cutoff, cutoff_rule = choose_val_cutoff(raw_train)
    train_ids = raw_train.loc[raw_train["ts"] < cutoff, "impression_id"]
    val_ids = raw_train.loc[raw_train["ts"] >= cutoff, "impression_id"]
    test_ids = raw_dev["impression_id"]
    split = pd.concat(
        [
            pd.DataFrame({"impression_id": train_ids.astype("int64"), "split": "train"}),
            pd.DataFrame({"impression_id": val_ids.astype("int64"), "split": "val"}),
            pd.DataFrame({"impression_id": test_ids.astype("int64"), "split": "test"}),
        ],
        ignore_index=True,
    )
    if split["impression_id"].duplicated().any():
        raise AssertionError("at least one impression appears in multiple splits")

    assert_split_temporal_order(impression_level, split)

    merged = split.merge(impression_level, on="impression_id", how="left", validate="one_to_one")
    ranges = {
        name: _range(group["ts"])
        for name, group in merged.groupby("split", observed=True)
    }
    users = {
        name: set(group["user_id"].astype(str))
        for name, group in merged.groupby("split", observed=True)
    }
    user_overlap = {
        "train_val": len(users.get("train", set()) & users.get("val", set())),
        "train_test": len(users.get("train", set()) & users.get("test", set())),
        "val_test": len(users.get("val", set()) & users.get("test", set())),
    }
    meta = {
        "created": pd.Timestamp.now().isoformat(),
        "split_version": "split_v1",
        "construction": "train=MINDsmall_train before cutoff, val=MINDsmall_train after cutoff, test=MINDsmall_dev",
        "val_cutoff_ts": cutoff.isoformat(sep=" "),
        "val_cutoff_rule": cutoff_rule,
        "train_ts_range": [ranges["train"]["min"], ranges["train"]["max"]],
        "val_ts_range": [ranges["val"]["min"], ranges["val"]["max"]],
        "test_ts_range": [ranges["test"]["min"], ranges["test"]["max"]],
        "raw_train_ts_range": [
            raw_train["ts"].min().isoformat(sep=" "),
            raw_train["ts"].max().isoformat(sep=" "),
        ],
        "raw_dev_ts_range": [
            raw_dev["ts"].min().isoformat(sep=" "),
            raw_dev["ts"].max().isoformat(sep=" "),
        ],
        "assertions": {
            "max(train.ts) <= min(val.ts)": True,
            "max(val.ts) <= min(test.ts)": True,
            "min(raw_dev.ts) >= max(raw_train.ts)": True,
        },
        "dev_starts_after_train_ends": True,
        "n_impressions": {
            name: int(count)
            for name, count in split["split"].value_counts().sort_index().items()
        },
        "n_users": {
            name: int(len(user_set))
            for name, user_set in users.items()
        },
        "user_overlap": user_overlap,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    split.to_parquet(out, index=False)
    meta_out.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return meta


def assert_split_temporal_order(impression_level: pd.DataFrame, split: pd.DataFrame) -> None:
    merged = split.merge(impression_level, on="impression_id", how="left", validate="one_to_one")
    train = merged.loc[merged["split"] == "train", "ts"]
    val = merged.loc[merged["split"] == "val", "ts"]
    test = merged.loc[merged["split"] == "test", "ts"]
    raw_train = impression_level.loc[impression_level["source_split"] == "train", "ts"]
    raw_dev = impression_level.loc[impression_level["source_split"] == "dev", "ts"]

    if train.max() > val.min():
        raise AssertionError("max(train.ts) <= min(val.ts)")
    if val.max() > test.min():
        raise AssertionError("max(val.ts) <= min(test.ts)")
    if raw_dev.min() < raw_train.max():
        raise AssertionError("min(raw_dev.ts) >= max(raw_train.ts)")


def verify_split(processed_dir: Path, split_file: Path) -> None:
    impression_level = load_impression_level(processed_dir)
    split = pd.read_parquet(split_file)
    assert_split_temporal_order(impression_level, split)


def _range(series: pd.Series) -> dict[str, object]:
    min_ts = series.min()
    max_ts = series.max()
    return {
        "min": min_ts.isoformat(sep=" "),
        "max": max_ts.isoformat(sep=" "),
        "span_days": (max_ts - min_ts).total_seconds() / 86400.0,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or verify temporal splits.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--processed", type=Path, default=Path("data/processed"))
    build.add_argument("--out", type=Path, default=Path("data/splits/split_v1.parquet"))
    build.add_argument("--meta-out", type=Path, default=Path("data/splits/split_v1_meta.json"))
    verify = subparsers.add_parser("verify")
    verify.add_argument("--processed", type=Path, default=Path("data/processed"))
    verify.add_argument("--split", type=Path, default=Path("data/splits/split_v1.parquet"))
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "build":
        meta = build_split(args.processed, args.out, args.meta_out)
        print(json.dumps(meta, indent=2, sort_keys=True))
    elif args.command == "verify":
        verify_split(args.processed, args.split)
        print("split verification passed")
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
