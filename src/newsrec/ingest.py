"""Ingest raw MIND TSV files into parquet contracts.

MIND does not ship publication timestamps. The `first_impression_ts` column
written here is the earliest time an article appears in any logged impression
candidate list across the train and dev behavior files. It is a proxy for first
observed exposure, not true publication time.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


TIME_FORMAT = "%m/%d/%Y %I:%M:%S %p"

BEHAVIOR_COLUMNS = [
    "raw_impression_id",
    "user_id",
    "time",
    "history_raw",
    "impressions_raw",
]

NEWS_COLUMNS_8 = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
    "abstract_entities",
]

NEWS_COLUMNS_7 = [
    "news_id",
    "category",
    "subcategory",
    "title",
    "abstract",
    "url",
    "title_entities",
]


@dataclass(frozen=True)
class DatasetDirs:
    train: str
    dev: str


DATASETS = {
    "small": DatasetDirs(train="MINDsmall_train", dev="MINDsmall_dev"),
    "demo": DatasetDirs(train="MINDdemo_train", dev="MINDdemo_dev"),
}


def validate_tsv_field_count(path: Path, expected: int | None = None) -> dict[str, int | bool]:
    """Return TSV field-count diagnostics and optionally enforce one count."""
    counts: dict[int, int] = {}
    bad_rows = 0
    first_count: int | None = None
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            field_count = len(line.rstrip("\n").split("\t"))
            if first_count is None:
                first_count = field_count
            counts[field_count] = counts.get(field_count, 0) + 1
            if expected is not None and field_count != expected:
                bad_rows += 1
    if first_count is None:
        raise ValueError(f"{path} is empty")
    if expected is not None and bad_rows:
        raise ValueError(f"{path} has {bad_rows} rows with field count != {expected}")
    return {
        "first_row_field_count": first_count,
        "bad_rows": bad_rows,
        "all_rows_match_expected": bad_rows == 0 if expected is not None else True,
        **{f"rows_with_{field_count}_fields": count for field_count, count in counts.items()},
    }


def read_behaviors(path: Path, source_split: str, impression_offset: int) -> pd.DataFrame:
    validate_tsv_field_count(path, expected=5)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=BEHAVIOR_COLUMNS,
        dtype={
            "raw_impression_id": "int64",
            "user_id": "string",
            "time": "string",
            "history_raw": "string",
            "impressions_raw": "string",
        },
        keep_default_na=False,
        na_filter=False,
    )
    parsed_ts = pd.to_datetime(df["time"], format=TIME_FORMAT, errors="coerce")
    nat_count = int(parsed_ts.isna().sum())
    if nat_count:
        raise ValueError(f"{path} has {nat_count} unparseable timestamps")

    df["ts"] = parsed_ts
    df["source_split"] = source_split
    df["raw_impression_id"] = df["raw_impression_id"].astype("int64")
    df["impression_id"] = df["raw_impression_id"] + impression_offset
    df["history_len"] = df["history_raw"].map(_count_space_separated).astype("int16")

    duplicated = int(df["impression_id"].duplicated().sum())
    if duplicated:
        raise ValueError(f"{path} produced {duplicated} duplicate global impression IDs")

    return df[
        [
            "impression_id",
            "raw_impression_id",
            "user_id",
            "ts",
            "source_split",
            "history_raw",
            "impressions_raw",
            "history_len",
        ]
    ]


def read_news(path: Path, source_split: str) -> tuple[pd.DataFrame, dict[str, int | bool]]:
    first_count = int(validate_tsv_field_count(path)["first_row_field_count"])
    if first_count == 8:
        columns = NEWS_COLUMNS_8
    elif first_count == 7:
        columns = NEWS_COLUMNS_7
    else:
        raise ValueError(f"{path} first row has {first_count} fields; expected 7 or 8")

    diagnostics = validate_tsv_field_count(path, expected=first_count)
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=columns,
        dtype="string",
        keep_default_na=False,
        na_filter=False,
    )
    if first_count == 7:
        df["abstract_entities"] = ""
    df["source_split"] = source_split
    return df[NEWS_COLUMNS_8 + ["source_split"]], diagnostics


def explode_impressions(behaviors: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[int, int, str, pd.Timestamp, str, str, int, int, int]] = []
    for record in behaviors.itertuples(index=False):
        tokens = str(record.impressions_raw).split()
        n_shown = len(tokens)
        if n_shown == 0:
            raise ValueError(f"impression_id={record.impression_id} has no impression candidates")
        for position, token in enumerate(tokens):
            try:
                news_id, label = token.rsplit("-", 1)
            except ValueError as exc:
                raise ValueError(f"bad impression token {token!r}") from exc
            if label not in {"0", "1"}:
                raise ValueError(f"bad click label {label!r} in token {token!r}")
            rows.append(
                (
                    int(record.impression_id),
                    int(record.raw_impression_id),
                    str(record.user_id),
                    record.ts,
                    str(record.source_split),
                    news_id,
                    int(label),
                    int(position),
                    int(n_shown),
                )
            )

    df = pd.DataFrame(
        rows,
        columns=[
            "impression_id",
            "raw_impression_id",
            "user_id",
            "ts",
            "source_split",
            "news_id",
            "label",
            "position",
            "n_shown",
        ],
    )
    df = df.astype(
        {
            "impression_id": "int64",
            "raw_impression_id": "int64",
            "user_id": "string",
            "source_split": "string",
            "news_id": "string",
            "label": "int8",
            "position": "int16",
            "n_shown": "int16",
        }
    )
    assert_contiguous_positions(df)
    return df


def explode_history(behaviors: pd.DataFrame) -> pd.DataFrame:
    rows: list[tuple[int, int, str, str, str, pd.Timestamp, int]] = []
    for record in behaviors.itertuples(index=False):
        history_raw = str(record.history_raw).strip()
        if not history_raw:
            continue
        for hist_position, hist_news_id in enumerate(history_raw.split()):
            rows.append(
                (
                    int(record.impression_id),
                    int(record.raw_impression_id),
                    str(record.user_id),
                    str(record.source_split),
                    hist_news_id,
                    record.ts,
                    int(hist_position),
                )
            )

    return pd.DataFrame(
        rows,
        columns=[
            "impression_id",
            "raw_impression_id",
            "user_id",
            "source_split",
            "hist_news_id",
            "ts",
            "hist_position",
        ],
    ).astype(
        {
            "impression_id": "int64",
            "raw_impression_id": "int64",
            "user_id": "string",
            "source_split": "string",
            "hist_news_id": "string",
            "hist_position": "int16",
        }
    )


def union_news(train_news: pd.DataFrame, dev_news: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    train_news = train_news.drop_duplicates("news_id", keep="first").copy()
    dev_news = dev_news.drop_duplicates("news_id", keep="first").copy()

    shared = train_news.merge(dev_news, on="news_id", suffixes=("_train", "_dev"))
    conflict_cols = ["title", "abstract", "category", "subcategory"]
    conflict_mask = pd.Series(False, index=shared.index)
    for col in conflict_cols:
        conflict_mask = conflict_mask | (shared[f"{col}_train"] != shared[f"{col}_dev"])
    conflict_count = int(conflict_mask.sum())

    train_ids = set(train_news["news_id"].astype(str))
    dev_ids = set(dev_news["news_id"].astype(str))
    both_ids = train_ids & dev_ids
    dev_only = dev_news.loc[~dev_news["news_id"].isin(train_ids)].copy()

    combined = pd.concat([train_news, dev_only], ignore_index=True)
    combined["source_split"] = combined["news_id"].map(
        lambda news_id: "both" if str(news_id) in both_ids else "train" if str(news_id) in train_ids else "dev"
    )
    diagnostics = {
        "train_only": len(train_ids - dev_ids),
        "dev_only": len(dev_ids - train_ids),
        "both": len(both_ids),
        "conflicting_shared_news_rows": conflict_count,
        "duplicates_dropped_train": int(len(train_news) - train_news["news_id"].nunique()),
        "duplicates_dropped_dev": int(len(dev_news) - dev_news["news_id"].nunique()),
    }
    return combined[NEWS_COLUMNS_8 + ["source_split"]], diagnostics


def attach_first_impression_ts(news: pd.DataFrame, impressions: pd.DataFrame) -> pd.DataFrame:
    first_seen = impressions.groupby("news_id", observed=True)["ts"].min()
    news = news.copy()
    news["first_impression_ts"] = news["news_id"].map(first_seen)
    return news


def assert_contiguous_positions(impressions: pd.DataFrame) -> None:
    grouped = impressions.groupby("impression_id", observed=True).agg(
        n_rows=("position", "size"),
        min_position=("position", "min"),
        max_position=("position", "max"),
        n_shown=("n_shown", "first"),
    )
    bad = grouped[
        (grouped["min_position"] != 0)
        | (grouped["max_position"] != grouped["n_shown"] - 1)
        | (grouped["n_rows"] != grouped["n_shown"])
    ]
    if not bad.empty:
        raise AssertionError(f"{len(bad)} impressions have non-contiguous candidate positions")


def ingest_dataset(raw_dir: Path, out_dir: Path, dataset: str) -> dict[str, object]:
    dirs = DATASETS[dataset]
    train_dir = raw_dir / dirs.train
    dev_dir = raw_dir / dirs.dev
    for required in ["behaviors.tsv", "news.tsv", "entity_embedding.vec", "relation_embedding.vec"]:
        for split_dir in [train_dir, dev_dir]:
            if not (split_dir / required).exists():
                raise FileNotFoundError(split_dir / required)

    train_behaviors = read_behaviors(train_dir / "behaviors.tsv", "train", impression_offset=0)
    dev_offset = int(train_behaviors["raw_impression_id"].max())
    dev_behaviors = read_behaviors(dev_dir / "behaviors.tsv", "dev", impression_offset=dev_offset)
    behaviors = pd.concat([train_behaviors, dev_behaviors], ignore_index=True)
    if behaviors["impression_id"].duplicated().any():
        raise ValueError("global impression_id collision after train/dev offset")

    train_news, train_news_diag = read_news(train_dir / "news.tsv", "train")
    dev_news, dev_news_diag = read_news(dev_dir / "news.tsv", "dev")
    news, news_union_diag = union_news(train_news, dev_news)

    impressions = explode_impressions(behaviors)
    history = explode_history(behaviors)
    news = attach_first_impression_ts(news, impressions)

    out_dir.mkdir(parents=True, exist_ok=True)
    news.to_parquet(out_dir / "news.parquet", index=False)
    impressions.to_parquet(out_dir / "impressions.parquet", index=False)
    history.to_parquet(out_dir / "history.parquet", index=False)

    manifest = {
        "dataset": dataset,
        "raw_train_dir": str(train_dir),
        "raw_dev_dir": str(dev_dir),
        "outputs": {
            "news": str(out_dir / "news.parquet"),
            "impressions": str(out_dir / "impressions.parquet"),
            "history": str(out_dir / "history.parquet"),
        },
        "behavior_field_counts": {
            "train": validate_tsv_field_count(train_dir / "behaviors.tsv", expected=5),
            "dev": validate_tsv_field_count(dev_dir / "behaviors.tsv", expected=5),
        },
        "news_field_counts": {
            "train": train_news_diag,
            "dev": dev_news_diag,
        },
        "news_union": news_union_diag,
        "body_text_field_present": False,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _count_space_separated(value: object) -> int:
    text = str(value).strip()
    return 0 if not text else len(text.split())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest MIND TSV files into parquet.")
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    manifest = ingest_dataset(args.raw, args.out, args.dataset)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
