from __future__ import annotations

import pandas as pd

from newsrec.baselines.popularity import naive_popularity_recommendations, recency_popularity_recommendations


def test_empty_recency_window_is_reported_when_fallback_matches_r0a() -> None:
    impressions = pd.DataFrame(
        {
            "impression_id": [1, 2, 2],
            "split": ["train", "test", "test"],
            "ts": [
                pd.Timestamp("2020-01-01 00:00:00"),
                pd.Timestamp("2020-01-03 00:00:00"),
                pd.Timestamp("2020-01-03 00:00:00"),
            ],
            "news_id": ["A", "B", "C"],
            "label": [1, 1, 0],
        }
    )
    corpus_ids = ["A", "B", "C"]

    recency, diagnostics = recency_popularity_recommendations(
        impressions,
        corpus_ids,
        window_hours=1,
        history_source="train",
        top_k=3,
    )
    naive = naive_popularity_recommendations(impressions, corpus_ids, top_k=3)

    assert diagnostics.empty_window_queries == 1
    assert diagnostics.empty_window_fraction == 1.0
    assert recency["news_id"].tolist() == naive["news_id"].tolist()
