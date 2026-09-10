from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from newsrec.metrics import TrackA, TrackB, gini


def test_track_a_per_impression_auc_is_macro_average() -> None:
    scores = pd.DataFrame(
        {
            "impression_id": [1, 1, 2, 2],
            "score": [0.9, 0.1, 0.2, 0.8],
            "label": [1, 0, 1, 0],
        }
    )
    result = TrackA.evaluate(scores)
    assert result.auc == pytest.approx(0.5)
    assert result.skipped_impressions == 0
    assert result.evaluated_impressions == 2


def test_track_a_skips_degenerate_impressions() -> None:
    scores = pd.DataFrame(
        {
            "impression_id": [1, 1, 2, 2],
            "score": [0.9, 0.1, 0.2, 0.8],
            "label": [1, 0, 0, 0],
        }
    )
    result = TrackA.evaluate(scores)
    assert result.auc == pytest.approx(1.0)
    assert result.skipped_impressions == 1


def test_track_b_recall_known_overlap() -> None:
    recs = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q1", "q2", "q2"],
            "news_id": ["a", "b", "c", "d", "e"],
            "rank": [1, 2, 3, 1, 2],
        }
    )
    truth = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2"],
            "news_id": ["b", "x", "d"],
        }
    )
    result = TrackB.evaluate(recs, truth, corpus_size=5, recall_k=(1, 2), ndcg_k=(2,))
    assert result.recall[1] == pytest.approx(0.5)
    assert result.recall[2] == pytest.approx(0.75)


def test_gini_endpoints() -> None:
    assert gini([1, 1, 1, 1]) == pytest.approx(0.0)
    assert gini([100, 0, 0, 0, 0]) == pytest.approx(0.8)


def test_track_namespaces_reject_wrong_shape() -> None:
    track_a_like = pd.DataFrame({"impression_id": [1], "score": [0.5], "label": [1]})
    track_b_like = pd.DataFrame({"query_id": ["1"], "news_id": ["N1"], "rank": [1]})
    truth = pd.DataFrame({"query_id": ["1"], "news_id": ["N1"]})
    with pytest.raises(ValueError):
        TrackA.evaluate(track_b_like)
    with pytest.raises(ValueError):
        TrackB.evaluate(track_a_like, truth, corpus_size=1)


def test_rank_sum_auc_matches_sklearn_with_ties_and_degenerate_groups():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(17)
    rows, expected, skipped = [], [], 0
    for impression in range(100):
        labels = rng.integers(0, 2, size=int(rng.integers(2, 60)))
        scores = rng.integers(-3, 4, size=len(labels)).astype(float)
        rows.extend((impression, float(score), int(label)) for score, label in zip(scores, labels))
        if len(set(labels)) == 1:
            skipped += 1
        else:
            expected.append(roc_auc_score(labels, scores))
    result = TrackA.evaluate(pd.DataFrame(rows, columns=['impression_id', 'score', 'label']))
    assert result.auc == pytest.approx(np.mean(expected), abs=1e-12)
    assert result.skipped_impressions == skipped
