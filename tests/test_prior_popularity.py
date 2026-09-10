import pandas as pd

from newsrec.baselines.popularity import recency_popularity_recommendations


def test_prior_fallback_does_not_include_current_or_future_validation_labels():
    data = pd.DataFrame({'impression_id': [1, 2, 3, 4],
                         'ts': pd.to_datetime(['2020-01-01', '2020-01-03', '2020-01-03', '2020-01-04']),
                         'news_id': ['B', 'A', 'A', 'A'], 'label': [1, 1, 1, 1],
                         'split': ['train', 'val', 'val', 'val']})
    recs, diag = recency_popularity_recommendations(data, ['A', 'B'], 6, 'prior', 'val', 2)
    head = recs[(recs.query_id == '2') & (recs['rank'] == 1)]
    assert head.news_id.item() == 'B'
    assert diag.empty_window_queries > 0


def test_counter_head_matches_full_sort_including_ties():
    from collections import Counter
    from newsrec.baselines.popularity import rank_from_counter, rank_from_counter_head
    corpus = [str(i) for i in range(200)]
    counts = Counter({str(i): i % 7 + 1 for i in range(140)})
    assert rank_from_counter_head(counts, sorted(corpus), 50) == rank_from_counter(counts, corpus)[:50]
