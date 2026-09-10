import pandas as pd
import pytest

from newsrec.finetune import assert_finetune_pairs_use_training_articles_only, build_pairs


def test_pair_guard_rejects_article_not_shown_in_training():
    with pytest.raises(AssertionError):
        assert_finetune_pairs_use_training_articles_only([('A', 'C')], {'A', 'B'}, {'C'})


def test_coclick_pairs_only_use_positive_labels_from_training_impressions():
    data = pd.DataFrame({'impression_id': [1, 1, 1, 2, 2], 'news_id': ['A', 'B', 'C', 'C', 'D'],
                         'label': [1, 1, 0, 1, 1], 'split': ['train', 'train', 'train', 'val', 'val'],
                         'ts': pd.to_datetime(['2020-01-01'] * 3 + ['2020-01-02'] * 2)})
    pairs, stats = build_pairs(data)
    assert pairs == [('A', 'B')]
    assert stats['n_pairs'] == 1


def test_saved_phase1_pairs_contain_no_test_only_articles():
    from pathlib import Path
    path = Path('data/embeddings/minilm_finetuned_phase1/training_pairs.parquet')
    if not path.exists():
        pytest.skip('Phase 1 artifact not yet built')
    frame = pd.read_parquet('data/processed/impressions.parquet').merge(
        pd.read_parquet('data/splits/split_v1.parquet'), on='impression_id')
    train = set(frame.loc[frame.split == 'train', 'news_id'].astype(str))
    test_only = set(frame.loc[frame.split == 'test', 'news_id'].astype(str)) - train
    pairs = list(pd.read_parquet(path).itertuples(index=False, name=None))
    assert_finetune_pairs_use_training_articles_only(pairs, train, test_only)
