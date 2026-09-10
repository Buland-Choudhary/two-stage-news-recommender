import numpy as np
import pandas as pd
import pytest
import torch

from newsrec.losses import in_batch_contrastive_loss
from newsrec.metrics import TrackB


def test_temperature_changes_perfect_match_cross_entropy():
    vectors = torch.eye(32)
    control = in_batch_contrastive_loss(vectors, vectors, temperature=1.0)
    scaled = in_batch_contrastive_loss(vectors, vectors, temperature=0.07)
    assert control.item() == pytest.approx(np.log(np.e + 31) - 1, rel=1e-6)
    assert scaled < 0.0001
    with pytest.raises(ValueError):
        in_batch_contrastive_loss(vectors, vectors, temperature=0)


def test_array_metrics_equal_dataframe_metrics():
    ranks = np.array([[0, 1, 2], [2, 3, 0]])
    relevant = [{1, 3}, {2}]
    recs = pd.DataFrame([(str(q), str(item), k + 1) for q, row in enumerate(ranks)
                         for k, item in enumerate(row)], columns=['query_id', 'news_id', 'rank'])
    truth = pd.DataFrame([(str(q), str(item)) for q, items in enumerate(relevant)
                          for item in items], columns=['query_id', 'news_id'])
    old = TrackB.evaluate(recs, truth, 4, (1, 2, 3), (1, 2))
    new = TrackB.evaluate_indices(ranks, relevant, 4, (1, 2, 3), (1, 2))
    assert new == old
