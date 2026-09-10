import numpy as np

from newsrec.baselines.tfidf import stable_topk
from newsrec.metrics import TrackB
from newsrec.report import selected_baseline


def test_baseline_selection_ignores_higher_test_score_of_unselected_window():
    rows = [dict(status='success', rung='R0b', recall50='0.1', run_id='chosen', notes='',
                 config=dict(val_selected=True, evaluation_split='test')),
            dict(status='success', rung='R0b', recall50='0.9', run_id='test_best', notes='',
                 config=dict(val_selected=False, evaluation_split='test'))]
    assert selected_baseline(rows, 'train')['run_id'] == 'chosen'


def test_restricted_search_can_recover_cold_item_below_full_cutoff():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    full = stable_topk(scores, np.arange(4), 2)[None, :]
    cold = np.array([2, 3])
    restricted = stable_topk(scores[cold], cold, 2)[None, :]
    truth = [{2}]
    assert TrackB.evaluate_indices(full, truth, 4, (2,), (2,)).recall[2] == 0
    assert TrackB.evaluate_indices(restricted, truth, 4, (2,), (2,)).recall[2] == 1


def test_topk_ties_are_broken_by_corpus_index():
    assert stable_topk(np.ones(4), np.array([3, 1, 2, 0]), 2).tolist() == [0, 1]
