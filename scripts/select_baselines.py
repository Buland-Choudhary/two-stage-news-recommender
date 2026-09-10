"""Validation-selected windows with all-time controls and held-out test results."""
import json
from dataclasses import asdict
from pathlib import Path

from newsrec.baselines.popularity import naive_popularity_recommendations, recency_popularity_recommendations
from newsrec.evaluation import EvaluationData, frame_to_indices, metric_fields
from newsrec.metrics import TrackB
from newsrec.runlog import append_run
from newsrec.train_retrieval import git_commit
from newsrec.report import load_runs


def main():
    data = EvaluationData()
    existing = {(r['config'].get('history_source'), r['config'].get('evaluation_split'), r['config'].get('window_hours')): r
                for r in load_runs('results/runs.csv') if r['status'] == 'success'
                and 'SUPERSEDED' not in r['notes']
                and r['rung'] in {'R0a', 'R0a-prior', 'R0b', 'R0b-prior'}
                and r['config'].get('selection_metric') == 'val_recall50'}
    selected = {}
    for source in ('train', 'prior'):
        val_results = []
        for split in ('val', 'test'):
            query_ids = data.queries(split).queries.query_id.tolist()
            truth = data.truth(split, query_ids)
            for window in (None, 6, 12, 24, 48):
                old = existing.get((source, split, window))
                if old:
                    if split == 'val' and window is not None:
                        val_results.append(dict(window_hours=window, val_recall50=float(old['recall50']),
                                                val_run_id=old['run_id'], history_source=source))
                    if split == 'test' and window == selected[source]['window_hours']:
                        selected[source].update(run_id=old['run_id'], recall50=float(old['recall50']), rung=old['rung'])
                    continue
                if window is None:
                    recs = naive_popularity_recommendations(data.impressions, data.ids, split, 100,
                                                            'train' if split == 'val' else source)
                    diagnostics = {}
                    rung = 'R0a' if source == 'train' else 'R0a-prior'
                else:
                    recs, diag = recency_popularity_recommendations(data.impressions, data.ids, window,
                                                                   'train' if split == 'val' else source, split, 100)
                    diagnostics = asdict(diag)
                    rung = 'R0b' if source == 'train' else 'R0b-prior'
                ranks = frame_to_indices(recs, query_ids, data.item_index)
                result = TrackB.evaluate_indices(ranks, truth, len(data.ids))
                is_selected = window is not None and split == 'test' and window == selected[source]['window_hours']
                config = dict(temperature=None, window_hours=window,
                              evaluation_split=split, selection_metric='val_recall50',
                              val_selected=is_selected, coverage_gini_k=100, **diagnostics)
                config['history_source'] = source
                config['validation_label_policy'] = 'held-out day frozen; no val labels during val selection'
                row = append_run(Path('results/runs.csv'), dict(status='success', rung=rung,
                                 stage='baseline_selection' if split == 'val' else 'retrieval',
                                 seed=0, split_version='split_v1', candidate_universe='full_corpus',
                                 **metric_fields(result), config_json=config, git_commit=git_commit(),
                                 notes='Window selected on held-out val day without val labels; test prior uses train+val only'))
                print(json.dumps(dict(run_id=row['run_id'], source=source, split=split, window=window,
                                      recall50=result.recall[50], selected=is_selected)), flush=True)
                if split == 'val' and window is not None:
                    val_results.append(dict(window_hours=window, val_recall50=result.recall[50],
                                            val_run_id=row['run_id'], history_source=source))
                if is_selected:
                    selected[source].update(run_id=row['run_id'], recall50=result.recall[50], rung=rung)
            if split == 'val':
                # Stable tie break prefers the shortest tied window.
                selected[source] = max(val_results, key=lambda x: (x['val_recall50'], -x['window_hours']))
    Path('results/baseline_selection.json').write_text(json.dumps(selected, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        append_run(Path('results/runs.csv'), dict(status='failed', rung='BASELINE_SWEEP',
                   stage='baseline_selection', candidate_universe='full_corpus',
                   config_json=dict(temperature=None), notes=f'{type(exc).__name__}: {exc}'))
        raise
