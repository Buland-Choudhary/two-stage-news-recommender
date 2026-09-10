"""Generate results tables from run rows, without selecting by test performance."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev


def load_runs(path):
    with Path(path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row['config'] = json.loads(row.get('config_json') or '{}')
    return rows


def selected_baseline(rows, source):
    candidates = [row for row in rows if row['status'] == 'success'
                  and 'SUPERSEDED' not in row['notes']
                  and row['rung'] == ('R0b' if source == 'train' else 'R0b-prior')
                  and row['config'].get('val_selected') is True
                  and row['config'].get('evaluation_split') == 'test']
    if not candidates:
        raise ValueError(f'No validation-selected {source} baseline; run select_baselines.py')
    return candidates[-1]


def current_test_rows(rows, rung):
    candidates = [row for row in rows if row['status'] == 'success' and row['rung'] == rung
                  and row['stage'] == 'retrieval' and 'SUPERSEDED' not in row['notes']
                  and row['config'].get('evaluation_split', 'test') == 'test']
    if rung not in {'R4', 'R5'}:
        return candidates[-1:]
    by_seed = {row['seed']: row for row in candidates}
    return [by_seed[seed] for seed in sorted(by_seed)]


def spread(values):
    return f'{values[0]:.6f}' if len(values) == 1 else f'{mean(values):.6f} +/- {pstdev(values):.6f}'


def generate_trackb_table(runs_file, stats_file, out_path):
    rows = load_runs(runs_file)
    stats = json.loads(Path(stats_file).read_text())
    train, prior = selected_baseline(rows, 'train'), selected_baseline(rows, 'prior')
    lines = ['# Track B Results Table v2', '',
             f"Full corpus: {stats['N_ARTICLES']:,} articles; {stats['NEWS_WITHOUT_FIRST_IMPRESSION_TS']:,} never appear in labelled impressions; "
             '5,369 are shown on the test day. Absolute Recall@K is structurally small.', '',
             'R0b-prior has access to Nov 14 labels that retrieval model training does not. '
             'Empty-history fallback uses that same prior baseline for every new retrieval model.', '',
             f"Windows selected on validation: train-only {train['config']['window_hours']}h (`{train['run_id']}`); "
             f"prior {prior['config']['window_hours']}h (`{prior['run_id']}`). Test-best windows are exploratory only.", '',
             '| Rung | Run IDs | Seeds | Recall@10 | Recall@50 | Delta R0b | x R0b | Delta R0b-prior | x R0b-prior | Recall@100 | nDCG@10 | nDCG@50 | Coverage@100 | Gini@100 |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rung in ('R0a', 'R0a-prior', 'R0b', 'R0b-prior', 'R1', 'R3', 'R4', 'C1', 'C2', 'C3', 'C4', 'R5'):
        group = [train] if rung == 'R0b' else [prior] if rung == 'R0b-prior' else current_test_rows(rows, rung)
        if not group:
            continue
        recalls = [float(row['recall50']) for row in group]
        metric = lambda key: spread([float(row[key]) for row in group])
        cells = [rung, ', '.join(row['run_id'] for row in group),
                 f'{len(group)} seeds' if len(group) > 1 else 'single seed', metric('recall10'), metric('recall50')]
        for baseline in (train, prior):
            value = float(baseline['recall50'])
            cells.extend([spread([r - value for r in recalls]), spread([r / value for r in recalls])])
        cells.extend(metric(key) for key in ('recall100', 'ndcg10', 'ndcg50', 'coverage', 'gini'))
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Spread is population standard deviation across the stated seeds, not a confidence interval. '
              'R1 is the historical ALS applicability diagnostic; its fallback was selected on test in Week 2, '
              'so its end-to-end score is not a selection-clean comparison. Superseded R3/R4 runs are excluded. '
              'Track A AUC is an OFF-OBJECTIVE DIAGNOSTIC and is intentionally absent from this retrieval table.']
    text = '\n'.join(lines) + '\n'
    Path(out_path).write_text(text)
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['trackb'])
    parser.add_argument('--runs', type=Path, default=Path('results/runs.csv'))
    parser.add_argument('--stats', type=Path, default=Path('data/stats/dataset_stats.json'))
    parser.add_argument('--out', type=Path, default=Path('results/trackb_results_v2.md'))
    args = parser.parse_args()
    print(generate_trackb_table(args.runs, args.stats, args.out))


if __name__ == '__main__':
    main()
