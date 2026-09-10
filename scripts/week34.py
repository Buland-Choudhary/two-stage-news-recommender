"""Reproducible Week 3 repair / Week 4 experiment commands."""
from __future__ import annotations

import argparse
import csv
import json
import traceback
from pathlib import Path

import faiss
import torch

from newsrec.runlog import append_run
from newsrec.train_retrieval import train_and_evaluate

RUNS = Path('results/runs.csv')
ROOT = Path('data/embeddings/week34')


def train(name, temperature, rung, seed=0, embedding='data/embeddings/minilm_frozen', logq=False):
    out = ROOT / name
    if (out / 'selection.json').exists():
        return json.loads((out / 'selection.json').read_text())
    try:
        return train_and_evaluate(
            Path('data/processed'), Path('data/splits/split_v1.parquet'), Path(embedding),
            RUNS, out, Path('data/index') / (name + '.faiss'), rung=rung, seed=seed,
            batch_size=2048, temperature=temperature, epochs=30, patience=5,
            validation_only=True, logq_correction=logq)
    except Exception as exc:
        append_run(RUNS, dict(status='failed', rung=rung, stage='validation_selection', seed=seed,
                             candidate_universe='full_corpus', split_version='split_v1', batch_size=2048,
                             config_json=dict(temperature=temperature, name=name, logq=logq),
                             notes=f'{type(exc).__name__}: {exc}'))
        traceback.print_exc()
        return None


def sweep():
    results = []
    for temperature in (0.02, 0.05, 0.07, 0.1, 0.2, 1.0):
        print(f'START temperature={temperature}', flush=True)
        result = train(f'sweep_t{temperature}', temperature, 'R3-sweep')
        if result:
            results.append(result)
    best = max(results, key=lambda x: x['config']['best_val_recall50']) if results else None
    summary = dict(results=results, best=best, s1=best is None or best['config']['best_val_recall50'] <= 0.0008)
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / 'sweep.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({'s1': summary['s1'], 'best': best['config'] if best else None}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['mark', 'mark_prequential', 'sweep', 'r3', 'r4', 'r5'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    if args.task in {'mark', 'mark_prequential'}:
        with RUNS.open(newline='') as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if args.task == 'mark' and row['run_id'] in {'r3_018', 'r4_019', 'r4_020', 'r4_021'} and 'SUPERSEDED' not in row['notes']:
                row['notes'] += '; SUPERSEDED: temperature=1.0 with normalized embeddings; invalid model-effect interpretation'
            config = json.loads(row.get('config_json') or '{}')
            if (args.task == 'mark_prequential' and config.get('selection_metric') == 'val_recall50'
                    and config.get('history_source') == 'prior' and 'validation_label_policy' not in config
                    and 'SUPERSEDED' not in row['notes']):
                row['notes'] += '; SUPERSEDED for selection: prequential validation updates do not match frozen-source test; retained as diagnostic'
        with RUNS.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    elif args.task == 'sweep':
        sweep()
    else:
        selection = json.loads((ROOT / 'sweep.json').read_text())
        if selection['s1'] and args.task in {'r3', 'r4'}:
            print('BLOCKED S1', flush=True)
            return
        temperature = selection['best']['config']['temperature'] if selection['best'] else 0.07
        seeds = (0,) if args.task == 'r3' else (0, 1, 2)
        for seed in seeds:
            embedding = 'data/embeddings/minilm_finetuned_phase1' if args.task == 'r5' else 'data/embeddings/minilm_frozen'
            train(f'{args.task}_seed{seed}', temperature, args.task.upper(), seed,
                  embedding=embedding, logq=args.task == 'r4')


if __name__ == '__main__':
    main()
