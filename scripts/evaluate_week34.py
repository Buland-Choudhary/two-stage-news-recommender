"""Evaluate repaired retrieval checkpoints and C1-C4 cold-start controls."""
import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import torch

from newsrec.baselines.category_pop import category_rankings
from newsrec.baselines.tfidf import tfidf_rankings
from newsrec.evaluation import EvaluationData, evaluate_checkpoint, log_evaluation, rank_dense
from newsrec.runlog import append_run


def controls(data):
    queries = data.queries('test')
    pools = data.cold_pools()
    positions = {str(q): i for i, q in enumerate(queries.queries.query_id)}
    for rung in ('C1', 'C2', 'C3'):
        config = dict(temperature=None, seed=0, fallback_protocol='validation-selected R0b-prior')
        if rung == 'C1':
            rankings = category_rankings(data.news, data.impressions, data.ids, queries.history_idx, pools)
            config['method'] = 'dominant history category first, train category click count, news_id tie break; no recency proxy'
        elif rung == 'C2':
            rankings, detail = tfidf_rankings(data.news, data.impressions, data.ids, queries.history_idx, pools)
            config.update(detail)
            config['method'] = 'cosine of mean history TF-IDF, train-fitted vocabulary, title + abstract'
        else:
            vectors = []
            for history in queries.history_idx:
                vector = data.frozen[history[history >= 0]].mean(axis=0)
                vectors.append(vector / max(np.linalg.norm(vector), 1e-12))
            users = np.asarray(vectors, dtype=np.float32)
            ranked = rank_dense(data, data.frozen, users, queries)
            restricted = {name: rank_dense(data, data.frozen, users, queries, pool) for name, pool in pools.items()}
            config['method'] = 'frozen MiniLM, normalized mean history, no trained projection or user tower'
        if rung != 'C3':
            all_rankings = {}
            for name, model_ranks in rankings.items():
                pool = None if name == 'full' else pools[name]
                combined = data.fallback(pool).copy()
                for i, query in enumerate(queries.nonempty_query_ids):
                    combined[positions[query]] = model_ranks[i]
                all_rankings[name] = combined
            ranked = all_rankings.pop('full')
            restricted = all_rankings
        row = log_evaluation(data, rung, ranked, restricted, config)
        print(json.dumps({'run_id': row['run_id'], 'rung': rung, 'recall50': row['recall50']}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['r3', 'r4', 'r5', 'controls', 'c4'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    faiss.omp_set_num_threads(4)
    data = EvaluationData()
    if args.task == 'controls':
        controls(data)
    else:
        for seed in ((0, 1, 2) if args.task in {'r4', 'r5'} else (0,)):
            model = 'r3' if args.task == 'c4' else args.task
            evaluate_checkpoint(data, Path('data/embeddings/week34') / f'{model}_seed{seed}', args.task.upper(), seed)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        append_run(Path('results/runs.csv'), dict(status='failed', rung='EVALUATION', stage='retrieval',
                   candidate_universe='full_corpus', config_json=dict(temperature=None),
                   notes=f'{type(exc).__name__}: {exc}'))
        raise
