"""Batched exact retrieval evaluation, including both cold-start universes."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from newsrec.baselines.popularity import build_truth, recency_popularity_recommendations
from newsrec.index import build_index, search_batched
from newsrec.metrics import TrackB
from newsrec.runlog import append_run
from newsrec.towers import TwoTowerRetrievalModel
from newsrec.train_retrieval import (
    build_query_histories, encode_query_histories, evaluate_track_a, git_commit,
    load_embedding_artifact, load_inputs, project_all_items,
)

RUNS = Path('results/runs.csv')


class EvaluationData:
    def __init__(self):
        self.news, self.impressions, self.history = load_inputs(
            Path('data/processed'), Path('data/splits/split_v1.parquet'))
        self.ids, self.frozen = load_embedding_artifact(Path('data/embeddings/minilm_frozen'))
        self.item_index = {item: i for i, item in enumerate(self.ids)}
        assert set(self.ids) == set(self.news.news_id.astype(str))
        self.query_cache = {}
        self.fallback_cache = {}

    def queries(self, split):
        if split not in self.query_cache:
            self.query_cache[split] = build_query_histories(
                self.impressions, self.history, self.item_index, split, 50)
        return self.query_cache[split]

    def truth(self, split, query_ids):
        frame = build_truth(self.impressions, split)
        lookup = {}
        for query, item in frame.itertuples(index=False, name=None):
            lookup.setdefault(str(query), set()).add(self.item_index[str(item)])
        return [lookup.get(str(query), set()) for query in query_ids]

    def cold_pools(self):
        shown = {split: set(self.impressions.loc[self.impressions.split == split, 'news_id'].astype(str))
                 for split in ('train', 'val', 'test')}
        return {name: np.array(sorted(self.item_index[item] for item in pool)) for name, pool in {
            'unseen_in_train': shown['test'] - shown['train'],
            'unseen_in_anything_prior': shown['test'] - (shown['train'] | shown['val']),
        }.items()}

    def history_cold_exposure(self):
        history_items = set(self.history.loc[self.history.split == 'train', 'hist_news_id'].astype(str))
        return {name: len(history_items & {self.ids[int(i)] for i in pool})
                for name, pool in self.cold_pools().items()}

    def fallback(self, pool=None):
        selection = json.loads(Path('results/baseline_selection.json').read_text())['prior']
        key = 'full' if pool is None else tuple(pool)
        if key not in self.fallback_cache:
            ids = self.ids if pool is None else [self.ids[int(i)] for i in pool]
            queries = self.queries('test')
            query_ids = queries.queries.query_id.astype(str).tolist()
            empty_ids = set(query_ids) - set(queries.nonempty_query_ids)
            # Counts outside a restricted universe cannot be returned as candidates.
            frame = self.impressions.loc[(self.impressions.split != 'test') |
                                         self.impressions.impression_id.astype(str).isin(empty_ids)].copy()
            if pool is not None:
                frame = frame.copy()
                frame.loc[~frame.news_id.astype(str).isin(ids), 'label'] = 0
            output = np.full((len(query_ids), min(200, len(ids))), -1, dtype=np.int64)
            if empty_ids:
                empty_order = [query for query in query_ids if query in empty_ids]
                recs, _ = recency_popularity_recommendations(frame, ids, selection['window_hours'],
                                                             'prior', 'test', min(200, len(ids)))
                empty_ranks = frame_to_indices(recs, empty_order, self.item_index)
                positions = [i for i, query in enumerate(query_ids) if query in empty_ids]
                output[positions] = empty_ranks
            self.fallback_cache[key] = output
        return self.fallback_cache[key]


def frame_to_indices(frame, query_ids, item_index):
    frame = frame.copy()
    frame['query_id'] = frame.query_id.astype(str)
    frame['item_idx'] = frame.news_id.astype(str).map(item_index)
    assert not frame.item_idx.isna().any()
    pivot = frame.pivot(index='query_id', columns='rank', values='item_idx')
    return pivot.reindex([str(q) for q in query_ids]).to_numpy(dtype=np.int64)


def metric_fields(result):
    return dict(recall10=result.recall[10], recall50=result.recall[50], recall100=result.recall[100],
                ndcg10=result.ndcg[10], ndcg50=result.ndcg[50], coverage=result.coverage, gini=result.gini)


def log_evaluation(data, rung, ranked, restricted, config, seed=0, extra=None):
    queries = data.queries('test')
    query_ids = queries.queries.query_id.tolist()
    relevant = data.truth('test', query_ids)
    result = TrackB.evaluate_indices(ranked, relevant, len(data.ids))
    cold = {}
    for definition, pool in data.cold_pools().items():
        pool_set = set(pool.tolist())
        cold_truth = [items & pool_set for items in relevant]
        full = TrackB.evaluate_indices(ranked, cold_truth, len(data.ids), (50,), (50,))
        limited = TrackB.evaluate_indices(restricted[definition], cold_truth, len(data.ids), (50,), (50,))
        cold[definition] = dict(cold_recall50_restricted=limited.recall[50],
                                cold_recall50_fullcorpus=full.recall[50],
                                cold_share_topk=float(np.isin(ranked[:, :50], pool).mean()),
                                n_cold_articles=len(pool), evaluated_queries=full.evaluated_queries)
    config = {**config, 'evaluation_split': 'test', 'cold_definitions': cold,
              'baseline_selection': json.loads(Path('results/baseline_selection.json').read_text()),
              'cold_articles_in_raw_training_histories': data.history_cold_exposure(),
              'cold_restricted_pool': 'test-period shown articles absent from exposure window',
              'cold_query_population': 'queries with at least one relevant cold click',
              'coverage_gini_k': 100, 'cold_share_k': 50,
              'empty_history': queries.stats, 'index_type': 'IndexFlatIP', 'search_batch_size': 1024}
    return _append_evaluation(rung, seed, result, cold, config, extra)


def _append_evaluation(rung, seed, result, cold, config, extra):
    primary = cold['unseen_in_train']
    return append_run(RUNS, dict(status='success', rung=rung, stage='retrieval', seed=seed,
                       candidate_universe='full_corpus', split_version='split_v1', git_commit=git_commit(),
                       machine='buland-HP-Pavilion-Gaming-Laptop-15-dk1xxx', **metric_fields(result),
                       cold_recall50_restricted=primary['cold_recall50_restricted'],
                       cold_recall50_fullcorpus=primary['cold_recall50_fullcorpus'],
                       cold_share_topk=primary['cold_share_topk'], config_json=config, **(extra or {})))


def rank_dense(data, item_vectors, query_vectors, queries, pool=None):
    candidate_indices = np.arange(len(data.ids)) if pool is None else pool
    index = build_index(item_vectors[candidate_indices])
    _, ranks = search_batched(index, query_vectors, min(200, len(candidate_indices)), 1024)
    ranks = candidate_indices[ranks]
    output = data.fallback(pool).copy()
    positions = {str(q): i for i, q in enumerate(queries.queries.query_id)}
    for row, query_id in enumerate(queries.nonempty_query_ids):
        output[positions[query_id]] = ranks[row]
    return output


def evaluate_checkpoint(data, directory, rung, seed=0):
    directory = Path(directory)
    selection = json.loads((directory / 'selection.json').read_text())
    config = selection['config']
    embedding = Path('data/embeddings/minilm_finetuned_phase1') if rung == 'R5' else Path('data/embeddings/minilm_frozen')
    ids, base = load_embedding_artifact(embedding)
    assert ids == data.ids
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(directory / 'model.pt', map_location=device)
    model = TwoTowerRetrievalModel().to(device)
    model.load_state_dict(checkpoint['model_state'])
    base_tensor = torch.from_numpy(base).to(device)
    item_vectors = project_all_items(model, base_tensor, 8192)
    queries = data.queries('test')
    user_vectors = encode_query_histories(model, base_tensor, queries.history_idx, 1024)
    ranked = rank_dense(data, item_vectors, user_vectors, queries)
    restricted = {name: rank_dense(data, item_vectors, user_vectors, queries, pool)
                  for name, pool in data.cold_pools().items()}
    tracka, diag = evaluate_track_a(model, base_tensor, item_vectors, data.impressions, queries, data.item_index, 1024)
    np.save(directory / 'test_ranked.npy', ranked)
    config = {**config, 'base_embeddings': str(embedding), 'track_a': {**diag, 'auc': tracka.auc},
              'architecture': dict(input_dim=base.shape[1], embed_dim=128, heads=4,
                                   max_history=50, dropout=0.2, user_id_embedding=False),
              'encoder_seed': 0, 'seed_scope': 'projection and user-tower training',
              'matched_control': 'C4 uses corrected R3 seed 0 checkpoint; R5 changes only base embeddings',
              'fallback_protocol': 'validation-selected R0b-prior; identical across models'}
    row = log_evaluation(data, rung, ranked, restricted, config, seed,
                         dict(auc_impression=tracka.auc, batch_size=2048,
                              embed_dim=128, negatives='in_batch',
                              item_tower_mode='minilm_finetuned_frozen_projection' if rung == 'R5' else 'minilm_frozen_projection',
                              train_minutes=selection['run']['train_minutes'],
                              peak_vram_mb=selection['run']['peak_vram_mb'],
                              logq=config['logq'], notes='Track A AUC is an OFF-OBJECTIVE DIAGNOSTIC'))
    print(json.dumps({'run_id': row['run_id'], 'rung': rung, **metric_fields(TrackB.evaluate_indices(
        ranked, data.truth('test', queries.queries.query_id.tolist()), len(ids)))}), flush=True)
    return row
