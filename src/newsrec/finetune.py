"""Option C Phase 1: co-click pairs from training impressions only.

Pairs use observed positive labels in the same impression, not MIND's
untimestamped history. Encoder gradients never consume validation/test text.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from newsrec.encoders import article_text, sha256_file
from newsrec.losses import in_batch_contrastive_loss
from newsrec.runlog import append_run
from newsrec.train_retrieval import git_commit, seed_everything


def assert_finetune_pairs_use_training_articles_only(pairs, training_articles, test_only_articles):
    used = {article for pair in pairs for article in pair}
    assert used <= training_articles, 'fine-tuning pair contains article not shown in training window'
    assert used.isdisjoint(test_only_articles), 'fine-tuning pair contains test-only article'


def build_pairs(impressions):
    training = impressions.loc[impressions.split == 'train']
    train_articles = set(training.news_id.astype(str))
    test_only = set(impressions.loc[impressions.split == 'test', 'news_id'].astype(str)) - train_articles
    positives = training.loc[training.label == 1]
    pairs = []
    for _, group in positives.groupby('impression_id', observed=True):
        pairs.extend(combinations(sorted(set(group.news_id.astype(str))), 2))
    assert_finetune_pairs_use_training_articles_only(pairs, train_articles, test_only)
    assert training.ts.max() < impressions.loc[impressions.split == 'val', 'ts'].min()
    return pairs, dict(n_pairs=len(pairs), n_distinct_pairs=len(set(pairs)),
                       n_articles=len({item for pair in pairs for item in pair}),
                       latest_source_timestamp=str(training.ts.max()),
                       train_articles_only_assertion=True, no_test_only_articles_assertion=True)


def fit(temperature, batch_size=32, max_tokens=32, epochs=3):
    from sentence_transformers import SentenceTransformer

    seed_everything(0)
    torch.set_num_threads(4)
    started = time.perf_counter()
    out = Path('data/embeddings/minilm_finetuned_phase1')
    out.mkdir(parents=True, exist_ok=True)
    news = pd.read_parquet('data/processed/news.parquet').sort_values('news_id')
    impressions = pd.read_parquet('data/processed/impressions.parquet').merge(
        pd.read_parquet('data/splits/split_v1.parquet'), on='impression_id', validate='many_to_one')
    pairs, pair_stats = build_pairs(impressions)
    assert pairs, 'no same-impression co-click pairs in train'
    pd.DataFrame(pairs, columns=['left', 'right']).to_parquet(out / 'training_pairs.parquet', index=False)
    del impressions
    ids = news.news_id.astype(str).tolist()
    texts = {str(row.news_id): article_text(row.title, row.abstract) for row in news.itertuples()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                cache_folder='data/embeddings/model_cache', device=device,
                                local_files_only=True)
    model.max_seq_length = max_tokens
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda')
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    curves = []
    rng = np.random.default_rng(0)
    train_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(pairs))
        total_loss, count = 0.0, 0
        for start in range(0, len(order), batch_size):
            batch = [pairs[int(i)] for i in order[start:start + batch_size]]
            if len(batch) < 2:
                continue
            left = {k: v.to(device) for k, v in model.tokenize([texts[a] for a, _ in batch]).items()}
            right = {k: v.to(device) for k, v in model.tokenize([texts[b] for _, b in batch]).items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=device == 'cuda', dtype=torch.float16):
                a = F.normalize(model(left)['sentence_embedding'], dim=-1)
                b = F.normalize(model(right)['sentence_embedding'], dim=-1)
                loss = (in_batch_contrastive_loss(a, b, temperature=temperature) +
                        in_batch_contrastive_loss(b, a, temperature=temperature)) / 2
            if not torch.isfinite(loss):
                raise FloatingPointError('non-finite Phase 1 loss with GradScaler enabled')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * len(batch)
            count += len(batch)
        curves.append(dict(epoch=epoch, train_loss=total_loss / count, n_pairs=count))
        (out / 'training_history.json').write_text(json.dumps(curves, indent=2))
        print(json.dumps(curves[-1]), flush=True)
    train_minutes = (time.perf_counter() - train_started) / 60
    model.save(str(out / 'encoder'))
    model.eval()
    vectors = model.encode([texts[item] for item in ids], batch_size=128,
                           normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True)
    assert vectors.shape == (len(ids), 384), 'fine-tuned embedding contract changed'
    assert np.isfinite(vectors).all(), 'fine-tuned embeddings contain non-finite values'
    np.save(out / 'emb.npy', vectors.astype(np.float32))
    (out / 'ids.json').write_text(json.dumps(ids, indent=2))
    meta = dict(variant='minilm_finetuned_phase1', base_model='all-MiniLM-L6-v2',
                finetuned=True, finetune_data='same-impression positive co-clicks, training window only',
                temperature=temperature, batch_size=batch_size, max_tokens=max_tokens,
                lr=2e-5, epochs=epochs, seed=0, amp_dtype='fp16', grad_scaler=True,
                fields='title + abstract', D=384, n_items=len(ids), normalized=True, dtype='float32',
                training_history=curves, pair_stats=pair_stats,
                total_pipeline_minutes=(time.perf_counter() - started) / 60,
                emb_sha256=sha256_file(out / 'emb.npy'), ids_sha256=sha256_file(out / 'ids.json'))
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    row = append_run(Path('results/runs.csv'), dict(status='success', rung='PHASE1', stage='encoder_training',
                     seed=0, batch_size=batch_size, candidate_universe='full_corpus', split_version='split_v1',
                     train_minutes=train_minutes,
                     peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else 0,
                     git_commit=git_commit(), config_json=meta,
                     notes='3 fixed epochs; no test evaluation; attribution requires matched R5-C4 comparison'))
    print(json.dumps({'run_id': row['run_id'], **pair_stats}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--temperature', type=float, required=True)
    args = parser.parse_args()
    for tokens in (32, 16, 8):
        for batch in (32, 16, 8, 4, 2):
            try:
                fit(args.temperature, batch, tokens)
                return
            except torch.cuda.OutOfMemoryError as exc:
                append_run(Path('results/runs.csv'), dict(status='failed', rung='PHASE1', stage='encoder_training',
                           candidate_universe='full_corpus', config_json=dict(temperature=args.temperature,
                           batch_size=batch, max_tokens=tokens), notes=str(exc)))
                torch.cuda.empty_cache()
            except Exception as exc:
                append_run(Path('results/runs.csv'), dict(status='failed', rung='PHASE1', stage='encoder_training',
                           candidate_universe='full_corpus', config_json=dict(temperature=args.temperature,
                           batch_size=batch, max_tokens=tokens), notes=f'{type(exc).__name__}: {exc}'))
                raise
    raise RuntimeError('S2: Phase 1 OOM at every attempted batch size and sequence length')


if __name__ == '__main__':
    main()
