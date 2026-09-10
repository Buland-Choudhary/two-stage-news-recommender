"""Verify completed Week 3/4 artifacts without changing experimental results."""
from __future__ import annotations

import csv
import io
import json
import math
import subprocess
from pathlib import Path

import faiss
import numpy as np

from newsrec.encoders import sha256_file
from newsrec.report import current_test_rows, load_runs, selected_baseline


FROZEN_HASHES = {
    'data/splits/split_v1.parquet': '7142255d119f0d7214b2955b7c6936497bdda27aad4c84157edce91d134d00eb',
    'data/splits/split_v1_meta.json': '1517d243efbf8530c7e16619e2f11ac2d3d018342eaecc4657ecdb4decfd9952',
    'data/embeddings/minilm_frozen/emb.npy': 'cd3d44e958630b14f1bfc3bca61d14639b42207025f99b822868b8bea30ae5f9',
}
METRICS = ('recall10', 'recall50', 'recall100', 'ndcg10', 'ndcg50', 'coverage', 'gini')
COLD_METRICS = ('cold_recall50_restricted', 'cold_recall50_fullcorpus', 'cold_share_topk')


def main():
    rows = load_runs('results/runs.csv')
    by_id = {row['run_id']: row for row in rows}
    assert len(by_id) == len(rows), 'duplicate run IDs'
    original = subprocess.check_output(['git', 'show', '64ce346:results/runs.csv'], text=True)
    historical = list(csv.DictReader(io.StringIO(original)))
    for old in historical:
        new = by_id[old['run_id']]
        for key, value in old.items():
            if key not in {'notes', 'config_json'}:
                assert new[key] == value, (old['run_id'], key, 'historical value changed')
    for path, expected in FROZEN_HASHES.items():
        assert sha256_file(Path(path)) == expected, f'frozen artifact changed: {path}'

    verified = {}
    for rung in ('R3', 'R4', 'C1', 'C2', 'C3', 'C4', 'R5'):
        group = current_test_rows(rows, rung)
        expected_seeds = {'0', '1', '2'} if rung in {'R4', 'R5'} else {'0'}
        assert {row['seed'] for row in group} == expected_seeds, f'incomplete seeds: {rung}'
        for row in group:
            assert row['candidate_universe'] == 'full_corpus'
            assert row['split_version'] == 'split_v1'
            assert 'temperature' in row['config']
            for metric in METRICS:
                assert math.isfinite(float(row[metric])) and 0 <= float(row[metric]) <= 1
            if rung in {'C1', 'C2', 'C3', 'C4', 'R5'}:
                for metric in COLD_METRICS:
                    assert math.isfinite(float(row[metric]))
                for definition, size in [('unseen_in_train', 3810), ('unseen_in_anything_prior', 2483)]:
                    cold = row['config']['cold_definitions'][definition]
                    assert cold['n_cold_articles'] == size
                    assert cold['evaluated_queries'] > 0
                    assert all(math.isfinite(cold[m]) and 0 <= cold[m] <= 1 for m in COLD_METRICS)
            if rung in {'R3', 'R4', 'C4', 'R5'}:
                assert row['config']['temperature'] == 0.05
                assert 'OFF-OBJECTIVE DIAGNOSTIC' in row['notes']
                assert math.isfinite(float(row['auc_impression']))
        verified[rung] = [row['run_id'] for row in group]

    control = current_test_rows(rows, 'C4')[0]['config']
    for row in current_test_rows(rows, 'R5'):
        config = row['config']
        for key in ('temperature', 'batch_size', 'patience', 'lr', 'epochs_requested', 'logq', 'architecture', 'evaluation_device'):
            assert config[key] == control[key], ('unmatched C4/R5 setting', key)
        assert config['logq'] is False
        assert config['base_embeddings'] != control['base_embeddings']

    index_checks = {}
    for name in ('r3_seed0', 'r4_seed0', 'r4_seed1', 'r4_seed2', 'r5_seed0', 'r5_seed1', 'r5_seed2'):
        index = faiss.read_index(f'data/index/week34_{name}.faiss')
        assert isinstance(index, faiss.IndexFlatIP) and index.ntotal == 65238 and index.d == 128
        vectors = np.load(f'data/embeddings/week34/{name}/item_emb_128.npy', mmap_mode='r')
        assert vectors.shape == (65238, 128) and np.isfinite(vectors).all()
        index_checks[name] = dict(n_items=index.ntotal, dimension=index.d, exact=True)
    vectors = np.load('data/embeddings/minilm_finetuned_phase1/emb.npy', mmap_mode='r')
    assert vectors.shape == (65238, 384) and vectors.dtype == np.float32 and np.isfinite(vectors).all()
    frozen_ids = json.loads(Path('data/embeddings/minilm_frozen/ids.json').read_text())
    assert json.loads(Path('data/embeddings/minilm_finetuned_phase1/ids.json').read_text()) == frozen_ids
    tracked = subprocess.check_output(['git', 'ls-files', '-z']).decode().split('\0')
    forbidden = ('.npy', '.pt', '.pth', '.ckpt', '.faiss', '.safetensors', '.bin')
    assert not [p for p in tracked if p.startswith(('data/raw/', 'data/processed/', 'data/embeddings/', 'data/index/'))
                or p.endswith(forbidden)]
    result = dict(status='passed', frozen_hashes=FROZEN_HASHES,
                  historical_rows_preserved=len(historical), total_run_rows=len(rows),
                  verified_runs=verified, indexes=index_checks,
                  baseline_runs={s: selected_baseline(rows, s)['run_id'] for s in ('train', 'prior')},
                  no_tracked_data_or_weights=True)
    Path('notes/WEEK34_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
