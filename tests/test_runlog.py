from concurrent.futures import ThreadPoolExecutor
import csv
import json

from newsrec.runlog import append_run


def test_append_run_preserves_prior_rows(tmp_path):
    path = tmp_path / 'runs.csv'
    append_run(path, {'run_id': 'first', 'status': 'success', 'rung': 'R0a'})
    append_run(path, {'run_id': 'second', 'status': 'failed', 'rung': 'R0b'})
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert [row['run_id'] for row in rows] == ['first', 'second']
    assert rows[1]['status'] == 'failed'


def test_concurrent_run_rows_remain_complete_and_have_unique_ids(tmp_path):
    path = tmp_path / 'runs.csv'
    def write(seed):
        return append_run(path, dict(rung='TOY', status='failed' if seed == 0 else 'success',
                                    seed=seed, candidate_universe='full_corpus',
                                    config_json=dict(temperature=0.07)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, range(12)))
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 12
    assert len({row['run_id'] for row in rows}) == 12
    assert sum(row['status'] == 'failed' for row in rows) == 1
    assert all(json.loads(row['config_json'])['temperature'] == 0.07 for row in rows)
