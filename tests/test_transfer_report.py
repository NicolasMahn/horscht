import json
from copy import deepcopy
from pathlib import Path

import pytest

from build_transfer_report import build


def record():
    root = Path(__file__).resolve().parents[1]
    examples = json.loads((root / 'data/transfer.json').read_text())
    measurement = {'accuracy': 0.5, 'loss': 1.2,
                   'families': {ex['family']: {'accuracy': 0.5} for ex in examples},
                   'predictions': [dict(id=ex['id'], predicted=0,
                                        probabilities=[1 / len(ex['options'])] * len(ex['options'])) for ex in examples]}
    return {'status': 'complete', 'experiment': 'novel-task-transfer',
            'manifest': {'candidate': 'qwen-base', 'dataset_manifest': {'data_sha256': 'abc', 'locked_at': '2026-09-28T00:00:00Z'},
                         'source_data_sha256': 'def', 'settings': {}, 'seeds': list(range(42, 52))},
            'examples': examples, 'uniform': {'accuracy': .3, 'loss': 1.23},
            'feature_cache': {'token_lengths': [80], 'source_feature_max_difference': 0},
            'runs': [dict(seed=seed, conditions={c: deepcopy(measurement) for c in ('initial', '16', '64')},
                          accuracy_delta_initial_to_16=0, accuracy_delta_initial_to_64=0,
                          accuracy_delta_16_to_64=0) for seed in range(42, 52)]}


def test_report_includes_all_questions_and_safe_embedded_data():
    data = record()
    data['examples'][0]['context'] = '</script><script>alert(1)</script>'
    html = build([data])
    assert html.count('<details>') == 64
    assert '</script><script>alert(1)</script>' not in html
    assert '&lt;script&gt;alert(1)&lt;/script&gt;' in html
    assert '1.200 ± 0.000' in html
    assert 'excluded from the existing development-probe selection policy' in html


def test_report_rejects_mismatched_seeds_or_question_order():
    data = record()
    data['runs'].pop()
    with pytest.raises(ValueError, match='ten matched seeds'):
        build([data])
    data = record()
    data['runs'][0]['conditions']['16']['predictions'].reverse()
    with pytest.raises(ValueError, match='Prediction order'):
        build([data])


def test_report_explicitly_counts_underfit_source_heads():
    data = record()
    data['source_training_checks'] = {'16': {'passed': 10, 'total': 10}, '64': {'passed': 5, 'total': 10}}
    html = build([data])
    assert '<td>5 / 10</td>' in html
    assert 'including heads that missed this target' in html
    assert 'do not change selection eligibility' in html
