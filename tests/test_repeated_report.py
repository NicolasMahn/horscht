import copy
import json
from pathlib import Path

import pytest

from build_repeated_report import curve_statistics, validate
from report_theme import MODELS

ROOT = Path(__file__).parents[1]


def test_stopped_runs_remain_in_the_mean_and_sample_sd():
    runs = [
        {'training': {'history': [{'loss': 4}, {'loss': 2}]}},
        {'training': {'history': [{'loss': 6}, {'loss': 4}, {'loss': 0}]}},
    ]
    steps, averages, deviations = curve_statistics(runs, 'loss')
    assert steps == [0, 1, 2]
    assert averages == [5, 3, 1]
    assert deviations == pytest.approx([2 ** .5] * 3)


def measured_records():
    return [json.loads(path.read_text()) for path in sorted((ROOT / 'results/repeated').glob('*.json'))]


def test_all_seeds_are_complete_and_seed_42_reproduces_original():
    records = measured_records()
    if not records:
        pytest.skip("Historical repeated-run aggregates are not installed")
    assert {r["manifest"]["candidate"] for r in records} == set(MODELS)
    validate(records)
    for record in records:
        assert [r['seed'] for r in record['runs']] == list(range(42, 52))
        assert all(r['passed'] and r['reloaded_head_logits_identical'] for r in record['runs'])
        model = record['manifest']['candidate']
        original = json.loads((ROOT / 'results/comparison' / f'{model}.json').read_text())
        assert record['runs'][0]['training'] == original['report']['training']
        assert record['runs'][0]['probe'] == original['report']['probe']


@pytest.mark.parametrize('problem', ['duplicate', 'different', 'missing_step', 'setting'])
def test_incompatible_runs_are_rejected(problem):
    settings = dict.fromkeys(('data_sha256', 'learning_rate', 'max_steps', 'format', 'dtype', 'threads',
                              'capacity', 'device', 'head', 'pooling', 'max_tokens', 'optimizer',
                              'chat_template', 'feature_dtype', 'backbone_trainable'), 'synthetic')
    history = [dict(step=step, loss=1., accuracy=.5, probe_loss=1., probe_accuracy=.5) for step in range(3)]
    baseline = {'examples': [], 'manifest': settings,
                'runs': [{'seed': seed, 'training': {'history': copy.deepcopy(history), 'steps': 2}}
                         for seed in (42, 43)]}
    records = [copy.deepcopy(baseline), copy.deepcopy(baseline)]
    validate(records)
    if problem == 'duplicate':
        records[0]['runs'][1]['seed'] = records[0]['runs'][0]['seed']
    elif problem == 'different':
        records[1]['runs'][0]['seed'] = 999
    elif problem == 'missing_step':
        records[0]['runs'][0]['training']['history'].pop(1)
    else:
        records[1]['manifest']['learning_rate'] = .5
    with pytest.raises(ValueError):
        validate(records)
