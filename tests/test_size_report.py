import copy
import re

import pytest

from build_size_report import build, paired_changes, selection, validate
from selection_policy import EXPECTED_SEEDS


def fixture(candidate='qwen-base'):
    examples = [{'id': f'{split}-{i}', 'split': split, 'options': ['A', 'B']}
                for split in ('train', 'probe') for i in range(64)]
    subsets = {str(size): [f'train-{i}' for i in range(size)] for size in (16, 64)}
    conditions = {}
    for size in ('16', '64'):
        accuracy = .5 if size == '16' else .75
        report = {'passed': True, 'reloaded_head_logits_identical': True,
                  'training': {'initial': {'loss': .7, 'accuracy': .5},
                               'final': {'loss': .01, 'accuracy': 1}, 'steps': 1,
                               'weight_change_l2': 1,
                               'history': [{'step': 0, 'loss': .7, 'accuracy': .5, 'probe_loss': 1, 'probe_accuracy': .5},
                                           {'step': 1, 'loss': .01, 'accuracy': 1, 'probe_loss': .8, 'probe_accuracy': accuracy}]},
                  'probe': {'loss': .8, 'accuracy': accuracy}}
        conditions[size] = {'training_ids': subsets[size],
                            'runs': [{'seed': seed, 'report': copy.deepcopy(report)} for seed in EXPECTED_SEEDS]}
    return {'schema_version': 1, 'experiment': 'matched-training-size',
            'manifest': {'candidate': candidate, 'data_sha256': 'fixture', 'dataset_locked_at': '2026-09-28T22:00:00Z', 'settings': {'learning_rate': .01},
                         'seeds': list(EXPECTED_SEEDS), 'subsets': subsets,
                         'probe_ids': [f'probe-{i}' for i in range(64)]},
            'examples': examples, 'conditions': conditions, 'status': 'complete', 'eligible': True,
            'cache_verification': {'features_close': True, 'predictions_identical': True}}


def test_paired_difference_uses_matching_seeds_not_run_order():
    record = fixture()
    record['conditions']['64']['runs'].reverse()
    assert paired_changes(record) == [.25] * 10


@pytest.mark.parametrize('problem', ['seeds', 'probe_ids', 'subsets', 'settings', 'history', 'endpoint'])
def test_invalid_comparisons_are_rejected(problem):
    records = [fixture(), fixture('qwen-instruct')]
    record = records[1]
    if problem == 'seeds':
        record['conditions']['64']['runs'].pop()
    elif problem == 'probe_ids':
        record['manifest']['probe_ids'][0] = 'train-0'
    elif problem == 'subsets':
        record['manifest']['subsets']['16'][0] = 'train-63'
    elif problem == 'settings':
        record['manifest']['settings']['learning_rate'] = .1
    elif problem == 'history':
        record['conditions']['16']['runs'][0]['report']['training']['history'].pop()
    else:
        record['conditions']['64']['runs'][0]['report']['probe']['accuracy'] = .9
    with pytest.raises(ValueError):
        validate(records)


def test_selection_keeps_exact_ties_and_excludes_failed_records():
    records = [fixture(), fixture('qwen-instruct')]
    assert selection(records, '64')['leaders'] == ['qwen-base', 'qwen-instruct']
    assert not selection(records, '64')['complete']
    records[1]['conditions']['64']['runs'][0]['report']['reloaded_head_logits_identical'] = False
    assert selection(records, '64')['leaders'] == ['qwen-base']


def test_report_is_separate_escaped_and_chart_ids_are_unique():
    record = fixture()
    record['manifest']['settings']['note'] = '</script><script>alert(1)</script>'
    html = build([record])
    assert 'not the previous report’s eight probes' in html
    assert 'comparison incomplete' in html
    assert record['manifest']['settings']['note'] not in html
    ids = re.findall(r'<div id="([^"]+)" class="plotly-graph-div"', html)
    assert len(ids) == len(set(ids)) == 10
    assert 'size-16-loss' in ids and 'size-64-loss' in ids
    assert not re.search(r'@@[A-Z_]+@@', html)


def test_failed_and_partial_bundles_render_explicit_exclusions(tmp_path):
    import json
    from build_size_report import load_records
    results = tmp_path / 'results' / 'size_comparison'
    results.mkdir(parents=True)
    (results / 'qwen-base.json').write_text(json.dumps({'status': 'failed', 'error': 'Model load failed'}))
    partial = fixture('qwen-instruct')
    partial['conditions']['64']['runs'].pop()
    (results / 'qwen-instruct.json').write_text(json.dumps(partial))
    records, unavailable = load_records(tmp_path)
    assert records == []
    html = build(records, unavailable)
    assert 'Model load failed' in html
    assert 'Partial measurements' in html
    assert 'No eligible candidate' in html


def test_underfit_complete_results_keep_metrics_but_explain_ineligibility():
    record = fixture()
    record['eligible'] = False
    report = record['conditions']['64']['runs'][0]['report']
    report['training']['final']['loss'] = .1
    report['training']['history'][-1]['loss'] = .1
    report['passed'] = False
    html = build([record])
    assert '64 questions, seed 42: overfitting target not reached' in html
    assert 'size-64-probe_accuracy' in html
    assert '75.00%' in html


def test_selection_eligibility_is_per_training_size():
    from build_size_report import eligible, summary_rows
    record = fixture()
    record['eligible'] = False
    report = record['conditions']['64']['runs'][0]['report']
    report['training']['final']['loss'] = .1
    report['training']['history'][-1]['loss'] = .1
    report['passed'] = False
    assert eligible(record, '16')
    assert not eligible(record, '64')
    assert selection([record], '16')['leaders'] == ['qwen-base']
    assert selection([record], '64')['leaders'] == []
    assert '64 questions, seed 42: overfitting target not reached' in summary_rows([record])
    assert '16 questions: eligible' in summary_rows([record])
