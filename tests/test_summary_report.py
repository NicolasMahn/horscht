import copy

import pytest

from build_summary_report import build, transfer_scores
from test_size_report import fixture


def test_pick_excludes_failed_training_and_retains_observed_score():
    strong = fixture('qwen-instruct')
    for run in strong['conditions']['64']['runs']:
        run['report']['probe']['accuracy'] = .9
        run['report']['training']['history'][-1]['probe_accuracy'] = .9
    strong['conditions']['64']['runs'][0]['report']['passed'] = False
    html = build([fixture(), strong])
    assert '<h1>Current pick: Qwen3 · Base</h1>' in html
    assert '90.0%' in html and 'Not eligible' in html
    assert '2 / 9 models measured · 1 eligible' in html


def test_accuracy_tie_uses_loss_and_transfer_does_not_select():
    records = [fixture(), fixture('qwen-instruct')]
    for run in records[1]['conditions']['64']['runs']:
        run['report']['probe']['loss'] = .5
        run['report']['training']['history'][-1]['probe_loss'] = .5
    transfers = []
    for record, score in zip(records, (.95, .1)):
        transfers.append({'status': 'complete', 'manifest': {
            'candidate': record['manifest']['candidate'], 'dataset_manifest': {},
            'source_data_sha256': 'fixture', 'settings': {}, 'seeds': list(range(42, 52))},
            'examples': list(range(64)), 'runs': [
                {'seed': seed, 'conditions': {'64': {'accuracy': score}}}
                for seed in range(42, 52)]})
    html = build(records, transfers)
    assert '<h1>Current pick: Qwen3 · Post-trained</h1>' in html
    assert 'Qwen3 · Base scores higher on new tasks (95.0%)' in html


def test_pending_and_errors_are_explicit_and_escaped():
    html = build([], unavailable=[('qwen-base', '<script>bad</script>')])
    assert 'No eligible pick yet' in html and 'Unavailable' in html
    assert 'Pending' in html and '—' in html
    assert '<script>bad</script>' not in html
    assert '&lt;script&gt;bad&lt;/script&gt;' in html
    assert '0 / 9 models measured' in html


def test_partial_runs_are_not_presented_as_ten_seed_mean():
    record = fixture()
    record['conditions']['64']['runs'].pop()
    with pytest.raises(ValueError, match='Incomplete'):
        build([record])


def test_all_measured_with_exclusion_is_not_reported_as_pending():
    from report_theme import MODELS
    records = [fixture(candidate) for candidate in MODELS]
    records[0]['conditions']['64']['runs'][0]['report']['passed'] = False
    html = build(records)
    assert 'All models measured; exclusions are shown below.' in html
    assert 'still incomplete' not in html
    assert '9 / 9 models measured · 8 eligible' in html
    assert 'Not eligible' in html
