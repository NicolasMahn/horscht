import copy

import pytest

from selection_policy import EXPECTED_SEEDS, select


def record(candidate, accuracy=.75, loss=.8):
    return {'manifest': {'candidate': candidate},
            'source_reload_verification': {'features_close': True, 'predictions_identical': True},
            'runs': [{'seed': seed, 'passed': True, 'reloaded_head_logits_identical': True,
                      'training': {'final': {'accuracy': 1, 'loss': .01}, 'weight_change_l2': 1},
                      'probe': {'accuracy': accuracy, 'loss': loss}}
                     for seed in EXPECTED_SEEDS]}


def test_accuracy_is_primary_even_when_lower_accuracy_has_better_loss():
    result = select([record('a', .8, 2), record('b', .7, .1)], ['a', 'b'])
    assert result['complete']
    assert result['leaders'] == ['a']


def test_lower_loss_breaks_accuracy_tie_but_exact_tie_remains_tied():
    result = select([record('a'), record('b', loss=.7), record('c', loss=.7)], ['a', 'b', 'c'])
    assert result['leaders'] == ['b', 'c']
    assert [r['candidate'] for r in result['ranked']] == ['b', 'c', 'a']


@pytest.mark.parametrize('problem', ['missing_seed', 'duplicate_seed', 'wrong_seed', 'failed',
                                    'head_reload', 'backbone_reload', 'fake_pass', 'nan'])
def test_ineligible_candidate_cannot_win_or_make_comparison_complete(problem):
    candidate = record('a', 1, .01)
    run = candidate['runs'][0]
    if problem == 'missing_seed':
        candidate['runs'].pop()
    elif problem == 'duplicate_seed':
        candidate['runs'][-1]['seed'] = run['seed']
    elif problem == 'wrong_seed':
        run['seed'] = 999
    elif problem == 'failed':
        run['passed'] = False
    elif problem == 'head_reload':
        run['reloaded_head_logits_identical'] = False
    elif problem == 'backbone_reload':
        candidate['source_reload_verification']['features_close'] = False
    elif problem == 'fake_pass':
        run['training']['final']['loss'] = .5
    else:
        run['probe']['loss'] = float('nan')
    result = select([candidate, record('b')], ['a', 'b'])
    assert result['leaders'] == ['b']
    assert not result['complete']
    assert result['excluded'][0]['candidate'] == 'a'


def test_missing_candidates_are_explicit_and_empty_results_have_no_winner():
    result = select([], ['a', 'b'])
    assert not result['complete']
    assert result['leaders'] == []
    assert len(result['excluded']) == 2


def test_duplicate_records_are_rejected():
    candidate = record('a')
    with pytest.raises(ValueError, match='Duplicate'):
        select([candidate, copy.deepcopy(candidate)], ['a'])


def test_report_loader_excludes_stale_results_after_source_failure(tmp_path):
    import json
    from build_repeated_report import load_records, selection_summary
    repeated = tmp_path / 'results' / 'repeated'
    comparison = tmp_path / 'results' / 'comparison'
    repeated.mkdir(parents=True)
    comparison.mkdir(parents=True)
    stale = record('qwen-base')
    stale['feature_source'] = 'old-run'
    (repeated / 'qwen-base.json').write_text(json.dumps(stale))
    (comparison / 'qwen-base.json').write_text(json.dumps({'candidate': 'qwen-base', 'status': 'failed'}))
    records, unavailable = load_records(tmp_path)
    assert records == []
    assert unavailable[0]['manifest']['candidate'] == 'qwen-base'
    html = selection_summary(records, unavailable)
    assert 'Latest backbone run failed' in html
    assert 'No eligible candidate' in html


def test_report_loader_marks_missing_and_unrepeated_results_incomplete(tmp_path):
    import json
    from build_repeated_report import load_records
    repeated = tmp_path / 'results' / 'repeated'
    comparison = tmp_path / 'results' / 'comparison'
    repeated.mkdir(parents=True)
    comparison.mkdir(parents=True)
    stale = record('qwen-base')
    stale['feature_source'] = 'old-run'
    (repeated / 'qwen-base.json').write_text(json.dumps(stale))
    (comparison / 'qwen-base.json').write_text(json.dumps({'run_directory': 'new-run'}))
    records, unavailable = load_records(tmp_path)
    result = select([*records, *unavailable], ['qwen-base', 'missing'])
    assert not result['complete']
    assert result['leaders'] == []
    assert len(result['excluded']) == 2
