import json
import re

from build_current_training_report import build, load, merge_history


def test_resume_replaces_rolled_back_work_and_coalesces_evaluation():
    original = [
        {'update': 0, 'dev': {'loss': 1}},
        {'update': 1, 'train_loss': .8},
        {'update': 2, 'train_loss': .7},
        {'update': 3, 'train_loss': .1, 'superseded': True},
    ]
    recovery = [{'update': 2, 'dev': {'loss': .6}}, {'update': 3, 'train_loss': .5}]
    merged = merge_history(original, recovery, 2)
    assert [p['update'] for p in merged] == [0, 1, 2, 3]
    assert merged[2] == {'update': 2, 'train_loss': .7, 'dev': {'loss': .6}}
    assert merged[3] == {'update': 3, 'train_loss': .5}


def test_completed_report_has_unique_curves_and_correct_denominators():
    data = load()
    history, report, status, config, resume = data
    assert len(history) == 8018
    assert sum('train_loss' in p for p in history) == 8017
    assert sum('dev' in p for p in history) == 18
    assert sum(p['update'] == resume for p in history) == 1
    assert history[-1]['questions_seen'] == 128263
    selected = report['selected_checkpoint_development']
    assert round(selected['hard_accuracy'] * 2740) == 2489
    assert selected['task_grounded_idk']['answerable']['questions'] == 3102
    assert selected['task_grounded_idk']['unanswerable']['questions'] == 748
    assert round(selected['task_grounded_idk']['answerable']['predicted_idk_fraction'] * 3102) == 11
    assert round(selected['task_grounded_idk']['unanswerable']['predicted_idk_fraction'] * 748) == 709
    html = build(*data)
    payload = json.loads(re.search(r'<script type="application/json" id="measurements">(.*?)</script>', html, re.S)[1])
    assert len(payload['history']) == 8018
    assert html.count('<svg ') == 4
    assert 'Development loss · full range</summary><svg' in html
    assert '<strong>not been run</strong>' in html
    assert '90.8%' in html and '32.7%' in html
    assert '0.4281' in html and '0.4393' in html
    assert not re.search(r'<(?:script|link)[^>]+(?:src|href)="https?://', html)


def test_verified_external_aggregates_replace_pending_status():
    from build_current_training_report import load_benchmark, validate_benchmark
    benchmark = load_benchmark()
    validate_benchmark(benchmark)
    html = build(*load(), benchmark=benchmark)
    assert '40.19%' in html and '45.57%' in html
    assert '-5.38 percentage points' in html
    assert '7,184 correct now versus 8,146' in html
    assert '1.7522' in html and '0.7245' in html
    assert 'not a causal test' in html and 'suite was reused' in html
    assert 'No fresh exact/lexical overlap audit' in html
    assert 'independent semantic problems' in html
    assert 'separate internal held-out test has not been run' in html
    assert 'Independent benchmarks and the held-out test have <strong>not been run</strong>' not in html
    assert 'No prior external benchmark result is reused here.' not in html


def test_external_counts_cannot_silently_disagree():
    import copy
    import pytest
    from build_current_training_report import load_benchmark, validate_benchmark
    benchmark = copy.deepcopy(load_benchmark())
    benchmark['metrics']['overall']['correct'] += 1
    with pytest.raises(ValueError, match='slice totals'):
        validate_benchmark(benchmark)


def test_original_qwen_is_distinct_exploratory_end_to_end_comparison():
    from build_current_training_report import load_benchmark, load_original_benchmark
    html = build(*load(), benchmark=load_benchmark(), original_benchmark=load_original_benchmark())
    assert '51.72%' in html and '9,245 / 17,874 correct' in html
    assert '+11.53 percentage points' in html
    assert '6.0296' in html and '0.8843' in html
    assert 'different prompt wrapper, choice labels and readout' in html
    assert 'does not isolate the effect of backbone training' in html
    assert 'post hoc diagnostics on a reused suite' in html
    assert 'results-before-original-qwen-2026-10-04.html' in html


def test_original_qwen_rejects_mismatched_suite():
    import copy
    import pytest
    from build_current_training_report import load_benchmark, load_original_benchmark, validate_original_benchmark
    original = copy.deepcopy(load_original_benchmark())
    original['suite_reuse']['canonical_items_sha256'] = 'different'
    with pytest.raises(ValueError, match='different suite'):
        validate_original_benchmark(original, load_benchmark())
