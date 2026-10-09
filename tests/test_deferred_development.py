import json
from pathlib import Path

import pytest
import torch

import scalable_training as engine
from test_initial_baseline_recovery import bind_baseline
from test_scalable_training_lifecycle import (
    _args, _assert_tree_equal, _checkpoint_for_report, _install_toy_transformers, _snapshot,
)


def setup(tmp_path, monkeypatch):
    harness = {'markers': {}, 'tokenizers': [], 'models': [], 'initial_model_states': [],
               'model_rows': [], 'model_calls': []}
    snapshot, lock, selection, ids, markers = _snapshot(tmp_path)
    harness['markers'].update(markers)
    _install_toy_transformers(monkeypatch, harness)
    monkeypatch.setattr(engine, 'require_training_snapshot', lambda path: None)
    monkeypatch.setattr(engine, 'require_memory', lambda *args: None)
    monkeypatch.setattr(engine, 'write_curves', lambda *args: None)
    original = Path(engine.__file__).parent
    for name in ['training_loop.py','training_checkpoint.py','paired_training.py','experiment.py','attention_head.py','probability_experiment.py']:
        (tmp_path / name).write_bytes((original / name).read_bytes())
    monkeypatch.setattr(engine, 'ROOT', tmp_path)
    def args(name, *, defer=False, resume=None):
        value = _args(tmp_path, 'last-block', snapshot, lock, selection, output_name=name, resume=resume)
        value.epochs = 1
        value.order_consistency_questions = 0
        value.fast_dev_questions = 0
        value.full_dev_every = 2
        value.lr_schedule = 'warmup-cosine'
        value.defer_full_development = defer
        return value
    return harness, args, selection


def test_deferred_scoring_preserves_candidates_selection_and_final_optimization(tmp_path, monkeypatch):
    harness, args, selection = setup(tmp_path, monkeypatch)
    reference = args('reference')
    assert engine.run_scalable_arm(reference) == 0
    expected = {p['update']:p['dev'] for p in json.loads((reference.output / 'history.json').read_text()) if 'dev' in p}
    real_eval = engine.evaluate_batches
    train_forward_counts = []
    def evaluate(*values, **kwargs):
        train_forward_counts.append(sum(call['grad_enabled'] for call in harness['model_calls']))
        return real_eval(*values, **kwargs)
    monkeypatch.setattr(engine, 'evaluate_batches', evaluate)
    calls_before = sum(call['grad_enabled'] for call in harness['model_calls'])
    deferred = args('deferred', defer=True)
    deferred.retain_checkpoints = 1
    assert engine.run_scalable_arm(deferred) == 0
    assert [count - calls_before for count in train_forward_counts] == [0, 4, 4]
    index = json.loads((deferred.output / 'deferred-candidates.json').read_text())
    assert set(index) == {'0','2','4'}
    for update, entry in index.items():
        assert (tmp_path / entry['path'] / 'checkpoint.json').is_file()
        assert entry['dev'] == expected[int(update)]
    ref_report = json.loads((reference.output / 'report.json').read_text())
    def_report = json.loads((deferred.output / 'report.json').read_text())
    assert def_report['best_update'] == ref_report['best_update']
    assert def_report['best_dev_soft_cross_entropy'] == ref_report['best_dev_soft_cross_entropy']
    for name in ('backbone.pt','head.pt','optimizer.pt','scheduler.pt','torch_rng.pt'):
        _assert_tree_equal(torch.load(_checkpoint_for_report(reference.output,'last_checkpoint_relative') / name, weights_only=True),
                          torch.load(_checkpoint_for_report(deferred.output,'last_checkpoint_relative') / name, weights_only=True))
    assert json.loads((deferred.output / 'deferred-selection.json').read_text())['duplicate_selected_development_inference'] is False


def test_nonzero_resume_reuses_baseline_and_continues_without_replaying_updates(tmp_path, monkeypatch):
    harness, args, selection = setup(tmp_path, monkeypatch)
    reference = args('uninterrupted')
    assert engine.run_scalable_arm(reference) == 0
    real_run = engine.run_epoch_updates
    def interrupt(plan, cursor, *, stop, step, observe, before_step=None):
        def observed(progress, metrics):
            observe(progress, metrics)
            stop.request('transition boundary')
        return real_run(plan,cursor,stop=stop,step=step,observe=observed,before_step=before_step)
    monkeypatch.setattr(engine,'run_epoch_updates',interrupt)
    old = args('old')
    assert engine.run_scalable_arm(old) == 130
    checkpoint = _checkpoint_for_report(old.output,'last_checkpoint_relative')
    restored = json.loads((checkpoint / 'checkpoint.json').read_text())
    binding = bind_baseline(tmp_path,old.output,checkpoint)
    policy = {'timing':'after all weight updates','candidate_updates':[0,2,4],
        'selection':'minimum full-development soft cross entropy; earliest update breaks ties',
        'baseline':'reuse audited complete initial development metrics',
        'selected_reload':'verified artifact checksums and tensor load; no duplicate full-development scoring',
        'fast_monitoring':'unchanged; never selects checkpoints'}
    source = tmp_path / 'old-engine.py'
    source.write_bytes(Path(engine.__file__).read_bytes())
    audit = {'schema_version':'training-resource-lifecycle-audit-v1','purpose':'development-evaluation-deferral',
        'development_execution':policy, 'checkpoint_manifest_sha256':engine.sha256(checkpoint / 'checkpoint.json'),
        'previous_run_manifest_sha256':engine.sha256(old.output / 'manifest.json'),
        'source_changes':{'runner_sha256':{'previous':restored['manifest']['runner_sha256'],'current':engine.sha256(source)}},
        'previous_source_files':{'runner_sha256':'old-engine.py'},'development_state_reuse':binding}
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps(audit))
    monkeypatch.setattr(engine,'run_epoch_updates',real_run)
    counts = []
    prior_training_calls = sum(call['grad_enabled'] for call in harness['model_calls'])
    real_eval = engine.evaluate_batches
    def evaluate(*values, **kwargs):
        counts.append(sum(call['grad_enabled'] for call in harness['model_calls']))
        return real_eval(*values, **kwargs)
    monkeypatch.setattr(engine,'evaluate_batches',evaluate)
    resumed = args('new',defer=True,resume=checkpoint)
    resumed.resume_lifecycle_audit = audit_path
    assert engine.run_scalable_arm(resumed) == 0
    assert counts == [prior_training_calls + 3] * 2
    report = json.loads((resumed.output / 'report.json').read_text())
    assert report['cursor']['global_update'] == 4
    assert report['training_questions'] == 4
    for name in ('backbone.pt','head.pt','optimizer.pt','scheduler.pt','torch_rng.pt'):
        _assert_tree_equal(torch.load(_checkpoint_for_report(reference.output,'last_checkpoint_relative') / name,weights_only=True),
                          torch.load(_checkpoint_for_report(resumed.output,'last_checkpoint_relative') / name,weights_only=True))


def test_deferred_mode_rejects_early_stopping(tmp_path, monkeypatch):
    _, args, _ = setup(tmp_path,monkeypatch)
    value = args('invalid',defer=True)
    value.early_stop_patience = 2
    with pytest.raises(ValueError,match='no early stopping'):
        engine.run_scalable_arm(value)


def test_interrupt_during_post_scoring_resumes_without_training_or_rescoring_complete_candidates(tmp_path, monkeypatch):
    _, args, selection = setup(tmp_path,monkeypatch)
    original_eval = engine.evaluate_batches
    calls = []
    def interrupt_final(*values, **kwargs):
        calls.append('full')
        if len(calls) == 3:
            values[0].harness['post_stopped'] = True
            raise InterruptedError('stop during final candidate')
        return original_eval(*values, **kwargs)
    monkeypatch.setattr(engine,'evaluate_batches',interrupt_final)
    first = args('first',defer=True)
    assert engine.run_scalable_arm(first) == 130
    checkpoint = _checkpoint_for_report(first.output,'last_checkpoint_relative')
    restored = json.loads((checkpoint / 'checkpoint.json').read_text())
    index = json.loads((first.output / 'deferred-candidates.json').read_text())
    assert 'dev' in index['2'] and 'dev' not in index['4']
    assert restored['progress']['cursor']['global_update'] == 4
    binding = bind_baseline(tmp_path,first.output,checkpoint)
    binding['artifacts']['deferred-candidates.json'] = engine.sha256(first.output / 'deferred-candidates.json')
    source = tmp_path / 'source-engine.py'
    source.write_bytes(Path(engine.__file__).read_bytes())
    audit = {'schema_version':'training-resource-lifecycle-audit-v1','purpose':'development-evaluation-deferral',
        'development_execution':restored['manifest']['development_execution'],
        'checkpoint_manifest_sha256':engine.sha256(checkpoint / 'checkpoint.json'),
        'previous_run_manifest_sha256':engine.sha256(first.output / 'manifest.json'),
        'source_changes':{'runner_sha256':{'previous':restored['manifest']['runner_sha256'],'current':engine.sha256(source)}},
        'previous_source_files':{'runner_sha256':'source-engine.py'},'development_state_reuse':binding}
    audit_path = tmp_path / 'post-audit.json'
    audit_path.write_text(json.dumps(audit))
    calls.clear()
    monkeypatch.setattr(engine,'evaluate_batches', lambda *values, **kwargs: (calls.append('full') or original_eval(*values,**kwargs)))
    monkeypatch.setattr(engine,'run_epoch_updates',lambda *values, **kwargs: pytest.fail('All updates are already complete'))
    second = args('second',defer=True,resume=checkpoint)
    second.resume_lifecycle_audit = audit_path
    assert engine.run_scalable_arm(second) == 0
    assert calls == ['full']
    assert json.loads((second.output / 'report.json').read_text())['cursor'] == restored['progress']['cursor']
