import copy
import json
from pathlib import Path

import pytest
import torch

import scalable_training as engine
from test_scalable_training_lifecycle import (
    _args, _assert_tree_equal, _checkpoint_for_report, _install_toy_transformers, _snapshot,
)


def bind_baseline(root, source, checkpoint):
    protocol = root / 'protocol.json'
    selection = root / 'selection.json'
    protocol.write_text(json.dumps({'selection_sha256': engine.sha256(selection)}))
    return {'artifacts': {name: engine.sha256(source / name) for name in
            ('baselines.json', 'history.json', 'development-protocol.json', 'manifest.json')},
            'protocol_path': 'protocol.json', 'protocol_sha256': engine.sha256(protocol),
            'selection_sha256': engine.sha256(selection)}


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, 'ROOT', tmp_path)
    source = tmp_path / 'source'
    checkpoint = source / 'checkpoints/initial'
    checkpoint.mkdir(parents=True)
    selection = tmp_path / 'selection.json'
    selection.write_text('{}')
    metrics = {'questions': 2, 'soft_cross_entropy': 1.25, 'kl_pq': .5,
               'categorical_brier_sum': .2, 'argmax_agreement': .5}
    dev = {'full_ids': ['a', 'b'], 'full_every_updates': 360, 'fast_ids': ['a']}
    manifest = {'selection_sha256': engine.sha256(selection)}
    progress = {'cursor': dict(epoch=0, global_update=0, next_update=0, questions_seen=0, tokens_seen=0),
                'reason': 'initial baseline', 'best_update': 0, 'epoch_plan_sha256': None,
                'best_loss': metrics['soft_cross_entropy']}
    restored = {'manifest': manifest, 'progress': progress}
    (checkpoint / 'checkpoint.json').write_text(json.dumps(restored))
    (source / 'manifest.json').write_text(json.dumps(manifest))
    (source / 'baselines.json').write_text(json.dumps({'warm_start_development': metrics,
                                                      'uniform_development': metrics}))
    (source / 'history.json').write_text(json.dumps([{'update': 0, 'dev': metrics}]))
    (source / 'development-protocol.json').write_text(json.dumps(dev))
    audit = {'record': {'baseline_reuse': bind_baseline(tmp_path, source, checkpoint)}}
    return checkpoint, restored, audit, dev, selection


def test_reuse_is_opt_in_and_does_not_consume_rng(fixture):
    checkpoint, restored, audit, dev, selection = fixture
    state = torch.get_rng_state().clone()
    assert engine.reuse_completed_initial_baseline(checkpoint, restored, None, dev, selection) is None
    result = engine.reuse_completed_initial_baseline(*fixture)
    assert result['history'][0]['dev']['questions'] == 2
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize('change', ['cursor', 'reason', 'metrics', 'history', 'protocol', 'selection', 'manifest', 'missing'])
def test_reuse_rejects_inexact_or_incomplete_saved_baseline(fixture, change):
    checkpoint, restored, audit, dev, selection = fixture
    source = checkpoint.parent.parent
    if change == 'cursor':
        restored['progress']['cursor']['global_update'] = 1
    elif change == 'reason':
        restored['progress']['reason'] = 'periodic'
    elif change == 'metrics':
        data = json.loads((source / 'baselines.json').read_text())
        del data['warm_start_development']['kl_pq']
        (source / 'baselines.json').write_text(json.dumps(data))
        audit['record']['baseline_reuse']['artifacts']['baselines.json'] = engine.sha256(source / 'baselines.json')
    elif change == 'history':
        (source / 'history.json').write_text('[]')
        audit['record']['baseline_reuse']['artifacts']['history.json'] = engine.sha256(source / 'history.json')
    elif change == 'protocol':
        dev['full_ids'].reverse()
    elif change == 'selection':
        selection.write_text('{"different":true}')
    elif change == 'manifest':
        restored['manifest']['new_field'] = True
    else:
        (source / 'baselines.json').unlink()
    with pytest.raises(ValueError, match='Baseline reuse'):
        engine.reuse_completed_initial_baseline(*fixture)


def test_exact_recovery_skips_baseline_and_matches_uninterrupted_training(tmp_path, monkeypatch):
    harness = {'markers': {}, 'tokenizers': [], 'models': [], 'initial_model_states': [],
               'model_rows': [], 'model_calls': []}
    snapshot, lock, selection, ids, markers = _snapshot(tmp_path)
    harness['markers'].update(markers)
    _install_toy_transformers(monkeypatch, harness)
    monkeypatch.setattr(engine, 'require_training_snapshot', lambda path: None)
    monkeypatch.setattr(engine, 'require_memory', lambda *args: None)
    monkeypatch.setattr(engine, 'write_curves', lambda *args: None)

    def args(name, resume=None):
        value = _args(tmp_path, 'last-block', snapshot, lock, selection, output_name=name, resume=resume)
        value.order_consistency_questions = 0
        value.fast_dev_questions = 0
        value.full_dev_every = 8
        value.lr_schedule = 'warmup-cosine'
        return value

    uninterrupted = args('uninterrupted')
    assert engine.run_scalable_arm(uninterrupted) == 0
    real_run = engine.run_epoch_updates
    monkeypatch.setattr(engine, 'run_epoch_updates', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('GPU-reset fixture')))
    failed = args('failed')
    with pytest.raises(RuntimeError, match='GPU-reset fixture'):
        engine.run_scalable_arm(failed)
    checkpoint = failed.output / json.loads((failed.output / 'latest-checkpoint.json').read_text())['path']
    previous = json.loads((checkpoint / 'checkpoint.json').read_text())['manifest']
    monkeypatch.setattr(engine, 'run_epoch_updates', real_run)
    source_hash = previous['runner_sha256']
    source_copy = tmp_path / 'engine-source.py'
    source_copy.write_bytes(Path(engine.__file__).read_bytes())
    monkeypatch.setattr(engine, 'ROOT', tmp_path)
    # All other source hashes still resolve to the real frozen fixture sources.
    real_sha = engine.sha256
    monkeypatch.setattr(engine, 'sha256', lambda path: real_sha(Path(engine.__file__).parent / Path(path).name)
                        if Path(path).parent == tmp_path and Path(path).name.endswith('.py')
                        and not Path(path).exists() else real_sha(path))
    binding = bind_baseline(tmp_path, failed.output, checkpoint)
    audit = {'schema_version': 'training-resource-lifecycle-audit-v1', 'purpose': 'resource-lifecycle-only',
             'checkpoint_manifest_sha256': engine.sha256(checkpoint / 'checkpoint.json'),
             'previous_run_manifest_sha256': engine.sha256(failed.output / 'manifest.json'),
             'source_changes': {'runner_sha256': {'previous': source_hash, 'current': source_hash}},
             'previous_source_files': {'runner_sha256': 'engine-source.py'}, 'baseline_reuse': binding}
    audit_path = tmp_path / 'audit.json'
    audit_path.write_text(json.dumps(audit))
    resumed = args('recovery', checkpoint)
    resumed.resume_lifecycle_audit = audit_path
    calls = []
    real_eval = engine.evaluate_batches
    monkeypatch.setattr(engine, 'evaluate_batches', lambda *args, **kwargs: (calls.append(args[3]) or real_eval(*args, **kwargs)))
    assert engine.run_scalable_arm(resumed) == 0
    # Epoch-end checks and selected-checkpoint reload verification remain scheduled;
    # the initial baseline forward pass is absent.
    assert len(calls) == 3
    assert json.loads((resumed.output / 'baseline-reuse.json').read_text())['inference_questions'] == 0
    final = _checkpoint_for_report(uninterrupted.output, 'last_checkpoint_relative')
    recovered = _checkpoint_for_report(resumed.output, 'last_checkpoint_relative')
    for name in ('backbone.pt', 'head.pt', 'optimizer.pt', 'scheduler.pt', 'torch_rng.pt'):
        _assert_tree_equal(torch.load(final / name, weights_only=True), torch.load(recovered / name, weights_only=True))
