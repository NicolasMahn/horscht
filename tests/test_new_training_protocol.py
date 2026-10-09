import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/new-data-last-block-2026-10-08'


def load_script(name):
    spec = importlib.util.spec_from_file_location('new_protocol_' + name, ROOT / ('scripts/' + name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_container_paths_resolve_same_project_artifact(monkeypatch):
    driver = load_script('run_experiment')
    smoke = load_script('smoke_longest_context')
    host = ROOT / 'data/training_snapshots/frozen/rows.jsonl'
    assert driver.workspace(host) == '/workspace/data/training_snapshots/frozen/rows.jsonl'
    monkeypatch.setattr(smoke, 'ROOT', Path('/workspace'))
    assert smoke.project_path(str(host), {'project_root':str(ROOT)}) == Path(driver.workspace(host))
    with pytest.raises(ValueError):
        driver.workspace(ROOT.parent / 'outside.jsonl')


def test_all_stages_run_memory_smoke_before_training_and_finish_after_main(tmp_path, monkeypatch):
    driver = load_script('run_experiment')
    calls = []
    monkeypatch.setattr(driver, 'verify_protocol', lambda path: {'seed':20261008})
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    (tmp_path / 'runs').mkdir()
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda *args, **kwargs: '')
    monkeypatch.setattr(driver, 'run_evaluation', lambda *args, **kwargs: calls.append(args[2]))
    monkeypatch.setattr(driver, 'train_arm', lambda *args, **kwargs: calls.append(args[2]))
    monkeypatch.setattr(driver, 'choose_rate', lambda *args: 1e-4)
    monkeypatch.setattr(driver, 'finish', lambda *args: calls.append('finish'))
    monkeypatch.setattr(driver, 'rebuild_report', lambda *args: None)
    run_dir = tmp_path / 'runs/new'
    run_dir.mkdir()
    (run_dir / 'protocol.json').write_text('{}')
    driver.run(run_dir, 'all')
    assert calls == ['container-preflight','full-context-smoke','pilot-lr-1e-4','pilot-lr-1e-3','main','finish']


def test_recovery_all_reuses_smoke_and_completed_pilots(tmp_path, monkeypatch):
    driver = load_script('run_experiment')
    driver.RECOVERY = {'revision': 'recovery-001'}
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    monkeypatch.setattr(driver, 'verify_protocol', lambda path: {})
    monkeypatch.setattr(driver.subprocess, 'check_output', lambda *args, **kwargs: '')
    monkeypatch.setattr(driver, 'run_evaluation', lambda *args, **kwargs: pytest.fail('Saved smoke must be reused'))
    calls = []
    monkeypatch.setattr(driver, 'train_arm', lambda *args, **kwargs: calls.append(args[2]))
    monkeypatch.setattr(driver, 'choose_rate', lambda *args: 1e-4)
    monkeypatch.setattr(driver, 'finish', lambda *args: calls.append('finish'))
    monkeypatch.setattr(driver, 'rebuild_report', lambda *args: None)
    (tmp_path / 'runs').mkdir()
    run = tmp_path / 'runs/new'
    run.mkdir()
    (run / 'protocol.json').write_text('{}')
    driver.run(run, 'all')
    assert calls == ['pilot-lr-1e-4', 'pilot-lr-1e-3', 'main', 'finish']


@pytest.mark.parametrize('defer', [False, True])
def test_recovery_main_command_preserves_training_recipe_and_adds_audited_durability(tmp_path, monkeypatch, defer):
    import json
    from types import SimpleNamespace
    driver = load_script('run_experiment')
    monkeypatch.setattr(driver, 'ROOT', tmp_path)
    run = tmp_path / 'runs/new'
    source = run / 'main'
    checkpoint = source / 'checkpoints/initial'
    checkpoint.mkdir(parents=True)
    (checkpoint / 'checkpoint.json').write_text('{}')
    (source / 'latest-checkpoint.json').write_text(json.dumps({'path': 'checkpoints/initial'}))
    (run / 'protocol.json').write_text('{}')
    (run / 'full-context-smoke.json').write_text(json.dumps({
        'protocol_sha256': driver.sha256(run / 'protocol.json'), 'backward_completed': True,
        'optimizer_updates': 0, 'compact_checkpoint_reload_verified': True, 'complete_input_tokens': 37609}))
    driver.RECOVERY = {'revision': 'recovery-001', 'engine_path': 'runs/new/recovery-001/scalable_training.py',
                       'lifecycle_audit_path': 'runs/new/recovery-001/lifecycle-audit.json',
                       'resume_checkpoint_manifest_sha256': driver.sha256(checkpoint / 'checkpoint.json'),
                       'checkpoint_every_updates': 8, 'checkpoint_seconds': 300}
    if defer:
        driver.RECOVERY.update(development_deferral=True, input_path='runs/new/recovery-001/paired_training.py',
                               runtime_environment={'HSA_CU_MASK':'0:0-23'}, run_generation='attempt-003',
                               launch_attempt_path='runs/new/attempt.json')
        (run / 'attempt.json').write_text(json.dumps({'expected_output':'runs/new/main-recovery-001'}))
        monkeypatch.setenv('INVOCATION_ID', 'invocation-003')
    protocol = {'main': {'updates':1438, 'fast_every_updates':64, 'full_every_updates':360,
                        'fast_development_questions':128}, 'seed':20261008, 'max_tokens':40960,
                'max_batch_tokens':40960, 'snapshot':str(tmp_path / 'snapshot'),
                'system_test_lock':str(tmp_path / 'lock'), 'max_complete_input_tokens':{'train':37609}}
    monkeypatch.setattr(driver, 'stage_output', lambda *args: source)
    monkeypatch.setattr(driver, 'rebuild_report', lambda *args: None)
    commands = []
    def popen(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(pid=100, wait=lambda:139)
    monkeypatch.setattr(driver.subprocess, 'Popen', popen)
    with pytest.raises(RuntimeError, match='exited 139'):
        driver.train_arm(run, protocol, 'main', 1e-4, pilot=False)
    cmd = commands[0]
    def option(name):
        return cmd[cmd.index(name) + 1]
    assert option('--checkpoint-every') == '8'
    assert option('--checkpoint-seconds') == '300'
    assert option('--resume') == '/workspace/runs/new/main/checkpoints/initial'
    assert option('--resume-lifecycle-audit') == '/workspace/runs/new/recovery-001/lifecycle-audit.json'
    assert option('--full-dev-every') == '360' and option('--eval-every') == '64'
    assert option('--head-learning-rate') == '0.0001'
    assert option('--accumulation') == '16' and option('--warmup-fraction') == '0.03'
    assert '--warm-start' not in cmd
    assert ('--defer-full-development' in cmd) is defer
    assert ('HSA_CU_MASK=0:0-23' in cmd) is defer
    if defer:
        receipt = json.loads((run / 'main-active.json').read_text())
        assert receipt['run_generation'] == 'attempt-003'
        assert receipt['systemd_invocation_id'] == 'invocation-003'
        assert receipt['output'] == 'runs/new/main-recovery-001'
    if defer:
        assert str(tmp_path / driver.RECOVERY['input_path']) + ':/workspace/paired_training.py:ro' in cmd
    assert str(tmp_path / driver.RECOVERY['engine_path']) + ':/workspace/scalable_training.py:ro' in cmd
    assert (run / 'main-recovery-001-command.json').exists()
    assert not (run / 'main-command.json').exists()


def test_crash_boundary_binds_one_lost_update_without_fabricating_stop():
    prep = load_script('prepare_deferred_transition')
    cursor = {'epoch':0,'global_update':64,'next_update':64,'questions_seen':1024,'tokens_seen':1266017}
    status = {'output':'/workspace/runs/new/main-recovery-001','global_update':65,'next_update':65,
              'questions_seen':1040,'tokens_seen':1285279,'training_running':True,
              'updated_at':'2026-10-08T18:56:01+00:00'}
    exit_receipt = {'exit_code':139,'completed_at':'2026-10-08T18:58:04+00:00'}
    active = {'output':'runs/new/main-recovery-001'}
    result = prep.verify_crash_boundary(status,cursor,exit_receipt,active,active['output'])
    assert result['lost_completed_updates'] == [65]
    assert result['updates_replayed_after_restore'] == [65]
    assert status['training_running'] is True
    for bad_status, bad_exit, bad_active in [
        ({**status,'global_update':66},exit_receipt,active),
        (status,{**exit_receipt,'exit_code':0},active),
        (status,exit_receipt,{'output':'runs/other/main'}),
        ({**status,'updated_at':'2026-10-08T19:00:00+00:00'},exit_receipt,active),
    ]:
        with pytest.raises(ValueError,match='Crash evidence'):
            prep.verify_crash_boundary(bad_status,cursor,bad_exit,bad_active,active['output'])
