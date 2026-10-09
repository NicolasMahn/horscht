import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import build_project_record


def write_json(root: Path, relative_path: str, value: object) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')
    return path


def test_new_data_panel_separates_new_development_and_follows_recovery(tmp_path, monkeypatch):
    monkeypatch.setattr(build_project_record, 'ROOT', tmp_path)
    monkeypatch.setattr(build_project_record, 'SOURCES', {})
    run = build_project_record.NEW_DATA_RUN
    write_json(tmp_path, f'{run}/experiment-status.json', {'status': 'failed', 'phase': 'calibration'})
    write_json(tmp_path, f'{run}/resolved-main.json', {'output': f'{run}/main-recovery-001'})
    write_json(tmp_path, f'{run}/main-recovery-001/report.json', {
        'status': 'completed', 'best_update': 100, 'best_dev_soft_cross_entropy': .75,
    })
    body = build_project_record.new_data_panel()
    assert 'saved state: failed' in body
    assert 'different development set' in body
    assert 'Selected update 100' in body
    write_json(tmp_path, f'{run}/resolved-main.json', {'output': '../outside'})
    with pytest.raises(ValueError, match='outside its run directory'):
        build_project_record.new_data_panel()


def test_option_attention_panel_uses_saved_state_and_development_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(build_project_record, 'ROOT', tmp_path)
    monkeypatch.setattr(build_project_record, 'SOURCES', {})
    state, _, _, _ = build_project_record.current_attention_state()
    assert state == 'Not started'

    run = build_project_record.ATTENTION_RUN
    write_json(tmp_path, f'{run}/process.json', {'service': 'attention.service'})
    write_json(tmp_path, f'{run}/completion-status.json', {'phase': 'waiting'})
    write_json(tmp_path, f'{run}/progress.json', {
        'status': 'preparing', 'phase': 'initial development baseline',
        'global_update': 0, 'questions_seen': 0,
        'observed_at': '2026-10-06T08:00:00+00:00',
        'latest_development': {'questions_done': 2048, 'questions': 7202},
    })
    monkeypatch.setattr(build_project_record.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout='active'))

    state, meaning, gpu, body = build_project_record.current_attention_state()

    assert state == 'Preparing · initial development baseline'
    assert '2,048 / 7,202' in meaning
    assert 'no optimizer update yet' in meaning
    assert '2,048 / 7,202 questions in the current pass' in body
    assert 'Exploratory external suite:' not in body
    assert gpu == 'GPU owned by option-attention training or evaluation'


@pytest.mark.parametrize('completion_phase', ['completed', 'starting blind external evaluation', 'failed'])
def test_option_attention_aggregate_requires_completed_training_and_matching_hash(tmp_path, monkeypatch, completion_phase):
    monkeypatch.setattr(build_project_record, 'ROOT', tmp_path)
    monkeypatch.setattr(build_project_record, 'SOURCES', {})
    monkeypatch.setattr(build_project_record.subprocess, 'run',
                        lambda *args, **kwargs: SimpleNamespace(stdout='inactive'))
    run = build_project_record.ATTENTION_RUN
    output = f'{run}/recovery-2026-10-06/attention-last-block'
    write_json(tmp_path, f'{run}/process.json', {
        'service': 'attention.service', 'training_output': output,
    })
    metrics = {
        'overall': {
            'examples': 2, 'correct': 1, 'accuracy': 0.5,
            'negative_log_likelihood': 0.7, 'multiclass_brier_score': 0.5,
        },
        'by_source_slice': {
            'fixture': {
                'examples': 2, 'correct': 1, 'accuracy': 0.5,
                'negative_log_likelihood': 0.7, 'multiclass_brier_score': 0.5,
            },
        },
    }
    aggregate_path = write_json(tmp_path, (
        f'{build_project_record.BENCHMARK_ROOT}/current-corpus-option-attention-2026-10-06/'
        'blind-aggregate-result.json'), {'candidate_results': {'last-block': metrics}})
    write_json(tmp_path, f'{output}/report.json', {
        'status': 'completed', 'selected_checkpoint_reload_verified': True,
        'best_update': 512, 'best_dev_soft_cross_entropy': 0.6,
    })
    write_json(tmp_path, f'{run}/completion-status.json', {'phase': completion_phase})
    write_json(tmp_path, f'{run}/benchmark-report.json', {
        'metrics': metrics,
        'aggregate_path': aggregate_path.relative_to(tmp_path).as_posix(),
        'aggregate_sha256': hashlib.sha256(aggregate_path.read_bytes()).hexdigest(),
    })

    state, meaning, _, body = build_project_record.current_attention_state()

    assert state == 'Training and external evaluation completed'
    assert '1 / 2 aggregate items' in meaning
    assert '50.00% accuracy (1 / 2)' in body
    assert 'last planned external test for the old training corpus' in body
    assert f'{output}/report.json' in body

    write_json(tmp_path, aggregate_path.relative_to(tmp_path).as_posix(), {
        'candidate_results': {'last-block': {**metrics, 'overall': {**metrics['overall'], 'correct': 2}}}
    })
    monkeypatch.setattr(build_project_record, 'SOURCES', {})
    with pytest.raises(ValueError, match='aggregate differs from its report binding'):
        build_project_record.current_attention_state()


def test_option_attention_output_pointer_cannot_escape_run_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(build_project_record, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='must remain inside its run directory'):
        build_project_record.attention_training_output({
            'training_output': f'{build_project_record.ATTENTION_RUN}/../../outside/attention-last-block',
        })


def test_dashboard_reads_resumed_attention_history_from_process_pointer(tmp_path, monkeypatch):
    monkeypatch.setattr(build_project_record, 'ROOT', tmp_path)
    monkeypatch.setattr(build_project_record, 'SOURCES', {})
    monkeypatch.setattr(build_project_record, 'matched_arms', lambda: {})
    full_history = [{'update': 8017, 'dev': {'hard_accuracy': 0.7, 'soft_cross_entropy': 0.5}}]
    full_report = {
        'status': 'completed', 'selected_checkpoint_reload_verified': True,
        'best_update': 8017, 'selected_checkpoint_development': full_history[0]['dev'],
    }
    monkeypatch.setattr(build_project_record, 'load_training',
                        lambda: (full_history, full_report, {}, {}, {}))
    output = f'{build_project_record.ATTENTION_RUN}/recovery-2026-10-06/attention-last-block'
    write_json(tmp_path, f'{build_project_record.ATTENTION_RUN}/process.json', {
        'training_output': output,
    })
    point = {'hard_accuracy': 0.4, 'soft_cross_entropy': 0.9}
    write_json(tmp_path, f'{output}/history.json', [
        {'update': 1024, 'dev': point},
    ])

    runs = build_project_record.dashboard_runs()

    resumed = next(run for run in runs if run['key'] == 'option-attention')
    assert resumed['dev'] == [(1024, point)]
    assert resumed['kept'] is None
