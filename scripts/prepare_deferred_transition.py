#!/usr/bin/env python3
"""Seal code before GPU stop, then bind the exact saved boundary with explicit crash evidence."""
from datetime import datetime, timezone
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from paired_training import sha256
from training_loop import atomic_json
from training_checkpoint import _verify_artifacts

RUN = ROOT / "runs/new-data-last-block-2026-10-08"
DEST = RUN / 'recovery-003'
def verify_crash_boundary(status, cursor, process_exit, active, source_relative):
    if (process_exit.get('exit_code') != 139 or not process_exit.get('completed_at')
            or active.get('output') != source_relative or cursor.get('global_update') != 64
            or cursor.get('next_update') != 64 or cursor.get('epoch') != 0
            or status.get('global_update') != 65 or status.get('next_update') != 65
            or status.get('questions_seen') != cursor['questions_seen'] + 16
            or status.get('tokens_seen', 0) <= cursor['tokens_seen']
            or status.get('output') != '/workspace/' + source_relative
            or datetime.fromisoformat(status['updated_at']) >= datetime.fromisoformat(process_exit['completed_at'])):
        raise ValueError('Crash evidence does not bind the saved64/completed65 boundary')
    return {'durable_update':64, 'last_observed_completed_update':65,
            'lost_completed_updates':[65], 'updates_replayed_after_restore':[65],
            'cause':'GPU hang/reset; initiating cause not established',
            'source_exit_code':139, 'source_completed_at':process_exit['completed_at']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal-code',action='store_true')
    parser.add_argument('--bind-boundary',action='store_true')
    args = parser.parse_args()
    protocol = json.loads((RUN / 'protocol.json').read_text())
    policy = {'timing':'after all weight updates','candidate_updates':[0,360,720,1080,1438],
              'selection':'minimum full-development soft cross entropy; earliest update breaks ties',
              'baseline':'reuse audited complete initial development metrics',
              'selected_reload':'verified artifact checksums and tensor load; no duplicate full-development scoring',
              'fast_monitoring':'unchanged; never selects checkpoints'}
    if args.seal_code:
        if DEST.exists():
            raise FileExistsError('Code revisions are immutable')
        DEST.mkdir()
        source_files = {'engine_path':'scalable_training.py','input_path':'paired_training.py',
                        'evaluation_path':'evaluate_new_training.py',
                        'snapshot_protocol_path':'evaluation/snapshot_bound_protocol.py',
                        'blind_runner_path':'evaluation/blind_runner.py'}
        for filename in source_files.values():
            target = DEST / filename
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT / filename,target)
        (DEST / 'run_experiment.py').write_text((RUN / 'run_experiment.py').read_text().replace(
            'ROOT = Path(__file__).resolve().parents[1]','ROOT = Path(__file__).resolve().parents[3]'))
        (DEST / 'prepare_deferred_transition.py').write_text(Path(__file__).read_text().replace(
            'ROOT = Path(__file__).resolve().parents[1]', 'ROOT = Path(__file__).resolve().parents[3]').replace(
            'RUN = ROOT / "runs/new-data-last-block-2026-10-08"', 'RUN = ROOT / "runs/new-data-last-block-2026-10-08"s[1]'))
        (DEST / 'run_pipeline.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            f'cd {ROOT}\nexec {ROOT}/.venv/bin/python -u {DEST}/run_experiment.py '
            f'--run-dir {RUN} --operational-recovery {DEST}/operational-recovery.json --stage all '
            f'>> {DEST}/driver.log 2>&1\n')
        hashes = {p.relative_to(ROOT).as_posix():sha256(p) for p in DEST.rglob('*') if p.is_file()}
        atomic_json(DEST / 'code-freeze.json', {'schema_version':'development-deferral-code-freeze-v1',
            'protocol_sha256':sha256(RUN / 'protocol.json'), 'development_execution':policy,
            'source_paths':{field:(DEST / filename).relative_to(ROOT).as_posix() for field,filename in source_files.items()},
            'code_hashes':hashes,'created_at':datetime.now(timezone.utc).isoformat(),
            'active_gpu_job_mutated':False})
        print(json.dumps({'status':'code_frozen_ready_for_root_boundary_stop','path':str(DEST),'gpu_job_launched':False}),flush=True)
    if args.bind_boundary:
        if (DEST / 'operational-recovery.json').exists():
            raise FileExistsError('Boundary receipts are immutable')
        freeze = json.loads((DEST / 'code-freeze.json').read_text())
        for path,expected in freeze['code_hashes'].items():
            if sha256(ROOT / path) != expected:
                raise ValueError('Sealed deferred code changed')
        engine_spec = importlib.util.spec_from_file_location('sealed_deferred_engine', ROOT / freeze['source_paths']['engine_path'])
        engine = importlib.util.module_from_spec(engine_spec)
        engine_spec.loader.exec_module(engine)
        if sha256(RUN / 'protocol.json') != freeze['protocol_sha256']:
            raise ValueError('Original frozen protocol changed')
        for filename,expected in protocol['code_hashes'].items():
            if sha256(RUN / 'code' / filename) != expected:
                raise ValueError('Original frozen source changed')
        active = json.loads((RUN / 'main-active.json').read_text())
        source = ROOT / active['output']
        status_path = RUN / 'training-status-crash-2-2026-10-08.json'
        exit_path = RUN / 'main-recovery-001-process-exit.json'
        status = json.loads(status_path.read_text())
        checkpoint = source / json.loads((source / 'latest-checkpoint.json').read_text())['path']
        restored = json.loads((checkpoint / 'checkpoint.json').read_text())
        cursor = restored['progress']['cursor']
        crash = verify_crash_boundary(status, cursor, json.loads(exit_path.read_text()), active,
                                      source.relative_to(ROOT).as_posix())
        crash['evidence'] = {path.relative_to(ROOT).as_posix():sha256(path) for path in
                             [status_path, exit_path, RUN / 'main-active.json', source / 'latest-checkpoint.json']}
        atomic_json(DEST / 'crash-boundary.json', crash)
        if cursor['global_update'] >= 360:
            raise ValueError('Initial timing amendment must precede the first original full-development candidate')
        _verify_artifacts(checkpoint,restored['files'])
        previous = restored['manifest']
        if previous.get('early_stop_patience') != 0:
            raise ValueError('Deferral cannot alter an active early-stop decision')
        prior_sources = {'runner_sha256':RUN / 'recovery-001/scalable_training.py',
                         'input_sha256':RUN / 'code/paired_training.py'}
        changes = {field:{'previous':previous[field],'current':sha256(ROOT / freeze['source_paths'][kind])}
                   for field,kind in [('runner_sha256','engine_path'),('input_sha256','input_path')]}
        for field,path in prior_sources.items():
            if sha256(path) != changes[field]['previous']:
                raise ValueError('Previous exact training source changed')
        binding = {'artifacts':{name:sha256(source / name) for name in
                    ['baselines.json','history.json','development-protocol.json','manifest.json']},
                   'protocol_path':(RUN / 'protocol.json').relative_to(ROOT).as_posix(),
                   'protocol_sha256':sha256(RUN / 'protocol.json'),'selection_sha256':sha256(RUN / 'selection.json')}
        audit = {'schema_version':'training-resource-lifecycle-audit-v1','purpose':'development-evaluation-deferral',
            'resume_checkpoint':checkpoint.relative_to(ROOT).as_posix(),
            'previous_run_manifest':(source / 'manifest.json').relative_to(ROOT).as_posix(),
            'checkpoint_manifest_sha256':sha256(checkpoint / 'checkpoint.json'),
            'previous_run_manifest_sha256':sha256(source / 'manifest.json'),
            'source_changes':changes,'previous_source_files':{field:path.relative_to(ROOT).as_posix() for field,path in prior_sources.items()},
            'development_state_reuse':binding,'development_execution':policy,
            'optimization_and_data_sequence':'exact durable cursor64 resume; completed but unsaved update65 replayed once',
            'created_at':datetime.now(timezone.utc).isoformat()}
        atomic_json(DEST / 'lifecycle-audit.json',audit)
        current = {**previous,**{field:change['current'] for field,change in changes.items()},'development_execution':policy}
        verified = engine.validate_resume_lifecycle_audit(DEST / 'lifecycle-audit.json',checkpoint,previous,current)
        reused = engine.reuse_resume_development_state(checkpoint,restored,verified,
                      json.loads((source / 'development-protocol.json').read_text()),RUN / 'selection.json')
        for name in ['training-status.json','experiment-status.json','main-active.json','main-command.json','main-recovery-001-command.json',
                     'main-recovery-001-process-exit.json','driver-process-exit.json']:
            if (RUN / name).exists():
                shutil.copyfile(RUN / name,DEST / ('source-' + name))
        generation = 'new-data-recovery-003-' + sha256(checkpoint / 'checkpoint.json')[:16]
        attempt = {'schema_version':'training-launch-attempt-v1','run_generation':generation,
                   'service_name':'horscht-newdata-deferred-20261008.service',
                   'expected_output':'runs/new-data-last-block-2026-10-08/main-recovery-002',
                   'runtime_environment':{'HSA_CU_MASK':'0:0-23'},
                   'scope':'process-only for main training and deferred development scoring',
                   'mitigation':'24 of 40 compute units; may reduce throughput; does not establish or fix reset cause',
                   'gpu_job_launched':False}
        if (RUN / 'main-recovery-002').exists():
            raise ValueError('Expected next output already exists; do not race another launch')
        atomic_json(DEST / 'launch-attempt.json',attempt)
        artifacts = dict(freeze['code_hashes'])
        for name in ['code-freeze.json','lifecycle-audit.json','crash-boundary.json','launch-attempt.json']:
            artifacts[(DEST / name).relative_to(ROOT).as_posix()] = sha256(DEST / name)
        record = {'schema_version':'new-mixture-operational-recovery-v1','revision':'recovery-003',
            'protocol_sha256':sha256(RUN / 'protocol.json'),'driver_sha256':sha256(DEST / 'run_experiment.py'),
            **freeze['source_paths'],'lifecycle_audit_path':(DEST / 'lifecycle-audit.json').relative_to(ROOT).as_posix(),
            'resume_checkpoint':checkpoint.relative_to(ROOT).as_posix(),
            'resume_checkpoint_manifest_sha256':sha256(checkpoint / 'checkpoint.json'),
            'checkpoint_every_updates':8,'checkpoint_seconds':300,'development_deferral':True,
            'development_execution':policy,'resume_cursor':cursor,'completed_updates_replayed':1,
            'crash_boundary':crash,'run_generation':generation,'runtime_environment':{'HSA_CU_MASK':'0:0-23'},
            'launch_attempt_path':(DEST / 'launch-attempt.json').relative_to(ROOT).as_posix(),
            'gpu_job_launched':False,'artifacts':artifacts,'prepared_at':datetime.now(timezone.utc).isoformat()}
        atomic_json(DEST / 'operational-recovery.json',record)
        spec = importlib.util.spec_from_file_location('sealed_deferral',DEST / 'run_experiment.py')
        driver = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(driver)
        driver.RECOVERY = driver.verify_recovery(RUN,DEST / 'operational-recovery.json')
        driver.verify_protocol(RUN)
        for name in ['pilot-lr-1e-4','pilot-lr-1e-3']:
            driver.stage_output(RUN,name)
        driver.choose_rate(RUN,protocol)
        atomic_json(DEST / 'preflight.json',{'status':'passed','gpu_job_launched':False,'resume_cursor':cursor,
            'baseline_questions':reused['baselines']['warm_start_development']['questions'],
            'checkpoint_artifacts_verified':True,'completed_updates_replayed':1,
            'crash_boundary_sha256':sha256(DEST / 'crash-boundary.json'),'runtime_environment':{'HSA_CU_MASK':'0:0-23'},
            'operational_recovery_sha256':sha256(DEST / 'operational-recovery.json'),
            'completed_at':datetime.now(timezone.utc).isoformat()})
        print(json.dumps({'status':'boundary_verified_ready_for_root_launch','resume_cursor':cursor,'gpu_job_launched':False}),flush=True)
