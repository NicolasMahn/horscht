#!/usr/bin/env python3
"""Run frozen development pilots, then the selected one-epoch continuation."""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paired_training import sha256
from training_loop import atomic_json

IMAGE = '896bcf7217a72aa39f91a25223152a78dc5e8ec8ae306f980c42881361f66d15'
RECOVERY = None


def hub_cache_mount():
    cache = Path(os.environ.get('HF_HUB_CACHE') or Path.home() / '.cache/huggingface/hub').expanduser().resolve()
    return f'{cache}:/hf/hub:ro'


def verify_recovery(run_dir, path):
    record = json.loads(path.read_text())
    if (record.get('schema_version') != 'new-mixture-operational-recovery-v1'
            or record.get('protocol_sha256') != sha256(run_dir / 'protocol.json')
            or record.get('checkpoint_every_updates') != 8 or record.get('checkpoint_seconds') != 300
            or record.get('driver_sha256') != sha256(Path(__file__))):
        raise ValueError('Recovery revision is not bound to this driver and frozen experiment')
    for relative, expected in record['artifacts'].items():
        item = ROOT / relative
        if Path(relative).is_absolute() or '..' in Path(relative).parts or item.is_symlink() or sha256(item) != expected:
            raise ValueError('Recovery artifact checksum mismatch')
    if record['artifacts'].get(record['engine_path']) != sha256(ROOT / record['engine_path']):
        raise ValueError('Recovery engine must be sealed in the revision artifacts')
    if record['artifacts'].get(record['lifecycle_audit_path']) != sha256(ROOT / record['lifecycle_audit_path']):
        raise ValueError('Recovery lifecycle audit must be sealed in the revision artifacts')
    if record['artifacts'].get(record['evaluation_path']) != sha256(ROOT / record['evaluation_path']):
        raise ValueError('Recovery evaluation runner must be sealed in the revision artifacts')
    for name in ['input_path','snapshot_protocol_path','blind_runner_path']:
        if name in record and record['artifacts'].get(record[name]) != sha256(ROOT / record[name]):
            raise ValueError('Deferred revision code must be sealed in the operational artifacts')
    if 'runtime_environment' in record:
        if (record['runtime_environment'] != {'HSA_CU_MASK':'0:0-23'}
                or not record.get('run_generation') or not record.get('launch_attempt_path')):
            raise ValueError('Recovery runtime environment must match the approved process-only mitigation')
        attempt = json.loads((ROOT / record['launch_attempt_path']).read_text())
        if (record['artifacts'].get(record['launch_attempt_path']) != sha256(ROOT / record['launch_attempt_path'])
                or attempt['run_generation'] != record['run_generation']
                or attempt['runtime_environment'] != record['runtime_environment']):
            raise ValueError('Recovery invocation environment is not bound to its sealed attempt')
    return record


def invocation_metadata():
    if RECOVERY is None or 'run_generation' not in RECOVERY:
        return {}
    return {'run_generation':RECOVERY['run_generation'],
            'systemd_invocation_id':os.environ.get('INVOCATION_ID'),
            'runtime_environment':RECOVERY.get('runtime_environment',{})}


def workspace(path):
    return '/workspace/' + Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def verify_protocol(run_dir):
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    for filename, expected in protocol['code_hashes'].items():
        if sha256(run_dir / 'code' / filename) != expected:
            raise ValueError(f'Frozen code checksum mismatch: {filename}')
    if RECOVERY is None and sha256(Path(__file__)) != protocol['code_hashes']['run_experiment.py']:
        raise ValueError('Executed experiment driver differs from its frozen source')
    for path, expected in [(Path(protocol['snapshot']) / 'manifest.json', protocol['snapshot_manifest_sha256']),
            (Path(protocol['system_test_lock']), protocol['system_test_manifest_sha256']),
            (run_dir / 'selection.json', protocol['selection_sha256']),
            (run_dir / 'pilot-selection.json', protocol['pilot_selection_sha256']),
            (Path(protocol['warm_start']['path']) / 'checkpoint.json', protocol['warm_start']['checkpoint_manifest_sha256'])]:
        if sha256(path) != expected:
            raise ValueError('Frozen experiment input checksum mismatch')
    return protocol


def stage_output(run_dir, name):
    receipt_path = run_dir / (name + '-receipt.json')
    if not receipt_path.exists():
        active = run_dir / (name + '-active.json')
        if active.exists():
            record = json.loads(active.read_text())
            if record['protocol_sha256'] != sha256(run_dir / 'protocol.json'):
                raise ValueError('Active stage belongs to another protocol')
            return ROOT / record['output']
        return run_dir / name
    receipt = json.loads(receipt_path.read_text())
    output = Path(receipt['output'])
    if receipt['protocol_sha256'] != sha256(run_dir / 'protocol.json'):
        raise ValueError('Completed stage receipt belongs to another protocol')
    for filename, expected in receipt['artifacts'].items():
        if sha256(output / filename) != expected:
            raise ValueError('Completed stage artifact differs from its receipt')
    report = json.loads((output / 'report.json').read_text())
    if sha256(output / report['best_checkpoint_relative'] / 'checkpoint.json') != receipt['selected_checkpoint_manifest_sha256']:
        raise ValueError('Selected checkpoint differs from the completed stage receipt')
    return output


def rebuild_report(run_dir):
    command = [str(ROOT / '.venv/bin/python'), str(ROOT / 'build_project_record.py')]
    with (run_dir / 'canonical-report.log').open('a') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    atomic_json(run_dir / 'canonical-report-receipt.json', {'command':command,
        'executed_code':{name:sha256(ROOT / name) for name in ['build_project_record.py','build_current_training_report.py','report_theme.py']},
        'rebuilt_at':datetime.now(timezone.utc).isoformat()})


def train_arm(run_dir, protocol, name, rate, *, pilot):
    smoke = json.loads((run_dir / 'full-context-smoke.json').read_text())
    if (smoke.get('protocol_sha256') != sha256(run_dir / 'protocol.json')
            or smoke.get('backward_completed') is not True or smoke.get('optimizer_updates') != 0
            or smoke.get('compact_checkpoint_reload_verified') is not True
            or smoke.get('complete_input_tokens') != protocol['max_complete_input_tokens']['train']):
        raise ValueError('A complete longest-input backward and compact reload proof is required before training')
    output = stage_output(run_dir, name)
    resume = None
    if output.exists():
        report_path = output / 'report.json'
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report['status'] == 'completed' and report['selected_checkpoint_reload_verified']:
                if (run_dir / (name + '-receipt.json')).exists():
                    return report
                raise ValueError('Completed training is missing its durable receipt; review before proceeding')
        latest = output / 'latest-checkpoint.json'
        if not latest.exists():
            raise ValueError(f'{name} has no durable checkpoint to resume')
        resume = output / json.loads(latest.read_text())['path']
        index = 1
        while (run_dir / f'{name}-recovery-{index:03d}').exists():
            index += 1
        output = run_dir / f'{name}-recovery-{index:03d}'
    if RECOVERY is not None and name == 'main':
        if resume is None or sha256(resume / 'checkpoint.json') != RECOVERY['resume_checkpoint_manifest_sha256']:
            raise ValueError('Operational recovery must restore its audited exact main checkpoint')
    if RECOVERY and 'launch_attempt_path' in RECOVERY and name == 'main':
        attempt = json.loads((ROOT / RECOVERY['launch_attempt_path']).read_text())
        if output.relative_to(ROOT).as_posix() != attempt['expected_output']:
            raise ValueError('Recovery output differs from its sealed launch attempt')
    plan = protocol['pilot' if pilot else 'main']
    selection = run_dir / ('pilot-selection.json' if pilot else 'selection.json')
    container_name = 'horscht-newdata-' + name + '-20261008'
    command = ['podman', 'run', '--pull=never', '--rm', '--name', container_name,
       '--workdir=/workspace', '--stop-timeout=600',
       '--network=none', '--memory=60g', '--device=/dev/kfd', '--device=/dev/dri',
       '--security-opt=label=disable', '--security-opt=seccomp=unconfined', '--read-only', '--tmpfs=/tmp:rw,size=1g',
       '-e', 'HF_HUB_OFFLINE=1', '-e', 'HF_HOME=/hf', '-e', 'HF_HUB_DISABLE_PROGRESS_BARS=1', '-e', 'PYTHONPATH=/workspace',
       '-v', f'{ROOT}:/workspace:ro', '-v', f'{run_dir}:{workspace(run_dir)}:rw',
       '-v', hub_cache_mount()]
    for filename in ['paired_training.py','scalable_training.py','training_loop.py','training_checkpoint.py',
                     'experiment.py','attention_head.py','probability_experiment.py','candidates.json','dataset_factory']:
        override = {'scalable_training.py':'engine_path', 'paired_training.py':'input_path'}
        source = ROOT / RECOVERY[override[filename]] if RECOVERY and override.get(filename) in RECOVERY else run_dir / 'code' / filename
        command += ['-v', f'{source}:/workspace/{filename}:ro']
    if RECOVERY and name == 'main':
        for key, value in RECOVERY.get('runtime_environment', {}).items():
            command += ['-e', key + '=' + value]
    command += [IMAGE, '/opt/venv/bin/python', '/workspace/paired_training.py', 'train-arm',
       '--snapshot',workspace(protocol['snapshot']), '--selection',workspace(selection), '--output',workspace(output),
       '--arm','last-block','--head-architecture','last-token-linear-v1','--model','qwen-instruct',
       '--seed',str(protocol['seed']), '--max-tokens',str(protocol['max_tokens']),
       '--max-batch-tokens',str(protocol['max_batch_tokens']), '--epochs','1','--microbatch','1','--accumulation','16',
       '--bucket-questions','256','--head-learning-rate',str(rate),'--backbone-learning-rate','0.00001',
       '--compact-last-block-checkpoints',
       '--lr-schedule','warmup-cosine','--warmup-fraction','0.03',
       '--eval-every',str(plan['fast_every_updates']), '--full-dev-every',str(plan['full_every_updates']),
       '--fast-dev-questions',str(plan['fast_development_questions']), '--order-consistency-questions','0',
       '--checkpoint-every',str(RECOVERY['checkpoint_every_updates'] if RECOVERY else plan.get('checkpoint_every_updates',plan['updates'])),
       '--checkpoint-seconds',str(RECOVERY['checkpoint_seconds'] if RECOVERY else 1800),
       '--retain-checkpoints','3','--feature-cache-mib','0','--threads','2','--reserve-gib','12',
       '--device','cuda','--attention-backend','flash','--gpu-allocation-cap-gib','44',
       '--system-test-manifest',workspace(protocol['system_test_lock']), '--launch-approved',
       '--status-file',workspace(run_dir / 'training-status.json')]
    command += ['--resume',workspace(resume)] if resume is not None else ['--warm-start',workspace(protocol['warm_start']['path'])]
    if RECOVERY is not None:
        command += ['--resume-lifecycle-audit',workspace(ROOT / RECOVERY['lifecycle_audit_path'])]
        if RECOVERY.get('development_deferral'):
            command += ['--defer-full-development']
    if pilot:
        command += ['--max-updates',str(plan['updates'])]
    atomic_json(run_dir / 'experiment-status.json', {'status':'running','phase':name,
               'updated_at':datetime.now(timezone.utc).isoformat(), 'output':str(output),
               'log':str(run_dir / (name + ('-' + RECOVERY['revision'] if RECOVERY else '') + '.log')), 'expected_updates':plan['updates'], **invocation_metadata()})
    log_name = name + '-' + RECOVERY['revision'] if RECOVERY else name
    atomic_json(run_dir / (log_name + '-command.json'), {'command':command, 'protocol_sha256':sha256(run_dir / 'protocol.json'),
                'operational_recovery':RECOVERY, **invocation_metadata()})
    atomic_json(run_dir / (name + '-active.json'), {'output':output.relative_to(ROOT).as_posix(),
                'protocol_sha256':sha256(run_dir / 'protocol.json'), **invocation_metadata()})
    rebuild_report(run_dir)
    print(json.dumps({'phase':name,'head_learning_rate':rate,'expected_updates':plan['updates'], **invocation_metadata()}), flush=True)
    with (run_dir / (log_name + '.log')).open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        atomic_json(run_dir / 'gpu-ownership.json', {'owner':'new mixture last-block continuation', 'pid':os.getpid(),
                    'container_launcher_pid':process.pid,'container':container_name,'phase':name,
                    'started_at':datetime.now(timezone.utc).isoformat(), **invocation_metadata()})
        previous = {}
        def forward(number, frame):
            if process.poll() is None:
                process.send_signal(number)
        for number in (signal.SIGINT,signal.SIGTERM):
            previous[number] = signal.signal(number, forward)
        try:
            code = process.wait()
        finally:
            for number, handler in previous.items():
                signal.signal(number,handler)
    atomic_json(run_dir / (log_name + '-process-exit.json'), {'exit_code':code,'completed_at':datetime.now(timezone.utc).isoformat(), **invocation_metadata()})
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect its log and status before deciding next action')
    report = json.loads((output / 'report.json').read_text())
    if report['status'] != 'completed' or not report['selected_checkpoint_reload_verified']:
        raise RuntimeError(f'{name} has not completed and verified its selected checkpoint')
    if report['cursor']['global_update'] != plan['updates']:
        raise RuntimeError(f'{name} optimizer exposure differs from the frozen plan')
    best = output / report['best_checkpoint_relative']
    atomic_json(run_dir / (name + '-receipt.json'), {'status':'completed','output':str(output),
         'protocol_sha256':sha256(run_dir / 'protocol.json'), 'completed_at':datetime.now(timezone.utc).isoformat(),
         'artifacts':{filename:sha256(output / filename) for filename in ['report.json','manifest.json','best-checkpoint.json']},
         'selected_checkpoint_manifest_sha256':sha256(best / 'checkpoint.json')})
    if name == 'main':
        atomic_json(run_dir / 'resolved-main.json', {'output':output.relative_to(ROOT).as_posix(),'protocol_sha256':sha256(run_dir / 'protocol.json')})
    rebuild_report(run_dir)
    return report


def choose_rate(run_dir, protocol):
    names = ['pilot-lr-1e-4','pilot-lr-1e-3']
    reports = {name:json.loads((stage_output(run_dir,name) / 'report.json').read_text()) for name in names}
    manifests = {name:json.loads((stage_output(run_dir,name) / 'manifest.json').read_text()) for name in names}
    for name, report in reports.items():
        if report['status'] != 'completed' or not report['selected_checkpoint_reload_verified'] or not math.isfinite(report['best_dev_soft_cross_entropy']):
            raise ValueError('Both pilots must complete with finite, verified full-development scores')
        if report['cursor']['global_update'] != protocol['pilot']['updates']:
            raise ValueError('Pilot exposure differs from the frozen budget')
    for field in ['training_questions','training_tokens']:
        if reports[names[0]][field] != reports[names[1]][field]:
            raise ValueError('Pilots do not have matched question/token exposure')
    for field in ['initial_head_state_sha256','selection_sha256','seed','warm_start','lr_schedule',
                  'warmup_fraction','scheduler_total_updates','backbone_learning_rate','runner_sha256','loop_sha256',
                  'checkpoint_sha256','input_sha256','head_sha256','loss_sha256']:
        if manifests[names[0]][field] != manifests[names[1]][field]:
            raise ValueError(f'Pilots do not have matched configuration: {field}')
    winner = min(names, key=lambda name: (reports[name]['best_dev_soft_cross_entropy'], names.index(name)))
    if RECOVERY is not None:
        record = json.loads((run_dir / 'pilot-choice.json').read_text())
        if (record['protocol_sha256'] != sha256(run_dir / 'protocol.json') or record['winner'] != winner
                or record['selected_head_learning_rate'] != manifests[winner]['head_learning_rate']
                or record['pilot_reports_sha256'] != {name:sha256(stage_output(run_dir,name) / 'report.json') for name in names}):
            raise ValueError('Recovery pilot choice differs from completed frozen pilot receipts')
        return record['selected_head_learning_rate']
    record = {'schema_version':'new-mixture-pilot-selection-v1', 'protocol_sha256':sha256(run_dir / 'protocol.json'),
              'selected_head_learning_rate':manifests[winner]['head_learning_rate'], 'winner':winner,
              'selection':'full pilot-development soft cross entropy only; tie prefers 1e-4',
              'pilot_reports_sha256':{name:sha256(stage_output(run_dir,name) / 'report.json') for name in names},
              'arms':{name:{'head_learning_rate':manifests[name]['head_learning_rate'],
                           'best_dev_soft_cross_entropy':reports[name]['best_dev_soft_cross_entropy'],
                           'best_update':reports[name]['best_update'], 'training_questions':reports[name]['training_questions'],
                           'training_tokens':reports[name]['training_tokens']} for name in names},
              'external_and_final_internal_tests':'not evaluated', 'selected_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(run_dir / 'pilot-choice.json', record)
    rebuild_report(run_dir)
    return record['selected_head_learning_rate']



def run_evaluation(run_dir, protocol, name, python_args, *, extra_mounts=(), cpu_only=False):
    command = ['podman','run','--pull=never','--rm','--name','horscht-newdata-' + name + '-20261008',
        '--workdir=/workspace','--stop-timeout=600',
        '--network=none','--memory=60g','--device=/dev/kfd','--device=/dev/dri',
        '--security-opt=label=disable','--security-opt=seccomp=unconfined','--read-only','--tmpfs=/tmp:rw,size=1g',
        '-e','HF_HUB_OFFLINE=1','-e','HF_HOME=/hf','-e','HF_HUB_DISABLE_PROGRESS_BARS=1','-e','PYTHONPATH=/workspace',
        '-v',f'{ROOT}:/workspace:ro','-v',f'{run_dir}:{workspace(run_dir)}:rw',
        '-v',hub_cache_mount()]
    for filename in protocol['code_hashes']:
        if filename.startswith('dataset_factory/'):
            continue
        if (filename.endswith('.py') and filename not in {'prepare_protocol.py','run_experiment.py'}) or filename == 'candidates.json':
            overrides = {'evaluate_new_training.py':'evaluation_path', 'evaluation/snapshot_bound_protocol.py':'snapshot_protocol_path',
                         'evaluation/blind_runner.py':'blind_runner_path'}
            source = ROOT / RECOVERY[overrides[filename]] if RECOVERY and overrides.get(filename) in RECOVERY else run_dir / 'code' / filename
            command += ['-v',f'{source}:/workspace/{filename}:ro']
    command += ['-v',f'{run_dir / "code/dataset_factory"}:/workspace/dataset_factory:ro']
    for source, destination in extra_mounts:
        command += ['-v',f'{source}:{destination}:rw']
    if cpu_only:
        command.remove('--device=/dev/kfd')
        command.remove('--device=/dev/dri')
    command += [IMAGE,'/opt/venv/bin/python',*python_args]
    atomic_json(run_dir / 'experiment-status.json', {'status':'running','phase':name,
                'updated_at':datetime.now(timezone.utc).isoformat(),'log':str(run_dir / (name + '.log'))})
    atomic_json(run_dir / (name + '-command.json'), {'command':command,'protocol_sha256':sha256(run_dir / 'protocol.json')})
    rebuild_report(run_dir)
    with (run_dir / (name + '.log')).open('a') as log:
        process = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        previous = {}
        def forward(number, frame):
            if process.poll() is None:
                process.send_signal(number)
        for number in (signal.SIGINT,signal.SIGTERM):
            previous[number] = signal.signal(number,forward)
        try:
            code = process.wait()
        finally:
            for number,handler in previous.items():
                signal.signal(number,handler)
    atomic_json(run_dir / (name + '-process-exit.json'), {'exit_code':code,
                'completed_at':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha256(run_dir / 'protocol.json')})
    if code:
        raise RuntimeError(f'{name} exited {code}; inspect safe status and logs before continuing')
    rebuild_report(run_dir)


def finish(run_dir, protocol):
    main = stage_output(run_dir,'main')
    for stage in ['calibration','test']:
        run_evaluation(run_dir,protocol,'internal-' + stage,
             ['/workspace/evaluate_new_training.py','--stage',stage,'--snapshot',workspace(protocol['snapshot']),
              '--run',workspace(main),'--device','cuda','--threads','2'])
    # The helper reads only bound metadata and hashes. Its source was frozen with
    # training, while importing at its normal project path keeps path roots valid.
    helper = ROOT / 'evaluation/snapshot_bound_protocol.py'
    if RECOVERY and 'snapshot_protocol_path' in RECOVERY:
        import importlib.util
        helper = ROOT / RECOVERY['snapshot_protocol_path']
        if sha256(helper) != RECOVERY['artifacts'][RECOVERY['snapshot_protocol_path']]:
            raise ValueError('Deferred external binder differs from its sealed source')
        spec = importlib.util.spec_from_file_location('deferred_snapshot_bound_protocol',helper)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bind_completed_development_protocol = module.bind_completed_development_protocol
    elif sha256(helper) != protocol['code_hashes']['evaluation/snapshot_bound_protocol.py']:
        raise ValueError('External protocol binder differs from its frozen source')
    else:
        from evaluation.snapshot_bound_protocol import bind_completed_development_protocol
    external_protocol = run_dir / 'blind-protocol.json'
    bind_completed_development_protocol(Path(protocol['snapshot']),Path(protocol['system_test_lock']),main,
                                         output_protocol=external_protocol)
    external_output = ROOT / 'data/external_evaluation/results/new-data-last-block-2026-10-08'
    external_output.mkdir(parents=True,exist_ok=True)
    arguments = ['/workspace/evaluation/blind_runner.py','--protocol',workspace(external_protocol),
                 '--head-run',workspace(main),'--output-directory',workspace(external_output)]
    if (external_output / 'blind-aggregate-result.json').exists() or any(external_output.glob('cursor-*.json')):
        arguments += ['--resume']
    run_evaluation(run_dir,protocol,'external',arguments,extra_mounts=[(external_output,workspace(external_output))])
    aggregate = external_output / 'blind-aggregate-result.json'
    result = json.loads(aggregate.read_text())
    atomic_json(run_dir / 'benchmark-report.json', {'metrics':result['candidate_results']['last-block'],
          'aggregate_path':aggregate.relative_to(ROOT).as_posix(),'aggregate_sha256':sha256(aggregate),
          'protocol_path':external_protocol.relative_to(ROOT).as_posix(),'protocol_sha256':sha256(external_protocol),
          'training_protocol_sha256':sha256(run_dir / 'protocol.json'),
          'scope':'exploratory external suite already observed in prior work; uncalibrated valid-choice head probabilities',
          'completed_at':datetime.now(timezone.utc).isoformat(), **invocation_metadata()})
    atomic_json(run_dir / 'final-evaluation-receipt.json', {'status':'completed',
          'training_protocol_sha256':sha256(run_dir / 'protocol.json'),
          'calibration_sha256':sha256(main / 'calibration-evaluation/calibration.json'),
          'final_internal_test_sha256':sha256(main / 'calibration-evaluation/final-test.json'),
          'external_aggregate_sha256':sha256(aggregate),'completed_at':datetime.now(timezone.utc).isoformat(), **invocation_metadata()})
    rebuild_report(run_dir)


def run(run_dir, stage):
    protocol = verify_protocol(run_dir)
    with (ROOT / 'runs' / 'gpu-owner.lock').open('a') as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if subprocess.check_output(['podman','ps','-q'],text=True).strip():
            raise RuntimeError('Another container is active; GPU ownership must be resolved before launch')
        if stage in {'smoke','pilot','all'} and RECOVERY is None:
            run_evaluation(run_dir,protocol,'container-preflight',['/workspace/smoke_longest_context.py',
                           '--protocol',workspace(run_dir / 'protocol.json'),'--output',workspace(run_dir / 'container-preflight.json'),
                           '--preflight-only'],cpu_only=True)
            run_evaluation(run_dir,protocol,'full-context-smoke',['/workspace/smoke_longest_context.py',
                           '--protocol',workspace(run_dir / 'protocol.json'),'--output',workspace(run_dir / 'full-context-smoke.json')])
        if stage in {'pilot','all'}:
            train_arm(run_dir, protocol, 'pilot-lr-1e-4', 1e-4, pilot=True)
            train_arm(run_dir, protocol, 'pilot-lr-1e-3', 1e-3, pilot=True)
            rate = choose_rate(run_dir, protocol)
        elif stage != 'smoke':
            rate = choose_rate(run_dir, protocol)
        else:
            rate = None
        if stage in {'main','all'}:
            train_arm(run_dir, protocol, 'main', rate, pilot=False)
        if stage in {'finish','all'}:
            finish(run_dir, protocol)
        atomic_json(run_dir / 'experiment-status.json', {'status':'completed' if stage in {'all','finish'} else 'ready','stage':stage,
                    'selected_head_learning_rate':rate,'completed_at':datetime.now(timezone.utc).isoformat(),
                    'protocol_sha256':sha256(run_dir / 'protocol.json'), **invocation_metadata()})
        rebuild_report(run_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['smoke','pilot','main','finish','all'],default='all')
    parser.add_argument('--run-dir',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--operational-recovery',type=Path)
    args = parser.parse_args()
    exit_code = 1
    try:
        if args.operational_recovery is not None:
            RECOVERY = verify_recovery(args.run_dir, args.operational_recovery)
        run(args.run_dir,args.stage)
        exit_code = 0
    except Exception as exc:
        atomic_json(args.run_dir / 'experiment-status.json', {'status':'failed','message':str(exc),
                    'updated_at':datetime.now(timezone.utc).isoformat(), **invocation_metadata()})
        try:
            rebuild_report(args.run_dir)
        except Exception:
            pass
        raise
    finally:
        atomic_json(args.run_dir / 'driver-process-exit.json', {'exit_code':exit_code,'stage':args.stage,
                    'completed_at':datetime.now(timezone.utc).isoformat(), **invocation_metadata()})
