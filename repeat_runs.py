"""Repeat frozen-feature head training with a fixed, shared set of seeds."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiment import fit_head, load_head, metrics, save_head, write_json
from report_theme import MODELS

ROOT = Path(__file__).parent


def repeat(source, seeds, output):
    manifest = source['manifest']
    run_dir = ROOT / source['run_directory']
    feature_path = run_dir / 'features.pt'
    features = torch.load(feature_path, weights_only=True)
    examples = source['examples']
    if json.loads((run_dir / 'examples.json').read_text()) != examples:
        raise ValueError('Cached feature examples do not match the source record')
    if json.loads((run_dir / 'manifest.json').read_text()) != manifest:
        raise ValueError('Cached feature manifest does not match the source record')
    if manifest['backbone_trainable']:
        raise ValueError('Feature reuse requires a frozen backbone')
    verification = source.get('reload_verification') or {}
    if not (verification.get('features_close') and verification.get('predictions_identical')):
        raise ValueError('Source features must have passed fresh-process verification')
    if features.shape[0] != len(examples):
        raise ValueError('Cached feature count does not match examples')
    labels = torch.tensor([ex['answer_index'] for ex in examples])
    counts = torch.tensor([len(ex['options']) for ex in examples])
    train = torch.tensor([ex['split'] == 'train' for ex in examples])
    torch.set_num_threads(manifest['threads'])
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for seed in seeds:
        started = time.perf_counter()
        head, training = fit_head(features[train], labels[train], counts[train],
                                  seed=seed, steps=manifest['max_steps'],
                                  learning_rate=manifest['learning_rate'],
                                  probe=(features[~train], labels[~train], counts[~train]))
        path = output / f'head-seed-{seed}.pt'
        save_head(path, head)
        reloaded = load_head(path)
        with torch.no_grad():
            logits = head(features, counts)
            identical = torch.equal(logits, reloaded(features, counts))
            probabilities = logits.softmax(-1)
        passed = (training['final']['accuracy'] == 1 and training['final']['loss'] <= .02
                  and training['weight_change_l2'] > 0 and identical)
        record = {
            'seed': seed, 'training': training,
            'probe': metrics(head, features[~train], labels[~train], counts[~train]),
            'passed': passed, 'reloaded_head_logits_identical': identical,
            'seconds': time.perf_counter() - started,
            'head_path': str(path.relative_to(ROOT)),
            'predictions': [
                {'id': ex['id'], 'predicted': int(logits[i].argmax()),
                 'expected': ex['answer_index'],
                 'probabilities': probabilities[i, :counts[i]].tolist()}
                for i, ex in enumerate(examples)
            ],
        }
        records.append(record)
        write_json(output / f'seed-{seed}.json', record)
        print(f"{manifest['candidate']} seed {seed}: pass={passed}, steps={training['steps']}, "
              f"probe={record['probe']['accuracy']:.1%}", flush=True)
    return {
        'manifest': manifest, 'examples': examples, 'runs': records,
        'feature_source': source['run_directory'],
        'features_sha256': hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        'source_reload_verification': verification,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'experiment_sha256': hashlib.sha256((ROOT / 'experiment.py').read_bytes()).hexdigest(),
        'finished_at': datetime.now(timezone.utc).isoformat(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(42, 52)))
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Use at least two distinct seeds')
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')
    batch = ROOT / 'runs' / f'{stamp}-repeated'
    results = ROOT / 'results' / 'repeated'
    results.mkdir(parents=True, exist_ok=True)
    failed = False
    for model in args.models:
        source = json.loads((ROOT / 'results' / 'comparison' / f'{model}.json').read_text())
        record = repeat(source, args.seeds, batch / model)
        write_json(results / f'{model}.json', record)
        failed |= any(not run['passed'] for run in record['runs'])
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
