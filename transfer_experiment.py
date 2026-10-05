"""Evaluate locked, novel-task probes using existing heads; never train or select."""

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

import experiment as core
from size_experiment import digest, write_json, SETTINGS

ROOT = Path(__file__).resolve().parent


def measure(head, features, examples):
    counts = torch.tensor([len(ex['options']) for ex in examples])
    labels = torch.tensor([ex['answer_index'] for ex in examples])
    with torch.inference_mode():
        logits = head(features, counts)
        probabilities = logits.softmax(-1)
        predicted = logits.argmax(-1)
    families = {}
    for family in sorted({ex['family'] for ex in examples}):
        indices = [i for i, ex in enumerate(examples) if ex['family'] == family]
        families[family] = core.metrics(head, features[indices], labels[indices], counts[indices])
    return {
        **core.metrics(head, features, labels, counts), 'families': families,
        'predictions': [dict(id=ex['id'], expected=ex['answer_index'], predicted=predicted[i].item(),
                             probabilities=probabilities[i, :counts[i]].tolist()) for i, ex in enumerate(examples)],
    }


def validate_source(directory):
    metadata = json.loads((directory / 'manifest.json').read_text())
    source_examples = directory / 'examples.json'
    if digest(source_examples) != metadata['data_sha256']:
        raise ValueError('Source training dataset checksum differs')
    for key in ('dtype', 'max_tokens', 'capacity', 'format', 'pooling', 'chat_template', 'backbone_trainable'):
        if metadata['settings'][key] != SETTINGS[key]:
            raise ValueError(f'Unsupported source extraction setting: {key}')
    verification = json.loads((directory / 'reload_verification.json').read_text())
    if not verification['features_close'] or not verification['predictions_identical']:
        raise ValueError('Source extraction verification failed')
    cache = json.loads((directory / 'cache.json').read_text())
    if digest(directory / 'features.pt') != cache['features_sha256']:
        raise ValueError('Source feature checksum differs')
    sources = {}
    for size in ('16', '64'):
        for seed in metadata['seeds']:
            folder = directory / size / str(seed)
            record = json.loads((folder / 'result.json').read_text())
            checksum = digest(folder / 'head.pt')
            if record['seed'] != seed or record['training_ids'] != metadata['subsets'][size]:
                raise ValueError('Source head seed or training subset differs')
            if checksum != record['head_sha256']:
                raise ValueError('Source head checksum differs')
            report = record['report']
            final = report['training']['final']
            if (not report['reloaded_head_logits_identical']
                    or report['training']['weight_change_l2'] <= 0
                    or not math.isfinite(final['loss'])
                    or not 0 <= final['accuracy'] <= 1):
                raise ValueError('Source head pipeline verification failed')
            sources[f'{size}/{seed}'] = checksum
    return metadata, sources


def training_checks(directory, metadata):
    checks = {}
    for size in ('16', '64'):
        runs = []
        for seed in metadata['seeds']:
            path = directory / size / str(seed) / 'result.json'
            record = json.loads(path.read_text())
            runs.append({'seed': seed, 'memorization_passed': record['report']['passed'],
                         'final': record['report']['training']['final'],
                         'result_sha256': digest(path)})
        checks[size] = {'passed': sum(run['memorization_passed'] for run in runs),
                        'total': len(runs), 'runs': runs}
    return checks


def evaluate(model, source_root, runs_root, results_root, data):
    source = source_root / model
    metadata, heads = validate_source(source)
    examples = json.loads(data.read_text())
    dataset_manifest = json.loads(data.with_name('transfer-manifest.json').read_text())
    # The generator locks the dataset before any model sees it.
    if dataset_manifest['data_sha256'] != digest(data):
        raise ValueError('Transfer dataset differs from its locked manifest')
    if len(examples) != 64 or any(ex['split'] != 'probe' for ex in examples):
        raise ValueError('Expected 64 untouched transfer probes')
    if {ex['id'] for ex in examples} & {ex['id'] for ex in json.loads((source / 'examples.json').read_text())}:
        raise ValueError('Transfer and source IDs overlap')
    provenance = {
        'schema_version': 1, 'candidate': model, 'repo': metadata['repo'], 'revision': metadata['revision'],
        'dataset_manifest': dataset_manifest, 'source_manifest_sha256': digest(source / 'manifest.json'),
        'source_data_sha256': metadata['data_sha256'], 'source_heads_sha256': heads,
        'source_directory': str(source.resolve()), 'settings': metadata['settings'], 'seeds': metadata['seeds'],
        'evaluator_sha256': digest(__file__), 'experiment_sha256': digest(core.__file__),
    }
    folder = runs_root / model
    folder.mkdir(parents=True, exist_ok=True)
    locked = folder / 'manifest.json'
    if locked.exists() and json.loads(locked.read_text()) != provenance:
        raise ValueError('Transfer provenance changed; use a new runs root')
    write_json(locked, provenance)
    cache_file = folder / 'features.pt'
    cache_metadata = folder / 'cache.json'
    if cache_file.exists() and cache_metadata.exists():
        cache = json.loads(cache_metadata.read_text())
        if digest(cache_file) != cache['features_sha256']:
            raise ValueError('Transfer feature checksum differs')
        features = torch.load(cache_file, weights_only=True, map_location='cpu')
    else:
        tokenizer, backbone = core.load_backbone(metadata['repo'], metadata['revision'], SETTINGS['dtype'])
        features, lengths = core.encode_examples(tokenizer, backbone, examples, SETTINGS['max_tokens'])
        # Re-extract source examples to ensure this loader reproduces the head's original inputs.
        original_examples = json.loads((source / 'examples.json').read_text())[:2]
        fresh, _ = core.encode_examples(tokenizer, backbone, original_examples, SETTINGS['max_tokens'])
        original = torch.load(source / 'features.pt', weights_only=True, map_location='cpu')[:2]
        difference = (fresh - original).abs().max().item()
        if not torch.allclose(fresh, original, atol=0.001, rtol=0.001):
            raise ValueError('Current loader does not reproduce source features')
        del tokenizer, backbone
        torch.save(features, cache_file)
        cache = {'features_sha256': digest(cache_file), 'token_lengths': lengths,
                 'source_feature_max_difference': difference, 'source_features_close': True}
        write_json(cache_metadata, cache)
    if features.ndim != 2 or len(features) != len(examples) or not torch.isfinite(features).all():
        raise ValueError('Invalid transfer features')
    runs = []
    for seed in metadata['seeds']:
        torch.manual_seed(seed)
        initial = core.DecisionHead(features.shape[1])
        measurements = {'initial': measure(initial, features, examples)}
        for size in ('16', '64'):
            measurements[size] = measure(core.load_head(source / size / str(seed) / 'head.pt'), features, examples)
        runs.append({'seed': seed, 'conditions': measurements,
                     'accuracy_delta_16_to_64': measurements['64']['accuracy'] - measurements['16']['accuracy'],
                     'accuracy_delta_initial_to_16': measurements['16']['accuracy'] - measurements['initial']['accuracy'],
                     'accuracy_delta_initial_to_64': measurements['64']['accuracy'] - measurements['initial']['accuracy']})
    result = {'schema_version': 1, 'experiment': 'novel-task-transfer', 'status': 'complete',
              'manifest': provenance, 'examples': examples, 'feature_cache': cache, 'runs': runs,
              'source_training_checks': training_checks(source, metadata),
              'uniform': {'accuracy': sum(1 / len(ex['options']) for ex in examples) / len(examples),
                          'loss': sum(math.log(len(ex['options'])) for ex in examples) / len(examples)},
              'finished_at': datetime.now(timezone.utc).isoformat(),
              'policy': 'Exploratory endpoint evaluation only; excluded from model selection and training.'}
    results_root.mkdir(parents=True, exist_ok=True)
    write_json(results_root / f'{model}.json', result)
    print(model, {condition: sum(run['conditions'][condition]['accuracy'] for run in runs) / len(runs)
                  for condition in ('initial', '16', '64')}, flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', required=True, choices=core.CANDIDATES)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'runs/size-comparison-original16')
    parser.add_argument('--runs-root', type=Path, default=ROOT / 'runs/transfer')
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results/transfer')
    parser.add_argument('--data', type=Path, default=ROOT / 'data/transfer.json')
    args = parser.parse_args()
    torch.set_num_threads(8)
    for model in args.models:
        evaluate(model, args.source_root, args.runs_root, args.results_root, args.data)


if __name__ == '__main__':
    main()
