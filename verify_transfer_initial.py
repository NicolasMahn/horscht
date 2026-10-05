"""Verify that recreated random heads match the recorded training initializations."""

import json
from pathlib import Path

import torch

import experiment as core
from size_experiment import write_json
from transfer_experiment import validate_source

ROOT = Path(__file__).resolve().parent


def verify_record(record):
    source = Path(record['manifest']['source_directory'])
    metadata, hashes = validate_source(source)
    if hashes != record['manifest']['source_heads_sha256']:
        raise ValueError('Source heads differ from evaluated transfer heads')
    examples = json.loads((source / 'examples.json').read_text())
    features = torch.load(source / 'features.pt', weights_only=True, map_location='cpu')
    indices_by_id = {example['id']: i for i, example in enumerate(examples)}
    labels = torch.tensor([example['answer_index'] for example in examples])
    counts = torch.tensor([len(example['options']) for example in examples])
    differences = []
    for seed in metadata['seeds']:
        torch.manual_seed(seed)
        head = core.DecisionHead(features.shape[1])
        for size in ('16', '64'):
            indices = [indices_by_id[key] for key in metadata['subsets'][size]]
            actual = core.metrics(head, features[indices], labels[indices], counts[indices])
            result = json.loads((source / size / str(seed) / 'result.json').read_text())
            expected = result['report']['training']['initial']
            differences.extend(abs(actual[key] - expected[key]) for key in ('accuracy', 'loss'))
    maximum = max(differences)
    if maximum != 0:
        raise ValueError(f'{metadata["candidate"]}: initial metrics differ by {maximum}')
    return {'candidate': metadata['candidate'], 'max_initial_metric_difference': maximum,
            'matched_seed_count': len(metadata['seeds']), 'source_features_checksum_verified': True}


def main():
    torch.set_num_threads(8)
    checks = [verify_record(json.loads(path.read_text()))
              for path in sorted((ROOT / 'results/transfer').glob('*.json'))]
    write_json(ROOT / 'results/transfer-initial-verification.json', checks)
    print(f'{len(checks)} models: initial metrics exactly match saved training records')


if __name__ == '__main__':
    main()
