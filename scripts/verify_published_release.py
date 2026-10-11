"""Verify anonymous release downloads and the standalone CPU inference interface."""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Directory for anonymously downloaded release files')
    args = parser.parse_args()
    receipt = json.loads((ROOT / 'release/publication-receipt.json').read_text())
    if {r['type'] for r in receipt['repositories']} != {'model', 'dataset'}:
        raise ValueError('Both published repositories are required')
    verified = []
    for repo in receipt['repositories']:
        info = HfApi(token=False).repo_info(repo['repo_id'], repo_type=repo['type'], revision=repo['revision'])
        if info.private:
            raise ValueError('Release repository is not public')
        total = 0
        for name, expected in repo['files'].items():
            path = Path(hf_hub_download(repo['repo_id'], name, repo_type=repo['type'],
                                       revision=repo['revision'], token=False, force_download=True,
                                       local_dir=args.output / repo['type']))
            if path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
                raise ValueError(f'Download verification failed: {repo["type"]}/{name}')
            total += path.stat().st_size
            print(f'Checked {repo["type"]}/{name}', flush=True)
        verified.append({'repo_id': repo['repo_id'], 'type': repo['type'], 'revision': repo['revision'],
                         'files_verified': len(repo['files']), 'bytes_verified': total})
    import torch
    torch.set_num_threads(8)
    model_dir = args.output / 'model'
    spec = importlib.util.spec_from_file_location('horscht_public_release', model_dir / 'horscht_model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.HorschtModel.from_pretrained(model_dir, device='cpu')
    result = model.predict([{'context': 'A box contains two red balls and one blue ball. Draw one ball uniformly.',
                             'question': 'Which colour is more likely?', 'options': ['Red', 'Blue']}])[0]
    probabilities = result['probabilities']
    if len(probabilities) != 2 or not all(0 <= p <= 1 for p in probabilities) or abs(sum(probabilities)-1) > 1e-5:
        raise ValueError('Invalid output distribution')
    report = {'verified_at': datetime.now(timezone.utc).isoformat(), 'anonymous_downloads': True,
              'repositories': verified, 'cpu_inference': {'passed': True, 'example_source': 'invented, not benchmark data',
              'result': result, 'torch': torch.__version__},
              'scope': 'All published files downloaded anonymously and checksum verified; synthetic CPU inference, not a benchmark rerun.'}
    (ROOT / 'release/publication-verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Anonymous downloads and standalone CPU inference verified.', flush=True)


if __name__ == '__main__':
    main()
