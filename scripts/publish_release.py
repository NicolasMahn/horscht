"""Upload only prepared release artifacts; never traverse training/evaluation stores."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = {
    'README.md', 'LICENSE', 'LICENSE_QWEN', 'LICENSE_CODE', 'NOTICE',
    'horscht_model.py', 'requirements.txt', 'release_manifest.json',
    'backbone_config.json', 'decision.safetensors', 'smoke_verification.json',
    'tokenizer/chat_template.jinja', 'tokenizer/tokenizer.json',
    'tokenizer/tokenizer_config.json',
}
DATA_FILES = {
    'README.md', 'LICENSE', 'NOTICE.md', 'manifest.json', 'checksums.json',
    'document-attribution.jsonl', 'public-source-manifest.json',
    'LICENSES/Apache-2.0.txt',
    'verification.json',
    'data/warmstart/train.parquet', 'data/warmstart/validation.parquet',
    'data/continuation/train.parquet', 'data/continuation/validation.parquet',
}


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(folder, allowed):
    missing = sorted(name for name in allowed if not (folder / name).is_file())
    if missing:
        raise ValueError(f'Incomplete {folder.name} package: {missing}')
    return {name: {'bytes': (folder / name).stat().st_size, 'sha256': digest(folder / name)}
            for name in sorted(allowed)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', help='Hugging Face account or organization; defaults to authenticated account')
    parser.add_argument('--private', action='store_true', help='Create private repositories instead of public releases')
    parser.add_argument('--dry-run', action='store_true', help='Validate and inventory without any remote mutations')
    args = parser.parse_args()
    packages = [
        ('model', 'horscht-qwen3-4b-v0.1', ROOT / 'release/model', MODEL_FILES),
        ('dataset', 'horscht-decisions-v0.1', ROOT / 'release/dataset', DATA_FILES),
    ]
    prepared = []
    for kind, name, folder, allowed in packages:
        files = inventory(folder, allowed)
        prepared.append((kind, name, folder, files))
    model_manifest = json.loads((ROOT / 'release/model/release_manifest.json').read_text())
    for name, info in model_manifest['files'].items():
        if name not in prepared[0][3] or prepared[0][3][name]['sha256'] != info['sha256']:
            raise ValueError('Model manifest does not match prepared files')
    data_checksums = json.loads((ROOT / 'release/dataset/checksums.json').read_text())
    for name, expected in data_checksums.items():
        if name not in prepared[1][3] or prepared[1][3][name]['sha256'] != expected:
            raise ValueError('Dataset checksum record does not match prepared files')
    verification = json.loads((ROOT / 'release/dataset/verification.json').read_text())
    if verification.get('schema_version') != 'horscht-dataset-verification-v1':
        raise ValueError('Missing verified dataset export')
    for config in ('warmstart', 'continuation'):
        check = verification['checks'][config]
        if (check['identical_input_target_and_explanation'] is not True
                or check['identity_allowlist_exact'] is not True
                or check['sealed_id_overlap'] != 0 or check['train_dev_group_overlap'] != 0):
            raise ValueError('Dataset identity or split checks did not pass')
    if args.dry_run:
        print(json.dumps({'status': 'validated-local-inventory', 'packages': [
            {'type': kind, 'name': name, 'files': len(files), 'bytes': sum(v['bytes'] for v in files.values())}
            for kind, name, _, files in prepared]}, indent=2))
        return
    api = HfApi()
    account = api.whoami()['name']
    namespace = args.namespace or account
    receipt_path = ROOT / 'release/publication-receipt.json'
    receipt = {'namespace': namespace, 'private': args.private, 'repositories': []}
    for kind, name, folder, files in prepared:
        repo_id = f'{namespace}/{name}'
        api.create_repo(repo_id=repo_id, repo_type=kind, private=args.private, exist_ok=True)
        commit = api.upload_folder(repo_id=repo_id, repo_type=kind, folder_path=folder,
                                   allow_patterns=list(files),
                                   commit_message='Publish Horscht v0.1 research release')
        repo_files = set(api.list_repo_files(repo_id, repo_type=kind, revision=commit.oid))
        if not set(files) <= repo_files:
            raise RuntimeError('Remote repository is missing release artifacts')
        entry = {'type': kind, 'repo_id': repo_id, 'revision': commit.oid,
                 'url': f'https://huggingface.co/{"datasets/" if kind == "dataset" else ""}{repo_id}',
                 'files': files}
        receipt['repositories'].append(entry)
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
        print(f'Published {entry["url"]} at {commit.oid}', flush=True)


if __name__ == '__main__':
    main()
