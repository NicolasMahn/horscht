"""Export a portable final-block replacement, proving frozen upstream identity."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def export(checkpoint: Path, upstream: Path, output: Path) -> dict:
    torch.set_num_threads(2)
    record = json.loads((checkpoint / 'checkpoint.json').read_text())
    manifest = record['manifest']
    if record['backbone_state_kind'] != 'trainable-overlay-v1':
        raise ValueError('Expected compact training checkpoint')
    if manifest['repo'] != 'Qwen/Qwen3-4B':
        raise ValueError('Expected Qwen3-4B')
    if upstream.name != manifest['revision']:
        raise ValueError('Upstream snapshot directory must be the pinned revision')
    names = set(record['backbone_parameter_names'])
    if names != set(manifest['trainable_backbone_parameter_names']):
        raise ValueError('Trainable name allowlists differ')
    # The complete block replaces every warm-start block tensor. Proving every
    # other tensor equals upstream makes the private warm-start file unnecessary.
    base = (checkpoint / record['base_checkpoint']['path']).resolve()
    if sha256(base / 'checkpoint.json') != record['base_checkpoint']['manifest_sha256']:
        raise ValueError('Warm-start manifest changed')
    base_record = json.loads((base / 'checkpoint.json').read_text())
    if base_record.get('backbone_state_kind', 'full') != 'full':
        raise ValueError('Expected full warm-start backbone')
    for directory, fields in [(base, ['backbone.pt']), (checkpoint, ['backbone.pt', 'head.pt'])]:
        source_record = json.loads((directory / 'checkpoint.json').read_text())
        for field in fields:
            if sha256(directory / field) != source_record['files'][field]:
                raise ValueError('Checkpoint artifact hash mismatch')
    frozen = torch.load(base / 'backbone.pt', map_location='cpu', weights_only=True, mmap=True)
    overlay = torch.load(checkpoint / 'backbone.pt', map_location='cpu', weights_only=True, mmap=True)
    head = torch.load(checkpoint / 'head.pt', map_location='cpu', weights_only=True, mmap=True)
    if set(overlay) != names or any(not n.startswith('layers.35.') for n in names):
        raise ValueError('Overlay is not precisely the final block')
    index = json.loads((upstream / 'model.safetensors.index.json').read_text())['weight_map']
    upstream_backbone_names = {name.removeprefix('model.') for name in index if name.startswith('model.')}
    if set(frozen) != upstream_backbone_names:
        raise ValueError('Upstream and warm-start backbone names differ')
    upstream_final = {name for name in upstream_backbone_names if name.startswith('layers.35.')}
    if names != upstream_final:
        raise ValueError('Overlay does not replace the complete final block')
    verified = 0
    for shard in sorted(set(index.values())):
        with safe_open(upstream / shard, framework='pt', device='cpu') as stream:
            for name in stream.keys():
                key = name.removeprefix('model.')
                if key not in frozen or key in names:
                    continue
                tensor = stream.get_tensor(name).float()
                if not torch.equal(frozen[key], tensor):
                    raise ValueError(f'Frozen upstream identity failed for {key}')
                verified += 1
    if set(head) != {'score.weight', 'score.bias'} or head['score.weight'].shape != (265, 2560):
        raise ValueError('Head contract mismatch')
    tensors = {f'backbone.{k}': v.contiguous() for k, v in overlay.items()}
    tensors.update({f'head.{k}': v.contiguous() for k, v in head.items()})
    if any(v.dtype != torch.float32 or not torch.isfinite(v).all() for v in tensors.values()):
        raise ValueError('Export requires finite FP32 tensors')
    output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, output / 'decision.safetensors', metadata={
        'format': 'horscht-final-block-replacement-v1',
        'notice': 'Modified Qwen3-4B final block by Nicolas; replacement values, not arithmetic deltas.',
    })
    shutil.copytree(checkpoint / 'tokenizer', output / 'tokenizer', dirs_exist_ok=True)
    shutil.copy2(checkpoint / 'backbone_config/config.json', output / 'backbone_config.json')
    files = ['decision.safetensors', 'backbone_config.json']
    files += sorted(p.relative_to(output).as_posix() for p in (output / 'tokenizer').iterdir() if p.is_file())
    report = {
        'schema_version': 'horscht-final-block-replacement-v1',
        'base_model': manifest['repo'], 'base_revision': manifest['revision'],
        'hidden_size': 2560, 'capacity': 265, 'max_tokens': manifest['max_tokens'],
        'default_inference_max_tokens': 32768,
        'formatter': manifest['formatter'], 'head_architecture': manifest['head_architecture'],
        'parameter_dtype': 'float32', 'inference_autocast_dtype': 'bfloat16',
        'overlay_parameter_names': sorted(names),
        'source_checkpoint_manifest_sha256': sha256(checkpoint / 'checkpoint.json'),
        'source_backbone_sha256': record['files']['backbone.pt'],
        'source_head_sha256': record['files']['head.pt'],
        'warm_start_manifest_sha256': record['base_checkpoint']['manifest_sha256'],
        'frozen_upstream_identity': {'verified_tensors': verified, 'exact_equal_after_upstream_float32_cast': True},
        'files': {f: {'sha256': sha256(output / f), 'bytes': (output / f).stat().st_size} for f in files},
        'selected_update': record['progress']['cursor']['global_update'],
    }
    (output / 'release_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--upstream-snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = export(args.checkpoint, args.upstream_snapshot, args.output)
    print(json.dumps({'verified_frozen_tensors': result['frozen_upstream_identity']['verified_tensors'],
                      'artifact_bytes': result['files']['decision.safetensors']['bytes']}))
