"""Normalize redundant public evidence and publish unique verified originals."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import time

from dataset_factory.game_data.breadth import (
    atomic_json, digest, mastermind_group_keys, mastermind_key, now, recompute,
)
from dataset_factory.game_data.mastermind import Mastermind


def finalize(directory: Path, status_path: Path, generation_snapshot: Path):
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if 'semantic_quality' in manifest:
        raise ValueError('quality refinement already published; refusing a repeated wave')
    started = time.perf_counter()
    original_sources = dict(manifest['provenance']['source_sha256'])
    generator_path = str(Path(__file__).with_name('breadth.py').resolve())
    if hashlib.sha256(generation_snapshot.read_bytes()).hexdigest() != original_sources[generator_path]:
        raise ValueError('original generator snapshot checksum mismatch')
    for path, checksum in original_sources.items():
        if path != generator_path and hashlib.sha256(Path(path).read_bytes()).hexdigest() != checksum:
            raise ValueError('teacher source changed during refinement')
    manifest.update({'state': 'quality_refinement', 'phase': 'normalize_and_relabel'})
    atomic_json(status_path, manifest)
    rows, rejected, kept = [], [], {}
    normalized_histories = 0
    for record in manifest['files']:
        path = Path(record['path']); content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record['sha256']:
            raise ValueError('pre-refinement data checksum mismatch')
        for line in content.splitlines():
            row = json.loads(line)
            old_id = row['id']
            if row['topic'] == 'mastermind':
                observation = json.loads(row['context'])
                history = tuple((tuple(g), b, w) for g, b, w in observation['history'])
                effective = tuple(dict.fromkeys(history))
                normalized_histories += len(effective) < len(history)
                state = Mastermind(observation['symbols'], observation['length'], effective)
                row['context'] = json.dumps(state.observation(), sort_keys=True)
                metadata = row['game_metadata']; query = metadata['query']
                family = mastermind_key(state)
                query['identity'] = {'kind': query['kind'], 'joint_canonical': mastermind_key(state, query)}
                row['id'] = 'game-breadth-v1:' + digest({'family': row['topic'], 'state_family': family, 'query': query['identity']})
                metadata['state_family_id'] = family
                metadata['split_group_keys'] = mastermind_group_keys(state)
                metadata['original_problem_id'] = row['id']
                if len(effective) != len(history):
                    metadata['quality_refinement'] = {'redundant_public_observations_removed': len(history)-len(effective),
                                                      'pre_refinement_id': old_id}
                    if row['answerability_status'] == 'answerable':
                        row['difficulty'] = 'easy' if len(effective) == 1 else 'medium' if len(effective) == 2 else 'hard'
            if row['id'] in kept:
                retained = kept[row['id']]
                if (row['split'], row['problem_group_id']) != (retained['split'], retained['problem_group_id']):
                    raise ValueError('equivalent public-evidence questions crossed family splits')
                rejected.append({'removed_id': old_id, 'normalized_id': row['id'], 'retained_id': retained['id'],
                                 'family': row['topic'], 'reason': 'redundant_feedback_equivalent_question'})
                continue
            recompute(row)
            kept[row['id']] = row
            rows.append(row)
    for record in manifest['prior']:
        if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest() != record['sha256']:
            raise ValueError('prior artifact changed during refinement')
    files=[]
    for split in ('train','development','quarantine'):
        subset=[r for r in rows if r['split']==split]; path=directory/(split+'.jsonl'); checksum=hashlib.sha256()
        temporary=path.with_suffix('.jsonl.tmp')
        with temporary.open('wb') as stream:
            for row in subset:
                line=(json.dumps(row,sort_keys=True)+'\n').encode();stream.write(line);checksum.update(line)
        temporary.replace(path)
        files.append({'path':str(path),'rows':len(subset),'bytes':path.stat().st_size,'sha256':checksum.hexdigest()})
    rejected_path=directory/'quality-exclusions.jsonl'
    exclusion_text=''.join(json.dumps(row,sort_keys=True)+'\n' for row in rejected)
    rejected_path.write_text(exclusion_text)
    accepted=[row for row in rows if row['split']!='quarantine']
    semantic=dict(Counter(r['game_metadata']['target_semantics'] for r in accepted))
    manifest.update({'state':'complete_quality_filtered','phase':'complete','completed_at':now(),
        'generated_candidate_rows':manifest.get('generated_candidate_rows',manifest['generated_new_originals']),
        'generated_new_originals':len(rows),'accepted_originals':len(accepted),'quarantined_originals':len(rows)-len(accepted),
        'quality_excluded_originals':len(rejected),'accepted_by_family':dict(Counter(r['topic'] for r in accepted)),
        'split_counts':dict(Counter(r['split'] for r in rows)),
        'split_by_family':{family:dict(Counter(r['split'] for r in rows if r['topic']==family)) for family in ('dice_decisions','mastermind','sokoban')},
        'files':files,'training_allowlist':[f for f in files if f['path'].endswith('/train.jsonl')],
        'development_allowlist':[f for f in files if f['path'].endswith('/development.jsonl')],
        'semantic_quality':{'redundant_history_records_normalized':normalized_histories,'excluded_duplicate_originals':len(rejected),
            'exclusion_ledger':{'path':str(rejected_path),'rows':len(rejected),'sha256':hashlib.sha256(exclusion_text.encode()).hexdigest()},
            'prior_group_assignments_preserved':True,'effective_public_histories_and_joint_queries_unique':True,
            'refinement_elapsed_seconds':round(time.perf_counter()-started,3)},
        'diversity':{**manifest['diversity'],'query_types':dict(Counter(r['operation_id'] for r in accepted)),
            'target_semantics':semantic,'answerability':dict(Counter(r['answerability_status'] for r in accepted)),
            'difficulty':dict(Counter(r['difficulty'] for r in accepted)),
            'candidate_counts':dict(Counter(str(len(r['options'])) for r in accepted))}})
    manifest['verification'].update({'verified_originals':len(rows),'effective_history_duplicates_removed':True,'verified_at':now()})
    generation_sources = dict(original_sources)
    generation_sources.pop(generator_path)
    generation_sources[str(generation_snapshot.resolve())] = original_sources[generator_path]
    manifest['provenance']['generation_source_sha256'] = generation_sources
    current_paths=[Path(p) for p in original_sources]+[Path(__file__).resolve()]
    manifest['provenance']['source_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in current_paths}
    sokoban_development = manifest['split_by_family']['sokoban'].get('development', 0)
    if manifest['accepted_by_family'].get('sokoban', 0) and sokoban_development < 30:
        manifest['limitations'].append(f'Sokoban has only {sokoban_development} development rows after indivisible near-terrain grouping; do not claim a stable per-family development metric.')
    for record in files:
        if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']:
            raise ValueError('final checksum mismatch')
    atomic_json(manifest_path,manifest);atomic_json(status_path,manifest)
    return manifest


if __name__=='__main__':
    result=finalize(Path('data/game_breadth/2026-10-03'),Path('runs/game-breadth-2026-10-03/status.json'),
                    Path('runs/game-breadth-2026-10-03/generation-source.py'))
    print(json.dumps({key:result[key] for key in ('state','accepted_originals','quality_excluded_originals','split_counts','accepted_by_family')},sort_keys=True))
