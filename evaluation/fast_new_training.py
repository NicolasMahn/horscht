"""Frozen, bounded evaluation amendment; no training or original-report mutation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import torch
from evaluate_new_training import (_snapshot_bundle, _load_tokenizer, _load_selected_model,
                                   _prepare_inference, _infer_rows, write_json_once)
from evaluation.temperature_scaling import fit_temperature, evaluate_final_test
from paired_training import read_jsonl, sha256, summarize_logits
from scalable_training import evaluate_batches
from training_checkpoint import load_training_checkpoint, _verify_artifacts
from training_loop import atomic_json, StopController

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/new-data-last-block-2026-10-08'
SEED = 20261009
UPDATES = [0,360,720,1080,1438]


def read(path):
    return json.loads(path.read_text())


def stratified_ids(rows, lengths, count, seed=SEED):
    """Proportional source/language cells with fixed marginal length coverage."""
    bins = [512,2048,8192,32768,40960]
    band = lambda identifier: next(i for i,b in enumerate(bins) if lengths[identifier] <= b)
    source_cells = defaultdict(list)
    for row in rows:
        source_cells[row.get('_new_dataset',{}).get('source_id','unknown')].append(row)
    rank = lambda identifier: hashlib.sha256(f'{seed}:{identifier}'.encode()).hexdigest()
    def proportional_quotas(cells, size):
        total = sum(len(ids) for ids in cells.values())
        quotas = {key:int(size*len(ids)/total) for key,ids in cells.items()}
        order = sorted(cells,key=lambda key:(-(size*len(cells[key])/total%1),key))
        for key in order[:size-sum(quotas.values())]:
            quotas[key] += 1
        return quotas
    source_quotas = proportional_quotas(source_cells,min(count,len(rows)))
    cells, quotas = {}, {}
    for source,source_rows in sorted(source_cells.items()):
        inner = defaultdict(list)
        for row in source_rows:
            inner[(source,row.get('language','unknown'),band(row['id']))].append(row['id'])
        cells.update(inner)
        quotas.update(proportional_quotas(inner,source_quotas[source]))
    selected = []
    for key,ids in sorted(cells.items()):
        selected.extend(sorted(ids,key=rank)[:quotas[key]])
    cell_by_id = {identifier:key for key,ids in cells.items() for identifier in ids}
    population_bands = {band(identifier) for identifier in cell_by_id}
    for missing in sorted(population_bands - {band(i) for i in selected}):
        replacement = min((i for i in cell_by_id if band(i) == missing),key=rank)
        counts = Counter(band(i) for i in selected)
        donors = [i for i in selected if cell_by_id[i][0] == cell_by_id[replacement][0] and counts[band(i)] > 1]
        if not donors:
            raise ValueError('Sample cannot preserve source quotas and length coverage')
        selected[selected.index(max(donors,key=rank))] = replacement
    return selected,{json.dumps(key):len(ids) for key,ids in cells.items()}


def validate_score(metrics, count):
    if metrics.get('questions') != count or any(not isinstance(metrics.get(k),(int,float))
            or not math.isfinite(metrics[k]) for k in ['soft_cross_entropy','kl_pq','categorical_brier_sum','argmax_agreement']):
        raise ValueError('Partial or nonfinite development score cannot select a checkpoint')


def prepare(output):
    if (output / 'manifest.json').exists():
        raise FileExistsError('Fast evaluation manifests are immutable')
    protocol = read(RUN / 'protocol.json')
    snapshot = Path(protocol['snapshot'])
    split_map, groups, sources, _, split_sha = _snapshot_bundle(snapshot)
    rows = read_jsonl(snapshot / 'rows.jsonl')
    selection = read(RUN / 'selection.json')
    full_dev = set(selection['ids_by_split']['dev'])
    pools = {'dev':[r for r in rows if r['id'] in full_dev],
             'calibration':[r for r in rows if split_map[r['id']] == 'calibration'],
             'test':[r for r in rows if split_map[r['id']] == 'test' and sources[r['id']] != 'original_replay']}
    lengths = read(RUN / 'token-feasibility-v3.json')['lengths_by_id']
    index_path = ROOT / read(RUN / 'main-active.json')['output'] / 'deferred-candidates.json'
    index = read(index_path)
    candidates = {}
    for update in UPDATES:
        entry = index[str(update)]
        checkpoint = ROOT / entry['path']
        record = read(checkpoint / 'checkpoint.json')
        if sha256(checkpoint / 'checkpoint.json') != entry['checkpoint_manifest_sha256'] or record['progress']['cursor']['global_update'] != update:
            raise ValueError('Candidate checkpoint binding changed')
        _verify_artifacts(checkpoint,record['files'])
        candidates[str(update)] = {'path':entry['path'],'checkpoint_manifest_sha256':entry['checkpoint_manifest_sha256']}
    manifest = read(ROOT / candidates['1438']['path'] / 'checkpoint.json')['manifest']
    tokenizer = _load_tokenizer(manifest,2)
    subsets, profiles = {}, {}
    for split,count in [('dev',128),('calibration',64)]:
        ids,cells = stratified_ids(pools[split],lengths,count)
        chosen = {r['id']:r for r in pools[split]}
        _,plan = _prepare_inference(tokenizer,[chosen[i] for i in ids],manifest)
        # Recompute exact full input lengths with the frozen tokenizer, never crop.
        from scalable_training import ordered_lengths
        actual = ordered_lengths([chosen[i] for i in ids],tokenizer,manifest['seed'],-1,manifest['max_tokens'])
        subsets[split] = ids
        profiles[split] = {'questions':len(ids),'total_complete_tokens':sum(actual.values()),
                           'max_complete_tokens':max(actual.values()),'population_cells':cells,
                           'source_counts':dict(Counter(sources[i] for i in ids))}
    weighted = profiles['dev']['total_complete_tokens']*5 + profiles['calibration']['total_complete_tokens']
    code = ['evaluation/fast_new_training.py','evaluate_new_training.py','evaluation/temperature_scaling.py',
            'scalable_training.py','paired_training.py','training_checkpoint.py','training_loop.py',
            'experiment.py','attention_head.py','probability_experiment.py']
    record = {'schema_version':'fast-evaluation-amendment-v1','created_at':datetime.now(timezone.utc).isoformat(),
              'snapshot':snapshot.relative_to(ROOT).as_posix(),'snapshot_manifest_sha256':sha256(snapshot/'manifest.json'),
              'splits_sha256':split_sha,'original_protocol_sha256':sha256(RUN/'protocol.json'),
              'candidate_index_sha256':sha256(index_path),'candidates':candidates,'candidate_updates':UPDATES,
              'seed':SEED,'subsets':subsets,'profiles':profiles,'total_internal_forward_tokens':weighted,
              'sampling':'proportional source quotas then proportional language/token-band cells within each source; global token-band coverage; hash-seeded within cells; no outcomes used',
              'selection':'minimum shared dev128 soft CE; earliest update breaks ties',
              'scope':'deadline-bounded exploratory subset evaluation; replaces remaining evaluation scope only; no full-suite claim',
              'runtime_environment':{'HSA_CU_MASK':'0:0-23'},'code_hashes':{p:sha256(ROOT/p) for p in code},
              'gpu_job_launched':False}
    write_json_once(output/'manifest.json',record)
    print(json.dumps({'status':'prepared','profiles':{k:{x:v[x] for x in ['questions','total_complete_tokens','max_complete_tokens']} for k,v in profiles.items()},'total_internal_forward_tokens':weighted}),flush=True)


def run(output, stage):
    frozen = read(output/'protocol.json')
    manifest_sha = sha256(output/'protocol.json')
    if os.environ.get('HSA_CU_MASK') != '0:0-23':
        raise ValueError('All fast evaluation stages require the pinned24CU mask')
    for path,expected in frozen['code_hashes'].items():
        if sha256(ROOT/path) != expected:
            raise ValueError('Fast evaluator code changed after subset freeze')
    snapshot = ROOT/frozen['snapshot']
    if sha256(snapshot/'manifest.json') != frozen['snapshot_manifest_sha256']:
        raise ValueError('Snapshot binding changed')
    split_map,groups,sources,_,split_sha = _snapshot_bundle(snapshot)
    rows_by_id = {r['id']:r for r in read_jsonl(snapshot/'rows.jsonl')}
    first = ROOT/frozen['candidates']['0']['path']
    model_manifest = read(first/'checkpoint.json')['manifest']
    tokenizer = _load_tokenizer(model_manifest,2)
    plans = {split:_prepare_inference(tokenizer,[rows_by_id[i] for i in ids],model_manifest)
             for split,ids in frozen['subsets'].items()}
    backbone,head = _load_selected_model(model_manifest,first,'cuda',2,tokenizer,output/'status.json')
    development = {}
    for update in UPDATES:
        checkpoint = ROOT/frozen['candidates'][str(update)]['path']
        if sha256(checkpoint/'checkpoint.json') != frozen['candidates'][str(update)]['checkpoint_manifest_sha256']:
            raise ValueError('Frozen candidate changed')
        result_path = output/f'dev-{update:04d}.json'
        if result_path.exists():
            result = read(result_path)
            if result['manifest_sha256'] != manifest_sha:
                raise ValueError('Saved fast result belongs to another plan')
        else:
            load_training_checkpoint(checkpoint,backbone=backbone,head=head,restore_rng=False)
            rows,plan = plans['dev']
            atomic_json(output/'status.json',{'status':'running','phase':'shared development','questions':len(frozen['subsets']['dev']),'update':update})
            args = SimpleNamespace(max_tokens=model_manifest['max_tokens'],device='cuda',head_architecture='last-token-linear-v1',evaluation_phase=f"fast dev{len(frozen['subsets']['dev'])} update{update}")
            ordered, logits = [], []
            with StopController() as stop:
                for update_batches in plan.updates:
                    _, batch_rows, scores = evaluate_batches(tokenizer,backbone,head,rows,
                           SimpleNamespace(updates=[update_batches]),args,None,return_logits=True,stop=stop)
                    ordered.extend(batch_rows)
                    logits.append(scores)
                    progress = {'status':'running','phase':'shared development','update':update,
                                'questions_done':len(ordered),'questions':len(rows)}
                    atomic_json(output/'status.json',progress)
                    print(json.dumps(progress),flush=True)
            metrics = summarize_logits(torch.cat(logits),ordered)
            validate_score(metrics,len(frozen['subsets']['dev']))
            result = {'update':update,'metrics':metrics,'manifest_sha256':manifest_sha,
                      'checkpoint_manifest_sha256':frozen['candidates'][str(update)]['checkpoint_manifest_sha256']}
            write_json_once(result_path,result)
        validate_score(result['metrics'],len(frozen['subsets']['dev']))
        development[str(update)] = result
    best = min(UPDATES,key=lambda u:(development[str(u)]['metrics']['soft_cross_entropy'],u))
    chosen = frozen['candidates'][str(best)]
    selected = {'checkpoint_path':chosen['path'],'checkpoint_manifest_sha256':chosen['checkpoint_manifest_sha256'],
                'update':best,'selection_manifest_sha256':manifest_sha,'selection_metrics_path':(output/'development-selection.json').relative_to(ROOT).as_posix(),
                'selected_checkpoint_reload_verified':True}
    load_training_checkpoint(ROOT/chosen['path'],backbone=backbone,head=head,restore_rng=False)
    for path,record in [(output/'development-selection.json',{'candidates':development,'selected_update':best,'manifest_sha256':manifest_sha}),
                        (output/'selected-checkpoint.json',selected)]:
        if path.exists():
            if read(path) != record:
                raise ValueError('Immutable selection result changed')
        else:
            write_json_once(path,record)
    if stage == 'selection':
        atomic_json(output/'status.json',{'status':'completed','phase':'fast checkpoint selection','selected_update':best})
        print(json.dumps({'status':'selection_completed','selected_update':best}),flush=True)
        return
    for split in ['calibration']:
        result_path = output/('calibration.json' if split == 'calibration' else 'final-test.json')
        if result_path.exists():
            saved = read(result_path)
            if saved['fast_manifest_sha256'] != manifest_sha or saved['checkpoint_sha256'] != chosen['checkpoint_manifest_sha256']:
                raise ValueError('Completed stage binding changed')
            continue
        rows,plan = plans[split]
        atomic_json(output/'status.json',{'status':'running','phase':f'fast {split}'})
        logits,targets,counts,ordered = _infer_rows(tokenizer,backbone,head,[rows_by_id[i] for i in frozen['subsets'][split]],rows,plan,
                      max_tokens=model_manifest['max_tokens'],device='cuda',evaluation_phase=f'fast {split}')
        common = dict(row_ids=[r['id'] for r in ordered],split_by_id=split_map,group_ids_by_id=groups,
                      checkpoint_sha256=chosen['checkpoint_manifest_sha256'],split_manifest_sha256=split_sha)
        if split == 'calibration':
            result = fit_temperature(logits,targets,counts,**common)
        else:
            result = evaluate_final_test(logits,targets,counts,calibration_record=read(output/'calibration.json'),**common)
        result.update(fast_manifest_sha256=manifest_sha,scope='bounded fresh-heldout subset; exploratory')
        write_json_once(result_path,result)
    atomic_json(output/'status.json',{'status':'completed','phase':'internal fast evaluation','selected_update':best})
    print(json.dumps({'status':'internal_completed','selected_update':best}),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','selection','calibration','internal'],required=True)
    parser.add_argument('--output',type=Path,default=RUN/'fast-evaluation-003')
    args = parser.parse_args()
    if args.stage == 'prepare':
        prepare(args.output.resolve())
    else:
        run(args.output.resolve(),args.stage)
