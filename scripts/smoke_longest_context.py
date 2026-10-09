#!/usr/bin/env python3
"""Measure a complete longest training input, including backward, without an update."""
from __future__ import annotations
import argparse
import json
import random
import importlib
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import Adafactor, AutoConfig, AutoModel, AutoTokenizer
from attention_head import make_head
from experiment import CANDIDATES
from paired_training import ROOT, available_memory_bytes, device_memory, ordered_row, peak_rss_gib, read_jsonl, require_memory, sha256, tensors_for
from probability_experiment import soft_cross_entropy
from scalable_training import configure_backbone_training, encode_batch, model_state_sha256, ordered_lengths
from training_checkpoint import load_training_checkpoint, save_training_checkpoint
from training_loop import StopController, atomic_json, wait_for_resources


def project_path(value, protocol):
    return ROOT / Path(value).relative_to(protocol['project_root'])


def container_preflight(protocol_path, output):
    """Check real imports and frozen files without initializing CUDA/HIP."""
    protocol = json.loads(protocol_path.read_text())
    expected_root = Path('/workspace')
    modules = ['paired_training','scalable_training','training_checkpoint','training_loop','experiment',
               'attention_head','probability_experiment','dataset_factory.new_dataset',
               'dataset_factory.training_snapshot','evaluation.isolation','evaluation.temperature_scaling',
               'evaluation.snapshot_bound_protocol','evaluate_new_training','evaluation.blind_runner']
    imports = {}
    for name in modules:
        module = importlib.import_module(name)
        source = Path(module.__file__)
        relative = source.relative_to(expected_root).as_posix()
        if sha256(source) != protocol['code_hashes'][relative]:
            raise ValueError('Container imported a source outside the frozen module contract')
        if hasattr(module,'ROOT') and Path(module.ROOT) != expected_root:
            raise ValueError('Container module resolved an incorrect project root')
        imports[name] = source.as_posix()
    if ROOT != expected_root:
        raise ValueError('Smoke entrypoint resolved an incorrect project root')
    snapshot = project_path(protocol['snapshot'],protocol)
    checkpoint = project_path(protocol['warm_start']['path'],protocol)
    for path, expected in [(snapshot / 'manifest.json',protocol['snapshot_manifest_sha256']),
                           (checkpoint / 'checkpoint.json',protocol['warm_start']['checkpoint_manifest_sha256'])]:
        if sha256(path) != expected:
            raise ValueError('Container cannot resolve an exact frozen experiment input')
    if torch.cuda.is_initialized():
        raise RuntimeError('CPU container preflight unexpectedly initialized CUDA/HIP')
    record = {'schema_version':'canonical-container-import-preflight-v1','status':'passed',
              'protocol_sha256':sha256(protocol_path),'project_root':ROOT.as_posix(),
              'entrypoint':Path(__file__).as_posix(),'import_paths':imports,'cuda_initialized':False,
              'completed_at':datetime.now(timezone.utc).isoformat()}
    atomic_json(output,record)
    print(json.dumps(record),flush=True)
    return record


def smoke(protocol_path, output, stop):
    protocol = json.loads(protocol_path.read_text())
    if output.exists():
        record = json.loads(output.read_text())
        if (record['protocol_sha256'] != sha256(protocol_path) or not record['backward_completed']
                or record['optimizer_updates'] != 0 or record.get('compact_checkpoint_reload_verified') is not True):
            raise ValueError('Existing smoke receipt does not match this frozen experiment')
        return record
    torch.set_num_threads(2)
    torch.manual_seed(protocol['seed'])
    random.seed(protocol['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA/HIP is unavailable for the full-context smoke')
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(44 * 1024**3 / total)
    torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    candidate = CANDIDATES['qwen-instruct']
    tokenizer = AutoTokenizer.from_pretrained(candidate['repo'],revision=candidate['revision'],local_files_only=True)
    tokenizer.padding_side = 'right'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(candidate['repo'],revision=candidate['revision'],local_files_only=True)
    with torch.device('meta'):
        skeleton = AutoModel.from_config(config)
    parameter_bytes = sum(parameter.numel() for parameter in skeleton.parameters()) * 4
    del skeleton
    snapshot = project_path(protocol['snapshot'], protocol)
    selection = json.loads((protocol_path.parent / 'selection.json').read_text())
    selected_ids = set(selection['ids_by_split']['train'])
    rows = [row for row in read_jsonl(snapshot / 'rows.jsonl') if row['id'] in selected_ids]
    lengths = ordered_lengths(rows,tokenizer,protocol['seed'],0,protocol['max_tokens'])
    longest_id = max(lengths,key=lengths.__getitem__)
    row = ordered_row(next(row for row in rows if row['id'] == longest_id),protocol['seed'],0)
    if lengths[longest_id] > config.max_position_embeddings:
        raise ValueError('Full context exceeds the pinned model native window')
    def await_memory(additional, phase):
        def publish(snapshot, waiting):
            atomic_json(output.with_name('full-context-smoke-status.json'), {'phase':phase,'memory_waiting':waiting,
                        'resource_memory':snapshot,'updated_at':datetime.now(timezone.utc).isoformat()})
        if not wait_for_resources(lambda: require_memory(additional,12,'cuda'),stop=stop,
                                  release_cache=torch.cuda.empty_cache,publish=publish):
            raise SystemExit(130)
    await_memory(parameter_bytes + 6 * 1024**3,'loading selected model')
    backbone = AutoModel.from_pretrained(candidate['repo'],revision=candidate['revision'],local_files_only=True,
                    dtype=torch.float32,attn_implementation='sdpa',device_map='cuda').to('cuda')
    backbone._horscht_attention_backend = 'flash'
    names = configure_backbone_training(backbone,'last-block')
    head = make_head(backbone.config.hidden_size,'last-token-linear-v1').to('cuda')
    checkpoint = project_path(protocol['warm_start']['path'], protocol)
    load_training_checkpoint(checkpoint,backbone=backbone,head=head,restore_rng=False)
    initial_head = model_state_sha256(head)
    initial_backbone = model_state_sha256(backbone)
    before_samples = {name:parameter.detach().flatten()[:2].clone() for name,parameter in backbone.named_parameters() if parameter.requires_grad}
    await_memory(6 * 1024**3,'full-context forward and backward')
    features = encode_batch(tokenizer,backbone,[row],protocol['max_tokens'],'cuda',None,'last-token-linear-v1')
    targets, _, _, counts = tensors_for([row],'cuda')
    loss = soft_cross_entropy(head(features,counts),targets,counts)
    if not torch.isfinite(loss):
        raise RuntimeError('Full-context smoke loss is non-finite')
    loss.backward()
    torch.cuda.synchronize()
    if any(parameter.grad is None or not torch.isfinite(parameter.grad).all() for parameter in head.parameters()):
        raise RuntimeError('Full-context head gradients are missing or non-finite')
    for name, parameter in backbone.named_parameters():
        if parameter.requires_grad and (parameter.grad is None or not torch.isfinite(parameter.grad).all()):
            raise RuntimeError('Full-context last-block gradients are missing or non-finite')
        if not parameter.requires_grad and parameter.grad is not None:
            raise RuntimeError('Full-context smoke reached frozen parameters')
    unchanged = all(torch.equal(before_samples[name],parameter.detach().flatten()[:2])
                    for name,parameter in backbone.named_parameters() if parameter.requires_grad)
    if not unchanged or model_state_sha256(head) != initial_head:
        raise RuntimeError('Smoke changed weights without an optimizer update')
    del features, loss
    backbone.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    optimizer = Adafactor([{'params':head.parameters(),'lr':1e-4},
                          {'params':[parameter for parameter in backbone.parameters() if parameter.requires_grad],'lr':1e-5}],
                          scale_parameter=False,relative_step=False,warmup_init=False,beta1=None,weight_decay=0,clip_threshold=1.)
    compact = output.with_name('full-context-smoke-checkpoint')
    if compact.exists():
        compact_record = json.loads((compact / 'checkpoint.json').read_text())
        if (compact_record['manifest'].get('protocol_sha256') != sha256(protocol_path)
                or compact_record['progress'].get('optimizer_updates') != 0):
            raise ValueError('Existing compact smoke checkpoint belongs to another experiment')
    else:
        save_training_checkpoint(compact,backbone=backbone,head=head,optimizer=optimizer,
                 progress={'optimizer_updates':0},manifest={'trainable_backbone_parameter_names':names,
                 'protocol_sha256':sha256(protocol_path),
                 'warm_start':{'checkpoint_manifest_sha256':sha256(checkpoint / 'checkpoint.json')}},
                 backbone_parameter_names=names,base_checkpoint=checkpoint)
    await_memory(2 * 1024**3,'compact checkpoint reload')
    backbone.requires_grad_(False)
    load_training_checkpoint(compact,backbone=backbone,head=head,restore_rng=False)
    compact_verified = model_state_sha256(backbone) == initial_backbone and model_state_sha256(head) == initial_head
    if not compact_verified:
        raise RuntimeError('Compact checkpoint did not restore the exact frozen backbone and head')
    record = {'schema_version':'complete-context-backward-smoke-v1','completed_at':datetime.now(timezone.utc).isoformat(),
              'protocol_sha256':sha256(protocol_path),'warm_start_manifest_sha256':sha256(checkpoint / 'checkpoint.json'),
              'questions':1,'complete_input_tokens':lengths[longest_id],'native_window':config.max_position_embeddings,
              'head_architecture':'last-token-linear-v1','trainable_backbone_names':names,'device':'cuda',
              'attention_backend':'flash','parameters_dtype':'float32','compute_dtype':'bfloat16',
              'gpu_allocation_cap_gib':44,'reserve_gib':12,'backward_completed':True,'optimizer_updates':0,
              'head_weight_hash_unchanged':True,'last_block_weight_samples_unchanged':unchanged,
              'compact_checkpoint_reload_verified':compact_verified,
              'compact_checkpoint_manifest_sha256':sha256(compact / 'checkpoint.json'),
              'initial_backbone_state_sha256':initial_backbone,'initial_head_state_sha256':initial_head,
              'peak_rss_gib':peak_rss_gib(),'available_ram_gib':available_memory_bytes()/1024**3,**device_memory('cuda')}
    atomic_json(output,record)
    print(json.dumps(record),flush=True)
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--preflight-only',action='store_true')
    args=parser.parse_args()
    if args.preflight_only:
        container_preflight(args.protocol,args.output)
    else:
        with StopController() as stop:
            smoke(args.protocol,args.output,stop)
