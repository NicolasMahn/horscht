"""Direct choice scoring with a pinned Qwen3-4B final-block replacement."""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


def format_input(example: Mapping[str, Any]) -> str:
    """Only these three fields enter the backbone; labels and metadata do not."""
    context, question, options = example['context'], example['question'], example['options']
    if not isinstance(context, str) or not isinstance(question, str) or not question.strip():
        raise ValueError('Context and question must be strings and question must be nonempty')
    if not isinstance(options, (list, tuple)) or not 2 <= len(options) <= 265:
        raise ValueError('Expected 2–265 options')
    if any(not isinstance(option, str) or not option for option in options) or len(set(options)) != len(options):
        raise ValueError('Expected distinct nonempty option strings')
    choices = '\n'.join(f'{i}: {text}' for i, text in enumerate(options))
    return f'Context: {context}\nQuestion: {question}\nChoices:\n{choices}\nSelected choice:'


def last_token(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positions = torch.arange(mask.shape[1], device=mask.device)
    indices = positions.expand_as(mask).masked_fill(~mask.bool(), -1).max(1).values
    if (indices < 0).any():
        raise ValueError('Cannot pool an empty input')
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


class DecisionHead(nn.Module):
    def __init__(self, hidden_size: int, capacity: int):
        super().__init__()
        self.normalize = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.score = nn.Linear(hidden_size, capacity)

    def forward(self, features: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        if ((counts < 2) | (counts > self.score.out_features)).any():
            raise ValueError('Invalid choice count')
        scores = self.score(self.normalize(features.float()))
        unavailable = torch.arange(scores.shape[-1], device=scores.device)[None, :] >= counts[:, None]
        return scores.masked_fill(unavailable, -torch.inf)


class HorschtModel:
    def __init__(self, backbone: nn.Module, head: DecisionHead, tokenizer: Any,
                 manifest: Mapping[str, Any], device: str):
        self.backbone, self.head, self.tokenizer = backbone, head, tokenizer
        self.manifest, self.device = manifest, torch.device(device)

    @classmethod
    def from_pretrained(cls, directory: str | Path, *, device: str = 'cpu',
                        local_files_only: bool = False) -> 'HorschtModel':
        directory = Path(directory)
        manifest = json.loads((directory / 'release_manifest.json').read_text())
        if manifest['schema_version'] != 'horscht-final-block-replacement-v1':
            raise ValueError('Unsupported release schema')
        for relative, expected in manifest['files'].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Invalid artifact path')
            digest = hashlib.sha256()
            with (directory / path).open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(chunk)
            if digest.hexdigest() != expected['sha256']:
                raise ValueError('Release artifact checksum mismatch')
        config = AutoConfig.for_model(**json.loads((directory / 'backbone_config.json').read_text()))
        backbone = AutoModel.from_pretrained(
            manifest['base_model'], revision=manifest['base_revision'], config=config,
            dtype=torch.float32, attn_implementation='sdpa', local_files_only=local_files_only,
        )
        state = load_file(directory / 'decision.safetensors', device='cpu')
        overlay = {k.removeprefix('backbone.'): v for k, v in state.items() if k.startswith('backbone.')}
        head_state = {k.removeprefix('head.'): v for k, v in state.items() if k.startswith('head.')}
        if set(overlay) != set(manifest['overlay_parameter_names']):
            raise ValueError('Backbone replacement names differ')
        expected_names = {k for k in backbone.state_dict() if k.startswith('layers.35.')}
        if set(overlay) != expected_names or len(state) != len(overlay) + len(head_state):
            raise ValueError('Expected the complete final block and head')
        backbone.load_state_dict(overlay, strict=False)
        head = DecisionHead(manifest['hidden_size'], manifest['capacity'])
        head.load_state_dict(head_state, strict=True)
        tokenizer = AutoTokenizer.from_pretrained(directory / 'tokenizer', local_files_only=True)
        tokenizer.padding_side = 'right'
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return cls(backbone.to(device).eval().requires_grad_(False),
                   head.to(device).eval().requires_grad_(False), tokenizer, manifest, device)

    @torch.inference_mode()
    def predict(self, examples: Sequence[Mapping[str, Any]], *, temperature: float = 1.0,
                autocast: bool = True, max_tokens: int = 32768) -> list[dict[str, Any]]:
        """Return probabilities in supplied option order, without text generation.

        Processing one question at a time avoids padding-dependent numerics and
        unbounded batch memory. Overlong complete prompts are refused, not cut.
        """
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError('Temperature must be finite and positive')
        if type(max_tokens) is not int or not 1 <= max_tokens <= self.manifest['max_tokens']:
            raise ValueError('Input token budget must be a positive integer within the training budget')
        results = []
        for example in examples:
            tokens = self.tokenizer(format_input(example), return_tensors='pt', add_special_tokens=True)
            length = tokens['input_ids'].shape[1]
            if length > max_tokens:
                raise ValueError(f'Input has {length} tokens; limit is {max_tokens}; refusing truncation')
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            mask = tokens.get('attention_mask', torch.ones_like(tokens['input_ids']))
            compute = (torch.autocast(self.device.type, dtype=torch.bfloat16, cache_enabled=False)
                       if autocast else nullcontext())
            with compute:
                hidden = self.backbone(input_ids=tokens['input_ids'], attention_mask=mask,
                                       use_cache=False).last_hidden_state
            count = len(example['options'])
            logits = self.head(last_token(hidden, mask).float(), torch.tensor([count], device=self.device))
            probabilities = (logits / temperature).softmax(-1)[0, :count].cpu().tolist()
            if any(not math.isfinite(p) for p in probabilities):
                raise RuntimeError('Nonfinite output probabilities')
            results.append({'choice_index': int(logits.argmax(-1).item()),
                            'probabilities': probabilities, 'input_tokens': length,
                            'temperature': temperature})
        return results
