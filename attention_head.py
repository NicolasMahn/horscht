"""Shared option queries attending to the existing context/question token states."""

import math

import torch
from torch import nn

from experiment import CAPACITY, format_input, mask_choices


ARCHITECTURE = "option-context-attention-v1"
ATTENTION_SIZE = 256


def input_spans(row):
    prefix = f"Context: {row['context']}\nQuestion: {row['question']}\n"
    prompt = prefix + "Choices:\n"
    spans = []
    for index, option in enumerate(row["options"]):
        prompt += f"{index}: "
        start = len(prompt)
        prompt += option
        spans.append((start, len(prompt)))
        prompt += "\n"
    prompt += "Selected choice:"
    if prompt != format_input(row):
        raise ValueError("Option metadata must preserve the original formatted input")
    return len(prefix), spans


def token_span_masks(rows, offsets, attention_mask):
    """Include boundary-straddling tokens by overlap, ignoring padding/specials."""
    context = torch.zeros_like(attention_mask, dtype=torch.bool)
    options = torch.zeros((len(rows), max(len(row["options"]) for row in rows),
                           attention_mask.shape[1]), dtype=torch.bool)
    starts, ends = offsets[..., 0], offsets[..., 1]
    real = attention_mask.bool() & (ends > starts)
    for batch, row in enumerate(rows):
        prefix_end, spans = input_spans(row)
        context[batch] = real[batch] & (starts[batch] < prefix_end)
        for option, (start, end) in enumerate(spans):
            options[batch, option] = real[batch] & (starts[batch] < end) & (ends[batch] > start)
            if not options[batch, option].any():
                raise ValueError("An available option has no text tokens")
        if not context[batch].any():
            raise ValueError("Context/question prefix has no text tokens")
    return context, options


class OptionAttentionHead(nn.Module):
    def __init__(self, hidden_size, capacity=CAPACITY, attention_size=ATTENTION_SIZE):
        super().__init__()
        self.capacity = capacity
        self.attention_size = attention_size
        self.normalize = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.query = nn.Linear(hidden_size, attention_size, bias=False)
        self.key = nn.Linear(hidden_size, attention_size, bias=False)
        self.value = nn.Linear(hidden_size, attention_size, bias=False)
        self.score = nn.Linear(3 * attention_size, 1)

    def forward(self, features, counts):
        hidden, context_mask, option_mask = features
        if option_mask.shape[1] > self.capacity:
            raise ValueError("Options exceed head capacity")
        valid = torch.arange(option_mask.shape[1], device=counts.device)[None, :] < counts[:, None]
        lengths = option_mask.sum(-1)
        if ((lengths == 0) & valid).any() or (~context_mask.any(-1)).any():
            raise ValueError("Available options and context need nonempty token spans")
        pooled = option_mask.float() @ hidden.float()
        pooled = pooled / lengths.clamp_min(1)[..., None]
        queries = self.query(self.normalize(pooled))
        normalized = self.normalize(hidden.float())
        keys, values = self.key(normalized), self.value(normalized)
        weights = (queries @ keys.transpose(-1, -2)) / math.sqrt(self.attention_size)
        weights = weights.masked_fill(~context_mask[:, None, :], -torch.inf).softmax(-1)
        attended = weights @ values
        logits = self.score(torch.cat((queries, attended, queries * attended), dim=-1)).squeeze(-1)
        logits = nn.functional.pad(logits, (0, self.capacity - logits.shape[1]), value=-torch.inf)
        return mask_choices(logits, counts)


def make_head(hidden_size, architecture="last-token-linear-v1", capacity=CAPACITY):
    if architecture == ARCHITECTURE:
        return OptionAttentionHead(hidden_size, capacity)
    if architecture == "last-token-linear-v1":
        from experiment import DecisionHead
        return DecisionHead(hidden_size, capacity)
    raise ValueError("Unknown decision-head architecture")
