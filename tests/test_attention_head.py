import pytest
import torch

from attention_head import ARCHITECTURE, OptionAttentionHead, input_spans, token_span_masks
from experiment import format_input
from scalable_training import configure_backbone_training, encode_batch
from training_checkpoint import load_training_checkpoint, save_training_checkpoint


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class CharacterTokenizer:
    def __call__(self, prompts, **kwargs):
        maximum = max(map(len, prompts))
        return {"input_ids": torch.tensor([[1 + ord(char) % 30 for char in prompt] + [0] * (maximum - len(prompt)) for prompt in prompts]),
                "attention_mask": torch.tensor([[1] * len(prompt) + [0] * (maximum - len(prompt)) for prompt in prompts]),
                "offset_mapping": torch.tensor([[(index, index + 1) for index in range(len(prompt))] + [(0, 0)] * (maximum - len(prompt)) for prompt in prompts])}


def example(options=None, context="facts"):
    return {"id": "synthetic", "context": context, "question": "Which?", "options": options or ["first", "second"]}


def test_spans_preserve_prompt_and_include_unicode_multiline_repeated_options():
    row = example(["α\nsecond", "second", "α"], context="second")
    prefix, spans = input_spans(row)
    prompt = format_input(row)
    assert prompt[:prefix].endswith("Question: Which?\n")
    assert [prompt[start:end] for start, end in spans] == row["options"]
    tokens = CharacterTokenizer()([prompt])
    context, options = token_span_masks([row], tokens["offset_mapping"], tokens["attention_mask"])
    assert context.sum() == prefix
    assert options.sum(-1).tolist() == [[len(option) for option in row["options"]]]


def test_overlapping_boundary_special_padding_and_contextless_question():
    row = example(context="")
    prefix, spans = input_spans(row)
    start, end = spans[0]
    offsets = torch.tensor([[(0, 0), (0, prefix + 1), (start - 1, start + 1), (start + 1, end + 1), spans[1], (0, 0)]])
    context, options = token_span_masks([row], offsets, torch.tensor([[1, 1, 1, 1, 1, 0]]))
    assert context.tolist() == [[False, True, False, False, False, False]]
    assert options[0, 0].tolist() == [False, False, True, True, False, False]
    assert options[0, 1, 4]


def test_shared_scorer_permutation_padding_context_and_gradients():
    torch.manual_seed(17)
    head = OptionAttentionHead(8, attention_size=4)
    hidden = torch.randn(2, 8, 8, requires_grad=True)
    context = torch.tensor([[1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    options = torch.zeros(2, 3, 8, dtype=torch.bool)
    options[:, 0, 3:5] = True
    options[:, 1, 5:7] = True
    options[0, 2, 7] = True
    counts = torch.tensor([3, 2])
    logits = head((hidden, context, options), counts)
    assert torch.isneginf(logits[0, 3:]).all() and torch.isneginf(logits[1, 2:]).all()
    swapped = head((hidden, context, options[:, [1, 0, 2]]), counts)
    torch.testing.assert_close(swapped[:, :2], logits[:, [1, 0]])
    altered = hidden.detach().clone()
    altered[1, 7] += 1000
    torch.testing.assert_close(head((altered, context, options), counts)[1], logits[1])
    altered[0, 0] = torch.randn(8)
    assert not torch.equal(head((altered, context, options), counts)[0, :3], logits[0, :3])
    torch.nn.functional.cross_entropy(logits, torch.tensor([2, 1])).backward()
    assert hidden.grad[:, :2].abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in head.parameters())


def test_capacity_265_and_invalid_spans():
    head = OptionAttentionHead(4, attention_size=2)
    hidden = torch.randn(1, 266, 4)
    context = torch.zeros(1, 266, dtype=torch.bool)
    context[0, 0] = True
    options = torch.zeros(1, 265, 266, dtype=torch.bool)
    options[0, torch.arange(265), torch.arange(1, 266)] = True
    assert torch.isfinite(head((hidden, context, options), torch.tensor([265]))).all()
    options[0, 3] = False
    with pytest.raises(ValueError, match="nonempty"):
        head((hidden, context, options), torch.tensor([265]))


def test_last_block_attention_update_checkpoint_roundtrip_keeps_prefix_frozen(tmp_path):
    from transformers import Qwen3Config, Qwen3Model

    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=4, attention_dropout=0.)
    backbone = Qwen3Model(config).eval()
    configure_backbone_training(backbone, "last-block")
    head = OptionAttentionHead(16, attention_size=4)
    before = {name: value.clone() for name, value in backbone.state_dict().items()}
    rows = [example(), example(["third", "fourth", "fifth"], context="")]
    tokenizer = CharacterTokenizer()
    features = encode_batch(tokenizer, backbone, rows, 256, "cpu", head_architecture=ARCHITECTURE)
    counts = torch.tensor([2, 3])
    optimizer = torch.optim.SGD([parameter for parameter in backbone.parameters() if parameter.requires_grad] + list(head.parameters()), lr=.01)
    torch.nn.functional.cross_entropy(head(features, counts), torch.tensor([1, 2])).backward()
    assert all((parameter.grad is not None) == name.startswith("layers.2.") for name, parameter in backbone.named_parameters())
    optimizer.step()
    assert any(not torch.equal(value, before[name]) for name, value in backbone.state_dict().items() if name.startswith("layers.2."))
    assert all(torch.equal(value, before[name]) for name, value in backbone.state_dict().items() if not name.startswith("layers.2."))
    with torch.no_grad():
        expected = head(encode_batch(tokenizer, backbone, rows, 256, "cpu", head_architecture=ARCHITECTURE), counts)
    destination = tmp_path / "checkpoint"
    save_training_checkpoint(destination, backbone=backbone, head=head, optimizer=optimizer, progress={}, manifest={"head_architecture": ARCHITECTURE})
    restored_head = OptionAttentionHead(16, attention_size=4)
    load_training_checkpoint(destination, backbone=backbone, head=restored_head)
    with torch.no_grad():
        actual = restored_head(encode_batch(tokenizer, backbone, rows, 256, "cpu", head_architecture=ARCHITECTURE), counts)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
