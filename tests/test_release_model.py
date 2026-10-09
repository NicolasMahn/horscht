import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from experiment import DecisionHead as TrainingHead, format_input as training_format
from release.model.horscht_model import DecisionHead, HorschtModel, format_input


EXAMPLE = {"context": "Synthetic context.", "question": "Which supplied option?",
           "options": ["first", "second"], "answer_index": 1,
           "explanation": "PRIVATE LABEL METADATA", "split": "test"}


def test_release_formatter_and_head_preserve_training_contract():
    assert format_input(EXAMPLE) == training_format(EXAMPLE)
    assert "PRIVATE LABEL METADATA" not in format_input(EXAMPLE)
    training = TrainingHead(4, 5)
    released = DecisionHead(4, 5)
    released.load_state_dict(training.state_dict())
    features = torch.tensor([[1., -1., 2., -2.], [2., 3., 1., 0.]])
    counts = torch.tensor([2, 4])
    assert torch.equal(released(features, counts), training(features, counts))


@pytest.mark.parametrize("relative,valid_hash", [("weights.bin", False), ("../weights.bin", True)])
def test_loader_rejects_unverified_artifacts_before_loading_base(tmp_path, monkeypatch, relative, valid_hash):
    import release.model.horscht_model as module
    payload = b"synthetic artifact"
    (tmp_path / "weights.bin").write_bytes(payload)
    manifest = {"schema_version": "horscht-final-block-replacement-v1",
                "files": {relative: {"sha256": hashlib.sha256(payload).hexdigest() if valid_hash else "0" * 64}}}
    (tmp_path / "release_manifest.json").write_text(json.dumps(manifest))

    def forbidden(*args, **kwargs):
        raise AssertionError("Base model must not load before artifact verification")

    monkeypatch.setattr(module.AutoModel, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="artifact"):
        HorschtModel.from_pretrained(tmp_path, local_files_only=True)


class SyntheticTokenizer:
    def __init__(self, length):
        self.length = length
        self.calls = []

    def __call__(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"input_ids": torch.ones(1, self.length, dtype=torch.long),
                "attention_mask": torch.ones(1, self.length, dtype=torch.long)}


class SyntheticBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, input_ids, **kwargs):
        self.calls += 1
        assert kwargs["use_cache"] is False
        values = torch.tensor([1., -1., 2., -2.]).expand(1, input_ids.shape[1], 4)
        return SimpleNamespace(last_hidden_state=values)


def synthetic_model(length=3):
    head = DecisionHead(4, 5)
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.bias.copy_(torch.tensor([1., 2., 100., 100., 100.]))
    return HorschtModel(SyntheticBackbone(), head, SyntheticTokenizer(length), {"max_tokens": 40960}, "cpu")


def test_prediction_masks_unavailable_choices_and_only_temperature_changes_probability():
    model = synthetic_model()
    ordinary = model.predict([EXAMPLE], autocast=False)[0]
    softened = model.predict([EXAMPLE], temperature=2, autocast=False)[0]
    assert ordinary["choice_index"] == softened["choice_index"] == 1
    assert len(ordinary["probabilities"]) == 2
    assert sum(ordinary["probabilities"]) == pytest.approx(1)
    assert ordinary["probabilities"][1] > softened["probabilities"][1] > .5
    assert model.tokenizer.calls[0][0] == training_format(EXAMPLE)


def test_overlong_input_is_refused_before_backbone_inference():
    model = synthetic_model(length=4)
    with pytest.raises(ValueError, match="refusing truncation"):
        model.predict([EXAMPLE], autocast=False, max_tokens=3)
    assert model.backbone.calls == 0


@pytest.mark.parametrize("temperature", [0, -1, float("nan"), float("inf")])
def test_invalid_temperature_is_refused_before_tokenization(temperature):
    model = synthetic_model()
    with pytest.raises(ValueError, match="finite and positive"):
        model.predict([EXAMPLE], temperature=temperature, autocast=False)
    assert not model.tokenizer.calls


def test_default_input_cap_requires_explicit_override_for_training_budget():
    model = synthetic_model(length=32769)
    with pytest.raises(ValueError, match="refusing truncation"):
        model.predict([EXAMPLE], autocast=False)
    assert model.backbone.calls == 0
    assert model.predict([EXAMPLE], autocast=False, max_tokens=40960)[0]["input_tokens"] == 32769
    with pytest.raises(ValueError, match="training budget"):
        model.predict([EXAMPLE], autocast=False, max_tokens=40961)
