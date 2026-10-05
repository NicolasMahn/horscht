import copy
import json
import re
from collections import Counter

import pytest
import torch
from torch.nn import functional as F

from experiment import (
    CAPACITY,
    DecisionHead,
    fit_head,
    format_input,
    last_token,
    load_examples,
    load_head,
    mask_choices,
    metrics,
    save_head,
)
from make_examples import make_examples


def test_masked_loss_and_predictions_ignore_unavailable_outputs():
    logits = torch.tensor([[1.0, 2.0, 100.0, 200.0], [3.0, 1.0, 2.0, 300.0]], requires_grad=True)
    counts = torch.tensor([2, 3])
    labels = torch.tensor([1, 2])
    masked = mask_choices(logits, counts)
    expected_loss = (
        F.cross_entropy(logits[:1, :2], labels[:1])
        + F.cross_entropy(logits[1:, :3], labels[1:])
    ) / 2
    loss = F.cross_entropy(masked, labels)
    torch.testing.assert_close(loss, expected_loss)
    assert masked.argmax(-1).tolist() == [1, 0]
    assert torch.isneginf(masked[0, 2:]).all()
    assert masked.softmax(-1)[0, 2:].sum().item() == 0
    loss.backward()
    assert logits.grad[0, 2:].count_nonzero().item() == 0
    assert logits.grad[1, 3:].count_nonzero().item() == 0
    assert logits.grad[:, :2].count_nonzero().item() > 0


@pytest.mark.parametrize("count", [0, 1, CAPACITY + 1])
def test_mask_rejects_invalid_option_counts(count):
    with pytest.raises(ValueError, match="number of choices"):
        mask_choices(torch.zeros(1, CAPACITY), torch.tensor([count]))


def test_last_token_handles_left_and_right_padding():
    hidden = torch.arange(3 * 5 * 2).reshape(3, 5, 2)
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1], [1, 1, 1, 1, 1]])
    torch.testing.assert_close(last_token(hidden, mask), torch.stack([hidden[0, 2], hidden[1, 4], hidden[2, 4]]))


def test_last_token_rejects_any_empty_input():
    with pytest.raises(ValueError, match="empty input"):
        last_token(torch.zeros(2, 3, 4), torch.tensor([[1, 1, 0], [0, 0, 0]]))


def test_formatter_excludes_labels_and_review_metadata():
    example = make_examples()[0]
    changed = copy.deepcopy(example)
    for key in ("id", "split", "family", "explanation", "provenance"):
        changed[key] = f"secret-{key}"
    changed["answer_index"] = 1 - example["answer_index"]
    assert format_input(changed) == format_input(example)
    assert example["context"] in format_input(example)
    assert example["question"] in format_input(example)
    for index, option in enumerate(example["options"]):
        assert f"{index}: {option}" in format_input(example)


def test_fixtures_are_deterministic_valid_and_have_correct_answers(tmp_path):
    examples = make_examples()
    assert examples == make_examples()
    path = tmp_path / "examples.json"
    path.write_text(json.dumps(examples))
    assert load_examples(path) == examples
    assert Counter(ex["split"] for ex in examples) == {"train": 16, "probe": 8}
    assert len({format_input(ex) for ex in examples}) == len(examples)
    assert {len(ex["options"]) for ex in examples} == {2, 3, 4}
    for example in examples:
        context = example["context"]
        match example["family"]:
            case "comparison":
                first, second = map(int, re.findall(r"\d+", context))
                expected = "Yes" if first > second else "No"
            case "addition":
                expected = str(sum(map(int, re.findall(r"\d+", context))))
            case "routing":
                keyword = context.rsplit(" ", 1)[-1].rstrip(".")
                expected = {"invoice": "Billing", "crash": "Engineering", "quote": "Sales"}[keyword]
            case "state":
                expected = context.rsplit(" ", 1)[-1].rstrip(".").title()
            case _:
                pytest.fail("Unverified fixture family")
        assert example["options"][example["answer_index"]] == expected


@pytest.mark.parametrize("mutation", ["duplicate_id", "duplicate_option", "outside_label", "fractional_label", "bool_label", "invalid_context", "unknown_split", "no_train"])
def test_invalid_dataset_rejected(tmp_path, mutation):
    examples = make_examples()
    if mutation == "duplicate_id":
        examples[1]["id"] = examples[0]["id"]
    elif mutation == "duplicate_option":
        examples[0]["options"][1] = examples[0]["options"][0]
    elif mutation == "outside_label":
        examples[0]["answer_index"] = len(examples[0]["options"])
    elif mutation == "fractional_label":
        examples[0]["answer_index"] = 0.5
    elif mutation == "bool_label":
        examples[0]["answer_index"] = True
    elif mutation == "invalid_context":
        examples[0]["context"] = None
    elif mutation == "unknown_split":
        examples[0]["split"] = "test"
    else:
        for example in examples:
            example["split"] = "probe"
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(examples))
    with pytest.raises(ValueError):
        load_examples(path)


def test_synthetic_features_can_be_memorized_and_reloaded(tmp_path):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        # Independent feature directions guarantee this is a solvable head-only task.
        features = torch.eye(8)
        labels = torch.tensor([0, 1, 2, 3, 1, 0, 2, 1])
        counts = torch.tensor([2, 2, 3, 4, 2, 3, 4, 2])
        head, report = fit_head(features, labels, counts, steps=200, learning_rate=0.05)
        assert report["final"]["accuracy"] == 1.0
        assert report["final"]["loss"] <= 0.02
        assert report["final"]["loss"] < report["initial"]["loss"]
        assert report["weight_change_l2"] > 0
        assert head.score.out_features == CAPACITY
        path = tmp_path / "head.pt"
        save_head(path, head)
        loaded = load_head(path)
        assert loaded is not head
        assert loaded.score.weight.data_ptr() != head.score.weight.data_ptr()
        assert not loaded.training
        with torch.no_grad():
            original = head(features, counts)
            restored = loaded(features, counts)
        torch.testing.assert_close(restored, original, rtol=0, atol=0)
        assert torch.equal(restored.argmax(-1), labels)
    finally:
        torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("step_limit", [3, 200])
def test_probe_observation_preserves_training_and_records_every_step(step_limit):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        features = torch.eye(8)
        labels = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        counts = torch.full((8,), 4)
        # These deliberately contradictory labels must never influence fitting or stopping.
        probe = (features.clone().requires_grad_(), (labels + 1) % 4, counts.clone())
        originals = tuple(tensor.detach().clone() for tensor in probe)
        plain_head, plain = fit_head(features, labels, counts, steps=step_limit, learning_rate=0.05)
        observed_head, observed = fit_head(
            features, labels, counts, steps=step_limit, learning_rate=0.05, probe=probe
        )

        assert observed["steps"] == plain["steps"]
        assert observed["initial"] == plain["initial"]
        assert observed["final"] == plain["final"]
        for key, tensor in plain_head.state_dict().items():
            torch.testing.assert_close(observed_head.state_dict()[key], tensor, rtol=0, atol=0)
        if step_limit == 200:
            assert observed["steps"] < step_limit
            assert observed["final"]["accuracy"] == 1.0
            assert observed["history"][-1]["probe_accuracy"] == 0.0
        else:
            assert observed["steps"] == step_limit

        history = observed["history"]
        assert [point["step"] for point in history] == list(range(observed["steps"] + 1))
        assert len(history) == len(plain["history"])
        for point, plain_point in zip(history, plain["history"]):
            assert {key: point[key] for key in ("step", "loss", "accuracy")} == plain_point
            assert set(point) == {"step", "loss", "accuracy", "probe_loss", "probe_accuracy"}
            assert 0 <= point["probe_accuracy"] <= 1
            assert torch.isfinite(torch.tensor(point["probe_loss"]))
        assert {key: history[0][key] for key in ("loss", "accuracy")} == observed["initial"]
        assert {key: history[-1][key] for key in ("loss", "accuracy")} == observed["final"]
        torch.manual_seed(42)
        initial_head = DecisionHead(features.shape[-1])
        expected_initial_probe = metrics(initial_head, *probe)
        assert {key: history[0][f"probe_{key}"] for key in expected_initial_probe} == expected_initial_probe
        expected_probe = metrics(observed_head, *probe)
        assert {key: history[-1][f"probe_{key}"] for key in expected_probe} == expected_probe
        for actual, original in zip(probe, originals):
            torch.testing.assert_close(actual.detach(), original, rtol=0, atol=0)
            assert actual.grad is None
    finally:
        torch.set_num_threads(previous_threads)
