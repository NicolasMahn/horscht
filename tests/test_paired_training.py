import copy
from types import SimpleNamespace

import pytest
import torch

from dataset_factory import probability_v2
from paired_training import choose_pilot_rows, encode, ordered_row, prompt_cache_key, require_memory, resume_best_checkpoint


def hard_row(identifier):
    return probability_v2.build_hard_row({
        "id": identifier, "problem_group_id": f"group-{identifier}", "split": "train",
        "language": "en", "context": "The marked number is two.",
        "question": "Which number is marked?", "options": ["one", "two", "three"],
        "answer_index": 1, "topic": "numbers", "family": "lookup", "difficulty": "easy",
        "explanation": "The context names two.", "provenance": "Luna-generated; label not independently verified.",
    })


def test_selection_uses_locked_groups_and_counts_complete_long_inputs():
    rows = [hard_row(str(index)) for index in range(8)]
    splits = [
        {"id": row["id"], "snapshot_group_id": "linked" if index < 2 else f"g-{index}",
         "split": "train" if index < 6 else "dev"}
        for index, row in enumerate(rows)
    ]
    lengths = {row["id"]: 100 for row in rows}
    lengths["5"] = 513
    first, policy = choose_pilot_rows(rows, splits, lengths, counts={"train": 4, "dev": 2},
                                      max_tokens=512, seed=9)
    second, _ = choose_pilot_rows(rows, splits, lengths, counts={"train": 4, "dev": 2},
                                 max_tokens=512, seed=9)
    assert first == second
    assert policy["excluded_complete_long_inputs_by_split"] == {"train": 1}
    assert "5" not in {row["id"] for row in first["train"]}
    assert len({entry["snapshot_group_id"] for entry in splits if entry["id"] in
                {row["id"] for row in first["train"]}}) == 4
    with pytest.raises(ValueError, match="distinct groups"):
        choose_pilot_rows(rows, splits, lengths, counts={"train": 5}, max_tokens=512, seed=9)


def test_seeded_joint_permutation_preserves_source_and_correct_choice_identity():
    row = hard_row("source")
    before = copy.deepcopy(row)
    correct_id = row["choice_ids"][row["answer_index"]]
    permuted = ordered_row(row, 13, 2)
    assert row == before
    assert permuted == ordered_row(row, 13, 2)
    assert permuted["choice_ids"][permuted["answer_index"]] == correct_id


def test_frozen_cache_key_includes_actual_choice_order():
    original = hard_row("same-id")
    reordered = probability_v2.shuffle_choices(original, [2, 0, 1])
    assert original["id"] == reordered["id"]
    assert prompt_cache_key(original) != prompt_cache_key(reordered)
    assert prompt_cache_key(original) == prompt_cache_key(copy.deepcopy(original))


class FakeTokenizer:
    def __init__(self, length):
        self.length = length
        self.seen = None

    def __call__(self, text, **kwargs):
        self.seen = text, kwargs
        return {"input_ids": torch.ones(1, self.length, dtype=torch.long),
                "attention_mask": torch.ones(1, self.length, dtype=torch.long)}


class FakeBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=input_ids[..., None] * self.weight)


def test_encoding_never_truncates_and_keeps_backbone_gradient():
    row = hard_row("input")
    tokenizer = FakeTokenizer(3)
    backbone = FakeBackbone()
    feature = encode(tokenizer, backbone, row, 3)
    feature.sum().backward()
    assert backbone.weight.grad.item() == 1
    assert tokenizer.seen[1]["truncation"] is False
    assert all(option in tokenizer.seen[0] for option in row["options"])
    too_long = FakeTokenizer(4)
    with pytest.raises(ValueError, match="refusing truncation"):
        encode(too_long, backbone, row, 3)
    assert backbone.calls == 1


def test_memory_guard_preserves_reserve(monkeypatch):
    monkeypatch.setattr("paired_training.available_memory_bytes", lambda: 20 * 1024**3)
    require_memory(8 * 1024**3, 12)
    with pytest.raises(RuntimeError, match="memory guard"):
        require_memory(9 * 1024**3, 12)


def test_resume_resolves_best_checkpoint_when_artifacts_move_from_container(tmp_path):
    best = tmp_path / "checkpoint-0006"
    best.mkdir()
    last = tmp_path / "last"
    last.mkdir()
    assert resume_best_checkpoint({"best_checkpoint": "checkpoint-0006"}, last) == best
    assert resume_best_checkpoint({"best_checkpoint": "/workspace/old-run/checkpoint-0006"}, last) == best
    with pytest.raises(FileNotFoundError, match="selected resume checkpoint is missing"):
        resume_best_checkpoint({"best_checkpoint": "checkpoint-0004"}, last)


@pytest.mark.parametrize("option", ["Both A and C", "All of the above", "Keine der obigen Antworten", "以上皆是"])
def test_order_references_remain_fixed_across_epochs(option):
    row = hard_row("reference")
    row["options"][0] = option
    assert ordered_row(row, 13, 1) == row
    assert ordered_row(row, 13, 2) == row


def test_online_orders_change_without_changing_stable_target_identity():
    row = hard_row("epoch-augmentation")
    target_id = row["choice_ids"][row["answer_index"]]
    variants = [ordered_row(row, 20261001, epoch) for epoch in range(8)]
    assert len({tuple(variant["choice_ids"]) for variant in variants}) > 1
    assert all(variant["choice_ids"][variant["answer_index"]] == target_id for variant in variants)
