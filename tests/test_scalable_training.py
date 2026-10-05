import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scalable_training import (
    checkpoint_space_bytes, compare_training_arms, encode_batch, excluded_by_protocol,
    model_state_sha256, run_scalable_arm, validate_system_test_lock,
    with_linked_exclusions,
)
from training_loop import FeatureCache


class BatchTokenizer:
    def __init__(self):
        self.prompts = []

    def __call__(self, prompts, **kwargs):
        self.prompts = prompts
        assert kwargs["truncation"] is False
        lengths = [int(prompt.split("Context: length=")[1].split("\n")[0]) for prompt in prompts]
        maximum = max(lengths)
        return {
            "input_ids": torch.tensor([list(range(1, length + 1)) + [0] * (maximum - length) for length in lengths]),
            "attention_mask": torch.tensor([[1] * length + [0] * (maximum - length) for length in lengths]),
        }


class BatchBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = 0

    def forward(self, input_ids, attention_mask, use_cache):
        self.calls += 1
        return SimpleNamespace(last_hidden_state=input_ids[..., None] * self.weight)


def row(identifier, length):
    return {"id": identifier, "context": f"length={length}", "question": "Which one?",
            "options": ["first", "second"], "problem_group_id": f"group-{identifier}",
            "family": "lookup", "operation_id": "lookup-v1",
            "_training_snapshot": {"snapshot_group_id": f"linked-{identifier}"}}


def test_true_batch_pools_nonpadding_and_reaches_backbone_gradient():
    tokenizer, backbone = BatchTokenizer(), BatchBackbone()
    rows = [row("short", 3), row("long", 5)]
    features = encode_batch(tokenizer, backbone, rows, 5, "cpu")
    torch.testing.assert_close(features, torch.tensor([[6.], [10.]]))
    features.sum().backward()
    assert backbone.weight.grad.item() == 8
    assert all(option in prompt for prompt in tokenizer.prompts for option in ["first", "second"])
    with pytest.raises(ValueError, match="refusing truncation"):
        encode_batch(tokenizer, backbone, rows, 4, "cpu")
    assert backbone.calls == 1


def test_frozen_batch_cache_reencodes_new_order_and_returns_trainable_normal_tensor():
    tokenizer, backbone = BatchTokenizer(), BatchBackbone()
    backbone.requires_grad_(False)
    cache = FeatureCache("pinned-backbone-and-format", 1024)
    rows = [row("same", 3)]
    first = encode_batch(tokenizer, backbone, rows, 5, "cpu", cache)
    second = encode_batch(tokenizer, backbone, rows, 5, "cpu", cache)
    assert backbone.calls == 1 and torch.equal(first, second)
    head = torch.nn.Linear(1, 2)
    head(second).sum().backward()
    assert head.weight.grad is not None
    reordered = [{**rows[0], "options": list(reversed(rows[0]["options"]))}]
    encode_batch(tokenizer, backbone, reordered, 5, "cpu", cache)
    assert backbone.calls == 2


def test_head_only_update_keeps_frozen_backbone_exactly_unchanged_without_cache():
    tokenizer, backbone = BatchTokenizer(), BatchBackbone()
    backbone.requires_grad_(False)
    before = {name: value.clone() for name, value in backbone.state_dict().items()}
    head = torch.nn.Linear(1, 2)
    head_before = head.weight.detach().clone()
    optimizer = torch.optim.SGD(head.parameters(), lr=0.1)
    features = encode_batch(tokenizer, backbone, [row("frozen", 3)], 5, "cpu",
                            FeatureCache("original-backbone", 0))
    assert not backbone.training and not features.requires_grad
    torch.nn.functional.cross_entropy(head(features), torch.tensor([1])).backward()
    optimizer.step()
    assert backbone.weight.grad is None
    for name, value in backbone.state_dict().items():
        assert torch.equal(value, before[name])
    assert not torch.equal(head.weight, head_before)


def test_launch_gate_rejects_before_output_or_model_allocation(tmp_path):
    args = SimpleNamespace(launch_approved=False, output=tmp_path / "output", status_file=tmp_path / "status.json")
    with pytest.raises(ValueError, match="explicit go-ahead"):
        run_scalable_arm(args)
    assert not args.output.exists() and not args.status_file.exists()


def test_independent_system_lock_hashes_and_source_snapshot_are_checked(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "manifest.json").write_text('{"snapshot": 1}\n')
    suite = tmp_path / "independent.jsonl"
    suite.write_text('{"reference_fixture": true}\n')
    lock = {
        "schema_version": "system-test-lock-v1", "status": "locked",
        "independent_from_training_generator": True, "locked_at": "2026-09-30T00:00:00Z",
        "training_snapshot_manifest_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest(),
        "files": {suite.name: hashlib.sha256(suite.read_bytes()).hexdigest()},
        "training_exclusions": {"family_ids": ["unseen-composition"]},
    }
    path = tmp_path / "lock.json"
    path.write_text(json.dumps(lock))
    assert validate_system_test_lock(path, snapshot) == lock
    suite.write_text('{"reference_fixture": false}\n')
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_system_test_lock(path, snapshot)
    lock["status"] = "draft"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="locked independent"):
        validate_system_test_lock(path, snapshot)


def test_protocol_excludes_linked_groups_and_families_without_mutating_rows():
    sample = row("one", 3)
    assert excluded_by_protocol(sample, {"training_exclusions": {"snapshot_group_ids": ["linked-one"]}})
    assert excluded_by_protocol(sample, {"training_exclusions": {"family_ids": ["lookup"]}})
    assert excluded_by_protocol(sample, {"training_exclusions": {"operation_ids": ["lookup-v1"]}})
    assert not excluded_by_protocol(sample, {"training_exclusions": {"row_ids": ["another"]}})
    assert sample["_training_snapshot"] == {"snapshot_group_id": "linked-one"}


def test_source_row_exclusion_expands_to_linked_translation_or_pair():
    original, translation = row("one", 3), row("two", 3)
    translation["_training_snapshot"]["snapshot_group_id"] = "linked-one"
    lock = {"training_exclusions": {"row_ids": ["one"]}}
    expanded = with_linked_exclusions(lock, [original, translation])
    assert excluded_by_protocol(translation, expanded)
    assert lock == {"training_exclusions": {"row_ids": ["one"]}}


def test_training_allowlist_rejects_external_and_symlinked_sources(tmp_path, monkeypatch):
    import experiment
    import paired_training
    from probability_experiment import load_probability_jsonl

    monkeypatch.setattr(experiment, "ROOT", tmp_path)
    monkeypatch.setattr(paired_training, "ROOT", tmp_path)
    allowed = tmp_path / "data" / "training_snapshots" / "v1"
    allowed.mkdir(parents=True)
    external = tmp_path / "data" / "external_evaluation"
    external.mkdir()
    artifact = external / "never-read.jsonl"
    artifact.write_text("invalid content that must never reach a parser")
    with pytest.raises(ValueError, match="External evaluation"):
        load_probability_jsonl(artifact)
    linked = tmp_path / "linked.jsonl"
    linked.symlink_to(artifact)
    with pytest.raises(ValueError, match="External evaluation"):
        experiment.load_examples(linked)
    paired_training.require_training_snapshot(allowed)
    (allowed / "rows.jsonl").symlink_to(artifact)
    with pytest.raises(ValueError, match="artifact resolves outside"):
        paired_training.require_training_snapshot(allowed)


def test_checkpoint_budget_includes_actual_optimizer_and_full_backbone_state():
    backbone, head = torch.nn.Linear(2, 3), torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()))
    before = checkpoint_space_bytes(backbone, head, optimizer, False)
    head(backbone(torch.ones(1, 2))).sum().backward()
    optimizer.step()
    after = checkpoint_space_bytes(backbone, head, optimizer, False)
    assert after > before
    backbone_bytes = sum(parameter.numel() * parameter.element_size() for parameter in backbone.parameters())
    assert checkpoint_space_bytes(backbone, head, optimizer, True) == after + backbone_bytes


def test_initial_state_digest_and_comparison_reject_different_exposure(tmp_path):
    head = torch.nn.Linear(3, 2)
    same = torch.nn.Linear(3, 2)
    same.load_state_dict(head.state_dict())
    assert model_state_sha256(head) == model_state_sha256(same)
    with torch.no_grad():
        same.weight[0, 0].add_(1)
    assert model_state_sha256(head) != model_state_sha256(same)
    reports = {
        "head": {"training_questions": 4, "training_tokens": 10, "cursor": {}},
        "full": {"training_questions": 4, "training_tokens": 11, "cursor": {}},
    }
    manifests = {name: {"schema_version": 2} for name in reports}
    with pytest.raises(ValueError, match="different exposure: training_tokens"):
        compare_training_arms({name: tmp_path / name for name in reports}, reports, manifests, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_comparison_matches_final_plateau_cursor_to_completed_epoch_cursor(tmp_path):
    arms = {name: tmp_path / name for name in ["head", "full"]}
    for directory in arms.values():
        directory.mkdir()
        (directory / "history.json").write_text(json.dumps([
            {"update": 1, "questions_seen": 16, "dev": {"soft_cross_entropy": 0.5}},
        ]))
    reports = {
        name: {"status": "completed", "training_questions": 16, "training_tokens": 80,
               "total_seconds": 10, "cursor": {"global_update": 1, "epoch": epoch, "next_update": next_update}}
        for name, epoch, next_update in [("head", 1, 0), ("full", 0, 1)]
    }
    manifests = {name: {"schema_version": 2, "selection_sha256": "1" * 64,
                        "system_test_manifest_sha256": "2" * 64,
                        "initial_head_state_sha256": "3" * 64} for name in arms}
    output = tmp_path / "comparison"
    assert compare_training_arms(arms, reports, manifests, output) == 0
    comparison = json.loads((output / "comparison.json").read_text())
    assert comparison["matched_questions"] == 16
    reports["full"]["cursor"]["global_update"] = 2
    with pytest.raises(ValueError, match="optimizer updates"):
        compare_training_arms(arms, reports, manifests, tmp_path / "mismatched")
