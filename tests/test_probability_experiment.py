import copy
import json
import math

import pytest
import torch
from torch.nn import functional as F

from dataset_factory import probability_v2 as v2
from experiment import CAPACITY, DecisionHead, load_head, save_head
from probability_experiment import (
    categorical_metrics,
    fit_head,
    load_probability_jsonl,
    soft_cross_entropy,
    targets_for_rows,
)


def _base_row(identifier="soft-question"):
    return {
        "id": identifier,
        "problem_group_id": f"group-{identifier}",
        "split": "train",
        "language": "en",
        "context": "A concise situation with stated evidence.",
        "question": "Which outcome follows?",
        "options": ["first outcome", "second outcome", "third outcome"],
        "topic": "decision making",
        "family": "evidence_choice",
        "difficulty": "easy",
        "explanation": "The target follows from the stated evidence.",
        "provenance": "Luna-generated; label not independently verified.",
    }


def _soft_row(probabilities=(0.7, 0.2, 0.1), identifier="soft-question"):
    base = _base_row(identifier)
    ids = v2.stable_choice_ids(base["id"], base["options"])
    return v2.build_soft_row(
        base,
        "teacher_estimate",
        distribution_evidence={
            "kind": "teacher_estimate",
            "choice_ids": ids,
            "model_id": "teacher-test",
            "model_version": "1",
            "estimation_method": "fixture allocations",
        },
        target_distribution=list(probabilities),
    )


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_metrics_match_analytic_categorical_values():
    logits = torch.tensor([[0.0, 0.0, -torch.inf]])
    target = torch.tensor([[0.75, 0.25, 0.0]])
    counts = torch.tensor([2])

    result = categorical_metrics(logits, target, counts, ["exact_derived"], [None])

    expected_kl = 0.75 * math.log(1.5) + 0.25 * math.log(0.5)
    assert result["soft_cross_entropy"] == pytest.approx(math.log(2))
    assert result["kl_pq"] == pytest.approx(expected_kl)
    assert result["categorical_brier_sum"] == pytest.approx(0.125)
    assert result["argmax_agreement"] == 1.0
    assert "hard_accuracy" not in result
    assert result["by_source_kind"]["exact_derived"]["interpretation"] == (
        "agreement_with_recorded_target_distribution"
    )


def test_hard_label_one_hot_matches_categorical_cross_entropy():
    row = v2.build_hard_row({**_base_row("hard-question"), "answer_index": 1})
    targets, sources, hard_labels = targets_for_rows([row])
    logits = torch.tensor([[1.0, 2.0, -1.0] + [0.0] * 262])
    counts = torch.tensor([3])

    loss = soft_cross_entropy(logits, targets, counts)
    metrics = categorical_metrics(logits, targets, counts, sources, hard_labels)

    torch.testing.assert_close(targets[0, :3], torch.tensor([0.0, 1.0, 0.0]))
    assert loss.item() == pytest.approx(F.cross_entropy(logits[:, :3], torch.tensor([1])).item())
    assert metrics["hard_accuracy"] == 1.0
    assert metrics["hard_label_questions"] == 1


def test_choice_shuffle_preserves_metrics_when_logits_and_targets_follow_ids(tmp_path):
    original = _soft_row()
    order = [2, 0, 1]
    shuffled = v2.shuffle_choices(original, order)
    path = tmp_path / "shuffled.jsonl"
    _write_rows(path, [shuffled])
    loaded = load_probability_jsonl(path)
    targets, sources, hard_labels = targets_for_rows(loaded)
    counts = torch.tensor([3])
    original_logits = torch.tensor([[1.4, 0.2, -0.3] + [0.0] * 262])
    shuffled_logits = torch.tensor([[original_logits[0, index].item() for index in order] + [0.0] * 262])
    original_target = torch.tensor([[0.7, 0.2, 0.1] + [0.0] * 262])
    original_result = categorical_metrics(
        original_logits, original_target, counts, ["teacher_estimate"], [None]
    )
    shuffled_result = categorical_metrics(shuffled_logits, targets, counts, sources, hard_labels)

    assert shuffled["distribution_evidence"]["choice_ids"] == shuffled["choice_ids"]
    assert shuffled_result["soft_cross_entropy"] == pytest.approx(original_result["soft_cross_entropy"])
    assert shuffled_result["kl_pq"] == pytest.approx(original_result["kl_pq"])
    assert shuffled_result["categorical_brier_sum"] == pytest.approx(
        original_result["categorical_brier_sum"]
    )
    assert shuffled_result["argmax_agreement"] == original_result["argmax_agreement"]
    assert shuffled_result["by_source_kind"]["teacher_estimate"]["interpretation"] == (
        "agreement_with_teacher_estimate; not truth or calibration"
    )


def test_variable_choice_masks_zero_targets_and_unused_capacity_have_finite_gradients():
    torch.manual_seed(7)
    head = DecisionHead(hidden_size=4, capacity=5)
    features = torch.randn(2, 4)
    counts = torch.tensor([2, 4])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                            [0.25, 0.75, 0.0, 0.0, 0.0]])

    loss = soft_cross_entropy(head(features, counts), targets, counts)
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all()
               for parameter in head.parameters())
    assert head.score.weight.grad[4].count_nonzero().item() == 0
    assert head.score.bias.grad[4].item() == 0


def test_265_choice_soft_target_load_train_mask_permute_and_checkpoint(tmp_path):
    options = [f"impossible distractor {index:03d}" for index in range(CAPACITY - 1)]
    options.append("supported outcome")
    soft_distribution = [0.0] * CAPACITY
    soft_distribution[7] = 0.25
    soft_distribution[-1] = 0.75
    soft = v2.build_soft_row(
        {**_base_row("wide-soft"), "options": options},
        "teacher_estimate",
        distribution_evidence={
            "kind": "teacher_estimate",
            "choice_ids": v2.stable_choice_ids("wide-soft", options),
            "model_id": "teacher-test",
            "model_version": "1",
            "estimation_method": "fixture allocations",
        },
        target_distribution=soft_distribution,
    )
    hard = v2.build_hard_row(
        {**_base_row("wide-hard"), "options": options, "answer_index": CAPACITY - 1}
    )
    short = _soft_row(identifier="short-row")
    data = tmp_path / "wide-choices.jsonl"
    _write_rows(data, [soft, hard, short])

    rows = load_probability_jsonl(data)
    targets, sources, hard_labels = targets_for_rows(rows)
    counts = torch.tensor([CAPACITY, CAPACITY, len(short["options"])])
    assert targets.shape == (3, CAPACITY)
    assert targets[0, 7].item() == pytest.approx(0.25)
    assert targets[0, CAPACITY - 1].item() == pytest.approx(0.75)
    assert targets[1, CAPACITY - 1].item() == 1.0
    assert targets[2, 3:].count_nonzero().item() == 0

    torch.manual_seed(41)
    head = DecisionHead(hidden_size=8)
    features = torch.randn(3, 8)
    logits = head(features, counts)
    logits.retain_grad()
    loss = soft_cross_entropy(logits, targets, counts)
    soft_loss = soft_cross_entropy(logits[:1], targets[:1], counts[:1])
    before = categorical_metrics(logits[:1], targets[:1], counts[:1], sources[:1], hard_labels[:1])
    assert head.score.out_features == CAPACITY
    assert logits.shape == (3, CAPACITY)
    assert torch.isfinite(logits[0]).all()
    assert torch.isneginf(logits[2, 3:]).all()
    assert logits.softmax(-1)[2, 3:].count_nonzero().item() == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert logits.grad[0, CAPACITY - 1].abs().item() > 0
    assert logits.grad[1, CAPACITY - 1].abs().item() > 0
    assert logits.grad[2, 3:].count_nonzero().item() == 0

    order = list(reversed(range(CAPACITY)))
    shuffled = v2.shuffle_choices(rows[0], order)
    shuffled_path = tmp_path / "wide-choices-shuffled.jsonl"
    _write_rows(shuffled_path, [shuffled])
    shuffled_rows = load_probability_jsonl(shuffled_path)
    shuffled_targets, shuffled_sources, shuffled_hard = targets_for_rows(shuffled_rows)
    shuffled_logits = logits.detach()[0:1, order]
    shuffled_count = torch.tensor([CAPACITY])
    after = categorical_metrics(
        shuffled_logits, shuffled_targets, shuffled_count, shuffled_sources, shuffled_hard
    )
    shuffled_loss = soft_cross_entropy(shuffled_logits, shuffled_targets, shuffled_count)
    assert shuffled_loss.item() == pytest.approx(soft_loss.detach().item())
    for key in ("soft_cross_entropy", "kl_pq", "categorical_brier_sum", "argmax_agreement"):
        assert after[key] == pytest.approx(before[key])

    trained_head, training = fit_head(
        features,
        targets,
        counts,
        sources,
        hard_labels,
        steps=1,
        learning_rate=0.01,
        seed=41,
    )
    assert trained_head.score.out_features == CAPACITY
    assert not torch.equal(trained_head.score.weight[CAPACITY - 1], head.score.weight[CAPACITY - 1])
    assert torch.isfinite(torch.tensor(training["final"]["soft_cross_entropy"]))

    checkpoint = tmp_path / "wide-head.pt"
    save_head(checkpoint, trained_head)
    reloaded = load_head(checkpoint)
    assert reloaded.score.out_features == CAPACITY
    with torch.no_grad():
        torch.testing.assert_close(reloaded(features, counts), trained_head(features, counts), rtol=0, atol=0)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda row: row.update(target_distribution=[1.0, 0.0]),
        lambda row: row.update(target_distribution=[0.6, 0.2, 0.1]),
        lambda row: row.update(target_distribution=[float("nan"), 0.0, 1.0]),
        lambda row: row.update(distribution_source_kind="mystery"),
        lambda row: row.update(choice_ids=list(reversed(row["choice_ids"]))),
    ],
)
def test_invalid_probability_v2_targets_or_choice_alignment_are_rejected(tmp_path, mutate):
    row = copy.deepcopy(_soft_row())
    mutate(row)
    path = tmp_path / "invalid.jsonl"
    _write_rows(path, [row])

    with pytest.raises(ValueError, match="invalid probability-v2 row|stable IDs"):
        load_probability_jsonl(path)
