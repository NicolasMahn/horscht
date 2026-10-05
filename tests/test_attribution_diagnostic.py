import pytest
import torch

from attribution_diagnostic import (
    activation_distance, exact_shapley, fixtures, masked_row,
    parameter_changes, summarize_scores, target_contrast,
)


def test_exact_shapley_allocates_interaction_and_obeys_efficiency():
    # The pair-only reward is split equally between its two participating facts.
    values = [2 + bool(mask & 1) * 0.3 + bool(mask & 2) * 0.1
              + bool(mask & 1 and mask & 2) * 0.8 for mask in range(8)]
    phi = exact_shapley(values, 3)
    assert phi == pytest.approx([0.7, 0.5, 0])
    assert sum(phi) == pytest.approx(values[-1] - values[0])


def test_masks_preserve_question_choices_and_choice_index():
    for fixture in fixtures():
        for mask in range(8):
            row = masked_row(fixture, mask)
            assert set(row) == {"context", "question", "options"}
            assert row["question"] == fixture["question"]
            assert row["options"] == fixture["options"]
            assert row["options"][fixture["gold"]] == fixture["options"][fixture["gold"]]
            expected = "\n".join(group["text"] for index, group in enumerate(fixture["groups"]) if mask & (1 << index))
            assert row["context"] == expected
        assert masked_row(fixture, 0)["context"] == ""


def test_target_does_not_follow_changed_prediction():
    fixture = fixtures()[1]
    scores = [[0.1, 0.9]] * 7 + [[0.8, 0.2]]
    summary = summarize_scores(fixture, scores)
    assert fixture["gold"] == 0
    assert summary["empty_target_probability"] == 0.1
    assert summary["full_target_probability"] == 0.8
    assert sum(summary["shapley"]) == pytest.approx(0.7)
    assert summary["efficiency_residual"] == pytest.approx(0)
    assert target_contrast([0.1, 0.9], 0) == pytest.approx(-0.4)


def test_coalition_scores_reject_choice_remapping():
    fixture = fixtures()[1]
    with pytest.raises(ValueError, match="option set"):
        summarize_scores(fixture, [[0.2, 0.3, 0.5]] * 8)
    with pytest.raises(ValueError):
        exact_shapley([0.0] * 7, 3)


def test_activation_comparison_measures_aligned_change():
    reference = torch.tensor([[1.0, 0], [0, 2]])
    scaled = activation_distance(reference, reference * 2)
    assert scaled["mean_token_cosine"] == pytest.approx(1)
    assert scaled["relative_l2"] == pytest.approx(1)
    with pytest.raises(ValueError, match="aligned"):
        activation_distance(reference, reference[:1])


def test_parameter_delta_distinguishes_head_only_from_backbone_changes():
    reference = {"layers.0.weight": torch.tensor([[1.0, 2.0]]), "norm.weight": torch.tensor([3.0])}
    unchanged = parameter_changes(reference, reference)
    assert unchanged["overall"]["changed"] == 0
    candidate = {"layers.0.weight": torch.tensor([[2.0, 2.0]]), "norm.weight": torch.tensor([3.0])}
    changed = parameter_changes(reference, candidate)
    assert changed["overall"]["changed"] == 1
    assert changed["overall"]["relative_l2"] == pytest.approx((1 / 14) ** 0.5)
    assert changed["groups"]["norm"]["changed"] == 0
