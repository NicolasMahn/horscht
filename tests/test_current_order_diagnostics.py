import json

import pytest
import torch

import training_order_diagnostics as module


def test_changed_order_metrics_align_stable_choices_and_separate_controls():
    first = {
        "control": (["left", "right"], torch.tensor([0.7, 0.3])),
        "changed": (["left", "right"], torch.tensor([0.9, 0.1])),
    }
    second = {
        "control": (["right", "left"], torch.tensor([0.3, 0.7])),
        "changed": (["right", "left"], torch.tensor([0.9, 0.1])),
    }
    controls = module.consistency(first, second, ["control"])
    changed = module.consistency(first, second, ["changed"])
    assert controls["mean_aligned_total_variation"] == 0
    assert controls["stable_choice_argmax_fraction"] == 1
    assert changed["mean_aligned_total_variation"] == pytest.approx(0.8)
    assert changed["stable_choice_argmax_fraction"] == 0
    assert module.consistency(first, second, [])["mean_aligned_total_variation"] is None


def test_order_diagnostics_reject_changed_choice_identities():
    first = {"row": (["a", "b"], torch.tensor([0.5, 0.5]))}
    second = {"row": (["a", "c"], torch.tensor([0.5, 0.5]))}
    with pytest.raises(ValueError, match="Choice identities changed"):
        module.consistency(first, second, ["row"])


def test_order_diagnostics_refuse_source_held_rows_before_model_allocation(tmp_path):
    (tmp_path / "splits.jsonl").write_text(
        json.dumps({"id": "development", "split": "dev"})
        + "\n"
        + json.dumps({"id": "held", "split": "test"})
        + "\n"
    )
    (tmp_path / "rows.jsonl").write_text(
        json.dumps({"id": "development"}) + "\n" + json.dumps({"id": "held"}) + "\n"
    )
    assert set(module.development_rows(tmp_path, ["development"])) == {"development"}
    with pytest.raises(ValueError, match="development rows only"):
        module.development_rows(tmp_path, ["held"])
    with pytest.raises(ValueError, match="unique development cohort"):
        module.development_rows(tmp_path, ["development", "development"])
