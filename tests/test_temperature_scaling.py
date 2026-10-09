import math

import pytest
import torch

from evaluation.temperature_scaling import (
    apply_temperature,
    evaluate_final_test,
    fit_temperature,
    summarize_distribution_metrics,
)


CHECKPOINT_SHA = "a" * 64
SPLITS_SHA = "b" * 64


def _groups(*row_ids):
    return {row_id: [f"problem:{row_id}", f"snapshot:{row_id}"] for row_id in row_ids}


def test_metrics_score_full_soft_targets_and_ignore_masked_choices():
    logits = torch.tensor([[2.0, 0.0, 99.0], [0.0, 1.0, -99.0]])
    targets = torch.tensor([[0.7, 0.3, 0.0], [0.2, 0.8, 0.0]])
    counts = torch.tensor([2, 2])

    metrics = summarize_distribution_metrics(logits, targets, counts)

    first = torch.softmax(torch.tensor([2.0, 0.0], dtype=torch.float64), dim=0)
    second = torch.softmax(torch.tensor([0.0, 1.0], dtype=torch.float64), dim=0)
    expected_ce = (
        -(torch.tensor([0.7, 0.3], dtype=torch.float64) * first.log()).sum()
        -(torch.tensor([0.2, 0.8], dtype=torch.float64) * second.log()).sum()
    ).item() / 2
    assert metrics["questions"] == 2
    assert metrics["soft_target_cross_entropy_nats"] == pytest.approx(expected_ce)
    assert metrics["target_argmax_agreement"] == 1.0
    assert metrics["confidence"]["mean_soft_target_mass_at_prediction"] == pytest.approx(0.75)


def test_fit_uses_full_soft_targets_and_positive_scaling_preserves_argmax():
    logits = torch.tensor([[4.0, 0.0, 20.0], [0.0, 4.0, -20.0]])
    targets = torch.tensor([[0.65, 0.35, 0.0], [0.35, 0.65, 0.0]])
    counts = torch.tensor([2, 2])
    row_ids = ["cal-a", "cal-b"]
    split_by_id = {"cal-a": "calibration", "cal-b": "calibration"}

    record = fit_temperature(
        logits,
        targets,
        counts,
        row_ids=row_ids,
        split_by_id=split_by_id,
        group_ids_by_id=_groups(*row_ids),
        checkpoint_sha256=CHECKPOINT_SHA,
        split_manifest_sha256=SPLITS_SHA,
    )
    calibrated = apply_temperature(logits, counts, record["temperature"])

    assert record["fit_split"] == "calibration"
    assert record["fit_method"].endswith("full_distribution_soft_target_cross_entropy")
    assert record["temperature"] > 1.0
    assert record["calibrated"]["soft_target_cross_entropy_nats"] < record["uncalibrated"]["soft_target_cross_entropy_nats"]
    assert torch.equal(logits[:, :2].argmax(dim=-1), calibrated[:, :2].argmax(dim=-1))
    assert torch.isneginf(calibrated[:, 2]).all()
    assert record["calibration_row_ids_sha256"] != CHECKPOINT_SHA
    assert "cal-a" not in str(record) and "cal-b" not in str(record)


def test_fit_rejects_dev_or_test_rows_and_invalid_soft_target_mass():
    logits = torch.tensor([[1.0, 0.0]])
    targets = torch.tensor([[0.6, 0.4]])
    counts = torch.tensor([2])
    with pytest.raises(ValueError, match="calibration split"):
        fit_temperature(
            logits,
            targets,
            counts,
            row_ids=["heldout"],
            split_by_id={"heldout": "test"},
            group_ids_by_id=_groups("heldout"),
            checkpoint_sha256=CHECKPOINT_SHA,
        )
    with pytest.raises(ValueError, match="sum to one"):
        fit_temperature(
            logits,
            torch.tensor([[0.6, 0.3]]),
            counts,
            row_ids=["cal"],
            split_by_id={"cal": "calibration"},
            group_ids_by_id=_groups("cal"),
            checkpoint_sha256=CHECKPOINT_SHA,
        )
    rounded = fit_temperature(
        logits,
        torch.tensor([[0.500001, 0.5]]),
        counts,
        row_ids=["rounded-cal"],
        split_by_id={"rounded-cal": "calibration"},
        group_ids_by_id=_groups("rounded-cal"),
        checkpoint_sha256=CHECKPOINT_SHA,
    )
    assert math.isfinite(rounded["calibrated"]["soft_target_cross_entropy_nats"])


def test_final_test_uses_frozen_calibration_record_and_requires_matching_checkpoint():
    calibration_logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    calibration_targets = torch.tensor([[0.6, 0.4], [0.4, 0.6]])
    counts = torch.tensor([2, 2])
    calibration = fit_temperature(
        calibration_logits,
        calibration_targets,
        counts,
        row_ids=["cal-a", "cal-b"],
        split_by_id={"cal-a": "calibration", "cal-b": "calibration"},
        group_ids_by_id=_groups("cal-a", "cal-b"),
        checkpoint_sha256=CHECKPOINT_SHA,
        split_manifest_sha256=SPLITS_SHA,
    )

    test_logits = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    test_targets = torch.tensor([[0.4, 0.6], [0.7, 0.3]])
    test_ids = ["test-a", "test-b"]
    test_splits = {"cal-a": "calibration", "cal-b": "calibration", "test-a": "test", "test-b": "test"}
    test_result = evaluate_final_test(
        test_logits,
        test_targets,
        torch.tensor([2, 2]),
        row_ids=test_ids,
        source_by_id={"test-a": "original_replay", "test-b": "documents"},
        split_by_id=test_splits,
        group_ids_by_id=_groups("cal-a", "cal-b", *test_ids),
        checkpoint_sha256=CHECKPOINT_SHA,
        calibration_record=calibration,
        split_manifest_sha256=SPLITS_SHA,
    )

    assert test_result["split"] == "test"
    assert test_result["temperature"] == calibration["temperature"]
    assert test_result["metrics"]["calibrated"]["questions"] == 2
    assert test_result["metrics"]["uncalibrated"]["questions"] == 2
    assert test_result["cohort_metrics"]["original_replay"]["questions"] == 1
    assert test_result["cohort_metrics"]["fresh"]["questions"] == 1
    assert test_result["cohort_metrics"]["by_source"]["documents"]["questions"] == 1
    assert test_result["source_question_counts"] == {"documents": 1, "original_replay": 1}
    assert "ece" not in str(test_result).lower()
    assert "test-a" not in str(test_result)

    with pytest.raises(ValueError, match="checkpoint digests differ"):
        evaluate_final_test(
            test_logits,
            test_targets,
            torch.tensor([2, 2]),
            row_ids=test_ids,
            source_by_id={"test-a": "original_replay", "test-b": "documents"},
            split_by_id=test_splits,
            group_ids_by_id=_groups("cal-a", "cal-b", *test_ids),
            checkpoint_sha256="c" * 64,
            calibration_record=calibration,
        )
    with pytest.raises(ValueError, match="test split"):
        evaluate_final_test(
            calibration_logits[:1],
            calibration_targets[:1],
            counts[:1],
            row_ids=["cal-a"],
            source_by_id={"cal-a": "original_replay"},
            split_by_id=test_splits,
            group_ids_by_id=_groups("cal-a", "cal-b", *test_ids),
            checkpoint_sha256=CHECKPOINT_SHA,
            calibration_record=calibration,
        )


def test_group_leakage_across_any_snapshot_splits_is_rejected():
    with pytest.raises(ValueError, match="assigned wholly to one split"):
        fit_temperature(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[0.6, 0.4]]),
            torch.tensor([2]),
            row_ids=["cal"],
            split_by_id={"cal": "calibration", "test": "test"},
            group_ids_by_id={
                "cal": ["problem:shared", "snapshot:cal"],
                "test": ["problem:shared", "snapshot:test"],
            },
            checkpoint_sha256=CHECKPOINT_SHA,
        )


def test_265_capacity_mask_is_preserved_and_temperature_is_positive():
    logits = torch.zeros((1, 265))
    logits[0, 1] = 1.0
    logits[0, 264] = 1000.0
    counts = torch.tensor([2])

    scaled = apply_temperature(logits, counts, 2.0)

    assert scaled.shape == (1, 265)
    assert scaled[0, :2].argmax().item() == 1
    assert torch.isneginf(scaled[0, 2:]).all()
    with pytest.raises(ValueError, match="finite positive"):
        apply_temperature(logits, counts, 0.0)
