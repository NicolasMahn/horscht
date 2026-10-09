"""Fit and apply one scalar temperature using grouped soft-target calibration rows.

The module consumes logits from a frozen checkpoint and full categorical target
distributions. It never selects choices, reconstructs labels from argmax, or
returns per-row outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import Tensor

from dataset_factory.decision_metrics import categorical_brier_sum


CAPACITY = 265
CALIBRATION_SCHEMA = "soft-target-temperature-v1"
DEFAULT_MIN_TEMPERATURE = 0.05
DEFAULT_MAX_TEMPERATURE = 20.0
_TARGET_TOLERANCE = 2e-6
FINAL_TEST_COHORT_INTERPRETATION = (
    "original_replay contains historically reused test rows and is exploratory; fresh contains new source groups. "
    "Combined metrics include both cohorts."
)


def apply_temperature(logits: Tensor, counts: Tensor, temperature: float) -> Tensor:
    """Scale valid choice logits and keep padded choices masked at ``-inf``."""
    scores, _, valid_mask = _validated_batch(logits, None, counts)
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("temperature must be a finite positive scalar")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive scalar")
    scaled = scores / temperature
    return scaled.masked_fill(~valid_mask, -torch.inf)


def summarize_distribution_metrics(
    logits: Tensor,
    targets: Tensor,
    counts: Tensor,
    *,
    temperature: float = 1.0,
) -> dict[str, Any]:
    """Return aggregate proper scores and descriptive confidence summaries.

    Confidence is summarized against the recorded target distribution. These
    comparisons do not establish empirical calibration or truth when targets
    are estimates or constructed distributions.
    """
    scores, target, _ = _validated_batch(logits, targets, counts)
    if target is None:
        raise ValueError("target distributions are required for scoring")
    scaled = apply_temperature(scores, torch.as_tensor(counts), temperature)
    question_metrics: list[tuple[float, float, int, float, float]] = []
    for row_index, count in enumerate(_choice_counts(counts, scores.shape[0], scores.shape[1])):
        row_logits = scaled[row_index, :count]
        row_target = target[row_index, :count]
        log_probabilities = torch.log_softmax(row_logits, dim=-1)
        probabilities = log_probabilities.exp()
        target_values = row_target.tolist()
        probability_values = probabilities.tolist()
        # Use log-softmax directly so small probabilities do not underflow to zero.
        cross_entropy = -(row_target * log_probabilities).sum().item()
        brier = categorical_brier_sum(target_values, probability_values)
        prediction = int(row_logits.argmax().item())
        target_argmax = int(row_target.argmax().item())
        confidence = probabilities.max().item()
        target_mass_at_prediction = row_target[prediction].item()
        question_metrics.append(
            (cross_entropy, brier, int(prediction == target_argmax), confidence,
             target_mass_at_prediction)
        )

    count = len(question_metrics)
    confidences = sorted(row[3] for row in question_metrics)
    mean_confidence = math.fsum(confidences) / count
    mean_target_mass = math.fsum(row[4] for row in question_metrics) / count
    return {
        "questions": count,
        "soft_target_cross_entropy_nats": math.fsum(row[0] for row in question_metrics) / count,
        "categorical_brier_sum": math.fsum(row[1] for row in question_metrics) / count,
        "target_argmax_agreement": math.fsum(row[2] for row in question_metrics) / count,
        "confidence": {
            "mean_max_probability": mean_confidence,
            "median_max_probability": _quantile(confidences, 0.5),
            "p90_max_probability": _quantile(confidences, 0.9),
            "fraction_max_probability_at_least_0_5": sum(value >= 0.5 for value in confidences) / count,
            "fraction_max_probability_at_least_0_8": sum(value >= 0.8 for value in confidences) / count,
            "fraction_max_probability_at_least_0_95": sum(value >= 0.95 for value in confidences) / count,
            "mean_soft_target_mass_at_prediction": mean_target_mass,
            "mean_confidence_minus_soft_target_mass": mean_confidence - mean_target_mass,
            "interpretation": (
                "Confidence is compared with recorded target mass at the predicted choice; "
                "this is not empirical correctness or a claim of truth."
            ),
        },
    }


def fit_temperature(
    logits: Tensor,
    targets: Tensor,
    counts: Tensor,
    *,
    row_ids: Sequence[str],
    split_by_id: Mapping[str, str],
    group_ids_by_id: Mapping[str, Sequence[str]],
    checkpoint_sha256: str,
    split_manifest_sha256: str | None = None,
    min_temperature: float = DEFAULT_MIN_TEMPERATURE,
    max_temperature: float = DEFAULT_MAX_TEMPERATURE,
) -> dict[str, Any]:
    """Fit a positive scalar on calibration rows using their full target vectors.

    The bounded one-dimensional objective is convex in inverse temperature.
    Its derivative is monotone, so bisection finds the minimum deterministically.
    A boundary fit is recorded explicitly.
    """
    scores, target, _ = _validated_batch(logits, targets, counts)
    ids = _validate_row_provenance(
        row_ids,
        split_by_id,
        group_ids_by_id,
        expected_split="calibration",
        expected_count=scores.shape[0],
    )
    _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    if split_manifest_sha256 is not None:
        _validate_sha256(split_manifest_sha256, "split_manifest_sha256")
    lower, upper = _validate_temperature_bounds(min_temperature, max_temperature)

    valid_counts = _choice_counts(counts, scores.shape[0], scores.shape[1])
    target_scores = torch.stack(
        [(target[index, :count] * scores[index, :count]).sum()
         for index, count in enumerate(valid_counts)]
    )

    def derivative(inverse_temperature: float) -> float:
        per_question: list[Tensor] = []
        for index, count in enumerate(valid_counts):
            valid_logits = scores[index, :count]
            probabilities = torch.softmax(valid_logits * inverse_temperature, dim=-1)
            expected_score = (probabilities * valid_logits).sum()
            per_question.append(expected_score - target_scores[index])
        return torch.stack(per_question).mean().item()

    inverse_low = 1.0 / upper
    inverse_high = 1.0 / lower
    derivative_low = derivative(inverse_low)
    derivative_high = derivative(inverse_high)
    if derivative_low >= 0:
        temperature = upper
        boundary = "max_temperature"
    elif derivative_high <= 0:
        temperature = lower
        boundary = "min_temperature"
    else:
        left, right = inverse_low, inverse_high
        for _ in range(96):
            midpoint = (left + right) / 2.0
            slope = derivative(midpoint)
            if abs(slope) <= 1e-12:
                left = right = midpoint
                break
            if slope < 0:
                left = midpoint
            else:
                right = midpoint
        temperature = 1.0 / ((left + right) / 2.0)
        boundary = None

    uncalibrated = summarize_distribution_metrics(scores, target, counts)
    calibrated = summarize_distribution_metrics(scores, target, counts, temperature=temperature)
    result: dict[str, Any] = {
        "schema_version": CALIBRATION_SCHEMA,
        "fit_method": "bounded_scalar_minimum_full_distribution_soft_target_cross_entropy",
        "fit_split": "calibration",
        "questions": len(ids),
        "calibration_row_ids_sha256": _ids_sha256(ids),
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "temperature": temperature,
        "temperature_bounds": {"min": lower, "max": upper},
        "fit_at_boundary": boundary is not None,
        "boundary": boundary,
        "uncalibrated": uncalibrated,
        "calibrated": calibrated,
    }
    if split_manifest_sha256 is not None:
        result["split_manifest_sha256"] = split_manifest_sha256.lower()
    return result


def evaluate_final_test(
    logits: Tensor,
    targets: Tensor,
    counts: Tensor,
    *,
    row_ids: Sequence[str],
    source_by_id: Mapping[str, str],
    split_by_id: Mapping[str, str],
    group_ids_by_id: Mapping[str, Sequence[str]],
    checkpoint_sha256: str,
    calibration_record: Mapping[str, Any],
    split_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Score the held-out test once with a frozen checkpoint and calibration record.

    This function has no fitting path. Callers should persist its aggregate
    result with create-once semantics so a final test is not repeatedly reused.
    """
    scores, target, _ = _validated_batch(logits, targets, counts)
    ids = _validate_row_provenance(
        row_ids,
        split_by_id,
        group_ids_by_id,
        expected_split="test",
        expected_count=scores.shape[0],
    )
    source_ids = _validate_source_provenance(ids, source_by_id)
    _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    if split_manifest_sha256 is not None:
        _validate_sha256(split_manifest_sha256, "split_manifest_sha256")
    temperature = validate_frozen_calibration_record(
        calibration_record, checkpoint_sha256, split_manifest_sha256=split_manifest_sha256
    )
    original_indices = [index for index, identifier in enumerate(ids) if source_ids[identifier] == "original_replay"]
    fresh_indices = [index for index, identifier in enumerate(ids) if source_ids[identifier] != "original_replay"]
    if not original_indices or not fresh_indices:
        raise ValueError("final test must include both original_replay and fresh source cohorts")

    def metric_pair(indices: Sequence[int]) -> dict[str, Any]:
        index_tensor = torch.tensor(indices, dtype=torch.int64)
        selected_scores = scores.index_select(0, index_tensor)
        selected_targets = target.index_select(0, index_tensor)
        selected_counts = torch.as_tensor(counts, device="cpu").index_select(0, index_tensor)
        return {
            "questions": len(indices),
            "uncalibrated": summarize_distribution_metrics(selected_scores, selected_targets, selected_counts),
            "calibrated": summarize_distribution_metrics(
                selected_scores, selected_targets, selected_counts, temperature=temperature
            ),
        }

    by_source = {
        source: metric_pair([index for index, identifier in enumerate(ids) if source_ids[identifier] == source])
        for source in sorted(set(source_ids.values()))
    }
    source_counts = {source: len(indices) for source, indices in (
        (source, [index for index, identifier in enumerate(ids) if source_ids[identifier] == source])
        for source in sorted(set(source_ids.values()))
    )}
    result: dict[str, Any] = {
        "schema_version": "soft-target-final-test-v1",
        "split": "test",
        "questions": len(ids),
        "test_row_ids_sha256": _ids_sha256(ids),
        "source_question_counts": source_counts,
        "checkpoint_sha256": checkpoint_sha256.lower(),
        "temperature": temperature,
        "calibration_row_ids_sha256": calibration_record["calibration_row_ids_sha256"],
        "calibration_record_sha256": calibration_record_sha256(calibration_record),
        "metrics": {
            "uncalibrated": summarize_distribution_metrics(scores, target, counts),
            "calibrated": summarize_distribution_metrics(scores, target, counts, temperature=temperature),
        },
        "cohort_metrics": {
            "original_replay": metric_pair(original_indices),
            "fresh": metric_pair(fresh_indices),
            "by_source": by_source,
        },
        "cohort_interpretation": FINAL_TEST_COHORT_INTERPRETATION,
    }
    if split_manifest_sha256 is not None:
        result["split_manifest_sha256"] = split_manifest_sha256.lower()
    return result


def _validate_source_provenance(row_ids: Sequence[str], source_by_id: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(source_by_id, Mapping) or set(source_by_id) != set(row_ids):
        raise ValueError("source metadata must cover exactly the final-test rows")
    result = dict(source_by_id)
    if any(not isinstance(source, str) or not source.strip() for source in result.values()):
        raise ValueError("each final-test row must have one nonempty source ID")
    return result


def _validate_calibration_record(record: Mapping[str, Any], checkpoint_sha256: str) -> float:
    if record.get("schema_version") != CALIBRATION_SCHEMA or record.get("fit_split") != "calibration":
        raise ValueError("final test requires a calibration-split temperature record")
    record_checkpoint = record.get("checkpoint_sha256")
    _validate_sha256(record_checkpoint, "calibration checkpoint_sha256")
    if record_checkpoint.lower() != checkpoint_sha256.lower():
        raise ValueError("calibration and test checkpoint digests differ")
    _validate_sha256(record.get("calibration_row_ids_sha256"), "calibration_row_ids_sha256")
    temperature = record.get("temperature")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise ValueError("calibration temperature is invalid")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("calibration temperature is invalid")
    return temperature


def validate_frozen_calibration_record(
    record: Mapping[str, Any],
    checkpoint_sha256: str,
    *,
    split_manifest_sha256: str | None = None,
) -> float:
    """Validate a saved temperature's checkpoint and split-manifest binding."""
    _validate_sha256(checkpoint_sha256, "checkpoint_sha256")
    if split_manifest_sha256 is not None:
        _validate_sha256(split_manifest_sha256, "split_manifest_sha256")
        if record.get("split_manifest_sha256") != split_manifest_sha256.lower():
            raise ValueError("test split manifest differs from the calibration split manifest")
    return _validate_calibration_record(record, checkpoint_sha256)


def _validated_batch(
    logits: Tensor,
    targets: Tensor | None,
    counts: Tensor,
) -> tuple[Tensor, Tensor | None, Tensor]:
    scores = torch.as_tensor(logits, dtype=torch.float64, device="cpu")
    count_values = torch.as_tensor(counts, device="cpu")
    if scores.ndim != 2 or scores.shape[0] < 1 or not 2 <= scores.shape[1] <= CAPACITY:
        raise ValueError("logits must have shape [questions, choices] with 2–265 choices")
    if count_values.ndim != 1 or count_values.shape[0] != scores.shape[0]:
        raise ValueError("choice counts must have one value per question")
    if count_values.dtype == torch.bool or count_values.is_floating_point() or count_values.is_complex():
        raise ValueError("choice counts must be integers")
    valid_counts = _choice_counts(count_values, scores.shape[0], scores.shape[1])
    valid_mask = torch.arange(scores.shape[1])[None, :] < count_values.to(torch.int64)[:, None]
    for index, count in enumerate(valid_counts):
        if not torch.isfinite(scores[index, :count]).all():
            raise ValueError("valid-choice logits must be finite")

    target_values: Tensor | None = None
    if targets is not None:
        target_values = torch.as_tensor(targets, dtype=torch.float64, device="cpu")
        if target_values.shape != scores.shape:
            raise ValueError("target distributions must match the logits shape")
        if not torch.isfinite(target_values).all():
            raise ValueError("target distributions must be finite")
        if (target_values < 0).any() or (target_values > 1).any():
            raise ValueError("target probabilities must be in [0, 1]")
        if (target_values.masked_select(~valid_mask) != 0).any():
            raise ValueError("padded choices must have zero target mass")
        for index, count in enumerate(valid_counts):
            total = target_values[index, :count].sum().item()
            if abs(total - 1.0) > _TARGET_TOLERANCE:
                raise ValueError("each valid target distribution must sum to one")
        # Correct only representation-level sum drift after the normalization check.
        target_values = target_values / target_values.sum(dim=1, keepdim=True)
    return scores, target_values, valid_mask


def _choice_counts(counts: Tensor, rows: int, width: int) -> list[int]:
    values = torch.as_tensor(counts, device="cpu")
    if values.ndim != 1 or values.shape[0] != rows:
        raise ValueError("choice counts must have one value per question")
    if values.dtype == torch.bool or values.is_floating_point() or values.is_complex():
        raise ValueError("choice counts must be integers")
    result = [int(value) for value in values.tolist()]
    if any(value < 2 or value > width for value in result):
        raise ValueError("choice counts must be between 2 and the logit width")
    return result


def _validate_row_provenance(
    row_ids: Sequence[str],
    split_by_id: Mapping[str, str],
    group_ids_by_id: Mapping[str, Sequence[str]],
    *,
    expected_split: str,
    expected_count: int,
) -> list[str]:
    ids = list(row_ids)
    if len(ids) != expected_count or any(not isinstance(value, str) or not value.strip() for value in ids):
        raise ValueError("row IDs must match the batch and be nonempty strings")
    if len(ids) != len(set(ids)):
        raise ValueError("row IDs must be unique")
    if not isinstance(split_by_id, Mapping) or any(split_by_id.get(identifier) != expected_split for identifier in ids):
        raise ValueError(f"all rows must belong to the {expected_split} split")
    validate_group_split_assignments(split_by_id, group_ids_by_id)
    return ids


def validate_group_split_assignments(
    split_by_id: Mapping[str, str],
    group_ids_by_id: Mapping[str, Sequence[str]],
) -> None:
    """Reject incomplete group metadata or any group assigned across splits."""
    if not isinstance(group_ids_by_id, Mapping) or set(group_ids_by_id) != set(split_by_id):
        raise ValueError("group metadata must cover the complete split mapping")
    if any(split not in {"train", "dev", "calibration", "test"} for split in split_by_id.values()):
        raise ValueError("split mapping contains an unsupported split")
    group_split: dict[str, str] = {}
    for identifier, split in split_by_id.items():
        groups = group_ids_by_id[identifier]
        if (
            not isinstance(groups, Sequence)
            or isinstance(groups, (str, bytes))
            or not groups
            or any(not isinstance(group, str) or not group.strip() for group in groups)
        ):
            raise ValueError("each split row must have nonempty group IDs")
        for group in set(groups):
            assigned_split = group_split.setdefault(group, split)
            if assigned_split != split:
                raise ValueError("problem groups must be assigned wholly to one split")


def _validate_sha256(value: Any, field: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{field} must be a SHA-256 hex digest")


def _validate_temperature_bounds(minimum: float, maximum: float) -> tuple[float, float]:
    if (
        isinstance(minimum, bool)
        or isinstance(maximum, bool)
        or not isinstance(minimum, (int, float))
        or not isinstance(maximum, (int, float))
    ):
        raise ValueError("temperature bounds must be finite positive scalars")
    lower, upper = float(minimum), float(maximum)
    if not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0 or lower >= upper:
        raise ValueError("temperature bounds must satisfy 0 < min < max")
    return lower, upper


def _ids_sha256(ids: Sequence[str]) -> str:
    digest = hashlib.sha256("\n".join(sorted(ids)).encode("utf-8")).hexdigest()
    return digest


def _quantile(sorted_values: Sequence[float], quantile: float) -> float:
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def calibration_record_sha256(record: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible calibration record for downstream provenance."""
    canonical = json.dumps(dict(record), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
