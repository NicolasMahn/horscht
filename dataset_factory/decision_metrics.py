"""Pure evaluation metrics for probability-v2 choices and rejection policies.

Task-grounded abstention is a normal choice in the categorical distribution.
Confidence rejection is an external action represented by ``None`` and always
has utility zero.  This module does not calibrate model probabilities.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


_ANSWERABILITY_STATUSES = frozenset({"answerable", "unanswerable"})
_DISTRIBUTION_TOLERANCE = 1e-9


def categorical_cross_entropy(
    target_distribution: Sequence[float], predicted_distribution: Sequence[float]
) -> float:
    """Return categorical cross-entropy in nats, including soft targets.

    A zero target mass contributes zero even when its predicted probability is
    also zero. Positive target mass assigned zero predicted probability yields
    positive infinity, as required by the logarithmic score.
    """
    target, predicted = _validate_distributions(target_distribution, predicted_distribution)
    terms: list[float] = []
    for target_mass, predicted_probability in zip(target, predicted):
        if target_mass == 0:
            continue
        if predicted_probability == 0:
            return math.inf
        terms.append(-target_mass * math.log(predicted_probability))
    return math.fsum(terms)


def categorical_brier_sum(
    target_distribution: Sequence[float], predicted_distribution: Sequence[float]
) -> float:
    """Return the summed multiclass Brier score, without dividing by classes."""
    target, predicted = _validate_distributions(target_distribution, predicted_distribution)
    return math.fsum(
        (predicted_probability - target_mass) ** 2
        for target_mass, predicted_probability in zip(target, predicted)
    )


def expected_decision_utility(
    target_distribution: Sequence[float],
    action: int | str | None,
    abstention_choice: int | str,
    answerability_status: str,
    *,
    wrong_answer_cost: float,
    choice_ids: Sequence[str] | None = None,
) -> float:
    """Score a selected choice or external rejection against a target distribution.

    Correct categorical choices earn ``+1`` and wrong choices cost
    ``wrong_answer_cost``. ``None`` means confidence rejection and earns zero.
    The status and abstention choice must agree with the target: answerable
    targets assign zero mass to the task IDK class, and unanswerable targets
    put all mass on it.
    """
    target = _validate_distribution(target_distribution, "target_distribution")
    abstention_index = _resolve_choice_index(abstention_choice, len(target), choice_ids)
    _validate_answerability_target(target, abstention_index, answerability_status)
    penalty = _validate_wrong_answer_cost(wrong_answer_cost)

    if action is None:
        return 0.0
    action_index = _resolve_choice_index(action, len(target), choice_ids)
    return math.fsum(
        target_mass * (1.0 if index == action_index else -penalty)
        for index, target_mass in enumerate(target)
    )


def threshold_action(
    predicted_distribution: Sequence[float],
    abstention_choice: int | str,
    *,
    wrong_answer_cost: float,
    choice_ids: Sequence[str] | None = None,
) -> int | None:
    """Choose the best substantive class at threshold ``cost / (1 + cost)``.

    The task-grounded abstention class is excluded from this confidence check.
    Return its selected substantive choice index, or ``None`` for external
    rejection. A probability exactly on the threshold is accepted. The caller
    supplies the probabilities; this function does not calibrate them.
    """
    predicted = _validate_distribution(predicted_distribution, "predicted_distribution")
    abstention_index = _resolve_choice_index(abstention_choice, len(predicted), choice_ids)
    penalty = _validate_wrong_answer_cost(wrong_answer_cost)
    threshold = penalty / (1.0 + penalty)
    substantive_indices = [index for index in range(len(predicted)) if index != abstention_index]
    best_index = max(substantive_indices, key=predicted.__getitem__)
    return best_index if predicted[best_index] >= threshold else None


def evaluate_decision_row(
    target_distribution: Sequence[float],
    predicted_distribution: Sequence[float],
    abstention_choice: int | str,
    answerability_status: str,
    *,
    wrong_answer_cost: float,
    choice_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return proper scores, task-class diagnostics, and rejection-policy utility.

    ``task_argmax_expected_utility`` scores the highest-probability class across
    the entire vector, including task IDK. ``threshold_expected_utility`` scores
    the substantive-only threshold policy, whose rejected action is external.
    """
    target, predicted = _validate_distributions(target_distribution, predicted_distribution)
    abstention_index = _resolve_choice_index(abstention_choice, len(target), choice_ids)
    _validate_answerability_target(target, abstention_index, answerability_status)
    penalty = _validate_wrong_answer_cost(wrong_answer_cost)

    task_argmax_index = max(range(len(predicted)), key=predicted.__getitem__)
    substantive_indices = [index for index in range(len(predicted)) if index != abstention_index]
    always_answer_index = max(substantive_indices, key=predicted.__getitem__)
    policy_action = threshold_action(
        predicted,
        abstention_index,
        wrong_answer_cost=penalty,
    )
    task_argmax_utility = _utility_for_index(target, task_argmax_index, penalty)
    policy_utility = (
        0.0 if policy_action is None else _utility_for_index(target, policy_action, penalty)
    )
    always_answer_utility = _utility_for_index(target, always_answer_index, penalty)

    answerable = answerability_status == "answerable"
    unanswerable = answerability_status == "unanswerable"
    return {
        "categorical_cross_entropy": categorical_cross_entropy(target, predicted),
        "categorical_brier_sum": categorical_brier_sum(target, predicted),
        "task_argmax_choice_index": task_argmax_index,
        "task_argmax_expected_utility": task_argmax_utility,
        "threshold": penalty / (1.0 + penalty),
        "threshold_action_choice_index": policy_action,
        "threshold_rejected": policy_action is None,
        "threshold_expected_utility": policy_utility,
        "always_answer_choice_index": always_answer_index,
        "always_answer_expected_utility": always_answer_utility,
        "all_reject_expected_utility": 0.0,
        "answerable_covered": (policy_action is not None) if answerable else None,
        "unanswerable_idk_correct": (task_argmax_index == abstention_index) if unanswerable else None,
        "false_idk": (task_argmax_index == abstention_index) if answerable else None,
    }


def evaluate_decision_rows(
    rows: Sequence[Mapping[str, Any]], *, wrong_answer_cost: float
) -> dict[str, Any]:
    """Aggregate proper scores, rejection metrics, task-IDK diagnostics, and baselines.

    Each row needs ``predicted_distribution``, ``answerability_status``, and
    either ``target_distribution`` or a hard-label ``answer_index``. The
    abstention choice can be given as ``abstention_choice_id`` alongside
    ``choice_ids``, as ``abstention_choice_index``, or as ``abstention_choice``.
    """
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or not rows:
        raise ValueError("rows must be a non-empty sequence of decision rows")
    penalty = _validate_wrong_answer_cost(wrong_answer_cost)
    evaluated: list[dict[str, Any]] = []
    statuses: list[str] = []
    selective_predictions: list[tuple[float, float] | None] = []
    source_kinds: list[str | None] = []
    for row_number, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"rows[{row_number}] must be a mapping")
        choice_ids = row.get("choice_ids")
        predicted = row.get("predicted_distribution")
        if "target_distribution" in row:
            target = row["target_distribution"]
        else:
            target = _one_hot_target(row.get("answer_index"), predicted)
        if "abstention_choice_id" in row:
            abstention_choice = row["abstention_choice_id"]
        elif "abstention_choice_index" in row:
            abstention_choice = row["abstention_choice_index"]
        elif "abstention_choice" in row:
            abstention_choice = row["abstention_choice"]
        else:
            raise ValueError(f"rows[{row_number}] is missing an abstention choice ID or index")
        try:
            row_metrics = evaluate_decision_row(
                target,
                predicted,
                abstention_choice,
                row.get("answerability_status"),
                wrong_answer_cost=penalty,
                choice_ids=choice_ids,
            )
        except ValueError as error:
            raise ValueError(f"rows[{row_number}]: {error}") from error
        evaluated.append(row_metrics)
        status = row["answerability_status"]
        statuses.append(status)

        source_kind = row.get("distribution_source_kind")
        if source_kind is not None and (not isinstance(source_kind, str) or not source_kind):
            raise ValueError(
                f"rows[{row_number}].distribution_source_kind must be a non-empty string"
            )
        source_kinds.append(source_kind)

        if status == "answerable":
            substantive_index = row_metrics["always_answer_choice_index"]
            selective_predictions.append(
                (
                    float(predicted[substantive_index]),
                    1.0 - float(target[substantive_index]),
                )
            )
        else:
            selective_predictions.append(None)

    report = _summarize_decision_rows(evaluated, statuses, selective_predictions, penalty)
    grouped_indices: dict[str | None, list[int]] = {}
    for index, source_kind in enumerate(source_kinds):
        grouped_indices.setdefault(source_kind, []).append(index)
    report["source_kind_stratification"] = [
        {
            "distribution_source_kind": source_kind,
            **_summarize_decision_rows(
                [evaluated[index] for index in indices],
                [statuses[index] for index in indices],
                [selective_predictions[index] for index in indices],
                penalty,
            ),
        }
        for source_kind, indices in sorted(
            grouped_indices.items(), key=lambda item: (item[0] is None, item[0] or "")
        )
    ]
    return report


def _summarize_decision_rows(
    evaluated: Sequence[Mapping[str, Any]],
    statuses: Sequence[str],
    selective_predictions: Sequence[tuple[float, float] | None],
    penalty: float,
) -> dict[str, Any]:
    count = len(evaluated)
    answerable_metrics = [
        metric for metric, status in zip(evaluated, statuses) if status == "answerable"
    ]
    unanswerable_metrics = [
        metric for metric, status in zip(evaluated, statuses) if status == "unanswerable"
    ]
    answerable_coverage = (
        sum(metric["answerable_covered"] for metric in answerable_metrics) / len(answerable_metrics)
        if answerable_metrics
        else None
    )
    unanswerable_idk_recall = (
        sum(metric["unanswerable_idk_correct"] for metric in unanswerable_metrics)
        / len(unanswerable_metrics)
        if unanswerable_metrics
        else None
    )
    false_idk_rate = (
        sum(metric["false_idk"] for metric in answerable_metrics) / len(answerable_metrics)
        if answerable_metrics
        else None
    )
    covered_answerable_errors = [
        prediction[1]
        for metric, status, prediction in zip(evaluated, statuses, selective_predictions)
        if status == "answerable"
        and metric["answerable_covered"]
        and prediction is not None
    ]
    selective_risk = (
        math.fsum(covered_answerable_errors) / len(covered_answerable_errors)
        if covered_answerable_errors
        else None
    )

    return {
        "questions": count,
        "wrong_answer_cost": penalty,
        "threshold": penalty / (1.0 + penalty),
        "proper_scores": {
            "categorical_cross_entropy": _mean(
                metric["categorical_cross_entropy"] for metric in evaluated
            ),
            "categorical_brier_sum": _mean(
                metric["categorical_brier_sum"] for metric in evaluated
            ),
        },
        "decision_metrics": {
            "expected_utility": _mean(
                metric["threshold_expected_utility"] for metric in evaluated
            ),
            "answerable_rows": len(answerable_metrics),
            "answerable_coverage": answerable_coverage,
            "covered_answerable_rows": len(covered_answerable_errors),
            "selective_risk": selective_risk,
            "unanswerable_rows": len(unanswerable_metrics),
            "unanswerable_idk_recall": unanswerable_idk_recall,
            "false_idk_rate": false_idk_rate,
            "task_argmax_expected_utility": _mean(
                metric["task_argmax_expected_utility"] for metric in evaluated
            ),
        },
        "baselines": {
            "always_answer_expected_utility": _mean(
                metric["always_answer_expected_utility"] for metric in evaluated
            ),
            "all_reject_expected_utility": _mean(
                metric["all_reject_expected_utility"] for metric in evaluated
            ),
        },
        "risk_coverage_curve": _risk_coverage_curve(statuses, selective_predictions),
    }


def _risk_coverage_curve(
    statuses: Sequence[str],
    selective_predictions: Sequence[tuple[float, float] | None],
) -> list[dict[str, Any]]:
    """Build confidence-threshold points, adding equal-confidence rows together."""
    errors_by_confidence: dict[float, list[float]] = {}
    answerable_count = 0
    for status, prediction in zip(statuses, selective_predictions):
        if status != "answerable":
            continue
        answerable_count += 1
        if prediction is None:
            raise ValueError("answerable rows need a substantive prediction for risk metrics")
        confidence, error_mass = prediction
        errors_by_confidence.setdefault(confidence, []).append(error_mass)

    if not answerable_count:
        return []

    curve = [
        {
            "confidence_threshold": None,
            "covered_answerable_rows": 0,
            "answerable_coverage": 0.0,
            "selective_risk": None,
        }
    ]
    covered = 0
    cumulative_group_errors: list[float] = []
    for confidence in sorted(errors_by_confidence, reverse=True):
        # Sorting makes aggregation independent of input order within ties.
        group_errors = math.fsum(sorted(errors_by_confidence[confidence]))
        cumulative_group_errors.append(group_errors)
        covered += len(errors_by_confidence[confidence])
        curve.append(
            {
                "confidence_threshold": confidence,
                "covered_answerable_rows": covered,
                "answerable_coverage": covered / answerable_count,
                "selective_risk": math.fsum(cumulative_group_errors) / covered,
            }
        )
    return curve


def _validate_distributions(
    target_distribution: Sequence[float], predicted_distribution: Sequence[float]
) -> tuple[list[float], list[float]]:
    target = _validate_distribution(target_distribution, "target_distribution")
    predicted = _validate_distribution(predicted_distribution, "predicted_distribution")
    if len(target) != len(predicted):
        raise ValueError("target and predicted distributions must have the same length")
    return target, predicted


def _validate_distribution(values: Sequence[float], name: str) -> list[float]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a categorical probability sequence")
    if len(values) < 2:
        raise ValueError(f"{name} must contain at least two choices")
    converted: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name}[{index}] must be a finite number")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name}[{index}] must be a finite number")
        if numeric < 0.0 or numeric > 1.0:
            raise ValueError(f"{name}[{index}] must be between zero and one")
        converted.append(numeric)
    if not math.isclose(
        math.fsum(converted), 1.0, rel_tol=0.0, abs_tol=_DISTRIBUTION_TOLERANCE
    ):
        raise ValueError(f"{name} must sum to one")
    return converted


def _resolve_choice_index(
    choice: int | str, choice_count: int, choice_ids: Sequence[str] | None
) -> int:
    if isinstance(choice, bool):
        raise ValueError("choice index must be an integer, not a boolean")
    if isinstance(choice, int):
        if not 0 <= choice < choice_count:
            raise ValueError("choice index is outside the probability vector")
        return choice
    if not isinstance(choice, str) or not choice:
        raise ValueError("choice must be a valid index or non-empty choice ID")
    if choice_ids is None:
        raise ValueError("choice_ids are required to resolve a choice ID")
    if (
        not isinstance(choice_ids, Sequence)
        or isinstance(choice_ids, (str, bytes))
        or len(choice_ids) != choice_count
        or any(not isinstance(value, str) or not value for value in choice_ids)
        or len(set(choice_ids)) != len(choice_ids)
    ):
        raise ValueError("choice_ids must be unique non-empty IDs parallel to the choices")
    try:
        return choice_ids.index(choice)
    except ValueError as error:
        raise ValueError("choice ID is not present in choice_ids") from error


def _validate_answerability_target(
    target: Sequence[float], abstention_index: int, answerability_status: str
) -> None:
    if not isinstance(answerability_status, str) or answerability_status not in _ANSWERABILITY_STATUSES:
        raise ValueError("answerability_status must be 'answerable' or 'unanswerable'")
    if answerability_status == "answerable":
        if target[abstention_index] > _DISTRIBUTION_TOLERANCE:
            raise ValueError("answerable target must assign zero mass to task abstention")
    elif not math.isclose(
        target[abstention_index], 1.0, rel_tol=0.0, abs_tol=_DISTRIBUTION_TOLERANCE
    ):
        raise ValueError("unanswerable target must be one-hot on task abstention")


def _validate_wrong_answer_cost(wrong_answer_cost: float) -> float:
    if (
        isinstance(wrong_answer_cost, bool)
        or not isinstance(wrong_answer_cost, (int, float))
        or not math.isfinite(float(wrong_answer_cost))
        or wrong_answer_cost < 0
    ):
        raise ValueError("wrong_answer_cost must be a finite non-negative number")
    return float(wrong_answer_cost)


def _utility_for_index(target: Sequence[float], action_index: int, penalty: float) -> float:
    return math.fsum(
        target_mass * (1.0 if index == action_index else -penalty)
        for index, target_mass in enumerate(target)
    )


def _one_hot_target(answer_index: Any, predicted_distribution: Any) -> list[float]:
    if isinstance(answer_index, bool) or not isinstance(answer_index, int):
        raise ValueError("row must contain target_distribution or an integer answer_index")
    if not isinstance(predicted_distribution, Sequence) or isinstance(
        predicted_distribution, (str, bytes)
    ):
        raise ValueError("predicted_distribution must be a categorical probability sequence")
    if not 0 <= answer_index < len(predicted_distribution):
        raise ValueError("answer_index is outside the predicted probability vector")
    return [float(index == answer_index) for index in range(len(predicted_distribution))]


def _mean(values: Any) -> float:
    values = list(values)
    return math.fsum(values) / len(values)
