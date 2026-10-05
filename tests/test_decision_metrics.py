import math

import pytest

from dataset_factory.decision_metrics import (
    categorical_brier_sum,
    categorical_cross_entropy,
    evaluate_decision_row,
    evaluate_decision_rows,
    expected_decision_utility,
    threshold_action,
)


def test_soft_target_proper_scores_and_expected_utility_are_analytic():
    target = [0.75, 0.25, 0.0]
    predicted = [0.5, 0.25, 0.25]

    assert categorical_cross_entropy(target, predicted) == pytest.approx(
        0.75 * math.log(2) + 0.25 * math.log(4)
    )
    assert categorical_brier_sum(target, predicted) == pytest.approx(0.125)
    assert expected_decision_utility(
        target,
        0,
        2,
        "answerable",
        wrong_answer_cost=2,
    ) == pytest.approx(0.25)


def test_hard_one_hot_target_scores_like_a_categorical_label():
    target = [0.0, 1.0, 0.0]
    predicted = [0.2, 0.7, 0.1]

    assert categorical_cross_entropy(target, predicted) == pytest.approx(-math.log(0.7))
    assert categorical_brier_sum(target, predicted) == pytest.approx(0.14)
    assert expected_decision_utility(
        target,
        1,
        2,
        "answerable",
        wrong_answer_cost=0.5,
    ) == 1.0
    assert expected_decision_utility(
        target,
        0,
        2,
        "answerable",
        wrong_answer_cost=0.5,
    ) == -0.5


def test_task_grounded_idk_is_a_correct_class_for_unanswerable_rows():
    target = [0.0, 0.0, 1.0]
    abstention_id = "idk"
    choice_ids = ["first", "second", abstention_id]

    assert expected_decision_utility(
        target,
        abstention_id,
        abstention_id,
        "unanswerable",
        wrong_answer_cost=3,
        choice_ids=choice_ids,
    ) == 1.0
    assert expected_decision_utility(
        target,
        0,
        2,
        "unanswerable",
        wrong_answer_cost=3,
    ) == -3.0


def test_false_task_idk_costs_like_a_wrong_answer_but_external_rejection_is_zero():
    target = [0.6, 0.4, 0.0]

    assert expected_decision_utility(
        target,
        2,
        2,
        "answerable",
        wrong_answer_cost=1.5,
    ) == -1.5
    assert expected_decision_utility(
        target,
        None,
        2,
        "answerable",
        wrong_answer_cost=1.5,
    ) == 0.0


def test_threshold_uses_substantive_maximum_and_accepts_the_boundary():
    # With cost 1, the decision threshold is 1 / (1 + 1) = 0.5.
    assert threshold_action(
        [0.5, 0.1, 0.4], 2, wrong_answer_cost=1
    ) == 0
    assert threshold_action(
        [0.499, 0.1, 0.401], 2, wrong_answer_cost=1
    ) is None
    # A high task-IDK probability does not count as substantive confidence.
    assert threshold_action(
        [0.04, 0.01, 0.95], 2, wrong_answer_cost=1
    ) is None


@pytest.mark.parametrize(
    ("target", "predicted"),
    [
        ([1.1, -0.1], [0.5, 0.5]),
        ([0.4, 0.4], [0.5, 0.5]),
        ([float("nan"), 0.0], [0.5, 0.5]),
        ([True, 0.0], [0.5, 0.5]),
        ([1.0, 0.0], [0.5, 0.25]),
    ],
)
def test_invalid_probability_distributions_are_rejected(target, predicted):
    with pytest.raises(ValueError):
        categorical_cross_entropy(target, predicted)


def test_zero_probability_edges_do_not_create_nan():
    assert categorical_cross_entropy([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert categorical_brier_sum([1.0, 0.0], [1.0, 0.0]) == 0.0
    assert math.isinf(categorical_cross_entropy([1.0, 0.0], [0.0, 1.0]))
    assert categorical_brier_sum([1.0, 0.0], [0.0, 1.0]) == 2.0


def test_row_and_aggregate_metrics_keep_proper_scores_and_decisions_separate():
    answerable = {
        "target_distribution": [1.0, 0.0, 0.0],
        "predicted_distribution": [0.8, 0.1, 0.1],
        "choice_ids": ["a", "b", "idk"],
        "abstention_choice_id": "idk",
        "answerability_status": "answerable",
    }
    unanswerable = {
        "answer_index": 2,
        "predicted_distribution": [0.1, 0.1, 0.8],
        "abstention_choice_index": 2,
        "answerability_status": "unanswerable",
    }

    row_metrics = evaluate_decision_row(
        [0.0, 0.0, 1.0],
        [0.1, 0.1, 0.8],
        2,
        "unanswerable",
        wrong_answer_cost=1,
    )
    assert row_metrics["threshold_rejected"] is True
    assert row_metrics["threshold_expected_utility"] == 0.0
    assert row_metrics["task_argmax_expected_utility"] == 1.0

    report = evaluate_decision_rows([answerable, unanswerable], wrong_answer_cost=1)
    assert report["questions"] == 2
    assert report["proper_scores"]["categorical_cross_entropy"] == pytest.approx(
        -math.log(0.8)
    )
    assert report["proper_scores"]["categorical_brier_sum"] == pytest.approx(0.06)
    assert report["decision_metrics"]["expected_utility"] == pytest.approx(0.5)
    assert report["decision_metrics"]["answerable_coverage"] == 1.0
    assert report["decision_metrics"]["unanswerable_idk_recall"] == 1.0
    assert report["decision_metrics"]["false_idk_rate"] == 0.0
    assert report["baselines"]["always_answer_expected_utility"] == 0.0
    assert report["baselines"]["all_reject_expected_utility"] == 0.0


def test_selective_risk_uses_expected_error_mass_for_soft_and_hard_targets():
    report = evaluate_decision_rows(
        [
            {
                "target_distribution": [0.75, 0.25, 0.0],
                "predicted_distribution": [0.9, 0.05, 0.05],
                "abstention_choice_index": 2,
                "answerability_status": "answerable",
                "distribution_source_kind": "teacher_estimate",
            },
            {
                "answer_index": 1,
                "predicted_distribution": [0.8, 0.1, 0.1],
                "abstention_choice_index": 2,
                "answerability_status": "answerable",
            },
        ],
        wrong_answer_cost=1,
    )

    assert report["decision_metrics"]["answerable_coverage"] == 1.0
    assert report["decision_metrics"]["covered_answerable_rows"] == 2
    # Soft risk is 1 - 0.75; the wrong hard-label prediction has risk 1.
    assert report["decision_metrics"]["selective_risk"] == pytest.approx(0.625)


def test_risk_coverage_groups_tied_confidence_thresholds_and_is_order_independent():
    rows = [
        {
            "target_distribution": [1.0, 0.0, 0.0],
            "predicted_distribution": [0.8, 0.1, 0.1],
            "abstention_choice_index": 2,
            "answerability_status": "answerable",
        },
        {
            "answer_index": 1,
            "predicted_distribution": [0.8, 0.1, 0.1],
            "abstention_choice_index": 2,
            "answerability_status": "answerable",
        },
        {
            "target_distribution": [0.2, 0.8, 0.0],
            "predicted_distribution": [0.4, 0.4, 0.2],
            "abstention_choice_index": 2,
            "answerability_status": "answerable",
        },
    ]

    report = evaluate_decision_rows(rows, wrong_answer_cost=1)
    reversed_report = evaluate_decision_rows(list(reversed(rows)), wrong_answer_cost=1)
    curve = report["risk_coverage_curve"]

    assert curve == reversed_report["risk_coverage_curve"]
    assert [point["confidence_threshold"] for point in curve] == [None, 0.8, 0.4]
    assert [point["covered_answerable_rows"] for point in curve] == [0, 2, 3]
    assert curve[0]["selective_risk"] is None
    assert curve[1]["answerable_coverage"] == pytest.approx(2 / 3)
    assert curve[1]["selective_risk"] == pytest.approx(0.5)
    assert curve[2]["answerable_coverage"] == 1.0
    assert curve[2]["selective_risk"] == pytest.approx(0.6)
    assert report["decision_metrics"]["selective_risk"] == pytest.approx(0.5)


def test_empty_answerability_statuses_and_no_covered_answerable_rows_are_explicit():
    unanswerable_only = evaluate_decision_rows(
        [
            {
                "answer_index": 1,
                "predicted_distribution": [0.2, 0.7, 0.1],
                "abstention_choice_index": 1,
                "answerability_status": "unanswerable",
            }
        ],
        wrong_answer_cost=1,
    )
    unanswerable_metrics = unanswerable_only["decision_metrics"]
    assert unanswerable_metrics["answerable_rows"] == 0
    assert unanswerable_metrics["answerable_coverage"] is None
    assert unanswerable_metrics["selective_risk"] is None
    assert unanswerable_metrics["false_idk_rate"] is None
    assert unanswerable_metrics["unanswerable_idk_recall"] == 1.0
    assert unanswerable_only["risk_coverage_curve"] == []

    answerable_only = evaluate_decision_rows(
        [
            {
                "answer_index": 0,
                "predicted_distribution": [0.8, 0.1, 0.1],
                "abstention_choice_index": 2,
                "answerability_status": "answerable",
            }
        ],
        wrong_answer_cost=1,
    )
    answerable_metrics = answerable_only["decision_metrics"]
    assert answerable_metrics["unanswerable_rows"] == 0
    assert answerable_metrics["unanswerable_idk_recall"] is None
    assert answerable_metrics["selective_risk"] == 0.0

    rejected_answerable_only = evaluate_decision_rows(
        [
            {
                "answer_index": 0,
                "predicted_distribution": [0.49, 0.01, 0.5],
                "abstention_choice_index": 2,
                "answerability_status": "answerable",
            }
        ],
        wrong_answer_cost=1,
    )
    rejected_metrics = rejected_answerable_only["decision_metrics"]
    assert rejected_metrics["answerable_coverage"] == 0.0
    assert rejected_metrics["covered_answerable_rows"] == 0
    assert rejected_metrics["selective_risk"] is None
    assert rejected_answerable_only["risk_coverage_curve"][0]["selective_risk"] is None


def test_rows_are_stratified_by_distribution_source_kind_with_missing_kind_kept_separate():
    report = evaluate_decision_rows(
        [
            {
                "target_distribution": [0.7, 0.3, 0.0],
                "predicted_distribution": [0.8, 0.1, 0.1],
                "abstention_choice_index": 2,
                "answerability_status": "answerable",
                "distribution_source_kind": "teacher_estimate",
            },
            {
                "target_distribution": [0.0, 1.0, 0.0],
                "predicted_distribution": [0.1, 0.8, 0.1],
                "abstention_choice_index": 2,
                "answerability_status": "answerable",
                "distribution_source_kind": "exact_derived",
            },
            {
                "answer_index": 0,
                "predicted_distribution": [0.8, 0.1, 0.1],
                "abstention_choice_index": 2,
                "answerability_status": "answerable",
            },
        ],
        wrong_answer_cost=1,
    )
    strata = {
        item["distribution_source_kind"]: item
        for item in report["source_kind_stratification"]
    }

    assert list(strata) == ["exact_derived", "teacher_estimate", None]
    assert strata["exact_derived"]["questions"] == 1
    assert strata["exact_derived"]["decision_metrics"]["selective_risk"] == 0.0
    assert strata["teacher_estimate"]["decision_metrics"]["selective_risk"] == pytest.approx(0.3)
    assert strata[None]["questions"] == 1
    assert strata[None]["decision_metrics"]["selective_risk"] == 0.0


def test_inconsistent_answerability_targets_and_invalid_costs_are_rejected():
    with pytest.raises(ValueError, match="answerable target"):
        expected_decision_utility(
            [0.5, 0.0, 0.5],
            0,
            2,
            "answerable",
            wrong_answer_cost=1,
        )
    with pytest.raises(ValueError, match="unanswerable target"):
        expected_decision_utility(
            [0.5, 0.5, 0.0],
            2,
            2,
            "unanswerable",
            wrong_answer_cost=1,
        )
    with pytest.raises(ValueError, match="wrong_answer_cost"):
        threshold_action([0.5, 0.5], 1, wrong_answer_cost=-1)
