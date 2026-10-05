import json
from pathlib import Path

import pytest

from dataset_factory import probability_v2 as v2


def _base_row(*, three_choices=False):
    options = ["accept the record", "hold for review"]
    if three_choices:
        options.append("reject the record")
    return {
        "id": "question-v2-1",
        "problem_group_id": "group-v2-1",
        "split": "train",
        "operation_id": "evidence_review",
        "topic": "archive records",
        "language": "en",
        "difficulty": "medium",
        "context": "A short archival record lists the review rule.",
        "question": "How should the record be handled?",
        "options": options,
        "explanation": "The record follows the stated review rule.",
        "provenance": "Luna-generated; label not independently verified.",
    }


def _exact_row(probabilities=None):
    base = _base_row()
    choice_ids = v2.stable_choice_ids(base["id"], base["options"])
    evidence = {
        "kind": "exact_derived",
        "choice_ids": choice_ids,
        "probabilities": [
            {"numerator": 3, "denominator": 4},
            {"numerator": 1, "denominator": 4},
        ],
        "derivation_method": "enumerate-compatible-outcomes-v1",
    }
    return v2.build_soft_row(
        base,
        "exact_derived",
        distribution_evidence=evidence,
        target_distribution=probabilities,
    )


def _teacher_row():
    return v2.build_soft_row(
        _base_row(),
        "teacher_estimate",
        distribution_evidence={
            "kind": "teacher_estimate",
            "model_id": "teacher-small",
            "model_version": "2026-09",
            "estimation_method": "independent-pairwise-elicitation",
        },
        target_distribution=[0.6, 0.4],
    )


@pytest.mark.parametrize(
    ("distribution", "expected_error"),
    [
        ([0.7, 0.2], "target_distribution_does_not_sum_to_one"),
        ([1.01, -0.01], "target_distribution[0]_is_out_of_range"),
        ([float("nan"), 0.0], "target_distribution[0]_is_not_finite"),
    ],
)
def test_soft_distribution_rejects_sum_range_and_nan(distribution, expected_error):
    row = _exact_row()
    row["target_distribution"] = distribution

    assert expected_error in v2.validate_v2_row(row)


def test_soft_distribution_rejects_vector_length_and_evidence_mismatch():
    row = _exact_row()
    row["target_distribution"] = [1.0]
    errors = v2.validate_v2_row(row)
    assert "target_distribution_length_does_not_match_options" in errors

    row["target_distribution"] = [0.5, 0.5]
    assert "target_distribution_does_not_match_evidence" in v2.validate_v2_row(row)


def test_exact_fraction_and_empirical_count_vectors_are_derived_and_checked():
    exact = v2.derive_distribution(
        "exact_derived",
        {
            "kind": "exact_derived",
            "probabilities": [
                {"numerator": 1, "denominator": 3},
                {"numerator": 2, "denominator": 3},
            ],
        },
    )
    empirical = v2.derive_distribution(
        "empirical_frequency",
        {"kind": "empirical_frequency", "counts": [2, 6], "sample_size": 8},
    )

    assert exact == pytest.approx([1 / 3, 2 / 3])
    assert empirical == [0.25, 0.75]
    with pytest.raises(v2.ContractError, match="sum to sample_size"):
        v2.derive_distribution(
            "empirical_frequency",
            {"kind": "empirical_frequency", "counts": [2, 6], "sample_size": 7},
        )


def test_shuffle_keeps_options_ids_distribution_and_evidence_aligned():
    row = _exact_row()
    row["distribution_evidence"]["favorable_counts"] = [7, 3]
    shuffled = v2.shuffle_choices(row, [1, 0])

    assert shuffled["options"] == list(reversed(row["options"]))
    assert shuffled["choice_ids"] == list(reversed(row["choice_ids"]))
    assert shuffled["target_distribution"] == [0.25, 0.75]
    assert shuffled["distribution_evidence"]["choice_ids"] == list(
        reversed(row["distribution_evidence"]["choice_ids"])
    )
    assert shuffled["distribution_evidence"]["probabilities"] == list(
        reversed(row["distribution_evidence"]["probabilities"])
    )
    assert shuffled["distribution_evidence"]["favorable_counts"] == [3, 7]
    assert v2.validate_v2_row(shuffled) == []


def test_probability_v2_accepts_265_aligned_choices_and_rejects_out_of_range_counts():
    options = [f"candidate {index}" for index in range(1, 266)]
    base = {**_base_row(), "id": "question-v2-265", "options": options}
    row = v2.build_soft_row(
        base,
        "exact_derived",
        distribution_evidence={
            "kind": "exact_derived",
            "probabilities": [{"numerator": 1, "denominator": 265} for _ in options],
            "method": "equal-outcome-enumeration",
        },
    )
    permutation = list(reversed(range(265)))
    shuffled = v2.shuffle_choices(row, permutation)

    assert len(shuffled["options"]) == 265
    assert shuffled["choice_ids"] == [row["choice_ids"][index] for index in permutation]
    assert shuffled["target_distribution"] == [row["target_distribution"][index] for index in permutation]
    assert shuffled["distribution_evidence"]["choice_ids"] == shuffled["choice_ids"]
    assert v2.validate_v2_row(shuffled) == []

    too_few = {**row, "options": ["only option"], "choice_ids": [row["choice_ids"][0]]}
    assert any(error.startswith("options_must_contain_two_to_") for error in v2.validate_v2_row(too_few))
    too_many = {**row, "options": options + ["candidate 266"]}
    assert any(error.startswith("options_must_contain_two_to_") for error in v2.validate_v2_row(too_many))


def test_answerability_and_abstention_metadata_constrain_hard_and_soft_targets():
    row = _teacher_row()
    abstention_id = row["choice_ids"][1]
    row.update(
        answerability_status="answerable",
        unanswerable_reason=None,
        abstention_choice_id=abstention_id,
        target_distribution=[1.0, 0.0],
    )
    assert v2.validate_v2_row(row) == []

    row["target_distribution"] = [0.5, 0.5]
    assert "answerable_abstention_choice_must_have_zero_target_mass" in v2.validate_v2_row(row)

    row.update(
        answerability_status="unanswerable",
        unanswerable_reason="missing_context",
        target_distribution=[0.5, 0.5],
    )
    assert "unanswerable_target_must_select_abstention_one_hot" in v2.validate_v2_row(row)
    row["target_distribution"] = [0.0, 1.0]
    assert v2.validate_v2_row(row) == []

    hard = v2.build_hard_row({**_base_row(), "answer_index": 0})
    hard.update(
        answerability_status="unanswerable",
        unanswerable_reason="underdetermined",
        abstention_choice_id=hard["choice_ids"][1],
    )
    assert "unanswerable_hard_label_must_select_abstention" in v2.validate_v2_row(hard)
    hard["answer_index"] = 1
    assert v2.validate_v2_row(hard) == []


def test_answerability_metadata_requires_a_reason_and_valid_choice_reference():
    row = _teacher_row()
    row["answerability_status"] = "unanswerable"
    row["unanswerable_reason"] = "confidence_is_low"
    assert "unanswerable_reason_is_unsupported" in v2.validate_v2_row(row)
    assert "answerability_metadata_requires_abstention_choice_id" in v2.validate_v2_row(row)

    row["unanswerable_reason"] = "missing_context"
    row["abstention_choice_id"] = "unknown-choice"
    errors = v2.validate_v2_row(row)
    assert "abstention_choice_id_must_match_exactly_one_choice_id" in errors
    assert "answerability_metadata_requires_abstention_choice_id" in errors


def test_paired_context_fixture_keeps_variants_in_one_group_and_split():
    first = _teacher_row()
    second = _teacher_row()
    first["id"] = "paired-context-a"
    second["id"] = "paired-context-b"
    first["problem_group_id"] = second["problem_group_id"] = "paired-context-group"
    first["context"] = "The ledger omits the delivery date."
    second["context"] = "The ledger includes the delivery date."

    assert first["id"] != second["id"]
    assert first["problem_group_id"] == second["problem_group_id"]
    assert first["split"] == second["split"] == "train"
    assert v2.validate_v2_row(first) == []
    assert v2.validate_v2_row(second) == []


def test_verification_status_marks_math_without_claiming_semantics_or_calibration():
    exact_status = v2.verification_status(_exact_row())
    teacher_status = v2.verification_status(_teacher_row())

    assert exact_status["schema_and_label_math"] == "machine_checked"
    assert exact_status["prose_semantics"] == "unverified"
    assert teacher_status["teacher_estimate_calibration"] == "unverified"
    assert "confidence" not in _teacher_row()["distribution_evidence"]

    row = _teacher_row()
    row["distribution_evidence"]["confidence"] = 0.99
    assert "distribution_evidence_must_not_use_confidence_fields" in v2.validate_v2_row(row)


def test_hard_and_legacy_labels_remain_separate_from_soft_distributions():
    hard = v2.build_hard_row({**_base_row(), "answer_index": 1})
    legacy = {**_base_row(), "answer_index": 0}
    misleading_legacy = {**legacy, "target_distribution": [1.0, 0.0]}

    assert hard["supervision_kind"] == "hard_label"
    assert "target_distribution" not in hard
    assert "distribution_source_kind" not in hard
    assert v2.classify_row(hard) == ("hard_label", [], "v2")
    assert v2.classify_row(legacy) == ("hard_label", [], "legacy")
    assert v2.classify_row(misleading_legacy)[0] is None


def test_summary_counts_sources_dimensions_entropy_and_hard_soft_totals(tmp_path):
    exact = _exact_row()
    hard = v2.build_hard_row({**_base_row(), "id": "hard-v2", "answer_index": 0})
    legacy = {**_base_row(), "id": "legacy-hard", "answer_index": 1}
    invalid = {**_teacher_row(), "id": "invalid", "target_distribution": [0.2, 0.2]}
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in (exact, hard, legacy, invalid)),
        encoding="utf-8",
    )

    summary = v2.summarize([path])

    assert summary["accepted_rows"] == 3
    assert summary["production_eligible_rows"] == 3
    assert summary["excluded_pilot_rows"] == 0
    assert summary["accepted_rows_by_source_kind"] == {
        "exact_derived": 1,
        "hard_label": 2,
    }
    assert summary["accepted_rows_by_language"] == {"en": 3}
    assert summary["accepted_rows_by_operation"] == {"evidence_review": 3}
    assert summary["soft_rows_by_entropy_bits"] == {"0.5-1.0": 1}
    assert "hard labels have no target_distribution" in summary["entropy_scope"]
    assert summary["supervision_totals"] == {"soft_distribution": 1, "hard_label": 2}
    assert summary["legacy_hard_label_rows"] == 1
    assert summary["verification_scope"]["prose_semantics"] == "unverified"
    assert "do not verify prose semantics or probability calibration" in summary["disclaimer"]


def test_ten_row_pilot_is_valid_but_excluded_from_production_count():
    pilot = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "generated_train"
        / "probability-v2-pilot.jsonl"
    )

    summary = v2.summarize([pilot])

    assert summary["accepted_rows"] == 10
    assert summary["rejected_rows"] == 0
    assert summary["production_eligible_rows"] == 0
    assert summary["production_eligible_rows_by_source_kind"] == {}
    assert summary["excluded_pilot_rows"] == 10


def test_summary_rejects_duplicate_ids_and_groups_spanning_splits(tmp_path):
    duplicate = _exact_row()
    duplicate_again = dict(duplicate)
    hard_train = v2.build_hard_row(
        {
            **_base_row(),
            "id": "leak-train",
            "problem_group_id": "leak-group",
            "answer_index": 0,
        }
    )
    hard_validation = {
        **hard_train,
        "id": "leak-validation",
        "split": "validation",
    }
    path = tmp_path / "identity-audit.jsonl"
    path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (duplicate, duplicate_again, hard_train, hard_validation)
        ),
        encoding="utf-8",
    )

    summary = v2.summarize([path])

    assert summary["accepted_rows"] == 1
    assert summary["duplicate_row_ids_rejected"] == 1
    assert summary["problem_group_split_conflicts"] == 1
    assert summary["rows_in_conflicting_problem_groups"] == 2
    assert summary["rejection_reasons"]["problem_group_spans_splits"] == 1
    assert summary["rejection_reasons"]["split_is_not_train"] == 1


def test_summary_excludes_skip_record_files_from_question_rows(tmp_path):
    question_path = tmp_path / "shard-0001.jsonl"
    question_path.write_text(json.dumps(_exact_row()) + "\n", encoding="utf-8")
    skipped_path = tmp_path / "skipped-shard-0001.jsonl"
    skipped_path.write_text(
        json.dumps(
            {
                "id": "assignment-1",
                "reason": "unsupported_language_tag: unsupported output language",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    directory_summary = v2.summarize([tmp_path])
    explicit_skip_summary = v2.summarize([skipped_path])

    assert directory_summary["files_scanned"] == [question_path.as_posix()]
    assert directory_summary["accepted_rows"] == 1
    assert directory_summary["rejected_rows"] == 0
    assert explicit_skip_summary["files_scanned"] == []
    assert explicit_skip_summary["rejected_rows"] == 0
