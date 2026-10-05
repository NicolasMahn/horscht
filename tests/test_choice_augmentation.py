import copy

import pytest

from dataset_factory import probability_v2 as v2
from dataset_factory.choice_augmentation import (
    AugmentationError,
    BalancedTargetSlotSchedule,
    augment_with_impossible_options,
)


PROVENANCE = "Luna-generated; label not independently verified."


def _base_row(options=None):
    options = options or ["red token", "blue token"]
    row = {
        "schema_version": "probability-v2",
        "id": "probability-v2-fixture-source",
        "problem_group_id": "probability-v2-fixture-group",
        "split": "train",
        "language": "en",
        "context": (
            "The sealed draw contains exactly a red token or a blue token, "
            "and each token is equally likely."
        ),
        "question": "Which listed token can be drawn?",
        "options": options,
        "choice_ids": v2.stable_choice_ids("probability-v2-fixture-source", options),
        "topic": "closed-world test fixture",
        "family": "finite_outcome_counting",
        "difficulty": "easy",
        "explanation": "The stated draw has exactly the two listed outcomes.",
        "provenance": PROVENANCE,
    }
    return row


def _exact_row():
    row = _base_row()
    return v2.build_soft_row(
        row,
        "exact_derived",
        distribution_evidence={
            "kind": "exact_derived",
            "choice_ids": list(row["choice_ids"]),
            "probabilities": [
                {"numerator": 1, "denominator": 2},
                {"numerator": 1, "denominator": 2},
            ],
            "favorable_counts": [1, 1],
            "sample_space_size": 2,
            "derivation_method": "enumerate-the-explicit-fixture-outcomes",
        },
    )


def _impossible_options(count):
    return [
        {
            "text": f"impossible fixture decoy {index:03d}",
            "reason": "The quoted closed-world rule lists the complete draw support.",
        }
        for index in range(1, count + 1)
    ]


def _soft_proof(row):
    return {
        "kind": "explicit_closed_world_outcomes",
        "context_quote": "The sealed draw contains exactly a red token or a blue token",
        "possible_choice_ids": list(row["choice_ids"]),
        "review_status": "deterministic_fixture",
        "reviewer": "test-fixture-oracle",
    }


def test_exact_augmentation_reaches_slot_264_without_changing_source():
    source = _exact_row()
    before = copy.deepcopy(source)
    additions = _impossible_options(263)
    augmented = augment_with_impossible_options(
        source,
        additions,
        proof=_soft_proof(source),
        target_slot=264,
        seed=42,
        variant_key="test-265-choice",
        allow_fixture_proof=True,
    )

    assert source == before
    assert augmented["id"] != source["id"]
    assert augmented["problem_group_id"] == source["problem_group_id"]
    assert augmented["split"] == source["split"]
    assert len(augmented["options"]) == 265
    assert augmented["choice_count"] == 265
    assert augmented["target_distribution"][264] == pytest.approx(0.5)
    assert augmented["target_distribution"][:264].count(0.0) == 263
    assert augmented["distribution_evidence"]["choice_ids"] == augmented["choice_ids"]
    assert augmented["distribution_evidence"]["favorable_counts"][264] == 1
    assert augmented["distribution_evidence"]["favorable_counts"].count(0) == 263
    assert len(set(augmented["choice_ids"])) == 265
    assert v2.validate_v2_row(augmented) == []
    assert augmented["_augmentation"]["mass_training_status"] == "requires_review"


def test_answerable_hard_row_keeps_its_answer_after_joint_permutation():
    source = _base_row()
    source.update({"supervision_kind": "hard_label", "answer_index": 0})
    additions = _impossible_options(1)
    proof = {
        "kind": "reviewed_impossible_distractors",
        "context_quote": "The sealed draw contains exactly a red token or a blue token",
        "correct_choice_id": source["choice_ids"][0],
        "review_status": "deterministic_fixture",
        "reviewer": "test-fixture-oracle",
        "distractor_proofs": additions,
    }

    augmented = augment_with_impossible_options(
        source,
        additions,
        proof=proof,
        target_slot=2,
        seed=19,
        variant_key="hard-fixture",
        allow_fixture_proof=True,
    )

    assert len(augmented["options"]) == 3
    assert augmented["answer_index"] == 2
    assert augmented["options"][augmented["answer_index"]] == source["options"][0]
    assert "target_distribution" not in augmented
    assert "distribution_evidence" not in augmented
    assert v2.validate_v2_row(augmented) == []


def test_unverified_proofs_and_nonexact_soft_sources_are_rejected():
    source = _exact_row()
    additions = _impossible_options(1)
    with pytest.raises(AugmentationError, match="must name a reviewer"):
        augment_with_impossible_options(
            source,
            additions,
            proof={**_soft_proof(source), "review_status": "not_reviewed"},
            target_slot=0,
            seed=1,
            variant_key="unreviewed",
        )

    empirical = v2.build_soft_row(
        _base_row(),
        "empirical_frequency",
        distribution_evidence={
            "kind": "empirical_frequency",
            "choice_ids": v2.stable_choice_ids("probability-v2-fixture-source", ["red token", "blue token"]),
            "counts": [1, 1],
            "sample_size": 2,
        },
    )
    with pytest.raises(AugmentationError, match="only exact/stochastic"):
        augment_with_impossible_options(
            empirical,
            additions,
            proof=_soft_proof(empirical),
            target_slot=0,
            seed=1,
            variant_key="empirical-rejected",
            allow_fixture_proof=True,
        )


def test_target_slot_schedule_balances_all_265_positions_and_resumes():
    schedule = BalancedTargetSlotSchedule(seed=20260929)
    first_cycle = [schedule.next_slot(265) for _ in range(265)]
    assert sorted(first_cycle) == list(range(265))

    resumed = BalancedTargetSlotSchedule.from_state_dict(schedule.state_dict())
    expected = [schedule.next_slot(265) for _ in range(9)]
    assert [resumed.next_slot(265) for _ in range(9)] == expected


def test_target_slot_schedule_rejects_out_of_range_cardinality():
    schedule = BalancedTargetSlotSchedule(seed=1)
    with pytest.raises(AugmentationError, match="between 2 and 265"):
        schedule.next_slot(266)
