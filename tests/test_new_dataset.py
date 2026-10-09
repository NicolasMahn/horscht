import json

from dataset_factory.new_dataset import (
    assign_splits,
    balanced_sample,
    grouped_rows,
    public_permutation,
)
from dataset_factory.new_public_data import convert_typed, validate_public_row


def row(identifier, context=None, source="documents", split="train"):
    return {
        "id": identifier,
        "problem_group_id": identifier,
        "context": context or identifier,
        "question": "Choose the matching entry.",
        "options": ["First", "Second"],
        "language": "en",
        "_new_dataset": {"source_id": source, "original_split": split},
        "operation_id": "fixture",
    }


def test_nested_document_relations_and_parent_links_are_transitive():
    a, b, c = [row(key) for key in ("a", "b", "c")]
    a["document_metadata"] = {
        "document_id": "edition-a",
        "relation_keys": ["article:1"],
    }
    b["document_metadata"] = {
        "document_id": "edition-b",
        "relation_keys": ["article:1", "article:2"],
    }
    c["document_metadata"] = {
        "document_id": "edition-c",
        "relation_keys": ["article:2"],
    }
    groups, _ = grouped_rows([a, b, c])
    assert len(set(groups)) == 1
    child = row("child")
    child["translation_of"] = "a"
    assert len(set(grouped_rows([a, child])[0])) == 1


def test_game_trajectory_and_old_group_aliases_join_before_splitting():
    a, b = row("a", source="games"), row("b", source="games")
    a["game_metadata"] = {
        "generation": {"scenario_id": "episode-1"},
        "state_family_id": "shared-state",
    }
    b["game_metadata"] = {
        "generation": {"scenario_id": "episode-1"},
        "prior_group_ids": ["old-group"],
    }
    old = row("old", source="original_replay", split="test")
    old["_training_snapshot"] = {"snapshot_group_id": "old-group"}
    groups, components = grouped_rows([a, b, old])
    assert len(set(groups)) == 1
    assigned, conflicts = assign_splits([a, b, old], components)
    assert not conflicts
    assert set(assigned.values()) == {"test"}


def test_warmstart_train_cannot_become_calibration_and_conflict_is_quarantined():
    a = row("old-train", context="same", source="original_replay")
    b = row("fresh", context="same")
    _, components = grouped_rows([a, b])
    assigned, conflicts = assign_splits([a, b], components)
    assert not conflicts and set(assigned.values()) == {"train"}
    c = row("old-dev", context="same", source="original_replay", split="dev")
    _, components = grouped_rows([a, b, c])
    assigned, conflicts = assign_splits([a, b, c], components)
    assert not assigned and len(conflicts) == 1


def test_balanced_cap_preserves_sparse_high_cardinality_stratum():
    abundant = [row(str(i)) for i in range(100)]
    high = row("high")
    high["options"] = [str(i) for i in range(265)]
    selected = balanced_sample(abundant + [high], 4)
    assert len(selected) == 4
    assert any(candidate["id"] == "high" for candidate in selected)
    assert selected == balanced_sample(abundant + [high], 4)


def test_permutation_remaps_hard_and_soft_labels_without_changing_group():
    hard = row("hard")
    hard.update(choice_ids=["a", "b"], answer_index=0)
    soft = row("soft")
    soft.update(
        choice_ids=["a", "b"],
        target_distribution=[0.2, 0.8],
        distribution_evidence={"choice_ids": ["a", "b"], "probabilities": [0.2, 0.8]},
    )
    for original in (hard, soft):
        permuted = public_permutation(original)
        assert permuted["problem_group_id"] == original["problem_group_id"]
        if "answer_index" in original:
            assert permuted["choice_ids"][permuted["answer_index"]] == "a"
        else:
            assert dict(
                zip(permuted["choice_ids"], permuted["target_distribution"])
            ) == {"a": 0.2, "b": 0.8}
            assert (
                permuted["distribution_evidence"]["probabilities"]
                == permuted["target_distribution"]
            )


def test_typed_boolean_without_criteria_preserves_published_rounding():
    metadata = {
        "repository": "LocalLLaMA/typed-decisions",
        "revision": "pinned",
        "split": "train",
    }
    record = {
        "id": "state-1",
        "workflow": "fixture",
        "split": "train",
        "state": '{"task":"fixture"}',
        "questions": json.dumps(
            {"duplicate": {"instructions": "Are these duplicates?", "type": "boolean"}}
        ),
        "gold": json.dumps(
            {
                "duplicate": {
                    "probabilities": {"false": 0.333333, "true": 0.666666},
                    "type": "boolean",
                }
            }
        ),
        "label_agreement": json.dumps({"duplicate": {"argmax_agree": True}}),
    }
    result = convert_typed([record], metadata)[0]
    assert result["options"] == ["false", "true"]
    assert result["target_distribution"] == [0.333333, 0.666666]
    assert not validate_public_row(result)
    result["target_distribution"] = [0.3, 0.6]
    assert "target_distribution_does_not_sum_to_one" in validate_public_row(result)


def test_typed_ordinal_criteria_align_by_published_label():
    metadata = {
        "repository": "LocalLLaMA/typed-decisions",
        "revision": "pinned",
        "split": "train",
    }
    record = {
        "id": "state-2",
        "workflow": "fixture",
        "split": "train",
        "state": "{}",
        "questions": json.dumps(
            {
                "severity": {
                    "instructions": "Rate severity.",
                    "type": "ordinal",
                    "criteria": ["none", "minor", "major", "critical"],
                }
            }
        ),
        "gold": json.dumps(
            {
                "severity": {
                    "probabilities": {"2": 0.8, "0": 0.1, "1": 0.1, "3": 0.0},
                    "type": "ordinal",
                }
            }
        ),
        "label_agreement": json.dumps({"severity": {"argmax_agree": True}}),
    }
    result = convert_typed([record], metadata)[0]
    assert result["options"] == ["2: major", "0: none", "1: minor", "3: critical"]
    assert result["target_distribution"] == [0.8, 0.1, 0.1, 0.0]
    assert not validate_public_row(result)


def test_runtime_adapter_accepts_real_source_envelopes_in_all_new_splits():
    import hashlib
    from dataset_factory.new_dataset import SPLITS, validate_snapshot_row
    from dataset_factory.progress import PROVENANCE
    from paired_training import ordered_row
    from probability_experiment import targets_for_rows

    for source in ("documents", "games", "typed-decisions", "sciq", "banking77"):
        for split in SPLITS:
            fixture = row(f"{source}:{split}", source=source)
            fixture.update(
                split=split,
                schema_version="probability-v2",
                supervision_kind="hard_label",
                answer_index=0,
                choice_ids=["a", "b"],
                explanation="Source supervision.",
                topic="fixture",
                difficulty="medium",
                provenance=PROVENANCE,
            )
            if source == "documents":
                fixture.update(
                    semantic_review_status="independent_luna_max_review_passed",
                    provenance="luna-generated; source-grounded",
                    document_metadata={
                        "author_model": "gpt-6-luna",
                        "content_sha256": hashlib.sha256(
                            fixture["context"].encode()
                        ).hexdigest(),
                    },
                )
            elif source == "games":
                fixture["provenance"] = (
                    "Local deterministic game-data-v1; exact fixture."
                )
            else:
                fixture.update(
                    provenance=f"Public source-native training split: {source}; see public_metadata",
                    public_metadata={"split": "train"},
                )
            assert not validate_snapshot_row(fixture)
            permuted = ordered_row(fixture, 17, 1)
            targets, _, _ = targets_for_rows([permuted])
            assert float(targets.sum()) == 1
            assert permuted["choice_ids"][permuted["answer_index"]] == "a"


def test_runtime_adapter_rejects_reassigned_original_and_native_holdouts():
    from dataset_factory.new_dataset import validate_snapshot_row

    fixture = row("old", source="original_replay", split="train")
    fixture["split"] = "calibration"
    assert validate_snapshot_row(fixture) == ["snapshot_original_split_changed"]
    fixture = row("game", source="games", split="development")
    fixture["split"] = "train"
    assert validate_snapshot_row(fixture) == [
        "snapshot_native_development_split_changed"
    ]
