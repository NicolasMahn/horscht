import copy
import json
from pathlib import Path

import pytest

from dataset_factory import probability_v2
from dataset_factory.training_snapshot import SnapshotError, build_snapshot, permute_choices


PROVENANCE = "luna-generated; label not independently verified"


def _base_row(identifier, *, context=None, options=None, answer_index=0, group=None):
    options = options or ["first outcome", "second outcome"]
    return {
        "id": identifier,
        "problem_group_id": group or f"group-{identifier}",
        "split": "train",
        "language": "en",
        "context": context or f"The fixture context for {identifier} states both outcomes.",
        "question": f"Which outcome follows in {identifier}?",
        "options": options,
        "answer_index": answer_index,
        "topic": "deterministic fixture",
        "family": "finite_outcome_counting",
        "difficulty": "easy",
        "explanation": "The fixture supplies the stated answer directly.",
        "provenance": PROVENANCE,
    }


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _make_soft_row(identifier, *, context=None, group=None, options=None):
    options = options or ["first outcome", "second outcome"]
    row = _base_row(identifier, context=context, options=options, group=group)
    row.pop("answer_index")
    row["choice_ids"] = probability_v2.stable_choice_ids(identifier, options)
    return probability_v2.build_soft_row(
        row,
        "exact_derived",
        distribution_evidence={
            "kind": "exact_derived",
            "choice_ids": list(row["choice_ids"]),
            "probabilities": [
                {"numerator": 1, "denominator": 1},
                {"numerator": 0, "denominator": 1},
            ],
            "favorable_counts": [1, 0],
            "sample_space_size": 1,
        },
    )


def test_snapshot_is_repeatable_grouped_and_keeps_review_cohort(tmp_path):
    source = tmp_path / "source.jsonl"
    issues = tmp_path / "known-issues.jsonl"
    rows = []
    for index in range(45):
        rows.append(_base_row(f"row-{index:02d}"))

    # A second question over the same exact passage is one indivisible group.
    pair_context = "A distinct shared passage describes a closed two-outcome process."
    rows.extend(
        [
            _base_row("paired-a", context=pair_context),
            _base_row("paired-b", context=pair_context, options=["left", "right"]),
        ]
    )
    pilot = _make_soft_row("pilot")
    pilot["pilot_status"] = probability_v2.EXCLUDED_PILOT_STATUS
    rows.append(pilot)
    pending = _base_row("pending-review")
    recommended = _base_row("recommended")
    invalid = _base_row("invalid-review")
    invalid["split"] = "test"
    rows.extend([pending, recommended, invalid])
    rows.append(copy.deepcopy(rows[0]))  # Duplicate ID follows canonical first-row handling.
    exact_duplicate = copy.deepcopy(rows[1])
    exact_duplicate.update({"id": "duplicate-content", "problem_group_id": "other-group"})
    rows.append(exact_duplicate)
    _write_jsonl(source, rows)
    _write_jsonl(
        issues,
        [
            {"issue_id": "issue-pending", "training_status": "exclude_pending_review", "question_ids": ["pending-review"]},
            {"issue_id": "issue-review", "training_status": "review_recommended", "question_ids": ["recommended"]},
            {"issue_id": "issue-overlap", "training_status": "exclude_pending_review", "question_ids": ["invalid-review"]},
        ],
    )
    source_bytes = source.read_bytes()
    issues_bytes = issues.read_bytes()

    manifest_a = build_snapshot(
        "snapshot-a",
        source_paths=[source],
        known_issues_path=issues,
        output_root=tmp_path / "snapshots-a",
        seed=41,
        split_ratios={"train": 0.5, "dev": 0.25, "test": 0.25},
    )
    manifest_b = build_snapshot(
        "snapshot-b",
        source_paths=[source],
        known_issues_path=issues,
        output_root=tmp_path / "snapshots-b",
        seed=41,
        split_ratios={"train": 0.5, "dev": 0.25, "test": 0.25},
    )
    output_a = tmp_path / "snapshots-a" / "snapshot-a"
    output_b = tmp_path / "snapshots-b" / "snapshot-b"

    assert source.read_bytes() == source_bytes
    assert issues.read_bytes() == issues_bytes
    assert (output_a / "rows.jsonl").read_bytes() == (output_b / "rows.jsonl").read_bytes()
    assert (output_a / "splits.jsonl").read_bytes() == (output_b / "splits.jsonl").read_bytes()
    assert manifest_a["rows"]["training_eligible_rows"] == manifest_b["rows"]["training_eligible_rows"]
    assert set(manifest_a["rows"]["by_split"]) == {"train", "dev", "test"}

    split_rows = [json.loads(line) for line in (output_a / "splits.jsonl").read_text().splitlines()]
    by_id = {row["id"]: row for row in split_rows}
    split_for_group = {}
    for row in split_rows:
        prior_split = split_for_group.setdefault(row["snapshot_group_id"], row["split"])
        assert prior_split == row["split"]
    assert by_id["paired-a"]["snapshot_group_id"] == by_id["paired-b"]["snapshot_group_id"]
    assert by_id["paired-a"]["split"] == by_id["paired-b"]["split"]
    assert by_id["recommended"]["cohorts"] == ["review_recommended"]
    assert "pending-review" not in by_id
    assert "pilot" not in by_id
    assert "invalid-review" not in by_id
    assert manifest_a["rows"]["review_recommended_rows"] == 1
    assert manifest_a["known_issue_sidecar"]["intersection_counts"]["pending_review_ids_in_structurally_accepted_rows"] == 1
    assert "not guaranteed" in manifest_a["grouping"]["semantic_duplicates"]

    exclusions = [json.loads(line) for line in (output_a / "excluded.jsonl").read_text().splitlines()]
    exclusion_by_id = {row["id"]: row for row in exclusions if row["id"]}
    assert "known_issue_excluded" in exclusion_by_id["pending-review"]["categories"]
    assert "structural_invalid" in exclusion_by_id["invalid-review"]["categories"]
    assert "known_issue_excluded" in exclusion_by_id["invalid-review"]["categories"]
    assert "pilot_excluded" in exclusion_by_id["pilot"]["categories"]
    assert manifest_a["exclusions"]["duplicate_ids_dropped"] == 1
    assert manifest_a["exclusions"]["exact_content_dropped"] == 1

    normalized = [json.loads(line) for line in (output_a / "rows.jsonl").read_text().splitlines()]
    legacy = next(row for row in normalized if row["id"] == "row-00")
    assert legacy["schema_version"] == probability_v2.SCHEMA_VERSION
    assert legacy["supervision_kind"] == probability_v2.HARD_LABEL
    assert legacy["answer_index"] == 0
    assert "target_distribution" not in legacy
    soft = next(row for row in normalized if row["id"] == "paired-a")
    assert soft["_training_snapshot"]["target_kind"] == "hard_onehot"


def test_permute_choices_keeps_hard_and_soft_targets_aligned_at_265_choices():
    options = [f"outcome-{index:03d}" for index in range(264)] + ["Cannot be determined."]
    hard_source = _base_row("hard-265", options=options, answer_index=264)
    hard_source["choice_ids"] = probability_v2.stable_choice_ids(hard_source["id"], options)
    hard_source.update(
        {
            "answerability_status": "unanswerable",
            "unanswerable_reason": "missing_context",
            "abstention_choice_id": hard_source["choice_ids"][264],
            "idk_index": 264,
        }
    )
    hard = probability_v2.build_hard_row(hard_source)
    hard_before = copy.deepcopy(hard)

    hard_permuted = permute_choices(hard, seed="epoch-3")
    hard_again = permute_choices(hard, seed="epoch-3")
    assert hard_permuted == hard_again
    assert hard == hard_before
    assert hard_permuted["options"] != options
    assert hard_permuted["choice_ids"][hard_permuted["answer_index"]] == hard["choice_ids"][hard["answer_index"]]
    assert hard_permuted["choice_ids"][hard_permuted["idk_index"]] == hard["abstention_choice_id"]
    assert "target_distribution" not in hard_permuted
    assert probability_v2.validate_v2_row(hard_permuted) == []

    soft_options = [f"soft-outcome-{index:03d}" for index in range(264)] + ["Cannot be determined."]
    soft_source = _base_row("soft-265", options=soft_options)
    soft_source.pop("answer_index")
    soft_source["choice_ids"] = probability_v2.stable_choice_ids(soft_source["id"], soft_options)
    soft_source.update(
        {
            "answerability_status": "unanswerable",
            "unanswerable_reason": "missing_context",
            "abstention_choice_id": soft_source["choice_ids"][264],
            "idk_index": 264,
        }
    )
    exact_probabilities = [
        {"numerator": 1 if index == 264 else 0, "denominator": 1}
        for index in range(265)
    ]
    soft = probability_v2.build_soft_row(
        soft_source,
        "exact_derived",
        distribution_evidence={
            "kind": "exact_derived",
            "choice_ids": list(soft_source["choice_ids"]),
            "probabilities": exact_probabilities,
            "favorable_counts": [1 if index == 264 else 0 for index in range(265)],
            "sample_space_size": 1,
        },
    )
    soft_before = copy.deepcopy(soft)
    soft_permuted = permute_choices(soft, seed="epoch-3")

    assert soft == soft_before
    assert soft_permuted["choice_ids"][soft_permuted["idk_index"]] == soft["abstention_choice_id"]
    assert soft_permuted["target_distribution"][soft_permuted["idk_index"]] == 1.0
    assert soft_permuted["target_distribution"].count(1.0) == 1
    assert soft_permuted["distribution_evidence"]["choice_ids"] == soft_permuted["choice_ids"]
    assert soft_permuted["distribution_evidence"]["probabilities"] == [
        exact_probabilities[index] for index in [soft["choice_ids"].index(identifier) for identifier in soft_permuted["choice_ids"]]
    ]
    assert soft_permuted["distribution_evidence"]["favorable_counts"][soft_permuted["idk_index"]] == 1
    assert probability_v2.validate_v2_row(soft_permuted) == []


def test_previous_snapshot_preserves_linked_held_assignment_and_checks_hashes(tmp_path, monkeypatch):
    import dataset_factory.training_snapshot as snapshot_module

    allowed_root = tmp_path / "training_snapshots"
    monkeypatch.setattr(snapshot_module, "SNAPSHOT_ROOT", allowed_root)
    source = tmp_path / "source.jsonl"
    issues = tmp_path / "known-issues.jsonl"
    issues.write_text("", encoding="utf-8")
    first = _base_row("historical")
    first["translation_group_id"] = "shared-history"
    _write_jsonl(source, [first])
    build_snapshot(
        "prior", source_paths=[source], known_issues_path=issues,
        output_root=allowed_root, seed=11,
    )
    prior_path = allowed_root / "prior"
    prior_split = json.loads((prior_path / "splits.jsonl").read_text().strip())["split"]

    newcomer = _base_row("new-linked")
    newcomer["translation_group_id"] = "shared-history"
    _write_jsonl(source, [first, newcomer])
    manifest = build_snapshot(
        "next", source_paths=[source], known_issues_path=issues,
        output_root=allowed_root, seed=99,
        split_ratios={"train": 0.98, "dev": 0.01, "test": 0.01},
        previous_snapshot=prior_path,
    )
    next_splits = {
        row["id"]: row["split"]
        for row in map(json.loads, (allowed_root / "next" / "splits.jsonl").read_text().splitlines())
    }
    assert next_splits == {"historical": prior_split, "new-linked": prior_split}
    assert manifest["previous_snapshot"]["manifest_sha256"]
    assert manifest["previous_snapshot"]["preservation_audit"]["groups_preserved"] == 1

    tampered = allowed_root / "prior" / "splits.jsonl"
    tampered.write_text(tampered.read_text() + "\n", encoding="utf-8")
    with pytest.raises(SnapshotError, match="hash"):
        build_snapshot(
            "bad-hash", source_paths=[source], known_issues_path=issues,
            output_root=allowed_root, previous_snapshot=prior_path,
        )
    tampered.unlink()
    outside_file = tmp_path / "outside.jsonl"
    outside_file.write_text("{}\n", encoding="utf-8")
    tampered.symlink_to(outside_file)
    with pytest.raises(SnapshotError, match="must not be a symlink"):
        build_snapshot(
            "symlink", source_paths=[source], known_issues_path=issues,
            output_root=allowed_root, previous_snapshot=prior_path,
        )


def test_previous_snapshot_must_be_under_training_snapshots(tmp_path):
    from dataset_factory.training_snapshot import _load_previous_assignments

    with pytest.raises(SnapshotError, match="inside data/training_snapshots"):
        _load_previous_assignments(tmp_path / "untrusted")
