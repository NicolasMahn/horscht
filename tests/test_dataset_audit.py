import json
from pathlib import Path

import pytest

from dataset_audit import audit
from dataset_factory.progress import PROVENANCE


def _row(identifier, *, language="en", operation="reasoning", group=None, split="train"):
    return {
        "id": identifier,
        "problem_group_id": group or f"group-{identifier}",
        "split": split,
        "language": language,
        "difficulty": "medium",
        "topic": "landforms",
        "family": operation,
        "task_id": operation,
        "context": f"A creek crosses marker {identifier}.",
        "question": "Which marker comes next?",
        "options": ["north", "south"],
        "answer_index": 0,
        "explanation": "do not embed this explanation",
        "provenance": PROVENANCE,
        "generation_metadata": {"seed": 123, "batch_id": "private-batch"},
    }


def test_embedding_input_uses_only_context_question_and_ordered_options():
    row = _row("sentinel")
    row.update({
        "answer_index": 1,
        "explanation": "ANSWER_SECRET",
        "target_distribution": [0.9, 0.1],
        "distribution_evidence": {"evidence_secret": "EVIDENCE_SECRET"},
    })

    text = audit.render_embedding_input(row)

    assert "A creek crosses marker sentinel." in text
    assert "Which marker comes next?" in text
    assert "- north\n- south" in text
    assert "ANSWER_SECRET" not in text
    assert "EVIDENCE_SECRET" not in text
    assert "do not embed" not in text
    assert "private-batch" not in text


def test_snapshot_excludes_duplicate_ids_and_cross_split_groups(tmp_path):
    root = tmp_path
    source = root / "data/generated_train/broad-v1"
    source.mkdir(parents=True)
    rows = [
        _row("conflict-train", group="shared"),
        _row("conflict-test", group="shared", split="test"),
        _row("kept", operation="causal_graphs"),
        _row("kept", operation="duplicate-id"),
    ]
    (source / "sample.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    output = root / "audit"

    manifest = audit._accepted_snapshot(root, output)
    sample = [json.loads(line) for line in (output / "sample.jsonl").read_text(encoding="utf-8").splitlines()]

    assert manifest["accepted_rows"] == 1
    assert manifest["duplicate_ids_skipped"] == 1
    assert manifest["problem_group_split_conflicts"] == 1
    assert [row["id"] for row in sample] == ["kept"]
    assert "answer_index" not in sample[0]
    assert "explanation" not in sample[0]
    assert "generation_metadata" not in sample[0]
    assert len(sample[0]["row_sha256"]) == 64
    assert len(manifest["source_snapshot_sha256"]) == 64


def test_sample_selection_is_seed_stable_and_empty_population_is_rejected():
    rows = [
        {"id": f"{language}-{index}", "language": language, "source_kind": "hard_label", "operation": "task"}
        for language in ("en", "fr")
        for index in range(6)
    ]
    first = audit.select_sample([dict(row) for row in rows], 8, 14)
    second = audit.select_sample([dict(row) for row in rows], 8, 14)

    assert [(row["id"], row["sample_weight"]) for row in first] == [
        (row["id"], row["sample_weight"]) for row in second
    ]
    assert {row["language"] for row in first} == {"en", "fr"}
    assert sum(row["sample_weight"] for row in first) == len(rows)
    with pytest.raises(ValueError):
        audit.select_sample([], 1, 14)


def test_embedding_cache_key_invalidates_when_any_sample_input_changes():
    manifest = {"source_snapshot_sha256": "source-a"}
    rows = [{"id": "row-1", "row_sha256": "row-a", "text_sha256": "text-a"}]
    identity = audit.embedding_cache_identity(manifest, rows)
    matching_cache = {"cache_key": audit.embedding_cache_key(identity), "rows": 1, "dimensions": 384}

    assert audit.embedding_cache_matches(matching_cache, identity, 1)
    changed_rows = [{**rows[0], "text_sha256": "text-b"}]
    changed_identity = audit.embedding_cache_identity(manifest, changed_rows)
    assert not audit.embedding_cache_matches(matching_cache, changed_identity, 1)
    changed_model = {**identity, "revision": "another-revision"}
    assert not audit.embedding_cache_matches(matching_cache, changed_model, 1)
