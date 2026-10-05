import gzip
import json
from pathlib import Path

import pytest

from dataset_factory import assignments, progress, reconcile


def _catalog() -> dict:
    return {
        "languages": [
            {"tag": "en", "name": "English", "family": "Germanic", "script": "Latin"},
            {"tag": "fr", "name": "French", "family": "Romance", "script": "Latin"},
        ],
        "domains": [
            {"id": "weaving", "name": "Weaving", "region": "Global"},
            {"id": "games", "name": "Games", "region": "Global"},
            {"id": "archives", "name": "Archives", "region": "Global"},
        ],
        "operations": [
            {
                "id": "sequence",
                "name": "Sequence reasoning",
                "compatible_domains": ["weaving", "games"],
                "question_forms": ["choose the valid next step", "identify the required order"],
            },
            {
                "id": "evidence",
                "name": "Evidence selection",
                "compatible_domains": ["archives", "games"],
                "question_forms": ["select the supported claim"],
            },
        ],
        "input_structures": [{"id": "prose"}, {"id": "table"}],
        "registers": [{"id": "plain"}, {"id": "formal"}],
        "difficulty_bands": [{"id": "easy"}, {"id": "medium"}, {"id": "hard"}],
        "context_length_bands": [{"id": "short"}, {"id": "medium"}, {"id": "long"}],
        "choice_counts": [2, 3, 4],
    }


def _pool() -> list[dict[str, str]]:
    return [
        {
            "word": f"cue-{index}",
            "source_language": "en" if index % 2 else "fr",
            "source_version": "fixture-1",
            "pool_id": "fixture-pool",
        }
        for index in range(20)
    ]


def test_assignment_draw_is_reproducible_and_covers_its_dimensions(tmp_path):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
    catalog = assignments.load_catalog(catalog_path)
    first = assignments.create_assignments(catalog, _pool(), 48, 171, 100)
    second = assignments.create_assignments(catalog, _pool(), 48, 171, 100)

    assert first == second
    assert len({row["id"] for row in first}) == 48
    assert len({row["problem_group_id"] for row in first}) == 48
    assert {row["language"] for row in first} == {"en", "fr"}
    assert len({row["operation_id"] for row in first}) == 2
    assert len({row["domain_id"] for row in first}) == 3
    for row in first:
        words = row["_generator"]["seed_words"]
        assert 3 <= len(words) <= 8
        assert len({word["word"] for word in words}) == len(words)
        assert 2 <= row["choice_count"] <= 4
        assert 0 <= row["_generator"]["answer_position"] < row["choice_count"]

    language_shard = assignments.create_assignments(
        catalog, _pool(), 8, 171, 300, language_tags=["fr"]
    )
    assert {row["language"] for row in language_shard} == {"fr"}
    with pytest.raises(assignments.ContractError, match="unknown catalog tags"):
        assignments.create_assignments(catalog, _pool(), 1, 171, 400, language_tags=["zz"])

    short_contexts = assignments.create_assignments(
        catalog, _pool(), 8, 171, 500, context_length_bands=["short"]
    )
    assert {row["context_length_band"]["id"] for row in short_contexts} == {"short"}
    assert short_contexts == assignments.create_assignments(
        catalog, _pool(), 8, 171, 500, context_length_bands=["short"]
    )
    with pytest.raises(assignments.ContractError, match="unknown catalog ids"):
        assignments.create_assignments(catalog, _pool(), 1, 171, 600, context_length_bands=["tiny"])


def test_assignment_writer_records_a_resumable_nonoverlapping_plan(tmp_path, monkeypatch):
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
    pool_path = tmp_path / "pool.jsonl.gz"
    with gzip.open(pool_path, "wt", encoding="utf-8") as stream:
        for row in _pool():
            stream.write(json.dumps(row) + "\n")

    assignment_dir = tmp_path / "assignments"
    batch_dir = tmp_path / "batches"
    monkeypatch.setattr(assignments, "BROAD_V1_DIR", tmp_path / "broad-v1")
    monkeypatch.setattr(assignments, "MULTILINGUAL_DIR", tmp_path / "multilingual")

    result = assignments.write_assignment_batch(
        count=8,
        seed=9,
        start_index=40,
        shard_index=1,
        catalog_path=catalog_path,
        word_pool_path=pool_path,
        assignment_dir=assignment_dir,
        batch_history_dir=batch_dir,
    )
    plans = [json.loads(line) for line in Path(result["output"]).read_text(encoding="utf-8").splitlines()]

    assert [row["_generator"]["global_index"] for row in plans] == list(range(40, 48))
    assert {row["status"] for row in plans} == {"assigned"}
    resumed = assignments.write_assignment_batch(
        count=10,
        seed=9,
        start_index=40,
        shard_index=1,
        resume=True,
        catalog_path=catalog_path,
        word_pool_path=pool_path,
        assignment_dir=assignment_dir,
        batch_history_dir=batch_dir,
    )
    assert resumed["new_assignments"] == 2
    resumed_plans = [json.loads(line) for line in Path(result["output"]).read_text(encoding="utf-8").splitlines()]
    assert [row["_generator"]["global_index"] for row in resumed_plans] == list(range(40, 50))
    with pytest.raises(assignments.ContractError, match="already exists"):
        assignments.write_assignment_batch(
            count=8,
            seed=9,
            start_index=40,
            shard_index=1,
            catalog_path=catalog_path,
            word_pool_path=pool_path,
            assignment_dir=assignment_dir,
            batch_history_dir=batch_dir,
        )


def test_progress_quarantines_duplicates_and_bad_answer_indexes(tmp_path, monkeypatch):
    training_dir = Path("data/generated_train")
    batch_dir = training_dir / "batches"
    multilingual_dir = training_dir / "multilingual"
    broad_dir = training_dir / "broad-v1"
    monkeypatch.setattr(progress, "TRAINING_DIR", training_dir)
    monkeypatch.setattr(progress, "SHARD_DIRS", (batch_dir, multilingual_dir, broad_dir))
    root = tmp_path
    (root / training_dir).mkdir(parents=True)
    (root / training_dir / "coverage-catalog-v1.json").write_text(
        json.dumps({"languages": [{"tag": "en"}]}), encoding="utf-8"
    )
    (root / broad_dir).mkdir(parents=True)

    valid = {
        "id": "question-1",
        "problem_group_id": "group-1",
        "split": "train",
        "operation_id": "evidence",
        "topic": "archival records",
        "language": "en",
        "difficulty": "medium",
        "context": "The record lists two dates and one source.",
        "question": "Which date is stated first?",
        "options": ["12 May", "14 May"],
        "answer_index": 0,
        "explanation": "The record lists 12 May before 14 May.",
        "provenance": "Luna-generated; label not independently verified.",
    }
    duplicate_content = dict(valid, id="question-2", problem_group_id="group-2")
    bad_answer = dict(valid, id="question-3", problem_group_id="group-3", answer_index=2)
    shard = root / broad_dir / "shard-001.jsonl"
    shard.write_text(
        "".join(json.dumps(row) + "\n" for row in (valid, duplicate_content, bad_answer)),
        encoding="utf-8",
    )
    staged = dict(
        valid,
        id="staged-question",
        problem_group_id="staged-group",
        context="A temporary staged question has different source data.",
        question="Which record is in the staged sample?",
        explanation="The staged record is separate from the published shard.",
    )
    (root / broad_dir / ".luna-staging.jsonl").write_text(
        json.dumps(staged) + "\n", encoding="utf-8"
    )

    report = progress.scan(root, target=10)

    assert assignments._reasons(valid, Path("broad-v1/shard-001.jsonl"), {"en"}) == []
    assert report["row_counts"]["accepted_rows"] == 1
    assert report["row_counts"]["duplicate_normalized_context_question_options_rows"] == 1
    assert report["row_counts"]["schema_rejected_rows"] == 1
    assert report["row_counts"]["accepted_rows"] == 1
    assert report["row_counts"]["campaign_accepted_rows"] == 0
    assert report["remaining_questions"] == 10
    assert all(not path.startswith("data/generated_train/broad-v1/.") for path in report["scope"]["files_scanned"])
    assert report["accepted_definition"].endswith("does not mean the answer, explanation, or reasoning was semantically verified.")


def test_progress_campaign_target_excludes_old_unassigned_examples(tmp_path, monkeypatch):
    training_dir = Path("data/generated_train")
    batch_dir = training_dir / "batches"
    multilingual_dir = training_dir / "multilingual"
    broad_dir = training_dir / "broad-v1"
    monkeypatch.setattr(progress, "TRAINING_DIR", training_dir)
    monkeypatch.setattr(progress, "SHARD_DIRS", (batch_dir, multilingual_dir, broad_dir))
    root = tmp_path
    (root / training_dir).mkdir(parents=True)
    (root / training_dir / "coverage-catalog-v1.json").write_text(
        json.dumps({"languages": [{"tag": "en"}]}), encoding="utf-8"
    )
    (root / broad_dir).mkdir(parents=True)

    campaign = {
        "id": "broad-v1-assignment-0001",
        "problem_group_id": "broad-v1-group-0001",
        "split": "train",
        "operation_id": "evidence",
        "topic": "archival records",
        "language": "en",
        "difficulty": "medium",
        "context": "A campaign record has a distinct source and date.",
        "question": "Which date does the campaign record state?",
        "options": ["12 May", "14 May"],
        "answer_index": 0,
        "explanation": "The source gives 12 May.",
        "provenance": "Luna-generated; label not independently verified.",
    }
    legacy = dict(
        campaign,
        id="broad-v1-luna-001-0001",
        problem_group_id="broad-v1-luna-001-g-0001",
        context="An older record has separate source data.",
        question="Which date does this older record state?",
        explanation="The older source gives 12 May.",
    )
    (root / broad_dir / "luna-001.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in (campaign, legacy)),
        encoding="utf-8",
    )

    report = progress.scan(root, target=10)

    assert report["row_counts"]["accepted_rows"] == 2
    assert report["row_counts"]["campaign_accepted_rows"] == 1
    assert report["remaining_questions"] == 9
    assert report["all_source_remaining_questions"] == 8


def test_reconcile_closes_only_rows_with_output_or_skip_records(tmp_path):
    assignment_dir = tmp_path / "data" / "generated_train" / "assignments"
    question_dir = tmp_path / "data" / "generated_train" / "broad-v1"
    assignment_dir.mkdir(parents=True)
    question_dir.mkdir(parents=True)
    plan_path = assignment_dir / "shard-001.jsonl"
    plans = [
        {"id": "assignment-1", "status": "in_flight", "output": "data/generated_train/broad-v1/luna-001.jsonl"},
        {"id": "assignment-2", "status": "in_flight", "output": "data/generated_train/broad-v1/luna-001.jsonl"},
        {"id": "assignment-3", "status": "in_flight", "output": "data/generated_train/broad-v1/luna-001.jsonl"},
    ]
    plan_path.write_text("".join(json.dumps(row) + "\n" for row in plans), encoding="utf-8")
    (question_dir / "luna-001.jsonl").write_text(
        json.dumps({"id": "assignment-1"}) + "\n", encoding="utf-8"
    )
    (assignment_dir / "skipped-luna-001.jsonl").write_text(
        json.dumps({"plan_id": "assignment-2", "reason": "unsupported language"}) + "\n",
        encoding="utf-8",
    )

    result = reconcile.reconcile_shard(plan_path, question_dir, tmp_path)
    updated = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]

    assert result["pending"] == 1
    assert [row["status"] for row in updated] == ["completed", "skipped", "in_flight"]
    assert updated[1]["skip_reason"] == "unsupported language"

    missing_plan = assignment_dir / "shard-002.jsonl"
    missing_plan.write_text(
        json.dumps({"id": "assignment-4", "status": "in_flight"}) + "\n",
        encoding="utf-8",
    )
    missing = reconcile.reconcile_shard(missing_plan, question_dir, tmp_path)
    assert missing["pending"] == 1
