import gzip
import fcntl
import copy
import hashlib
import os
import json
from collections import Counter
from pathlib import Path

import pytest

from dataset_factory import probability_v2, probability_v2_assignments as planner


def _catalog():
    return {
        "languages": [
            {"tag": tag, "name": tag, "family": "test", "script": "Latin"}
            for tag in ("en", "es", "ja")
        ],
        "domains": [
            {"id": "archive", "name": "Archive", "region": "Global"},
            {"id": "textile", "name": "Textile", "region": "Global"},
        ],
        "operations": [
            {
                "id": operation,
                "name": operation,
                "compatible_domains": ["archive", "textile"],
                "question_forms": ["choose the supported option", "identify the next step"],
            }
            for operation in ("evidence_review", "sequence_reasoning")
        ],
        "input_structures": [{"id": "prose"}, {"id": "table"}],
        "registers": [{"id": "plain"}, {"id": "formal"}],
        "difficulty_bands": [{"id": tag} for tag in ("easy", "medium", "hard")],
        "context_length_bands": [{"id": tag} for tag in ("short", "medium", "long")],
        "choice_counts": [2, 3, 4],
    }


def _write_inputs(root: Path):
    catalog_path = root / "catalog.json"
    catalog_path.write_text(json.dumps(_catalog()), encoding="utf-8")
    pool_path = root / "pool.jsonl.gz"
    with gzip.open(pool_path, "wt", encoding="utf-8") as stream:
        for language in ("en", "fr", "ja", "ar"):
            for index in range(4):
                stream.write(
                    json.dumps(
                        {
                            "word": f"{language}-seed-{index}",
                            "source_language": language,
                            "pool_id": "test-pool",
                            "source_version": "fixture-1",
                        }
                    )
                    + "\n"
                )
    return catalog_path, pool_path


def _small_targets(exact=20):
    return {
        "exact_derived": exact,
        "explicit_stochastic_process": 0,
        "empirical_frequency": 0,
        "teacher_estimate": 0,
        "hard_label": 0,
    }


def _question_for(plan_row, *, choice_count_override=None):
    choice_count = choice_count_override or plan_row["choice_count"]
    options = [f"candidate action {index + 1}" for index in range(choice_count)]
    abstention_text = "I cannot determine this from the provided information"
    options[-1] = abstention_text
    choice_ids = probability_v2.stable_choice_ids(plan_row["id"], options)
    abstention_index = len(options) - 1
    base = {
        "id": plan_row["id"],
        "problem_group_id": plan_row["problem_group_id"],
        "split": "train",
        "operation_id": plan_row["operation_id"],
        "domain_id": plan_row["domain_id"],
        "choice_count": choice_count,
        "choice_ids": choice_ids,
        "answerability_status": plan_row.get("answerability_status"),
        "unanswerable_reason": plan_row.get("unanswerable_reason"),
        "abstention_choice_id": choice_ids[abstention_index],
        "topic": plan_row["domain_id"],
        "language": plan_row["language"],
        "difficulty": "medium",
        "context": "A record states which action the procedure permits.",
        "question": "Which action is supported?",
        "options": options,
        "explanation": "The procedure requires review before disposal.",
        "provenance": "Luna-generated; label not independently verified.",
    }
    return probability_v2.build_soft_row(
        base,
        "exact_derived",
        distribution_evidence={
            "kind": "exact_derived",
            "choice_ids": choice_ids,
            "probabilities": [
                (
                    {"numerator": 1, "denominator": 1}
                    if plan_row.get("answerability_status") == "unanswerable"
                    else {"numerator": 0, "denominator": 1}
                )
                if index == abstention_index
                else (
                    {"numerator": 0, "denominator": 1}
                    if plan_row.get("answerability_status") == "unanswerable"
                    else {"numerator": 1, "denominator": choice_count - 1}
                )
                for index in range(choice_count)
            ],
            "method": "fixture-rational-method",
        },
    )


def test_plan_shard_is_homogeneous_balanced_deterministic_and_resumable(tmp_path):
    catalog_path, pool_path = _write_inputs(tmp_path)
    assignment_dir = tmp_path / "v2" / "assignments"
    question_dir = tmp_path / "v2" / "questions"
    legacy_dir = tmp_path / "legacy"
    targets = _small_targets()

    result = planner.write_assignment_shard(
        count=8,
        source_kind="exact_derived",
        language_tag="es",
        seed=79,
        catalog_path=catalog_path,
        word_pool_path=pool_path,
        assignment_dir=assignment_dir,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=targets,
    )
    plan_path = Path(result["output"])
    rows = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]

    assert result["language"] == "es"
    assert result["distribution_source_kind"] == "exact_derived"
    assert len(rows) == 8
    assert {row["language"] for row in rows} == {"es"}
    assert {row["distribution_source_kind"] for row in rows} == {"exact_derived"}
    assert {row["status"] for row in rows} == {"assigned"}
    assert {
        row["_planner"]["authoring_guidance_version"] for row in rows
    } == {planner.AUTHORING_GUIDANCE_VERSION}
    assert len({row["id"] for row in rows}) == len(rows)
    assert len({row["problem_group_id"] for row in rows}) == len(rows)
    assert [row["_generator"]["global_index"] for row in rows] == list(range(1, 9))
    assert all(3 <= len(row["_generator"]["seed_words"]) <= 8 for row in rows)
    assert all(len({word["source_language"] for word in row["_generator"]["seed_words"]}) >= 3 for row in rows)
    assert all("question" not in row and "context" not in row and "options" not in row for row in rows)
    cells = Counter(f"{row['operation_id']}|{row['domain_id']}" for row in rows)
    assert max(cells.values()) - min(cells.values()) <= 1
    assert result["progress_file"] == (tmp_path / "v2" / "progress.json").as_posix()
    progress = json.loads(Path(result["progress_file"]).read_text(encoding="utf-8"))
    assert progress["active_assignments"] == 8
    assert progress["source_quotas"] == targets
    assert progress["legacy_hard_label_rows"] == 0
    assert sum(progress["domain_coverage"].values()) == 8
    assert sum(progress["operation_coverage"].values()) == 8
    progress_markdown = Path(result["progress_file"]).with_suffix(".md").read_text(encoding="utf-8")
    assert "Accepted **0 / 20** production rows; **20** still need to be accepted." in progress_markdown
    assert "Source quotas" in progress_markdown
    assert "| Source | Accepted | Active | Target | Still to accept | Not yet assigned |" in progress_markdown
    assert "Prose semantics, language fluency" in progress_markdown

    first_rows = rows[:2]
    resumed = planner.write_assignment_shard(
        count=8,
        source_kind="exact_derived",
        language_tag="es",
        seed=79,
        shard_index=1,
        resume=True,
        catalog_path=catalog_path,
        word_pool_path=pool_path,
        assignment_dir=assignment_dir,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=targets,
    )
    assert resumed["new_assignments"] == 0
    with pytest.raises(planner.ContractError, match="--language-tag is required"):
        planner.write_assignment_shard(
            count=5,
            source_kind="exact_derived",
            seed=79,
            shard_index=1,
            resume=True,
            catalog_path=catalog_path,
            word_pool_path=pool_path,
            assignment_dir=assignment_dir,
            question_dir=question_dir,
            legacy_dir=legacy_dir,
            pilot_file=tmp_path / "missing-pilot.jsonl",
            source_targets=targets,
        )

    extended = planner.write_assignment_shard(
        count=10,
        source_kind="exact_derived",
        language_tag="es",
        seed=79,
        shard_index=1,
        resume=True,
        catalog_path=catalog_path,
        word_pool_path=pool_path,
        assignment_dir=assignment_dir,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=targets,
    )
    updated = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
    assert extended["new_assignments"] == 2
    assert updated[:2] == first_rows
    assert [row["_generator"]["global_index"] for row in updated] == list(range(1, 11))


def test_high_cardinality_plan_sampling_is_deterministic_and_structure_compatible(tmp_path):
    catalog_path, pool_path = _write_inputs(tmp_path)
    catalog = _catalog()
    catalog["operations"] = [
        {
            "id": "taxonomy_outliers",
            "name": "Taxonomy outliers",
            "compatible_domains": ["archive", "textile"],
            "question_forms": ["classify each listed item using the supplied key"],
        }
    ]
    catalog["input_structures"] = [{"id": "table_or_grid"}, {"id": "taxonomy_key"}]
    catalog["choice_counts"] = [2, 3, 4, 5, 16, 32, 64, 128, 265]
    catalog_path.write_text(json.dumps(catalog), encoding="utf-8")
    targets = _small_targets(exact=1_000)

    def plan(directory):
        result = planner.write_assignment_shard(
            count=1_000,
            source_kind="exact_derived",
            language_tag="en",
            seed=13,
            catalog_path=catalog_path,
            word_pool_path=pool_path,
            assignment_dir=directory / "assignments",
            question_dir=directory / "questions",
            legacy_dir=directory / "legacy",
            pilot_file=directory / "missing-pilot.jsonl",
            source_targets=targets,
        )
        return [json.loads(line) for line in Path(result["output"]).read_text(encoding="utf-8").splitlines()]

    rows = plan(tmp_path / "first")
    repeated = plan(tmp_path / "second")
    signature = lambda row: (
        row["choice_count"],
        row["_planner"]["choice_count_tier"],
        row["operation_id"],
        row["input_structure"]["id"],
    )

    assert [signature(row) for row in rows] == [signature(row) for row in repeated]
    assert {row["_planner"]["choice_count_tier"] for row in rows} == {"low", "medium", "high"}
    assert any(row["choice_count"] > 4 for row in rows)
    assert any(row["choice_count"] == 265 for row in rows)
    assert all(row["_planner"]["choice_count"] == row["choice_count"] for row in rows)
    assert {row["_planner"]["choice_count_policy_version"] for row in rows} == {
        planner.CHOICE_COUNT_POLICY_VERSION
    }
    answerability = Counter(row["answerability_status"] for row in rows)
    assert answerability["answerable"] == pytest.approx(800, abs=40)
    assert answerability["unanswerable"] == pytest.approx(200, abs=40)
    assert {row["_planner"]["answerability_policy_version"] for row in rows} == {
        planner.ANSWERABILITY_POLICY_VERSION
    }
    assert all(row["_planner"]["abstention_choice_required"] is True for row in rows)
    reasons = Counter(
        row["unanswerable_reason"] for row in rows if row["answerability_status"] == "unanswerable"
    )
    assert set(reasons) == set(planner.UNANSWERABLE_REASONS)
    assert max(reasons.values()) - min(reasons.values()) <= 1
    assert all(row["input_structure"]["id"] in {"table_or_grid", "taxonomy_key"} for row in rows)


def test_reconcile_closes_completed_skipped_rows_and_refreshes_progress(tmp_path):
    catalog_path, pool_path = _write_inputs(tmp_path)
    assignment_dir = tmp_path / "v2" / "assignments"
    question_dir = tmp_path / "v2" / "questions"
    legacy_dir = tmp_path / "legacy"
    targets = _small_targets(exact=10)
    result = planner.write_assignment_shard(
        count=2,
        source_kind="exact_derived",
        language_tag="es",
        seed=13,
        catalog_path=catalog_path,
        word_pool_path=pool_path,
        assignment_dir=assignment_dir,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=targets,
    )
    plan_path = Path(result["output"])
    plans = [json.loads(line) for line in plan_path.read_text(encoding="utf-8").splitlines()]
    output_path = Path(plans[0]["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_question_for(plans[0])) + "\n", encoding="utf-8")
    with output_path.open("a", encoding="utf-8") as stream:
        stream.write('{"partial":\n')

    partial_progress = planner.check_plan_status(
        catalog_path=catalog_path,
        assignment_dir=assignment_dir,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=targets,
    )
    assert partial_progress["production_rows_accepted"] == 1
    assert partial_progress["active_assignment_rows"] == 1
    assert partial_progress["progress"]["production_rows_remaining_to_accept"] == 9
    assert partial_progress["progress"]["unassigned_target_balance"] == 8
    assert partial_progress["production_target_deficit"] == 8
    assert partial_progress["progress"]["rejected_question_rows"] == 1

    first_reconcile = planner.reconcile_shard(
        plan_path,
        assignment_dir=assignment_dir,
        catalog_path=catalog_path,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=targets,
    )
    assert first_reconcile["statuses"] == {"completed": 1, "pending": 1}
    progress = json.loads(Path(first_reconcile["progress_file"]).read_text(encoding="utf-8"))
    assert progress["accepted_production_rows"] == 1
    assert progress["active_assignments"] == 1
    assert progress["production_rows_remaining_to_accept"] == 9
    assert progress["unassigned_target_balance"] == 8
    assert progress["production_target_deficit"] == 8

    skip_path = Path(plans[0]["skip_file"])
    skip_path.parent.mkdir(parents=True, exist_ok=True)
    skip_path.write_text(
        json.dumps({"plan_id": plans[1]["id"], "reason": "unsupported language tag"}) + "\n",
        encoding="utf-8",
    )
    skipped_progress = planner.check_plan_status(
        catalog_path=catalog_path,
        assignment_dir=assignment_dir,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=targets,
    )
    assert skipped_progress["production_rows_accepted"] == 1
    assert skipped_progress["active_assignment_rows"] == 0
    assert skipped_progress["progress"]["production_rows_remaining_to_accept"] == 9
    assert skipped_progress["production_target_deficit"] == 9

    second_reconcile = planner.reconcile_shard(
        plan_path,
        assignment_dir=assignment_dir,
        catalog_path=catalog_path,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=targets,
    )
    assert second_reconcile["statuses"] == {"completed": 1, "skipped": 1}
    progress = json.loads(Path(second_reconcile["progress_file"]).read_text(encoding="utf-8"))
    assert progress["active_assignments"] == 0
    assert progress["skipped_assignments"] == 1
    assert progress["accepted_production_rows"] == 1
    assert progress["rejected_question_rows"] == 1
    assert progress["unsupported_languages"] == ["es"]

    with pytest.raises(planner.ContractError, match="marked unsupported"):
        planner.write_assignment_shard(
            count=1,
            source_kind="exact_derived",
            language_tag="es",
            seed=13,
            catalog_path=catalog_path,
            word_pool_path=pool_path,
            assignment_dir=assignment_dir,
            question_dir=question_dir,
            legacy_dir=legacy_dir,
            pilot_file=tmp_path / "missing-pilot.jsonl",
            source_targets=targets,
        )


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("operation_id", "wrong_operation", "output_operation_id_mismatch"),
        ("domain_id", "wrong_domain", "output_domain_id_mismatch"),
        ("choice_count", 3, "output_choice_count_mismatch"),
    ],
)
def test_reconcile_fails_when_output_coverage_does_not_match_assignment(
    tmp_path, field, value, expected_reason
):
    catalog_path, pool_path = _write_inputs(tmp_path)
    assignment_dir = tmp_path / "v2" / "assignments"
    question_dir = tmp_path / "v2" / "questions"
    result = planner.write_assignment_shard(
        count=1,
        source_kind="exact_derived",
        language_tag="es",
        seed=31,
        catalog_path=catalog_path,
        word_pool_path=pool_path,
        assignment_dir=assignment_dir,
        question_dir=question_dir,
        legacy_dir=tmp_path / "legacy",
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=_small_targets(exact=2),
    )
    plan_path = Path(result["output"])
    plan_row = json.loads(plan_path.read_text(encoding="utf-8").splitlines()[0])
    if field == "choice_count":
        alternative_count = 2 if plan_row["choice_count"] != 2 else 3
        output = _question_for(plan_row, choice_count_override=alternative_count)
    else:
        output = _question_for(plan_row)
        output[field] = value
    output_path = Path(plan_row["output"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output) + "\n", encoding="utf-8")

    reconciled = planner.reconcile_shard(
        plan_path,
        assignment_dir=assignment_dir,
        catalog_path=catalog_path,
        question_dir=question_dir,
        legacy_dir=tmp_path / "legacy",
        pilot_file=tmp_path / "missing-pilot.jsonl",
        source_targets=_small_targets(exact=2),
    )

    updated_plan = json.loads(plan_path.read_text(encoding="utf-8").splitlines()[0])
    assert reconciled["statuses"] == {"failed": 1}
    assert updated_plan["failure_reason"] == expected_reason


def test_least_covered_language_skips_explicitly_unsupported_tags_and_source_deficits_include_legacy_hard(tmp_path):
    catalog = _catalog()
    catalog["_cells"] = [
        (operation, domain)
        for operation in catalog["operations"]
        for domain_id in operation["compatible_domains"]
        for domain in catalog["domains"]
        if domain["id"] == domain_id
    ]
    history = planner.PlanningHistory(catalog)
    history.unsupported_languages.add("en")
    chosen = planner._choose_language(catalog, history, seed=7, requested_tag=None)
    assert chosen in {"es", "ja"}
    with pytest.raises(planner.ContractError, match="no fallback"):
        planner._choose_language(catalog, history, seed=7, requested_tag="en")

    saved = _accounting_history(tmp_path, [_hard_fixture(f"legacy-{index}") for index in range(4)])
    report = planner.budget_report(
        saved.summary,
        {},
        {
            "exact_derived": 4,
            "explicit_stochastic_process": 4,
            "empirical_frequency": 2,
            "teacher_estimate": 2,
            "hard_label": 4,
        },
    )
    assert report["production_target"] == 16
    assert report["production_rows_remaining_to_accept"] == 12
    assert report["unassigned_target_balance"] == 12
    assert report["source_rows_remaining_to_accept"]["hard_label"] == 0
    assert report["source_target_deficits"]["hard_label"] == 0
    assert report["production_target_deficit"] == 12


def _batch_kwargs(tmp_path):
    catalog_path, pool_path = _write_inputs(tmp_path)
    return {
        "catalog_path": catalog_path,
        "word_pool_path": pool_path,
        "assignment_dir": tmp_path / "v2" / "assignments",
        "question_dir": tmp_path / "v2" / "questions",
        "legacy_dir": tmp_path / "legacy",
        "pilot_file": tmp_path / "missing-pilot.jsonl",
        "source_targets": _small_targets(exact=30),
    }


def _batch_rows(result):
    return [
        json.loads(line)
        for shard in result["shards"]
        for line in Path(shard["output"]).read_text(encoding="utf-8").splitlines()
    ]


def test_batch_retains_history_and_scans_once_with_deterministic_shard_seeds(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    first_kwargs, second_kwargs = _batch_kwargs(first), _batch_kwargs(second)
    specs = [
        {"language_tag": "en", "source_kind": "exact_derived", "count": 4},
        {"language_tag": "es", "source_kind": "exact_derived", "count": 4, "seed": 97},
        {"language_tag": "en", "source_kind": "exact_derived", "count": 4},
    ]
    calls = Counter()
    original_history, original_summary = planner.load_history, probability_v2.summarize

    def history(*args, **kwargs):
        calls["history"] += 1
        return original_history(*args, **kwargs)

    def summary(*args, **kwargs):
        calls["summary"] += 1
        return original_summary(*args, **kwargs)

    monkeypatch.setattr(planner, "load_history", history)
    monkeypatch.setattr(probability_v2, "summarize", summary)
    result = planner.write_assignment_batch(specs, seed=11, **first_kwargs)
    assert calls == {"history": 1}
    rows = _batch_rows(result)
    assert [row["_generator"]["global_index"] for row in rows] == list(range(1, 13))
    assert [row["_generator"]["base_seed"] for row in rows] == [11] * 4 + [97] * 4 + [11] * 4
    assert [shard["shard_index"] for shard in result["shards"]] == [1, 1, 2]
    assert len({row["id"] for row in rows}) == 12
    assert len({row["problem_group_id"] for row in rows}) == 12
    assert [row["_planner"]["source_target_deficit_before_assignment"] for row in rows] == list(range(30, 18, -1))
    assert result["budget_after_planning"]["active_assignment_rows"] == 12
    cells = Counter((row["operation_id"], row["domain_id"]) for row in rows)
    assert set(cells.values()) == {3}
    progress = json.loads(Path(result["progress_file"]).read_text(encoding="utf-8"))
    assert progress["active_assignments_by_language"] == {"en": 8, "es": 4}
    assert progress["planned_assignments_by_status_and_source_kind"] == {"assigned:exact_derived": 12}
    repeated = planner.write_assignment_batch(specs, seed=11, **second_kwargs)
    normalize = lambda row: {key: value for key, value in row.items() if key not in {"output", "skip_file"}}
    assert [normalize(row) for row in rows] == [normalize(row) for row in _batch_rows(repeated)]
    followup = planner.write_assignment_batch(specs[:1], seed=11, **first_kwargs)
    assert followup["shards"][0]["start_index"] == 13
    assert followup["shards"][0]["shard_index"] == 3
    assert followup["budget_after_planning"]["active_assignment_rows"] == 16


def test_batch_reserves_invalid_orphan_outputs_skip_ids_and_existing_paths(tmp_path):
    kwargs = _batch_kwargs(tmp_path)
    output_dir = kwargs["question_dir"] / "exact_derived" / "en"
    output_dir.mkdir(parents=True)
    orphan = output_dir / "shard-0040.jsonl"
    original = json.dumps({"id": "probability-v2-assignment-0000000100"}) + "\n{partial\n"
    orphan.write_text(original, encoding="utf-8")
    skip = output_dir / "skipped-shard-0041.jsonl"
    skip.write_text(json.dumps({"plan_id": "probability-v2-assignment-0000000120"}) + "\n", encoding="utf-8")
    result = planner.write_assignment_batch([
        {"language_tag": "en", "source_kind": "exact_derived", "count": 2}
    ], **kwargs)
    assert result["shards"][0]["start_index"] == 121
    assert result["shards"][0]["shard_index"] == 42
    assert orphan.read_text(encoding="utf-8") == original
    assert kwargs["source_targets"]["exact_derived"] == 30


def test_batch_preflights_combined_quotas_and_language_before_writing(tmp_path):
    kwargs = _batch_kwargs(tmp_path)
    specs = [{"language_tag": "en", "source_kind": "exact_derived", "count": 16}] * 2
    with pytest.raises(planner.ContractError, match="below requested batch count"):
        planner.write_assignment_batch(specs, **kwargs)
    assert not list(kwargs["assignment_dir"].rglob("shard-*.jsonl"))
    planner.set_language_status(
        "es", "unsupported", "fixture", catalog_path=kwargs["catalog_path"],
        assignment_dir=kwargs["assignment_dir"],
    )
    with pytest.raises(planner.ContractError, match="marked unsupported"):
        planner.write_assignment_batch([
            {"language_tag": "en", "source_kind": "exact_derived", "count": 2},
            {"language_tag": "es", "source_kind": "exact_derived", "count": 2},
        ], **kwargs)
    assert not list(kwargs["assignment_dir"].rglob("shard-*.jsonl"))


def test_batch_credits_accepted_rows_preserves_old_plans_and_tracks_each_source(tmp_path):
    kwargs = _batch_kwargs(tmp_path)
    kwargs["source_targets"] = {**_small_targets(exact=4), "hard_label": 3}
    initial = planner.write_assignment_shard(
        count=1, source_kind="exact_derived", language_tag="en", **kwargs
    )
    old_path = Path(initial["output"])
    old_bytes = old_path.read_bytes()
    old_row = json.loads(old_bytes)
    question = Path(initial["question_output"])
    question.parent.mkdir(parents=True)
    question.write_text(json.dumps(_question_for(old_row)) + "\n", encoding="utf-8")
    result = planner.write_assignment_batch([
        {"language_tag": "es", "source_kind": "exact_derived", "count": 2},
        {"language_tag": "en", "source_kind": "hard_label", "count": 3},
    ], **kwargs)
    assert old_path.read_bytes() == old_bytes
    assert result["budget_before"]["production_rows_accepted"] == 1
    assert result["budget_after_planning"]["source_rows_committed"]["exact_derived"] == 3
    assert result["budget_after_planning"]["source_rows_committed"]["hard_label"] == 3
    assert result["budget_after_planning"]["production_target_deficit"] == 1
    assert [row["_generator"]["global_index"] for row in _batch_rows(result)] == list(range(2, 7))


def test_plan_batch_cli_accepts_json_specs(tmp_path, capsys):
    kwargs = _batch_kwargs(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([
        {"language_tag": "es", "source_kind": "exact_derived", "count": 2, "seed": 17}
    ]), encoding="utf-8")
    args = ["plan-batch", "--spec", str(spec), "--seed", "31"]
    for key in ("catalog_path", "word_pool_path", "assignment_dir", "question_dir", "legacy_dir", "pilot_file"):
        cli_key = {"catalog_path": "catalog", "word_pool_path": "word-pool"}.get(key, key.replace("_", "-"))
        args.extend([f"--{cli_key}", str(kwargs[key])])
    assert planner._cli(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["new_assignments"] == 2
    assert {row["_generator"]["base_seed"] for row in _batch_rows(result)} == {17}


def test_batch_and_single_shard_reject_concurrent_planner(tmp_path):
    kwargs = _batch_kwargs(tmp_path)
    kwargs["assignment_dir"].mkdir(parents=True)
    with (kwargs["assignment_dir"] / ".planning.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(planner.ContractError, match="another planner"):
            planner.write_assignment_batch([
                {"language_tag": "en", "source_kind": "exact_derived", "count": 2}
            ], **kwargs)
        with pytest.raises(planner.ContractError, match="another planner"):
            planner.write_assignment_shard(count=2, source_kind="exact_derived", **kwargs)


def test_atomic_new_shard_does_not_replace_racing_file(tmp_path, monkeypatch):
    target = tmp_path / "shard-0001.jsonl"
    original_link = planner.os.link

    def racing_link(source, destination):
        Path(destination).write_text("owned by another writer\n", encoding="utf-8")
        return original_link(source, destination)

    monkeypatch.setattr(planner.os, "link", racing_link)
    with pytest.raises(planner.ContractError, match="already exists"):
        planner._atomic_jsonl(target, [{"id": "new"}], overwrite=False)
    assert target.read_text(encoding="utf-8") == "owned by another writer\n"
    assert list(tmp_path.iterdir()) == [target]


def _hard_fixture(identifier):
    return {
        "id": identifier, "problem_group_id": f"group-{identifier}", "split": "train",
        "language": "en", "domain_id": "archive", "operation_id": "evidence_review",
        "context": f"The synthetic record {identifier} requires preservation.",
        "question": "Which action does this record require?", "options": ["preserve", "discard"],
        "answer_index": 0, "difficulty": "medium", "explanation": "The record states preservation.",
        "provenance": "luna-generated; label not independently verified",
    }


def _accounting_history(tmp_path, rows, *, issues=()):
    catalog_path, _ = _write_inputs(tmp_path)
    source = tmp_path / "questions.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    issue_file = tmp_path / "known-issues.jsonl"
    issue_file.write_text("".join(json.dumps(row) + "\n" for row in issues), encoding="utf-8")
    catalog = planner.assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    return planner.load_history(catalog, question_paths=[source], assignment_dir=tmp_path / "assignments", known_issues_path=issue_file)


def test_usable_original_accounting_excludes_pending_keeps_review_and_deduplicates_variants(tmp_path):
    first = _hard_fixture("first")
    exact = {**first, "id": "exact-copy", "problem_group_id": "another-group"}
    grouped = {**_hard_fixture("group-copy"), "problem_group_id": first["problem_group_id"]}
    marked = {**_hard_fixture("marked-variant"), "variant_of": "first"}
    augmented = {**_hard_fixture("augmented"), "_augmentation": {"source_question_id": "first"}}
    excluded, review = _hard_fixture("excluded"), _hard_fixture("review")
    pilot = {**_hard_fixture("pilot"), "pilot_status": probability_v2.EXCLUDED_PILOT_STATUS}
    history = _accounting_history(tmp_path, [excluded, first, exact, grouped, marked, augmented, review, pilot], issues=[
        {"training_status": "exclude_pending_review", "question_ids": ["excluded"]},
        {"training_status": "review_recommended", "question_ids": ["review"]},
    ])
    summary = history.summary
    assert summary["accepted_rows"] == 8
    assert summary["structural_production_rows"] == 7
    assert summary["eligible_rows"] == 6
    assert summary["eligible_original_rows"] == 4
    assert summary["unique_eligible_original_rows"] == 2
    assert summary["known_issue_excluded_rows"] == 1
    assert summary["review_recommended_eligible_rows"] == 1
    assert summary["explicit_variant_rows"] == 2
    assert summary["grouped_variant_rows"] == summary["duplicate_exact_content_rows"] == 1
    assert summary["legacy_hard_label_rows"] == 2
    assert summary["structural_legacy_hard_label_rows"] == 8
    report = planner.budget_report(summary, {}, {**_small_targets(exact=0), "hard_label": 8})
    assert report["production_rows_accepted"] == 2
    assert report["production_rows_remaining_to_accept"] == 6
    assert report["source_target_deficits"]["hard_label"] == 6
    assert "excluded" in history.question_ids
    assert "excluded" in history.identifiers


def test_canonical_identity_and_split_checks_match_structural_summary(tmp_path):
    valid, conflicted = _hard_fixture("valid"), _hard_fixture("conflict")
    invalid_split = {**_hard_fixture("conflict-test"), "split": "test", "problem_group_id": conflicted["problem_group_id"]}
    history = _accounting_history(tmp_path, [valid, copy.deepcopy(valid), conflicted, invalid_split])
    canonical = probability_v2.summarize([tmp_path / "questions.jsonl"])
    assert history.summary["accepted_rows"] == canonical["accepted_rows"] == 1
    assert history.summary["structural_production_rows"] == canonical["production_eligible_rows"]
    assert history.summary["duplicate_row_ids_rejected"] == 1
    assert history.summary["problem_group_split_conflicts"] == 1
    assert history.summary["unique_eligible_original_rows"] == 1
    assert "conflict" in history.identifiers


def test_exact_content_attribution_is_stable_across_source_file_and_option_order(tmp_path):
    catalog_path, _ = _write_inputs(tmp_path)
    catalog = planner.assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    def soft(identifier, source):
        row = _hard_fixture(identifier)
        row.pop("answer_index")
        row["context"] = "One synthetic process always retains the record."
        return probability_v2.build_soft_row(row, source, distribution_evidence={
            "kind": source, "probabilities": [{"numerator": 1, "denominator": 1}, {"numerator": 0, "denominator": 1}],
        })
    earlier = soft("process-first", "explicit_stochastic_process")
    later = probability_v2.shuffle_choices(soft("exact-second", "exact_derived"), [1, 0])
    first, second = tmp_path / "a-source.jsonl", tmp_path / "z-source.jsonl"
    first.write_text(json.dumps(earlier) + "\n")
    second.write_text(json.dumps(later) + "\n")
    issues = tmp_path / "issues.jsonl"
    issues.write_text("")
    histories = [planner.load_history(catalog, question_paths=paths, assignment_dir=tmp_path / "assignments", known_issues_path=issues) for paths in ([first, second], [second, first])]
    for history in histories:
        assert history.summary["unique_eligible_original_rows"] == 1
        assert history.summary["production_eligible_rows_by_source_kind"] == {"explicit_stochastic_process": 1}
        assert history.summary["duplicate_exact_content_rows"] == 1


def test_excluded_completed_assignments_stay_closed_and_replacement_plans_use_new_ids(tmp_path):
    kwargs = _batch_kwargs(tmp_path)
    kwargs["source_targets"] = _small_targets(exact=2)
    initial = planner.write_assignment_shard(count=2, source_kind="exact_derived", language_tag="en", **kwargs)
    plan_path = Path(initial["output"])
    plans = [json.loads(line) for line in plan_path.read_text().splitlines()]
    old_plan = plan_path.read_bytes()
    output = Path(initial["question_output"])
    output.parent.mkdir(parents=True)
    output.write_text("".join(json.dumps(_question_for(plan)) + "\n" for plan in plans))
    old_output = output.read_bytes()
    issues = kwargs["assignment_dir"].parent / "known-issues-v1.jsonl"
    issues.write_text(json.dumps({"training_status": "exclude_pending_review", "question_ids": [plan["id"] for plan in plans]}) + "\n")
    result = planner.write_assignment_batch([
        {"language_tag": "en", "source_kind": "exact_derived", "count": 2}
    ], **kwargs)
    assert result["budget_before"]["production_rows_accepted"] == 0
    assert result["budget_before"]["active_assignment_rows"] == 0
    assert result["budget_before"]["production_target_deficit"] == 2
    assert result["shards"][0]["start_index"] == 3
    assert result["budget_after_planning"]["active_assignment_rows"] == 2
    assert plan_path.read_bytes() == old_plan
    assert output.read_bytes() == old_output


def test_overquota_source_cannot_satisfy_a_different_source_target(tmp_path):
    history = _accounting_history(tmp_path, [_hard_fixture(f"hard-{index}") for index in range(4)])
    targets = {**_small_targets(exact=2), "hard_label": 2}
    report = planner.budget_report(history.summary, {}, targets)
    assert report["production_rows_accepted"] == report["production_target"] == 4
    assert report["production_target_deficit"] == report["production_rows_remaining_to_accept"] == 2
    assert report["source_target_deficits"]["exact_derived"] == 2
    assert report["source_targets"] == targets


@pytest.mark.parametrize("change", ["raw_summary", "total", "source", "alias", "exclusion", "bool"])
def test_budget_fails_closed_on_missing_or_inconsistent_accounting(tmp_path, change):
    summary = copy.deepcopy(_accounting_history(tmp_path, [_hard_fixture("row")]).summary)
    if change == "raw_summary":
        summary.pop("accounting_version")
    elif change == "total":
        summary["eligible_rows"] = 3
    elif change == "source":
        summary["production_eligible_rows_by_source_kind"] = {"teacher_estimate": 2}
    elif change == "alias":
        summary["production_eligible_rows"] = 5
    elif change == "exclusion":
        summary["known_issue_excluded_rows"] = 1
    else:
        summary["eligible_rows"] = True
    with pytest.raises(planner.ContractError):
        planner.budget_report(summary, {}, _small_targets())


def test_planning_fails_closed_on_changed_issues_and_ram_reserve(tmp_path, monkeypatch):
    history = _accounting_history(tmp_path, [_hard_fixture("row")])
    issues = tmp_path / "known-issues.jsonl"
    original_fingerprint = planner.training_snapshot.exact_content_fingerprint
    def changed_issues(row):
        with issues.open("a") as stream:
            stream.write(json.dumps({"training_status": "exclude_pending_review", "question_ids": [row["id"]]}) + "\n")
        return original_fingerprint(row)
    monkeypatch.setattr(planner.training_snapshot, "exact_content_fingerprint", changed_issues)
    catalog_path = tmp_path / "catalog.json"
    loaded = planner.assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    with pytest.raises(planner.ContractError, match="sidecar changed"):
        planner.load_history(loaded, question_paths=[tmp_path / "questions.jsonl"], assignment_dir=tmp_path / "assignments", known_issues_path=issues)
    monkeypatch.setattr(planner.training_snapshot, "exact_content_fingerprint", original_fingerprint)
    original_read = Path.read_text
    def limited_ram(path, *args, **kwargs):
        if path == Path("/proc/meminfo"):
            return "MemAvailable: 12582911 kB\n"
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", limited_ram)
    with pytest.raises(planner.ContractError, match="12 GiB"):
        planner.load_history(loaded, question_paths=[tmp_path / "questions.jsonl"], assignment_dir=tmp_path / "assignments", known_issues_path=issues)


def test_default_production_exclusion_sidecar_is_required(tmp_path, monkeypatch):
    catalog_path, _ = _write_inputs(tmp_path)
    catalog = planner.assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    assignment_dir = tmp_path / "assignments"
    monkeypatch.setattr(planner, "ASSIGNMENT_DIR", assignment_dir)
    with pytest.raises(planner.ContractError, match="sidecar is missing"):
        planner.load_history(catalog, question_paths=[], assignment_dir=assignment_dir)


@pytest.mark.parametrize("data", [
    '{"training_status":"exclude_pending_review","question_ids":"not-a-list"}\n',
    '{"training_status":"exclude_pending_review","question_ids":["row"]}',
    '{broken}\n',
])
def test_invalid_exclusion_sidecar_cannot_fall_back_to_raw_credit(tmp_path, data):
    _accounting_history(tmp_path, [_hard_fixture("row")])
    issues = tmp_path / "known-issues.jsonl"
    issues.write_text(data)
    catalog = planner.assignments.load_catalog(tmp_path / "catalog.json", max_choice_count=probability_v2.MAX_CHOICES)
    with pytest.raises(planner.ContractError):
        planner.load_history(catalog, question_paths=[tmp_path / "questions.jsonl"], assignment_dir=tmp_path / "assignments", known_issues_path=issues)


def test_frozen_prefix_allows_append_and_new_files_without_double_crediting_assignments(tmp_path, monkeypatch):
    kwargs = _batch_kwargs(tmp_path)
    first = planner.write_assignment_shard(count=3, source_kind="exact_derived", language_tag="en", **kwargs)
    second = planner.write_assignment_shard(count=1, source_kind="exact_derived", language_tag="es", **kwargs)
    plans = [json.loads(line) for line in Path(first["output"]).read_text().splitlines()]
    later_plan = json.loads(Path(second["output"]).read_text())
    source = Path(first["question_output"])
    source.parent.mkdir(parents=True)
    initial = (json.dumps(_question_for(plans[0])) + "\n").encode()
    source.write_bytes(initial)
    new_file = Path(second["question_output"])
    original = planner.training_snapshot.exact_content_fingerprint
    calls = Counter()

    def append_during_parse(row):
        calls[row["id"]] += 1
        with source.open("ab") as stream:
            stream.write((json.dumps(_question_for(plans[1])) + "\n").encode())
        new_file.parent.mkdir(parents=True)
        new_file.write_text(json.dumps(_question_for(later_plan)) + "\n")
        return original(row)

    monkeypatch.setattr(planner.training_snapshot, "exact_content_fingerprint", append_during_parse)
    catalog = planner.assignments.load_catalog(kwargs["catalog_path"], max_choice_count=probability_v2.MAX_CHOICES)
    history = planner.load_history(catalog, question_paths=[kwargs["question_dir"]], assignment_dir=kwargs["assignment_dir"])
    assert calls == {plans[0]["id"]: 1}
    assert history.summary["unique_eligible_original_rows"] == 1
    assert history.active_assignments == 3
    assert plans[1]["id"] not in history.question_ids
    assert later_plan["id"] not in history.question_ids
    assert later_plan["id"] in history.assignment_ids
    assert history.max_index == 4
    budget = planner.budget_report(history.summary, history.active_by_source, kwargs["source_targets"])
    assert budget["production_rows_committed"] == 4
    prefix = history.summary["question_prefixes"][0]
    assert prefix["prefix_bytes"] == prefix["initial_size_bytes"] == len(initial)
    assert prefix["prefix_sha256"] == hashlib.sha256(initial).hexdigest()
    assert prefix["state"] == "append_growth"
    assert prefix["growth_bytes"] == source.stat().st_size - len(initial)
    assert history.summary["new_files_omitted"] == [new_file.as_posix()]
    interval = history.summary["scan_interval"]
    assert interval["started_at"] <= interval["finished_at"]


@pytest.mark.parametrize("mutation", ["rewrite", "rewrite_and_append", "truncate", "replace", "rewrite_restore_mtime"])
def test_frozen_prefix_rejects_destructive_changes_during_parse(tmp_path, monkeypatch, mutation):
    _accounting_history(tmp_path, [_hard_fixture("initial")])
    source = tmp_path / "questions.jsonl"
    initial = source.read_bytes()
    before = source.stat()
    original = planner.training_snapshot.exact_content_fingerprint
    def mutate(row):
        changed = initial.replace(b"preservation", b"destruction!")
        if mutation == "truncate":
            source.write_bytes(b"")
        elif mutation == "replace":
            replacement = tmp_path / "replacement.jsonl"
            replacement.write_bytes(initial)
            os.replace(replacement, source)
        elif mutation == "rewrite_and_append":
            source.write_bytes(changed + (json.dumps(_hard_fixture("appended")) + "\n").encode())
        else:
            source.write_bytes(changed)
            if mutation == "rewrite_restore_mtime":
                os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        return original(row)
    monkeypatch.setattr(planner.training_snapshot, "exact_content_fingerprint", mutate)
    catalog = planner.assignments.load_catalog(tmp_path / "catalog.json", max_choice_count=probability_v2.MAX_CHOICES)
    with pytest.raises(planner.ContractError, match="rewritten|truncated|replaced"):
        planner.load_history(catalog, question_paths=[source], assignment_dir=tmp_path / "assignments", known_issues_path=tmp_path / "known-issues.jsonl")


def test_frozen_prefix_rejects_rewriting_a_file_before_its_parse_even_with_growth(tmp_path, monkeypatch):
    catalog_path, _ = _write_inputs(tmp_path)
    first, second = tmp_path / "a-first.jsonl", tmp_path / "z-later.jsonl"
    first.write_text(json.dumps(_hard_fixture("first")) + "\n")
    second.write_text(json.dumps(_hard_fixture("later")) + "\n")
    original = planner.training_snapshot.exact_content_fingerprint
    def rewrite_future(row):
        if row["id"] == "first":
            second.write_text(json.dumps(_hard_fixture("rewritten")) + "\n" + json.dumps(_hard_fixture("new")) + "\n")
        return original(row)
    monkeypatch.setattr(planner.training_snapshot, "exact_content_fingerprint", rewrite_future)
    catalog = planner.assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    with pytest.raises(planner.ContractError, match="frozen prefix"):
        planner.load_history(catalog, question_paths=[second, first], assignment_dir=tmp_path / "assignments")


def test_initial_partial_tail_is_not_parsed_when_writer_completes_it(tmp_path, monkeypatch):
    _accounting_history(tmp_path, [_hard_fixture("complete")])
    source = tmp_path / "questions.jsonl"
    complete = source.read_bytes()
    tail = (json.dumps(_hard_fixture("tail")) + "\n").encode()
    halfway = len(tail) // 2
    with source.open("ab") as stream:
        stream.write(tail[:halfway])
    original = planner.training_snapshot.exact_content_fingerprint
    def finish_tail(row):
        with source.open("ab") as stream:
            stream.write(tail[halfway:])
        return original(row)
    monkeypatch.setattr(planner.training_snapshot, "exact_content_fingerprint", finish_tail)
    catalog = planner.assignments.load_catalog(tmp_path / "catalog.json", max_choice_count=probability_v2.MAX_CHOICES)
    history = planner.load_history(catalog, question_paths=[source], assignment_dir=tmp_path / "assignments")
    assert history.summary["accepted_rows"] == 1
    assert history.summary["malformed_json_rows"] == 0
    assert "tail" not in history.question_ids
    prefix = history.summary["question_prefixes"][0]
    assert prefix["prefix_bytes"] == len(complete)
    assert prefix["initial_size_bytes"] == len(complete) + halfway
    assert prefix["prefix_sha256"] == hashlib.sha256(complete).hexdigest()
    assert prefix["state"] == "append_growth"
