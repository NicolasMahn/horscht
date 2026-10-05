import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from dataset_factory import authoring_writer as writer, probability_v2 as v2


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _job(tmp_path, *, source="exact_derived", status="answerable", count=3, size=10):
    plan, output, skip = (tmp_path / name for name in ("plan.jsonl", "output.jsonl", "skipped.jsonl"))
    plans = [
        {
            "schema_version": "probability-v2-assignment-v1",
            "id": f"assignment-{index}", "problem_group_id": f"group-{index}",
            "split": "train", "language": "es", "domain_id": "archive",
            "operation_id": "evidence_review", "choice_count": count,
            "difficulty_band": {"id": "medium"}, "status": "assigned",
            "answerability_status": status,
            "unanswerable_reason": "missing_context" if status == "unanswerable" else None,
            "distribution_source_kind": source,
            "output": str(output), "skip_file": str(skip),
            "_generator": {"global_index": index, "base_seed": 71},
            "_planner": {"authoring_guidance_version": "probability-v2-semantic-grounding-v2"},
        }
        for index in range(size)
    ]
    _jsonl(plan, plans)
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps({
        "plan": str(plan), "output": str(output), "skip_file": str(skip),
        "source_kind": source, "language": "es",
        "remaining_ids": [row["id"] for row in plans],
    }), encoding="utf-8")
    return job_path, plans, output, skip


def _draft(plan):
    count = plan["choice_count"]
    unanswerable = plan["answerability_status"] == "unanswerable"
    target = [0] * count
    target[-1 if unanswerable else 0] = 1
    draft = {
        "assignment_id": plan["id"],
        "context": "La ficha indica que el documento debe conservarse.",
        "question": "¿Qué acción establece la ficha?",
        "options": [f"Acción {index + 1}" for index in range(count - 1)] + ["No se puede determinar con la información dada"],
        "explanation": "La acción indicada está escrita en la ficha.",
        "abstention_index": count - 1,
        "abstention_text": "No se puede determinar con la información dada",
    }
    source = plan["distribution_source_kind"]
    if source == "hard_label":
        draft["answer_index"] = count - 1 if unanswerable else 0
    elif source == "teacher_estimate":
        draft["target_distribution"] = target
        draft["distribution_evidence"] = {
            "kind": source, "model_id": "fixture", "model_version": "1",
            "estimation_method": "fixture-estimate",
        }
    elif source == "empirical_frequency":
        draft["distribution_evidence"] = {"kind": source, "counts": target, "sample_size": 1}
    else:
        draft["distribution_evidence"] = {
            "kind": source,
            "probabilities": [{"numerator": value, "denominator": 1} for value in target],
        }
    return draft


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("source", ["exact_derived", "explicit_stochastic_process", "empirical_frequency", "teacher_estimate", "hard_label"])
@pytest.mark.parametrize("status", ["answerable", "unanswerable"])
def test_builds_plan_envelope_for_each_source_and_abstention_target(tmp_path, source, status):
    job, plans, output, _ = _job(tmp_path, source=source, status=status)
    drafts = tmp_path / "drafts.jsonl"
    _jsonl(drafts, [_draft(plans[5]), _draft(plans[1])])
    result = writer.append_drafts(job, drafts)
    assert result == {"appended": 2, "already_present": 0}
    rows = _read(output)
    assert [row["id"] for row in rows] == ["assignment-5", "assignment-1"]
    assert [row["_generator"]["global_index"] for row in rows] == [5, 1]
    for row, plan in zip(rows, [plans[5], plans[1]]):
        assert not v2.validate_v2_row(row)
        assert row["semantic_review_status"] == "author_self_review_only"
        assert row["language_review_status"] == "author_self_review_only"
        assert row["_assignment_metadata"] == plan
        for field in ("id", "problem_group_id", "choice_count", "language", "operation_id", "domain_id", "answerability_status", "unanswerable_reason", "_generator", "_planner"):
            assert row[field] == plan[field]
        assert row["abstention_choice_id"] == row["choice_ids"][-1]
        if source == "hard_label":
            assert "distribution_source_kind" not in row
        else:
            assert row["distribution_source_kind"] == source


def _required_plan(plan):
    plan["question_form"] = {"id": "synthetic-law", "question_kind": "outcome_distribution"}
    plan["_planner"].update(
        target_shape_requirement="multiple_substantive_support-v1",
        target_shape_selection={"status": "required"},
    )


def _law_draft(plan):
    draft = _draft(plan)
    evidence = draft["distribution_evidence"]
    evidence.update(question_kind="outcome_distribution", mechanism_assumptions=["Synthetic stated conditioning"],
                    outcome_partition="Two mutually exclusive substantive categories cover the stated sample.")
    source = plan["distribution_source_kind"]
    if source == "teacher_estimate":
        draft["target_distribution"] = [0.25, 0.75, 0]
    elif source == "empirical_frequency":
        evidence.update(counts=[1, 3, 0], sample_size=4)
    else:
        evidence["probabilities"] = [{"numerator": n, "denominator": 4} for n in (1, 3, 0)]
    return draft


@pytest.mark.parametrize("source", ["exact_derived", "explicit_stochastic_process", "empirical_frequency", "teacher_estimate"])
def test_marked_targets_are_derived_unchanged_and_retry_does_not_double_count(tmp_path, source):
    job, plans, output, _ = _job(tmp_path, source=source, size=1)
    _required_plan(plans[0])
    _jsonl(tmp_path / "plan.jsonl", plans)
    drafts = tmp_path / "drafts.jsonl"
    draft = _law_draft(plans[0])
    before = json.dumps(draft, sort_keys=True)
    _jsonl(drafts, [draft])
    result = writer.append_drafts(job, drafts)
    assert result["multiple_substantive_support_appended"] == 1
    assert result["required_multiple_support_appended"] == 1
    row = _read(output)[0]
    assert row["target_distribution"] == [0.25, 0.75, 0]
    assert row["distribution_evidence"]["mechanism_assumptions"] == ["Synthetic stated conditioning"]
    assert row["_assignment_metadata"] == plans[0]
    assert json.dumps(draft, sort_keys=True) == before
    preserved = output.read_bytes()
    retry = writer.append_drafts(job, drafts)
    assert retry["multiple_substantive_support_appended"] == retry["required_multiple_support_appended"] == 0
    assert output.read_bytes() == preserved


@pytest.mark.parametrize("failure,code", [
    ("one_hot", "required_multiple_support_missing"),
    ("assumptions", "missing_distribution_task_evidence"),
    ("partition", "missing_distribution_task_evidence"),
    ("question_kind", "missing_distribution_task_evidence"),
    ("idk", "required_abstention_mass_not_zero"),
    ("evidence_mismatch", "invalid_v2_row"),
])
def test_marked_contract_failures_never_append_or_redistribute(tmp_path, failure, code):
    job, plans, output, _ = _job(tmp_path, size=2, source="teacher_estimate")
    for plan in plans:
        _required_plan(plan)
    _jsonl(tmp_path / "plan.jsonl", plans)
    valid, invalid = _law_draft(plans[0]), _law_draft(plans[1])
    if failure == "one_hot":
        invalid["target_distribution"] = [1, 0, 0]
    elif failure == "assumptions":
        invalid["distribution_evidence"]["mechanism_assumptions"] = []
    elif failure == "partition":
        invalid["distribution_evidence"].pop("outcome_partition")
    elif failure == "question_kind":
        invalid["distribution_evidence"]["question_kind"] = "unknown_realized_outcome"
    elif failure == "idk":
        invalid["target_distribution"] = [.25, .75 - 1e-12, 1e-12]
    else:
        plans[1]["distribution_source_kind"] = "exact_derived"
        invalid["distribution_evidence"] = {"kind": "exact_derived", "probabilities": [
            {"numerator": n, "denominator": 4} for n in (1, 3, 0)]}
        invalid["target_distribution"] = [.5, .5, 0]
        # Exercise the post-build check directly without changing a homogeneous job.
        with pytest.raises(writer.WriterError, match=code):
            writer._build_row(plans[1], invalid)
        assert not output.exists()
        return
    drafts = tmp_path / "drafts.jsonl"
    _jsonl(drafts, [valid, invalid])
    with pytest.raises(writer.WriterError, match=code):
        writer.append_drafts(job, drafts)
    assert not output.exists()


@pytest.mark.parametrize("source,status,count", [
    ("hard_label", "answerable", 3), ("exact_derived", "unanswerable", 3), ("exact_derived", "answerable", 2),
])
def test_marked_incompatible_plan_fails_without_changing_unmarked_contract(tmp_path, source, status, count):
    _, plans, _, _ = _job(tmp_path, source=source, status=status, count=count, size=1)
    draft = _draft(plans[0])
    assert writer._build_row(plans[0], draft)
    _required_plan(plans[0])
    with pytest.raises(writer.WriterError, match="incompatible_target_shape_requirement"):
        writer._build_row(plans[0], draft)


def test_marked_support_uses_abstention_identity_instead_of_fixed_position(tmp_path):
    _, plans, _, _ = _job(tmp_path, size=1)
    _required_plan(plans[0])
    draft = _law_draft(plans[0])
    draft["options"][1], draft["options"][2] = draft["options"][2], draft["options"][1]
    draft["abstention_index"] = 1
    draft["distribution_evidence"]["probabilities"] = [
        {"numerator": n, "denominator": 4} for n in (1, 0, 3)]
    row = writer._build_row(plans[0], draft)
    assert row["target_distribution"] == [.25, 0, .75]
    assert row["abstention_choice_id"] == row["choice_ids"][1]


@pytest.mark.parametrize("failure", ["wrong_id", "not_allowlisted", "duplicate_id", "choice_count", "missing_abstention", "wrong_abstention_text", "wrong_target", "metadata_override"])
def test_invalid_whole_batch_does_not_append_earlier_valid_draft(tmp_path, failure):
    job, plans, output, _ = _job(tmp_path, status="unanswerable")
    drafts = tmp_path / "drafts.jsonl"
    valid, invalid = _draft(plans[0]), _draft(plans[1])
    expected = {
        "wrong_id": "assignment_not_allowed", "not_allowlisted": "assignment_not_allowed",
        "duplicate_id": "duplicate_draft_identity", "choice_count": "choice_count_mismatch",
        "missing_abstention": "invalid_abstention_choice", "wrong_abstention_text": "invalid_abstention_choice",
        "wrong_target": "invalid_v2_row", "metadata_override": "invalid_draft_fields",
    }[failure]
    if failure == "wrong_id":
        invalid["assignment_id"] = "assignment-from-another-shard"
    elif failure == "not_allowlisted":
        data = json.loads(job.read_text())
        data["remaining_ids"].remove(plans[1]["id"])
        job.write_text(json.dumps(data))
    elif failure == "duplicate_id":
        invalid["assignment_id"] = valid["assignment_id"]
    elif failure == "choice_count":
        invalid["options"].append("Otra acción")
    elif failure == "missing_abstention":
        invalid.pop("abstention_index")
    elif failure == "wrong_abstention_text":
        invalid["abstention_text"] = "Texto no presente"
    elif failure == "wrong_target":
        invalid["distribution_evidence"]["probabilities"] = [
            {"numerator": 1 if index == 0 else 0, "denominator": 1} for index in range(3)
        ]
    else:
        invalid["language"] = "en"
    _jsonl(drafts, [valid, invalid])
    with pytest.raises(writer.WriterError, match=expected):
        writer.append_drafts(job, drafts)
    assert not output.exists()


def test_incremental_append_preserves_prior_bytes_and_retry_is_idempotent(tmp_path, monkeypatch):
    job, plans, output, _ = _job(tmp_path)
    drafts = tmp_path / "drafts.jsonl"
    fsync_calls = []
    original_fsync = writer.os.fsync

    def fsync(descriptor):
        fsync_calls.append(descriptor)
        original_fsync(descriptor)

    monkeypatch.setattr(writer.os, "fsync", fsync)
    _jsonl(drafts, [_draft(plans[0])])
    writer.append_drafts(job, drafts)
    first_bytes = output.read_bytes()
    assert len(fsync_calls) == 2  # The new file and its directory must both be durable.
    _jsonl(drafts, [_draft(plans[0]), _draft(plans[7])])
    assert writer.append_drafts(job, drafts) == {"appended": 1, "already_present": 1}
    complete = output.read_bytes()
    assert complete.startswith(first_bytes)
    assert writer.append_drafts(job, drafts) == {"appended": 0, "already_present": 2}
    assert output.read_bytes() == complete
    conflicting = _draft(plans[0])
    conflicting["context"] = "Un texto diferente."
    _jsonl(drafts, [_draft(plans[2]), conflicting])
    with pytest.raises(writer.WriterError, match="retry_content_conflict"):
        writer.append_drafts(job, drafts)
    assert output.read_bytes() == complete


@pytest.mark.parametrize("data, code", [(b'{"id":"assignment-0"}', "incomplete_jsonl_boundary"), (b'{broken}\n', "invalid_jsonl")])
@pytest.mark.parametrize("ledger", ["output", "skip"])
def test_refuses_damaged_local_ledger_without_rewriting(tmp_path, data, code, ledger):
    job, plans, output, skip = _job(tmp_path)
    path = output if ledger == "output" else skip
    path.write_bytes(data)
    drafts = tmp_path / "drafts.jsonl"
    _jsonl(drafts, [_draft(plans[1])])
    with pytest.raises(writer.WriterError, match=code):
        writer.append_drafts(job, drafts)
    assert path.read_bytes() == data


def test_skips_are_explicit_immutable_and_cannot_convert_source_rows(tmp_path):
    job, plans, output, skip = _job(tmp_path)
    drafts = tmp_path / "drafts.jsonl"
    _jsonl(drafts, [{"assignment_id": "assignment-0", "reason": "author cannot support this language"}])
    assert writer.skip_assignments(job, drafts) == {"appended": 1, "already_present": 0}
    preserved = skip.read_bytes()
    assert _read(skip)[0] == {
        "id": "assignment-0", "assignment_id": "assignment-0", "status": "skipped",
        "reason": "author cannot support this language", "language": "es", "source_kind": "exact_derived",
    }
    assert writer.skip_assignments(job, drafts) == {"appended": 0, "already_present": 1}
    assert skip.read_bytes() == preserved
    _jsonl(drafts, [_draft(plans[0])])
    with pytest.raises(writer.WriterError, match="assignment_already_skipped"):
        writer.append_drafts(job, drafts)
    _jsonl(drafts, [_draft(plans[1])])
    writer.append_drafts(job, drafts)
    output_bytes = output.read_bytes()
    _jsonl(drafts, [{"assignment_id": "assignment-1", "reason": "cannot replace written row"}])
    with pytest.raises(writer.WriterError, match="assignment_already_written"):
        writer.skip_assignments(job, drafts)
    assert skip.read_bytes() == preserved
    assert output.read_bytes() == output_bytes


def test_max_eight_drafts_and_own_path_lock(tmp_path):
    job, plans, output, _ = _job(tmp_path)
    drafts = tmp_path / "drafts.jsonl"
    _jsonl(drafts, [_draft(plan) for plan in plans[:9]])
    with pytest.raises(writer.WriterError, match="draft_batch_must_have_1_to_8_rows"):
        writer.append_drafts(job, drafts)
    _jsonl(drafts, [_draft(plan) for plan in plans[:8]])
    with output.with_name(f".{output.name}.authoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(writer.WriterError, match="output_locked"):
            writer.append_drafts(job, drafts)
    assert writer.append_drafts(job, drafts) == {"appended": 8, "already_present": 0}


def test_cli_errors_do_not_print_draft_content_or_paths(tmp_path, capsys):
    job, plans, _, _ = _job(tmp_path)
    drafts = tmp_path / "drafts.jsonl"
    draft = _draft(plans[0])
    draft["context"] = "private content must not appear in error logs"
    draft["assignment_id"] = "unknown-private-id"
    _jsonl(drafts, [draft])
    assert writer._cli(["append", "--job", str(job), "--drafts", str(drafts)]) == 2
    captured = capsys.readouterr()
    assert captured.out == '{"error_code": "assignment_not_allowed"}\n'
    assert captured.err == ""


@pytest.mark.parametrize("identity_key", ["id", "assignment_id", "plan_id"])
def test_skip_retry_accepts_prior_identity_formats_without_changing_bytes(tmp_path, identity_key):
    job, _, _, skip = _job(tmp_path)
    _jsonl(skip, [{identity_key: "assignment-0", "reason": "unsupported_language", "audit_note": "preserved"}])
    preserved = skip.read_bytes()
    drafts = tmp_path / "drafts.jsonl"
    _jsonl(drafts, [{"assignment_id": "assignment-0", "reason": "unsupported_language"}])
    assert writer.skip_assignments(job, drafts) == {"appended": 0, "already_present": 1}
    assert skip.read_bytes() == preserved


def test_pinned_plan_checksum_is_verified_before_append(tmp_path):
    job, plans, output, _ = _job(tmp_path)
    data = json.loads(job.read_text())
    plan = Path(data["plan"])
    data["plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    job.write_text(json.dumps(data))
    drafts = tmp_path / "drafts.jsonl"
    _jsonl(drafts, [_draft(plans[0])])
    assert writer.append_drafts(job, drafts)["appended"] == 1
    preserved = output.read_bytes()
    changed = dict(plans[0], status="pending")
    _jsonl(plan, [changed, *plans[1:]])
    _jsonl(drafts, [_draft(plans[1])])
    with pytest.raises(writer.WriterError, match="plan_checksum_mismatch"):
        writer.append_drafts(job, drafts)
    assert output.read_bytes() == preserved


def test_interrupted_between_rows_leaves_durable_prefix_and_retry_appends_only_missing(tmp_path, monkeypatch):
    job, plans, output, _ = _job(tmp_path)
    drafts = tmp_path / "drafts.jsonl"
    _jsonl(drafts, [_draft(plans[0]), _draft(plans[1])])
    original_write, original_fsync = writer.os.write, writer.os.fsync
    events = []
    writes = 0

    def write(descriptor, payload):
        nonlocal writes
        writes += 1
        events.append("write")
        if writes == 2:
            raise OSError("synthetic interruption")
        return original_write(descriptor, payload)

    def fsync(descriptor):
        events.append("fsync")
        return original_fsync(descriptor)

    monkeypatch.setattr(writer.os, "write", write)
    monkeypatch.setattr(writer.os, "fsync", fsync)
    with pytest.raises(writer.WriterError, match="input_or_append_failure"):
        writer.append_drafts(job, drafts)
    assert events == ["write", "fsync", "fsync", "write"]
    assert [row["id"] for row in _read(output)] == ["assignment-0"]
    preserved = output.read_bytes()
    assert writer.append_drafts(job, drafts) == {"appended": 1, "already_present": 1}
    assert output.read_bytes().startswith(preserved)
    assert [row["id"] for row in _read(output)] == ["assignment-0", "assignment-1"]


@pytest.mark.parametrize("empty_file", [False, True])
def test_status_missing_or_empty_output_counts_exact_launch_allowlist_without_writes(tmp_path, empty_file):
    job, plans, output, skip = _job(tmp_path)
    if empty_file:
        output.touch()
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    result = writer.job_status(job, next_count=3)
    assert result["written"] == result["skipped"] == 0
    assert result["unaccounted"] == 10
    assert result["next_ids"] == [plan["id"] for plan in plans[:3]]
    assert result["output_state"] == ("empty" if empty_file else "missing")
    assert result["skip_state"] == "missing"
    assert result["state"] == "unfinished"
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before
    assert not skip.exists()


def test_status_credits_only_allowlisted_ids_and_preserves_saved_prefix(tmp_path):
    job, plans, output, skip = _job(tmp_path)
    _jsonl(output, [{"id": plans[index]["id"]} for index in (0, 1, 8)])
    _jsonl(skip, [{"assignment_id": plans[9]["id"], "reason": "explicit"}])
    data = json.loads(job.read_text())
    data["remaining_ids"] = [plans[8]["id"], plans[9]["id"]]
    job.write_text(json.dumps(data))
    before = {path: path.read_bytes() for path in tmp_path.iterdir()}
    result = writer.job_status(job)
    assert result["allowlisted"] == 2
    assert result["written"] == result["skipped"] == 1
    assert result["unaccounted"] == 0
    assert result["next_ids"] == []
    assert result["complete"] is True
    assert {path: path.read_bytes() for path in tmp_path.iterdir()} == before


def test_status_in_flight_lock_withholds_counts_and_next_ids(tmp_path):
    job, _, output, _ = _job(tmp_path)
    with output.with_name(f".{output.name}.authoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.write_bytes(b'{"id":')
        result = writer.job_status(job)
    assert result["state"] == "in_flight"
    assert result["written"] is result["skipped"] is result["unaccounted"] is None
    assert result["complete"] is False
    assert result["next_ids"] == []
    result = writer.job_status(job)
    assert result["state"] == result["output_state"] == "incomplete_boundary"
    assert result["snapshot_available"] is False
    assert output.read_bytes() == b'{"id":'


def test_status_changing_unlocked_file_withholds_snapshot(tmp_path, monkeypatch):
    job, plans, output, _ = _job(tmp_path)
    _jsonl(output, [{"id": plans[0]["id"]}])
    original = writer._status_bytes

    def changed(path):
        data = original(path)
        if path == output:
            with path.open("a") as stream:
                stream.write(json.dumps({"id": plans[1]["id"]}) + "\n")
        return data

    monkeypatch.setattr(writer, "_status_bytes", changed)
    result = writer.job_status(job)
    assert result["state"] == "in_flight"
    assert result["snapshot_available"] is False
    assert result["next_ids"] == []


@pytest.mark.parametrize("failure", ["duplicate", "collision", "unknown"])
def test_status_fails_closed_for_duplicate_or_colliding_ledger_ids(tmp_path, failure):
    job, plans, output, skip = _job(tmp_path)
    row = {"id": plans[0]["id"]}
    _jsonl(output, [row, row] if failure == "duplicate" else [row])
    expected = "duplicate_ledger_identity"
    if failure == "collision":
        _jsonl(skip, [{"plan_id": row["id"], "reason": "conflict"}])
        expected = "source_skip_ledger_conflict"
    elif failure == "unknown":
        _jsonl(output, [{"id": "unknown-private-id", "context": "not printed"}])
        expected = "ledger_unknown_assignment"
    with pytest.raises(writer.WriterError, match=expected):
        writer.job_status(job)


def _cohort_job(tmp_path, name, *, complete=False):
    directory = tmp_path / name
    directory.mkdir()
    job, plans, output, skip = _job(directory, size=2)
    for plan in plans:
        plan["id"] = f"{name}-{plan['id']}"
        plan["problem_group_id"] = f"{name}-{plan['problem_group_id']}"
    data = json.loads(job.read_text())
    _jsonl(Path(data["plan"]), plans)
    data["remaining_ids"] = [plan["id"] for plan in plans]
    data["worker_status"] = "missing_runtime"
    job.write_text(json.dumps(data))
    if complete:
        _jsonl(output, [{"id": plan["id"]} for plan in plans])
    return job, plans, output, skip


def test_cohort_deduplicates_completed_jobs_and_retains_missing_runtime_unfinished_slot(tmp_path):
    completed, _, _, _ = _cohort_job(tmp_path, "complete", complete=True)
    missing, _, _, _ = _cohort_job(tmp_path, "missing")
    queued, _, _, _ = _cohort_job(tmp_path, "queued")
    alias = tmp_path / "completed-alias.json"
    alias.symlink_to(completed)
    manifest = tmp_path / "dispatcher.json"
    manifest.write_text(json.dumps({
        "launched": [str(completed), str(alias), str(missing), str(completed)],
        "queued": [str(queued)], "lane_limit": 2,
    }))
    result = writer.cohort_status(manifest)
    assert result["launched_jobs"] == 2
    assert result["completed_launched_jobs"] == 1
    assert result["unfinished_launched_jobs"] == 1
    assert result["available_slots"] == 1
    assert result["queued_jobs"] == 1
    repeated = writer.cohort_status(manifest)
    assert repeated == result


def test_cohort_in_flight_job_occupies_launched_slot(tmp_path):
    job, _, output, _ = _cohort_job(tmp_path, "active")
    manifest = tmp_path / "dispatcher.json"
    manifest.write_text(json.dumps({"launched": [str(job)], "queued": [], "lane_limit": 1}))
    with output.with_name(f".{output.name}.authoring.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = writer.cohort_status(manifest)
    assert result["unfinished_launched_jobs"] == 1
    assert result["available_slots"] == 0
    assert result["jobs"][0]["state"] == "in_flight"


@pytest.mark.parametrize("failure", ["both_groups", "different_job_same_outputs", "different_plan_same_ids"])
def test_cohort_rejects_overlapping_groups_and_ownership(tmp_path, failure):
    first, _, _, _ = _cohort_job(tmp_path, "first")
    second, _, _, _ = _cohort_job(tmp_path, "second")
    expected = "cohort_ownership_collision"
    if failure == "both_groups":
        second = first
        expected = "job_both_launched_and_queued"
    elif failure == "different_job_same_outputs":
        second.write_bytes(first.read_bytes())
    else:
        first_data, second_data = json.loads(first.read_text()), json.loads(second.read_text())
        first_plans = _read(Path(first_data["plan"]))
        second_plans = _read(Path(second_data["plan"]))
        for source, target in zip(first_plans, second_plans):
            target["id"] = source["id"]
        _jsonl(Path(second_data["plan"]), second_plans)
        second_data["remaining_ids"] = first_data["remaining_ids"]
        second.write_text(json.dumps(second_data))
    manifest = tmp_path / "dispatcher.json"
    manifest.write_text(json.dumps({"launched": [str(first)], "queued": [str(second)]}))
    with pytest.raises(writer.WriterError, match=expected):
        writer.cohort_status(manifest)


def test_status_and_cohort_cli_print_ids_and_counts_only(tmp_path, capsys):
    job, plans, _, _ = _cohort_job(tmp_path, "cli")
    assert writer._cli(["status", "--job", str(job), "--next-count", "1"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["next_ids"] == [plans[0]["id"]]
    manifest = tmp_path / "dispatcher.json"
    manifest.write_text(json.dumps({"launched": [str(job)], "queued": []}))
    assert writer._cli(["cohort", "--manifest", str(manifest)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["unfinished_launched_jobs"] == 1
    assert result["available_slots"] == 15
