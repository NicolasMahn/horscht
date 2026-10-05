import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "runs/overnight-2026-09-30/inspect-assignment-contract.py"
spec = importlib.util.spec_from_file_location("inspect_assignment_contract", SCRIPT)
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


@pytest.fixture
def metadata(tmp_path, monkeypatch):
    run = tmp_path / "runs/local-run"
    (run / "jobs").mkdir(parents=True)
    dataset = tmp_path / "data/generated_train/probability-v2"
    plan = dataset / "assignments/own-plan.jsonl"
    output = dataset / "questions/own-output.jsonl"
    skip = dataset / "questions/own-skip.jsonl"
    plan.parent.mkdir(parents=True)
    output.parent.mkdir(parents=True)
    rows = []
    for index in range(10):
        planner = {"target_shape_selection": {"status": "not_selected", "requirement": "multiple_substantive_support-v1"}}
        if index == 0:
            planner["target_shape_requirement"] = "multiple_substantive_support-v1"
            planner["target_shape_selection"]["status"] = "required"
        elif index == 1:
            planner["target_shape_selection"]["status"] = "unanswerable"
        rows.append({"id": f"assignment-{index}", "schema_version": "probability-v2-assignment-v1",
                     "language": "en", "distribution_source_kind": "exact_from_context",
                     "output": str(output), "skip_file": str(skip), "_planner": planner,
                     "answerability_status": "unanswerable" if index == 1 else "answerable",
                     "choice_count": 3, "seed": "private-seed", "question": "private-question"})
    plan.write_text("".join(json.dumps(row) + "\n" for row in rows))
    job = run / "jobs/own-job.json"
    job.write_text(json.dumps({"plan": str(plan), "output": str(output), "skip_file": str(skip),
                               "language": "en", "source_kind": "exact_from_context",
                               "plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
                               "remaining_ids": [row["id"] for row in rows]}))
    monkeypatch.setattr(inspector, "ROOT", tmp_path)
    monkeypatch.setattr(inspector, "RUN", run)
    return tmp_path, job, plan, output, skip


def repin(job, plan):
    data = json.loads(job.read_bytes())
    data["plan_sha256"] = hashlib.sha256(plan.read_bytes()).hexdigest()
    job.write_text(json.dumps(data))


def test_marker_is_direct_plan_field_only_and_cli_emits_no_content(metadata, capsys):
    _, job, _, _, _ = metadata
    assert inspector.main(["--job", str(job), "--ids", "assignment-0", "assignment-1", "assignment-2"]) == 0
    text = capsys.readouterr().out
    rows = json.loads(text)["assignments"]
    assert [row["required_multiple_support"] for row in rows] == [True, False, False]
    assert rows[1]["answerability_status"] == "unanswerable"
    assert rows[1]["selection_status"] == "unanswerable"
    assert set(rows[0]) == {"id", "answerability_status", "distribution_source_kind", "choice_count",
                            "required_multiple_support", "selection_status"}
    assert "private" not in text and "seed" not in text and "question" not in text


def test_default_uses_fresh_helper_next_eight_and_is_read_only(metadata):
    root, job, _, _, skip = metadata
    skip.write_text(json.dumps({"id": "assignment-0", "status": "skipped"}) + "\n")
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    rows = inspector.inspect_contract(job)["assignments"]
    assert [row["id"] for row in rows] == [f"assignment-{index}" for index in range(1, 9)]
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("ids", [["outside"], ["assignment-0"] * 2, [f"assignment-{i}" for i in range(9)], []])
def test_explicit_ids_reject_unknown_duplicate_or_excess(metadata, ids):
    _, job, _, _, _ = metadata
    with pytest.raises(inspector.InspectionError):
        inspector.inspect_contract(job, ids)


@pytest.mark.parametrize("field", ["job", "plan", "output", "skip_file"])
def test_resolved_path_scope_rejects_external_or_evaluation_paths(metadata, field):
    root, job, _, _, _ = metadata
    external = root / "evaluation-only/private.jsonl"
    if field == "job":
        with pytest.raises(inspector.InspectionError, match="job_path_outside_run"):
            inspector.inspect_contract(external)
    else:
        value = json.loads(job.read_bytes())
        value[field] = str(external)
        job.write_text(json.dumps(value))
        with pytest.raises(inspector.InspectionError, match="job_artifact_outside_training_scope"):
            inspector.inspect_contract(job)


def test_symlink_into_external_scope_is_rejected_before_read(metadata):
    root, job, plan, _, _ = metadata
    external = root / "external-plan.jsonl"
    plan.unlink()
    plan.symlink_to(external)
    with pytest.raises(inspector.InspectionError, match="job_artifact_outside_training_scope"):
        inspector.inspect_contract(job)


def test_pin_and_allowlist_validation_remain_fail_closed(metadata):
    _, job, plan, _, _ = metadata
    plan.write_bytes(plan.read_bytes() + b"\n")
    with pytest.raises(inspector.InspectionError, match="plan_checksum_mismatch"):
        inspector.inspect_contract(job)
    repin(job, plan)
    value = json.loads(job.read_bytes())
    value["remaining_ids"].append("not-in-plan")
    job.write_text(json.dumps(value))
    with pytest.raises(inspector.writer.WriterError, match="invalid_job_allowlist"):
        inspector.inspect_contract(job)


def test_unavailable_fresh_status_emits_only_error_code(metadata, capsys):
    _, job, _, output, _ = metadata
    output.write_bytes(b'{"question":"do-not-emit')
    assert inspector.main(["--job", str(job)]) == 2
    assert json.loads(capsys.readouterr().out) == {"error_code": "fresh_status_unavailable"}
