import copy
import fcntl
import hashlib
import json
from pathlib import Path

import pytest

from dataset_factory import cross_context_authoring as contract


PROJECT = Path(__file__).resolve().parents[1]


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def evidence(context, effect="changes_target"):
    n = len(context)
    return {"blocks": [{"block_id": region, "start": start, "end": end, "region": region,
                         "role": "substantive", "essential": True, "dependency_effect": effect,
                         "ablation_rationale": "Author-only removal rationale"}
                        for region, start, end in (("beginning", 0, n // 3), ("middle", n // 3, 2 * n // 3), ("end", 2 * n // 3, n))],
            "combined_dependency_rationale": "Author-only combination rationale", "no_summary_shortcut": True,
            "no_padding": True, "review_status": "author_self_review_only"}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    run = tmp_path / "runs/toy"
    (run / "jobs").mkdir(parents=True)
    dataset = tmp_path / "data/generated_train/probability-v2"
    (dataset / "assignments").mkdir(parents=True)
    (dataset / "questions").mkdir()
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    for name in contract.TOKENIZER_FILES:
        (tokenizer_dir / name).write_bytes(b"synthetic pinned tokenizer bytes")
    (tmp_path / "experiment.py").write_bytes((PROJECT / "experiment.py").read_bytes())
    (tmp_path / "docs").mkdir()
    document = tmp_path / "docs/contract.md"
    document.write_text("Synthetic pinned guidance and contract.")
    profile = {"name": contract.CONTRACT, "start_index": contract.START, "max_assignments": 4096,
               "selection_probability": 0.2, "seed": 9281, "prompt_token_bands": contract.BANDS,
               "context_length_policy": "selected_full_prompt_band_overrides_legacy_context_length_target",
               "max_prompt_tokens": 10240,
               "formatter_sha256": contract.sha((tmp_path / "experiment.py").read_bytes()),
               "helper_sha256": contract.sha(Path(contract.writer.__file__).read_bytes()),
               "contract": {"path": str(document), "sha256": contract.sha(document.read_bytes())},
               "tokenizer": {"path": str(tokenizer_dir), "files": {name: contract.sha((tokenizer_dir / name).read_bytes())
                                                                        for name in contract.TOKENIZER_FILES}}}
    profile_path = run / "profile.json"
    profile_path.write_text(json.dumps(profile))
    monkeypatch.setattr(contract, "ROOT", tmp_path)
    monkeypatch.setattr(contract, "available_memory", lambda: 100 * 1024**3)
    return tmp_path, run, dataset, profile, profile_path


def make_job(setup, *, selected=True, source="exact_derived", answerability="answerable", operation="policy_exceptions", count=3):
    _, run, dataset, profile, profile_path = setup
    identifier = next(f"assignment-{index}" for index in range(1000)
                      if contract.assignment_selection(profile, f"assignment-{index}", contract.START)["required_cross_context_dependency"] == selected)
    plan_path, output, skip = dataset / "assignments/own.jsonl", dataset / "questions/own.jsonl", dataset / "questions/skipped.jsonl"
    plan = {"schema_version": "probability-v2-assignment-v1", "id": identifier, "problem_group_id": "toy-group",
            "status": "assigned", "split": "train", "language": "es", "operation_id": operation,
            "domain_id": "archive", "difficulty_band": {"id": "medium"},
            "context_length_band": {"id": "short"}, "choice_count": count, "answerability_status": answerability,
            "unanswerable_reason": "missing_context" if answerability == "unanswerable" else None,
            "distribution_source_kind": source, "output": str(output), "skip_file": str(skip),
            "_generator": {"global_index": contract.START}, "_planner": {}}
    jsonl(plan_path, [plan])
    job_path = run / "jobs/own.json"
    scratch = run / "work/own"
    scratch.mkdir(parents=True)
    job_path.write_text(json.dumps({"plan": str(plan_path), "output": str(output), "skip_file": str(skip),
                                   "source_kind": source, "language": "es", "remaining_ids": [identifier],
                                   "plan_sha256": contract.sha(plan_path.read_bytes()),
                                   "cross_context_profile": {"path": str(profile_path), "sha256": contract.sha(profile_path.read_bytes())},
                                   "cross_context_checker_sha256": contract.sha(Path(contract.__file__).read_bytes()),
                                   "scratch_prefix": str(scratch),
                                   "guidance_path": profile["contract"]["path"], "guidance_sha256": profile["contract"]["sha256"],
                                   "dependency_evidence_file": str(run / "dependency-evidence/own.jsonl")}))
    context = "ABC" * 30
    target = [0] * count
    target[-1 if answerability == "unanswerable" else 0] = 1
    draft = {"assignment_id": identifier, "context": context, "question": "Synthetic combined-block question",
             "options": [f"Option {i}" for i in range(count - 1)] + ["IDK"], "explanation": "Private explanation",
             "abstention_index": count - 1, "abstention_text": "IDK"}
    if source == "hard_label":
        draft["answer_index"] = count - 1 if answerability == "unanswerable" else 0
    elif source == "teacher_estimate":
        draft["distribution_evidence"] = {"kind": source, "model_id": "toy", "model_version": "1", "estimation_method": "toy"}
        draft["target_distribution"] = target
    elif source == "empirical_frequency":
        draft["distribution_evidence"] = {"kind": source, "counts": target, "sample_size": 1}
    else:
        draft["distribution_evidence"] = {"kind": source, "probabilities": [{"numerator": value, "denominator": 1} for value in target]}
    drafts = scratch / "drafts.jsonl"
    evidence_path = scratch / "author-evidence.jsonl"
    jsonl(drafts, [draft])
    claim = evidence(context, "changes_missing_information_analysis" if answerability == "unanswerable" else "changes_target")
    jsonl(evidence_path, [{"assignment_id": identifier, "author_dependency_evidence": claim}] if selected else [])
    selection = contract.assignment_selection(profile, identifier, contract.START)
    length = selection["prompt_token_band"][0] if selected else 100
    return job_path, plan, output, draft, drafts, evidence_path, length


class Tokenizer:
    def __init__(self, length):
        self.length = length
        self.prompts = []

    def __call__(self, prompt, *, add_special_tokens, truncation):
        assert add_special_tokens is True and truncation is False
        self.prompts.append(prompt)
        return {"input_ids": range(self.length)}


def test_selection_is_bounded_seeded_and_preallocation_compatible(setup):
    _, _, _, profile, path = setup
    loaded = contract.load_profile(path)
    first = [contract.assignment_selection(loaded, f"id-{i}", contract.START + i) for i in range(4096)]
    assert first == [contract.assignment_selection(loaded, f"id-{i}", contract.START + i) for i in range(4096)]
    selected = [row for row in first if row["required_cross_context_dependency"]]
    assert 700 < len(selected) < 950
    assert {tuple(row["prompt_token_band"]) for row in selected} == {tuple(band) for band in contract.BANDS}
    other = dict(profile, seed=loaded["seed"] + 1)
    assert first != [contract.assignment_selection(other, f"id-{i}", contract.START + i) for i in range(4096)]
    for index in [contract.START - 1, contract.START + 4096, True]:
        with pytest.raises(contract.ContractError, match="assignment_outside_dependency_window"):
            contract.assignment_selection(loaded, "id", index)


@pytest.mark.parametrize("field,value", [("start_index", 137730), ("max_assignments", 8192), ("selection_probability", 1),
                                          ("max_prompt_tokens", 40960), ("context_length_policy", "legacy_also_satisfied")])
def test_profile_cannot_reanchor_expand_or_hide_length_override(setup, field, value):
    _, _, _, profile, path = setup
    profile[field] = value
    path.write_text(json.dumps(profile))
    with pytest.raises(contract.ContractError, match="invalid_dependency_profile_bounds"):
        contract.load_profile(path)


@pytest.mark.parametrize("source,operation,answerability,count", [
    ("exact_derived", "policy_exceptions", "answerable", 3),
    ("explicit_stochastic_process", "graph_kinship", "unanswerable", 4),
    ("empirical_frequency", "tradeoff_optimization", "answerable", 17),
    ("teacher_estimate", "procedure_recovery", "unanswerable", 2),
    ("hard_label", "evidence_review", "answerable", 265),
])
def test_append_preserves_source_task_targets_and_keeps_claims_outside_learner(setup, source, operation, answerability, count):
    job, plan, output, draft, drafts, evidence_path, length = make_job(setup, source=source, operation=operation,
                                                                     answerability=answerability, count=count)
    original_plan = Path(json.loads(job.read_bytes())["plan"]).read_bytes()
    tokenizer = Tokenizer(length)
    result = contract.append(job, drafts, evidence_path, tokenizer=tokenizer)
    assert result["appended"] == 1 and result["author_claims_appended"] == 1 and result["selected_checked"] == 1
    row = json.loads(output.read_text())
    expected = contract.writer._build_row(plan, draft)
    assert row == expected
    assert row["context_length_band"] == {"id": "short"}
    assert "Author-only" not in tokenizer.prompts[0] and "Private explanation" not in tokenizer.prompts[0]
    assert f"{count - 1}: IDK" in tokenizer.prompts[0]
    assert "author_dependency_evidence" not in row
    assert Path(json.loads(job.read_bytes())["plan"]).read_bytes() == original_plan
    audit = contract.audit_job(job)
    assert audit["selected"] == audit["accepted_claims"] == audit["selected_accepted_claims"] == 1
    assert audit["claimed_not_written"] == 0
    receipt = json.loads(Path(json.loads(job.read_bytes())["dependency_evidence_file"]).read_text())
    assert receipt["legacy_context_length_target_overridden"] is True and receipt["full_prompt_tokens"] == length


def test_unmarked_assignment_stays_normal_without_claimed_dependencies(setup):
    job, _, output, _, drafts, evidence_path, _ = make_job(setup, selected=False)
    result = contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(100))
    assert result["appended"] == 1 and result["selected_checked"] == 0
    assert "author_dependency_evidence" not in json.loads(output.read_text())
    assert contract.audit_job(job)["selected"] == 0


@pytest.mark.parametrize("failure", ["zero", "summary_only", "filler", "overlap", "empty_slice", "empty_rationale", "idk_effect_answerable"])
def test_invalid_or_summary_only_dependencies_reject_before_source_or_claim_append(setup, failure):
    job, _, output, draft, drafts, evidence_path, length = make_job(setup)
    data = json.loads(evidence_path.read_text())
    claim = data["author_dependency_evidence"]
    if failure == "zero":
        claim["blocks"] = []
    elif failure == "summary_only":
        claim["blocks"] = [claim["blocks"][-1]]
    elif failure == "filler":
        claim["blocks"][0]["essential"] = False
    elif failure == "overlap":
        claim["blocks"].append({**claim["blocks"][0], "block_id": "second-beginning"})
    elif failure == "empty_slice":
        draft["context"] = " " * 30 + draft["context"][30:]
        jsonl(drafts, [draft])
    elif failure == "empty_rationale":
        claim["blocks"][1]["ablation_rationale"] = " "
    else:
        claim["blocks"][1]["dependency_effect"] = "changes_missing_information_analysis"
    jsonl(evidence_path, [data])
    with pytest.raises(contract.ContractError):
        contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    assert not output.exists()
    assert not Path(json.loads(job.read_bytes())["dependency_evidence_file"]).exists()


def test_duplicate_evidence_ids_fail_before_writing(setup):
    job, _, output, _, drafts, evidence_path, length = make_job(setup)
    data = json.loads(evidence_path.read_text())
    jsonl(evidence_path, [data, data])
    with pytest.raises(contract.ContractError, match="duplicate_or_outside_evidence_id"):
        contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    assert not output.exists()


@pytest.mark.parametrize("length,code", [(999, "prompt_outside_assigned_band"), (10241, "prompt_exceeds_verified_capacity")])
def test_complete_prompt_length_rejected_without_truncation(setup, length, code):
    job, _, output, _, drafts, evidence_path, _ = make_job(setup)
    with pytest.raises(contract.ContractError, match=code):
        contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    assert not output.exists()


def test_claim_before_append_crash_is_not_acceptance_and_exact_retry_is_idempotent(setup, monkeypatch):
    job, _, output, draft, drafts, evidence_path, length = make_job(setup)
    original = contract.writer.append_drafts
    def crash(*args):
        raise contract.ContractError("synthetic_preappend_crash")
    monkeypatch.setattr(contract.writer, "append_drafts", crash)
    with pytest.raises(contract.ContractError, match="synthetic_preappend_crash"):
        contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    audit = contract.audit_job(job)
    assert audit["author_claims"] == audit["claimed_not_written"] == 1
    assert audit["accepted_claims"] == audit["written"] == 0
    monkeypatch.setattr(contract.writer, "append_drafts", original)
    result = contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    assert result["appended"] == 1 and result["author_claims_appended"] == 0
    before = output.read_bytes()
    result = contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    assert result["already_present"] == 1 and output.read_bytes() == before
    draft["explanation"] = "Different content"
    jsonl(drafts, [draft])
    with pytest.raises(contract.ContractError, match="dependency_evidence_retry_conflict"):
        contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    assert output.read_bytes() == before


def test_heavy_import_seam_is_serialized_before_tokenizer_load(setup, monkeypatch):
    root, _, _, _, _ = setup
    job, _, _, _, drafts, evidence_path, length = make_job(setup)
    def load(profile):
        with (root / "runs/.cross-context-tokenizer.lock").open("a") as independent:
            with pytest.raises(BlockingIOError):
                fcntl.flock(independent, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return Tokenizer(length)
    monkeypatch.setattr(contract, "load_tokenizer", load)
    assert contract.append(job, drafts, evidence_path)["appended"] == 1


def test_low_memory_and_broken_pins_fail_closed(setup, monkeypatch):
    job, _, output, _, drafts, evidence_path, length = make_job(setup)
    monkeypatch.setattr(contract, "available_memory", lambda: 12 * 1024**3)
    with pytest.raises(contract.ContractError, match="insufficient_tokenizer_memory_reserve"):
        contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    assert not output.exists()
    data = json.loads(job.read_bytes())
    data["cross_context_checker_sha256"] = "0" * 64
    job.write_text(json.dumps(data))
    with pytest.raises(contract.ContractError, match="checker_pin_mismatch"):
        contract.inspect_job(job)


def test_audit_reports_skips_and_raw_rows_without_claims_separately(setup):
    job, plan, output, draft, _, _, _ = make_job(setup)
    jsonl(output, [contract.writer._build_row(plan, draft)])
    audit = contract.audit_job(job)
    assert audit["written"] == audit["written_missing_claims"] == 1
    assert audit["accepted_claims"] == 0
    output.unlink()
    skip = Path(json.loads(job.read_bytes())["skip_file"])
    jsonl(skip, [{"id": plan["id"], "status": "skipped", "reason": "Synthetic incompatible task"}])
    audit = contract.audit_job(job)
    assert audit["skipped"] == audit["selected_skipped"] == 1
    assert audit["accepted_claims"] == audit["written_missing_claims"] == audit["unaccounted"] == 0


@pytest.mark.parametrize("field", ["guidance_sha256", "plan_sha256", "cross_context_profile", "dependency_evidence_file"])
def test_job_binding_errors_fail_before_any_append(setup, field):
    job, _, output, _, drafts, evidence_path, length = make_job(setup)
    data = json.loads(job.read_bytes())
    data[field] = str(setup[0] / "evaluation-only/private.jsonl") if field == "dependency_evidence_file" else (
        {**data[field], "sha256": "0" * 64} if field == "cross_context_profile" else "0" * 64)
    job.write_text(json.dumps(data))
    with pytest.raises((contract.ContractError, contract.writer.WriterError)):
        contract.append(job, drafts, evidence_path, tokenizer=Tokenizer(length))
    assert not output.exists()


def test_inspection_is_read_only_and_returns_only_selection_metadata(setup):
    root, _, _, _, _ = setup
    job, plan, _, _, _, _, _ = make_job(setup)
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    result = contract.inspect_job(job)
    assert result["assignments"][0]["id"] == plan["id"]
    assert set(result["assignments"][0]) == {"id", "required_cross_context_dependency", "prompt_token_band"}
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_another_authors_scratch_is_rejected(setup):
    _, run, _, _, _ = setup
    job, _, output, _, drafts, evidence_path, length = make_job(setup)
    other = run / "work/another-author"
    other.mkdir()
    foreign = other / drafts.name
    foreign.write_bytes(drafts.read_bytes())
    with pytest.raises(contract.ContractError, match="draft_or_evidence_outside_own_scratch"):
        contract.append(job, foreign, evidence_path, tokenizer=Tokenizer(length))
    assert not output.exists()
