import json
import hashlib
import math
from pathlib import Path

import pytest

from evaluation import blind_runner
from experiment import format_input


def _row(item_id, question, answer_index, group="mmlu-pro-test"):
    row = {
        "item_id": item_id,
        "context": "",
        "question": question,
        "options": ["first synthetic option", "second synthetic option"],
        "answer_index": answer_index,
    }
    if group.startswith("belebele-"):
        row["language"] = group.removeprefix("belebele-")
    elif group.startswith("bbh-"):
        row["category"] = group.removeprefix("bbh-").replace("-", "_")
    return row


def _write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_model_allowlist_uses_the_training_formatter_and_hides_gold_metadata():
    raw = _row("synthetic-private-id", "Synthetic question?", 1)
    raw["category"] = "private synthetic category"
    item_id, model_row, answer, group = blind_runner._validate_item(raw, {"mmlu-pro-test": 1})

    assert item_id == "synthetic-private-id"
    assert answer == 1
    assert group == "mmlu-pro-test"
    assert set(model_row) == {"id", "context", "question", "options"}
    assert model_row["id"] == "external-item"
    prompt = format_input(model_row)
    assert prompt == (
        "Context: \nQuestion: Synthetic question?\nChoices:\n"
        "0: first synthetic option\n1: second synthetic option\nSelected choice:"
    )
    assert "synthetic-private-id" not in prompt
    assert "private synthetic category" not in prompt


def test_cursor_resumes_from_committed_aggregate_without_replaying_items(tmp_path):
    items = tmp_path / "synthetic-items.jsonl"
    _write_rows(items, [
        _row("synthetic-a", "Synthetic question A?", 0),
        _row("synthetic-b", "Synthetic question B?", 1),
    ])
    state_path = tmp_path / "cursor.json"
    expected_groups = {"mmlu-pro-test": 2}
    observed = []

    def score(model_row):
        assert set(model_row) == {"id", "context", "question", "options"}
        observed.append(model_row["question"])
        return [0.0, 0.0]

    first = blind_runner._process_candidate_items(
        items,
        state_path,
        protocol_sha="a" * 64,
        lock_sha="b" * 64,
        arm="full",
        expected_count=2,
        expected_groups=expected_groups,
        score_one=score,
        resume=False,
        stop_after_items=1,
    )
    assert first["status"] == "paused"
    assert first["next_item_index"] == 1
    assert observed == ["Synthetic question A?"]

    state_text = state_path.read_text(encoding="utf-8")
    assert "synthetic-a" not in state_text
    assert "Synthetic question" not in state_text
    assert "first synthetic option" not in state_text
    assert "predictions" not in state_text

    second = blind_runner._process_candidate_items(
        items,
        state_path,
        protocol_sha="a" * 64,
        lock_sha="b" * 64,
        arm="full",
        expected_count=2,
        expected_groups=expected_groups,
        score_one=score,
        resume=True,
    )
    assert second["status"] == "complete"
    assert second["aggregate"]["examples"] == 2
    assert second["aggregate"]["correct"] == 1
    assert observed == ["Synthetic question A?", "Synthetic question B?"]
    uniform_metrics = {
        "accuracy": 0.5,
        "negative_log_likelihood": math.log(2.0),
        "multiclass_brier_score": 0.5,
    }
    expected_metrics = {
        "examples": 2,
        "correct": 1,
        "accuracy": 0.5,
        "negative_log_likelihood": math.log(2.0),
        "multiclass_brier_score": 0.5,
        "uniform_baseline": uniform_metrics,
    }
    assert blind_runner._metrics_from_state(second) == {
        "overall": expected_metrics,
        "by_source_slice": {"mmlu-pro-test": expected_metrics},
    }


def test_unclean_inflight_cursor_fails_closed_without_replaying(tmp_path):
    items = tmp_path / "synthetic-items.jsonl"
    _write_rows(items, [_row("synthetic-a", "Synthetic question?", 0)])
    state_path = tmp_path / "cursor.json"
    state = blind_runner._empty_state("a" * 64, "b" * 64, "head", 1)
    state["inflight_item_index"] = 0
    blind_runner._write_cursor(state_path, state)
    calls = []

    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner._process_candidate_items(
            items,
            state_path,
            protocol_sha="a" * 64,
            lock_sha="b" * 64,
            arm="head",
            expected_count=1,
            expected_groups={"mmlu-pro-test": 1},
        score_one=lambda _row: calls.append("called") or [0.0, 0.0],
            resume=True,
        )

    assert raised.value.safe_code == "previous_run_inflight_item_will_not_be_replayed"
    assert calls == []


def test_malformed_synthetic_source_error_does_not_echo_line_payload(tmp_path):
    items = tmp_path / "synthetic-malformed.jsonl"
    items.write_text('{"private synthetic payload"\n', encoding="utf-8")

    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        list(blind_runner._iter_items(items))

    assert raised.value.safe_code == "canonical_items_json_invalid"
    assert "private synthetic payload" not in str(raised.value)


def test_overlong_preflight_returns_only_safe_aggregate_diagnostic(tmp_path):
    items = tmp_path / "synthetic-items.jsonl"
    _write_rows(items, [_row("synthetic-a", "Synthetic long-input check?", 0)])

    class SyntheticTokenizer:
        def __call__(self, prompt, *, add_special_tokens, truncation):
            assert "Synthetic long-input check?" in prompt
            assert add_special_tokens is True
            assert truncation is False
            return {"input_ids": list(range(12))}

    result = blind_runner.preflight_inputs(
        items,
        SyntheticTokenizer(),
        max_tokens=8,
        expected_count=1,
        expected_groups={"mmlu-pro-test": 1},
    )
    assert result == {
        "items_checked": 1,
        "groups_checked": 1,
        "max_input_tokens": 12,
        "over_limit_items": 1,
    }


def test_protocol_is_fixed_to_locked_bundle_and_candidate_order():
    protocol, digest = blind_runner._load_protocol()
    assert protocol["candidates"] == ["full", "head"]
    assert protocol["inference"]["batch_size"] == 1
    assert protocol["inference"]["truncation"] is False
    assert protocol["prompt_format"]["choice_order"].startswith("Canonical source order")
    assert sum(protocol["expected_group_counts"].values()) == protocol["canonical_item_count"]
    assert protocol["scoring"]["metrics"] == [
        "examples", "correct", "accuracy", "negative_log_likelihood", "multiclass_brier_score",
    ]
    assert "1/K" in protocol["scoring"]["uniform_baseline"]
    assert len(digest) == 64


def test_current_corpus_protocol_reuses_bundle_and_pins_one_selected_checkpoint():
    protocol_path = blind_runner.ROOT / "evaluation/protocols/current-corpus-blind-mcq-v1.json"
    protocol, digest = blind_runner._load_protocol(protocol_path)
    lock, items_path = blind_runner._validate_system_bundle(protocol)
    blind_runner._verify_exclusion_rebind(protocol, lock)
    assert protocol["candidates"] == ["full"]
    assert protocol["suite_reuse"]["status"] == "reused_suite_comparison_not_virgin"
    assert protocol["selection_binding"]["checkpoint_manifest_sha256"]
    assert protocol["exclusion_rebind"]["current_excluded_groups"] == 0
    assert len(digest) == 64
    assert lock["training_snapshot_manifest_sha256"] == protocol["training_snapshot_manifest_sha256"]
    assert items_path.is_file()


@pytest.mark.parametrize("arm", ["head", "last-block"])
def test_current_single_candidate_protocol_keeps_completed_report_binding_contract(tmp_path, monkeypatch, arm):
    protocol = json.loads((blind_runner.ROOT / "evaluation/protocols/current-corpus-blind-mcq-v1.json").read_text())
    monkeypatch.setattr(blind_runner, "ROOT", tmp_path)
    protocol["protocol_id"] = f"independent-mcq-current-{arm}-2026-10-v1"
    protocol["candidates"] = [arm]
    protocol["run_paths"] = {arm: f"runs/synthetic-{arm}/{arm}"}
    path = tmp_path / f"{arm}-protocol.json"
    path.write_text(json.dumps(protocol))
    assert blind_runner._load_protocol(path)[0]["candidates"] == [arm]
    protocol["selection_binding"].pop("report_sha256")
    path.write_text(json.dumps(protocol))
    with pytest.raises(blind_runner.BlindEvaluationError, match="protocol_selection_binding_invalid"):
        blind_runner._load_protocol(path)


def test_head_only_prepare_uses_its_frozen_tokenizer_without_full_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(blind_runner, "ROOT", tmp_path)
    (tmp_path / "manifest.json").write_text(json.dumps({"torch": "test", "transformers": "test"}))
    protocol = {"candidates": ["head"], "canonical_item_count": 17874,
                "system_test_lock_sha256": "1" * 64,
                "inference": {"input_limit_tokens": 32768}, "expected_group_counts": {}}
    candidate = {"arm": "head", "run_manifest_path": "manifest.json"}
    monkeypatch.setattr(blind_runner, "_load_protocol", lambda path: (protocol, "2" * 64))
    monkeypatch.setattr(blind_runner, "_validate_system_bundle", lambda value: ({}, tmp_path / "synthetic-unused-items.jsonl"))
    monkeypatch.setattr(blind_runner, "_load_protocol_candidates", lambda *args: {"head": candidate})
    monkeypatch.setattr(blind_runner, "_verify_selection_binding", lambda *args: {})
    monkeypatch.setattr(blind_runner, "_make_or_validate_lock", lambda *args: (tmp_path / "lock.json", {}, "3" * 64))
    loaded = []
    def tokenizer(selected):
        loaded.append(selected["arm"])
        return object(), object()
    monkeypatch.setattr(blind_runner, "_load_tokenizer", tokenizer)
    monkeypatch.setattr(blind_runner, "preflight_inputs", lambda *args, **kwargs: {
        "items_checked": 17874, "max_input_tokens": 1665, "over_limit_items": 0})
    result = blind_runner.run_evaluation(output_directory=tmp_path / "result", prepare_only=True)
    assert result["phase"] == "prepared" and loaded == ["head"]


def test_probability_scores_and_uniform_baseline_are_aggregate_only(tmp_path):
    items = tmp_path / "synthetic-items.jsonl"
    _write_rows(items, [
        _row("synthetic-a", "Synthetic question A?", 1),
        _row("synthetic-b", "Synthetic question B?", 1),
    ])
    state_path = tmp_path / "cursor.json"
    first, tied = [], []

    def score(model_row):
        question = model_row["question"]
        if question.endswith("A?"):
            first.append(True)
            return [0.0, math.log(3.0)]
        tied.append(True)
        return [0.0, 0.0]

    state = blind_runner._process_candidate_items(
        items,
        state_path,
        protocol_sha="a" * 64,
        lock_sha="b" * 64,
        arm="full",
        expected_count=2,
        expected_groups={"mmlu-pro-test": 2},
        score_one=score,
        resume=False,
    )
    metrics = blind_runner._metrics_from_state(state)
    overall = metrics["overall"]
    assert overall["examples"] == 2
    assert overall["correct"] == 1
    assert overall["accuracy"] == 0.5
    assert math.isclose(overall["negative_log_likelihood"], (math.log(4.0 / 3.0) + math.log(2.0)) / 2)
    assert math.isclose(overall["multiclass_brier_score"], 0.3125)
    assert overall["uniform_baseline"] == {
        "accuracy": 0.5,
        "negative_log_likelihood": math.log(2.0),
        "multiclass_brier_score": 0.5,
    }
    assert metrics["by_source_slice"]["mmlu-pro-test"] == overall
    assert first == [True] and tied == [True]

    cursor_text = state_path.read_text(encoding="utf-8")
    assert "Synthetic question" not in cursor_text
    assert "synthetic-a" not in cursor_text
    assert "predictions" not in cursor_text
    assert "logits" not in cursor_text
    assert "probabilities" not in cursor_text


def test_invalid_probability_input_fails_with_safe_code():
    for logits in ([0.0], [0.0, math.inf], [0.0, "bad"]):
        with pytest.raises(blind_runner.BlindEvaluationError) as raised:
            blind_runner._score_outcome(logits, answer=0, choice_count=2)
        assert raised.value.safe_code == "model_output_invalid"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _candidate_fixture(root, arm):
    run = root / "runs" / arm
    checkpoint = run / "checkpoints" / "best"
    (checkpoint / "tokenizer").mkdir(parents=True)
    (checkpoint / "backbone_config").mkdir()
    contents = {
        "tokenizer/tokenizer.json": b"synthetic tokenizer artifact",
        "backbone_config/config.json": b"synthetic backbone config",
        "head.pt": b"synthetic head weights",
    }
    if arm in {"full", "last-block"}:
        contents["backbone.pt"] = b"synthetic backbone weights"
    if arm == "last-block":
        contents["backbone_config/config.json"] = json.dumps({"num_hidden_layers": 3}).encode()
    for relative, payload in contents.items():
        path = checkpoint / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    files = {relative: _sha256(checkpoint / relative) for relative in contents}
    run_manifest = {
        "schema_version": 2,
        "arm": arm,
        "model": "qwen-instruct",
        "repo": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "snapshot_manifest_sha256": "1" * 64,
        "selection_sha256": "2" * 64,
        "system_test_manifest_sha256": "3" * 64,
        "seed": 17,
        "epochs": 2,
        "device": "cuda",
        "threads": 4,
        "formatter": "plain-context-question-choices-v1",
        "pooling": "last non-padding token",
        "capacity": 265,
        "max_tokens": 10240,
        "attention_backend": "flash",
        "parameters_dtype": "float32",
        "compute_autocast": "bfloat16; cache disabled",
        "initial_head_state_sha256": "4" * 64,
        "gpu_allocation_cap_gib": 44,
        "torch": "synthetic-torch",
        "transformers": "synthetic-transformers",
    }
    if arm == "last-block":
        run_manifest.update({
            "backbone_training_scope": "last-block",
            "backbone_trainable": True,
            "trainable_backbone_parameter_names": [
                "layers.2.self_attn.q_proj.weight", "layers.2.post_attention_layernorm.weight",
            ],
            "trainable_backbone_parameter_count": 32,
        })
    run.mkdir(parents=True, exist_ok=True)
    run_manifest_path = run / "manifest.json"
    run_manifest_path.write_text(json.dumps(run_manifest), encoding="utf-8")
    (run / "report.json").write_text(json.dumps({
        "arm": arm,
        "status": "completed",
        "selected_checkpoint_reload_verified": True,
        "selected_checkpoint_development": {"soft_cross_entropy": 0.5},
        "best_checkpoint_relative": "checkpoints/best",
    }), encoding="utf-8")
    checkpoint_manifest = {
        "schema_version": 1,
        "include_backbone": arm in {"full", "last-block"},
        "manifest": dict(run_manifest),
        "files": files,
    }
    (checkpoint / "checkpoint.json").write_text(json.dumps(checkpoint_manifest), encoding="utf-8")
    return run


def test_selected_candidate_and_reflection_receipt_bind_checkpoint_payload_hashes(monkeypatch, tmp_path):
    monkeypatch.setattr(blind_runner, "ROOT", tmp_path)
    protocol = {
        "model_key": "qwen-instruct",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "run_paths": {"full": "runs/full", "head": "runs/head"},
        "prompt_format": {"id": "plain-context-question-choices-v1"},
        "inference": {
            "pooling": "last non-padding token",
            "choice_head_capacity": 265,
            "input_limit_tokens": 10240,
            "attention_backend": "flash",
            "parameters_dtype": "float32",
            "backbone_autocast": "bfloat16; cache disabled",
        },
    }
    candidates = {
        arm: blind_runner._selected_candidate(arm, _candidate_fixture(tmp_path, arm), protocol)
        for arm in ("full", "head")
    }
    assert len(candidates["full"]["checkpoint_payload_sha256"]) == 64
    assert "backbone.pt" in candidates["full"]["checkpoint_files"]
    assert "backbone.pt" not in candidates["head"]["checkpoint_files"]
    assert candidates["full"]["tokenizer_files"] == candidates["head"]["tokenizer_files"]

    artifact = tmp_path / "runs" / "internal-dev-summary.json"
    artifact.write_text('{"synthetic": true}\n', encoding="utf-8")
    receipt = tmp_path / "runs" / "generation-reflection-receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": "generation-reflection-receipt-v1",
        "status": "committed",
        "committed_at": "2026-09-30T00:00:00Z",
        "external_test_used": False,
        "internal_dev_artifacts": [{"path": "runs/internal-dev-summary.json", "sha256": _sha256(artifact)}],
        "decision_artifact": {"path": "runs/internal-dev-summary.json", "sha256": _sha256(artifact)},
        "guidance_artifacts": [],
        "selected_checkpoint_refs": [
            {
                "arm": arm,
                "manifest_path": candidates[arm]["checkpoint_manifest_path"],
                "manifest_sha256": candidates[arm]["checkpoint_manifest_sha256"],
            }
            for arm in ("full", "head")
        ],
    }), encoding="utf-8")
    verified = blind_runner.verify_generation_receipt(receipt, candidates)
    assert verified["sha256"] == _sha256(receipt)
    assert set(verified["selected_checkpoint_refs"]) == {"full", "head"}

    tampered_candidates = {arm: dict(candidate) for arm, candidate in candidates.items()}
    tampered_candidates["full"]["checkpoint_manifest_sha256"] = "0" * 64
    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner.verify_generation_receipt(receipt, tampered_candidates)
    assert raised.value.safe_code == "candidate_checkpoint_manifest_hash_mismatch"

    invalid_receipt = json.loads(receipt.read_text(encoding="utf-8"))
    invalid_receipt["committed_at"] = "Synthetic question contents"
    receipt.write_text(json.dumps(invalid_receipt), encoding="utf-8")
    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner.verify_generation_receipt(receipt, candidates)
    assert raised.value.safe_code == "generation_reflection_timestamp_invalid"
    assert "Synthetic question contents" not in str(raised.value)


@pytest.mark.parametrize("arm", ["full", "head", "last-block"])
def test_single_candidate_selection_binding_allows_only_pinned_recovery_target(monkeypatch, tmp_path, arm):
    monkeypatch.setattr(blind_runner, "ROOT", tmp_path)
    source_run = _candidate_fixture(tmp_path, arm)
    recovery_run = tmp_path / "runs" / "recovery" / arm
    recovery_run.mkdir(parents=True)
    source_manifest = json.loads((source_run / "manifest.json").read_text(encoding="utf-8"))
    (recovery_run / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    relative = f"../../{arm}/checkpoints/best"
    report_path = recovery_run / "report.json"
    report_path.write_text(json.dumps({
        "arm": arm,
        "status": "completed",
        "selected_checkpoint_reload_verified": True,
        "selected_checkpoint_development": {"soft_cross_entropy": 0.5},
        "best_checkpoint_relative": relative,
    }), encoding="utf-8")
    checkpoint_manifest = source_run / "checkpoints" / "best" / "checkpoint.json"
    protocol = {
        "model_key": "qwen-instruct",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "training_snapshot_manifest_sha256": "1" * 64,
        "system_test_lock_sha256": "3" * 64,
        "run_paths": {arm: f"runs/recovery/{arm}"},
        "prompt_format": {"id": "plain-context-question-choices-v1"},
        "inference": {
            "pooling": "last non-padding token",
            "choice_head_capacity": 265,
            "input_limit_tokens": 10240,
            "attention_backend": "flash",
            "parameters_dtype": "float32",
            "backbone_autocast": "bfloat16; cache disabled",
        },
        "selection_binding": {
            "kind": "completed-development-report",
            "report_path": f"runs/recovery/{arm}/report.json",
            "report_sha256": _sha256(report_path),
            "best_checkpoint_relative": relative,
            "selected_checkpoint_path": f"runs/{arm}/checkpoints/best",
            "checkpoint_manifest_sha256": _sha256(checkpoint_manifest),
            "run_manifest_sha256": _sha256(recovery_run / "manifest.json"),
            "checkpoint_source_run_manifest_path": f"runs/{arm}/manifest.json",
            "checkpoint_source_run_manifest_sha256": _sha256(source_run / "manifest.json"),
        },
    }
    candidate = blind_runner._selected_candidate(arm, recovery_run, protocol)
    evidence = blind_runner._verify_selection_binding(protocol, {arm: candidate}, tmp_path / "unused-receipt.json")
    assert candidate["checkpoint_path"] == f"runs/{arm}/checkpoints/best"
    assert evidence["kind"] == "completed-development-report"
    tampered = dict(protocol)
    tampered["selection_binding"] = dict(protocol["selection_binding"], report_sha256="0" * 64)
    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner._verify_selection_binding(tampered, {arm: candidate}, tmp_path / "unused-receipt.json")
    assert raised.value.safe_code == "candidate_development_selection_binding_mismatch"


@pytest.fixture
def last_block_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(blind_runner, "ROOT", tmp_path)
    run = _candidate_fixture(tmp_path, "last-block")
    protocol = {
        "model_key": "qwen-instruct",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "run_paths": {"last-block": "runs/last-block"},
        "prompt_format": {"id": "plain-context-question-choices-v1"},
        "inference": {
            "pooling": "last non-padding token",
            "choice_head_capacity": 265,
            "input_limit_tokens": 10240,
            "attention_backend": "flash",
            "parameters_dtype": "float32",
            "backbone_autocast": "bfloat16; cache disabled",
        },
    }
    return run, protocol


@pytest.mark.parametrize("missing", ["state-policy", "backbone-payload", "both"])
def test_last_block_selected_checkpoint_requires_the_full_backbone(last_block_candidate, missing):
    run, protocol = last_block_candidate
    checkpoint = run / "checkpoints/best"
    path = checkpoint / "checkpoint.json"
    metadata = json.loads(path.read_text())
    if missing in {"state-policy", "both"}:
        metadata["include_backbone"] = False
    if missing in {"backbone-payload", "both"}:
        del metadata["files"]["backbone.pt"]
        (checkpoint / "backbone.pt").unlink()
    path.write_text(json.dumps(metadata))
    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner._selected_candidate("last-block", run, protocol)
    assert raised.value.safe_code == "candidate_checkpoint_protocol_mismatch"


@pytest.mark.parametrize("field,value", [
    ("status", "interrupted"), ("selected_checkpoint_reload_verified", False),
])
def test_last_block_candidate_requires_a_completed_verified_report(last_block_candidate, field, value):
    run, protocol = last_block_candidate
    path = run / "report.json"
    report = json.loads(path.read_text())
    report[field] = value
    path.write_text(json.dumps(report))
    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner._selected_candidate("last-block", run, protocol)
    assert raised.value.safe_code == "candidate_run_protocol_mismatch"


@pytest.mark.parametrize("field,value", [
    ("backbone_training_scope", None),
    ("backbone_training_scope", "full"),
    ("backbone_trainable", False),
    ("trainable_backbone_parameter_names", None),
    ("trainable_backbone_parameter_names", []),
    ("trainable_backbone_parameter_names", [17]),
    ("trainable_backbone_parameter_names", ["layers.2.self_attn.q_proj.weight"] * 2),
    ("trainable_backbone_parameter_names", ["layers.1.self_attn.q_proj.weight"]),
    ("trainable_backbone_parameter_names", ["layers.2.self_attn.q_proj.weight", "embed_tokens.weight"]),
    ("trainable_backbone_parameter_names", ["norm.weight"]),
    ("trainable_backbone_parameter_count", 0),
    ("trainable_backbone_parameter_count", True),
])
def test_last_block_candidate_rejects_unsafe_training_declarations(last_block_candidate, field, value):
    run, protocol = last_block_candidate
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if value is None:
        del manifest[field]
    else:
        manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))
    checkpoint_path = run / "checkpoints/best/checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    checkpoint["manifest"] = manifest
    checkpoint_path.write_text(json.dumps(checkpoint))
    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner._selected_candidate("last-block", run, protocol)
    assert raised.value.safe_code == "candidate_last_block_scope_invalid"


@pytest.mark.parametrize("field,value", [
    ("backbone_training_scope", "full"),
    ("trainable_backbone_parameter_names", ["layers.2.mlp.up_proj.weight"]),
    ("trainable_backbone_parameter_count", 64),
])
def test_last_block_checkpoint_training_declaration_matches_its_source_run(last_block_candidate, field, value):
    run, protocol = last_block_candidate
    path = run / "checkpoints/best/checkpoint.json"
    checkpoint = json.loads(path.read_text())
    checkpoint["manifest"][field] = value
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner._selected_candidate("last-block", run, protocol)
    assert raised.value.safe_code == "candidate_checkpoint_run_manifest_mismatch"


def test_last_block_recovery_manifest_scope_matches_the_checkpoint_source(last_block_candidate):
    source_run, protocol = last_block_candidate
    root = source_run.parent.parent
    recovery_run = root / "runs/recovery/last-block"
    recovery_run.mkdir(parents=True)
    manifest = json.loads((source_run / "manifest.json").read_text())
    manifest["backbone_training_scope"] = "full"
    manifest_path = recovery_run / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    report = json.loads((source_run / "report.json").read_text())
    relative = "../../last-block/checkpoints/best"
    report["best_checkpoint_relative"] = relative
    report_path = recovery_run / "report.json"
    report_path.write_text(json.dumps(report))
    protocol.update({
        "training_snapshot_manifest_sha256": "1" * 64,
        "system_test_lock_sha256": "3" * 64,
        "run_paths": {"last-block": "runs/recovery/last-block"},
        "selection_binding": {
            "kind": "completed-development-report",
            "report_path": "runs/recovery/last-block/report.json",
            "report_sha256": _sha256(report_path),
            "best_checkpoint_relative": relative,
            "selected_checkpoint_path": "runs/last-block/checkpoints/best",
            "checkpoint_manifest_sha256": _sha256(source_run / "checkpoints/best/checkpoint.json"),
            "run_manifest_sha256": _sha256(manifest_path),
            "checkpoint_source_run_manifest_path": "runs/last-block/manifest.json",
            "checkpoint_source_run_manifest_sha256": _sha256(source_run / "manifest.json"),
        },
    })
    with pytest.raises(blind_runner.BlindEvaluationError) as raised:
        blind_runner._selected_candidate("last-block", recovery_run, protocol)
    assert raised.value.safe_code == "candidate_checkpoint_run_manifest_mismatch"
