import json
import os
import shutil

import pytest

import evaluate_new_training as evaluator
from evaluate_new_training import _next_test_attempt, _require_stage_available, write_json_once
from paired_training import sha256


def test_evaluation_records_are_create_once(tmp_path):
    path = tmp_path / "calibration.json"
    write_json_once(path, {"temperature": 1.25})

    with pytest.raises(FileExistsError, match="Immutable evaluation record"):
        write_json_once(path, {"temperature": 2.0})

    assert json.loads(path.read_text()) == {"temperature": 1.25}


def test_final_test_stage_requires_frozen_calibration_and_reuses_completed_result(tmp_path):
    calibration = tmp_path / "calibration.json"
    attempts = tmp_path / "test-attempts"
    final = tmp_path / "final-test.json"

    with pytest.raises(FileNotFoundError, match="Freeze calibration"):
        _require_stage_available("test", calibration, attempts, final)

    calibration.write_text("{}")
    assert _require_stage_available("test", calibration, attempts, final) == "run"
    attempts.mkdir()
    assert _require_stage_available("test", calibration, attempts, final) == "run"
    final.write_text("{}")
    assert _require_stage_available("test", calibration, attempts, final) == "complete"


def test_incomplete_test_attempt_can_retry_only_with_same_frozen_inputs(tmp_path):
    attempts = tmp_path / "test-attempts"
    binding = {
        "checkpoint_sha256": "a" * 64,
        "calibration_record_sha256": "b" * 64,
        "test_row_ids_sha256": "c" * 64,
        "questions": 12,
    }

    first = _next_test_attempt(attempts, binding)
    write_json_once(attempts / f"attempt-{first:04d}-started.json", {
        "schema_version": "soft-target-final-test-attempt-start-v1",
        "attempt": first,
        **binding,
    })
    second = _next_test_attempt(attempts, binding)

    assert first == 1
    assert second == 2
    assert (attempts / "attempt-0001-recovered.json").is_file()
    with pytest.raises(ValueError, match="different frozen inputs"):
        _next_test_attempt(attempts, {**binding, "checkpoint_sha256": "d" * 64})


def test_canonical_training_snapshot_path_is_accepted(tmp_path, monkeypatch):
    root = tmp_path / "project"
    snapshot = root / "data" / "training_snapshots" / "2026-10-08-v4"
    run = root / "runs" / "main"
    output = run / "calibration-evaluation"
    snapshot.mkdir(parents=True)
    output.mkdir(parents=True)
    monkeypatch.setattr(evaluator, "ROOT", root)
    monkeypatch.setattr(evaluator, "require_training_snapshot", lambda _snapshot: None)

    resolved_snapshot, resolved_run, resolved_output = evaluator._require_project_paths(snapshot, run, output)

    assert resolved_snapshot == snapshot.resolve()
    assert resolved_run == run.resolve()
    assert resolved_output == output.resolve()


def test_report_checkpoint_can_resolve_to_an_earlier_main_family_sibling(tmp_path):
    family = tmp_path / "new-experiment"
    main = family / "main"
    recovery = family / "main-recovery-001"
    checkpoint = main / "checkpoints" / "best"
    checkpoint.mkdir(parents=True)
    recovery.mkdir()

    resolved_checkpoint, source_run = evaluator._resolve_report_checkpoint(
        recovery, "../main/checkpoints/best"
    )

    assert resolved_checkpoint == checkpoint
    assert source_run == main


@pytest.mark.parametrize(
    "active_name,relative",
    [
        ("main", "../main/checkpoints/best"),
        ("main-recovery-001", "../main-recovery-002/checkpoints/best"),
        ("main-recovery-001", "../other/checkpoints/best"),
        ("main-recovery-001", "../../other/main/checkpoints/best"),
        ("main-recovery-001", "checkpoints/../outside"),
        ("main-recovery-001", "checkpoints\\best"),
    ],
)
def test_report_checkpoint_rejects_paths_outside_earlier_main_family(
    tmp_path, active_name, relative
):
    run = tmp_path / active_name
    run.mkdir()

    with pytest.raises(ValueError):
        evaluator._resolve_report_checkpoint(run, relative)


def test_selected_recovery_checkpoint_binds_active_and_source_runs(tmp_path, monkeypatch):
    root = tmp_path / "project"
    family = root / "runs" / "last-block-family"
    main = family / "main"
    recovery = family / "main-recovery-001"
    checkpoint = main / "checkpoints" / "best"
    checkpoint.mkdir(parents=True)
    recovery.mkdir()
    snapshot_hash = "a" * 64
    source_manifest = {
        "arm": "last-block",
        "backbone_training_scope": "last-block",
        "head_architecture": "last-token-linear-v1",
        "capacity": 265,
        "snapshot_manifest_sha256": snapshot_hash,
        "checkpoint_backbone_state": evaluator._FULL_CHECKPOINT_STORAGE,
        "repo": "org/model",
        "revision": "revision-1",
        "seed": 17,
        "output": str(main),
        "started_at": "2026-10-01T00:00:00Z",
    }
    active_manifest = {
        **source_manifest,
        "output": str(recovery),
        "started_at": "2026-10-02T00:00:00Z",
    }
    (main / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    (recovery / "manifest.json").write_text(json.dumps(active_manifest), encoding="utf-8")
    (checkpoint / "checkpoint.json").write_text(json.dumps({
        "schema_version": 1,
        "include_backbone": True,
        "backbone_state_kind": "full",
        "manifest": source_manifest,
        "files": {"backbone.pt": "backbone", "head.pt": "head"},
    }), encoding="utf-8")
    (recovery / "report.json").write_text(json.dumps({
        "status": "completed",
        "arm": "last-block",
        "selected_checkpoint_reload_verified": True,
        "best_checkpoint_relative": "../main/checkpoints/best",
    }), encoding="utf-8")
    (recovery / "best-checkpoint.json").write_text(json.dumps({
        "path": "../main/checkpoints/best",
    }), encoding="utf-8")
    monkeypatch.setattr(evaluator, "ROOT", root)

    selected = evaluator._selected_checkpoint(recovery, snapshot_hash)

    assert selected.active_manifest == active_manifest
    assert selected.checkpoint == checkpoint
    assert selected.best_checkpoint_relative == "../main/checkpoints/best"
    assert selected.source_run == main
    assert selected.source_run_manifest == source_manifest
    assert selected.checkpoint_sha256 == sha256(checkpoint / "checkpoint.json")


def test_frozen_manifest_match_allows_run_identity_but_rejects_semantic_change():
    base = {
        "seed": 10,
        "max_batch_tokens": 32768,
        "started_at": "2026-10-01T00:00:00Z",
        "output": "/runs/main",
    }
    recovery = {
        **base,
        "started_at": "2026-10-02T00:00:00Z",
        "output": "/runs/main-recovery-001",
    }

    evaluator._validate_frozen_manifest_match(base, recovery, candidate_label="recovery")

    with pytest.raises(ValueError, match="frozen inputs"):
        evaluator._validate_frozen_manifest_match(
            base, {**recovery, "max_batch_tokens": 40960}, candidate_label="recovery"
        )


def test_frozen_manifest_match_accepts_only_audited_lifecycle_hash_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    source = tmp_path / "training_loop.py"
    source.write_text("old source", encoding="utf-8")
    previous_hash = sha256(source)
    current_hash = "b" * 64
    base = {"loop_sha256": previous_hash, "seed": 10}
    candidate = {"loop_sha256": current_hash, "seed": 10}
    record = {
        "schema_version": "training-resource-lifecycle-audit-v1",
        "purpose": "resource-lifecycle-only",
        "source_changes": {
            "loop_sha256": {"previous": previous_hash, "current": current_hash},
        },
        "previous_source_files": {"loop_sha256": "training_loop.py"},
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(record), encoding="utf-8")
    candidate["resource_lifecycle_audit"] = {
        "path": str(audit_path),
        "sha256": sha256(audit_path),
        "record": record,
    }

    evaluator._validate_frozen_manifest_match(base, candidate, candidate_label="recovery")

    candidate["seed"] = 11
    with pytest.raises(ValueError, match="frozen inputs"):
        evaluator._validate_frozen_manifest_match(base, candidate, candidate_label="recovery")


def test_frozen_manifest_match_rejects_wrong_audit_previous_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    source = tmp_path / "training_loop.py"
    source.write_text("old source", encoding="utf-8")
    previous_hash = sha256(source)
    current_hash = "b" * 64
    base = {"loop_sha256": "c" * 64}
    candidate = {"loop_sha256": current_hash}
    record = {
        "schema_version": "training-resource-lifecycle-audit-v1",
        "purpose": "resource-lifecycle-only",
        "source_changes": {"loop_sha256": {"previous": previous_hash, "current": current_hash}},
        "previous_source_files": {"loop_sha256": "training_loop.py"},
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(record), encoding="utf-8")
    candidate["resource_lifecycle_audit"] = {
        "path": str(audit_path), "sha256": sha256(audit_path), "record": record,
    }

    with pytest.raises(ValueError, match="frozen inputs"):
        evaluator._validate_frozen_manifest_match(base, candidate, candidate_label="recovery")


def test_frozen_manifest_match_rejects_changed_audited_source_file(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    source = tmp_path / "training_loop.py"
    source.write_text("old source", encoding="utf-8")
    previous_hash = sha256(source)
    current_hash = "b" * 64
    base = {"loop_sha256": previous_hash}
    candidate = {"loop_sha256": current_hash}
    record = {
        "schema_version": "training-resource-lifecycle-audit-v1",
        "purpose": "resource-lifecycle-only",
        "source_changes": {"loop_sha256": {"previous": previous_hash, "current": current_hash}},
        "previous_source_files": {"loop_sha256": "training_loop.py"},
    }
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(record), encoding="utf-8")
    candidate["resource_lifecycle_audit"] = {
        "path": str(audit_path), "sha256": sha256(audit_path), "record": record,
    }
    source.write_text("replaced source", encoding="utf-8")

    with pytest.raises(ValueError, match="source checksum"):
        evaluator._validate_frozen_manifest_match(base, candidate, candidate_label="recovery")


def test_snapshot_bundle_collects_source_cohorts_without_exposing_rows(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    split_entries = [
        {"id": "train-row", "split": "train", "problem_group_id": "p1", "snapshot_group_id": "s1", "cohorts": ["documents"]},
        {"id": "dev-row", "split": "dev", "problem_group_id": "p2", "snapshot_group_id": "s2", "cohorts": ["documents"]},
        {"id": "cal-row", "split": "calibration", "problem_group_id": "p3", "snapshot_group_id": "s3", "cohorts": ["games"]},
        {"id": "test-old", "split": "test", "problem_group_id": "p4", "snapshot_group_id": "s4", "cohorts": ["original_replay"]},
        {"id": "test-new", "split": "test", "problem_group_id": "p5", "snapshot_group_id": "s5", "cohorts": ["sciq"]},
    ]
    row_entries = [{"id": entry["id"]} for entry in split_entries]
    rows_path = snapshot / "rows.jsonl"
    splits_path = snapshot / "splits.jsonl"
    rows_path.write_text("".join(json.dumps(row) + "\n" for row in row_entries), encoding="utf-8")
    splits_path.write_text("".join(json.dumps(entry) + "\n" for entry in split_entries), encoding="utf-8")
    (snapshot / "manifest.json").write_text(json.dumps({
        "artifacts": {
            "rows.jsonl": {"sha256": sha256(rows_path)},
            "splits.jsonl": {"sha256": sha256(splits_path)},
        }
    }), encoding="utf-8")
    split_by_id, group_ids, source_by_id, rows_by_split, _ = evaluator._snapshot_bundle(snapshot)

    assert split_by_id["test-old"] == "test"
    assert group_ids["test-old"] == ["problem:p4", "snapshot:s4"]
    assert source_by_id == {entry["id"]: entry["cohorts"][0] for entry in split_entries}
    assert [row["id"] for row in rows_by_split["test"]] == ["test-old", "test-new"]


def _compact_storage_fixture(tmp_path):
    runs_root = tmp_path / "runs"
    base = runs_root / "warm-start" / "checkpoints" / "base"
    selected = runs_root / "new-run" / "checkpoints" / "best"
    base.mkdir(parents=True)
    selected.mkdir(parents=True)
    (base / "backbone.pt").write_bytes(b"full-backbone-state")
    base_manifest = {
        "arm": "last-block",
        "backbone_training_scope": "last-block",
        "repo": "org/model",
        "revision": "revision-1",
        "capacity": 265,
        "head_architecture": "last-token-linear-v1",
        "snapshot_manifest_sha256": "c" * 64,
        "trainable_backbone_parameter_names": ["layer.weight"],
    }
    base_record = {
        "schema_version": 1,
        "include_backbone": True,
        "backbone_state_kind": "full",
        "manifest": base_manifest,
        "files": {"backbone.pt": sha256(base / "backbone.pt")},
    }
    (base / "checkpoint.json").write_text(json.dumps(base_record), encoding="utf-8")
    base_digest = sha256(base / "checkpoint.json")
    warm_start = {
        "checkpoint": str(base),
        "checkpoint_manifest_sha256": base_digest,
        "source_snapshot_manifest_sha256": "c" * 64,
    }
    run_manifest = {
        "checkpoint_backbone_state": evaluator._OVERLAY_CHECKPOINT_STORAGE,
        "warm_start": warm_start,
        "trainable_backbone_parameter_names": ["layer.weight"],
        "arm": "last-block",
        "backbone_training_scope": "last-block",
        "repo": "org/model",
        "revision": "revision-1",
        "capacity": 265,
        "head_architecture": "last-token-linear-v1",
    }
    checkpoint_record = {
        "schema_version": 1,
        "include_backbone": True,
        "backbone_state_kind": "trainable-overlay-v1",
        "backbone_parameter_names": ["layer.weight"],
        "base_checkpoint": {
            "path": os.path.relpath(base, selected),
            "manifest_sha256": base_digest,
        },
        "manifest": run_manifest,
        "files": {"backbone.pt": "overlay-file-sha", "head.pt": "head-file-sha"},
    }
    return runs_root, base, selected, run_manifest, checkpoint_record


def test_compact_checkpoint_requires_matching_immutable_base(tmp_path):
    runs_root, _base, selected, run_manifest, checkpoint_record = _compact_storage_fixture(tmp_path)

    evaluator._validate_checkpoint_storage(
        selected,
        checkpoint_record,
        run_manifest,
        runs_root=runs_root,
        current_run=selected.parents[1],
    )


def test_compact_checkpoint_rejects_changed_or_missing_base(tmp_path):
    runs_root, base, selected, run_manifest, checkpoint_record = _compact_storage_fixture(tmp_path)
    (base / "checkpoint.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="base manifest checksum mismatch"):
        evaluator._validate_checkpoint_storage(
            selected,
            checkpoint_record,
            run_manifest,
            runs_root=runs_root,
            current_run=selected.parents[1],
        )

    runs_root, base, selected, run_manifest, checkpoint_record = _compact_storage_fixture(tmp_path / "missing")
    shutil.rmtree(base)
    with pytest.raises(ValueError, match="base must be a real checkpoint"):
        evaluator._validate_checkpoint_storage(
            selected,
            checkpoint_record,
            run_manifest,
            runs_root=runs_root,
            current_run=selected.parents[1],
        )
