import json
import os
from pathlib import Path

import pytest
from types import SimpleNamespace

from evaluation import blind_runner
from evaluation.snapshot_bound_protocol import (
    DEFAULT_PROTOCOL_TEMPLATE,
    DEFERRED_DEVELOPMENT_EXECUTION_V1,
    bind_fast_development_protocol,
    bind_completed_development_protocol,
    create_snapshot_bound_lock,
    sha256_file,
    validate_fast_development_selection,
    validate_compact_checkpoint_binding,
)
from scalable_training import validate_system_test_lock
from paired_training import TrainingMemoryPressure


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _snapshot_root(tmp_path):
    root = tmp_path
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n")
    external = root / "data/external_evaluation"
    snapshot = root / "data/training_snapshots/snapshot-v3"
    external.mkdir(parents=True)
    rows = [
        {"id": "training-row-1", "family": "family-1", "operation_id": "operation-1"},
        {"id": "training-row-2"},
    ]
    splits = [
        {"id": "training-row-1", "problem_group_id": "problem-1", "snapshot_group_id": "snapshot-1"},
        {"id": "training-row-2", "problem_group_id": "problem-2", "snapshot_group_id": "snapshot-2"},
    ]
    _write_jsonl(snapshot / "rows.jsonl", rows)
    _write_jsonl(snapshot / "splits.jsonl", splits)
    artifact_hashes = {
        name: {"records": 2, "sha256": sha256_file(snapshot / name)}
        for name in ("rows.jsonl", "splits.jsonl")
    }
    _write_json(snapshot / "manifest.json", {
        "schema_version": "training-snapshot-v2", "version": "snapshot-v3",
        "artifacts": artifact_hashes,
    })

    # The helper may hash this opaque file but must never parse or display it.
    canonical = external / "canonical/bundle/items.jsonl"
    canonical.parent.mkdir(parents=True)
    canonical.write_bytes(b"opaque canonical bytes\n")
    canonical_hash = sha256_file(canonical)
    bundle = {
        "schema_version": "external-test-bundle-v1", "canonical_item_count": 17874,
        "canonical_items_path": "canonical/bundle/items.jsonl",
        "canonical_items_sha256": canonical_hash,
    }
    _write_json(external / "bundle-manifest.json", bundle)
    extra = external / "staging/reference.bin"
    extra.parent.mkdir(parents=True)
    extra.write_bytes(b"immutable suite artifact")
    suite_files = {
        "bundle-manifest.json": sha256_file(external / "bundle-manifest.json"),
        "canonical/bundle/items.jsonl": canonical_hash,
        "staging/reference.bin": sha256_file(extra),
    }
    suite_lock = external / "system-test-lock-current-corpus-2026-10-01.json"
    _write_json(suite_lock, {
        "schema_version": "system-test-lock-v1", "status": "locked",
        "independent_from_training_generator": True, "files": suite_files,
    })

    audit_dir = external / "overlap-audit-v3"
    audit_dir.mkdir()
    quarantine_path = audit_dir / "quarantine-training-ids.jsonl"
    overlay_path = audit_dir / "exact-match-exclusion-overlay.jsonl"
    _write_jsonl(quarantine_path, [{"training_id": "training-row-1", "overlap_types": ["near_lexical_overlap"]}])
    _write_jsonl(overlay_path, [])
    report_path = audit_dir / "overlap-report.json"
    _write_json(report_path, {
        "schema_version": "blind-overlap-audit-v1",
        "benchmark_items_sha256": canonical_hash,
        "training_snapshot_version": "snapshot-v3",
        "training_snapshot_manifest_sha256": sha256_file(snapshot / "manifest.json"),
        "training_rows_sha256": sha256_file(snapshot / "rows.jsonl"),
        "training_rows_checked": 2,
        "benchmark_rows_checked": 17874,
        "exact_prompt_overlap_test_rows": 0,
        "exact_prompt_overlap_training_ids": 0,
        "near_lexical_overlap_test_rows": 1,
        "near_lexical_overlap_training_ids": 1,
        "near_candidate_pairs_checked": 1,
        "near_candidate_pairs_truncated": 0,
        "semantic_overlap_checked": False,
        "pretraining_contamination_ruled_out": False,
        "quarantine_training_ids_sha256": sha256_file(quarantine_path),
        "exact_match_exclusion_overlay_sha256": sha256_file(overlay_path),
    })
    return root, snapshot, audit_dir, suite_lock, canonical_hash


def test_snapshot_bound_lock_maps_only_training_ids_to_linked_groups(tmp_path):
    root, snapshot, audit_dir, suite_lock, _ = _snapshot_root(tmp_path)
    lock_path = root / "data/external_evaluation/system-test-lock-v3.json"

    lock = create_snapshot_bound_lock(snapshot, audit_dir, lock_path, suite_lock_path=suite_lock)

    assert lock["snapshot_binding"]["manifest_sha256"] == sha256_file(snapshot / "manifest.json")
    assert lock["training_exclusions"] == {
        "row_ids": ["training-row-1"],
        "problem_group_ids": ["problem-1"],
        "snapshot_group_ids": ["snapshot-1"],
        "family_ids": ["family-1"],
        "operation_ids": ["operation-1"],
    }
    assert lock["overlap_audit"]["near_candidate_pairs_truncated"] == 0
    assert validate_system_test_lock(lock_path, snapshot) == lock
    with pytest.raises(FileExistsError, match="immutable"):
        create_snapshot_bound_lock(snapshot, audit_dir, lock_path, suite_lock_path=suite_lock)


def test_completed_development_protocol_is_bound_to_snapshot_audit_and_checkpoint(tmp_path, monkeypatch):
    root, snapshot, audit_dir, suite_lock, canonical_hash = _snapshot_root(tmp_path)
    external = root / "data/external_evaluation"
    lock_path = external / "system-test-lock-v3.json"
    lock = create_snapshot_bound_lock(snapshot, audit_dir, lock_path, suite_lock_path=suite_lock)

    template_path = root / DEFAULT_PROTOCOL_TEMPLATE
    template_path.parent.mkdir(parents=True)
    template = json.loads((blind_runner.ROOT / DEFAULT_PROTOCOL_TEMPLATE).read_text())
    template["bundle_manifest_sha256"] = sha256_file(external / "bundle-manifest.json")
    template["canonical_items_sha256"] = canonical_hash
    template["canonical_items_path"] = "data/external_evaluation/canonical/bundle/items.jsonl"
    template["bundle_manifest_path"] = "data/external_evaluation/bundle-manifest.json"
    _write_json(template_path, template)

    run_directory = root / "runs/new-mixture"
    main = run_directory / "main"
    checkpoint = main / "checkpoints/selected"
    checkpoint.mkdir(parents=True)
    (checkpoint / "backbone.pt").write_bytes(b"synthetic backbone")
    (checkpoint / "head.pt").write_bytes(b"synthetic head")
    checkpoint_manifest = {
        "schema_version": 1, "include_backbone": True,
        "files": {name: sha256_file(checkpoint / name) for name in ("backbone.pt", "head.pt")},
        "manifest": {
            "arm": "last-block", "repo": template["model_id"], "revision": template["model_revision"],
            "formatter": template["prompt_format"]["id"],
            "max_tokens": template["inference"]["input_limit_tokens"], "capacity": 265,
            "snapshot_manifest_sha256": sha256_file(snapshot / "manifest.json"),
            "system_test_manifest_sha256": sha256_file(lock_path),
        },
    }
    _write_json(checkpoint / "checkpoint.json", checkpoint_manifest)
    run_manifest = {
        "schema_version": 2, "arm": "last-block", "model": template["model_key"],
        "repo": template["model_id"], "revision": template["model_revision"],
        "snapshot_manifest_sha256": sha256_file(snapshot / "manifest.json"),
        "system_test_manifest_sha256": sha256_file(lock_path), "formatter": template["prompt_format"]["id"],
        "pooling": template["inference"]["pooling"], "capacity": 265,
        "max_tokens": template["inference"]["input_limit_tokens"], "device": "cuda",
        "attention_backend": template["inference"]["attention_backend"],
        "parameters_dtype": template["inference"]["parameters_dtype"],
        "compute_autocast": template["inference"]["backbone_autocast"],
        "head_architecture": "last-token-linear-v1",
    }
    _write_json(main / "manifest.json", run_manifest)
    _write_json(main / "report.json", {
        "status": "completed", "arm": "last-block", "selected_checkpoint_reload_verified": True,
        "best_checkpoint_relative": "checkpoints/selected",
    })

    protocol_path = run_directory / "blind-protocol.json"
    protocol = bind_completed_development_protocol(snapshot, lock_path, main, protocol_path, template_path)

    assert protocol["protocol_id"] == "independent-mcq-current-last-block-2026-10-v3"
    assert protocol["training_snapshot_manifest_sha256"] == sha256_file(snapshot / "manifest.json")
    assert protocol["overlap_audit_report_sha256"] == lock["overlap_audit"]["report_sha256"]
    assert protocol["selection_binding"]["checkpoint_manifest_sha256"] == sha256_file(checkpoint / "checkpoint.json")
    assert "exclusion_rebind" not in protocol
    assert bind_completed_development_protocol(snapshot, lock_path, main, protocol_path, template_path) == protocol

    monkeypatch.setattr(blind_runner, "ROOT", root)
    monkeypatch.setattr(blind_runner, "EXTERNAL_DATA_ROOT", external)
    parsed, _ = blind_runner._load_protocol(protocol_path)
    validated_lock, items_path = blind_runner._validate_system_bundle(parsed)
    assert validated_lock["snapshot_binding"]["manifest_sha256"] == protocol["training_snapshot_manifest_sha256"]
    assert items_path == external / "canonical/bundle/items.jsonl"

    deferred_protocol = json.loads(json.dumps(protocol))
    deferred_protocol["protocol_id"] = "independent-mcq-current-last-block-2026-10-v4"
    deferred_protocol["development_execution"] = DEFERRED_DEVELOPMENT_EXECUTION_V1
    deferred_protocol["selection_binding"].update({
        "base_checkpoint_binding": {"path": "runs/base/checkpoints/full"},
        "deferred_selection_sha256": "a" * 64,
        "deferred_candidate_index_sha256": "b" * 64,
        "selected_checkpoint_update": 360,
    })
    deferred_protocol_path = run_directory / "deferred-protocol.json"
    _write_json(deferred_protocol_path, deferred_protocol)
    parsed_deferred, _ = blind_runner._load_protocol(deferred_protocol_path)
    assert parsed_deferred["development_execution"] == DEFERRED_DEVELOPMENT_EXECUTION_V1

    invalid_deferred_protocol = json.loads(json.dumps(deferred_protocol))
    invalid_deferred_protocol["development_execution"]["candidate_updates"][-1] = 1437
    invalid_deferred_path = run_directory / "invalid-deferred-protocol.json"
    _write_json(invalid_deferred_path, invalid_deferred_protocol)
    with pytest.raises(blind_runner.BlindEvaluationError, match="protocol_deferred_development_contract_invalid"):
        blind_runner._load_protocol(invalid_deferred_path)

    recovery = run_directory / "main-recovery-001"
    recovery.mkdir()
    _write_json(recovery / "manifest.json", json.loads((main / "manifest.json").read_text()))
    recovery_report = json.loads((main / "report.json").read_text())
    recovery_report["best_checkpoint_relative"] = "../main/checkpoints/selected"
    _write_json(recovery / "report.json", recovery_report)
    recovered_protocol_path = recovery / "blind-protocol.json"
    recovered_protocol = bind_completed_development_protocol(
        snapshot, lock_path, recovery, recovered_protocol_path, template_path,
    )
    assert recovered_protocol["selection_binding"]["active_run_manifest_path"] == (
        "runs/new-mixture/main-recovery-001/manifest.json"
    )
    assert recovered_protocol["selection_binding"]["checkpoint_source_run_manifest_path"] == (
        "runs/new-mixture/main/manifest.json"
    )
    assert recovered_protocol["selection_binding"]["best_checkpoint_relative"] == "../main/checkpoints/selected"


def test_fast_development_binding_pins_winner_to_exact_small_dev_selection(tmp_path, monkeypatch):
    root, snapshot, audit_dir, suite_lock, canonical_hash = _snapshot_root(tmp_path)
    snapshot_manifest_path = snapshot / "manifest.json"
    snapshot_manifest = json.loads(snapshot_manifest_path.read_text())
    snapshot_manifest["version"] = "snapshot-v4"
    _write_json(snapshot_manifest_path, snapshot_manifest)
    audit_report = audit_dir / "overlap-report.json"
    report = json.loads(audit_report.read_text())
    report["training_snapshot_version"] = "snapshot-v4"
    report["training_snapshot_manifest_sha256"] = sha256_file(snapshot_manifest_path)
    _write_json(audit_report, report)

    external = root / "data/external_evaluation"
    lock_path = external / "system-test-lock-v4.json"
    create_snapshot_bound_lock(snapshot, audit_dir, lock_path, suite_lock_path=suite_lock)
    lock_sha = sha256_file(lock_path)
    template_path = root / DEFAULT_PROTOCOL_TEMPLATE
    template_path.parent.mkdir(parents=True)
    template = json.loads((blind_runner.ROOT / DEFAULT_PROTOCOL_TEMPLATE).read_text())
    template["bundle_manifest_sha256"] = sha256_file(external / "bundle-manifest.json")
    template["canonical_items_sha256"] = canonical_hash
    template["canonical_items_path"] = "data/external_evaluation/canonical/bundle/items.jsonl"
    template["bundle_manifest_path"] = "data/external_evaluation/bundle-manifest.json"
    _write_json(template_path, template)

    family = root / "runs/new-plan"
    main = family / "main"
    warm_start = family / "warmstart/checkpoints/base"
    base_names = ["layers.35.self_attn.q_proj.weight"]
    base_backbone = warm_start / "backbone.pt"
    base_head = warm_start / "head.pt"
    base_backbone.parent.mkdir(parents=True)
    base_backbone.write_bytes(b"pinned full base backbone")
    base_head.write_bytes(b"pinned base head")
    base_run_manifest = {
        "arm": "last-block", "repo": template["model_id"], "revision": template["model_revision"],
        "head_architecture": "last-token-linear-v1", "trainable_backbone_parameter_names": base_names,
    }
    base_data = {
        "schema_version": 1, "include_backbone": True, "backbone_state_kind": "full",
        "base_checkpoint": None, "backbone_parameter_names": None,
        "files": {"backbone.pt": sha256_file(base_backbone), "head.pt": sha256_file(base_head)},
        "manifest": base_run_manifest,
    }
    _write_json(warm_start / "checkpoint.json", base_data)
    base_checkpoint_sha = sha256_file(warm_start / "checkpoint.json")

    names = ["layers.35.self_attn.q_proj.weight"]
    run_manifest = {
        "schema_version": 2, "arm": "last-block", "model": template["model_key"],
        "repo": template["model_id"], "revision": template["model_revision"],
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "system_test_manifest_sha256": lock_sha,
        "selection_sha256": "a" * 64, "seed": 20261009, "epochs": 1, "device": "cuda", "threads": 1,
        "formatter": template["prompt_format"]["id"], "pooling": template["inference"]["pooling"],
        "capacity": template["inference"]["choice_head_capacity"],
        "max_tokens": 40960,
        "attention_backend": template["inference"]["attention_backend"],
        "parameters_dtype": template["inference"]["parameters_dtype"],
        "compute_autocast": template["inference"]["backbone_autocast"],
        "head_architecture": "last-token-linear-v1", "backbone_training_scope": "last-block",
        "backbone_trainable": True, "trainable_backbone_parameter_names": names,
        "trainable_backbone_parameter_count": 1, "warm_start": {
            "checkpoint": str(warm_start), "checkpoint_manifest_sha256": base_checkpoint_sha,
            "backbone_sha256": sha256_file(base_backbone), "head_sha256": sha256_file(base_head),
        },
    }
    _write_json(main / "manifest.json", run_manifest)
    index = {}
    fast_candidates = {}
    scores = {0: 1.3, 360: 1.2, 720: 0.9, 1080: 1.1, 1438: 1.0}
    for update in (0, 360, 720, 1080, 1438):
        checkpoint = main / "checkpoints" / f"update-{update:08d}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "backbone.pt").write_bytes(f"overlay {update}".encode())
        (checkpoint / "head.pt").write_bytes(f"head {update}".encode())
        (checkpoint / "tokenizer").mkdir()
        (checkpoint / "tokenizer/tokenizer.json").write_text("{}\n")
        (checkpoint / "backbone_config").mkdir()
        _write_json(checkpoint / "backbone_config/config.json", {"num_hidden_layers": 36})
        files = {
            "backbone.pt": sha256_file(checkpoint / "backbone.pt"),
            "head.pt": sha256_file(checkpoint / "head.pt"),
            "tokenizer/tokenizer.json": sha256_file(checkpoint / "tokenizer/tokenizer.json"),
            "backbone_config/config.json": sha256_file(checkpoint / "backbone_config/config.json"),
        }
        relative_base = os.path.relpath(warm_start, checkpoint).replace(os.sep, "/")
        checkpoint_record = {
            "schema_version": 1, "include_backbone": True,
            "backbone_state_kind": "trainable-overlay-v1", "base_checkpoint": {
                "path": relative_base, "manifest_sha256": base_checkpoint_sha,
            },
            "backbone_parameter_names": names, "files": files,
            "progress": {"cursor": {"global_update": update}},
            "manifest": dict(run_manifest, output="runs/new-plan/main"),
        }
        _write_json(checkpoint / "checkpoint.json", checkpoint_record)
        checkpoint_sha = sha256_file(checkpoint / "checkpoint.json")
        checkpoint_relative = checkpoint.relative_to(root).as_posix()
        index[str(update)] = {
            "path": checkpoint_relative, "checkpoint_manifest_sha256": checkpoint_sha,
            "dev": None, "score_origin": None,
        }
        fast_candidates[str(update)] = {
            "path": checkpoint_relative, "checkpoint_manifest_sha256": checkpoint_sha,
        }
    candidate_index_path = main / "deferred-candidates.json"
    _write_json(candidate_index_path, index)

    fast_dir = family / "fast-evaluation-003"
    original_protocol_path = family / "protocol.json"
    _write_json(original_protocol_path, {
        "snapshot": str(snapshot), "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "system_test_lock": str(lock_path), "system_test_manifest_sha256": lock_sha,
    })
    code_path = fast_dir / "frozen-runner.py"
    code_path.parent.mkdir(parents=True)
    code_path.write_text("frozen code\n")
    fast_manifest_path = fast_dir / "protocol.json"
    fast_manifest = {
        "schema_version": "fast-evaluation-amendment-v1",
        "snapshot": snapshot.relative_to(root).as_posix(),
        "snapshot_manifest_sha256": sha256_file(snapshot_manifest_path),
        "splits_sha256": sha256_file(snapshot / "splits.jsonl"),
        "system_test_lock": lock_path.relative_to(root).as_posix(),
        "system_test_manifest_sha256": lock_sha,
        "original_protocol_path": original_protocol_path.relative_to(root).as_posix(),
        "original_protocol_sha256": sha256_file(original_protocol_path),
        "candidate_index_path": candidate_index_path.relative_to(root).as_posix(),
        "candidate_index_sha256": sha256_file(candidate_index_path),
        "candidate_updates": [0, 360, 720, 1080, 1438], "candidates": fast_candidates,
        "profiles": {"dev": {"questions": 128}},
        "runtime_environment": {"HSA_CU_MASK": "0:0-23"},
        "code_hashes": {code_path.relative_to(root).as_posix(): sha256_file(code_path)},
    }
    _write_json(fast_manifest_path, fast_manifest)
    fast_sha = sha256_file(fast_manifest_path)
    metrics_path = fast_dir / "development-selection.json"
    metric_candidates = {
        str(update): {
            "update": update, "metrics": {"soft_cross_entropy": scores[update]},
            "manifest_sha256": fast_sha,
            "checkpoint_manifest_sha256": fast_candidates[str(update)]["checkpoint_manifest_sha256"],
        }
        for update in scores
    }
    _write_json(metrics_path, {"manifest_sha256": fast_sha, "selected_update": 720, "candidates": metric_candidates})
    selection_path = fast_dir / "selected-checkpoint.json"
    _write_json(selection_path, {
        "checkpoint_path": fast_candidates["720"]["path"],
        "checkpoint_manifest_sha256": fast_candidates["720"]["checkpoint_manifest_sha256"],
        "update": 720, "selection_manifest_sha256": fast_sha,
        "selection_metrics_path": metrics_path.relative_to(root).as_posix(),
        "selected_checkpoint_reload_verified": True,
    })

    protocol_path = fast_dir / "blind-protocol.json"
    protocol = bind_fast_development_protocol(
        snapshot, lock_path, fast_manifest_path, selection_path, protocol_path, template_path,
    )
    assert protocol["protocol_id"] == "independent-mcq-current-last-block-2026-10-v4"
    assert protocol["candidates"] == ["last-block"]
    assert protocol["selection_binding"]["kind"] == "fast-development-selection"
    assert protocol["selection_binding"]["selected_checkpoint_update"] == 720
    assert protocol["selection_binding"]["training_max_tokens"] == 40960
    assert protocol["selection_binding"]["evaluation_input_limit_tokens"] == 32768
    assert protocol["inference"]["input_limit_tokens"] == 32768

    monkeypatch.setattr(blind_runner, "ROOT", root)
    monkeypatch.setattr(blind_runner, "EXTERNAL_DATA_ROOT", external)
    parsed, _ = blind_runner._load_protocol(protocol_path)
    validated_lock, _ = blind_runner._validate_system_bundle(parsed)
    assert validated_lock["training_snapshot_manifest_sha256"] == protocol["training_snapshot_manifest_sha256"]
    candidates = blind_runner._load_protocol_candidates(parsed, root / "unused-full", root / "unused-head")
    receipt = blind_runner._verify_selection_binding(parsed, candidates, root / "unused-receipt.json")
    assert receipt["kind"] == "fast-development-selection"
    assert receipt["selected_checkpoint_update"] == 720

    too_large_template = dict(template)
    too_large_template["inference"] = dict(template["inference"], input_limit_tokens=65536)
    too_large_template_path = root / "runs/new-plan/too-large-template.json"
    _write_json(too_large_template_path, too_large_template)
    with pytest.raises(ValueError, match="development_run_protocol_mismatch"):
        bind_fast_development_protocol(
            snapshot, lock_path, fast_manifest_path, selection_path,
            root / "runs/new-plan/rejected-protocol.json", too_large_template_path,
        )

    metrics = json.loads(metrics_path.read_text())
    metrics["candidates"]["360"]["metrics"]["soft_cross_entropy"] = 0.1
    _write_json(metrics_path, metrics)
    with pytest.raises(blind_runner.BlindEvaluationError, match="candidate_fast_selection_binding_invalid"):
        blind_runner._verify_selection_binding(parsed, candidates, root / "unused-receipt.json")


def test_snapshot_bound_external_memory_wait_releases_cache_and_honors_stop(tmp_path, monkeypatch):
    class WakeEvent:
        def __init__(self):
            self.waits = []
            self.is_set = False

        def set(self):
            self.is_set = True

        def clear(self):
            self.is_set = False

        def wait(self, seconds):
            self.waits.append(seconds)
            return True

    event = WakeEvent()
    monkeypatch.setattr(blind_runner, "_STOP_EVENT", event)
    monkeypatch.setattr(blind_runner, "_STOP_REQUESTED", False)
    checks = 0
    released = []
    snapshot = {
        "host_available_bytes": 2 * 1024**3,
        "driver_free_bytes": 1 * 1024**3,
        "device_total_bytes": 48 * 1024**3,
        "device_allocated_bytes": 20 * 1024**3,
        "device_reserved_bytes": 20 * 1024**3,
        "process_cap_headroom_bytes": 20 * 1024**3,
        "effective_available_bytes": 1 * 1024**3,
    }

    def require_memory(additional_bytes, reserve_gib, device):
        nonlocal checks
        assert reserve_gib == 12 and device == "cuda"
        checks += 1
        if checks == 2:
            blind_runner._request_stop(15, None)
        raise TrainingMemoryPressure(snapshot, additional_bytes, int(reserve_gib * 1024**3))

    monkeypatch.setattr("paired_training.require_memory", require_memory)
    torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: released.append(True)))
    status_path = tmp_path / "run-status.json"

    result = blind_runner._wait_for_model_memory(
        10 * 1024**3, {"arm": "last-block"}, status_path, torch
    )

    assert result is False
    assert released == [True]
    assert event.waits == [30]
    assert event.is_set is True
    status = json.loads(status_path.read_text())
    assert status["phase"] == "waiting_for_model_memory"
    assert status["memory_waiting"] is True
    assert status["candidate"] == "last-block"
    assert status["memory"]["host_available"] == 2.0


def test_compact_checkpoint_binding_uses_exact_full_warm_start_base(tmp_path):
    root = tmp_path
    names = ["layers.35.self_attn.q_proj.weight", "layers.35.self_attn.q_proj.bias"]
    base = root / "runs/original/checkpoints/update-00008017"
    base.mkdir(parents=True)
    base_backbone = base / "backbone.pt"
    base_backbone.write_bytes(b"full immutable backbone")
    base_head = base / "head.pt"
    base_head.write_bytes(b"base head")
    base_data = {
        "schema_version": 1, "include_backbone": True, "backbone_state_kind": "full",
        "backbone_parameter_names": None, "base_checkpoint": None,
        "files": {
            "backbone.pt": sha256_file(base_backbone),
            "head.pt": sha256_file(base_head),
        },
        "manifest": {
            "arm": "last-block", "repo": "model/repo", "revision": "revision-hash",
            "head_architecture": "last-token-linear-v1", "trainable_backbone_parameter_names": names,
        },
    }
    _write_json(base / "checkpoint.json", base_data)

    current = root / "runs/new-mixture/main/checkpoints/selected"
    current.mkdir(parents=True)
    (current / "backbone.pt").write_bytes(b"last-block overlay")
    reference_path = os.path.relpath(base, current).replace(os.sep, "/")
    current_data = {
        "backbone_state_kind": "trainable-overlay-v1",
        "backbone_parameter_names": names,
        "base_checkpoint": {"path": reference_path, "manifest_sha256": sha256_file(base / "checkpoint.json")},
        "manifest": {
            "head_architecture": "last-token-linear-v1", "trainable_backbone_parameter_names": names,
        },
    }
    run_manifest = {
        "repo": "model/repo", "revision": "revision-hash",
        "warm_start": {
            "checkpoint": "/workspace/" + base.relative_to(root).as_posix(),
            "checkpoint_manifest_sha256": sha256_file(base / "checkpoint.json"),
            "backbone_sha256": sha256_file(base_backbone),
            "head_sha256": sha256_file(base_head),
        },
    }

    binding = validate_compact_checkpoint_binding(root, current, current_data, run_manifest, verify_base_backbone=True)
    assert binding["path"] == base.relative_to(root).as_posix()
    assert binding["backbone_sha256"] == sha256_file(base_backbone)
    assert binding["backbone_parameter_count"] == len(names)

    base_backbone.write_bytes(b"changed backbone")
    with pytest.raises(ValueError, match="base_backbone_mismatch"):
        validate_compact_checkpoint_binding(root, current, current_data, run_manifest, verify_base_backbone=True)


def test_compact_checkpoint_binding_rejects_unmapped_absolute_base_path(tmp_path):
    root = tmp_path
    current = root / "runs/new-mixture/main/checkpoints/selected"
    current.mkdir(parents=True)
    with pytest.raises(ValueError, match="path_outside_project"):
        validate_compact_checkpoint_binding(
            root,
            current,
            {
                "backbone_state_kind": "trainable-overlay-v1",
                "base_checkpoint": {"path": "../../base", "manifest_sha256": "a" * 64},
                "backbone_parameter_names": ["layers.35.attn.q_proj.weight"],
                "manifest": {"trainable_backbone_parameter_names": ["layers.35.attn.q_proj.weight"]},
            },
            {
                "repo": "model/repo", "revision": "revision",
                "warm_start": {"checkpoint": "/tmp/not-the-project/runs/base"},
            },
        )
