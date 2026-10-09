import copy
import json

import pytest

from evaluation.snapshot_bound_protocol import (
    DEFERRED_DEVELOPMENT_EXECUTION_V1,
    sha256_file,
    validate_deferred_development_selection,
    validate_resource_lifecycle_audit_chain,
)


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _deferred_run(root):
    family = root / "runs" / "experiment"
    main = family / "main"
    recovery = family / "main-recovery-001"
    active = family / "main-recovery-002"
    snapshot_hash = "b" * 64
    lock_hash = "c" * 64
    selection_inputs = {"ids_by_split": {"dev": [f"dev-{i}" for i in range(128)]}}
    _write_json(family / "selection.json", selection_inputs)
    selection_hash = sha256_file(family / "selection.json")
    _write_json(family / "protocol.json", {"selection_sha256": selection_hash})
    metrics = {"questions": 128, "soft_cross_entropy": 0.5, "kl_pq": 0.2}
    common = {
        "schema_version": 2,
        "arm": "last-block",
        "snapshot_manifest_sha256": snapshot_hash,
        "selection_sha256": selection_hash,
        "system_test_manifest_sha256": lock_hash,
        "epochs": 1,
        "scheduler_total_updates": 1438,
        "full_dev_every": 360,
        "fast_dev_questions": 128,
        "early_stop_patience": 0,
        "max_updates": 0,
        "order_consistency_questions": 0,
        "runner_sha256": "",
        "input_sha256": "d" * 64,
    }
    old_source = family / "code" / "scalable_training.py"
    old_source.parent.mkdir(parents=True, exist_ok=True)
    old_source.write_bytes(b"original engine")
    old_input_source = family / "code" / "paired_training.py"
    old_input_source.write_bytes(b"original input")
    main_manifest = {
        **common, "output": "runs/experiment/main", "runner_sha256": sha256_file(old_source),
        "input_sha256": sha256_file(old_input_source),
    }
    _write_json(main / "manifest.json", main_manifest)
    dev_protocol = {
        "full_ids": selection_inputs["ids_by_split"]["dev"],
        "fast_ids": selection_inputs["ids_by_split"]["dev"],
        "fast_every_updates": 64,
        "full_every_updates": 360,
    }
    _write_json(main / "baselines.json", {"warm_start_development": metrics})
    _write_json(main / "history.json", [{"update": 0, "dev": metrics}])
    _write_json(main / "development-protocol.json", dev_protocol)
    main_checkpoint = main / "checkpoints" / "update-00000000"
    _write_json(main_checkpoint / "checkpoint.json", {
        "manifest": main_manifest, "progress": {"cursor": {"global_update": 0}},
    })

    recovery_source = recovery / "scalable_training.py"
    recovery.mkdir(parents=True, exist_ok=True)
    recovery_source.write_bytes(b"recovery engine")
    recovery_manifest = {
        **main_manifest,
        "output": "runs/experiment/main-recovery-001",
        "runner_sha256": sha256_file(recovery_source),
    }
    recovery_audit = {
        "schema_version": "training-resource-lifecycle-audit-v1",
        "purpose": "resource-lifecycle-only",
        "resume_checkpoint": "runs/experiment/main/checkpoints/update-00000000",
        "previous_run_manifest": "runs/experiment/main/manifest.json",
        "checkpoint_manifest_sha256": sha256_file(main_checkpoint / "checkpoint.json"),
        "previous_run_manifest_sha256": sha256_file(main / "manifest.json"),
        "source_changes": {"runner_sha256": {
            "previous": main_manifest["runner_sha256"], "current": recovery_manifest["runner_sha256"],
        }},
        "previous_source_files": {"runner_sha256": "runs/experiment/code/scalable_training.py"},
        "baseline_reuse": {
            "artifacts": {
                name: sha256_file(main / name)
                for name in ("baselines.json", "history.json", "development-protocol.json", "manifest.json")
            },
            "protocol_path": "runs/experiment/protocol.json",
            "protocol_sha256": sha256_file(family / "protocol.json"),
            "selection_sha256": selection_hash,
        },
    }
    recovery_audit_path = recovery / "lifecycle-audit.json"
    _write_json(recovery_audit_path, recovery_audit)
    recovery_manifest["resource_lifecycle_audit"] = {
        "path": "runs/experiment/main-recovery-001/lifecycle-audit.json",
        "sha256": sha256_file(recovery_audit_path), "record": recovery_audit,
    }
    _write_json(recovery / "manifest.json", recovery_manifest)
    recovery_checkpoint = recovery / "checkpoints" / "update-00000000"
    _write_json(recovery_checkpoint / "checkpoint.json", {
        "manifest": recovery_manifest, "progress": {"cursor": {"global_update": 0}},
    })
    _write_json(recovery / "baselines.json", {"warm_start_development": metrics})
    _write_json(recovery / "history.json", [{"update": 0, "dev": metrics}])
    _write_json(recovery / "development-protocol.json", dev_protocol)

    active_source = active / "scalable_training.py"
    input_source = active / "paired_training.py"
    active.mkdir(parents=True, exist_ok=True)
    active_source.write_bytes(b"deferred engine")
    input_source.write_bytes(b"deferred input")
    active_manifest = {
        **recovery_manifest,
        "output": "runs/experiment/main-recovery-002",
        "runner_sha256": sha256_file(active_source),
        "input_sha256": sha256_file(input_source),
        "development_execution": copy.deepcopy(DEFERRED_DEVELOPMENT_EXECUTION_V1),
    }
    binding = {
        "artifacts": {
            name: sha256_file(recovery / name)
            for name in ("baselines.json", "history.json", "development-protocol.json", "manifest.json")
        },
        "protocol_path": "runs/experiment/protocol.json",
        "protocol_sha256": sha256_file(family / "protocol.json"),
        "selection_sha256": selection_hash,
    }
    active_audit = {
        "schema_version": "training-resource-lifecycle-audit-v1",
        "purpose": "development-evaluation-deferral",
        "resume_checkpoint": "runs/experiment/main-recovery-001/checkpoints/update-00000000",
        "previous_run_manifest": "runs/experiment/main-recovery-001/manifest.json",
        "checkpoint_manifest_sha256": sha256_file(recovery_checkpoint / "checkpoint.json"),
        "previous_run_manifest_sha256": sha256_file(recovery / "manifest.json"),
        "source_changes": {
            "runner_sha256": {
                "previous": recovery_manifest["runner_sha256"], "current": active_manifest["runner_sha256"],
            },
            "input_sha256": {
                "previous": recovery_manifest["input_sha256"], "current": active_manifest["input_sha256"],
            },
        },
        "previous_source_files": {
            "runner_sha256": "runs/experiment/main-recovery-001/scalable_training.py",
            "input_sha256": "runs/experiment/code/paired_training.py",
        },
        "development_state_reuse": binding,
        "development_execution": copy.deepcopy(DEFERRED_DEVELOPMENT_EXECUTION_V1),
    }
    active_audit_path = active / "lifecycle-audit.json"
    _write_json(active_audit_path, active_audit)
    active_manifest["resource_lifecycle_audit"] = {
        "path": "runs/experiment/main-recovery-002/lifecycle-audit.json",
        "sha256": sha256_file(active_audit_path), "record": active_audit,
    }
    _write_json(active / "manifest.json", active_manifest)
    _write_json(active / "development-protocol.json", dev_protocol)
    _write_json(active / "baselines.json", {"warm_start_development": metrics})
    _write_json(active / "history.json", [{"update": 0, "dev": metrics}])

    candidate_index = {}
    for update in DEFERRED_DEVELOPMENT_EXECUTION_V1["candidate_updates"]:
        if update == 0:
            checkpoint = main_checkpoint
            candidate_metrics = metrics
            score_origin = "audited completed baseline"
        else:
            checkpoint = active / "checkpoints" / f"update-{update:08d}"
            candidate_metrics = {"questions": 128, "soft_cross_entropy": 1.0 + update / 10000}
            score_origin = "single post-training full-development pass"
        _write_json(checkpoint / "checkpoint.json", {
            "manifest": main_manifest if update == 0 else active_manifest,
            "progress": {"cursor": {"global_update": update}},
        })
        candidate_index[str(update)] = {
            "path": checkpoint.relative_to(root).as_posix(),
            "checkpoint_manifest_sha256": sha256_file(checkpoint / "checkpoint.json"),
            "dev": candidate_metrics,
            "score_origin": score_origin,
        }
    _write_json(active / "deferred-candidates.json", candidate_index)
    selected_checkpoint = main_checkpoint
    relative = "../main/checkpoints/update-00000000"
    _write_json(active / "best-checkpoint.json", {
        "path": relative, "update": 0, "loss": metrics["soft_cross_entropy"],
    })
    selection = {
        "schema_version": "post-training-development-selection-v1",
        "candidate_updates": copy.deepcopy(DEFERRED_DEVELOPMENT_EXECUTION_V1["candidate_updates"]),
        "best_update": 0,
        "best_dev_soft_cross_entropy": metrics["soft_cross_entropy"],
        "selected_checkpoint_manifest_sha256": candidate_index["0"]["checkpoint_manifest_sha256"],
        "candidate_index_sha256": sha256_file(active / "deferred-candidates.json"),
        "selected_reload_verification": "artifact checksums and exact tensor load; stored single-pass development score reused",
        "duplicate_selected_development_inference": False,
    }
    _write_json(active / "deferred-selection.json", selection)
    report = {
        "status": "completed",
        "cursor": {"global_update": 1438},
        "early_stopped": False,
        "development_execution": copy.deepcopy(DEFERRED_DEVELOPMENT_EXECUTION_V1),
        "deferred_selection_sha256": sha256_file(active / "deferred-selection.json"),
        "best_update": 0,
        "best_dev_soft_cross_entropy": metrics["soft_cross_entropy"],
        "best_checkpoint_relative": relative,
        "selected_checkpoint_development": metrics,
    }
    _write_json(active / "report.json", report)
    return active, active_manifest, report, selected_checkpoint


def _deferred_retry(root, previous_run, previous_manifest, previous_report):
    family = previous_run.parent
    current_run = family / "main-recovery-003"
    previous_index = json.loads((previous_run / "deferred-candidates.json").read_text())
    selection_hash = previous_manifest["selection_sha256"]
    policy = copy.deepcopy(DEFERRED_DEVELOPMENT_EXECUTION_V1)
    resume_checkpoint = previous_run / "checkpoints" / "update-00001438"
    previous_source = previous_run / "scalable_training.py"
    current_manifest = {
        **previous_manifest,
        "output": "runs/experiment/main-recovery-003",
    }
    binding = {
        "artifacts": {
            name: sha256_file(previous_run / name)
            for name in (
                "baselines.json", "history.json", "development-protocol.json", "manifest.json",
                "deferred-candidates.json",
            )
        },
        "protocol_path": "runs/experiment/protocol.json",
        "protocol_sha256": sha256_file(family / "protocol.json"),
        "selection_sha256": selection_hash,
    }
    record = {
        "schema_version": "training-resource-lifecycle-audit-v1",
        "purpose": "development-evaluation-deferral",
        "resume_checkpoint": resume_checkpoint.relative_to(root).as_posix(),
        "previous_run_manifest": (previous_run / "manifest.json").relative_to(root).as_posix(),
        "checkpoint_manifest_sha256": sha256_file(resume_checkpoint / "checkpoint.json"),
        "previous_run_manifest_sha256": sha256_file(previous_run / "manifest.json"),
        "source_changes": {"runner_sha256": {
            "previous": previous_manifest["runner_sha256"],
            "current": previous_manifest["runner_sha256"],
        }},
        "previous_source_files": {
            "runner_sha256": previous_source.relative_to(root).as_posix(),
        },
        "development_state_reuse": binding,
        "development_execution": policy,
    }
    audit_path = current_run / "lifecycle-audit.json"
    _write_json(audit_path, record)
    current_manifest["resource_lifecycle_audit"] = {
        "path": audit_path.relative_to(root).as_posix(),
        "sha256": sha256_file(audit_path), "record": record,
    }
    _write_json(current_run / "manifest.json", current_manifest)
    for name in ("development-protocol.json", "baselines.json", "history.json",
                 "deferred-candidates.json", "deferred-selection.json", "best-checkpoint.json"):
        (current_run / name).parent.mkdir(parents=True, exist_ok=True)
        (current_run / name).write_bytes((previous_run / name).read_bytes())
    _write_json(current_run / "report.json", previous_report)
    return current_run, current_manifest, previous_index


def test_deferred_selection_accepts_audited_baseline_winner_and_two_link_chain(tmp_path):
    active, manifest, report, checkpoint = _deferred_run(tmp_path)

    chain = validate_resource_lifecycle_audit_chain(tmp_path, manifest)
    binding = validate_deferred_development_selection(tmp_path, active, manifest, report, checkpoint)

    assert [entry["purpose"] for entry in chain] == [
        "development-evaluation-deferral", "resource-lifecycle-only",
    ]
    assert binding["selected_checkpoint_update"] == 0
    assert binding["development_execution"] == DEFERRED_DEVELOPMENT_EXECUTION_V1


def test_deferred_state_reuse_rejects_changed_family_selection_input(tmp_path):
    active, manifest, _, _ = _deferred_run(tmp_path)
    selection_path = active.parent / "selection.json"
    selection = json.loads(selection_path.read_text())
    selection["ids_by_split"]["dev"][0] = "different-dev-id"
    _write_json(selection_path, selection)

    with pytest.raises(ValueError, match="deferred_development_state_inputs_changed"):
        validate_resource_lifecycle_audit_chain(tmp_path, manifest)


def test_deferred_selection_rejects_policy_drift_and_unbound_candidate_edits(tmp_path):
    active, manifest, report, checkpoint = _deferred_run(tmp_path)
    altered_manifest = copy.deepcopy(manifest)
    altered_manifest["development_execution"]["candidate_updates"][-1] = 1437
    with pytest.raises(ValueError, match="development_evaluation_deferral_policy_invalid"):
        validate_resource_lifecycle_audit_chain(tmp_path, altered_manifest)

    index_path = active / "deferred-candidates.json"
    index = json.loads(index_path.read_text())
    index["0"]["dev"]["soft_cross_entropy"] = 0.75
    _write_json(index_path, index)
    with pytest.raises(ValueError, match="deferred_development_selection_binding_mismatch"):
        validate_deferred_development_selection(tmp_path, active, manifest, report, checkpoint)


def test_deferred_selection_recomputes_minimum_and_earliest_tie_break(tmp_path):
    active, manifest, report, _ = _deferred_run(tmp_path)
    index = json.loads((active / "deferred-candidates.json").read_text())
    selection = json.loads((active / "deferred-selection.json").read_text())
    update = 360
    index[str(update)]["dev"]["soft_cross_entropy"] = index["0"]["dev"]["soft_cross_entropy"]
    _write_json(active / "deferred-candidates.json", index)
    selection.update({
        "best_update": update,
        "best_dev_soft_cross_entropy": index[str(update)]["dev"]["soft_cross_entropy"],
        "selected_checkpoint_manifest_sha256": index[str(update)]["checkpoint_manifest_sha256"],
        "candidate_index_sha256": sha256_file(active / "deferred-candidates.json"),
    })
    _write_json(active / "deferred-selection.json", selection)
    selected = active / "checkpoints" / "update-00000360"
    report.update({
        "deferred_selection_sha256": sha256_file(active / "deferred-selection.json"),
        "best_update": update,
        "best_dev_soft_cross_entropy": selection["best_dev_soft_cross_entropy"],
        "best_checkpoint_relative": "checkpoints/update-00000360",
        "selected_checkpoint_development": index[str(update)]["dev"],
    })
    _write_json(active / "report.json", report)
    _write_json(active / "best-checkpoint.json", {
        "path": "checkpoints/update-00000360", "update": update,
        "loss": selection["best_dev_soft_cross_entropy"],
    })

    with pytest.raises(ValueError, match="deferred_development_selected_checkpoint_mismatch"):
        validate_deferred_development_selection(tmp_path, active, manifest, report, selected)


def test_post_scoring_retry_reuses_only_bound_sibling_candidate_records(tmp_path):
    previous_run, previous_manifest, previous_report, _ = _deferred_run(tmp_path)
    retry_run, retry_manifest, prior_index = _deferred_retry(
        tmp_path, previous_run, previous_manifest, previous_report,
    )
    selected = tmp_path / prior_index["0"]["path"]

    chain = validate_resource_lifecycle_audit_chain(tmp_path, retry_manifest)
    binding = validate_deferred_development_selection(
        tmp_path, retry_run, retry_manifest, previous_report, selected,
    )

    assert [entry["purpose"] for entry in chain] == [
        "development-evaluation-deferral",
        "development-evaluation-deferral",
        "resource-lifecycle-only",
    ]
    assert binding["selected_checkpoint_update"] == 0

    index_path = retry_run / "deferred-candidates.json"
    index = json.loads(index_path.read_text())
    index["360"]["path"] = "runs/experiment/main-recovery-002/checkpoints/update-00000720"
    _write_json(index_path, index)
    selection_path = retry_run / "deferred-selection.json"
    selection = json.loads(selection_path.read_text())
    selection["candidate_index_sha256"] = sha256_file(index_path)
    _write_json(selection_path, selection)
    retry_report = copy.deepcopy(previous_report)
    retry_report["deferred_selection_sha256"] = sha256_file(selection_path)
    _write_json(retry_run / "report.json", retry_report)
    with pytest.raises(ValueError, match="deferred_development_reused_candidate_mismatch"):
        validate_deferred_development_selection(
            tmp_path, retry_run, retry_manifest, retry_report, selected,
        )


def test_post_scoring_retry_scores_only_a_prior_pending_candidate(tmp_path):
    previous_run, previous_manifest, previous_report, _ = _deferred_run(tmp_path)
    previous_index_path = previous_run / "deferred-candidates.json"
    previous_index = json.loads(previous_index_path.read_text())
    pending = previous_index["720"]
    previous_index["720"] = {
        "path": pending["path"],
        "checkpoint_manifest_sha256": pending["checkpoint_manifest_sha256"],
    }
    _write_json(previous_index_path, previous_index)
    retry_run, retry_manifest, _ = _deferred_retry(
        tmp_path, previous_run, previous_manifest, previous_report,
    )
    retry_index_path = retry_run / "deferred-candidates.json"
    retry_index = json.loads(retry_index_path.read_text())
    retry_index["720"].update({
        "dev": {"questions": 128, "soft_cross_entropy": 1.072},
        "score_origin": "single post-training full-development pass",
    })
    _write_json(retry_index_path, retry_index)
    selection_path = retry_run / "deferred-selection.json"
    selection = json.loads(selection_path.read_text())
    selection["candidate_index_sha256"] = sha256_file(retry_index_path)
    _write_json(selection_path, selection)
    retry_report = copy.deepcopy(previous_report)
    retry_report["deferred_selection_sha256"] = sha256_file(selection_path)
    _write_json(retry_run / "report.json", retry_report)
    selected = tmp_path / retry_index["0"]["path"]

    binding = validate_deferred_development_selection(
        tmp_path, retry_run, retry_manifest, retry_report, selected,
    )

    assert binding["selected_checkpoint_update"] == 0


def test_post_scoring_retry_rejects_a_resume_before_all_updates_complete(tmp_path):
    previous_run, previous_manifest, previous_report, _ = _deferred_run(tmp_path)
    retry_run, retry_manifest, _ = _deferred_retry(
        tmp_path, previous_run, previous_manifest, previous_report,
    )
    audit_path = retry_run / "lifecycle-audit.json"
    record = json.loads(audit_path.read_text())
    incomplete = previous_run / "checkpoints" / "update-00000720"
    record["resume_checkpoint"] = incomplete.relative_to(tmp_path).as_posix()
    record["checkpoint_manifest_sha256"] = sha256_file(incomplete / "checkpoint.json")
    _write_json(audit_path, record)
    retry_manifest["resource_lifecycle_audit"] = {
        "path": audit_path.relative_to(tmp_path).as_posix(),
        "sha256": sha256_file(audit_path), "record": record,
    }

    with pytest.raises(ValueError, match="deferred_development_retry_requires_completed_weights"):
        validate_resource_lifecycle_audit_chain(tmp_path, retry_manifest)
