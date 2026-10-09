"""Bind training snapshots, aggregate overlap audits, and frozen blind protocols.

This module only reads training metadata and aggregate overlap-audit output. It
never parses canonical benchmark items; their bytes are hashed by the audit and
the evaluator under the blind-benchmark rule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


EXTERNAL_ROOT_PARTS = ("data", "external_evaluation")
SNAPSHOT_ROOT_PARTS = ("data", "training_snapshots")
OLD_SUITE_LOCK = "data/external_evaluation/system-test-lock-current-corpus-2026-10-01.json"
DEFAULT_PROTOCOL_TEMPLATE = "evaluation/protocols/last-block-blind-template-v1.json"
PROTOCOL_ID_PREFIX = "independent-mcq-current-last-block-2026-10-"
_MAIN_RUN_DIRECTORY = re.compile(r"main(?:-recovery-([0-9]{3}))?\Z")
FAST_SELECTION_UPDATES = [0, 360, 720, 1080, 1438]
FAST_SELECTION_QUESTION_COUNT = 128
FAST_SELECTION_BINDING_KIND = "fast-development-selection"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(code) from None
    if not isinstance(value, dict):
        raise ValueError(code)
    return value


def _project_root(path: Path) -> Path:
    resolved = path.resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "data" / "external_evaluation").is_dir():
            return candidate
    raise ValueError("project_root_not_found")


def _rooted(root: Path, value: str | Path, *, allowed: tuple[str, ...] | None = None) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(candidate))
    root = root.resolve()
    if not lexical.is_relative_to(root) or ".." in raw.parts:
        raise ValueError("path_outside_project")
    current = root
    for part in lexical.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinked_artifact")
    path = lexical.resolve()
    if not path.is_relative_to(root):
        raise ValueError("path_outside_project")
    if allowed is not None and not path.is_relative_to((root.joinpath(*allowed)).resolve()):
        raise ValueError("path_outside_allowed_root")
    return path


def _relative(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _selected_main_checkpoint(root: Path, run_path: Path, relative_path: str) -> tuple[Path, Path]:
    """Resolve a main/recovery selection only within its experiment run family."""
    root = Path(root).resolve()
    run_path = _rooted(root, run_path, allowed=("runs",))
    if not _MAIN_RUN_DIRECTORY.fullmatch(run_path.name):
        raise ValueError("main_run_directory_invalid")
    family = run_path.parent.resolve()
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or "\\" in relative_path or not relative.parts:
        raise ValueError("selected_checkpoint_path_invalid")
    current = run_path
    for part in relative.parts:
        if part == ".":
            raise ValueError("selected_checkpoint_path_invalid")
        current = current.parent if part == ".." else current / part
        current = Path(os.path.abspath(current))
        if not current.is_relative_to(family) or current.is_symlink():
            raise ValueError("selected_checkpoint_outside_main_run_family")
    checkpoint = current.resolve()
    source_run = checkpoint.parent.parent
    if (
        checkpoint.parent.name != "checkpoints"
        or source_run.parent.resolve() != family
        or not _MAIN_RUN_DIRECTORY.fullmatch(source_run.name)
        or not checkpoint.is_dir()
    ):
        raise ValueError("selected_checkpoint_outside_main_run_family")
    return checkpoint, source_run


def _runtime_rooted(root: Path, value: str | Path, *, allowed: tuple[str, ...] | None = None) -> Path:
    raw = Path(value)
    if raw.is_absolute() and raw.parts[:2] == ("/", "workspace") and root.resolve() != Path("/workspace"):
        raw = Path(*raw.parts[2:])
    return _rooted(root, raw, allowed=allowed)


_LIFECYCLE_HASH_FIELDS = {"runner_sha256", "loop_sha256", "checkpoint_sha256", "input_sha256"}
DEFERRED_DEVELOPMENT_EXECUTION_V1 = {
    "timing": "after all weight updates",
    "candidate_updates": [0, 360, 720, 1080, 1438],
    "selection": "minimum full-development soft cross entropy; earliest update breaks ties",
    "baseline": "reuse audited complete initial development metrics",
    "selected_reload": "verified artifact checksums and tensor load; no duplicate full-development scoring",
    "fast_monitoring": "unchanged; never selects checkpoints",
}


def _expected_development_execution(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if (
        manifest.get("epochs") != 1
        or manifest.get("scheduler_total_updates") != 1438
        or manifest.get("full_dev_every") != 360
        or manifest.get("fast_dev_questions") != 128
        or manifest.get("early_stop_patience") != 0
        or manifest.get("max_updates") not in (0, None)
        or manifest.get("order_consistency_questions") != 0
    ):
        raise ValueError("deferred_development_training_contract_invalid")
    return {**DEFERRED_DEVELOPMENT_EXECUTION_V1, "candidate_updates": list(DEFERRED_DEVELOPMENT_EXECUTION_V1["candidate_updates"])}


def _validate_development_state_reuse(
    root: Path,
    candidate: Mapping[str, Any],
    previous: Mapping[str, Any],
    record: Mapping[str, Any],
    previous_run: Path,
) -> None:
    binding = record.get("development_state_reuse")
    required = {"baselines.json", "history.json", "development-protocol.json", "manifest.json"}
    if previous.get("development_execution") is not None:
        required.add("deferred-candidates.json")
    if not isinstance(binding, Mapping) or set(binding) != {
        "artifacts", "protocol_path", "protocol_sha256", "selection_sha256"
    } or not isinstance(binding.get("artifacts"), Mapping) or set(binding["artifacts"]) != required:
        raise ValueError("deferred_development_state_binding_invalid")
    for name, expected in binding["artifacts"].items():
        artifact = previous_run / name
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError("deferred_development_state_artifact_mismatch")
    protocol_path = _runtime_rooted(root, binding["protocol_path"], allowed=("runs",))
    if (
        protocol_path.resolve() != (previous_run.parent / "protocol.json").resolve()
        or protocol_path.is_symlink()
        or not protocol_path.is_file()
        or sha256_file(protocol_path) != binding["protocol_sha256"]
    ):
        raise ValueError("deferred_development_protocol_binding_invalid")
    protocol = _read_object(protocol_path, "deferred_development_protocol_invalid")
    if (
        protocol.get("selection_sha256") != binding["selection_sha256"]
        or binding["selection_sha256"] != previous.get("selection_sha256")
    ):
        raise ValueError("deferred_development_selection_input_changed")
    current_output_value = candidate.get("output")
    if not isinstance(current_output_value, str):
        raise ValueError("deferred_development_output_missing")
    current_run = _runtime_rooted(root, current_output_value, allowed=("runs",))
    selection_path = _runtime_rooted(root, current_run.parent / "selection.json", allowed=("runs",))
    if (
        current_run.parent.resolve() != previous_run.parent.resolve()
        or
        not (previous_run / "development-protocol.json").is_file()
        or not (current_run / "development-protocol.json").is_file()
        or sha256_file(previous_run / "development-protocol.json")
        != sha256_file(current_run / "development-protocol.json")
        or not selection_path.is_file()
        or sha256_file(selection_path) != binding["selection_sha256"]
        or candidate.get("selection_sha256") != binding["selection_sha256"]
        or previous.get("selection_sha256") != binding["selection_sha256"]
    ):
        raise ValueError("deferred_development_state_inputs_changed")


def _validate_initial_baseline_reuse(
    root: Path, record: Mapping[str, Any], previous: Mapping[str, Any], previous_run: Path
) -> None:
    binding = record.get("baseline_reuse")
    required = {"baselines.json", "history.json", "development-protocol.json", "manifest.json"}
    if not isinstance(binding, Mapping) or not isinstance(binding.get("artifacts"), Mapping) or set(binding["artifacts"]) != required:
        raise ValueError("initial_development_baseline_binding_invalid")
    for name, expected in binding["artifacts"].items():
        artifact = previous_run / name
        if artifact.is_symlink() or not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError("initial_development_baseline_artifact_mismatch")
    protocol_path = _runtime_rooted(root, binding.get("protocol_path", ""), allowed=("runs",))
    if (
        protocol_path.resolve() != (previous_run.parent / "protocol.json").resolve()
        or protocol_path.is_symlink()
        or not protocol_path.is_file()
        or sha256_file(protocol_path) != binding.get("protocol_sha256")
    ):
        raise ValueError("initial_development_baseline_protocol_mismatch")
    protocol = _read_object(protocol_path, "initial_development_baseline_protocol_invalid")
    if (
        binding.get("selection_sha256") != previous.get("selection_sha256")
        or protocol.get("selection_sha256") != binding.get("selection_sha256")
    ):
        raise ValueError("initial_development_baseline_selection_mismatch")
    baselines = _read_object(previous_run / "baselines.json", "initial_development_baseline_metrics_invalid")
    metrics = baselines.get("warm_start_development") or baselines.get("random_head_development")
    try:
        history = json.loads((previous_run / "history.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("initial_development_baseline_history_invalid") from None
    development = _read_object(previous_run / "development-protocol.json", "initial_development_baseline_protocol_invalid")
    resume_checkpoint = _runtime_rooted(root, record.get("resume_checkpoint", ""), allowed=("runs",))
    checkpoint = _read_object(resume_checkpoint / "checkpoint.json", "initial_development_baseline_checkpoint_invalid")
    if (
        not isinstance(metrics, Mapping)
        or not isinstance(development.get("full_ids"), list)
        or metrics.get("questions") != len(development["full_ids"])
        or not isinstance(history, list)
        or not history
        or history[0].get("update") != 0
        or history[0].get("dev") != metrics
        or resume_checkpoint.parent.name != "checkpoints"
        or resume_checkpoint.parent.parent.name != "main"
        or checkpoint.get("progress", {}).get("cursor", {}).get("global_update") != 0
    ):
        raise ValueError("initial_development_baseline_incomplete")


def validate_resource_lifecycle_audit_chain(
    root: Path, manifest: Mapping[str, Any], *, max_links: int = 3
) -> list[dict[str, Any]]:
    """Validate the bounded baseline and deferred-development resume chain."""
    root = Path(root).resolve()
    chain: list[dict[str, Any]] = []

    def visit(candidate: Mapping[str, Any], depth: int) -> None:
        audit = candidate.get("resource_lifecycle_audit")
        if audit is None:
            return
        if depth >= max_links or not isinstance(audit, Mapping):
            raise ValueError("resource_lifecycle_audit_chain_invalid")
        audit_path = _runtime_rooted(root, audit.get("path", ""), allowed=("runs",))
        record = audit.get("record")
        if (
            not isinstance(record, Mapping)
            or not audit_path.is_file()
            or audit_path.is_symlink()
            or sha256_file(audit_path) != audit.get("sha256")
            or _read_object(audit_path, "resource_lifecycle_audit_invalid") != record
            or record.get("schema_version") != "training-resource-lifecycle-audit-v1"
            or record.get("purpose") not in {"resource-lifecycle-only", "development-evaluation-deferral"}
        ):
            raise ValueError("resource_lifecycle_audit_invalid")
        resume_checkpoint = _runtime_rooted(root, record.get("resume_checkpoint", ""), allowed=("runs",))
        previous_manifest_path = _runtime_rooted(root, record.get("previous_run_manifest", ""), allowed=("runs",))
        checkpoint_manifest_path = resume_checkpoint / "checkpoint.json"
        if (
            resume_checkpoint.parent.name != "checkpoints"
            or not resume_checkpoint.is_dir()
            or previous_manifest_path.resolve() != (resume_checkpoint.parent.parent / "manifest.json").resolve()
            or not checkpoint_manifest_path.is_file()
            or not previous_manifest_path.is_file()
            or sha256_file(checkpoint_manifest_path) != record.get("checkpoint_manifest_sha256")
            or sha256_file(previous_manifest_path) != record.get("previous_run_manifest_sha256")
        ):
            raise ValueError("resource_lifecycle_resume_binding_invalid")
        checkpoint_record = _read_object(checkpoint_manifest_path, "resource_lifecycle_checkpoint_invalid")
        previous = _read_object(previous_manifest_path, "resource_lifecycle_previous_manifest_invalid")
        if checkpoint_record.get("manifest") != previous:
            raise ValueError("resource_lifecycle_checkpoint_manifest_mismatch")
        changes = record.get("source_changes")
        sources = record.get("previous_source_files")
        if (
            not isinstance(changes, Mapping)
            or not changes
            or set(changes) - _LIFECYCLE_HASH_FIELDS
            or not isinstance(sources, Mapping)
            or set(sources) != set(changes)
        ):
            raise ValueError("resource_lifecycle_source_changes_invalid")
        purpose = record["purpose"]
        is_deferred_retry = False
        if purpose == "development-evaluation-deferral":
            expected_policy = _expected_development_execution(candidate)
            is_deferred_retry = previous.get("development_execution") == expected_policy
            if (
                record.get("development_execution") != expected_policy
                or candidate.get("development_execution") != expected_policy
                or previous.get("early_stop_patience") != 0
                or record["development_execution"] != candidate.get("development_execution")
                or previous.get("development_execution") not in (None, expected_policy)
            ):
                raise ValueError("development_evaluation_deferral_policy_invalid")
            saved_update = checkpoint_record.get("progress", {}).get("cursor", {}).get("global_update")
            if is_deferred_retry and saved_update != 1438:
                raise ValueError("deferred_development_retry_requires_completed_weights")
            if not is_deferred_retry and (saved_update is None or saved_update >= 360):
                raise ValueError("deferred_development_initial_transition_boundary_invalid")
            _validate_development_state_reuse(root, candidate, previous, record, resume_checkpoint.parent.parent)
        elif candidate.get("development_execution") != previous.get("development_execution"):
            raise ValueError("resource_lifecycle_changed_development_policy")

        for field, transition in changes.items():
            source_name = sources.get(field)
            if (
                not isinstance(transition, Mapping)
                or set(transition) != {"previous", "current"}
                or (
                    transition.get("previous") == transition.get("current")
                    and not (
                        purpose == "development-evaluation-deferral"
                        and is_deferred_retry
                        and field in {"runner_sha256", "input_sha256"}
                    )
                )
                or (
                    purpose == "development-evaluation-deferral"
                    and is_deferred_retry
                    and transition.get("previous") != transition.get("current")
                )
                or transition.get("previous") != previous.get(field)
                or transition.get("current") != candidate.get(field)
                or not isinstance(source_name, str)
            ):
                raise ValueError("resource_lifecycle_source_transition_invalid")
            source_path = _runtime_rooted(root, source_name, allowed=("runs",))
            if not source_path.is_file() or sha256_file(source_path) != transition["previous"]:
                raise ValueError("resource_lifecycle_source_hash_invalid")

        ignored = {"started_at", "output", "resource_lifecycle_audit"}
        if purpose == "development-evaluation-deferral":
            ignored.add("development_execution")
        missing = object()
        for field in (set(previous) | set(candidate)) - ignored:
            before, after = previous.get(field, missing), candidate.get(field, missing)
            if before == after:
                continue
            transition = changes.get(field)
            if not isinstance(transition, Mapping) or transition.get("previous") != before or transition.get("current") != after:
                raise ValueError("resource_lifecycle_noncode_manifest_change")

        chain.append(dict(record))
        if previous.get("resource_lifecycle_audit") is not None:
            visit(previous, depth + 1)

    visit(manifest, 0)
    if chain and chain[0].get("purpose") == "development-evaluation-deferral":
        purposes = [entry.get("purpose") for entry in chain]
        if purposes not in (
            ["development-evaluation-deferral", "resource-lifecycle-only"],
            ["development-evaluation-deferral", "development-evaluation-deferral", "resource-lifecycle-only"],
        ):
            raise ValueError("resource_lifecycle_audit_chain_invalid")
        previous_manifest_path = _runtime_rooted(root, chain[-1]["previous_run_manifest"], allowed=("runs",))
        previous_run = previous_manifest_path.parent
        previous_manifest = _read_object(previous_manifest_path, "resource_lifecycle_previous_manifest_invalid")
        _validate_initial_baseline_reuse(root, chain[-1], previous_manifest, previous_run)
    return chain


def validate_deferred_development_selection(
    root: Path,
    run_path: Path,
    run_manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    selected_checkpoint: Path,
    *,
    system_test_manifest_sha256: str | None = None,
) -> dict[str, Any] | None:
    policy = run_manifest.get("development_execution")
    if policy is None:
        return None
    root = Path(root).resolve()
    run_path = _rooted(root, run_path, allowed=("runs",))
    expected_policy = _expected_development_execution(run_manifest)
    chain = validate_resource_lifecycle_audit_chain(root, run_manifest)
    if (
        not chain
        or chain[0].get("purpose") != "development-evaluation-deferral"
        or policy != expected_policy
        or report.get("development_execution") != expected_policy
        or report.get("status") != "completed"
        or report.get("cursor", {}).get("global_update") != 1438
        or report.get("early_stopped") is not False
    ):
        raise ValueError("deferred_development_report_contract_invalid")
    development_protocol_path = run_path / "development-protocol.json"
    selection_path = _runtime_rooted(root, run_path.parent / "selection.json", allowed=("runs",))
    index_path = run_path / "deferred-candidates.json"
    selection_record_path = run_path / "deferred-selection.json"
    best_pointer_path = run_path / "best-checkpoint.json"
    for path in (development_protocol_path, selection_path, index_path, selection_record_path, best_pointer_path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("deferred_development_selection_artifact_missing")
    development_protocol = _read_object(development_protocol_path, "deferred_development_protocol_invalid")
    selection_inputs = _read_object(selection_path, "deferred_development_selection_input_invalid")
    ids_by_split = selection_inputs.get("ids_by_split")
    full_ids = development_protocol.get("full_ids")
    fast_ids = development_protocol.get("fast_ids")
    if (
        not isinstance(full_ids, list)
        or any(not isinstance(identifier, str) or not identifier for identifier in full_ids)
        or not full_ids
        or len(set(full_ids)) != len(full_ids)
        or not isinstance(ids_by_split, Mapping)
        or ids_by_split.get("dev") != full_ids
        or not isinstance(fast_ids, list)
        or any(not isinstance(identifier, str) or not identifier for identifier in fast_ids)
        or len(fast_ids) != 128
        or len(set(fast_ids)) != len(fast_ids)
        or not set(fast_ids).issubset(full_ids)
        or development_protocol.get("fast_every_updates") != 64
        or development_protocol.get("full_every_updates") != 360
        or run_manifest.get("selection_sha256") != sha256_file(selection_path)
        or (
            system_test_manifest_sha256 is not None
            and run_manifest.get("system_test_manifest_sha256") != system_test_manifest_sha256
        )
    ):
        raise ValueError("deferred_development_selection_inputs_invalid")

    candidates = _read_object(index_path, "deferred_development_candidate_index_invalid")
    selection = _read_object(selection_record_path, "deferred_development_selection_invalid")
    updates = expected_policy["candidate_updates"]
    previous_manifest_path = _runtime_rooted(root, chain[0]["previous_run_manifest"], allowed=("runs",))
    previous_run = previous_manifest_path.parent
    previous_manifest = _read_object(previous_manifest_path, "deferred_development_previous_manifest_invalid")
    baseline_data = _read_object(previous_run / "baselines.json", "deferred_development_baseline_invalid")
    baseline_metrics = baseline_data.get("warm_start_development") or baseline_data.get("random_head_development")
    prior_candidates = None
    if previous_manifest.get("development_execution") is not None:
        state_binding = chain[0].get("development_state_reuse")
        expected_index_hash = state_binding.get("artifacts", {}).get("deferred-candidates.json") if isinstance(state_binding, Mapping) else None
        prior_index_path = previous_run / "deferred-candidates.json"
        if (
            not isinstance(expected_index_hash, str)
            or prior_index_path.is_symlink()
            or not prior_index_path.is_file()
            or sha256_file(prior_index_path) != expected_index_hash
        ):
            raise ValueError("deferred_development_prior_candidate_index_mismatch")
        prior_candidates = _read_object(prior_index_path, "deferred_development_prior_candidate_index_invalid")
    if (
        set(candidates) != {str(update) for update in updates}
        or set(selection) != {
            "schema_version", "candidate_updates", "best_update", "best_dev_soft_cross_entropy",
            "selected_checkpoint_manifest_sha256", "candidate_index_sha256",
            "selected_reload_verification", "duplicate_selected_development_inference",
        }
        or selection.get("schema_version") != "post-training-development-selection-v1"
        or selection.get("candidate_updates") != updates
        or selection.get("candidate_index_sha256") != sha256_file(index_path)
        or selection.get("selected_reload_verification")
        != "artifact checksums and exact tensor load; stored single-pass development score reused"
        or selection.get("duplicate_selected_development_inference") is not False
        or report.get("deferred_selection_sha256") != sha256_file(selection_record_path)
        or report.get("best_update") != selection.get("best_update")
        or report.get("best_dev_soft_cross_entropy") != selection.get("best_dev_soft_cross_entropy")
    ):
        raise ValueError("deferred_development_selection_binding_mismatch")

    family = run_path.parent.resolve()
    selected_entry = None
    selected_path = Path(selected_checkpoint).resolve()
    for update in updates:
        entry = candidates[str(update)]
        if not isinstance(entry, Mapping) or set(entry) != {"path", "checkpoint_manifest_sha256", "dev", "score_origin"}:
            raise ValueError("deferred_development_candidate_invalid")
        checkpoint = _rooted(root, entry["path"], allowed=("runs",))
        if checkpoint.parent.name != "checkpoints" or checkpoint.parent.parent.parent.resolve() != family:
            raise ValueError("deferred_development_candidate_outside_family")
        source_run = checkpoint.parent.parent
        source_index = 0 if source_run.name == "main" else _MAIN_RUN_DIRECTORY.fullmatch(source_run.name)
        active_index = 0 if run_path.name == "main" else _MAIN_RUN_DIRECTORY.fullmatch(run_path.name)
        source_number = int(source_index.group(1)) if hasattr(source_index, "group") else source_index
        active_number = int(active_index.group(1)) if hasattr(active_index, "group") else active_index
        if source_number is None or active_number is None or source_number > active_number:
            raise ValueError("deferred_development_candidate_run_order_invalid")
        if update == 0 and source_run.name != "main":
            raise ValueError("deferred_development_baseline_checkpoint_invalid")
        if update > 0 and source_run.resolve() != run_path.resolve():
            if prior_candidates is None or source_run.resolve() != previous_run.resolve():
                raise ValueError("deferred_development_checkpoint_not_in_active_or_audited_prior_run")
            prior_entry = prior_candidates.get(str(update))
            if (
                not isinstance(prior_entry, Mapping)
                or set(prior_entry) not in (
                    {"path", "checkpoint_manifest_sha256"},
                    {"path", "checkpoint_manifest_sha256", "dev", "score_origin"},
                )
                or entry.get("path") != prior_entry.get("path")
                or entry.get("checkpoint_manifest_sha256") != prior_entry.get("checkpoint_manifest_sha256")
                or (
                    "dev" in prior_entry
                    and dict(entry) != dict(prior_entry)
                )
            ):
                raise ValueError("deferred_development_reused_candidate_mismatch")
        checkpoint_record = _read_object(checkpoint / "checkpoint.json", "deferred_development_checkpoint_invalid")
        cursor = checkpoint_record.get("progress", {}).get("cursor", {})
        checkpoint_run_manifest = checkpoint_record.get("manifest")
        if (
            checkpoint.is_symlink()
            or sha256_file(checkpoint / "checkpoint.json") != entry.get("checkpoint_manifest_sha256")
            or cursor.get("global_update") != update
            or not isinstance(checkpoint_run_manifest, Mapping)
            or checkpoint_run_manifest.get("snapshot_manifest_sha256") != run_manifest.get("snapshot_manifest_sha256")
            or checkpoint_run_manifest.get("selection_sha256") != run_manifest.get("selection_sha256")
            or (system_test_manifest_sha256 is not None
                and checkpoint_run_manifest.get("system_test_manifest_sha256") != system_test_manifest_sha256)
        ):
            raise ValueError("deferred_development_checkpoint_binding_mismatch")
        metrics = entry.get("dev")
        if (
            not isinstance(metrics, Mapping)
            or type(metrics.get("questions")) is not int
            or metrics.get("questions") != len(full_ids)
            or type(metrics.get("soft_cross_entropy")) not in (int, float)
            or not math.isfinite(metrics.get("soft_cross_entropy", float("nan")))
            or entry.get("score_origin") != (
                "audited completed baseline" if update == 0 else "single post-training full-development pass"
            )
        ):
            raise ValueError("deferred_development_candidate_score_invalid")
        if update == 0 and (not isinstance(baseline_metrics, Mapping) or dict(metrics) != dict(baseline_metrics)):
            raise ValueError("deferred_development_baseline_score_mismatch")
        if checkpoint.resolve() == selected_path:
            selected_entry = entry
    pointer = _read_object(best_pointer_path, "deferred_development_best_pointer_invalid")
    selected_update = selection.get("best_update")
    selected_entry = candidates.get(str(selected_update))
    expected_best_update = min(
        updates,
        key=lambda update: (candidates[str(update)]["dev"]["soft_cross_entropy"], update),
    )
    if (
        selected_update not in updates
        or selected_update != expected_best_update
        or not isinstance(selected_entry, Mapping)
        or Path(report.get("best_checkpoint_relative", "")).is_absolute()
        or pointer.get("update") != selected_update
        or pointer.get("path") != report.get("best_checkpoint_relative")
        or pointer.get("loss") != selected_entry.get("dev", {}).get("soft_cross_entropy")
        or Path(os.path.relpath(selected_path, run_path)).as_posix() != report.get("best_checkpoint_relative")
        or selected_entry.get("path") is None
        or _rooted(root, selected_entry["path"], allowed=("runs",)).resolve() != selected_path
        or selected_entry.get("checkpoint_manifest_sha256") != selection.get("selected_checkpoint_manifest_sha256")
        or selected_entry.get("dev") != report.get("selected_checkpoint_development")
        or selected_entry.get("dev", {}).get("soft_cross_entropy") != selection.get("best_dev_soft_cross_entropy")
    ):
        raise ValueError("deferred_development_selected_checkpoint_mismatch")
    return {
        "development_execution": expected_policy,
        "deferred_selection_sha256": sha256_file(selection_record_path),
        "deferred_candidate_index_sha256": sha256_file(index_path),
        "selected_checkpoint_update": selected_update,
    }


def _jsonl(path: Path) -> Iterable[dict[str, Any]]:
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError:
        raise ValueError("metadata_jsonl_unavailable") from None
    with stream:
        for encoded in stream:
            if not encoded.strip():
                continue
            try:
                value = json.loads(encoded)
            except json.JSONDecodeError:
                raise ValueError("metadata_jsonl_invalid") from None
            if not isinstance(value, dict):
                raise ValueError("metadata_jsonl_record_invalid")
            yield value


def _snapshot_metadata(snapshot: Path) -> tuple[Path, dict[str, Any], dict[str, str]]:
    snapshot = snapshot.resolve()
    root = _project_root(snapshot)
    snapshot = _rooted(root, snapshot, allowed=SNAPSHOT_ROOT_PARTS)
    manifest_path = snapshot / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("snapshot_manifest_missing")
    manifest = _read_object(manifest_path, "snapshot_manifest_invalid")
    if manifest.get("schema_version") != "training-snapshot-v2" or not isinstance(manifest.get("version"), str):
        raise ValueError("snapshot_manifest_contract_invalid")
    declared = manifest.get("artifacts")
    if not isinstance(declared, dict) or not {"rows.jsonl", "splits.jsonl"}.issubset(declared):
        raise ValueError("snapshot_artifacts_missing")
    hashes: dict[str, str] = {}
    for name, item in declared.items():
        if not isinstance(name, str) or not isinstance(item, dict):
            raise ValueError("snapshot_artifact_manifest_invalid")
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("snapshot_artifact_path_invalid")
        artifact = _rooted(root, snapshot.joinpath(*relative.parts), allowed=SNAPSHOT_ROOT_PARTS)
        if not artifact.is_file() or not isinstance(item.get("sha256"), str) or sha256_file(artifact) != item["sha256"]:
            raise ValueError("snapshot_artifact_hash_mismatch")
        hashes[name] = item["sha256"]
    return root, manifest, hashes


def _audit_metadata(root: Path, snapshot: Path, manifest: Mapping[str, Any], artifact_hashes: Mapping[str, str],
                    audit_directory: Path, suite_lock: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Path], dict[str, set[str]]]:
    audit_directory = _rooted(root, audit_directory, allowed=EXTERNAL_ROOT_PARTS)
    paths = {
        "report": audit_directory / "overlap-report.json",
        "quarantine": audit_directory / "quarantine-training-ids.jsonl",
        "exact_overlay": audit_directory / "exact-match-exclusion-overlay.jsonl",
    }
    for path in paths.values():
        if path.is_symlink() or not path.is_file():
            raise ValueError("overlap_audit_artifact_missing")
    report = _read_object(paths["report"], "overlap_report_invalid")
    if report.get("schema_version") != "blind-overlap-audit-v1":
        raise ValueError("overlap_report_contract_invalid")
    expected_rows = artifact_hashes.get("rows.jsonl")
    expected_manifest = sha256_file(snapshot / "manifest.json")
    if (
        report.get("training_snapshot_version") != manifest.get("version")
        or report.get("training_snapshot_manifest_sha256") != expected_manifest
        or report.get("training_rows_sha256") != expected_rows
        or report.get("training_rows_checked") != manifest.get("artifacts", {}).get("rows.jsonl", {}).get("records")
        or report.get("semantic_overlap_checked") is not False
        or report.get("pretraining_contamination_ruled_out") is not False
        or type(report.get("near_candidate_pairs_truncated")) is not int
        or report["near_candidate_pairs_truncated"] != 0
    ):
        raise ValueError("overlap_report_snapshot_binding_invalid")
    files = suite_lock.get("files")
    if not isinstance(files, dict):
        raise ValueError("suite_lock_files_invalid")
    canonical_hash = files.get("canonical/bundle/items.jsonl")
    bundle_hash = files.get("bundle-manifest.json")
    if (
        not isinstance(canonical_hash, str)
        or report.get("benchmark_items_sha256") != canonical_hash
        or not isinstance(bundle_hash, str)
    ):
        raise ValueError("overlap_report_suite_binding_invalid")
    bundle_path = root / "data" / "external_evaluation" / "bundle-manifest.json"
    bundle = _read_object(bundle_path, "bundle_manifest_invalid")
    canonical_count = bundle.get("canonical_item_count")
    canonical_path = bundle.get("canonical_items_path")
    if (
        sha256_file(bundle_path) != bundle_hash
        or canonical_count != report.get("benchmark_rows_checked")
        or bundle.get("canonical_items_sha256") != canonical_hash
        or not isinstance(canonical_path, str)
        or canonical_path != "canonical/bundle/items.jsonl"
    ):
        raise ValueError("overlap_report_canonical_bundle_mismatch")
    if (
        sha256_file(paths["quarantine"]) != report.get("quarantine_training_ids_sha256")
        or sha256_file(paths["exact_overlay"]) != report.get("exact_match_exclusion_overlay_sha256")
    ):
        raise ValueError("overlap_audit_artifact_hash_mismatch")

    quarantined: dict[str, set[str]] = {}
    for record in _jsonl(paths["quarantine"]):
        identifier = record.get("training_id")
        kinds = record.get("overlap_types")
        if not isinstance(identifier, str) or not identifier or not isinstance(kinds, list) or not kinds:
            raise ValueError("quarantine_training_id_invalid")
        if any(kind not in {"exact_prompt_overlap", "near_lexical_overlap"} for kind in kinds):
            raise ValueError("quarantine_overlap_kind_invalid")
        quarantined[identifier] = set(kinds)
    exact_ids: set[str] = set()
    for record in _jsonl(paths["exact_overlay"]):
        identifier = record.get("training_id")
        if (
            not isinstance(identifier, str) or not identifier
            or record.get("exclude_from_future_training_snapshot") is not True
            or record.get("reason") != "exact_prompt_overlap"
        ):
            raise ValueError("exact_overlay_record_invalid")
        exact_ids.add(identifier)
    expected_exact = {identifier for identifier, kinds in quarantined.items() if "exact_prompt_overlap" in kinds}
    expected_near = {identifier for identifier, kinds in quarantined.items() if "near_lexical_overlap" in kinds and "exact_prompt_overlap" not in kinds}
    if (
        exact_ids != expected_exact
        or len(expected_exact) != report.get("exact_prompt_overlap_training_ids")
        or len(expected_near) != report.get("near_lexical_overlap_training_ids")
    ):
        raise ValueError("overlap_quarantine_count_mismatch")
    return report, paths, {"training_ids": set(quarantined), "exact_ids": exact_ids}


def _training_exclusions(snapshot: Path, overlap_ids: set[str]) -> dict[str, list[str]]:
    exclusions: dict[str, set[str]] = {
        "row_ids": set(overlap_ids), "problem_group_ids": set(), "snapshot_group_ids": set(),
        "family_ids": set(), "operation_ids": set(),
    }
    if not overlap_ids:
        return {kind: [] for kind in exclusions}
    found: set[str] = set()
    for record in _jsonl(snapshot / "splits.jsonl"):
        identifier = record.get("id")
        if identifier in overlap_ids:
            found.add(identifier)
            for field, kind in (("problem_group_id", "problem_group_ids"), ("snapshot_group_id", "snapshot_group_ids")):
                if isinstance(record.get(field), str) and record[field]:
                    exclusions[kind].add(record[field])
    for record in _jsonl(snapshot / "rows.jsonl"):
        identifier = record.get("id")
        if identifier in overlap_ids:
            for field, kind in (("family", "family_ids"), ("operation_id", "operation_ids")):
                if isinstance(record.get(field), str) and record[field]:
                    exclusions[kind].add(record[field])
    if found != overlap_ids:
        raise ValueError("quarantine_training_ids_not_in_snapshot")
    return {kind: sorted(values) for kind, values in exclusions.items()}


def _write_exclusive_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(dict(data), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError:
        raise FileExistsError("system_test_lock_is_immutable") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _write_or_validate_json(path: Path, data: Mapping[str, Any]) -> None:
    if path.is_symlink():
        raise ValueError("protocol_file_symlinked")
    if path.exists():
        if _read_object(path, "protocol_file_invalid") != dict(data):
            raise ValueError("existing_protocol_does_not_match_completed_development_run")
        return
    try:
        _write_exclusive_json(path, data)
    except FileExistsError:
        if _read_object(path, "protocol_file_invalid") != dict(data):
            raise ValueError("existing_protocol_does_not_match_completed_development_run") from None


def _verify_checkpoint_artifacts(directory: Path, files: Mapping[str, Any]) -> None:
    actual: set[str] = set()
    for artifact in directory.rglob("*"):
        if artifact.is_symlink():
            raise ValueError("selected_checkpoint_contains_symlink")
        if artifact.is_file() and artifact.name != "checkpoint.json":
            actual.add(artifact.relative_to(directory).as_posix())
    if actual != set(files):
        raise ValueError("selected_checkpoint_artifact_set_mismatch")
    for name, expected in files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not isinstance(expected, str):
            raise ValueError("selected_checkpoint_artifact_path_invalid")
        artifact = directory.joinpath(*relative.parts)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError("selected_checkpoint_artifact_hash_mismatch")


def _manifest_project_path(root: Path, value: Any, *, allowed: tuple[str, ...]) -> Path:
    """Resolve a host path or the runner's explicit /workspace alias safely."""
    if not isinstance(value, str) or not value:
        raise ValueError("checkpoint_reference_path_invalid")
    raw = Path(value)
    if ".." in raw.parts:
        raise ValueError("checkpoint_reference_path_invalid")
    workspace = Path("/workspace")
    if raw.is_absolute() and raw.is_relative_to(workspace):
        mapped = root.joinpath(*raw.relative_to(workspace).parts)
    else:
        mapped = raw if raw.is_absolute() else root / raw
    return _rooted(root, mapped, allowed=allowed)


def validate_compact_checkpoint_binding(
    root: Path,
    checkpoint: Path,
    checkpoint_data: Mapping[str, Any],
    run_manifest: Mapping[str, Any],
    *,
    verify_base_backbone: bool = False,
) -> dict[str, Any]:
    """Verify a last-block overlay against the exact full warm-start checkpoint."""
    root = Path(root).resolve()
    checkpoint = _rooted(root, checkpoint, allowed=("runs",))
    if checkpoint_data.get("backbone_state_kind") != "trainable-overlay-v1":
        raise ValueError("compact_checkpoint_state_kind_invalid")
    reference = checkpoint_data.get("base_checkpoint")
    names = checkpoint_data.get("backbone_parameter_names")
    manifest = checkpoint_data.get("manifest")
    warm_start = run_manifest.get("warm_start")
    if (
        not isinstance(reference, Mapping)
        or set(reference) != {"path", "manifest_sha256"}
        or not isinstance(reference.get("path"), str)
        or not isinstance(reference.get("manifest_sha256"), str)
        or not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or not isinstance(manifest, Mapping)
        or not isinstance(warm_start, Mapping)
    ):
        raise ValueError("compact_checkpoint_reference_invalid")

    base = _manifest_project_path(root, warm_start.get("checkpoint"), allowed=("runs",))
    relative_base = os.path.relpath(base, checkpoint).replace(os.sep, "/")
    reference_path = PurePosixPath(reference["path"])
    if (
        reference_path.is_absolute()
        or "\\" in reference["path"]
        or reference["path"] != relative_base
        or not base.is_dir()
        or base.is_symlink()
    ):
        raise ValueError("compact_checkpoint_base_path_mismatch")

    base_manifest_path = base / "checkpoint.json"
    if base_manifest_path.is_symlink() or not base_manifest_path.is_file():
        raise ValueError("compact_checkpoint_base_manifest_missing")
    base_manifest_sha256 = sha256_file(base_manifest_path)
    if (
        reference["manifest_sha256"] != base_manifest_sha256
        or warm_start.get("checkpoint_manifest_sha256") != base_manifest_sha256
    ):
        raise ValueError("compact_checkpoint_base_manifest_mismatch")
    base_data = _read_object(base_manifest_path, "compact_checkpoint_base_manifest_invalid")
    base_files = base_data.get("files")
    base_training_manifest = base_data.get("manifest")
    backbone_path = base / "backbone.pt"
    backbone_sha256 = base_files.get("backbone.pt") if isinstance(base_files, Mapping) else None
    head_path = base / "head.pt"
    head_sha256 = base_files.get("head.pt") if isinstance(base_files, Mapping) else None
    if (
        base_data.get("schema_version") != 1
        or base_data.get("include_backbone") is not True
        or base_data.get("backbone_state_kind", "full") != "full"
        or base_data.get("base_checkpoint") not in (None, {})
        or base_data.get("backbone_parameter_names") not in (None, [])
        or not isinstance(base_files, Mapping)
        or not isinstance(backbone_sha256, str)
        or backbone_path.is_symlink()
        or not backbone_path.is_file()
        or warm_start.get("backbone_sha256") != backbone_sha256
        or not isinstance(head_sha256, str)
        or head_path.is_symlink()
        or not head_path.is_file()
        or warm_start.get("head_sha256") != head_sha256
        or sha256_file(head_path) != head_sha256
        or not isinstance(base_training_manifest, Mapping)
    ):
        raise ValueError("compact_checkpoint_base_contract_invalid")
    if (
        base_training_manifest.get("arm") != "last-block"
        or base_training_manifest.get("repo") != run_manifest.get("repo")
        or base_training_manifest.get("revision") != run_manifest.get("revision")
        or base_training_manifest.get("head_architecture", "last-token-linear-v1")
        != manifest.get("head_architecture", "last-token-linear-v1")
        or base_training_manifest.get("trainable_backbone_parameter_names") != names
        or manifest.get("trainable_backbone_parameter_names") != names
    ):
        raise ValueError("compact_checkpoint_parameter_scope_mismatch")
    if verify_base_backbone and sha256_file(backbone_path) != backbone_sha256:
        raise ValueError("compact_checkpoint_base_backbone_mismatch")

    names_sha256 = hashlib.sha256(
        json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "path": _relative(root, base),
        "checkpoint_manifest_sha256": base_manifest_sha256,
        "backbone_sha256": backbone_sha256,
        "head_sha256": head_sha256,
        "backbone_parameter_names_sha256": names_sha256,
        "backbone_parameter_count": len(names),
        "backbone_state_kind": "full",
    }


def create_snapshot_bound_lock(
    snapshot: Path,
    audit_directory: Path,
    output_lock: Path | None = None,
    *,
    suite_lock_path: Path | None = None,
) -> dict[str, Any]:
    """Create an immutable system-test lock from a frozen snapshot and audit."""
    snapshot = Path(snapshot).resolve()
    root, manifest, artifacts = _snapshot_metadata(snapshot)
    external_root = root / "data" / "external_evaluation"
    suite_lock_path = Path(suite_lock_path) if suite_lock_path else root / OLD_SUITE_LOCK
    suite_lock_path = _rooted(root, suite_lock_path, allowed=EXTERNAL_ROOT_PARTS)
    suite_lock = _read_object(suite_lock_path, "suite_lock_invalid")
    if (
        suite_lock.get("schema_version") != "system-test-lock-v1"
        or suite_lock.get("status") != "locked"
        or suite_lock.get("independent_from_training_generator") is not True
    ):
        raise ValueError("suite_lock_not_independent_and_locked")
    suite_files = suite_lock.get("files")
    if not isinstance(suite_files, dict) or not suite_files:
        raise ValueError("suite_lock_files_invalid")
    locked_files: dict[str, str] = {}
    for name, expected in suite_files.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or not isinstance(expected, str):
            raise ValueError("suite_lock_artifact_path_invalid")
        if name.startswith("overlap-audit/"):
            continue
        artifact = _rooted(root, external_root.joinpath(*relative.parts), allowed=EXTERNAL_ROOT_PARTS)
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise ValueError("suite_lock_artifact_hash_mismatch")
        locked_files[name] = expected

    report, audit_paths, matches = _audit_metadata(root, snapshot, manifest, artifacts, Path(audit_directory), suite_lock)
    audit_relative = {}
    for key, path in audit_paths.items():
        external_relative = path.relative_to(external_root).as_posix()
        digest = sha256_file(path)
        locked_files[external_relative] = digest
        audit_relative[key] = _relative(root, path)
    manifest_sha = sha256_file(snapshot / "manifest.json")
    snapshot_binding = {
        "path": _relative(root, snapshot),
        "version": manifest["version"],
        "manifest_sha256": manifest_sha,
        "rows_sha256": artifacts["rows.jsonl"],
        "splits_sha256": artifacts["splits.jsonl"],
        "artifact_sha256": dict(sorted(artifacts.items())),
    }
    overlap_binding = {
        "report_path": audit_relative["report"],
        "report_sha256": sha256_file(audit_paths["report"]),
        "quarantine_training_ids_path": audit_relative["quarantine"],
        "quarantine_training_ids_sha256": sha256_file(audit_paths["quarantine"]),
        "exact_match_exclusion_overlay_path": audit_relative["exact_overlay"],
        "exact_match_exclusion_overlay_sha256": sha256_file(audit_paths["exact_overlay"]),
        "benchmark_items_sha256": report["benchmark_items_sha256"],
        "benchmark_rows_checked": report["benchmark_rows_checked"],
        "training_rows_checked": report["training_rows_checked"],
        "exact_prompt_overlap_test_rows": report["exact_prompt_overlap_test_rows"],
        "exact_prompt_overlap_training_ids": report["exact_prompt_overlap_training_ids"],
        "near_lexical_overlap_test_rows": report["near_lexical_overlap_test_rows"],
        "near_lexical_overlap_training_ids": report["near_lexical_overlap_training_ids"],
        "near_candidate_pairs_checked": report["near_candidate_pairs_checked"],
        "near_candidate_pairs_truncated": report["near_candidate_pairs_truncated"],
        "semantic_overlap_checked": False,
        "pretraining_contamination_ruled_out": False,
    }
    lock = {
        "schema_version": "system-test-lock-v1",
        "status": "locked",
        "independent_from_training_generator": True,
        "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "files": dict(sorted(locked_files.items())),
        "training_snapshot_manifest_sha256": manifest_sha,
        "snapshot_binding": snapshot_binding,
        "overlap_audit": overlap_binding,
        "training_exclusions": _training_exclusions(snapshot, matches["training_ids"]),
        "source_suite_lock_path": _relative(root, suite_lock_path),
        "source_suite_lock_sha256": sha256_file(suite_lock_path),
        "scope": "Snapshot-bound lock for development-selected training; existing external suite is immutable exploratory reuse. Exact and bounded lexical checks are aggregate-only; semantic overlap and pretrained-model exposure remain unknown.",
    }
    version_slug = manifest["version"]
    if not version_slug or Path(version_slug).name != version_slug or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in version_slug):
        raise ValueError("snapshot_version_invalid")
    output_lock = Path(output_lock) if output_lock else external_root / f"system-test-lock-{version_slug}.json"
    output_lock = _rooted(root, output_lock, allowed=EXTERNAL_ROOT_PARTS)
    _write_exclusive_json(output_lock, lock)
    validate_snapshot_bound_lock(lock, snapshot, output_lock, project_root=root)
    return lock


def validate_snapshot_bound_lock(
    lock: Mapping[str, Any], snapshot: Path, lock_path: Path, *, project_root: Path | None = None,
) -> bool:
    """Verify a snapshot/audit binding; return False for legacy locks."""
    binding = lock.get("snapshot_binding")
    audit = lock.get("overlap_audit")
    if binding is None and audit is None:
        return False
    if not isinstance(binding, Mapping) or not isinstance(audit, Mapping):
        raise ValueError("snapshot_bound_lock_contract_invalid")
    snapshot = Path(snapshot).resolve()
    root = Path(project_root).resolve() if project_root else _project_root(snapshot)
    expected_snapshot = _rooted(root, binding.get("path", ""), allowed=SNAPSHOT_ROOT_PARTS)
    if snapshot != expected_snapshot:
        raise ValueError("snapshot_bound_lock_path_mismatch")
    actual_root, manifest, artifacts = _snapshot_metadata(snapshot)
    manifest_hash = sha256_file(snapshot / "manifest.json")
    if (
        actual_root.resolve() != root.resolve()
        or binding.get("version") != manifest.get("version")
        or binding.get("manifest_sha256") != manifest_hash
        or lock.get("training_snapshot_manifest_sha256") != manifest_hash
        or binding.get("rows_sha256") != artifacts.get("rows.jsonl")
        or binding.get("splits_sha256") != artifacts.get("splits.jsonl")
        or binding.get("artifact_sha256") != dict(sorted(artifacts.items()))
    ):
        raise ValueError("snapshot_bound_lock_hash_mismatch")
    audit_paths = {
        "report": audit.get("report_path"),
        "quarantine": audit.get("quarantine_training_ids_path"),
        "exact_overlay": audit.get("exact_match_exclusion_overlay_path"),
    }
    expected_hashes = {
        "report": audit.get("report_sha256"),
        "quarantine": audit.get("quarantine_training_ids_sha256"),
        "exact_overlay": audit.get("exact_match_exclusion_overlay_sha256"),
    }
    files = lock.get("files")
    if not isinstance(files, Mapping):
        raise ValueError("snapshot_bound_lock_files_invalid")
    for key, value in audit_paths.items():
        if not isinstance(value, str):
            raise ValueError("snapshot_bound_lock_audit_path_invalid")
        path = _rooted(root, value, allowed=EXTERNAL_ROOT_PARTS)
        relative_external = path.relative_to(root / "data" / "external_evaluation").as_posix()
        if (
            not path.is_file()
            or sha256_file(path) != expected_hashes[key]
            or files.get(relative_external) != expected_hashes[key]
        ):
            raise ValueError("snapshot_bound_lock_audit_hash_mismatch")
    report = _read_object(_rooted(root, audit_paths["report"], allowed=EXTERNAL_ROOT_PARTS), "overlap_report_invalid")
    if (
        report.get("training_snapshot_manifest_sha256") != manifest_hash
        or report.get("training_rows_sha256") != artifacts.get("rows.jsonl")
        or report.get("benchmark_items_sha256") != audit.get("benchmark_items_sha256")
        or report.get("benchmark_rows_checked") != audit.get("benchmark_rows_checked")
        or report.get("near_candidate_pairs_truncated") != 0
        or report.get("exact_prompt_overlap_training_ids") != audit.get("exact_prompt_overlap_training_ids")
        or report.get("near_lexical_overlap_training_ids") != audit.get("near_lexical_overlap_training_ids")
    ):
        raise ValueError("snapshot_bound_lock_overlap_report_mismatch")
    if Path(lock_path).is_symlink() or not Path(lock_path).is_file():
        raise ValueError("snapshot_bound_lock_missing")
    return True


def validate_fast_development_selection(
    root: Path,
    fast_manifest_path: Path,
    selection_path: Path,
    *,
    snapshot: Path,
    lock_path: Path,
    verify_checkpoint_artifacts: bool = True,
    verify_base_backbone: bool = True,
) -> dict[str, Any]:
    """Verify a frozen small-development selection and its five pinned checkpoints."""
    root = Path(root).resolve()
    snapshot = _rooted(root, snapshot, allowed=SNAPSHOT_ROOT_PARTS)
    lock_path = _rooted(root, lock_path, allowed=EXTERNAL_ROOT_PARTS)
    manifest_path = _rooted(root, fast_manifest_path, allowed=("runs",))
    selection_path = _rooted(root, selection_path, allowed=("runs",))
    if (manifest_path.is_symlink() or selection_path.is_symlink()
            or not manifest_path.is_file() or not selection_path.is_file()
            or manifest_path.parent != selection_path.parent):
        raise ValueError("fast_selection_artifact_paths_invalid")

    fast = _read_object(manifest_path, "fast_selection_manifest_invalid")
    selection = _read_object(selection_path, "fast_selection_record_invalid")
    fast_sha = sha256_file(manifest_path)
    snapshot_sha = sha256_file(snapshot / "manifest.json")
    lock_sha = sha256_file(lock_path)
    expected_selection_fields = {
        "checkpoint_path", "checkpoint_manifest_sha256", "update",
        "selection_manifest_sha256", "selection_metrics_path",
        "selected_checkpoint_reload_verified",
    }
    if (
        fast.get("schema_version") != "fast-evaluation-amendment-v1"
        or fast.get("snapshot") != _relative(root, snapshot)
        or fast.get("snapshot_manifest_sha256") != snapshot_sha
        or fast.get("splits_sha256") != _snapshot_metadata(snapshot)[2].get("splits.jsonl")
        or fast.get("candidate_updates") != FAST_SELECTION_UPDATES
        or not isinstance(fast.get("profiles"), Mapping)
        or not isinstance(fast["profiles"].get("dev"), Mapping)
        or fast["profiles"]["dev"].get("questions") != FAST_SELECTION_QUESTION_COUNT
        or fast.get("runtime_environment") != {"HSA_CU_MASK": "0:0-23"}
        or set(selection) != expected_selection_fields
        or selection.get("selection_manifest_sha256") != fast_sha
        or selection.get("selected_checkpoint_reload_verified") is not True
        or type(selection.get("update")) is not int
    ):
        raise ValueError("fast_selection_manifest_contract_invalid")

    original_protocol_raw = fast.get("original_protocol_path")
    if not isinstance(original_protocol_raw, str):
        raise ValueError("fast_selection_original_protocol_path_invalid")
    original_protocol_path = _rooted(root, original_protocol_raw, allowed=("runs",))
    original_protocol = _read_object(original_protocol_path, "fast_selection_original_protocol_invalid")
    original_snapshot_raw = original_protocol.get("snapshot")
    original_lock_raw = original_protocol.get("system_test_lock") or original_protocol.get("system_test_lock_path")
    if not isinstance(original_snapshot_raw, str) or not isinstance(original_lock_raw, str):
        raise ValueError("fast_selection_original_protocol_binding_invalid")
    original_snapshot_path = _rooted(root, original_snapshot_raw, allowed=SNAPSHOT_ROOT_PARTS)
    original_lock_path = _rooted(root, original_lock_raw, allowed=EXTERNAL_ROOT_PARTS)
    fast_lock_raw = fast.get("system_test_lock") or fast.get("system_test_lock_path")
    fast_lock_sha = fast.get("system_test_manifest_sha256") or fast.get("system_test_lock_sha256")
    if (
        original_protocol_path.is_symlink()
        or fast.get("original_protocol_sha256") != sha256_file(original_protocol_path)
        or original_snapshot_path.resolve() != snapshot.resolve()
        or original_protocol.get("snapshot_manifest_sha256") != snapshot_sha
        or original_lock_path.resolve() != lock_path.resolve()
        or (original_protocol.get("system_test_manifest_sha256") or original_protocol.get("system_test_lock_sha256")) != lock_sha
        or fast_lock_raw != _relative(root, lock_path)
        or fast_lock_sha != lock_sha
    ):
        raise ValueError("fast_selection_original_protocol_binding_invalid")

    code_hashes = fast.get("code_hashes")
    if not isinstance(code_hashes, Mapping) or not code_hashes:
        raise ValueError("fast_selection_code_hashes_invalid")
    for relative, expected in code_hashes.items():
        code_path = _rooted(root, relative, allowed=None)
        if code_path.is_symlink() or not code_path.is_file() or sha256_file(code_path) != expected:
            raise ValueError("fast_selection_code_hash_mismatch")

    candidate_index_raw = fast.get("candidate_index_path")
    if not isinstance(candidate_index_raw, str):
        raise ValueError("fast_selection_candidate_index_path_invalid")
    candidate_index_path = _rooted(root, candidate_index_raw, allowed=("runs",))
    if (candidate_index_path.is_symlink() or not candidate_index_path.is_file()
            or sha256_file(candidate_index_path) != fast.get("candidate_index_sha256")):
        raise ValueError("fast_selection_candidate_index_hash_mismatch")
    candidate_index = _read_object(candidate_index_path, "fast_selection_candidate_index_invalid")
    candidates = fast.get("candidates")
    metrics_path_raw = selection.get("selection_metrics_path")
    if not isinstance(metrics_path_raw, str):
        raise ValueError("fast_selection_metrics_path_invalid")
    metrics_path = _rooted(root, metrics_path_raw, allowed=("runs",))
    if metrics_path.parent != manifest_path.parent or metrics_path.is_symlink() or not metrics_path.is_file():
        raise ValueError("fast_selection_metrics_path_invalid")
    metrics = _read_object(metrics_path, "fast_selection_metrics_invalid")
    metrics_sha = sha256_file(metrics_path)
    if (
        not isinstance(candidates, Mapping)
        or set(candidates) != {str(update) for update in FAST_SELECTION_UPDATES}
        or set(candidate_index) != set(candidates)
        or metrics.get("manifest_sha256") != fast_sha
        or metrics.get("selected_update") != selection.get("update")
        or not isinstance(metrics.get("candidates"), Mapping)
        or set(metrics["candidates"]) != set(candidates)
    ):
        raise ValueError("fast_selection_candidate_set_invalid")

    scores: dict[int, float] = {}
    checkpoint_records: dict[int, dict[str, Any]] = {}
    active_run_path = candidate_index_path.parent
    active_manifest_path = active_run_path / "manifest.json"
    if active_manifest_path.is_symlink() or not active_manifest_path.is_file():
        raise ValueError("fast_selection_active_run_manifest_missing")
    active_manifest = _read_object(active_manifest_path, "fast_selection_active_run_manifest_invalid")
    if (
        active_manifest.get("snapshot_manifest_sha256") != snapshot_sha
        or active_manifest.get("system_test_manifest_sha256") != lock_sha
        or active_manifest.get("arm") != "last-block"
        or _MAIN_RUN_DIRECTORY.fullmatch(active_run_path.name) is None
    ):
        raise ValueError("fast_selection_active_run_manifest_mismatch")
    family = active_run_path.parent.resolve()
    if manifest_path.parent.parent.resolve() != family:
        raise ValueError("fast_selection_outside_candidate_family")
    validate_resource_lifecycle_audit_chain(root, active_manifest)

    invariant_fields = (
        "schema_version", "arm", "model", "repo", "revision", "snapshot_manifest_sha256",
        "selection_sha256", "system_test_manifest_sha256", "seed", "epochs", "device", "threads",
        "sampler", "microbatch_questions", "accumulation_questions", "max_batch_tokens", "bucket_questions",
        "formatter", "pooling", "capacity", "max_tokens", "attention_backend", "parameters_dtype",
        "compute_autocast", "optimizer", "lr_schedule", "warmup_fraction", "scheduler_total_updates",
        "max_updates", "head_learning_rate", "backbone_learning_rate", "fast_dev_questions", "full_dev_every",
        "reserve_gib", "gpu_allocation_cap_gib", "torch", "transformers", "cuda_version", "hip_version",
        "device_name", "initial_head_state_sha256", "head_architecture", "head_attention_size",
        "option_span_metadata", "backbone_training_scope", "backbone_trainable", "backbone_parameter_count",
        "trainable_backbone_parameter_names", "trainable_backbone_parameter_count", "gradient_checkpointing",
        "gradient_buffers", "checkpoint_backbone_state", "selection_metric", "target_policy",
        "early_stop_patience", "early_stop_min_delta", "early_stop_min_epochs", "choice_order_policy",
        "order_consistency_questions", "order_consistency_sampling", "head_parameter_count", "attention_head_sha256",
        "loss_sha256", "resource_lifecycle",
    )
    selected_record: dict[str, Any] | None = None
    for update in FAST_SELECTION_UPDATES:
        key = str(update)
        candidate = candidates[key]
        indexed = candidate_index[key]
        score_record = metrics["candidates"][key]
        if (
            not isinstance(candidate, Mapping)
            or set(candidate) != {"path", "checkpoint_manifest_sha256"}
            or not isinstance(indexed, Mapping)
            or indexed.get("path") != candidate.get("path")
            or indexed.get("checkpoint_manifest_sha256") != candidate.get("checkpoint_manifest_sha256")
            or not isinstance(score_record, Mapping)
            or score_record.get("update") != update
            or score_record.get("manifest_sha256") != fast_sha
            or score_record.get("checkpoint_manifest_sha256") != candidate.get("checkpoint_manifest_sha256")
            or not isinstance(score_record.get("metrics"), Mapping)
        ):
            raise ValueError("fast_selection_candidate_binding_invalid")
        score = score_record["metrics"].get("soft_cross_entropy")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError("fast_selection_score_invalid")
        scores[update] = float(score)

        checkpoint = _rooted(root, candidate["path"], allowed=("runs",))
        if (checkpoint.parent.name != "checkpoints"
                or _MAIN_RUN_DIRECTORY.fullmatch(checkpoint.parent.parent.name) is None
                or checkpoint.parent.parent.parent.resolve() != family
                or not checkpoint.is_dir()):
            raise ValueError("fast_selection_candidate_outside_family")
        source_manifest_path = checkpoint.parent.parent / "manifest.json"
        if source_manifest_path.is_symlink() or not source_manifest_path.is_file():
            raise ValueError("fast_selection_source_manifest_missing")
        source_manifest = _read_object(source_manifest_path, "fast_selection_source_manifest_invalid")
        if any(source_manifest.get(field) != active_manifest.get(field) for field in invariant_fields):
            raise ValueError("fast_selection_candidate_run_mismatch")
        validate_resource_lifecycle_audit_chain(root, source_manifest)
        checkpoint_manifest_path = checkpoint / "checkpoint.json"
        if checkpoint_manifest_path.is_symlink() or not checkpoint_manifest_path.is_file():
            raise ValueError("fast_selection_checkpoint_manifest_missing")
        checkpoint_sha = sha256_file(checkpoint_manifest_path)
        checkpoint_data = _read_object(checkpoint_manifest_path, "fast_selection_checkpoint_manifest_invalid")
        checkpoint_copy = checkpoint_data.get("manifest")
        if (
            checkpoint_sha != candidate["checkpoint_manifest_sha256"]
            or checkpoint_data.get("schema_version") != 1
            or checkpoint_data.get("include_backbone") is not True
            or checkpoint_data.get("progress", {}).get("cursor", {}).get("global_update") != update
            or not isinstance(checkpoint_copy, Mapping)
            or checkpoint_copy.get("arm") != "last-block"
            or checkpoint_copy.get("snapshot_manifest_sha256") != snapshot_sha
            or checkpoint_copy.get("system_test_manifest_sha256") != lock_sha
            or checkpoint_copy.get("repo") != active_manifest.get("repo")
            or checkpoint_copy.get("revision") != active_manifest.get("revision")
            or checkpoint_copy.get("formatter") != active_manifest.get("formatter")
            or checkpoint_copy.get("max_tokens") != active_manifest.get("max_tokens")
        ):
            raise ValueError("fast_selection_checkpoint_binding_invalid")
        if verify_checkpoint_artifacts:
            files = checkpoint_data.get("files")
            if not isinstance(files, Mapping):
                raise ValueError("fast_selection_checkpoint_files_invalid")
            _verify_checkpoint_artifacts(checkpoint, files)
        if checkpoint_data.get("backbone_state_kind", "full") == "trainable-overlay-v1":
            base_binding = validate_compact_checkpoint_binding(
                root, checkpoint, checkpoint_data, source_manifest,
                verify_base_backbone=verify_base_backbone and update == selection.get("update"),
            )
        else:
            if checkpoint_data.get("base_checkpoint") not in (None, {}) or checkpoint_data.get("backbone_parameter_names") not in (None, []):
                raise ValueError("fast_selection_full_checkpoint_storage_invalid")
            base_binding = None
        checkpoint_records[update] = {
            "path": _relative(root, checkpoint),
            "manifest_sha256": checkpoint_sha,
            "manifest": checkpoint_data,
            "source_run_manifest_path": _relative(root, source_manifest_path),
            "source_run_manifest_sha256": sha256_file(source_manifest_path),
            "source_run_manifest": source_manifest,
            "base_checkpoint_binding": base_binding,
        }
        if update == selection.get("update"):
            selected_record = checkpoint_records[update]

    winner = min(FAST_SELECTION_UPDATES, key=lambda update: (scores[update], update))
    selected = checkpoint_records.get(selection.get("update"))
    if (
        winner != selection.get("update")
        or selected_record is None
        or selected is None
        or selection.get("checkpoint_path") != selected["path"]
        or selection.get("checkpoint_manifest_sha256") != selected["manifest_sha256"]
        or selection.get("selection_metrics_path") != _relative(root, metrics_path)
    ):
        raise ValueError("fast_selection_winner_binding_invalid")

    return {
        "manifest_path": _relative(root, manifest_path),
        "manifest_sha256": fast_sha,
        "selection_path": _relative(root, selection_path),
        "selection_sha256": sha256_file(selection_path),
        "metrics_path": _relative(root, metrics_path),
        "metrics_sha256": metrics_sha,
        "candidate_index_path": _relative(root, candidate_index_path),
        "candidate_index_sha256": sha256_file(candidate_index_path),
        "candidate_updates": list(FAST_SELECTION_UPDATES),
        "selected_update": selection["update"],
        "selected_checkpoint_path": selected["path"],
        "checkpoint_manifest_sha256": selected["manifest_sha256"],
        "source_run_manifest_path": selected["source_run_manifest_path"],
        "source_run_manifest_sha256": selected["source_run_manifest_sha256"],
        "source_run_manifest": selected["source_run_manifest"],
        "checkpoint_manifest": selected["manifest"],
        "base_checkpoint_binding": selected["base_checkpoint_binding"],
        "adapter_source_run_manifest_path": _relative(root, active_manifest_path),
        "adapter_source_run_manifest": active_manifest,
        "scores": scores,
    }


def bind_fast_development_protocol(
    snapshot: Path,
    lock_path: Path,
    fast_manifest_path: Path,
    selection_path: Path,
    output_protocol: Path | None = None,
    template_path: Path | None = None,
) -> dict[str, Any]:
    """Bind the frozen fast-development winner to the unchanged full blind suite."""
    snapshot = Path(snapshot).resolve()
    root, snapshot_manifest, artifacts = _snapshot_metadata(snapshot)
    lock_path = _rooted(root, lock_path, allowed=EXTERNAL_ROOT_PARTS)
    lock = _read_object(lock_path, "system_test_lock_invalid")
    validate_snapshot_bound_lock(lock, snapshot, lock_path, project_root=root)
    evidence = validate_fast_development_selection(
        root, fast_manifest_path, selection_path, snapshot=snapshot, lock_path=lock_path,
    )
    source_manifest = evidence["source_run_manifest"]
    checkpoint_data = evidence["checkpoint_manifest"]
    if snapshot_manifest["version"].endswith("-v4") and checkpoint_data.get("backbone_state_kind") != "trainable-overlay-v1":
        raise ValueError("v4_checkpoint_must_use_pinned_trainable_overlay")

    template_path = Path(template_path) if template_path else root / DEFAULT_PROTOCOL_TEMPLATE
    template_path = _rooted(root, template_path)
    protocol = _read_object(template_path, "protocol_template_invalid")
    if protocol.get("schema_version") != "blind-external-evaluation-protocol-v1" or protocol.get("candidates") != ["last-block"]:
        raise ValueError("protocol_template_contract_invalid")
    protocol_model = (protocol.get("model_key"), protocol.get("model_id"), protocol.get("model_revision"))
    checkpoint_manifest = checkpoint_data
    training_max_tokens = source_manifest.get("max_tokens")
    evaluation_input_limit = protocol.get("inference", {}).get("input_limit_tokens")
    if (
        protocol_model != (source_manifest.get("model"), source_manifest.get("repo"), source_manifest.get("revision"))
        or source_manifest.get("formatter") != protocol.get("prompt_format", {}).get("id")
            or source_manifest.get("pooling") != protocol.get("inference", {}).get("pooling")
            or source_manifest.get("capacity") != protocol.get("inference", {}).get("choice_head_capacity")
            or source_manifest.get("attention_backend") != protocol.get("inference", {}).get("attention_backend")
        or source_manifest.get("parameters_dtype") != protocol.get("inference", {}).get("parameters_dtype")
        or source_manifest.get("compute_autocast") != protocol.get("inference", {}).get("backbone_autocast")
        or type(training_max_tokens) is not int
        or type(evaluation_input_limit) is not int
        or evaluation_input_limit <= 0
        or evaluation_input_limit > training_max_tokens
        or checkpoint_manifest.get("manifest", {}).get("max_tokens") != training_max_tokens
        or checkpoint_manifest.get("manifest", {}).get("arm") != "last-block"
        or checkpoint_manifest.get("manifest", {}).get("repo") != protocol.get("model_id")
        or checkpoint_manifest.get("manifest", {}).get("revision") != protocol.get("model_revision")
        or checkpoint_manifest.get("manifest", {}).get("formatter") != protocol.get("prompt_format", {}).get("id")
    ):
        raise ValueError("development_run_protocol_mismatch")

    # The adapter manifest gives the existing blind runner its normal model/runtime
    # metadata. It deliberately omits the full-development policy so this fast,
    # separately verified selection cannot be mistaken for a completed full-dev run.
    adapter_run = Path(output_protocol).parent / "blind-suite-adapter" if output_protocol else Path(selection_path).parent / "blind-protocol.json"
    if adapter_run.name == "blind-protocol.json":
        adapter_run = adapter_run.parent / "blind-suite-adapter"
    adapter_run = _rooted(root, adapter_run, allowed=("runs",))
    adapter_run.mkdir(parents=True, exist_ok=True)
    active_manifest = {
        key: value for key, value in source_manifest.items()
        if key not in {"development_execution", "development_state_reuse", "resource_lifecycle_audit"}
    }
    active_manifest["status"] = "completed"
    active_manifest["fast_selection_adapter"] = {
        "schema_version": "fast-selection-run-adapter-v1",
        "fast_manifest_path": evidence["manifest_path"],
        "fast_manifest_sha256": evidence["manifest_sha256"],
        "selection_path": evidence["selection_path"],
        "selection_sha256": evidence["selection_sha256"],
    }
    active_manifest_path = adapter_run / "manifest.json"
    _write_or_validate_json(active_manifest_path, active_manifest)

    selected_path = root / evidence["selected_checkpoint_path"]
    relative_checkpoint = os.path.relpath(selected_path, adapter_run).replace(os.sep, "/")
    binding = {
        "kind": FAST_SELECTION_BINDING_KIND,
        "report_path": evidence["selection_path"],
        "report_sha256": evidence["selection_sha256"],
        "best_checkpoint_relative": relative_checkpoint,
        "selected_checkpoint_path": evidence["selected_checkpoint_path"],
        "checkpoint_manifest_sha256": evidence["checkpoint_manifest_sha256"],
        "run_manifest_sha256": sha256_file(active_manifest_path),
        "active_run_manifest_path": _relative(root, active_manifest_path),
        "checkpoint_source_run_manifest_path": evidence["source_run_manifest_path"],
        "checkpoint_source_run_manifest_sha256": evidence["source_run_manifest_sha256"],
        "fast_manifest_path": evidence["manifest_path"],
        "fast_manifest_sha256": evidence["manifest_sha256"],
        "fast_selection_path": evidence["selection_path"],
        "fast_selection_sha256": evidence["selection_sha256"],
        "fast_metrics_path": evidence["metrics_path"],
        "fast_metrics_sha256": evidence["metrics_sha256"],
        "fast_candidate_index_path": evidence["candidate_index_path"],
        "fast_candidate_index_sha256": evidence["candidate_index_sha256"],
        "selected_checkpoint_update": evidence["selected_update"],
        "training_max_tokens": training_max_tokens,
        "evaluation_input_limit_tokens": evaluation_input_limit,
    }
    if evidence["base_checkpoint_binding"] is not None:
        binding["base_checkpoint_binding"] = evidence["base_checkpoint_binding"]

    version_suffix = snapshot_manifest["version"].rsplit("-", 1)[-1]
    if not version_suffix.startswith("v") or not version_suffix[1:].isdigit():
        raise ValueError("snapshot_version_invalid")
    audit = lock["overlap_audit"]
    protocol.update({
        "protocol_id": PROTOCOL_ID_PREFIX + version_suffix,
        "candidates": ["last-block"],
        "run_paths": {"last-block": _relative(root, adapter_run)},
        "system_test_lock_path": _relative(root, lock_path),
        "system_test_lock_sha256": sha256_file(lock_path),
        "training_snapshot_path": lock["snapshot_binding"]["path"],
        "training_snapshot_manifest_sha256": sha256_file(snapshot / "manifest.json"),
        "training_snapshot_rows_sha256": artifacts["rows.jsonl"],
        "training_snapshot_splits_sha256": artifacts["splits.jsonl"],
        "overlap_audit_report_sha256": audit["report_sha256"],
        "checkpoint_selection": (
            "Freeze the selected checkpoint from the pinned 128-question shared development subset and all five "
            "candidate hashes before external inference; no external result is used for selection or calibration."
        ),
        "report_gate": "Write aggregate metrics only after the single locked candidate completes the exact canonical item set.",
        "selection_binding": binding,
        "contamination_limit": (
            f"The bound exact-prompt and 3-word-shingle lexical audit found "
            f"{audit['exact_prompt_overlap_test_rows']} exact and {audit['near_lexical_overlap_test_rows']} near-overlap suite rows; "
            "candidate pairs were not truncated. Semantic overlap and pretrained-model exposure remain unknown."
        ),
        "suite_reuse": {
            **protocol.get("suite_reuse", {}),
            "canonical_items_sha256": protocol["canonical_items_sha256"],
            "training_max_tokens": training_max_tokens,
            "evaluation_input_limit_tokens": evaluation_input_limit,
            "input_limit_note": (
                "The evaluator keeps the locked suite's existing input cap; it does not increase the limit to the "
                "larger token budget used during training."
            ),
            "comparison_note": (
                "Exploratory reuse of the same previously observed 17,874-item suite. The checkpoint was selected "
                "only on the frozen 128-question internal development subset; results are descriptive, not a virgin benchmark result."
            ),
            "previous_protocol_sha256": sha256_file(template_path),
            "status": "reused_suite_comparison_not_virgin",
        },
    })
    protocol.pop("exclusion_rebind", None)
    scoring = dict(protocol.get("scoring", {}))
    scoring.pop("publish_aggregate_only_after_both_candidates_complete", None)
    scoring["publish_aggregate_only_after_all_locked_candidates_complete"] = True
    scoring["no_temperature_scaling_or_calibration"] = True
    protocol["scoring"] = scoring
    output_protocol = Path(output_protocol) if output_protocol else Path(selection_path).parent / "blind-protocol.json"
    output_protocol = _rooted(root, output_protocol, allowed=("runs",))
    _write_or_validate_json(output_protocol, protocol)
    return protocol


def bind_completed_development_protocol(
    snapshot: Path,
    lock_path: Path,
    resolved_main_output: Path,
    output_protocol: Path | None = None,
    template_path: Path | None = None,
) -> dict[str, Any]:
    """Bind a completed, reload-verified development selection to the locked suite."""
    snapshot = Path(snapshot).resolve()
    root, manifest, artifacts = _snapshot_metadata(snapshot)
    lock_path = _rooted(root, lock_path, allowed=EXTERNAL_ROOT_PARTS)
    lock = _read_object(lock_path, "system_test_lock_invalid")
    validate_snapshot_bound_lock(lock, snapshot, lock_path, project_root=root)
    run_path = _rooted(root, resolved_main_output, allowed=("runs",))
    report_path = run_path / "report.json"
    run_manifest_path = run_path / "manifest.json"
    report = _read_object(report_path, "development_report_invalid")
    run_manifest = _read_object(run_manifest_path, "development_run_manifest_invalid")
    if (
        report.get("status") != "completed"
        or report.get("selected_checkpoint_reload_verified") is not True
        or report.get("arm") != "last-block"
        or run_manifest.get("arm") != "last-block"
        or run_manifest.get("snapshot_manifest_sha256") != sha256_file(snapshot / "manifest.json")
        or run_manifest.get("system_test_manifest_sha256") != sha256_file(lock_path)
        or run_manifest.get("device") != "cuda"
        or run_manifest.get("head_architecture", "last-token-linear-v1") != "last-token-linear-v1"
    ):
        raise ValueError("completed_development_run_not_bound_to_lock")
    checkpoint_relative = report.get("best_checkpoint_relative")
    if not isinstance(checkpoint_relative, str):
        raise ValueError("selected_checkpoint_path_missing")
    checkpoint, source_run = _selected_main_checkpoint(root, run_path, checkpoint_relative)
    source_manifest_path = _rooted(root, source_run / "manifest.json", allowed=("runs",))
    if source_manifest_path.is_symlink() or not source_manifest_path.is_file():
        raise ValueError("selected_checkpoint_source_manifest_missing")
    source_run_manifest = _read_object(source_manifest_path, "selected_checkpoint_source_manifest_invalid")
    invariant_fields = (
        "schema_version", "arm", "model", "repo", "revision", "snapshot_manifest_sha256",
        "selection_sha256", "system_test_manifest_sha256", "seed", "epochs", "device", "threads",
        "sampler", "microbatch_questions", "accumulation_questions", "max_batch_tokens", "bucket_questions",
        "formatter", "pooling", "capacity", "max_tokens", "attention_backend", "parameters_dtype",
        "compute_autocast", "optimizer", "lr_schedule", "warmup_fraction", "scheduler_total_updates",
        "max_updates", "head_learning_rate", "backbone_learning_rate", "fast_dev_questions", "full_dev_every",
        "reserve_gib", "gpu_allocation_cap_gib", "torch", "transformers", "cuda_version", "hip_version",
        "device_name", "initial_head_state_sha256", "head_architecture", "head_attention_size",
        "option_span_metadata", "backbone_training_scope", "backbone_trainable", "backbone_parameter_count",
        "trainable_backbone_parameter_names", "trainable_backbone_parameter_count", "gradient_checkpointing",
        "gradient_buffers", "checkpoint_backbone_state", "selection_metric", "target_policy",
        "early_stop_patience", "early_stop_min_delta", "early_stop_min_epochs", "choice_order_policy",
        "order_consistency_questions", "order_consistency_sampling", "head_parameter_count", "attention_head_sha256",
        "loss_sha256", "resource_lifecycle",
    )
    if any(source_run_manifest.get(field) != run_manifest.get(field) for field in invariant_fields):
        raise ValueError("selected_checkpoint_source_run_mismatch")
    if source_run_manifest.get("warm_start") != run_manifest.get("warm_start"):
        raise ValueError("selected_checkpoint_warm_start_mismatch")
    checkpoint_manifest = checkpoint / "checkpoint.json"
    checkpoint_data = _read_object(checkpoint_manifest, "selected_checkpoint_manifest_invalid")
    checkpoint_state_kind = checkpoint_data.get("backbone_state_kind", "full")
    if (
        not checkpoint.is_dir()
        or checkpoint_data.get("include_backbone") is not True
        or not isinstance(checkpoint_data.get("files"), dict)
        or not {"backbone.pt", "head.pt"}.issubset(checkpoint_data["files"])
        or checkpoint_state_kind not in {"full", "trainable-overlay-v1"}
        or checkpoint_data.get("manifest", {}).get("snapshot_manifest_sha256") != sha256_file(snapshot / "manifest.json")
        or checkpoint_data.get("manifest", {}).get("system_test_manifest_sha256") != sha256_file(lock_path)
    ):
        raise ValueError("selected_checkpoint_not_snapshot_bound")
    if manifest["version"].endswith("-v4") and checkpoint_state_kind != "trainable-overlay-v1":
        raise ValueError("v4_checkpoint_must_use_pinned_trainable_overlay")
    if checkpoint_state_kind == "trainable-overlay-v1":
        base_checkpoint_binding = validate_compact_checkpoint_binding(
            root, checkpoint, checkpoint_data, source_run_manifest, verify_base_backbone=True,
        )
    else:
        if checkpoint_data.get("base_checkpoint") not in (None, {}) or checkpoint_data.get("backbone_parameter_names") not in (None, []):
            raise ValueError("full_checkpoint_must_not_reference_a_delta_base")
        base_checkpoint_binding = None
    validate_resource_lifecycle_audit_chain(root, run_manifest)
    deferred_binding = validate_deferred_development_selection(
        root,
        run_path,
        run_manifest,
        report,
        checkpoint,
        system_test_manifest_sha256=sha256_file(lock_path),
    )
    template_path = Path(template_path) if template_path else root / DEFAULT_PROTOCOL_TEMPLATE
    template_path = _rooted(root, template_path)
    protocol = _read_object(template_path, "protocol_template_invalid")
    if protocol.get("schema_version") != "blind-external-evaluation-protocol-v1" or protocol.get("candidates") != ["last-block"]:
        raise ValueError("protocol_template_contract_invalid")
    protocol_model = (protocol.get("model_key"), protocol.get("model_id"), protocol.get("model_revision"))
    if (
        protocol_model != (run_manifest.get("model"), run_manifest.get("repo"), run_manifest.get("revision"))
        or run_manifest.get("formatter") != protocol.get("prompt_format", {}).get("id")
        or run_manifest.get("pooling") != protocol.get("inference", {}).get("pooling")
        or run_manifest.get("capacity") != protocol.get("inference", {}).get("choice_head_capacity")
        or run_manifest.get("max_tokens") != protocol.get("inference", {}).get("input_limit_tokens")
        or run_manifest.get("attention_backend") != protocol.get("inference", {}).get("attention_backend")
        or run_manifest.get("parameters_dtype") != protocol.get("inference", {}).get("parameters_dtype")
        or run_manifest.get("compute_autocast") != protocol.get("inference", {}).get("backbone_autocast")
        or checkpoint_data.get("manifest", {}).get("arm") != "last-block"
        or checkpoint_data.get("manifest", {}).get("repo") != protocol.get("model_id")
        or checkpoint_data.get("manifest", {}).get("revision") != protocol.get("model_revision")
        or checkpoint_data.get("manifest", {}).get("formatter") != protocol.get("prompt_format", {}).get("id")
        or checkpoint_data.get("manifest", {}).get("max_tokens") != protocol.get("inference", {}).get("input_limit_tokens")
    ):
        raise ValueError("development_run_protocol_mismatch")
    _verify_checkpoint_artifacts(checkpoint, checkpoint_data["files"])
    lock_relative = _relative(root, lock_path)
    run_relative = _relative(root, run_path)
    report_relative = _relative(root, report_path)
    checkpoint_relative_root = _relative(root, checkpoint)
    source_manifest_relative = _relative(root, source_manifest_path)
    source_manifest_hash = sha256_file(source_manifest_path)
    audit = lock["overlap_audit"]
    version_suffix = manifest["version"].rsplit("-", 1)[-1]
    if not version_suffix.startswith("v") or not version_suffix[1:].isdigit():
        raise ValueError("snapshot_version_invalid")
    selection_binding = {
        "kind": "completed-development-report",
        "report_path": report_relative,
        "report_sha256": sha256_file(report_path),
        "best_checkpoint_relative": checkpoint_relative,
        "selected_checkpoint_path": checkpoint_relative_root,
        "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest),
        "active_run_manifest_path": _relative(root, run_manifest_path),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "checkpoint_source_run_manifest_path": source_manifest_relative,
        "checkpoint_source_run_manifest_sha256": source_manifest_hash,
    }
    if base_checkpoint_binding is not None:
        selection_binding["base_checkpoint_binding"] = base_checkpoint_binding
    if deferred_binding is not None:
        selection_binding.update({
            "deferred_selection_sha256": deferred_binding["deferred_selection_sha256"],
            "deferred_candidate_index_sha256": deferred_binding["deferred_candidate_index_sha256"],
            "selected_checkpoint_update": deferred_binding["selected_checkpoint_update"],
        })
    protocol.update({
        "protocol_id": PROTOCOL_ID_PREFIX + version_suffix,
        "candidates": ["last-block"],
        "run_paths": {"last-block": run_relative},
        "system_test_lock_path": lock_relative,
        "system_test_lock_sha256": sha256_file(lock_path),
        "training_snapshot_path": lock["snapshot_binding"]["path"],
        "training_snapshot_manifest_sha256": sha256_file(snapshot / "manifest.json"),
        "training_snapshot_rows_sha256": artifacts["rows.jsonl"],
        "training_snapshot_splits_sha256": artifacts["splits.jsonl"],
        "overlap_audit_report_sha256": audit["report_sha256"],
        "checkpoint_selection": "Freeze the completed last-block report and its development-selected checkpoint before external inference; no external test selection or calibration.",
        "report_gate": "Write aggregate metrics only after the single locked candidate completes the exact canonical item set.",
        "selection_binding": selection_binding,
        "contamination_limit": (
            f"The bound exact-prompt and 3-word-shingle lexical audit found "
            f"{audit['exact_prompt_overlap_test_rows']} exact and {audit['near_lexical_overlap_test_rows']} near-overlap suite rows; "
            "candidate pairs were not truncated. Semantic overlap and pretrained-model exposure remain unknown."
        ),
        "suite_reuse": {
            **protocol.get("suite_reuse", {}),
            "canonical_items_sha256": protocol["canonical_items_sha256"],
            "comparison_note": "Exploratory reuse of the same previously observed 17,874-item suite; this is not a virgin benchmark result. The checkpoint was selected only on development data.",
            "previous_protocol_sha256": sha256_file(template_path),
            "status": "reused_suite_comparison_not_virgin",
        },
    })
    if deferred_binding is not None:
        protocol["development_execution"] = deferred_binding["development_execution"]
    protocol.pop("exclusion_rebind", None)
    scoring = dict(protocol.get("scoring", {}))
    scoring.pop("publish_aggregate_only_after_both_candidates_complete", None)
    scoring["publish_aggregate_only_after_all_locked_candidates_complete"] = True
    scoring["no_temperature_scaling_or_calibration"] = True
    protocol["scoring"] = scoring
    output_protocol = Path(output_protocol) if output_protocol else run_path.parent / "blind-protocol.json"
    output_protocol = _rooted(root, output_protocol, allowed=("runs",))
    _write_or_validate_json(output_protocol, protocol)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    lock_parser = subparsers.add_parser("lock", help="create the immutable snapshot-bound training lock")
    lock_parser.add_argument("--snapshot", type=Path, required=True)
    lock_parser.add_argument("--audit-directory", type=Path, required=True)
    lock_parser.add_argument("--output-lock", type=Path)
    bind_parser = subparsers.add_parser("bind", help="bind a completed main development run to the blind suite")
    bind_parser.add_argument("--snapshot", type=Path, required=True)
    bind_parser.add_argument("--system-test-lock", type=Path, required=True)
    bind_parser.add_argument("--main-output", "--run-directory", dest="main_output", type=Path, required=True)
    bind_parser.add_argument("--output-protocol", type=Path)
    fast_parser = subparsers.add_parser("bind-fast", help="bind a frozen fast-development winner to the full blind suite")
    fast_parser.add_argument("--snapshot", type=Path, required=True)
    fast_parser.add_argument("--system-test-lock", type=Path, required=True)
    fast_parser.add_argument("--fast-manifest", type=Path, required=True)
    fast_parser.add_argument("--selection-file", type=Path, required=True)
    fast_parser.add_argument("--output-protocol", type=Path)
    args = parser.parse_args()
    if args.command == "lock":
        lock = create_snapshot_bound_lock(args.snapshot, args.audit_directory, args.output_lock)
        lock_path = args.output_lock or (
            _project_root(args.snapshot) / "data" / "external_evaluation"
            / f"system-test-lock-{lock['snapshot_binding']['version']}.json"
        )
        print(json.dumps({"status": lock["status"], "snapshot_manifest_sha256": lock["training_snapshot_manifest_sha256"],
                          "system_test_lock_sha256": sha256_file(lock_path),
                          "overlap_counts": {key: lock["overlap_audit"][key] for key in (
                              "exact_prompt_overlap_test_rows", "near_lexical_overlap_test_rows", "near_candidate_pairs_truncated")}}, sort_keys=True))
    elif args.command == "bind":
        protocol = bind_completed_development_protocol(args.snapshot, args.system_test_lock, args.main_output,
                                                       args.output_protocol)
        print(json.dumps({"protocol_id": protocol["protocol_id"], "candidates": protocol["candidates"],
                          "protocol_sha256": sha256_file(args.output_protocol or args.main_output.parent / "blind-protocol.json")}, sort_keys=True))
    else:
        protocol = bind_fast_development_protocol(
            args.snapshot, args.system_test_lock, args.fast_manifest, args.selection_file,
            args.output_protocol,
        )
        protocol_path = args.output_protocol or args.selection_file.parent / "blind-protocol.json"
        print(json.dumps({
            "protocol_id": protocol["protocol_id"],
            "candidates": protocol["candidates"],
            "selected_checkpoint_update": protocol["selection_binding"]["selected_checkpoint_update"],
            "protocol_sha256": sha256_file(protocol_path),
        }, sort_keys=True))


if __name__ == "__main__":
    main()
