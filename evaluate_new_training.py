"""Fit a soft-target temperature and score the frozen snapshot test split once."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable

import torch

from attention_head import make_head
from evaluation.temperature_scaling import (
    FINAL_TEST_COHORT_INTERPRETATION,
    calibration_record_sha256,
    evaluate_final_test,
    fit_temperature,
    validate_frozen_calibration_record,
    validate_group_split_assignments,
)
from paired_training import ordered_row, read_jsonl, require_memory, sha256, tensors_for
from scalable_training import evaluate_batches, ordered_lengths, require_training_snapshot
from training_checkpoint import load_training_checkpoint
from training_loop import StopController, atomic_json, build_epoch_plan, wait_for_resources


ROOT = Path(__file__).resolve().parent
DEFAULT_RUN = ROOT / "runs" / "new-data-last-block-2026-10-08" / "main"


def _read_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"Invalid metadata artifact: {path.name}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"Invalid metadata artifact: {path.name}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Invalid metadata artifact: {path.name}")
    return value


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    """Atomically create one JSON record without replacing an earlier result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            os.chmod(temporary, 0o600)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A hard link is an atomic create-if-absent operation on this filesystem.
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except FileExistsError:
        raise FileExistsError(f"Immutable evaluation record already exists: {path.name}") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _require_project_paths(snapshot: Path, run: Path, output: Path) -> tuple[Path, Path, Path]:
    original_run = run
    if original_run.is_symlink():
        raise ValueError("The frozen training run cannot be a symlink")
    original_output = output
    if original_output.is_symlink():
        raise ValueError("The evaluation output directory cannot be a symlink")
    snapshot = snapshot.resolve()
    run = run.resolve()
    output = output.resolve()
    snapshot_roots = [
        (ROOT / "data" / "training_snapshots").resolve(),
        (ROOT / "data" / "newtraining").resolve(),
    ]
    runs_root = (ROOT / "runs").resolve()
    if not any(snapshot.is_relative_to(snapshot_root) for snapshot_root in snapshot_roots):
        raise ValueError("This evaluator accepts only frozen snapshots under data/training_snapshots or data/newtraining")
    if not run.is_relative_to(runs_root) or not run.is_dir() or run.is_symlink():
        raise ValueError("The frozen training run must be a real directory under runs")
    if not output.is_relative_to(run):
        raise ValueError("Evaluation artifacts must stay inside the frozen run directory")
    require_training_snapshot(snapshot)
    return snapshot, run, output


_FULL_CHECKPOINT_STORAGE = "full"
_OVERLAY_CHECKPOINT_STORAGE = "trainable-overlay-v1 with immutable warm-start reference"


def _reject_symlink_path(path: Path) -> None:
    if not path.is_absolute():
        raise ValueError("Checkpoint paths must be absolute after resolution")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        if component == ".":
            continue
        if component == "..":
            current = current.parent
            continue
        current = current / component
        if current.is_symlink():
            raise ValueError("Checkpoint storage paths cannot contain symlinks")


def _validate_checkpoint_storage(
    checkpoint: Path,
    checkpoint_record: dict[str, Any],
    run_manifest: dict[str, Any],
    *,
    runs_root: Path,
    current_run: Path,
) -> None:
    """Validate full or warm-start-overlay storage before allocating the model.

    `include_backbone` means the checkpoint contains a backbone state artifact;
    `backbone_state_kind` determines whether that artifact is full or an overlay.
    The loader verifies artifact bytes and reconstructs an overlay from its base.
    """
    storage = run_manifest.get("checkpoint_backbone_state")
    if storage == _FULL_CHECKPOINT_STORAGE:
        expected_state_kind = "full"
    elif storage == _OVERLAY_CHECKPOINT_STORAGE:
        expected_state_kind = "trainable-overlay-v1"
    else:
        raise ValueError("Training manifest has no supported checkpoint storage contract")

    files = checkpoint_record.get("files")
    inner_manifest = checkpoint_record.get("manifest")
    if (
        checkpoint_record.get("include_backbone") is not True
        or checkpoint_record.get("backbone_state_kind") != expected_state_kind
        or not isinstance(files, dict)
        or not {"backbone.pt", "head.pt"} <= set(files)
        or not isinstance(inner_manifest, dict)
        or inner_manifest.get("checkpoint_backbone_state") != storage
        or inner_manifest.get("warm_start") != run_manifest.get("warm_start")
    ):
        raise ValueError("Selected checkpoint does not match its declared backbone storage contract")

    if expected_state_kind == "full":
        if checkpoint_record.get("base_checkpoint") is not None or checkpoint_record.get("backbone_parameter_names") is not None:
            raise ValueError("Full checkpoint must not declare an overlay base or parameter allowlist")
        return

    names = checkpoint_record.get("backbone_parameter_names")
    expected_names = run_manifest.get("trainable_backbone_parameter_names")
    base_reference = checkpoint_record.get("base_checkpoint")
    warm_start = run_manifest.get("warm_start")
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
        or names != expected_names
        or inner_manifest.get("trainable_backbone_parameter_names") != names
        or not isinstance(base_reference, dict)
        or set(base_reference) != {"path", "manifest_sha256"}
        or not isinstance(base_reference.get("path"), str)
        or not isinstance(warm_start, dict)
        or not isinstance(warm_start.get("checkpoint"), str)
        or warm_start.get("checkpoint_manifest_sha256") != base_reference.get("manifest_sha256")
        or not isinstance(warm_start.get("source_snapshot_manifest_sha256"), str)
        or len(warm_start["source_snapshot_manifest_sha256"]) != 64
        or any(
            character not in "0123456789abcdefABCDEF"
            for character in warm_start["source_snapshot_manifest_sha256"]
        )
    ):
        raise ValueError("Compact checkpoint does not declare its exact pinned warm-start base")

    relative_base = Path(base_reference["path"])
    if relative_base.is_absolute():
        raise ValueError("Compact checkpoint base reference must be relative to the checkpoint directory")
    raw_base = checkpoint / relative_base
    _reject_symlink_path(raw_base)
    base = raw_base.resolve()
    runs_root = runs_root.resolve()
    current_run = current_run.resolve()
    if (
        not base.is_relative_to(runs_root)
        or base.is_relative_to(current_run)
        or not base.is_dir()
    ):
        raise ValueError("Compact checkpoint base must be a real checkpoint under runs")

    warm_start_path = Path(warm_start["checkpoint"])
    if not warm_start_path.is_absolute():
        warm_start_path = ROOT / warm_start_path
    _reject_symlink_path(warm_start_path)
    if warm_start_path.resolve() != base:
        raise ValueError("Compact checkpoint base path differs from the recorded warm-start checkpoint")

    base_manifest_path = base / "checkpoint.json"
    base_hash = sha256(base_manifest_path)
    if base_hash != base_reference["manifest_sha256"]:
        raise ValueError("Compact checkpoint base manifest checksum mismatch")
    base_record = _read_object(base_manifest_path)
    base_run_manifest = base_record.get("manifest")
    if (
        base_record.get("include_backbone") is not True
        or base_record.get("backbone_state_kind", "full") != "full"
        or not isinstance(base_record.get("files"), dict)
        or "backbone.pt" not in base_record["files"]
        or not isinstance(base_run_manifest, dict)
        or base_run_manifest.get("arm") != "last-block"
        or base_run_manifest.get("backbone_training_scope") != "last-block"
        or base_run_manifest.get("repo") != run_manifest.get("repo")
        or base_run_manifest.get("revision") != run_manifest.get("revision")
        or base_run_manifest.get("capacity") != run_manifest.get("capacity")
        or base_run_manifest.get("head_architecture", "last-token-linear-v1")
        != run_manifest.get("head_architecture", "last-token-linear-v1")
        or base_run_manifest.get("trainable_backbone_parameter_names") != names
        or base_run_manifest.get("snapshot_manifest_sha256")
        != warm_start.get("source_snapshot_manifest_sha256")
    ):
        raise ValueError("Compact checkpoint base is not the matching full last-block warm-start")


@dataclass(frozen=True)
class _SelectedCheckpoint:
    active_manifest: dict[str, Any]
    active_manifest_sha256: str
    active_report_sha256: str
    checkpoint: Path
    checkpoint_sha256: str
    best_checkpoint_relative: str
    source_run: Path
    source_run_manifest: dict[str, Any]
    source_run_manifest_sha256: str
    base_checkpoint_binding: dict[str, Any] | None


def _run_recovery_index(name: str) -> int | None:
    if name == "main":
        return 0
    match = re.fullmatch(r"main-recovery-(\d{3})", name)
    return int(match.group(1)) if match else None


def _resolve_report_checkpoint(run: Path, relative: str) -> tuple[Path, Path]:
    """Resolve a best checkpoint inside this run or from an earlier run-family sibling."""
    current_index = _run_recovery_index(run.name)
    parsed = PurePosixPath(relative)
    if current_index is None or parsed.is_absolute() or not parsed.parts:
        raise ValueError("Selected checkpoint path is outside the supported main run family")
    if "\\" in relative or "." in parsed.parts:
        raise ValueError("Selected checkpoint path has an unsupported separator")

    parts = parsed.parts
    if ".." not in parts:
        if parts[0] != "checkpoints" or len(parts) < 2:
            raise ValueError("Selected checkpoint path must stay under checkpoints")
        source_run = run
        relative_checkpoint = parts
    else:
        if (
            current_index == 0
            or len(parts) < 4
            or parts[0] != ".."
            or parts[2] != "checkpoints"
            or ".." in parts[1:]
        ):
            raise ValueError("Selected checkpoint path may only target an earlier run-family checkpoint")
        source_name = parts[1]
        source_index = _run_recovery_index(source_name)
        if source_index is None or source_index >= current_index or source_name == run.name:
            raise ValueError("Selected checkpoint source is not an earlier main-family run")
        raw_source_run = run.parent / source_name
        _reject_symlink_path(raw_source_run)
        source_run = raw_source_run.resolve()
        if source_run.parent != run.parent.resolve() or not source_run.is_dir():
            raise ValueError("Selected checkpoint source is outside the current run family")
        relative_checkpoint = parts[2:]

    raw_checkpoint = source_run.joinpath(*relative_checkpoint)
    _reject_symlink_path(raw_checkpoint)
    checkpoint = raw_checkpoint.resolve()
    checkpoints_root = (source_run / "checkpoints").resolve()
    if (
        not checkpoint.is_relative_to(checkpoints_root)
        or not checkpoint.is_dir()
        or checkpoint.is_symlink()
    ):
        raise FileNotFoundError("The development-selected checkpoint is missing")
    return checkpoint, source_run


_LIFECYCLE_HASH_FIELDS = {"runner_sha256", "loop_sha256", "checkpoint_sha256", "input_sha256"}


def _validate_frozen_manifest_match(
    reference: dict[str, Any], candidate: dict[str, Any], *, candidate_label: str
) -> None:
    """Compare frozen training semantics, allowing only run identity and audited code changes."""
    audit = candidate.get("resource_lifecycle_audit")
    changes: dict[str, Any] = {}
    if audit is not None:
        record = audit.get("record") if isinstance(audit, dict) else None
        audit_path_value = audit.get("path") if isinstance(audit, dict) else None
        audit_hash = audit.get("sha256") if isinstance(audit, dict) else None
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != "training-resource-lifecycle-audit-v1"
            or record.get("purpose") not in {"resource-lifecycle-only", "development-evaluation-deferral"}
            or not isinstance(record.get("source_changes"), dict)
            or not record["source_changes"]
            or set(record["source_changes"]) - _LIFECYCLE_HASH_FIELDS
            or not isinstance(record.get("previous_source_files"), dict)
            or set(record["previous_source_files"]) != set(record["source_changes"])
            or not isinstance(audit_path_value, str)
            or not audit_path_value
            or not isinstance(audit_hash, str)
            or len(audit_hash) != 64
        ):
            raise ValueError(f"{candidate_label} has an invalid resource lifecycle audit")
        audit_path = Path(audit_path_value)
        if not audit_path.is_absolute():
            audit_path = ROOT / audit_path
        _reject_symlink_path(audit_path)
        if (
            not audit_path.is_file()
            or sha256(audit_path) != audit_hash
            or _read_object(audit_path) != record
        ):
            raise ValueError(f"{candidate_label} resource lifecycle audit checksum mismatch")
        if record.get("purpose") == "development-evaluation-deferral" or {
            "resume_checkpoint", "previous_run_manifest", "checkpoint_manifest_sha256",
            "previous_run_manifest_sha256",
        } <= set(record):
            from evaluation.snapshot_bound_protocol import validate_resource_lifecycle_audit_chain

            validate_resource_lifecycle_audit_chain(ROOT, candidate)
        changes = record["source_changes"]
        for field, change in changes.items():
            source_name = record["previous_source_files"].get(field)
            if (
                not isinstance(change, dict)
                or set(change) != {"previous", "current"}
                or not isinstance(source_name, str)
            ):
                raise ValueError(f"{candidate_label} has an invalid resource lifecycle source")
            source_relative = Path(source_name)
            if source_relative.is_absolute() or ".." in source_relative.parts:
                raise ValueError(f"{candidate_label} resource lifecycle source escapes the project")
            source_path = ROOT / source_relative
            _reject_symlink_path(source_path)
            if not source_path.is_file() or sha256(source_path) != change.get("previous"):
                raise ValueError(f"{candidate_label} resource lifecycle source checksum mismatch")

    # These identify an invocation or its output directory, not the training contract.
    ignored = {"started_at", "output", "resource_lifecycle_audit"}
    if audit is not None and record.get("purpose") == "development-evaluation-deferral":
        # The chain validator above checks the policy's exact values and the
        # old manifest from which this amendment was made.
        ignored.add("development_execution")
    missing = object()
    for key in (set(reference) | set(candidate)) - ignored:
        before = reference.get(key, missing)
        after = candidate.get(key, missing)
        if before == after:
            continue
        change = changes.get(key)
        if (
            key not in _LIFECYCLE_HASH_FIELDS
            or not isinstance(change, dict)
            or set(change) != {"previous", "current"}
            or change["current"] != after
            or not isinstance(change["previous"], str)
            or len(change["previous"]) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in change["previous"])
            or (
                record.get("purpose") == "resource-lifecycle-only"
                and change["previous"] != before
            )
        ):
            raise ValueError(f"{candidate_label} differs from the selected checkpoint's frozen inputs")
    for key, change in changes.items():
        if (
            not isinstance(change, dict)
            or set(change) != {"previous", "current"}
            or change["current"] != candidate.get(key)
        ):
            raise ValueError(f"{candidate_label} has an inconsistent resource lifecycle audit")


def _require_stage_available(stage: str, calibration_path: Path, attempts_dir: Path, final_test_path: Path) -> str:
    if stage == "calibration":
        if calibration_path.exists():
            return "complete"
        if attempts_dir.exists() or final_test_path.exists():
            raise FileExistsError("Final-test evaluation has started before calibration was frozen")
        return "run"
    elif stage == "test":
        if not calibration_path.is_file():
            raise FileNotFoundError("Freeze calibration.json before starting final-test evaluation")
        return "complete" if final_test_path.exists() else "run"
    else:
        raise ValueError("stage must be calibration or test")


def _selected_checkpoint(run: Path, snapshot_sha256: str) -> _SelectedCheckpoint:
    manifest = _read_object(run / "manifest.json")
    report = _read_object(run / "report.json")
    pointer = _read_object(run / "best-checkpoint.json")
    if (
        manifest.get("arm") != "last-block"
        or manifest.get("backbone_training_scope") != "last-block"
        or manifest.get("head_architecture", "last-token-linear-v1") != "last-token-linear-v1"
        or manifest.get("capacity") != 265
        or manifest.get("snapshot_manifest_sha256") != snapshot_sha256
    ):
        raise ValueError("Training run does not match the requested frozen last-block snapshot")
    if (
        report.get("status") != "completed"
        or report.get("selected_checkpoint_reload_verified") is not True
        or report.get("arm", "last-block") != "last-block"
    ):
        raise ValueError("Training must finish and verify its development-selected checkpoint first")
    relative = report.get("best_checkpoint_relative")
    pointer_relative = pointer.get("path")
    if not isinstance(relative, str) or not relative or pointer_relative != relative:
        raise ValueError("The best-checkpoint pointer does not match the completed training report")
    checkpoint, source_run = _resolve_report_checkpoint(run, relative)
    source_manifest_path = source_run / "manifest.json"
    if source_manifest_path.is_symlink() or not source_manifest_path.is_file():
        raise FileNotFoundError("The selected checkpoint source run manifest is missing")
    source_manifest = _read_object(source_manifest_path)
    for manifest_label, run_path, run_manifest in (
        ("active", run, manifest),
        ("checkpoint source", source_run, source_manifest),
    ):
        recorded_output = run_manifest.get("output")
        recorded_output_path = Path(recorded_output) if isinstance(recorded_output, str) else None
        if recorded_output_path is not None and not recorded_output_path.is_absolute():
            recorded_output_path = ROOT / recorded_output_path
        if (
            recorded_output_path is None
            or not recorded_output
            or recorded_output_path.resolve() != run_path.resolve()
        ):
            raise ValueError(f"The {manifest_label} manifest does not identify its own run directory")
    _validate_frozen_manifest_match(
        source_manifest, manifest, candidate_label="Active recovery manifest"
    )
    checkpoint_manifest_path = checkpoint / "checkpoint.json"
    checkpoint_manifest = _read_object(checkpoint_manifest_path)
    checkpoint_run_manifest = checkpoint_manifest.get("manifest")
    if (
        checkpoint_manifest.get("schema_version") != 1
        or checkpoint_manifest.get("include_backbone") is not True
        or not isinstance(checkpoint_manifest.get("files"), dict)
        or not isinstance(checkpoint_run_manifest, dict)
        or checkpoint_run_manifest.get("snapshot_manifest_sha256") != snapshot_sha256
        or checkpoint_run_manifest.get("backbone_training_scope") != "last-block"
        or checkpoint_run_manifest.get("head_architecture", "last-token-linear-v1") != "last-token-linear-v1"
    ):
        raise ValueError("The selected checkpoint is incomplete or belongs to another snapshot")
    _validate_frozen_manifest_match(
        source_manifest, checkpoint_run_manifest, candidate_label="Selected checkpoint manifest"
    )
    _validate_checkpoint_storage(
        checkpoint,
        checkpoint_manifest,
        source_manifest,
        runs_root=ROOT / "runs",
        current_run=source_run,
    )
    base_checkpoint_binding = None
    if checkpoint_manifest.get("backbone_state_kind") == "trainable-overlay-v1":
        from evaluation.snapshot_bound_protocol import validate_compact_checkpoint_binding

        base_checkpoint_binding = validate_compact_checkpoint_binding(
            ROOT,
            checkpoint,
            checkpoint_manifest,
            source_manifest,
            verify_base_backbone=True,
        )
    from evaluation.snapshot_bound_protocol import validate_deferred_development_selection

    validate_deferred_development_selection(
        ROOT,
        run,
        manifest,
        report,
        checkpoint,
        system_test_manifest_sha256=manifest.get("system_test_manifest_sha256"),
    )
    return _SelectedCheckpoint(
        active_manifest=manifest,
        active_manifest_sha256=sha256(run / "manifest.json"),
        active_report_sha256=sha256(run / "report.json"),
        checkpoint=checkpoint,
        checkpoint_sha256=sha256(checkpoint_manifest_path),
        best_checkpoint_relative=relative,
        source_run=source_run,
        source_run_manifest=source_manifest,
        source_run_manifest_sha256=sha256(source_manifest_path),
        base_checkpoint_binding=base_checkpoint_binding,
    )


def _snapshot_bundle(snapshot: Path) -> tuple[
    dict[str, str], dict[str, list[str]], dict[str, str], dict[str, list[dict[str, Any]]], str
]:
    manifest = _read_object(snapshot / "manifest.json")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("Snapshot manifest has no artifact checksums")
    for filename in ("rows.jsonl", "splits.jsonl"):
        expected = artifacts.get(filename, {}).get("sha256")
        if not isinstance(expected, str) or sha256(snapshot / filename) != expected:
            raise ValueError(f"Snapshot checksum mismatch: {filename}")

    try:
        split_rows = read_jsonl(snapshot / "splits.jsonl")
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("Snapshot split manifest is invalid") from None
    split_by_id: dict[str, str] = {}
    group_ids_by_id: dict[str, list[str]] = {}
    source_by_id: dict[str, str] = {}
    for entry in split_rows:
        if not isinstance(entry, dict):
            raise ValueError("Snapshot split manifest contains an invalid row")
        identifier = entry.get("id")
        split = entry.get("split")
        problem_group = entry.get("problem_group_id")
        snapshot_group = entry.get("snapshot_group_id")
        cohorts = entry.get("cohorts")
        if (
            not isinstance(identifier, str)
            or not identifier
            or identifier in split_by_id
            or split not in {"train", "dev", "calibration", "test"}
            or not isinstance(problem_group, str)
            or not problem_group
            or not isinstance(snapshot_group, str)
            or not snapshot_group
            or not isinstance(cohorts, list)
            or len(cohorts) != 1
            or not isinstance(cohorts[0], str)
            or not cohorts[0].strip()
        ):
            raise ValueError("Snapshot split rows need unique IDs, supported splits, group IDs, and one source cohort")
        split_by_id[identifier] = split
        group_ids_by_id[identifier] = [f"problem:{problem_group}", f"snapshot:{snapshot_group}"]
        source_by_id[identifier] = cohorts[0]
    validate_group_split_assignments(split_by_id, group_ids_by_id)

    rows_by_split: dict[str, list[dict[str, Any]]] = {"calibration": [], "test": []}
    seen_ids: set[str] = set()
    try:
        with (snapshot / "rows.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                identifier = row.get("id") if isinstance(row, dict) else None
                if not isinstance(identifier, str) or identifier not in split_by_id or identifier in seen_ids:
                    raise ValueError("Snapshot rows and split IDs do not match")
                seen_ids.add(identifier)
                split = split_by_id[identifier]
                if split in rows_by_split:
                    rows_by_split[split].append(row)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError("Snapshot rows are invalid") from None
    if seen_ids != set(split_by_id):
        raise ValueError("Snapshot rows and split IDs do not match")
    if not rows_by_split["calibration"] or not rows_by_split["test"]:
        raise ValueError("Snapshot needs nonempty calibration and final-test splits")
    return split_by_id, group_ids_by_id, source_by_id, rows_by_split, sha256(snapshot / "splits.jsonl")


def _load_tokenizer(run_manifest: dict[str, Any], threads: int) -> Any:
    from transformers import AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        run_manifest["repo"], revision=run_manifest["revision"], local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_selected_model(
    run_manifest: dict[str, Any], checkpoint: Path, device: str, threads: int,
    tokenizer: Any, status_path: Path,
) -> tuple[Any, Any]:
    from transformers import AutoConfig, AutoModel

    torch.set_num_threads(threads)
    config = AutoConfig.from_pretrained(
        run_manifest["repo"], revision=run_manifest["revision"], local_files_only=True
    )
    with torch.device("meta"):
        skeleton = AutoModel.from_config(config)
    parameter_bytes = sum(parameter.numel() for parameter in skeleton.parameters()) * 4
    del skeleton

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for frozen-checkpoint inference")
        cap_gib = run_manifest.get("gpu_allocation_cap_gib", 44)
        if isinstance(cap_gib, bool) or not isinstance(cap_gib, (int, float)) or cap_gib <= 0:
            raise ValueError("Frozen training manifest has an invalid GPU allocation cap")
        _, total_memory = torch.cuda.mem_get_info()
        if cap_gib * 1024**3 >= total_memory:
            raise ValueError("Frozen GPU allocation cap must be below addressable memory")
        torch.cuda.set_per_process_memory_fraction(cap_gib * 1024**3 / total_memory)

    reserve_gib = run_manifest.get("reserve_gib", 12)
    if isinstance(reserve_gib, bool) or not isinstance(reserve_gib, (int, float)) or reserve_gib <= 0:
        raise ValueError("Frozen training manifest has an invalid memory reserve")
    additional_bytes = parameter_bytes + 6 * 1024**3
    resource_status = {
        "schema_version": "evaluation-memory-status-v1",
        "phase": "loading frozen evaluation model",
        "updated_at_utc": _time_now(),
    }
    atomic_json(status_path, {**resource_status, "status": "preparing"})

    def publish_memory(snapshot: dict[str, Any], waiting: bool) -> None:
        atomic_json(status_path, {
            **resource_status,
            "status": "waiting_for_resources" if waiting else "ready",
            "memory": snapshot,
        })

    with StopController() as stop:
        ready = wait_for_resources(
            lambda: require_memory(additional_bytes, float(reserve_gib), device),
            stop=stop,
            release_cache=torch.cuda.empty_cache if device == "cuda" else None,
            publish=publish_memory,
        )
    if not ready:
        atomic_json(status_path, {
            **resource_status,
            "status": "interrupted",
            "stop_reason": stop.reason,
        })
        raise InterruptedError("Frozen evaluation stopped while waiting for memory")
    atomic_json(status_path, {**resource_status, "status": "ready"})

    attention_backend = run_manifest.get("attention_backend", "auto")
    if attention_backend not in {"auto", "flash"}:
        raise ValueError("Frozen training manifest has an unsupported attention backend")
    if attention_backend == "flash":
        if device != "cuda":
            raise ValueError("The frozen fused-attention path requires CUDA/HIP inference")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)

    load_options = {"device_map": "cuda"} if device == "cuda" else {}
    backbone = AutoModel.from_pretrained(
        run_manifest["repo"],
        revision=run_manifest["revision"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
        **load_options,
    ).to(device)
    backbone._horscht_attention_backend = attention_backend
    head = make_head(config.hidden_size, run_manifest.get("head_architecture", "last-token-linear-v1")).to(device)
    restored = load_training_checkpoint(checkpoint, backbone=backbone, head=head, restore_rng=False)
    if restored["manifest"].get("backbone_training_scope") != "last-block":
        raise ValueError("Selected checkpoint is not a last-block checkpoint")
    backbone.eval()
    head.eval()
    return backbone, head


def _prepare_inference(
    tokenizer: Any,
    rows: list[dict[str, Any]],
    run_manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], Any]:
    try:
        permuted_rows = [ordered_row(row, run_manifest["seed"], -1) for row in rows]
        lengths = ordered_lengths(
            rows, tokenizer, run_manifest["seed"], -1, run_manifest["max_tokens"]
        )
        plan = build_epoch_plan(
            lengths,
            seed=run_manifest["seed"],
            epoch=-1,
            microbatch=run_manifest["microbatch_questions"],
            accumulation_questions=run_manifest["accumulation_questions"],
            max_batch_tokens=run_manifest["max_batch_tokens"],
            bucket_questions=run_manifest["bucket_questions"],
        )
    except Exception:
        raise RuntimeError("Deterministic evaluation planning failed; per-row details are suppressed") from None
    return permuted_rows, plan


def _infer_rows(
    tokenizer: Any,
    backbone: Any,
    head: Any,
    rows: list[dict[str, Any]],
    permuted_rows: list[dict[str, Any]],
    plan: Any,
    *,
    max_tokens: int,
    device: str,
    evaluation_phase: str,
    before_inference: Callable[[], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    args = SimpleNamespace(
        max_tokens=max_tokens,
        device=device,
        head_architecture="last-token-linear-v1",
        evaluation_phase=evaluation_phase,
    )
    if before_inference is not None:
        before_inference()
    try:
        _, ordered, logits = evaluate_batches(
            tokenizer, backbone, head, permuted_rows, plan, args, None, return_logits=True
        )
    except Exception:
        raise RuntimeError("Frozen-checkpoint inference failed; per-row details are suppressed") from None
    ordered_ids = [row["id"] for row in ordered]
    if len(ordered_ids) != len(set(ordered_ids)) or set(ordered_ids) != {row["id"] for row in rows}:
        raise ValueError("Inference did not cover the requested split exactly once")
    targets, _, _, counts = tensors_for(ordered, "cpu")
    return logits, targets, counts, ordered


def _time_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _ids_sha256(row_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(row_ids)).encode("utf-8")).hexdigest()


def _selection_binding(selected: _SelectedCheckpoint, run: Path) -> dict[str, Any]:
    def relative(path: Path) -> str:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()

    binding: dict[str, Any] = {
        "kind": "completed-development-report",
        "report_path": relative(run / "report.json"),
        "report_sha256": selected.active_report_sha256,
        "best_checkpoint_relative": selected.best_checkpoint_relative,
        "selected_checkpoint_path": relative(selected.checkpoint),
        "checkpoint_manifest_sha256": selected.checkpoint_sha256,
        "active_run_manifest_path": relative(run / "manifest.json"),
        "run_manifest_sha256": selected.active_manifest_sha256,
        "checkpoint_source_run_manifest_path": relative(selected.source_run / "manifest.json"),
        "checkpoint_source_run_manifest_sha256": selected.source_run_manifest_sha256,
    }
    if selected.base_checkpoint_binding is not None:
        binding["base_checkpoint_binding"] = selected.base_checkpoint_binding
    return binding


def _next_test_attempt(attempts_dir: Path, binding: dict[str, Any]) -> int:
    starts = sorted(attempts_dir.glob("attempt-*-started.json")) if attempts_dir.exists() else []
    numbers: list[int] = []
    for started_path in starts:
        try:
            number = int(started_path.name.split("-")[1])
        except (IndexError, ValueError):
            raise ValueError("Final-test attempt receipt has an invalid name") from None
        start = _read_object(started_path)
        if (
            start.get("schema_version") != "soft-target-final-test-attempt-start-v1"
            or start.get("attempt") != number
            or any(start.get(key) != value for key, value in binding.items())
        ):
            raise ValueError("A prior final-test attempt used different frozen inputs")
        numbers.append(number)
        prefix = f"attempt-{number:04d}"
        completed = attempts_dir / f"{prefix}-completed.json"
        failed = attempts_dir / f"{prefix}-failed.json"
        recovered = attempts_dir / f"{prefix}-recovered.json"
        for terminal in (failed, recovered):
            if terminal.exists():
                receipt = _read_object(terminal)
                if receipt.get("attempt") != number or any(receipt.get(key) != value for key, value in binding.items()):
                    raise ValueError("A prior final-test attempt receipt is inconsistent")
        if completed.is_file():
            raise ValueError("A completed attempt is missing its final aggregate record")
        if not failed.exists() and not recovered.exists():
            write_json_once(recovered, {
                "schema_version": "soft-target-final-test-attempt-recovered-v1",
                "attempt": number,
                "recovered_at_utc": _time_now(),
                "reason": "previous attempt did not record a completed aggregate",
                **binding,
            })
    return max(numbers, default=0) + 1


def _validate_completed_test(
    result: dict[str, Any],
    *,
    binding: dict[str, Any],
    calibration_record: dict[str, Any],
) -> dict[str, Any]:
    expected = {
        "schema_version": "soft-target-final-test-v1",
        "split": "test",
        **binding,
        "temperature": calibration_record["temperature"],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError("Completed final-test artifact does not match the frozen evaluation inputs")
    if not isinstance(result.get("metrics"), dict) or set(result["metrics"]) != {"uncalibrated", "calibrated"}:
        raise ValueError("Completed final-test artifact has no paired aggregate metrics")
    if any(
        not isinstance(result["metrics"].get(name), dict)
        or result["metrics"][name].get("questions") != binding["questions"]
        for name in ("uncalibrated", "calibrated")
    ):
        raise ValueError("Completed final-test aggregate counts are inconsistent")
    if not isinstance(result.get("cohort_metrics"), dict) or set(result["cohort_metrics"]) != {
        "original_replay", "fresh", "by_source"
    }:
        raise ValueError("Completed final-test artifact has no source-cohort aggregates")
    if result.get("cohort_interpretation") != FINAL_TEST_COHORT_INTERPRETATION:
        raise ValueError("Completed final-test artifact has an invalid source-cohort interpretation")
    if result.get("source_question_counts") != binding.get("source_question_counts"):
        raise ValueError("Completed final-test source counts do not match the frozen split metadata")
    expected_original = result["source_question_counts"].get("original_replay", 0)
    expected_fresh = sum(
        count for source, count in result["source_question_counts"].items() if source != "original_replay"
    )
    if expected_original < 1 or expected_fresh < 1:
        raise ValueError("Completed final-test artifact is missing a required source cohort")
    for name, questions in (("original_replay", expected_original), ("fresh", expected_fresh)):
        cohort = result["cohort_metrics"][name]
        if not isinstance(cohort, dict) or cohort.get("questions") != questions:
            raise ValueError("Completed final-test source-cohort count is inconsistent")
        if any(
            not isinstance(cohort.get(metric), dict) or cohort[metric].get("questions") != questions
            for metric in ("uncalibrated", "calibrated")
        ):
            raise ValueError("Completed final-test source-cohort metrics are incomplete")
    if not isinstance(result["cohort_metrics"]["by_source"], dict) or set(
        result["cohort_metrics"]["by_source"]
    ) != set(result["source_question_counts"]):
        raise ValueError("Completed final-test per-source aggregates are incomplete")
    for source, questions in result["source_question_counts"].items():
        cohort = result["cohort_metrics"]["by_source"][source]
        if not isinstance(cohort, dict) or cohort.get("questions") != questions:
            raise ValueError("Completed final-test per-source count is inconsistent")
        if any(
            not isinstance(cohort.get(metric), dict) or cohort[metric].get("questions") != questions
            for metric in ("uncalibrated", "calibrated")
        ):
            raise ValueError("Completed final-test per-source metrics are incomplete")
    if type(result.get("attempt")) is not int or result["attempt"] < 1:
        raise ValueError("Completed final-test artifact has no valid attempt receipt")
    return result


def _ensure_completed_attempt(attempts_dir: Path, result: dict[str, Any], binding: dict[str, Any]) -> None:
    attempt = result["attempt"]
    receipt_path = attempts_dir / f"attempt-{attempt:04d}-completed.json"
    expected = {
        "schema_version": "soft-target-final-test-attempt-completed-v1",
        "attempt": attempt,
        "final_result_sha256": calibration_record_sha256(result),
        **binding,
    }
    if receipt_path.exists():
        receipt = _read_object(receipt_path)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError("Final-test completion receipt does not match the aggregate artifact")
        return
    write_json_once(receipt_path, {**expected, "completed_at_utc": _time_now()})


def run_stage(
    stage: str,
    *,
    snapshot: Path,
    run: Path,
    output: Path | None = None,
    device: str | None = None,
    threads: int = 4,
) -> dict[str, Any]:
    if stage not in {"calibration", "test"}:
        raise ValueError("stage must be calibration or test")
    if type(threads) is not int or threads < 1:
        raise ValueError("threads must be a positive integer")
    default_output = run / "calibration-evaluation"
    snapshot, run, output = _require_project_paths(snapshot, run, output or default_output)
    calibration_path = output / "calibration.json"
    attempts_dir = output / "test-attempts"
    final_test_path = output / "final-test.json"
    stage_state = _require_stage_available(stage, calibration_path, attempts_dir, final_test_path)

    snapshot_manifest_sha256 = sha256(snapshot / "manifest.json")
    selected_checkpoint = _selected_checkpoint(run, snapshot_manifest_sha256)
    run_manifest = selected_checkpoint.active_manifest
    checkpoint = selected_checkpoint.checkpoint
    checkpoint_sha256 = selected_checkpoint.checkpoint_sha256
    selection_binding = _selection_binding(selected_checkpoint, run)
    split_by_id, group_ids_by_id, source_by_id, rows_by_split, split_manifest_sha256 = _snapshot_bundle(snapshot)
    run_manifest_sha256 = selected_checkpoint.active_manifest_sha256
    evaluation_device = device or run_manifest.get("device", "cuda")
    if evaluation_device not in {"cuda", "cpu"}:
        raise ValueError("device must be cuda or cpu")
    inference_profile = {
        "device": evaluation_device,
        "attention_backend": run_manifest.get("attention_backend", "auto"),
        "parameters_dtype": "float32",
        "autocast_dtype": "bfloat16",
    }
    calibration_record = _read_object(calibration_path) if calibration_path.is_file() else None
    if calibration_record is not None:
        calibration_ids = [row["id"] for row in rows_by_split["calibration"]]
        if (
            calibration_record.get("run_manifest_sha256") != run_manifest_sha256
            or calibration_record.get("selection_binding") != selection_binding
            or calibration_record.get("snapshot_manifest_sha256") != snapshot_manifest_sha256
            or calibration_record.get("questions") != len(calibration_ids)
            or calibration_record.get("calibration_row_ids_sha256") != _ids_sha256(calibration_ids)
        ):
            raise ValueError("Calibration record does not belong to these frozen inputs")
        validate_frozen_calibration_record(
            calibration_record,
            checkpoint_sha256,
            split_manifest_sha256=split_manifest_sha256,
        )
        if calibration_record.get("inference_profile") != inference_profile:
            raise ValueError("Calibration and evaluation inference profiles differ")
    if stage == "calibration" and stage_state == "complete":
        if calibration_record is None:
            raise ValueError("Frozen calibration record is missing")
        return calibration_record

    test_binding: dict[str, Any] | None = None
    if stage == "test":
        if calibration_record is None:
            raise FileNotFoundError("Freeze calibration.json before starting final-test evaluation")
        test_rows = rows_by_split["test"]
        test_ids = [row["id"] for row in test_rows]
        source_question_counts: dict[str, int] = {}
        for identifier in test_ids:
            source = source_by_id[identifier]
            source_question_counts[source] = source_question_counts.get(source, 0) + 1
        test_binding = {
            "questions": len(test_ids),
            "source_question_counts": source_question_counts,
            "test_row_ids_sha256": _ids_sha256(test_ids),
            "checkpoint_sha256": checkpoint_sha256,
            "calibration_row_ids_sha256": calibration_record["calibration_row_ids_sha256"],
            "calibration_record_sha256": calibration_record_sha256(calibration_record),
            "split_manifest_sha256": split_manifest_sha256,
            "snapshot_manifest_sha256": snapshot_manifest_sha256,
            "run_manifest_sha256": run_manifest_sha256,
            "selection_binding": selection_binding,
            "inference_profile": inference_profile,
        }
        if stage_state == "complete":
            result = _validate_completed_test(
                _read_object(final_test_path),
                binding=test_binding,
                calibration_record=calibration_record,
            )
            _ensure_completed_attempt(attempts_dir, result, test_binding)
            return result

    if evaluation_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for frozen-checkpoint inference")
    selected_rows = rows_by_split["calibration" if stage == "calibration" else "test"]
    tokenizer = _load_tokenizer(run_manifest, threads)
    permuted_rows, plan = _prepare_inference(tokenizer, selected_rows, run_manifest)
    backbone, head = _load_selected_model(
        run_manifest, checkpoint, evaluation_device, threads, tokenizer, output / "memory-status.json"
    )

    if stage == "calibration":
        logits, targets, counts, ordered_rows = _infer_rows(
            tokenizer,
            backbone,
            head,
            selected_rows,
            permuted_rows,
            plan,
            max_tokens=run_manifest["max_tokens"],
            device=evaluation_device,
            evaluation_phase="calibration inference",
        )
        row_ids = [row["id"] for row in ordered_rows]
        result = fit_temperature(
            logits,
            targets,
            counts,
            row_ids=row_ids,
            split_by_id=split_by_id,
            group_ids_by_id=group_ids_by_id,
            checkpoint_sha256=checkpoint_sha256,
            split_manifest_sha256=split_manifest_sha256,
        )
        result["run_manifest_sha256"] = run_manifest_sha256
        result["snapshot_manifest_sha256"] = snapshot_manifest_sha256
        result["selection_binding"] = selection_binding
        result["inference_profile"] = inference_profile
        write_json_once(calibration_path, result)
        return result

    if calibration_record is None or test_binding is None:
        raise RuntimeError("Calibration or final-test provenance disappeared")
    attempt_number = _next_test_attempt(attempts_dir, test_binding)
    started_path = attempts_dir / f"attempt-{attempt_number:04d}-started.json"

    def start_test_inference() -> None:
        write_json_once(started_path, {
            "schema_version": "soft-target-final-test-attempt-start-v1",
            "attempt": attempt_number,
            "started_at_utc": _time_now(),
            **test_binding,
        })

    try:
        logits, targets, counts, ordered_rows = _infer_rows(
            tokenizer,
            backbone,
            head,
            selected_rows,
            permuted_rows,
            plan,
            max_tokens=run_manifest["max_tokens"],
            device=evaluation_device,
            evaluation_phase="final test inference",
            before_inference=start_test_inference,
        )
        row_ids = [row["id"] for row in ordered_rows]
        result = evaluate_final_test(
            logits,
            targets,
            counts,
            row_ids=row_ids,
            source_by_id={identifier: source_by_id[identifier] for identifier in row_ids},
            split_by_id=split_by_id,
            group_ids_by_id=group_ids_by_id,
            checkpoint_sha256=checkpoint_sha256,
            calibration_record=calibration_record,
            split_manifest_sha256=split_manifest_sha256,
        )
        result.update(
            run_manifest_sha256=run_manifest_sha256,
            snapshot_manifest_sha256=snapshot_manifest_sha256,
            selection_binding=selection_binding,
            inference_profile=inference_profile,
            attempt=attempt_number,
        )
        write_json_once(final_test_path, result)
        write_json_once(attempts_dir / f"attempt-{attempt_number:04d}-completed.json", {
            "schema_version": "soft-target-final-test-attempt-completed-v1",
            "attempt": attempt_number,
            "completed_at_utc": _time_now(),
            "final_result_sha256": calibration_record_sha256(result),
            **test_binding,
        })
        return result
    except BaseException as exc:
        if started_path.is_file() and not final_test_path.exists():
            failure_path = attempts_dir / f"attempt-{attempt_number:04d}-failed.json"
            if not failure_path.exists():
                write_json_once(failure_path, {
                    "schema_version": "soft-target-final-test-attempt-failed-v1",
                    "attempt": attempt_number,
                    "failed_at_utc": _time_now(),
                    "failure_kind": type(exc).__name__,
                    **test_binding,
                })
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "test"), required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(argv)
    result = run_stage(
        args.stage,
        snapshot=args.snapshot,
        run=args.run,
        output=args.output,
        device=args.device,
        threads=args.threads,
    )
    print(json.dumps({
        "stage": args.stage,
        "questions": result["questions"],
        "temperature": result.get("temperature"),
        "checkpoint_sha256": result["checkpoint_sha256"],
        "result_path": str((args.output or args.run / "calibration-evaluation") / (
            "calibration.json" if args.stage == "calibration" else "final-test.json"
        )),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
