"""Run the locked external MCQ protocol without emitting item-level data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping

from evaluation.isolation import EXTERNAL_DATA_ROOT


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "evaluation" / "protocols" / "blind-mcq-v1.json"
DEFAULT_OUTPUT = EXTERNAL_DATA_ROOT / "results" / "paired-overnight-2026-09-30"
DEFAULT_RECEIPT = ROOT / "runs" / "overnight-2026-09-30" / "generation-reflection-receipt.json"
DEFAULT_FULL_RUN = ROOT / "runs" / "paired-overnight-2026-09-30" / "full"
DEFAULT_HEAD_RUN = ROOT / "runs" / "paired-overnight-2026-09-30" / "head"
_STOP_REQUESTED = False
_SCORE_SUM_KEYS = (
    "negative_log_likelihood_sum",
    "multiclass_brier_score_sum",
    "uniform_accuracy_sum",
    "uniform_negative_log_likelihood_sum",
    "uniform_multiclass_brier_score_sum",
)


class BlindEvaluationError(RuntimeError):
    """A fixed safe code for an external-evaluation setup failure."""

    def __init__(self, safe_code: str, safe_metadata: Mapping[str, int] | None = None):
        super().__init__(safe_code)
        self.safe_code = safe_code
        metadata = dict(safe_metadata or {})
        allowed = {"items_checked", "max_input_tokens", "over_limit_items"}
        if set(metadata) - allowed or any(type(value) is not int or value < 0 for value in metadata.values()):
            raise ValueError("unsafe evaluation status metadata")
        self.safe_metadata = metadata


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any], *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            if private:
                os.chmod(temporary, 0o600)
            json.dump(dict(value), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        raise BlindEvaluationError("atomic_metadata_write_failed") from None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _read_object(path: Path, safe_code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BlindEvaluationError(safe_code) from None
    if not isinstance(value, dict):
        raise BlindEvaluationError(safe_code)
    return value


def _repo_file(relative: str, safe_code: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise BlindEvaluationError(safe_code)
    path = ROOT.joinpath(*candidate.parts)
    current = ROOT
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise BlindEvaluationError(safe_code)
    if not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise BlindEvaluationError(safe_code)
    return path


def _external_file(relative: str, safe_code: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise BlindEvaluationError(safe_code)
    path = EXTERNAL_DATA_ROOT.joinpath(*candidate.parts)
    current = EXTERNAL_DATA_ROOT
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise BlindEvaluationError(safe_code)
    if not path.is_file() or not path.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
        raise BlindEvaluationError(safe_code)
    return path


def _load_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise BlindEvaluationError("protocol_file_invalid")
    protocol = _read_object(path, "protocol_file_invalid")
    candidates = protocol.get("candidates")
    protocol_id = protocol.get("protocol_id")
    allowed_protocols = {
        "independent-mcq-full-head-2026-09-v1": ["full", "head"],
        "independent-mcq-current-full-2026-10-v1": ["full"],
        "independent-mcq-current-head-2026-10-v1": ["head"],
        "independent-mcq-current-last-block-2026-10-v1": ["last-block"],
    }
    if not isinstance(candidates, list) or protocol_id not in allowed_protocols or candidates != allowed_protocols[protocol_id]:
        raise BlindEvaluationError("protocol_candidate_contract_invalid")
    if not isinstance(protocol.get("run_paths"), dict) or set(protocol["run_paths"]) != set(candidates):
        raise BlindEvaluationError("protocol_run_paths_invalid")
    selection_binding = protocol.get("selection_binding")
    if protocol_id in {"independent-mcq-current-full-2026-10-v1", "independent-mcq-current-head-2026-10-v1", "independent-mcq-current-last-block-2026-10-v1"}:
        required_binding_fields = {
            "kind", "report_path", "report_sha256", "best_checkpoint_relative",
            "selected_checkpoint_path", "checkpoint_manifest_sha256", "run_manifest_sha256",
            "checkpoint_source_run_manifest_path", "checkpoint_source_run_manifest_sha256",
        }
        if not isinstance(selection_binding, dict) or set(selection_binding) != required_binding_fields or selection_binding.get("kind") != "completed-development-report":
            raise BlindEvaluationError("protocol_selection_binding_invalid")
    if (
        protocol.get("schema_version") != "blind-external-evaluation-protocol-v1"
        or protocol.get("model_key") != "qwen-instruct"
        or protocol.get("canonical_item_count") != 17874
        or protocol.get("inference", {}).get("input_limit_tokens") not in {10240, 32768}
        or protocol.get("inference", {}).get("batch_size") != 1
        or protocol.get("inference", {}).get("truncation") is not False
        or protocol.get("inference", {}).get("choice_order") is not None
        or protocol.get("prompt_format", {}).get("choice_order")
        != "Canonical source order after the pinned deterministic adapter; do not shuffle, relabel, or permute."
        or protocol.get("prompt_format", {}).get("gold_answer_and_metadata_excluded") is not True
        or protocol.get("resumption", {}).get("write_ahead_inflight_marker") is not True
        or protocol.get("scoring", {}).get("primary_metric") != "accuracy"
        or protocol.get("scoring", {}).get("metrics") != [
            "examples", "correct", "accuracy", "negative_log_likelihood", "multiclass_brier_score",
        ]
        or protocol.get("scoring", {}).get("no_temperature_scaling_or_calibration") is not True
        or protocol.get("scoring", {}).get("no_hyperparameter_or_prompt_tuning") is not True
        or protocol.get("scoring", {}).get("no_per_item_predictions_or_errors_in_reports") is not True
        or protocol.get("scoring", {}).get("publish_aggregate_only_after_both_candidates_complete") is not True
        or protocol.get("scoring", {}).get("probabilities")
        != "Softmax over valid masked choice logits; no temperature scaling or calibration."
        or protocol.get("scoring", {}).get("negative_log_likelihood")
        != "Mean -log(p_gold) for the one-hot gold choice."
        or protocol.get("scoring", {}).get("multiclass_brier_score")
        != "Mean sum_j((p_j - y_j)^2) over valid choices, where y is one-hot gold."
        or protocol.get("scoring", {}).get("uniform_baseline")
        != "Per-item expected accuracy 1/K, NLL log(K), and multiclass Brier score 1 - 1/K for K valid choices; aggregate by the same source slices."
    ):
        raise BlindEvaluationError("protocol_contract_invalid")
    return protocol, _sha256(path)


def _validate_system_bundle(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    lock_path = _repo_file(protocol["system_test_lock_path"], "system_test_lock_path_invalid")
    lock_hash = _sha256(lock_path)
    if lock_hash != protocol.get("system_test_lock_sha256"):
        raise BlindEvaluationError("system_test_lock_hash_mismatch")
    lock = _read_object(lock_path, "system_test_lock_invalid")
    if (
        lock.get("schema_version") != "system-test-lock-v1"
        or lock.get("status") != "locked"
        or lock.get("independent_from_training_generator") is not True
        or lock.get("training_snapshot_manifest_sha256") != protocol.get("training_snapshot_manifest_sha256")
    ):
        raise BlindEvaluationError("system_test_lock_contract_invalid")
    files = lock.get("files")
    if not isinstance(files, dict):
        raise BlindEvaluationError("system_test_lock_files_invalid")
    bundle_rel = protocol.get("bundle_manifest_path")
    items_rel = protocol.get("canonical_items_path")
    if not isinstance(bundle_rel, str) or not isinstance(items_rel, str):
        raise BlindEvaluationError("protocol_bundle_paths_invalid")
    external_prefix = "data/external_evaluation/"
    if not bundle_rel.startswith(external_prefix) or not items_rel.startswith(external_prefix):
        raise BlindEvaluationError("protocol_bundle_paths_invalid")
    bundle_path = _external_file(bundle_rel.removeprefix(external_prefix), "bundle_manifest_path_invalid")
    items_path = _external_file(items_rel.removeprefix(external_prefix), "canonical_items_path_invalid")
    relative_bundle = bundle_path.relative_to(EXTERNAL_DATA_ROOT).as_posix()
    relative_items = items_path.relative_to(EXTERNAL_DATA_ROOT).as_posix()
    if files.get(relative_bundle) != protocol.get("bundle_manifest_sha256") or _sha256(bundle_path) != protocol.get("bundle_manifest_sha256"):
        raise BlindEvaluationError("bundle_manifest_hash_mismatch")
    if files.get(relative_items) != protocol.get("canonical_items_sha256") or _sha256(items_path) != protocol.get("canonical_items_sha256"):
        raise BlindEvaluationError("canonical_items_hash_mismatch")
    bundle = _read_object(bundle_path, "bundle_manifest_invalid")
    if (
        bundle.get("canonical_items_path") != relative_items
        or bundle.get("canonical_items_sha256") != protocol.get("canonical_items_sha256")
        or bundle.get("canonical_item_count") != protocol.get("canonical_item_count")
    ):
        raise BlindEvaluationError("bundle_manifest_contract_invalid")
    return lock, items_path


def _verify_exclusion_rebind(protocol: Mapping[str, Any], system_lock: Mapping[str, Any]) -> None:
    binding = protocol.get("exclusion_rebind")
    if binding is None:
        return
    if not isinstance(binding, Mapping):
        raise BlindEvaluationError("exclusion_rebind_binding_invalid")
    report_path = _repo_file(binding.get("report_path", ""), "exclusion_rebind_report_invalid")
    if _sha256(report_path) != binding.get("report_sha256"):
        raise BlindEvaluationError("exclusion_rebind_report_hash_mismatch")
    report = _read_object(report_path, "exclusion_rebind_report_invalid")
    expected_counts = binding.get("excluded_identifier_counts_by_kind")
    if (
        report.get("path") != protocol.get("system_test_lock_path")
        or report.get("rebound_lock_sha256") != protocol.get("system_test_lock_sha256")
        or report.get("original_lock_sha256") != binding.get("original_lock_sha256")
        or report.get("test_suite_files") != len(system_lock.get("files", {}))
        or report.get("test_content_opened") is not False
        or report.get("external_inference_authorized") is not False
        or report.get("historical_excluded_groups") != binding.get("historical_excluded_groups")
        or report.get("current_excluded_groups") != binding.get("current_excluded_groups")
        or report.get("excluded_identifiers_by_kind") != expected_counts
        or binding.get("current_inference_authorization") != "explicit user authorization for this frozen evaluation"
    ):
        raise BlindEvaluationError("exclusion_rebind_contract_mismatch")
    prior_lock = _repo_file("data/external_evaluation/system-test-lock.json", "exclusion_rebind_original_lock_invalid")
    if _sha256(prior_lock) != binding.get("original_lock_sha256"):
        raise BlindEvaluationError("exclusion_rebind_original_lock_hash_mismatch")


def _verify_checkpoint_files(directory: Path, files: Mapping[str, Any]) -> None:
    if directory.is_symlink() or not directory.is_dir() or not isinstance(files, Mapping) or not files:
        raise BlindEvaluationError("selected_checkpoint_files_invalid")
    actual: set[str] = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise BlindEvaluationError("selected_checkpoint_contains_symlink")
        if path.is_file() and path.name != "checkpoint.json":
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(files):
        raise BlindEvaluationError("selected_checkpoint_artifact_set_mismatch")
    for relative, expected in files.items():
        path_part = PurePosixPath(relative)
        if path_part.is_absolute() or ".." in path_part.parts or not isinstance(expected, str):
            raise BlindEvaluationError("selected_checkpoint_artifact_path_invalid")
        path = directory.joinpath(*path_part.parts)
        if not path.is_file() or _sha256(path) != expected:
            raise BlindEvaluationError("selected_checkpoint_artifact_hash_mismatch")


def _selected_candidate(arm: str, run_path: Path, protocol: Mapping[str, Any]) -> dict[str, Any]:
    expected_run = ROOT / protocol["run_paths"][arm]
    if run_path.resolve() != expected_run.resolve() or run_path.is_symlink() or not run_path.is_dir():
        raise BlindEvaluationError("candidate_run_path_mismatch")
    run_manifest_path = run_path / "manifest.json"
    report_path = run_path / "report.json"
    if run_manifest_path.is_symlink() or report_path.is_symlink():
        raise BlindEvaluationError("candidate_run_metadata_symlinked")
    run_manifest = _read_object(run_manifest_path, "candidate_run_manifest_invalid")
    report = _read_object(report_path, "candidate_run_report_invalid")
    model_spec = protocol.get("model_id"), protocol.get("model_revision")
    if (
        run_manifest.get("schema_version") != 2
        or run_manifest.get("arm") != arm
        or run_manifest.get("model") != protocol.get("model_key")
        or (run_manifest.get("repo"), run_manifest.get("revision")) != model_spec
        or run_manifest.get("status") not in {None, "completed"}
        or run_manifest.get("formatter") != protocol["prompt_format"]["id"]
        or run_manifest.get("pooling") != protocol["inference"]["pooling"]
        or run_manifest.get("capacity") != protocol["inference"]["choice_head_capacity"]
        or run_manifest.get("max_tokens") != protocol["inference"]["input_limit_tokens"]
        or run_manifest.get("device") != "cuda"
        or run_manifest.get("attention_backend") != protocol["inference"]["attention_backend"]
        or run_manifest.get("parameters_dtype") != protocol["inference"]["parameters_dtype"]
        or run_manifest.get("compute_autocast") != protocol["inference"]["backbone_autocast"]
        or report.get("arm") != arm
        or report.get("status") != "completed"
        or report.get("selected_checkpoint_reload_verified") is not True
    ):
        raise BlindEvaluationError("candidate_run_protocol_mismatch")
    if not isinstance(report.get("selected_checkpoint_development"), dict):
        raise BlindEvaluationError("candidate_development_selection_missing")
    relative = report.get("best_checkpoint_relative")
    if not isinstance(relative, str) or not relative:
        raise BlindEvaluationError("candidate_selected_checkpoint_missing")
    relative_path = PurePosixPath(relative)
    selection_binding = protocol.get("selection_binding")
    if relative_path.is_absolute():
        raise BlindEvaluationError("candidate_selected_checkpoint_path_invalid")
    if selection_binding is None:
        if ".." in relative_path.parts:
            raise BlindEvaluationError("candidate_selected_checkpoint_path_invalid")
        current = run_path
        for part in relative_path.parts:
            current = current / part
            if current.is_symlink():
                raise BlindEvaluationError("candidate_selected_checkpoint_symlinked")
        checkpoint_dir = current.resolve()
        if not checkpoint_dir.is_relative_to(run_path.resolve()) or not checkpoint_dir.is_dir():
            raise BlindEvaluationError("candidate_selected_checkpoint_outside_run")
    else:
        if protocol.get("run_paths", {}).keys() != {arm} or relative != selection_binding.get("best_checkpoint_relative"):
            raise BlindEvaluationError("candidate_selected_checkpoint_binding_mismatch")
        expected_relative = selection_binding.get("selected_checkpoint_path")
        if not isinstance(expected_relative, str):
            raise BlindEvaluationError("candidate_selected_checkpoint_binding_invalid")
        expected_path = _repo_directory(expected_relative, "candidate_selected_checkpoint_binding_invalid")
        resolved_from_report = (run_path.joinpath(*relative_path.parts)).resolve()
        if resolved_from_report != expected_path.resolve():
            raise BlindEvaluationError("candidate_selected_checkpoint_binding_mismatch")
        checkpoint_dir = expected_path.resolve()
    checkpoint_manifest_path = checkpoint_dir / "checkpoint.json"
    if checkpoint_manifest_path.is_symlink():
        raise BlindEvaluationError("candidate_checkpoint_manifest_symlinked")
    checkpoint_manifest = _read_object(checkpoint_manifest_path, "candidate_checkpoint_manifest_invalid")
    files = checkpoint_manifest.get("files")
    manifest_copy = checkpoint_manifest.get("manifest")
    if selection_binding is None:
        checkpoint_source_manifest_path = run_manifest_path
        checkpoint_source_manifest = run_manifest
    else:
        checkpoint_source_manifest_path = checkpoint_dir.parent.parent / "manifest.json"
        if checkpoint_source_manifest_path.is_symlink():
            raise BlindEvaluationError("candidate_checkpoint_source_manifest_symlinked")
        checkpoint_source_manifest = _read_object(
            checkpoint_source_manifest_path, "candidate_checkpoint_source_manifest_invalid",
        )
        expected_source_path = selection_binding.get("checkpoint_source_run_manifest_path")
        expected_source_hash = selection_binding.get("checkpoint_source_run_manifest_sha256")
        if (
            not isinstance(expected_source_path, str)
            or checkpoint_source_manifest_path.resolve() != _repo_file(
                expected_source_path, "candidate_checkpoint_source_manifest_invalid",
            ).resolve()
            or _sha256(checkpoint_source_manifest_path) != expected_source_hash
            or checkpoint_source_manifest.get("snapshot_manifest_sha256") != protocol.get("training_snapshot_manifest_sha256")
            or checkpoint_source_manifest.get("system_test_manifest_sha256") != protocol.get("system_test_lock_sha256")
        ):
            raise BlindEvaluationError("candidate_checkpoint_source_manifest_mismatch")
    if (
        checkpoint_manifest.get("schema_version") != 1
        or not isinstance(manifest_copy, dict)
        or manifest_copy.get("arm") != arm
        or manifest_copy.get("repo") != model_spec[0]
        or manifest_copy.get("revision") != model_spec[1]
        or manifest_copy.get("formatter") != protocol["prompt_format"]["id"]
        or manifest_copy.get("max_tokens") != protocol["inference"]["input_limit_tokens"]
        or checkpoint_manifest.get("include_backbone") is not (arm in {"full", "last-block"})
        or not isinstance(files, dict)
        or not any(name.startswith("tokenizer/") for name in files)
        or not any(name.startswith("backbone_config/") for name in files)
        or (arm in {"full", "last-block"}) != ("backbone.pt" in files)
        or "head.pt" not in files
    ):
        raise BlindEvaluationError("candidate_checkpoint_protocol_mismatch")
    for field in (
        "model", "repo", "revision", "snapshot_manifest_sha256", "selection_sha256",
        "system_test_manifest_sha256", "seed", "epochs", "device", "threads",
        "formatter", "pooling", "capacity", "max_tokens", "attention_backend",
        "parameters_dtype", "compute_autocast", "initial_head_state_sha256",
    ):
        if manifest_copy.get(field) != checkpoint_source_manifest.get(field):
            raise BlindEvaluationError("candidate_checkpoint_run_manifest_mismatch")
    _verify_checkpoint_files(checkpoint_dir, files)
    if arm == "last-block":
        config = _read_object(checkpoint_dir / "backbone_config" / "config.json", "candidate_checkpoint_config_invalid")
        layer_count = config.get("num_hidden_layers")
        names = checkpoint_source_manifest.get("trainable_backbone_parameter_names")
        count = checkpoint_source_manifest.get("trainable_backbone_parameter_count")
        if (
            type(layer_count) is not int or layer_count < 1
            or checkpoint_source_manifest.get("backbone_training_scope") != "last-block"
            or checkpoint_source_manifest.get("backbone_trainable") is not True
            or not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.startswith(f"layers.{layer_count - 1}.") for name in names)
            or len(names) != len(set(names))
            or type(count) is not int or count <= 0
        ):
            raise BlindEvaluationError("candidate_last_block_scope_invalid")
        for field in ("backbone_training_scope", "backbone_trainable", "trainable_backbone_parameter_names", "trainable_backbone_parameter_count"):
            if (manifest_copy.get(field) != checkpoint_source_manifest.get(field)
                    or run_manifest.get(field) != checkpoint_source_manifest.get(field)):
                raise BlindEvaluationError("candidate_checkpoint_run_manifest_mismatch")
    tokenizer_hashes = {name: digest for name, digest in files.items() if name.startswith("tokenizer/")}
    if not tokenizer_hashes:
        raise BlindEvaluationError("candidate_checkpoint_tokenizer_missing")
    return {
        "arm": arm,
        "run_path": run_path.relative_to(ROOT).as_posix(),
        "run_manifest_path": run_manifest_path.relative_to(ROOT).as_posix(),
        "run_manifest_sha256": _sha256(run_manifest_path),
        "checkpoint_source_run_manifest_path": checkpoint_source_manifest_path.relative_to(ROOT).as_posix(),
        "checkpoint_source_run_manifest_sha256": _sha256(checkpoint_source_manifest_path),
        "run_report_path": report_path.relative_to(ROOT).as_posix(),
        "run_report_sha256": _sha256(report_path),
        "best_checkpoint_relative": relative,
        "checkpoint_path": checkpoint_dir.relative_to(ROOT).as_posix(),
        "checkpoint_manifest_path": checkpoint_manifest_path.relative_to(ROOT).as_posix(),
        "checkpoint_manifest_sha256": _sha256(checkpoint_manifest_path),
        "checkpoint_payload_sha256": _mapping_sha256(dict(sorted(files.items()))),
        "checkpoint_files": dict(sorted(files.items())),
        "tokenizer_files": dict(sorted(tokenizer_hashes.items())),
        "model_id": model_spec[0],
        "model_revision": model_spec[1],
        "device": run_manifest["device"],
        "attention_backend": run_manifest["attention_backend"],
        "threads": run_manifest["threads"],
        "gpu_allocation_cap_gib": run_manifest.get("gpu_allocation_cap_gib"),
    }


def _repo_directory(relative: str, safe_code: str) -> Path:
    candidate = PurePosixPath(relative)
    if candidate.is_absolute() or ".." in candidate.parts or not candidate.parts:
        raise BlindEvaluationError(safe_code)
    path = ROOT.joinpath(*candidate.parts)
    current = ROOT
    for part in candidate.parts:
        current = current / part
        if current.is_symlink():
            raise BlindEvaluationError(safe_code)
    if not path.is_dir() or not path.resolve().is_relative_to(ROOT.resolve()):
        raise BlindEvaluationError(safe_code)
    return path


def _verify_receipt(receipt_path: Path, candidates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    if receipt_path.is_symlink() or not receipt_path.is_file() or not receipt_path.resolve().is_relative_to(ROOT.resolve()):
        raise BlindEvaluationError("generation_reflection_receipt_missing")
    receipt = _read_object(receipt_path, "generation_reflection_receipt_invalid")
    if (
        receipt.get("schema_version") != "generation-reflection-receipt-v1"
        or receipt.get("status") != "committed"
        or receipt.get("external_test_used") is not False
    ):
        raise BlindEvaluationError("generation_reflection_not_committed_or_external_used")
    committed_at = receipt.get("committed_at")
    try:
        parsed_committed_at = datetime.fromisoformat(committed_at.replace("Z", "+00:00"))
    except (AttributeError, TypeError, ValueError):
        raise BlindEvaluationError("generation_reflection_timestamp_invalid") from None
    if parsed_committed_at.tzinfo is None:
        raise BlindEvaluationError("generation_reflection_timestamp_invalid")
    artifact_refs: list[Mapping[str, Any]] = []
    for field in ("internal_dev_artifacts", "guidance_artifacts"):
        references = receipt.get(field)
        if not isinstance(references, list) or any(not isinstance(reference, dict) for reference in references):
            raise BlindEvaluationError("generation_reflection_artifact_refs_invalid")
        artifact_refs.extend(references)
    decision = receipt.get("decision_artifact")
    if not isinstance(decision, dict):
        raise BlindEvaluationError("generation_reflection_decision_missing")
    artifact_refs.append(decision)
    for reference in artifact_refs:
        path_value, expected = reference.get("path"), reference.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str) or len(expected) != 64:
            raise BlindEvaluationError("generation_reflection_artifact_ref_invalid")
        if path_value.startswith("data/external_evaluation/"):
            raise BlindEvaluationError("generation_reflection_uses_external_artifact")
        path = _repo_file(path_value, "generation_reflection_artifact_missing")
        if _sha256(path) != expected:
            raise BlindEvaluationError("generation_reflection_artifact_hash_mismatch")
    refs = receipt.get("selected_checkpoint_refs")
    if not isinstance(refs, list) or len(refs) != 2:
        raise BlindEvaluationError("generation_reflection_checkpoint_refs_invalid")
    refs_by_arm: dict[str, dict[str, Any]] = {}
    for reference in refs:
        if not isinstance(reference, dict) or reference.get("arm") not in {"full", "head"}:
            raise BlindEvaluationError("generation_reflection_checkpoint_refs_invalid")
        arm = reference["arm"]
        if arm in refs_by_arm:
            raise BlindEvaluationError("generation_reflection_checkpoint_refs_invalid")
        refs_by_arm[arm] = reference
    if set(refs_by_arm) != set(candidates):
        raise BlindEvaluationError("generation_reflection_checkpoint_refs_invalid")
    for arm, candidate in candidates.items():
        reference = refs_by_arm[arm]
        if (
            reference.get("manifest_path") != candidate["checkpoint_manifest_path"]
            or reference.get("manifest_sha256") != candidate["checkpoint_manifest_sha256"]
        ):
            raise BlindEvaluationError("generation_reflection_candidate_binding_mismatch")
    return {
        "path": receipt_path.relative_to(ROOT).as_posix(),
        "sha256": _sha256(receipt_path),
        "committed_at": committed_at,
        "selected_checkpoint_refs": refs_by_arm,
    }


def verify_generation_receipt(
    receipt_path: str | Path,
    selected_candidates: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify a committed reflection receipt against frozen checkpoint manifests.

    ``selected_candidates`` is keyed by ``full`` and ``head``. Each value must
    include the selected checkpoint's repository-relative manifest path and
    expected SHA-256, as returned by ``_selected_candidate``.
    """
    if not isinstance(selected_candidates, Mapping) or set(selected_candidates) != {"full", "head"}:
        raise BlindEvaluationError("generation_reflection_checkpoint_refs_invalid")
    for candidate in selected_candidates.values():
        if not isinstance(candidate, Mapping):
            raise BlindEvaluationError("generation_reflection_checkpoint_refs_invalid")
        manifest_path_value = candidate.get("checkpoint_manifest_path")
        expected_hash = candidate.get("checkpoint_manifest_sha256")
        if not isinstance(manifest_path_value, str) or not isinstance(expected_hash, str):
            raise BlindEvaluationError("generation_reflection_checkpoint_refs_invalid")
        manifest_path = _repo_file(manifest_path_value, "candidate_checkpoint_manifest_missing")
        if _sha256(manifest_path) != expected_hash:
            raise BlindEvaluationError("candidate_checkpoint_manifest_hash_mismatch")
    path = Path(receipt_path)
    if not path.is_absolute():
        path = ROOT / path
    return _verify_receipt(path, selected_candidates)


def _verify_selection_binding(
    protocol: Mapping[str, Any],
    selected_candidates: Mapping[str, Mapping[str, Any]],
    receipt_path: Path,
) -> dict[str, Any]:
    binding = protocol.get("selection_binding")
    if binding is None:
        if receipt_path.resolve() != (ROOT / protocol["reflection_receipt_path"]).resolve():
            raise BlindEvaluationError("generation_reflection_receipt_path_mismatch")
        return {
            **verify_generation_receipt(receipt_path, selected_candidates),
            "kind": "generation-reflection-receipt",
        }
    if len(selected_candidates) != 1 or set(selected_candidates) - {"full", "head", "last-block"} or not isinstance(binding, Mapping):
        raise BlindEvaluationError("protocol_selection_binding_invalid")
    candidate = next(iter(selected_candidates.values()))
    checks = {
        "report_path": candidate.get("run_report_path"),
        "report_sha256": candidate.get("run_report_sha256"),
        "best_checkpoint_relative": candidate.get("best_checkpoint_relative"),
        "selected_checkpoint_path": candidate.get("checkpoint_path"),
        "checkpoint_manifest_sha256": candidate.get("checkpoint_manifest_sha256"),
        "run_manifest_sha256": candidate.get("run_manifest_sha256"),
        "checkpoint_source_run_manifest_path": candidate.get("checkpoint_source_run_manifest_path"),
        "checkpoint_source_run_manifest_sha256": candidate.get("checkpoint_source_run_manifest_sha256"),
    }
    if binding.get("kind") != "completed-development-report" or any(binding.get(key) != value for key, value in checks.items()):
        raise BlindEvaluationError("candidate_development_selection_binding_mismatch")
    report_path = _repo_file(binding["report_path"], "candidate_run_report_invalid")
    report = _read_object(report_path, "candidate_run_report_invalid")
    if report.get("status") != "completed" or report.get("selected_checkpoint_reload_verified") is not True:
        raise BlindEvaluationError("candidate_development_selection_not_verified")
    return {
        "kind": "completed-development-report",
        "path": binding["report_path"],
        "sha256": binding["report_sha256"],
    }


def _evaluation_code_hashes(protocol_path: Path) -> dict[str, str]:
    paths = (
        Path("evaluation/blind_runner.py"),
        protocol_path.relative_to(ROOT),
        Path("experiment.py"),
        Path("scalable_training.py"),
        Path("training_checkpoint.py"),
    )
    result = {}
    for relative in paths:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise BlindEvaluationError("evaluation_code_artifact_missing")
        result[relative.as_posix()] = _sha256(path)
    return result


def _make_or_validate_lock(
    protocol: Mapping[str, Any],
    protocol_sha: str,
    system_lock: Mapping[str, Any],
    items_path: Path,
    candidates: Mapping[str, Mapping[str, Any]],
    receipt: Mapping[str, Any],
    output_directory: Path,
    protocol_path: Path,
) -> tuple[Path, dict[str, Any], str]:
    if not output_directory.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
        raise BlindEvaluationError("evaluation_output_outside_external_root")
    output_directory.mkdir(parents=True, exist_ok=True)
    lock_path = output_directory / "blind-evaluation-lock.json"
    candidate_records = [dict(candidates[arm]) for arm in protocol["candidates"]]
    data = {
        "schema_version": "blind-external-evaluation-lock-v1",
        "status": "locked",
        "locked_at_utc": _now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "system_test_lock_sha256": protocol["system_test_lock_sha256"],
        "bundle_manifest_sha256": protocol["bundle_manifest_sha256"],
        "canonical_items_sha256": protocol["canonical_items_sha256"],
        "training_snapshot_manifest_sha256": protocol["training_snapshot_manifest_sha256"],
        "model_id": protocol["model_id"],
        "model_revision": protocol["model_revision"],
        "candidates": candidate_records,
        "evaluation_code_sha256": _evaluation_code_hashes(protocol_path),
        "scoring": protocol["scoring"],
        "prompt_format": protocol["prompt_format"],
        "inference": protocol["inference"],
        "expected_group_counts": protocol["expected_group_counts"],
    }
    if protocol.get("suite_reuse"):
        data["suite_reuse"] = protocol["suite_reuse"]
    if protocol.get("exclusion_rebind"):
        data["exclusion_rebind"] = protocol["exclusion_rebind"]
    if receipt.get("kind") == "generation-reflection-receipt":
        data["reflection_receipt_path"] = receipt["path"]
        data["reflection_receipt_sha256"] = receipt["sha256"]
    else:
        data["selection_evidence_kind"] = receipt["kind"]
        data["selection_evidence_path"] = receipt["path"]
        data["selection_evidence_sha256"] = receipt["sha256"]
    if lock_path.exists():
        existing = _read_object(lock_path, "evaluation_lock_invalid")
        comparable = dict(existing)
        comparable.pop("locked_at_utc", None)
        proposed = dict(data)
        proposed.pop("locked_at_utc", None)
        if comparable != proposed:
            raise BlindEvaluationError("existing_evaluation_lock_does_not_match_frozen_protocol")
        lock = existing
    else:
        _atomic_json(lock_path, data, private=True)
        lock = data
    lock_sha = _sha256(lock_path)
    # The pretraining lock has no mutable status field; only its committed hashes are bound here.
    if system_lock.get("status") != "locked" or _sha256(items_path) != protocol.get("canonical_items_sha256"):
        raise BlindEvaluationError("locked_external_inputs_changed")
    return lock_path, lock, lock_sha


def _group_name(row: Mapping[str, Any], expected_groups: Mapping[str, int]) -> str:
    language = row.get("language")
    if isinstance(language, str):
        name = f"belebele-{language}"
        if name not in expected_groups:
            raise BlindEvaluationError("canonical_group_contract_invalid")
        return name
    category = row.get("category")
    if isinstance(category, str):
        bbh_name = f"bbh-{category.replace('_', '-')}"
        if bbh_name in expected_groups:
            return bbh_name
    return "mmlu-pro-test"


def _validate_item(row: Any, expected_groups: Mapping[str, int]) -> tuple[str, dict[str, Any], int, str]:
    if not isinstance(row, dict):
        raise BlindEvaluationError("canonical_item_record_invalid")
    item_id = row.get("item_id")
    context = row.get("context")
    question = row.get("question")
    options = row.get("options")
    answer = row.get("answer_index")
    if (
        not isinstance(item_id, str) or not item_id
        or not isinstance(context, str)
        or not isinstance(question, str) or not question.strip()
        or not isinstance(options, list) or not 2 <= len(options) <= 265
        or any(not isinstance(option, str) or not option.strip() for option in options)
        or isinstance(answer, bool) or not isinstance(answer, int) or not 0 <= answer < len(options)
    ):
        raise BlindEvaluationError("canonical_item_contract_invalid")
    model_row = {
        # The opaque ordinal is consumed only by the training helper's length guard;
        # experiment.format_input never serializes it.
        "id": "external-item",
        "context": context,
        "question": question,
        "options": options,
    }
    group = _group_name(row, expected_groups)
    return item_id, model_row, answer, group


def _iter_items(items_path: Path) -> Iterator[Any]:
    try:
        source = items_path.open("rb")
    except OSError:
        raise BlindEvaluationError("canonical_items_could_not_be_opened") from None
    with source:
        for encoded in source:
            if not encoded.strip():
                continue
            try:
                yield json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise BlindEvaluationError("canonical_items_json_invalid") from None


def _token_count(tokenizer: Any, prompt: str) -> int:
    try:
        values = tokenizer(prompt, add_special_tokens=True, truncation=False)
        ids = values["input_ids"]
        if hasattr(ids, "shape"):
            if len(ids.shape) == 2:
                return int(ids.shape[1])
            return int(ids.shape[0])
        if ids and isinstance(ids[0], list):
            if len(ids) != 1:
                raise BlindEvaluationError("tokenizer_contract_invalid")
            return len(ids[0])
        return len(ids)
    except BlindEvaluationError:
        raise
    except Exception:
        raise BlindEvaluationError("tokenizer_preflight_failed") from None


def preflight_inputs(
    items_path: Path,
    tokenizer: Any,
    *,
    max_tokens: int,
    expected_count: int,
    expected_groups: Mapping[str, int],
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    seen_item_ids: set[str] = set()
    rows = 0
    over_limit = 0
    maximum = 0
    for raw in _iter_items(items_path):
        item_id, model_row, _answer, group = _validate_item(raw, expected_groups)
        if item_id in seen_item_ids:
            raise BlindEvaluationError("canonical_item_identity_duplicate")
        seen_item_ids.add(item_id)
        rows += 1
        counts[group] = counts.get(group, 0) + 1
        from experiment import format_input

        length = _token_count(tokenizer, format_input(model_row))
        if length < 1:
            raise BlindEvaluationError("tokenizer_returned_empty_input")
        maximum = max(maximum, length)
        over_limit += length > max_tokens
    if rows != expected_count or counts != dict(expected_groups):
        raise BlindEvaluationError("canonical_item_aggregate_contract_invalid")
    return {
        "items_checked": rows,
        "groups_checked": len(counts),
        "max_input_tokens": maximum,
        "over_limit_items": over_limit,
    }


def _empty_state(protocol_sha: str, lock_sha: str, arm: str, item_count: int) -> dict[str, Any]:
    return {
        "schema_version": "blind-evaluation-cursor-v1",
        "status": "running",
        "protocol_sha256": protocol_sha,
        "evaluation_lock_sha256": lock_sha,
        "candidate": arm,
        "item_count": item_count,
        "next_item_index": 0,
        "inflight_item_index": None,
        "aggregate": {"examples": 0, "correct": 0, **{key: 0.0 for key in _SCORE_SUM_KEYS}},
        "by_group": {},
        "updated_at_utc": _now(),
    }


def _write_cursor(path: Path, state: Mapping[str, Any]) -> None:
    payload = dict(state)
    payload.pop("cursor_sha256", None)
    payload["cursor_sha256"] = _mapping_sha256(payload)
    _atomic_json(path, payload, private=True)


def _validate_state(state: Mapping[str, Any], protocol_sha: str, lock_sha: str, arm: str, item_count: int) -> None:
    checksum = state.get("cursor_sha256")
    payload = dict(state)
    payload.pop("cursor_sha256", None)
    if not isinstance(checksum, str) or checksum != _mapping_sha256(payload):
        raise BlindEvaluationError("evaluation_cursor_checksum_invalid")
    if (
        state.get("schema_version") != "blind-evaluation-cursor-v1"
        or state.get("protocol_sha256") != protocol_sha
        or state.get("evaluation_lock_sha256") != lock_sha
        or state.get("candidate") != arm
        or state.get("item_count") != item_count
        or type(state.get("next_item_index")) is not int
        or not 0 <= state["next_item_index"] <= item_count
        or state.get("status") not in {"running", "paused", "complete"}
        or not isinstance(state.get("aggregate"), dict)
        or not isinstance(state.get("by_group"), dict)
    ):
        raise BlindEvaluationError("evaluation_cursor_contract_invalid")
    if state.get("inflight_item_index") is not None:
        raise BlindEvaluationError("previous_run_inflight_item_will_not_be_replayed")
    aggregate = state["aggregate"]
    _validate_sufficient_stats(aggregate, state["next_item_index"], "evaluation_cursor_aggregate_invalid")
    if aggregate["examples"] != state["next_item_index"]:
        raise BlindEvaluationError("evaluation_cursor_aggregate_invalid")
    group_examples = 0
    group_correct = 0
    group_sums = {key: 0.0 for key in _SCORE_SUM_KEYS}
    for values in state["by_group"].values():
        if not isinstance(values, dict):
            raise BlindEvaluationError("evaluation_cursor_aggregate_invalid")
        _validate_sufficient_stats(values, values.get("examples"), "evaluation_cursor_aggregate_invalid")
        group_examples += values["examples"]
        group_correct += values["correct"]
        for key in _SCORE_SUM_KEYS:
            group_sums[key] += values[key]
    if group_examples != aggregate["examples"] or group_correct != aggregate["correct"] or any(
        not math.isclose(group_sums[key], aggregate[key], rel_tol=1e-10, abs_tol=1e-9)
        for key in _SCORE_SUM_KEYS
    ):
        raise BlindEvaluationError("evaluation_cursor_aggregate_invalid")


def _validate_sufficient_stats(values: Mapping[str, Any], examples: Any, safe_code: str) -> None:
    expected_keys = {"examples", "correct", *_SCORE_SUM_KEYS}
    if set(values) != expected_keys:
        raise BlindEvaluationError(safe_code)
    count, correct = values["examples"], values["correct"]
    if type(count) is not int or count < 0 or type(correct) is not int or not 0 <= correct <= count:
        raise BlindEvaluationError(safe_code)
    if examples != count:
        raise BlindEvaluationError(safe_code)
    for key in _SCORE_SUM_KEYS:
        value = values[key]
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise BlindEvaluationError(safe_code)
    if values["multiclass_brier_score_sum"] > 2 * count + 1e-8:
        raise BlindEvaluationError(safe_code)
    if values["uniform_accuracy_sum"] > count / 2 + 1e-8:
        raise BlindEvaluationError(safe_code)
    if values["uniform_negative_log_likelihood_sum"] > count * math.log(265) + 1e-8:
        raise BlindEvaluationError(safe_code)
    if values["uniform_multiclass_brier_score_sum"] > count + 1e-8:
        raise BlindEvaluationError(safe_code)


def _validate_complete_state(state: Mapping[str, Any], expected_groups: Mapping[str, int]) -> None:
    if state.get("status") != "complete" or state.get("next_item_index") != state.get("item_count"):
        raise BlindEvaluationError("completed_candidate_cursor_invalid")
    actual_groups = {
        name: values.get("examples")
        for name, values in state.get("by_group", {}).items()
        if isinstance(values, dict)
    }
    if actual_groups != dict(expected_groups):
        raise BlindEvaluationError("completed_candidate_group_counts_invalid")
    for values in state.get("by_group", {}).values():
        if not isinstance(values, dict):
            raise BlindEvaluationError("completed_candidate_aggregate_invalid")
        _validate_sufficient_stats(values, values.get("examples"), "completed_candidate_aggregate_invalid")


def _progress_status(
    path: Path,
    *,
    phase: str,
    arm: str | None = None,
    processed: int | None = None,
    total: int | None = None,
    safe_error_code: str | None = None,
    items_checked: int | None = None,
    max_input_tokens: int | None = None,
    over_limit_items: int | None = None,
) -> None:
    value: dict[str, Any] = {"phase": phase, "updated_at_utc": _now()}
    if arm is not None:
        value["candidate"] = arm
    if processed is not None:
        value["processed_items"] = processed
    if total is not None:
        value["total_items"] = total
    if safe_error_code is not None:
        value["safe_error_code"] = safe_error_code
    safe_counts = {
        "items_checked": items_checked,
        "max_input_tokens": max_input_tokens,
        "over_limit_items": over_limit_items,
    }
    for key, count in safe_counts.items():
        if count is not None:
            if type(count) is not int or count < 0:
                raise BlindEvaluationError("status_metadata_invalid")
            value[key] = count
    _atomic_json(path, value, private=True)


def _process_candidate_items(
    items_path: Path,
    state_path: Path,
    *,
    protocol_sha: str,
    lock_sha: str,
    arm: str,
    expected_count: int,
    expected_groups: Mapping[str, int],
    score_one: Callable[[dict[str, Any]], list[float]],
    resume: bool,
    stop_after_items: int | None = None,
    status_path: Path | None = None,
) -> dict[str, Any]:
    if state_path.exists():
        if not resume:
            raise BlindEvaluationError("evaluation_cursor_exists_requires_resume")
        state = _read_object(state_path, "evaluation_cursor_invalid")
        _validate_state(state, protocol_sha, lock_sha, arm, expected_count)
        if state.get("status") == "complete":
            return state
        if state.get("status") not in {"paused", "running"}:
            raise BlindEvaluationError("evaluation_cursor_not_safely_resumable")
        state["status"] = "running"
    else:
        if resume:
            raise BlindEvaluationError("evaluation_cursor_missing_for_resume")
        state = _empty_state(protocol_sha, lock_sha, arm, expected_count)
    _write_cursor(state_path, state)
    if _STOP_REQUESTED:
        state["status"] = "paused"
        state["updated_at_utc"] = _now()
        _write_cursor(state_path, state)
        if status_path is not None:
            _progress_status(status_path, phase="paused", arm=arm, processed=state["next_item_index"], total=expected_count)
        return state

    seen_rows = 0
    for raw in _iter_items(items_path):
        item_id, model_row, answer, group = _validate_item(raw, expected_groups)
        ordinal = seen_rows
        seen_rows += 1
        if ordinal < state["next_item_index"]:
            continue
        if ordinal != state["next_item_index"]:
            raise BlindEvaluationError("evaluation_cursor_order_invalid")
        state["inflight_item_index"] = ordinal
        state["updated_at_utc"] = _now()
        _write_cursor(state_path, state)
        logits = score_one(model_row)
        outcome = _score_outcome(logits, answer, len(model_row["options"]))
        prediction = outcome["prediction"]
        aggregate = state["aggregate"]
        _add_outcome(aggregate, outcome, prediction == answer)
        group_stats = state["by_group"].setdefault(
            group,
            {"examples": 0, "correct": 0, **{key: 0.0 for key in _SCORE_SUM_KEYS}},
        )
        _add_outcome(group_stats, outcome, prediction == answer)
        state["next_item_index"] = ordinal + 1
        state["inflight_item_index"] = None
        state["status"] = "running"
        state["updated_at_utc"] = _now()
        _write_cursor(state_path, state)
        if status_path is not None:
            _progress_status(
                status_path, phase="candidate_running", arm=arm,
                processed=state["next_item_index"], total=expected_count,
            )
        if _STOP_REQUESTED or (stop_after_items is not None and state["next_item_index"] >= stop_after_items):
            state["status"] = "paused"
            state["updated_at_utc"] = _now()
            _write_cursor(state_path, state)
            if status_path is not None:
                _progress_status(
                    status_path, phase="paused", arm=arm,
                    processed=state["next_item_index"], total=expected_count,
                )
            return state
    if seen_rows != expected_count or state["next_item_index"] != expected_count:
        raise BlindEvaluationError("canonical_item_count_changed_during_inference")
    actual_groups = {name: values["examples"] for name, values in state["by_group"].items()}
    if actual_groups != dict(expected_groups):
        raise BlindEvaluationError("canonical_group_counts_changed_during_inference")
    aggregate = state["aggregate"]
    if aggregate["examples"] != expected_count:
        raise BlindEvaluationError("candidate_prediction_coverage_invalid")
    state["status"] = "complete"
    state["updated_at_utc"] = _now()
    _write_cursor(state_path, state)
    return state


def _metrics_from_state(state: Mapping[str, Any]) -> dict[str, Any]:
    def one(counts: Mapping[str, Any]) -> dict[str, Any]:
        examples, correct = counts["examples"], counts["correct"]
        try:
            _validate_sufficient_stats(counts, examples, "aggregate_state_invalid")
        except BlindEvaluationError:
            raise
        if examples < 1:
            raise BlindEvaluationError("aggregate_state_invalid")
        return {
            "examples": examples,
            "correct": correct,
            "accuracy": correct / examples,
            "negative_log_likelihood": counts["negative_log_likelihood_sum"] / examples,
            "multiclass_brier_score": counts["multiclass_brier_score_sum"] / examples,
            "uniform_baseline": {
                "accuracy": counts["uniform_accuracy_sum"] / examples,
                "negative_log_likelihood": counts["uniform_negative_log_likelihood_sum"] / examples,
                "multiclass_brier_score": counts["uniform_multiclass_brier_score_sum"] / examples,
            },
        }

    groups = state.get("by_group")
    if not isinstance(groups, dict):
        raise BlindEvaluationError("aggregate_state_invalid")
    return {
        "overall": one(state["aggregate"]),
        "by_source_slice": {name: one(counts) for name, counts in sorted(groups.items())},
    }


def _score_outcome(logits: Any, answer: int, choice_count: int) -> dict[str, Any]:
    if (
        not isinstance(logits, (list, tuple))
        or len(logits) != choice_count
        or isinstance(answer, bool)
        or not isinstance(answer, int)
        or not 2 <= choice_count <= 265
        or not 0 <= answer < choice_count
    ):
        raise BlindEvaluationError("model_output_invalid")
    values = []
    for value in logits:
        if type(value) not in {int, float} or not math.isfinite(value):
            raise BlindEvaluationError("model_output_invalid")
        values.append(float(value))
    maximum = max(values)
    exponentials = [math.exp(value - maximum) for value in values]
    normalizer = math.fsum(exponentials)
    probabilities = [value / normalizer for value in exponentials]
    prediction = max(range(choice_count), key=values.__getitem__)
    nll = (maximum - values[answer]) + math.log(normalizer)
    brier = math.fsum(
        (probability - (1.0 if index == answer else 0.0)) ** 2
        for index, probability in enumerate(probabilities)
    )
    if not all(math.isfinite(value) and value >= 0 for value in (nll, brier)):
        raise BlindEvaluationError("model_output_invalid")
    return {
        "prediction": prediction,
        "negative_log_likelihood": nll,
        "multiclass_brier_score": brier,
        "uniform_accuracy": 1.0 / choice_count,
        "uniform_negative_log_likelihood": math.log(choice_count),
        "uniform_multiclass_brier_score": 1.0 - 1.0 / choice_count,
    }


def _add_outcome(stats: dict[str, Any], outcome: Mapping[str, Any], correct: bool) -> None:
    stats["examples"] += 1
    stats["correct"] += int(correct)
    stats["negative_log_likelihood_sum"] += outcome["negative_log_likelihood"]
    stats["multiclass_brier_score_sum"] += outcome["multiclass_brier_score"]
    stats["uniform_accuracy_sum"] += outcome["uniform_accuracy"]
    stats["uniform_negative_log_likelihood_sum"] += outcome["uniform_negative_log_likelihood"]
    stats["uniform_multiclass_brier_score_sum"] += outcome["uniform_multiclass_brier_score"]


def _load_tokenizer(candidate: Mapping[str, Any]):
    import torch
    import transformers
    from transformers import AutoTokenizer

    if transformers.__version__ != candidate.get("transformers_version"):
        raise BlindEvaluationError("transformers_version_mismatch")
    if torch.__version__ != candidate.get("torch_version"):
        raise BlindEvaluationError("torch_version_mismatch")
    checkpoint_dir = ROOT / candidate["checkpoint_path"]
    tokenizer_path = checkpoint_dir / "tokenizer"
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    except Exception:
        raise BlindEvaluationError("selected_checkpoint_tokenizer_load_failed") from None
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return torch, tokenizer


def _load_model(candidate: Mapping[str, Any], protocol: Mapping[str, Any]):
    import torch
    import transformers
    from transformers import AutoModel

    if transformers.__version__ != candidate.get("transformers_version"):
        raise BlindEvaluationError("transformers_version_mismatch")
    if torch.__version__ != candidate.get("torch_version"):
        raise BlindEvaluationError("torch_version_mismatch")
    device = candidate["device"]
    if device != "cuda" or not torch.cuda.is_available():
        raise BlindEvaluationError("locked_accelerator_unavailable")
    _, total_memory = torch.cuda.mem_get_info()
    cap_gib = candidate.get("gpu_allocation_cap_gib")
    if not isinstance(cap_gib, (int, float)) or cap_gib <= 0 or cap_gib * 1024**3 >= total_memory:
        raise BlindEvaluationError("locked_gpu_memory_cap_invalid")
    torch.cuda.set_per_process_memory_fraction(cap_gib * 1024**3 / total_memory)
    torch.set_num_threads(candidate["threads"])

    checkpoint_dir = ROOT / candidate["checkpoint_path"]
    AutoConfig = transformers.AutoConfig
    config_path = checkpoint_dir / "backbone_config"
    try:
        config = AutoConfig.from_pretrained(config_path, local_files_only=True)
        skeleton = _meta_model(AutoModel, config)
        model_bytes = sum(parameter.numel() for parameter in skeleton.parameters()) * 4
        del skeleton
    except Exception:
        raise BlindEvaluationError("selected_checkpoint_config_load_failed") from None
    try:
        from paired_training import require_memory

        require_memory(model_bytes, 12, device)
    except Exception:
        raise BlindEvaluationError("locked_gpu_memory_reserve_unavailable")
    try:
        backbone = AutoModel.from_pretrained(
            candidate["model_id"], revision=candidate["model_revision"], local_files_only=True,
            dtype=torch.float32, attn_implementation="sdpa", device_map="cuda",
        ).to(device)
    except Exception:
        raise BlindEvaluationError("pinned_backbone_load_failed") from None
    backbone._horscht_attention_backend = protocol["inference"]["attention_backend"]
    backbone.requires_grad_(False)
    backbone.eval()

    from experiment import DecisionHead
    head = DecisionHead(backbone.config.hidden_size, protocol["inference"]["choice_head_capacity"]).to(device)
    head.requires_grad_(False)
    restored = _load_locked_training_checkpoint(checkpoint_dir, backbone, head)
    if restored.get("manifest", {}).get("arm") != candidate["arm"]:
        raise BlindEvaluationError("loaded_checkpoint_arm_mismatch")
    head.eval()
    return torch, backbone, head


def _meta_model(AutoModel: Any, config: Any):
    import torch

    with torch.device("meta"):
        model = AutoModel.from_config(config)
    return model


def _load_locked_training_checkpoint(checkpoint_dir: Path, backbone: Any, head: Any) -> dict[str, Any]:
    from training_checkpoint import load_training_checkpoint

    try:
        return load_training_checkpoint(checkpoint_dir, backbone=backbone, head=head)
    except Exception:
        raise BlindEvaluationError("selected_checkpoint_load_or_hash_validation_failed") from None


def _model_scorer(torch: Any, tokenizer: Any, backbone: Any, head: Any, protocol: Mapping[str, Any], candidate: Mapping[str, Any]):
    from scalable_training import encode_batch

    args = SimpleNamespace(device=candidate["device"], max_tokens=protocol["inference"]["input_limit_tokens"])

    def score(model_row: dict[str, Any]) -> list[float]:
        if set(model_row) != {"id", "context", "question", "options"}:
            raise BlindEvaluationError("model_input_field_allowlist_failed")
        counts = torch.tensor([len(model_row["options"])], device=candidate["device"])
        try:
            with torch.inference_mode():
                features = encode_batch(tokenizer, backbone, [model_row], args.max_tokens, candidate["device"], cache=None)
                logits = head(features, counts)
        except Exception:
            raise BlindEvaluationError("model_inference_failed") from None
        choice_count = counts.item()
        valid = logits[0, :choice_count]
        if not valid.numel() or not torch.isfinite(valid).all():
            raise BlindEvaluationError("model_output_invalid")
        return [float(value) for value in valid.detach().float().cpu().tolist()]

    return score


def _load_protocol_candidates(protocol: Mapping[str, Any], full_run: Path, head_run: Path) -> dict[str, dict[str, Any]]:
    run_paths = {"full": full_run, "head": head_run}
    if "last-block" in protocol["candidates"]:
        run_paths["last-block"] = _repo_directory(protocol["run_paths"]["last-block"], "candidate_run_path_invalid")
    return {
        arm: _selected_candidate(arm, run_paths[arm], protocol)
        for arm in protocol["candidates"]
    }


def _write_final_result(
    output_directory: Path,
    lock_path: Path,
    lock_sha: str,
    protocol_sha: str,
    protocol: Mapping[str, Any],
    states: Mapping[str, Mapping[str, Any]],
) -> Path:
    result_path = output_directory / "blind-aggregate-result.json"
    if result_path.exists():
        raise BlindEvaluationError("final_external_result_already_exists")
    if any(states.get(arm, {}).get("status") != "complete" for arm in protocol["candidates"]):
        raise BlindEvaluationError("all_locked_candidates_must_complete")
    result = {
        "schema_version": "blind-external-aggregate-result-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "evaluation_lock_path": lock_path.relative_to(ROOT).as_posix(),
        "evaluation_lock_sha256": lock_sha,
        "system_test_lock_sha256": protocol["system_test_lock_sha256"],
        "evaluated_at_utc": _now(),
        "candidate_results": {arm: _metrics_from_state(states[arm]) for arm in protocol["candidates"]},
        "benchmark_scope_note": "Belebele language slices are translations and are reported separately; they are not independent semantic problems.",
        "contamination_limit": protocol.get(
            "contamination_limit",
            "The locked overlap audit does not rule out semantic overlap or pretrained-model exposure.",
        ),
    }
    if protocol.get("suite_reuse"):
        result["suite_reuse"] = protocol["suite_reuse"]
    _atomic_json(result_path, result, private=True)
    return result_path


def run_evaluation(
    *,
    full_run: Path = DEFAULT_FULL_RUN,
    head_run: Path = DEFAULT_HEAD_RUN,
    receipt_path: Path = DEFAULT_RECEIPT,
    output_directory: Path = DEFAULT_OUTPUT,
    protocol_path: Path = PROTOCOL_PATH,
    resume: bool = False,
    stop_after_items: int | None = None,
    prepare_only: bool = False,
) -> dict[str, Any]:
    def rooted(path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    full_run = rooted(full_run)
    head_run = rooted(head_run)
    receipt_path = rooted(receipt_path)
    output_directory = rooted(output_directory)
    protocol_path = rooted(protocol_path)
    protocol, protocol_sha = _load_protocol(protocol_path)
    system_lock, items_path = _validate_system_bundle(protocol)
    _verify_exclusion_rebind(protocol, system_lock)
    candidates = _load_protocol_candidates(protocol, full_run, head_run)
    receipt = _verify_selection_binding(protocol, candidates, receipt_path)
    if len(candidates) > 1 and any(
        candidates[arm]["tokenizer_files"] != candidates["full"]["tokenizer_files"]
        for arm in candidates if arm != "full"
    ):
        raise BlindEvaluationError("candidate_tokenizer_hashes_differ")
    for arm, candidate in candidates.items():
        run_manifest = _read_object(ROOT / candidate["run_manifest_path"], "candidate_run_manifest_invalid")
        candidate["torch_version"] = run_manifest.get("torch")
        candidate["transformers_version"] = run_manifest.get("transformers")
    lock_path, lock, lock_sha = _make_or_validate_lock(
        protocol, protocol_sha, system_lock, items_path, candidates, receipt, output_directory, protocol_path,
    )
    status_path = output_directory / "run-status.json"
    safe_base = {
        "protocol_sha256": protocol_sha,
        "evaluation_lock_path": lock_path.relative_to(ROOT).as_posix(),
        "evaluation_lock_sha256": lock_sha,
        "canonical_item_count": protocol["canonical_item_count"],
        "candidate_count": len(candidates),
    }
    if prepare_only:
        torch, tokenizer = _load_tokenizer(next(iter(candidates.values())))
        preflight = preflight_inputs(
            items_path, tokenizer,
            max_tokens=protocol["inference"]["input_limit_tokens"],
            expected_count=protocol["canonical_item_count"],
            expected_groups=protocol["expected_group_counts"],
        )
        if preflight["over_limit_items"]:
            raise BlindEvaluationError(
                "complete_input_exceeds_locked_token_limit",
                safe_metadata={
                    "items_checked": preflight["items_checked"],
                    "max_input_tokens": preflight["max_input_tokens"],
                    "over_limit_items": preflight["over_limit_items"],
                },
            )
        _progress_status(
            status_path, phase="prepared", processed=preflight["items_checked"],
            total=protocol["canonical_item_count"], max_input_tokens=preflight["max_input_tokens"],
        )
        del tokenizer
        return {"phase": "prepared", "max_input_tokens": preflight["max_input_tokens"], **safe_base}

    final_path = output_directory / "blind-aggregate-result.json"
    if final_path.exists():
        if resume:
            return {"phase": "complete", "result_path": final_path.relative_to(ROOT).as_posix(), **safe_base}
        raise BlindEvaluationError("final_external_result_already_exists")

    torch, tokenizer = _load_tokenizer(next(iter(candidates.values())))
    preflight = preflight_inputs(
        items_path, tokenizer,
        max_tokens=protocol["inference"]["input_limit_tokens"],
        expected_count=protocol["canonical_item_count"],
        expected_groups=protocol["expected_group_counts"],
    )
    if preflight["over_limit_items"]:
        raise BlindEvaluationError(
            "complete_input_exceeds_locked_token_limit",
            safe_metadata={
                "items_checked": preflight["items_checked"],
                "max_input_tokens": preflight["max_input_tokens"],
                "over_limit_items": preflight["over_limit_items"],
            },
        )
    _progress_status(
        status_path, phase="input_preflight_complete", processed=preflight["items_checked"],
        total=protocol["canonical_item_count"],
    )
    del tokenizer
    states: dict[str, dict[str, Any]] = {}
    state_paths = {arm: output_directory / f"cursor-{arm}.json" for arm in protocol["candidates"]}
    if resume and not any(path.is_file() for path in state_paths.values()):
        raise BlindEvaluationError("evaluation_cursor_missing_for_resume")
    for arm in protocol["candidates"]:
        state_path = state_paths[arm]
        if state_path.exists():
            state = _read_object(state_path, "evaluation_cursor_invalid")
            _validate_state(state, protocol_sha, lock_sha, arm, protocol["canonical_item_count"])
            if state.get("status") == "complete":
                _validate_complete_state(state, protocol["expected_group_counts"])
                states[arm] = state
                continue
            if not resume:
                raise BlindEvaluationError("evaluation_cursor_exists_requires_resume")
            if state.get("status") not in {"paused", "running"}:
                raise BlindEvaluationError("evaluation_cursor_not_safely_resumable")
        elif resume:
            # A not-yet-started candidate has no state to resume and can start once.
            state = None
        else:
            state = None
        if state is not None and state.get("status") == "complete":
            continue

        torch, tokenizer = _load_tokenizer(candidates[arm])
        torch, backbone, head = _load_model(candidates[arm], protocol)
        scorer = _model_scorer(torch, tokenizer, backbone, head, protocol, candidates[arm])
        state = _process_candidate_items(
            items_path,
            state_path,
            protocol_sha=protocol_sha,
            lock_sha=lock_sha,
            arm=arm,
            expected_count=protocol["canonical_item_count"],
            expected_groups=protocol["expected_group_counts"],
            score_one=scorer,
            resume=state_path.is_file(),
            stop_after_items=stop_after_items,
            status_path=status_path,
        )
        states[arm] = state
        if state.get("status") == "paused":
            return {
                "phase": "paused",
                "candidate": arm,
                "processed_items": state["next_item_index"],
                "total_items": state["item_count"],
                **safe_base,
            }
        del backbone, head, tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(states) != len(candidates):
        raise BlindEvaluationError("candidate_run_state_incomplete")
    result_path = _write_final_result(output_directory, lock_path, lock_sha, protocol_sha, protocol, states)
    _progress_status(
        status_path, phase="complete", processed=protocol["canonical_item_count"],
        total=protocol["canonical_item_count"],
    )
    return {
        "phase": "complete",
        "result_path": result_path.relative_to(ROOT).as_posix(),
        "processed_items_per_candidate": protocol["canonical_item_count"],
        **safe_base,
    }


def _request_stop(_signum: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen blind MCQ evaluator")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--full-run", type=Path, default=DEFAULT_FULL_RUN)
    parser.add_argument("--head-run", type=Path, default=DEFAULT_HEAD_RUN)
    parser.add_argument("--reflection-receipt", type=Path, default=DEFAULT_RECEIPT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--stop-after-items", type=int)
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    args = _parse_args()
    try:
        if args.stop_after_items is not None and args.stop_after_items < 1:
            raise BlindEvaluationError("stop_after_items_invalid")
        result = run_evaluation(
            full_run=args.full_run,
            head_run=args.head_run,
            receipt_path=args.reflection_receipt,
            output_directory=args.output_directory,
            protocol_path=args.protocol,
            resume=args.resume,
            stop_after_items=args.stop_after_items,
            prepare_only=args.prepare_only,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0 if result["phase"] in {"complete", "prepared"} else 75
    except BlindEvaluationError as exc:
        try:
            output_directory = args.output_directory if args.output_directory.is_absolute() else ROOT / args.output_directory
            if output_directory.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
                output_directory.mkdir(parents=True, exist_ok=True)
                status = {"phase": "failed", "updated_at_utc": _now(), "safe_error_code": exc.safe_code}
                status.update(exc.safe_metadata)
                _atomic_json(output_directory / "run-status.json", status, private=True)
        except Exception:
            pass
        print(json.dumps({"phase": "failed", "safe_error_code": exc.safe_code}, sort_keys=True), flush=True)
        return 2
    except Exception:
        try:
            output_directory = args.output_directory if args.output_directory.is_absolute() else ROOT / args.output_directory
            if output_directory.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
                output_directory.mkdir(parents=True, exist_ok=True)
                _progress_status(output_directory / "run-status.json", phase="failed", safe_error_code="unexpected_failure")
        except Exception:
            pass
        print(json.dumps({"phase": "failed", "safe_error_code": "unexpected_failure"}, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
