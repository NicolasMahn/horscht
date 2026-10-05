"""Experiment locks and aggregate-only external evaluation metrics."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from evaluation.isolation import EXTERNAL_DATA_ROOT, BenchmarkIsolationError, sha256_file


_LOCK_CHECKSUM_FIELDS = (
    "benchmark_manifest_sha256",
    "canonical_items_sha256",
    "training_snapshot_sha256",
    "checkpoint_sha256",
    "evaluation_code_sha256",
)
_LOCK_TEXT_FIELDS = (
    "experiment_id",
    "model_id",
    "model_revision",
    "prompt_format",
    "benchmark_id",
    "benchmark_revision",
    "training_snapshot_version",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(value), stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_experiment_lock(path: Path, metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Create an immutable checkpoint/data lock before touching external-test inputs."""
    if not path.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
        raise BenchmarkIsolationError("evaluation_lock_outside_external_evaluation_root")
    if path.exists():
        raise BenchmarkIsolationError("evaluation_lock_is_immutable")
    missing = [field for field in (*_LOCK_TEXT_FIELDS, *_LOCK_CHECKSUM_FIELDS) if not metadata.get(field)]
    if missing:
        raise BenchmarkIsolationError("evaluation_lock_metadata_incomplete")
    if any(not isinstance(metadata[field], str) or not metadata[field].strip() for field in _LOCK_TEXT_FIELDS):
        raise BenchmarkIsolationError("evaluation_lock_text_metadata_invalid")
    if any(
        not isinstance(metadata[field], str)
        or len(metadata[field]) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in metadata[field])
        for field in _LOCK_CHECKSUM_FIELDS
    ):
        raise BenchmarkIsolationError("evaluation_lock_checksum_invalid")

    lock = {
        "schema_version": "blind-evaluation-lock-v1",
        **{field: metadata[field] for field in (*_LOCK_TEXT_FIELDS, *_LOCK_CHECKSUM_FIELDS)},
        "locked_at_utc": _now(),
    }
    _atomic_json(path, lock)
    return lock


def _metrics_for(gold: list[int], predicted: list[int]) -> dict[str, Any]:
    if len(gold) != len(predicted) or not gold:
        raise BenchmarkIsolationError("aggregate_prediction_count_invalid")
    correct = sum(actual == guess for actual, guess in zip(gold, predicted))
    return {
        "examples": len(gold),
        "correct": correct,
        "accuracy": correct / len(gold),
    }


def _validate_aggregate_metrics(value: Mapping[str, Any]) -> None:
    if set(value) - {"overall", "by_language"} or "overall" not in value:
        raise BenchmarkIsolationError("result_must_contain_aggregate_metrics_only")
    groups = [value["overall"]]
    by_language = value.get("by_language", {})
    if not isinstance(by_language, Mapping) or any(not isinstance(language, str) for language in by_language):
        raise BenchmarkIsolationError("language_metrics_invalid")
    groups.extend(by_language.values())
    for group in groups:
        if not isinstance(group, Mapping) or set(group) != {"examples", "correct", "accuracy"}:
            raise BenchmarkIsolationError("result_must_contain_aggregate_metrics_only")
        examples, correct, accuracy = group["examples"], group["correct"], group["accuracy"]
        if (
            isinstance(examples, bool)
            or not isinstance(examples, int)
            or examples < 1
            or isinstance(correct, bool)
            or not isinstance(correct, int)
            or not 0 <= correct <= examples
            or isinstance(accuracy, bool)
            or not isinstance(accuracy, (int, float))
            or abs(float(accuracy) - correct / examples) > 1e-12
        ):
            raise BenchmarkIsolationError("aggregate_metric_values_invalid")


def aggregate_predictions(
    items: Iterable[Mapping[str, Any]],
    predictions_by_item_id: Mapping[str, int],
) -> dict[str, Any]:
    """Summarize predictions without retaining per-item outcomes or identifiers."""
    overall_gold: list[int] = []
    overall_predicted: list[int] = []
    language_gold: dict[str, list[int]] = {}
    language_predicted: dict[str, list[int]] = {}
    seen_ids: set[str] = set()
    for row in items:
        item_id = row.get("item_id")
        answer = row.get("answer_index")
        options = row.get("options")
        if (
            not isinstance(item_id, str)
            or isinstance(answer, bool)
            or not isinstance(answer, int)
            or not isinstance(options, list)
            or not 0 <= answer < len(options)
        ):
            raise BenchmarkIsolationError("canonical_item_contract_invalid_during_aggregation")
        if item_id in seen_ids or item_id not in predictions_by_item_id:
            raise BenchmarkIsolationError("prediction_coverage_or_identity_invalid")
        prediction = predictions_by_item_id[item_id]
        if isinstance(prediction, bool) or not isinstance(prediction, int) or not 0 <= prediction < len(options):
            raise BenchmarkIsolationError("prediction_choice_index_invalid")
        seen_ids.add(item_id)
        overall_gold.append(answer)
        overall_predicted.append(prediction)
        language = row.get("language")
        if isinstance(language, str) and language.strip():
            language_gold.setdefault(language, []).append(answer)
            language_predicted.setdefault(language, []).append(prediction)
    if not overall_gold or len(seen_ids) != len(predictions_by_item_id):
        raise BenchmarkIsolationError("prediction_coverage_or_identity_invalid")

    metrics: dict[str, Any] = {"overall": _metrics_for(overall_gold, overall_predicted)}
    if language_gold:
        metrics["by_language"] = {
            language: _metrics_for(language_gold[language], language_predicted[language])
            for language in sorted(language_gold)
        }
    return metrics


def write_aggregate_result(
    output_directory: Path,
    lock_path: Path,
    aggregate_metrics: Mapping[str, Any],
    *,
    exploratory: bool = False,
) -> Path:
    """Write one aggregate result; later uses must be labeled exploratory."""
    if not output_directory.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
        raise BenchmarkIsolationError("evaluation_result_outside_external_evaluation_root")
    if not lock_path.is_file() or not lock_path.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
        raise BenchmarkIsolationError("evaluation_lock_missing_or_outside_external_root")
    try:
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BenchmarkIsolationError("evaluation_lock_invalid") from None
    if lock.get("schema_version") != "blind-evaluation-lock-v1":
        raise BenchmarkIsolationError("evaluation_lock_schema_invalid")
    if any(not isinstance(lock.get(field), str) or not lock[field].strip() for field in _LOCK_TEXT_FIELDS):
        raise BenchmarkIsolationError("evaluation_lock_metadata_invalid")
    if any(
        not isinstance(lock.get(field), str)
        or len(lock[field]) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in lock[field])
        for field in _LOCK_CHECKSUM_FIELDS
    ):
        raise BenchmarkIsolationError("evaluation_lock_checksum_invalid")
    if not isinstance(aggregate_metrics, Mapping):
        raise BenchmarkIsolationError("result_must_contain_aggregate_metrics_only")
    _validate_aggregate_metrics(aggregate_metrics)

    final_path = output_directory / "final-metrics.json"
    if final_path.exists() and not exploratory:
        raise BenchmarkIsolationError("benchmark_result_already_exists_mark_reuse_exploratory")
    if exploratory:
        suffix = hashlib.sha256(f"{_now()}\0{os.getpid()}".encode()).hexdigest()[:12]
        output_path = output_directory / f"exploratory-metrics-{suffix}.json"
    else:
        output_path = final_path
    if output_path.exists():
        raise BenchmarkIsolationError("evaluation_result_is_immutable")
    result = {
        "schema_version": "blind-evaluation-result-v1",
        "experiment_id": lock["experiment_id"],
        "benchmark_id": lock["benchmark_id"],
        "benchmark_revision": lock["benchmark_revision"],
        "lock_sha256": sha256_file(lock_path),
        "evaluated_at_utc": _now(),
        "exploratory_reuse": exploratory,
        "aggregate_metrics": dict(aggregate_metrics),
    }
    _atomic_json(output_path, result)
    return output_path
