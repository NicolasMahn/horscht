"""Download and normalize evaluation-only benchmark artifacts.

This module deliberately emits metadata and counts only. Canonical question
content stays in the external-evaluation data tree.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_DATA_ROOT = ROOT / "data" / "external_evaluation"
CANONICAL_SCHEMA = "blind-mcq-v1"


class BenchmarkIsolationError(ValueError):
    """A safe-to-display benchmark setup error without item content."""


@dataclass(frozen=True)
class ReleaseSpec:
    benchmark_id: str
    source_url: str
    revision: str
    artifact_name: str
    source_format: str
    data_license: str
    allowed_hosts: tuple[str, ...]
    official_metadata_url: str
    code_license: str | None = None
    code_revision: str | None = None
    expected_source_rows: int | None = None
    expected_sha256: str | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(dict(value), stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _safe_component(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise BenchmarkIsolationError(f"invalid_{label}")
    if value in {".", ".."}:
        raise BenchmarkIsolationError(f"invalid_{label}")
    return value


def _release_directory(spec: ReleaseSpec) -> Path:
    benchmark_id = _safe_component(spec.benchmark_id, "benchmark_id")
    revision = _safe_component(spec.revision, "revision")
    path = EXTERNAL_DATA_ROOT / benchmark_id / revision
    resolved_root = EXTERNAL_DATA_ROOT.resolve()
    if not path.resolve().is_relative_to(resolved_root):
        raise BenchmarkIsolationError("release_path_outside_external_evaluation_root")
    return path


def _validate_release_spec(spec: ReleaseSpec) -> None:
    _release_directory(spec)
    _safe_component(spec.artifact_name, "artifact_name")
    if spec.source_format not in {"json", "jsonl", "parquet"}:
        raise BenchmarkIsolationError("unsupported_source_format")
    if not spec.data_license.strip() or not spec.official_metadata_url.strip():
        raise BenchmarkIsolationError("release_license_or_metadata_url_missing")
    parsed = urllib.parse.urlparse(spec.source_url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise BenchmarkIsolationError("source_url_must_be_https")
    hosts = {host.lower() for host in spec.allowed_hosts}
    if not hosts or parsed.hostname.lower() not in hosts:
        raise BenchmarkIsolationError("source_host_not_allowlisted")
    pinned_path_parts = {urllib.parse.unquote(part) for part in parsed.path.split("/")}
    pinned_query = urllib.parse.parse_qs(parsed.query).get("revision", [])
    if spec.revision not in pinned_path_parts and spec.revision not in pinned_query:
        raise BenchmarkIsolationError("source_url_does_not_pin_revision")
    metadata_url = urllib.parse.urlparse(spec.official_metadata_url)
    if metadata_url.scheme != "https" or not metadata_url.hostname:
        raise BenchmarkIsolationError("official_metadata_url_must_be_https")
    if spec.expected_source_rows is not None and spec.expected_source_rows < 1:
        raise BenchmarkIsolationError("invalid_expected_source_rows")
    if spec.expected_sha256 is not None and not re.fullmatch(r"[0-9a-fA-F]{64}", spec.expected_sha256):
        raise BenchmarkIsolationError("invalid_expected_sha256")


def _release_identity(spec: ReleaseSpec) -> dict[str, Any]:
    return {
        "schema_version": "external-release-v1",
        "benchmark_id": spec.benchmark_id,
        "source_url": spec.source_url,
        "revision": spec.revision,
        "artifact_name": spec.artifact_name,
        "source_format": spec.source_format,
        "data_license": spec.data_license,
        "code_license": spec.code_license,
        "official_metadata_url": spec.official_metadata_url,
        "code_revision": spec.code_revision,
        "expected_source_rows": spec.expected_source_rows,
    }


def download_release(spec: ReleaseSpec, *, timeout_seconds: int = 60) -> Path:
    """Download one pinned source file and record an immutable provenance manifest."""
    _validate_release_spec(spec)
    destination = _release_directory(spec)
    parent = destination.parent
    source_path = destination / spec.artifact_name
    manifest_path = destination / "release-manifest.json"
    if destination.exists():
        if not source_path.is_file() or not manifest_path.is_file():
            raise BenchmarkIsolationError("release_directory_exists_without_complete_manifest")
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise BenchmarkIsolationError("existing_release_manifest_is_invalid") from None
        digest = sha256_file(source_path)
        if (
            any(previous.get(key) != value for key, value in _release_identity(spec).items())
            or previous.get("source_sha256") != digest
            or (spec.expected_sha256 and digest.lower() != spec.expected_sha256.lower())
        ):
            raise BenchmarkIsolationError("existing_release_does_not_match_pinned_metadata")
        return source_path

    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.revision}.", dir=parent))
    staged_source = staging / spec.artifact_name
    digest = hashlib.sha256()
    size_bytes = 0
    try:
        request = urllib.request.Request(
            spec.source_url,
            headers={"User-Agent": "horscht-blind-evaluation/1.0"},
        )
        try:
            response = urllib.request.urlopen(request, timeout=timeout_seconds)
        except (OSError, urllib.error.URLError, urllib.error.HTTPError):
            raise BenchmarkIsolationError("source_download_failed") from None
        with response:
            final_host = urllib.parse.urlparse(response.geturl()).hostname
            if final_host is None or final_host.lower() not in {host.lower() for host in spec.allowed_hosts}:
                raise BenchmarkIsolationError("source_redirect_host_not_allowlisted")
            with staged_source.open("wb") as stream:
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(block)
                    digest.update(block)
                    size_bytes += len(block)
                stream.flush()
                os.fsync(stream.fileno())

        source_sha256 = digest.hexdigest()
        if size_bytes == 0:
            raise BenchmarkIsolationError("source_artifact_is_empty")
        if spec.expected_sha256 and source_sha256.lower() != spec.expected_sha256.lower():
            raise BenchmarkIsolationError("source_checksum_mismatch")

        manifest = {
            **_release_identity(spec),
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "source_size_bytes": size_bytes,
            "source_sha256": source_sha256,
        }
        _atomic_json(staging / "release-manifest.json", manifest)
        try:
            os.replace(staging, destination)
        except OSError:
            if destination.exists():
                raise BenchmarkIsolationError("release_directory_created_concurrently") from None
            raise BenchmarkIsolationError("could_not_publish_release_directory") from None
        staging = Path()
        return source_path
    finally:
        if staging != Path() and staging.exists():
            shutil.rmtree(staging)


def iter_json_records(path: Path, source_format: str) -> Iterator[Mapping[str, Any]]:
    """Read JSON/JSONL source objects without ever echoing a malformed line."""
    if source_format == "jsonl":
        try:
            stream = path.open("rb")
        except OSError:
            raise BenchmarkIsolationError("source_artifact_could_not_be_opened") from None
        with stream:
            for line_number, encoded_line in enumerate(stream, 1):
                if not encoded_line.strip():
                    continue
                try:
                    record = json.loads(encoded_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise BenchmarkIsolationError("invalid_source_json") from None
                if not isinstance(record, dict):
                    raise BenchmarkIsolationError("source_record_not_an_object")
                yield record
        return

    if source_format == "json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            raise BenchmarkIsolationError("source_json_is_invalid") from None
        if not isinstance(value, list):
            raise BenchmarkIsolationError("source_json_must_be_an_array")
        for record in value:
            if not isinstance(record, dict):
                raise BenchmarkIsolationError("source_record_not_an_object")
            yield record
        return

    raise BenchmarkIsolationError("parquet_requires_a_pinned_adapter")


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(normalized.split())


def canonical_prompt_fingerprint(item: Mapping[str, Any]) -> str:
    """Fingerprint context, question and choices without exposing their text."""
    context = item.get("context", "")
    question = item.get("question", "")
    options = item.get("options")
    if (
        not isinstance(context, str)
        or not isinstance(question, str)
        or not isinstance(options, list)
        or any(not isinstance(option, str) for option in options)
    ):
        raise BenchmarkIsolationError("canonical_prompt_fields_invalid")
    content = {
        "context": _normalize_text(context),
        "question": _normalize_text(question),
        "options": sorted(_normalize_text(option) for option in options),
    }
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def canonicalize_records(
    records: Iterable[Mapping[str, Any]],
    transform: Callable[[Mapping[str, Any], int], Mapping[str, Any]],
    output_path: Path,
    *,
    benchmark_id: str,
    revision: str,
    source_manifest_path: Path,
    source_artifact_path: Path,
    transform_id: str,
    expected_items: int | None = None,
) -> dict[str, Any]:
    """Write canonical MCQs, removing exact duplicate prompts deterministically."""
    _safe_component(benchmark_id, "benchmark_id")
    _safe_component(revision, "revision")
    _safe_component(transform_id, "transform_id")
    resolved_root = EXTERNAL_DATA_ROOT.resolve()
    if not output_path.resolve().is_relative_to(resolved_root):
        raise BenchmarkIsolationError("canonical_output_outside_external_evaluation_root")
    if not source_manifest_path.resolve().is_relative_to(resolved_root):
        raise BenchmarkIsolationError("source_manifest_outside_external_evaluation_root")
    if not source_artifact_path.resolve().is_relative_to(resolved_root):
        raise BenchmarkIsolationError("source_artifact_outside_external_evaluation_root")
    if not source_manifest_path.is_file():
        raise BenchmarkIsolationError("source_manifest_missing")
    if not source_artifact_path.is_file():
        raise BenchmarkIsolationError("source_artifact_missing")
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BenchmarkIsolationError("source_manifest_invalid") from None
    if (
        source_manifest.get("schema_version") != "external-release-v1"
        or source_manifest.get("benchmark_id") != benchmark_id
        or source_manifest.get("revision") != revision
        or source_manifest.get("artifact_name") != source_artifact_path.name
        or source_manifest.get("source_sha256") != sha256_file(source_artifact_path)
    ):
        raise BenchmarkIsolationError("source_artifact_and_manifest_do_not_match")
    if output_path.exists():
        raise BenchmarkIsolationError("canonical_artifact_is_immutable")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    records_read = 0
    rows_written = 0
    duplicates_dropped = 0
    seen_prompts: dict[str, str] = {}
    seen_ids: dict[str, str] = {}
    digest = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output_path.parent, prefix=f".{output_path.name}.",
            suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for source_row_number, source_record in enumerate(records, 1):
                records_read += 1
                try:
                    mapped = transform(source_record, source_row_number)
                except BenchmarkIsolationError:
                    raise
                except Exception:
                    raise BenchmarkIsolationError("adapter_failed") from None
                item = _validate_canonical_item(mapped)
                fingerprint = canonical_prompt_fingerprint(item)
                correct_option = _normalize_text(item["options"][item["answer_index"]])
                previous_target = seen_prompts.get(fingerprint)
                if previous_target is not None:
                    if previous_target != correct_option:
                        raise BenchmarkIsolationError("conflicting_duplicate_prompt")
                    duplicates_dropped += 1
                    continue
                prior_item_fingerprint = seen_ids.get(item["item_id"])
                if prior_item_fingerprint is not None:
                    raise BenchmarkIsolationError("duplicate_item_id")
                seen_prompts[fingerprint] = correct_option
                seen_ids[item["item_id"]] = fingerprint
                encoded = (_canonical_json(item) + "\n").encode("utf-8")
                stream.write(encoded)
                digest.update(encoded)
                rows_written += 1
            stream.flush()
            os.fsync(stream.fileno())

        if rows_written == 0:
            raise BenchmarkIsolationError("canonical_dataset_is_empty")
        expected_source_rows = source_manifest.get("expected_source_rows")
        if expected_source_rows is not None and records_read != expected_source_rows:
            raise BenchmarkIsolationError("source_record_count_does_not_match_pinned_expectation")
        if expected_items is not None and rows_written != expected_items:
            raise BenchmarkIsolationError("canonical_item_count_does_not_match_expectation")

        raw_adapter_stats = getattr(transform, "aggregate_stats", {})
        if not isinstance(raw_adapter_stats, Mapping) or any(
            not isinstance(key, str)
            or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key)
            or isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for key, value in raw_adapter_stats.items()
        ):
            raise BenchmarkIsolationError("adapter_aggregate_metadata_invalid")

        manifest_path = output_path.with_name(f"{output_path.stem}.manifest.json")
        if manifest_path.exists():
            raise BenchmarkIsolationError("canonical_manifest_already_exists")
        source_sha256 = sha256_file(source_manifest_path)
        manifest = {
            "schema_version": "canonical-benchmark-v1",
            "benchmark_id": benchmark_id,
            "revision": revision,
            "canonical_schema": CANONICAL_SCHEMA,
            "transform_id": transform_id,
            "source_manifest_sha256": source_sha256,
            "source_sha256": source_manifest["source_sha256"],
            "source_record_count": records_read,
            "canonical_record_count": rows_written,
            "duplicate_prompt_count": duplicates_dropped,
            "adapter_aggregates": dict(sorted(raw_adapter_stats.items())),
            "canonical_sha256": digest.hexdigest(),
        }
        os.replace(temporary, output_path)
        temporary = None
        _atomic_json(manifest_path, manifest)
        return manifest
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _validate_canonical_item(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkIsolationError("adapter_record_not_an_object")
    item_id = value.get("item_id")
    context = value.get("context", "")
    question = value.get("question")
    options = value.get("options")
    answer_index = value.get("answer_index")
    if not isinstance(item_id, str) or not item_id.strip():
        raise BenchmarkIsolationError("canonical_item_id_invalid")
    if not isinstance(context, str) or not isinstance(question, str) or not question.strip():
        raise BenchmarkIsolationError("canonical_prompt_invalid")
    if (
        not isinstance(options, list)
        or not 2 <= len(options) <= 265
        or any(not isinstance(option, str) or not option.strip() for option in options)
    ):
        raise BenchmarkIsolationError("canonical_options_invalid")
    if isinstance(answer_index, bool) or not isinstance(answer_index, int) or not 0 <= answer_index < len(options):
        raise BenchmarkIsolationError("canonical_answer_index_invalid")

    item: dict[str, Any] = {
        "item_id": item_id.strip(),
        "context": context,
        "question": question,
        "options": list(options),
        "answer_index": answer_index,
    }
    for field in ("language", "category", "group_id"):
        field_value = value.get(field)
        if field_value is not None:
            if not isinstance(field_value, str) or not field_value.strip():
                raise BenchmarkIsolationError(f"canonical_{field}_invalid")
            item[field] = field_value.strip()
    return item


def model_input(item: Mapping[str, Any]) -> str:
    """Return the only canonical fields allowed into benchmark inference."""
    validated = _validate_canonical_item(item)
    choices = "\n".join(f"{index}: {text}" for index, text in enumerate(validated["options"]))
    context = f"Context: {validated['context']}\n" if validated["context"].strip() else ""
    return f"{context}Question: {validated['question']}\nChoices:\n{choices}\nSelected choice:"
