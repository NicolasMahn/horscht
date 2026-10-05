"""Build the selected isolated bundle, overlap report, and training lock."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from evaluation import adapters, isolation, overlap


ROOT = isolation.ROOT
EXTERNAL_ROOT = ROOT / "data" / "external_evaluation"
STAGING_ROOT = EXTERNAL_ROOT / "staging"
CANONICAL_ROOT = EXTERNAL_ROOT / "canonical"
SNAPSHOT = ROOT / "data" / "training_snapshots" / "2026-09-29-v1"
STATUS_PATH = EXTERNAL_ROOT / "prepare-bundle-status.json"
LOG_PATH = EXTERNAL_ROOT / "prepare-bundle.log"

MMLU_ID = "mmlu-pro-test"
MMLU_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
BELEBELE_REVISION = "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2"
BBH_REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
BELEBELE_SLICES = ("eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "zho_Hans")
BBH_SLICES = (
    "logical_deduction_three_objects",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return isolation.sha256_file(path)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _status(phase: str, **metadata: Any) -> None:
    _atomic_json(STATUS_PATH, {"phase": phase, "updated_at_utc": _now(), **metadata})


def _source_path(benchmark_id: str, revision: str, filename: str) -> Path:
    return STAGING_ROOT / benchmark_id / revision / filename


def _source_manifest(benchmark_id: str, revision: str) -> Path:
    return STAGING_ROOT / benchmark_id / revision / "release-manifest.json"


def _canonicalize_one(
    records: Iterable[dict[str, Any]],
    adapter: Any,
    *,
    benchmark_id: str,
    revision: str,
    source_filename: str,
    output_relative: str,
    transform_id: str,
    expected_items: int | None = None,
) -> tuple[Path, dict[str, Any]]:
    source_path = _source_path(benchmark_id, revision, source_filename)
    source_manifest = _source_manifest(benchmark_id, revision)
    output_path = CANONICAL_ROOT / output_relative
    manifest = isolation.canonicalize_records(
        records,
        adapter,
        output_path,
        benchmark_id=benchmark_id,
        revision=revision,
        source_manifest_path=source_manifest,
        source_artifact_path=source_path,
        transform_id=transform_id,
        expected_items=expected_items,
    )
    return output_path, manifest


def _load_mmlu() -> tuple[Path, dict[str, Any]]:
    source_path = _source_path(MMLU_ID, MMLU_REVISION, "test-00000-of-00001.parquet")
    adapter = adapters.MMLUProAdapter()
    return _canonicalize_one(
        adapters.iter_parquet_records(source_path),
        adapter,
        benchmark_id=MMLU_ID,
        revision=MMLU_REVISION,
        source_filename=source_path.name,
        output_relative="mmlu-pro-test/items.jsonl",
        transform_id="mmlu-pro-official-placeholder-filter-v1",
    )


def _load_belebele(language_file: str) -> tuple[Path, dict[str, Any]]:
    benchmark_id = f"belebele-{language_file}"
    source_path = _source_path(benchmark_id, BELEBELE_REVISION, f"{language_file}.jsonl")
    adapter = adapters.BelebeleAdapter(language_file)
    return _canonicalize_one(
        isolation.iter_json_records(source_path, "jsonl"),
        adapter,
        benchmark_id=benchmark_id,
        revision=BELEBELE_REVISION,
        source_filename=source_path.name,
        output_relative=f"belebele-{language_file}/items.jsonl",
        transform_id=f"belebele-schema-v1-{language_file}",
        expected_items=900,
    )


def _load_bbh(task_id: str) -> tuple[Path, dict[str, Any]]:
    benchmark_id = f"bbh-{task_id.replace('_', '-')}"
    source_path = _source_path(benchmark_id, BBH_REVISION, f"{task_id}.json")
    adapter = adapters.BBHMultipleChoiceAdapter(task_id)
    records = adapters.iter_bbh_records(source_path)
    path, manifest = _canonicalize_one(
        records,
        adapter,
        benchmark_id=benchmark_id,
        revision=BBH_REVISION,
        source_filename=source_path.name,
        output_relative=f"bbh-{task_id.replace('_', '-')}/items.jsonl",
        transform_id=f"bbh-explicit-choice-parser-v1-{task_id}",
    )
    if manifest["source_record_count"] > 250:
        raise isolation.BenchmarkIsolationError("bbh_selected_slice_exceeds_250_rows")
    return path, manifest


def _merge_canonical_items(paths: list[Path], output: Path) -> tuple[int, str]:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise isolation.BenchmarkIsolationError("merged_canonical_bundle_is_immutable")
    digest = hashlib.sha256()
    rows = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for path in paths:
                with path.open("rb") as source:
                    for line in source:
                        if not line.strip():
                            continue
                        try:
                            item = json.loads(line)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            raise isolation.BenchmarkIsolationError("canonical_source_json_invalid") from None
                        if not isinstance(item, dict):
                            raise isolation.BenchmarkIsolationError("canonical_source_row_invalid")
                        stream.write(line if line.endswith(b"\n") else line + b"\n")
                        digest.update(line if line.endswith(b"\n") else line + b"\n")
                        rows += 1
            stream.flush()
            os.fsync(stream.fileno())
        if rows == 0:
            raise isolation.BenchmarkIsolationError("merged_canonical_bundle_empty")
        os.replace(temporary, output)
        temporary = None
        return rows, digest.hexdigest()
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _exact_training_exclusions(exact_overlay_path: Path) -> dict[str, list[str]]:
    exact_ids: set[str] = set()
    try:
        with exact_overlay_path.open("rb") as stream:
            for encoded in stream:
                if not encoded.strip():
                    continue
                try:
                    record = json.loads(encoded)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise isolation.BenchmarkIsolationError("overlap_overlay_json_invalid") from None
                identifier = record.get("training_id") if isinstance(record, dict) else None
                if not isinstance(identifier, str) or not identifier:
                    raise isolation.BenchmarkIsolationError("overlap_overlay_id_invalid")
                exact_ids.add(identifier)
    except OSError:
        raise isolation.BenchmarkIsolationError("overlap_overlay_could_not_be_read") from None

    row_ids: set[str] = set()
    problem_group_ids: set[str] = set()
    snapshot_group_ids: set[str] = set()
    training_rows = SNAPSHOT / "rows.jsonl"
    try:
        stream = training_rows.open("rb")
    except OSError:
        raise isolation.BenchmarkIsolationError("frozen_training_snapshot_rows_missing") from None
    with stream:
        for encoded in stream:
            if not encoded.strip():
                continue
            try:
                row = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise isolation.BenchmarkIsolationError("frozen_training_snapshot_json_invalid") from None
            if not isinstance(row, dict):
                raise isolation.BenchmarkIsolationError("frozen_training_snapshot_row_invalid")
            identifier = row.get("id")
            if identifier not in exact_ids:
                continue
            row_ids.add(identifier)
            problem_group = row.get("problem_group_id")
            if isinstance(problem_group, str) and problem_group:
                problem_group_ids.add(problem_group)
            snapshot = row.get("_training_snapshot")
            if isinstance(snapshot, dict):
                group = snapshot.get("snapshot_group_id")
                if isinstance(group, str) and group:
                    snapshot_group_ids.add(group)
    if row_ids != exact_ids:
        raise isolation.BenchmarkIsolationError("overlap_overlay_training_ids_not_in_frozen_snapshot")
    return {
        "row_ids": sorted(row_ids),
        "problem_group_ids": sorted(problem_group_ids),
        "snapshot_group_ids": sorted(snapshot_group_ids),
        "family_ids": [],
        "operation_ids": [],
    }


def _write_system_test_lock(
    *,
    bundle_metadata: dict[str, Any],
    overlap_metadata: dict[str, Any],
    training_exclusions: dict[str, list[str]],
    artifact_paths: list[Path],
) -> tuple[Path, dict[str, Any]]:
    lock_path = EXTERNAL_ROOT / "system-test-lock.json"
    if lock_path.exists():
        raise isolation.BenchmarkIsolationError("system_test_lock_is_immutable")
    files: dict[str, str] = {}
    for artifact in sorted(set(artifact_paths)):
        if artifact.is_symlink() or not artifact.is_file():
            raise isolation.BenchmarkIsolationError("system_test_artifact_missing_or_symlinked")
        if not artifact.resolve().is_relative_to(EXTERNAL_ROOT.resolve()):
            raise isolation.BenchmarkIsolationError("system_test_artifact_outside_lock_directory")
        relative = artifact.relative_to(EXTERNAL_ROOT).as_posix()
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise isolation.BenchmarkIsolationError("system_test_artifact_path_invalid")
        files[relative] = _sha256(artifact)
    if not files:
        raise isolation.BenchmarkIsolationError("system_test_artifacts_missing")

    snapshot_manifest_path = SNAPSHOT / "manifest.json"
    lock = {
        "schema_version": "system-test-lock-v1",
        "status": "locked",
        "locked_at": _now(),
        "independent_from_training_generator": True,
        "training_snapshot_manifest_sha256": _sha256(snapshot_manifest_path),
        "training_exclusions": training_exclusions,
        "files": files,
        "benchmark_bundle": bundle_metadata,
        "overlap_audit": overlap_metadata,
        "pipeline_sha256": {
            relative: _sha256(ROOT / relative)
            for relative in (
                "evaluation/adapters.py",
                "evaluation/isolation.py",
                "evaluation/overlap.py",
                "evaluation/prepare_bundle.py",
            )
        },
        "training_exclusion_policy": "exclude exact overlap rows plus their source and snapshot groups; preserve raw rows",
        "contamination_limit": "exact and lexical audit does not rule out semantic overlap or base-model pretraining exposure",
    }
    _atomic_json(lock_path, lock)
    return lock_path, lock


def prepare() -> dict[str, Any]:
    source_specs: list[tuple[str, str]] = [(MMLU_ID, MMLU_REVISION)]
    source_specs.extend((f"belebele-{language_file}", BELEBELE_REVISION) for language_file in BELEBELE_SLICES)
    source_specs.extend((f"bbh-{task_id.replace('_', '-')}", BBH_REVISION) for task_id in BBH_SLICES)
    expected_sources = len(source_specs)
    for benchmark_id, revision in source_specs:
        if not _source_manifest(benchmark_id, revision).is_file():
            raise isolation.BenchmarkIsolationError("pinned_source_release_missing")

    _status("canonicalizing", expected_sources=expected_sources, completed_sources=0)
    canonical_paths: list[Path] = []
    canonical_manifests: list[dict[str, Any]] = []
    path, manifest = _load_mmlu()
    canonical_paths.append(path)
    canonical_manifests.append(manifest)
    _status("canonicalizing", expected_sources=expected_sources, completed_sources=1)
    for index, language_file in enumerate(BELEBELE_SLICES, 2):
        path, manifest = _load_belebele(language_file)
        canonical_paths.append(path)
        canonical_manifests.append(manifest)
        _status("canonicalizing", expected_sources=expected_sources, completed_sources=index)
    for offset, task_id in enumerate(BBH_SLICES, 2 + len(BELEBELE_SLICES)):
        path, manifest = _load_bbh(task_id)
        canonical_paths.append(path)
        canonical_manifests.append(manifest)
        _status("canonicalizing", expected_sources=expected_sources, completed_sources=offset)

    merged_items_path = CANONICAL_ROOT / "bundle" / "items.jsonl"
    canonical_rows, merged_sha256 = _merge_canonical_items(canonical_paths, merged_items_path)
    release_metadata = []
    for benchmark_id, revision in source_specs:
        manifest_path = _source_manifest(benchmark_id, revision)
        release = json.loads(manifest_path.read_text(encoding="utf-8"))
        canonical_manifest_path = next(
            path.with_name(f"{path.stem}.manifest.json")
            for path in canonical_paths
            if path.parent.name == benchmark_id or path.parent.name == benchmark_id.replace("_", "-")
        )
        canonical_manifest = json.loads(canonical_manifest_path.read_text(encoding="utf-8"))
        release_metadata.append({
            "benchmark_id": benchmark_id,
            "revision": revision,
            "source_url": release["source_url"],
            "official_metadata_url": release["official_metadata_url"],
            "data_license": release["data_license"],
            "code_license": release.get("code_license"),
            "code_revision": release.get("code_revision"),
            "source_bytes": release["source_size_bytes"],
            "source_sha256": release["source_sha256"],
            "source_record_count": canonical_manifest["source_record_count"],
            "canonical_record_count": canonical_manifest["canonical_record_count"],
            "duplicate_prompt_count": canonical_manifest["duplicate_prompt_count"],
            "canonical_sha256": canonical_manifest["canonical_sha256"],
            "transform_id": canonical_manifest["transform_id"],
        })

    bundle_manifest_path = EXTERNAL_ROOT / "bundle-manifest.json"
    bundle_metadata = {
        "schema_version": "external-test-bundle-v1",
        "bundle_id": "independent-mcq-2026-09-v1",
        "created_at_utc": _now(),
        "training_snapshot_version": "2026-09-29-v1",
        "source_sets": release_metadata,
        "selected_source_count": len(source_specs),
        "canonical_item_count": canonical_rows,
        "canonical_items_path": merged_items_path.relative_to(EXTERNAL_ROOT).as_posix(),
        "canonical_items_sha256": merged_sha256,
        "mmlu_pro_placeholder_handling": "remove only choices exactly equal to the official N/A sentinel and remap answer indices; aggregate removal counts are in the canonical manifest",
        "belebele_note": "five complete language slices; translated rows share semantic question groups",
        "bbh_note": "six selected explicit-choice task slices; this is not the full benchmark score",
    }
    _atomic_json(bundle_manifest_path, bundle_metadata)

    _status("overlap_audit", canonical_items=canonical_rows)
    overlap_dir = EXTERNAL_ROOT / "overlap-audit"
    overlap_report = overlap.run_overlap_audit(
        SNAPSHOT / "rows.jsonl",
        SNAPSHOT / "manifest.json",
        merged_items_path,
        overlap_dir,
        benchmark_id="independent-mcq-2026-09-v1",
        benchmark_revision="pinned-multi-source-bundle-v1",
        benchmark_items_sha256=merged_sha256,
    )
    exclusions = _exact_training_exclusions(overlap_dir / "exact-match-exclusion-overlay.jsonl")

    overlap_metadata = {
        key: value
        for key, value in overlap_report.items()
        if key not in {"benchmark_items_sha256", "training_rows_sha256", "training_snapshot_manifest_sha256"}
    }
    bundle_metadata["overlap_summary"] = overlap_metadata
    _atomic_json(bundle_manifest_path, bundle_metadata)

    artifact_paths: list[Path] = [bundle_manifest_path, merged_items_path]
    for benchmark_id, revision in source_specs:
        source_dir = STAGING_ROOT / benchmark_id / revision
        artifact_paths.extend(path for path in source_dir.iterdir() if path.is_file())
        canonical = next(
            path for path in canonical_paths
            if path.parent.name == benchmark_id or path.parent.name == benchmark_id.replace("_", "-")
        )
        artifact_paths.extend([canonical, canonical.with_name(f"{canonical.stem}.manifest.json")])
    artifact_paths.extend([
        overlap_dir / "overlap-report.json",
        overlap_dir / "quarantine-training-ids.jsonl",
        overlap_dir / "exact-match-exclusion-overlay.jsonl",
    ])
    lock_path, lock = _write_system_test_lock(
        bundle_metadata=bundle_metadata,
        overlap_metadata=overlap_metadata,
        training_exclusions=exclusions,
        artifact_paths=artifact_paths,
    )

    from scalable_training import validate_system_test_lock

    validate_system_test_lock(lock_path, SNAPSHOT)
    _status(
        "complete",
        source_files=len(source_specs),
        canonical_items=canonical_rows,
        exact_overlap_test_rows=overlap_report["exact_prompt_overlap_test_rows"],
        exact_overlap_training_ids=overlap_report["exact_prompt_overlap_training_ids"],
        exact_overlap_problem_groups=len(exclusions["problem_group_ids"]),
        near_overlap_test_rows=overlap_report["near_lexical_overlap_test_rows"],
        near_overlap_training_ids=overlap_report["near_lexical_overlap_training_ids"],
        lock_path=lock_path.relative_to(ROOT).as_posix(),
        lock_sha256=_sha256(lock_path),
        total_locked_files=len(lock["files"]),
    )
    return {
        "source_files": len(source_specs),
        "canonical_items": canonical_rows,
        "overlap": overlap_report,
        "training_exclusions": exclusions,
        "lock_path": lock_path,
        "lock_sha256": _sha256(lock_path),
    }


def main() -> int:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = prepare()
        safe = {
            "phase": "complete",
            "source_files": result["source_files"],
            "canonical_items": result["canonical_items"],
            "exact_overlap_test_rows": result["overlap"]["exact_prompt_overlap_test_rows"],
            "exact_overlap_training_ids": result["overlap"]["exact_prompt_overlap_training_ids"],
            "near_overlap_test_rows": result["overlap"]["near_lexical_overlap_test_rows"],
            "near_overlap_training_ids": result["overlap"]["near_lexical_overlap_training_ids"],
            "lock_path": result["lock_path"].relative_to(ROOT).as_posix(),
            "lock_sha256": result["lock_sha256"],
        }
        print(json.dumps(safe, sort_keys=True), flush=True)
        return 0
    except isolation.BenchmarkIsolationError as exc:
        _status("failed", safe_error_code=str(exc))
        print(json.dumps({"phase": "failed", "safe_error_code": str(exc)}, sort_keys=True), flush=True)
        return 2
    except Exception:
        _status("failed", safe_error_code="internal_error")
        print(json.dumps({"phase": "failed", "safe_error_code": "internal_error"}, sort_keys=True), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
