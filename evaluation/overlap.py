"""Blind exact and cheap lexical overlap screening against a frozen snapshot."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator, Mapping

from evaluation.isolation import (
    EXTERNAL_DATA_ROOT,
    ROOT,
    BenchmarkIsolationError,
    _canonical_json,
    _normalize_text,
    canonical_prompt_fingerprint,
    sha256_file,
)

TRAINING_SNAPSHOT_ROOT = ROOT / "data" / "training_snapshots"

_MINHASH_MULTIPLIERS = (
    0x9E3779B185EBCA87,
    0xC2B2AE3D27D4EB4F,
    0x165667B19E3779F9,
    0x85EBCA77C2B2AE63,
    0x27D4EB2F165667C5,
    0x94D049BB133111EB,
    0xDB4F0B9175AE2165,
    0xBBE0563303A4615F,
)
_MINHASH_OFFSETS = (
    0x243F6A8885A308D3,
    0x13198A2E03707344,
    0xA4093822299F31D0,
    0x082EFA98EC4E6C89,
    0x452821E638D01377,
    0xBE5466CF34E90C6C,
    0xC0AC29B7C97C50DD,
    0x3F84D5B5B5470917,
)
_MASK_64 = (1 << 64) - 1
_NEAR_THRESHOLD = 0.85
_MAX_NEAR_CANDIDATES = 5000


def _read_rows(path: Path, *, label: str) -> Iterator[dict[str, Any]]:
    try:
        stream = path.open("rb")
    except OSError:
        raise BenchmarkIsolationError(f"{label}_jsonl_could_not_be_opened") from None
    with stream:
        for line_number, encoded in enumerate(stream, 1):
            if not encoded.strip():
                continue
            try:
                record = json.loads(encoded)
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise BenchmarkIsolationError(f"invalid_{label}_json") from None
            if not isinstance(record, dict):
                raise BenchmarkIsolationError(f"{label}_record_not_an_object")
            yield record


def _row_text(row: Mapping[str, Any]) -> str:
    options = row.get("options")
    if not isinstance(options, list) or any(not isinstance(option, str) for option in options):
        raise BenchmarkIsolationError("overlap_options_invalid")
    fields = [str(row.get("context", "")), str(row.get("question", "")), *sorted(options)]
    return _normalize_text(" ".join(fields))


def _shingles(text: str) -> frozenset[str]:
    tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold(), flags=re.UNICODE)
    if not tokens:
        return frozenset()
    if len(tokens) < 3:
        return frozenset(tokens)
    return frozenset("\x1f".join(tokens[index:index + 3]) for index in range(len(tokens) - 2))


def _minhash_signature(features: frozenset[str]) -> tuple[int, ...] | None:
    if not features:
        return None
    signature = [_MASK_64] * len(_MINHASH_MULTIPLIERS)
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        for index, (multiplier, offset) in enumerate(zip(_MINHASH_MULTIPLIERS, _MINHASH_OFFSETS)):
            mixed = (value * multiplier + offset) & _MASK_64
            if mixed < signature[index]:
                signature[index] = mixed
    return tuple(signature)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(dict(value), stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def _atomic_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            encoded = (_canonical_json(dict(row)) + "\n").encode("utf-8")
            stream.write(encoded.decode("utf-8"))
            digest.update(encoded)
    return digest.hexdigest()


def run_overlap_audit(
    training_rows_path: Path,
    training_manifest_path: Path,
    benchmark_items_path: Path,
    output_directory: Path,
    *,
    benchmark_id: str,
    benchmark_revision: str,
    benchmark_items_sha256: str | None = None,
) -> dict[str, Any]:
    """Write aggregate-only overlap metadata and a training-ID quarantine overlay.

    The lexical pass uses a fixed 8-hash MinHash candidate index, followed by
    exact word-shingle Jaccard scoring. Candidate caps are recorded so a bounded
    run cannot be mistaken for a complete semantic contamination check.
    """
    external_root = EXTERNAL_DATA_ROOT.resolve()
    training_root = TRAINING_SNAPSHOT_ROOT.resolve()
    if not output_directory.resolve().is_relative_to(external_root):
        raise BenchmarkIsolationError("overlap_outputs_outside_external_evaluation_root")
    if output_directory.exists():
        raise BenchmarkIsolationError("overlap_audit_directory_is_immutable")
    if not benchmark_items_path.resolve().is_relative_to(external_root):
        raise BenchmarkIsolationError("benchmark_items_outside_external_evaluation_root")
    if (
        not training_rows_path.resolve().is_relative_to(training_root)
        or not training_manifest_path.resolve().is_relative_to(training_root)
    ):
        raise BenchmarkIsolationError("overlap_inputs_must_use_frozen_training_snapshot")
    try:
        snapshot_manifest = json.loads(training_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BenchmarkIsolationError("training_snapshot_manifest_invalid") from None
    expected_rows_hash = (
        snapshot_manifest.get("artifacts", {}).get("rows.jsonl", {}).get("sha256")
        if isinstance(snapshot_manifest, dict)
        else None
    )
    actual_rows_hash = sha256_file(training_rows_path)
    if not isinstance(expected_rows_hash, str) or expected_rows_hash != actual_rows_hash:
        raise BenchmarkIsolationError("training_rows_checksum_does_not_match_snapshot_manifest")
    if not isinstance(snapshot_manifest.get("version"), str) or not snapshot_manifest["version"].strip():
        raise BenchmarkIsolationError("training_snapshot_version_missing")

    train_ids: list[str] = []
    train_texts: list[str] = []
    train_fingerprints: dict[str, list[int]] = defaultdict(list)
    minhash_indexes: list[dict[int, list[int]]] = [defaultdict(list) for _ in _MINHASH_MULTIPLIERS]
    training_count = 0
    for row in _read_rows(training_rows_path, label="training_snapshot"):
        training_count += 1
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            raise BenchmarkIsolationError("training_snapshot_id_invalid")
        text = _row_text(row)
        try:
            fingerprint = canonical_prompt_fingerprint(row)
        except BenchmarkIsolationError:
            raise BenchmarkIsolationError("training_snapshot_prompt_invalid") from None
        row_index = len(train_ids)
        train_ids.append(identifier.strip())
        train_texts.append(text)
        train_fingerprints[fingerprint].append(row_index)
        signature = _minhash_signature(_shingles(text))
        if signature is not None:
            for index, value in enumerate(signature):
                minhash_indexes[index][value].append(row_index)
    if training_count == 0:
        raise BenchmarkIsolationError("training_snapshot_is_empty")

    items_path = benchmark_items_path
    actual_items_hash = sha256_file(items_path)
    if benchmark_items_sha256 is not None and benchmark_items_sha256 != actual_items_hash:
        raise BenchmarkIsolationError("benchmark_items_checksum_mismatch")
    items_digest = actual_items_hash
    exact_train_ids: set[str] = set()
    near_train_ids: set[str] = set()
    exact_test_rows = 0
    near_test_rows = 0
    benchmark_count = 0
    candidate_pairs_checked = 0
    candidate_pairs_truncated = 0
    cached_train_features: dict[int, frozenset[str]] = {}

    for row in _read_rows(items_path, label="benchmark_item"):
        benchmark_count += 1
        if not isinstance(row.get("item_id"), str) or not row["item_id"].strip():
            raise BenchmarkIsolationError("benchmark_item_id_invalid")
        try:
            fingerprint = canonical_prompt_fingerprint(row)
            text = _row_text(row)
        except BenchmarkIsolationError:
            raise BenchmarkIsolationError("benchmark_item_invalid") from None
        exact_indexes = train_fingerprints.get(fingerprint, ())
        if exact_indexes:
            exact_test_rows += 1
            exact_train_ids.update(train_ids[index] for index in exact_indexes)
            continue

        query_features = _shingles(text)
        query_signature = _minhash_signature(query_features)
        if query_signature is None:
            continue
        candidates: set[int] = set()
        for index, value in enumerate(query_signature):
            candidates.update(minhash_indexes[index].get(value, ()))
        ordered_candidates = sorted(candidates)
        if len(ordered_candidates) > _MAX_NEAR_CANDIDATES:
            candidate_pairs_truncated += len(ordered_candidates) - _MAX_NEAR_CANDIDATES
            ordered_candidates = ordered_candidates[:_MAX_NEAR_CANDIDATES]
        matched: list[int] = []
        for train_index in ordered_candidates:
            candidate_pairs_checked += 1
            features = cached_train_features.get(train_index)
            if features is None:
                features = _shingles(train_texts[train_index])
                cached_train_features[train_index] = features
            union_size = len(query_features | features)
            if union_size and len(query_features & features) / union_size >= _NEAR_THRESHOLD:
                matched.append(train_index)
        if matched:
            near_test_rows += 1
            near_train_ids.update(train_ids[index] for index in matched)

    if benchmark_count == 0:
        raise BenchmarkIsolationError("benchmark_item_set_is_empty")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir()
    overlap_types_by_train_id: dict[str, set[str]] = defaultdict(set)
    for identifier in exact_train_ids:
        overlap_types_by_train_id[identifier].add("exact_prompt_overlap")
    for identifier in near_train_ids - exact_train_ids:
        overlap_types_by_train_id[identifier].add("near_lexical_overlap")
    quarantine_rows = [
        {"training_id": identifier, "overlap_types": sorted(types)}
        for identifier, types in sorted(overlap_types_by_train_id.items())
    ]
    exact_overlay_rows = [
        {"training_id": identifier, "exclude_from_future_training_snapshot": True, "reason": "exact_prompt_overlap"}
        for identifier in sorted(exact_train_ids)
    ]
    quarantine_sha256 = _atomic_jsonl(output_directory / "quarantine-training-ids.jsonl", quarantine_rows)
    overlay_sha256 = _atomic_jsonl(output_directory / "exact-match-exclusion-overlay.jsonl", exact_overlay_rows)
    report = {
        "schema_version": "blind-overlap-audit-v1",
        "benchmark_id": benchmark_id,
        "benchmark_revision": benchmark_revision,
        "benchmark_items_sha256": items_digest,
        "training_snapshot_version": snapshot_manifest["version"],
        "training_snapshot_manifest_sha256": sha256_file(training_manifest_path),
        "training_rows_sha256": actual_rows_hash,
        "training_rows_checked": training_count,
        "benchmark_rows_checked": benchmark_count,
        "exact_prompt_overlap_test_rows": exact_test_rows,
        "exact_prompt_overlap_training_ids": len(exact_train_ids),
        "near_lexical_overlap_test_rows": near_test_rows,
        "near_lexical_overlap_training_ids": len(near_train_ids - exact_train_ids),
        "near_method": "8-hash MinHash candidate index; 3-word-shingle Jaccard >= 0.85",
        "near_candidate_pairs_checked": candidate_pairs_checked,
        "near_candidate_pairs_truncated": candidate_pairs_truncated,
        "semantic_overlap_checked": False,
        "pretraining_contamination_ruled_out": False,
        "quarantine_training_ids_sha256": quarantine_sha256,
        "exact_match_exclusion_overlay_sha256": overlay_sha256,
    }
    _atomic_json(output_directory / "overlap-report.json", report)
    return report
