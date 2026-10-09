"""Freeze validated training rows and a deterministic, group-safe split.

Run ``python -m dataset_factory.training_snapshot create VERSION`` from the
repository root. The command reads existing generated questions only; saved
assignments are not training examples.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from dataset_factory import probability_v2, progress


ROOT = Path(__file__).resolve().parents[1]
TRAINING_ROOT = ROOT / "data" / "generated_train"
DEFAULT_SOURCE_PATHS = (
    TRAINING_ROOT / "broad-v1",
    TRAINING_ROOT / "probability-v2" / "questions",
    TRAINING_ROOT / "probability-v2-pilot.jsonl",
    TRAINING_ROOT / "probability-v2-pilot-cardinality-answerability-v1.jsonl",
)
DEFAULT_KNOWN_ISSUES = TRAINING_ROOT / "probability-v2" / "known-issues-v1.jsonl"
SNAPSHOT_ROOT = ROOT / "data" / "training_snapshots"
EXTERNAL_EVALUATION_ROOT = ROOT / "data" / "external_evaluation"
SNAPSHOT_SCHEMA_VERSION = "training-snapshot-v1"
DEFAULT_SPLIT_RATIOS = {"train": 0.90, "dev": 0.05, "test": 0.05}
_LINK_FIELDS = (
    "translation_group_id",
    "paired_context_group_id",
    "paired_group_id",
    "pair_id",
    "translation_of",
    "paired_with_id",
    "parent_question_id",
    "source_row_id",
    "variant_of",
)
_INDEX_FIELDS = frozenset(
    {
        "answer_index",
        "abstention_index",
        "idk_index",
        "unknown_index",
        "cannot_determine_index",
        "correct_choice_index",
        "target_index",
    }
)
_VECTOR_FIELDS = frozenset({"choice_ids", "probabilities", "target_distribution", "counts", "favorable_counts"})


class SnapshotError(ValueError):
    """Raised when a snapshot cannot be made reproducibly and without leakage."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(dict(value), ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    """Write sorted-key JSONL and return its SHA-256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                line = (_canonical_json(dict(row)) + "\n").encode("utf-8")
                stream.write(line)
                digest.update(line)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return digest.hexdigest()


def _source_files(paths: Iterable[Path]) -> list[Path]:
    paths = tuple(Path(path) for path in paths)
    _reject_external_evaluation_sources(paths)
    try:
        return probability_v2._input_files(paths)
    except FileNotFoundError as exc:
        raise SnapshotError(f"training source does not exist: {exc}") from exc


def _reject_external_evaluation_sources(paths: Iterable[Path]) -> None:
    """Keep evaluation-only artifacts out of every snapshot build."""
    lexical_root = EXTERNAL_EVALUATION_ROOT.absolute()
    resolved_root = EXTERNAL_EVALUATION_ROOT.resolve()
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        candidates = [path.absolute(), path.resolve()]
        if not path.is_absolute():
            rooted_path = ROOT / path
            candidates.extend([rooted_path.absolute(), rooted_path.resolve()])
        for candidate in candidates:
            if candidate.is_relative_to(lexical_root) or candidate.is_relative_to(resolved_root):
                raise SnapshotError("external_evaluation data cannot be used as a training source")


def _known_issue_maps(path: Path) -> tuple[dict[str, set[str]], dict[str, list[str]], dict[str, Any]]:
    """Return per-ID flags/details and a hash record for the issue sidecar."""
    if not path.is_file():
        raise SnapshotError(f"known-issue sidecar does not exist: {path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise SnapshotError(f"known-issue sidecar changed while being read: {path}")
    id_flags: dict[str, set[str]] = defaultdict(set)
    id_issue_ids: dict[str, list[str]] = defaultdict(list)
    line_errors: list[dict[str, Any]] = []
    for line_number, encoded_line in enumerate(raw.splitlines(), 1):
        if not encoded_line.strip():
            continue
        try:
            issue = json.loads(encoded_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            line_errors.append({"line": line_number, "reason": f"invalid_known_issue_json:{exc}"})
            continue
        if not isinstance(issue, dict):
            line_errors.append({"line": line_number, "reason": "known_issue_is_not_an_object"})
            continue
        status = issue.get("training_status")
        if status not in {"exclude_pending_review", "review_recommended"}:
            continue
        for identifier in issue.get("question_ids", []):
            if not _nonempty(identifier):
                continue
            flag = "known_issue_excluded" if status == "exclude_pending_review" else "review_recommended"
            id_flags[identifier].add(flag)
            id_issue_ids[identifier].append(str(issue.get("issue_id", "unknown")))
    if line_errors:
        raise SnapshotError(f"known-issue sidecar contains malformed lines: {line_errors[:3]}")
    return dict(id_flags), dict(id_issue_ids), {
        "path": path.resolve().relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "sha256": _sha256(raw),
        "size_bytes": len(raw),
        "mtime_ns": after.st_mtime_ns,
    }


def _decode_row(encoded_line: bytes) -> tuple[Any | None, str | None]:
    try:
        return json.loads(encoded_line), None
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{exc}"


def _row_schema_kind(row: Mapping[str, Any]) -> str:
    return "legacy" if "schema_version" not in row else str(row.get("schema_version"))


def _target_signature(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Represent labels by option text so answer-order variants compare safely."""
    options = row.get("options")
    if not isinstance(options, list):
        return []
    normalized_options = [progress.normalize(str(option)) for option in options]
    supervision = row.get("supervision_kind")
    if supervision == probability_v2.HARD_LABEL or "schema_version" not in row:
        answer_index = row.get("answer_index")
        if isinstance(answer_index, int) and not isinstance(answer_index, bool) and 0 <= answer_index < len(options):
            return sorted((option, "1" if index == answer_index else "0") for index, option in enumerate(normalized_options))
        return []
    target = row.get("target_distribution")
    if isinstance(target, list) and len(target) == len(options):
        return sorted((option, repr(float(target[index]))) for index, option in enumerate(normalized_options))
    return []


def exact_content_fingerprint(row: Mapping[str, Any]) -> str:
    """Hash normalized prompt content and labels, ignoring volatile metadata."""
    options = row.get("options") if isinstance(row.get("options"), list) else []
    content = {
        "context": progress.normalize(str(row.get("context", ""))),
        "question": progress.normalize(str(row.get("question", ""))),
        "options_and_targets": _target_signature(row)
        or sorted(progress.normalize(str(option)) for option in options),
    }
    return _sha256(_canonical_json(content).encode("utf-8"))


def _context_fingerprint(row: Mapping[str, Any]) -> str:
    return _sha256(progress.normalize(str(row.get("context", ""))).encode("utf-8"))


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def _related_keys(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    group_id = row.get("problem_group_id")
    if _nonempty(group_id):
        keys.append(("problem_group_id", group_id.strip()))
    for field in _LINK_FIELDS:
        value = row.get(field)
        if _nonempty(value):
            keys.append((field, value.strip()))
    augmentation = row.get("_augmentation")
    if isinstance(augmentation, Mapping) and _nonempty(augmentation.get("parent_id")):
        keys.append(("row_id", augmentation["parent_id"].strip()))
    if _nonempty(row.get("id")):
        keys.append(("row_id", row["id"].strip()))
    keys.append(("exact_context", _context_fingerprint(row)))
    keys.append(("exact_content", exact_content_fingerprint(row)))
    return keys


def _stable_split(group_ids: Sequence[str], seed: int, ratios: Mapping[str, float]) -> str:
    material = f"training-snapshot-split-v1\0{seed}\0{'\0'.join(group_ids)}".encode("utf-8")
    fraction = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") / 2**64
    train_end = ratios["train"]
    dev_end = train_end + ratios["dev"]
    return "train" if fraction < train_end else "dev" if fraction < dev_end else "test"


def _validate_ratios(ratios: Mapping[str, float]) -> dict[str, float]:
    if set(ratios) != {"train", "dev", "test"}:
        raise SnapshotError("split_ratios must define exactly train, dev, and test")
    result: dict[str, float] = {}
    for name in ("train", "dev", "test"):
        value = ratios[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
            raise SnapshotError(f"split ratio {name!r} must be between zero and one")
        result[name] = float(value)
    if abs(sum(result.values()) - 1.0) > 1e-12:
        raise SnapshotError("split ratios must sum to one")
    return result


def _exclusion_entry(
    *, path: Path, line_number: int, row: Mapping[str, Any] | None,
    categories: Sequence[str], reasons: Sequence[str], issue_ids: Sequence[str] = (),
) -> dict[str, Any]:
    return {
        "id": row.get("id") if row else None,
        "problem_group_id": row.get("problem_group_id") if row else None,
        "source_file": path.resolve().relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "source_line": line_number,
        "categories": sorted(set(categories)),
        "reasons": sorted(set(reasons)),
        "known_issue_ids": sorted(set(issue_ids)),
    }


def _snapshot_group_id(row_ids: Sequence[str]) -> str:
    digest = _sha256("\0".join(sorted(row_ids)).encode("utf-8"))[:20]
    return f"snapshot-group-{digest}"


def _load_previous_assignments(path: Path) -> tuple[dict[tuple[str, str], set[str]], str]:
    """Load only the immutable row, split, and manifest artifacts of an allowed snapshot."""
    path = Path(path).expanduser().resolve()
    try:
        path.relative_to(SNAPSHOT_ROOT.resolve())
    except ValueError as exc:
        raise SnapshotError("previous snapshot must be inside data/training_snapshots") from exc
    if not path.is_dir():
        raise SnapshotError(f"previous snapshot directory does not exist: {path}")
    for name in ("manifest.json", "rows.jsonl", "splits.jsonl"):
        artifact_path = path / name
        if artifact_path.is_symlink():
            raise SnapshotError(f"previous snapshot artifact must not be a symlink: {name}")
        try:
            artifact_path.resolve().relative_to(path)
        except ValueError as exc:
            raise SnapshotError(f"previous snapshot artifact escapes its directory: {name}") from exc
    try:
        manifest_bytes = (path / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        rows_bytes = (path / "rows.jsonl").read_bytes()
        splits_bytes = (path / "splits.jsonl").read_bytes()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError("previous snapshot is missing or has malformed required artifacts") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SNAPSHOT_SCHEMA_VERSION:
        raise SnapshotError("previous snapshot manifest has an unsupported schema")
    artifacts = manifest.get("artifacts", {})
    for name, content in (("rows.jsonl", rows_bytes), ("splits.jsonl", splits_bytes)):
        record = artifacts.get(name, {})
        if not isinstance(record, dict) or _sha256(content) != record.get("sha256"):
            raise SnapshotError(f"previous snapshot {name} hash does not match its manifest")
    try:
        rows = [json.loads(line) for line in rows_bytes.splitlines() if line.strip()]
        splits = [json.loads(line) for line in splits_bytes.splitlines() if line.strip()]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SnapshotError("previous snapshot rows or splits are malformed") from exc
    split_by_id: dict[str, str] = {}
    for item in splits:
        if not isinstance(item, dict) or not _nonempty(item.get("id")) or item.get("split") not in {"train", "dev", "test"}:
            raise SnapshotError("previous snapshot contains an invalid split record")
        identifier = item["id"]
        if identifier in split_by_id:
            raise SnapshotError("previous snapshot contains duplicate split IDs")
        split_by_id[identifier] = item["split"]
    assignments: dict[tuple[str, str], set[str]] = defaultdict(set)
    row_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or not _nonempty(row.get("id")):
            raise SnapshotError("previous snapshot contains an invalid row")
        identifier = row["id"]
        split = row.get("_training_snapshot", {}).get("split")
        if split not in {"train", "dev", "test"} or split_by_id.get(identifier) != split:
            raise SnapshotError("previous snapshot rows and split assignments disagree")
        if identifier in row_ids:
            raise SnapshotError("previous snapshot contains duplicate row IDs")
        row_ids.add(identifier)
        for key in _related_keys(row):
            assignments[key].add(split)
    if row_ids != set(split_by_id):
        raise SnapshotError("previous snapshot rows and split IDs do not match")
    return dict(assignments), _sha256(manifest_bytes)


def _write_frozen_directory(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise SnapshotError(f"snapshot already exists and is immutable: {destination}")
    os.replace(staging, destination)
    directory_fd = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def build_snapshot(
    version: str,
    *,
    source_paths: Sequence[Path] = DEFAULT_SOURCE_PATHS,
    known_issues_path: Path = DEFAULT_KNOWN_ISSUES,
    output_root: Path = SNAPSHOT_ROOT,
    seed: int = 20260929,
    split_ratios: Mapping[str, float] = DEFAULT_SPLIT_RATIOS,
    previous_snapshot: Path | None = None,
) -> dict[str, Any]:
    """Build a frozen normalized corpus without changing any source file."""
    if not isinstance(version, str) or not version.strip() or "/" in version or "\\" in version:
        raise SnapshotError("version must be a non-empty directory name")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SnapshotError("seed must be an integer")
    ratios = _validate_ratios(split_ratios)
    previous_assignments: dict[tuple[str, str], set[str]] = {}
    previous_manifest_hash: str | None = None
    if previous_snapshot is not None:
        previous_assignments, previous_manifest_hash = _load_previous_assignments(previous_snapshot)
    destination = output_root / version
    if destination.exists():
        raise SnapshotError(f"snapshot already exists and is immutable: {destination}")

    source_files = _source_files(source_paths)
    issue_flags, issue_ids, issue_record = _known_issue_maps(known_issues_path)
    before_stats = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in source_files}
    input_hashes: dict[str, dict[str, Any]] = {}
    raw_rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    observed_group_splits: dict[str, set[str]] = defaultdict(set)
    source_lines = 0

    for path in source_files:
        digest = hashlib.sha256()
        file_size = 0
        relative_path = path.resolve().relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
        with path.open("rb") as stream:
            for line_number, encoded_line in enumerate(stream, 1):
                digest.update(encoded_line)
                file_size += len(encoded_line)
                if not encoded_line.strip():
                    continue
                source_lines += 1
                row, parse_error = _decode_row(encoded_line)
                if parse_error:
                    exclusions.append(_exclusion_entry(
                        path=path, line_number=line_number, row=None,
                        categories=["structural_invalid"], reasons=[parse_error],
                    ))
                    continue
                if not isinstance(row, dict):
                    exclusions.append(_exclusion_entry(
                        path=path, line_number=line_number, row=None,
                        categories=["structural_invalid"], reasons=["row_is_not_an_object"],
                    ))
                    continue

                identifier = row.get("id")
                known_flags = issue_flags.get(identifier, set()) if _nonempty(identifier) else set()
                known_statuses = [flag for flag in known_flags if flag == "known_issue_excluded"]
                split = row.get("split")
                group_id = row.get("problem_group_id")
                if _nonempty(group_id) and _nonempty(split):
                    observed_group_splits[group_id.strip()].add(split.strip())

                source_kind, errors, schema = probability_v2.classify_row(row)
                if errors or source_kind is None:
                    categories = ["structural_invalid"]
                    if known_statuses:
                        categories.append("known_issue_excluded")
                    exclusions.append(_exclusion_entry(
                        path=path, line_number=line_number, row=row,
                        categories=categories,
                        reasons=[*errors, *("known_issue_exclude_pending_review" for _ in known_statuses)],
                        issue_ids=issue_ids.get(identifier, ()) if _nonempty(identifier) else (),
                    ))
                    continue

                record = {
                    "row": copy.deepcopy(row),
                    "id": row["id"].strip(),
                    "problem_group_id": row["problem_group_id"].strip(),
                    "source_kind": source_kind,
                    "schema_kind": schema or _row_schema_kind(row),
                    "source_file": relative_path,
                    "source_line": line_number,
                    "known_flags": sorted(known_flags),
                    "known_issue_ids": sorted(issue_ids.get(identifier, ())) if _nonempty(identifier) else [],
                }
                raw_rows.append(record)
        after = path.stat()
        if before_stats[path] != (after.st_size, after.st_mtime_ns) or file_size != after.st_size:
            raise SnapshotError(f"training source changed while being read: {path}")
        input_hashes[relative_path] = {
            "sha256": digest.hexdigest(),
            "size_bytes": file_size,
            "mtime_ns": after.st_mtime_ns,
        }

    # Match the project's canonical acceptance summary before applying training
    # exclusions and exact-content deduplication.
    canonical = probability_v2.summarize(source_files)
    for path, expected_stat in before_stats.items():
        current_stat = path.stat()
        if expected_stat != (current_stat.st_size, current_stat.st_mtime_ns):
            raise SnapshotError(f"training source changed during canonical validation: {path}")
    issue_stat = known_issues_path.stat()
    if (issue_stat.st_size, issue_stat.st_mtime_ns) != (issue_record["size_bytes"], issue_record["mtime_ns"]):
        raise SnapshotError(f"known-issue sidecar changed during snapshot construction: {known_issues_path}")
    candidate_by_id: dict[str, dict[str, Any]] = {}
    valid_duplicate_ids = 0
    for record in raw_rows:
        identifier = record["id"]
        if identifier in candidate_by_id:
            valid_duplicate_ids += 1
            exclusions.append(_exclusion_entry(
                path=Path(record["source_file"]), line_number=record["source_line"],
                row=record["row"], categories=["duplicate_id"],
                reasons=["duplicate_row_id"], issue_ids=record["known_issue_ids"],
            ))
            continue
        candidate_by_id[identifier] = record

    # Drop every otherwise valid row in a group whose original source splits
    # disagree, matching probability_v2.summarize's no-cross-split rule.
    candidates = list(candidate_by_id.values())
    filtered_candidates: list[dict[str, Any]] = []
    conflicting_group_ids = {group for group, splits in observed_group_splits.items() if len(splits) > 1}
    group_conflict_rows = 0
    for record in candidates:
        if record["problem_group_id"] in conflicting_group_ids:
            group_conflict_rows += 1
            categories = ["structural_invalid", "group_split_conflict"]
            if "known_issue_excluded" in record["known_flags"]:
                categories.append("known_issue_excluded")
            exclusions.append(_exclusion_entry(
                path=Path(record["source_file"]), line_number=record["source_line"],
                row=record["row"], categories=categories,
                reasons=["problem_group_spans_source_splits"], issue_ids=record["known_issue_ids"],
            ))
            continue
        filtered_candidates.append(record)

    production_before_manual_exclusions = sum(
        record["row"].get("pilot_status") != probability_v2.EXCLUDED_PILOT_STATUS
        for record in filtered_candidates
    )
    canonical_production = canonical["production_eligible_rows"]
    if (
        len(filtered_candidates) != canonical["accepted_rows"]
        or len(candidates) != canonical["accepted_rows"] + group_conflict_rows
    ):
        raise SnapshotError(
            "row extraction disagrees with probability_v2.summarize: "
            f"extracted={len(candidates)}, summary={canonical['accepted_rows']}, "
            f"group_conflicts={group_conflict_rows}"
        )
    if production_before_manual_exclusions != canonical_production:
        raise SnapshotError(
            "production row count disagrees with probability_v2.summarize: "
            f"extracted={production_before_manual_exclusions}, summary={canonical_production}"
        )

    # Exclude pilot and pending-review rows from mass training, while keeping
    # review-recommended rows in the corpus as a visible cohort.
    production_rows: list[dict[str, Any]] = []
    for record in filtered_candidates:
        row = record["row"]
        categories: list[str] = []
        reasons: list[str] = []
        if row.get("pilot_status") == probability_v2.EXCLUDED_PILOT_STATUS:
            categories.append("pilot_excluded")
            reasons.append("illustrative_pilot_not_mass_training")
        if "known_issue_excluded" in record["known_flags"]:
            categories.append("known_issue_excluded")
            reasons.append("known_issue_exclude_pending_review")
        if categories:
            exclusions.append(_exclusion_entry(
                path=Path(record["source_file"]), line_number=record["source_line"],
                row=row, categories=categories, reasons=reasons,
                issue_ids=record["known_issue_ids"],
            ))
            continue
        if "review_recommended" in record["known_flags"]:
            record["cohorts"] = ["review_recommended"]
        else:
            record["cohorts"] = []
        record["content_fingerprint"] = exact_content_fingerprint(row)
        production_rows.append(record)

    # Deduplicate equivalent source content even when IDs differ. Keep the
    # first by stable source ordering, and report every discarded source row.
    content_seen: dict[str, dict[str, Any]] = {}
    eligible: list[dict[str, Any]] = []
    duplicate_content_rows = 0
    for record in production_rows:
        fingerprint = record["content_fingerprint"]
        previous = content_seen.get(fingerprint)
        if previous is not None:
            previous["cohorts"] = sorted(set(previous["cohorts"]) | set(record["cohorts"]))
            duplicate_content_rows += 1
            exclusions.append(_exclusion_entry(
                path=Path(record["source_file"]), line_number=record["source_line"],
                row=record["row"], categories=["duplicate_exact_content"],
                reasons=["same_normalized_prompt_options_and_target"],
                issue_ids=record["known_issue_ids"],
            ))
            continue
        content_seen[fingerprint] = record
        eligible.append(record)

    # Union all explicitly known pairings, exact contexts, and exact content.
    # This catches answer-order copies and paired questions on identical context.
    union_find = _UnionFind(len(eligible))
    key_owner: dict[tuple[str, str], int] = {}
    for index, record in enumerate(eligible):
        for key in _related_keys(record["row"]):
            previous = key_owner.setdefault(key, index)
            union_find.union(previous, index)
    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in range(len(eligible)):
        members_by_root[union_find.find(index)].append(index)

    group_for_index: dict[int, str] = {}
    split_for_group: dict[str, str] = {}
    preservation = {"groups_preserved": 0, "groups_preserved_as_train": 0,
                    "groups_promoted_away_from_train": 0, "historical_dev_to_test_conflicts": 0}
    for member_indices in members_by_root.values():
        ids = sorted(eligible[index]["id"] for index in member_indices)
        group_id = _snapshot_group_id(ids)
        historical_splits: set[str] = set()
        if previous_assignments:
            for index in member_indices:
                for key in _related_keys(eligible[index]["row"]):
                    historical_splits.update(previous_assignments.get(key, ()))
        if historical_splits:
            split = next(name for name in ("test", "dev", "train") if name in historical_splits)
            preservation["groups_preserved"] += 1
            preservation["groups_preserved_as_train"] += split == "train"
            preservation["groups_promoted_away_from_train"] += "train" in historical_splits and split != "train"
            preservation["historical_dev_to_test_conflicts"] += "dev" in historical_splits and "test" in historical_splits
        else:
            split = _stable_split(ids, seed, ratios)
        split_for_group[group_id] = split
        for index in member_indices:
            group_for_index[index] = group_id

    normalized_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    for index, record in enumerate(eligible):
        row = record["row"]
        group_id = group_for_index[index]
        split = split_for_group[group_id]
        normalized = copy.deepcopy(row)
        if record["schema_kind"] == "legacy":
            normalized = probability_v2.build_hard_row(normalized)
        normalized["_training_snapshot"] = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "source_file": record["source_file"],
            "source_line": record["source_line"],
            "source_schema_kind": record["schema_kind"],
            "source_kind": record["source_kind"],
            "target_kind": "hard_onehot" if record["source_kind"] == probability_v2.HARD_LABEL else "soft_full_distribution",
            "snapshot_group_id": group_id,
            "split": split,
            "source_split": row.get("split"),
            "cohorts": list(record["cohorts"]),
        }
        normalized_rows.append(normalized)
        split_rows.append({
            "id": record["id"],
            "problem_group_id": record["problem_group_id"],
            "snapshot_group_id": group_id,
            "split": split,
            "source_split": row.get("split"),
            "source_kind": record["source_kind"],
            "cohorts": list(record["cohorts"]),
        })

    normalized_rows.sort(key=lambda row: (row["id"], row["_training_snapshot"]["source_file"]))
    split_rows.sort(key=lambda row: row["id"])
    exclusions.sort(key=lambda item: (str(item.get("id") or ""), item["source_file"], item["source_line"], item["categories"]))

    split_counts = Counter(row["_training_snapshot"]["split"] for row in normalized_rows)
    source_counts = Counter(row["_training_snapshot"]["source_kind"] for row in normalized_rows)
    language_counts = Counter(progress._language_tag(row) or "unknown" for row in normalized_rows)
    cohort_counts = Counter(
        cohort for row in normalized_rows for cohort in row["_training_snapshot"]["cohorts"]
    )
    clean_subset_count = len(normalized_rows) - cohort_counts["review_recommended"]
    exclusion_counts = Counter(category for item in exclusions for category in item["categories"])
    intersections = Counter(tuple(item["categories"]) for item in exclusions if len(item["categories"]) > 1)
    accepted_known_issue_ids = {
        record["id"] for record in filtered_candidates if "known_issue_excluded" in record["known_flags"]
    }
    accepted_review_ids = {
        record["id"] for record in production_rows if "review_recommended" in record["known_flags"]
    }
    known_issue_id_counts = {
        "exclude_pending_review": sum("known_issue_excluded" in flags for flags in issue_flags.values()),
        "review_recommended": sum("review_recommended" in flags for flags in issue_flags.values()),
    }
    known_issue_intersection = {
        "pending_review_ids_in_structurally_accepted_rows": sum(
            "known_issue_excluded" in record["known_flags"] and record["row"].get("pilot_status") != probability_v2.EXCLUDED_PILOT_STATUS
            for record in filtered_candidates
        ),
        "review_recommended_ids_in_training_rows_before_content_dedup": len(accepted_review_ids),
    }
    source_hashes = dict(sorted(input_hashes.items()))
    source_hashes[issue_record["path"]] = {
        "sha256": issue_record["sha256"],
        "size_bytes": issue_record["size_bytes"],
        "mtime_ns": issue_record["mtime_ns"],
        "role": "known_issue_sidecar",
    }

    staging: Path | None = None
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        staging = Path(tempfile.mkdtemp(prefix=f".{version}.", dir=output_root))
        row_hash = _atomic_jsonl(staging / "rows.jsonl", normalized_rows)
        split_hash = _atomic_jsonl(staging / "splits.jsonl", split_rows)
        exclusion_hash = _atomic_jsonl(staging / "excluded.jsonl", exclusions)
        manifest: dict[str, Any] = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "seed": seed,
            "split_ratios": ratios,
            "split_assignment": "SHA-256 of seed and complete sorted group member IDs; groups stay in one split",
            "previous_snapshot": None if previous_snapshot is None else {
                "path": Path(previous_snapshot).resolve().relative_to(SNAPSHOT_ROOT.resolve()).as_posix(),
                "manifest_sha256": previous_manifest_hash,
                "preservation_audit": preservation,
            },
            "grouping": {
                "included_links": ["problem_group_id", *_LINK_FIELDS, "_augmentation.parent_id", "exact normalized context", "exact normalized prompt/options/target"],
                "semantic_duplicates": "unknown translations and paraphrases without explicit links are not guaranteed to be detected",
            },
            "source_files": source_hashes,
            "known_issue_sidecar": {
                "path": issue_record["path"],
                "sha256": issue_record["sha256"],
                "unique_question_ids_by_training_status": dict(sorted(known_issue_id_counts.items())),
                "intersection_counts": known_issue_intersection,
            },
            "canonical_acceptance": {
                "summary_schema_version": canonical["schema_version"],
                "structurally_accepted_including_pilot": canonical["accepted_rows"],
                "production_eligible_excluding_pilot": canonical["production_eligible_rows"],
                "legacy_hard_label_rows_in_accepted_source": canonical["legacy_hard_label_rows"],
                "canonical_rejected_rows_including_duplicates_and_group_conflicts": canonical["rejected_rows"],
                "group_split_conflicts": canonical["problem_group_split_conflicts"],
                "malformed_json_rows": canonical["malformed_json_rows"],
                "summary_disclaimer": canonical["disclaimer"],
            },
            "rows": {
                "source_lines": source_lines,
                "structurally_accepted_rows": len(filtered_candidates),
                "production_rows_before_known_issue_exclusions": production_before_manual_exclusions,
                "after_known_issue_and_pilot_exclusions": len(production_rows),
                "exact_content_deduplicated": duplicate_content_rows,
                "training_eligible_rows": len(normalized_rows),
                "by_split": dict(sorted(split_counts.items())),
                "by_source_kind": dict(sorted(source_counts.items())),
                "by_language": dict(sorted(language_counts.items())),
                "review_recommended_rows": cohort_counts["review_recommended"],
                "without_review_recommended_rows": clean_subset_count,
                "target_representation": {
                    "hard_label": "answer_index, tagged target_kind=hard_onehot; no soft target_distribution is synthesized",
                    "soft_distribution": "source target_distribution retained in option order, tagged target_kind=soft_full_distribution",
                },
            },
            "exclusions": {
                "sidecar_rows": len(exclusions),
                "counts_by_category": dict(sorted(exclusion_counts.items())),
                "category_intersections": {
                    "+".join(categories): count for categories, count in sorted(intersections.items())
                },
                "duplicate_ids_dropped": valid_duplicate_ids,
                "exact_content_dropped": duplicate_content_rows,
                "known_issue_accepted_id_intersection": len(accepted_known_issue_ids),
            },
            "artifacts": {
                "rows.jsonl": {"sha256": row_hash, "records": len(normalized_rows)},
                "splits.jsonl": {"sha256": split_hash, "records": len(split_rows)},
                "excluded.jsonl": {"sha256": exclusion_hash, "records": len(exclusions)},
            },
            "semantic_quality": "not reviewed; schema and distribution arithmetic only",
        }
        _atomic_json(staging / "manifest.json", manifest)
        _write_frozen_directory(staging, destination)
        staging = None
        return manifest
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def permute_choices(row: Mapping[str, Any], seed: int | str, *,
                    validator: Callable[[dict[str, Any]], list[str]] | None = None) -> dict[str, Any]:
    """Return a copy with every choice-aligned target/evidence field permuted.

    The supplied seed is combined with the stable row ID so each item gets a
    deterministic but independent order in an epoch. Hard labels remain
    index-only; soft targets remain full distributions.
    """
    if not isinstance(row, Mapping):
        raise SnapshotError("row must be a mapping")
    copied = copy.deepcopy(dict(row))
    validate = validator or (lambda value: probability_v2.classify_row(value)[1])
    errors = validate(copied)
    if errors:
        raise SnapshotError("cannot permute an invalid training row: " + "; ".join(errors[:3]))
    options = copied.get("options")
    if not isinstance(options, list) or len(options) < 2:
        raise SnapshotError("row.options must contain at least two choices")
    if "choice_ids" not in copied:
        copied["choice_ids"] = probability_v2.stable_choice_ids(str(copied["id"]), options)
    choice_ids = copied["choice_ids"]
    if not isinstance(choice_ids, list) or len(choice_ids) != len(options):
        raise SnapshotError("row.choice_ids must align with row.options")

    old_positions = list(range(len(options)))
    rng = random.Random(f"{seed}\0{copied['id']}")
    rng.shuffle(old_positions)  # new position -> old position
    new_position = {old_position: index for index, old_position in enumerate(old_positions)}

    def reorder_evidence(value: Any, key: str = "") -> Any:
        if isinstance(value, list):
            if key in _VECTOR_FIELDS and len(value) == len(options):
                return [copy.deepcopy(value[old_index]) for old_index in old_positions]
            return [reorder_evidence(item) for item in value]
        if isinstance(value, dict):
            result: dict[str, Any] = {}
            for child_key, child_value in value.items():
                if child_key in _INDEX_FIELDS and isinstance(child_value, int) and not isinstance(child_value, bool):
                    result[child_key] = new_position.get(child_value, child_value)
                else:
                    result[child_key] = reorder_evidence(child_value, child_key)
            return result
        return copy.deepcopy(value)

    copied["options"] = [copy.deepcopy(options[index]) for index in old_positions]
    copied["choice_ids"] = [copy.deepcopy(choice_ids[index]) for index in old_positions]
    answer_index = copied.get("answer_index")
    if isinstance(answer_index, int) and not isinstance(answer_index, bool):
        copied["answer_index"] = new_position[answer_index]
    for key in _INDEX_FIELDS - {"answer_index"}:
        value = copied.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            copied[key] = new_position.get(value, value)
    target = copied.get("target_distribution")
    if isinstance(target, list) and len(target) == len(options):
        copied["target_distribution"] = [copy.deepcopy(target[index]) for index in old_positions]
    evidence = copied.get("distribution_evidence")
    if isinstance(evidence, dict):
        copied["distribution_evidence"] = reorder_evidence(evidence)
    errors = validate(copied)
    if errors:
        raise SnapshotError("permutation broke the training row contract: " + "; ".join(errors[:3]))
    return copied


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create", help="freeze the currently accepted training corpus")
    create.add_argument("version", help="new immutable directory name under data/training_snapshots")
    create.add_argument("--seed", type=int, default=20260929)
    create.add_argument("--output-root", type=Path, default=SNAPSHOT_ROOT)
    create.add_argument("--previous-snapshot", type=Path, help="preserve assignments from an immutable snapshot under data/training_snapshots")
    args = parser.parse_args(argv)
    manifest = build_snapshot(args.version, seed=args.seed, output_root=args.output_root,
                              previous_snapshot=args.previous_snapshot)
    print(json.dumps({
        "snapshot": str(args.output_root / args.version),
        "training_eligible_rows": manifest["rows"]["training_eligible_rows"],
        "by_split": manifest["rows"]["by_split"],
        "by_source_kind": manifest["rows"]["by_source_kind"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
