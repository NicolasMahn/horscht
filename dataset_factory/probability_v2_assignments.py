"""Plan resumable probability-v2 training assignments without writing questions.

Create plans with ``python -m dataset_factory.probability_v2_assignments plan``.
Each shard is pinned to one language and one supervision source kind.
"""

from __future__ import annotations

import argparse
import fcntl
import functools
import hashlib
import json
import os
import random
import re
import sqlite3
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator, Mapping

from dataset_factory import assignments, probability_v2, training_snapshot


ROOT = Path(__file__).resolve().parents[1]
TRAINING_DIR = ROOT / "data" / "generated_train"
CATALOG = TRAINING_DIR / "probability-v2-catalog-v1.json"
WORD_POOL = TRAINING_DIR / "seed-word-pool.jsonl.gz"
V2_ROOT = TRAINING_DIR / "probability-v2"
ASSIGNMENT_DIR = V2_ROOT / "assignments"
QUESTION_DIR = V2_ROOT / "questions"
LEGACY_DIR = TRAINING_DIR / "broad-v1"
PILOT_FILE = TRAINING_DIR / "probability-v2-pilot.jsonl"
ASSIGNMENT_SCHEMA_VERSION = "probability-v2-assignment-v1"
AUTHORING_GUIDANCE_VERSION = "probability-v2-semantic-grounding-v2"
CHOICE_COUNT_POLICY_FILE = TRAINING_DIR / "probability-v2-cardinality-policy-v1.json"
_CARDINALITY_POLICY = json.loads(CHOICE_COUNT_POLICY_FILE.read_text(encoding="utf-8"))
CHOICE_COUNT_POLICY_VERSION = _CARDINALITY_POLICY["choice_count_policy_version"]
CHOICE_COUNT_POLICY = _CARDINALITY_POLICY["tier_counts"]
CHOICE_COUNT_TIER_WEIGHTS = _CARDINALITY_POLICY["tier_weights"]
MEDIUM_CARDINALITY_OPERATIONS = frozenset(_CARDINALITY_POLICY["medium_operations"])
HIGH_CARDINALITY_OPERATIONS = frozenset(_CARDINALITY_POLICY["high_operations"])
MEDIUM_CARDINALITY_STRUCTURES = frozenset(_CARDINALITY_POLICY["medium_input_structures"])
HIGH_CARDINALITY_STRUCTURES = frozenset(_CARDINALITY_POLICY["high_input_structures"])
ANSWERABILITY_POLICY_FILE = TRAINING_DIR / "probability-v2-answerability-policy-v1.json"
_ANSWERABILITY_POLICY = json.loads(ANSWERABILITY_POLICY_FILE.read_text(encoding="utf-8"))
ANSWERABILITY_POLICY_VERSION = _ANSWERABILITY_POLICY["answerability_policy_version"]
ANSWERABILITY_MIX_PERCENT = _ANSWERABILITY_POLICY["default_mix_percent"]
UNANSWERABLE_REASONS = tuple(_ANSWERABILITY_POLICY["unanswerable_reasons"])
ABSTENTION_CHOICE_GUIDANCE = _ANSWERABILITY_POLICY["abstention_choice_guidance"]
if (
    set(ANSWERABILITY_MIX_PERCENT) != {"answerable", "unanswerable"}
    or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in ANSWERABILITY_MIX_PERCENT.values()
    )
    or sum(ANSWERABILITY_MIX_PERCENT.values()) != 100
    or set(UNANSWERABLE_REASONS) != probability_v2.UNANSWERABLE_REASONS
):
    raise RuntimeError("invalid probability-v2 answerability policy")
MAX_BATCH = 10_000
MAX_PLAN_BATCH_ROWS = 100_000
MIN_AVAILABLE_MEMORY = 12 * 1024**3
ACCOUNTING_VERSION = "probability-v2-usable-originals-v1"
TOTAL_TRAIN_TARGET = 1_000_000
SOURCE_KINDS = (
    "exact_derived",
    "explicit_stochastic_process",
    "empirical_frequency",
    "teacher_estimate",
    probability_v2.HARD_LABEL,
)
# The accepted mix gives a combined 600k exact/process allocation; an even split
# keeps each single-kind shard quota explicit and easy to change later.
DEFAULT_SOURCE_TARGETS = {
    "exact_derived": 300_000,
    "explicit_stochastic_process": 300_000,
    "empirical_frequency": 100_000,
    "teacher_estimate": 100_000,
    probability_v2.HARD_LABEL: 200_000,
}
ACTIVE_STATUSES = frozenset({"assigned", "in_flight", "pending"})
KNOWN_STATUSES = frozenset({"assigned", "in_flight", "pending", "completed", "skipped", "failed", "superseded"})
PROGRESS_FILE = V2_ROOT / "progress.json"
DEV_PROFILE_NAME = "dev-guidance-v3"
DEV_PROFILE_LIMIT = 4_096
DEV_PROFILE_BANDS = {
    "2-4": (2, 4, 70), "5-16": (5, 16, 20),
    "17-64": (17, 64, 8), "65-265": (65, 265, 2),
}
DEV_PROFILE_OPERATION_WEIGHTS = {
    "policy_exceptions": 1.5, "graph_kinship": 1.5,
    "tradeoff_optimization": 1.5, "procedure_recovery": 1.5,
}
MULTIPLE_SUPPORT_REQUIREMENT = "multiple_substantive_support-v1"
MULTIPLE_SUPPORT_PROFILE = "multiple-substantive-support-v1-pilot"
MULTIPLE_SUPPORT_START = 128_513
MULTIPLE_SUPPORT_FOLLOW_ON_PREFIX = "multiple-substantive-support-v1-follow-on-"
PROFILE_LIMITS = {DEV_PROFILE_NAME: DEV_PROFILE_LIMIT, MULTIPLE_SUPPORT_PROFILE: 1_024}
DISTRIBUTION_TASK_FORMS = {
    "finite_population_distribution": {
        "source_kinds": ["exact_derived", "empirical_frequency"],
        "operations": ["taxonomy_outliers", "policy_exceptions"],
        "input_structures": ["inventory_ledger", "table_or_grid", "taxonomy_key", "rule_excerpt"],
    },
    "stochastic_outcome_distribution": {
        "source_kinds": ["explicit_stochastic_process"],
        "operations": ["game_sport_rules", "physical_mechanisms"],
        "input_structures": ["state_transition_table", "table_or_grid", "rule_excerpt"],
    },
    "estimated_hypothesis_distribution": {
        "source_kinds": ["teacher_estimate"],
        "operations": ["argument_evidence", "experiment_design"],
        "input_structures": ["claim_evidence_bundle", "measurement_sheet", "multi_document_packet"],
    },
}


class ContractError(ValueError):
    """Raised when a planning request conflicts with the durable v2 contract."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _support_profile_start(name: Any) -> int | None:
    if name == MULTIPLE_SUPPORT_PROFILE:
        return MULTIPLE_SUPPORT_START
    if not isinstance(name, str):
        return None
    match = re.fullmatch(re.escape(MULTIPLE_SUPPORT_FOLLOW_ON_PREFIX) + r"([1-9][0-9]*)", name)
    return int(match[1]) if match and int(match[1]) >= 129_537 else None


def _planning_profile_limit(name: Any) -> int | None:
    if not isinstance(name, str):
        return None
    return PROFILE_LIMITS.get(name, 4_096 if _support_profile_start(name) is not None else None)


def _jsonl_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        if path.is_file():
            if path.suffix.lower() == ".jsonl":
                files.append(path)
        elif path.is_dir():
            files.extend(sorted(file for file in path.rglob("*.jsonl") if file.is_file()))
    return sorted(set(files))


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            result = []
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
                if not isinstance(row, dict):
                    raise ContractError(f"expected an object at {path}:{line_no}")
                result.append(row)
            return result
    except (OSError, UnicodeError) as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc


def _atomic_jsonl(
    path: Path, rows: Iterable[Mapping[str, Any]], *, overwrite: bool = True
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise ContractError(f"assignment shard already exists: {path}") from exc
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _derived_seed(seed: int, global_index: int) -> int:
    material = f"horscht-probability-v2-assignment-v1:{seed}:{global_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def _choose_answerability(
    rng: random.Random, history: "PlanningHistory"
) -> tuple[str, str | None]:
    """Assign answerability from the versioned mix and balance unanswerable reasons."""
    mix = ANSWERABILITY_MIX_PERCENT
    answerability_status = rng.choices(
        ("answerable", "unanswerable"),
        weights=(mix["answerable"], mix["unanswerable"]),
        k=1,
    )[0]
    history.answerability_counts[answerability_status] += 1
    if answerability_status == "answerable":
        return answerability_status, None

    fewest = min(history.unanswerable_reason_counts[reason] for reason in UNANSWERABLE_REASONS)
    candidates = [
        reason
        for reason in UNANSWERABLE_REASONS
        if history.unanswerable_reason_counts[reason] == fewest
    ]
    reason = rng.choice(candidates)
    history.unanswerable_reason_counts[reason] += 1
    return answerability_status, reason


def _cell_key(operation_id: str, domain_id: str) -> str:
    return f"{operation_id}|{domain_id}"


def _normalize_source_targets(source_targets: Mapping[str, int] | None = None) -> dict[str, int]:
    targets = dict(DEFAULT_SOURCE_TARGETS if source_targets is None else source_targets)
    if set(targets) != set(SOURCE_KINDS):
        raise ContractError(f"source targets must define exactly: {', '.join(SOURCE_KINDS)}")
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in targets.values()):
        raise ContractError("source targets must be non-negative integers")
    return targets


def _read_language_statuses(assignment_dir: Path, catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    status_path = assignment_dir / "language-status.jsonl"
    if not status_path.exists():
        return {}
    known = {row["tag"] for row in catalog["languages"]}
    latest: dict[str, dict[str, Any]] = {}
    for row in _rows(status_path):
        tag = row.get("language")
        status = row.get("language_status")
        if tag not in known or status not in {"supported", "unsupported"}:
            raise ContractError(f"invalid language status record in {status_path}")
        latest[tag] = row
    return latest


def set_language_status(
    language_tag: str,
    language_status: str,
    reason: str,
    *,
    catalog_path: Path = CATALOG,
    assignment_dir: Path = ASSIGNMENT_DIR,
) -> dict[str, Any]:
    """Persist an explicit language capability status without changing shard plans."""
    if language_status not in {"supported", "unsupported"}:
        raise ContractError("language_status must be supported or unsupported")
    if not _nonempty(reason):
        raise ContractError("reason must be a non-empty string")
    catalog = assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    known = {row["tag"] for row in catalog["languages"]}
    if language_tag not in known:
        raise ContractError(f"unknown language tag: {language_tag}")
    path = assignment_dir / "language-status.jsonl"
    statuses = _read_language_statuses(assignment_dir, catalog)
    record = {
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
        "language": language_tag,
        "language_status": language_status,
        "reason": reason.strip(),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    statuses[language_tag] = record
    _atomic_jsonl(path, [statuses[tag] for tag in sorted(statuses)])
    return record


class PlanningHistory:
    """Observed coverage, closed outputs, and durable identities from saved history."""

    def __init__(self, catalog: dict[str, Any]) -> None:
        self.coverage = assignments.Coverage(catalog)
        self.identifiers: set[str] = set()
        self.groups: set[str] = set()
        self.indexes: set[int] = set()
        self.question_ids: set[str] = set()
        self.skip_ids: set[str] = set()
        self.question_index_by_id: dict[str, int] = {}
        self.assignment_ids: set[str] = set()
        self.assignment_groups: set[str] = set()
        self.assignment_indexes: set[int] = set()
        self.max_index = 0
        self.active_by_source: Counter[str] = Counter()
        self.active_by_language: Counter[str] = Counter()
        self.status_by_source: Counter[str] = Counter()
        self.status_by_language: Counter[str] = Counter()
        self.answerability_counts: Counter[str] = Counter()
        self.unanswerable_reason_counts: Counter[str] = Counter()
        self.active_assignments = 0
        self.unsupported_languages: set[str] = set()
        self.language_statuses: dict[str, dict[str, Any]] = {}
        self.summary: dict[str, Any] = {}
        self.planning_profiles: dict[str, dict[str, Any]] = {}


def _require_planning_memory() -> None:
    """Leave the training owner a physical-RAM reserve during streaming scans."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                if int(line.split()[1]) * 1024 < MIN_AVAILABLE_MEMORY:
                    raise ContractError("planning requires at least 12 GiB available physical RAM")
                return
    except ContractError:
        raise
    except (OSError, ValueError):
        raise ContractError("cannot verify planning physical-RAM reserve") from None
    raise ContractError("cannot verify planning physical-RAM reserve")


def _file_stamp(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _complete_boundary(stream: BinaryIO, size: int) -> int:
    """Find the last complete line without loading a potentially long tail."""
    position = size
    while position:
        start = max(0, position - 64 * 1024)
        stream.seek(start)
        chunk = stream.read(position - start)
        if len(chunk) != position - start:
            raise ContractError("question history was truncated during prefix capture")
        newline = chunk.rfind(b"\n")
        if newline >= 0:
            return start + newline + 1
        position = start
    return 0


def _prefix_hash(path: Path, boundary: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = boundary
        while remaining:
            _require_planning_memory()
            chunk = stream.read(min(remaining, 1024 * 1024))
            if not chunk:
                raise ContractError("question history was truncated inside a frozen prefix")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _capture_prefix(path: Path) -> dict[str, Any]:
    before = _file_stamp(path)
    with path.open("rb") as stream:
        boundary = _complete_boundary(stream, before[2])
    digest = _prefix_hash(path, boundary)
    after = _file_stamp(path)
    if before[:2] != after[:2] or after[2] < before[2]:
        raise ContractError("question history was replaced or truncated during prefix capture")
    return {
        "path": path.as_posix(), "device": before[0], "inode": before[1],
        "initial_size_bytes": before[2], "prefix_bytes": boundary,
        "initial_mtime_ns": before[3], "initial_ctime_ns": before[4], "prefix_sha256": digest,
    }


def _prefix_lines(path: Path, prefix: Mapping[str, Any]) -> Iterator[bytes]:
    """Parse only the pinned prefix; unobserved append IDs stay plan commitments."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = prefix["prefix_bytes"]
        while remaining:
            line = stream.readline(remaining)
            if not line or not line.endswith(b"\n"):
                raise ContractError("question history changed inside a frozen prefix")
            digest.update(line)
            remaining -= len(line)
            yield line
    if digest.hexdigest() != prefix["prefix_sha256"]:
        raise ContractError("question history was rewritten inside a frozen prefix")


def _verify_prefix(path: Path, prefix: dict[str, Any]) -> None:
    current = _file_stamp(path)
    if current[:2] != (prefix["device"], prefix["inode"]):
        raise ContractError("question history file was replaced during planning snapshot")
    if current[2] < prefix["initial_size_bytes"]:
        raise ContractError("question history file was truncated during planning snapshot")
    if current[3:] != (prefix["initial_mtime_ns"], prefix["initial_ctime_ns"]) and current[2] == prefix["initial_size_bytes"]:
        raise ContractError("question history file was rewritten without append growth")
    grew = current[2] > prefix["initial_size_bytes"]
    if grew and _prefix_hash(path, prefix["prefix_bytes"]) != prefix["prefix_sha256"]:
        raise ContractError("question history was rewritten inside an appended file")
    after = _file_stamp(path)
    if after[:2] != current[:2] or after[2] < current[2]:
        raise ContractError("question history changed destructively during prefix verification")
    prefix["verified_size_bytes"] = after[2]
    prefix["growth_bytes"] = after[2] - prefix["initial_size_bytes"]
    prefix["state"] = "append_growth" if prefix["growth_bytes"] else "unchanged"


def _known_exclusions(path: Path) -> tuple[set[str], set[str], dict[str, Any], tuple[int, ...]]:
    excluded: set[str] = set()
    recommended: set[str] = set()
    try:
        before = _file_stamp(path)
    except FileNotFoundError:
        raise ContractError("required known-issue sidecar is missing") from None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                if not line.strip():
                    continue
                if not line.endswith(b"\n"):
                    raise ContractError("incomplete known-issue sidecar boundary")
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ContractError("invalid known-issue sidecar row")
                status = row.get("training_status")
                if status not in {"exclude_pending_review", "review_recommended"}:
                    continue
                identifiers = row.get("question_ids")
                if not isinstance(identifiers, list) or any(not _nonempty(value) for value in identifiers):
                    raise ContractError("invalid known-issue question_ids")
                target = excluded if status == "exclude_pending_review" else recommended
                target.update(value.strip() for value in identifiers)
    except (UnicodeError, json.JSONDecodeError):
        raise ContractError("invalid known-issue sidecar JSON") from None
    if before != _file_stamp(path):
        raise ContractError("known-issue sidecar changed while being read")
    return excluded, recommended, {"path": path.as_posix(), "sha256": digest.hexdigest()}, before


def _is_explicit_variant(row: Mapping[str, Any]) -> bool:
    return "_augmentation" in row or any(
        _nonempty(row.get(key))
        for key in ("translation_of", "parent_question_id", "source_row_id", "variant_of")
    )


def _scan_questions(
    catalog: dict[str, Any], history: PlanningHistory, files: list[Path],
    *, known_issues_path: Path | None,
) -> dict[str, Any]:
    """One accounting pass over pinned prefixes, plus opaque byte-hash checks."""
    _require_planning_memory()
    scan_started = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    excluded, recommended, issue_record = set(), set(), None
    issue_stamp = None
    if known_issues_path is not None:
        excluded, recommended, issue_record, issue_stamp = _known_exclusions(known_issues_path)
    rejected: Counter[str] = Counter()
    valid_raw = duplicate_ids = malformed = 0
    prefixes = {path: _capture_prefix(path) for path in files}
    with tempfile.TemporaryDirectory(prefix="probability-v2-planning-") as temporary:
        with sqlite3.connect(Path(temporary) / "identities.sqlite3") as connection:
            connection.executescript("""
                PRAGMA journal_mode=OFF;
                PRAGMA synchronous=OFF;
                PRAGMA cache_size=-8192;
                PRAGMA temp_store=FILE;
                CREATE TABLE groups_seen (group_id TEXT, split TEXT);
                CREATE INDEX groups_seen_index ON groups_seen(group_id);
                CREATE TABLE candidates (
                    ordinal INTEGER PRIMARY KEY, id TEXT UNIQUE, group_id TEXT,
                    source TEXT, language TEXT, schema_kind TEXT, fingerprint TEXT,
                    pilot INTEGER, excluded INTEGER, recommended INTEGER, variant INTEGER
                );
                CREATE TABLE conflicted_groups (group_id TEXT PRIMARY KEY);
                CREATE TABLE original_groups (group_id TEXT PRIMARY KEY);
                CREATE TABLE fingerprints (value TEXT PRIMARY KEY);
            """)
            line_count = 0
            for path in files:
                is_skip = path.name.startswith("skipped-")
                for line in _prefix_lines(path, prefixes[path]):
                    if not line.strip():
                        continue
                    line_count += 1
                    if line_count % 1024 == 0:
                        _require_planning_memory()
                    try:
                        row = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError):
                        if not is_skip:
                            malformed += 1
                            rejected["invalid_json"] += 1
                        continue
                    if isinstance(row, dict):
                        _reserve_question_identity(history, row)
                    if is_skip:
                        if isinstance(row, dict):
                            identifier = row.get("id", row.get("assignment_id", row.get("plan_id")))
                            if _nonempty(identifier):
                                history.skip_ids.add(identifier.strip())
                        continue
                    if isinstance(row, dict):
                        identifier = row.get("id")
                        if _nonempty(identifier) and identifier.strip() in excluded:
                            history.question_ids.add(identifier.strip())
                        group, split = row.get("problem_group_id"), row.get("split")
                        if _nonempty(group) and _nonempty(split):
                            connection.execute("INSERT INTO groups_seen VALUES (?, ?)", (group.strip(), split.strip()))
                    source, errors, schema = probability_v2.classify_row(row)
                    if errors or source is None:
                        rejected[errors[0] if errors else "invalid_row"] += 1
                        continue
                    valid_raw += 1
                    identifier = row["id"].strip()
                    history.question_ids.add(identifier)
                    history.coverage.add(row, catalog, is_plan=False)
                    inserted = connection.execute(
                        """INSERT OR IGNORE INTO candidates(
                            id,group_id,source,language,schema_kind,fingerprint,
                            pilot,excluded,recommended,variant
                        ) VALUES (?,?,?,?,?,?,?,?,?,?)""",
                        (identifier, row["problem_group_id"].strip(), source,
                         probability_v2._language(row), schema,
                         training_snapshot.exact_content_fingerprint(row),
                         row.get("pilot_status") == probability_v2.EXCLUDED_PILOT_STATUS,
                         identifier in excluded, identifier in recommended, _is_explicit_variant(row)),
                    ).rowcount
                    if not inserted:
                        duplicate_ids += 1
                        rejected["duplicate_row_id"] += 1
            connection.execute("""INSERT INTO conflicted_groups
                SELECT group_id FROM groups_seen GROUP BY group_id HAVING COUNT(DISTINCT split)>1""")
            conflicts = connection.execute("SELECT COUNT(*) FROM conflicted_groups").fetchone()[0]
            conflict_rows = connection.execute("SELECT COUNT(*) FROM candidates JOIN conflicted_groups USING(group_id)").fetchone()[0]
            rejected["problem_group_spans_splits"] += conflict_rows
            counts: Counter[str] = Counter()
            sources = {name: Counter() for name in ("structural", "eligible", "original", "unique")}
            languages: Counter[str] = Counter()
            candidates = connection.execute("""SELECT group_id,source,language,schema_kind,fingerprint,
                pilot,excluded,recommended,variant FROM candidates
                WHERE NOT EXISTS (SELECT 1 FROM conflicted_groups WHERE conflicted_groups.group_id=candidates.group_id)
                ORDER BY ordinal""")
            for group, source, language, schema, fingerprint, pilot, excluded_row, review, variant in candidates:
                counts["accepted"] += 1
                if counts["accepted"] % 1024 == 0:
                    _require_planning_memory()
                languages[language] += 1
                counts["legacy"] += source == probability_v2.HARD_LABEL and schema == "legacy"
                if pilot:
                    counts["pilot"] += 1
                    continue
                sources["structural"][source] += 1
                if excluded_row:
                    counts["excluded"] += 1
                    continue
                sources["eligible"][source] += 1
                counts["review_recommended"] += review
                if variant:
                    counts["explicit_variants"] += 1
                    continue
                sources["original"][source] += 1
                if not connection.execute("INSERT OR IGNORE INTO original_groups VALUES (?)", (group,)).rowcount:
                    counts["grouped_variants"] += 1
                    continue
                if not connection.execute("INSERT OR IGNORE INTO fingerprints VALUES (?)", (fingerprint,)).rowcount:
                    counts["exact_duplicates"] += 1
                    continue
                sources["unique"][source] += 1
                counts["legacy_unique"] += source == probability_v2.HARD_LABEL and schema == "legacy"
    for path, prefix in prefixes.items():
        _verify_prefix(path, prefix)
    if known_issues_path is not None and _file_stamp(known_issues_path) != issue_stamp:
        raise ContractError("known-issue sidecar changed during planning snapshot")
    return {
        "accounting_version": ACCOUNTING_VERSION, "known_issue_sidecar": issue_record,
        "scan_interval": {"started_at": scan_started, "finished_at": datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")},
        "question_prefixes": list(prefixes.values()),
        "original_attribution_rule": "sorted source paths, then line order; first eligible unmarked row per group, then first exact-content fingerprint",
        "uniqueness_scope": "explicit variant links, problem groups, and exact normalized prompt/options/targets; semantic independence is unverified",
        "structurally_valid_rows_before_identity_checks": valid_raw,
        "accepted_rows": counts["accepted"], "accepted_rows_by_language": dict(languages),
        "structural_production_rows": sum(sources["structural"].values()),
        "eligible_rows": sum(sources["eligible"].values()),
        "eligible_original_rows": sum(sources["original"].values()),
        "unique_eligible_original_rows": sum(sources["unique"].values()),
        "production_eligible_rows": sum(sources["unique"].values()),
        "production_eligible_rows_by_source_kind": dict(sources["unique"]),
        "structural_production_rows_by_source_kind": dict(sources["structural"]),
        "eligible_rows_by_source_kind": dict(sources["eligible"]),
        "eligible_original_rows_by_source_kind": dict(sources["original"]),
        "known_issue_excluded_rows": counts["excluded"],
        "review_recommended_eligible_rows": counts["review_recommended"],
        "explicit_variant_rows": counts["explicit_variants"],
        "grouped_variant_rows": counts["grouped_variants"],
        "duplicate_exact_content_rows": counts["exact_duplicates"],
        "legacy_hard_label_rows": counts["legacy_unique"],
        "structural_legacy_hard_label_rows": counts["legacy"], "excluded_pilot_rows": counts["pilot"],
        "rejected_rows": sum(rejected.values()), "duplicate_row_ids_rejected": duplicate_ids,
        "problem_group_split_conflicts": conflicts, "malformed_json_rows": malformed,
    }


def _reserve_question_identity(history: PlanningHistory, row: Mapping[str, Any]) -> None:
    """Rejected and orphaned outputs still own their durable identities."""
    identifier = row.get("id", row.get("assignment_id", row.get("plan_id")))
    group = row.get("problem_group_id")
    if _nonempty(identifier):
        identifier = identifier.strip()
        history.identifiers.add(identifier)
    if _nonempty(group):
        history.groups.add(group.strip())
    generator = row.get("_generator")
    index = generator.get("global_index") if isinstance(generator, dict) else None
    reserved_indexes: set[int] = set()
    if isinstance(index, int) and not isinstance(index, bool) and index >= 0:
        reserved_indexes.add(index)
    for value, prefix in (
        (identifier, "probability-v2-assignment-"),
        (group, "probability-v2-group-"),
    ):
        if _nonempty(value) and value.strip().startswith(prefix):
            suffix = value.strip().removeprefix(prefix)
            if suffix.isdecimal():
                reserved_indexes.add(int(suffix))
    history.indexes.update(reserved_indexes)
    history.max_index = max([history.max_index, *reserved_indexes])
    if _nonempty(identifier) and len(reserved_indexes) == 1:
        history.question_index_by_id[identifier] = next(iter(reserved_indexes))


def load_history(
    catalog: dict[str, Any],
    *,
    question_paths: Iterable[Path],
    assignment_dir: Path,
    known_issues_path: Path | None = None,
) -> PlanningHistory:
    """Read legacy/v2 question rows and v2 plans; never mutate history files."""
    question_paths = tuple(question_paths)
    try:
        training_snapshot._reject_external_evaluation_sources(question_paths)
    except training_snapshot.SnapshotError:
        raise ContractError("evaluation-only paths cannot be planning inputs") from None
    history = PlanningHistory(catalog)
    if known_issues_path is None:
        nearby = assignment_dir.parent / "known-issues-v1.jsonl"
        if nearby.exists() or assignment_dir.resolve() == ASSIGNMENT_DIR.resolve():
            known_issues_path = nearby
    question_files = _jsonl_files(question_paths)
    history.summary = _scan_questions(catalog, history, question_files, known_issues_path=known_issues_path)
    current_files = _jsonl_files(question_paths)
    captured_files = set(question_files)
    if captured_files - set(current_files):
        raise ContractError("captured question history files were removed during planning snapshot")
    history.summary["new_files_omitted"] = [path.as_posix() for path in current_files if path not in captured_files]

    catalog_languages = {item["tag"] for item in catalog["languages"]}
    known_cells = {
        _cell_key(operation["id"], domain["id"])
        for operation, domain in catalog["_cells"]
    }
    known_operations = {item["id"] for item in catalog["operations"]}
    known_domains = {item["id"] for item in catalog["domains"]}
    history.language_statuses = _read_language_statuses(assignment_dir, catalog)
    history.unsupported_languages = {
        tag
        for tag, record in history.language_statuses.items()
        if record.get("language_status") == "unsupported"
    }
    plan_files = sorted(assignment_dir.rglob("shard-*.jsonl")) if assignment_dir.is_dir() else []
    for path in plan_files:
        _require_planning_memory()
        for row in _rows(path):
            if row.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION:
                raise ContractError(f"unsupported assignment schema in {path}")
            source_kind = row.get("distribution_source_kind")
            if source_kind not in SOURCE_KINDS:
                raise ContractError(f"invalid distribution_source_kind in {path}")
            status = row.get("status")
            if status not in KNOWN_STATUSES:
                raise ContractError(f"invalid assignment status in {path}")
            identifier = row.get("id")
            group_id = row.get("problem_group_id")
            language = row.get("language")
            operation_id = row.get("operation_id")
            domain_id = row.get("domain_id")
            if not all(_nonempty(value) for value in (identifier, group_id, language, operation_id, domain_id)):
                raise ContractError(f"incomplete assignment identity or coverage fields in {path}")
            if language not in catalog_languages or operation_id not in known_operations or domain_id not in known_domains:
                raise ContractError(f"assignment references unknown catalog ids in {path}")
            if _cell_key(operation_id, domain_id) not in known_cells:
                raise ContractError(f"assignment references incompatible operation/domain pair in {path}")
            if identifier.strip() in history.assignment_ids:
                raise ContractError(f"duplicate assignment id {identifier!r} in {path}")
            if group_id.strip() in history.assignment_groups:
                raise ContractError(f"duplicate assignment problem_group_id {group_id!r} in {path}")
            history.identifiers.add(identifier.strip())
            history.groups.add(group_id.strip())
            generator = row.get("_generator")
            index = generator.get("global_index") if isinstance(generator, dict) else None
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                raise ContractError(f"assignment needs a non-negative _generator.global_index in {path}")
            planner_metadata = row.get("_planner")
            profile = planner_metadata.get("planning_profile") if isinstance(planner_metadata, Mapping) else None
            if profile is not None:
                if (not isinstance(profile, dict) or not _nonempty(profile.get("name"))
                    or _planning_profile_limit(profile["name"]) is None):
                    raise ContractError("unsupported saved planning profile")
                start = profile.get("start_index")
                limit = _planning_profile_limit(profile["name"])
                fixed_start = _support_profile_start(profile["name"])
                if (isinstance(start, bool) or not isinstance(start, int)
                    or profile.get("max_assignments") != limit
                    or fixed_start is not None and start != fixed_start
                    or not start <= index < start + limit):
                    raise ContractError("saved planning profile exceeds its adoption window")
                previous = history.planning_profiles.setdefault(profile["name"], profile)
                if previous != profile:
                    raise ContractError("conflicting saved planning profile bindings")
            if index in history.assignment_indexes:
                raise ContractError(f"duplicate global_index {index} in assignment history")
            output_index = history.question_index_by_id.get(identifier.strip())
            if output_index is not None and output_index != index:
                raise ContractError(f"assignment/output global_index mismatch for {identifier!r}")
            if index in history.indexes and output_index != index:
                raise ContractError(f"global_index {index} overlaps a different historical row")
            history.assignment_ids.add(identifier.strip())
            history.assignment_groups.add(group_id.strip())
            history.assignment_indexes.add(index)
            history.indexes.add(index)
            history.max_index = max(history.max_index, index)
            history.status_by_source[f"{status}:{source_kind}"] += 1
            history.status_by_language[f"{status}:{language}"] += 1
            answerability_status = row.get("answerability_status")
            if answerability_status in {"answerable", "unanswerable"}:
                history.answerability_counts[answerability_status] += 1
                reason = row.get("unanswerable_reason")
                if answerability_status == "unanswerable" and reason in UNANSWERABLE_REASONS:
                    history.unanswerable_reason_counts[reason] += 1
            coverage_row = row
            assignment_id = identifier.strip()
            if status in ACTIVE_STATUSES:
                if assignment_id in history.question_ids:
                    coverage_row = dict(row, status="completed")
                elif assignment_id in history.skip_ids or language in history.unsupported_languages:
                    coverage_row = dict(row, status="skipped")
            history.coverage.add(coverage_row, catalog, is_plan=True)
            if (
                status in ACTIVE_STATUSES
                and language not in history.unsupported_languages
                and assignment_id not in history.question_ids
                and assignment_id not in history.skip_ids
            ):
                history.active_by_source[source_kind] += 1
                history.active_by_language[language] += 1
                history.active_assignments += 1

    return history


def _question_history_paths(
    *,
    root: Path = ROOT,
    question_dir: Path = QUESTION_DIR,
    legacy_dir: Path = LEGACY_DIR,
    pilot_file: Path = PILOT_FILE,
) -> list[Path]:
    paths = [question_dir, legacy_dir, pilot_file]
    return [path if path.is_absolute() else root / path for path in paths]


def _choose_language(
    catalog: dict[str, Any],
    history: PlanningHistory,
    *,
    seed: int,
    requested_tag: str | None,
) -> str:
    known = sorted(item["tag"] for item in catalog["languages"])
    if requested_tag is not None:
        if requested_tag not in known:
            raise ContractError(f"unknown output language tag: {requested_tag}")
        if requested_tag in history.unsupported_languages:
            raise ContractError(f"language tag {requested_tag} is marked unsupported; no fallback is applied")
        return requested_tag
    candidates = [tag for tag in known if tag not in history.unsupported_languages]
    if not candidates:
        raise ContractError("no supported language tags remain in the catalog")
    least = min(history.coverage.languages[tag] for tag in candidates)
    tied = [tag for tag in candidates if history.coverage.languages[tag] == least]
    return random.Random(_derived_seed(seed, 0)).choice(tied)


def _pick_word_seeds(
    by_language: Mapping[str, list[dict[str, str]]], rng: random.Random
) -> list[dict[str, str]]:
    languages = sorted(tag for tag, rows in by_language.items() if rows)
    if len(languages) < 3:
        raise ContractError("seed pool must contain words from at least three source languages")
    count = rng.randint(3, min(8, len(languages)))
    selected = rng.sample(languages, count)
    return [rng.choice(by_language[tag]).copy() for tag in selected]


def _cell_counts(catalog: dict[str, Any], history: PlanningHistory) -> Counter[str]:
    return Counter(
        {
            _cell_key(operation["id"], domain["id"]): history.coverage.cells[
                _cell_key(operation["id"], domain["id"])
            ]
            for operation, domain in catalog["_cells"]
        }
    )


def _pick_cell(catalog: dict[str, Any], counts: Counter[str], rng: random.Random) -> tuple[dict[str, Any], dict[str, Any]]:
    cells = catalog["_cells"]
    minimum = min(counts[_cell_key(operation["id"], domain["id"])] for operation, domain in cells)
    least_used = [
        (operation, domain)
        for operation, domain in cells
        if counts[_cell_key(operation["id"], domain["id"])] == minimum
    ]
    return rng.choice(sorted(least_used, key=lambda pair: _cell_key(pair[0]["id"], pair[1]["id"])))


def _opaque_file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_planning_profile(path: Path | None) -> dict[str, Any] | None:
    """Bind a coordinator-owned adoption artifact without interpreting prose."""
    if path is None:
        return None
    path = _resolve_path(path).resolve()
    try:
        raw = path.read_bytes()
        config = json.loads(raw)
        if not isinstance(config, dict) or set(config) != {
            "schema_version", "name", "start_index", "max_assignments",
            "authoring_guidance_version", "bindings",
        }:
            raise ContractError("invalid planning profile artifact fields")
        start = config["start_index"]
        if (config["schema_version"] != "probability-v2-planning-profile-v1"
            or _planning_profile_limit(config["name"]) is None
            or isinstance(start, bool) or not isinstance(start, int) or start < 1
            or isinstance(config["max_assignments"], bool) or not isinstance(config["max_assignments"], int)
            or config["max_assignments"] != _planning_profile_limit(config["name"])
            or not _nonempty(config["authoring_guidance_version"])):
            raise ContractError("invalid bounded planning profile")
        fixed_start = _support_profile_start(config["name"])
        if fixed_start is not None and start != fixed_start:
            raise ContractError(f"multiple-support profile must begin at index {fixed_start}")
        bindings = config["bindings"]
        if not isinstance(bindings, dict) or set(bindings) != {"guidance", "decision"}:
            raise ContractError("profile requires guidance and decision bindings")
        resolved = {}
        for name, binding in bindings.items():
            if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                raise ContractError("invalid planning profile binding")
            if not _nonempty(binding["path"]) or not isinstance(binding["sha256"], str):
                raise ContractError("invalid planning profile binding")
            bound = _resolve_path(binding["path"]).resolve()
            if bound == path or _opaque_file_hash(bound) != binding["sha256"]:
                raise ContractError("planning profile binding hash mismatch")
            resolved[name] = {"path": bound.as_posix(), "sha256": binding["sha256"]}
        policy = {
            "choice_bands": {name: {"minimum": low, "maximum": high, "weight": weight}
                             for name, (low, high, weight) in DEV_PROFILE_BANDS.items()},
            "operation_weights": dict(DEV_PROFILE_OPERATION_WEIGHTS),
        } if config["name"] == DEV_PROFILE_NAME else {
            "target_shape_requirement": MULTIPLE_SUPPORT_REQUIREMENT,
            "selection_probability": 0.25,
            "marked_choice_counts": [3, 4],
            "distribution_task_forms": DISTRIBUTION_TASK_FORMS,
        }
        return {
            **config, "bindings": resolved,
            "manifest_path": path.as_posix(), "manifest_sha256": hashlib.sha256(raw).hexdigest(),
            **policy,
        }
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ContractError("cannot read planning profile or bound artifact") from None


def _profile_preflight(profile: dict[str, Any] | None, history: PlanningHistory, start: int, count: int) -> None:
    if profile is None:
        return
    previous = history.planning_profiles.get(profile["name"])
    if previous is not None and previous != profile:
        raise ContractError("planning profile bindings differ from immutable saved plans")
    if previous is None and profile["start_index"] != max(history.max_index + 1, 1):
        raise ContractError("new planning profile must begin at the next unused global index")
    if previous is None:
        for saved in history.planning_profiles.values():
            if (profile["start_index"] < saved["start_index"] + saved["max_assignments"]
                and saved["start_index"] < profile["start_index"] + profile["max_assignments"]):
                raise ContractError("planning profile windows overlap immutable saved profiles")
    limit = profile["max_assignments"]
    if start < profile["start_index"] or start + count > profile["start_index"] + limit:
        raise ContractError(f"requested new plans exceed the {limit:,}-assignment profile window")


def _pick_profile_task(catalog: dict[str, Any], counts: Counter[str], rng: random.Random):
    structures = assignments._items(catalog["input_structures"], "input_structures")
    feasible = {}
    for band in DEV_PROFILE_BANDS:
        operations = MEDIUM_CARDINALITY_OPERATIONS if band == "5-16" else HIGH_CARDINALITY_OPERATIONS
        structure_ids = MEDIUM_CARDINALITY_STRUCTURES if band == "5-16" else HIGH_CARDINALITY_STRUCTURES
        cells = [pair for pair in catalog["_cells"] if band == "2-4" or pair[0]["id"] in operations]
        inputs = [item for item in structures if band == "2-4" or item.get("id") in structure_ids]
        if cells and inputs:
            feasible[band] = (cells, inputs)
    if not feasible:
        raise ContractError("catalog has no task compatible with the planning profile")
    band = rng.choices(list(feasible), weights=[DEV_PROFILE_BANDS[name][2] for name in feasible], k=1)[0]
    cells, inputs = feasible[band]
    operations = {operation["id"]: operation for operation, _ in cells}
    identifiers = sorted(operations)
    operation_id = rng.choices(identifiers, weights=[DEV_PROFILE_OPERATION_WEIGHTS.get(name, 1.0)
                                                   for name in identifiers], k=1)[0]
    candidates = [pair for pair in cells if pair[0]["id"] == operation_id]
    minimum = min(counts[_cell_key(operation_id, domain["id"])] for _, domain in candidates)
    operation, domain = rng.choice(sorted(
        [pair for pair in candidates if counts[_cell_key(operation_id, pair[1]["id"])] == minimum],
        key=lambda pair: pair[1]["id"],
    ))
    low, high, _ = DEV_PROFILE_BANDS[band]
    total_weight = sum(DEV_PROFILE_BANDS[name][2] for name in feasible)
    metadata = {
        "choice_band": band,
        "effective_band_weights": {name: DEV_PROFILE_BANDS[name][2] / total_weight for name in feasible},
        "omitted_bands": [name for name in DEV_PROFILE_BANDS if name not in feasible],
        "operation_weight": DEV_PROFILE_OPERATION_WEIGHTS.get(operation_id, 1.0),
        "operation_effective_probability": DEV_PROFILE_OPERATION_WEIGHTS.get(operation_id, 1.0)
            / sum(DEV_PROFILE_OPERATION_WEIGHTS.get(name, 1.0) for name in identifiers),
        "omitted_priority_operations": [name for name in DEV_PROFILE_OPERATION_WEIGHTS if name not in operations],
        "operation_weight_scope": "operations compatible with sampled band",
        "complete_inventory_required": band == "65-265",
    }
    return operation, domain, rng.randint(low, high), rng.choice(inputs), metadata


def _distribution_candidates(catalog: dict[str, Any], source_kind: str):
    candidates = []
    for form, compatibility in DISTRIBUTION_TASK_FORMS.items():
        if source_kind not in compatibility["source_kinds"]:
            continue
        inputs = [item for item in catalog["input_structures"]
                  if item.get("id") in compatibility["input_structures"]]
        for operation, domain in catalog["_cells"]:
            if operation["id"] in compatibility["operations"] and inputs:
                candidates.append((form, operation, domain, inputs))
    return sorted(candidates, key=lambda item: (item[0], item[1]["id"], item[2]["id"]))


def probability_task_compatibility(
    catalog: dict[str, Any], source_kinds: Iterable[str] = SOURCE_KINDS,
) -> dict[str, Any]:
    """Read catalog metadata only; explicitly report omissions before allocation."""
    report = {}
    for source in sorted(set(source_kinds)):
        if source not in SOURCE_KINDS:
            raise ContractError("unsupported source in probability-task compatibility request")
        candidates = _distribution_candidates(catalog, source)
        report[source] = {
            "compatible_cells": len(candidates),
            "forms": sorted({item[0] for item in candidates}),
            "omission_reason": None if candidates else (
                "hard_label" if source == probability_v2.HARD_LABEL else "unsupported_distribution_task"
            ),
        }
    return report


def _pick_distribution_task(candidates, counts: Counter[str], rng: random.Random):
    minimum = min(counts[_cell_key(operation["id"], domain["id"])]
                  for _, operation, domain, _ in candidates)
    eligible = [item for item in candidates if counts[_cell_key(item[1]["id"], item[2]["id"])] == minimum]
    form, operation, domain, structures = rng.choice(eligible)
    question_form = {
        "id": form, "question_kind": "outcome_distribution",
        "requirements": ["state conditioning and mechanism assumptions in context",
                         "define a mutually exclusive exhaustive substantive outcome partition",
                         "ask for its probability law, not a uniquely correct action or an unseen realized outcome"],
    }
    return operation, domain, question_form, rng.choice(structures), rng.choice((3, 4))


def _target_shape_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    reasons = Counter(row["_planner"]["target_shape_selection"]["status"] for row in rows)
    return {"planned_marked": reasons["required"], "selection_status_counts": dict(sorted(reasons.items())),
            "interpretation": "planned requirements only; actual target shapes and soft-format share are separate"}


def budget_report(
    summary: Mapping[str, Any],
    active_by_source: Mapping[str, int],
    source_targets: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Credit only verified accounting of unique eligible originals toward quotas."""
    targets = _normalize_source_targets(source_targets)
    if summary.get("accounting_version") != ACCOUNTING_VERSION:
        raise ContractError("usable-original accounting snapshot is required; raw summaries cannot satisfy quotas")
    stages = (
        ("structural_production_rows", "structural_production_rows_by_source_kind"),
        ("eligible_rows", "eligible_rows_by_source_kind"),
        ("eligible_original_rows", "eligible_original_rows_by_source_kind"),
        ("unique_eligible_original_rows", "production_eligible_rows_by_source_kind"),
    )
    def count(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ContractError("accounting counters must be non-negative integers")
        return value
    previous = None
    for total_key, source_key in stages:
        total = count(summary.get(total_key))
        values = summary.get(source_key)
        if not isinstance(values, Mapping) or set(values) - set(SOURCE_KINDS):
            raise ContractError("invalid accounting source counters")
        if sum(count(value) for value in values.values()) != total:
            raise ContractError("accounting total disagrees with source counters")
        if previous is not None and any(values.get(kind, 0) > previous.get(kind, 0) for kind in SOURCE_KINDS):
            raise ContractError("accounting stages disagree")
        previous = values
    if (
        count(summary.get("production_eligible_rows")) != summary["unique_eligible_original_rows"]
        or count(summary.get("accepted_rows")) - count(summary.get("excluded_pilot_rows")) != summary["structural_production_rows"]
        or summary["structural_production_rows"] - summary["eligible_rows"] != count(summary.get("known_issue_excluded_rows"))
        or summary["eligible_rows"] - summary["eligible_original_rows"] != count(summary.get("explicit_variant_rows"))
        or summary["eligible_original_rows"] - summary["unique_eligible_original_rows"] != count(summary.get("grouped_variant_rows")) + count(summary.get("duplicate_exact_content_rows"))
        or count(summary.get("review_recommended_eligible_rows")) > summary["eligible_rows"]
    ):
        raise ContractError("accounting stages disagree")
    if set(active_by_source) - set(SOURCE_KINDS):
        raise ContractError("invalid active source counters")
    for value in active_by_source.values():
        count(value)
    content_by_source = summary["production_eligible_rows_by_source_kind"]
    committed_by_source = {
        kind: int(content_by_source.get(kind, 0)) + int(active_by_source.get(kind, 0))
        for kind in SOURCE_KINDS
    }
    deficits = {
        kind: max(0, target - committed_by_source[kind])
        for kind, target in targets.items()
    }
    content_total = summary["unique_eligible_original_rows"]
    active_total = sum(int(value) for value in active_by_source.values())
    total_target = sum(targets.values())
    return {
        "accounting": dict(summary),
        "production_target": total_target,
        "production_rows_accepted": content_total,
        "production_rows_remaining_to_accept": sum(max(0, target - content_by_source.get(kind, 0)) for kind, target in targets.items()),
        "active_assignment_rows": active_total,
        "production_rows_committed": content_total + active_total,
        "unassigned_target_balance": sum(deficits.values()),
        "production_target_deficit": sum(deficits.values()),
        "source_targets": targets,
        "source_rows_remaining_to_accept": {
            kind: max(0, target - int(content_by_source.get(kind, 0)))
            for kind, target in targets.items()
        },
        "source_rows_committed": committed_by_source,
        "source_target_deficits": deficits,
    }


def _progress_payload(
    summary: Mapping[str, Any],
    history: PlanningHistory,
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "probability-v2-progress-v1",
        "accounting": dict(summary),
        "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_quotas": dict(budget["source_targets"]),
        "accepted_production_rows": int(summary.get("production_eligible_rows", 0)),
        "accepted_production_rows_by_source_kind": dict(
            sorted(summary.get("production_eligible_rows_by_source_kind", {}).items())
        ),
        "production_target": budget["production_target"],
        "production_rows_remaining_to_accept": budget["production_rows_remaining_to_accept"],
        "unassigned_target_balance": budget["unassigned_target_balance"],
        "source_rows_remaining_to_accept": dict(budget["source_rows_remaining_to_accept"]),
        "production_target_deficit": budget["production_target_deficit"],
        "source_target_deficits": dict(budget["source_target_deficits"]),
        "active_assignments": history.active_assignments,
        "active_assignments_by_source_kind": dict(sorted(history.active_by_source.items())),
        "active_assignments_by_language": dict(sorted(history.active_by_language.items())),
        "planned_assignments_by_status_and_source_kind": dict(sorted(history.status_by_source.items())),
        "planned_assignments_by_status_and_language": dict(sorted(history.status_by_language.items())),
        "skipped_assignments": sum(
            count for key, count in history.status_by_source.items() if key.startswith("skipped:")
        ),
        "failed_assignments": sum(
            count for key, count in history.status_by_source.items() if key.startswith("failed:")
        ),
        "legacy_hard_label_rows": int(summary.get("legacy_hard_label_rows", 0)),
        "excluded_pilot_rows": int(summary.get("excluded_pilot_rows", 0)),
        "rejected_question_rows": int(summary.get("rejected_rows", 0)),
        "duplicate_question_ids_rejected": int(summary.get("duplicate_row_ids_rejected", 0)),
        "problem_group_split_conflicts": int(summary.get("problem_group_split_conflicts", 0)),
        "language_coverage": dict(sorted(history.coverage.languages.items())),
        "domain_coverage": dict(sorted(history.coverage.domains.items())),
        "operation_coverage": dict(sorted(history.coverage.operations.items())),
        "operation_domain_coverage": dict(sorted(history.coverage.cells.items())),
        "language_statuses": history.language_statuses,
        "unsupported_languages": sorted(history.unsupported_languages),
    }


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


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _progress_markdown(payload: Mapping[str, Any]) -> str:
    accepted = int(payload.get("accepted_production_rows", 0))
    target = int(payload.get("production_target", 0))
    remaining_to_accept = int(payload.get("production_rows_remaining_to_accept", 0))
    unassigned_balance = int(payload.get("unassigned_target_balance", 0))
    source_targets = payload.get("source_quotas", {})
    accepted_by_source = payload.get("accepted_production_rows_by_source_kind", {})
    active_by_source = payload.get("active_assignments_by_source_kind", {})
    source_remaining = payload.get("source_rows_remaining_to_accept", {})
    source_deficits = payload.get("source_target_deficits", {})
    lines = [
        "# Probability-v2 generation progress",
        "",
        f"Updated: {payload.get('updated_at', 'unknown')}",
        "",
        f"Accepted **{accepted:,} / {target:,}** production rows; **{remaining_to_accept:,}** still need to be accepted.",
        f"Active assignments: {int(payload.get('active_assignments', 0)):,}; unassigned target balance: {unassigned_balance:,}.",
        f"Skipped assignments: {int(payload.get('skipped_assignments', 0)):,}; failed: {int(payload.get('failed_assignments', 0)):,}.",
        f"Legacy hard-label rows credited: {int(payload.get('legacy_hard_label_rows', 0)):,}; pilot rows excluded: {int(payload.get('excluded_pilot_rows', 0)):,}.",
        f"Structurally rejected rows: {int(payload.get('rejected_question_rows', 0)):,}; problem-group split conflicts: {int(payload.get('problem_group_split_conflicts', 0)):,}.",
        "",
        "Quota credit uses unique eligible originals after known exclusions, explicit variants, group copies, and exact-content copies are removed. These checks do not prove semantic independence.",
        "",
        "Schema and available distribution-math checks are machine-checked. Prose semantics, language fluency, synthetic-sample claims, and teacher calibration remain unverified.",
        "",
        "## Source quotas",
        "",
        "| Source | Accepted | Active | Target | Still to accept | Not yet assigned |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for source in SOURCE_KINDS:
        lines.append(
            f"| `{source}` | {int(accepted_by_source.get(source, 0)):,} | "
            f"{int(active_by_source.get(source, 0)):,} | {int(source_targets.get(source, 0)):,} | "
            f"{int(source_remaining.get(source, 0)):,} | {int(source_deficits.get(source, 0)):,} |"
        )

    for title, key in (
        ("Languages", "language_coverage"),
        ("Domains", "domain_coverage"),
        ("Operations", "operation_coverage"),
    ):
        values = payload.get(key, {})
        if not isinstance(values, Mapping):
            continue
        lines.extend(["", f"## {title} ({len(values)} catalog entries with rows or active assignments)", ""])
        if values:
            lines.extend(f"- `{name}`: {int(count):,}" for name, count in sorted(values.items()))
        else:
            lines.append("- No coverage recorded yet.")

    unsupported = payload.get("unsupported_languages", [])
    lines.extend(["", "## Languages marked unsupported", ""])
    lines.append(", ".join(f"`{tag}`" for tag in unsupported) if unsupported else "None recorded.")
    return "\n".join(lines) + "\n"


def _refresh_progress(
    *,
    summary: Mapping[str, Any],
    history: PlanningHistory,
    budget: Mapping[str, Any],
    progress_path: Path,
) -> dict[str, Any]:
    payload = _progress_payload(summary, history, budget)
    _atomic_json(progress_path, payload)
    _atomic_text(progress_path.with_suffix(".md"), _progress_markdown(payload))
    return payload


def create_plan_rows(
    *,
    count: int,
    seed: int,
    start_index: int,
    source_kind: str,
    language_tag: str,
    catalog: dict[str, Any],
    word_pool: list[dict[str, str]],
    history: PlanningHistory,
    budget: Mapping[str, Any],
    output_path: Path,
    planning_profile: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_BATCH:
        raise ContractError(f"count must be 1–{MAX_BATCH}")
    if isinstance(start_index, bool) or not isinstance(start_index, int) or start_index < 0:
        raise ContractError("start_index must be zero or greater")
    if source_kind not in SOURCE_KINDS:
        raise ContractError(f"unsupported source kind: {source_kind}")
    if budget["source_target_deficits"][source_kind] < count:
        raise ContractError(
            f"{source_kind} deficit is {budget['source_target_deficits'][source_kind]}, below requested count {count}"
        )
    if budget["production_target_deficit"] < count:
        raise ContractError(
            f"production target deficit is {budget['production_target_deficit']}, below requested count {count}"
        )
    _profile_preflight(planning_profile, history, start_index, count)

    by_language: dict[str, list[dict[str, str]]] = defaultdict(list)
    for word in word_pool:
        tag = word.get("source_language")
        if _nonempty(tag):
            by_language[tag].append(word)
    groups = len({word.get("word", "").casefold() for word in word_pool if _nonempty(word.get("word"))})
    if groups < 3:
        raise ContractError("word pool must have at least three distinct words")

    context_lengths = assignments._items(catalog["context_length_bands"], "context_length_bands")
    operations_domains = catalog["_cells"]
    if not operations_domains:
        raise ContractError("catalog contains no compatible operation/domain cells")
    cell_counts = _cell_counts(catalog, history)
    multiple_support_pilot = planning_profile is not None and _support_profile_start(planning_profile["name"]) is not None
    distribution_candidates = _distribution_candidates(catalog, source_kind) if multiple_support_pilot else []
    output_relative = output_path
    try:
        output_value = output_relative.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        output_value = output_relative.as_posix()

    rows: list[dict[str, Any]] = []
    source_deficit = int(budget["source_target_deficits"][source_kind])
    total_deficit = int(budget["production_target_deficit"])
    source_target = int(budget["source_targets"][source_kind])
    total_target = int(budget["production_target"])
    for offset in range(count):
        index = start_index + offset
        derived_seed = _derived_seed(seed, index)
        rng = random.Random(derived_seed)
        if planning_profile is not None and not multiple_support_pilot:
            operation, domain, choice_count, input_structure, profile_sampling = _pick_profile_task(catalog, cell_counts, rng)
            tier = "low" if choice_count <= 4 else "medium" if choice_count <= 16 else "high"
            question_form = rng.choice(assignments._items(operation["question_forms"], "question_forms"))
        else:
            operation, domain = _pick_cell(catalog, cell_counts, rng)
            question_form = rng.choice(assignments._items(operation["question_forms"], "question_forms"))
            # Larger lists require a task and input form that can define many real outcomes.
            low_structures = assignments._items(catalog["input_structures"], "input_structures")
            feasible_tiers: list[str] = ["low"]
            if operation["id"] in MEDIUM_CARDINALITY_OPERATIONS:
                feasible_tiers.append("medium")
            if operation["id"] in HIGH_CARDINALITY_OPERATIONS:
                feasible_tiers.append("high")
            tier = rng.choices(
                feasible_tiers,
                weights=[CHOICE_COUNT_TIER_WEIGHTS[item] for item in feasible_tiers],
                k=1,
            )[0]
            if tier == "low":
                choice_count = rng.choice(CHOICE_COUNT_POLICY["low"])
                input_structures = low_structures
            elif tier == "medium":
                choice_count = rng.choice(CHOICE_COUNT_POLICY["medium"])
                input_structures = [
                    item for item in low_structures if item.get("id") in MEDIUM_CARDINALITY_STRUCTURES
                ]
            else:
                choice_count = rng.choice(CHOICE_COUNT_POLICY["high"])
                input_structures = [
                    item for item in low_structures if item.get("id") in HIGH_CARDINALITY_STRUCTURES
                ]
            if not input_structures:
                raise ContractError(f"catalog has no input structures compatible with {tier} cardinality")
            input_structure = rng.choice(input_structures)
        answerability_status, unanswerable_reason = _choose_answerability(rng, history)
        if multiple_support_pilot:
            selection_material = f"{MULTIPLE_SUPPORT_REQUIREMENT}:{seed}:{index}".encode()
            selection_rng = random.Random(int.from_bytes(hashlib.sha256(selection_material).digest()[:8], "big"))
            if source_kind == probability_v2.HARD_LABEL:
                selection_status = "hard_label"
            elif answerability_status != "answerable":
                selection_status = "unanswerable"
            elif selection_rng.random() >= 0.25:
                selection_status = "not_selected"
            elif not distribution_candidates:
                selection_status = "unsupported_distribution_task"
            else:
                selection_status = "required"
                operation, domain, question_form, input_structure, choice_count = _pick_distribution_task(
                    distribution_candidates, cell_counts, selection_rng,
                )
                tier = "low"
        seed_words = _pick_word_seeds(by_language, rng)
        identifier = f"probability-v2-assignment-{index:010d}"
        group_id = f"probability-v2-group-{index:010d}"
        if identifier in history.identifiers or group_id in history.groups:
            raise ContractError(f"assignment identity already exists at global index {index}")
        row = {
            "schema_version": ASSIGNMENT_SCHEMA_VERSION,
            "id": identifier,
            "problem_group_id": group_id,
            "split": "train",
            "language": language_tag,
            "language_profile": next(
                item for item in catalog["languages"] if item["tag"] == language_tag
            ),
            "domain_id": domain["id"],
            "domain": domain,
            "operation_id": operation["id"],
            "operation": operation,
            "question_form": question_form,
            "input_structure": input_structure,
            "register": rng.choice(assignments._items(catalog["registers"], "registers")),
            "difficulty_band": rng.choice(assignments._items(catalog["difficulty_bands"], "difficulty_bands")),
            "context_length_band": rng.choice(context_lengths),
            "choice_count": choice_count,
            "answerability_status": answerability_status,
            "unanswerable_reason": unanswerable_reason,
            "distribution_source_kind": source_kind,
            "supervision_kind": (
                probability_v2.HARD_LABEL if source_kind == probability_v2.HARD_LABEL else "soft_distribution"
            ),
            "output": output_value,
            "skip_file": output_value.replace("/shard-", "/skipped-shard-"),
            "status": "assigned",
            "language_status": "pending",
            "_generator": {
                "planner_schema_version": ASSIGNMENT_SCHEMA_VERSION,
                "base_seed": seed,
                "derived_seed": derived_seed,
                "global_index": index,
                "seed_words": seed_words,
                "seed_word_source_languages": [word["source_language"] for word in seed_words],
            },
            "_planner": {
                "authoring_guidance_version": AUTHORING_GUIDANCE_VERSION,
                "choice_count": choice_count,
                "choice_count_policy_version": CHOICE_COUNT_POLICY_VERSION,
                "choice_count_tier": tier,
                "answerability_policy_version": ANSWERABILITY_POLICY_VERSION,
                "abstention_choice_required": True,
                "abstention_choice_guidance": ABSTENTION_CHOICE_GUIDANCE,
                "production_target": total_target,
                "production_target_deficit_before_assignment": total_deficit - offset,
                "source_target": source_target,
                "source_target_deficit_before_assignment": source_deficit - offset,
                "source_quota_mix": dict(budget["source_targets"]),
            },
        }
        if planning_profile is not None:
            row["_planner"].update(
                authoring_guidance_version=planning_profile["authoring_guidance_version"],
                planning_profile=planning_profile,
                profile_assignment_number=index - planning_profile["start_index"] + 1,
            )
            if multiple_support_pilot:
                row["_planner"]["target_shape_selection"] = {
                    "status": selection_status, "selection_probability": 0.25,
                    "requirement": MULTIPLE_SUPPORT_REQUIREMENT,
                }
                if selection_status == "required":
                    row["_planner"]["target_shape_requirement"] = MULTIPLE_SUPPORT_REQUIREMENT
            else:
                row["_planner"].update(choice_count_policy_version=DEV_PROFILE_NAME, profile_sampling=profile_sampling)
            history.planning_profiles[planning_profile["name"]] = planning_profile
        rows.append(row)
        cell_counts[_cell_key(operation["id"], domain["id"])] += 1
        history.identifiers.add(identifier)
        history.groups.add(group_id)
        history.indexes.add(index)
        history.assignment_ids.add(identifier)
        history.assignment_groups.add(group_id)
        history.assignment_indexes.add(index)
        history.max_index = max(history.max_index, index)
        history.active_by_source[source_kind] += 1
        history.active_by_language[language_tag] += 1
        history.active_assignments += 1
        history.status_by_source[f"assigned:{source_kind}"] += 1
        history.status_by_language[f"assigned:{language_tag}"] += 1
        history.coverage.add(row, catalog, is_plan=True)
    return rows


def _next_shard(directory: Path) -> int:
    numbers = [
        int(path.stem.removeprefix("shard-"))
        for path in directory.glob("shard-*.jsonl")
        if path.stem.removeprefix("shard-").isdecimal()
    ] if directory.exists() else []
    return max(numbers, default=0) + 1


def _exclusive_planning(function):
    """Keep cooperating planners from allocating the same snapshot concurrently."""
    @functools.wraps(function)
    def locked(*args, **kwargs):
        directory = kwargs.get("assignment_dir", ASSIGNMENT_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".planning.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ContractError("another planner owns this assignment directory") from exc
            return function(*args, **kwargs)
    return locked


@_exclusive_planning
def write_assignment_shard(
    *,
    count: int,
    source_kind: str,
    seed: int = 0,
    language_tag: str | None = None,
    start_index: int | None = None,
    shard_index: int | None = None,
    resume: bool = False,
    catalog_path: Path = CATALOG,
    word_pool_path: Path = WORD_POOL,
    assignment_dir: Path = ASSIGNMENT_DIR,
    question_dir: Path = QUESTION_DIR,
    legacy_dir: Path = LEGACY_DIR,
    pilot_file: Path = PILOT_FILE,
    source_targets: Mapping[str, int] | None = None,
    progress_path: Path | None = None,
    known_issues_path: Path | None = None,
    planning_profile_path: Path | None = None,
) -> dict[str, Any]:
    """Write or resume one homogeneous JSONL assignment shard atomically."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_BATCH:
        raise ContractError(f"count must be 1–{MAX_BATCH}")
    if source_kind not in SOURCE_KINDS:
        raise ContractError(f"unsupported source kind: {source_kind}")
    if resume and shard_index is None:
        raise ContractError("--shard-index is required with --resume")
    if resume and language_tag is None:
        raise ContractError("--language-tag is required with --resume so the existing shard is identified")
    if resume and planning_profile_path is not None:
        raise ContractError("planning profiles apply only to new plans; resume keeps its original profile")
    profile = load_planning_profile(planning_profile_path)

    _require_planning_memory()
    catalog = assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    word_pool = assignments.load_word_pool(word_pool_path)
    question_paths = [question_dir, legacy_dir, pilot_file]
    history = load_history(catalog, question_paths=question_paths, assignment_dir=assignment_dir, known_issues_path=known_issues_path)
    summary = history.summary
    budget = budget_report(summary, history.active_by_source, source_targets)
    selected_language = _choose_language(
        catalog, history, seed=seed, requested_tag=language_tag
    )
    compatibility = probability_task_compatibility(catalog, [source_kind]) if (
        profile is not None and _support_profile_start(profile["name"]) is not None
    ) else None

    source_dir = assignment_dir / source_kind / selected_language
    shard_number = shard_index if shard_index is not None else _next_shard(source_dir)
    if shard_number < 1:
        raise ContractError("shard_index must be one or greater")
    target = source_dir / f"shard-{shard_number:04d}.jsonl"
    if target.exists() != resume:
        expectation = "already exists; use --resume" if target.exists() else "does not exist for --resume"
        raise ContractError(f"{target} {expectation}")

    old = _rows(target) if resume else []
    if resume:
        for row in old:
            if row.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION:
                raise ContractError(f"cannot resume non-v2 assignment shard: {target}")
            if row.get("language") != selected_language or row.get("distribution_source_kind") != source_kind:
                raise ContractError("a shard must keep its original language and source kind")
        indexes = [row.get("_generator", {}).get("global_index") for row in old]
        if not old or any(not isinstance(index, int) or isinstance(index, bool) for index in indexes):
            raise ContractError(f"cannot resume {target}: every row needs a global_index")
        if indexes != list(range(indexes[0], indexes[0] + len(indexes))):
            raise ContractError(f"cannot resume {target}: global_index values must be contiguous and ordered")
        start = indexes[0]
        if start_index is not None and start_index != start:
            raise ContractError(f"start_index must match existing shard start {start}")
        if count < len(old):
            raise ContractError(f"count {count} is less than existing shard size {len(old)}")
        next_index = indexes[-1] + 1
        add_count = count - len(old)
        if add_count and any(row.get("_planner", {}).get("planning_profile") is not None for row in old):
            raise ContractError("bounded profile shards cannot be extended on resume")
    else:
        next_index = max(history.max_index + 1, 1) if start_index is None else start_index
        overlaps = sorted(index for index in history.indexes if next_index <= index < next_index + count)
        if overlaps:
            raise ContractError(f"global_index range overlaps existing history at {overlaps[0]}")
        add_count = count

    if add_count == 0:
        progress = _refresh_progress(
            summary=summary,
            history=history,
            budget=budget,
            progress_path=progress_path or assignment_dir.parent / "progress.json",
        )
        return {
            "output": target.as_posix(),
            "shard_index": shard_number,
            "start_index": next_index - len(old),
            "new_assignments": 0,
            "shard_assignments": len(old),
            "language": selected_language,
            "distribution_source_kind": source_kind,
            "budget_before": budget,
            "status": "already_at_requested_size",
            "progress": progress,
        }
    if any(next_index + offset in history.indexes for offset in range(add_count)):
        raise ContractError("new global_index range overlaps existing assignment history")

    rows = create_plan_rows(
        count=add_count,
        seed=seed,
        start_index=next_index,
        source_kind=source_kind,
        language_tag=selected_language,
        catalog=catalog,
        word_pool=word_pool,
        history=history,
        budget=budget,
        output_path=question_dir / source_kind / selected_language / f"shard-{shard_number:04d}.jsonl",
        planning_profile=profile,
    )
    if profile is not None and load_planning_profile(planning_profile_path) != profile:
        raise ContractError("planning profile changed before plan commit")
    _atomic_jsonl(target, [*old, *rows], overwrite=resume)
    budget_after = budget_report(summary, history.active_by_source, source_targets)
    progress_file = progress_path or assignment_dir.parent / "progress.json"
    _refresh_progress(summary=summary, history=history, budget=budget_after, progress_path=progress_file)
    return {
        "output": target.as_posix(),
        "question_output": rows[0]["output"],
        "shard_index": shard_number,
        "start_index": next_index - len(old) if old else next_index,
        "new_assignments": len(rows),
        "shard_assignments": len(old) + len(rows),
        "language": selected_language,
        "language_status": "pending",
        "distribution_source_kind": source_kind,
        "batch_counts": dict(Counter(row["operation_id"] for row in rows)),
        "budget_before": budget,
        "budget_after_planning": budget_after,
        "unsupported_languages": sorted(history.unsupported_languages),
        "progress_file": progress_file.as_posix(),
        **({"planning_profile": profile} if profile is not None else {}),
        **({"target_shape_planning": {"preallocation_compatibility": compatibility, **_target_shape_report(rows)}}
           if compatibility is not None else {}),
    }


@_exclusive_planning
def write_assignment_batch(
    specs: Iterable[Mapping[str, Any]],
    *,
    seed: int = 0,
    catalog_path: Path = CATALOG,
    word_pool_path: Path = WORD_POOL,
    assignment_dir: Path = ASSIGNMENT_DIR,
    question_dir: Path = QUESTION_DIR,
    legacy_dir: Path = LEGACY_DIR,
    pilot_file: Path = PILOT_FILE,
    source_targets: Mapping[str, int] | None = None,
    progress_path: Path | None = None,
    known_issues_path: Path | None = None,
    planning_profile_path: Path | None = None,
) -> dict[str, Any]:
    """Allocate pinned new shards from one snapshot, bounded to 100,000 rows.

    Each spec requires language_tag, source_kind, and count, and may override seed.
    Existing assignments are never extended. A failed write may leave earlier
    shards committed; retrying reloads their identities and quotas from disk.
    """
    requests: list[dict[str, Any]] = []
    requested_rows = 0
    for spec in specs:
        if not isinstance(spec, Mapping) or set(spec) - {"language_tag", "source_kind", "count", "seed"}:
            raise ContractError("each batch spec needs language_tag, source_kind, count, and optional seed")
        count = spec.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= MAX_BATCH:
            raise ContractError(f"count must be 1–{MAX_BATCH}")
        source = spec.get("source_kind")
        if source not in SOURCE_KINDS:
            raise ContractError(f"unsupported source kind: {source}")
        language = spec.get("language_tag")
        if not _nonempty(language):
            raise ContractError("each batch spec requires a language_tag")
        shard_seed = spec.get("seed", seed)
        if isinstance(shard_seed, bool) or not isinstance(shard_seed, int):
            raise ContractError("seed must be an integer")
        requests.append(dict(count=count, source_kind=source, language_tag=language, seed=shard_seed))
        requested_rows += count
        if requested_rows > MAX_PLAN_BATCH_ROWS:
            raise ContractError(f"batch must contain at most {MAX_PLAN_BATCH_ROWS} rows")
    if not requests:
        raise ContractError("batch spec must contain at least one shard")
    profile = load_planning_profile(planning_profile_path)

    _require_planning_memory()
    catalog = assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    word_pool = assignments.load_word_pool(word_pool_path)
    question_paths = [question_dir, legacy_dir, pilot_file]
    history = load_history(catalog, question_paths=question_paths, assignment_dir=assignment_dir, known_issues_path=known_issues_path)
    summary = history.summary
    initial_budget = budget_report(summary, history.active_by_source, source_targets)
    requested_by_source = Counter()
    for request in requests:
        _choose_language(catalog, history, seed=request["seed"], requested_tag=request["language_tag"])
        requested_by_source[request["source_kind"]] += request["count"]
    for source, count in requested_by_source.items():
        deficit = initial_budget["source_target_deficits"][source]
        if count > deficit:
            raise ContractError(f"{source} deficit is {deficit}, below requested batch count {count}")
    if sum(requested_by_source.values()) > initial_budget["production_target_deficit"]:
        raise ContractError("requested batch exceeds production target deficit")
    _profile_preflight(profile, history, max(history.max_index + 1, 1), requested_rows)
    compatibility = probability_task_compatibility(catalog, requested_by_source) if (
        profile is not None and _support_profile_start(profile["name"]) is not None
    ) else None
    shape_counts: Counter[str] = Counter()
    shape_counts_by_source: dict[str, Counter[str]] = defaultdict(Counter)

    results: list[dict[str, Any]] = []
    next_shards: dict[tuple[str, str], int] = {}
    for request in requests:
        _require_planning_memory()
        source, language = request["source_kind"], request["language_tag"]
        pair = (source, language)
        plan_dir = assignment_dir / source / language
        output_dir = question_dir / source / language
        if pair not in next_shards:
            # Empty or malformed orphan files own their paths even without readable IDs.
            orphan_numbers = [
                int(path.stem.removeprefix("skipped-").removeprefix("shard-"))
                for path in output_dir.glob("*.jsonl")
                if path.stem.removeprefix("skipped-").removeprefix("shard-").isdecimal()
            ]
            next_shards[pair] = max(_next_shard(plan_dir), max(orphan_numbers, default=0) + 1)
        shard_number = next_shards[pair]
        next_shards[pair] += 1
        filename = f"shard-{shard_number:04d}.jsonl"
        target, output = plan_dir / filename, output_dir / filename
        skip = output_dir / f"skipped-{filename}"
        if target.exists() or output.exists() or skip.exists():
            raise ContractError(f"batch shard path already exists: {target}")
        budget = budget_report(summary, history.active_by_source, source_targets)
        start = max(history.max_index + 1, 1)
        rows = create_plan_rows(
            count=request["count"], seed=request["seed"], start_index=start,
            source_kind=source, language_tag=language, catalog=catalog,
            word_pool=word_pool, history=history, budget=budget, output_path=output,
            planning_profile=profile,
        )
        if profile is not None and load_planning_profile(planning_profile_path) != profile:
            raise ContractError("planning profile changed before plan commit")
        if compatibility is not None:
            shape_counts.update(row["_planner"]["target_shape_selection"]["status"] for row in rows)
            shape_counts_by_source[source].update(row["_planner"]["target_shape_selection"]["status"] for row in rows)
        _atomic_jsonl(target, rows, overwrite=False)
        results.append({
            "output": target.as_posix(), "question_output": rows[0]["output"],
            "shard_index": shard_number, "start_index": start,
            "new_assignments": len(rows), "shard_assignments": len(rows),
            "language": language, "language_status": "pending",
            "distribution_source_kind": source, "seed": request["seed"],
            "batch_counts": dict(Counter(row["operation_id"] for row in rows)),
        })
    final_budget = budget_report(summary, history.active_by_source, source_targets)
    progress_file = progress_path or assignment_dir.parent / "progress.json"
    _refresh_progress(summary=summary, history=history, budget=final_budget, progress_path=progress_file)
    return {
        "shards": results, "new_assignments": sum(requested_by_source.values()),
        "budget_before": initial_budget, "budget_after_planning": final_budget,
        "unsupported_languages": sorted(history.unsupported_languages),
        "progress_file": progress_file.as_posix(),
        **({"planning_profile": profile} if profile is not None else {}),
        **({"target_shape_planning": {
            "preallocation_compatibility": compatibility, "planned_marked": shape_counts["required"],
            "selection_status_counts": dict(sorted(shape_counts.items())),
            "selection_status_counts_by_source": {source: dict(sorted(counts.items()))
                                                   for source, counts in sorted(shape_counts_by_source.items())},
            "interpretation": "planned requirements only; actual target shapes and soft-format share are separate",
        }} if compatibility is not None else {}),
    }


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _output_operation_id(row: Mapping[str, Any]) -> str | None:
    """Read operation_id, with v1 task_id/family and operation.id aliases."""
    for key in ("operation_id", "task_id", "family"):
        value = row.get(key)
        if _nonempty(value):
            return value.strip()
    operation = row.get("operation")
    if isinstance(operation, Mapping):
        value = operation.get("id")
        if _nonempty(value):
            return value.strip()
    if _nonempty(operation):
        return operation.strip()
    return None


def _output_domain_id(row: Mapping[str, Any]) -> str | None:
    """Read domain_id or domain.id; topic is accepted only when it equals the ID."""
    value = row.get("domain_id")
    if _nonempty(value):
        return value.strip()
    domain = row.get("domain")
    if isinstance(domain, Mapping):
        value = domain.get("id")
        if _nonempty(value):
            return value.strip()
    if _nonempty(domain):
        return domain.strip()
    topic = row.get("topic")
    return topic.strip() if _nonempty(topic) else None


def reconcile_shard(
    plan_path: Path,
    *,
    output_path: Path | None = None,
    skip_path: Path | None = None,
    assignment_dir: Path = ASSIGNMENT_DIR,
    catalog_path: Path = CATALOG,
    question_dir: Path = QUESTION_DIR,
    legacy_dir: Path = LEGACY_DIR,
    pilot_file: Path = PILOT_FILE,
    source_targets: Mapping[str, int] | None = None,
    progress_path: Path | None = None,
    known_issues_path: Path | None = None,
) -> dict[str, Any]:
    """Close plan rows against valid output or explicit skip records, then refresh progress."""
    plans = _rows(plan_path)
    if not plans:
        raise ContractError(f"assignment plan is empty: {plan_path}")
    if any(row.get("schema_version") != ASSIGNMENT_SCHEMA_VERSION for row in plans):
        raise ContractError(f"unsupported assignment schema in {plan_path}")
    shard_language = plans[0].get("language")
    shard_source = plans[0].get("distribution_source_kind")
    if any(row.get("language") != shard_language or row.get("distribution_source_kind") != shard_source for row in plans):
        raise ContractError("assignment shard must have one language and one source kind")

    resolved_output = output_path or _resolve_path(plans[0]["output"])
    declared_skip = plans[0].get("skip_file")
    resolved_skip = skip_path or (_resolve_path(declared_skip) if _nonempty(declared_skip) else None)
    plan_by_id = {row["id"]: row for row in plans if _nonempty(row.get("id"))}
    if len(plan_by_id) != len(plans):
        raise ContractError("assignment IDs must be non-empty and unique")

    output_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    malformed_output_rows = 0
    orphan_output_rows = 0
    if resolved_output.is_file():
        with resolved_output.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    malformed_output_rows += 1
                    continue
                if not isinstance(row, dict):
                    malformed_output_rows += 1
                    continue
                identifier = row.get("id")
                if isinstance(identifier, str) and identifier in plan_by_id:
                    output_by_id[identifier].append(row)
                else:
                    orphan_output_rows += 1

    skips_by_id: dict[str, dict[str, Any]] = {}
    if resolved_skip is not None and resolved_skip.is_file():
        for row in _rows(resolved_skip):
            identifier = row.get("id", row.get("plan_id"))
            if isinstance(identifier, str) and identifier in plan_by_id:
                skips_by_id[identifier] = row

    counts: Counter[str] = Counter()
    unsupported_reason_seen = False
    for plan in plans:
        identifier = plan["id"]
        previous = plan.get("status")
        candidates = output_by_id.get(identifier, [])
        skip = skips_by_id.get(identifier)
        if candidates and skip is not None:
            plan["status"] = "failed"
            plan["failure_reason"] = "assignment_has_both_output_and_skip_record"
            plan.pop("skip_reason", None)
        elif len(candidates) > 1:
            plan["status"] = "failed"
            plan["failure_reason"] = "duplicate_output_rows_for_assignment_id"
            plan.pop("skip_reason", None)
        elif candidates:
            output = candidates[0]
            errors = probability_v2.validate_v2_row(output)
            reason: str | None = None
            if errors:
                reason = "invalid_v2_output:" + ",".join(errors[:3])
            elif output.get("problem_group_id") != plan.get("problem_group_id"):
                reason = "output_problem_group_id_mismatch"
            elif output.get("language") != plan.get("language"):
                reason = "output_language_mismatch"
            elif output.get("split") != "train":
                reason = "output_split_is_not_train"
            elif output.get("supervision_kind") != plan.get("supervision_kind"):
                reason = "output_supervision_kind_mismatch"
            elif _output_operation_id(output) != plan.get("operation_id"):
                reason = "output_operation_id_mismatch"
            elif _output_domain_id(output) != plan.get("domain_id"):
                reason = "output_domain_id_mismatch"
            elif len(output["options"]) != plan.get("choice_count"):
                reason = "output_choice_count_mismatch"
            elif plan.get("distribution_source_kind") == probability_v2.HARD_LABEL:
                if output.get("supervision_kind") != probability_v2.HARD_LABEL:
                    reason = "hard_label_assignment_has_soft_output"
                elif any(key in output for key in ("target_distribution", "distribution_source_kind", "distribution_evidence")):
                    reason = "hard_label_output_contains_distribution_fields"
            elif output.get("distribution_source_kind") != plan.get("distribution_source_kind"):
                reason = "output_distribution_source_kind_mismatch"

            if reason is None:
                plan["status"] = "completed"
                plan.pop("failure_reason", None)
                plan.pop("skip_reason", None)
            else:
                plan["status"] = "failed"
                plan["failure_reason"] = reason
                plan.pop("skip_reason", None)
        elif skip is not None:
            plan["status"] = "skipped"
            reason = skip.get("reason")
            plan["skip_reason"] = reason.strip() if _nonempty(reason) else "explicit_skip_record"
            plan.pop("failure_reason", None)
            normalized_reason = reason.casefold().replace(" ", "_") if isinstance(reason, str) else ""
            if "unsupported_language" in normalized_reason or "unsupported_tag" in normalized_reason:
                unsupported_reason_seen = True
                plan["language_status"] = "unsupported"
        elif previous in {"completed", "skipped", "failed", "superseded"}:
            pass
        else:
            plan["status"] = "pending"
            plan.pop("failure_reason", None)
            plan.pop("skip_reason", None)
        counts[plan["status"]] += 1

    _atomic_jsonl(plan_path, plans)
    if unsupported_reason_seen:
        set_language_status(
            str(shard_language),
            "unsupported",
            "assignment skip records reported unsupported language",
            catalog_path=catalog_path,
            assignment_dir=assignment_dir,
        )
    refreshed = check_plan_status(
        catalog_path=catalog_path,
        assignment_dir=assignment_dir,
        question_dir=question_dir,
        legacy_dir=legacy_dir,
        pilot_file=pilot_file,
        source_targets=source_targets,
        progress_path=progress_path,
        known_issues_path=known_issues_path,
    )
    return {
        "plan": plan_path.as_posix(),
        "output": resolved_output.as_posix(),
        "skip_file": resolved_skip.as_posix() if resolved_skip else None,
        "plan_rows": len(plans),
        "statuses": dict(sorted(counts.items())),
        "malformed_output_rows": malformed_output_rows,
        "orphan_output_rows": orphan_output_rows,
        "production_target_deficit": refreshed["production_target_deficit"],
        "progress_file": refreshed["progress_file"],
    }


def check_plan_status(
    *,
    catalog_path: Path = CATALOG,
    assignment_dir: Path = ASSIGNMENT_DIR,
    question_dir: Path = QUESTION_DIR,
    legacy_dir: Path = LEGACY_DIR,
    pilot_file: Path = PILOT_FILE,
    source_targets: Mapping[str, int] | None = None,
    seed: int = 0,
    progress_path: Path | None = None,
    known_issues_path: Path | None = None,
) -> dict[str, Any]:
    catalog = assignments.load_catalog(catalog_path, max_choice_count=probability_v2.MAX_CHOICES)
    history = load_history(
        catalog,
        question_paths=[question_dir, legacy_dir, pilot_file],
        assignment_dir=assignment_dir,
        known_issues_path=known_issues_path,
    )
    summary = history.summary
    budget = budget_report(summary, history.active_by_source, source_targets)
    progress_file = progress_path or assignment_dir.parent / "progress.json"
    progress = _refresh_progress(
        summary=summary, history=history, budget=budget, progress_path=progress_file
    )
    recommended = _choose_language(catalog, history, seed=seed, requested_tag=None)
    return {
        **budget,
        "accepted_rows_by_language": summary["accepted_rows_by_language"],
        "active_assignments_by_language": dict(sorted(history.active_by_language.items())),
        "recommended_next_language": recommended,
        "unsupported_languages": sorted(history.unsupported_languages),
        "language_statuses": history.language_statuses,
        "hard_label_rows_credited": summary["production_eligible_rows_by_source_kind"].get(
            probability_v2.HARD_LABEL, 0
        ),
        "pilot_rows_excluded": summary["excluded_pilot_rows"],
        "progress_file": progress_file.as_posix(),
        "progress": progress,
    }


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="create or resume one language/source-kind shard")
    plan.add_argument("--count", type=int, required=True, help="final shard size; with --resume, desired total size")
    plan.add_argument("--source-kind", choices=SOURCE_KINDS, required=True)
    plan.add_argument("--language-tag", help="pin the shard language; otherwise choose the least-covered tag")
    plan.add_argument("--seed", type=int, default=0)
    plan.add_argument("--planning-profile", type=Path, help="pinned bounded profile artifact; new plans only")
    plan.add_argument("--start-index", type=int)
    plan.add_argument("--shard-index", type=int)
    plan.add_argument("--resume", action="store_true")
    plan.add_argument("--catalog", type=Path, default=CATALOG)
    plan.add_argument("--word-pool", type=Path, default=WORD_POOL)
    plan.add_argument("--assignment-dir", type=Path, default=ASSIGNMENT_DIR)
    plan.add_argument("--question-dir", type=Path, default=QUESTION_DIR)
    plan.add_argument("--legacy-dir", type=Path, default=LEGACY_DIR)
    plan.add_argument("--pilot-file", type=Path, default=PILOT_FILE)
    plan.add_argument("--progress-file", type=Path)
    plan.add_argument("--exact-derived-target", type=int, default=DEFAULT_SOURCE_TARGETS["exact_derived"])
    plan.add_argument("--explicit-stochastic-process-target", type=int, default=DEFAULT_SOURCE_TARGETS["explicit_stochastic_process"])
    plan.add_argument("--empirical-frequency-target", type=int, default=DEFAULT_SOURCE_TARGETS["empirical_frequency"])
    plan.add_argument("--teacher-estimate-target", type=int, default=DEFAULT_SOURCE_TARGETS["teacher_estimate"])
    plan.add_argument("--hard-label-target", type=int, default=DEFAULT_SOURCE_TARGETS[probability_v2.HARD_LABEL])

    batch = subparsers.add_parser("plan-batch", help="allocate pinned new shards from one planning snapshot")
    batch.add_argument("--spec", type=Path, required=True, help="JSON array of language_tag/source_kind/count specs; optional seed per shard")
    batch.add_argument("--seed", type=int, default=0)
    batch.add_argument("--planning-profile", type=Path, help="pinned bounded profile artifact")
    batch.add_argument("--catalog", type=Path, default=CATALOG)
    batch.add_argument("--word-pool", type=Path, default=WORD_POOL)
    batch.add_argument("--assignment-dir", type=Path, default=ASSIGNMENT_DIR)
    batch.add_argument("--question-dir", type=Path, default=QUESTION_DIR)
    batch.add_argument("--legacy-dir", type=Path, default=LEGACY_DIR)
    batch.add_argument("--pilot-file", type=Path, default=PILOT_FILE)
    batch.add_argument("--progress-file", type=Path)
    batch.add_argument("--exact-derived-target", type=int, default=DEFAULT_SOURCE_TARGETS["exact_derived"])
    batch.add_argument("--explicit-stochastic-process-target", type=int, default=DEFAULT_SOURCE_TARGETS["explicit_stochastic_process"])
    batch.add_argument("--empirical-frequency-target", type=int, default=DEFAULT_SOURCE_TARGETS["empirical_frequency"])
    batch.add_argument("--teacher-estimate-target", type=int, default=DEFAULT_SOURCE_TARGETS["teacher_estimate"])
    batch.add_argument("--hard-label-target", type=int, default=DEFAULT_SOURCE_TARGETS[probability_v2.HARD_LABEL])

    check = subparsers.add_parser("check", help="show accepted rows, active plans, deficits, and next language")
    check.add_argument("--seed", type=int, default=0)
    check.add_argument("--catalog", type=Path, default=CATALOG)
    check.add_argument("--assignment-dir", type=Path, default=ASSIGNMENT_DIR)
    check.add_argument("--question-dir", type=Path, default=QUESTION_DIR)
    check.add_argument("--legacy-dir", type=Path, default=LEGACY_DIR)
    check.add_argument("--pilot-file", type=Path, default=PILOT_FILE)
    check.add_argument("--progress-file", type=Path)
    check.add_argument("--exact-derived-target", type=int, default=DEFAULT_SOURCE_TARGETS["exact_derived"])
    check.add_argument("--explicit-stochastic-process-target", type=int, default=DEFAULT_SOURCE_TARGETS["explicit_stochastic_process"])
    check.add_argument("--empirical-frequency-target", type=int, default=DEFAULT_SOURCE_TARGETS["empirical_frequency"])
    check.add_argument("--teacher-estimate-target", type=int, default=DEFAULT_SOURCE_TARGETS["teacher_estimate"])
    check.add_argument("--hard-label-target", type=int, default=DEFAULT_SOURCE_TARGETS[probability_v2.HARD_LABEL])

    reconcile = subparsers.add_parser("reconcile", help="match a plan shard to its output and skip files")
    reconcile.add_argument("--plan", type=Path, required=True)
    reconcile.add_argument("--output", type=Path)
    reconcile.add_argument("--skip-file", type=Path)
    reconcile.add_argument("--catalog", type=Path, default=CATALOG)
    reconcile.add_argument("--assignment-dir", type=Path, default=ASSIGNMENT_DIR)
    reconcile.add_argument("--question-dir", type=Path, default=QUESTION_DIR)
    reconcile.add_argument("--legacy-dir", type=Path, default=LEGACY_DIR)
    reconcile.add_argument("--pilot-file", type=Path, default=PILOT_FILE)
    reconcile.add_argument("--progress-file", type=Path)
    reconcile.add_argument("--exact-derived-target", type=int, default=DEFAULT_SOURCE_TARGETS["exact_derived"])
    reconcile.add_argument("--explicit-stochastic-process-target", type=int, default=DEFAULT_SOURCE_TARGETS["explicit_stochastic_process"])
    reconcile.add_argument("--empirical-frequency-target", type=int, default=DEFAULT_SOURCE_TARGETS["empirical_frequency"])
    reconcile.add_argument("--teacher-estimate-target", type=int, default=DEFAULT_SOURCE_TARGETS["teacher_estimate"])
    reconcile.add_argument("--hard-label-target", type=int, default=DEFAULT_SOURCE_TARGETS[probability_v2.HARD_LABEL])

    for command in (plan, batch, check, reconcile):
        command.add_argument("--known-issues-file", type=Path, help="exclusion sidecar; default: assignment directory's sibling known-issues-v1.jsonl")

    mark = subparsers.add_parser("mark-language-status", help="record a supported or unsupported catalog tag")
    mark.add_argument("--language-tag", required=True)
    mark.add_argument("--language-status", choices=("supported", "unsupported"), required=True)
    mark.add_argument("--reason", required=True)
    mark.add_argument("--catalog", type=Path, default=CATALOG)
    mark.add_argument("--assignment-dir", type=Path, default=ASSIGNMENT_DIR)

    args = parser.parse_args(argv)
    try:
        if args.command == "mark-language-status":
            result = set_language_status(
                args.language_tag,
                args.language_status,
                args.reason,
                catalog_path=args.catalog,
                assignment_dir=args.assignment_dir,
            )
        else:
            targets = {
                "exact_derived": args.exact_derived_target,
                "explicit_stochastic_process": args.explicit_stochastic_process_target,
                "empirical_frequency": args.empirical_frequency_target,
                "teacher_estimate": args.teacher_estimate_target,
                probability_v2.HARD_LABEL: args.hard_label_target,
            }
            if args.command == "plan-batch":
                try:
                    specs = json.loads(args.spec.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ContractError(f"cannot read batch spec: {args.spec}") from exc
                if not isinstance(specs, list):
                    raise ContractError("batch spec must be a JSON array")
                result = write_assignment_batch(
                    specs, seed=args.seed, catalog_path=args.catalog,
                    word_pool_path=args.word_pool, assignment_dir=args.assignment_dir,
                    question_dir=args.question_dir, legacy_dir=args.legacy_dir,
                    pilot_file=args.pilot_file, source_targets=targets,
                    progress_path=args.progress_file,
                    known_issues_path=args.known_issues_file,
                    planning_profile_path=args.planning_profile,
                )
            elif args.command == "reconcile":
                result = reconcile_shard(
                    args.plan,
                    output_path=args.output,
                    skip_path=args.skip_file,
                    catalog_path=args.catalog,
                    assignment_dir=args.assignment_dir,
                    question_dir=args.question_dir,
                    legacy_dir=args.legacy_dir,
                    pilot_file=args.pilot_file,
                    source_targets=targets,
                    progress_path=args.progress_file,
                    known_issues_path=args.known_issues_file,
                )
            elif args.command == "check":
                result = check_plan_status(
                    catalog_path=args.catalog,
                    assignment_dir=args.assignment_dir,
                    question_dir=args.question_dir,
                    legacy_dir=args.legacy_dir,
                    pilot_file=args.pilot_file,
                    source_targets=targets,
                    seed=args.seed,
                    progress_path=args.progress_file,
                    known_issues_path=args.known_issues_file,
                )
            else:
                result = write_assignment_shard(
                    count=args.count,
                    source_kind=args.source_kind,
                    seed=args.seed,
                    language_tag=args.language_tag,
                    start_index=args.start_index,
                    shard_index=args.shard_index,
                    resume=args.resume,
                    catalog_path=args.catalog,
                    word_pool_path=args.word_pool,
                    assignment_dir=args.assignment_dir,
                    question_dir=args.question_dir,
                    legacy_dir=args.legacy_dir,
                    pilot_file=args.pilot_file,
                    source_targets=targets,
                    progress_path=args.progress_file,
                    known_issues_path=args.known_issues_file,
                    planning_profile_path=args.planning_profile,
                )
    except (ContractError, assignments.ContractError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
