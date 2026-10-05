"""Stream generated training shards and atomically refresh their progress reports.

Run from the repository root with ``python -m dataset_factory.progress``.
Only non-hidden ``shard-*.jsonl`` files in the generated-train batch and
multilingual directories, plus non-hidden JSONL files in the broad-v1 directory,
are treated as question data. Assignment manifests are inspected separately
and never contribute to question counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TARGET_QUESTIONS = 1_000_000
PROVENANCE = "luna-generated; label not independently verified"
CAMPAIGN_ID_PREFIX = "broad-v1-assignment-"
TRAINING_DIR = Path("data/generated_train")
SHARD_DIRS = (
    TRAINING_DIR / "batches",
    TRAINING_DIR / "multilingual",
    TRAINING_DIR / "broad-v1",
)
SHARD_PATTERNS = ("shard-*.jsonl", "shard-*.jsonl", "*.jsonl")
DIMENSIONS = (
    "language",
    "domain_or_topic",
    "operation",
    "input_structure",
    "register",
    "context_length",
    "choice_count",
    "semantic_outcome",
    "seed_source_language",
)


def normalize(value: str) -> str:
    """Normalize text for deduplication and stable count labels."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _catalog_tags(root: Path) -> set[str] | None:
    catalog_path = root / TRAINING_DIR / "coverage-catalog-v1.json"
    try:
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        languages = catalog.get("languages") if isinstance(catalog, dict) else None
        tags = {
            entry["tag"].strip()
            for entry in languages or []
            if isinstance(entry, dict) and nonempty_string(entry.get("tag"))
        }
        return tags or None
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _language_tag(row: dict[str, Any]) -> str | None:
    language = row.get("language")
    if isinstance(language, dict):
        language = language.get("tag")
    if isinstance(language, str) and language.strip():
        return language.strip()
    profile = row.get("language_profile")
    if isinstance(profile, dict) and nonempty_string(profile.get("tag")):
        return profile["tag"].strip()
    return None


def validate_row(
    row: Any,
    valid_languages: set[str] | None = None,
    *,
    max_choices: int = 4,
) -> list[str]:
    errors: list[str] = []
    if not isinstance(row, dict):
        return ["row_is_not_an_object"]

    required_strings = (
        "id",
        "problem_group_id",
        "split",
        "context",
        "question",
        "explanation",
        "provenance",
    )
    for key in required_strings:
        if not nonempty_string(row.get(key)):
            errors.append(f"missing_or_empty_{key}")

    has_domain = any(nonempty_string(row.get(key)) for key in ("topic", "domain", "domain_id"))
    if isinstance(row.get("domain"), dict):
        has_domain = has_domain or nonempty_string(row["domain"].get("id"))
    if not has_domain:
        errors.append("missing_or_empty_topic_or_domain")
    has_operation = any(
        nonempty_string(row.get(key)) for key in ("family", "task_id", "operation", "operation_id")
    )
    if isinstance(row.get("operation"), dict):
        has_operation = has_operation or nonempty_string(row["operation"].get("id"))
    if not has_operation:
        errors.append("missing_or_empty_operation")
    if row.get("split") != "train":
        errors.append("split_is_not_train")
    language = _language_tag(row)
    if language is None:
        errors.append("missing_or_empty_language")
    elif valid_languages is not None and language not in valid_languages:
        errors.append("language_not_in_coverage_catalog")

    difficulty = row.get("difficulty", row.get("difficulty_band"))
    if isinstance(difficulty, dict):
        difficulty = difficulty.get("id")
    if difficulty not in {"easy", "medium", "hard"}:
        errors.append("unsupported_difficulty")

    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= max_choices:
        errors.append(f"options_must_contain_two_to_{max_choices}_strings")
    elif any(not nonempty_string(option) for option in options):
        errors.append(f"options_must_contain_two_to_{max_choices}_strings")
    elif len({normalize(option) for option in options}) != len(options):
        errors.append("options_are_not_distinct")

    answer_index = row.get("answer_index")
    if not isinstance(answer_index, int) or isinstance(answer_index, bool):
        errors.append("answer_index_is_not_an_integer")
    elif isinstance(options, list) and not 0 <= answer_index < len(options):
        errors.append("answer_index_is_out_of_range")
    choice_count = row.get("choice_count")
    if choice_count is not None:
        if isinstance(choice_count, bool) or not isinstance(choice_count, int):
            errors.append("choice_count_is_not_an_integer")
        elif isinstance(options, list) and choice_count != len(options):
            errors.append("choice_count_does_not_match_options")

    provenance = row.get("provenance")
    if nonempty_string(provenance):
        normalized_provenance = normalize(provenance).rstrip(".")
        if normalized_provenance != PROVENANCE:
            errors.append("provenance_does_not_match_required_label")
    return errors


def _explicit_label(row: dict[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        value = row.get(key)
        if nonempty_string(value):
            return " ".join(unicodedata.normalize("NFKC", value).split())
        if isinstance(value, dict):
            identifier = value.get("id", value.get("tag"))
            if nonempty_string(identifier):
                return " ".join(unicodedata.normalize("NFKC", identifier).split())
    return None


def _generation_metadata(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("_generator", "generation_metadata"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def infer_input_structure(context: str) -> str:
    explicit_lines = context.splitlines()
    if sum("|" in line for line in explicit_lines) >= 2:
        return "table"
    if any(re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line) for line in explicit_lines):
        return "list"
    if re.search(r"(?:^|\s)(?:[A-Z]{1,4}[-_]\d+|[A-Z]\d+)\s*[—–:-]", context):
        return "record_list"
    if re.search(r"(?:→|->|\bfirst\b.*\bthen\b|\bzuerst\b.*\bdann\b)", context, re.I):
        return "ordered_steps"
    if re.search(r"\b(if|when|unless|policy|rule|regel|wenn|falls|vorgabe)\b", context, re.I):
        return "rule_prose"
    return "prose"


def infer_semantic_outcome(row: dict[str, Any]) -> str:
    explicit = _explicit_label(row, ("semantic_outcome", "outcome_type"))
    if explicit is None:
        explicit = _explicit_label(_generation_metadata(row), ("outcome_class",))
    if explicit:
        return explicit

    options = row.get("options", [])
    answer = options[row["answer_index"]]
    answer_normalized = normalize(answer).strip(" .!?。")
    if answer_normalized in {"yes", "no", "ja", "nein", "true", "false"}:
        return "binary"

    operation = normalize(
        _explicit_label(row, ("operation_id", "operation", "task_id", "family")) or ""
    )
    if re.search(r"\b(arithmetic|numeric|calculate|calculation|count|total|duration|arrival|remaining|fee|amount|math)\b", operation):
        return "numeric_result"
    if re.search(r"\b(order|sequence|schedule|route|path|workflow|rank|sort|trip)\b", operation):
        return "ordered_or_route_choice"
    if re.search(r"\b(classif|categor|label|status|band|validate|violation)\b", operation):
        return "classification"
    if re.search(r"^[-+]?\d+(?:[.,]\d+)?(?:\s*[a-z%€$£]+)?$", answer_normalized):
        return "numeric_result"
    if re.search(r"\b(stop|hold|allow|deny|approve|reject|send|pause|use|choose|select|handle|review|keep|delete|store|label|print|deploy|archive)\b", answer_normalized):
        return "action_choice"
    return "categorical_choice"


def dimensions_for(row: dict[str, Any]) -> dict[str, Any]:
    options = row["options"]
    context = row["context"]
    word_count = len(context.split())
    if word_count <= 40:
        context_length = "0-40"
    elif word_count <= 80:
        context_length = "41-80"
    elif word_count <= 120:
        context_length = "81-120"
    else:
        context_length = "121+"

    operation = _explicit_label(row, ("operation_id", "operation", "operation_type", "family", "task_id"))
    structure = _explicit_label(row, ("input_structure", "input_format"))
    source_languages: set[str] = set()
    explicit_sources = row.get("seed_source_languages", row.get("seed_source_language"))
    if isinstance(explicit_sources, str) and explicit_sources.strip():
        source_languages.add(normalize(explicit_sources))
    elif isinstance(explicit_sources, list):
        for source in explicit_sources:
            if isinstance(source, str) and source.strip():
                source_languages.add(normalize(source))
            elif isinstance(source, dict) and nonempty_string(source.get("tag")):
                source_languages.add(normalize(source["tag"]))
    if not source_languages:
        source = _explicit_label(row, ("source_language", "seed_language"))
        if source:
            source_languages.add(normalize(source))
    if not source_languages:
        seed_words = _generation_metadata(row).get("seed_words")
        if isinstance(seed_words, list):
            source_languages = {
                normalize(word["source_language"])
                for word in seed_words
                if isinstance(word, dict) and nonempty_string(word.get("source_language"))
            }
    domain = _explicit_label(row, ("domain_id", "domain", "topic")) or "not_recorded"
    register = _explicit_label(row, ("register", "style_register")) or "not_recorded"
    context_band = _explicit_label(row, ("context_length_band", "context_length"))
    return {
        "language": _language_tag(row) or "not_recorded",
        "domain_or_topic": normalize(domain),
        "operation": normalize(operation or "not_recorded"),
        "input_structure": normalize(structure) if structure else infer_input_structure(context),
        "register": normalize(register),
        "context_length": normalize(context_band) if context_band else context_length,
        "choice_count": str(len(options)),
        "semantic_outcome": normalize(infer_semantic_outcome(row)),
        "seed_source_language": sorted(source_languages) if source_languages else ["not_recorded"],
    }


def _jsonl_shards(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    found: list[list[Path]] = [[], [], []]
    for index, (relative_dir, pattern) in enumerate(zip(SHARD_DIRS, SHARD_PATTERNS)):
        directory = root / relative_dir
        if directory.is_dir():
            found[index] = sorted(
                path
                for path in directory.glob(pattern)
                if path.is_file() and not path.name.startswith(".")
            )
    return found[0], found[1], found[2]


def _observed_identity(row: Any) -> tuple[str | None, str | None, str | None]:
    if not isinstance(row, dict):
        return None, None, None

    def clean(key: str) -> str | None:
        value = row.get(key)
        return value.strip() if isinstance(value, str) and value.strip() else None

    return clean("id"), clean("problem_group_id"), clean("split")


def _assignment_status(root: Path) -> dict[str, Any]:
    """Read recognizable assignment manifests without treating them as shards."""
    training_root = root / TRAINING_DIR
    manifests: list[Path] = []
    if training_root.is_dir():
        for path in training_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl"}:
                continue
            relative = str(path.relative_to(training_root)).casefold()
            if "assign" not in relative and "worker-status" not in path.name.casefold():
                continue
            manifests.append(path)

    states: Counter[str] = Counter()
    expected_rows: Counter[str] = Counter()
    recognized = 0

    def visit(value: Any, implicit_assignment: bool = False) -> None:
        nonlocal recognized
        if isinstance(value, dict):
            status = value.get("status")
            normalized = normalize(status).replace(" ", "_").replace("-", "_") if isinstance(status, str) else None
            record = any(
                nonempty_string(value.get(key))
                for key in ("id", "assignment_id", "worker", "output")
            )
            if normalized in {"assigned", "queued", "pending", "in_flight", "running", "in_progress"} and record:
                state = "active_assigned" if normalized in {"assigned", "queued", "pending"} else "active_in_flight"
                states[state] += 1
                count = value.get("expected_rows")
                if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    expected_rows[state] += count
                recognized += 1
                return
            if normalized is not None and record:
                return
            if implicit_assignment and record:
                state = "active_assigned"
                states[state] += 1
                count = value.get("expected_rows")
                if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    expected_rows[state] += count
                recognized += 1
                return
            for child in value.values():
                visit(child, implicit_assignment)
        elif isinstance(value, list):
            for child in value:
                visit(child, implicit_assignment)

    for path in manifests:
        try:
            if path.suffix.lower() == ".jsonl":
                with path.open("r", encoding="utf-8") as stream:
                    for line in stream:
                        if line.strip():
                            visit(json.loads(line), "assignments" in path.parts)
            else:
                visit(json.loads(path.read_text(encoding="utf-8")), "assignments" in path.parts)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

    return {
        "manifest_files": [path.relative_to(root).as_posix() for path in sorted(manifests)],
        "recognized_status_records": recognized,
        "active_assigned": states["active_assigned"],
        "active_in_flight": states["active_in_flight"],
        "active_assigned_expected_rows": expected_rows["active_assigned"],
        "active_in_flight_expected_rows": expected_rows["active_in_flight"],
        "status": "discovered" if recognized else "no_recognized_assignment_statuses",
    }


def scan(root: Path, target: int) -> dict[str, Any]:
    batch_files, multilingual_files, broad_files = _jsonl_shards(root)
    files = batch_files + multilingual_files + broad_files
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    valid_languages = _catalog_tags(root)
    raw_lines = 0
    malformed_json = 0
    schema_rejected = 0
    structurally_valid = 0
    campaign_parseable = 0
    campaign_schema_rejected = 0
    campaign_structurally_valid = 0
    campaign_accepted = 0
    duplicate_ids = 0
    duplicate_content = 0
    source_summaries: dict[str, dict[str, Any]] = {
        path.as_posix(): {
            "parseable_rows": 0,
            "parseable_rows_by_language": Counter(),
            "structurally_valid_rows": 0,
        }
        for path in SHARD_DIRS
    }

    with tempfile.TemporaryDirectory(prefix="dataset-progress-") as temp_dir:
        db_path = Path(temp_dir) / "progress.sqlite3"
        connection = sqlite3.connect(db_path)
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE audit (
                row_no INTEGER PRIMARY KEY,
                path TEXT NOT NULL,
                line_no INTEGER NOT NULL,
                status TEXT NOT NULL,
                errors TEXT NOT NULL,
                item_id TEXT,
                group_id TEXT,
                split TEXT,
                duplicate_id INTEGER NOT NULL DEFAULT 0,
                duplicate_content INTEGER NOT NULL DEFAULT 0,
                dimensions TEXT
            );
            CREATE INDEX audit_group_idx ON audit(group_id);
            CREATE INDEX audit_id_idx ON audit(item_id);
            CREATE TABLE seen_ids (value TEXT PRIMARY KEY);
            CREATE TABLE seen_content (value TEXT PRIMARY KEY);
            CREATE TABLE observed_identity (
                row_no INTEGER PRIMARY KEY,
                item_id TEXT NOT NULL,
                group_id TEXT NOT NULL,
                split TEXT NOT NULL
            );
            CREATE INDEX observed_group_idx ON observed_identity(group_id);
            CREATE INDEX observed_id_idx ON observed_identity(item_id);
            """
        )

        for path in files:
            relative_path = path.relative_to(root).as_posix()
            with path.open("rb") as stream:
                for line_no, raw_bytes in enumerate(stream, 1):
                    raw_lines += 1
                    row_no = raw_lines
                    try:
                        raw_line = raw_bytes.decode("utf-8")
                        row = json.loads(raw_line)
                    except (json.JSONDecodeError, UnicodeError):
                        malformed_json += 1
                        connection.execute(
                            "INSERT INTO audit(row_no,path,line_no,status,errors) VALUES(?,?,?,?,?)",
                            (row_no, relative_path, line_no, "malformed_json", json.dumps(["invalid_json"])),
                        )
                        continue

                    campaign_row = (
                        isinstance(row, dict)
                        and isinstance(row.get("id"), str)
                        and row["id"].startswith(CAMPAIGN_ID_PREFIX)
                    )
                    campaign_parseable += int(campaign_row)

                    source_dir = next(
                        (part.as_posix() for part in SHARD_DIRS if relative_path.startswith(part.as_posix() + "/")),
                        None,
                    )
                    if source_dir is not None and isinstance(row, dict):
                        source_summaries[source_dir]["parseable_rows"] += 1
                        language = _language_tag(row)
                        if language:
                            source_summaries[source_dir]["parseable_rows_by_language"][language] += 1

                    item_id, group_id, split = _observed_identity(row)
                    if item_id and group_id and split:
                        connection.execute(
                            "INSERT INTO observed_identity(row_no,item_id,group_id,split) VALUES(?,?,?,?)",
                            (row_no, item_id, group_id, split),
                        )

                    errors = validate_row(row, valid_languages)
                    if errors:
                        schema_rejected += 1
                        campaign_schema_rejected += int(campaign_row)
                        connection.execute(
                            "INSERT INTO audit(row_no,path,line_no,status,errors,item_id,group_id,split) VALUES(?,?,?,?,?,?,?,?)",
                            (
                                row_no,
                                relative_path,
                                line_no,
                                "schema_rejected",
                                json.dumps(errors, ensure_ascii=False),
                                item_id,
                                group_id,
                                split,
                            ),
                        )
                        continue

                    structurally_valid += 1
                    campaign_structurally_valid += int(campaign_row)
                    if source_dir is not None:
                        source_summaries[source_dir]["structurally_valid_rows"] += 1
                    id_value = row["id"].strip()
                    duplicate_id = connection.execute(
                        "INSERT OR IGNORE INTO seen_ids(value) VALUES(?)", (id_value,)
                    ).rowcount == 0
                    canonical = json.dumps(
                        [
                            normalize(row["context"]),
                            normalize(row["question"]),
                            [normalize(option) for option in row["options"]],
                        ],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    content_key = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                    duplicate_text = connection.execute(
                        "INSERT OR IGNORE INTO seen_content(value) VALUES(?)", (content_key,)
                    ).rowcount == 0
                    duplicate_ids += int(duplicate_id)
                    duplicate_content += int(duplicate_text)
                    dims = dimensions_for(row)
                    connection.execute(
                        """INSERT INTO audit(
                            row_no,path,line_no,status,errors,item_id,group_id,split,
                            duplicate_id,duplicate_content,dimensions
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            row_no,
                            relative_path,
                            line_no,
                            "valid",
                            "[]",
                            id_value,
                            row["problem_group_id"].strip(),
                            row["split"].strip(),
                            int(duplicate_id),
                            int(duplicate_text),
                            json.dumps(dims, ensure_ascii=False, sort_keys=True),
                        ),
                    )

        connection.executescript(
            """
            CREATE TABLE repeated_groups AS
                SELECT group_id FROM observed_identity GROUP BY group_id HAVING COUNT(*) > 1;
            CREATE UNIQUE INDEX repeated_groups_idx ON repeated_groups(group_id);
            CREATE TABLE split_groups AS
                SELECT group_id FROM observed_identity GROUP BY group_id HAVING COUNT(DISTINCT split) > 1;
            CREATE UNIQUE INDEX split_groups_idx ON split_groups(group_id);
            CREATE TABLE split_ids AS
                SELECT item_id FROM observed_identity GROUP BY item_id HAVING COUNT(DISTINCT split) > 1;
            CREATE UNIQUE INDEX split_ids_idx ON split_ids(item_id);
            """
        )

        duplicate_group_ids = connection.execute("SELECT COUNT(*) FROM repeated_groups").fetchone()[0]
        duplicate_group_rows = connection.execute(
            "SELECT COUNT(*) FROM observed_identity WHERE group_id IN repeated_groups"
        ).fetchone()[0]
        groups_spanning_splits = connection.execute("SELECT COUNT(*) FROM split_groups").fetchone()[0]
        rows_in_split_groups = connection.execute(
            "SELECT COUNT(*) FROM observed_identity WHERE group_id IN split_groups"
        ).fetchone()[0]
        ids_spanning_splits = connection.execute("SELECT COUNT(*) FROM split_ids").fetchone()[0]
        rows_in_split_ids = connection.execute(
            "SELECT COUNT(*) FROM observed_identity WHERE item_id IN split_ids"
        ).fetchone()[0]

        dimensions: dict[str, Counter[str]] = {dimension: Counter() for dimension in DIMENSIONS}
        accepted_count = 0
        quarantine: list[dict[str, Any]] = []
        batch_valid_count = 0

        cursor = connection.execute(
            """SELECT a.row_no,a.path,a.line_no,a.status,a.errors,a.item_id,a.group_id,a.split,
                      a.duplicate_id,a.duplicate_content,a.dimensions,
                      (g.group_id IS NOT NULL),(sg.group_id IS NOT NULL),(si.item_id IS NOT NULL)
               FROM audit AS a
               LEFT JOIN repeated_groups AS g ON a.group_id=g.group_id
               LEFT JOIN split_groups AS sg ON a.group_id=sg.group_id
               LEFT JOIN split_ids AS si ON a.item_id=si.item_id
               ORDER BY a.row_no"""
        )
        for (
            row_no,
            relative_path,
            line_no,
            status,
            errors_json,
            item_id,
            group_id,
            split,
            duplicate_id,
            duplicate_text,
            dimensions_json,
            group_reused,
            group_crosses_splits,
            id_crosses_splits,
        ) in cursor:
            reasons = json.loads(errors_json)
            if status == "valid":
                if duplicate_id:
                    reasons.append("duplicate_id")
                if duplicate_text:
                    reasons.append("duplicate_normalized_context_question_options")
                if group_reused:
                    reasons.append("problem_group_id_reused")
                if group_crosses_splits:
                    reasons.append("problem_group_spans_splits")
                if id_crosses_splits:
                    reasons.append("id_spans_splits")

                if not reasons:
                    accepted_count += 1
                    campaign_accepted += int(
                        isinstance(item_id, str) and item_id.startswith(CAMPAIGN_ID_PREFIX)
                    )
                    for dimension, value in json.loads(dimensions_json).items():
                        values = value if isinstance(value, list) else [value]
                        for category in set(values):
                            dimensions[dimension][category] += 1

            if reasons:
                quarantine.append(
                    {"file": relative_path, "line": line_no, "reasons": sorted(set(reasons))}
                )

        connection.close()

    remaining = max(0, target - campaign_accepted)
    assignment = _assignment_status(root)
    return {
        "generated_at": timestamp,
        "target_questions": target,
        "remaining_questions": remaining,
        "all_source_remaining_questions": max(0, target - accepted_count),
        "over_target_by": max(0, campaign_accepted - target),
        "scope": {
            "scanned_directories": [path.as_posix() for path in SHARD_DIRS],
            "file_patterns": {
                path.as_posix(): pattern for path, pattern in zip(SHARD_DIRS, SHARD_PATTERNS)
            },
            "files_scanned": [path.relative_to(root).as_posix() for path in files],
            "assignment_files_are_not_question_data": True,
        },
        "row_counts": {
            "physical_lines": raw_lines,
            "malformed_json_rows": malformed_json,
            "schema_rejected_rows": schema_rejected,
            "malformed_or_schema_rejected_rows": malformed_json + schema_rejected,
            "structurally_valid_rows_before_deduplication": structurally_valid,
            "duplicate_id_rows": duplicate_ids,
            "duplicate_normalized_context_question_options_rows": duplicate_content,
            "duplicate_problem_group_ids": duplicate_group_ids,
            "rows_in_reused_problem_groups": duplicate_group_rows,
            "groups_spanning_splits": groups_spanning_splits,
            "rows_in_groups_spanning_splits": rows_in_split_groups,
            "ids_spanning_splits": ids_spanning_splits,
            "rows_with_ids_spanning_splits": rows_in_split_ids,
            "accepted_rows": accepted_count,
            "campaign_parseable_rows": campaign_parseable,
            "campaign_schema_rejected_rows": campaign_schema_rejected,
            "campaign_structurally_valid_rows_before_deduplication": campaign_structurally_valid,
            "campaign_accepted_rows": campaign_accepted,
            "quarantined_rows": len(quarantine),
        },
        "campaign_target_definition": (
            f"Accepted rows with IDs beginning {CAMPAIGN_ID_PREFIX!r}; rows from older datasets "
            "are reported in all-source counts but do not advance the new campaign target."
        ),
        "accepted_definition": (
            "A row is accepted only when its structure passes this schema, its ID and normalized "
            "context+question+ordered-options are unique, and it has no reused problem-group or "
            "cross-split identity conflict. Acceptance does not mean the answer, explanation, or "
            "reasoning was semantically verified."
        ),
        "schema": {
            "required": [
                "id", "problem_group_id", "split=train", "language tag",
                "family/task_id or operation/operation_id", "topic/domain/domain_id",
                "difficulty or difficulty_band=easy|medium|hard", "context", "question",
                "options=2..4 distinct nonempty strings", "answer_index=zero-based in range",
                "explanation", "provenance=Luna-generated; label not independently verified",
            ],
            "additional_fields": "Allowed; supported metadata fields are used when present.",
        },
        "deduplication": {
            "text_normalization": "Unicode NFKC, case-fold, and collapse whitespace; option order is preserved.",
            "content_key": "SHA-256 of normalized context, question, and ordered options.",
            "first_occurrence_wins_for_id_and_content": True,
        },
        "counts_by_dimension": {key: dict(sorted(value.items())) for key, value in dimensions.items()},
        "source_summaries": {
            path: {
                "shard_file_count": len(
                    [file for file in files if file.relative_to(root).as_posix().startswith(path + "/")]
                ),
                "parseable_rows": summary["parseable_rows"],
                "parseable_rows_by_language": dict(sorted(summary["parseable_rows_by_language"].items())),
                "structurally_valid_rows": summary["structurally_valid_rows"],
            }
            for path, summary in source_summaries.items()
        },
        "language_validation": {
            "catalog_file": (TRAINING_DIR / "coverage-catalog-v1.json").as_posix(),
            "method": "coverage catalog tags" if valid_languages is not None else "nonempty language tag",
            "valid_catalog_tag_count": len(valid_languages) if valid_languages is not None else None,
        },
        "assignment_status": assignment,
        "quarantine_references": quarantine,
        "source_lines_preserved": True,
        "interpretation_notes": [
            "Input structure and semantic outcome are inferred heuristically when explicit metadata is absent.",
            "Register is not_recorded unless explicit. Seed-source-language uses explicit labels or generator seed-word metadata, otherwise not_recorded; rows with multiple source languages count under each tag.",
            "Context length uses an explicit context_length_band when present, otherwise whitespace-delimited word-count buckets: 0-40, 41-80, 81-120, and 121+.",
            "A question with multiple seed-source languages is counted once under each source tag.",
            "Quarantine references point to original file and line; source shard lines are never rewritten or deleted.",
        ],
    }


def _markdown(report: dict[str, Any]) -> str:
    counts = report["row_counts"]
    summaries = report["source_summaries"]
    batch_summary = summaries[(TRAINING_DIR / "batches").as_posix()]
    assignment = report["assignment_status"]
    lines = [
        "# Generated training data progress",
        "",
        f"Updated: `{report['generated_at']}`",
        "",
        f"Luna assignment questions accepted: **{counts['campaign_accepted_rows']:,} / {report['target_questions']:,}**; "
        f"remaining: **{report['remaining_questions']:,}**.",
        "",
        f"All-source accepted rows: **{counts['accepted_rows']:,}**. Older or unassigned rows are reported here "
        "but do not advance the new campaign target.",
        "",
        "Accepted means only that a row passed structural checks, deduplication, and split/group conflict checks. "
        "It does not mean the answer, explanation, or reasoning has been semantically verified.",
        "",
        "Source JSONL lines remain unchanged. Quarantine references identify rejected or duplicate rows by file and line.",
        "",
        "## Scan counts",
        "",
        "| Measure | Count |",
        "|---|---:|",
    ]
    for label, key in (
        ("Shard files scanned", None),
        ("Physical lines", "physical_lines"),
        ("Malformed JSON rows", "malformed_json_rows"),
        ("Schema-rejected rows", "schema_rejected_rows"),
        ("Structurally valid rows before deduplication", "structurally_valid_rows_before_deduplication"),
        ("Duplicate ID rows", "duplicate_id_rows"),
        ("Duplicate normalized context/question/options rows", "duplicate_normalized_context_question_options_rows"),
        ("Reused problem-group IDs", "duplicate_problem_group_ids"),
        ("Groups spanning splits", "groups_spanning_splits"),
        ("IDs spanning splits", "ids_spanning_splits"),
        ("Accepted rows", "accepted_rows"),
        ("Luna assignment campaign rows accepted", "campaign_accepted_rows"),
        ("Quarantined rows", "quarantined_rows"),
    ):
        count = len(report["scope"]["files_scanned"]) if key is None else counts[key]
        lines.append(f"| {label} | {count:,} |")

    lines.extend(
        [
            "",
            "## Initial EN/DE batch shards",
            "",
            f"{batch_summary['shard_file_count']} files contain {batch_summary['structurally_valid_rows']:,} structurally valid rows. "
            "Parseable rows by output-language tag: "
            + (", ".join(
                f"{language} {count:,}"
                for language, count in batch_summary["parseable_rows_by_language"].items()
            ) or "none")
            + ". Language-tag membership does not assess translation or language quality.",
            "",
            "## Assigned and in-flight work",
            "",
        ]
    )
    if assignment["recognized_status_records"]:
        lines.append(
            f"Recognized {assignment['active_assigned']:,} assigned and "
            f"{assignment['active_in_flight']:,} in-flight records in assignment manifests "
            f"({assignment['active_assigned_expected_rows']:,} and "
            f"{assignment['active_in_flight_expected_rows']:,} expected rows, respectively). "
            "These manifests are excluded from generated-question counts."
        )
    else:
        lines.append("No recognizable assignment-status manifest was found under `data/generated_train/`.")

    lines.extend(["", "## Counts by dimension", ""])
    for dimension, values in report["counts_by_dimension"].items():
        lines.extend([f"### {dimension.replace('_', ' ').title()}", "", "| Value | Accepted rows |", "|---|---:|"])
        if values:
            ordered_values = sorted(values.items(), key=lambda item: (-item[1], item[0]))
            limit = len(ordered_values) if dimension == "language" else min(15, len(ordered_values))
            lines.extend(
                f"| {value.replace('|', '\\|')} | {count:,} |"
                for value, count in ordered_values[:limit]
            )
            omitted = len(ordered_values) - limit
            if omitted:
                lines.append(f"| {omitted:,} additional categories | See `progress.json` |")
        else:
            lines.append("| No accepted rows | 0 |")
        lines.append("")

    lines.extend(
        [
            "## Counting rules",
            "",
            "The scanner reads non-hidden `shard-*.jsonl` files line by line from `data/generated_train/batches/` and "
            "`data/generated_train/multilingual/`, plus `*.jsonl` from `data/generated_train/broad-v1/`. "
            f"Only rows whose ID begins `{CAMPAIGN_ID_PREFIX}` advance the new campaign target. "
            "It requires a train row, a nonempty output-language tag (validated against the coverage catalog "
            "when available), easy/medium/hard difficulty, the question fields listed in `progress.json`, two to four distinct "
            "options, an in-range zero-based answer index, and the required Luna provenance label.",
            "",
            "Dimension counts include accepted rows only. Output-language tags are counted as labels; catalog "
            "membership does not imply translation or language quality. Operation uses explicit operation metadata, "
            "then family, then task ID. Domain uses `domain` when present and otherwise `topic`. Input structure and "
            "semantic outcome use explicit metadata when available and otherwise a lightweight text heuristic. "
            "Register is `not_recorded` unless explicit. Seed-source-language uses explicit labels or generator seed-word metadata, otherwise `not_recorded`; rows with multiple source languages count under each tag.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp_path, path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Count generated training questions and refresh progress reports.")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1], help="Repository root.")
    parser.add_argument("--target", type=int, default=TARGET_QUESTIONS, help="Accepted-question target.")
    args = parser.parse_args(argv)
    if args.target < 0:
        parser.error("--target must be non-negative")

    root = args.root.resolve()
    report = scan(root, args.target)
    json_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    markdown_text = _markdown(report)
    _atomic_write(root / TRAINING_DIR / "progress.json", json_text)
    _atomic_write(root / TRAINING_DIR / "progress.md", markdown_text)
    print(
        f"Campaign accepted {report['row_counts']['campaign_accepted_rows']:,} / {args.target:,}; "
        f"wrote data/generated_train/progress.json and progress.md"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
