"""Validate and summarize the versioned probability-supervision dataset contract.

Run ``python -m dataset_factory.probability_v2 summary PATH [PATH ...]`` to
count accepted v2 rows and structurally valid legacy hard-label rows.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sqlite3
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dataset_factory import progress


SCHEMA_VERSION = "probability-v2"
MIN_CHOICES = 2
MAX_CHOICES = 265
ANSWERABILITY_STATUSES = frozenset({"answerable", "unanswerable"})
UNANSWERABLE_REASONS = frozenset(
    {
        "missing_context",
        "underdetermined",
        "external_fact_unavailable",
        "realized_random_outcome_unknown",
    }
)
SOFT_SOURCE_KINDS = frozenset(
    {
        "exact_derived",
        "explicit_stochastic_process",
        "empirical_frequency",
        "teacher_estimate",
    }
)
HARD_LABEL = "hard_label"
EXCLUDED_PILOT_STATUS = "illustrative_not_mass_training"
SUM_TOLERANCE = 1e-9
ENTROPY_BANDS = (
    (0.0, 0.5, "0-0.5"),
    (0.5, 1.0, "0.5-1.0"),
    (1.0, 1.5, "1.0-1.5"),
    (1.5, 2.0, "1.5-2.0"),
)
_CONFIDENCE_KEYS = {"confidence", "confidence_score", "answer_confidence"}


class ContractError(ValueError):
    """Raised by helpers when a requested v2 row cannot be represented safely."""


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _choice_ids(row: Mapping[str, Any]) -> list[str] | None:
    values = row.get("choice_ids")
    if not isinstance(values, list) or any(not _nonempty(value) for value in values):
        return None
    if len(set(values)) != len(values):
        return None
    return values


def stable_choice_ids(question_id: str, options: Sequence[str]) -> list[str]:
    """Create opaque IDs tied to a question and option text, independent of order."""
    if not _nonempty(question_id):
        raise ContractError("question_id must be a non-empty string")
    if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
        raise ContractError("options must be a sequence of strings")
    normalized: list[str] = []
    for option in options:
        if not _nonempty(option):
            raise ContractError("options must contain non-empty strings")
        normalized.append(progress.normalize(option))
    if len(set(normalized)) != len(normalized):
        raise ContractError("options must be distinct after normalization")
    identifiers = []
    for option in normalized:
        digest = hashlib.sha256(f"{question_id}\0{option}".encode("utf-8")).hexdigest()[:20]
        identifiers.append(f"choice-{digest}")
    return identifiers


def _fraction(value: Any, location: str) -> Fraction:
    if not isinstance(value, dict):
        raise ContractError(f"{location} must be an object with numerator and denominator")
    numerator = value.get("numerator")
    denominator = value.get("denominator")
    if not _is_int(numerator) or not _is_int(denominator) or denominator <= 0:
        raise ContractError(f"{location} needs an integer numerator and positive integer denominator")
    result = Fraction(numerator, denominator)
    if result < 0 or result > 1:
        raise ContractError(f"{location} must be between zero and one")
    return result


def derive_distribution(
    source_kind: str, evidence: Mapping[str, Any]
) -> list[float]:
    """Derive exact or empirical probabilities from machine-checkable evidence."""
    if source_kind not in {"exact_derived", "explicit_stochastic_process", "empirical_frequency"}:
        raise ContractError(f"{source_kind!r} does not have a derivable evidence vector")
    if not isinstance(evidence, Mapping):
        raise ContractError("distribution_evidence must be an object")
    expected_kind = "empirical_frequency" if source_kind == "empirical_frequency" else source_kind
    if evidence.get("kind") != expected_kind:
        raise ContractError(f"distribution_evidence.kind must be {expected_kind!r}")

    if source_kind in {"exact_derived", "explicit_stochastic_process"}:
        values = evidence.get("probabilities")
        if not isinstance(values, list) or not values:
            raise ContractError("exact distribution evidence needs a non-empty probabilities list")
        fractions = [_fraction(value, f"probabilities[{index}]") for index, value in enumerate(values)]
        if sum(fractions, Fraction(0, 1)) != 1:
            raise ContractError("exact rational probabilities must sum to one")
        return [float(value) for value in fractions]

    counts = evidence.get("counts")
    sample_size = evidence.get("sample_size")
    if not isinstance(counts, list) or not counts:
        raise ContractError("empirical evidence needs a non-empty counts list")
    if not _is_int(sample_size) or sample_size <= 0:
        raise ContractError("empirical evidence sample_size must be a positive integer")
    if any(not _is_int(count) or count < 0 for count in counts):
        raise ContractError("empirical evidence counts must be non-negative integers")
    if sum(counts) != sample_size:
        raise ContractError("empirical evidence counts must sum to sample_size")
    return [count / sample_size for count in counts]


def build_soft_row(
    row: Mapping[str, Any],
    distribution_source_kind: str,
    *,
    distribution_evidence: Mapping[str, Any],
    target_distribution: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Build a v2 soft row, deriving exact and empirical vectors from evidence."""
    if not isinstance(row, Mapping):
        raise ContractError("row must be an object")
    if distribution_source_kind not in SOFT_SOURCE_KINDS:
        raise ContractError("distribution_source_kind is unsupported")
    result = copy.deepcopy(dict(row))
    options = result.get("options")
    if not isinstance(options, list):
        raise ContractError("row.options must be a list")
    identifiers = result.get("choice_ids")
    if identifiers is None:
        identifiers = stable_choice_ids(str(result.get("id", "")), options)
    if not isinstance(identifiers, list) or len(identifiers) != len(options):
        raise ContractError("choice_ids must be parallel to options")

    evidence = copy.deepcopy(dict(distribution_evidence))
    if "choice_ids" not in evidence:
        evidence["choice_ids"] = list(identifiers)
    elif evidence["choice_ids"] != identifiers:
        raise ContractError("distribution_evidence.choice_ids must match row.choice_ids")

    if distribution_source_kind == "teacher_estimate":
        if target_distribution is None:
            raise ContractError("teacher estimates require an explicit target_distribution")
        probabilities = list(target_distribution)
    else:
        probabilities = derive_distribution(distribution_source_kind, evidence)
        if target_distribution is not None:
            if len(target_distribution) != len(probabilities) or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not math.isclose(float(value), expected, rel_tol=0.0, abs_tol=SUM_TOLERANCE)
                for value, expected in zip(target_distribution, probabilities)
            ):
                raise ContractError("target_distribution does not match its evidence")

    result.update(
        {
            "schema_version": SCHEMA_VERSION,
            "supervision_kind": "soft_distribution",
            "choice_ids": list(identifiers),
            "distribution_source_kind": distribution_source_kind,
            "target_distribution": probabilities,
            "distribution_evidence": evidence,
        }
    )
    errors = validate_v2_row(result)
    if errors:
        raise ContractError("invalid soft row: " + "; ".join(errors))
    return result


def build_hard_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Build a v2 hard-label row using answer_index as its only label encoding."""
    if not isinstance(row, Mapping):
        raise ContractError("row must be an object")
    result = copy.deepcopy(dict(row))
    if any(
        key in result
        for key in ("target_distribution", "distribution_source_kind", "distribution_evidence")
    ):
        raise ContractError("hard rows cannot contain distribution fields")
    options = result.get("options")
    if not isinstance(options, list):
        raise ContractError("row.options must be a list")
    if "choice_ids" not in result:
        result["choice_ids"] = stable_choice_ids(str(result.get("id", "")), options)
    result["schema_version"] = SCHEMA_VERSION
    result["supervision_kind"] = HARD_LABEL
    errors = validate_v2_row(result)
    if errors:
        raise ContractError("invalid hard row: " + "; ".join(errors))
    return result


def _contains_confidence_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            (isinstance(key, str) and key.casefold() in _CONFIDENCE_KEYS)
            or _contains_confidence_key(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_confidence_key(child) for child in value)
    return False


def _validate_choice_ids(row: Mapping[str, Any], options: list[Any], errors: list[str]) -> None:
    ids = _choice_ids(row)
    if ids is None:
        errors.append("choice_ids_must_be_unique_nonempty_strings")
    elif len(ids) != len(options):
        errors.append("choice_ids_length_does_not_match_options")


def _base_row_errors(row: Any, *, hard: bool) -> list[str]:
    """Apply the existing content envelope while allowing soft rows without answer_index."""
    errors = progress.validate_row(row, max_choices=MAX_CHOICES)
    if not hard:
        errors = [error for error in errors if error != "answer_index_is_not_an_integer"]
    return errors


def _answerability_errors(row: Mapping[str, Any], *, hard: bool) -> list[str]:
    """Check optional answerability and abstention metadata against the row label."""
    errors: list[str] = []
    has_status = "answerability_status" in row
    has_reason = "unanswerable_reason" in row
    status = row.get("answerability_status")
    reason = row.get("unanswerable_reason")
    identifiers = _choice_ids(row)
    abstention_id = row.get("abstention_choice_id")

    abstention_index: int | None = None
    if abstention_id is not None:
        if not _nonempty(abstention_id):
            errors.append("abstention_choice_id_must_be_a_nonempty_string")
        elif identifiers is None or abstention_id not in identifiers:
            errors.append("abstention_choice_id_must_match_exactly_one_choice_id")
        else:
            abstention_index = identifiers.index(abstention_id)

    if has_status != has_reason:
        errors.append("answerability_status_and_unanswerable_reason_must_appear_together")
        return errors
    if not has_status:
        return errors
    if status not in ANSWERABILITY_STATUSES:
        errors.append("answerability_status_is_unsupported")
        return errors
    if status == "answerable":
        if reason is not None:
            errors.append("answerable_unanswerable_reason_must_be_null")
    elif reason not in UNANSWERABLE_REASONS:
        errors.append("unanswerable_reason_is_unsupported")

    if abstention_index is None:
        errors.append("answerability_metadata_requires_abstention_choice_id")
        return errors

    if hard:
        answer_index = row.get("answer_index")
        if _is_int(answer_index):
            if status == "answerable" and answer_index == abstention_index:
                errors.append("answerable_hard_label_must_not_select_abstention")
            elif status == "unanswerable" and answer_index != abstention_index:
                errors.append("unanswerable_hard_label_must_select_abstention")
        return errors

    target = row.get("target_distribution")
    if not isinstance(target, list) or len(target) != len(identifiers or []):
        return errors
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in target
    ):
        return errors
    if status == "answerable":
        if float(target[abstention_index]) > SUM_TOLERANCE:
            errors.append("answerable_abstention_choice_must_have_zero_target_mass")
    elif any(
        not math.isclose(float(value), 1.0 if index == abstention_index else 0.0,
                         rel_tol=0.0, abs_tol=SUM_TOLERANCE)
        for index, value in enumerate(target)
    ):
        errors.append("unanswerable_target_must_select_abstention_one_hot")
    return errors


def validate_v2_row(row: Any) -> list[str]:
    """Return schema/math errors for a v2 row; this does not judge prose semantics."""
    if not isinstance(row, dict):
        return ["row_is_not_an_object"]
    if row.get("schema_version") != SCHEMA_VERSION:
        return ["schema_version_is_not_probability_v2"]

    supervision = row.get("supervision_kind")
    if supervision == HARD_LABEL:
        errors = _base_row_errors(row, hard=True)
        options = row.get("options")
        if isinstance(options, list):
            _validate_choice_ids(row, options, errors)
        if "target_distribution" in row:
            errors.append("hard_label_must_not_have_target_distribution")
        if "distribution_source_kind" in row:
            errors.append("hard_label_must_not_have_distribution_source_kind")
        if "distribution_evidence" in row:
            errors.append("hard_label_must_not_have_distribution_evidence")
        errors.extend(_answerability_errors(row, hard=True))
        return errors

    if supervision != "soft_distribution":
        return ["supervision_kind_is_unsupported"]

    errors = _base_row_errors(row, hard=False)
    if "answer_index" in row:
        errors.append("soft_distribution_must_not_have_answer_index")
    options = row.get("options")
    ids = _choice_ids(row)
    if isinstance(options, list):
        _validate_choice_ids(row, options, errors)

    source_kind = row.get("distribution_source_kind")
    if source_kind not in SOFT_SOURCE_KINDS:
        errors.append("distribution_source_kind_is_unsupported")

    probabilities = row.get("target_distribution")
    if not isinstance(probabilities, list):
        errors.append("target_distribution_is_not_a_list")
    else:
        if isinstance(options, list) and len(probabilities) != len(options):
            errors.append("target_distribution_length_does_not_match_options")
        valid_numbers = True
        for index, probability in enumerate(probabilities):
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                errors.append(f"target_distribution[{index}]_is_not_numeric")
                valid_numbers = False
            elif not math.isfinite(float(probability)):
                errors.append(f"target_distribution[{index}]_is_not_finite")
                valid_numbers = False
            elif not 0 <= probability <= 1:
                errors.append(f"target_distribution[{index}]_is_out_of_range")
                valid_numbers = False
        if valid_numbers and not math.isclose(
            sum(float(probability) for probability in probabilities),
            1.0,
            rel_tol=0.0,
            abs_tol=SUM_TOLERANCE,
        ):
            errors.append("target_distribution_does_not_sum_to_one")

    evidence = row.get("distribution_evidence")
    if not isinstance(evidence, dict):
        errors.append("distribution_evidence_is_not_an_object")
    elif _contains_confidence_key(evidence):
        errors.append("distribution_evidence_must_not_use_confidence_fields")
    else:
        favorable_counts = evidence.get("favorable_counts")
        if favorable_counts is not None:
            if not isinstance(favorable_counts, list):
                errors.append("distribution_evidence_favorable_counts_is_not_a_list")
            elif isinstance(options, list) and len(favorable_counts) != len(options):
                errors.append("distribution_evidence_favorable_counts_length_does_not_match_options")
            elif any(not _is_int(value) or value < 0 for value in favorable_counts):
                errors.append("distribution_evidence_favorable_counts_must_be_nonnegative_integers")

        expected_evidence_kind = (
            "empirical_frequency" if source_kind == "empirical_frequency" else source_kind
        )
        if source_kind in SOFT_SOURCE_KINDS and evidence.get("kind") != expected_evidence_kind:
            errors.append("distribution_evidence_kind_does_not_match_source_kind")

        if ids is not None and evidence.get("choice_ids") != ids:
            errors.append("distribution_evidence_choice_ids_do_not_match")

        if source_kind in {"exact_derived", "explicit_stochastic_process", "empirical_frequency"}:
            try:
                derived = derive_distribution(source_kind, evidence)
            except ContractError as exc:
                errors.append(f"distribution_evidence_is_invalid:{exc}")
            else:
                if isinstance(options, list) and len(derived) != len(options):
                    errors.append("distribution_evidence_vector_length_does_not_match_options")
                elif isinstance(probabilities, list) and len(probabilities) == len(derived):
                    if any(
                        not isinstance(value, (int, float))
                        or isinstance(value, bool)
                        or not math.isfinite(float(value))
                        or not math.isclose(
                            float(value), expected, rel_tol=0.0, abs_tol=SUM_TOLERANCE
                        )
                        for value, expected in zip(probabilities, derived)
                    ):
                        errors.append("target_distribution_does_not_match_evidence")
        elif source_kind == "teacher_estimate":
            for key in ("model_id", "model_version", "estimation_method"):
                if not _nonempty(evidence.get(key)):
                    errors.append(f"teacher_evidence_missing_{key}")

    if _contains_confidence_key(row):
        errors.append("soft_distribution_must_not_use_confidence_fields")
    errors.extend(_answerability_errors(row, hard=False))
    return errors


def classify_row(row: Any) -> tuple[str | None, list[str], str | None]:
    """Classify v2 rows and cheaply validated legacy answer-index rows."""
    if isinstance(row, dict) and row.get("schema_version") == SCHEMA_VERSION:
        errors = validate_v2_row(row)
        if errors:
            return None, errors, "v2"
        if row["supervision_kind"] == HARD_LABEL:
            return HARD_LABEL, [], "v2"
        return str(row["distribution_source_kind"]), [], "v2"

    if isinstance(row, dict) and "schema_version" not in row:
        if any(
            key in row
            for key in (
                "supervision_kind",
                "target_distribution",
                "distribution_source_kind",
                "distribution_evidence",
            )
        ):
            return None, ["v2_fields_require_probability_v2_schema_version"], "unknown"
        errors = progress.validate_row(row)
        if not errors:
            return HARD_LABEL, [], "legacy"
        return None, [f"legacy_{error}" for error in errors], "legacy"
    return None, ["unsupported_schema_version"], "unknown"


def verification_status(row: Any) -> dict[str, str]:
    """Describe what validation established for one accepted row."""
    source, errors, _schema = classify_row(row)
    if errors or source is None:
        raise ContractError("verification status is available only for accepted rows")
    return {
        "schema_and_label_math": "machine_checked",
        "prose_semantics": "unverified",
        "empirical_sample_to_claim_match": (
            "unverified" if source == "empirical_frequency" else "not_applicable"
        ),
        "teacher_estimate_calibration": (
            "unverified" if source == "teacher_estimate" else "not_applicable"
        ),
    }


def shuffle_choices(row: Mapping[str, Any], order: Sequence[int]) -> dict[str, Any]:
    """Return a reordered copy with every contract-defined choice vector aligned."""
    if not isinstance(row, Mapping):
        raise ContractError("row must be an object")
    result = copy.deepcopy(dict(row))
    options = result.get("options")
    if not isinstance(options, list):
        raise ContractError("row.options must be a list")
    if len(order) != len(options) or any(not _is_int(index) for index in order):
        raise ContractError("order must be a permutation of choice indexes")
    if sorted(order) != list(range(len(options))):
        raise ContractError("order must be a permutation of choice indexes")

    old_answer = result.get("answer_index")
    for key in ("options", "choice_ids", "target_distribution"):
        vector = result.get(key)
        if isinstance(vector, list):
            if len(vector) != len(options):
                raise ContractError(f"{key} must be parallel to options before shuffling")
            result[key] = [vector[index] for index in order]

    evidence = result.get("distribution_evidence")
    if isinstance(evidence, dict):
        evidence_ids = evidence.get("choice_ids")
        if isinstance(evidence_ids, list):
            if len(evidence_ids) != len(options):
                raise ContractError("distribution_evidence.choice_ids must be parallel to options")
            evidence["choice_ids"] = [evidence_ids[index] for index in order]
        for key in ("probabilities", "counts"):
            vector = evidence.get(key)
            if isinstance(vector, list):
                if len(vector) != len(options):
                    raise ContractError(f"distribution_evidence.{key} must be parallel to options")
                evidence[key] = [vector[index] for index in order]
        favorable_counts = evidence.get("favorable_counts")
        if isinstance(favorable_counts, list):
            if len(favorable_counts) != len(options):
                raise ContractError("distribution_evidence.favorable_counts must be parallel to options")
            evidence["favorable_counts"] = [favorable_counts[index] for index in order]

    if _is_int(old_answer):
        if not 0 <= old_answer < len(options):
            raise ContractError("answer_index must refer to an option before shuffling")
        result["answer_index"] = list(order).index(old_answer)
    return result


def _operation(row: Mapping[str, Any]) -> str:
    for key in ("operation_id", "task_id", "family"):
        value = row.get(key)
        if _nonempty(value):
            return value.strip()
    operation = row.get("operation")
    if isinstance(operation, dict):
        identifier = operation.get("id")
        if _nonempty(identifier):
            return identifier.strip()
    if _nonempty(operation):
        return operation.strip()
    return "not_recorded"


def _language(row: Mapping[str, Any]) -> str:
    value = row.get("language")
    if isinstance(value, dict):
        value = value.get("tag")
    if _nonempty(value):
        return value.strip()
    profile = row.get("language_profile")
    if isinstance(profile, dict) and _nonempty(profile.get("tag")):
        return profile["tag"].strip()
    return "not_recorded"


def entropy_bits(probabilities: Sequence[float]) -> float:
    return -sum(float(probability) * math.log2(float(probability)) for probability in probabilities if probability > 0)


def entropy_band(probabilities: Sequence[float]) -> str:
    entropy = entropy_bits(probabilities)
    for lower, upper, label in ENTROPY_BANDS:
        if lower <= entropy < upper or (label == "1.5-2.0" and entropy <= upper + SUM_TOLERANCE):
            return label
    return "2.0+"


def _input_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_file():
            if not (path.suffix.lower() == ".jsonl" and path.name.startswith("skipped-")):
                files.append(path)
        elif path.is_dir():
            files.extend(
                sorted(
                    file
                    for file in path.rglob("*.jsonl")
                    if file.is_file() and not file.name.startswith("skipped-")
                )
            )
        else:
            raise FileNotFoundError(path)
    return sorted(set(files))


def summarize(paths: Iterable[Path]) -> dict[str, Any]:
    """Count schema/math-accepted rows without making semantic or calibration claims."""
    files = _input_files(paths)
    by_source: Counter[str] = Counter()
    by_language: Counter[str] = Counter()
    by_operation: Counter[str] = Counter()
    supervision: Counter[str] = Counter()
    soft_entropy: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    legacy_hard = 0
    malformed = 0
    duplicate_ids = 0
    group_split_conflicts = 0
    group_split_conflict_rows = 0

    with tempfile.TemporaryDirectory(prefix="probability-v2-summary-") as temp_dir:
        connection = sqlite3.connect(Path(temp_dir) / "summary.sqlite3")
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE row_identities (
                problem_group_id TEXT NOT NULL,
                split TEXT NOT NULL
            );
            CREATE INDEX row_identities_group_idx ON row_identities(problem_group_id);
            CREATE TABLE accepted_candidates (
                row_id TEXT PRIMARY KEY,
                problem_group_id TEXT NOT NULL,
                split TEXT NOT NULL,
                source TEXT NOT NULL,
                language TEXT NOT NULL,
                operation TEXT NOT NULL,
                entropy_band TEXT,
                schema_kind TEXT NOT NULL,
                pilot_status TEXT
            );
            CREATE TABLE conflicting_groups (problem_group_id TEXT PRIMARY KEY);
            """
        )

        for path in files:
            with path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        malformed += 1
                        rejected["invalid_json"] += 1
                        continue

                    if isinstance(row, dict):
                        group_id = row.get("problem_group_id")
                        split = row.get("split")
                        if _nonempty(group_id) and _nonempty(split):
                            connection.execute(
                                "INSERT INTO row_identities(problem_group_id, split) VALUES(?, ?)",
                                (group_id.strip(), split.strip()),
                            )

                    source, errors, schema = classify_row(row)
                    if errors or source is None:
                        rejected[errors[0] if errors else "invalid_row"] += 1
                        continue

                    identifier = row["id"].strip()
                    group_id = row["problem_group_id"].strip()
                    split = row["split"].strip()
                    entropy = (
                        entropy_band(row["target_distribution"])
                        if source != HARD_LABEL
                        else None
                    )
                    inserted = connection.execute(
                        """INSERT OR IGNORE INTO accepted_candidates(
                            row_id, problem_group_id, split, source, language,
                            operation, entropy_band, schema_kind, pilot_status
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            identifier,
                            group_id,
                            split,
                            source,
                            _language(row),
                            _operation(row),
                            entropy,
                            schema or "unknown",
                            row.get("pilot_status") if isinstance(row.get("pilot_status"), str) else None,
                        ),
                    ).rowcount
                    if not inserted:
                        duplicate_ids += 1
                        rejected["duplicate_row_id"] += 1

        connection.execute(
            """INSERT INTO conflicting_groups(problem_group_id)
               SELECT problem_group_id FROM row_identities
               GROUP BY problem_group_id HAVING COUNT(DISTINCT split) > 1"""
        )
        group_split_conflicts = connection.execute(
            "SELECT COUNT(*) FROM conflicting_groups"
        ).fetchone()[0]
        group_split_conflict_rows = connection.execute(
            """SELECT COUNT(*) FROM row_identities AS identities
               JOIN conflicting_groups AS groups USING(problem_group_id)"""
        ).fetchone()[0]
        excluded_group_rows = connection.execute(
            """SELECT COUNT(*) FROM accepted_candidates AS candidates
               JOIN conflicting_groups AS groups USING(problem_group_id)"""
        ).fetchone()[0]
        if excluded_group_rows:
            rejected["problem_group_spans_splits"] += excluded_group_rows

        accepted_rows = connection.execute(
            """SELECT source, language, operation, entropy_band, schema_kind, pilot_status
               FROM accepted_candidates AS candidates
               WHERE NOT EXISTS (
                   SELECT 1 FROM conflicting_groups AS groups
                   WHERE groups.problem_group_id = candidates.problem_group_id
               )"""
        )
        accepted = 0
        production_eligible = 0
        excluded_pilot = 0
        production_by_source: Counter[str] = Counter()
        for source, language, operation, entropy, schema, pilot_status in accepted_rows:
            accepted += 1
            by_source[source] += 1
            by_language[language] += 1
            by_operation[operation] += 1
            if source == HARD_LABEL:
                supervision[HARD_LABEL] += 1
                legacy_hard += int(schema == "legacy")
            else:
                supervision["soft_distribution"] += 1
                soft_entropy[entropy] += 1
            if pilot_status == EXCLUDED_PILOT_STATUS:
                excluded_pilot += 1
            else:
                production_eligible += 1
                production_by_source[source] += 1
        connection.close()

    return {
        "schema_version": SCHEMA_VERSION,
        "files_scanned": [path.as_posix() for path in files],
        "accepted_rows": accepted,
        "production_eligible_rows": production_eligible,
        "production_eligible_rows_by_source_kind": dict(sorted(production_by_source.items())),
        "excluded_pilot_rows": excluded_pilot,
        "accepted_rows_by_source_kind": dict(sorted(by_source.items())),
        "accepted_rows_by_language": dict(sorted(by_language.items())),
        "accepted_rows_by_operation": dict(sorted(by_operation.items())),
        "soft_rows_by_entropy_bits": dict(sorted(soft_entropy.items())),
        "entropy_scope": "soft distributions only; hard labels have no target_distribution",
        "supervision_totals": {
            "soft_distribution": supervision["soft_distribution"],
            "hard_label": supervision[HARD_LABEL],
        },
        "legacy_hard_label_rows": legacy_hard,
        "duplicate_row_ids_rejected": duplicate_ids,
        "problem_group_split_conflicts": group_split_conflicts,
        "rows_in_conflicting_problem_groups": group_split_conflict_rows,
        "rejected_rows": sum(rejected.values()),
        "malformed_json_rows": malformed,
        "rejection_reasons": dict(sorted(rejected.items())),
        "disclaimer": "Schema and math checks do not verify prose semantics or probability calibration.",
        "verification_scope": {
            "schema_and_distribution_math": "machine_checked for accepted rows",
            "prose_semantics": "unverified",
            "empirical_sample_to_claim_match": "unverified",
            "teacher_estimate_calibration": "unverified",
        },
    }


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    summary_parser = subparsers.add_parser(
        "summary", help="summarize accepted v2 and legacy hard-label JSONL rows"
    )
    summary_parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    if args.command == "summary":
        print(json.dumps(summarize(args.paths), ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
