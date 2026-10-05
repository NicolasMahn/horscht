"""Verify centrally approved withdrawals of exact, uncollected author records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_VERSION = "author-issue-corrections-v1"


class CorrectionError(ValueError):
    """Content-free verification failure; no correction should be applied."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _path(value: Any, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise CorrectionError("invalid_correction_path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def _sha(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise CorrectionError("invalid_sha256")
    return value


def _ids(values: Any) -> frozenset[str]:
    if (
        not isinstance(values, list) or not values
        or any(not isinstance(value, str) or not value or value != value.strip() for value in values)
        or len(set(values)) != len(values)
    ):
        raise CorrectionError("invalid_exact_affected_ids")
    return frozenset(values)


def record_scope(record: Mapping[str, Any], allowed_job_ids: Iterable[str]) -> frozenset[str]:
    """Normalize explicit scope only; callers must retain flags on any failure.

    Ten-digit author aliases resolve only within this job. Multiple explicit
    fields must agree, so an inconsistent summary cannot silently widen or
    narrow an exclusion. Boolean review flags carry no scope themselves.
    """
    if not isinstance(record, Mapping):
        raise CorrectionError("invalid_scope_record")
    allowed = _ids(list(allowed_job_ids))
    scopes = []
    for field in ("scope_ids", "ids", "question_ids", "id", "assignment_id", "question_id",
                  "exclude_pending_review", "review_recommended"):
        if field not in record:
            continue
        value = record[field]
        if field in ("exclude_pending_review", "review_recommended") and isinstance(value, bool):
            continue
        values = [value] if field in ("id", "assignment_id", "question_id") else value
        identifiers = _ids(values)
        normalized = []
        for identifier in identifiers:
            if identifier in allowed:
                normalized.append(identifier)
                continue
            if not re.fullmatch(r"[0-9]{10}", identifier):
                raise CorrectionError("scope_id_outside_job")
            matches = [candidate for candidate in allowed if candidate.endswith("-" + identifier)]
            if len(matches) != 1:
                raise CorrectionError("scope_alias_ambiguous" if matches else "scope_id_outside_job")
            normalized.append(matches[0])
        if len(set(normalized)) != len(normalized):
            raise CorrectionError("duplicate_normalized_scope_id")
        scopes.append(frozenset(normalized))
    if not scopes:
        raise CorrectionError("record_scope_missing")
    if any(scope != scopes[0] for scope in scopes[1:]):
        raise CorrectionError("conflicting_record_scopes")
    return scopes[0]


def _object(data: bytes) -> dict[str, Any]:
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError):
        raise CorrectionError("invalid_correction_json") from None
    if not isinstance(value, dict):
        raise CorrectionError("correction_json_not_object")
    return value


@dataclass(frozen=True)
class RecordKey:
    path: Path
    line: int
    sha256: str


@dataclass(frozen=True)
class Withdrawal:
    record: RecordKey
    affected_ids: frozenset[str]
    receipt: RecordKey


@dataclass(frozen=True)
class VerifiedWithdrawals:
    withdrawals: tuple[Withdrawal, ...]

    def should_withdraw(
        self, path: Path, line: int, raw_line: bytes, affected_ids: Iterable[str],
    ) -> bool:
        """Filter this exact record before unioning flags; never subtract an ID."""
        key = RecordKey(path.resolve(), line, hashlib.sha256(raw_line).hexdigest())
        identifiers = frozenset(affected_ids)
        return any(item.record == key and item.affected_ids == identifiers for item in self.withdrawals)

    def counts(self) -> dict[str, int]:
        return {
            "verified_withdrawals": len(self.withdrawals),
            "unique_affected_ids": len({identifier for item in self.withdrawals for identifier in item.affected_ids}),
            "author_ledgers": len({item.record.path for item in self.withdrawals}),
            "clarification_receipts": len({item.receipt for item in self.withdrawals}),
        }


def _binding(value: Mapping[str, Any], root: Path) -> tuple[RecordKey, frozenset[str]]:
    line = value.get("line")
    if isinstance(line, bool) or not isinstance(line, int) or line < 1:
        raise CorrectionError("invalid_record_line")
    key = RecordKey(_path(value.get("ledger_path"), root), line, _sha(value.get("record_sha256")))
    return key, _ids(value.get("question_ids"))


def _record_line(key: RecordKey) -> bytes:
    with key.path.open("rb") as stream:
        for number, raw_line in enumerate(stream, 1):
            if number == key.line:
                if not raw_line.endswith(b"\n"):
                    raise CorrectionError("incomplete_target_record")
                if hashlib.sha256(raw_line).hexdigest() != key.sha256:
                    raise CorrectionError("target_record_hash_mismatch")
                return raw_line
    raise CorrectionError("target_record_line_missing")


def load_withdrawals(
    manifest_path: Path,
    author_ledger_job_ids: Mapping[Path, Iterable[str]],
    *,
    already_collected: Iterable[RecordKey] = (),
    root: Path = ROOT,
) -> VerifiedWithdrawals:
    """Validate all bindings before returning any pre-collection withdrawals.

    The caller supplies the author-ledger allowlist and known collected record
    keys. Existing known-issue exclusions are retained independently; this API
    does not authorize reversing a collected issue or verify prose semantics.
    """
    try:
        manifest = _object(manifest_path.read_bytes())
        if manifest.get("schema_version") != MANIFEST_VERSION or not isinstance(manifest.get("withdrawals"), list):
            raise CorrectionError("invalid_correction_manifest")
        allowed: dict[Path, frozenset[str]] = {}
        for path, identifiers in author_ledger_job_ids.items():
            resolved = _path(str(path), root)
            scope = _ids(list(identifiers))
            if resolved in allowed and allowed[resolved] != scope:
                raise CorrectionError("conflicting_ledger_job_allowlists")
            allowed[resolved] = scope
        collected = {(_path(str(key.path), root), key.line) for key in already_collected}
        verified: dict[RecordKey, Withdrawal] = {}
        for entry in manifest["withdrawals"]:
            if not isinstance(entry, dict) or entry.get("phase") != "before_collection":
                raise CorrectionError("only_precollection_withdrawals_supported")
            key, affected = _binding(entry, root)
            if key.path not in allowed:
                raise CorrectionError("target_not_in_author_ledger_allowlist")
            if (key.path, key.line) in collected:
                raise CorrectionError("collected_issue_requires_separate_reversal")
            record = _object(_record_line(key))
            if record_scope(record, allowed[key.path]) != affected:
                raise CorrectionError("target_affected_ids_mismatch")
            exclusion = record.get("exclude_pending_review")
            if not (exclusion is True or isinstance(exclusion, list) and exclusion
                    or record.get("training_status") == "exclude_pending_review"):
                raise CorrectionError("target_is_not_author_exclusion")
            receipt_binding = entry.get("clarification_receipt")
            if not isinstance(receipt_binding, dict):
                raise CorrectionError("invalid_clarification_receipt_binding")
            receipt_line = receipt_binding.get("line")
            if isinstance(receipt_line, bool) or not isinstance(receipt_line, int) or receipt_line < 1:
                raise CorrectionError("invalid_record_line")
            receipt = RecordKey(_path(receipt_binding.get("path"), root), receipt_line,
                                _sha(receipt_binding.get("sha256")))
            if (receipt.path, receipt.line) == (key.path, key.line) or receipt.path == manifest_path.resolve():
                raise CorrectionError("receipt_overlaps_target")
            _object(_record_line(receipt))
            if (
                entry.get("decision") != "withdraw_exact_record"
                or entry.get("replacement_training_status") != "review_recommended"
                or entry.get("approved_by") != "central_coordinator"
                or entry.get("author_clarification_received") is not True
                or entry.get("concrete_defect_confirmed") is not False
                or any(not isinstance(entry.get(field), str) or not entry[field].strip()
                       for field in ("author_session_id", "clarification_id"))
            ):
                raise CorrectionError("invalid_clarification_receipt_binding")
            withdrawal = Withdrawal(key, affected, receipt)
            if key in verified and verified[key] != withdrawal:
                raise CorrectionError("conflicting_duplicate_withdrawal")
            verified[key] = withdrawal
        return VerifiedWithdrawals(tuple(verified.values()))
    except CorrectionError:
        raise
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        raise CorrectionError("correction_input_failure") from None


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--job-allowlists", type=Path, required=True,
                        help="JSON object mapping explicit author-ledger paths to canonical job ID lists")
    args = parser.parse_args(argv)
    try:
        allowlists = _object(args.job_allowlists.read_bytes())
        result = load_withdrawals(args.manifest, allowlists).counts()
    except CorrectionError as exc:
        print(json.dumps({"error_code": exc.code}))
        return 2
    except OSError:
        print(json.dumps({"error_code": "correction_input_failure"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
