"""Append authored drafts and reconcile exact job/cohort IDs without corpus scans."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dataset_factory import probability_v2 as v2, progress


ROOT = Path(__file__).resolve().parents[1]
MAX_DRAFTS = 8
MULTIPLE_SUPPORT_REQUIREMENT = "multiple_substantive_support-v1"
CONTENT_FIELDS = frozenset({"context", "question", "options", "explanation"})
DRAFT_FIELDS = CONTENT_FIELDS | {
    "assignment_id", "abstention_index", "abstention_text", "answer_index",
    "distribution_evidence", "target_distribution",
}
PLAN_FIELDS = (
    "id", "problem_group_id", "split", "language", "language_profile",
    "domain_id", "domain", "operation_id", "operation", "question_form",
    "input_structure", "register", "difficulty_band", "context_length_band",
    "choice_count", "answerability_status", "unanswerable_reason", "_generator", "_planner",
)


class WriterError(ValueError):
    """A content-free failure code suitable for author and coordinator logs."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


def _path(value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WriterError("invalid_job_path")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def _jsonl(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return []
        raise WriterError("missing_input") from None
    return _parse_jsonl(data)


def _parse_jsonl(data: bytes) -> list[dict[str, Any]]:
    if data and not data.endswith(b"\n"):
        raise WriterError("incomplete_jsonl_boundary")
    try:
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError):
        raise WriterError("invalid_jsonl") from None
    if any(not isinstance(row, dict) for row in rows):
        raise WriterError("jsonl_row_not_object")
    return rows


def _load_job(job_path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise WriterError("invalid_job_json") from None
    if not isinstance(job, dict):
        raise WriterError("invalid_job_json")
    plan_path, output, skip = (_path(job.get(key)) for key in ("plan", "output", "skip_file"))
    if len({plan_path, output, skip, job_path.resolve()}) != 4:
        raise WriterError("job_paths_overlap")
    if "plan_sha256" in job and hashlib.sha256(plan_path.read_bytes()).hexdigest() != job["plan_sha256"]:
        raise WriterError("plan_checksum_mismatch")
    plans: dict[str, dict[str, Any]] = {}
    for row in _jsonl(plan_path):
        identifier = row.get("id")
        if not progress.nonempty_string(identifier) or identifier in plans:
            raise WriterError("invalid_plan_identity")
        if (
            row.get("schema_version") != "probability-v2-assignment-v1"
            or row.get("language") != job.get("language")
            or row.get("distribution_source_kind") != job.get("source_kind")
            or _path(row.get("output")) != output
            or _path(row.get("skip_file")) != skip
        ):
            raise WriterError("job_plan_mismatch")
        plans[identifier] = row
    allowed = job.get("remaining_ids")
    if (
        not isinstance(allowed, list)
        or any(not isinstance(identifier, str) or identifier not in plans for identifier in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        raise WriterError("invalid_job_allowlist")
    job = dict(job, plan=plan_path, output=output, skip_file=skip, remaining_ids=set(allowed))
    return job, plans


def _build_row(plan: dict[str, Any], draft: dict[str, Any]) -> dict[str, Any]:
    if set(draft) - DRAFT_FIELDS or not CONTENT_FIELDS <= draft.keys():
        raise WriterError("invalid_draft_fields")
    options = draft["options"]
    if not isinstance(options, list) or len(options) != plan.get("choice_count"):
        raise WriterError("choice_count_mismatch")
    abstention = draft.get("abstention_index")
    text = draft.get("abstention_text")
    if (
        not isinstance(abstention, int) or isinstance(abstention, bool)
        or not 0 <= abstention < len(options)
        or not progress.nonempty_string(text) or options[abstention] != text
    ):
        raise WriterError("invalid_abstention_choice")
    if plan.get("answerability_status") not in v2.ANSWERABILITY_STATUSES:
        raise WriterError("invalid_plan_answerability")
    planner_metadata = plan.get("_planner")
    requirement = planner_metadata.get("target_shape_requirement") if isinstance(planner_metadata, dict) else None
    if requirement is not None:
        if requirement != MULTIPLE_SUPPORT_REQUIREMENT:
            raise WriterError("unsupported_target_shape_requirement")
        form = plan.get("question_form")
        if (plan["answerability_status"] != "answerable" or plan["distribution_source_kind"] == v2.HARD_LABEL
            or len(options) < 3 or not isinstance(form, dict) or form.get("question_kind") != "outcome_distribution"):
            raise WriterError("incompatible_target_shape_requirement")
    row = {key: copy.deepcopy(plan[key]) for key in PLAN_FIELDS if key in plan}
    row.update({key: copy.deepcopy(draft[key]) for key in CONTENT_FIELDS})
    row["topic"] = plan.get("domain_id")
    difficulty = plan.get("difficulty_band")
    row["difficulty"] = difficulty.get("id") if isinstance(difficulty, dict) else difficulty
    row["provenance"] = progress.PROVENANCE
    row["semantic_review_status"] = "author_self_review_only"
    row["language_review_status"] = "author_self_review_only"
    row["_assignment_metadata"] = copy.deepcopy(plan)
    try:
        row["choice_ids"] = v2.stable_choice_ids(plan["id"], options)
        row["abstention_choice_id"] = row["choice_ids"][abstention]
        source = plan["distribution_source_kind"]
        if source == v2.HARD_LABEL:
            if "answer_index" not in draft or {"distribution_evidence", "target_distribution"} & draft.keys():
                raise WriterError("invalid_hard_draft_fields")
            row["answer_index"] = draft["answer_index"]
            row = v2.build_hard_row(row)
        else:
            evidence = draft.get("distribution_evidence")
            if "answer_index" in draft or not isinstance(evidence, dict):
                raise WriterError("invalid_soft_draft_fields")
            row = v2.build_soft_row(
                row, source, distribution_evidence=evidence,
                target_distribution=draft.get("target_distribution"),
            )
    except v2.ContractError:
        raise WriterError("invalid_v2_row") from None
    if v2.validate_v2_row(row):
        raise WriterError("invalid_v2_row")
    if requirement is not None:
        evidence = row["distribution_evidence"]
        assumptions = evidence.get("mechanism_assumptions")
        if (evidence.get("question_kind") != "outcome_distribution"
            or not isinstance(assumptions, list) or not assumptions
            or any(not progress.nonempty_string(item) for item in assumptions)
            or not progress.nonempty_string(evidence.get("outcome_partition"))):
            raise WriterError("missing_distribution_task_evidence")
        target = row["target_distribution"]
        if target[abstention] != 0:
            raise WriterError("required_abstention_mass_not_zero")
        if sum(value > 0 for index, value in enumerate(target) if index != abstention) < 2:
            raise WriterError("required_multiple_support_missing")
    return row


def _ledger(path: Path, plans: dict[str, dict[str, Any]], *, skip: bool = False) -> dict[str, dict[str, Any]]:
    return _ledger_rows(_jsonl(path, missing_ok=True), plans, skip=skip)


def _ledger_rows(source_rows: list[dict[str, Any]], plans: dict[str, dict[str, Any]], *, skip: bool = False) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        if skip:
            identities = [row[key] for key in ("id", "assignment_id", "plan_id") if key in row]
            if any(not isinstance(value, str) for value in identities) or len(set(identities)) != 1:
                raise WriterError("invalid_skip_identity")
            identifier = identities[0]
        else:
            identifier = row.get("id")
        if not isinstance(identifier, str) or identifier not in plans:
            raise WriterError("ledger_unknown_assignment")
        if identifier in rows:
            raise WriterError("duplicate_ledger_identity")
        rows[identifier] = row
    return rows


def _durable_append(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        payloads = [
            (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
            for row in rows
        ]
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            for index, payload in enumerate(payloads):
                remaining = memoryview(payload)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written == 0:
                        raise WriterError("append_failed")
                    remaining = remaining[written:]
                os.fsync(descriptor)
                if index == 0:
                    _sync_directory(path.parent)
        finally:
            os.close(descriptor)
    else:
        # A retry also completes durability after an earlier fsync failure.
        with path.open("rb") as stream:
            os.fsync(stream.fileno())
        _sync_directory(path.parent)


def _sync_directory(path: Path) -> None:
    directory = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _write(job_path: Path, drafts_path: Path, *, skip: bool) -> dict[str, int]:
    job, plans = _load_job(job_path)
    if drafts_path.resolve() in {job["output"], job["skip_file"], job_path.resolve(), job["plan"]}:
        raise WriterError("draft_path_overlaps_ledger")
    drafts = _jsonl(drafts_path)
    if not 1 <= len(drafts) <= MAX_DRAFTS:
        raise WriterError("draft_batch_must_have_1_to_8_rows")
    staged: dict[str, dict[str, Any]] = {}
    for draft in drafts:
        identifier = draft.get("assignment_id")
        if not isinstance(identifier, str) or identifier not in job["remaining_ids"]:
            raise WriterError("assignment_not_allowed")
        if identifier in staged:
            raise WriterError("duplicate_draft_identity")
        if skip:
            if set(draft) != {"assignment_id", "reason"} or not progress.nonempty_string(draft.get("reason")):
                raise WriterError("invalid_skip_draft")
            staged[identifier] = {
                "id": identifier, "assignment_id": identifier, "status": "skipped",
                "reason": draft["reason"], "language": plans[identifier]["language"],
                "source_kind": plans[identifier]["distribution_source_kind"],
            }
        else:
            staged[identifier] = _build_row(plans[identifier], draft)

    output, skip_path = job["output"], job["skip_file"]
    output.parent.mkdir(parents=True, exist_ok=True)
    skip_path.parent.mkdir(parents=True, exist_ok=True)
    with output.with_name(f".{output.name}.authoring.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WriterError("output_locked") from None
        written, skipped = _ledger(output, plans), _ledger(skip_path, plans, skip=True)
        if written.keys() & skipped.keys():
            raise WriterError("source_skip_ledger_conflict")
        own, other = (skipped, written) if skip else (written, skipped)
        new: list[dict[str, Any]] = []
        for identifier, row in staged.items():
            if identifier in other:
                raise WriterError("assignment_already_skipped" if not skip else "assignment_already_written")
            if identifier in own:
                existing = own[identifier]
                if skip:
                    matches = existing.get("reason") == row["reason"]
                else:
                    matches = existing == row
                if not matches:
                    raise WriterError("retry_content_conflict")
            else:
                if plans[identifier].get("status") not in {"assigned", "in_flight", "pending"}:
                    raise WriterError("assignment_inactive")
                new.append(row)
        _durable_append(skip_path if skip else output, new)
    result = {"appended": len(new), "already_present": len(staged) - len(new)}
    if not skip and any("target_shape_selection" in row.get("_planner", {}) for row in staged.values()):
        soft = [row for row in new if row.get("distribution_source_kind") in v2.SOFT_SOURCE_KINDS]
        answerable = [row for row in soft if row["answerability_status"] == "answerable"]
        result.update(
            soft_appended=len(soft), answerable_soft_appended=len(answerable),
            multiple_substantive_support_appended=sum(
                sum(value > 0 for identifier, value in zip(row["choice_ids"], row["target_distribution"])
                    if identifier != row["abstention_choice_id"]) >= 2 for row in answerable
            ),
            required_multiple_support_appended=sum(
                row["_planner"].get("target_shape_requirement") == MULTIPLE_SUPPORT_REQUIREMENT for row in new
            ),
        )
    return result


def append_drafts(job_path: Path, drafts_path: Path) -> dict[str, int]:
    """Build and validate the whole draft batch before appending any source row."""
    return _run(job_path, drafts_path, skip=False)


def skip_assignments(job_path: Path, drafts_path: Path) -> dict[str, int]:
    """Append explicit skip records; never replace source rows or existing skips."""
    return _run(job_path, drafts_path, skip=True)


def _run(job_path: Path, drafts_path: Path, *, skip: bool) -> dict[str, int]:
    try:
        return _write(job_path, drafts_path, skip=skip)
    except WriterError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise WriterError("input_or_append_failure") from None


def _file_stamp(path: Path) -> tuple[int, ...] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _status_bytes(path: Path) -> tuple[bytes, str]:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return b"", "missing"
    state = "empty" if not data else "complete" if data.endswith(b"\n") else "incomplete_boundary"
    return data, state


def _unavailable_status(job: dict[str, Any], state: str, output_state: str, skip_state: str) -> dict[str, Any]:
    return {
        "state": state, "complete": False, "snapshot_available": False,
        "allowlisted": len(job["remaining_ids"]),
        "written": None, "skipped": None, "unaccounted": None, "next_ids": [],
        "output_state": output_state, "skip_state": skip_state,
    }


def _job_snapshot(job: dict[str, Any], plans: dict[str, dict[str, Any]], next_count: int) -> dict[str, Any]:
    output, skip = job["output"], job["skip_file"]
    before = (_file_stamp(output), _file_stamp(skip))
    output_bytes, output_state = _status_bytes(output)
    skip_bytes, skip_state = _status_bytes(skip)
    if before != (_file_stamp(output), _file_stamp(skip)):
        return _unavailable_status(job, "in_flight", "in_flight", "in_flight")
    if "incomplete_boundary" in (output_state, skip_state):
        return _unavailable_status(job, "incomplete_boundary", output_state, skip_state)
    written = _ledger_rows(_parse_jsonl(output_bytes), plans)
    skipped = _ledger_rows(_parse_jsonl(skip_bytes), plans, skip=True)
    if written.keys() & skipped.keys():
        raise WriterError("source_skip_ledger_conflict")
    allowed = job["remaining_ids"]
    missing = [identifier for identifier in plans if identifier in allowed and identifier not in written and identifier not in skipped]
    return {
        "state": "unfinished" if missing else "complete", "complete": not missing,
        "snapshot_available": True, "allowlisted": len(allowed),
        "written": len(allowed & written.keys()), "skipped": len(allowed & skipped.keys()),
        "unaccounted": len(missing), "next_ids": missing[:next_count],
        "output_state": output_state, "skip_state": skip_state,
    }


def _job_status(job: dict[str, Any], plans: dict[str, dict[str, Any]], next_count: int) -> dict[str, Any]:
    if isinstance(next_count, bool) or not isinstance(next_count, int) or not 1 <= next_count <= MAX_DRAFTS:
        raise WriterError("next_count_must_be_1_to_8")
    output = job["output"]
    lock_path = output.with_name(f".{output.name}.authoring.lock")
    try:
        lock = lock_path.open("rb")
    except FileNotFoundError:
        result = _job_snapshot(job, plans, next_count)
        if lock_path.exists():
            return _unavailable_status(job, "in_flight", "in_flight", "in_flight")
        return result
    with lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return _unavailable_status(job, "in_flight", "in_flight", "in_flight")
        return _job_snapshot(job, plans, next_count)


def job_status(job_path: Path, *, next_count: int = MAX_DRAFTS) -> dict[str, Any]:
    """Read only this job's plan and ledgers; never infer completion from runtime."""
    try:
        job, plans = _load_job(job_path)
        return _job_status(job, plans, next_count)
    except WriterError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise WriterError("status_input_failure") from None


def cohort_status(manifest_path: Path, *, next_count: int = MAX_DRAFTS) -> dict[str, Any]:
    """Count each canonical launched job once; unfinished jobs retain their slot."""
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or set(manifest) - {"launched", "queued", "lane_limit"}:
            raise WriterError("invalid_cohort_manifest")
        limit = manifest.get("lane_limit", 16)
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 16:
            raise WriterError("lane_limit_must_be_1_to_16")
        groups: dict[str, list[Path]] = {}
        for group in ("launched", "queued"):
            values = manifest.get(group)
            if not isinstance(values, list):
                raise WriterError("invalid_cohort_manifest")
            groups[group] = list(dict.fromkeys(_path(value) for value in values))
        if set(groups["launched"]) & set(groups["queued"]):
            raise WriterError("job_both_launched_and_queued")
        owned_paths: set[Path] = set()
        owned_ids: set[str] = set()
        jobs: list[dict[str, Any]] = []
        for group, paths in groups.items():
            for path in paths:
                job, plans = _load_job(path)
                paths_owned = {job["output"], job["skip_file"]}
                if paths_owned & owned_paths or plans.keys() & owned_ids:
                    raise WriterError("cohort_ownership_collision")
                owned_paths.update(paths_owned)
                owned_ids.update(plans)
                jobs.append({"job": path.as_posix(), "group": group, **_job_status(job, plans, next_count)})
        unfinished = sum(row["group"] == "launched" and not row["complete"] for row in jobs)
        return {
            "lane_limit": limit, "launched_jobs": len(groups["launched"]),
            "queued_jobs": len(groups["queued"]), "unfinished_launched_jobs": unfinished,
            "completed_launched_jobs": len(groups["launched"]) - unfinished,
            "available_slots": max(0, limit - unfinished), "jobs": jobs,
        }
    except WriterError:
        raise
    except (OSError, ValueError, TypeError, KeyError):
        raise WriterError("cohort_input_failure") from None


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("append", "skip", "status", "cohort"))
    parser.add_argument("--job", type=Path)
    parser.add_argument("--drafts", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--next-count", type=int, default=MAX_DRAFTS)
    args = parser.parse_args(argv)
    if args.action == "cohort" and args.manifest is None:
        parser.error("--manifest is required for cohort")
    if args.action != "cohort" and args.job is None:
        parser.error("--job is required")
    if args.action in {"append", "skip"} and args.drafts is None:
        parser.error("--drafts is required for append/skip")
    try:
        if args.action == "status":
            result = job_status(args.job, next_count=args.next_count)
        elif args.action == "cohort":
            result = cohort_status(args.manifest, next_count=args.next_count)
        else:
            function = append_drafts if args.action == "append" else skip_assignments
            result = function(args.job, args.drafts)
    except WriterError as exc:
        print(json.dumps({"error_code": exc.code}))
        return 2
    except (OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"error_code": "input_or_append_failure"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
