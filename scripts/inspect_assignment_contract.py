"""Inspect explicit local assignment metadata without emitting authored content."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/overnight-2026-09-30"
sys.path.insert(0, str(ROOT))

from dataset_factory import authoring_writer as writer


class InspectionError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise InspectionError(code)


def local_path(value):
    require(isinstance(value, (str, Path)) and bool(str(value).strip()), "invalid_local_path")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def inspect_contract(job_path, ids=None):
    path = local_path(job_path)
    require(path.parent == RUN / "jobs" and path.suffix == ".json", "job_path_outside_run")
    job_bytes = path.read_bytes()
    job = json.loads(job_bytes)
    require(isinstance(job, dict), "invalid_job_json")
    dataset = ROOT / "data/generated_train/probability-v2"
    for field, directory in (("plan", "assignments"), ("output", "questions"), ("skip_file", "questions")):
        target = local_path(job.get(field))
        require(target.is_relative_to(dataset / directory) and target.suffix == ".jsonl",
                "job_artifact_outside_training_scope")
    plan_path = local_path(job["plan"])
    # Checking a pin before and after helper validation also closes its separate
    # hash/read calls against a concurrent plan rewrite.
    pin = job.get("plan_sha256")
    require(isinstance(pin, str) and len(pin) == 64 and all(char in "0123456789abcdef" for char in pin),
            "plan_pin_required")
    require(hashlib.sha256(plan_path.read_bytes()).hexdigest() == pin, "plan_checksum_mismatch")
    loaded_job, plans = writer._load_job(path)
    if ids is None:
        status = writer._job_status(loaded_job, plans, writer.MAX_DRAFTS)
        require(status["snapshot_available"], "fresh_status_unavailable")
        ids = status["next_ids"]
    else:
        require(isinstance(ids, list) and 1 <= len(ids) <= writer.MAX_DRAFTS
                and all(isinstance(identifier, str) for identifier in ids)
                and len(set(ids)) == len(ids), "explicit_ids_must_be_unique_1_to_8")
        require(all(identifier in loaded_job["remaining_ids"] for identifier in ids), "id_outside_job_allowlist")
    assignments = []
    for identifier in ids:
        plan = plans[identifier]
        planner = plan.get("_planner", {})
        require(isinstance(planner, dict), "invalid_planner_metadata")
        selection = planner.get("target_shape_selection", {})
        require(isinstance(selection, dict), "invalid_selection_metadata")
        selection_status = selection.get("status")
        require(selection_status is None or isinstance(selection_status, str), "invalid_selection_status")
        count = plan.get("choice_count")
        require(isinstance(count, int) and not isinstance(count, bool) and 2 <= count <= 265,
                "invalid_choice_count")
        require(plan.get("answerability_status") in ("answerable", "unanswerable"), "invalid_answerability")
        assignments.append({
            "id": identifier,
            "answerability_status": plan["answerability_status"],
            "distribution_source_kind": plan["distribution_source_kind"],
            "choice_count": count,
            "required_multiple_support": planner.get("target_shape_requirement") == "multiple_substantive_support-v1",
            "selection_status": selection_status,
        })
    require(path.read_bytes() == job_bytes and hashlib.sha256(plan_path.read_bytes()).hexdigest() == pin,
            "metadata_changed_during_inspection")
    return {"assignments": assignments}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", default=None)
    args = parser.parse_args(argv)
    try:
        result = inspect_contract(args.job, args.ids)
    except (InspectionError, writer.WriterError) as exc:
        print(json.dumps({"error_code": str(exc)}))
        return 2
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        print(json.dumps({"error_code": "assignment_contract_inspection_failed"}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
