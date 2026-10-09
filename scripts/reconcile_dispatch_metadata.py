"""Reconcile explicit dispatcher metadata; coordinator invokes --apply to commit.

Launch reservations and spawn receipts never establish live running turns.
No plans, question outputs, skip ledgers, or evaluation artifacts are opened.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone


OWNERS = ("resume_dispatch_b", "resume_dispatch_c", "new_dispatch_d", "new_dispatch_e",
          "batch2_dispatch_f", "batch2_dispatch_g")
WRITER_SCHEMAS = {
    "resume_dispatch_b": ("writers", "job_path", "spawn_utc"),
    "new_dispatch_d": ("records", "job", "spawned_at_utc"),
    "batch2_dispatch_f": ("jobs", "job_path", "spawn_utc"),
    "batch2_dispatch_g": ("mappings", "job", "actual_spawn_utc"),
}
ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/overnight-2026-09-30"


class MetadataError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise MetadataError(code)


def path_at(value, root):
    require(isinstance(value, (str, Path)) and bool(str(value).strip()), "invalid_metadata_path")
    path = Path(value)
    return (path if path.is_absolute() else root / path).resolve()


def timestamp(value):
    require(isinstance(value, str), "invalid_spawn_timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise MetadataError("invalid_spawn_timestamp") from None
    require(parsed.utcoffset() is not None and parsed.utcoffset().total_seconds() == 0, "spawn_timestamp_not_utc")
    return value


def merge_receipt(existing, update):
    require(isinstance(existing, (dict, str)), "invalid_actual_dispatch_record")
    merged = copy.deepcopy(existing) if isinstance(existing, dict) else {}
    if isinstance(existing, str):
        if existing != update["canonical"]:
            merged["previous_registry_metadata"] = existing
    elif any(field in existing and existing[field] != value for field, value in update.items()):
        merged.setdefault("previous_dispatch_records", []).append(
            {field: existing[field] for field in update if field in existing}
        )
    return {**merged, **update}


class Inputs:
    def __init__(self):
        self.hashes = {}

    def read(self, path):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        require(path not in self.hashes or self.hashes[path] == digest, "metadata_changed_during_read")
        self.hashes[path] = digest
        value = json.loads(data)
        require(isinstance(value, dict), "metadata_object_required")
        return value

    def verify(self):
        for path, expected in self.hashes.items():
            require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, "metadata_changed_before_commit")


def reconcile(run=RUN, root=ROOT, adoptions=None):
    run, root = run.resolve(), root.resolve()
    inputs = Inputs()
    registry = inputs.read(run / "dispatch-registry.json")
    status = inputs.read(run / "status.json")
    require(isinstance(status.get("generation"), dict), "generation_status_missing")
    jobs, indices, owned_paths, allowed_ids = {}, {}, {}, set()
    manifest_counts = {}
    launched = set()
    for owner in OWNERS:
        manifest = inputs.read(run / "dispatcher-state" / f"{owner}.json")
        require(isinstance(manifest.get("lane_limit"), int) and not isinstance(manifest["lane_limit"], bool)
                and 1 <= manifest["lane_limit"] <= 16, "invalid_lane_limit")
        manifest_counts[owner] = {}
        for group in ("launched", "queued"):
            values = manifest.get(group)
            require(isinstance(values, list), "invalid_owner_manifest")
            manifest_counts[owner][group] = len(values)
            for value in values:
                path = path_at(value, root)
                require(path not in jobs, "duplicate_job_reservation")
                job = inputs.read(path)
                index = job.get("job_index")
                require(isinstance(index, int) and not isinstance(index, bool) and index >= 0, "invalid_job_index")
                require(index not in indices, "duplicate_job_index")
                ledger_paths = [path_at(job.get(field), root) for field in ("plan", "output", "skip_file")]
                require(len(set(ledger_paths)) == 3 and path not in ledger_paths, "job_paths_overlap")
                for ledger in ledger_paths:
                    require(ledger not in owned_paths, "job_ledger_ownership_collision")
                    owned_paths[ledger] = index
                ids = job.get("remaining_ids")
                require(isinstance(ids, list) and all(isinstance(item, str) and item for item in ids)
                        and len(set(ids)) == len(ids), "invalid_job_allowlist")
                require(not allowed_ids.intersection(ids), "assignment_reservation_collision")
                allowed_ids.update(ids)
                batch = job.get("batch") or path.stem.split("-")[0]
                require(isinstance(batch, str) and batch, "invalid_batch_metadata")
                jobs[path] = {"index": index, "owner": owner, "group": group, "batch": batch,
                              "plan_path": ledger_paths[0], "plan_sha256": job.get("plan_sha256"),
                              "allowlist_count": len(ids)}
                indices[index] = path
                if group == "launched":
                    launched.add(index)

    transfers = inputs.read(run / "job-owner-transfers.json")
    require(transfers.get("schema_version") == "generation-job-owner-transfers-v1"
            and isinstance(transfers.get("transfers"), list), "invalid_transfer_ledger")
    final_transfers = {}
    for transfer in transfers["transfers"]:
        require(isinstance(transfer, dict), "invalid_transfer_record")
        path = path_at(transfer.get("job_path"), root)
        source, target = transfer.get("from_dispatcher"), transfer.get("to_dispatcher")
        require(source in OWNERS and target in OWNERS and source != target, "invalid_transfer_owners")
        require(path in jobs and jobs[path]["index"] == transfer.get("job_index"), "transfer_job_mismatch")
        require(inputs.hashes[path] == transfer.get("job_sha256"), "transfer_job_hash_mismatch")
        require(path_at(transfer.get("previous_owner_manifest_path"), root)
                == (run / "dispatcher-state" / f"{source}.json").resolve(), "transfer_manifest_mismatch")
        require(transfer.get("unlaunched_and_output_skip_absent") is True, "transfer_authorization_missing")
        timestamp(transfer.get("effective_at"))
        if path in final_transfers:
            require(final_transfers[path]["to_dispatcher"] == source, "transfer_chain_mismatch")
        final_transfers[path] = transfer
    for path, transfer in final_transfers.items():
        require(jobs[path]["owner"] == transfer["to_dispatcher"], "transfer_mapping_mismatch")

    actual = copy.deepcopy(registry.get("actual_dispatches_by_batch", {}))
    require(isinstance(actual, dict), "invalid_actual_dispatch_registry")
    # Retain all previous pilot receipts, including other dispatcher owners.
    for index, record in registry.get("actual_pilot_dispatches", {}).items():
        require(isinstance(record, dict) and str(index).isdigit(), "invalid_old_pilot_receipt")
        path = indices.get(int(index))
        if path is not None:
            actual.setdefault(jobs[path]["batch"], {}).setdefault(str(index), copy.deepcopy(record))
    writer_jobs, canonical_jobs = {}, {}
    uncertified_records = 0
    recent = registry.setdefault("recent_reported_writer_names", {})
    require(isinstance(recent, dict), "invalid_recent_writer_registry")
    for owner, (records_field, job_field, time_field) in WRITER_SCHEMAS.items():
        mapping = inputs.read(run / "dispatcher-state" / f"{owner}-writers.json")
        records = mapping.get(records_field)
        require(isinstance(records, list), "invalid_writer_mapping")
        for record in records:
            require(isinstance(record, dict), "invalid_writer_record")
            if owner == "new_dispatch_d" and record.get("event") != "spawn":
                continue
            spawned = record.get(time_field)
            observed = spawned is None and "first_certified_observation_at_utc" in record
            if spawned is None and not observed:
                uncertified_records += 1
                continue  # Missing timing alone never establishes an actual owner.
            if spawned is not None:
                spawned = timestamp(spawned)
            path = path_at(record.get(job_field), root)
            canonical = record.get("canonical_writer")
            # Some owner maps recorded the actual child task name without its
            # parent path. Preserve it; the explicit map and job pin bind owner.
            short_name = (isinstance(canonical, str) and bool(canonical)
                          and "/" not in canonical and canonical not in (".", "..")
                          and not any(char.isspace() for char in canonical))
            require(isinstance(canonical, str) and (short_name or
                    (canonical.startswith(f"/root/overnight_coordinator/{owner}/")
                     and canonical.rsplit("/", 1)[-1])), "writer_owner_mismatch")
            require(path in jobs and jobs[path]["owner"] == owner and jobs[path]["group"] == "launched",
                    "spawn_reservation_mismatch")
            if not observed and (short_name or owner in ("batch2_dispatch_f", "batch2_dispatch_g") or "job_sha256" in record):
                require(record.get("job_sha256") == inputs.hashes[path], "writer_job_pin_mismatch")
            require(canonical not in canonical_jobs or canonical_jobs[canonical] == path, "writer_job_collision")
            require(path not in writer_jobs or writer_jobs[path] == canonical, "conflicting_writer_mapping")
            canonical_jobs[canonical], writer_jobs[path] = path, canonical
            key, batch = str(jobs[path]["index"]), jobs[path]["batch"]
            existing = actual.setdefault(batch, {}).get(key, {})
            update = {"canonical": canonical, "actual_spawn_at": spawned, "reported_by": owner}
            if observed:
                require(record.get("job_sha256") == inputs.hashes[path], "observed_owner_job_pin_mismatch")
                recorded_plan_path = record.get("plan_path")
                # The exact job hash already pins its authoritative plan path.
                # Older certified receipts omitted this redundant path field.
                require((recorded_plan_path is None or path_at(recorded_plan_path, root) == jobs[path]["plan_path"])
                        and isinstance(jobs[path]["plan_sha256"], str) and len(jobs[path]["plan_sha256"]) == 64
                        and all(char in "0123456789abcdef" for char in jobs[path]["plan_sha256"])
                        and record.get("plan_sha256") == jobs[path]["plan_sha256"], "observed_owner_plan_pin_mismatch")
                helper = record.get("helper_counts")
                require(helper is None or isinstance(helper, dict), "invalid_observed_owner_helper_metadata")
                available = helper.get("snapshot_available") if helper is not None else None
                require(available is None or isinstance(available, bool), "invalid_observed_owner_helper_availability")
                # Certified visibility is historical evidence. A later locked
                # boundary cannot undo it; current helper availability is separate.
                fields = ("written", "skipped", "unaccounted")
                if helper is not None and available is not False and (available is True or any(field in helper for field in fields)):
                    counts = [helper.get(field) for field in fields]
                    require(all(isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in counts)
                            and sum(counts) == jobs[path]["allowlist_count"], "observed_owner_helper_counts_mismatch")
                update.update(
                    first_certified_observation_at=timestamp(record["first_certified_observation_at_utc"]),
                    timing_definition="observed_not_spawn",
                    helper_snapshot_available=available,
                )
            actual[batch][key] = merge_receipt(existing, update)
            if key not in recent or recent[key] != canonical:
                recent[key] = merge_receipt(recent.get(key, {}), update)

    now = datetime.now(timezone.utc).isoformat()
    previous = registry.get("job_indices_reported_dispatched")
    require(isinstance(previous, list) and all(isinstance(index, int) and not isinstance(index, bool)
            for index in previous), "invalid_launch_history")
    registry.update(recorded_at=now, job_indices_reported_dispatched=sorted(set(previous) | launched),
                    actual_dispatches_by_batch=actual)
    generation = status["generation"]
    generation["queued_job_counts_by_dispatcher"] = {owner: counts["queued"] for owner, counts in manifest_counts.items()}
    generation["dispatch_metadata_reconciled_at"] = now
    mappings = adoptions or {
        "next_bounded_pilot": run / "support-pilot-adoption-current.json",
        "bounded_follow_on": run / "batch11-support-adoption-current.json",
    }
    batch_counts = {batch: len(records) for batch, records in actual.items()}
    for field, path in mappings.items():
        adoption = inputs.read(path_at(path, root))
        require(isinstance(adoption.get("batch"), str) and isinstance(adoption.get("counts"), dict), "invalid_adoption_metadata")
        require(isinstance(generation.get(field), dict), "adoption_status_object_missing")
        target = generation[field]
        batch = adoption["batch"]
        records = actual.get(batch, {})
        reservations = {owner: sorted(job["index"] for job in jobs.values()
                                    if job["batch"] == batch and job["owner"] == owner) for owner in OWNERS}
        target.update(latest_adoption=copy.deepcopy(adoption), actual_reported_launches=len(records),
                      launches_sampled_at=now, dispatchers={owner: ids for owner, ids in reservations.items() if ids})
        batch_counts[batch] = len(records)
        if field == "bounded_follow_on":
            registry["actual_follow_on_dispatches"] = len(records)
    status["updated_at"] = now
    report = {"unique_reservations": len(jobs), "manifest_counts": manifest_counts,
              "transfers_verified": len(final_transfers), "actual_launches_by_batch": batch_counts,
              "uncertified_writer_records_not_credited": uncertified_records,
              "running_turn_counts_updated": False}
    return registry, status, report, inputs


def atomic_json(path, value):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False, mode="w") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.chmod(path.stat().st_mode & 0o777)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--adoption", action="append", default=[], metavar="STATUS_FIELD=PATH")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    try:
        adoptions = {}
        for item in args.adoption:
            field, separator, path = item.partition("=")
            require(bool(field and separator and path) and field not in adoptions, "invalid_adoption_argument")
            adoptions[field] = path_at(path, args.repo_root)
        registry, status, report, inputs = reconcile(args.run_dir, args.repo_root, adoptions or None)
        inputs.verify()
        if args.apply:
            atomic_json(args.run_dir / "dispatch-registry.json", registry)
            atomic_json(args.run_dir / "status.json", status)
        print(json.dumps({"applied": args.apply, **report}, sort_keys=True))
        return 0
    except (MetadataError, OSError, ValueError, TypeError, KeyError):
        print(json.dumps({"error_code": "dispatch_metadata_reconciliation_failed"}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
