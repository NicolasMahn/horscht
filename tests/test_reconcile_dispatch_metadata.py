import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "runs/overnight-2026-09-30/reconcile-dispatch-metadata.py"
spec = importlib.util.spec_from_file_location("reconcile_dispatch_metadata", SCRIPT)
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)


def write(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def metadata(tmp_path):
    run = tmp_path / "run"
    (run / "dispatcher-state").mkdir(parents=True)
    (run / "jobs").mkdir()
    registry = {"job_indices_reported_dispatched": [9], "actual_pilot_dispatches": {},
                "recent_reported_writer_names": {"9": "preserve-old-name"}, "custom": {"keep": True}}
    status = {"generation": {"active_writers": 2, "active_writers_sampled_at": "prior-runtime-sample",
                             "next_bounded_pilot": {"custom": 10}, "bounded_follow_on": {"custom": 11}},
              "unrelated_status": {"keep": True}}
    write(run / "dispatch-registry.json", registry)
    write(run / "status.json", status)
    jobs = {}
    for name, index in (("batch11-007", 555), ("batch11-008", 556), ("batch11-031", 999), ("batch9-008", 9008)):
        path = run / "jobs" / f"{name}.json"
        write(path, {"job_index": index, "plan": str(run / f"{name}-plan.jsonl"),
                     "output": str(run / f"{name}-output.jsonl"), "skip_file": str(run / f"{name}-skip.jsonl"),
                     "remaining_ids": [f"allowed-{index}"]})
        jobs[name] = path
    for owner in dispatch.OWNERS:
        launched, queued = [], []
        if owner == "resume_dispatch_b":
            launched, queued = [str(jobs["batch11-007"])], [str(jobs["batch11-008"])]
        elif owner == "new_dispatch_d":
            launched = [str(jobs["batch11-031"])]
        elif owner == "resume_dispatch_c":
            launched = [str(jobs["batch9-008"])]
        write(run / "dispatcher-state" / f"{owner}.json", {"lane_limit": 16, "launched": launched, "queued": queued})
    b = {"job_path": str(jobs["batch11-007"]), "canonical_writer": "/root/overnight_coordinator/resume_dispatch_b/actual_unpadded_name",
         "spawn_utc": "2026-09-30T06:00:00Z"}
    d = {"event": "spawn", "job": str(jobs["batch11-031"].relative_to(tmp_path)),
         "canonical_writer": "/root/overnight_coordinator/new_dispatch_d/actual_other_name",
         "spawned_at_utc": "2026-09-30T06:00:01Z"}
    write(run / "dispatcher-state/resume_dispatch_b-writers.json", {"writers": [b, b]})
    write(run / "dispatcher-state/new_dispatch_d-writers.json", {"records": [d, {**d, "event": "followup"}]})
    write(run / "dispatcher-state/batch2_dispatch_f-writers.json", {"jobs": []})
    write(run / "dispatcher-state/batch2_dispatch_g-writers.json", {"mappings": []})
    transfer = {"job_path": str(jobs["batch9-008"]), "job_index": 9008,
                "job_sha256": hashlib.sha256(jobs["batch9-008"].read_bytes()).hexdigest(),
                "from_dispatcher": "batch2_dispatch_f", "to_dispatcher": "resume_dispatch_c",
                "previous_owner_manifest_path": str(run / "dispatcher-state/batch2_dispatch_f.json"),
                "unlaunched_and_output_skip_absent": True, "effective_at": "2026-09-30T05:00:00Z"}
    write(run / "job-owner-transfers.json", {"schema_version": "generation-job-owner-transfers-v1", "transfers": [transfer]})
    for name, batch in (("support-pilot-adoption-current.json", "batch10"), ("batch11-support-adoption-current.json", "batch11")):
        write(run / name, {"batch": batch, "counts": {"eligible_before_dedup": 3}, "scope": "synthetic metadata"})
    return tmp_path, run, jobs


def test_actual_schemas_deduplicate_spawns_preserve_metadata_and_leave_runtime_counts(metadata):
    root, run, _ = metadata
    original_registry = json.loads((run / "dispatch-registry.json").read_bytes())
    original_status = json.loads((run / "status.json").read_bytes())
    registry, status, report, inputs = dispatch.reconcile(run, root)
    inputs.verify()
    assert registry["custom"] == original_registry["custom"]
    assert registry["recent_reported_writer_names"]["9"] == original_registry["recent_reported_writer_names"]["9"]
    assert registry["recent_reported_writer_names"]["555"]["canonical"].endswith("/actual_unpadded_name")
    assert registry["job_indices_reported_dispatched"] == [9, 555, 999, 9008]
    actual = registry["actual_dispatches_by_batch"]["batch11"]
    assert set(actual) == {"555", "999"}
    assert actual["555"]["canonical"].endswith("/actual_unpadded_name")
    assert status["generation"]["bounded_follow_on"]["actual_reported_launches"] == 2
    assert status["generation"]["bounded_follow_on"]["dispatchers"]["resume_dispatch_b"] == [555, 556]
    assert status["generation"]["bounded_follow_on"]["custom"] == 11
    assert status["generation"]["bounded_follow_on"]["latest_adoption"]["counts"]["eligible_before_dedup"] == 3
    assert status["generation"]["active_writers"] == 2
    assert status["generation"]["active_writers_sampled_at"] == "prior-runtime-sample"
    assert status["generation"]["queued_job_counts_by_dispatcher"]["resume_dispatch_b"] == 1
    assert status["unrelated_status"] == original_status["unrelated_status"]
    assert report["unique_reservations"] == 4 and report["transfers_verified"] == 1


@pytest.mark.parametrize("failure,code", [
    ("reservation", "duplicate_job_reservation"), ("output", "job_ledger_ownership_collision"),
    ("transfer", "transfer_mapping_mismatch"), ("writer", "writer_owner_mismatch"),
])
def test_invalid_metadata_fails_without_central_writes(metadata, failure, code):
    root, run, jobs = metadata
    before = {path: path.read_bytes() for path in (run / "status.json", run / "dispatch-registry.json")}
    if failure == "reservation":
        path = run / "dispatcher-state/batch2_dispatch_f.json"
        value = json.loads(path.read_bytes())
        value["queued"] = [str(jobs["batch11-007"])]
    elif failure == "output":
        path = jobs["batch11-031"]
        value = json.loads(path.read_bytes())
        value["output"] = json.loads(jobs["batch11-007"].read_bytes())["output"]
    elif failure == "transfer":
        path = run / "job-owner-transfers.json"
        value = json.loads(path.read_bytes())
        value["transfers"][0]["to_dispatcher"] = "resume_dispatch_b"
    else:
        path = run / "dispatcher-state/resume_dispatch_b-writers.json"
        value = json.loads(path.read_bytes())
        value["writers"][0]["canonical_writer"] = "/root/overnight_coordinator/new_dispatch_d/wrong_owner"
    write(path, value)
    with pytest.raises(dispatch.MetadataError, match=code):
        dispatch.reconcile(run, root)
    assert all(path.read_bytes() == raw for path, raw in before.items())


def test_metadata_change_before_commit_is_rejected(metadata):
    root, run, jobs = metadata
    _, _, _, inputs = dispatch.reconcile(run, root)
    jobs["batch11-007"].write_text(jobs["batch11-007"].read_text() + "\n")
    with pytest.raises(dispatch.MetadataError, match="metadata_changed_before_commit"):
        inputs.verify()


def test_corrected_actual_names_retain_previous_registry_metadata(metadata):
    root, run, _ = metadata
    path = run / "dispatch-registry.json"
    registry = json.loads(path.read_bytes())
    registry["recent_reported_writer_names"]["555"] = {"canonical": "previous-wrong-name", "custom": "keep"}
    write(path, registry)
    reconciled, _, _, _ = dispatch.reconcile(run, root)
    record = reconciled["recent_reported_writer_names"]["555"]
    assert record["custom"] == "keep"
    assert record["canonical"].endswith("/actual_unpadded_name")
    assert record["previous_dispatch_records"] == [{"canonical": "previous-wrong-name"}]


def observed_owner(metadata):
    _, run, jobs = metadata
    path = jobs["batch11-007"]
    job = json.loads(path.read_bytes())
    job["plan_sha256"] = "a" * 64
    write(path, job)
    mapping_path = run / "dispatcher-state/resume_dispatch_b-writers.json"
    mapping = json.loads(mapping_path.read_bytes())
    record = mapping["writers"][0]
    record.update(spawn_utc=None, first_certified_observation_at_utc="2026-09-30T06:52:00Z",
                  job_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), plan_path=job["plan"],
                  plan_sha256=job["plan_sha256"],
                  helper_counts={"snapshot_available": True, "written": 0, "skipped": 0, "unaccounted": 1})
    mapping["writers"] = [record]
    write(mapping_path, mapping)
    return mapping_path, mapping


def test_certified_observation_counts_actual_owner_without_inventing_spawn_time(metadata):
    root, run, _ = metadata
    observed_owner(metadata)
    registry, status, report, _ = dispatch.reconcile(run, root)
    actual = registry["actual_dispatches_by_batch"]["batch11"]["555"]
    assert actual["actual_spawn_at"] is None
    assert actual["first_certified_observation_at"] == "2026-09-30T06:52:00Z"
    assert actual["timing_definition"] == "observed_not_spawn"
    assert status["generation"]["bounded_follow_on"]["actual_reported_launches"] == 2
    assert report["running_turn_counts_updated"] is False


@pytest.mark.parametrize("missing_path", ["null", "absent"])
def test_certified_observation_uses_exact_pinned_job_when_record_plan_path_is_missing(metadata, missing_path):
    root, run, _ = metadata
    path, mapping = observed_owner(metadata)
    if missing_path == "null":
        mapping["writers"][0]["plan_path"] = None
    else:
        mapping["writers"][0].pop("plan_path")
    write(path, mapping)
    registry, _, report, _ = dispatch.reconcile(run, root)
    actual = registry["actual_dispatches_by_batch"]["batch11"]["555"]
    assert actual["actual_spawn_at"] is None and actual["timing_definition"] == "observed_not_spawn"
    assert report["actual_launches_by_batch"]["batch11"] == 2


@pytest.mark.parametrize("failure,code", [
    ("time", "spawn_timestamp_not_utc"), ("job_pin", "observed_owner_job_pin_mismatch"),
    ("plan_pin", "observed_owner_plan_pin_mismatch"), ("helper", "observed_owner_helper_counts_mismatch"),
    ("plan_path", "observed_owner_plan_pin_mismatch"),
])
def test_observation_requires_certified_time_matching_pins_and_available_helper(metadata, failure, code):
    root, run, _ = metadata
    path, mapping = observed_owner(metadata)
    record = mapping["writers"][0]
    if failure == "time":
        record["first_certified_observation_at_utc"] = "2026-09-30T06:52:00"
    elif failure == "job_pin":
        record["job_sha256"] = "0" * 64
    elif failure == "plan_pin":
        record["plan_sha256"] = "0" * 64
    elif failure == "plan_path":
        record["plan_path"] = str(run / "wrong-plan.jsonl")
    else:
        record["helper_counts"]["written"] = -1
    write(path, mapping)
    with pytest.raises(dispatch.MetadataError, match=code):
        dispatch.reconcile(run, root)


@pytest.mark.parametrize("helper", [
    {"snapshot_available": False, "written": None, "skipped": None, "unaccounted": None},
    {"written": 0, "skipped": 0, "unaccounted": 1},
    None,
])
def test_certified_owner_survives_later_unavailable_or_legacy_helper_metadata(metadata, helper):
    root, run, _ = metadata
    path, mapping = observed_owner(metadata)
    mapping["writers"][0]["plan_path"] = None
    mapping["writers"][0]["helper_counts"] = helper
    write(path, mapping)
    registry, status, _, _ = dispatch.reconcile(run, root)
    actual = registry["actual_dispatches_by_batch"]["batch11"]["555"]
    assert actual["actual_spawn_at"] is None
    assert actual["helper_snapshot_available"] == (helper.get("snapshot_available") if helper else None)
    assert status["generation"]["bounded_follow_on"]["actual_reported_launches"] == 2


def test_arbitrary_missing_timestamp_is_not_credited(metadata):
    root, run, _ = metadata
    path = run / "dispatcher-state/resume_dispatch_b-writers.json"
    mapping = json.loads(path.read_bytes())
    for record in mapping["writers"]:
        record["spawn_utc"] = None
    write(path, mapping)
    registry, _, report, _ = dispatch.reconcile(run, root)
    assert set(registry["actual_dispatches_by_batch"]["batch11"]) == {"999"}
    assert report["uncertified_writer_records_not_credited"] == 2


def test_short_actual_owner_name_requires_exact_job_pin(metadata):
    root, run, jobs = metadata
    path = run / "dispatcher-state/resume_dispatch_b-writers.json"
    mapping = json.loads(path.read_bytes())
    for record in mapping["writers"]:
        record["canonical_writer"] = "actual_short_writer"
    write(path, mapping)
    with pytest.raises(dispatch.MetadataError, match="writer_job_pin_mismatch"):
        dispatch.reconcile(run, root)
    for record in mapping["writers"]:
        record["job_sha256"] = hashlib.sha256(jobs["batch11-007"].read_bytes()).hexdigest()
    write(path, mapping)
    registry, _, _, _ = dispatch.reconcile(run, root)
    assert registry["actual_dispatches_by_batch"]["batch11"]["555"]["canonical"] == "actual_short_writer"


def test_f_and_g_actual_spawn_schemas_use_job_indices_pins_and_ignore_g_event_labels(metadata):
    root, run, _ = metadata
    for owner, field, job_field, time_field, index in (
        ("batch2_dispatch_f", "jobs", "job_path", "spawn_utc", 56600),
        ("batch2_dispatch_g", "mappings", "job", "actual_spawn_utc", 56611),
    ):
        job_path = run / "jobs" / f"batch12-{owner}.json"
        write(job_path, {"job_index": index, "plan": str(run / f"{owner}-new-plan.jsonl"),
                         "output": str(run / f"{owner}-new-output.jsonl"), "skip_file": str(run / f"{owner}-new-skip.jsonl"),
                         "remaining_ids": [f"new-allowed-{index}"]})
        manifest_path = run / "dispatcher-state" / f"{owner}.json"
        manifest = json.loads(manifest_path.read_bytes())
        manifest["launched"].append(str(job_path))
        write(manifest_path, manifest)
        record = {job_field: str(job_path), "canonical_writer": f"actual_distinct_{owner}",
                  time_field: "2026-09-30T07:00:00Z", "job_sha256": hashlib.sha256(job_path.read_bytes()).hexdigest(),
                  "event": "canonical_owner_dispatch"}
        write(run / "dispatcher-state" / f"{owner}-writers.json", {field: [record, record]})
    registry, _, report, _ = dispatch.reconcile(run, root)
    records = registry["actual_dispatches_by_batch"]["batch12"]
    assert set(records) == {"56600", "56611"}
    assert records["56600"]["reported_by"] == "batch2_dispatch_f"
    assert records["56611"]["reported_by"] == "batch2_dispatch_g"
    assert records["56611"]["canonical"] == "actual_distinct_batch2_dispatch_g"
    assert report["actual_launches_by_batch"]["batch12"] == 2
    path = run / "dispatcher-state/batch2_dispatch_g-writers.json"
    mapping = json.loads(path.read_bytes())
    mapping["mappings"][0]["job_sha256"] = "0" * 64
    write(path, mapping)
    with pytest.raises(dispatch.MetadataError, match="writer_job_pin_mismatch"):
        dispatch.reconcile(run, root)


def test_cli_default_is_read_only_and_explicit_apply_uses_only_synthetic_metadata(metadata, capsys):
    root, run, _ = metadata
    before = (run / "status.json").read_bytes()
    argv = ["--run-dir", str(run), "--repo-root", str(root)]
    assert dispatch.main(argv) == 0
    assert json.loads(capsys.readouterr().out)["applied"] is False
    assert (run / "status.json").read_bytes() == before
    assert dispatch.main(argv + ["--apply"]) == 0
    assert json.loads(capsys.readouterr().out)["applied"] is True
    assert json.loads((run / "status.json").read_bytes())["generation"]["active_writers"] == 2
