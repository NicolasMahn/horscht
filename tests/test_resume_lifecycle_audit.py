import hashlib
import json

import pytest

import scalable_training


@pytest.fixture
def audit_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(scalable_training, "ROOT", tmp_path)
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "selected"
    checkpoint.mkdir(parents=True)
    previous, current, sources, changes = {}, {}, {}, {}
    for field in scalable_training.LIFECYCLE_HASH_FIELDS:
        source = tmp_path / f"{field}.py"
        source.write_text(f"old resource implementation: {field}")
        previous[field] = hashlib.sha256(source.read_bytes()).hexdigest()
        current[field] = hashlib.sha256(f"new resource implementation: {field}".encode()).hexdigest()
        sources[field] = source.name
        changes[field] = {"previous": previous[field], "current": current[field]}
    (run / "manifest.json").write_text(json.dumps(previous))
    (checkpoint / "checkpoint.json").write_text(json.dumps({"manifest": previous}))
    audit = {"schema_version": "training-resource-lifecycle-audit-v1", "purpose": "resource-lifecycle-only",
             "checkpoint_manifest_sha256": scalable_training.sha256(checkpoint / "checkpoint.json"),
             "previous_run_manifest_sha256": scalable_training.sha256(run / "manifest.json"),
             "source_changes": changes, "previous_source_files": sources}
    path = tmp_path / "audit.json"
    return path, checkpoint, previous, current, audit


def test_audit_verifies_checkpoint_and_actual_frozen_source_hashes(audit_fixture):
    path, checkpoint, previous, current, audit = audit_fixture
    path.write_text(json.dumps(audit))
    verified = scalable_training.validate_resume_lifecycle_audit(path, checkpoint, previous, current)
    assert verified["record"] == audit
    assert verified["sha256"] == scalable_training.sha256(path)
    field = next(iter(audit["source_changes"]))
    (path.parent / audit["previous_source_files"][field]).write_text("unreviewed replacement")
    with pytest.raises(ValueError, match="frozen source checksum"):
        scalable_training.validate_resume_lifecycle_audit(path, checkpoint, previous, current)


@pytest.mark.parametrize("field", ["head_sha256", "loss_sha256", "head_learning_rate", "selection_sha256"])
def test_audit_cannot_relax_non_lifecycle_fields(audit_fixture, field):
    path, checkpoint, previous, current, audit = audit_fixture
    audit["source_changes"][field] = {"previous": "old", "current": "new"}
    audit["previous_source_files"][field] = "irrelevant"
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="allowed implementation fields"):
        scalable_training.validate_resume_lifecycle_audit(path, checkpoint, previous, current)


@pytest.mark.parametrize("field", ["checkpoint_manifest_sha256", "previous_run_manifest_sha256"])
def test_audit_is_bound_to_one_saved_checkpoint(audit_fixture, field):
    path, checkpoint, previous, current, audit = audit_fixture
    audit[field] = "0" * 64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="bound to this checkpoint"):
        scalable_training.validate_resume_lifecycle_audit(path, checkpoint, previous, current)


def test_audit_rejects_unreviewed_current_code(audit_fixture):
    path, checkpoint, previous, current, audit = audit_fixture
    current[next(iter(current))] = "0" * 64
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="source mismatch"):
        scalable_training.validate_resume_lifecycle_audit(path, checkpoint, previous, current)
