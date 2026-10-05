import hashlib
import io
import json
from pathlib import Path

import pytest

from evaluation import isolation, metrics, overlap
from dataset_factory.training_snapshot import SnapshotError, build_snapshot


class _FakeResponse:
    def __init__(self, payload: bytes, final_url: str):
        self._stream = io.BytesIO(payload)
        self._final_url = final_url

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self._stream.close()

    def geturl(self):
        return self._final_url

    def read(self, size=-1):
        return self._stream.read(size)


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _item(item_id, *, context="A synthetic setting.", question="Which choice follows?", options=None, answer_index=0, language=None):
    item = {
        "item_id": item_id,
        "context": context,
        "question": question,
        "options": options or ["first synthetic choice", "second synthetic choice"],
        "answer_index": answer_index,
    }
    if language:
        item["language"] = language
    return item


def test_pinned_download_records_metadata_and_reuses_identical_release(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation, "EXTERNAL_DATA_ROOT", tmp_path / "external_evaluation")
    payload = b"synthetic source artifact"
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request.full_url, timeout))
        return _FakeResponse(payload, "https://approved.example/release.jsonl")

    monkeypatch.setattr(isolation.urllib.request, "urlopen", fake_urlopen)
    spec = isolation.ReleaseSpec(
        benchmark_id="synthetic-suite",
        source_url="https://approved.example/a1b2c3d4/source.jsonl",
        revision="a1b2c3d4",
        artifact_name="source.jsonl",
        source_format="jsonl",
        data_license="CC0-1.0 synthetic fixture",
        code_license=None,
        allowed_hosts=("approved.example",),
        official_metadata_url="https://approved.example/metadata",
        expected_source_rows=1,
    )

    path = isolation.download_release(spec, timeout_seconds=17)
    manifest = json.loads((path.parent / "release-manifest.json").read_text(encoding="utf-8"))
    assert path.read_bytes() == payload
    assert manifest["source_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["revision"] == "a1b2c3d4"
    assert manifest["data_license"] == "CC0-1.0 synthetic fixture"
    assert isolation.download_release(spec) == path
    assert len(calls) == 1


def test_canonical_adapter_deduplicates_and_drops_unapproved_source_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation, "EXTERNAL_DATA_ROOT", tmp_path / "external_evaluation")
    release_dir = isolation.EXTERNAL_DATA_ROOT / "synthetic-suite" / "r1"
    release_dir.mkdir(parents=True)
    source_manifest = release_dir / "release-manifest.json"
    source_artifact = release_dir / "source.jsonl"
    source_artifact.write_bytes(b"synthetic canonicalization source")
    source_manifest.write_text(
        json.dumps({
            "schema_version": "external-release-v1",
            "benchmark_id": "synthetic-suite",
            "revision": "r1",
            "artifact_name": "source.jsonl",
            "source_sha256": hashlib.sha256(source_artifact.read_bytes()).hexdigest(),
            "expected_source_rows": 2,
        }),
        encoding="utf-8",
    )
    output = release_dir / "canonical" / "items.jsonl"
    source_rows = [
        {
            "source_id": "synthetic-a",
            "context": "A synthetic red token.",
            "question": "Which token is described?",
            "options": ["red", "blue"],
            "answer_index": 0,
            "explanation": "synthetic-only explanation field",
        },
        {
            "source_id": "synthetic-copy",
            "context": " A synthetic red token. ",
            "question": "Which token is described?",
            "options": ["blue", "red"],
            "answer_index": 1,
            "explanation": "another synthetic-only field",
        },
    ]

    def transform(row, _row_number):
        return {
            "item_id": row["source_id"],
            "context": row["context"],
            "question": row["question"],
            "options": row["options"],
            "answer_index": row["answer_index"],
        }

    manifest = isolation.canonicalize_records(
        source_rows,
        transform,
        output,
        benchmark_id="synthetic-suite",
        revision="r1",
        source_manifest_path=source_manifest,
        source_artifact_path=source_artifact,
        transform_id="fixture-v1",
        expected_items=1,
    )
    canonical = json.loads(output.read_text(encoding="utf-8"))
    prompt = isolation.model_input(canonical)
    assert manifest["source_record_count"] == 2
    assert manifest["canonical_record_count"] == 1
    assert manifest["duplicate_prompt_count"] == 1
    assert canonical["answer_index"] == 0
    assert set(canonical) == {"item_id", "context", "question", "options", "answer_index"}
    assert "synthetic-a" not in prompt
    assert "explanation" not in prompt


def test_canonicalizer_preserves_duplicate_choice_positions(monkeypatch, tmp_path):
    monkeypatch.setattr(isolation, "EXTERNAL_DATA_ROOT", tmp_path / "external_evaluation")
    release_dir = isolation.EXTERNAL_DATA_ROOT / "synthetic-suite" / "r1"
    release_dir.mkdir(parents=True)
    source_manifest = release_dir / "release-manifest.json"
    source_artifact = release_dir / "source.jsonl"
    source_artifact.write_bytes(b"synthetic source bytes")
    source_manifest.write_text(
        json.dumps({
            "schema_version": "external-release-v1",
            "benchmark_id": "synthetic-suite",
            "revision": "r1",
            "artifact_name": "source.jsonl",
            "source_sha256": hashlib.sha256(source_artifact.read_bytes()).hexdigest(),
            "expected_source_rows": 1,
        }),
        encoding="utf-8",
    )
    output = release_dir / "canonical" / "items.jsonl"

    def transform(_row, _source_ordinal):
        return {
            "item_id": "synthetic-item",
            "context": "Synthetic context.",
            "question": "Which repeated choice is selected?",
            "options": ["same synthetic choice", "same synthetic choice", "other synthetic choice"],
            "answer_index": 1,
        }

    isolation.canonicalize_records(
        [{}],
        transform,
        output,
        benchmark_id="synthetic-suite",
        revision="r1",
        source_manifest_path=source_manifest,
        source_artifact_path=source_artifact,
        transform_id="fixture-duplicate-choice-v1",
        expected_items=1,
    )
    canonical = json.loads(output.read_text(encoding="utf-8"))
    assert canonical["options"] == [
        "same synthetic choice",
        "same synthetic choice",
        "other synthetic choice",
    ]
    assert canonical["answer_index"] == 1


def test_overlap_audit_writes_only_aggregate_and_training_id_outputs(monkeypatch, tmp_path):
    monkeypatch.setattr(overlap, "EXTERNAL_DATA_ROOT", tmp_path / "external_evaluation")
    monkeypatch.setattr(overlap, "TRAINING_SNAPSHOT_ROOT", tmp_path / "training_snapshots")
    external_root = overlap.EXTERNAL_DATA_ROOT
    training_path = overlap.TRAINING_SNAPSHOT_ROOT / "synthetic-v1" / "rows.jsonl"
    training_manifest_path = training_path.parent / "manifest.json"
    canonical_path = external_root / "synthetic-suite" / "r1" / "items.jsonl"
    output = external_root / "synthetic-suite" / "r1" / "overlap"

    long_context = " ".join(f"syntheticword{index}" for index in range(100))
    changed_context = long_context.replace("syntheticword50", "alternateword50")
    train_rows = [
        {"id": "train-exact", "context": "A sealed blue shape has two synthetic sides.", "question": "Which side is sealed?", "options": ["blue", "green"]},
        {"id": "train-near", "context": long_context, "question": "Which synthetic sequence is stated?", "options": ["first route", "second route"]},
        {"id": "train-clear", "context": "A separate synthetic fact.", "question": "What is separate?", "options": ["north", "south"]},
    ]
    benchmark_rows = [
        _item(
            "external-secret-exact-id",
            context="A sealed blue shape has two synthetic sides.",
            question="Which side is sealed?",
            options=["green", "blue"],
            answer_index=1,
        ),
        _item(
            "external-secret-near-id",
            context=changed_context,
            question="Which synthetic sequence is stated?",
            options=["first route", "second route"],
        ),
    ]
    _write_jsonl(training_path, train_rows)
    training_manifest_path.write_text(
        json.dumps({
            "version": "synthetic-v1",
            "artifacts": {"rows.jsonl": {"sha256": hashlib.sha256(training_path.read_bytes()).hexdigest()}},
        }),
        encoding="utf-8",
    )
    _write_jsonl(canonical_path, benchmark_rows)

    report = overlap.run_overlap_audit(
        training_path,
        training_manifest_path,
        canonical_path,
        output,
        benchmark_id="synthetic-suite",
        benchmark_revision="r1",
    )
    quarantine = (output / "quarantine-training-ids.jsonl").read_text(encoding="utf-8")
    exact_overlay = (output / "exact-match-exclusion-overlay.jsonl").read_text(encoding="utf-8")
    aggregate = (output / "overlap-report.json").read_text(encoding="utf-8")
    assert report["exact_prompt_overlap_test_rows"] == 1
    assert report["exact_prompt_overlap_training_ids"] == 1
    assert report["near_lexical_overlap_test_rows"] == 1
    assert report["near_lexical_overlap_training_ids"] == 1
    assert "train-exact" in quarantine and "train-near" in quarantine
    assert "train-exact" in exact_overlay and "train-near" not in exact_overlay
    for output_text in (quarantine, exact_overlay, aggregate):
        assert "external-secret-" not in output_text
        assert "syntheticword" not in output_text
        assert "sealed blue shape" not in output_text


def test_locked_aggregate_metrics_hide_item_identifiers_and_require_exploratory_reuse(monkeypatch, tmp_path):
    monkeypatch.setattr(metrics, "EXTERNAL_DATA_ROOT", tmp_path / "external_evaluation")
    output_directory = metrics.EXTERNAL_DATA_ROOT / "synthetic-suite" / "r1" / "run-1"
    lock_path = output_directory / "evaluation-lock.json"
    checksum_fields = (
        "benchmark_manifest_sha256",
        "canonical_items_sha256",
        "training_snapshot_sha256",
        "checkpoint_sha256",
        "evaluation_code_sha256",
    )
    metadata = {
        "experiment_id": "synthetic-run-1",
        "model_id": "synthetic-model",
        "model_revision": "r1",
        "prompt_format": "fixture-prompt-v1",
        "benchmark_id": "synthetic-suite",
        "benchmark_revision": "r1",
        "training_snapshot_version": "synthetic-training-v1",
        **{field: "b" * 64 for field in checksum_fields},
    }
    metrics.write_experiment_lock(lock_path, metadata)
    aggregate = metrics.aggregate_predictions(
        [
            _item("private-external-item-a", answer_index=0, language="en"),
            _item("private-external-item-b", answer_index=1, language="de"),
        ],
        {"private-external-item-a": 0, "private-external-item-b": 0},
    )
    result_path = metrics.write_aggregate_result(output_directory, lock_path, aggregate)
    result_text = result_path.read_text(encoding="utf-8")
    result = json.loads(result_text)
    assert result["aggregate_metrics"]["overall"] == {"examples": 2, "correct": 1, "accuracy": 0.5}
    assert "private-external-item" not in result_text
    with pytest.raises(isolation.BenchmarkIsolationError, match="already_exists"):
        metrics.write_aggregate_result(output_directory, lock_path, aggregate)
    exploratory_path = metrics.write_aggregate_result(output_directory, lock_path, aggregate, exploratory=True)
    assert json.loads(exploratory_path.read_text(encoding="utf-8"))["exploratory_reuse"] is True


def test_training_snapshot_rejects_external_evaluation_paths_before_reading(tmp_path):
    external_source = isolation.ROOT / "data" / "external_evaluation" / "synthetic" / "items.jsonl"
    with pytest.raises(SnapshotError, match="external_evaluation"):
        build_snapshot(
            "blocked-synthetic-build",
            source_paths=[external_source],
            known_issues_path=tmp_path / "unused-issues.jsonl",
            output_root=tmp_path / "snapshots",
        )
    alias = tmp_path / "external-alias"
    alias.symlink_to(external_source.parent, target_is_directory=True)
    with pytest.raises(SnapshotError, match="external_evaluation"):
        build_snapshot(
            "blocked-symlink-build",
            source_paths=[alias / external_source.name],
            known_issues_path=tmp_path / "unused-issues.jsonl",
            output_root=tmp_path / "snapshots",
        )
