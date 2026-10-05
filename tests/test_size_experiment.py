import copy
import json
from types import SimpleNamespace

import pytest
import torch

import size_experiment as size


def examples():
    return [
        {"id": f"example-{i}", "context": f"Value {i}.", "question": "Choose the value.",
         "options": ["zero", "one"], "answer_index": i % 2,
         "split": "train" if i < 64 else "probe", "train_subset_16": i < 16}
        for i in range(128)
    ]


def test_partition_requires_nested_training_and_disjoint_fixed_probes():
    rows = examples()
    subsets, probes = size.partition(rows)
    assert subsets == {"16": list(range(16)), "64": list(range(64))}
    assert probes == list(range(64, 128))
    rows[64]["train_subset_16"] = True
    with pytest.raises(ValueError, match="Probe examples"):
        size.partition(rows)
    rows[64]["train_subset_16"] = False
    rows[0]["train_subset_16"] = 1
    with pytest.raises(ValueError, match="boolean"):
        size.partition(rows)


def test_changed_probe_labels_do_not_change_fitted_head(tmp_path, monkeypatch):
    monkeypatch.setitem(size.SETTINGS, "max_steps", 10)
    torch.set_num_threads(1)
    torch.manual_seed(5)
    features = torch.randn(128, 8)
    rows = examples()
    subsets, probes = size.partition(rows)
    first = size.fit_condition(tmp_path / "first", features, rows, subsets["16"], probes, 42)
    changed = copy.deepcopy(rows)
    for i in probes:
        changed[i]["answer_index"] = 1 - changed[i]["answer_index"]
    second = size.fit_condition(tmp_path / "second", features, changed, subsets["16"], probes, 42)
    a = size.core.load_head(tmp_path / "first" / "head.pt")
    b = size.core.load_head(tmp_path / "second" / "head.pt")
    for name, value in a.state_dict().items():
        assert torch.equal(value, b.state_dict()[name])
    assert first["report"]["training"]["final"] == second["report"]["training"]["final"]
    assert len(first["report"]["predictions"]) == 80
    assert first["report"]["reloaded_head_logits_identical"]
    assert all(item["id"] not in {f"example-{i}" for i in range(16, 64)}
               for item in first["report"]["predictions"])


def test_feature_cache_reuses_extraction_and_rejects_changed_provenance(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(size.core, "load_backbone", lambda *args: (None, None))
    def encode(*args):
        calls.append(True)
        return torch.ones(128, 8), [20] * 128
    monkeypatch.setattr(size.core, "encode_examples", encode)
    metadata = {"repo": "local-fixture", "revision": "fixed", "data_sha256": "one"}
    first, _ = size.prepare_cache(tmp_path, metadata, examples())
    second, _ = size.prepare_cache(tmp_path, metadata, examples())
    assert torch.equal(first, second)
    assert len(calls) == 1
    with pytest.raises(ValueError, match="provenance differs"):
        size.prepare_cache(tmp_path, {**metadata, "data_sha256": "two"}, examples())
    changed = examples()
    changed[0]["context"] = "Changed after cache creation."
    size.write_json(tmp_path / "examples.json", changed)
    with pytest.raises(ValueError, match="Cached examples differ"):
        size.prepare_cache(tmp_path, metadata, examples())
    size.write_json(tmp_path / "examples.json", examples())
    (tmp_path / "features.pt").write_bytes(b"interrupted-or-modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        size.prepare_cache(tmp_path, metadata, examples())


def test_saved_head_resume_rejects_corruption(tmp_path, monkeypatch):
    monkeypatch.setitem(size.SETTINGS, "max_steps", 1)
    rows = examples()
    subsets, probe = size.partition(rows)
    features = torch.randn(128, 8)
    first = size.fit_condition(tmp_path, features, rows, subsets["16"], probe, 42)
    monkeypatch.setattr(size.core, "fit_head", lambda *args, **kwargs: pytest.fail("Should reuse completed fit"))
    assert size.fit_condition(tmp_path, features, rows, subsets["16"], probe, 42) == first
    with pytest.raises(ValueError, match="seed and training subset"):
        size.fit_condition(tmp_path, features, rows, subsets["16"], probe, 43)
    (tmp_path / "head.pt").write_bytes(b"corruption")
    with pytest.raises(ValueError, match="checksum mismatch"):
        size.fit_condition(tmp_path, features, rows, subsets["16"], probe, 42)


def test_failures_are_retained_as_ineligible_results(tmp_path, monkeypatch):
    data = tmp_path / "expanded.json"
    data.write_text(json.dumps(examples()))
    monkeypatch.setattr(size, "prepare_cache", lambda *args: (_ for _ in ()).throw(RuntimeError("encoding failure")))
    result = size.run_model("ministral-base", data, tmp_path / "runs", tmp_path / "results")
    assert result["status"] == "failed" and not result["eligible"]
    assert result["error"] == "RuntimeError: encoding failure"
    assert json.loads((tmp_path / "results" / "ministral-base.json").read_text()) == result


def test_fresh_verification_checks_every_condition_head(tmp_path, monkeypatch):
    rows = examples()
    size.write_json(tmp_path / "examples.json", rows)
    size.write_json(tmp_path / "manifest.json", {"repo": "fixture", "revision": "fixed"})
    features = torch.randn(128, 8)
    torch.save(features, tmp_path / "features.pt")
    monkeypatch.setattr(size.core, "load_backbone", lambda *args: (None, None))
    monkeypatch.setattr(size.core, "encode_examples", lambda *args: (features[:2], [10, 10]))
    loaded = []
    def load_head(path):
        loaded.append(path)
        return size.core.DecisionHead(8).eval()
    monkeypatch.setattr(size.core, "load_head", load_head)
    assert size.verify_cache(tmp_path) == 0
    result = json.loads((tmp_path / "reload_verification.json").read_text())
    assert len(loaded) == 20 and len(result["heads"]) == 20
    assert result["features_close"] and result["predictions_identical"]


def test_protocol_is_locked_before_measurement_and_cannot_change(tmp_path):
    protocol = {"data_sha256": "fixed", "subsets": {"16": ["a"]},
                "probe_ids": ["p"], "settings": size.SETTINGS, "seeds": size.SEEDS}
    timestamp = size.lock_dataset(tmp_path, protocol)
    assert timestamp.endswith("+00:00")
    assert size.lock_dataset(tmp_path, protocol) == timestamp
    with pytest.raises(ValueError, match="Locked dataset"):
        size.lock_dataset(tmp_path, {**protocol, "probe_ids": ["new-probe"]})


def test_completed_but_underfit_condition_is_ineligible(tmp_path, monkeypatch):
    data = tmp_path / "expanded.json"
    data.write_text(json.dumps(examples()))
    monkeypatch.setattr(size, "prepare_cache", lambda *args: (torch.ones(128, 8), {}))
    def fit(directory, features, rows, train, probe, seed):
        return {"seed": seed, "report": {"passed": len(train) == 16}}
    monkeypatch.setattr(size, "fit_condition", fit)
    def verify(command, **kwargs):
        directory = tmp_path / "runs" / "ministral-base"
        size.write_json(directory / "reload_verification.json", {
            "features_close": True, "predictions_identical": True})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(size.subprocess, "run", verify)
    (tmp_path / "runs" / "ministral-base").mkdir(parents=True)
    result = size.run_model("ministral-base", data, tmp_path / "runs", tmp_path / "results")
    assert result["status"] == "complete"
    assert not result["eligible"]
    assert len(result["conditions"]["64"]["runs"]) == 10
    assert "not eligible" in result["error"]
