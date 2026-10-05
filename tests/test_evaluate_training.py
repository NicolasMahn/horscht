import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import evaluate_training
from evaluation import blind_runner
from dataset_factory import probability_v2
from experiment import CANDIDATES, DecisionHead, format_input
from paired_training import ordered_row, sha256
from scalable_training import build_epoch_plan, evaluate_batches, ordered_lengths
from training_checkpoint import save_training_checkpoint


class ToyConfig:
    hidden_size = 4

    def save_pretrained(self, directory):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text('{"hidden_size": 4}\n')


class ToyTokenizer:
    def __init__(self, harness):
        self.harness = harness
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.padding_side = "right"
        self.seen_rows = []
        self.row_for_encoding = {}

    def _encode(self, prompt):
        tokens = [257, *(byte + 1 for byte in prompt.encode("utf-8")), 258]
        identifier = next((row_id for marker, row_id in self.harness["markers"].items() if marker in prompt), None)
        assert identifier is not None, f"Unknown fixture prompt: {prompt[:80]!r}"
        self.seen_rows.append(identifier)
        self.harness.setdefault("tokenized_rows", []).append(identifier)
        self.row_for_encoding[tuple(tokens)] = identifier
        return tokens

    def __call__(self, prompts, *, return_tensors=None, **kwargs):
        assert kwargs.get("truncation") is False
        values = [prompts] if isinstance(prompts, str) else list(prompts)
        encoded = [self._encode(prompt) for prompt in values]
        if return_tensors != "pt":
            return {"input_ids": encoded[0] if isinstance(prompts, str) else encoded}

        width = max(map(len, encoded))
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, tokens in enumerate(encoded):
            input_ids[index, :len(tokens)] = torch.tensor(tokens)
            attention_mask[index, :len(tokens)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def rows_for(self, input_ids, attention_mask):
        rows = []
        for identifiers, mask in zip(input_ids, attention_mask):
            length = int(mask.sum().item())
            rows.append(self.row_for_encoding[tuple(identifiers[:length].tolist())])
        return rows

    def save_pretrained(self, directory):
        directory.mkdir(parents=True)
        (directory / "tokenizer.json").write_text('{"fixture": true}\n')


class ToyBackbone(torch.nn.Module):
    def __init__(self, config, tracker=None):
        super().__init__()
        self.config = config
        self.embedding = torch.nn.Embedding(259, config.hidden_size)
        self.projection = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.tracker = tracker

    def forward(self, input_ids, attention_mask, use_cache):
        assert use_cache is False
        if self.tracker is not None:
            self.tracker["model_rows"].extend(self.tracker["tokenizer"].rows_for(input_ids, attention_mask))
            self.tracker["attention_backend"].append(getattr(self, "_horscht_attention_backend", None))
        hidden = torch.tanh(self.projection(self.embedding(input_ids)))
        return SimpleNamespace(last_hidden_state=hidden)


def _install_fake_transformers(monkeypatch, harness):
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            tokenizer = ToyTokenizer(harness)
            harness["tokenizer"] = tokenizer
            return tokenizer

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return ToyConfig()

    class AutoModel:
        @staticmethod
        def from_config(config):
            return ToyBackbone(config)

        @staticmethod
        def from_pretrained(*args, **kwargs):
            harness["model_loads"] += 1
            model = ToyBackbone(ToyConfig(), harness)
            for parameter in model.parameters():
                torch.nn.init.constant_(parameter, 0.05)
            return model

    fake = SimpleNamespace(
        __version__="toy-transformers",
        AutoConfig=AutoConfig,
        AutoModel=AutoModel,
        AutoTokenizer=AutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake)


def _fixture_row(identifier, split, index):
    marker = f"FIXTURE_ROW_{split.upper()}_{index}"
    row = probability_v2.build_hard_row({
        "id": identifier,
        "problem_group_id": f"group-{identifier}",
        "split": "train",
        "language": "en",
        "context": f"{marker}: a closed deterministic process.",
        "question": "Which outcome follows from the context?",
        "options": ["first outcome", "second outcome"],
        "answer_index": index % 2,
        "topic": "toy internal evaluation",
        "family": "toy_operation",
        "operation_id": "toy-operation-v1",
        "difficulty": "easy",
        "explanation": "The fixture context states the answer.",
        "provenance": "luna-generated; label not independently verified",
    })
    row["answerability_status"] = "answerable" if index == 0 else "unanswerable"
    row["unanswerable_reason"] = None if index == 0 else "underdetermined"
    row["abstention_choice_id"] = row["choice_ids"][1]
    if index == 1:
        row.pop("operation_id")
    row["_training_snapshot"] = {
        "snapshot_group_id": f"snapshot-{identifier}",
        "split": split,
        "source_schema_kind": "v2",
        "cohorts": [],
    }
    return row, marker


def _fixture(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    rows, splits, markers = [], [], {}
    ids_by_split = {split: [] for split in ("train", "dev", "test")}
    for split, count in (("train", 2), ("dev", 2), ("test", 2)):
        for index in range(count):
            identifier = f"{split}-{index}"
            row, marker = _fixture_row(identifier, split, index)
            rows.append(row)
            markers[marker] = identifier
            splits.append({
                "id": identifier,
                "split": split,
                "snapshot_group_id": row["_training_snapshot"]["snapshot_group_id"],
            })
            ids_by_split[split].append(identifier)
    for filename, values in (("rows.jsonl", rows), ("splits.jsonl", splits)):
        (snapshot / filename).write_text(
            "".join(json.dumps(value, sort_keys=True) + "\n" for value in values), encoding="utf-8"
        )
    snapshot_manifest = {
        "artifacts": {
            filename: {"sha256": sha256(snapshot / filename)}
            for filename in ("rows.jsonl", "splits.jsonl")
        },
    }
    (snapshot / "manifest.json").write_text(json.dumps(snapshot_manifest, sort_keys=True) + "\n")

    harness = {
        "markers": markers,
        "model_loads": 0,
        "model_rows": [],
        "tokenized_rows": [],
        "attention_backend": [],
    }
    tokenizer = ToyTokenizer(harness)
    source_by_id = {row["id"]: row for row in rows}
    seed = 29
    lengths_by_epoch = {}
    for split, epoch in (("dev", -1), ("test", -2)):
        lengths_by_epoch[str(epoch)] = {
            identifier: len(tokenizer._encode(format_input(ordered_row(source_by_id[identifier], seed, epoch))))
            for identifier in ids_by_split[split]
        }

    selection = tmp_path / "selection.json"
    selection_data = {
        "schema_version": 2,
        "model": "qwen-instruct",
        "seed": seed,
        "max_tokens": 512,
        "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
        "system_test_manifest_sha256": "a" * 64,
        "ids_by_split": ids_by_split,
        "ordered_lengths_by_epoch": lengths_by_epoch,
    }
    selection.write_text(json.dumps(selection_data, sort_keys=True) + "\n")

    candidate = CANDIDATES["qwen-instruct"]
    run_manifests = {}
    for arm in ("head", "full"):
        run_dir = tmp_path / f"{arm}-run"
        checkpoint_dir = run_dir / "checkpoints" / "best"
        checkpoint_dir.parent.mkdir(parents=True)
        manifest = {
            "schema_version": 2,
            "arm": arm,
            "model": "qwen-instruct",
            **candidate,
            "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
            "selection_sha256": sha256(selection),
            "system_test_manifest_sha256": selection_data["system_test_manifest_sha256"],
            "seed": seed,
            "epochs": 2,
            "max_tokens": 512,
            "sampler": "length-bucket-shuffle-v1",
            "microbatch_questions": 1,
            "accumulation_questions": 1,
            "max_batch_tokens": 512,
            "bucket_questions": 2,
            "device": "cpu",
            "threads": 1,
            "torch": torch.__version__,
            "transformers": "toy-transformers",
            "parameters_dtype": "float32",
            "compute_autocast": "bfloat16; cache disabled",
            "formatter": "plain-context-question-choices-v1",
            "pooling": "last non-padding token",
            "capacity": 265,
            "attention_backend": "auto",
            "gpu_allocation_cap_gib": None,
            "head_learning_rate": 0.01,
            "backbone_learning_rate": 0.001 if arm == "full" else None,
            "initial_head_state_sha256": "b" * 64,
        }
        run_manifests[arm] = manifest
        (run_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True) + "\n")

        backbone = ToyBackbone(ToyConfig())
        head = DecisionHead(4)
        for parameter in head.parameters():
            torch.nn.init.zeros_(parameter)
        parameters = list(head.parameters())
        if arm == "full":
            parameters.extend(backbone.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.01)
        save_training_checkpoint(
            checkpoint_dir,
            backbone=backbone,
            tokenizer=tokenizer,
            head=head,
            optimizer=optimizer,
            progress={
                "cursor": {"epoch": 1, "next_update": 0, "global_update": 2,
                           "questions_seen": 4, "tokens_seen": 40},
                "best_loss": math.log(2),
                "best_update": 2,
                "best_checkpoint": "best",
                "reason": "best development",
            },
            manifest=manifest,
            include_backbone=arm == "full",
        )
        report = {
            "status": "completed",
            "arm": arm,
            "cursor": {"epoch": 2, "next_update": 0, "global_update": 4,
                       "questions_seen": 4, "tokens_seen": 40},
            "training_questions": 4,
            "training_tokens": 40,
            "best_update": 2,
            "best_checkpoint_relative": "checkpoints/best",
            "selected_checkpoint_reload_verified": True,
            "selected_checkpoint_development": {"soft_cross_entropy": math.log(2)},
        }
        (run_dir / "report.json").write_text(json.dumps(report, sort_keys=True) + "\n")
    return snapshot, selection, ids_by_split, harness, run_manifests


def _patch_runtime(monkeypatch, harness, snapshot):
    def allow_fixture_snapshot(path):
        assert path.resolve() == snapshot.resolve()
        manifest = json.loads((path / "manifest.json").read_text())
        for filename in ("rows.jsonl", "splits.jsonl"):
            assert sha256(path / filename) == manifest["artifacts"][filename]["sha256"]

    def memory(required_bytes, reserve_gib, device):
        harness.setdefault("memory_checks", []).append((required_bytes, reserve_gib, device))

    monkeypatch.setattr(evaluate_training, "require_training_snapshot", allow_fixture_snapshot)
    monkeypatch.setattr(evaluate_training, "require_memory", memory)
    monkeypatch.setattr(evaluate_training, "ROOT", snapshot.parent)
    monkeypatch.setattr(blind_runner, "ROOT", snapshot.parent)
    harness["tokenized_rows"].clear()
    _install_fake_transformers(monkeypatch, harness)


def test_default_stage_reports_development_diagnostics_without_using_test_rows(tmp_path, monkeypatch):
    snapshot, selection, ids_by_split, harness, manifests = _fixture(tmp_path)
    _patch_runtime(monkeypatch, harness, snapshot)

    output = evaluate_training.evaluate_training(
        snapshot=snapshot,
        selection=selection,
        head_run=tmp_path / "head-run",
        full_run=tmp_path / "full-run",
        output=tmp_path / "evaluation",
    )

    result = json.loads(output.read_text())
    assert result["stage"] == "dev"
    assert result["scope"] == "internal development diagnostics; source-held-out test questions were not evaluated"
    assert result["selected_questions"] == len(ids_by_split["dev"]) == 2
    assert result["development_permutation_epochs"] == [-1, -3]
    assert result["selection_sha256"] == sha256(selection)
    assert result["snapshot_manifest_sha256"] == sha256(snapshot / "manifest.json")
    assert set(result["arms"]) == {"head", "full"}
    for arm, evaluated in result["arms"].items():
        development = evaluated["development_orders"]
        assert development["baseline_permutation_epoch"] == -1
        assert development["stress_permutation_epoch"] == -3
        assert abs(development["stress_minus_baseline_soft_cross_entropy"]) < 1e-6
        assert development["baseline"]["overall"]["questions"] == 2
        assert development["baseline"]["overall"]["hard_correct_count"] in {0, 1, 2}
        assert set(development["baseline"]["grouped"]) == {
            "by_language", "by_choice_count", "by_answerability", "by_provenance", "by_review", "by_operation",
        }
        assert set(development["baseline"]["grouped"]["by_operation"]) == {
            "toy-operation-v1", "toy_operation",
        }
        confidence = development["baseline"]["confidence_and_idk"]
        assert sum(item["questions"] for item in confidence["max_probability_bins"]) == 2
        assert confidence["idk"]["questions_with_abstention_option"] == 2
        assert set(confidence["idk_by_answerability"]) == {"answerable", "unanswerable"}
        assert development["choice_order_sensitivity"]["questions"] == 2
        assert 0 <= development["choice_order_sensitivity"]["stable_choice_id_argmax_agreement_fraction"] <= 1
        assert development["choice_order_sensitivity"]["mean_aligned_distribution_total_variation"] == pytest.approx(0)
        assert evaluated["selected_checkpoint_artifacts"]["head.pt"]
        assert evaluated["selected_checkpoint_manifest_sha256"]
    assert "source_held_out_test" not in result
    assert harness["model_loads"] == 2
    assert len(harness["memory_checks"]) == 2
    assert all(reserve == 12 and device == "cpu" for _, reserve, device in harness["memory_checks"])
    assert set(harness["model_rows"]) <= set(ids_by_split["dev"])
    assert set(harness["tokenized_rows"]) <= set(ids_by_split["dev"])
    assert set(harness["attention_backend"]) == {"auto"}
    assert all(
        json.loads((tmp_path / f"{arm}-run" / "manifest.json").read_text()) == manifests[arm]
        for arm in ("head", "full")
    )
    encoded_output = output.read_text()
    assert "FIXTURE_ROW_" not in encoded_output
    assert all(identifier not in encoded_output for identifier in ids_by_split["dev"] + ids_by_split["test"])


def _write_generation_receipt(tmp_path, selection_sha=""):
    artifact = tmp_path / "runs" / "internal-dev-summary.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text('{"synthetic": true}\n', encoding="utf-8")
    candidate_refs = []
    for arm in ("full", "head"):
        manifest_path = tmp_path / f"{arm}-run" / "checkpoints" / "best" / "checkpoint.json"
        candidate_refs.append({
            "arm": arm,
            "manifest_path": manifest_path.relative_to(tmp_path).as_posix(),
            "manifest_sha256": sha256(manifest_path),
        })
    receipt = tmp_path / "generation-reflection-receipt.json"
    receipt.write_text(json.dumps({
        "schema_version": "generation-reflection-receipt-v1",
        "status": "committed",
        "committed_at": "2026-09-30T00:00:00Z",
        "external_test_used": False,
        "internal_dev_artifacts": [{"path": "runs/internal-dev-summary.json", "sha256": sha256(artifact)}],
        "decision_artifact": {"path": "runs/internal-dev-summary.json", "sha256": sha256(artifact)},
        "guidance_artifacts": [],
        "selected_checkpoint_refs": candidate_refs,
    }, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def test_source_test_stage_requires_receipt_and_keeps_test_metrics_separate(tmp_path, monkeypatch):
    snapshot, selection, ids_by_split, harness, _ = _fixture(tmp_path)
    _patch_runtime(monkeypatch, harness, snapshot)
    receipt = _write_generation_receipt(tmp_path)

    output = evaluate_training.evaluate_training(
        snapshot=snapshot,
        selection=selection,
        head_run=tmp_path / "head-run",
        full_run=tmp_path / "full-run",
        output=tmp_path / "source-test",
        stage="test",
        generation_receipt=receipt,
    )

    result = json.loads(output.read_text())
    assert result["stage"] == "test"
    assert result["selected_questions"] == len(ids_by_split["test"]) == 2
    assert result["generation_reflection_receipt"]["sha256"] == sha256(receipt)
    assert "development_permutation_epochs" not in result
    assert all("development_orders" not in arm for arm in result["arms"].values())
    assert all(arm["source_held_out_test"]["questions"] == 2 for arm in result["arms"].values())
    assert set(harness["model_rows"]) <= set(ids_by_split["test"])
    assert set(harness["tokenized_rows"]) <= set(ids_by_split["test"])


def test_source_test_stage_refuses_missing_receipt_before_model_or_tokenizer_use(tmp_path, monkeypatch):
    snapshot, selection, _, harness, _ = _fixture(tmp_path)
    _patch_runtime(monkeypatch, harness, snapshot)

    with pytest.raises(ValueError, match="requires a committed generation-reflection receipt"):
        evaluate_training.evaluate_training(
            snapshot=snapshot,
            selection=selection,
            head_run=tmp_path / "head-run",
            full_run=tmp_path / "full-run",
            output=tmp_path / "source-test",
            stage="test",
        )
    assert harness["model_loads"] == 0
    assert not harness["tokenized_rows"]
    assert not harness["model_rows"]


def test_source_test_stage_refuses_receipt_bound_to_different_checkpoints(tmp_path, monkeypatch):
    snapshot, selection, _, harness, _ = _fixture(tmp_path)
    _patch_runtime(monkeypatch, harness, snapshot)
    receipt = _write_generation_receipt(tmp_path)
    data = json.loads(receipt.read_text())
    data["selected_checkpoint_refs"][0]["manifest_sha256"] = "0" * 64
    receipt.write_text(json.dumps(data, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(blind_runner.BlindEvaluationError):
        evaluate_training.evaluate_training(
            snapshot=snapshot,
            selection=selection,
            head_run=tmp_path / "head-run",
            full_run=tmp_path / "full-run",
            output=tmp_path / "source-test",
            stage="test",
            generation_receipt=receipt,
        )
    assert harness["model_loads"] == 0
    assert not harness["tokenized_rows"]
    assert not harness["model_rows"]


def test_refuses_existing_output_before_model_loading(tmp_path, monkeypatch):
    harness = {"markers": {}, "model_loads": 0}
    _install_fake_transformers(monkeypatch, harness)
    output = tmp_path / "already-used"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        evaluate_training.evaluate_training(
            snapshot=tmp_path / "missing-snapshot",
            selection=tmp_path / "missing-selection.json",
            head_run=tmp_path / "missing-head",
            full_run=tmp_path / "missing-full",
            output=output,
        )
    assert harness["model_loads"] == 0


def test_refuses_corrupt_selected_checkpoint_before_test_evaluation(tmp_path, monkeypatch):
    snapshot, selection, _, harness, _ = _fixture(tmp_path)
    _patch_runtime(monkeypatch, harness, snapshot)
    head_file = tmp_path / "head-run" / "checkpoints" / "best" / "head.pt"
    head_file.write_bytes(head_file.read_bytes() + b"corrupt")

    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate_training.evaluate_training(
            snapshot=snapshot,
            selection=selection,
            head_run=tmp_path / "head-run",
            full_run=tmp_path / "full-run",
            output=tmp_path / "evaluation",
        )
    assert harness["model_loads"] == 0
    assert not harness.get("model_rows")
    assert not (tmp_path / "evaluation").exists()


@pytest.mark.parametrize("mutation", ["interrupted", "exposure_mismatch"])
def test_refuses_incomplete_or_unmatched_runs_before_model_loading(tmp_path, monkeypatch, mutation):
    snapshot, selection, _, harness, _ = _fixture(tmp_path)
    _patch_runtime(monkeypatch, harness, snapshot)
    head_report = tmp_path / "head-run" / "report.json"
    report = json.loads(head_report.read_text())
    if mutation == "interrupted":
        report["status"] = "interrupted"
    else:
        report["training_tokens"] += 1
    head_report.write_text(json.dumps(report, sort_keys=True) + "\n")

    with pytest.raises(ValueError, match="completed|different exposure|recorded training exposure"):
        evaluate_training.evaluate_training(
            snapshot=snapshot,
            selection=selection,
            head_run=tmp_path / "head-run",
            full_run=tmp_path / "full-run",
            output=tmp_path / "evaluation",
        )
    assert harness["model_loads"] == 0
    assert not (tmp_path / "evaluation").exists()
