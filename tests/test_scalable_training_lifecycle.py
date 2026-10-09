import hashlib
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import scalable_training
from dataset_factory import probability_v2
from training_checkpoint import load_training_checkpoint


class ToyConfig:
    hidden_size = 4

    def save_pretrained(self, directory):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text('{"hidden_size": 4}\n')


class ToyTokenizer:
    def __init__(self, harness):
        self.harness = harness
        self.padding_side = "right"
        self.pad_token_id = 0
        self.eos_token = "<eos>"
        self.prompts_seen = []
        self.row_for_encoding = {}

    def _encode(self, prompt):
        encoded = [257, *(byte + 1 for byte in prompt.encode("utf-8")), 258]
        row_id = next((identifier for marker, identifier in self.harness["markers"].items()
                       if marker in prompt), None)
        assert row_id is not None, f"Unrecognized toy prompt: {prompt[:80]!r}"
        self.prompts_seen.append((row_id, prompt))
        self.row_for_encoding[tuple(encoded)] = row_id
        return encoded

    def __call__(self, prompts, *, return_tensors=None, padding=False, **kwargs):
        assert kwargs.get("truncation") is False
        values = [prompts] if isinstance(prompts, str) else list(prompts)
        encoded = [self._encode(prompt) for prompt in values]
        if return_tensors != "pt":
            return {"input_ids": encoded[0] if isinstance(prompts, str) else encoded}

        width = max(map(len, encoded))
        input_ids = torch.zeros((len(encoded), width), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for index, sequence in enumerate(encoded):
            input_ids[index, :len(sequence)] = torch.tensor(sequence)
            attention_mask[index, :len(sequence)] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def rows_for(self, input_ids, attention_mask):
        result = []
        for identifiers, mask in zip(input_ids, attention_mask):
            length = int(mask.sum().item())
            result.append(self.row_for_encoding[tuple(identifiers[:length].tolist())])
        return result

    def save_pretrained(self, directory):
        directory.mkdir(parents=True)
        (directory / "tokenizer.json").write_text('{"fixture": true}\n')


class ToyBackbone(torch.nn.Module):
    def __init__(self, config, tokenizer, harness, *, record_forward):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.harness = harness
        self.record_forward = record_forward
        self.embedding = torch.nn.Embedding(259, config.hidden_size)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(config.hidden_size, config.hidden_size)])
        self.dropout = torch.nn.Dropout(0.2)
        self.gradient_checkpointing_enabled = False

    def gradient_checkpointing_enable(self, **kwargs):
        assert kwargs == {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
        self.gradient_checkpointing_enabled = True

    def forward(self, input_ids, attention_mask, use_cache):
        assert use_cache is False
        if self.record_forward:
            row_ids = self.tokenizer.rows_for(input_ids, attention_mask)
            self.harness["model_rows"].extend(row_ids)
            self.harness["model_calls"].append({
                "rows": row_ids,
                "training": self.training,
                "grad_enabled": torch.is_grad_enabled(),
            })
        hidden = self.embedding(input_ids)
        hidden = self.dropout(torch.tanh(self.layers[-1](hidden)))
        return SimpleNamespace(last_hidden_state=hidden)


def _install_toy_transformers(monkeypatch, harness):
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            tokenizer = ToyTokenizer(harness)
            harness["tokenizers"].append(tokenizer)
            return tokenizer

    class AutoConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return ToyConfig()

    class AutoModel:
        @staticmethod
        def from_config(config):
            tokenizer = harness["tokenizers"][-1]
            return ToyBackbone(config, tokenizer, harness, record_forward=False)

        @staticmethod
        def from_pretrained(*args, **kwargs):
            tokenizer = harness["tokenizers"][-1]
            model = ToyBackbone(ToyConfig(), tokenizer, harness, record_forward=True)
            harness["models"].append(model)
            harness["initial_model_states"].append({
                name: tensor.detach().clone() for name, tensor in model.state_dict().items()
            })
            return model

    def adafactor(params, **kwargs):
        assert kwargs == {
            "scale_parameter": False,
            "relative_step": False,
            "warmup_init": False,
            "beta1": None,
            "weight_decay": 0,
            "clip_threshold": 1.0,
        }
        return torch.optim.AdamW(params, weight_decay=0)

    fake_transformers = SimpleNamespace(
        __version__="toy-transformers",
        Adafactor=adafactor,
        AutoConfig=AutoConfig,
        AutoModel=AutoModel,
        AutoTokenizer=AutoTokenizer,
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _snapshot(tmp_path):
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    rows = []
    split_rows = []
    ids_by_split = {name: [] for name in ("train", "dev", "test")}
    markers = {}
    for split, count in (("train", 4), ("dev", 2), ("test", 2)):
        for index in range(count):
            identifier = f"{split}-{index}"
            marker = f"ROW_MARKER_{split.upper()}_{index}"
            markers[marker] = identifier
            raw = {
                "id": identifier,
                "problem_group_id": f"group-{identifier}",
                "split": "train",
                "language": "en",
                "context": f"{marker}: a short deterministic fixture context.",
                "question": "Which outcome follows from the context?",
                "options": ["first outcome", "second outcome"],
                "answer_index": index % 2,
                "topic": "toy training fixture",
                "family": "toy_operation",
                "operation_id": "toy-operation-v1",
                "difficulty": "easy",
                "explanation": "The fixture context states the answer.",
                "provenance": "luna-generated; label not independently verified",
            }
            row = probability_v2.build_hard_row(raw)
            row["_training_snapshot"] = {
                "snapshot_group_id": f"snapshot-{identifier}",
                "source_schema_kind": "v2",
                "cohorts": [],
            }
            rows.append(row)
            split_rows.append({"id": identifier, "split": split, "snapshot_group_id": f"snapshot-{identifier}"})
            ids_by_split[split].append(identifier)

    _write_jsonl(snapshot / "rows.jsonl", rows)
    _write_jsonl(snapshot / "splits.jsonl", split_rows)
    artifacts = {}
    for filename in ("rows.jsonl", "splits.jsonl"):
        artifacts[filename] = {"sha256": hashlib.sha256((snapshot / filename).read_bytes()).hexdigest()}
    (snapshot / "manifest.json").write_text(json.dumps({"artifacts": artifacts}, sort_keys=True) + "\n")

    system_test_fixture = tmp_path / "independent-system-test.jsonl"
    system_test_fixture.write_text('{"fixture": "independent reference artifact"}\n')
    system_lock = tmp_path / "system-test-lock.json"
    lock = {
        "schema_version": "system-test-lock-v1",
        "status": "locked",
        "independent_from_training_generator": True,
        "locked_at": "2026-09-30T08:00:00Z",
        "training_snapshot_manifest_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest(),
        "files": {system_test_fixture.name: hashlib.sha256(system_test_fixture.read_bytes()).hexdigest()},
        "training_exclusions": {},
    }
    system_lock.write_text(json.dumps(lock, sort_keys=True) + "\n")

    selection = tmp_path / "selection.json"
    selection_data = {
        "model": "qwen-instruct",
        "seed": 314,
        "max_tokens": 512,
        "snapshot_manifest_sha256": hashlib.sha256((snapshot / "manifest.json").read_bytes()).hexdigest(),
        "system_test_manifest_sha256": hashlib.sha256(system_lock.read_bytes()).hexdigest(),
        "ids_by_split": ids_by_split,
    }
    selection.write_text(json.dumps(selection_data, sort_keys=True) + "\n")
    return snapshot, system_lock, selection, ids_by_split, markers


def _args(tmp_path, arm, snapshot, system_lock, selection, *, output_name, resume=None):
    return SimpleNamespace(
        snapshot=snapshot,
        selection=selection,
        output=tmp_path / output_name,
        arm=arm,
        head_learning_rate=0.02,
        backbone_learning_rate=0.005,
        accumulation=1,
        eval_every=1,
        threads=1,
        reserve_gib=0.0,
        resume=resume,
        device="cpu",
        system_test_manifest=system_lock,
        launch_approved=True,
        epochs=2,
        microbatch=1,
        max_batch_tokens=512,
        bucket_questions=2,
        checkpoint_every=1,
        checkpoint_seconds=10**9,
        retain_checkpoints=32,
        feature_cache_mib=1,
        status_file=tmp_path / f"{output_name}-status.json",
        model="qwen-instruct",
        seed=314,
        max_tokens=512,
    )


def _checkpoint_for_report(output, report_key):
    return output / json.loads((output / "report.json").read_text())[report_key]


def _assert_tree_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys()
        for key in left:
            _assert_tree_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert isinstance(right, type(left)) and len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_tree_equal(left_value, right_value)
    else:
        assert left == right


@pytest.mark.parametrize("arm", ["head", "full"])
def test_scalable_arm_trains_two_epochs_and_resumes_without_touching_test_rows(
    tmp_path, monkeypatch, arm
):
    torch_state = torch.get_rng_state()
    python_state = random.getstate()
    previous_threads = torch.get_num_threads()
    try:
        harness = {
            "markers": {},
            "tokenizers": [],
            "models": [],
            "initial_model_states": [],
            "model_rows": [],
            "model_calls": [],
            "evaluation_rows": [],
            "memory_checks": [],
            "curve_writes": [],
        }
        snapshot, system_lock, selection, ids_by_split, markers = _snapshot(tmp_path)
        harness["markers"].update(markers)
        _install_toy_transformers(monkeypatch, harness)

        def require_fixture_snapshot(path):
            assert path.resolve() == snapshot.resolve()
            manifest = json.loads((path / "manifest.json").read_text())
            rows = [json.loads(line) for line in (path / "rows.jsonl").read_text().splitlines()]
            splits = [json.loads(line) for line in (path / "splits.jsonl").read_text().splitlines()]
            assert all(not probability_v2.validate_v2_row(row) for row in rows)
            assert {row["id"] for row in rows} == {row["id"] for row in splits}
            for filename in ("rows.jsonl", "splits.jsonl"):
                actual_hash = hashlib.sha256((path / filename).read_bytes()).hexdigest()
                assert manifest["artifacts"][filename]["sha256"] == actual_hash
            harness.setdefault("snapshot_guard_checks", []).append(path)

        monkeypatch.setattr(scalable_training, "require_training_snapshot", require_fixture_snapshot)

        def check_memory(required_bytes, reserve_gib, device):
            harness["memory_checks"].append((required_bytes, reserve_gib, device))

        monkeypatch.setattr(scalable_training, "require_memory", check_memory)
        monkeypatch.setattr(scalable_training, "available_memory_bytes", lambda: 8 * 1024**3)
        monkeypatch.setattr(scalable_training, "peak_rss_gib", lambda: 0.01)
        monkeypatch.setattr(scalable_training, "device_memory", lambda device: {
            "peak_device_allocated_gib": 0.0,
            "peak_device_reserved_gib": 0.0,
        })
        monkeypatch.setattr(scalable_training, "synchronize", lambda device: None)
        monkeypatch.setattr(scalable_training.shutil, "disk_usage", lambda path: SimpleNamespace(free=16 * 1024**3))

        def write_curves(path, history):
            harness["curve_writes"].append(len(history))
            path.write_text(f"toy curves for {len(history)} observations\n")

        monkeypatch.setattr(scalable_training, "write_curves", write_curves)
        real_evaluate = scalable_training.evaluate_batches

        def trace_evaluation(tokenizer, backbone, head, rows, *args):
            identifiers = [row["id"] for row in rows]
            harness["evaluation_rows"].extend(identifiers)
            assert set(identifiers) <= set(ids_by_split["dev"])
            return real_evaluate(tokenizer, backbone, head, rows, *args)

        monkeypatch.setattr(scalable_training, "evaluate_batches", trace_evaluation)

        baseline_args = _args(
            tmp_path, arm, snapshot, system_lock, selection, output_name=f"{arm}-baseline"
        )
        assert scalable_training.run_scalable_arm(baseline_args) == 0
        baseline_output = baseline_args.output
        baseline_report = json.loads((baseline_output / "report.json").read_text())
        assert baseline_report["status"] == "completed"
        assert baseline_report["cursor"]["epoch"] == 2
        assert baseline_report["cursor"]["global_update"] == 8
        assert baseline_report["selected_checkpoint_reload_verified"] is True
        assert baseline_report["training_questions"] == 8

        initial_checkpoint = next(
            path for path in (baseline_output / "checkpoints").iterdir()
            if json.loads((path / "checkpoint.json").read_text())["progress"]["cursor"]["global_update"] == 0
        )
        final_checkpoint = _checkpoint_for_report(baseline_output, "last_checkpoint_relative")
        initial_metadata = json.loads((initial_checkpoint / "checkpoint.json").read_text())
        final_metadata = json.loads((final_checkpoint / "checkpoint.json").read_text())
        assert initial_metadata["manifest"]["backbone_trainable"] is (arm == "full")
        assert final_metadata["progress"]["cursor"]["epoch"] == 2
        assert {"head.pt", "optimizer.pt", "torch_rng.pt", "python_rng.json"} <= set(final_metadata["files"])
        assert "tokenizer/tokenizer.json" in final_metadata["files"]
        assert "backbone_config/config.json" in final_metadata["files"]
        assert ("backbone.pt" in final_metadata["files"]) is (arm == "full")
        assert (initial_metadata["files"]["head.pt"] != final_metadata["files"]["head.pt"])
        if arm == "full":
            assert initial_metadata["files"]["backbone.pt"] != final_metadata["files"]["backbone.pt"]
            assert all(parameter.requires_grad for parameter in harness["models"][0].parameters())
            assert harness["models"][0].gradient_checkpointing_enabled
            assert any(call["training"] and call["grad_enabled"] for call in harness["model_calls"])
        else:
            assert all(not parameter.requires_grad for parameter in harness["models"][0].parameters())
            assert all(
                torch.equal(parameter, harness["initial_model_states"][0][name])
                for name, parameter in harness["models"][0].state_dict().items()
            )

        for checkpoint_path in (baseline_output / "checkpoints").iterdir():
            metadata = json.loads((checkpoint_path / "checkpoint.json").read_text())
            artifacts = metadata["files"]
            actual = {
                path.relative_to(checkpoint_path).as_posix()
                for path in checkpoint_path.rglob("*")
                if path.is_file() and path.name != "checkpoint.json"
            }
            assert actual == set(artifacts)
            for relative, expected_hash in artifacts.items():
                actual_hash = hashlib.sha256((checkpoint_path / relative).read_bytes()).hexdigest()
                assert actual_hash == expected_hash

        initial_checkpoint_state = load_training_checkpoint(
            initial_checkpoint,
            backbone=harness["models"][0],
            head=scalable_training.DecisionHead(4),
        )
        assert initial_checkpoint_state["progress"]["cursor"]["global_update"] == 0

        real_run_epoch_updates = scalable_training.run_epoch_updates
        armed = {"value": True}

        def interrupt_after_first_update(plan, cursor, *, stop, step, observe, before_step=None):
            def stop_after_observe(progress, metrics):
                observe(progress, metrics)
                if armed["value"]:
                    armed["value"] = False
                    stop.request("toy update-boundary interruption")

            return real_run_epoch_updates(
                plan, cursor, stop=stop, step=step, observe=stop_after_observe, before_step=before_step
            )

        monkeypatch.setattr(scalable_training, "run_epoch_updates", interrupt_after_first_update)
        interrupted_args = _args(
            tmp_path, arm, snapshot, system_lock, selection, output_name=f"{arm}-interrupted"
        )
        assert scalable_training.run_scalable_arm(interrupted_args) == 130
        interrupted_report = json.loads((interrupted_args.output / "report.json").read_text())
        assert interrupted_report["status"] == "interrupted"
        assert interrupted_report["cursor"]["global_update"] == 1
        resume_checkpoint = _checkpoint_for_report(interrupted_args.output, "last_checkpoint_relative")
        monkeypatch.setattr(scalable_training, "run_epoch_updates", real_run_epoch_updates)
        resumed_args = _args(
            tmp_path, arm, snapshot, system_lock, selection, output_name=f"{arm}-resumed",
            resume=resume_checkpoint,
        )
        assert scalable_training.run_scalable_arm(resumed_args) == 0
        resumed_output = resumed_args.output
        resumed_report = json.loads((resumed_output / "report.json").read_text())
        assert resumed_report["cursor"] == baseline_report["cursor"]
        assert resumed_report["training_questions"] == baseline_report["training_questions"]
        assert resumed_report["best_update"] == baseline_report["best_update"]
        assert abs(resumed_report["best_dev_soft_cross_entropy"] - baseline_report["best_dev_soft_cross_entropy"]) < 1e-8

        resumed_checkpoint = _checkpoint_for_report(resumed_output, "last_checkpoint_relative")
        for filename in ("head.pt", "optimizer.pt", "torch_rng.pt"):
            _assert_tree_equal(
                torch.load(final_checkpoint / filename, map_location="cpu", weights_only=True),
                torch.load(resumed_checkpoint / filename, map_location="cpu", weights_only=True),
            )
        if arm == "full":
            _assert_tree_equal(
                torch.load(final_checkpoint / "backbone.pt", map_location="cpu", weights_only=True),
                torch.load(resumed_checkpoint / "backbone.pt", map_location="cpu", weights_only=True),
            )
        assert json.loads((final_checkpoint / "python_rng.json").read_text()) == json.loads(
            (resumed_checkpoint / "python_rng.json").read_text()
        )

        test_ids = set(ids_by_split["test"])
        assert test_ids.isdisjoint(harness["evaluation_rows"])
        assert test_ids.isdisjoint(harness["model_rows"])
        for tokenizer in harness["tokenizers"]:
            assert test_ids.isdisjoint(row_id for row_id, _ in tokenizer.prompts_seen)
            assert all("ROW_MARKER_TEST_" not in prompt for _, prompt in tokenizer.prompts_seen)
        assert harness["memory_checks"]
        assert len(harness["snapshot_guard_checks"]) == 3
        assert all(device == "cpu" for _, _, device in harness["memory_checks"])
        assert harness["curve_writes"]
        assert (baseline_output / "curves.html").is_file()
        plateau_args = _args(tmp_path, arm, snapshot, system_lock, selection,
                             output_name=f"{arm}-plateau")
        plateau_args.epochs = 4
        plateau_args.early_stop_patience = 2
        plateau_args.early_stop_min_delta = 100.0
        plateau_args.early_stop_min_epochs = 1
        assert scalable_training.run_scalable_arm(plateau_args) == 0
        plateau = json.loads((plateau_args.output / "report.json").read_text())
        assert plateau["early_stopped"] is True
        assert plateau["status"] == "completed"
        assert plateau["stop_reason"] == "development plateau"
        assert plateau["training_questions"] == len(ids_by_split["train"])
        assert plateau["selected_checkpoint_reload_verified"] is True
        assert set(harness["model_rows"]).isdisjoint(test_ids)

        from paired_training import TrainingMemoryPressure
        memory_calls = {"count": 0}

        def pressure_before_third_update(required_bytes, reserve_gib, device):
            memory_calls["count"] += 1
            if memory_calls["count"] == 4:
                raise TrainingMemoryPressure({"effective_available_bytes": 0}, required_bytes, 0)

        monkeypatch.setattr(scalable_training, "require_memory", pressure_before_third_update)
        monkeypatch.setattr(scalable_training.StopController, "wait",
                            lambda controller, seconds: controller.request("resource test interruption"))
        pressure_args = _args(tmp_path, arm, snapshot, system_lock, selection, output_name=f"{arm}-pressure")
        pressure_args.eval_every = pressure_args.checkpoint_every = 100
        assert scalable_training.run_scalable_arm(pressure_args) == 130
        pressure_report = json.loads((pressure_args.output / "report.json").read_text())
        assert pressure_report["cursor"]["global_update"] == 2
        assert pressure_report["training_questions"] == 2
        pressure_checkpoint = _checkpoint_for_report(pressure_args.output, "last_checkpoint_relative")
        pressure_metadata = json.loads((pressure_checkpoint / "checkpoint.json").read_text())
        assert pressure_metadata["progress"]["reason"] == "memory pressure before next update"
        assert pressure_metadata["progress"]["cursor"] == pressure_report["cursor"]
        assert set(pressure_metadata["files"]) >= {"optimizer.pt", "torch_rng.pt", "python_rng.json"}

        monkeypatch.setattr(scalable_training, "require_memory", check_memory)
        pressure_resume = _args(tmp_path, arm, snapshot, system_lock, selection,
                                output_name=f"{arm}-pressure-resumed", resume=pressure_checkpoint)
        pressure_resume.eval_every = pressure_resume.checkpoint_every = 100
        assert scalable_training.run_scalable_arm(pressure_resume) == 0
        pressure_final = _checkpoint_for_report(pressure_resume.output, "last_checkpoint_relative")
        for filename in ("head.pt", "optimizer.pt", "torch_rng.pt") + (("backbone.pt",) if arm == "full" else ()):
            _assert_tree_equal(torch.load(final_checkpoint / filename, map_location="cpu", weights_only=True),
                               torch.load(pressure_final / filename, map_location="cpu", weights_only=True))
        assert json.loads((pressure_final / "python_rng.json").read_text()) == json.loads((final_checkpoint / "python_rng.json").read_text())

        models_before = len(harness["models"])
        monkeypatch.setattr(scalable_training, "require_memory",
                            lambda required, reserve, device: (_ for _ in ()).throw(TrainingMemoryPressure({"effective_available_bytes": 0}, required, 0)))
        startup_args = _args(tmp_path, arm, snapshot, system_lock, selection, output_name=f"{arm}-startup-wait")
        assert scalable_training.run_scalable_arm(startup_args) == 130
        assert len(harness["models"]) == models_before
        startup_status = json.loads((startup_args.output / "status.json").read_text())
        assert startup_status["status"] == "interrupted" and startup_status["global_update"] == 0
    finally:
        torch.set_rng_state(torch_state)
        random.setstate(python_state)
        torch.set_num_threads(previous_threads)


def test_scheduled_pilot_selects_full_dev_only_and_can_continue_from_weights(tmp_path, monkeypatch):
    harness = {'markers': {}, 'tokenizers': [], 'models': [], 'initial_model_states': [],
               'model_rows': [], 'model_calls': []}
    snapshot, lock, selection, ids, markers = _snapshot(tmp_path)
    harness['markers'].update(markers)
    _install_toy_transformers(monkeypatch, harness)
    monkeypatch.setattr(scalable_training, 'require_training_snapshot', lambda path: None)
    monkeypatch.setattr(scalable_training, 'require_memory', lambda *args: None)
    monkeypatch.setattr(scalable_training, 'write_curves', lambda *args: None)
    args = _args(tmp_path, 'last-block', snapshot, lock, selection, output_name='source')
    args.max_updates = 3
    args.lr_schedule = 'warmup-cosine'
    args.warmup_fraction = .3
    args.fast_dev_questions = 1
    args.full_dev_every = 3
    args.order_consistency_questions = 0
    assert scalable_training.run_scalable_arm(args) == 0
    history = json.loads((args.output / 'history.json').read_text())
    assert [point['update'] for point in history if 'dev' in point] == [0, 3]
    assert [point['update'] for point in history if 'fast_dev' in point] == [1, 2, 3]
    report = json.loads((args.output / 'report.json').read_text())
    assert report['cursor']['global_update'] == 3
    assert report['best_update'] in (0, 3)
    assert report['selected_checkpoint_reload_verified']
    checkpoint = _checkpoint_for_report(args.output, 'last_checkpoint_relative')
    assert (checkpoint / 'scheduler.pt').is_file()
    continued = _args(tmp_path, 'last-block', snapshot, lock, selection, output_name='continued')
    continued.compact_last_block_checkpoints = True
    continued.warm_start = checkpoint
    continued.max_updates = 3
    continued.full_dev_every = 3
    continued.lr_schedule = 'warmup-cosine'
    continued.order_consistency_questions = 0
    assert scalable_training.run_scalable_arm(continued) == 0
    manifest = json.loads((continued.output / 'manifest.json').read_text())
    metadata = json.loads((checkpoint / 'checkpoint.json').read_text())
    source_head = scalable_training.DecisionHead(4)
    source_head.load_state_dict(torch.load(checkpoint / 'head.pt', weights_only=True))
    assert manifest['initial_head_state_sha256'] == scalable_training.model_state_sha256(source_head)
    assert manifest['warm_start']['checkpoint_manifest_sha256'] == scalable_training.sha256(checkpoint / 'checkpoint.json')
    continued_initial = next((continued.output / 'checkpoints').glob('update-00000000-*'))
    assert torch.load(continued_initial / 'optimizer.pt', weights_only=True)['state'] == {}
    assert json.loads((continued_initial / 'checkpoint.json').read_text())['progress']['cursor']['global_update'] == 0
    assert set(harness['model_rows']).isdisjoint(ids['test'])
    import copy
    real_run_updates = scalable_training.run_epoch_updates
    def interrupt_first(plan, cursor, *, stop, step, observe, before_step=None):
        def observe_and_stop(progress, metrics):
            observe(progress, metrics)
            stop.request('continuation interruption')
        return real_run_updates(plan, cursor, stop=stop, step=step, observe=observe_and_stop,
                                before_step=before_step)
    interrupted = copy.copy(continued)
    interrupted.output = tmp_path / 'continued-interrupted'
    monkeypatch.setattr(scalable_training, 'run_epoch_updates', interrupt_first)
    assert scalable_training.run_scalable_arm(interrupted) == 130
    monkeypatch.setattr(scalable_training, 'run_epoch_updates', real_run_updates)
    resumed = copy.copy(continued)
    resumed.output = tmp_path / 'continued-resumed'
    resumed.warm_start = None
    resumed.resume = _checkpoint_for_report(interrupted.output, 'last_checkpoint_relative')
    assert scalable_training.run_scalable_arm(resumed) == 0
    expected = _checkpoint_for_report(continued.output, 'last_checkpoint_relative')
    actual = _checkpoint_for_report(resumed.output, 'last_checkpoint_relative')
    for filename in ['head.pt','backbone.pt','optimizer.pt','scheduler.pt','torch_rng.pt']:
        _assert_tree_equal(torch.load(expected / filename, weights_only=True),
                           torch.load(actual / filename, weights_only=True))
