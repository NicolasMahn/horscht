"""Launch-gated minibatch training with shuffled epochs and resumable stops."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiment import CANDIDATES, DecisionHead, format_input, last_token
from paired_training import (
    ROOT, available_memory_bytes, choose_pilot_rows, device_memory, ordered_row, peak_rss_gib,
    read_jsonl, require_memory, require_training_snapshot, resume_best_checkpoint, sha256, summarize_logits,
    synchronize, tensors_for,
)
from probability_experiment import soft_cross_entropy
from training_checkpoint import load_training_checkpoint, save_training_checkpoint
from training_loop import (
    EpochPlan, FeatureCache, StatusPublisher, StopController, TrainingCursor,
    atomic_json, build_epoch_plan, development_plateau, run_epoch_updates,
    wait_for_resources, warmup_cosine_scheduler,
)


def validate_system_test_lock(path: Path, snapshot: Path) -> dict[str, Any]:
    lock = json.loads(path.read_text())
    if lock.get("schema_version") != "system-test-lock-v1" or lock.get("status") != "locked":
        raise ValueError("A locked independent system-test manifest is required before training")
    if lock.get("independent_from_training_generator") is not True:
        raise ValueError("System-test independence must be declared and reviewed")
    exclusions = lock.get("training_exclusions", {})
    if not isinstance(exclusions, dict):
        raise ValueError("System-test training exclusions must be a mapping")
    for kind, identifiers in exclusions.items():
        if kind not in {"row_ids", "problem_group_ids", "snapshot_group_ids", "family_ids", "operation_ids"}:
            raise ValueError(f"Unknown system-test exclusion kind: {kind}")
        if not isinstance(identifiers, list) or any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError(f"System-test exclusions must contain lists of nonempty IDs: {kind}")
    if lock.get("training_snapshot_manifest_sha256") != sha256(snapshot / "manifest.json"):
        raise ValueError("System-test exclusions do not belong to this training snapshot")
    timestamp = datetime.fromisoformat(lock["locked_at"].replace("Z", "+00:00"))
    if timestamp.tzinfo is None:
        raise ValueError("The system-test lock timestamp must include its timezone")
    files = lock.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("The system-test suite and reference artifacts must have locked hashes")
    for filename, expected in files.items():
        relative = Path(filename)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid system-test artifact path: {filename}")
        artifact = path.parent / relative
        if artifact.is_symlink() or not artifact.resolve().is_relative_to(path.parent.resolve()):
            raise ValueError(f"System-test artifact escapes its locked directory: {filename}")
        if not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"System-test checksum mismatch: {filename}")
    if lock.get("snapshot_binding") is not None or lock.get("overlap_audit") is not None:
        from evaluation.snapshot_bound_protocol import validate_snapshot_bound_lock

        validate_snapshot_bound_lock(lock, snapshot, path)
    return lock


def model_state_sha256(module):
    import hashlib

    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def configure_backbone_training(backbone, arm):
    """Limit autograd to the selected scope; a frozen prefix builds no graph."""
    if arm not in {"head", "full", "last-block"}:
        raise ValueError(f"Unknown training arm: {arm}")
    backbone.requires_grad_(arm == "full")
    if arm == "last-block":
        if not hasattr(backbone, "layers") or not backbone.layers:
            raise ValueError("Last-block training requires a decoder with explicit layers")
        backbone.layers[-1].requires_grad_(True)
        prefix = f"layers.{len(backbone.layers) - 1}."
        expected = {name for name, _ in backbone.named_parameters() if name.startswith(prefix)}
        actual = {name for name, parameter in backbone.named_parameters() if parameter.requires_grad}
        if not expected or actual != expected:
            raise ValueError("Last-block trainable parameter allowlist mismatch")
    if arm == "full":
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        backbone.eval()
    return [name for name, parameter in backbone.named_parameters() if parameter.requires_grad]


def checkpoint_space_bytes(backbone, head, optimizer, include_backbone, backbone_parameter_names=None):
    def tensor_bytes(value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        if isinstance(value, dict):
            return sum(map(tensor_bytes, value.values()))
        if isinstance(value, (list, tuple)):
            return sum(map(tensor_bytes, value))
        return 0

    states = [head.state_dict(), optimizer.state_dict()]
    if include_backbone:
        state = backbone.state_dict()
        states.append({name:state[name] for name in backbone_parameter_names} if backbone_parameter_names is not None else state)
    return sum(map(tensor_bytes, states)) + 64 * 1024**2


def excluded_by_protocol(row: dict[str, Any], lock: dict[str, Any]) -> bool:
    excluded = lock.get("training_exclusions", {})
    snapshot = row.get("_training_snapshot", {})
    return (
        row["id"] in excluded.get("row_ids", [])
        or row.get("problem_group_id") in excluded.get("problem_group_ids", [])
        or snapshot.get("snapshot_group_id") in excluded.get("snapshot_group_ids", [])
        or row.get("family") in excluded.get("family_ids", [])
        or row.get("operation_id") in excluded.get("operation_ids", [])
    )


def with_linked_exclusions(lock, rows):
    """A source-row exclusion also excludes all of its linked variants."""
    linked = set(lock.get("training_exclusions", {}).get("snapshot_group_ids", []))
    for row in rows:
        if excluded_by_protocol(row, lock):
            group = row.get("_training_snapshot", {}).get("snapshot_group_id")
            if group:
                linked.add(group)
    return {**lock, "training_exclusions": {**lock.get("training_exclusions", {}),
                                            "snapshot_group_ids": sorted(linked)}}


def ordered_lengths(rows, tokenizer, seed, epoch, max_tokens):
    result = {}
    for offset in range(0, len(rows), 256):
        batch = [ordered_row(row, seed, epoch) for row in rows[offset:offset + 256]]
        tokens = tokenizer([format_input(row) for row in batch], add_special_tokens=True, truncation=False)
        for row, identifiers in zip(batch, tokens["input_ids"]):
            if len(identifiers) > max_tokens:
                raise ValueError(f"{row['id']}: complete shuffled input has {len(identifiers)} tokens; refusing truncation")
            result[row["id"]] = len(identifiers)
    return result


def prepare_full_selection(args) -> int:
    from transformers import AutoTokenizer
    from paired_training import tokenize_snapshot

    require_training_snapshot(args.snapshot)
    lock = validate_system_test_lock(args.system_test_manifest, args.snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = CANDIDATES[args.model]
    tokenizer = AutoTokenizer.from_pretrained(candidate["repo"], revision=candidate["revision"], local_files_only=True)
    rows, lengths = tokenize_snapshot(args.snapshot, args.output, tokenizer)
    lock = with_linked_exclusions(lock, rows)
    splits = {entry["id"]: entry for entry in read_jsonl(args.snapshot / "splits.jsonl")}
    selected = {"train": [], "dev": [], "calibration": [], "test": []}
    exclusions = []
    for row in rows:
        split = splits[row["id"]]["split"]
        if split in {"train", "dev"} and excluded_by_protocol(row, lock):
            exclusions.append({"id": row["id"], "split": split, "reason": "independent system-test family/group exclusion"})
            continue
        if lengths[row["id"]] > args.max_tokens:
            raise ValueError(f"{row['id']}: complete input exceeds {args.max_tokens}; full selection refuses silent omissions")
        selected[split].append(row["id"])
    if not selected["train"] or not selected["dev"]:
        raise ValueError("Protocol exclusions leave no separate training/development partition")
    atomic_json(args.output / "selection.json", {
        "schema_version": 2, "scope": "full eligible corpus after locked system-family exclusions",
        "model": args.model, "seed": args.seed, "max_tokens": args.max_tokens,
        "snapshot_manifest_sha256": sha256(args.snapshot / "manifest.json"),
        "system_test_manifest_sha256": sha256(args.system_test_manifest), "ids_by_split": selected,
        "training_questions_per_epoch": len(selected["train"]),
        "unshuffled_training_tokens": sum(lengths[identifier] for identifier in selected["train"]),
        "excluded_by_system_protocol": exclusions,
        "quality_scope": "structural eligibility; known review-recommended cohort retained and reported; no semantic verification claim",
    })
    print(json.dumps({"questions_by_split": {split: len(identifiers) for split, identifiers in selected.items()},
                      "protocol_exclusions": len(exclusions)}), flush=True)
    return 0


def encode_batch(tokenizer, backbone, rows, max_tokens, device, cache=None, head_architecture="last-token-linear-v1"):
    from attention_head import ARCHITECTURE, token_span_masks

    attention_head = head_architecture == ARCHITECTURE
    if attention_head and cache is not None:
        raise ValueError("Token-state attention requires an uncached backbone forward")
    prompts = [format_input(row) for row in rows]
    tokens = tokenizer(prompts, return_tensors="pt", add_special_tokens=True,
                       truncation=False, padding=True, **({"return_offsets_mapping": True} if attention_head else {}))
    mask = tokens.get("attention_mask", torch.ones_like(tokens["input_ids"]))
    lengths = mask.sum(-1).tolist()
    for row, length in zip(rows, lengths):
        if length > max_tokens:
            raise ValueError(f"{row['id']}: complete input has {length} tokens; refusing truncation")
    key = cache.key(prompts, tokens["input_ids"].shape[1]) if cache is not None else None
    if cache is not None:
        found = cache.get(key, device)
        if found is not None:
            return found
        backbone.eval()
        with torch.no_grad():
            features = _forward_batch(backbone, tokens["input_ids"], mask, device)
        cache.put(key, features)
        return features
    if attention_head:
        context_mask, option_mask = token_span_masks(rows, tokens["offset_mapping"], mask)
        hidden = _forward_batch(backbone, tokens["input_ids"], mask, device, all_tokens=True)
        return hidden, context_mask.to(device), option_mask.to(device)
    return _forward_batch(backbone, tokens["input_ids"], mask, device)


def _forward_batch(backbone, input_ids, mask, device, all_tokens=False):
    input_ids, mask = input_ids.to(device), mask.to(device)
    backend = getattr(backbone, "_horscht_attention_backend", "auto")
    if backend == "flash":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    else:
        attention = nullcontext()
    with attention, torch.autocast(device, dtype=torch.bfloat16, cache_enabled=False):
        hidden = backbone(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
    return hidden.float() if all_tokens else last_token(hidden, mask).float()


def evaluate_batches(tokenizer, backbone, head, rows, plan, args, cache, return_predictions=False, return_logits=False, stop=None):
    rows_by_id = {row["id"]: row for row in rows}
    ordered, logits = [], []
    next_progress = 256
    backbone.eval()
    head.eval()
    with torch.inference_mode():
        for update in plan.updates:
            for identifiers in update:
                if stop is not None and stop.requested:
                    raise InterruptedError("Deferred development scoring stopped between complete batches")
                batch = [rows_by_id[identifier] for identifier in identifiers]
                features = encode_batch(tokenizer, backbone, batch, args.max_tokens, args.device, cache,
                                        getattr(args, "head_architecture", "last-token-linear-v1"))
                counts = torch.tensor([len(row["options"]) for row in batch], device=args.device)
                logits.append(head(features, counts).cpu())
                ordered.extend(batch)
                if len(ordered) >= next_progress:
                    print(json.dumps({"phase": getattr(args, "evaluation_phase", "development evaluation"), "questions_done": len(ordered),
                                      "questions": len(rows)}), flush=True)
                    next_progress += 256
    merged = torch.cat(logits)
    summary = summarize_logits(merged, ordered)
    if return_logits:
        return summary, ordered, merged
    if return_predictions:
        return summary, {row["id"]: (row["choice_ids"], merged[index, :len(row["options"])].softmax(-1))
                         for index, row in enumerate(ordered)}
    return summary


def write_curves(path, history):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(rows=1, cols=2, subplot_titles=("Training loss", "Development cross-entropy"))
    measured = [point for point in history if "train_loss" in point]
    development = [point for point in history if "dev" in point]
    figure.add_trace(go.Scatter(x=[point["questions_seen"] for point in measured],
                               y=[point["train_loss"] for point in measured], name="Training"), row=1, col=1)
    figure.add_trace(go.Scatter(x=[point["questions_seen"] for point in development],
                               y=[point["dev"]["soft_cross_entropy"] for point in development],
                               mode="lines+markers", name="Development"), row=1, col=2)
    figure.update_xaxes(title_text="Original question exposures")
    figure.update_layout(height=550)
    temporary = path.with_suffix(".tmp.html")
    figure.write_html(temporary, include_plotlyjs=True)
    os.replace(temporary, path)


def compare_training_arms(arms, reports, manifests, output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    for key in ["schema_version", "repo", "revision", "selection_sha256", "system_test_manifest_sha256",
                "seed", "max_tokens", "sampler", "microbatch_questions", "accumulation_questions",
                "max_batch_tokens", "bucket_questions", "head_learning_rate", "device", "torch",
                "transformers", "parameters_dtype", "compute_autocast", "initial_head_state_sha256",
                "attention_backend"]:
        if manifests["head"].get(key) != manifests["full"].get(key):
            raise ValueError(f"Arms are not matched: {key}")
    for key in ["training_questions", "training_tokens"]:
        if reports["head"][key] != reports["full"][key]:
            raise ValueError(f"Arms have different exposure: {key}")
    # A plateau stop on the final update leaves the epoch cursor at its end;
    # normal completion advances it. Both have the same optimizer exposure.
    if reports["head"]["cursor"].get("global_update") != reports["full"]["cursor"].get("global_update"):
        raise ValueError("Arms have different exposure: optimizer updates")
    if any(report["status"] != "completed" for report in reports.values()):
        raise ValueError("Both matched training exposures must complete before comparison")
    history = {name: json.loads((directory / "history.json").read_text()) for name, directory in arms.items()}
    comparison = {
        "scope": "matched training/development comparison; final held-out and independent system tests not evaluated",
        "matched_questions": reports["head"]["training_questions"],
        "matched_tokens": reports["head"]["training_tokens"],
        "cursor": reports["head"]["cursor"],
        "selection_sha256": manifests["head"]["selection_sha256"],
        "system_test_manifest_sha256": manifests["head"]["system_test_manifest_sha256"],
        "initial_head_state_sha256": manifests["head"]["initial_head_state_sha256"],
        "arms": reports,
    }
    atomic_json(output / "comparison.json", comparison)
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Development cross-entropy", "Measured cost"))
    for name, points in history.items():
        measured = [point for point in points if "dev" in point]
        figure.add_trace(go.Scatter(x=[point["questions_seen"] for point in measured],
                                   y=[point["dev"]["soft_cross_entropy"] for point in measured],
                                   mode="lines+markers", name=name), row=1, col=1)
    figure.add_trace(go.Bar(x=list(arms), y=[reports[name]["total_seconds"] for name in arms],
                           name="Elapsed seconds"), row=1, col=2)
    figure.update_xaxes(title_text="Original question exposures", row=1, col=1)
    figure.update_yaxes(title_text="Seconds", row=1, col=2)
    figure.update_layout(height=550)
    output.mkdir(parents=True, exist_ok=True)
    figure.write_html(output / "comparison.html", include_plotlyjs=True)
    return 0


LIFECYCLE_HASH_FIELDS = {"runner_sha256", "loop_sha256", "checkpoint_sha256", "input_sha256"}


def validate_resume_lifecycle_audit(path, checkpoint, previous, current):
    """Permit only the reviewed resource implementation for one exact checkpoint."""
    if path is None:
        return None
    if path.is_symlink():
        raise ValueError("Lifecycle audit cannot be a symlink")
    audit = json.loads(path.read_text())
    changes = audit.get("source_changes")
    sources = audit.get("previous_source_files")
    if (audit.get("schema_version") != "training-resource-lifecycle-audit-v1"
            or audit.get("purpose") not in {"resource-lifecycle-only", "development-evaluation-deferral"}
            or audit.get("checkpoint_manifest_sha256") != sha256(checkpoint / "checkpoint.json")
            or audit.get("previous_run_manifest_sha256") != sha256(checkpoint.parent.parent / "manifest.json")
            or not isinstance(changes, dict) or not changes
            or set(changes) - LIFECYCLE_HASH_FIELDS
            or not isinstance(sources, dict) or set(sources) != set(changes)):
        raise ValueError("Lifecycle audit is not bound to this checkpoint and allowed implementation fields")
    for field, transition in changes.items():
        if (not isinstance(transition, dict) or set(transition) != {"previous", "current"}
                or transition["previous"] != previous.get(field)
                or transition["current"] != current.get(field)):
            raise ValueError(f"Lifecycle audit source mismatch: {field}")
        relative = Path(sources[field])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Lifecycle audit source must stay within the repository")
        source = ROOT / relative
        if (source.is_symlink() or not source.is_file()
                or not source.resolve().is_relative_to(ROOT.resolve())
                or sha256(source) != transition["previous"]):
            raise ValueError(f"Lifecycle audit frozen source checksum mismatch: {field}")
    if audit["purpose"] == "development-evaluation-deferral":
        policy = audit.get("development_execution")
        if (not isinstance(policy, dict) or policy != current.get("development_execution")
                or policy.get("timing") != "after all weight updates"
                or current.get("early_stop_patience") != 0
                or previous.get("early_stop_patience") != 0
                or previous.get("development_execution") not in (None, policy)):
            raise ValueError("Development deferral audit does not bind the explicit execution policy")
    return {"path": str(path), "sha256": sha256(path), "record": audit}


def reuse_resume_development_state(checkpoint, restored, audit, development_protocol, selection_path):
    """Bind saved development state to a nonzero exact optimizer boundary."""
    binding = audit["record"].get("development_state_reuse") if audit is not None else None
    if not isinstance(binding, dict):
        raise ValueError("Deferred development requires an audited saved development state")
    source = checkpoint.parent.parent
    required = {"baselines.json", "history.json", "development-protocol.json", "manifest.json"}
    if restored["manifest"].get("development_execution") is not None:
        required.add("deferred-candidates.json")
    if set(binding.get("artifacts", {})) != required:
        raise ValueError("Deferred resume needs all saved development artifacts")
    for name, expected in binding["artifacts"].items():
        artifact = source / name
        if artifact.is_symlink() or not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"Deferred resume saved artifact checksum mismatch: {name}")
    protocol_relative = Path(binding["protocol_path"])
    if protocol_relative.is_absolute() or ".." in protocol_relative.parts:
        raise ValueError("Deferred resume protocol path escapes the project")
    protocol_path = ROOT / protocol_relative
    selection_hash = sha256(selection_path)
    if (protocol_path.is_symlink() or sha256(protocol_path) != binding["protocol_sha256"]
            or json.loads(protocol_path.read_text())["selection_sha256"] != selection_hash
            or binding["selection_sha256"] != selection_hash
            or restored["manifest"]["selection_sha256"] != selection_hash
            or json.loads((source / "manifest.json").read_text()) != restored["manifest"]
            or json.loads((source / "development-protocol.json").read_text()) != development_protocol):
        raise ValueError("Deferred resume frozen selection/development state changed")
    baselines = json.loads((source / "baselines.json").read_text())
    metrics = baselines.get("warm_start_development") or baselines.get("random_head_development")
    count = len(development_protocol["full_ids"])
    metric_fields = ["soft_cross_entropy", "kl_pq", "categorical_brier_sum", "argmax_agreement"]
    if (not isinstance(metrics, dict) or metrics.get("questions") != count
            or not count or any(type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key]) for key in metric_fields)
            or not isinstance(baselines.get("uniform_development"), dict)
            or baselines["uniform_development"].get("questions") != count
            or restored["progress"].get("best_update") != 0
            or restored["progress"].get("best_loss") != metrics["soft_cross_entropy"]):
        raise ValueError("Deferred resume has no complete baseline metrics")
    history = json.loads((source / "history.json").read_text())
    cursor = restored["progress"]["cursor"]
    history = [point for point in history if point.get("update", 0) <= cursor["global_update"]]
    if not history or history[0].get("update") != 0 or history[0].get("dev") != metrics:
        raise ValueError("Deferred resume baseline history mismatch")
    if cursor["global_update"] and history[-1].get("update") != cursor["global_update"]:
        raise ValueError("Deferred resume history does not reach the saved optimizer boundary")
    return {"baselines": baselines, "history": history, "binding": binding,
            "checkpoint_manifest_sha256": sha256(checkpoint / "checkpoint.json")}


def reuse_completed_initial_baseline(checkpoint, restored, audit, development_protocol, selection_path):
    """Reuse an audited update-zero baseline without consuming inference or RNG."""
    binding = audit["record"].get("baseline_reuse") if audit is not None else None
    if binding is None:
        return None
    source = checkpoint.parent.parent
    expected_cursor = {"epoch": 0, "global_update": 0, "next_update": 0,
                       "questions_seen": 0, "tokens_seen": 0}
    progress = restored["progress"]
    if (progress.get("cursor") != expected_cursor or progress.get("reason") != "initial baseline"
            or progress.get("best_update") != 0 or progress.get("epoch_plan_sha256") is not None):
        raise ValueError("Baseline reuse requires the exact completed initial baseline checkpoint")
    required = {"baselines.json", "history.json", "development-protocol.json", "manifest.json"}
    if not isinstance(binding, dict) or set(binding.get("artifacts", {})) != required:
        raise ValueError("Baseline reuse requires all saved metrics and development artifacts")
    for name, expected in binding["artifacts"].items():
        artifact = source / name
        if artifact.is_symlink() or not artifact.is_file() or sha256(artifact) != expected:
            raise ValueError(f"Baseline reuse artifact checksum mismatch: {name}")
    protocol_path = ROOT / binding["protocol_path"]
    if (Path(binding["protocol_path"]).is_absolute() or ".." in Path(binding["protocol_path"]).parts
            or protocol_path.is_symlink() or sha256(protocol_path) != binding["protocol_sha256"]):
        raise ValueError("Baseline reuse protocol binding mismatch")
    protocol = json.loads(protocol_path.read_text())
    selection_hash = sha256(selection_path)
    if (binding["selection_sha256"] != selection_hash or protocol["selection_sha256"] != selection_hash
            or restored["manifest"]["selection_sha256"] != selection_hash):
        raise ValueError("Baseline reuse selection binding mismatch")
    if json.loads((source / "manifest.json").read_text()) != restored["manifest"]:
        raise ValueError("Baseline reuse source manifest differs from the checkpoint")
    if json.loads((source / "development-protocol.json").read_text()) != development_protocol:
        raise ValueError("Baseline reuse full development protocol changed")
    baselines = json.loads((source / "baselines.json").read_text())
    metrics = baselines.get("warm_start_development") or baselines.get("random_head_development")
    count = len(development_protocol["full_ids"])
    metric_fields = ["soft_cross_entropy", "kl_pq", "categorical_brier_sum", "argmax_agreement"]
    if (not isinstance(metrics, dict) or metrics.get("questions") != count or not count
            or any(type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key])
                   for key in metric_fields)
            or not isinstance(baselines.get("uniform_development"), dict)
            or baselines["uniform_development"].get("questions") != count
            or metrics["soft_cross_entropy"] != progress.get("best_loss")):
        raise ValueError("Baseline reuse requires complete finite full-development metrics")
    history = json.loads((source / "history.json").read_text())
    if (not isinstance(history, list) or len(history) != 1 or history[0].get("update") != 0
            or history[0].get("questions_seen", 0) != 0 or history[0].get("dev") != metrics):
        raise ValueError("Baseline reuse history does not match the saved full baseline")
    return {"baselines": baselines, "history": history, "binding": binding,
            "checkpoint_manifest_sha256": sha256(checkpoint / "checkpoint.json")}


def run_scalable_arm(args) -> int:
    args._created_output = False
    try:
        with StopController() as stop:
            return _run_scalable_arm(args, stop)
    except BaseException as exc:
        if args._created_output:
            status_path = args.output / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update({"schema_version": 1, "status": "failed", "training_running": False,
                           "message": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()})
            atomic_json(status_path, status)
            atomic_json(args.status_file, status)
        raise


def _run_scalable_arm(args, stop) -> int:
    import transformers
    from transformers import Adafactor, AutoConfig, AutoModel, AutoTokenizer

    if not args.launch_approved:
        raise ValueError("Training launch is held until the coordinating agent gives explicit go-ahead")
    if getattr(args, "resume_lifecycle_audit", None) is not None and args.resume is None:
        raise ValueError("A lifecycle audit requires the exact resume checkpoint")
    warm_start = getattr(args, "warm_start", None)
    if warm_start is not None and args.resume is not None:
        raise ValueError("Choose an exact resume or a fresh-optimizer warm start")
    resume_origin = (json.loads((args.resume / "checkpoint.json").read_text())["manifest"]
                     if args.resume is not None else None)
    require_training_snapshot(args.snapshot)
    lock = validate_system_test_lock(args.system_test_manifest, args.snapshot)
    selection = json.loads(args.selection.read_text())
    if selection.get("system_test_manifest_sha256") != sha256(args.system_test_manifest):
        raise ValueError("Training selection must be frozen against the independent system-test lock")
    if selection["snapshot_manifest_sha256"] != sha256(args.snapshot / "manifest.json"):
        raise ValueError("Training selection does not belong to this snapshot")
    if any(selection[key] != getattr(args, key) for key in ["model", "seed", "max_tokens"]):
        raise ValueError("Model, seed and complete-input limit must match the locked selection")
    snapshot_manifest = json.loads((args.snapshot / "manifest.json").read_text())
    for filename in ["rows.jsonl", "splits.jsonl"]:
        if sha256(args.snapshot / filename) != snapshot_manifest["artifacts"][filename]["sha256"]:
            raise ValueError(f"Snapshot checksum mismatch: {filename}")
    splits = {entry["id"]: entry for entry in read_jsonl(args.snapshot / "splits.jsonl")}
    for split in ["train", "dev"]:
        identifiers = selection["ids_by_split"][split]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Selection repeats original questions in {split}")
    selected_ids = {identifier for split in ["train", "dev"] for identifier in selection["ids_by_split"][split]}
    all_rows = read_jsonl(args.snapshot / "rows.jsonl")
    lock = with_linked_exclusions(lock, all_rows)
    source_rows = {row["id"]: row for row in all_rows if row["id"] in selected_ids}
    del all_rows
    train = [source_rows[identifier] for identifier in selection["ids_by_split"]["train"]]
    dev = [ordered_row(source_rows[identifier], args.seed, -1) for identifier in selection["ids_by_split"]["dev"]]
    if any(splits[row["id"]]["split"] != "train" or excluded_by_protocol(row, lock) for row in train):
        raise ValueError("Training rows cross a held-out partition or independent system-test family exclusion")
    if any(splits[row["id"]]["split"] != "dev" or excluded_by_protocol(row, lock) for row in dev):
        raise ValueError("Development rows cross a locked partition or independent system-test family exclusion")
    if not train or not dev:
        raise ValueError("Separate nonempty training and development sets are required")
    args.output.mkdir(parents=True, exist_ok=False)
    args._created_output = True
    out = args.output
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    publisher = StatusPublisher(args.status_file, out / "status.json", {
        "arm": args.arm, "output": str(out), "started_at": started_at,
        "total_epochs": args.epochs, "total_questions": len(train) * args.epochs,
        "launch_gate": "locked independent system test and explicit go-ahead recorded",
        "curves_path": str(out / "curves.html"),
    })
    cursor = TrainingCursor()
    if args.resume:
        cursor = TrainingCursor.from_state_dict(json.loads((args.resume / "checkpoint.json").read_text())["progress"]["cursor"])
    publisher.publish("preparing", cursor, phase="loading pinned model", message="Training has not started.")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("This environment has no available CUDA/HIP device")
        torch.cuda.reset_peak_memory_stats()
        _, total_memory = torch.cuda.mem_get_info()
        cap_gib = getattr(args, "gpu_allocation_cap_gib", 44)
        if cap_gib <= 0 or cap_gib * 1024**3 >= total_memory:
            raise ValueError("GPU allocation cap must be positive and below addressable memory")
        torch.cuda.set_per_process_memory_fraction(cap_gib * 1024**3 / total_memory)
    attention_backend = getattr(args, "attention_backend", "auto")
    if attention_backend == "flash" and args.device != "cuda":
        raise ValueError("The measured fused-attention path requires the CUDA/HIP device")
    if attention_backend == "flash":
        # Non-reentrant activation checkpoints replay during backward, after the
        # forward context exits. Enforce the measured fused path for that replay.
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_cudnn_sdp(False)
    candidate = CANDIDATES[args.model]
    tokenizer = AutoTokenizer.from_pretrained(candidate["repo"], revision=candidate["revision"], local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(candidate["repo"], revision=candidate["revision"], local_files_only=True)
    with torch.device("meta"):
        skeleton = AutoModel.from_config(config)
    parameter_bytes = sum(parameter.numel() for parameter in skeleton.parameters()) * 4
    del skeleton
    full = args.arm == "full"
    trained_backbone = args.arm != "head"
    def memory_status(snapshot, waiting, phase):
        post_training = phase in {"deferred candidate reload", "selected deferred checkpoint reload"}
        publisher.publish("preparing" if waiting or post_training else "running", cursor,
                          phase=f"waiting for memory: {phase}" if waiting else phase,
                          resource_memory=snapshot, memory_waiting=waiting,
                          message="Saved state is intact; memory checks consume no questions or updates.",
                          **({"weight_updates_complete": True} if post_training else {}))

    def await_memory(additional, phase, before_wait=None):
        return wait_for_resources(lambda: require_memory(additional, args.reserve_gib, args.device),
                                  stop=stop, release_cache=torch.cuda.empty_cache if args.device == "cuda" else None,
                                  before_wait=before_wait,
                                  publish=lambda snapshot, waiting: memory_status(snapshot, waiting, phase))

    if not await_memory(parameter_bytes * (2 if full else 1) + 6 * 1024**3, "loading pinned model"):
        publisher.publish("interrupted", cursor, phase="startup stopped", stop_reason=stop.reason)
        return 130
    load_options = {"device_map": "cuda"} if args.device == "cuda" else {}
    backbone = AutoModel.from_pretrained(candidate["repo"], revision=candidate["revision"], local_files_only=True,
                                         dtype=torch.float32, attn_implementation="sdpa", **load_options).to(args.device)
    backbone._horscht_attention_backend = attention_backend
    if stop.requested:
        publisher.publish("interrupted", cursor, phase="startup stopped", stop_reason=stop.reason)
        return 130
    trainable_backbone_names = configure_backbone_training(backbone, args.arm)
    torch.manual_seed(args.seed)
    from attention_head import ARCHITECTURE, ATTENTION_SIZE, make_head
    head_architecture = getattr(args, "head_architecture", "last-token-linear-v1")
    if head_architecture == ARCHITECTURE and args.arm != "last-block":
        raise ValueError("The attention experiment is restricted to last-block training")
    head = make_head(backbone.config.hidden_size, head_architecture).to(args.device)
    warm_start_provenance = resume_origin.get("warm_start") if resume_origin is not None else None
    if warm_start is not None:
        warm_record = json.loads((warm_start / "checkpoint.json").read_text())
        previous = warm_record["manifest"]
        for key, expected in {"arm": args.arm, "repo": candidate["repo"], "revision": candidate["revision"],
                              "capacity": 265, "head_architecture": head_architecture}.items():
            if previous.get(key, "last-token-linear-v1" if key == "head_architecture" else None) != expected:
                raise ValueError(f"Warm-start model configuration mismatch: {key}")
        load_training_checkpoint(warm_start, backbone=backbone, head=head, restore_rng=False)
        warm_start_provenance = {"checkpoint": str(warm_start), "checkpoint_manifest_sha256": sha256(warm_start / "checkpoint.json"),
                                 "backbone_sha256":warm_record["files"]["backbone.pt"], "head_sha256":warm_record["files"]["head.pt"],
                                 "mode": "backbone and head weights; fresh optimizer, scheduler, cursor and seeded RNG",
                                 "source_snapshot_manifest_sha256": previous["snapshot_manifest_sha256"]}
    groups = [{"params": head.parameters(), "lr": args.head_learning_rate}]
    compact_checkpoint = getattr(args, "compact_last_block_checkpoints", False)
    if compact_checkpoint and (args.arm != "last-block" or warm_start_provenance is None):
        raise ValueError("Compact checkpoints require a last-block continuation with an immutable warm-start base")
    if trained_backbone:
        groups.append({"params": [parameter for parameter in backbone.parameters() if parameter.requires_grad],
                       "lr": args.backbone_learning_rate})
    optimizer = Adafactor(groups, scale_parameter=False, relative_step=False, warmup_init=False,
                         beta1=None, weight_decay=0, clip_threshold=1.0)
    max_updates = getattr(args, "max_updates", 0)
    total_updates = math.ceil(len(train) / args.accumulation) * args.epochs
    if max_updates:
        total_updates = min(total_updates, max_updates)
    lr_schedule = getattr(args, "lr_schedule", "constant")
    warmup_fraction = getattr(args, "warmup_fraction", .03)
    scheduler = (warmup_cosine_scheduler(optimizer, total_updates=total_updates, warmup_fraction=warmup_fraction)
                 if lr_schedule == "warmup-cosine" else None)
    fast_dev_questions = min(getattr(args, "fast_dev_questions", 0), len(dev))
    full_dev_every = getattr(args, "full_dev_every", 0) or args.eval_every
    order_questions = min(getattr(args, "order_consistency_questions", 512), len(dev))
    defer_full_dev = getattr(args, "defer_full_development", False)
    if defer_full_dev and (getattr(args, "early_stop_patience", 0) or order_questions or args.epochs != 1 or max_updates):
        raise ValueError("Deferred development requires one complete epoch, no early stopping or order diagnostics")
    candidate_updates = sorted({0, *range(full_dev_every, total_updates + 1, full_dev_every), total_updates})
    manifest = {
        "schema_version": 2, "arm": args.arm, "model": args.model, **candidate,
        "started_at": started_at, "seed": args.seed, "epochs": args.epochs,
        "snapshot_manifest_sha256": sha256(args.snapshot / "manifest.json"),
        "selection_sha256": sha256(args.selection), "system_test_manifest_sha256": sha256(args.system_test_manifest),
        "sampler": "length-bucket-shuffle-v1", "microbatch_questions": args.microbatch,
        "accumulation_questions": args.accumulation, "max_batch_tokens": args.max_batch_tokens,
        "bucket_questions": args.bucket_questions, "max_tokens": args.max_tokens,
        "formatter": "plain-context-question-choices-v1", "pooling": "option-text mean; context/question token attention" if head_architecture == ARCHITECTURE else "last non-padding token", "capacity": 265,
        "head_architecture": head_architecture,
        "head_attention_size": ATTENTION_SIZE if head_architecture == ARCHITECTURE else None,
        "option_span_metadata": "fast-tokenizer offsets; any text overlap; context/question prefix keys; no prompt changes" if head_architecture == ARCHITECTURE else None,
        "device": args.device, "threads": args.threads,
        "reserve_gib": args.reserve_gib,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "parameters_dtype": "float32", "compute_autocast": "bfloat16; cache disabled",
        "optimizer": "Adafactor; factored FP32 second moments; beta1=None; fixed learning rates" if scheduler is None else "Adafactor; factored FP32 second moments; beta1=None; shared warmup/cosine multiplier",
        "lr_schedule": lr_schedule, "warmup_fraction": warmup_fraction if scheduler is not None else None,
        "scheduler_total_updates": total_updates if scheduler is not None else None,
        "max_updates": max_updates, "warm_start": warm_start_provenance,
        "checkpoint_backbone_state": "trainable-overlay-v1 with immutable warm-start reference" if compact_checkpoint else "full",
        "fast_dev_questions": fast_dev_questions, "full_dev_every": full_dev_every,
        "head_learning_rate": args.head_learning_rate, "backbone_learning_rate": args.backbone_learning_rate if trained_backbone else None,
        "gradient_checkpointing": full, "gradient_buffers": "retained and zeroed between updates",
        "attention_backend": attention_backend,
        "gpu_allocation_cap_gib": getattr(args, "gpu_allocation_cap_gib", 44) if args.device == "cuda" else None,
        "backbone_trainable": trained_backbone, "backbone_parameter_count": parameter_bytes // 4,
        "backbone_training_scope": args.arm,
        "trainable_backbone_parameter_names": trainable_backbone_names,
        "trainable_backbone_parameter_count": sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad),
        "head_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "initial_head_state_sha256": resume_origin["initial_head_state_sha256"] if resume_origin is not None else model_state_sha256(head),
        "selection_metric": "minimum development soft_cross_entropy (KL differs by constant target entropy); Brier and aligned order consistency are diagnostics",
        "target_policy": "raw soft target sum checked within 2e-6, then normalized at tensor boundary; hard labels one-hot",
        "early_stop_patience": getattr(args, "early_stop_patience", 0),
        "early_stop_min_delta": getattr(args, "early_stop_min_delta", 0.0001),
        "early_stop_min_epochs": getattr(args, "early_stop_min_epochs", 1),
        "choice_order_policy": "conservative reference detector: ambiguous letter/relative-order references retain original order; all other inputs permute aligned choices/targets/IDK",
        "order_consistency_questions": order_questions,
        "order_consistency_sampling": "seeded source/cardinality/answerability cohort round-robin; one question per linked development group",
        "runner_sha256": sha256(Path(__file__)), "loop_sha256": sha256(ROOT / "training_loop.py"),
        "checkpoint_sha256": sha256(ROOT / "training_checkpoint.py"),
        "input_sha256": sha256(ROOT / "paired_training.py"), "head_sha256": sha256(ROOT / "experiment.py"),
        "attention_head_sha256": sha256(ROOT / "attention_head.py"),
        "resource_lifecycle": "checkpoint-and-wait-v1; mmap checkpoint loading; cache-aware physical budget",
        "loss_sha256": sha256(ROOT / "probability_experiment.py"),
        "cuda_version": torch.version.cuda, "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
    }
    if defer_full_dev:
        manifest["output"] = str(out)
        manifest["development_execution"] = {
            "timing": "after all weight updates", "candidate_updates": candidate_updates,
            "selection": "minimum full-development soft cross entropy; earliest update breaks ties",
            "baseline": "reuse audited complete initial development metrics",
            "selected_reload": "verified artifact checksums and tensor load; no duplicate full-development scoring",
            "fast_monitoring": "unchanged; never selects checkpoints",
        }
    atomic_json(out / "manifest.json", manifest)
    cache = None if trained_backbone else FeatureCache(json.dumps({key: manifest[key] for key in
                                                      ["repo", "revision", "formatter", "pooling", "device", "torch", "transformers", "attention_backend"]}, sort_keys=True),
                                          args.feature_cache_mib * 1024**2)
    dev_lengths = ordered_lengths([source_rows[identifier] for identifier in selection["ids_by_split"]["dev"]],
                                  tokenizer, args.seed, -1, args.max_tokens)
    dev_plan = build_epoch_plan(dev_lengths, seed=args.seed, epoch=-1, microbatch=args.microbatch,
                               accumulation_questions=args.accumulation, max_batch_tokens=args.max_batch_tokens,
                               bucket_questions=args.bucket_questions)
    fast_cohort, fast_policy = choose_pilot_rows(
        [source_rows[identifier] for identifier in selection["ids_by_split"]["dev"]],
        list(splits.values()), dev_lengths, counts={"dev": fast_dev_questions}, max_tokens=args.max_tokens, seed=args.seed)
    fast_dev = [ordered_row(row, args.seed, -1) for row in fast_cohort["dev"]]
    fast_plan = build_epoch_plan({row["id"]: dev_lengths[row["id"]] for row in fast_dev}, seed=args.seed, epoch=-1,
                                microbatch=args.microbatch, accumulation_questions=args.accumulation,
                                max_batch_tokens=args.max_batch_tokens, bucket_questions=args.bucket_questions) if fast_dev else None
    atomic_json(out / "development-protocol.json", {"fast_ids": [row["id"] for row in fast_dev],
                "fast_sampling": fast_policy, "fast_every_updates": args.eval_every,
                "full_ids": selection["ids_by_split"]["dev"], "full_every_updates": full_dev_every,
                "selection": "full development soft cross entropy only; initial and final budget boundaries included"})
    order_cohort, _ = choose_pilot_rows(
        [source_rows[identifier] for identifier in selection["ids_by_split"]["dev"]],
        list(splits.values()), dev_lengths, counts={"dev": order_questions},
        max_tokens=args.max_tokens, seed=args.seed)
    order_sources = order_cohort["dev"]
    atomic_json(out / "order-consistency-dev-ids.json", [row["id"] for row in order_sources])
    def development_metrics():
        summary, first = evaluate_batches(tokenizer, backbone, head, dev, dev_plan, args, cache, True)
        if not order_sources:
            return summary
        second_rows = [ordered_row(row, args.seed, -3) for row in order_sources]
        second_lengths = ordered_lengths(order_sources, tokenizer, args.seed, -3, args.max_tokens)
        second_plan = build_epoch_plan(second_lengths, seed=args.seed, epoch=-3, microbatch=args.microbatch,
                                      accumulation_questions=args.accumulation, max_batch_tokens=args.max_batch_tokens,
                                      bucket_questions=args.bucket_questions)
        _, second = evaluate_batches(tokenizer, backbone, head, second_rows, second_plan, args, cache, True)
        distances, agreements = [], []
        for identifier, (ids, probabilities) in second.items():
            original_ids, original = first[identifier]
            aligned = probabilities[[ids.index(choice) for choice in original_ids]]
            distances.append(float((original - aligned).abs().sum() / 2))
            agreements.append(int(original.argmax() == aligned.argmax()))
        summary["order_consistency"] = {"questions": len(distances),
            "mean_aligned_total_variation": sum(distances) / len(distances),
            "stable_choice_argmax_fraction": sum(agreements) / len(agreements)}
        return summary

    history = []
    saved_paths = []
    pinned_paths = set()
    best_loss, best_update, best_path = None, 0, None
    elapsed_before_resume = 0.0
    active_plan: EpochPlan | None = None
    first_update = True

    def checkpoint(reason):
        nonlocal best_path
        destination = out / "checkpoints" / f"update-{cursor.global_update:08d}-epoch-{cursor.epoch:03d}-next-{cursor.next_update:06d}"
        if best_path is None:
            best_path = destination
        progress = {"cursor": cursor.state_dict(),
                    "epoch_plan_sha256": active_plan.sha256 if active_plan and active_plan.epoch == cursor.epoch else None,
                    "best_loss": best_loss, "best_update": best_update,
                    "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started,
                    "best_checkpoint": os.path.relpath(best_path, destination.parent), "reason": reason}
        if not destination.exists():
            compact_names = trainable_backbone_names if compact_checkpoint else None
            needed = checkpoint_space_bytes(backbone, head, optimizer, trained_backbone, compact_names) + 2 * 1024**3
            if shutil.disk_usage(out).free < needed:
                raise MemoryError(f"Checkpoint needs {needed / 1024**3:.2f} GiB of free disk including its reserve")
            save_training_checkpoint(destination, backbone=backbone, tokenizer=tokenizer, head=head,
                                     optimizer=optimizer, scheduler=scheduler, progress=progress, manifest=manifest, include_backbone=trained_backbone,
                                     backbone_parameter_names=compact_names,
                                     base_checkpoint=warm_start_provenance["checkpoint"] if compact_checkpoint else None)
            saved_paths.append(destination)
        atomic_json(out / "latest-checkpoint.json", {"path": os.path.relpath(destination, out), "reason": reason})
        atomic_json(out / "best-checkpoint.json", {"path": os.path.relpath(best_path, out), "loss": best_loss, "update": best_update})
        # Retention applies only to complete checkpoints this invocation created.
        protected = set(saved_paths[-args.retain_checkpoints:]) | {best_path} | pinned_paths
        for stale in list(saved_paths):
            if stale not in protected:
                if stale.parent.resolve() != (out / "checkpoints").resolve() or stale.is_symlink():
                    raise RuntimeError("Checkpoint retention refused an unexpected path")
                shutil.rmtree(stale)
                saved_paths.remove(stale)
        return destination

    if args.resume:
        restored = load_training_checkpoint(args.resume, backbone=backbone, head=head, optimizer=optimizer, scheduler=scheduler)
        previous = restored["manifest"]
        audit = validate_resume_lifecycle_audit(getattr(args, "resume_lifecycle_audit", None), args.resume, previous, manifest)
        for key in ["schema_version", "arm", "repo", "revision", "epochs", "selection_sha256", "system_test_manifest_sha256",
                    "seed", "sampler", "microbatch_questions", "accumulation_questions", "max_batch_tokens",
                    "bucket_questions", "max_tokens", "device", "threads", "torch", "transformers", "parameters_dtype",
                    "compute_autocast", "head_learning_rate", "backbone_learning_rate", "optimizer",
                    "formatter", "pooling", "capacity", "snapshot_manifest_sha256", "initial_head_state_sha256",
                    "runner_sha256", "loop_sha256", "checkpoint_sha256", "input_sha256", "head_sha256", "loss_sha256",
                    "cuda_version", "hip_version", "device_name", "attention_backend", "gpu_allocation_cap_gib",
                    "early_stop_patience", "early_stop_min_delta", "early_stop_min_epochs", "choice_order_policy",
                    "head_architecture", "head_attention_size", "option_span_metadata", "attention_head_sha256"]:
            if previous.get(key) != manifest[key]:
                if audit is not None and key in audit["record"]["source_changes"]:
                    continue
                raise ValueError(f"Exact training resume configuration mismatch: {key}")
        for key in ["lr_schedule", "warmup_fraction", "scheduler_total_updates", "max_updates", "warm_start",
                    "fast_dev_questions", "full_dev_every", "order_consistency_questions", "checkpoint_backbone_state"]:
            if key in previous and previous[key] != manifest[key]:
                raise ValueError(f"Exact training resume configuration mismatch: {key}")
        if audit is not None:
            manifest["resource_lifecycle_audit"] = audit
            atomic_json(out / "manifest.json", manifest)
        cursor = TrainingCursor.from_state_dict(restored["progress"]["cursor"])
        best_loss = restored["progress"]["best_loss"]
        best_update = restored["progress"]["best_update"]
        best_path = resume_best_checkpoint(restored["progress"], args.resume)
        elapsed_before_resume = restored["progress"].get("elapsed_seconds", 0.0)
        previous_history = args.resume.parent.parent / "history.json"
        if previous_history.is_file():
            history = [point for point in json.loads(previous_history.read_text())
                       if point.get("update", 0) <= cursor.global_update]
    if defer_full_dev and args.resume and (audit is None or audit["record"]["purpose"] != "development-evaluation-deferral"):
        raise ValueError("Deferred resume requires the explicit development deferral audit")
    reused = reuse_resume_development_state(
        args.resume, restored, audit, json.loads((out / "development-protocol.json").read_text()), args.selection
    ) if defer_full_dev and args.resume else reuse_completed_initial_baseline(
        args.resume, restored, audit, json.loads((out / "development-protocol.json").read_text()), args.selection
    ) if args.resume else None
    if reused is not None:
        atomic_json(out / "baselines.json", reused["baselines"])
        atomic_json(out / "baseline-reuse.json", {"schema_version": "exact-initial-baseline-reuse-v1",
                    "source_checkpoint": str(args.resume), **reused,
                    "inference_questions": 0, "rng_consumed": False})
        history = reused["history"]
        atomic_json(out / "latest-checkpoint.json", {"path": os.path.relpath(args.resume, out),
                    "reason": "audited completed baseline reused"})
        atomic_json(out / "best-checkpoint.json", {"path": os.path.relpath(best_path, out),
                    "loss": best_loss, "update": best_update})
        publisher.publish("preparing", cursor, phase="completed baseline restored",
                          message="Audited full-development baseline reused; no baseline inference repeated.")
    else:
        publisher.publish("preparing", cursor, phase="initial development baseline",
                          message="All development rows are scored before any learned update; progress is logged every 256 questions.")
        initial = development_metrics()
        atomic_json(out / "baselines.json", {
            "random_head_development": initial if not args.resume and warm_start is None else None,
            "warm_start_development": initial if warm_start is not None else None,
            "uniform_development": summarize_logits(torch.zeros(len(dev), 265), dev),
            "scope": "development only; final source-held-out and independent system tests are untouched",
        })
        history.append({"update": cursor.global_update, "questions_seen": cursor.questions_seen, "dev": initial})
        if best_loss is None:
            best_loss = initial["soft_cross_entropy"]
            checkpoint("initial baseline")
    atomic_json(out / "history.json", history)
    write_curves(out / "curves.html", history)
    resume_plan_hash = restored["progress"].get("epoch_plan_sha256") if args.resume else None
    last_checkpoint_time = time.monotonic()
    deferred_candidates = {}
    if defer_full_dev:
        previous_candidates = args.resume.parent.parent / "deferred-candidates.json" if args.resume else None
        if previous_candidates is not None and previous_candidates.is_file():
            deferred_candidates = json.loads(previous_candidates.read_text())
            for update, entry in deferred_candidates.items():
                path = ROOT / entry["path"]
                if int(update) not in candidate_updates or sha256(path / "checkpoint.json") != entry["checkpoint_manifest_sha256"]:
                    raise ValueError("Deferred selection candidate changed")
                pinned_paths.add(path)
        else:
            if cursor.global_update >= candidate_updates[1]:
                raise ValueError("Cannot defer an already-passed candidate without a saved pinned candidate index")
            deferred_candidates["0"] = {"path": best_path.resolve().relative_to(ROOT.resolve()).as_posix(),
                "checkpoint_manifest_sha256": sha256(best_path / "checkpoint.json"),
                "dev": history[0]["dev"], "score_origin": "audited completed baseline"}
            pinned_paths.add(best_path)
        atomic_json(out / "deferred-candidates.json", deferred_candidates)

    def step(batches):
        nonlocal first_update
        step_started = time.perf_counter()
        optimizer.zero_grad(set_to_none=False)
        backbone.train(full)
        head.train()
        questions = sum(map(len, batches))
        loss_sum = 0.0
        for identifiers in batches:
            batch = [ordered_row(source_rows[identifier], args.seed, cursor.epoch) for identifier in identifiers]
            features = encode_batch(tokenizer, backbone, batch, args.max_tokens, args.device, cache, head_architecture)
            targets, _, _, counts = tensors_for(batch, args.device)
            loss = soft_cross_entropy(head(features, counts), targets, counts)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite categorical training loss")
            (loss * len(batch) / questions).backward()
            loss_sum += loss.item() * len(batch)
        if trained_backbone and first_update:
            missing = [name for name, parameter in backbone.named_parameters() if parameter.requires_grad and parameter.grad is None]
            if missing:
                raise RuntimeError(f"Training did not reach all allowed backbone parameters: {missing}")
            unexpected = [name for name, parameter in backbone.named_parameters() if not parameter.requires_grad and parameter.grad is not None]
            if unexpected:
                raise RuntimeError(f"Frozen backbone parameters received gradients: {unexpected}")
            atomic_json(out / "first-gradient-proof.json", {
                "training_scope": args.arm, "trainable_names": trainable_backbone_names,
                "all_trainable_backbone_gradients_present": True, "frozen_backbone_gradients_absent": True,
                "head_gradients_present": all(parameter.grad is not None for parameter in head.parameters()),
            })
        first_update = False
        norm = torch.nn.utils.clip_grad_norm_([parameter for group in optimizer.param_groups for parameter in group["params"]],
                                             1.0, error_if_nonfinite=True)
        learning_rates = [group["lr"] for group in optimizer.param_groups]
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        synchronize(args.device)
        return {"train_loss": loss_sum / questions, "gradient_norm": float(norm), "learning_rates": learning_rates,
                "update_compute_seconds": time.perf_counter() - step_started,
                "peak_rss_gib": peak_rss_gib(), "available_ram_gib": available_memory_bytes() / 1024**3,
                **device_memory(args.device)}

    def before_step():
        missing_gradient_bytes = sum(parameter.numel() * parameter.element_size()
                                     for group in optimizer.param_groups for parameter in group["params"]
                                     if parameter.grad is None)

        def preserve_boundary(snapshot):
            publisher.publish("preparing", cursor, phase="saving checkpoint before memory wait",
                              resource_memory=snapshot, memory_waiting=True)
            atomic_json(out / "history.json", history)
            checkpoint("memory pressure before next update")

        return await_memory(missing_gradient_bytes + 6 * 1024**3, "training", preserve_boundary)

    with nullcontext(stop):
        try:
            while cursor.epoch < args.epochs and not stop.requested and (not max_updates or cursor.global_update < max_updates):
                lengths = ordered_lengths(train, tokenizer, args.seed, cursor.epoch, args.max_tokens)
                active_plan = build_epoch_plan(lengths, seed=args.seed, epoch=cursor.epoch,
                                               microbatch=args.microbatch, accumulation_questions=args.accumulation,
                                               max_batch_tokens=args.max_batch_tokens, bucket_questions=args.bucket_questions)
                if resume_plan_hash is not None and resume_plan_hash != active_plan.sha256:
                    raise ValueError("Resume sampler/permutation plan changed")
                resume_plan_hash = None
                atomic_json(out / f"epoch-{cursor.epoch:03d}-plan.json", {
                    "epoch": cursor.epoch, "sha256": active_plan.sha256,
                    "updates": active_plan.updates, "ordered_token_lengths": lengths,
                })

                def observe(progress, metrics):
                    nonlocal best_loss, best_update, best_path, last_checkpoint_time
                    point = {**progress.state_dict(), "update": progress.global_update,
                             "elapsed_seconds": elapsed_before_resume + time.perf_counter() - started, **metrics}
                    final_epoch_update = progress.next_update == len(active_plan.updates)
                    final_budget_update = bool(max_updates and progress.global_update >= max_updates)
                    improved = False
                    if fast_plan is not None and not stop.requested and progress.global_update % args.eval_every == 0:
                        point["fast_dev"] = evaluate_batches(tokenizer, backbone, head, fast_dev, fast_plan, args, cache)
                    if not defer_full_dev and not stop.requested and (progress.global_update % full_dev_every == 0 or final_epoch_update or final_budget_update):
                        point["dev"] = development_metrics()
                        if point["dev"]["soft_cross_entropy"] < best_loss:
                            best_loss = point["dev"]["soft_cross_entropy"]
                            best_update = progress.global_update
                            best_path = out / "checkpoints" / f"update-{progress.global_update:08d}-epoch-{progress.epoch:03d}-next-{progress.next_update:06d}"
                            improved = True
                        patience = getattr(args, "early_stop_patience", 0)
                        if patience and progress.questions_seen >= len(train) * getattr(args, "early_stop_min_epochs", 1):
                            evaluations = [entry["dev"]["soft_cross_entropy"] for entry in history if "dev" in entry]
                            evaluations.append(point["dev"]["soft_cross_entropy"])
                            if development_plateau(evaluations, patience=patience,
                                                   min_delta=getattr(args, "early_stop_min_delta", 0.0001)):
                                stop.request("development plateau")
                    history.append(point)
                    with (out / "updates.jsonl").open("a") as stream:
                        stream.write(json.dumps(point, allow_nan=False) + "\n")
                    publisher.publish("running", progress, phase="training", latest_metrics=metrics,
                                      message="Stop requests are saved after the current complete optimizer update.")
                    if "dev" in point:
                        write_curves(out / "curves.html", history)
                    pinned_candidate = defer_full_dev and progress.global_update in candidate_updates
                    if pinned_candidate:
                        pinned_paths.add(out / "checkpoints" / f"update-{progress.global_update:08d}-epoch-{progress.epoch:03d}-next-{progress.next_update:06d}")
                    if pinned_candidate or improved or stop.requested or final_epoch_update or progress.global_update % args.checkpoint_every == 0 or time.monotonic() - last_checkpoint_time >= args.checkpoint_seconds:
                        # Persist the complete resumable history at checkpoint boundaries;
                        # append-only update events avoid rewriting a growing file each step.
                        atomic_json(out / "history.json", history)
                        saved = checkpoint("deferred development candidate" if pinned_candidate else "stop requested" if stop.requested else "best development" if improved else "periodic")
                        if pinned_candidate:
                            deferred_candidates[str(progress.global_update)] = {
                                "path": saved.resolve().relative_to(ROOT.resolve()).as_posix(),
                                "checkpoint_manifest_sha256": sha256(saved / "checkpoint.json")}
                            atomic_json(out / "deferred-candidates.json", deferred_candidates)
                        last_checkpoint_time = time.monotonic()
                    print(json.dumps({"arm": args.arm, "epoch": progress.epoch + 1,
                                      "update": progress.global_update, "questions_seen": progress.questions_seen,
                                      "tokens_seen": progress.tokens_seen, **metrics}), flush=True)
                    if final_budget_update:
                        stop.request("completed development pilot budget")

                publisher.publish("running", cursor, phase="training", message="Training uses the locked selection and complete shuffled inputs.")
                if not run_epoch_updates(active_plan, cursor, stop=stop, step=step, observe=observe, before_step=before_step):
                    break
            early_stopped = stop.reason == "development plateau"
            pilot_completed = stop.reason == "completed development pilot budget"
            interrupted = stop.requested and cursor.epoch < args.epochs and not early_stopped and not pilot_completed
            last = checkpoint("interrupted" if interrupted else "completed training exposure")
            state = "interrupted" if interrupted else "completed"
            verified_dev = None
            if not interrupted and defer_full_dev:
                optimizer.zero_grad(set_to_none=True)
                if sorted(map(int, deferred_candidates)) != candidate_updates:
                    raise ValueError("Deferred development candidate schedule is incomplete")
                for update in candidate_updates:
                    entry = deferred_candidates[str(update)]
                    candidate = ROOT / entry["path"]
                    if sha256(candidate / "checkpoint.json") != entry["checkpoint_manifest_sha256"]:
                        raise ValueError("Pinned development checkpoint changed")
                    if "dev" not in entry:
                        publisher.publish("preparing", cursor, phase=f"post-training full development: update {update}",
                                          candidate_update=update, weight_updates_complete=True,
                                          message="All weight updates are complete; original pinned candidates are scored once before selection.")
                        if not await_memory(2 * 1024**3, "deferred candidate reload"):
                            interrupted, state = True, "interrupted"
                            break
                        load_training_checkpoint(candidate, backbone=backbone, head=head, restore_rng=False)
                        try:
                            entry["dev"] = evaluate_batches(tokenizer, backbone, head, dev, dev_plan, args, cache, stop=stop)
                        except InterruptedError:
                            interrupted, state = True, "interrupted"
                            break
                        entry["score_origin"] = "single post-training full-development pass"
                        atomic_json(out / "deferred-candidates.json", deferred_candidates)
                    if entry["dev"].get("questions") != len(dev) or not math.isfinite(entry["dev"]["soft_cross_entropy"]):
                        raise ValueError("Deferred candidate has incomplete development metrics")
                if not interrupted:
                    best_update = min(candidate_updates, key=lambda update: (deferred_candidates[str(update)]["dev"]["soft_cross_entropy"], update))
                    selected = deferred_candidates[str(best_update)]
                    best_loss = selected["dev"]["soft_cross_entropy"]
                    best_path = ROOT / selected["path"]
                    if not await_memory(2 * 1024**3, "selected deferred checkpoint reload"):
                        interrupted, state = True, "interrupted"
                    else:
                        load_training_checkpoint(best_path, backbone=backbone, head=head, restore_rng=False)
                        verified_dev = selected["dev"]
                        for update in candidate_updates:
                            for point in history:
                                if point.get("update") == update:
                                    point["dev"] = deferred_candidates[str(update)]["dev"]
                                    break
                        atomic_json(out / "history.json", history)
                        atomic_json(out / "best-checkpoint.json", {"path": os.path.relpath(best_path, out), "loss": best_loss, "update": best_update})
                        atomic_json(out / "deferred-selection.json", {"schema_version": "post-training-development-selection-v1",
                            "candidate_updates": candidate_updates, "best_update": best_update,
                            "best_dev_soft_cross_entropy": best_loss, "selected_checkpoint_manifest_sha256": selected["checkpoint_manifest_sha256"],
                            "candidate_index_sha256": sha256(out / "deferred-candidates.json"),
                            "selected_reload_verification": "artifact checksums and exact tensor load; stored single-pass development score reused",
                            "duplicate_selected_development_inference": False})
                        write_curves(out / "curves.html", history)
            if not interrupted and not defer_full_dev:
                # The resumable last state is already saved; release gradients before
                # loading the selected state on this shared-memory accelerator.
                optimizer.zero_grad(set_to_none=True)
                if early_stopped or pilot_completed:
                    stop.acknowledge()
                if not await_memory(2 * 1024**3, "selected checkpoint reload"):
                    state = "interrupted"
                    interrupted = True
                else:
                    load_training_checkpoint(best_path, backbone=backbone, head=head)
                    verified_dev = evaluate_batches(tokenizer, backbone, head, dev, dev_plan, args, cache)
                    if abs(verified_dev["soft_cross_entropy"] - best_loss) > 1e-5:
                        raise RuntimeError("Selected checkpoint did not reproduce its development score")
            atomic_json(out / "report.json", {
                "status": state, "arm": args.arm, "cursor": cursor.state_dict(),
                "training_questions": cursor.questions_seen, "training_tokens": cursor.tokens_seen,
                "best_dev_soft_cross_entropy": best_loss, "best_update": best_update,
                "best_checkpoint_relative": os.path.relpath(best_path, out),
                "last_checkpoint_relative": os.path.relpath(last, out),
                "final_test": "not evaluated by training; both final candidates must be frozen first",
                "total_seconds": elapsed_before_resume + time.perf_counter() - started, **device_memory(args.device),
                "peak_rss_gib": peak_rss_gib(),
                "selected_checkpoint_reload_verified": verified_dev is not None,
                "selected_checkpoint_development": verified_dev,
                "early_stopped": early_stopped, "stop_reason": stop.reason,
                **({"development_execution": manifest["development_execution"],
                    "deferred_selection_sha256": sha256(out / "deferred-selection.json") if not interrupted else None} if defer_full_dev else {}),
            })
            publisher.publish(state, cursor, phase="checkpoint saved", completed_at=datetime.now(timezone.utc).isoformat(),
                              message="Selected arm saved. Final held-out and independent system tests have not been evaluated.")
            return 130 if interrupted else 0
        except BaseException as exc:
            publisher.publish("failed", cursor, phase="training stopped", message=str(exc))
            raise
