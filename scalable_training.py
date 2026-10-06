"""Launch-gated minibatch training with shuffled epochs and resumable stops."""

from __future__ import annotations

import json
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


def checkpoint_space_bytes(backbone, head, optimizer, include_backbone):
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
        states.append(backbone.state_dict())
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
    selected = {"train": [], "dev": [], "test": []}
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


def encode_batch(tokenizer, backbone, rows, max_tokens, device, cache=None):
    prompts = [format_input(row) for row in rows]
    tokens = tokenizer(prompts, return_tensors="pt", add_special_tokens=True,
                       truncation=False, padding=True)
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
    return _forward_batch(backbone, tokens["input_ids"], mask, device)


def _forward_batch(backbone, input_ids, mask, device):
    input_ids, mask = input_ids.to(device), mask.to(device)
    backend = getattr(backbone, "_horscht_attention_backend", "auto")
    if backend == "flash":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        attention = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
    else:
        attention = nullcontext()
    with attention, torch.autocast(device, dtype=torch.bfloat16, cache_enabled=False):
        hidden = backbone(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
    return last_token(hidden, mask).float()


def evaluate_batches(tokenizer, backbone, head, rows, plan, args, cache, return_predictions=False):
    rows_by_id = {row["id"]: row for row in rows}
    ordered, logits = [], []
    next_progress = 256
    backbone.eval()
    head.eval()
    with torch.inference_mode():
        for update in plan.updates:
            for identifiers in update:
                batch = [rows_by_id[identifier] for identifier in identifiers]
                features = encode_batch(tokenizer, backbone, batch, args.max_tokens, args.device, cache)
                counts = torch.tensor([len(row["options"]) for row in batch], device=args.device)
                logits.append(head(features, counts).cpu())
                ordered.extend(batch)
                if len(ordered) >= next_progress:
                    print(json.dumps({"phase": "development evaluation", "questions_done": len(ordered),
                                      "questions": len(rows)}), flush=True)
                    next_progress += 256
    merged = torch.cat(logits)
    summary = summarize_logits(merged, ordered)
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


def run_scalable_arm(args) -> int:
    args._created_output = False
    try:
        return _run_scalable_arm(args)
    except BaseException as exc:
        if args._created_output:
            status_path = args.output / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            status.update({"schema_version": 1, "status": "failed", "training_running": False,
                           "message": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()})
            atomic_json(status_path, status)
            atomic_json(args.status_file, status)
        raise


def _run_scalable_arm(args) -> int:
    import transformers
    from transformers import Adafactor, AutoConfig, AutoModel, AutoTokenizer

    if not args.launch_approved:
        raise ValueError("Training launch is held until the coordinating agent gives explicit go-ahead")
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
    require_memory(parameter_bytes * (2 if full else 1) + 6 * 1024**3, args.reserve_gib, args.device)
    load_options = {"device_map": "cuda"} if args.device == "cuda" else {}
    backbone = AutoModel.from_pretrained(candidate["repo"], revision=candidate["revision"], local_files_only=True,
                                         dtype=torch.float32, attn_implementation="sdpa", **load_options).to(args.device)
    backbone._horscht_attention_backend = attention_backend
    trainable_backbone_names = configure_backbone_training(backbone, args.arm)
    torch.manual_seed(args.seed)
    head = DecisionHead(backbone.config.hidden_size).to(args.device)
    groups = [{"params": head.parameters(), "lr": args.head_learning_rate}]
    if trained_backbone:
        groups.append({"params": [parameter for parameter in backbone.parameters() if parameter.requires_grad],
                       "lr": args.backbone_learning_rate})
    optimizer = Adafactor(groups, scale_parameter=False, relative_step=False, warmup_init=False,
                         beta1=None, weight_decay=0, clip_threshold=1.0)
    manifest = {
        "schema_version": 2, "arm": args.arm, "model": args.model, **candidate,
        "started_at": started_at, "seed": args.seed, "epochs": args.epochs,
        "snapshot_manifest_sha256": sha256(args.snapshot / "manifest.json"),
        "selection_sha256": sha256(args.selection), "system_test_manifest_sha256": sha256(args.system_test_manifest),
        "sampler": "length-bucket-shuffle-v1", "microbatch_questions": args.microbatch,
        "accumulation_questions": args.accumulation, "max_batch_tokens": args.max_batch_tokens,
        "bucket_questions": args.bucket_questions, "max_tokens": args.max_tokens,
        "formatter": "plain-context-question-choices-v1", "pooling": "last non-padding token", "capacity": 265,
        "device": args.device, "threads": args.threads,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "parameters_dtype": "float32", "compute_autocast": "bfloat16; cache disabled",
        "optimizer": "Adafactor; factored FP32 second moments; beta1=None; fixed learning rates",
        "head_learning_rate": args.head_learning_rate, "backbone_learning_rate": args.backbone_learning_rate if trained_backbone else None,
        "gradient_checkpointing": full, "gradient_buffers": "retained and zeroed between updates",
        "attention_backend": attention_backend,
        "gpu_allocation_cap_gib": getattr(args, "gpu_allocation_cap_gib", 44) if args.device == "cuda" else None,
        "backbone_trainable": trained_backbone, "backbone_parameter_count": parameter_bytes // 4,
        "backbone_training_scope": args.arm,
        "trainable_backbone_parameter_names": trainable_backbone_names,
        "trainable_backbone_parameter_count": sum(parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad),
        "head_parameter_count": sum(parameter.numel() for parameter in head.parameters()),
        "initial_head_state_sha256": model_state_sha256(head),
        "selection_metric": "minimum development soft_cross_entropy (KL differs by constant target entropy); Brier and aligned order consistency are diagnostics",
        "early_stop_patience": getattr(args, "early_stop_patience", 0),
        "early_stop_min_delta": getattr(args, "early_stop_min_delta", 0.0001),
        "early_stop_min_epochs": getattr(args, "early_stop_min_epochs", 1),
        "choice_order_policy": "conservative reference detector: ambiguous letter/relative-order references retain original order; all other inputs permute aligned choices/targets/IDK",
        "order_consistency_questions": min(512, len(dev)),
        "order_consistency_sampling": "seeded source/cardinality/answerability cohort round-robin; one question per linked development group",
        "runner_sha256": sha256(Path(__file__)), "loop_sha256": sha256(ROOT / "training_loop.py"),
        "checkpoint_sha256": sha256(ROOT / "training_checkpoint.py"),
        "input_sha256": sha256(ROOT / "paired_training.py"), "head_sha256": sha256(ROOT / "experiment.py"),
        "loss_sha256": sha256(ROOT / "probability_experiment.py"),
        "cuda_version": torch.version.cuda, "hip_version": torch.version.hip,
        "device_name": torch.cuda.get_device_name() if args.device == "cuda" else "CPU",
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
    order_cohort, _ = choose_pilot_rows(
        [source_rows[identifier] for identifier in selection["ids_by_split"]["dev"]],
        list(splits.values()), dev_lengths, counts={"dev": min(512, len(dev))},
        max_tokens=args.max_tokens, seed=args.seed)
    order_sources = order_cohort["dev"]
    atomic_json(out / "order-consistency-dev-ids.json", [row["id"] for row in order_sources])
    def development_metrics():
        summary, first = evaluate_batches(tokenizer, backbone, head, dev, dev_plan, args, cache, True)
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
            needed = checkpoint_space_bytes(backbone, head, optimizer, trained_backbone) + 2 * 1024**3
            if shutil.disk_usage(out).free < needed:
                raise MemoryError(f"Checkpoint needs {needed / 1024**3:.2f} GiB of free disk including its reserve")
            save_training_checkpoint(destination, backbone=backbone, tokenizer=tokenizer, head=head,
                                     optimizer=optimizer, progress=progress, manifest=manifest, include_backbone=trained_backbone)
            saved_paths.append(destination)
        atomic_json(out / "latest-checkpoint.json", {"path": os.path.relpath(destination, out), "reason": reason})
        atomic_json(out / "best-checkpoint.json", {"path": os.path.relpath(best_path, out), "loss": best_loss, "update": best_update})
        # Retention applies only to complete checkpoints this invocation created.
        protected = set(saved_paths[-args.retain_checkpoints:]) | {best_path}
        for stale in list(saved_paths):
            if stale not in protected:
                if stale.parent.resolve() != (out / "checkpoints").resolve() or stale.is_symlink():
                    raise RuntimeError("Checkpoint retention refused an unexpected path")
                shutil.rmtree(stale)
                saved_paths.remove(stale)
        return destination

    if args.resume:
        restored = load_training_checkpoint(args.resume, backbone=backbone, head=head, optimizer=optimizer)
        previous = restored["manifest"]
        for key in ["schema_version", "arm", "repo", "revision", "selection_sha256", "system_test_manifest_sha256",
                    "seed", "sampler", "microbatch_questions", "accumulation_questions", "max_batch_tokens",
                    "bucket_questions", "max_tokens", "device", "threads", "torch", "transformers", "parameters_dtype",
                    "compute_autocast", "head_learning_rate", "backbone_learning_rate", "optimizer",
                    "formatter", "pooling", "capacity", "snapshot_manifest_sha256", "initial_head_state_sha256",
                    "runner_sha256", "loop_sha256", "checkpoint_sha256", "input_sha256", "head_sha256", "loss_sha256",
                    "cuda_version", "hip_version", "device_name", "attention_backend", "gpu_allocation_cap_gib",
                    "early_stop_patience", "early_stop_min_delta", "early_stop_min_epochs", "choice_order_policy"]:
            if previous.get(key) != manifest[key]:
                raise ValueError(f"Exact training resume configuration mismatch: {key}")
        cursor = TrainingCursor.from_state_dict(restored["progress"]["cursor"])
        best_loss = restored["progress"]["best_loss"]
        best_update = restored["progress"]["best_update"]
        best_path = resume_best_checkpoint(restored["progress"], args.resume)
        elapsed_before_resume = restored["progress"].get("elapsed_seconds", 0.0)
        previous_history = args.resume.parent.parent / "history.json"
        if previous_history.is_file():
            history = [point for point in json.loads(previous_history.read_text())
                       if point.get("update", 0) <= cursor.global_update]
    publisher.publish("preparing", cursor, phase="initial development baseline",
                      message="All development rows are scored before any learned update; progress is logged every 256 questions.")
    initial = development_metrics()
    atomic_json(out / "baselines.json", {
        "random_head_development": initial if not args.resume else None,
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

    def step(batches):
        nonlocal first_update
        step_started = time.perf_counter()
        missing_gradient_bytes = sum(parameter.numel() * parameter.element_size()
                                     for group in optimizer.param_groups for parameter in group["params"]
                                     if parameter.grad is None)
        require_memory(missing_gradient_bytes + 6 * 1024**3, args.reserve_gib, args.device)
        optimizer.zero_grad(set_to_none=False)
        backbone.train(full)
        head.train()
        questions = sum(map(len, batches))
        loss_sum = 0.0
        for identifiers in batches:
            batch = [ordered_row(source_rows[identifier], args.seed, cursor.epoch) for identifier in identifiers]
            features = encode_batch(tokenizer, backbone, batch, args.max_tokens, args.device, cache)
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
        optimizer.step()
        synchronize(args.device)
        return {"train_loss": loss_sum / questions, "gradient_norm": float(norm),
                "update_compute_seconds": time.perf_counter() - step_started,
                "peak_rss_gib": peak_rss_gib(), "available_ram_gib": available_memory_bytes() / 1024**3,
                **device_memory(args.device)}

    with StopController() as stop:
        try:
            while cursor.epoch < args.epochs and not stop.requested:
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
                    improved = False
                    if not stop.requested and (progress.global_update % args.eval_every == 0 or final_epoch_update):
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
                    if improved or stop.requested or final_epoch_update or progress.global_update % args.checkpoint_every == 0 or time.monotonic() - last_checkpoint_time >= args.checkpoint_seconds:
                        # Persist the complete resumable history at checkpoint boundaries;
                        # append-only update events avoid rewriting a growing file each step.
                        atomic_json(out / "history.json", history)
                        checkpoint("stop requested" if stop.requested else "best development" if improved else "periodic")
                        last_checkpoint_time = time.monotonic()
                    print(json.dumps({"arm": args.arm, "epoch": progress.epoch + 1,
                                      "update": progress.global_update, "questions_seen": progress.questions_seen,
                                      "tokens_seen": progress.tokens_seen, **metrics}), flush=True)

                publisher.publish("running", cursor, phase="training", message="Training uses the locked selection and complete shuffled inputs.")
                if not run_epoch_updates(active_plan, cursor, stop=stop, step=step, observe=observe):
                    break
            early_stopped = stop.reason == "development plateau"
            interrupted = stop.requested and cursor.epoch < args.epochs and not early_stopped
            last = checkpoint("interrupted" if interrupted else "completed training exposure")
            state = "interrupted" if interrupted else "completed"
            verified_dev = None
            if not interrupted:
                # The resumable last state is already saved; release gradients before
                # loading the selected state on this shared-memory accelerator.
                optimizer.zero_grad(set_to_none=True)
                require_memory((parameter_bytes if full else 0) + 2 * 1024**3, args.reserve_gib, args.device)
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
            })
            publisher.publish(state, cursor, phase="checkpoint saved", completed_at=datetime.now(timezone.utc).isoformat(),
                              message="Selected arm saved. Final held-out and independent system tests have not been evaluated.")
            return 130 if interrupted else 0
        except BaseException as exc:
            publisher.publish("failed", cursor, phase="training stopped", message=str(exc))
            raise
