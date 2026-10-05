"""Evaluate two frozen training arms on their selected source-held-out rows."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import torch

from experiment import CANDIDATES, DecisionHead
from dataset_factory import probability_v2
from probability_experiment import categorical_metrics, targets_for_rows
from paired_training import (
    ordered_row, require_memory, require_training_snapshot, sha256,
)
from scalable_training import build_epoch_plan, encode_batch, evaluate_batches, ordered_lengths
from training_checkpoint import load_training_checkpoint


OUTPUT_FILE = "evaluation.json"
ROOT = Path(__file__).resolve().parent
INTERNAL_RESERVE_GIB = 12
MODEL_BYTES_RESERVE = 6 * 1024**3
TEST_PERMUTATION_EPOCH = -2
DEV_PERMUTATION_EPOCHS = (-1, -3)
CONFIDENCE_BIN_EDGES = (0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_MATCHED_MANIFEST_FIELDS = (
    "schema_version", "model", "repo", "revision", "snapshot_manifest_sha256",
    "selection_sha256", "system_test_manifest_sha256", "seed", "max_tokens", "sampler",
    "microbatch_questions", "accumulation_questions", "max_batch_tokens", "bucket_questions",
    "device", "threads", "torch", "transformers", "parameters_dtype", "compute_autocast",
    "formatter", "pooling", "capacity", "attention_backend", "gpu_allocation_cap_gib",
    "head_learning_rate", "initial_head_state_sha256",
)
_CHECKPOINT_MATCH_FIELDS = (
    "schema_version", "model", "repo", "revision", "snapshot_manifest_sha256",
    "selection_sha256", "system_test_manifest_sha256", "seed", "max_tokens", "sampler",
    "microbatch_questions", "accumulation_questions", "max_batch_tokens", "bucket_questions",
    "device", "threads", "torch", "transformers", "parameters_dtype", "compute_autocast",
    "formatter", "pooling", "capacity", "attention_backend", "gpu_allocation_cap_gib",
    "head_learning_rate", "initial_head_state_sha256",
)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def _checkpoint_for_run(run_dir: Path, report: dict[str, Any]) -> tuple[Path, dict[str, Any], str]:
    relative = report.get("best_checkpoint_relative")
    if not isinstance(relative, str) or not relative:
        raise ValueError("Completed run has no development-selected checkpoint reference")
    relative_path = PurePosixPath(relative)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError("Development-selected checkpoint path must stay inside its run directory")
    run_dir = run_dir.resolve()
    referenced_dir = run_dir.joinpath(*relative_path.parts)
    current = run_dir
    for part in relative_path.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Development-selected checkpoint path cannot contain symlinks")
    checkpoint_dir = referenced_dir.resolve()
    if not checkpoint_dir.is_relative_to(run_dir) or not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Development-selected checkpoint is missing: {checkpoint_dir}")
    checkpoint_file = checkpoint_dir / "checkpoint.json"
    if checkpoint_file.is_symlink() or not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint manifest is missing: {checkpoint_file}")
    checkpoint = _read_json(checkpoint_file)
    if checkpoint.get("schema_version") != 1 or not isinstance(checkpoint.get("files"), dict):
        raise ValueError("Unsupported or incomplete selected checkpoint manifest")
    return checkpoint_dir, checkpoint, sha256(checkpoint_file)


def _verify_checkpoint_files(directory: Path, files: dict[str, str]) -> None:
    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Selected checkpoint contains a symlink: {path.relative_to(directory)}")
        if path.is_file() and path.name != "checkpoint.json":
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(files):
        raise ValueError("Selected checkpoint artifact list differs from its manifest")

    for relative, expected in files.items():
        artifact_path = PurePosixPath(relative)
        if artifact_path.is_absolute() or ".." in artifact_path.parts:
            raise ValueError(f"Invalid selected checkpoint artifact path: {relative}")
        path = directory.joinpath(*artifact_path.parts)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Selected checkpoint checksum mismatch: {relative}")


def _validate_runs(
    snapshot: Path,
    selection_path: Path,
    selection: dict[str, Any],
    arm_dirs: dict[str, Path],
    stage: str,
) -> dict[str, dict[str, Any]]:
    snapshot_hash = sha256(snapshot / "manifest.json")
    selection_hash = sha256(selection_path)
    if selection.get("snapshot_manifest_sha256") != snapshot_hash:
        raise ValueError("Frozen selection does not belong to this training snapshot")
    if selection.get("schema_version", 2) != 2:
        raise ValueError("Evaluation requires a version 2 frozen selection")

    ids_by_split = selection.get("ids_by_split")
    if not isinstance(ids_by_split, dict):
        raise ValueError("Frozen selection has no split ID mapping")
    split_ids: dict[str, list[str]] = {}
    for split in ("train", "dev", "test"):
        identifiers = ids_by_split.get(split)
        if not isinstance(identifiers, list) or not identifiers:
            raise ValueError(f"Frozen selection requires nonempty {split} IDs")
        if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
            raise ValueError(f"Frozen selection has an invalid {split} ID")
        if len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Frozen selection repeats an original question in {split}")
        split_ids[split] = identifiers
    if set(split_ids["train"]) & set(split_ids["dev"]):
        raise ValueError("Frozen training and development IDs overlap")
    if set(split_ids["train"]) & set(split_ids["test"]):
        raise ValueError("Frozen training and source-held-out test IDs overlap")
    if set(split_ids["dev"]) & set(split_ids["test"]):
        raise ValueError("Frozen development and source-held-out test IDs overlap")

    system_lock_hash = selection.get("system_test_manifest_sha256")
    if (not isinstance(system_lock_hash, str) or len(system_lock_hash) != 64
            or any(character not in "0123456789abcdef" for character in system_lock_hash)):
        raise ValueError("Frozen selection is not bound to an independent system-test lock")
    if selection.get("model") not in CANDIDATES:
        raise ValueError("Frozen selection names an unsupported pinned model")
    if type(selection.get("seed")) is not int or type(selection.get("max_tokens")) is not int:
        raise ValueError("Frozen selection is missing its seed or complete-input limit")

    result = {}
    for arm, run_dir in arm_dirs.items():
        report_path = run_dir / "report.json"
        manifest_path = run_dir / "manifest.json"
        if not report_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(f"The {arm} arm has no completed report and manifest")
        report, manifest = _read_json(report_path), _read_json(manifest_path)
        if manifest.get("schema_version") != 2:
            raise ValueError(f"The {arm} arm does not use the supported scalable-run schema")
        if type(manifest.get("epochs")) is not int or manifest["epochs"] < 1:
            raise ValueError(f"The {arm} arm has an invalid epoch budget")
        if report.get("status") != "completed":
            raise ValueError(f"The {arm} arm has not completed training")
        if report.get("arm") != arm or manifest.get("arm") != arm:
            raise ValueError(f"The run directory does not contain the declared {arm} arm")
        if report.get("selected_checkpoint_reload_verified") is not True:
            raise ValueError(f"The {arm} arm did not verify its development-selected checkpoint")
        selected_dev = report.get("selected_checkpoint_development")
        if not isinstance(selected_dev, dict) or not isinstance(selected_dev.get("soft_cross_entropy"), (int, float)):
            raise ValueError(f"The {arm} arm has no selected-checkpoint development score")
        if manifest.get("selection_sha256") != selection_hash:
            raise ValueError(f"The {arm} arm used a different frozen selection")
        if manifest.get("snapshot_manifest_sha256") != snapshot_hash:
            raise ValueError(f"The {arm} arm used a different training snapshot")
        if manifest.get("system_test_manifest_sha256") != selection["system_test_manifest_sha256"]:
            raise ValueError(f"The {arm} arm used a different independent system-test lock")
        if any(manifest.get(key) != selection.get(key) for key in ("model", "seed", "max_tokens")):
            raise ValueError(f"The {arm} arm does not match the frozen model, seed and input limit")
        if manifest.get("parameters_dtype") != "float32" or manifest.get("compute_autocast") != "bfloat16; cache disabled":
            raise ValueError("Evaluation supports the recorded FP32 parameter and BF16 autocast policy")
        if manifest.get("attention_backend") not in {"auto", "flash"}:
            raise ValueError("The run manifest must record its explicit attention backend")
        if manifest.get("device") not in {"cpu", "cuda"}:
            raise ValueError("The run manifest has an unsupported inference device")
        if manifest.get("attention_backend") == "flash" and manifest["device"] != "cuda":
            raise ValueError("The recorded flash attention path requires CUDA/HIP")
        if manifest.get("formatter") != "plain-context-question-choices-v1" or manifest.get("pooling") != "last non-padding token":
            raise ValueError("The selected arm uses an unsupported input formatter or pooling rule")
        if manifest.get("capacity") != 265:
            raise ValueError("The selected arm does not use the 265-choice decision head")
        cursor = report.get("cursor")
        if not isinstance(cursor, dict) or cursor.get("epoch") != manifest.get("epochs"):
            raise ValueError(f"The {arm} arm did not complete its recorded epoch budget")
        cursor_fields = ("epoch", "next_update", "global_update", "questions_seen", "tokens_seen")
        if any(type(cursor.get(field)) is not int or cursor[field] < 0 for field in cursor_fields):
            raise ValueError(f"The {arm} arm has an invalid completed-training cursor")
        if cursor["next_update"] != 0 or cursor["global_update"] < 1:
            raise ValueError(f"The {arm} arm did not finish on an optimizer-update boundary")
        if (cursor["questions_seen"] != report.get("training_questions")
                or cursor["tokens_seen"] != report.get("training_tokens")):
            raise ValueError(f"The {arm} arm report does not match its recorded training exposure")
        if any(type(report.get(field)) is not int or report[field] < 1
               for field in ("training_questions", "training_tokens")):
            raise ValueError(f"The {arm} arm has no recorded training exposure")

        checkpoint_dir, checkpoint, checkpoint_hash = _checkpoint_for_run(run_dir, report)
        _verify_checkpoint_files(checkpoint_dir, checkpoint["files"])
        checkpoint_manifest = checkpoint.get("manifest")
        if not isinstance(checkpoint_manifest, dict):
            raise ValueError(f"The {arm} selected checkpoint has no training manifest")
        if any(checkpoint_manifest.get(key) != manifest.get(key) for key in _CHECKPOINT_MATCH_FIELDS):
            raise ValueError(f"The {arm} selected checkpoint does not match its completed run")
        if checkpoint.get("include_backbone") is not (arm == "full"):
            raise ValueError(f"The {arm} selected checkpoint has the wrong backbone state policy")
        progress = checkpoint.get("progress", {})
        if progress.get("best_update") != report.get("best_update"):
            raise ValueError(f"The {arm} selected checkpoint does not match the report's selected update")
        if abs(float(progress.get("best_loss", float("nan"))) - float(selected_dev["soft_cross_entropy"])) > 1e-5:
            raise ValueError(f"The {arm} selected checkpoint does not match its development score")
        result[arm] = {
            "run_dir": run_dir,
            "report": report,
            "manifest": manifest,
            "checkpoint_dir": checkpoint_dir,
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_hash,
        }

    head_manifest = result["head"]["manifest"]
    full_manifest = result["full"]["manifest"]
    for key in _MATCHED_MANIFEST_FIELDS:
        if head_manifest.get(key) != full_manifest.get(key):
            raise ValueError(f"The two arms are not matched: {key}")
    for key in ("training_questions", "training_tokens", "cursor"):
        if result["head"]["report"].get(key) != result["full"]["report"].get(key):
            raise ValueError(f"The two arms have different exposure: {key}")

    selection_lengths = selection.get("ordered_lengths_by_epoch")
    if not isinstance(selection_lengths, dict):
        raise ValueError("Frozen selection has no recorded per-epoch input lengths")
    prepared_splits = (("dev", -1),) if stage == "dev" else (("test", TEST_PERMUTATION_EPOCH),)
    for split, epoch in prepared_splits:
        lengths = selection_lengths.get(str(epoch))
        identifiers = selection["ids_by_split"][split]
        if not isinstance(lengths, dict) or set(lengths) != set(identifiers):
            raise ValueError(f"Frozen selection has incomplete input lengths for {split} epoch {epoch}")
        if any(type(lengths[identifier]) is not int or not 0 < lengths[identifier] <= selection["max_tokens"]
               for identifier in identifiers):
            raise ValueError(f"Frozen selection has invalid input lengths for {split} epoch {epoch}")
    return result


def _read_selected_rows(snapshot: Path, ids: list[str], split_by_id: dict[str, dict[str, Any]], split: str):
    requested = set(ids)
    found: dict[str, dict[str, Any]] = {}
    with (snapshot / "rows.jsonl").open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            identifier = row.get("id")
            if identifier not in requested:
                continue
            if identifier in found:
                raise ValueError(f"Snapshot repeats selected {split} question {identifier!r}")
            if split_by_id[identifier].get("split") != split:
                raise ValueError(f"Selected {split} ID {identifier!r} belongs to another split")
            snapshot_metadata = row.get("_training_snapshot", {})
            if (snapshot_metadata.get("split") != split
                    or snapshot_metadata.get("snapshot_group_id") != split_by_id[identifier].get("snapshot_group_id")):
                raise ValueError(f"Selected {split} row {identifier!r} disagrees with frozen split metadata")
            errors = probability_v2.validate_v2_row(row)
            if errors:
                raise ValueError(f"Selected {split} row {identifier!r} is invalid: {', '.join(errors)}")
            found[identifier] = row
    if set(found) != requested:
        raise ValueError(f"Snapshot is missing selected {split} questions: {sorted(requested - set(found))[:5]}")
    return [found[identifier] for identifier in ids]


def _selected_split_metadata(snapshot: Path, selected_ids: set[str]) -> dict[str, dict[str, Any]]:
    found = {}
    with (snapshot / "splits.jsonl").open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            entry = json.loads(line)
            identifier = entry.get("id")
            if identifier in selected_ids:
                if identifier in found:
                    raise ValueError(f"Snapshot repeats split metadata for selected ID {identifier!r}")
                found[identifier] = entry
    if set(found) != selected_ids:
        raise ValueError("Snapshot split metadata is missing selected questions")
    return found


def _plan_for_rows(rows, tokenizer, manifest, epoch):
    lengths = ordered_lengths(rows, tokenizer, manifest["seed"], epoch, manifest["max_tokens"])
    plan = build_epoch_plan(
        lengths,
        seed=manifest["seed"],
        epoch=epoch,
        microbatch=manifest["microbatch_questions"],
        accumulation_questions=manifest["accumulation_questions"],
        max_batch_tokens=manifest["max_batch_tokens"],
        bucket_questions=manifest["bucket_questions"],
    )
    return lengths, plan


def _verify_prepared_lengths(selection, lengths, epoch):
    prepared = selection["ordered_lengths_by_epoch"].get(str(epoch))
    if not isinstance(prepared, dict):
        raise ValueError(f"Frozen selection has no input lengths for permutation epoch {epoch}")
    if prepared != lengths:
        raise ValueError(f"Retokenized inputs differ from the frozen selection at epoch {epoch}")


def _load_tokenizer_and_config(manifest):
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    if transformers.__version__ != manifest["transformers"]:
        raise ValueError("Installed Transformers version differs from the frozen training environment")
    if torch.__version__ != manifest["torch"]:
        raise ValueError("Installed PyTorch version differs from the frozen training environment")
    candidate = CANDIDATES[manifest["model"]]
    if candidate["repo"] != manifest["repo"] or candidate["revision"] != manifest["revision"]:
        raise ValueError("Run manifest does not match the pinned local model revision")
    tokenizer = AutoTokenizer.from_pretrained(candidate["repo"], revision=candidate["revision"], local_files_only=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained(candidate["repo"], revision=candidate["revision"], local_files_only=True)
    return tokenizer, config


def _parameter_bytes(AutoModel, config):
    with torch.device("meta"):
        skeleton = AutoModel.from_config(config)
    try:
        return sum(parameter.numel() * 4 for parameter in skeleton.parameters())
    finally:
        del skeleton


def _evaluate_logits(tokenizer, backbone, head, rows, plan, args):
    rows_by_id = {row["id"]: row for row in rows}
    logits_by_id = {}
    backbone.eval()
    head.eval()
    with torch.inference_mode():
        for update in plan.updates:
            for identifiers in update:
                batch = [rows_by_id[identifier] for identifier in identifiers]
                features = encode_batch(tokenizer, backbone, batch, args.max_tokens, args.device, cache=None)
                counts = torch.tensor([len(row["options"]) for row in batch], device=args.device)
                batch_logits = head(features, counts).float().cpu()
                for row, logits in zip(batch, batch_logits):
                    logits_by_id[row["id"]] = logits
    if set(logits_by_id) != set(rows_by_id):
        raise RuntimeError("Development inference did not return one logit vector per selected row")
    return torch.stack([logits_by_id[row["id"]] for row in rows])


def _metric_summary(rows, logits):
    targets, source_kinds, hard_labels = targets_for_rows(rows)
    counts = torch.tensor([len(row["options"]) for row in rows], dtype=torch.long)
    result = categorical_metrics(logits, targets, counts, source_kinds, hard_labels)
    hard_correct = sum(
        int(int(logits[index, :count].argmax().item()) == label)
        for index, (count, label) in enumerate(zip(counts.tolist(), hard_labels))
        if label is not None
    )
    result["hard_correct_count"] = hard_correct
    return result, targets, source_kinds, hard_labels, counts


def _mean(values):
    return sum(values) / len(values) if values else None


def _confidence_idk_summary(rows, logits, targets, hard_labels, counts):
    details = []
    for index, row in enumerate(rows):
        count = int(counts[index])
        probabilities = torch.softmax(logits[index, :count], dim=-1)
        prediction = int(probabilities.argmax().item())
        choice_ids = row["choice_ids"]
        abstention_id = row.get("abstention_choice_id")
        abstention_index = choice_ids.index(abstention_id) if abstention_id in choice_ids else None
        label = hard_labels[index]
        details.append({
            "confidence": float(probabilities.max().item()),
            "target_mass_at_prediction": float(targets[index, prediction].item()),
            "hard_correct": None if label is None else int(prediction == label),
            "prediction_is_abstention": None if abstention_index is None else prediction == abstention_index,
            "target_mass_on_abstention": None if abstention_index is None else float(targets[index, abstention_index].item()),
            "answerability": row.get("answerability_status") or "not_recorded",
        })

    bins = []
    for index, (lower, upper) in enumerate(zip(CONFIDENCE_BIN_EDGES, CONFIDENCE_BIN_EDGES[1:])):
        members = [
            detail for detail in details
            if lower <= detail["confidence"] and (
                detail["confidence"] <= upper if index == len(CONFIDENCE_BIN_EDGES) - 2
                else detail["confidence"] < upper
            )
        ]
        hard = [detail["hard_correct"] for detail in members if detail["hard_correct"] is not None]
        bins.append({
            "range": f"[{lower:.2f}, {upper:.2f}{']' if index == len(CONFIDENCE_BIN_EDGES) - 2 else ')'}",
            "questions": len(members),
            "mean_max_probability": _mean([detail["confidence"] for detail in members]),
            "mean_target_mass_at_prediction": _mean([detail["target_mass_at_prediction"] for detail in members]),
            "hard_label_questions": len(hard),
            "hard_correct_count": sum(hard),
        })

    def idk_metrics(members):
        available = [detail for detail in members if detail["prediction_is_abstention"] is not None]
        hard = [detail["hard_correct"] for detail in members if detail["hard_correct"] is not None]
        return {
            "questions": len(members),
            "questions_with_abstention_option": len(available),
            "predicted_abstention_count": sum(detail["prediction_is_abstention"] for detail in available),
            "predicted_abstention_fraction": _mean([float(detail["prediction_is_abstention"]) for detail in available]),
            "mean_target_mass_on_abstention": _mean([
                detail["target_mass_on_abstention"] for detail in available
            ]),
            "hard_label_questions": len(hard),
            "hard_correct_count": sum(hard),
        }

    by_answerability = {}
    for answerability in sorted({detail["answerability"] for detail in details}):
        by_answerability[answerability] = idk_metrics(
            [detail for detail in details if detail["answerability"] == answerability]
        )
    return {
        "confidence_interpretation": "Target mass is support in the recorded target distribution; it is not a teacher-truth or calibration claim.",
        "mean_target_mass_at_prediction": _mean([detail["target_mass_at_prediction"] for detail in details]),
        "max_probability_bins": bins,
        "idk": idk_metrics(details),
        "idk_by_answerability": by_answerability,
    }


def _development_metrics(rows, logits):
    overall, targets, source_kinds, hard_labels, counts = _metric_summary(rows, logits)
    dimensions = {
        "language": lambda row: row.get("language") or "not_recorded",
        "choice_count": lambda row: str(len(row["options"])),
        "answerability": lambda row: row.get("answerability_status") or "not_recorded",
        "provenance": lambda row: row.get("_training_snapshot", {}).get("source_schema_kind", "not_recorded"),
        "review": lambda row: (
            "review_recommended"
            if "review_recommended" in row.get("_training_snapshot", {}).get("cohorts", [])
            else "no_known_review_flag"
        ),
        "operation": lambda row: row.get("operation_id") or row.get("family") or "not_recorded",
    }
    grouped = {}
    for name, get_key in dimensions.items():
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(str(get_key(row)), []).append(index)
        grouped[f"by_{name}"] = {
            key: _metric_summary([rows[index] for index in indices], logits[indices])[0]
            for key, indices in sorted(groups.items())
        }
    return {
        "questions": len(rows),
        "overall": overall,
        "grouped": grouped,
        "confidence_and_idk": _confidence_idk_summary(rows, logits, targets, hard_labels, counts),
    }


def _choice_order_agreement(rows_by_epoch, logits_by_epoch):
    first_rows = {row["id"]: (row, logits_by_epoch["-1"][index])
                  for index, row in enumerate(rows_by_epoch["-1"])}
    other_rows = {row["id"]: (row, logits_by_epoch["-3"][index])
                  for index, row in enumerate(rows_by_epoch["-3"])}
    stable_prediction_matches = 0
    variations = []
    for identifier, (first, first_logits) in first_rows.items():
        other, other_logits = other_rows[identifier]
        first_probabilities = torch.softmax(first_logits[:len(first["options"])], dim=-1)
        other_probabilities = torch.softmax(other_logits[:len(other["options"])], dim=-1)
        first_distribution = dict(zip(first["choice_ids"], first_probabilities.tolist()))
        other_distribution = dict(zip(other["choice_ids"], other_probabilities.tolist()))
        if set(first_distribution) != set(other_distribution):
            raise ValueError("Choice IDs changed across development permutations")
        first_prediction = first["choice_ids"][int(first_probabilities.argmax().item())]
        other_prediction = other["choice_ids"][int(other_probabilities.argmax().item())]
        stable_prediction_matches += int(first_prediction == other_prediction)
        variations.append(0.5 * sum(
            abs(first_distribution[choice_id] - other_distribution[choice_id])
            for choice_id in first_distribution
        ))
    questions = len(first_rows)
    return {
        "questions": questions,
        "stable_choice_id_argmax_agreement_count": stable_prediction_matches,
        "stable_choice_id_argmax_agreement_fraction": stable_prediction_matches / questions,
        "mean_aligned_distribution_total_variation": _mean(variations),
    }


def _checkpoint_bindings(runs, root):
    bindings = {}
    for arm, run in runs.items():
        manifest_path = (run["checkpoint_dir"] / "checkpoint.json").resolve()
        try:
            relative_path = manifest_path.relative_to(root.resolve()).as_posix()
        except ValueError as exc:
            raise ValueError("Selected checkpoints must be inside the repository for receipt binding") from exc
        bindings[arm] = {
            "checkpoint_manifest_path": relative_path,
            "checkpoint_manifest_sha256": run["checkpoint_sha256"],
        }
    return bindings


def _verify_generation_receipt(path, checkpoint_bindings):
    if path is None:
        raise ValueError("The source-test stage requires a committed generation-reflection receipt")
    from evaluation.blind_runner import verify_generation_receipt

    return verify_generation_receipt(path, checkpoint_bindings)


def _evaluate_arm(arm, run, selection, rows_by_split, tokenizer, config, args, stage):
    from transformers import AutoModel

    manifest = run["manifest"]
    if manifest["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("The selected runs require CUDA/HIP, but no accelerator is available")
    if manifest["attention_backend"] == "flash" and manifest["device"] != "cuda":
        raise ValueError("Cannot reproduce the run's flash attention backend on this device")
    if manifest["device"] == "cuda":
        _, total_memory = torch.cuda.mem_get_info()
        cap_gib = manifest.get("gpu_allocation_cap_gib")
        if not isinstance(cap_gib, (int, float)) or cap_gib <= 0 or cap_gib * 1024**3 >= total_memory:
            raise ValueError("The selected run has an invalid GPU allocation cap")
        torch.cuda.set_per_process_memory_fraction(cap_gib * 1024**3 / total_memory)

    torch.set_num_threads(manifest["threads"])
    torch.manual_seed(manifest["seed"])
    random.seed(manifest["seed"])
    required = args.parameter_bytes + MODEL_BYTES_RESERVE
    require_memory(required, INTERNAL_RESERVE_GIB, manifest["device"])

    candidate = CANDIDATES[manifest["model"]]
    load_options = {"device_map": "cuda"} if manifest["device"] == "cuda" else {}
    backbone = AutoModel.from_pretrained(
        candidate["repo"], revision=candidate["revision"], local_files_only=True,
        dtype=torch.float32, attn_implementation="sdpa", **load_options,
    ).to(manifest["device"])
    backbone._horscht_attention_backend = manifest["attention_backend"]
    head = DecisionHead(backbone.config.hidden_size).to(manifest["device"])
    restored = load_training_checkpoint(
        run["checkpoint_dir"], backbone=backbone, head=head,
    )
    for key in _CHECKPOINT_MATCH_FIELDS:
        if restored["manifest"].get(key) != manifest.get(key):
            raise ValueError(f"Loaded {arm} checkpoint changed training configuration: {key}")

    evaluation_args = SimpleNamespace(device=manifest["device"], max_tokens=manifest["max_tokens"])
    try:
        if stage == "dev":
            rows_by_epoch = {}
            logits_by_epoch = {}
            results = {}
            dev_original = rows_by_split["dev"]
            for epoch in DEV_PERMUTATION_EPOCHS:
                epoch_rows = [ordered_row(row, manifest["seed"], epoch) for row in dev_original]
                lengths, plan = _plan_for_rows(dev_original, tokenizer, manifest, epoch)
                if epoch == -1:
                    _verify_prepared_lengths(selection, lengths, epoch)
                rows_by_epoch[str(epoch)] = epoch_rows
                logits_by_epoch[str(epoch)] = _evaluate_logits(
                    tokenizer, backbone, head, epoch_rows, plan, evaluation_args,
                )
                results[str(epoch)] = _development_metrics(epoch_rows, logits_by_epoch[str(epoch)])
            selected_dev_score = run["report"]["selected_checkpoint_development"]["soft_cross_entropy"]
            baseline_score = results["-1"]["overall"]["soft_cross_entropy"]
            if abs(baseline_score - selected_dev_score) > 1e-5:
                raise RuntimeError(f"Reloaded {arm} checkpoint does not reproduce its selected development score")
            return {
                "development_orders": {
                    "baseline_permutation_epoch": -1,
                    "stress_permutation_epoch": -3,
                    "baseline": results["-1"],
                    "stress": results["-3"],
                    "stress_minus_baseline_soft_cross_entropy": (
                        results["-3"]["overall"]["soft_cross_entropy"] - baseline_score
                    ),
                    "choice_order_sensitivity": _choice_order_agreement(rows_by_epoch, logits_by_epoch),
                },
            }

        test_original = rows_by_split["test"]
        test_rows = [ordered_row(row, manifest["seed"], TEST_PERMUTATION_EPOCH) for row in test_original]
        test_lengths, test_plan = _plan_for_rows(test_original, tokenizer, manifest, TEST_PERMUTATION_EPOCH)
        _verify_prepared_lengths(selection, test_lengths, TEST_PERMUTATION_EPOCH)
        test_metrics = evaluate_batches(
            tokenizer, backbone, head, test_rows, test_plan, evaluation_args, cache=None,
        )
        return {
            "source_held_out_test": test_metrics,
            "test_permutation_epoch": TEST_PERMUTATION_EPOCH,
            "test_epoch_plan_sha256": test_plan.sha256,
        }
    finally:
        del backbone, head
        gc.collect()
        if manifest["device"] == "cuda":
            torch.cuda.empty_cache()



def evaluate_training(
    *,
    snapshot: str | Path,
    selection: str | Path,
    head_run: str | Path,
    full_run: str | Path,
    output: str | Path,
    stage: str = "dev",
    generation_receipt: str | Path | None = None,
) -> Path:
    """Write isolated development diagnostics or receipt-gated source-test aggregates."""
    if stage not in {"dev", "test"}:
        raise ValueError("stage must be 'dev' or 'test'")
    if stage == "test" and generation_receipt is None:
        raise ValueError("The source-test stage requires a committed generation-reflection receipt")

    output_dir = Path(output)
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Evaluation output already exists: {output_dir}")

    snapshot = Path(snapshot)
    selection_path = Path(selection)
    arm_dirs = {"head": Path(head_run), "full": Path(full_run)}
    require_training_snapshot(snapshot)
    snapshot_manifest = _read_json(snapshot / "manifest.json")
    for filename in ("rows.jsonl", "splits.jsonl"):
        artifact = snapshot_manifest.get("artifacts", {}).get(filename, {})
        if not (snapshot / filename).is_file() or sha256(snapshot / filename) != artifact.get("sha256"):
            raise ValueError(f"Training snapshot checksum mismatch: {filename}")
    selection_data = _read_json(selection_path)
    runs = _validate_runs(snapshot, selection_path, selection_data, arm_dirs, stage)
    checkpoint_bindings = _checkpoint_bindings(runs, ROOT)
    receipt = _verify_generation_receipt(generation_receipt, checkpoint_bindings) if stage == "test" else None

    split_ids = selection_data["ids_by_split"]
    relevant_ids = set(split_ids["train"]) | set(split_ids["dev"]) | set(split_ids["test"])
    split_by_id = _selected_split_metadata(snapshot, relevant_ids)
    group_sets = {split: set() for split in ("train", "dev", "test")}
    for split in group_sets:
        for identifier in split_ids[split]:
            entry = split_by_id[identifier]
            if entry.get("split") != split:
                raise ValueError(f"Selected {split} ID {identifier!r} belongs to another snapshot split")
            group = entry.get("snapshot_group_id")
            if not isinstance(group, str) or not group:
                raise ValueError(f"Selected ID {identifier!r} has no frozen problem group")
            group_sets[split].add(group)
    if group_sets["train"] & group_sets["dev"] or group_sets["train"] & group_sets["test"] or group_sets["dev"] & group_sets["test"]:
        raise ValueError("Selected train, development and test problem groups overlap")

    if stage == "dev":
        rows_by_split = {"dev": _read_selected_rows(snapshot, split_ids["dev"], split_by_id, "dev")}
        evaluated_ids = split_ids["dev"]
    else:
        rows_by_split = {"test": _read_selected_rows(snapshot, split_ids["test"], split_by_id, "test")}
        evaluated_ids = split_ids["test"]

    common = runs["head"]["manifest"]
    import transformers
    if transformers.__version__ != common["transformers"] or torch.__version__ != common["torch"]:
        raise ValueError("Current PyTorch/Transformers versions differ from the frozen training runs")
    tokenizer, config = _load_tokenizer_and_config(common)
    parameter_bytes = _parameter_bytes(transformers.AutoModel, config)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    started_at = datetime.now(timezone.utc).isoformat()
    original_torch_rng = torch.get_rng_state()
    original_python_rng = random.getstate()
    original_cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    result = {
        "schema_version": 2,
        "stage": stage,
        "scope": (
            "internal development diagnostics; source-held-out test questions were not evaluated"
            if stage == "dev"
            else "internal source-held-out test evaluation after committed development reflection"
        ),
        "started_at": started_at,
        "evaluated_at": None,
        "snapshot_manifest_sha256": sha256(snapshot / "manifest.json"),
        "selection_sha256": sha256(selection_path),
        "selected_ids_sha256": hashlib.sha256(
            json.dumps(evaluated_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "selected_questions": len(evaluated_ids),
        "model": common["model"],
        "repo": common["repo"],
        "revision": common["revision"],
        "seed": common["seed"],
        "device": common["device"],
        "parameters_dtype": common["parameters_dtype"],
        "compute_autocast": common["compute_autocast"],
        "attention_backend": common["attention_backend"],
        "internal_memory_reserve_gib": INTERNAL_RESERVE_GIB,
        "input_limit_tokens": common["max_tokens"],
        "selected_checkpoints": checkpoint_bindings,
        "arms": {},
    }
    if stage == "test":
        result["test_permutation_epoch"] = TEST_PERMUTATION_EPOCH
        result["generation_reflection_receipt"] = {
            "path": receipt["path"],
            "sha256": receipt["sha256"],
            "committed_at": receipt.get("committed_at"),
        }
    else:
        result["development_permutation_epochs"] = list(DEV_PERMUTATION_EPOCHS)
    try:
        for arm in ("head", "full"):
            run = runs[arm]
            arm_result = _evaluate_arm(
                arm, run, selection_data, rows_by_split, tokenizer, config,
                SimpleNamespace(parameter_bytes=parameter_bytes), stage,
            )
            result["arms"][arm] = {
                "run_manifest_sha256": sha256(run["run_dir"] / "manifest.json"),
                "report_sha256": sha256(run["run_dir"] / "report.json"),
                "selected_checkpoint_manifest_sha256": run["checkpoint_sha256"],
                "selected_checkpoint_artifacts": run["checkpoint"]["files"],
                **arm_result,
            }
        for run in runs.values():
            if sha256(run["checkpoint_dir"] / "checkpoint.json") != run["checkpoint_sha256"]:
                raise RuntimeError("A selected checkpoint manifest changed during evaluation")
        result["evaluated_at"] = datetime.now(timezone.utc).isoformat()
        (temporary / OUTPUT_FILE).write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        if output_dir.exists() or output_dir.is_symlink():
            raise FileExistsError(f"Evaluation output already exists: {output_dir}")
        os.rename(temporary, output_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        torch.set_rng_state(original_torch_rng)
        random.setstate(original_python_rng)
        if original_cuda_rng is not None and torch.cuda.is_initialized():
            torch.cuda.set_rng_state_all(original_cuda_rng)
    return output_dir / OUTPUT_FILE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--head-run", type=Path, required=True)
    parser.add_argument("--full-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("dev", "test"), default="dev")
    parser.add_argument("--generation-receipt", type=Path)
    args = parser.parse_args()
    path = evaluate_training(
        snapshot=args.snapshot,
        selection=args.selection,
        head_run=args.head_run,
        full_run=args.full_run,
        output=args.output,
        stage=args.stage,
        generation_receipt=args.generation_receipt,
    )
    print(json.dumps({"evaluation": str(path)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
