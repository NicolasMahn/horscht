"""Prepare and compare matched head-only and full-backbone training runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import resource
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from experiment import CANDIDATES, DecisionHead, format_input, last_token
from probability_experiment import categorical_metrics, soft_cross_entropy, targets_for_rows


ROOT = Path(__file__).parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def available_memory_bytes() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("Cannot read MemAvailable for the training memory guard")


def memory_snapshot(device: str = "cpu") -> dict[str, int | None]:
    host = available_memory_bytes()
    snapshot = {"host_available_bytes": host, "driver_free_bytes": None, "device_total_bytes": None,
                "device_allocated_bytes": 0, "device_reserved_bytes": 0,
                "reusable_device_bytes": 0, "effective_available_bytes": host,
                "process_cap_bytes": None, "process_cap_headroom_bytes": None}
    if device == "cuda":
        free, total = torch.cuda.mem_get_info()
        allocated, reserved = torch.cuda.memory_allocated(), torch.cuda.memory_reserved()
        reusable = max(0, reserved - allocated)
        cap = int(total * torch.cuda.get_per_process_memory_fraction())
        # Cached allocations can serve the next GPU operation. On a shared-memory
        # accelerator, crediting them to MemAvailable too would count RAM twice.
        snapshot.update(driver_free_bytes=free, device_total_bytes=total,
                        device_allocated_bytes=allocated, device_reserved_bytes=reserved,
                        reusable_device_bytes=reusable, effective_available_bytes=min(host, free + reusable),
                        process_cap_bytes=cap,
                        process_cap_headroom_bytes=max(0, cap - allocated))
    return snapshot


class TrainingMemoryPressure(RuntimeError):
    def __init__(self, snapshot, additional_bytes, reserve_bytes):
        self.snapshot = snapshot
        self.additional_bytes = additional_bytes
        self.reserve_bytes = reserve_bytes
        message = (f"Training memory guard: {snapshot['effective_available_bytes'] / 1024**3:.1f} GiB available; "
                   f"need {additional_bytes / 1024**3:.1f} GiB additional plus {reserve_bytes / 1024**3:.1f} GiB reserve")
        if snapshot.get("process_cap_headroom_bytes") is not None:
            message += (f"; host free {snapshot['host_available_bytes'] / 1024**3:.1f} GiB"
                        f"; driver free {snapshot['driver_free_bytes'] / 1024**3:.1f} GiB"
                        f"; process-cap headroom {snapshot['process_cap_headroom_bytes'] / 1024**3:.1f} GiB")
        super().__init__(message)


def require_memory(additional_bytes: int, reserve_gib: float, device: str = "cpu"):
    snapshot = memory_snapshot(device)
    reserve = int(reserve_gib * 1024**3)
    cap_headroom = snapshot.get("process_cap_headroom_bytes")
    driver_free = snapshot.get("driver_free_bytes")
    if (snapshot["effective_available_bytes"] < additional_bytes + reserve
            or driver_free is not None and driver_free < reserve
            or cap_headroom is not None and cap_headroom < additional_bytes):
        raise TrainingMemoryPressure(snapshot, additional_bytes, reserve)
    return snapshot


def peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2


def device_memory(device: str) -> dict[str, float]:
    if device == "cpu":
        return {"peak_device_allocated_gib": 0.0, "peak_device_reserved_gib": 0.0}
    return {"peak_device_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_device_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3}


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def permutation_seed(seed: int, question_id: str, epoch: int) -> int:
    payload = json.dumps([seed, question_id, epoch], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def choice_order_risks(row: dict[str, Any]) -> list[str]:
    """Keep ambiguous references in place rather than rewrite question meaning."""
    text = "\n".join(str(row.get(key, "")) for key in ("context", "question"))
    text += "\n" + "\n".join(map(str, row["options"]))
    risks = []
    if re.search(r"(?<!\w)[A-Z](?!\w)|\([a-z]\)", text):
        risks.append("possible_choice_letter_reference")
    if re.search(r"\b(?:option|choice|answer|alternative|réponse|opción|antwort|選択肢)\s*(?:number\s*)?[1-9]\d*\b", text, re.IGNORECASE):
        risks.append("possible_numeric_choice_reference")
    if re.search(r"\b(?:above|below|preceding|following|previous|latter|former|first|second|third|last|obigen|oben|unten|vorherigen|folgenden|ci-dessus|ci-dessous|anteriores|arriba|abajo|sopra|sotto)\b|上述|以上|以下|上記|下記|أعلاه|أدناه", text, re.IGNORECASE):
        risks.append("possible_relative_order_reference")
    return risks


def ordered_row(row: dict[str, Any], seed: int, epoch: int) -> dict[str, Any]:
    from dataset_factory.training_snapshot import permute_choices

    validator = None
    if row.get("_new_dataset"):
        from dataset_factory.new_dataset import validate_snapshot_row
        validator = validate_snapshot_row
        errors = validator(row)
        if errors:
            raise ValueError("Invalid source-aware snapshot row: " + "; ".join(errors[:3]))
    if choice_order_risks(row):
        import copy
        return copy.deepcopy(row)
    return permute_choices(row, permutation_seed(seed, row["id"], epoch), validator=validator)


def prompt_cache_key(row: dict[str, Any]) -> str:
    return hashlib.sha256(format_input(row).encode("utf-8")).hexdigest()


def require_training_snapshot(snapshot: Path) -> None:
    roots = [(ROOT / "data" / name).resolve() for name in ("training_snapshots", "newtraining")]
    allowed = next((root for root in roots if snapshot.resolve().is_relative_to(root)), None)
    if allowed is None:
        raise ValueError("Training accepts only frozen snapshots under data/training_snapshots or data/newtraining")
    for filename in ["manifest.json", "rows.jsonl", "splits.jsonl"]:
        if not (snapshot / filename).resolve().is_relative_to(allowed):
            raise ValueError("A training snapshot artifact resolves outside its allowlisted directory")


def resume_best_checkpoint(progress: dict[str, Any], checkpoint: Path) -> Path:
    saved = Path(progress["best_checkpoint"])
    if not saved.is_absolute():
        saved = checkpoint.parent / saved
    elif not saved.exists():
        # Container paths may differ from the host while sibling artifacts stay together.
        saved = checkpoint.parent / saved.name
    if not saved.is_dir():
        raise FileNotFoundError(f"The development-selected resume checkpoint is missing: {saved}")
    return saved


def cohort(row: dict[str, Any]) -> tuple[str, str, str]:
    count = len(row["options"])
    tier = "2-4" if count <= 4 else "5-16" if count <= 16 else "17-64" if count <= 64 else "65-265"
    source = row.get("distribution_source_kind", row.get("supervision_kind", "unknown"))
    if row.get("_new_dataset", {}).get("source_id"):
        source = f"{row['_new_dataset']['source_id']}:{source}"
    return source, tier, row.get("answerability_status") or "not_recorded"


def choose_pilot_rows(
    rows: list[dict[str, Any]],
    splits: list[dict[str, Any]],
    lengths: dict[str, int],
    *,
    counts: dict[str, int],
    max_tokens: int,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Take deterministic cohort round-robin subsets from already locked splits."""
    split_by_id = {entry["id"]: entry for entry in splits}
    rng = random.Random(seed)
    buckets: dict[str, dict[tuple[str, str, str], list[dict[str, Any]]]] = {
        split: {} for split in counts
    }
    long_counts: Counter[str] = Counter()
    for row in rows:
        entry = split_by_id[row["id"]]
        split = entry["split"]
        if split not in buckets:
            continue
        if lengths[row["id"]] > max_tokens:
            long_counts[split] += 1
            continue
        buckets[split].setdefault(cohort(row), []).append(row)
    result: dict[str, list[dict[str, Any]]] = {}
    for split, groups in buckets.items():
        for group in groups.values():
            rng.shuffle(group)
        keys = sorted(groups)
        rng.shuffle(keys)
        selected: list[dict[str, Any]] = []
        used_problem_groups: set[str] = set()
        while len(selected) < counts[split] and keys:
            remaining = []
            for key in keys:
                bucket = groups[key]
                while bucket:
                    row = bucket.pop()
                    entry = split_by_id[row["id"]]
                    group = str(entry.get("snapshot_group_id", entry.get("problem_group_id", row["id"])))
                    if group in used_problem_groups:
                        continue
                    used_problem_groups.add(group)
                    selected.append(row)
                    break
                if bucket:
                    remaining.append(key)
                if len(selected) == counts[split]:
                    break
            keys = remaining
        if len(selected) != counts[split]:
            raise ValueError(f"Only {len(selected)} eligible distinct groups for requested {counts[split]} {split} rows")
        result[split] = selected
    return result, {
        "max_tokens": max_tokens,
        "excluded_complete_long_inputs_by_split": dict(long_counts),
        "selected_questions_by_split": {split: len(values) for split, values in result.items()},
        "selection_policy": "seeded source/cardinality/answerability cohort round-robin; one question per known group",
    }


def tokenize_snapshot(snapshot: Path, output: Path, tokenizer) -> tuple[list[dict[str, Any]], dict[str, int]]:
    require_training_snapshot(snapshot)
    rows = read_jsonl(snapshot / "rows.jsonl")
    lengths: dict[str, int] = {}
    started = time.perf_counter()
    for index in range(0, len(rows), 256):
        batch = rows[index:index + 256]
        encoded = tokenizer([format_input(row) for row in batch], add_special_tokens=True, truncation=False)
        lengths.update((row["id"], len(tokens)) for row, tokens in zip(batch, encoded["input_ids"]))
    ordered_lengths = sorted(lengths.values())
    by_cardinality: dict[str, list[int]] = {}
    for row in rows:
        by_cardinality.setdefault(str(len(row["options"])), []).append(lengths[row["id"]])
    write_json(output / "token-audit.json", {
        "snapshot_rows_sha256": sha256(snapshot / "rows.jsonl"),
        "questions": len(rows),
        "min": ordered_lengths[0],
        "median": ordered_lengths[len(rows) // 2],
        "p95": ordered_lengths[int(len(rows) * .95)],
        "p99": ordered_lengths[int(len(rows) * .99)],
        "max": ordered_lengths[-1],
        "above_512": sum(value > 512 for value in ordered_lengths),
        "by_choice_count": {
            count: {"questions": len(values), "min": min(values), "max": max(values)}
            for count, values in sorted(by_cardinality.items(), key=lambda item: int(item[0]))
        },
        "seconds": time.perf_counter() - started,
        "lengths_by_id": lengths,
    })
    return rows, lengths


def tensors_for(rows: list[dict[str, Any]], device: str = "cpu"):
    targets, sources, labels = targets_for_rows(rows)
    counts = torch.tensor([len(row["options"]) for row in rows])
    return targets.to(device), sources, labels, counts.to(device)


def encode(tokenizer, backbone, row: dict[str, Any], max_tokens: int, device: str = "cpu") -> torch.Tensor:
    tokens = tokenizer(format_input(row), return_tensors="pt", add_special_tokens=True, truncation=False)
    if tokens["input_ids"].shape[1] > max_tokens:
        raise ValueError(f"{row['id']}: reordered complete input exceeds {max_tokens} tokens; refusing truncation")
    input_ids = tokens["input_ids"].to(device)
    mask = tokens.get("attention_mask", torch.ones_like(tokens["input_ids"])).to(device)
    with torch.autocast(device, dtype=torch.bfloat16, cache_enabled=False):
        hidden = backbone(input_ids=input_ids, attention_mask=mask, use_cache=False).last_hidden_state
    return last_token(hidden, mask).float()


def evaluate(tokenizer, backbone, head, rows, max_tokens, cache=None, device="cpu"):
    backbone.eval()
    head.eval()
    logits = []
    with torch.inference_mode():
        for row in rows:
            feature = cache[prompt_cache_key(row)] if cache is not None else encode(tokenizer, backbone, row, max_tokens, device)
            logits.append(head(feature, torch.tensor([len(row["options"])], device=device)))
    return summarize_logits(torch.cat(logits).cpu(), rows)


def summarize_logits(batch_logits: torch.Tensor, rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets, sources, labels, counts = tensors_for(rows, str(batch_logits.device))
    result = categorical_metrics(batch_logits, targets, counts, sources, labels)
    source_metrics = result["by_source_kind"].values()
    result["macro_source_cross_entropy"] = sum(value["soft_cross_entropy"] for value in source_metrics) / len(result["by_source_kind"])
    idk_groups = {"answerable": [], "unanswerable": []}
    for index, row in enumerate(rows):
        status = row.get("answerability_status")
        identifier = row.get("abstention_choice_id")
        if status in idk_groups and identifier in row["choice_ids"]:
            predicted_index = int(batch_logits[index, :len(row["options"])].argmax().item())
            idk_groups[status].append(row["choice_ids"][predicted_index] == identifier)
    result["task_grounded_idk"] = {
        status: {"questions": len(values), "predicted_idk_fraction": sum(values) / len(values)}
        for status, values in idk_groups.items() if values
    }
    dimensions = {
        "language": lambda row: row.get("language", "not_recorded"),
        "choice_count": lambda row: str(len(row["options"])),
        "answerability": lambda row: row.get("answerability_status") or "not_recorded",
        "provenance": lambda row: row.get("_training_snapshot", {}).get("source_schema_kind", "not_recorded"),
        "review": lambda row: "review_recommended" if "review_recommended" in row.get("_training_snapshot", {}).get("cohorts", []) else "no_known_review_flag",
    }
    for name, key in dimensions.items():
        groups = {}
        for index, row in enumerate(rows):
            groups.setdefault(key(row), []).append(index)
        result[f"by_{name}"] = {
            group: categorical_metrics(batch_logits[indices], targets[indices], counts[indices],
                                       [sources[index] for index in indices], [labels[index] for index in indices])
            for group, indices in sorted(groups.items())
        }
    return result


def run_arm(args: argparse.Namespace) -> int:
    from scalable_training import run_scalable_arm

    return run_scalable_arm(args)


def compare_arms(head_dir: Path, full_dir: Path, output: Path) -> int:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    arms = {"head": head_dir, "full": full_dir}
    reports = {name: json.loads((directory / "report.json").read_text()) for name, directory in arms.items()}
    manifests = {name: json.loads((directory / "manifest.json").read_text()) for name, directory in arms.items()}
    if any(manifest.get("schema_version") == 2 for manifest in manifests.values()):
        from scalable_training import compare_training_arms
        return compare_training_arms(arms, reports, manifests, output)
    for key in ["selection_sha256", "repo", "revision", "seed", "max_tokens", "head_learning_rate", "device", "torch"]:
        if manifests["head"][key] != manifests["full"][key]:
            raise ValueError(f"Arms are not matched: {key}")
    for key in ["training_questions", "training_tokens", "epochs", "updates"]:
        if reports["head"][key] != reports["full"][key]:
            raise ValueError(f"Arms have different exposure: {key}")
    initial_head_hashes = {
        name: json.loads((directory / "checkpoint-0000" / "checkpoint.json").read_text())["files"]["head.pt"]
        for name, directory in arms.items()
    }
    if len(set(initial_head_hashes.values())) != 1:
        raise ValueError("Arms started from different head weights")
    comparison = {
        "scope": "bounded paired pilot; tiny fixed holdout; development-selected checkpoints; not a full-corpus verdict",
        "matched_questions": reports["head"]["training_questions"],
        "matched_tokens": reports["head"]["training_tokens"],
        "selection_sha256": manifests["head"]["selection_sha256"],
        "same_initial_head_sha256": initial_head_hashes["head"],
        "arms": {name: {key: report[key] for key in ["best_dev_soft_cross_entropy", "best_update",
                                                            "final_test", "total_seconds", "peak_rss_gib", "peak_device_allocated_gib",
                                                            "peak_device_reserved_gib",
                                                            "head_weight_change_l2", "backbone_tensors_with_changed_sample",
                                                            "selected_checkpoint_reload_verified", "best_checkpoint", "last_checkpoint"]}
                 for name, report in reports.items()},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output.with_suffix(".json"), comparison)
    memory_metric = "peak_device_allocated_gib" if reports["head"]["peak_device_allocated_gib"] else "peak_rss_gib"
    memory_title = "Peak GPU allocations" if memory_metric == "peak_device_allocated_gib" else "Peak resident memory"
    figure = make_subplots(rows=2, cols=2, subplot_titles=("Development cross-entropy", "Development Brier score",
                                                          "Elapsed wall time", memory_title))
    for name, directory in arms.items():
        history = json.loads((directory / "history.json").read_text())
        measured = [point for point in history if "dev" in point]
        for column, metric in [(1, "soft_cross_entropy"), (2, "categorical_brier_sum")]:
            figure.add_trace(go.Scatter(x=[point["questions_seen"] for point in measured],
                                       y=[point["dev"][metric] for point in measured],
                                       mode="lines+markers", name=name, legendgroup=name,
                                       showlegend=column == 1), row=1, col=column)
    for column, metric in [(1, "total_seconds"), (2, memory_metric)]:
        figure.add_trace(go.Bar(x=list(arms), y=[reports[name][metric] for name in arms], showlegend=False),
                         row=2, col=column)
    figure.update_xaxes(title_text="Training questions seen", row=1)
    figure.update_yaxes(title_text="Seconds", row=2, col=1)
    figure.update_yaxes(title_text="GiB", row=2, col=2)
    figure.update_layout(title="Matched Qwen3-4B pilot: head only and full model", height=850)
    figure.write_html(output.with_suffix(".html"), include_plotlyjs=True)
    print(json.dumps({"comparison": str(output.with_suffix(".json")), "curves": str(output.with_suffix(".html"))}), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    prepare = subcommands.add_parser("prepare")
    prepare.add_argument("--snapshot", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--train-questions", type=int, default=32)
    prepare.add_argument("--dev-questions", type=int, default=16)
    prepare.add_argument("--test-questions", type=int, default=16)
    prepare.add_argument("--system-test-manifest", type=Path)
    prepare.add_argument("--epochs", type=int, default=1)
    run = subcommands.add_parser("train-arm", aliases=["run-arm"])
    run.add_argument("--snapshot", type=Path, required=True)
    run.add_argument("--selection", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--arm", choices=["head", "full", "last-block"], required=True)
    run.add_argument("--head-architecture", choices=["last-token-linear-v1", "option-context-attention-v1"], default="last-token-linear-v1")
    run.add_argument("--head-learning-rate", type=float, default=.001)
    run.add_argument("--backbone-learning-rate", type=float, default=.00001)
    run.add_argument("--accumulation", type=int, default=16)
    run.add_argument("--eval-every", type=int, default=500)
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--reserve-gib", type=float, default=12)
    run.add_argument("--resume", type=Path)
    run.add_argument("--warm-start", type=Path, help="Restore backbone/head weights with fresh optimizer, scheduler and RNG")
    run.add_argument("--lr-schedule", choices=["constant", "warmup-cosine"], default="constant")
    run.add_argument("--warmup-fraction", type=float, default=.03)
    run.add_argument("--max-updates", type=int, default=0, help="Bound a development-only pilot; zero means all epochs")
    run.add_argument("--fast-dev-questions", type=int, default=0)
    run.add_argument("--full-dev-every", type=int, default=0, help="Full-development interval; zero uses eval-every")
    run.add_argument("--order-consistency-questions", type=int, default=512)
    run.add_argument("--compact-last-block-checkpoints", action="store_true")
    run.add_argument("--resume-lifecycle-audit", type=Path)
    run.add_argument("--defer-full-development", action="store_true",
                     help="Pin the original selection checkpoints and score full development after all weight updates")
    run.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run.add_argument("--attention-backend", choices=["auto", "flash"], default="auto")
    run.add_argument("--gpu-allocation-cap-gib", type=float, default=44)
    run.add_argument("--system-test-manifest", type=Path, required=True)
    run.add_argument("--launch-approved", action="store_true")
    run.add_argument("--epochs", type=int, default=1)
    run.add_argument("--early-stop-patience", type=int, default=0)
    run.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    run.add_argument("--early-stop-min-epochs", type=int, default=1)
    run.add_argument("--microbatch", type=int, default=1)
    run.add_argument("--max-batch-tokens", type=int, default=16384)
    run.add_argument("--bucket-questions", type=int, default=256)
    run.add_argument("--checkpoint-every", type=int, default=500)
    run.add_argument("--checkpoint-seconds", type=float, default=1800)
    run.add_argument("--retain-checkpoints", type=int, default=3)
    run.add_argument("--feature-cache-mib", type=int, default=1024)
    run.add_argument("--status-file", type=Path, default=ROOT / "runs" / "training-status.json")
    prepare_full = subcommands.add_parser("prepare-full")
    prepare_full.add_argument("--snapshot", type=Path, required=True)
    prepare_full.add_argument("--output", type=Path, required=True)
    prepare_full.add_argument("--system-test-manifest", type=Path, required=True)
    prepare_full.add_argument("--model", default="qwen-instruct", choices=CANDIDATES)
    prepare_full.add_argument("--seed", type=int, default=20260929)
    prepare_full.add_argument("--max-tokens", type=int, default=16384)
    compare = subcommands.add_parser("compare")
    compare.add_argument("--head", type=Path, required=True)
    compare.add_argument("--full", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    for command in [prepare, run]:
        command.add_argument("--model", default="qwen-instruct", choices=CANDIDATES)
        command.add_argument("--seed", type=int, default=20260929)
        command.add_argument("--max-tokens", type=int, default=512 if command is prepare else 16384)
    args = parser.parse_args()
    if args.command == "compare":
        return compare_arms(args.head, args.full, args.output)
    if args.command == "prepare-full":
        from scalable_training import prepare_full_selection
        return prepare_full_selection(args)
    if args.command in ["run-arm", "train-arm"]:
        if args.resume and args.warm_start:
            parser.error("Choose either an exact resume or a fresh-optimizer warm start")
        if not 0 <= args.warmup_fraction < 1 or min(args.max_updates, args.fast_dev_questions, args.full_dev_every, args.order_consistency_questions) < 0:
            parser.error("Warmup fraction must be in [0, 1); optional budgets must be nonnegative")
        if min(args.head_learning_rate, args.backbone_learning_rate, args.accumulation,
               args.eval_every, args.threads, args.reserve_gib, args.max_tokens, args.epochs,
               args.microbatch, args.max_batch_tokens, args.bucket_questions, args.checkpoint_every,
               args.checkpoint_seconds, args.retain_checkpoints) <= 0 or args.feature_cache_mib < 0:
            parser.error("Learning rates, batch/check intervals, threads, reserve and token limit must be positive")
        return run_arm(args)
    from transformers import AutoTokenizer
    lock = None
    if args.system_test_manifest:
        from scalable_training import excluded_by_protocol, validate_system_test_lock, with_linked_exclusions
        lock = validate_system_test_lock(args.system_test_manifest, args.snapshot)
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = CANDIDATES[args.model]
    tokenizer = AutoTokenizer.from_pretrained(candidate["repo"], revision=candidate["revision"], local_files_only=True)
    rows, lengths = tokenize_snapshot(args.snapshot, args.output, tokenizer)
    protocol_exclusions = []
    if lock is not None:
        lock = with_linked_exclusions(lock, rows)
        protocol_exclusions = [row["id"] for row in rows if excluded_by_protocol(row, lock)]
        rows = [row for row in rows if not excluded_by_protocol(row, lock)]
    selected, policy = choose_pilot_rows(rows, read_jsonl(args.snapshot / "splits.jsonl"), lengths,
                                        counts={"train": args.train_questions, "dev": args.dev_questions,
                                                "test": args.test_questions},
                                        max_tokens=args.max_tokens, seed=args.seed)
    ordered_lengths = {}
    exposure_lengths = {}
    for split, epoch in [("train", epoch) for epoch in range(args.epochs)] + [("dev", -1), ("test", -2)]:
        for row in selected[split]:
            length = len(tokenizer(format_input(ordered_row(row, args.seed, epoch)), add_special_tokens=True)["input_ids"])
            if length > args.max_tokens:
                raise ValueError(f"{row['id']}: fixed permutation exceeds pilot token budget")
            ordered_lengths[row["id"]] = length
            exposure_lengths.setdefault(str(epoch), {})[row["id"]] = length
    write_json(args.output / "selection.json", {
        **policy, "seed": args.seed, "model": args.model,
        "snapshot_manifest_sha256": sha256(args.snapshot / "manifest.json"),
        "ids_by_split": {split: [row["id"] for row in values] for split, values in selected.items()},
        "ordered_token_lengths": ordered_lengths,
        "training_tokens": sum(ordered_lengths[row["id"]] for row in selected["train"]),
        "prepared_epochs": args.epochs, "ordered_lengths_by_epoch": exposure_lengths,
        "system_test_manifest_sha256": sha256(args.system_test_manifest) if args.system_test_manifest else None,
        "excluded_by_system_protocol_ids": protocol_exclusions,
        "scope": "bounded grouped cohort; remaining eligible snapshot questions are not used in this run",
        "unselected_snapshot_questions": len(lengths) - sum(map(len, selected.values())) - len(protocol_exclusions),
    })
    print(json.dumps(policy), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
