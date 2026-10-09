"""Frozen-backbone training with probability-v2 distributions and hard labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from dataset_factory import probability_v2
from experiment import CAPACITY, DecisionHead, format_input, last_token, reject_external_training_input


ROOT = Path(__file__).parent
SOFT_SOURCE_KINDS = probability_v2.SOFT_SOURCE_KINDS
HARD_LABEL = probability_v2.HARD_LABEL


def load_probability_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read strict probability-v2 rows and verify every choice-aligned field."""
    reject_external_training_input(path)
    examples: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"line {line_number}: invalid JSON: {exc.msg}") from exc
            errors = probability_v2.validate_v2_row(row)
            if errors:
                raise ValueError(f"line {line_number}: invalid probability-v2 row: {', '.join(errors)}")
            if row["id"] in seen_ids:
                raise ValueError(f"line {line_number}: duplicate example id {row['id']!r}")
            seen_ids.add(row["id"])

            expected_ids = probability_v2.stable_choice_ids(row["id"], row["options"])
            if row["choice_ids"] != expected_ids:
                raise ValueError(f"line {line_number}: choice_ids are not stable IDs for options order")
            if row["supervision_kind"] == "soft_distribution":
                evidence = row["distribution_evidence"]
                if row["distribution_source_kind"] not in SOFT_SOURCE_KINDS:
                    raise ValueError(f"line {line_number}: unsupported distribution source kind")
                if evidence["choice_ids"] != row["choice_ids"]:
                    raise ValueError(f"line {line_number}: evidence choice_ids do not match options order")
                if len(row["target_distribution"]) != len(row["options"]):
                    raise ValueError(f"line {line_number}: target length does not match options")
            elif row["supervision_kind"] != HARD_LABEL:
                raise ValueError(f"line {line_number}: unsupported supervision kind")
            examples.append(row)

    if not examples:
        raise ValueError("Expected at least one probability-v2 JSONL row")
    return examples


def targets_for_rows(rows: Sequence[dict[str, Any]]) -> tuple[Tensor, list[str], list[int | None]]:
    """Represent soft rows directly and hard labels as categorical one-hot targets."""
    distributions: list[list[float]] = []
    source_kinds: list[str] = []
    hard_labels: list[int | None] = []
    for row in rows:
        count = len(row["options"])
        if row["supervision_kind"] == "soft_distribution":
            distribution = [float(value) for value in row["target_distribution"]]
            total = math.fsum(distribution)
            if (len(distribution) != count or any(not math.isfinite(value) or value < 0 for value in distribution)
                    or not math.isclose(total, 1., rel_tol=0., abs_tol=2e-6)):
                raise ValueError("Soft targets must be valid distributions; only rounding drift up to 2e-6 is accepted")
            # Public typed sources round probabilities to six decimals. Keep their
            # stored values intact and remove that drift at the tensor boundary.
            distribution = [value / total for value in distribution]
            source = row["distribution_source_kind"]
            hard_label = None
        elif row["supervision_kind"] == HARD_LABEL:
            hard_label = row["answer_index"]
            distribution = [float(index == hard_label) for index in range(count)]
            source = HARD_LABEL
        else:
            raise ValueError(f"Unsupported supervision kind: {row['supervision_kind']!r}")
        distributions.append(distribution + [0.0] * (CAPACITY - count))
        source_kinds.append(source)
        hard_labels.append(hard_label)
    return torch.tensor(distributions, dtype=torch.float32), source_kinds, hard_labels


def soft_cross_entropy(logits: Tensor, targets: Tensor, counts: Tensor) -> Tensor:
    """Average categorical cross-entropy over valid choices in each question.

    Slicing each row before log-softmax keeps padded ``-inf`` logits out of the
    target product, including when target mass is exactly zero.
    """
    _validate_batch_shapes(logits, targets, counts)
    losses: list[Tensor] = []
    for index, count in enumerate(counts.tolist()):
        valid_logits = logits[index, :count]
        target = targets[index, :count]
        log_probabilities = F.log_softmax(valid_logits, dim=-1)
        losses.append(-(target * log_probabilities).sum())
    return torch.stack(losses).mean()


def categorical_metrics(
    logits: Tensor,
    targets: Tensor,
    counts: Tensor,
    source_kinds: Sequence[str],
    hard_labels: Sequence[int | None],
) -> dict[str, Any]:
    """Compute per-question categorical losses and label-source breakdowns."""
    _validate_batch_shapes(logits, targets, counts)
    if len(source_kinds) != logits.shape[0] or len(hard_labels) != logits.shape[0]:
        raise ValueError("source kinds and hard labels must match batch size")

    rows: list[dict[str, Any]] = []
    for index, count in enumerate(counts.tolist()):
        valid_logits = logits[index, :count]
        target = targets[index, :count]
        if not torch.isfinite(valid_logits).all():
            raise ValueError("valid-choice logits must be finite")
        if not torch.isfinite(target).all():
            raise ValueError("valid-choice targets must be finite")
        log_probabilities = F.log_softmax(valid_logits, dim=-1)
        probabilities = log_probabilities.exp()
        positive = target > 0
        cross_entropy = -(target * log_probabilities).sum()
        kl_divergence = (
            target[positive] * (target[positive].log() - log_probabilities[positive])
        ).sum()
        brier = (probabilities - target).square().sum()
        prediction = int(valid_logits.argmax().item())
        target_choice = int(target.argmax().item())
        rows.append(
            {
                "cross_entropy": float(cross_entropy.item()),
                "kl_pq": float(kl_divergence.item()),
                "brier": float(brier.item()),
                "argmax_agreement": float(prediction == target_choice),
                "hard_correct": (
                    float(prediction == hard_labels[index])
                    if hard_labels[index] is not None
                    else None
                ),
            }
        )

    result = _summarize_metric_rows(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_kind, row_metrics in zip(source_kinds, rows):
        grouped[source_kind].append(row_metrics)
    result["by_source_kind"] = {
        kind: {
            **_summarize_metric_rows(group_rows),
            "interpretation": (
                "agreement_with_teacher_estimate; not truth or calibration"
                if kind == "teacher_estimate"
                else "agreement_with_recorded_target_distribution"
                if kind != HARD_LABEL
                else "hard-label evaluation"
            ),
        }
        for kind, group_rows in sorted(grouped.items())
    }
    return result


def _summarize_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    result: dict[str, Any] = {
        "questions": count,
        "soft_cross_entropy": sum(row["cross_entropy"] for row in rows) / count,
        "kl_pq": sum(row["kl_pq"] for row in rows) / count,
        "categorical_brier_sum": sum(row["brier"] for row in rows) / count,
        "argmax_agreement": sum(row["argmax_agreement"] for row in rows) / count,
    }
    hard_rows = [row for row in rows if row["hard_correct"] is not None]
    if hard_rows:
        result["hard_label_questions"] = len(hard_rows)
        result["hard_accuracy"] = sum(row["hard_correct"] for row in hard_rows) / len(hard_rows)
    return result


def _validate_batch_shapes(logits: Tensor, targets: Tensor, counts: Tensor) -> None:
    if logits.ndim != 2 or targets.shape != logits.shape:
        raise ValueError("logits and target distributions must have matching [batch, capacity] shape")
    if counts.ndim != 1 or counts.shape[0] != logits.shape[0]:
        raise ValueError("counts must have one value per question")
    if ((counts < 2) | (counts > logits.shape[-1])).any():
        raise ValueError("every question must have at least two choices within capacity")
    if not torch.isfinite(targets).all() or (targets < 0).any():
        raise ValueError("targets must be finite and non-negative")
    for index, count in enumerate(counts.tolist()):
        valid_target = targets[index, :count]
        padded_target = targets[index, count:]
        if padded_target.numel() and torch.count_nonzero(padded_target).item():
            raise ValueError("padded choices must have zero target mass")
        # JSON values are checked at the schema's tighter tolerance before their
        # conversion to float32; summation then includes float32 representation error.
        if not math.isclose(float(valid_target.sum().item()), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError("each valid target distribution must sum to one")


def fit_head(
    features: Tensor,
    targets: Tensor,
    counts: Tensor,
    source_kinds: Sequence[str],
    hard_labels: Sequence[int | None],
    *,
    steps: int,
    learning_rate: float,
    seed: int,
) -> tuple[DecisionHead, dict[str, Any]]:
    torch.manual_seed(seed)
    head = DecisionHead(features.shape[-1], CAPACITY)
    initial_weights = head.score.weight.detach().clone()
    initial = categorical_metrics(head(features, counts), targets, counts, source_kinds, hard_labels)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0)
    history = [{"step": 0, **initial}]
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = soft_cross_entropy(head(features, counts), targets, counts)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite soft-target training loss")
        loss.backward()
        if any(parameter.grad is not None and not torch.isfinite(parameter.grad).all()
               for parameter in head.parameters()):
            raise RuntimeError("Non-finite decision-head gradient")
        optimizer.step()
        if step == steps or step % max(1, min(10, steps)) == 0:
            history.append(
                {"step": step, **categorical_metrics(head(features, counts), targets, counts,
                                                      source_kinds, hard_labels)}
            )
    head.eval()
    final = categorical_metrics(head(features, counts), targets, counts, source_kinds, hard_labels)
    return head, {
        "initial": initial,
        "final": final,
        "steps": steps,
        "weight_change_l2": (head.score.weight.detach() - initial_weights).norm().item(),
        "history": history,
    }


def load_cached_backbone(repo: str, revision: str, dtype: str):
    """Load only a local Hugging Face cache entry; never fetch model files."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        repo, revision=revision, local_files_only=True,
    )
    wrapper = AutoModel.from_pretrained(
        repo,
        revision=revision,
        dtype=getattr(torch, dtype),
        attn_implementation="sdpa",
        local_files_only=True,
    )
    backbone = wrapper.language_model if hasattr(wrapper, "language_model") else wrapper
    backbone.eval().requires_grad_(False)
    return tokenizer, backbone


def encode_examples(tokenizer, backbone, examples: Sequence[dict[str, Any]], max_tokens: int):
    vectors: list[Tensor] = []
    lengths: list[int] = []
    with torch.inference_mode():
        for index, example in enumerate(examples):
            tokens = tokenizer(format_input(example), return_tensors="pt", add_special_tokens=True)
            length = tokens["input_ids"].shape[1]
            if length > max_tokens:
                raise ValueError(f"{example['id']}: {length} tokens exceeds {max_tokens}; refusing truncation")
            attention_mask = tokens.get("attention_mask", torch.ones_like(tokens["input_ids"]))
            output = backbone(input_ids=tokens["input_ids"], attention_mask=attention_mask, use_cache=False)
            vectors.append(last_token(output.last_hidden_state, attention_mask).float().clone())
            lengths.append(length)
            print(f"Encoded {index + 1}/{len(examples)} ({length} tokens)", flush=True)
    return torch.cat(vectors).clone(), lengths


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    from experiment import CANDIDATES, save_head
    import transformers

    if args.model not in CANDIDATES:
        raise ValueError(f"Unknown model candidate: {args.model}")
    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc)
    candidate = CANDIDATES[args.model]
    train_rows = load_probability_jsonl(args.data)
    eval_rows = load_probability_jsonl(args.evaluation_data) if args.evaluation_data else train_rows
    revision = args.revision or candidate["revision"]
    output_dir = args.output or ROOT / "runs" / (
        f"probability-v2-softtarget-{timestamp.strftime('%Y-%m-%dT%H-%M-%SZ')}-{args.model}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "schema_version": 1,
        "started_at": timestamp.isoformat(),
        "model": args.model,
        "repo": candidate["repo"],
        "revision": revision,
        "device": "cpu",
        "dtype": args.dtype,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "max_steps": args.steps,
        "optimizer": "AdamW, weight_decay=0",
        "max_tokens": args.max_tokens,
        "capacity": CAPACITY,
        "format": "plain-context-question-choices-v1",
        "pooling": "last non-padding token",
        "backbone_trainable": False,
        "feature_dtype": "float32",
        "head": "existing non-affine LayerNorm followed by linear projection",
        "target_encoding": "soft distributions and hard-label one-hot distributions",
        "train_data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "evaluation_data_sha256": hashlib.sha256(
            args.evaluation_data.read_bytes() if args.evaluation_data else args.data.read_bytes()
        ).hexdigest(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    write_json(output_dir / "manifest.json", metadata)
    write_json(output_dir / "training_examples.json", train_rows)
    if args.evaluation_data:
        write_json(output_dir / "evaluation_examples.json", eval_rows)

    tokenizer, backbone = load_cached_backbone(candidate["repo"], revision, args.dtype)
    encode_started = time.perf_counter()
    train_features, train_lengths = encode_examples(tokenizer, backbone, train_rows, args.max_tokens)
    if args.evaluation_data:
        eval_features, eval_lengths = encode_examples(tokenizer, backbone, eval_rows, args.max_tokens)
    else:
        eval_features, eval_lengths = train_features, train_lengths
    encode_seconds = time.perf_counter() - encode_started
    del backbone, tokenizer
    torch.save(train_features, output_dir / "training_features.pt")

    train_targets, train_sources, train_hard = targets_for_rows(train_rows)
    train_counts = torch.tensor([len(row["options"]) for row in train_rows], dtype=torch.long)
    fit_started = time.perf_counter()
    head, training = fit_head(
        train_features,
        train_targets,
        train_counts,
        train_sources,
        train_hard,
        steps=args.steps,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    fit_seconds = time.perf_counter() - fit_started
    save_head(output_dir / "head.pt", head)

    eval_targets, eval_sources, eval_hard = targets_for_rows(eval_rows)
    eval_counts = torch.tensor([len(row["options"]) for row in eval_rows], dtype=torch.long)
    with torch.no_grad():
        evaluation = categorical_metrics(head(eval_features, eval_counts), eval_targets,
                                         eval_counts, eval_sources, eval_hard)
        predictions = head(eval_features, eval_counts).argmax(-1).tolist()
    is_illustrative_pilot = all(
        row.get("pilot_status") == probability_v2.EXCLUDED_PILOT_STATUS
        for row in train_rows + eval_rows
    )
    report = {
        "status": "completed",
        "scope": (
            "illustrative probability-v2 plumbing; no generalization claim"
            if is_illustrative_pilot
            else "frozen-backbone decision-head training/evaluation; no broad generalization claim"
        ),
        "pilot_status": probability_v2.EXCLUDED_PILOT_STATUS if is_illustrative_pilot else None,
        "evaluation_scope": "separate_probability_v2_file" if args.evaluation_data else "training_rows",
        "teacher_estimate_scope": "agreement with unverified estimates; not truth or calibration",
        "training_questions": len(train_rows),
        "evaluation_questions": len(eval_rows),
        "training_source_kinds": _counts(train_sources),
        "evaluation_source_kinds": _counts(eval_sources),
        "training": training,
        "evaluation": evaluation,
        "token_lengths": {"training": train_lengths, "evaluation": eval_lengths},
        "feature_extraction_seconds": encode_seconds,
        "head_training_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "predictions": [
            {
                "id": row["id"],
                "source_kind": source,
                "predicted_index": predicted,
                "target_argmax_index": int(target.argmax().item()),
                "target_distribution": target[:count].tolist(),
            }
            for row, source, predicted, target, count in zip(
                eval_rows, eval_sources, predictions, eval_targets, eval_counts.tolist()
            )
        ],
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "predictions"}, indent=2))
    print(f"Saved run: {output_dir}", flush=True)
    return 0


def _counts(source_kinds: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for source in source_kinds:
        result[source] = result.get(source, 0) + 1
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    training = subparsers.add_parser("run", help="fit and evaluate a frozen-backbone decision head")
    training.add_argument("--data", type=Path, default=ROOT / "data" / "generated_train" / "probability-v2-pilot.jsonl")
    training.add_argument("--evaluation-data", type=Path)
    training.add_argument("--output", type=Path)
    training.add_argument("--model", default="qwen35-reasoning")
    training.add_argument("--revision")
    training.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    training.add_argument("--max-tokens", type=int, default=384)
    training.add_argument("--steps", type=int, default=100)
    training.add_argument("--learning-rate", type=float, default=0.0005)
    training.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "run" and (args.steps < 1 or args.learning_rate <= 0 or args.max_tokens < 1):
        parser.error("--steps, --learning-rate, and --max-tokens must be positive")
    torch.set_num_threads(8)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
