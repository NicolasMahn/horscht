"""Frozen-backbone decision-head experiment. No text generation or backbone training."""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).parent
CAPACITY = 265
CANDIDATES = json.loads((ROOT / "candidates.json").read_text())


def reject_external_training_input(path):
    if Path(path).resolve().is_relative_to((ROOT / "data" / "external_evaluation").resolve()):
        raise ValueError("External evaluation artifacts cannot be loaded by training utilities")


def load_examples(path):
    reject_external_training_input(path)
    examples = json.loads(Path(path).read_text())
    if not isinstance(examples, list) or not examples:
        raise ValueError("Expected a nonempty list of examples")
    seen = set()
    for example in examples:
        if example["id"] in seen:
            raise ValueError("Example IDs must be unique")
        seen.add(example["id"])
        if not all(isinstance(example[key], str) and example[key].strip()
                   for key in ("context", "question")):
            raise ValueError("Context and question must be nonempty strings")
        options = example["options"]
        if not 2 <= len(options) <= CAPACITY or len(set(options)) != len(options):
            raise ValueError("Expected 2–265 distinct options")
        if not all(isinstance(option, str) and option for option in options):
            raise ValueError("Options must be nonempty strings")
        if type(example["answer_index"]) is not int or not 0 <= example["answer_index"] < len(options):
            raise ValueError("Answer index is outside the available choices")
        if example["split"] not in ("train", "probe"):
            raise ValueError("Unknown split")
    if not any(example["split"] == "train" for example in examples):
        raise ValueError("At least one training example is required")
    return examples


def format_input(example):
    # Labels, explanations, IDs, and split names must never reach the backbone.
    choices = "\n".join(f"{i}: {text}" for i, text in enumerate(example["options"]))
    return (f"Context: {example['context']}\nQuestion: {example['question']}\n"
            f"Choices:\n{choices}\nSelected choice:")


def last_token(hidden, attention_mask):
    positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)
    indices = positions.expand_as(attention_mask).masked_fill(~attention_mask.bool(), -1).max(1).values
    if (indices < 0).any():
        raise ValueError("Cannot pool an empty input")
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


def mask_choices(logits, counts):
    if ((counts < 2) | (counts > logits.shape[-1])).any():
        raise ValueError("Invalid number of choices")
    unavailable = torch.arange(logits.shape[-1], device=logits.device)[None, :] >= counts[:, None]
    return logits.masked_fill(unavailable, -torch.inf)


class DecisionHead(nn.Module):
    def __init__(self, hidden_size, capacity=CAPACITY):
        super().__init__()
        self.normalize = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.score = nn.Linear(hidden_size, capacity)

    def forward(self, features, counts):
        return mask_choices(self.score(self.normalize(features)), counts)


def metrics(head, features, labels, counts):
    with torch.no_grad():
        logits = head(features, counts)
        return {
            "loss": F.cross_entropy(logits, labels).item(),
            "accuracy": (logits.argmax(-1) == labels).float().mean().item(),
        }


def fit_head(features, labels, counts, *, steps=1000, learning_rate=0.01, seed=42, probe=None):
    torch.manual_seed(seed)
    head = DecisionHead(features.shape[-1])
    initial_weights = head.score.weight.detach().clone()
    initial = metrics(head, features, labels, counts)
    def observe(step):
        point = {"step": step, **metrics(head, features, labels, counts)}
        if probe is not None:
            point.update({f"probe_{key}": value for key, value in metrics(head, *probe).items()})
        return point

    history = [observe(0)]
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate, weight_decay=0)
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(head(features, counts), labels)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        optimizer.step()
        current = observe(step)
        history.append(current)
        if step % 10 == 0 or step == steps:
            if current["accuracy"] == 1.0 and current["loss"] <= 0.02:
                break
    head.eval()
    return head, {
        "initial": initial,
        "final": metrics(head, features, labels, counts),
        "steps": step,
        "weight_change_l2": (head.score.weight.detach() - initial_weights).norm().item(),
        "history": history,
    }


def load_backbone(repo, revision, dtype):
    from transformers import AutoModel, AutoTokenizer

    print(f"Loading {repo}@{revision} on CPU ({dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(repo, revision=revision)
    wrapper = AutoModel.from_pretrained(
        repo, revision=revision, dtype=getattr(torch, dtype), attn_implementation="sdpa",
    )
    # The multimodal checkpoint retains the required input embeddings here.
    backbone = wrapper.language_model if hasattr(wrapper, "language_model") else wrapper
    backbone.eval().requires_grad_(False)
    return tokenizer, backbone


def encode_examples(tokenizer, backbone, examples, max_tokens):
    vectors = []
    lengths = []
    with torch.inference_mode():
        for i, example in enumerate(examples):
            tokens = tokenizer(format_input(example), return_tensors="pt", add_special_tokens=True)
            length = tokens["input_ids"].shape[1]
            if length > max_tokens:
                raise ValueError(f"{example['id']}: {length} tokens exceeds {max_tokens}; refusing truncation")
            input_ids = tokens["input_ids"]
            attention_mask = tokens.get("attention_mask", torch.ones_like(input_ids))
            output = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            vectors.append(last_token(output.last_hidden_state, attention_mask).float().clone())
            lengths.append(length)
            print(f"Encoded {i + 1}/{len(examples)} ({length} tokens)", flush=True)
    # A normal tensor is needed for the trainable head's backward pass.
    return torch.cat(vectors).clone(), lengths


def save_head(path, head):
    torch.save({
        "schema_version": 1,
        "hidden_size": head.score.in_features,
        "capacity": head.score.out_features,
        "state_dict": head.state_dict(),
    }, path)


def load_head(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["schema_version"] != 1:
        raise ValueError("Unsupported head checkpoint version")
    head = DecisionHead(checkpoint["hidden_size"], checkpoint["capacity"])
    head.load_state_dict(checkpoint["state_dict"])
    return head.eval()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(args):
    from huggingface_hub import HfApi
    import transformers

    started = time.perf_counter()
    timestamp = datetime.now(timezone.utc)
    candidate = CANDIDATES[args.model]
    examples = load_examples(args.data)
    output_dir = args.output or ROOT / "runs" / f"{timestamp.strftime('%Y-%m-%dT%H-%M-%SZ')}-{args.model}"
    output_dir.mkdir(parents=True, exist_ok=False)
    revision = HfApi().model_info(candidate["repo"], revision=args.revision or candidate["revision"]).sha
    metadata = {
        "schema_version": 1,
        "started_at": timestamp.isoformat(),
        "candidate": args.model,
        "repo": candidate["repo"],
        "revision": revision,
        "device": "cpu",
        "dtype": args.dtype,
        "threads": args.threads,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "max_steps": args.steps,
        "optimizer": "AdamW, weight_decay=0",
        "max_tokens": args.max_tokens,
        "capacity": CAPACITY,
        "format": "plain-context-question-choices-v1",
        "chat_template": False,
        "pooling": "last non-padding token",
        "backbone_trainable": False,
        "feature_dtype": "float32",
        "head": "non-affine LayerNorm followed by linear projection",
        "history_interval": 1,
        "experiment_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    write_json(output_dir / "manifest.json", metadata)
    write_json(output_dir / "examples.json", examples)
    tokenizer, backbone = load_backbone(candidate["repo"], revision, args.dtype)
    extract_started = time.perf_counter()
    features, lengths = encode_examples(tokenizer, backbone, examples, args.max_tokens)
    extract_seconds = time.perf_counter() - extract_started
    del backbone, tokenizer
    torch.save(features, output_dir / "features.pt")
    labels = torch.tensor([example["answer_index"] for example in examples])
    counts = torch.tensor([len(example["options"]) for example in examples])
    train = torch.tensor([example["split"] == "train" for example in examples])
    fit_started = time.perf_counter()
    head, training = fit_head(features[train], labels[train], counts[train],
                              steps=args.steps, learning_rate=args.learning_rate, seed=args.seed,
                              probe=(features[~train], labels[~train], counts[~train]) if (~train).any() else None)
    fit_seconds = time.perf_counter() - fit_started
    save_head(output_dir / "head.pt", head)
    reloaded = load_head(output_dir / "head.pt")
    with torch.no_grad():
        logits = head(features, counts)
        reload_equal = torch.equal(logits, reloaded(features, counts))
        probabilities = logits.softmax(-1)
        predictions = logits.argmax(-1).tolist()
    passed = (training["final"]["accuracy"] == 1.0 and training["final"]["loss"] <= 0.02
              and training["weight_change_l2"] > 0 and reload_equal)
    report = {
        "passed": passed,
        "scope": "head-only memorization; not backbone adaptation or generalization",
        "training_examples": int(train.sum()),
        "probe_examples": int((~train).sum()),
        "exercised_output_positions": sorted(set(labels.tolist())),
        "training": training,
        "probe": metrics(head, features[~train], labels[~train], counts[~train]) if (~train).any() else None,
        "reloaded_head_logits_identical": reload_equal,
        "token_lengths": lengths,
        "feature_extraction_seconds": extract_seconds,
        "head_training_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "predictions": [{"id": ex["id"], "split": ex["split"], "expected": ex["answer_index"],
                         "predicted": prediction, "probabilities": probabilities[i, :counts[i]].tolist()}
                        for i, (ex, prediction) in enumerate(zip(examples, predictions))],
    }
    write_json(output_dir / "report.json", report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("predictions", "training")}, indent=2))
    print(f"Training: {training['initial']} -> {training['final']}", flush=True)
    print(f"Saved run: {output_dir}", flush=True)
    return 0 if passed else 1


def verify(args):
    metadata = json.loads((args.run / "manifest.json").read_text())
    examples = load_examples(args.run / "examples.json")[:2]
    tokenizer, backbone = load_backbone(metadata["repo"], metadata["revision"], metadata["dtype"])
    fresh, _ = encode_examples(tokenizer, backbone, examples, metadata["max_tokens"])
    cached = torch.load(args.run / "features.pt", weights_only=True)[:2]
    head = load_head(args.run / "head.pt")
    counts = torch.tensor([len(ex["options"]) for ex in examples])
    with torch.no_grad():
        fresh_logits, cached_logits = head(fresh, counts), head(cached, counts)
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "examples": [ex["id"] for ex in examples],
        "max_feature_difference": (fresh - cached).abs().max().item(),
        "features_close": torch.allclose(fresh, cached, atol=0.001, rtol=0.001),
        "predictions_identical": torch.equal(fresh_logits.argmax(-1), cached_logits.argmax(-1)),
    }
    write_json(args.run / "reload_verification.json", result)
    print(json.dumps(result, indent=2))
    return 0 if result["features_close"] and result["predictions_identical"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("candidates")
    training = subparsers.add_parser("run")
    training.add_argument("--model", choices=CANDIDATES, default="ministral-base")
    training.add_argument("--revision", help="Override the pinned candidate revision")
    training.add_argument("--data", type=Path, default=ROOT / "data" / "smoke.json")
    training.add_argument("--output", type=Path)
    training.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    training.add_argument("--max-tokens", type=int, default=256)
    training.add_argument("--steps", type=int, default=1000)
    training.add_argument("--learning-rate", type=float, default=0.01)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--threads", type=int, default=8)
    verification = subparsers.add_parser("verify")
    verification.add_argument("run", type=Path)
    verification.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.command == "candidates":
        print(json.dumps(CANDIDATES, indent=2))
        return 0
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "run" and (args.steps < 1 or args.learning_rate <= 0):
        parser.error("--steps and --learning-rate must be positive")
    torch.set_num_threads(args.threads)
    return run(args) if args.command == "run" else verify(args)


if __name__ == "__main__":
    raise SystemExit(main())
