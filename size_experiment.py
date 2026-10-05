"""Separate matched 16/64 training comparison, with one fixed 64-question holdout.

Run only after the original comparison finishes. Reuses frozen features within this
experiment; never reads or writes the original smoke experiment's result files.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import experiment as core

ROOT = Path(__file__).resolve().parent
SEEDS = list(range(42, 52))
SETTINGS = {
    "device": "cpu", "dtype": "bfloat16", "threads": 8,
    "learning_rate": 0.01, "max_steps": 1000, "max_tokens": 256,
    "capacity": core.CAPACITY, "optimizer": "AdamW, weight_decay=0",
    "format": "plain-context-question-choices-v1", "chat_template": False,
    "pooling": "last non-padding token", "backbone_trainable": False,
    "feature_dtype": "float32", "head": "non-affine LayerNorm followed by linear projection",
    "history_interval": 1,
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    # Interrupted writes must not destroy an otherwise resumable checkpoint.
    temporary = path.with_suffix(path.suffix + ".tmp")
    core.write_json(temporary, value)
    temporary.replace(path)


def partition(examples):
    train = [i for i, ex in enumerate(examples) if ex["split"] == "train"]
    probe = [i for i, ex in enumerate(examples) if ex["split"] == "probe"]
    if len(train) != 64 or len(probe) != 64:
        raise ValueError("Size experiment requires exactly 64 train and 64 probe examples")
    if any(type(ex.get("train_subset_16")) is not bool for ex in examples):
        raise ValueError("Every example requires a boolean train_subset_16 marker")
    if any(examples[i]["train_subset_16"] for i in probe):
        raise ValueError("Probe examples cannot belong to the training subset")
    small = [i for i in train if examples[i]["train_subset_16"]]
    if len(small) != 16:
        raise ValueError("The nested training subset must contain exactly 16 examples")
    return {"16": small, "64": train}, probe


def manifest(model, data, examples):
    import transformers

    subsets, probe = partition(examples)
    return {
        "schema_version": 1, "candidate": model, **core.CANDIDATES[model],
        "data_sha256": digest(data), "experiment_sha256": digest(core.__file__),
        "runner_sha256": digest(__file__), "settings": SETTINGS, "seeds": SEEDS,
        "subsets": {size: [examples[i]["id"] for i in indices] for size, indices in subsets.items()},
        "probe_ids": [examples[i]["id"] for i in probe],
        "torch": torch.__version__, "transformers": transformers.__version__,
        "selection_policy": "Fixed subsets, seeds and stopping rule; no probe-driven tuning",
    }


def lock_dataset(runs_root, metadata):
    """Record the shared, fixed protocol before the first feature extraction."""
    runs_root.mkdir(parents=True, exist_ok=True)
    path = runs_root / "dataset-lock.json"
    protocol = {key: metadata[key] for key in ("data_sha256", "subsets", "probe_ids", "settings", "seeds")}
    if path.exists():
        locked = json.loads(path.read_text())
        if locked["protocol"] != protocol:
            raise ValueError("Locked dataset or training protocol differs; use a fresh --runs-root")
    else:
        locked = {"locked_at": datetime.now(timezone.utc).isoformat(), "protocol": protocol}
        write_json(path, locked)
    return locked["locked_at"]


def prepare_cache(directory, metadata, examples):
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != metadata:
            raise ValueError("Existing size cache provenance differs; use a fresh --runs-root")
        if json.loads((directory / "examples.json").read_text()) != examples:
            raise ValueError("Cached examples differ from the locked dataset")
    else:
        write_json(manifest_path, metadata)
        write_json(directory / "examples.json", examples)
    feature_path = directory / "features.pt"
    cache_path = directory / "cache.json"
    if feature_path.exists() and cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if digest(feature_path) != cache["features_sha256"]:
            raise ValueError("Cached feature file checksum mismatch")
        features = torch.load(feature_path, map_location="cpu", weights_only=True)
    else:
        tokenizer, backbone = core.load_backbone(metadata["repo"], metadata["revision"], SETTINGS["dtype"])
        started = time.perf_counter()
        features, lengths = core.encode_examples(tokenizer, backbone, examples, SETTINGS["max_tokens"])
        cache = {"token_lengths": lengths, "feature_extraction_seconds": time.perf_counter() - started}
        del backbone, tokenizer
        temporary = feature_path.with_suffix(".tmp")
        torch.save(features, temporary)
        temporary.replace(feature_path)
        cache["features_sha256"] = digest(feature_path)
        write_json(cache_path, cache)
    if features.ndim != 2 or features.shape[0] != len(examples) or not torch.isfinite(features).all():
        raise ValueError("Invalid cached feature tensor")
    return features, cache


def fit_condition(directory, features, examples, train_indices, probe_indices, seed):
    directory.mkdir(parents=True, exist_ok=True)
    record_path = directory / "result.json"
    counts = torch.tensor([len(ex["options"]) for ex in examples])
    labels = torch.tensor([ex["answer_index"] for ex in examples])
    if record_path.exists():
        record = json.loads(record_path.read_text())
        if record["seed"] != seed or record["training_ids"] != [examples[i]["id"] for i in train_indices]:
            raise ValueError("Saved head does not match requested seed and training subset")
        if digest(directory / "head.pt") != record["head_sha256"]:
            raise ValueError("Saved head checksum mismatch")
        return record
    started = time.perf_counter()
    head, training = core.fit_head(
        features[train_indices], labels[train_indices], counts[train_indices],
        steps=SETTINGS["max_steps"], learning_rate=SETTINGS["learning_rate"], seed=seed,
        probe=(features[probe_indices], labels[probe_indices], counts[probe_indices]),
    )
    seconds = time.perf_counter() - started
    core.save_head(directory / "head.pt", head)
    reloaded = core.load_head(directory / "head.pt")
    with torch.no_grad():
        logits = head(features, counts)
        equal = torch.equal(logits, reloaded(features, counts))
        probabilities = logits.softmax(-1)
        predictions = logits.argmax(-1)
    passed = (training["final"]["accuracy"] == 1.0 and training["final"]["loss"] <= 0.02
              and training["weight_change_l2"] > 0 and equal)
    active = sorted(train_indices + probe_indices)
    record = {
        "seed": seed, "training_ids": [examples[i]["id"] for i in train_indices],
        "head_sha256": digest(directory / "head.pt"),
        "report": {
            "passed": passed, "training_examples": len(train_indices), "probe_examples": len(probe_indices),
            "training": training, "probe": core.metrics(head, features[probe_indices], labels[probe_indices], counts[probe_indices]),
            "reloaded_head_logits_identical": equal, "head_training_seconds": seconds,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "predictions": [
                {"id": examples[i]["id"], "split": examples[i]["split"],
                 "expected": examples[i]["answer_index"], "predicted": predictions[i].item(),
                 "probabilities": probabilities[i, :counts[i]].tolist()} for i in active
            ],
        },
    }
    write_json(record_path, record)
    return record


def verify_cache(directory):
    """Fresh process/backbone verification, including every saved head."""
    metadata = json.loads((directory / "manifest.json").read_text())
    examples = core.load_examples(directory / "examples.json")
    tokenizer, backbone = core.load_backbone(metadata["repo"], metadata["revision"], SETTINGS["dtype"])
    fresh, _ = core.encode_examples(tokenizer, backbone, examples[:2], SETTINGS["max_tokens"])
    cached = torch.load(directory / "features.pt", map_location="cpu", weights_only=True)[:2]
    counts = torch.tensor([len(ex["options"]) for ex in examples[:2]])
    heads = []
    with torch.no_grad():
        for size in ("16", "64"):
            for seed in SEEDS:
                head = core.load_head(directory / size / str(seed) / "head.pt")
                heads.append({"size": int(size), "seed": seed,
                              "predictions_identical": torch.equal(head(fresh, counts).argmax(-1), head(cached, counts).argmax(-1))})
    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "example_ids": [ex["id"] for ex in examples[:2]], "heads": heads,
        "max_feature_difference": (fresh - cached).abs().max().item(),
        "features_close": torch.allclose(fresh, cached, atol=0.001, rtol=0.001),
        "predictions_identical": all(head["predictions_identical"] for head in heads),
    }
    write_json(directory / "reload_verification.json", result)
    return 0 if result["features_close"] and result["predictions_identical"] else 1


def run_model(model, data, runs_root, results_root):
    directory = runs_root / model
    bundle = {"schema_version": 1, "experiment": "matched-training-size", "run_directory": str(directory.resolve()),
              "status": "failed", "eligible": False, "conditions": {}}
    results_root.mkdir(parents=True, exist_ok=True)
    try:
        examples = core.load_examples(data)
        subsets, probe = partition(examples)
        metadata = manifest(model, data, examples)
        metadata["dataset_locked_at"] = lock_dataset(runs_root, metadata)
        bundle.update(manifest=metadata, examples=examples)
        features, cache = prepare_cache(directory, metadata, examples)
        bundle["feature_cache"] = cache
        for size, indices in subsets.items():
            condition = {"training_ids": metadata["subsets"][size], "runs": []}
            bundle["conditions"][size] = condition
            for seed in SEEDS:
                print(f"{model}: {size} train examples, seed {seed}", flush=True)
                condition["runs"].append(fit_condition(directory / size / str(seed), features, examples, indices, probe, seed))
        # Verification reloads the backbone in a separate interpreter, not the extraction instance.
        with (directory / "verification.log").open("w") as log:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), "verify", str(directory)], stdout=log, stderr=subprocess.STDOUT)
        bundle["verification_exit_code"] = completed.returncode
        verification_path = directory / "reload_verification.json"
        bundle["cache_verification"] = json.loads(verification_path.read_text()) if verification_path.exists() else None
        bundle["status"] = "complete" if completed.returncode == 0 else "failed"
        bundle["eligible"] = completed.returncode == 0 and all(
            record["report"]["passed"] for condition in bundle["conditions"].values() for record in condition["runs"])
        if not bundle["eligible"]:
            bundle["error"] = "One or more memorization or reload checks failed; not eligible for selection"
    except Exception as error:
        bundle["error"] = f"{type(error).__name__}: {error}"
    write_json(results_root / f"{model}.json", bundle)
    print(f"{model}: status={bundle['status']}, eligible={bundle['eligible']}", flush=True)
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--models", nargs="+", choices=core.CANDIDATES, default=list(core.CANDIDATES))
    run.add_argument("--data", type=Path, default=ROOT / "data" / "expanded.json")
    run.add_argument("--runs-root", type=Path, default=ROOT / "runs" / "size-comparison")
    run.add_argument("--results-root", type=Path, default=ROOT / "results" / "size_comparison")
    verify = commands.add_parser("verify")
    verify.add_argument("directory", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(SETTINGS["threads"])
    if args.command == "verify":
        return verify_cache(args.directory)
    results = [run_model(model, args.data, args.runs_root, args.results_root) for model in args.models]
    return 0 if all(result["eligible"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
