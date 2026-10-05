"""Predeclared development-only order diagnostics on the frozen selected model."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModel, AutoTokenizer

from experiment import DecisionHead
from paired_training import (
    choice_order_risks,
    ordered_row,
    require_memory,
    require_training_snapshot,
    sha256,
)
from scalable_training import build_epoch_plan, evaluate_batches, ordered_lengths
from training_checkpoint import load_training_checkpoint
from training_loop import atomic_json


def consistency(first, second, identifiers):
    distances = []
    agreements = []
    for identifier in identifiers:
        ids, probabilities = first[identifier]
        other_ids, other = second[identifier]
        if set(ids) != set(other_ids):
            raise ValueError("Choice identities changed between development orders")
        aligned = other[[other_ids.index(choice) for choice in ids]]
        distances.append(float((probabilities - aligned).abs().sum() / 2))
        agreements.append(int(probabilities.argmax() == aligned.argmax()))
    return {
        "questions": len(distances),
        "mean_aligned_total_variation": sum(distances) / len(distances)
        if distances
        else None,
        "stable_choice_argmax_fraction": sum(agreements) / len(agreements)
        if agreements
        else None,
    }


def development_rows(snapshot: Path, identifiers: list[str]):
    """Require development membership before allocating a model."""
    wanted = set(identifiers)
    if not wanted or len(wanted) != len(identifiers):
        raise ValueError("A nonempty unique development cohort is required")
    with (snapshot / "splits.jsonl").open() as stream:
        splits = {
            row["id"]: row["split"]
            for row in map(json.loads, stream)
            if row["id"] in wanted
        }
    if set(splits) != wanted or set(splits.values()) != {"dev"}:
        raise ValueError("Order diagnostics accept development rows only")
    with (snapshot / "rows.jsonl").open() as stream:
        rows = {
            row["id"]: row for row in map(json.loads, stream) if row["id"] in wanted
        }
    if set(rows) != wanted:
        raise ValueError("The development cohort is missing source rows")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=12)
    cli = parser.parse_args()
    out = cli.run.resolve()
    report = json.loads((out / "full/report.json").read_text())
    if (
        report["status"] != "completed"
        or report["selected_checkpoint_reload_verified"] is not True
    ):
        raise ValueError("A completed, reload-verified selected checkpoint is required")
    manifest = json.loads((out / "full/manifest.json").read_text())
    checkpoint = (out / "full" / report["best_checkpoint_relative"]).resolve()
    if not checkpoint.is_relative_to((out / "full/checkpoints").resolve()):
        raise ValueError("Selected checkpoint escapes its run directory")
    snapshot = cli.snapshot.resolve()
    require_training_snapshot(snapshot)
    if sha256(snapshot / "manifest.json") != manifest["snapshot_manifest_sha256"]:
        raise ValueError("Training snapshot manifest checksum changed")
    snapshot_manifest = json.loads((snapshot / "manifest.json").read_text())
    for name in ("rows.jsonl", "splits.jsonl"):
        if sha256(snapshot / name) != snapshot_manifest["artifacts"][name]["sha256"]:
            raise ValueError("Training snapshot artifact checksum changed")
    identifiers = json.loads((out / "full/order-consistency-dev-ids.json").read_text())
    rows = development_rows(snapshot, identifiers)
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError("The pinned ROCm GPU environment is required")
    torch.set_num_threads(2)
    _, total = torch.cuda.mem_get_info()
    torch.cuda.set_per_process_memory_fraction(
        manifest["gpu_allocation_cap_gib"] * 1024**3 / total
    )
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    require_memory(
        manifest["backbone_parameter_count"] * 4 + 6 * 1024**3, cli.reserve_gib, "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        manifest["repo"], revision=manifest["revision"], local_files_only=True
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        manifest["repo"],
        revision=manifest["revision"],
        local_files_only=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
        device_map="cuda",
    )
    backbone._horscht_attention_backend = "flash"
    backbone.requires_grad_(False).eval()
    head = (
        DecisionHead(backbone.config.hidden_size)
        .to("cuda")
        .requires_grad_(False)
        .eval()
    )
    load_training_checkpoint(checkpoint, backbone=backbone, head=head)
    args = SimpleNamespace(max_tokens=manifest["max_tokens"], device="cuda")
    predictions = {}
    rendered = {}
    metrics = {}
    for epoch in (-1, -3):
        epoch_rows = [
            ordered_row(rows[identifier], manifest["seed"], epoch)
            for identifier in identifiers
        ]
        lengths = ordered_lengths(
            list(rows.values()),
            tokenizer,
            manifest["seed"],
            epoch,
            manifest["max_tokens"],
        )
        plan = build_epoch_plan(
            lengths,
            seed=manifest["seed"],
            epoch=epoch,
            microbatch=1,
            accumulation_questions=manifest["accumulation_questions"],
            max_batch_tokens=manifest["max_batch_tokens"],
            bucket_questions=manifest["bucket_questions"],
        )
        metrics[str(epoch)], predictions[epoch] = evaluate_batches(
            tokenizer, backbone, head, epoch_rows, plan, args, None, True
        )
        rendered[epoch] = {row["id"]: row["choice_ids"] for row in epoch_rows}
    groups = {
        "all": identifiers,
        "changed_order": [i for i in identifiers if rendered[-1][i] != rendered[-3][i]],
        "fixed_order_controls": [i for i in identifiers if choice_order_risks(rows[i])],
        "identical_seeded_order_controls": [
            i
            for i in identifiers
            if not choice_order_risks(rows[i]) and rendered[-1][i] == rendered[-3][i]
        ],
    }
    result = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "stage": "development only; selected checkpoint frozen before this diagnostic",
        "checkpoint_metadata_sha256": sha256(checkpoint / "checkpoint.json"),
        "cohort_ids_sha256": sha256(out / "full/order-consistency-dev-ids.json"),
        "scope": "changed-order sensitivity is reported separately from identical-input controls; no held or external inputs",
        "orders": metrics,
        "consistency_by_group": {
            name: consistency(predictions[-1], predictions[-3], ids)
            for name, ids in groups.items()
        },
    }
    atomic_json(out / "selected-development-order-diagnostics.json", result)
    print(
        json.dumps(
            {
                "completed_at": result["completed_at"],
                "consistency_by_group": result["consistency_by_group"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
