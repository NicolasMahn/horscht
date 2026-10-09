"""Aggregate receipts and real source/split runtime integration, without row previews."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from dataset_factory.new_dataset import ROOT, rows_at, sha, validate_snapshot_row
from dataset_factory.new_dataset_quarantine import FINAL as OUTPUT


def verify_runtime(
    snapshot: Path = OUTPUT, *, tokenizer_name: str = "qwen-instruct"
) -> dict:
    import torch
    from transformers import AutoTokenizer
    from experiment import CANDIDATES, format_input
    from paired_training import ordered_row
    from probability_experiment import targets_for_rows

    torch.set_num_threads(1)
    counts = Counter()
    representatives = {}
    selected = []
    for original in rows_at(snapshot / "rows.jsonl"):
        ordered = ordered_row(original, 20261008, 3)
        assert not validate_snapshot_row(ordered)
        assert ordered["context"] == original["context"]
        assert ordered["split"] == original["split"]
        key = (original["_new_dataset"]["source_id"], original["split"])
        counts[key] += 1
        if key not in representatives or len(format_input(ordered)) > len(
            format_input(representatives[key])
        ):
            representatives[key] = ordered
        selected.append(ordered)
        if len(selected) == 512:
            targets, _, _ = targets_for_rows(selected)
            assert torch.isfinite(targets).all()
            assert torch.allclose(
                targets.sum(-1), torch.ones(len(selected)), atol=2e-6, rtol=0
            )
            selected.clear()
    if selected:
        targets, _, _ = targets_for_rows(selected)
        assert torch.isfinite(targets).all()
        assert torch.allclose(
            targets.sum(-1), torch.ones(len(selected)), atol=2e-6, rtol=0
        )
    candidate = CANDIDATES[tokenizer_name]
    tokenizer = AutoTokenizer.from_pretrained(
        candidate["repo"], revision=candidate["revision"], local_files_only=True
    )
    tokenizer_counts = {}
    for key, ordered in sorted(representatives.items()):
        encoded = tokenizer(
            format_input(ordered), add_special_tokens=True, truncation=False
        )
        tokenizer_counts[":".join(key)] = len(encoded["input_ids"])
    return {
        "snapshot_manifest_sha256": sha(snapshot / "manifest.json"),
        "ordered_rows_checked": sum(counts.values()),
        "source_split_cells": len(counts),
        "cell_rows": {":".join(key): value for key, value in sorted(counts.items())},
        "tokenizer_representatives_checked": len(tokenizer_counts),
        "representative_token_counts": tokenizer_counts,
        "tokenizer": candidate,
        "context_truncations": 0,
        "errors": 0,
    }


def admission(snapshot: Path = OUTPUT) -> dict:
    manifest = json.loads((snapshot / "manifest.json").read_text())
    counts, choice_bands, languages, soft_kinds = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    non_onehot = Counter()
    highest = Counter()
    for row in rows_at(snapshot / "rows.jsonl"):
        if row["split"] != "train":
            continue
        source = row["_new_dataset"]["source_id"]
        choices = len(row["options"])
        counts[source] += 1
        band = (
            "2-4"
            if choices <= 4
            else "5-16"
            if choices <= 16
            else "17-64"
            if choices <= 64
            else "65-265"
        )
        choice_bands[band] += 1
        languages[row["language"]] += 1
        if row.get("supervision_kind") == "soft_distribution":
            soft_kinds[row["distribution_source_kind"]] += 1
            if sum(value > 0 for value in row["target_distribution"]) > 1:
                non_onehot[source] += 1
        if choices == 265:
            highest[source] += 1
    overlap_path = (
        ROOT
        / f"data/external_evaluation/overlap-audit-{snapshot.name}/overlap-report.json"
    )
    overlap = json.loads(overlap_path.read_text())
    return {
        "schema_version": "new-dataset-admission-v1",
        "snapshot": str(snapshot.relative_to(ROOT)),
        "manifest_sha256": sha(snapshot / "manifest.json"),
        "split_counts": manifest["rows"]["by_split"],
        "source_counts": manifest["source_selection"],
        "train_source_counts": dict(counts),
        "train_choice_bands": dict(choice_bands),
        "train_languages": dict(languages),
        "train_language_count": len(languages),
        "train_soft_source_kinds": dict(soft_kinds),
        "train_actual_non_onehot_targets": dict(non_onehot),
        "train_265_choices": dict(highest),
        "original_dev_rows": 7202,
        "original_test_rows": 7164,
        "new_dev_rows": manifest["rows"]["by_split"]["dev"] - 7202,
        "new_test_rows": manifest["rows"]["by_split"]["test"] - 7164,
        "source_files": manifest["source_files"],
        "artifacts": manifest["artifacts"],
        "overlap_report_sha256": sha(overlap_path),
        "overlap": {
            key: overlap[key]
            for key in (
                "training_rows_checked",
                "benchmark_rows_checked",
                "exact_prompt_overlap_training_ids",
                "near_lexical_overlap_training_ids",
                "near_candidate_pairs_truncated",
            )
        },
        "document_source_files": {
            str(path.relative_to(ROOT)): sha(path)
            for path in (
                ROOT / "data/document_luna_max/2026-10-02/documents.jsonl",
                ROOT / "data/document_breadth/2026-10-03/documents.jsonl",
                ROOT / "data/document_expansion/2026-10-02/documents.jsonl",
            )
        },
        "checkpoint_restriction": manifest["public_sources"]["checkpoint_restriction"],
        "coverage_limits": [
            *manifest["coverage_limits"],
            "Only 16 selected game rows have exactly 265 choices: 11 train, 5 development; none calibration/test.",
            "Document train choice counts range 3–12; no reviewed 265-choice document row is admitted.",
        ],
    }


if __name__ == "__main__":
    report = admission()
    report["runtime_integration"] = verify_runtime()
    output = ROOT / "data/newtraining/admission-2026-10-08-v4.json"
    if output.exists():
        raise SystemExit("admission_receipt_is_immutable")
    output.write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "receipt": str(output.relative_to(ROOT)),
                "split_counts": report["split_counts"],
                "runtime_rows": report["runtime_integration"]["ordered_rows_checked"],
                "source_split_cells": report["runtime_integration"][
                    "source_split_cells"
                ],
                "tokenizer_representatives": report["runtime_integration"][
                    "tokenizer_representatives_checked"
                ],
                "train_choice_bands": report["train_choice_bands"],
                "train_actual_non_onehot_targets": report[
                    "train_actual_non_onehot_targets"
                ],
                "language_count": report["train_language_count"],
                "train_265_choices": report["train_265_choices"],
            }
        )
    )
