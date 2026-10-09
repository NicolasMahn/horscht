"""Run aggregate-only overlap screening; benchmark content stays isolated."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from evaluation.isolation import (
    EXTERNAL_DATA_ROOT,
    BenchmarkIsolationError,
    sha256_file,
)
from evaluation.overlap import run_overlap_audit


def audit(snapshot: Path) -> dict:
    items = EXTERNAL_DATA_ROOT / "canonical/bundle/items.jsonl"
    output = EXTERNAL_DATA_ROOT / f"overlap-audit-{snapshot.name}"
    report = run_overlap_audit(
        snapshot / "rows.jsonl",
        snapshot / "manifest.json",
        items,
        output,
        benchmark_id="locked-independent-bundle",
        benchmark_revision=sha256_file(items),
        benchmark_items_sha256=sha256_file(items),
    )
    return {
        key: report[key]
        for key in (
            "training_rows_checked",
            "benchmark_rows_checked",
            "exact_prompt_overlap_training_ids",
            "near_lexical_overlap_training_ids",
            "near_candidate_pairs_truncated",
        )
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    args = parser.parse_args()
    try:
        print(json.dumps(audit(args.snapshot.resolve()), sort_keys=True))
    except (BenchmarkIsolationError, OSError, ValueError):
        # Suppress tracebacks that could include content from parser exceptions.
        raise SystemExit("overlap_audit_failed_safe") from None
