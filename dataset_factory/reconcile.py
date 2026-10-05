"""Match completed Luna output and skip files back to durable assignment plans."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT_DIR = Path("data/generated_train/assignments")
QUESTION_DIR = Path("data/generated_train/broad-v1")


def _read_records(path: Path, keys: tuple[str, ...]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            identifier = next((row.get(key) for key in keys if isinstance(row.get(key), str)), None)
            if identifier:
                records[identifier] = row
    return records


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def reconcile_shard(plan_path: Path, question_dir: Path, root: Path = ROOT) -> dict[str, Any]:
    """Close each plan with a corresponding output row or explicit skip record."""
    rows: list[dict[str, Any]] = []
    with plan_path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid plan JSON at {plan_path}:{line_no}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"plan row at {plan_path}:{line_no} needs an id")
            rows.append(row)

    if not rows:
        raise ValueError(f"assignment plan is empty: {plan_path}")
    shard_id = plan_path.stem.removeprefix("shard-")
    output = rows[0].get("output")
    if isinstance(output, str) and output:
        output_path = Path(output)
        question_path = output_path if output_path.is_absolute() else root / output_path
    else:
        question_path = question_dir / f"luna-{shard_id}.jsonl"
    skip_path = plan_path.parent / f"skipped-luna-{shard_id}.jsonl"
    output_records = _read_records(question_path, ("id",))
    skipped_records = _read_records(skip_path, ("id", "plan_id"))
    output_ids = set(output_records)
    skipped_ids = set(skipped_records)
    overlap = output_ids & skipped_ids
    if overlap:
        raise ValueError(f"IDs appear in both output and skip files for {plan_path}: {len(overlap)}")

    counts: Counter[str] = Counter()
    pending = 0
    for row in rows:
        identifier = row["id"]
        previous = row.get("status")
        if previous == "superseded":
            counts["superseded"] += 1
            continue
        if identifier in output_ids:
            row["status"] = "completed"
            row.pop("skip_reason", None)
            counts["completed"] += 1
        elif identifier in skipped_ids:
            row["status"] = "skipped"
            reason = skipped_records[identifier].get("reason")
            if isinstance(reason, str) and reason.strip():
                row["skip_reason"] = reason.strip()
            counts["skipped"] += 1
        elif previous in {"completed", "skipped"}:
            counts[previous] += 1
        else:
            pending += 1
            counts[previous if isinstance(previous, str) else "pending"] += 1

    _atomic_jsonl(plan_path, rows)
    return {
        "plan": plan_path.as_posix(),
        "output": question_path.as_posix(),
        "skip_file": skip_path.as_posix(),
        "plan_rows": len(rows),
        "pending": pending,
        "statuses": dict(sorted(counts.items())),
    }


def reconcile_all(root: Path = ROOT, assignment_dir: Path = ASSIGNMENT_DIR, question_dir: Path = QUESTION_DIR) -> dict[str, Any]:
    """Reconcile every numbered assignment shard and summarize its durable state."""
    resolved_assignment_dir = root / assignment_dir
    resolved_question_dir = root / question_dir
    shards = sorted(resolved_assignment_dir.glob("shard-*.jsonl"))
    results = [reconcile_shard(path, resolved_question_dir, root) for path in shards]
    totals: Counter[str] = Counter()
    planned = 0
    pending = 0
    for result in results:
        planned += result["plan_rows"]
        pending += result["pending"]
        totals.update(result["statuses"])
    return {
        "plan_files": len(results),
        "planned_rows": planned,
        "pending_rows": pending,
        "statuses": dict(sorted(totals.items())),
        "shards": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--assignment-dir", type=Path, default=ASSIGNMENT_DIR)
    parser.add_argument("--question-dir", type=Path, default=QUESTION_DIR)
    args = parser.parse_args(argv)
    result = reconcile_all(args.root.resolve(), args.assignment_dir, args.question_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
