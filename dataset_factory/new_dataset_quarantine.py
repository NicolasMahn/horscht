"""Derive an immutable snapshot by quarantining complete unsupported groups."""

from __future__ import annotations
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from dataset_factory.new_dataset import ROOT, rows_at, sha, bands
from dataset_factory.training_snapshot import _atomic_json, _atomic_jsonl

PARENT = ROOT / "data/training_snapshots/2026-10-08-v3"
FINAL = ROOT / "data/training_snapshots/2026-10-08-v4"
TOKEN_AUDIT = ROOT / "runs/new-data-last-block-2026-10-08/token-feasibility-v3.json"


def derive(
    parent: Path = PARENT, output: Path = FINAL, token_audit: Path = TOKEN_AUDIT
) -> dict:
    if output.exists():
        raise ValueError("derived_snapshot_directory_is_immutable")
    manifest = json.loads((parent / "manifest.json").read_text())
    audit = json.loads(token_audit.read_text())
    if audit["snapshot_manifest_sha256"] != sha(parent / "manifest.json"):
        raise ValueError("token_audit_parent_hash_mismatch")
    window = audit["model_max_position_embeddings"]
    rows = list(rows_at(parent / "rows.jsonl"))
    lengths = audit["lengths_by_id"]
    groups = {
        row["_training_snapshot"]["snapshot_group_id"]
        for row in rows
        if lengths[row["id"]] > window
    }
    quarantined = [
        row for row in rows if row["_training_snapshot"]["snapshot_group_id"] in groups
    ]
    rows = [
        row
        for row in rows
        if row["_training_snapshot"]["snapshot_group_id"] not in groups
    ]
    assert all(lengths[row["id"]] <= window for row in rows)
    retained_ids = {row["id"] for row in rows}
    splits = [
        entry
        for entry in rows_at(parent / "splits.jsonl")
        if entry["id"] in retained_ids
    ]
    exclusions = list(rows_at(parent / "excluded.jsonl"))
    exclusions.extend(
        {
            "id": row["id"],
            "snapshot_group_id": row["_training_snapshot"]["snapshot_group_id"],
            "source_file": row["_new_dataset"]["source_file"],
            "original_split": row["split"],
            "reasons": ["linked_group_contains_context_above_native_model_window"],
            "native_model_window": window,
            "tokens": lengths[row["id"]],
        }
        for row in quarantined
    )
    output.mkdir(parents=True)
    for name, values in [
        ("rows.jsonl", rows),
        ("splits.jsonl", splits),
        ("excluded.jsonl", exclusions),
    ]:
        manifest["artifacts"][name] = {
            "records": len(values),
            "sha256": _atomic_jsonl(output / name, values),
        }
    shutil.copyfile(
        parent / "original-split-registry.jsonl",
        output / "original-split-registry.jsonl",
    )
    manifest.update(
        version=output.name,
        created_at=datetime.now(timezone.utc).isoformat(),
        parent_snapshot={
            "path": str(parent.relative_to(ROOT)),
            "manifest_sha256": sha(parent / "manifest.json"),
        },
        native_context_admission={
            "model_max_position_embeddings": window,
            "token_audit_path": str(token_audit.relative_to(ROOT)),
            "token_audit_sha256": sha(token_audit),
            "policy": "Quarantine whole linked group if any complete formatted context exceeds native window; never crop.",
            "quarantined_rows": len(quarantined),
            "quarantined_groups": len(groups),
            "quarantined_split_counts": dict(
                Counter(row["split"] for row in quarantined)
            ),
            "all_retained_rows_within_native_window": True,
        },
    )
    manifest["rows"] = {
        "training_eligible_rows": len(rows),
        "by_split": dict(Counter(row["split"] for row in rows)),
    }
    for source, selection in manifest["source_selection"].items():
        selected = [row for row in rows if row["_new_dataset"]["source_id"] == source]
        selection["selected"] = len(selected)
        selection["split_counts"] = dict(Counter(row["split"] for row in selected))
        selection["native_window_quarantine_rows"] = sum(
            row["_new_dataset"]["source_id"] == source for row in quarantined
        )
    for split, distribution in manifest["distributions"].items():
        selected = [row for row in rows if row["split"] == split]
        distribution.update(
            rows=len(selected),
            source=dict(Counter(row["_new_dataset"]["source_id"] for row in selected)),
            language=dict(Counter(row["language"] for row in selected)),
            choices=dict(
                sorted(
                    Counter(str(len(row["options"])) for row in selected).items(),
                    key=lambda x: int(x[0]),
                )
            ),
            context_band=dict(Counter(bands(row)[1] for row in selected)),
            target_kind=dict(
                Counter(row["_training_snapshot"]["target_kind"] for row in selected)
            ),
            task=dict(
                Counter(
                    row.get("operation_id", row.get("family", "unknown"))
                    for row in selected
                )
            ),
        )
    _atomic_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    manifest = derive()
    print(
        json.dumps(
            {
                "snapshot": str(FINAL.relative_to(ROOT)),
                "split_counts": manifest["rows"]["by_split"],
                "native_context_admission": manifest["native_context_admission"],
                "manifest_sha256": sha(FINAL / "manifest.json"),
            }
        )
    )
