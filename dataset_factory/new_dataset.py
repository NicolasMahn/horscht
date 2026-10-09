"""Freeze source-aware, group-safe data while preserving a warm-start corpus."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from dataset_factory import probability_v2, progress, training_snapshot
from dataset_factory.probability_v2 import shuffle_choices
from dataset_factory.game_data import validate_game_row
from dataset_factory.new_public_data import PUBLIC_ROOT, validate_public_row
from evaluation.isolation import canonical_prompt_fingerprint

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "data/training_snapshots/2026-10-01-v1"
OUTPUT = ROOT / "data/training_snapshots/2026-10-08-v3"
SEED = 20261008
CAPS = {
    "original_replay": 8000,
    "games": 8000,
    "documents": 8000,
    "typed-decisions": 2400,
    "sciq": 2400,
    "banking77": 2400,
}
SPLITS = ("train", "dev", "calibration", "test")


def sha(path: Path) -> str:
    return training_snapshot._sha256(path.read_bytes())


def rows_at(path: Path) -> Iterable[dict]:
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def grouping_keys(row: dict) -> set[tuple[str, str]]:
    """Links share namespaces so forward references join their actual parent IDs."""
    keys = set()
    for field in (
        "id",
        "translation_of",
        "paired_with_id",
        "parent_question_id",
        "source_row_id",
        "variant_of",
        "augmentation_of",
    ):
        if isinstance(row.get(field), str) and row[field]:
            keys.add(("row", row[field]))
    for field in (
        "problem_group_id",
        "translation_group_id",
        "paired_context_group_id",
        "paired_group_id",
        "pair_id",
    ):
        if isinstance(row.get(field), str) and row[field]:
            keys.add(("group", row[field]))
    for envelope in ("_training_snapshot", "_augmentation"):
        metadata = row.get(envelope, {})
        if metadata.get("snapshot_group_id"):
            keys.add(("group", metadata["snapshot_group_id"]))
        if metadata.get("parent_id"):
            keys.add(("row", metadata["parent_id"]))
    doc = row.get("document_metadata", {})
    for field in ("document_id", "authoring_freeze_group_id"):
        if doc.get(field):
            keys.add(("document", str(doc[field])))
    for value in doc.get("relation_keys", []):
        keys.add(("document_relation", str(value)))
    game = row.get("game_metadata", {})
    for field in (
        "state_family_id",
        "scenario_id",
        "trajectory_id",
        "episode_id",
        "original_problem_id",
    ):
        value = game.get(field, game.get("generation", {}).get(field))
        if value:
            keys.add(("game_" + field, str(value)))
    for value in game.get("split_group_keys", []):
        keys.add(("game_link", str(value)))
    for value in game.get("prior_group_ids", []):
        keys.add(("group", str(value)))
    context = progress.normalize(row.get("context", ""))
    if context:
        keys.add(("context", hashlib.sha256(context.encode()).hexdigest()))
    keys.add(("prompt", canonical_prompt_fingerprint(row)))
    return keys


def grouped_rows(rows: list[dict]) -> tuple[list[str], dict[str, list[int]]]:
    union = training_snapshot._UnionFind(len(rows))
    first = {}
    for index, row in enumerate(rows):
        for key in grouping_keys(row):
            if key in first:
                union.union(index, first[key])
            else:
                first[key] = index
    components = defaultdict(list)
    for index in range(len(rows)):
        components[union.find(index)].append(index)
    by_group, group_ids = {}, [""] * len(rows)
    for indexes in components.values():
        group = training_snapshot._snapshot_group_id(
            [rows[index]["id"] for index in indexes]
        )
        by_group[group] = indexes
        for index in indexes:
            group_ids[index] = group
    return group_ids, by_group


def fresh_split(group: str) -> str:
    fraction = (
        int(
            hashlib.sha256(f"new-dataset-v1:{SEED}:{group}".encode()).hexdigest()[:16],
            16,
        )
        / 2**64
    )
    return (
        "train"
        if fraction < 0.80
        else "dev"
        if fraction < 0.88
        else "calibration"
        if fraction < 0.94
        else "test"
    )


def assign_splits(
    rows: list[dict], groups: dict[str, list[int]]
) -> tuple[dict[str, str], set[str]]:
    assignments, conflicts = {}, set()
    for group, indexes in groups.items():
        original = {
            rows[index]["_new_dataset"]["original_split"]
            for index in indexes
            if rows[index]["_new_dataset"]["source_id"] == "original_replay"
        }
        if len(original) > 1:
            # Retain the immutable original assignment; reject new members of a conflict.
            conflicts.add(group)
            continue
        native_dev = any(
            rows[index]["_new_dataset"]["original_split"] in {"dev", "development"}
            for index in indexes
            if rows[index]["_new_dataset"]["source_id"] != "original_replay"
        )
        if original and native_dev and original != {"dev"}:
            conflicts.add(group)
            continue
        assignments[group] = (
            next(iter(original))
            if original
            else "dev"
            if native_dev
            else fresh_split(group)
        )
    return assignments, conflicts


def bands(row: dict) -> tuple[str, str]:
    choices, words = len(row["options"]), len(row["context"].split())
    choice = (
        "2-4"
        if choices <= 4
        else "5-8"
        if choices <= 8
        else "9-32"
        if choices <= 32
        else "33-128"
        if choices <= 128
        else "129-265"
    )
    context = (
        "0-80"
        if words <= 80
        else "81-400"
        if words <= 400
        else "401-1600"
        if words <= 1600
        else "1601+"
    )
    return choice, context


def balanced_sample(rows: list[dict], cap: int) -> list[dict]:
    """Round-robin sparse strata before repeated draws from abundant strata."""
    buckets = defaultdict(list)
    for row in rows:
        choice, context = bands(row)
        task = row.get("operation_id", row.get("family", "unknown"))
        buckets[(str(task), row.get("language", "unknown"), choice, context)].append(
            row
        )
    queues = {}
    for key, values in buckets.items():
        values.sort(
            key=lambda row: hashlib.sha256(f"{SEED}:{row['id']}".encode()).digest()
        )
        queues[key] = deque(values)
    active = deque(
        sorted(queues, key=lambda key: hashlib.sha256(repr(key).encode()).digest())
    )
    selected = []
    while active and len(selected) < cap:
        key = active.popleft()
        selected.append(queues[key].popleft())
        if queues[key]:
            active.append(key)
    return selected


def validate_fresh(row: dict, source: str, documents: dict) -> list[str]:
    if source == "games":
        return validate_game_row(row, allowed_splits=("train", "development", "dev"))
    if source != "documents":
        return validate_public_row(row)
    errors = [
        error
        for error in probability_v2.validate_v2_row(row)
        if error
        not in {"split_is_not_train", "provenance_does_not_match_required_label"}
    ]
    metadata = row.get("document_metadata", {})
    if not str(row.get("provenance", "")).startswith(
        "luna-generated;"
    ) or not metadata.get("author_model"):
        errors.append("document_source_provenance_missing")
    if row.get("semantic_review_status") != "independent_luna_max_review_passed":
        errors.append("document_lacks_independent_review")
    key = (metadata.get("document_id"), metadata.get("content_sha256"))
    document = documents.get(key)
    if document is None or row.get("context") != document.get("text"):
        errors.append("document_context_not_complete_source")
    return errors


def validate_snapshot_row(row: dict) -> list[str]:
    """Validate frozen source semantics independently of the intake train envelope."""
    source = row.get("_new_dataset", {}).get("source_id")
    if row.get("split") not in SPLITS:
        return ["snapshot_split_is_not_explicitly_allowed"]
    original_split = row.get("_new_dataset", {}).get("original_split")
    if source == "original_replay" and row["split"] != original_split:
        return ["snapshot_original_split_changed"]
    if (
        source != "original_replay"
        and original_split in {"dev", "development"}
        and row["split"] != "dev"
    ):
        return ["snapshot_native_development_split_changed"]
    if source == "games":
        return validate_game_row(row, allowed_splits=SPLITS)
    if source in {"typed-decisions", "sciq", "banking77"}:
        return validate_public_row(row)
    if source not in {"documents", "original_replay"}:
        return ["snapshot_source_is_not_explicitly_allowed"]
    errors = [
        error
        for error in probability_v2.classify_row(row)[1]
        if error != "split_is_not_train"
    ]
    if source == "documents":
        errors = [
            error
            for error in errors
            if error != "provenance_does_not_match_required_label"
        ]
        metadata = row.get("document_metadata", {})
        if row.get("semantic_review_status") != "independent_luna_max_review_passed":
            errors.append("document_lacks_independent_review")
        if not str(row.get("provenance", "")).startswith(
            "luna-generated;"
        ) or not metadata.get("author_model"):
            errors.append("document_source_provenance_missing")
        if hashlib.sha256(
            str(row.get("context", "")).encode()
        ).hexdigest() != metadata.get("content_sha256"):
            errors.append("document_context_source_hash_mismatch")
    return errors


def public_permutation(row: dict) -> dict:
    order = list(range(len(row["options"])))
    random.Random(
        hashlib.sha256(f"public-canonical-permutation-v1:{row['id']}".encode()).digest()
    ).shuffle(order)
    result = shuffle_choices(row, order)
    result["_new_dataset"]["canonical_choice_permutation"] = order
    return result


def build(output: Path = OUTPUT) -> dict:
    if output.exists():
        raise ValueError("new_snapshot_directory_is_immutable")
    sources, documents = [], {}
    doc_roots = [
        ROOT / "data/document_luna_max/2026-10-02",
        ROOT / "data/document_breadth/2026-10-03",
        ROOT / "data/document_expansion/2026-10-02",
    ]
    for root in doc_roots:
        for doc in rows_at(root / "documents.jsonl"):
            documents[(doc["id"], doc["content_sha256"])] = doc
        sources.append(("documents", root / "accepted.jsonl"))
    for root in (
        ROOT / "data/game_expansion/2026-10-02",
        ROOT / "data/game_breadth/2026-10-03",
    ):
        for filename in ("train.jsonl", "development.jsonl"):
            sources.append(("games", root / filename))
    for name in ("typed-decisions", "sciq", "banking77"):
        sources.append((name, PUBLIC_ROOT / f"{name}.jsonl"))
    old_splits = {row["id"]: row for row in rows_at(OLD / "splits.jsonl")}
    records, exclusions, seen, source_report = [], [], set(), {}
    all_sources = [("original_replay", OLD / "rows.jsonl"), *sources]
    for source, path in all_sources:
        counts = Counter()
        for line, row in enumerate(rows_at(path), 1):
            counts["input_rows"] += 1
            if row["id"] in seen:
                counts["duplicate_id"] += 1
                continue
            errors = (
                []
                if source == "original_replay"
                else validate_fresh(row, source, documents)
            )
            if errors:
                counts["invalid_or_unreviewed"] += 1
                exclusions.append(
                    {
                        "id": row["id"],
                        "source_file": str(path.relative_to(ROOT)),
                        "reasons": errors,
                    }
                )
                continue
            seen.add(row["id"])
            original = (
                old_splits[row["id"]]["split"]
                if source == "original_replay"
                else row["split"]
            )
            row["_new_dataset"] = {
                "source_id": source,
                "source_file": str(path.relative_to(ROOT)),
                "source_line": line,
                "original_split": original,
            }
            records.append(row)
            counts["validated_rows"] += 1
        source_report[str(path.relative_to(ROOT))] = {
            "source_id": source,
            "sha256": sha(path),
            **dict(counts),
        }
    group_ids, groups = grouped_rows(records)
    assignments, conflicts = assign_splits(records, groups)
    eligible = []
    old_registry = []
    fingerprints = set()
    for index, row in enumerate(records):
        source = row["_new_dataset"]["source_id"]
        group = group_ids[index]
        if source == "original_replay":
            original = row["_new_dataset"]["original_split"]
            old_registry.append(
                {
                    "id": row["id"],
                    "split": original,
                    "snapshot_group_id": group,
                    "original_snapshot_group_id": old_splits[row["id"]][
                        "snapshot_group_id"
                    ],
                }
            )
        if group in conflicts:
            exclusions.append(
                {"id": row["id"], "reasons": ["original_split_component_conflict"]}
            )
            continue
        split = assignments[group]
        fingerprint = training_snapshot.exact_content_fingerprint(row)
        if fingerprint in fingerprints and source != "original_replay":
            exclusions.append({"id": row["id"], "reasons": ["exact_content_duplicate"]})
            continue
        fingerprints.add(fingerprint)
        row["split"] = split
        source_kind = row.get("distribution_source_kind", "hard_label")
        row["_training_snapshot"] = {
            "snapshot_group_id": group,
            "source_kind": source_kind,
            "target_kind": "soft_full_distribution"
            if row.get("supervision_kind") == "soft_distribution"
            else "hard_onehot",
            "source_file": row["_new_dataset"]["source_file"],
            "source_line": row["_new_dataset"]["source_line"],
        }
        eligible.append(row)
    selected = []
    selection_counts = {}
    for source, cap in CAPS.items():
        candidate = [
            row for row in eligible if row["_new_dataset"]["source_id"] == source
        ]
        if source == "original_replay":
            train = [row for row in candidate if row["split"] == "train"]
            chosen = balanced_sample(train, cap) + [
                row for row in candidate if row["split"] != "train"
            ]
        else:
            chosen = balanced_sample(candidate, cap)
        selected.extend(chosen)
        selection_counts[source] = {
            "eligible": len(candidate),
            "selected": len(chosen),
            "cap": cap,
            "split_counts": dict(Counter(row["split"] for row in chosen)),
        }
    selected = [
        public_permutation(row)
        if row["_new_dataset"]["source_id"] in {"sciq", "banking77", "typed-decisions"}
        else row
        for row in selected
    ]
    selected.sort(key=lambda row: row["id"])
    output.mkdir(parents=True)
    artifacts = {}

    def write(name, rows):
        count = len(rows)
        artifacts[name] = {
            "records": count,
            "sha256": training_snapshot._atomic_jsonl(output / name, rows),
        }

    write("rows.jsonl", selected)
    splits = [
        {
            "id": row["id"],
            "split": row["split"],
            "problem_group_id": row["problem_group_id"],
            "snapshot_group_id": row["_training_snapshot"]["snapshot_group_id"],
            "source_kind": row["_training_snapshot"]["source_kind"],
            "source_split": row["_new_dataset"]["original_split"],
            "cohorts": [row["_new_dataset"]["source_id"]],
        }
        for row in selected
    ]
    write("splits.jsonl", splits)
    write("excluded.jsonl", exclusions)
    write("original-split-registry.jsonl", old_registry)
    counts = Counter(row["split"] for row in selected)
    distributions = {}
    for split in SPLITS:
        rows = [row for row in selected if row["split"] == split]
        distributions[split] = {
            "rows": len(rows),
            "source": dict(Counter(row["_new_dataset"]["source_id"] for row in rows)),
            "language": dict(Counter(row["language"] for row in rows)),
            "choices": dict(
                sorted(
                    Counter(str(len(row["options"])) for row in rows).items(),
                    key=lambda x: int(x[0]),
                )
            ),
            "context_band": dict(Counter(bands(row)[1] for row in rows)),
            "target_kind": dict(
                Counter(row["_training_snapshot"]["target_kind"] for row in rows)
            ),
            "task": dict(
                Counter(
                    row.get("operation_id", row.get("family", "unknown"))
                    for row in rows
                )
            ),
        }
    manifest = {
        "schema_version": "training-snapshot-v2",
        "version": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "artifacts": artifacts,
        "rows": {"training_eligible_rows": len(selected), "by_split": dict(counts)},
        "source_files": source_report,
        "source_selection": selection_counts,
        "distributions": distributions,
        "original_snapshot": {
            "path": str(OLD.relative_to(ROOT)),
            "manifest_sha256": sha(OLD / "manifest.json"),
            "registry_rows": len(old_registry),
            "original_train_reassigned_to_holdout": 0,
        },
        "grouping": {
            "component_count_before_sampling": len(groups),
            "original_component_conflicts": len(conflicts),
            "links": [
                "all old rows before replay sampling",
                "normalized nonempty exact context",
                "choice-order-invariant prompt",
                "row parent/translation/augmentation IDs",
                "nested document relations and IDs",
                "game state/scenario/trajectory/split/prior groups",
            ],
        },
        "fresh_split_ratios": {
            "train": 0.80,
            "dev": 0.08,
            "calibration": 0.06,
            "test": 0.06,
        },
        "native_dev_policy": "Fresh source-native dev stays dev; original dev/test retain assignments; calibration only fresh unlinked groups.",
        "augmentation_policy": "No new bulk authoring. Public canonical choice permutation happens only after group freeze; labels remapped exactly.",
        "public_sources": json.loads((PUBLIC_ROOT / "manifest.json").read_text()),
        "coverage_limits": [
            "Documents concentrated in existing independently reviewed public corpus; no claim of all languages/tasks/265-choice document coverage.",
            "Original heldout test has prior experimental history and is not called virgin.",
            "Exact overlap does not rule out semantic or pretrained-model contamination.",
        ],
        "context_policy": "Complete source contexts preserved; tokenization must not truncate or crop rows.",
        "soft_target_rounding": "Typed-decisions six-decimal source probabilities preserved; source-specific sum tolerance 2e-6.",
    }
    training_snapshot._atomic_json(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    manifest = build()
    print(
        json.dumps(
            {
                "snapshot": str(OUTPUT.relative_to(ROOT)),
                "counts": manifest["rows"]["by_split"],
                "sources": manifest["source_selection"],
                "grouping": manifest["grouping"],
            }
        )
    )
