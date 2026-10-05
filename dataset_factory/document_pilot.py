"""Validate and export a frozen, explicitly reviewed document pilot, never training."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from dataset_factory.document_data import assemble_row, validate_document_row
from dataset_factory.document_model_routing import (
    apply_draft_model_route,
    apply_review_model_route,
    load_model_routing_audit,
)
from dataset_factory.probability_v2 import shuffle_choices


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def export(batch: Path, *, maximum: int = 96) -> dict:
    """Stage all admitted originals before writing either split; reject invalid rows."""
    source_bytes = (batch / "documents.jsonl").read_bytes()
    frozen = json.loads((batch / "allocation.json").read_text())
    if hashlib.sha256(source_bytes).hexdigest() != frozen["documents_sha256"]:
        raise ValueError("Source manifest changed after allocation freeze")
    allocation = frozen["allocation"]
    documents = {d["id"]: d for d in _jsonl(batch / "documents.jsonl")}
    if documents.keys() != allocation.keys():
        raise ValueError("Allocation and source document sets differ")
    review_bytes = (batch / "review.json").read_bytes()
    review = json.loads(review_bytes.decode("utf-8"))
    model_routing = load_model_routing_audit(batch)
    effective_reviewer_model, reviewer_route = apply_review_model_route(
        review, model_routing, hashlib.sha256(review_bytes).hexdigest()
    )
    review_status = (
        "independent_luna_review" if effective_reviewer_model == "gpt-6-luna"
        else "independent_sol_review"
    )
    decisions = {}
    for decision in review["decisions"]:
        key = (decision["document_id"], decision["index"])
        if key in decisions or decision["status"] not in {"accepted", "rejected"}:
            raise ValueError("Invalid or repeated review decision")
        decisions[key] = decision
    staged, rejected = [], []
    identifiers = set()
    keys = set()
    for doc_id, document in documents.items():
        for index, draft in enumerate(_jsonl(batch / "drafts" / f"{doc_id}.jsonl")):
            key = (doc_id, index)
            keys.add(key)
            decision = decisions.get(key)
            if decision is None:
                raise ValueError("Draft lacks an explicit review decision")
            draft_hash = hashlib.sha256(json.dumps(draft, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            if decision.get("reviewed_draft_sha256") != draft_hash:
                raise ValueError("Draft changed after semantic review")
            if decision["status"] == "rejected":
                rejected.append(decision)
                continue
            routed_draft, route_metadata = apply_draft_model_route(draft, model_routing, doc_id)
            row = assemble_row(document, routed_draft, allocation)
            if route_metadata is not None:
                row["document_metadata"]["model_routing_audit"] = route_metadata
            row["semantic_review_status"] = review_status
            row["language_review_status"] = review_status
            row["document_metadata"]["review"] = decision
            if reviewer_route is not None:
                row["document_metadata"]["review_model_routing_audit"] = reviewer_route
            errors = validate_document_row(row, documents, allocation)
            if errors:
                raise ValueError("Admitted draft fails structural validation: " + ",".join(errors))
            if row["id"] in identifiers:
                raise ValueError("Duplicate original question identity")
            identifiers.add(row["id"])
            # Deterministic permutation exercises the same head mapping used by
            # training; the row remains one original, not an augmentation.
            order = sorted(range(len(row["options"])), key=lambda i: hashlib.sha256(f"{row['id']}:{i}".encode()).digest())
            row = shuffle_choices(row, order)
            if validate_document_row(row, documents, allocation):
                raise ValueError("Permutation failed document validation")
            staged.append(row)
    if keys != decisions.keys():
        raise ValueError("Review ledger contains unknown drafts")
    if not 1 <= len(staged) <= maximum:
        raise ValueError("Accepted pilot size is outside its bound")
    for split in ("train", "dev"):
        (batch / f"{split}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in staged if r["split"] == split))
    (batch / "rejected.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rejected))
    summary = {"accepted_originals": len(staged), "rejected": len(rejected),
        "source_documents": len(documents),
        "publishers": dict(Counter(documents[r["document_metadata"]["document_id"]].get("publisher", r["document_metadata"]["source_family"]) for r in staged)),
        "languages": dict(Counter(r["language"] for r in staged)),
        "source_families": dict(Counter(r["document_metadata"]["source_family"] for r in staged)),
        "splits": dict(Counter(r["split"] for r in staged)),
        "task_kinds": dict(Counter(r["document_metadata"]["task_kind"] for r in staged)),
        "supervision": dict(Counter(r["supervision_kind"] for r in staged)),
        "answerability": dict(Counter(r["answerability_status"] for r in staged)),
        "choice_counts": dict(Counter(len(r["options"]) for r in staged)),
        "context_chars": {"minimum": min(len(r["context"]) for r in staged), "maximum": max(len(r["context"]) for r in staged)},
        "integration_rows_with_evidence_gap_over_1500_chars": sum(
            r["document_metadata"]["task_kind"] == "integration" and any(
                max(a["start"], b["start"]) - min(a["end"], b["end"]) > 1500
                for i, a in enumerate(r["document_metadata"]["evidence_spans"])
                for b in r["document_metadata"]["evidence_spans"][i + 1:]) for r in staged),
        "validation_errors": 0, "permutation_verified_originals": len(staged),
        "training_integration": "isolated pilot only; legacy training envelope intentionally unchanged",
        "external_benchmark_access": False}
    (batch / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    print(json.dumps(export(args.batch)))
