"""Incremental admission and isolated export for frozen document expansions.

This module only manages the document expansion's own source, ledger, and split
exports. It does not discover or modify training snapshots.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Sequence
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from dataset_factory.document_data import (
    DOCUMENT_PERMUTATION_TYPES,
    _group_ids,
    _validate_source,
    assemble_row,
    build_document_permutation_augmentation,
    freeze_groups,
    validate_document_augmentation,
    validate_document_row,
)
from dataset_factory.document_model_routing import (
    KNOWN_MODELS,
    ROUTING_LEDGER,
    apply_draft_model_route,
    apply_row_model_route,
    artifact_sha256,
    load_model_routing_audit,
    provenance_prefix,
)
from dataset_factory.probability_v2 import build_soft_row, shuffle_choices


SEED = "expansion20261002"
MAX_ORIGINALS_PER_DOCUMENT = 20
MAX_ORIGINALS_PER_BATCH = 2400
MAX_PERMUTATION_AUGMENTATIONS_PER_EXPORT = 512
PERMUTATION_AUGMENTATION_LEDGER = "permutation-augmentations.jsonl"
MAX_CHOICES = 265
RUN_NAME = "document-expansion-2026-10-02"
_SOFT_EVIDENCE_FIELDS = (
    "target_meaning",
    "derivation",
    "mechanism_assumptions",
    "outcome_partition",
    "mechanism_family",
)
_DRAFT_FIELDS = {
    "document_id", "question", "options", "abstention_index", "abstention_text",
    "explanation", "evidence_spans", "task_kind", "answerability_status",
    "unanswerable_reason", "author_name", "author_model", "answer_index",
    "distribution_source_kind", "distribution_evidence", "target_distribution", "difficulty",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _jsonl_bytes(rows: list[Mapping[str, Any]]) -> bytes:
    return "".join(_canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read valid JSON from {path.name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_jsonl(path: Path, *, missing_ok: bool = False) -> list[dict[str, Any]]:
    try:
        payload = path.read_bytes()
    except FileNotFoundError:
        if missing_ok:
            return []
        raise ValueError(f"Missing required file: {path.name}") from None
    try:
        decoded = payload.decode("utf-8")
        rows = [json.loads(line) for line in decoded.splitlines() if line.strip()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path.name} is not valid UTF-8 JSONL") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} must contain JSON objects")
    if payload and not payload.endswith(b"\n"):
        raise ValueError(f"{path.name} must end at a complete JSONL boundary")
    return rows


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _document_filename(document_id: str) -> str:
    if not document_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in document_id):
        raise ValueError("document ids must be safe single-component filenames")
    if document_id in {".", ".."}:
        raise ValueError("document ids must be safe single-component filenames")
    return f"{document_id}.json"


def _load_source_inputs(batch: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    combined: list[dict[str, Any]] = []
    origins: dict[str, str] = {}
    for filename, origin in (
        ("wiki-documents.jsonl", "wiki"),
        ("reporting-documents.jsonl", "reporting"),
        ("public-documents.jsonl", "public"),
    ):
        rows = _read_jsonl(batch / filename, missing_ok=True)
        for document in rows:
            _validate_source(document)
            doc_id = document["id"]
            _document_filename(doc_id)
            if doc_id in origins:
                raise ValueError(f"duplicate source document id: {doc_id}")
            combined.append(document)
            origins[doc_id] = origin
    if not combined:
        raise ValueError("source inputs contain no documents")
    combined.sort(key=lambda document: document["id"])
    return combined, origins


def _source_fingerprints(documents: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    return (
        {document["id"]: hashlib.sha256((_canonical_json(document) + "\n").encode("utf-8")).hexdigest()
         for document in documents},
        {document["id"]: list(document["relation_keys"]) for document in documents},
    )


def _first_snapshot_group_ids(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Recover immutable authoring groups from manifests written before the field existed."""
    snapshots = manifest.get("freeze_snapshots")
    if not isinstance(snapshots, list) or not snapshots:
        raise ValueError("freeze snapshots are required to recover authoring group ids")
    result: dict[str, str] = {}
    for snapshot in snapshots:
        if not isinstance(snapshot, dict):
            raise ValueError("frozen source snapshot is malformed")
        source_hashes = snapshot.get("document_sha256")
        group_ids = snapshot.get("group_ids")
        if not isinstance(source_hashes, dict) or not isinstance(group_ids, dict) or source_hashes.keys() != group_ids.keys():
            raise ValueError("frozen source snapshot cannot recover authoring group ids")
        for doc_id, group_id in group_ids.items():
            result.setdefault(doc_id, group_id)
    if any(not isinstance(group_id, str) or not group_id for group_id in result.values()):
        raise ValueError("frozen source snapshot contains invalid authoring group ids")
    return result


def _manifest_for(
    documents: list[dict[str, Any]],
    origins: Mapping[str, str],
    source_hash: str,
    previous: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    groups = _group_ids(documents)
    source_hashes, relation_keys = _source_fingerprints(documents)
    seeded = freeze_groups(documents, seed=SEED)
    if previous is None:
        allocation = seeded
        authoring_group_ids = dict(groups)
        added_ids = sorted(allocation)
        events = [{"documents_sha256": source_hash, "allocation": {doc_id: allocation[doc_id] for doc_id in added_ids}}]
        snapshots = [{
            "documents_sha256": source_hash,
            "document_sha256": dict(source_hashes),
            "relation_keys": dict(relation_keys),
            "group_ids": dict(groups),
            "allocation": dict(allocation),
        }]
    else:
        old_hashes = previous.get("document_sha256")
        old_relations = previous.get("relation_keys")
        old_origins = previous.get("source_origin")
        old_allocation = previous.get("allocation")
        if not all(isinstance(value, dict) for value in (old_hashes, old_relations, old_origins, old_allocation)):
            raise ValueError("existing freeze lacks bound source and allocation records")
        old_authoring_groups = previous.get("authoring_freeze_group_ids")
        if old_authoring_groups is None:
            old_authoring_groups = _first_snapshot_group_ids(previous)
        if not isinstance(old_authoring_groups, dict) or set(old_authoring_groups) != set(old_hashes):
            raise ValueError("existing freeze lacks stable authoring group ids")
        old_ids = set(old_hashes)
        if not old_ids <= source_hashes.keys():
            raise ValueError("source documents cannot be removed from an existing freeze")
        for doc_id in old_ids:
            if (old_hashes.get(doc_id) != source_hashes[doc_id]
                    or old_relations.get(doc_id) != relation_keys[doc_id]
                    or old_origins.get(doc_id) != origins.get(doc_id)):
                raise ValueError("an existing frozen document or its relations changed")
            if old_allocation.get(doc_id) not in {"train", "dev"}:
                raise ValueError("existing freeze has an invalid document allocation")

        allocation: dict[str, str] = {}
        group_members: dict[str, list[str]] = {}
        for doc_id, group_id in groups.items():
            group_members.setdefault(group_id, []).append(doc_id)
        for group_id, members in group_members.items():
            inherited = {old_allocation[doc_id] for doc_id in members if doc_id in old_ids}
            if len(inherited) > 1:
                raise ValueError("new source relations join existing train and dev groups")
            split = next(iter(inherited)) if inherited else seeded[members[0]]
            for doc_id in members:
                allocation[doc_id] = split
        for doc_id in old_ids:
            if allocation[doc_id] != old_allocation[doc_id]:
                raise ValueError("source extension would change an existing allocation")

        authoring_group_ids = dict(old_authoring_groups)
        for doc_id in set(source_hashes) - old_ids:
            authoring_group_ids[doc_id] = groups[doc_id]

        added_ids = sorted(set(source_hashes) - old_ids)
        events = list(previous.get("allocation_events", []))
        if not isinstance(events, list) or not events:
            raise ValueError("existing freeze lacks its allocation history")
        if added_ids:
            events.append({
                "documents_sha256": source_hash,
                "allocation": {doc_id: allocation[doc_id] for doc_id in added_ids},
            })
        snapshots = list(previous.get("freeze_snapshots", []))
        if not isinstance(snapshots, list) or not snapshots:
            raise ValueError("existing freeze lacks its source snapshots")
        if added_ids:
            snapshots.append({
                "documents_sha256": source_hash,
                "document_sha256": dict(source_hashes),
                "relation_keys": dict(relation_keys),
                "group_ids": dict(groups),
                "allocation": dict(allocation),
            })
        elif previous.get("documents_sha256") == source_hash:
            return {**previous, "authoring_freeze_group_ids": authoring_group_ids}
        else:
            raise ValueError("source bytes changed without adding a new document")

    return {
        "schema_version": "document-expansion-freeze-v1",
        "seed": SEED,
        "documents_sha256": source_hash,
        "allocation": allocation,
        "group_ids": groups,
        "authoring_freeze_group_ids": authoring_group_ids,
        "source_origin": dict(origins),
        "document_sha256": source_hashes,
        "relation_keys": relation_keys,
        "allocation_events": events,
        "freeze_snapshots": snapshots,
    }


def freeze(batch: str | Path) -> dict[str, Any]:
    """Freeze immutable wiki/reporting source files and write source copies.

    Existing frozen artifacts are verified. New sources append an allocation
    event while preserving every prior document's content and split.
    """
    batch_path = Path(batch).resolve()
    if not batch_path.is_dir():
        raise ValueError("batch directory must already exist with both source JSONL files")
    documents, origins = _load_source_inputs(batch_path)
    source_payload = _jsonl_bytes(documents)
    source_hash = hashlib.sha256(source_payload).hexdigest()
    allocation_path = batch_path / "allocation.json"
    destination = batch_path / "documents.jsonl"
    previous = None
    if allocation_path.exists() or destination.exists():
        if not allocation_path.is_file() or not destination.is_file():
            raise ValueError("incomplete existing source freeze; use a new batch directory")
        previous = _read_json(allocation_path)
        frozen_payload = destination.read_bytes()
        if hashlib.sha256(frozen_payload).hexdigest() != previous.get("documents_sha256"):
            raise ValueError("frozen source documents changed after allocation freeze")
        frozen_documents = _read_jsonl(destination)
        frozen_by_id = {document.get("id"): document for document in frozen_documents}
        old_hashes = previous.get("document_sha256")
        if (len(frozen_by_id) != len(frozen_documents) or not isinstance(old_hashes, dict)
                or set(old_hashes) != set(frozen_by_id)):
            raise ValueError("existing freeze has invalid per-document source bindings")
        for document in frozen_documents:
            _validate_source(document)
            filename = _document_filename(document["id"])
            per_document = batch_path / "documents" / filename
            expected = (_canonical_json(document) + "\n").encode("utf-8")
            expected_hash = hashlib.sha256(expected).hexdigest()
            if (old_hashes.get(document["id"]) != expected_hash or not per_document.is_file()
                    or per_document.read_bytes() != expected):
                raise ValueError("an existing frozen source copy changed")
        _load_frozen(batch_path)
        old_ids = set(old_hashes)
        incoming_ids = {document["id"] for document in documents}
        if not old_ids <= incoming_ids:
            raise ValueError("source documents cannot be removed from an existing freeze")
    manifest = _manifest_for(documents, origins, source_hash, previous)
    manifest_payload = (_canonical_json(manifest) + "\n").encode("utf-8")
    source_dir = batch_path / "documents"
    if previous is not None:
        if destination.read_bytes() == source_payload:
            if allocation_path.read_bytes() != manifest_payload:
                # Migration changes only the manifest; source records and their
                # per-document copies remain byte-for-byte untouched.
                _atomic_write(allocation_path, manifest_payload)
            _load_frozen(batch_path)
            return manifest
        for document in documents:
            per_document = source_dir / _document_filename(document["id"])
            expected = (_canonical_json(document) + "\n").encode("utf-8")
            if document["id"] in old_ids:
                if not per_document.is_file() or per_document.read_bytes() != expected:
                    raise ValueError("per-document source copy changed after freeze")
            elif per_document.exists():
                try:
                    cached_document = json.loads(per_document.read_text(encoding="utf-8"))
                except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                    raise ValueError("new per-document source copy is not valid JSON") from exc
                if cached_document != document:
                    raise ValueError("new per-document source copy does not match admitted source")

    staged_files = [(destination, source_payload)]
    staged_files.extend(
        (source_dir / _document_filename(document["id"]), (_canonical_json(document) + "\n").encode("utf-8"))
        for document in documents
    )
    # The allocation file is the commit marker and is written after every source copy.
    for path, payload in staged_files:
        _atomic_write(path, payload)
    _atomic_write(allocation_path, manifest_payload)
    return manifest


def _load_frozen(batch: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    documents_payload = (batch / "documents.jsonl").read_bytes()
    manifest = _read_json(batch / "allocation.json")
    if hashlib.sha256(documents_payload).hexdigest() != manifest.get("documents_sha256"):
        raise ValueError("source manifest changed after allocation freeze")
    if manifest.get("seed") != SEED or manifest.get("schema_version") != "document-expansion-freeze-v1":
        raise ValueError("unsupported document expansion freeze")
    documents = _read_jsonl(batch / "documents.jsonl")
    documents_by_id = {document.get("id"): document for document in documents}
    if len(documents_by_id) != len(documents) or None in documents_by_id:
        raise ValueError("frozen source document ids are not unique")
    allocation = manifest.get("allocation")
    group_ids = manifest.get("group_ids")
    authoring_group_ids = manifest.get("authoring_freeze_group_ids")
    origins = manifest.get("source_origin")
    if not isinstance(allocation, dict) or allocation.keys() != documents_by_id.keys():
        raise ValueError("allocation and source document sets differ")
    if group_ids != _group_ids(documents) or not isinstance(origins, dict) or origins.keys() != documents_by_id.keys():
        raise ValueError("source grouping or origin metadata is invalid")
    if any(value not in {"train", "dev"} for value in allocation.values()):
        raise ValueError("allocation contains an unsupported split")
    group_splits: dict[str, set[str]] = {}
    for doc_id, group_id in group_ids.items():
        group_splits.setdefault(group_id, set()).add(allocation[doc_id])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError("a related source group crosses the frozen split")
    source_hashes, relation_keys = _source_fingerprints(documents)
    if manifest.get("document_sha256") != source_hashes or manifest.get("relation_keys") != relation_keys:
        raise ValueError("frozen per-document source bindings are invalid")
    if not all(origins.get(doc_id) in {"wiki", "reporting", "public"} for doc_id in documents_by_id):
        raise ValueError("frozen source origins are invalid")
    events = manifest.get("allocation_events")
    event_allocation: dict[str, str] = {}
    if not isinstance(events, list) or not events:
        raise ValueError("frozen allocation history is missing")
    for event in events:
        event_rows = event.get("allocation") if isinstance(event, dict) else None
        if not isinstance(event_rows, dict):
            raise ValueError("frozen allocation history is malformed")
        for doc_id, split in event_rows.items():
            if doc_id in event_allocation or doc_id not in allocation or split != allocation[doc_id]:
                raise ValueError("frozen allocation history conflicts with current allocation")
            event_allocation[doc_id] = split
    if event_allocation != allocation:
        raise ValueError("frozen allocation history does not cover every source")
    first_event = events[0]
    first_ids = set(first_event["allocation"])
    first_documents = [document for document in documents if document["id"] in first_ids]
    if freeze_groups(first_documents, seed=SEED) != first_event["allocation"]:
        raise ValueError("initial allocation event does not match the frozen seed")
    snapshots = manifest.get("freeze_snapshots")
    if not isinstance(snapshots, list) or not snapshots or not isinstance(snapshots[-1], dict):
        raise ValueError("frozen source snapshots are incomplete")
    first_seen_groups = _first_snapshot_group_ids(manifest)
    if authoring_group_ids is None:
        authoring_group_ids = first_seen_groups
    if (not isinstance(authoring_group_ids, dict)
            or authoring_group_ids.keys() != documents_by_id.keys()
            or any(not isinstance(group_id, str) or not group_id for group_id in authoring_group_ids.values())):
        raise ValueError("stable authoring group ids are invalid")
    if first_seen_groups != authoring_group_ids:
        raise ValueError("stable authoring group ids do not match first source snapshots")
    latest = snapshots[-1]
    if (latest.get("documents_sha256") != manifest.get("documents_sha256")
            or latest.get("document_sha256") != source_hashes
            or latest.get("relation_keys") != relation_keys
            or latest.get("group_ids") != group_ids
            or latest.get("allocation") != allocation):
        raise ValueError("latest source snapshot does not match the frozen manifest")
    source_holds: dict[str, Any] = {"document_ids": [], "sidecar_sha256": None}
    sidecar_path = batch / "source-screening-limitations.json"
    if sidecar_path.is_file():
        sidecar = _read_json(sidecar_path)
        source_pool_hash = sidecar.get("frozen_source_pool_sha256")
        held_ids = sidecar.get("quarantined_biography_ids")
        if not _is_sha256(source_pool_hash) or not isinstance(held_ids, list):
            raise ValueError("source-screening metadata is malformed")
        if (any(not isinstance(doc_id, str) or not doc_id for doc_id in held_ids)
                or len(set(held_ids)) != len(held_ids)):
            raise ValueError("source-screening held ids must be unique document ids")
        source_snapshot = next((snapshot for snapshot in snapshots
                                if isinstance(snapshot, dict)
                                and snapshot.get("documents_sha256") == source_pool_hash), None)
        if source_snapshot is None:
            raise ValueError("source-screening sidecar is not bound to a frozen source snapshot")
        source_snapshot_ids = source_snapshot.get("document_sha256")
        if not isinstance(source_snapshot_ids, dict) or not set(held_ids) <= source_snapshot_ids.keys():
            raise ValueError("source-screening ids are outside their frozen source pool")
        source_holds = {
            "document_ids": sorted(held_ids),
            "sidecar_sha256": hashlib.sha256(sidecar_path.read_bytes()).hexdigest(),
        }
    for document in documents:
        _validate_source(document)
        source_file = batch / "documents" / _document_filename(document["id"])
        expected = (_canonical_json(document) + "\n").encode("utf-8")
        if not source_file.is_file() or source_file.read_bytes() != expected:
            raise ValueError("per-document source copy changed after freeze")
    manifest = dict(manifest)
    manifest["authoring_freeze_group_ids"] = authoring_group_ids
    manifest["source_held_document_ids"] = source_holds["document_ids"]
    manifest["source_screening_sidecar_sha256"] = source_holds["sidecar_sha256"]
    return documents, documents_by_id, manifest


def _validate_soft_draft(draft: Mapping[str, Any]) -> None:
    if "distribution_evidence" not in draft:
        if "target_distribution" in draft or "distribution_source_kind" in draft:
            raise ValueError("hard drafts must not include soft distribution fields")
        return
    if "answer_index" in draft:
        raise ValueError("soft drafts must not include answer_index")
    source_kind = draft.get("distribution_source_kind")
    if not isinstance(source_kind, str) or not source_kind.strip():
        raise ValueError("soft drafts need a named distribution_source_kind")
    evidence = draft.get("distribution_evidence")
    if not isinstance(evidence, dict):
        raise ValueError("soft drafts need distribution_evidence")
    missing = [key for key in _SOFT_EVIDENCE_FIELDS if not _nonempty_evidence(evidence.get(key))]
    if missing:
        raise ValueError("soft distribution evidence is missing " + ",".join(missing))
    if source_kind != "teacher_estimate" and "target_distribution" in draft:
        raise ValueError("derived soft drafts must omit target_distribution")
    if source_kind == "teacher_estimate" and not isinstance(draft.get("target_distribution"), list):
        raise ValueError("teacher estimates need an explicit target_distribution")


def _nonempty_evidence(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value) and all(isinstance(item, str) and item.strip() for item in value)
    return False


def _choice_order(row: Mapping[str, Any]) -> list[int]:
    return sorted(
        range(len(row["options"])),
        key=lambda index: hashlib.sha256(f"{row['id']}:{row['choice_ids'][index]}".encode("utf-8")).digest(),
    )


def _parse_draft(record: dict[str, Any], documents_by_id: Mapping[str, Mapping[str, Any]]) -> tuple[Mapping[str, Any], dict[str, Any]]:
    document_id = record.get("document_id")
    document = documents_by_id.get(document_id)
    if document is None:
        raise ValueError("draft document_id is not in the frozen source set")
    extra_fields = set(record) - _DRAFT_FIELDS
    if extra_fields:
        raise ValueError("draft contains unsupported fields: " + ",".join(sorted(extra_fields)))
    draft = {key: value for key, value in record.items() if key != "document_id"}
    options = draft.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= MAX_CHOICES:
        raise ValueError(f"originals must have between 2 and {MAX_CHOICES} choices")
    if "difficulty" in draft and (
            not isinstance(draft["difficulty"], str)
            or draft["difficulty"] not in {"easy", "medium", "hard"}):
        raise ValueError("draft difficulty must be easy, medium, or hard")
    _validate_soft_draft(draft)
    return document, draft


def _record_digest(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(row).encode("utf-8")).hexdigest()


def _draft_digest(row: Mapping[str, Any]) -> str:
    # Independent semantic-review receipts bind compact canonical UTF-8 JSON.
    # Accepted-row and audit ledgers keep their existing spaced hash format.
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _assemble_original(document: Mapping[str, Any], draft: Mapping[str, Any], allocation: Mapping[str, str]) -> dict[str, Any]:
    """Assemble through the document contract, adapting only teacher estimates."""
    if draft.get("distribution_source_kind") != "teacher_estimate":
        return assemble_row(document, draft, allocation)

    target = draft["target_distribution"]
    evidence = deepcopy(draft["distribution_evidence"])
    base_draft = {key: deepcopy(value) for key, value in draft.items()
                  if key not in {"distribution_source_kind", "distribution_evidence", "target_distribution"}}
    base_draft["answer_index"] = (
        draft["abstention_index"]
        if draft.get("answerability_status") == "unanswerable"
        else 0
    )
    row = assemble_row(document, base_draft, allocation)
    row.pop("answer_index")
    split, provenance = row["split"], row["provenance"]
    row["split"] = "train"
    row["provenance"] = "luna-generated; label not independently verified"
    row = build_soft_row(row, "teacher_estimate", distribution_evidence=evidence, target_distribution=target)
    row["split"], row["provenance"] = split, provenance
    return row


def _build_original_row(
    draft_record: dict[str, Any],
    documents_by_id: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    model_routing: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    routed_record, route_metadata = apply_draft_model_route(
        draft_record, model_routing or {"drafts": {}}
    )
    if routed_record.get("document_id") in set(manifest.get("source_held_document_ids", [])):
        raise ValueError("source-held document is not eligible for authoring or release")
    document, draft = _parse_draft(routed_record, documents_by_id)
    row = _assemble_original(document, draft, manifest["allocation"])
    if "difficulty" in draft:
        row["difficulty"] = draft["difficulty"]
    row["document_metadata"]["source_origin"] = manifest["source_origin"][document["id"]]
    row["document_metadata"]["authoring_freeze_group_id"] = manifest["authoring_freeze_group_ids"][document["id"]]
    if route_metadata is not None:
        row["document_metadata"]["model_routing_audit"] = route_metadata
    row["expansion_record_type"] = "original"
    row["semantic_review_status"] = "author_self_review_only"
    row["language_review_status"] = "author_self_review_only"
    row = shuffle_choices(row, _choice_order(row))
    errors = validate_document_row(row, documents_by_id, manifest["allocation"])
    if errors:
        raise ValueError("draft fails document validation: " + ",".join(errors))
    return row


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = row.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("accepted ledger contains a row without an id")
        if identifier in result:
            raise ValueError("accepted ledger contains duplicate original ids")
        result[identifier] = row
    return result


def _append_ledger_bytes(path: Path, new_rows: list[Mapping[str, Any]]) -> bytes:
    """Keep prior ledger bytes intact and append only complete canonical records."""
    try:
        existing = path.read_bytes()
    except FileNotFoundError:
        existing = b""
    if existing and not existing.endswith(b"\n"):
        raise ValueError(f"{path.name} does not end at a complete JSONL boundary")
    return existing + _jsonl_bytes(new_rows)


def _normalize_legacy_row(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Adapt rows admitted before stable authoring-group metadata was added."""
    normalized = deepcopy(dict(row))
    metadata = normalized.get("document_metadata")
    if not isinstance(metadata, dict):
        return normalized
    document_id = metadata.get("document_id")
    expected_group = manifest["authoring_freeze_group_ids"].get(document_id)
    if expected_group is None:
        raise ValueError("accepted row refers to an unknown frozen document")
    metadata.pop("freeze_group_id", None)
    metadata.setdefault("authoring_freeze_group_id", expected_group)
    if metadata["authoring_freeze_group_id"] != expected_group:
        raise ValueError("accepted row authoring group does not match its first freeze")
    metadata.setdefault("source_origin", manifest["source_origin"][document_id])
    normalized.setdefault("expansion_record_type", "original")
    normalized.setdefault("semantic_review_status", "author_self_review_only")
    normalized.setdefault("language_review_status", "author_self_review_only")
    return normalized


_CORRECTION_FIXED_ROW_FIELDS = (
    "id", "problem_group_id", "split", "operation_id", "topic", "language",
    "difficulty", "context", "question", "options", "choice_count", "choice_ids",
    "abstention_choice_id", "provenance", "expansion_record_type",
)
_CORRECTION_MUTABLE_METADATA_FIELDS = {"evidence_spans", "review", "revision_author"}
_SHA256_HEX_LENGTH = 64


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _validate_correction_transition(old: Mapping[str, Any], new: Mapping[str, Any]) -> None:
    if any(old.get(key) != new.get(key) for key in _CORRECTION_FIXED_ROW_FIELDS):
        raise ValueError("correction cannot change original identity, source, context, or choices")
    old_metadata = old.get("document_metadata")
    new_metadata = new.get("document_metadata")
    if not isinstance(old_metadata, dict) or not isinstance(new_metadata, dict):
        raise ValueError("correction requires frozen document metadata")
    metadata_keys = (set(old_metadata) | set(new_metadata)) - _CORRECTION_MUTABLE_METADATA_FIELDS
    if any(old_metadata.get(key) != new_metadata.get(key) for key in metadata_keys):
        raise ValueError("correction cannot change source or author metadata")


def _build_corrected_row(
    draft_record: dict[str, Any],
    documents_by_id: Mapping[str, Mapping[str, Any]],
    manifest: Mapping[str, Any],
    model_routing: Mapping[str, Any],
    primary_row: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep original authorship while recording who supplied a correction."""
    routed_record, revision_route = apply_draft_model_route(draft_record, model_routing)
    revision_author = {
        "author_name": routed_record.get("author_name"),
        "author_model": routed_record.get("author_model"),
    }
    if revision_route is not None:
        revision_author["model_routing_audit"] = revision_route

    primary_metadata = primary_row.get("document_metadata")
    if not isinstance(primary_metadata, dict):
        raise ValueError("correction requires frozen primary author metadata")
    routed_record["author_name"] = primary_metadata.get("author_name")
    routed_record["author_model"] = primary_metadata.get("author_model")
    row = _build_original_row(routed_record, documents_by_id, manifest, {})
    metadata = row["document_metadata"]
    primary_route = primary_metadata.get("model_routing_audit")
    if primary_route is not None:
        metadata["model_routing_audit"] = deepcopy(primary_route)

    metadata["revision_author"] = revision_author
    return row, revision_author


def _load_accepted(
    batch: Path,
    documents_by_id: Mapping[str, Mapping[str, Any]] | None = None,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the active row view plus immutable ledgers and latest row versions."""
    if documents_by_id is None or manifest is None:
        _, loaded_documents, loaded_manifest = _load_frozen(batch)
        documents_by_id = loaded_documents
        manifest = loaded_manifest

    accepted_path = batch / "accepted.jsonl"
    correction_path = batch / "corrections.jsonl"
    quarantine_path = batch / "quarantine.jsonl"
    raw_rows = _read_jsonl(accepted_path, missing_ok=True)
    base_by_id = _rows_by_id(raw_rows)
    model_routing = load_model_routing_audit(batch)
    current_raw_by_id = dict(base_by_id)
    current_by_id = {
        identifier: _normalize_legacy_row(row, manifest)
        for identifier, row in base_by_id.items()
    }
    for identifier, row in list(current_by_id.items()):
        current_by_id[identifier] = apply_row_model_route(
            row, model_routing, _record_digest(base_by_id[identifier])
        )
    corrections = _read_jsonl(correction_path, missing_ok=True)
    latest_correction_event_by_id: dict[str, dict[str, Any]] = {}
    for event in corrections:
        identifier = event.get("row_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("correction ledger entry has an invalid row id")
        previous_raw = current_raw_by_id.get(identifier)
        corrected = event.get("row")
        if previous_raw is None or not isinstance(corrected, dict):
            raise ValueError("correction ledger refers to an unknown original")
        if not isinstance(event.get("reason"), str) or not event["reason"].strip():
            raise ValueError("correction ledger entry has no reason")
        if event.get("prior_row_sha256") != _record_digest(previous_raw):
            raise ValueError("correction ledger prior row hash does not match")
        if event.get("corrected_row_sha256") != _record_digest(corrected) or corrected.get("id") != identifier:
            raise ValueError("correction ledger corrected row hash does not match")
        source_draft_hash = event.get("source_draft_sha256")
        if source_draft_hash is not None and not _is_sha256(source_draft_hash):
            raise ValueError("correction ledger source draft hash is malformed")
        revision_author = event.get("revision_author")
        corrected_metadata = corrected.get("document_metadata")
        if revision_author is not None and (
                not isinstance(revision_author, dict)
                or not isinstance(corrected_metadata, dict)
                or corrected_metadata.get("revision_author") != revision_author):
            raise ValueError("correction ledger revision author does not match its row")
        normalized = _normalize_legacy_row(corrected, manifest)
        normalized = apply_row_model_route(
            normalized, model_routing, _record_digest(base_by_id[identifier])
        )
        _validate_correction_transition(current_by_id[identifier], normalized)
        errors = validate_document_row(normalized, documents_by_id, manifest["allocation"])
        if errors:
            raise ValueError("corrected row fails document validation: " + ",".join(errors))
        current_raw_by_id[identifier] = corrected
        current_by_id[identifier] = normalized
        latest_correction_event_by_id[identifier] = event

    quarantine_events = _read_jsonl(quarantine_path, missing_ok=True)
    latest_quarantine_event_by_id: dict[str, dict[str, Any]] = {}
    for event in quarantine_events:
        identifier = event.get("row_id")
        digest = event.get("prior_row_sha256")
        if (not isinstance(identifier, str) or identifier not in base_by_id
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise ValueError("quarantine ledger entry is malformed or refers to an unknown original")
        if event.get("event") != "quarantine" or not isinstance(event.get("reason"), str) or not event["reason"].strip():
            raise ValueError("quarantine ledger entry is malformed")
        latest_quarantine_event_by_id[identifier] = event

    release_events = _read_jsonl(batch / "quarantine-releases.jsonl", missing_ok=True)
    released_ids: set[str] = set()
    for event in release_events:
        identifier = event.get("row_id")
        if (not isinstance(identifier, str) or identifier not in base_by_id
                or event.get("event") != "release"
                or event.get("reviewer_model") != "gpt-6-luna"
                or event.get("status") != "accepted"
                or not _is_sha256(event.get("quarantine_event_sha256"))
                or not _is_sha256(event.get("overlay_row_sha256"))
                or not _is_sha256(event.get("review_sha256"))
                or not _is_sha256(event.get("reviewed_draft_sha256"))
                or (event.get("correction_event_sha256") is not None
                    and not _is_sha256(event.get("correction_event_sha256")))
                or not isinstance(event.get("document_id"), str)
                or not isinstance(event.get("draft_index"), int)
                or isinstance(event.get("draft_index"), bool)
                or event["draft_index"] < 0):
            raise ValueError("quarantine release ledger entry is malformed")
        decision = event.get("review_decision")
        if (not isinstance(decision, dict)
                or decision.get("document_id") != event["document_id"]
                or decision.get("index") != event["draft_index"]
                or decision.get("reviewed_draft_sha256") != event["reviewed_draft_sha256"]
                or decision.get("status") != "accepted"):
            raise ValueError("quarantine release review evidence is malformed")
        if event.get("draft_file") is not None:
            relative_file = Path(event["draft_file"]) if isinstance(event["draft_file"], str) else None
            if (relative_file is None or relative_file.is_absolute() or ".." in relative_file.parts
                    or relative_file.suffix != ".jsonl"
                    or decision.get("draft_file") != event["draft_file"]):
                raise ValueError("quarantine release draft-file evidence is malformed")
        latest_quarantine = latest_quarantine_event_by_id.get(identifier)
        latest_correction = latest_correction_event_by_id.get(identifier)
        current = current_raw_by_id[identifier]
        if (latest_quarantine is not None
                and event["quarantine_event_sha256"] == _record_digest(latest_quarantine)
                and event["overlay_row_sha256"] == _record_digest(current)
                and event.get("correction_event_sha256") == (
                    _record_digest(latest_correction) if latest_correction is not None else None
                )
                and event["document_id"] == current_by_id[identifier]["document_metadata"]["document_id"]):
            released_ids.add(identifier)

    quarantined_ids = set(latest_quarantine_event_by_id) - released_ids

    active_rows = [
        current_by_id[row["id"]]
        for row in raw_rows
        if row["id"] not in quarantined_ids
    ]
    for row in current_by_id.values():
        errors = validate_document_row(row, documents_by_id, manifest["allocation"])
        if errors:
            raise ValueError("accepted row fails document validation: " + ",".join(errors))
    return {
        "raw_rows": raw_rows,
        "base_by_id": base_by_id,
        "current_raw_by_id": current_raw_by_id,
        "current_by_id": current_by_id,
        "corrections": corrections,
        "quarantine_events": quarantine_events,
        "latest_quarantine_event_by_id": latest_quarantine_event_by_id,
        "latest_correction_event_by_id": latest_correction_event_by_id,
        "release_events": release_events,
        "released_ids": released_ids,
        "quarantined_ids": quarantined_ids,
        "active_rows": active_rows,
        "model_routing": model_routing,
    }


def _load_permutation_augmentation_events(batch: Path) -> list[dict[str, Any]]:
    events = _read_jsonl(batch / PERMUTATION_AUGMENTATION_LEDGER, missing_ok=True)
    seen_child_ids: set[str] = set()
    for event in events:
        row = event.get("row")
        source_id = event.get("source_row_id")
        source_digest = event.get("source_row_sha256")
        augmentation_type = event.get("augmentation_type")
        child_id = event.get("augmentation_id")
        if (event.get("event") != "permutation_augmentation"
                or not isinstance(source_id, str) or not source_id
                or not _is_sha256(source_digest)
                or augmentation_type not in DOCUMENT_PERMUTATION_TYPES
                or not isinstance(child_id, str) or not child_id
                or not isinstance(row, dict)
                or event.get("augmentation_row_sha256") != _record_digest(row)
                or row.get("id") != child_id
                or row.get("augmentation_of") != source_id
                or row.get("augmentation_type") != augmentation_type):
            raise ValueError("permutation augmentation ledger entry is malformed")
        if child_id in seen_child_ids:
            raise ValueError("permutation augmentation ledger repeats a child id")
        seen_child_ids.add(child_id)
    return events


def _active_permutation_augmentation_rows(
    events: list[dict[str, Any]],
    active_rows: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    parents = {row["id"]: row for row in active_rows}
    documents_by_id = {document["id"]: document for document in documents}
    active_variants: list[dict[str, Any]] = []
    for event in events:
        parent = parents.get(event["source_row_id"])
        if parent is None or _record_digest(parent) != event["source_row_sha256"]:
            continue
        row = event["row"]
        errors = validate_document_augmentation(
            row, parent, documents_by_id, manifest["allocation"]
        )
        if errors:
            raise ValueError("permutation augmentation fails document validation: " + ",".join(errors))
        if row["document_metadata"]["document_id"] != parent["document_metadata"]["document_id"]:
            raise ValueError("permutation augmentation changed its frozen source document")
        if manifest["group_ids"][row["document_metadata"]["document_id"]] != manifest["group_ids"][
                parent["document_metadata"]["document_id"]]:
            raise ValueError("permutation augmentation crossed a frozen relation group")
        active_variants.append(row)
    return active_variants


def _commit_exports(
    batch: Path,
    active_rows: list[dict[str, Any]],
    documents: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    *,
    raw_original_attempts: int,
    quarantined_ids: set[str],
    corrections: list[dict[str, Any]],
    release_events: list[dict[str, Any]],
    released_ids: set[str],
    ledger_updates: Mapping[Path, bytes],
    permutation_augmentation_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if permutation_augmentation_events is None:
        permutation_augmentation_events = _load_permutation_augmentation_events(batch)
    active_augmentation_rows = _active_permutation_augmentation_rows(
        permutation_augmentation_events, active_rows, documents, manifest
    )
    source_origins = manifest.get("source_origin", {})
    augmentation_status = {
        "rows": len(active_augmentation_rows),
        "events": len(permutation_augmentation_events),
        "types": dict(sorted(Counter(row.get("augmentation_type") for row in active_augmentation_rows).items())),
        "source_groups": dict(sorted(Counter(
            manifest["group_ids"][row["document_metadata"]["document_id"]]
            for row in active_augmentation_rows
        ).items())),
        "source_origins": dict(sorted(Counter(
            source_origins[row["document_metadata"]["document_id"]]
            for row in active_augmentation_rows
        ).items())),
    }
    status_path = _run_status_path(batch)
    status = {
        "schema_version": "document-expansion-status-v1",
        "seed": SEED,
        "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "counts": _counts(active_rows, documents, manifest),
        "raw_original_attempts": raw_original_attempts,
        "quarantined_originals": len(quarantined_ids),
        "correction_events": len(corrections),
        "corrected_originals": len({event["row_id"] for event in corrections}),
        "quarantine_release_events": len(release_events),
        "released_originals": len(released_ids),
        "augmentations": augmentation_status,
        "review_policy": "author self-review only; targeted independent Luna audit is tracked separately",
        "quarantine_release_policy": "corrections stay quarantined until a separate independent review decision",
        "training_integration": "isolated document expansion; no training snapshot discovery or writes",
        "external_benchmark_access": False,
    }
    status["counts"]["quarantined_originals"] = len(quarantined_ids)
    if status_path.is_file():
        existing_status = _read_json(status_path)
        old_counts = existing_status.get("counts")
        merged_counts = {**old_counts, **status["counts"]} if isinstance(old_counts, dict) else status["counts"]
        status = {**existing_status, **status, "counts": merged_counts}
    status_payload = (_canonical_json(status) + "\n").encode("utf-8")
    outputs = [
        (batch / "train.jsonl", _jsonl_bytes(sorted(
            (row for row in active_rows if row["split"] == "train"), key=lambda row: row["id"]))),
        (batch / "dev.jsonl", _jsonl_bytes(sorted(
            (row for row in active_rows if row["split"] == "dev"), key=lambda row: row["id"]))),
        (batch / "augmentations.jsonl", _jsonl_bytes(sorted(
            active_augmentation_rows, key=lambda row: (row["augmentation_of"], row["augmentation_type"])))),
        (status_path, status_payload),
        *ledger_updates.items(),
    ]
    temp_outputs: list[tuple[Path, Path]] = []
    try:
        for destination, payload in outputs:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
            temporary = Path(temporary_name)
            os.fchmod(descriptor, 0o644)
            temp_outputs.append((temporary, destination))
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        for temporary, destination in temp_outputs:
            os.replace(temporary, destination)
            directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    except BaseException:
        for temporary, _ in temp_outputs:
            temporary.unlink(missing_ok=True)
        raise

    return {
        "accepted_originals": len(active_rows),
        "quarantined_originals": len(quarantined_ids),
        "raw_original_attempts": raw_original_attempts,
        "correction_events": len(corrections),
        "quarantine_release_events": len(release_events),
        "released_originals": len(released_ids),
        "augmentation_rows": len(active_augmentation_rows),
        "permutation_augmentation_events": len(permutation_augmentation_events),
        "counts": status["counts"],
        "status_path": str(status_path),
    }


def _counts(rows: list[dict[str, Any]], documents: list[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    def counted(values: list[Any]) -> dict[str, int]:
        counts = Counter(str(value) for value in values)
        return dict(sorted(counts.items()))

    represented_docs = {row["document_metadata"]["document_id"] for row in rows}
    group_ids = manifest["group_ids"]
    soft = [row for row in rows if row.get("supervision_kind") == "soft_distribution"]
    non_one_hot = []
    multiple_positive = []
    for row in soft:
        values = row.get("target_distribution", [])
        positives = sum(isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0 for value in values)
        one_hot = positives == 1 and sum(value == 1 for value in values) == 1 and sum(values) == 1
        if not one_hot:
            non_one_hot.append(row)
        if positives >= 2:
            multiple_positive.append(row)
    return {
        "accepted_originals": len(rows),
        "source_documents": len(documents),
        "represented_documents": len(represented_docs),
        "independent_groups": len({group_ids[doc_id] for doc_id in represented_docs}),
        "languages": counted([row.get("language") for row in rows]),
        "sources": counted([row.get("document_metadata", {}).get("source_family") for row in rows]),
        "source_origins": counted([row.get("document_metadata", {}).get("source_origin") for row in rows]),
        "splits": counted([row.get("split") for row in rows]),
        "tasks": counted([row.get("document_metadata", {}).get("task_kind") for row in rows]),
        "choices": counted([len(row.get("options", [])) for row in rows]),
        "soft_rows": len(soft),
        "soft_non_one_hot": len(non_one_hot),
        "soft_multiple_positive_choices": len(multiple_positive),
        "soft_source_kinds": counted([row.get("distribution_source_kind") for row in soft]),
        "soft_mechanism_families": counted([
            row.get("distribution_evidence", {}).get("mechanism_family") for row in soft
        ]),
    }


def _run_status_path(batch: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    try:
        batch.relative_to(project_root)
    except ValueError:
        project_root = batch
    return project_root / "runs" / RUN_NAME / "status.json"


def admit(batch: str | Path, draft_path: str | Path) -> dict[str, Any]:
    """Validate and append one complete draft chunk to the isolated ledger.

    Identical IDs are idempotent across retries. Any validation, identity, or
    cap error rejects the entire incoming chunk before the accepted ledger moves.
    """
    batch_path = Path(batch).resolve()
    if not batch_path.is_dir():
        raise ValueError("batch directory does not exist")
    draft_file = Path(draft_path)
    if not draft_file.is_absolute():
        draft_file = batch_path / draft_file
    documents, documents_by_id, manifest = _load_frozen(batch_path)
    incoming_drafts = _read_jsonl(draft_file)
    if not incoming_drafts:
        raise ValueError("draft chunk must contain at least one original")

    ledger_path = batch_path / "accepted.jsonl"
    lock_path = batch_path / ".document-expansion.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _load_accepted(batch_path, documents_by_id, manifest)
        staged_by_id: dict[str, dict[str, Any]] = {}
        for draft_record in incoming_drafts:
            row = _build_original_row(draft_record, documents_by_id, manifest, state["model_routing"])
            identifier = row["id"]
            existing = state["current_by_id"].get(identifier)
            if existing is not None:
                if identifier in state["quarantined_ids"]:
                    raise ValueError("a quarantined original cannot be readmitted; use a new row id")
                if _record_digest(existing) != _record_digest(row):
                    raise ValueError("same original id was reingested with conflicting content")
                continue
            previous = staged_by_id.get(identifier)
            if previous is not None:
                if _record_digest(previous) != _record_digest(row):
                    raise ValueError("same original id appears with conflicting content in the chunk")
                continue
            staged_by_id[identifier] = row

        new_rows = list(staged_by_id.values())
        candidate = [*state["active_rows"], *new_rows]
        candidate_by_document = Counter(row["document_metadata"]["document_id"] for row in candidate)
        if any(count > MAX_ORIGINALS_PER_DOCUMENT for count in candidate_by_document.values()):
            raise ValueError(f"a document cannot exceed {MAX_ORIGINALS_PER_DOCUMENT} admitted originals")
        if len(candidate) > MAX_ORIGINALS_PER_BATCH:
            raise ValueError(f"a batch cannot exceed {MAX_ORIGINALS_PER_BATCH} admitted originals")

        candidate_raw = [*state["raw_rows"], *new_rows]
        ledger_payload = _append_ledger_bytes(ledger_path, new_rows)
        result = _commit_exports(
            batch_path,
            candidate,
            documents,
            manifest,
            raw_original_attempts=len(candidate_raw),
            quarantined_ids=state["quarantined_ids"],
            corrections=state["corrections"],
            release_events=state["release_events"],
            released_ids=state["released_ids"],
            ledger_updates={ledger_path: ledger_payload} if new_rows else {},
        )
        result["new_originals"] = len(new_rows)
        result["idempotent_reingestions"] = len(incoming_drafts) - len(new_rows)
        return result


def export_permutation_augmentations(
    batch: str | Path,
    row_ids: Sequence[str],
    *,
    max_augmentations: int = MAX_PERMUTATION_AUGMENTATIONS_PER_EXPORT,
) -> dict[str, Any]:
    """Append aligned choice-order variants for selected active 265-choice rows."""
    batch_path = Path(batch).resolve()
    if not batch_path.is_dir():
        raise ValueError("batch directory does not exist")
    if not isinstance(row_ids, Sequence) or isinstance(row_ids, (str, bytes)) or not row_ids:
        raise ValueError("augmentation row_ids must be a non-empty sequence")
    if any(not isinstance(identifier, str) or not identifier for identifier in row_ids):
        raise ValueError("augmentation row_ids must be non-empty original row ids")
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("augmentation row_ids must be unique")
    if (not isinstance(max_augmentations, int) or isinstance(max_augmentations, bool)
            or not 1 <= max_augmentations <= MAX_PERMUTATION_AUGMENTATIONS_PER_EXPORT):
        raise ValueError(
            f"max_augmentations must be between 1 and {MAX_PERMUTATION_AUGMENTATIONS_PER_EXPORT}"
        )

    documents, documents_by_id, manifest = _load_frozen(batch_path)
    lock_path = batch_path / ".document-expansion.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _load_accepted(batch_path, documents_by_id, manifest)
        active_by_id = {row["id"]: row for row in state["active_rows"]}
        unknown = set(row_ids) - state["base_by_id"].keys()
        if unknown:
            raise ValueError("augmentation row_ids include an unknown original")
        inactive = set(row_ids) - active_by_id.keys()
        if inactive:
            raise ValueError("permutation augmentations require active original rows")

        events = _load_permutation_augmentation_events(batch_path)
        existing_by_key = {
            (event["source_row_id"], event["source_row_sha256"], event["augmentation_type"]): event
            for event in events
        }
        staged: list[dict[str, Any]] = []
        candidate_count = 0
        for identifier in row_ids:
            source_row = active_by_id[identifier]
            source_digest = _record_digest(source_row)
            for augmentation_type in DOCUMENT_PERMUTATION_TYPES:
                candidate_count += 1
                augmentation = build_document_permutation_augmentation(source_row, augmentation_type)
                errors = validate_document_augmentation(
                    augmentation, source_row, documents_by_id, manifest["allocation"]
                )
                if errors:
                    raise ValueError("generated augmentation fails document validation: " + ",".join(errors))
                key = (identifier, source_digest, augmentation_type)
                existing = existing_by_key.get(key)
                if existing is not None:
                    if _record_digest(existing["row"]) != _record_digest(augmentation):
                        raise ValueError("existing permutation augmentation conflicts with its deterministic row")
                    continue
                staged.append({
                    "event": "permutation_augmentation",
                    "source_row_id": identifier,
                    "source_row_sha256": source_digest,
                    "augmentation_id": augmentation["id"],
                    "augmentation_type": augmentation_type,
                    "augmentation_row_sha256": _record_digest(augmentation),
                    "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    "row": augmentation,
                })

        if len(staged) > max_augmentations:
            raise ValueError("permutation augmentation chunk exceeds max_augmentations")
        all_events = [*events, *staged]
        ledger_path = batch_path / PERMUTATION_AUGMENTATION_LEDGER
        result = _commit_exports(
            batch_path,
            state["active_rows"],
            documents,
            manifest,
            raw_original_attempts=len(state["raw_rows"]),
            quarantined_ids=state["quarantined_ids"],
            corrections=state["corrections"],
            release_events=state["release_events"],
            released_ids=state["released_ids"],
            ledger_updates={ledger_path: _append_ledger_bytes(ledger_path, staged)} if staged else {},
            permutation_augmentation_events=all_events,
        )
        result["new_augmentations"] = len(staged)
        result["idempotent_augmentation_requests"] = candidate_count - len(staged)
        result["augmentation_ids"] = [event["augmentation_id"] for event in staged]
        return result


def quarantine(batch: str | Path, ids: Sequence[str], reason: str) -> dict[str, Any]:
    """Append quarantine decisions and regenerate exports without deleting rows."""
    batch_path = Path(batch).resolve()
    if not batch_path.is_dir():
        raise ValueError("batch directory does not exist")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
        raise ValueError("quarantine ids must be a non-empty sequence")
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError("quarantine ids must be non-empty row ids")
    if len(set(ids)) != len(ids):
        raise ValueError("quarantine ids must be unique")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("quarantine reason must be non-empty")
    documents, documents_by_id, manifest = _load_frozen(batch_path)
    lock_path = batch_path / ".document-expansion.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _load_accepted(batch_path, documents_by_id, manifest)
        unknown = set(ids) - state["base_by_id"].keys()
        if unknown:
            raise ValueError("quarantine ids include an unknown original")
        new_ids = [identifier for identifier in ids if identifier not in state["quarantined_ids"]]
        additions = [
            {
                "event": "quarantine",
                "row_id": identifier,
                "prior_row_sha256": _record_digest(state["current_raw_by_id"][identifier]),
                "reason": reason.strip(),
                "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            for identifier in new_ids
        ]
        quarantined_ids = state["quarantined_ids"] | set(new_ids)
        active_rows = [row for row in state["active_rows"] if row["id"] not in set(new_ids)]
        path = batch_path / "quarantine.jsonl"
        result = _commit_exports(
            batch_path,
            active_rows,
            documents,
            manifest,
            raw_original_attempts=len(state["raw_rows"]),
            quarantined_ids=quarantined_ids,
            corrections=state["corrections"],
            release_events=state["release_events"],
            released_ids=state["released_ids"] - set(new_ids),
            ledger_updates={path: _append_ledger_bytes(path, additions)} if additions else {},
        )
        result["new_quarantines"] = len(additions)
        return result


def correct(batch: str | Path, draft_path: str | Path, reason: str) -> dict[str, Any]:
    """Append validated row corrections while keeping original attempts immutable."""
    batch_path = Path(batch).resolve()
    if not batch_path.is_dir():
        raise ValueError("batch directory does not exist")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("correction reason must be non-empty")
    draft_file = Path(draft_path)
    if not draft_file.is_absolute():
        draft_file = batch_path / draft_file
    correction_drafts = _read_jsonl(draft_file)
    if not correction_drafts:
        raise ValueError("correction chunk must contain at least one draft")
    documents, documents_by_id, manifest = _load_frozen(batch_path)
    lock_path = batch_path / ".document-expansion.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _load_accepted(batch_path, documents_by_id, manifest)
        staged: dict[str, dict[str, Any]] = {}
        new_events: list[dict[str, Any]] = []
        for draft_record in correction_drafts:
            candidate = _build_original_row(draft_record, documents_by_id, manifest, state["model_routing"])
            identifier = candidate["id"]
            old = state["current_by_id"].get(identifier)
            old_raw = state["current_raw_by_id"].get(identifier)
            if old is None or old_raw is None:
                raise ValueError("correction row id is unknown; question or options changes need a lineage replacement")
            row, revision_author = _build_corrected_row(
                draft_record, documents_by_id, manifest, state["model_routing"], old
            )
            _validate_correction_transition(old, row)
            if _record_digest(old) == _record_digest(row):
                continue
            previous_staged = staged.get(identifier)
            if previous_staged is not None:
                if _record_digest(previous_staged) != _record_digest(row):
                    raise ValueError("correction chunk contains conflicting rows for one original")
                continue
            staged[identifier] = row
            new_events.append({
                "row_id": identifier,
                "prior_row_sha256": _record_digest(old_raw),
                "corrected_row_sha256": _record_digest(row),
                "source_draft_sha256": _draft_digest(draft_record),
                "revision_author": revision_author,
                "reason": reason.strip(),
                "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "row": row,
            })

        current_by_id = dict(state["current_by_id"])
        current_by_id.update(staged)
        quarantined_ids = state["quarantined_ids"] | (set(staged) & state["released_ids"])
        active_rows = [
            current_by_id[row["id"]]
            for row in state["raw_rows"]
            if row["id"] not in quarantined_ids
        ]
        all_corrections = [*state["corrections"], *new_events]
        path = batch_path / "corrections.jsonl"
        result = _commit_exports(
            batch_path,
            active_rows,
            documents,
            manifest,
            raw_original_attempts=len(state["raw_rows"]),
            quarantined_ids=quarantined_ids,
            corrections=all_corrections,
            release_events=state["release_events"],
            released_ids=state["released_ids"] - set(staged),
            ledger_updates={path: _append_ledger_bytes(path, new_events)} if new_events else {},
        )
        result["new_corrections"] = len(new_events)
        return result


def audit_model_routing(
    batch: str | Path,
    *,
    requested_model: str,
    recorded_model: str,
    effective_model: str,
    reason: str,
    author_names: Sequence[str],
) -> dict[str, Any]:
    """Append hash-bound model-routing audits for explicitly selected authors.

    Raw accepted rows and source drafts stay unchanged. Derived exports and later
    admissions read the overlay by exact row/draft hash and author identity.
    """
    batch_path = Path(batch).resolve()
    if not batch_path.is_dir():
        raise ValueError("batch directory does not exist")
    for model in (requested_model, recorded_model, effective_model):
        if not isinstance(model, str) or model not in KNOWN_MODELS:
            raise ValueError("model-routing audit names an unsupported model")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("model-routing audit reason must be non-empty")
    if (not isinstance(author_names, Sequence) or isinstance(author_names, (str, bytes))
            or not author_names
            or any(not isinstance(name, str) or not name.strip() for name in author_names)
            or len(set(author_names)) != len(author_names)):
        raise ValueError("model-routing audit needs a unique exact author-name allowlist")
    selected_authors = set(author_names)

    lock_path = batch_path / ".document-expansion.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        routing_before = load_model_routing_audit(batch_path)
        additions: list[dict[str, Any]] = []
        matched_artifacts = 0

        accepted_path = batch_path / "accepted.jsonl"
        accepted_bytes_before = accepted_path.read_bytes() if accepted_path.exists() else None
        accepted_rows = _read_jsonl(accepted_path, missing_ok=True)
        for row in accepted_rows:
            metadata = row.get("document_metadata")
            if (not isinstance(metadata, dict) or not isinstance(metadata.get("author_name"), str)
                    or metadata["author_name"] not in selected_authors):
                continue
            matched_artifacts += 1
            recorded = metadata.get("author_model")
            if recorded != recorded_model:
                raise ValueError("selected accepted row does not have the declared recorded model")
            identifier = row.get("id")
            document_id = metadata.get("document_id")
            if not isinstance(identifier, str) or not identifier or not isinstance(document_id, str) or not document_id:
                raise ValueError("selected accepted row has incomplete model-routing identity")
            digest = _record_digest(row)
            key = (identifier, digest)
            event = {
                "event": "model_routing_audit",
                "artifact_type": "accepted_row",
                "row_id": identifier,
                "document_id": document_id,
                "author_name": metadata["author_name"],
                "artifact_sha256": digest,
                "requested_model": requested_model,
                "recorded_model": recorded_model,
                "effective_model": effective_model,
                "reason": reason.strip(),
                "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            existing = routing_before["accepted_rows"].get(key)
            if existing is not None:
                if (existing.get("author_name") != event["author_name"]
                        or {name: existing.get(name) for name in (
                            "requested_model", "recorded_model", "effective_model", "reason")}
                        != {name: event[name] for name in (
                            "requested_model", "recorded_model", "effective_model", "reason")}):
                    raise ValueError("a different routing audit already binds this accepted row hash")
                continue
            additions.append(event)

        drafts_dir = batch_path / "drafts"
        seen_drafts: set[tuple[str, str]] = set()
        if drafts_dir.is_dir():
            for draft_file in sorted(drafts_dir.glob("*.jsonl")):
                for index, draft in enumerate(_read_jsonl(draft_file)):
                    author_name = draft.get("author_name")
                    if not isinstance(author_name, str) or author_name not in selected_authors:
                        continue
                    matched_artifacts += 1
                    if draft.get("author_model") != recorded_model:
                        raise ValueError("selected draft does not have the declared recorded model")
                    document_id = draft.get("document_id") or draft_file.stem
                    if not isinstance(document_id, str) or not document_id:
                        raise ValueError("selected draft has no document_id")
                    digest = artifact_sha256(draft)
                    key = (document_id, digest)
                    if key in seen_drafts:
                        continue
                    seen_drafts.add(key)
                    event = {
                        "event": "model_routing_audit",
                        "artifact_type": "draft",
                        "document_id": document_id,
                        "draft_index": index,
                        "author_name": author_name,
                        "artifact_sha256": digest,
                        "requested_model": requested_model,
                        "recorded_model": recorded_model,
                        "effective_model": effective_model,
                        "reason": reason.strip(),
                        "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                    }
                    existing = routing_before["drafts"].get(key)
                    if existing is not None:
                        if (existing.get("author_name") != author_name
                                or {name: existing.get(name) for name in (
                                    "requested_model", "recorded_model", "effective_model", "reason")}
                                != {name: event[name] for name in (
                                    "requested_model", "recorded_model", "effective_model", "reason")}):
                            raise ValueError("a different routing audit already binds this draft hash")
                        continue
                    additions.append(event)

        if matched_artifacts == 0:
            raise ValueError("the author allowlist matched no accepted rows or drafts")
        ledger_path = batch_path / ROUTING_LEDGER
        if additions:
            _atomic_write(ledger_path, _append_ledger_bytes(ledger_path, additions))

        result: dict[str, Any] = {
            "audited_accepted_rows": sum(event["artifact_type"] == "accepted_row" for event in additions),
            "audited_drafts": sum(event["artifact_type"] == "draft" for event in additions),
            "audit_events": len(additions),
            "requested_model": requested_model,
            "recorded_model": recorded_model,
            "effective_model": effective_model,
            "raw_accepted_ledger_preserved": accepted_bytes_before == (
                accepted_path.read_bytes() if accepted_path.exists() else None
            ),
            "routing_ledger": str(ledger_path),
        }
        if result["raw_accepted_ledger_preserved"] is not True:
            raise RuntimeError("model-routing audit changed immutable accepted bytes")

        if accepted_path.exists():
            documents, documents_by_id, manifest = _load_frozen(batch_path)
            state = _load_accepted(batch_path, documents_by_id, manifest)
            all_release_events = state["release_events"]
            export_result = _commit_exports(
                batch_path,
                state["active_rows"],
                documents,
                manifest,
                raw_original_attempts=len(state["raw_rows"]),
                quarantined_ids=state["quarantined_ids"],
                corrections=state["corrections"],
                release_events=all_release_events,
                released_ids=state["released_ids"],
                ledger_updates={},
            )
            result["exported_accepted_originals"] = export_result["accepted_originals"]
        elif (batch_path / "review.json").is_file():
            from dataset_factory.document_pilot import export

            result["pilot_export"] = export(batch_path)
        return result


def audit_review_model_routing(
    batch: str | Path,
    *,
    requested_model: str,
    recorded_model: str,
    effective_model: str,
    reason: str,
) -> dict[str, Any]:
    """Record reviewer routing against the exact raw pilot review bytes."""
    batch_path = Path(batch).resolve()
    if not batch_path.is_dir():
        raise ValueError("batch directory does not exist")
    for model in (requested_model, recorded_model, effective_model):
        if not isinstance(model, str) or model not in KNOWN_MODELS:
            raise ValueError("review-routing audit names an unsupported model")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("review-routing audit reason must be non-empty")
    review_path = batch_path / "review.json"
    try:
        review_bytes = review_path.read_bytes()
        review = json.loads(review_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("pilot review artifact is not readable UTF-8 JSON") from exc
    if not isinstance(review, dict) or review.get("reviewer_model") != recorded_model:
        raise ValueError("review artifact does not contain the declared recorded reviewer model")
    digest = hashlib.sha256(review_bytes).hexdigest()

    lock_path = batch_path / ".document-expansion.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        routes = load_model_routing_audit(batch_path)
        event = {
            "event": "model_routing_audit",
            "artifact_type": "review_artifact",
            "review_path": "review.json",
            "artifact_sha256": digest,
            "requested_model": requested_model,
            "recorded_model": recorded_model,
            "effective_model": effective_model,
            "reason": reason.strip(),
            "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        existing = routes["reviews"].get(digest)
        if existing is not None:
            if any(existing.get(key) != event[key] for key in (
                    "requested_model", "recorded_model", "effective_model", "reason")):
                raise ValueError("a different reviewer route already binds this review artifact hash")
            new_events: list[dict[str, Any]] = []
        else:
            new_events = [event]
            ledger_path = batch_path / ROUTING_LEDGER
            _atomic_write(ledger_path, _append_ledger_bytes(ledger_path, new_events))

        from dataset_factory.document_pilot import export

        exported = export(batch_path)
        return {
            "new_review_routing_events": len(new_events),
            "review_sha256": digest,
            "raw_review_preserved": review_path.read_bytes() == review_bytes,
            "effective_reviewer_model": effective_model,
            "pilot_export": exported,
        }


def _read_review_artifact(path: Path) -> tuple[dict[str, Any], str]:
    try:
        payload = path.read_bytes()
        review = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("review artifact is not readable UTF-8 JSON") from exc
    if not isinstance(review, dict):
        raise ValueError("review artifact must contain a JSON object")
    if review.get("reviewer_model") != "gpt-6-luna":
        raise ValueError("quarantine release requires an independent gpt-6-luna review")
    if "selected_rows" in review and "decisions" in review:
        raise ValueError("review artifact must use one selection list")
    selections = review.get("selected_rows", review.get("decisions"))
    if not isinstance(selections, list) or not selections:
        raise ValueError("review artifact has no row selections")
    seen: set[tuple[str, int]] = set()
    for selection in selections:
        if not isinstance(selection, dict):
            raise ValueError("review selection must be a JSON object")
        document_id, index = selection.get("document_id"), selection.get("index")
        digest, status = selection.get("reviewed_draft_sha256"), selection.get("status")
        if (not isinstance(document_id, str) or not document_id
                or not isinstance(index, int) or isinstance(index, bool) or index < 0
                or not _is_sha256(digest)
                or not isinstance(status, str) or status not in {"accepted", "rejected"}):
            raise ValueError("review selection is malformed")
        declared_file = selection.get("draft_file")
        if declared_file is not None:
            relative_file = Path(declared_file) if isinstance(declared_file, str) else None
            if (relative_file is None or not declared_file.strip() or relative_file.is_absolute()
                    or "\\" in declared_file or ".." in relative_file.parts
                    or relative_file.suffix != ".jsonl"):
                raise ValueError("review selection draft_file must be a safe relative JSONL path")
        key = (document_id, index)
        if key in seen:
            raise ValueError("review artifact repeats a document/index selection")
        seen.add(key)
    return review, hashlib.sha256(payload).hexdigest()


def _review_draft_path(batch: Path, selection: Mapping[str, Any]) -> tuple[Path, str]:
    document_id = selection["document_id"]
    declared = selection.get("draft_file")
    if declared is None:
        filename = _document_filename(document_id)
        relative = Path("drafts") / f"{filename[:-5]}.jsonl"
    else:
        relative = Path(declared)
    try:
        resolved = (batch / relative).resolve(strict=True)
        resolved.relative_to(batch.resolve())
    except (OSError, ValueError) as exc:
        raise ValueError("review selection draft_file must stay inside the batch") from exc
    if not resolved.is_file() or resolved.suffix != ".jsonl":
        raise ValueError("review selection draft_file must name a JSONL file")
    return resolved, relative.as_posix()


def release(batch: str | Path, ids: Sequence[str], review_path: str | Path) -> dict[str, Any]:
    """Release held originals only when an independent review covers their current overlay.

    The release ledger binds a row to its latest quarantine event, latest correction,
    exact current row digest, and the selected draft hash stored in the review artifact.
    """
    batch_path = Path(batch).resolve()
    if not batch_path.is_dir():
        raise ValueError("batch directory does not exist")
    if not isinstance(ids, Sequence) or isinstance(ids, (str, bytes)) or not ids:
        raise ValueError("release ids must be a non-empty sequence")
    if any(not isinstance(identifier, str) or not identifier for identifier in ids):
        raise ValueError("release ids must be non-empty row ids")
    if len(set(ids)) != len(ids):
        raise ValueError("release ids must be unique")
    review_file = Path(review_path)
    if not review_file.is_absolute():
        review_file = batch_path / review_file
    review, review_hash = _read_review_artifact(review_file)
    selections = review.get("selected_rows", review.get("decisions"))

    documents, documents_by_id, manifest = _load_frozen(batch_path)
    lock_path = batch_path / ".document-expansion.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        state = _load_accepted(batch_path, documents_by_id, manifest)
        unknown = set(ids) - state["base_by_id"].keys()
        if unknown:
            raise ValueError("release ids include an unknown original")

        expected_docs = {
            identifier: state["current_by_id"][identifier]["document_metadata"]["document_id"]
            for identifier in ids
        }
        if set(expected_docs.values()) & set(manifest.get("source_held_document_ids", [])):
            raise ValueError("source-held document cannot be released for authoring")
        wanted_docs = set(expected_docs.values())
        matched: dict[str, tuple[dict[str, Any], dict[str, Any], str]] = {}
        draft_cache: dict[Path, list[dict[str, Any]]] = {}
        draft_file_names: dict[Path, str] = {}
        for selection in selections:
            document_id = selection["document_id"]
            if document_id not in wanted_docs:
                continue
            if document_id not in documents_by_id:
                raise ValueError("review selection refers to a source outside the frozen batch")
            draft_path, relative_draft_path = _review_draft_path(batch_path, selection)
            if draft_path not in draft_cache:
                draft_cache[draft_path] = _read_jsonl(draft_path)
                draft_file_names[draft_path] = relative_draft_path
            draft_rows = draft_cache[draft_path]
            index = selection["index"]
            if index >= len(draft_rows):
                raise ValueError("review selection index is outside the current draft")
            draft_record = draft_rows[index]
            if draft_record.get("document_id") != document_id:
                raise ValueError("current draft document_id does not match its review selection")
            draft_hash = _draft_digest(draft_record)
            if selection["reviewed_draft_sha256"] != draft_hash:
                raise ValueError("draft changed after independent review")
            reviewed_row = _build_original_row(draft_record, documents_by_id, manifest, state["model_routing"])
            identifier = reviewed_row["id"]
            if identifier not in expected_docs:
                continue
            if expected_docs[identifier] != document_id:
                raise ValueError("reviewed row source does not match the held original")
            if identifier in matched:
                raise ValueError("review artifact selects the same original more than once")
            latest_correction = state["latest_correction_event_by_id"].get(identifier)
            if latest_correction is not None:
                expected_draft_hash = latest_correction.get("source_draft_sha256")
                if expected_draft_hash is not None and expected_draft_hash != draft_hash:
                    raise ValueError("reviewed draft is not the latest corrected draft")
                is_legacy_correction = (
                    expected_draft_hash is None
                    and latest_correction.get("revision_author") is None
                )
                # Legacy corrections used the original row reconstruction above.
                if not is_legacy_correction:
                    reviewed_row, _ = _build_corrected_row(
                        draft_record, documents_by_id, manifest, state["model_routing"],
                        state["current_by_id"][identifier],
                    )
            matched[identifier] = (
                reviewed_row,
                {**selection, "draft_file": draft_file_names[draft_path]},
                draft_hash,
            )

        additions: list[dict[str, Any]] = []
        for identifier in ids:
            match = matched.get(identifier)
            if match is None:
                raise ValueError("review artifact does not select the requested original")
            reviewed_row, selection, draft_hash = match
            if selection["status"] != "accepted":
                raise ValueError("review did not accept the requested original")
            current = state["current_by_id"][identifier]
            if _record_digest(reviewed_row) != _record_digest(current):
                raise ValueError("reviewed draft does not reconstruct the current corrected overlay")
            latest_quarantine = state["latest_quarantine_event_by_id"].get(identifier)
            if latest_quarantine is None:
                raise ValueError("requested original has no quarantine event to release")
            latest_correction = state["latest_correction_event_by_id"].get(identifier)
            release_event = {
                "event": "release",
                "row_id": identifier,
                "quarantine_event_sha256": _record_digest(latest_quarantine),
                "correction_event_sha256": _record_digest(latest_correction) if latest_correction is not None else None,
                "overlay_row_sha256": _record_digest(state["current_raw_by_id"][identifier]),
                "review_sha256": review_hash,
                "reviewer_model": review["reviewer_model"],
                "document_id": selection["document_id"],
                "draft_index": selection["index"],
                "reviewed_draft_sha256": draft_hash,
                "status": selection["status"],
                "review_decision": deepcopy(selection),
                "draft_file": selection["draft_file"],
                "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            }
            same_release_exists = any(
                old.get("row_id") == identifier
                and all(old.get(field) == release_event[field] for field in (
                    "quarantine_event_sha256", "correction_event_sha256", "overlay_row_sha256",
                    "review_sha256", "reviewer_model", "document_id", "draft_index",
                    "reviewed_draft_sha256", "status"))
                for old in state["release_events"]
            )
            if not same_release_exists:
                additions.append(release_event)

        released_ids = state["released_ids"] | set(ids)
        quarantined_ids = state["quarantined_ids"] - set(ids)
        active_rows = [
            state["current_by_id"][row["id"]]
            for row in state["raw_rows"]
            if row["id"] not in quarantined_ids
        ]
        active_per_document = Counter(
            row["document_metadata"]["document_id"] for row in active_rows
        )
        if any(count > MAX_ORIGINALS_PER_DOCUMENT for count in active_per_document.values()):
            raise ValueError(f"release would exceed {MAX_ORIGINALS_PER_DOCUMENT} active originals for a document")
        if len(active_rows) > MAX_ORIGINALS_PER_BATCH:
            raise ValueError(f"release would exceed {MAX_ORIGINALS_PER_BATCH} active originals for the batch")

        all_releases = [*state["release_events"], *additions]
        ledger_path = batch_path / "quarantine-releases.jsonl"
        result = _commit_exports(
            batch_path,
            active_rows,
            documents,
            manifest,
            raw_original_attempts=len(state["raw_rows"]),
            quarantined_ids=quarantined_ids,
            corrections=state["corrections"],
            release_events=all_releases,
            released_ids=released_ids,
            ledger_updates={ledger_path: _append_ledger_bytes(ledger_path, additions)} if additions else {},
        )
        result["new_releases"] = len(additions)
        return result
