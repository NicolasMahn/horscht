"""Deterministic, source-grounded rows for the isolated document pilot."""
from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence

from dataset_factory import progress
from dataset_factory.document_model_routing import KNOWN_MODELS, artifact_sha256, provenance_prefix
from dataset_factory.probability_v2 import (
    ContractError,
    build_hard_row,
    build_soft_row,
    stable_choice_ids,
    shuffle_choices,
    validate_v2_row,
)

VERSION = "document-pilot-v1"
DOCUMENT_PERMUTATION_TYPES = (
    "highest_substantive_at_head",
    "abstention_at_head",
)


def _document_permutation_order(row: Mapping, augmentation_type: str) -> list[int]:
    if augmentation_type not in DOCUMENT_PERMUTATION_TYPES:
        raise ValueError("unsupported document permutation augmentation type")
    options = row.get("options")
    choice_ids = row.get("choice_ids")
    probabilities = row.get("target_distribution")
    if (not isinstance(options, list) or len(options) != 265
            or not isinstance(choice_ids, list) or len(choice_ids) != len(options)
            or not isinstance(probabilities, list) or len(probabilities) != len(options)):
        raise ValueError("document permutation augmentation requires a 265-choice soft row")
    abstention_id = row.get("abstention_choice_id")
    if not isinstance(abstention_id, str) or abstention_id not in choice_ids:
        raise ValueError("document permutation augmentation requires a stable abstention choice id")
    abstention_index = choice_ids.index(abstention_id)
    if augmentation_type == "abstention_at_head":
        target_index = abstention_index
    else:
        substantive = [
            index for index, value in enumerate(probabilities)
            if index != abstention_index and isinstance(value, (int, float))
            and not isinstance(value, bool) and value > 0
        ]
        if not substantive:
            raise ValueError("document permutation source has no positive substantive choice")
        target_index = max(substantive, key=lambda index: (probabilities[index], -index))

    source_digest = artifact_sha256(row)
    remainder = [index for index in range(len(options)) if index != target_index]
    remainder.sort(key=lambda index: hashlib.sha256(
        f"{row['id']}\0{source_digest}\0{augmentation_type}\0{choice_ids[index]}".encode("utf-8")
    ).digest())
    return [*remainder, target_index]


def _document_permutation_id(source_id: str, source_digest: str, augmentation_type: str) -> str:
    digest = hashlib.sha256(
        f"{source_id}\0{source_digest}\0{augmentation_type}".encode("utf-8")
    ).hexdigest()[:20]
    return f"document-permutation-augmentation-{digest}"


def build_document_permutation_augmentation(source_row: Mapping, augmentation_type: str) -> dict:
    """Build a deterministic choice-order variant without changing its source row."""
    if not isinstance(source_row, Mapping):
        raise ValueError("document permutation source must be an object")
    source_id = source_row.get("id")
    if not isinstance(source_id, str) or not source_id:
        raise ValueError("document permutation source needs an id")
    source_digest = artifact_sha256(source_row)
    order = _document_permutation_order(source_row, augmentation_type)
    row = shuffle_choices(source_row, order)
    row["id"] = _document_permutation_id(source_id, source_digest, augmentation_type)
    row["augmentation_of"] = source_id
    row["augmentation_type"] = augmentation_type
    return row


def validate_document_augmentation(
    augmentation: Mapping,
    original: Mapping,
    documents_by_id: Mapping[str, Mapping],
    allocation: Mapping[str, str],
) -> list[str]:
    """Check that an augmentation is exactly an aligned permutation of one source row."""
    errors: list[str] = []
    if not isinstance(augmentation, Mapping) or not isinstance(original, Mapping):
        return ["document_augmentation_or_source_is_not_an_object"]
    original_errors = validate_document_row(dict(original), documents_by_id, allocation)
    if original_errors:
        errors.append("document_augmentation_source_row_is_invalid")
    augmentation_type = augmentation.get("augmentation_type")
    try:
        expected = build_document_permutation_augmentation(original, augmentation_type)
    except (KeyError, TypeError, ValueError):
        return [*errors, "document_augmentation_type_or_source_is_invalid"]
    if dict(augmentation) != expected:
        errors.append("document_augmentation_does_not_match_expected_aligned_permutation")

    restored = copy.deepcopy(dict(augmentation))
    restored.pop("augmentation_of", None)
    restored.pop("augmentation_type", None)
    restored["id"] = original.get("id")
    expected_restored = shuffle_choices(original, _document_permutation_order(original, augmentation_type))
    if restored != expected_restored:
        errors.append("document_augmentation_does_not_restore_to_expected_source_permutation")
    errors.extend(validate_document_row(restored, documents_by_id, allocation))
    return errors


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _group_ids(documents: Sequence[Mapping]) -> dict[str, str]:
    """Build connected components from relation keys using deterministic union-find."""
    parents: dict[str, str] = {}
    ids = []
    for document in documents:
        doc_id = document.get("id")
        keys = document.get("relation_keys")
        if not isinstance(doc_id, str) or not doc_id.strip():
            raise ValueError("each document needs a nonempty id")
        if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or not k.strip() for k in keys):
            raise ValueError("each document needs nonempty relation_keys")
        if doc_id in ids:
            raise ValueError("document ids must be unique")
        ids.append(doc_id)

        def find(key):
            parents.setdefault(key, key)
            if parents[key] != key:
                parents[key] = find(parents[key])
            return parents[key]

        first = keys[0]
        for key in keys[1:]:
            a, b = find(first), find(key)
            parents[max(a, b)] = min(a, b)
        for key in keys:
            find(key)

    result = {}
    for document in documents:
        root = min(find_key for find_key in document["relation_keys"])
        # Resolve transitive aliases after all unions have been processed.
        while parents[root] != root:
            root = parents[root]
        result[document["id"]] = "document-group:" + _hash(root)[:24]
    return result


def freeze_groups(documents: Sequence[Mapping], seed: str = "document-pilot-2026-10-02") -> dict[str, str]:
    """Freeze transitive relation groups into train/dev before drafting."""
    groups = _group_ids(documents)
    unique = sorted(set(groups.values()))
    dev = {group for group in unique if int(_hash(f"{seed}\0{group}")[:16], 16) % 5 == 0}
    if len(unique) >= 2 and not dev:
        dev.add(min(unique, key=lambda group: _hash(f"{seed}\0{group}")))
    if len(unique) >= 2 and len(dev) == len(unique):
        dev.remove(max(unique, key=lambda group: _hash(f"{seed}\0{group}")))
    return {doc_id: ("dev" if group in dev else "train") for doc_id, group in groups.items()}


def _row_id(document: Mapping, draft: Mapping) -> str:
    identity = {"document": document["id"], "question": draft.get("question"),
                "options": sorted(draft.get("options", []), key=progress.normalize)}
    return "document:" + _hash(json.dumps(identity, sort_keys=True, ensure_ascii=False))[:24]


def assemble_row(document: Mapping, draft: Mapping, allocation: Mapping[str, str]) -> dict:
    """Add source envelope, stable choices and IDK to an authored draft."""
    _validate_source(document)
    if document["id"] not in allocation or allocation[document["id"]] not in {"train", "dev"}:
        raise ValueError("document requires a frozen train/dev allocation")
    options = draft.get("options")
    if not isinstance(options, list) or len(options) < 1 or any(not isinstance(o, str) or not o.strip() for o in options):
        raise ContractError("draft options must be nonempty strings")
    abstention_index = draft.get("abstention_index")
    if not isinstance(abstention_index, int) or isinstance(abstention_index, bool) or not 0 <= abstention_index < len(options):
        raise ContractError("abstention_index must select the native IDK option")
    if "abstention_text" in draft and draft["abstention_text"] != options[abstention_index]:
        raise ContractError("abstention_text must match the selected option")
    ordered_options = list(options)
    author_model = draft.get("author_model")
    author_prefix = provenance_prefix(author_model) if isinstance(author_model, str) else "unknown-model-generated"
    row = {
        "id": _row_id(document, draft), "problem_group_id": "document:" + document["id"],
        "split": allocation[document["id"]], "operation_id": "document_grounded_qa",
        "topic": document.get("topic", document["source_family"]),
        "language": document["language"], "difficulty": document.get("difficulty", "medium"),
        "context": document["text"], "question": draft.get("question"), "options": ordered_options,
        "choice_count": len(ordered_options), "explanation": draft.get("explanation"),
        "provenance": f"{author_prefix}; label not independently verified; relations={_hash(json.dumps(sorted(document['relation_keys']), ensure_ascii=False))[:24]}",
        "document_metadata": {
            "version": VERSION, "document_id": document["id"], "source_family": document["source_family"],
            "title": document["title"], "url": document["url"], "revision": document["revision"],
            "retrieved_at": document["retrieved_at"], "content_sha256": document["content_sha256"],
            "attribution": document["attribution"], "license": copy.deepcopy(document["license"]),
            "rights_evidence": document["rights_evidence"], "relation_keys": list(document["relation_keys"]),
            "author_name": draft.get("author_name"), "author_model": draft.get("author_model"),
            "evidence_spans": copy.deepcopy(draft.get("evidence_spans", [])),
            "task_kind": draft.get("task_kind", "document_qa"),
        },
    }
    row["choice_ids"] = stable_choice_ids(row["id"], ordered_options)
    for key in ("publisher", "published_at", "source_updated_at", "original_translation", "original_url", "translations", "extraction", "changes"):
        if key in document:
            row["document_metadata"][key] = copy.deepcopy(document[key])
    row["abstention_choice_id"] = row["choice_ids"][abstention_index]
    answerability = draft.get("answerability_status", "answerable")
    row["answerability_status"] = answerability
    row["unanswerable_reason"] = draft.get("unanswerable_reason")

    pilot_split, pilot_provenance = row["split"], row["provenance"]
    # Existing v2 builders enforce the mass-training envelope. Adapt only while
    # constructing, then restore the truthful frozen pilot metadata.
    row["split"] = "train"
    row["provenance"] = "luna-generated; label not independently verified"
    if "distribution_evidence" in draft:
        if "answer_index" in draft:
            raise ContractError("soft drafts must not include answer_index")
        source_kind = draft.get("distribution_source_kind", "exact_derived")
        evidence = copy.deepcopy(draft["distribution_evidence"])
        evidence["choice_ids"] = row["choice_ids"]
        row = build_soft_row(row, source_kind, distribution_evidence=evidence)
    else:
        if "distribution_source_kind" in draft:
            raise ContractError("hard drafts must not include distribution_source_kind")
        index = draft.get("answer_index")
        if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(options):
            raise ContractError("answer_index must select a drafted option")
        row["answer_index"] = index
        row = build_hard_row(row)
    row["split"] = pilot_split
    row["provenance"] = pilot_provenance
    return row


def _validate_source(document: Mapping) -> None:
    required = ("id", "source_family", "language", "title", "url", "revision", "retrieved_at",
                "content_sha256", "text")
    if any(not isinstance(document.get(key), str) or not document[key].strip() for key in required):
        raise ValueError("document source metadata is incomplete")
    if not document.get("attribution") or not document.get("rights_evidence"):
        raise ValueError("document attribution and rights evidence are required")
    if _hash(document["text"]) != document["content_sha256"]:
        raise ValueError("document content hash does not match source text")
    license_info = document.get("license")
    if not isinstance(license_info, Mapping) or not all(license_info.get(key) is not None for key in ("id", "url", "share_alike")):
        raise ValueError("document rights metadata is incomplete")
    if type(license_info["share_alike"]) is not bool:
        raise ValueError("share-alike status must be explicit")
    if license_info["id"].startswith("CC-BY-SA") and not license_info["share_alike"]:
        raise ValueError("share-alike source cannot drop its obligation")
    keys = document.get("relation_keys")
    if not isinstance(keys, list) or not keys or any(not isinstance(k, str) or not k.strip() for k in keys):
        raise ValueError("document relation_keys are required")


def validate_document_row(row: dict, documents_by_id: Mapping[str, Mapping], allocation: Mapping[str, str]) -> list[str]:
    """Validate probability-v2 structure plus truthful source and frozen-split envelope."""
    ignored = {"split_is_not_train", "provenance_does_not_match_required_label"}
    errors = [error for error in validate_v2_row(row) if error not in ignored]
    meta = row.get("document_metadata")
    if not isinstance(meta, dict):
        return errors + ["document_metadata_is_missing"]
    document = documents_by_id.get(meta.get("document_id"))
    if document is None:
        return errors + ["document_source_is_unknown"]
    try:
        _validate_source(document)
    except (ValueError, TypeError):
        errors.append("document_source_metadata_is_invalid")
        return errors
    if row.get("split") != allocation.get(document["id"]):
        errors.append("document_split_does_not_match_frozen_allocation")
    if row.get("split") not in {"train", "dev"}:
        errors.append("document_split_must_be_train_or_dev")
    if row.get("context") != document["text"]:
        errors.append("document_context_does_not_match_source_text")
    for key in ("source_family", "title", "url", "revision", "retrieved_at", "content_sha256", "attribution", "rights_evidence"):
        if meta.get(key) != document.get(key):
            errors.append("document_metadata_does_not_match_source")
            break
    if meta.get("license") != document.get("license"):
        errors.append("document_license_does_not_match_source")
    if meta.get("relation_keys") != document.get("relation_keys"):
        errors.append("document_relation_keys_do_not_match_source")
    if any(meta.get(key) != document.get(key) for key in (
            "publisher", "published_at", "source_updated_at", "original_translation",
            "original_url", "translations", "extraction", "changes")):
        errors.append("document_origin_or_changes_do_not_match_source")
    author_model = meta.get("author_model")
    if (not isinstance(meta.get("author_name"), str) or not meta["author_name"].strip()
            or not isinstance(author_model, str) or author_model not in KNOWN_MODELS):
        errors.append("document_author_metadata_is_missing")
    revision_author = meta.get("revision_author")
    if revision_author is not None:
        if (not isinstance(revision_author, dict)
                or not isinstance(revision_author.get("author_name"), str)
                or not revision_author["author_name"].strip()
                or not isinstance(revision_author.get("author_model"), str)
                or revision_author["author_model"] not in KNOWN_MODELS):
            errors.append("document_revision_author_metadata_is_malformed")
        else:
            revision_route = revision_author.get("model_routing_audit")
            if revision_route is not None:
                if not isinstance(revision_route, dict):
                    errors.append("document_revision_author_routing_audit_is_malformed")
                else:
                    revision_audit_core = {key: revision_route.get(key) for key in (
                        "requested_model", "recorded_model", "effective_model", "reason")
                    }
                    if (any(not isinstance(revision_audit_core[key], str) or revision_audit_core[key] not in KNOWN_MODELS for key in (
                            "requested_model", "recorded_model", "effective_model"))
                            or not isinstance(revision_audit_core["reason"], str)
                            or not revision_audit_core["reason"].strip()
                            or revision_route.get("audit_sha256") != artifact_sha256(revision_audit_core)
                            or revision_audit_core["effective_model"] != revision_author.get("author_model")):
                        errors.append("document_revision_author_routing_audit_does_not_match_author_model")
    spans = meta.get("evidence_spans")
    if not isinstance(spans, list) or not spans:
        errors.append("document_evidence_spans_are_missing")
    else:
        for span in spans:
            if (not isinstance(span, dict) or type(span.get("start")) is not int or type(span.get("end")) is not int
                    or not isinstance(span.get("quote"), str) or span["start"] < 0 or span["end"] <= span["start"]
                    or span["end"] > len(document["text"])
                    or document["text"][span["start"]:span["end"]] != span["quote"]):
                errors.append("document_evidence_span_is_not_exact")
                break
    expected_relation_stamp = _hash(json.dumps(sorted(document["relation_keys"]), ensure_ascii=False))[:24]
    expected_provenance = (
        f"{provenance_prefix(author_model)}; label not independently verified; relations={expected_relation_stamp}"
        if isinstance(author_model, str) and author_model in KNOWN_MODELS else None
    )
    if row.get("provenance") != expected_provenance:
        errors.append("document_provenance_does_not_match_author_model")
    routing_audit = meta.get("model_routing_audit")
    if routing_audit is not None:
        if not isinstance(routing_audit, dict):
            errors.append("document_model_routing_audit_is_malformed")
        else:
            audit_core = {key: routing_audit.get(key) for key in (
                "requested_model", "recorded_model", "effective_model", "reason")
            }
            if (any(not isinstance(audit_core[key], str) or audit_core[key] not in KNOWN_MODELS for key in (
                    "requested_model", "recorded_model", "effective_model"))
                    or not isinstance(audit_core["reason"], str) or not audit_core["reason"].strip()
                    or routing_audit.get("audit_sha256") != artifact_sha256(audit_core)
                    or audit_core["effective_model"] != author_model):
                errors.append("document_model_routing_audit_does_not_match_author_model")
    if meta.get("version") != VERSION:
        errors.append("document_metadata_version_is_unsupported")
    # A row's connected relation component must have one frozen split.
    group_ids = _group_ids(list(documents_by_id.values()))
    peers = [doc_id for doc_id, group in group_ids.items() if group == group_ids[document["id"]]]
    if any(allocation.get(peer) != row.get("split") for peer in peers):
        errors.append("document_related_sources_span_splits")
    if meta.get("task_kind") == "integration":
        valid_spans = spans if isinstance(spans, list) else []
        distant_pair = any(
            max(a["start"], b["start"]) - min(a["end"], b["end"]) > 500
            for i, a in enumerate(valid_spans) for b in valid_spans[i + 1:]
            if isinstance(a, dict) and isinstance(b, dict)
            and type(a.get("start")) is int and type(a.get("end")) is int
            and type(b.get("start")) is int and type(b.get("end")) is int
        )
        if not distant_pair:
            errors.append("document_integration_requires_distant_evidence")
    return errors
