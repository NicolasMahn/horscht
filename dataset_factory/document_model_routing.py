"""Hash-bound audit overlays for document-generator model routing."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping


ROUTING_LEDGER = "model-routing-audit.jsonl"
KNOWN_MODELS = frozenset({"gpt-6-luna", "gpt-6.1-sol"})


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def artifact_sha256(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def provenance_prefix(model: str) -> str:
    if model not in KNOWN_MODELS:
        return "unknown-model-generated"
    return "luna-generated" if model == "gpt-6-luna" else "gpt-6.1-sol-generated"


def _route_metadata(event: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "requested_model": event["requested_model"],
        "recorded_model": event["recorded_model"],
        "effective_model": event["effective_model"],
        "reason": event["reason"],
    }
    return {**core, "audit_sha256": artifact_sha256(core)}


def load_model_routing_audit(batch: Path) -> dict[str, Any]:
    """Load routes keyed by immutable source hashes, rejecting conflicting records."""
    path = batch / ROUTING_LEDGER
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        lines = []
    except (OSError, UnicodeError) as exc:
        raise ValueError("model-routing audit ledger is not readable UTF-8") from exc

    accepted: dict[tuple[str, str], dict[str, Any]] = {}
    drafts: dict[tuple[str, str], dict[str, Any]] = {}
    reviews: dict[str, dict[str, Any]] = {}
    for line in lines:
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError("model-routing audit ledger contains invalid JSON") from exc
        if not isinstance(event, dict):
            raise ValueError("model-routing audit entry must be an object")
        artifact_type = event.get("artifact_type")
        if (event.get("event") != "model_routing_audit"
                or not isinstance(artifact_type, str) or artifact_type not in {"accepted_row", "draft", "review_artifact"}
                or not isinstance(event.get("artifact_sha256"), str)
                or len(event["artifact_sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in event["artifact_sha256"])
                or not isinstance(event.get("requested_model"), str) or event["requested_model"] not in KNOWN_MODELS
                or not isinstance(event.get("recorded_model"), str) or event["recorded_model"] not in KNOWN_MODELS
                or not isinstance(event.get("effective_model"), str) or event["effective_model"] not in KNOWN_MODELS
                or not isinstance(event.get("reason"), str) or not event["reason"].strip()):
            raise ValueError("model-routing audit entry is malformed")
        if artifact_type == "review_artifact":
            key = event["artifact_sha256"]
            table = reviews
        elif artifact_type == "accepted_row":
            row_id, document_id = event.get("row_id"), event.get("document_id")
            if (not isinstance(row_id, str) or not row_id or not isinstance(document_id, str) or not document_id
                    or not isinstance(event.get("author_name"), str) or not event["author_name"].strip()):
                raise ValueError("accepted-row routing audit identity is malformed")
            key = (row_id, event["artifact_sha256"])
            table = accepted
        else:
            document_id, index = event.get("document_id"), event.get("draft_index")
            if (not isinstance(document_id, str) or not document_id
                    or not isinstance(index, int) or isinstance(index, bool) or index < 0):
                raise ValueError("draft routing audit identity is malformed")
            if not isinstance(event.get("author_name"), str) or not event["author_name"].strip():
                raise ValueError("draft routing audit author is malformed")
            key = (document_id, event["artifact_sha256"])
            table = drafts
        existing = table.get(key)
        if existing is not None and _route_metadata(existing) != _route_metadata(event):
            raise ValueError("conflicting model-routing audits bind the same artifact hash")
        table[key] = event
    return {"accepted_rows": accepted, "drafts": drafts, "reviews": reviews}


def apply_draft_model_route(
    draft: Mapping[str, Any],
    routes: Mapping[str, Any],
    document_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return an in-memory effective-model draft without changing its source bytes."""
    result = deepcopy(dict(draft))
    document_id = result.get("document_id") or document_id
    author_name = result.get("author_name")
    event = routes.get("drafts", {}).get((document_id, artifact_sha256(result)))
    if event is None:
        return result, None
    if event.get("author_name") != author_name:
        raise ValueError("draft author does not match its model-routing audit")
    if result.get("author_model") != event.get("recorded_model"):
        raise ValueError("draft recorded model does not match its model-routing audit")
    result["author_model"] = event["effective_model"]
    return result, _route_metadata(event)


def apply_row_model_route(
    row: Mapping[str, Any],
    routes: Mapping[str, Any],
    source_sha256: str,
) -> dict[str, Any]:
    """Apply a matching immutable accepted-row route to an exported row copy."""
    result = deepcopy(dict(row))
    metadata = result.get("document_metadata")
    if not isinstance(metadata, dict):
        return result
    event = routes.get("accepted_rows", {}).get((result.get("id"), source_sha256))
    if event is None:
        return result
    if (event.get("author_name") != metadata.get("author_name")
            or event.get("document_id") != metadata.get("document_id")):
        raise ValueError("accepted row does not match its model-routing audit identity")
    route_metadata = _route_metadata(event)
    if (metadata.get("author_model") == event.get("effective_model")
            and metadata.get("model_routing_audit") == route_metadata):
        return result
    if metadata.get("author_model") != event.get("recorded_model"):
        raise ValueError("accepted row recorded model does not match its model-routing audit")
    effective_model = event["effective_model"]
    metadata["author_model"] = effective_model
    metadata["model_routing_audit"] = route_metadata
    old_prefix = f"{provenance_prefix(event['recorded_model'])}; label not independently verified"
    new_prefix = f"{provenance_prefix(effective_model)}; label not independently verified"
    if not isinstance(result.get("provenance"), str) or not result["provenance"].startswith(old_prefix):
        raise ValueError("accepted row provenance cannot be reconciled with its model-routing audit")
    result["provenance"] = new_prefix + result["provenance"][len(old_prefix):]
    return result


def apply_review_model_route(
    review: Mapping[str, Any],
    routes: Mapping[str, Any],
    review_sha256: str,
) -> tuple[str, dict[str, Any] | None]:
    """Resolve the reviewer model for a byte-identical review artifact."""
    recorded_model = review.get("reviewer_model")
    event = routes.get("reviews", {}).get(review_sha256)
    if event is None:
        if not isinstance(recorded_model, str) or recorded_model not in KNOWN_MODELS:
            raise ValueError("review artifact has no supported reviewer model")
        return recorded_model, None
    if recorded_model != event.get("recorded_model"):
        raise ValueError("reviewer model does not match its model-routing audit")
    return event["effective_model"], _route_metadata(event)
