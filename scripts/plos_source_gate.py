"""Canonical, source-scoped PLOS proof digests shared by acquisition and intake."""
from __future__ import annotations

import hashlib
import json
from typing import Any


PLOS_IDENTITY_SCOPE_SCHEMA = "plos-identity-scope-v1"
PLOS_IDENTITY_SCOPE_FIELDS = (
    "decision",
    "reason_codes",
    "canonical_doi",
    "canonical_url",
    "solr_metadata_raw_file",
    "solr_metadata_raw_sha256",
    "plos_jats_raw_file",
    "plos_jats_raw_sha256",
    "front_article_ids",
    "front_article_title",
    "front_title_match",
    "front_dois",
    "front_pmcids",
    "solr_metadata_license_hint",
    "jats_governing_license",
    "jats_governing_license_evidence",
    "same_item_rights_verified",
    "raw_source_json_sha256",
    "fulltext_raw_file",
    "fulltext_raw_sha256",
    "source_title",
    "source_language",
    "topic",
    "topic_review_record_sha256",
)


def canonical_sha256(value: Any) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def topic_review_row_sha256(row: dict[str, Any]) -> str:
    return canonical_sha256(row)


def identity_scope_payload(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("PLOS identity audit row is malformed")
    return {field: record.get(field) for field in PLOS_IDENTITY_SCOPE_FIELDS}


def identity_scope_sha256(record: dict[str, Any]) -> str:
    return canonical_sha256(identity_scope_payload(record))
