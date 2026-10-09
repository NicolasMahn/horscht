#!/usr/bin/env python3
"""Bounded, resumable PMC/GOV.UK source acquisition for the breadth wave.

Only official PMC retrieval APIs and the official GOV.UK Content API are used
for source bodies. Progress is written to source-local status files; this
script does not touch the wave coordinator or generate questions.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import html
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser
from zoneinfo import ZoneInfo

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RUN = ROOT / "runs/document-breadth-2026-10-03/acquisition"
from scripts.plos_source_gate import identity_scope_sha256 as plos_identity_scope_sha256
from scripts.plos_source_gate import topic_review_row_sha256 as plos_topic_review_row_sha256
DATA = ROOT / "data/document_breadth/2026-10-03"
DOCS = DATA / "documents"
RAW_SCIENCE = RUN / "raw/science"
RAW_PUBLIC = RUN / "raw/public"
SCIENCE_MANIFEST = RUN / "science-documents.jsonl"
PUBLIC_MANIFEST = RUN / "public-documents.jsonl"
SCIENCE_STATUS = RUN / "science-status.json"
PUBLIC_STATUS = RUN / "public-status.json"
SCIENCE_IDENTIFIER_AUDIT_REVISIONS = RUN / "science-source-identity-audit-revisions"
PRIOR_MANIFESTS = (
    ROOT / "data/document_pilot/2026-10-02/documents.jsonl",
    ROOT / "data/document_expansion/2026-10-02/documents.jsonl",
    ROOT / "data/document_luna_max/2026-10-02/documents.jsonl",
)

PMC_OAI = "https://pmc.ncbi.nlm.nih.gov/api/oai/v1/mh/"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PMC_OAI_POLICY = "https://pmc.ncbi.nlm.nih.gov/tools/oai/"
PMC_DATA_POLICY = "https://pmc.ncbi.nlm.nih.gov/tools/textmining/"
PMC_LANGUAGE_GUIDE = "https://pmc.ncbi.nlm.nih.gov/about/userguide/"
PMC_ROBOTS = "https://pmc.ncbi.nlm.nih.gov/robots.txt"
NCBI_ROBOTS = "https://eutils.ncbi.nlm.nih.gov/robots.txt"
GOV_CONTENT_API_DOCS = "https://content-api.publishing.service.gov.uk/"
GOV_REUSE_POLICY = "https://www.gov.uk/help/reuse-govuk-content"
GOV_REUSE_CONTENT = "https://www.gov.uk/api/content/help/reuse-govuk-content"
GOV_ROBOTS = "https://www.gov.uk/robots.txt"
GOV_API_ROOT = "https://www.gov.uk/api/content"
GOV_SEARCH_API = "https://www.gov.uk/api/search.json"
GOV_SEARCH_API_DOCS = "https://docs.publishing.service.gov.uk/repos/search-api/using-the-search-api.html"

TARGET_SCIENCE = 90
TARGET_PUBLIC = 30
TARGET_PLOS_PROOF_SOURCES = 36
SCIENCE_TOPIC_LIMITS = {"biological_science": 20}
MAX_SOURCE_CHARS = 100_000
MIN_SCIENCE_CHARS = 2_500
MIN_PUBLIC_CHARS = 1_500
MAX_RESPONSE_BYTES = 12_000_000
MAX_NETWORK_BYTES = 300_000_000
REQUEST_SPACING_SECONDS = 1.05
USER_AGENT_PRODUCT = "HorschtDocumentBreadth/1.0"
LICENSE_FILTER = '"cc by license"[filter] OR "cc0 license"[filter]'

PLOS_JOURNALS = (
    "PLOS ONE", "PLOS Biology", "PLOS Medicine", "PLOS Computational Biology",
    "PLOS Genetics", "PLOS Pathogens", "PLOS Neglected Tropical Diseases",
    "PLOS Global Public Health", "PLOS Climate", "PLOS Water",
)
PLOS_API_SEARCH = "https://api.plos.org/search"
PLOS_API_ROBOTS = "https://api.plos.org/robots.txt"
PLOS_JOURNALS_ROBOTS = "https://journals.plos.org/robots.txt"
PLOS_SEARCH_API_DOCS = "https://api.plos.org/solr/faq"
PLOS_SOLR_FIELDS_DOCS = "https://api.plos.org/solr/search-fields/"
PLOS_TDM_API_DOCS = "https://api.plos.org/text-and-data-mining.html"
PLOS_TDM_POLICY = "https://plos.org/text-and-data-mining/"
PLOS_SUBMISSION_GUIDELINES = "https://journals.plos.org/{journal_slug}/s/submission-guidelines"
PLOS_SEARCH_SPACING_SECONDS = 6.1
PLOS_ARTICLE_SPACING_SECONDS = 30.0
PLOS_QUERY_TOPICS = {
    "environment_ecology": "title:(climate OR ecology OR biodiversity OR conservation OR ecosystem OR forest OR wildlife OR marine OR ocean OR pollution OR water OR river OR wetland OR habitat OR restoration)",
    "agriculture_food": "title:(agriculture OR farming OR farmer OR crop OR irrigation OR soil OR food OR livestock OR aquaculture OR fisheries OR agroecology)",
    "society_behavior": "title:(education OR student OR school OR community OR communities OR inequality OR employment OR household OR governance OR policy OR migration OR social-science OR behavior OR behaviour)",
    "public_health": "title:(\"public health\" OR epidemiology OR \"health services\" OR \"health policy\" OR nutrition OR \"health system\" OR breastfeeding OR surveillance)",
    "physical_science": "title:(energy OR renewable OR solar OR battery OR engineering OR geology OR geophysics OR materials OR \"resource efficiency\" OR water-energy)",
}
SCIENCE_QUERIES = (
    ("environmental_science", '(ecology[title/abstract] OR biodiversity[title/abstract] OR "climate change"[title/abstract] OR conservation[title/abstract])'),
    ("agriculture_food", '(agriculture[title/abstract] OR crop[title/abstract] OR farming[title/abstract] OR food systems[title/abstract])'),
    ("public_health", '("public health"[title/abstract] OR epidemiology[title/abstract] OR prevention[title/abstract])'),
    ("behavior_psychology", '(psychology[title/abstract] OR behavior[title/abstract] OR cognition[title/abstract])'),
    ("microbiology", '(microbiology[title/abstract] OR microbial[title/abstract] OR bacteria[title/abstract])'),
    ("methods_data", '(methodology[title/abstract] OR "data analysis"[title/abstract] OR algorithm[title/abstract])'),
    ("systems_engineering", '(engineering[title/abstract] OR "system design"[title/abstract] OR optimization[title/abstract])'),
    ("population_health", '(nutrition[title/abstract] OR "health services"[title/abstract] OR "health policy"[title/abstract])'),
)

# A bounded public route list seeds a source-native, topic-spread sample. Full
# text is always retrieved through the documented Content API after lookup.
GOV_PATHS = (
    "guidance/apply-for-a-passport",
    "renew-adult-passport",
    "apply-for-a-provisional-driving-licence",
    "renew-driving-licence",
    "register-to-vote",
    "register-a-birth",
    "register-a-death",
    "check-mot-history",
    "vehicle-tax",
    "pay-vehicle-tax",
    "self-assessment-tax-returns",
    "pay-self-assessment-tax-bill",
    "register-for-self-assessment",
    "register-for-vat",
    "apply-for-tax-free-childcare",
    "apply-for-child-benefit",
    "universal-credit",
    "claim-benefits",
    "benefits-calculators",
    "check-benefits-financial-support",
    "apply-for-a-blue-badge",
    "get-a-free-bus-pass",
    "check-uk-visa",
    "apply-to-come-to-the-uk",
    "apply-for-a-uk-visa",
    "set-up-a-limited-company",
    "register-your-company",
    "start-up-a-business",
    "prepare-for-flooding",
    "report-flooding",
    "flooding-and-coastal-change",
    "apply-for-free-school-meals",
    "get-free-school-meals",
    "find-a-job",
    "check-state-pension",
    "get-state-pension",
    "apply-for-state-pension",
    "apply-for-a-criminal-record-check",
    "report-a-crime",
    "report-benefit-fraud",
    "report-tax-fraud",
    "pay-council-tax",
    "pay-income-tax",
    "pay-national-insurance",
    "apply-for-a-driving-licence",
    "book-your-theory-test",
    "book-an-mot-test",
    "report-vehicle-sale",
    "get-vehicle-tax-refund",
    "apply-for-a-licence-to-carry-out-an-activity",
    "apply-for-a-personal-licence",
    "apply-for-a-planning-permission",
    "get-help-with-childcare-costs",
    "apply-for-a-free-prescription-certificate",
    "apply-for-a-free-nhs-eye-test",
    "register-as-a-carer",
    "apply-for-carers-allowance",
    "apply-for-pip",
    "claim-attendance-allowance",
    "report-a-missing-person",
    "report-a-death",
    "apply-for-a-certificate-of-entitlement-to-the-right-of-abode",
    "check-if-you-need-a-uk-visa",
    "apply-for-a-visitor-visa",
    "register-a-death",
    "apply-for-a-marriage-allowance",
    "claim-a-tax-refund",
    "submit-vat-return",
    "pay-your-self-assessment-tax-bill",
    "renew-your-passport",
    "get-a-passport-for-your-child",
    "apply-for-a-disabled-persons-railcard",
    "apply-for-a-blue-badge",
    "report-a-fault-with-a-traffic-light",
    "report-a-pothole",
    "make-a-subject-access-request",
    "get-a-copy-of-a-birth-certificate",
    "get-a-copy-of-a-death-certificate",
    "apply-for-a-certificate-of-no-impediment",
    "appeal-a-benefits-decision",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def canonical_science_review_row_sha256(row: dict[str, Any]) -> str:
    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha(canonical)


def science_identity_scope_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Return stable evidence fields for a per-source identity digest."""
    request_fields = ("url", "final_url", "http_status", "raw_file", "sha256", "response_bytes_decoded")

    def stable_request(value: Any) -> dict[str, Any] | None:
        if not isinstance(value, dict):
            return None
        return {key: value.get(key) for key in request_fields}

    list_evidence = row.get("dc_list_record_evidence")
    if isinstance(list_evidence, dict):
        list_evidence = {
            key: list_evidence.get(key)
            for key in ("raw_file", "raw_sha256", "header_identifier", "canonical_pmcids", "canonical_dois", "title", "license")
        }
    else:
        list_evidence = None
    return {
        "decision": row.get("decision"),
        "reason_codes": sorted(row.get("reason_codes") or []),
        "canonical_pmcid": row.get("canonical_pmcid"),
        "canonical_doi": row.get("canonical_doi"),
        "canonical_url": row.get("canonical_url"),
        "requested_oai_identifier": row.get("requested_oai_identifier"),
        "oai_identifier": row.get("oai_identifier"),
        "raw_source_json_sha256": str(row.get("raw_source_json_sha256") or "").lower() or None,
        "fulltext_raw_file": row.get("fulltext_raw_file"),
        "fulltext_raw_sha256": str(row.get("fulltext_raw_sha256") or "").lower() or None,
        "front_article_id_evidence": row.get("front_article_id_evidence"),
        "original_fulltext_request_log_evidence": stable_request(row.get("original_fulltext_request_log_evidence")),
        "dc_metadata_raw_file": row.get("dc_metadata_raw_file"),
        "dc_metadata_raw_sha256": str(row.get("dc_metadata_raw_sha256") or "").lower() or None,
        "dc_metadata_pmcids": sorted(row.get("dc_metadata_pmcids") or []),
        "dc_metadata_dois": sorted(row.get("dc_metadata_dois") or []),
        "dc_metadata_title": row.get("dc_metadata_title"),
        "dc_metadata_title_match": row.get("dc_metadata_title_match"),
        "dc_metadata_title_exact_match": row.get("dc_metadata_title_exact_match"),
        "dc_metadata_title_match_method": row.get("dc_metadata_title_match_method"),
        "dc_getrecord_request": stable_request(row.get("dc_getrecord_request")),
        "dc_list_record_evidence": list_evidence,
        "dc_rights_statements": row.get("dc_rights_statements"),
        "dc_governing_license": row.get("dc_governing_license"),
        "jats_governing_license": row.get("jats_governing_license"),
        "jats_license_evidence": row.get("jats_license_evidence"),
        "source_json_license_statement": row.get("source_json_license_statement"),
        "license_preflight_oai_identifier": row.get("license_preflight_oai_identifier"),
        "license": row.get("license"),
        "same_item_rights_verified": row.get("same_item_rights_verified"),
        "source_title": row.get("source_title"),
        "topic_review_record_sha256": row.get("topic_review_record_sha256"),
    }


def science_identity_scope_sha256(row: dict[str, Any]) -> str:
    canonical = json.dumps(science_identity_scope_payload(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha(canonical)


def archive_science_identity_audit(science: dict[str, Any]) -> tuple[str | None, str | None]:
    """Preserve the previous audit bytes by content hash before replacement."""
    if not SCIENCE_IDENTIFIER_AUDIT.exists():
        return None, None
    previous_bytes = SCIENCE_IDENTIFIER_AUDIT.read_bytes()
    previous_sha = sha(previous_bytes)
    revision_path = SCIENCE_IDENTIFIER_AUDIT_REVISIONS / f"{previous_sha}.json"
    if revision_path.exists():
        if sha(revision_path.read_bytes()) != previous_sha:
            raise RuntimeError("content-addressed science identity audit revision has a hash mismatch")
    else:
        atomic_write(revision_path, previous_bytes)
    revision_ref = raw_rel(revision_path)
    revision_rows = science.setdefault("identity_audit_revisions", [])
    if not any(row.get("sha256") == previous_sha for row in revision_rows if isinstance(row, dict)):
        revision_rows.append({"sha256": previous_sha, "file": revision_ref, "archived_at": utc_now()})
    science["superseded_identity_audit_sha256"] = previous_sha
    science["superseded_identity_audit_revision_file"] = revision_ref
    return previous_sha, revision_ref


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return default


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def norm_url(url: str) -> str:
    parts = urlsplit(url)
    host = (parts.hostname or "").removeprefix("www.").lower()
    path = re.sub(r"/+", "/", parts.path).rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), host, path, "", ""))


def normalize_doi(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value)
    return value.rstrip(" .")


def walk_values(value: Any, wanted: set[str], found: set[str]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            low = str(key).lower()
            if low in wanted and isinstance(item, (str, int)):
                found.add(str(item))
            walk_values(item, wanted, found)
    elif isinstance(value, list):
        for item in value:
            walk_values(item, wanted, found)


def load_prior_keys() -> dict[str, set[str]]:
    keys: dict[str, set[str]] = {"doi": set(), "pmcid": set(), "url": set(), "relation": set(), "work": set()}
    for path in PRIOR_MANIFESTS:
        if not path.exists():
            continue
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doi_values: set[str] = set()
                pmc_values: set[str] = set()
                work_values: set[str] = set()
                walk_values(doc, {"doi", "article_doi", "elocation_id"}, doi_values)
                walk_values(doc, {"pmcid", "pmc_id"}, pmc_values)
                walk_values(doc, {"source_work_id", "wikidata_qid", "content_id"}, work_values)
                for val in doi_values:
                    if val and "/" in val:
                        keys["doi"].add(normalize_doi(val))
                for val in pmc_values | {str(doc.get("source_work_id", "")), str(doc.get("id", ""))}:
                    match = re.search(r"PMC\d+", val, flags=re.I)
                    if match:
                        keys["pmcid"].add(match.group(0).upper())
                for val in work_values:
                    if val and val != "None":
                        keys["work"].add(val.casefold())
                for key in doc.get("relation_keys", []) or []:
                    if isinstance(key, str):
                        keys["relation"].add(key.casefold())
                if isinstance(doc.get("url"), str):
                    keys["url"].add(norm_url(doc["url"]))
    return keys


def existing_records(manifest: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if manifest.exists():
        with manifest.open() as stream:
            for line in stream:
                if line.strip():
                    items.append(json.loads(line))
    return items


def save_manifest(path: Path, docs: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(doc, ensure_ascii=False, separators=(",", ":")) + "\n" for doc in docs)
    atomic_write(path, content.encode("utf-8"))


def raw_rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def request_contact() -> str:
    try:
        value = subprocess.check_output(
            ["git", "config", "--global", "--get", "user.email"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("A configured contact address is required for these source APIs") from exc
    if not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
        raise RuntimeError("The configured contact address is not valid for source API identification")
    return value


def robots_allows(text: str, url: str, user_agent: str) -> bool:
    parser = RobotFileParser()
    parser.parse(text.splitlines())
    return parser.can_fetch(user_agent.split("/", 1)[0], url)


def local_language_records() -> list[dict[str, str]]:
    path = Path("/usr/share/iso-codes/json/iso_639-2.json")
    if not path.exists():
        return []
    parsed = json.loads(path.read_text())
    rows: dict[str, dict[str, str]] = {}
    for item in parsed.get("639-2", []):
        code = item.get("bibliographic") or item.get("alpha_3")
        name = item.get("name")
        if code and name and code not in {"und", "zxx", "mul"}:
            rows[code] = {"code": code, "name": name}
    return sorted(rows.values(), key=lambda item: item["code"])


class SerialHttp:
    def __init__(self, science_status: dict[str, Any], public_status: dict[str, Any], persist):
        self.session = requests.Session()
        self.contact = request_contact()
        self.user_agent = f"{USER_AGENT_PRODUCT} (bounded source acquisition; mailto:{self.contact})"
        self.session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Accept": "application/xml,application/json,text/html,text/plain,*/*",
        })
        self.statuses = {"science": science_status, "public": public_status}
        self.persist = persist
        self.last_request_at = 0.0

    def get(self, url: str, raw_path: Path, owner: str, *, params: dict[str, str] | None = None,
            max_bytes: int = MAX_RESPONSE_BYTES, allowed_status: tuple[int, ...] = (200,),
            store_raw: bool = True, min_spacing_seconds: float = REQUEST_SPACING_SECONDS) -> tuple[bytes, dict[str, Any]]:
        status = self.statuses[owner]
        request_url = requests.Request("GET", url, params=params).prepare().url or url
        if store_raw and raw_path.exists():
            body = raw_path.read_bytes()
            raw_sha = sha(body)
            original_request = next((
                item for item in reversed(status.get("requests", []))
                if item.get("raw_file") == raw_rel(raw_path)
                and item.get("sha256") == raw_sha
                and item.get("cached") is not True
            ), None)
            original_retrieved_at = (original_request or {}).get("retrieved_at")
            row = {
                "url": request_url,
                "final_url": request_url,
                "http_status": 200,
                "response_bytes_decoded": len(body),
                "sha256": raw_sha,
                "raw_file": raw_rel(raw_path),
                "cached": True,
                "retrieved_at": original_retrieved_at,
                "original_retrieved_at": original_retrieved_at,
                "verification_checked_at": utc_now(),
            }
            return body, row
        wait = max(REQUEST_SPACING_SECONDS, min_spacing_seconds) - (time.monotonic() - self.last_request_at)
        if wait > 0:
            time.sleep(wait)
        total_across_sources = sum(int(item.get("network_bytes_decoded", 0)) for item in self.statuses.values())
        if total_across_sources >= MAX_NETWORK_BYTES:
            raise RuntimeError("source network cap reached")
        response: requests.Response | None = None
        body_parts: list[bytes] = []
        total = 0
        started = utc_now()
        try:
            response = self.session.get(url, params=params, timeout=(20, 60), stream=True, allow_redirects=True)
            self.last_request_at = time.monotonic()
            status_code = response.status_code
            if status_code not in allowed_status:
                response.close()
                row = {
                    "url": request_url,
                    "final_url": response.url,
                    "http_status": status_code,
                    "response_bytes_decoded": 0,
                    "retrieved_at": started,
                    "error_category": "http_status",
                }
                status.setdefault("requests", []).append(row)
                status["last_request"] = row
                self.persist()
                if status_code in (401, 403, 404, 429, 451):
                    raise RuntimeError(f"source route returned {status_code}; no retry or alternate-route bypass")
                raise RuntimeError(f"source route returned HTTP {status_code}")
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                total_across_sources = sum(int(item.get("network_bytes_decoded", 0)) for item in self.statuses.values())
                if total > max_bytes or total_across_sources + total > MAX_NETWORK_BYTES:
                    response.close()
                    raise RuntimeError("response or network byte cap would be exceeded")
                body_parts.append(chunk)
            body = b"".join(body_parts)
            response.close()
            if store_raw:
                atomic_write(raw_path, body)
            row = {
                "url": request_url,
                "final_url": response.url,
                "http_status": status_code,
                "content_type": response.headers.get("Content-Type"),
                "response_bytes_decoded": len(body),
                "sha256": sha(body),
                "raw_file": raw_rel(raw_path) if store_raw else None,
                "cached": False,
                "retrieved_at": started,
            }
            status["network_bytes_decoded"] = int(status.get("network_bytes_decoded", 0)) + len(body)
            status["request_count"] = int(status.get("request_count", 0)) + 1
            status.setdefault("requests", []).append(row)
            status["last_request"] = row
            self.persist()
            return body, row
        except requests.RequestException as exc:
            if response is not None:
                response.close()
            row = {
                "url": request_url,
                "http_status": None,
                "response_bytes_decoded": total,
                "retrieved_at": started,
                "error_category": type(exc).__name__,
            }
            status.setdefault("requests", []).append(row)
            status["last_request"] = row
            self.persist()
            raise RuntimeError(f"source connection failed ({type(exc).__name__}); stopped without alternate route") from exc


def source_status(path: Path, kind: str, target: int) -> dict[str, Any]:
    current = read_json(path, {})
    if current:
        current.setdefault("requests", [])
        current.setdefault("network_bytes_decoded", 0)
        current.setdefault("request_count", 0)
        current.setdefault("accepted_work_count", 0)
        current.setdefault("accepted_document_count", 0)
        current.setdefault("held", {})
        current.setdefault("rejected_counts", {})
        current.setdefault("phase", "resuming")
        return current
    return {
        "schema_version": "document-breadth-acquisition-v1",
        "source_family": kind,
        "started_at": utc_now(),
        "updated_at": utc_now(),
        "phase": "preflight_pending",
        "target_independent_works": target,
        "accepted_work_count": 0,
        "accepted_document_count": 0,
        "request_count": 0,
        "network_bytes_decoded": 0,
        "network_byte_cap": MAX_NETWORK_BYTES,
        "max_source_chars": MAX_SOURCE_CHARS,
        "requests_serial": True,
        "request_spacing_seconds": REQUEST_SPACING_SECONDS,
        "external_benchmark_access": False,
        "requests": [],
        "rejected_counts": {},
        "held": {},
    }


def process_science_status(*, write_public: bool = True) -> dict[str, Any]:
    science = source_status(SCIENCE_STATUS, "scientific", TARGET_SCIENCE)
    public = source_status(PUBLIC_STATUS, "public_information", TARGET_PUBLIC)
    def persist() -> None:
        science["updated_at"] = utc_now()
        write_json(SCIENCE_STATUS, science)
        if write_public:
            public["updated_at"] = utc_now()
            write_json(PUBLIC_STATUS, public)
    write_json(SCIENCE_STATUS, science)
    if write_public:
        write_json(PUBLIC_STATUS, public)
    return science, public, persist


def xml_local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def text_content(element: ET.Element | None, *, skip: set[str] | None = None) -> str:
    if element is None:
        return ""
    skip = skip or {"graphic", "inline-graphic", "media", "object-id"}
    chunks: list[str] = []
    def visit(node: ET.Element) -> None:
        if xml_local(node.tag) in skip:
            return
        if node.text:
            chunks.append(node.text)
        for child in list(node):
            visit(child)
            if child.tail:
                chunks.append(child.tail)
    visit(element)
    value = "".join(chunks).replace("\xa0", " ")
    return re.sub(r"[ \t\r\f\v]+", " ", value).strip()


def first_text(root: ET.Element, *path: str) -> str:
    for node in root.iter():
        if xml_local(node.tag) == path[-1]:
            value = text_content(node)
            if value:
                return value
    return ""


def article_meta_values(root: ET.Element, name: str) -> list[str]:
    return [text_content(node) for node in root.iter() if xml_local(node.tag) == name and text_content(node)]


def ref_attr(element: ET.Element | None, suffix: str) -> str:
    if element is None:
        return ""
    return next((value for key, value in element.attrib.items() if key.endswith(suffix)), "")


def parse_date(article_meta: ET.Element | None) -> str | None:
    if article_meta is None:
        return None
    for pub_date in (n for n in article_meta.iter() if xml_local(n.tag) == "pub-date"):
        parts = {xml_local(n.tag): text_content(n) for n in list(pub_date)}
        year = parts.get("year")
        if not year:
            continue
        if parts.get("month") and parts.get("day"):
            try:
                month = int(parts["month"])
            except ValueError:
                month = datetime.strptime(parts["month"][:3].title(), "%b").month
            return f"{year}-{month:02d}-{int(parts['day']):02d}"
        if parts.get("month"):
            try:
                month = int(parts["month"])
            except ValueError:
                month = datetime.strptime(parts["month"][:3].title(), "%b").month
            return f"{year}-{month:02d}"
        return year
    return None


def jats_governing_license_evidence(article_meta: ET.Element | None) -> tuple[list[tuple[str, str]], list[dict[str, Any]]]:
    """Collect every article-meta license and governing permissions statement."""
    terms: list[tuple[str, str]] = []
    evidence: list[dict[str, Any]] = []
    if article_meta is None:
        return terms, evidence
    for node in article_meta.iter():
        name = xml_local(node.tag)
        if name not in {"license", "permissions", "copyright-statement", "license-p"}:
            continue
        hrefs = [value for key, value in node.attrib.items() if key.endswith("href") and value]
        statement = text_content(node)
        if not statement and not hrefs:
            continue
        if hrefs:
            for href in hrefs:
                terms.append((href, statement))
                evidence.append({"node": name, "href": href, "statement": statement, "canonical_license": canonical_cc_license(href, statement)})
        else:
            terms.append(("", statement))
            evidence.append({"node": name, "href": "", "statement": statement, "canonical_license": canonical_cc_license("", statement)})
    return terms, evidence


def section_blocks(element: ET.Element) -> tuple[str, list[dict[str, Any]]]:
    blocks: list[dict[str, Any]] = []
    for child in list(element):
        name = xml_local(child.tag)
        if name in {"title", "label", "sec", "ref-list", "fig-group"}:
            continue
        if name in {"p", "abstract", "caption", "boxed-text", "disp-quote", "statement", "supplementary-material"}:
            value = text_content(child)
            if value:
                blocks.append({"kind": name, "text": value})
        elif name in {"list", "def-list"}:
            list_type = child.attrib.get("list-type", "bullet")
            items: list[str] = []
            for item in child.iter():
                if xml_local(item.tag) in {"list-item", "def-item"}:
                    item_text = text_content(item)
                    if item_text:
                        items.append(item_text)
            for index, item_text in enumerate(items, start=1):
                blocks.append({"kind": "list_item", "list_type": list_type, "ordinal": index, "text": item_text})
        elif name in {"table-wrap", "table"}:
            table = parse_table(child)
            table["excluded_for_third_party_rights"] = bool(re.search(r"copyright|reproduced|adapted from|permission|courtesy", table.get("rights_note") or "", re.I))
            blocks.append({"kind": "table", **table})
        elif name in {"fig", "fig-group"}:
            caption = " ".join(text_content(n) for n in child.iter() if xml_local(n.tag) == "caption").strip()
            label = " ".join(text_content(n) for n in child.iter() if xml_local(n.tag) == "label").strip()
            attrib = " ".join(text_content(n) for n in child.iter() if xml_local(n.tag) in {"attrib", "permissions", "license"}).strip()
            if caption or label:
                third_party = bool(re.search(r"copyright|reproduced|adapted from|permission|courtesy", attrib, re.I))
                blocks.append({"kind": "figure_caption", "label": label, "text": caption, "rights_note": attrib or None, "excluded_for_third_party_rights": third_party})
        else:
            value = text_content(child)
            if value:
                blocks.append({"kind": name, "text": value})
    heading = next((text_content(child) for child in list(element) if xml_local(child.tag) == "title"), "")
    return heading, blocks


def parse_table(element: ET.Element) -> dict[str, Any]:
    label = next((text_content(node) for node in element.iter() if xml_local(node.tag) == "label"), "")
    caption = next((text_content(node) for node in element.iter() if xml_local(node.tag) == "caption"), "")
    rows: list[list[str]] = []
    for row in (node for node in element.iter() if xml_local(node.tag) == "tr"):
        cells = [text_content(cell) for cell in list(row) if xml_local(cell.tag) in {"th", "td"}]
        if cells:
            rows.append(cells)
    footnotes = [text_content(node) for node in element.iter() if xml_local(node.tag) == "fn" and text_content(node)]
    rights_note = " ".join(text_content(node) for node in element.iter() if xml_local(node.tag) in {"attrib", "permissions", "license"}).strip()
    return {"label": label, "caption": caption, "rows": rows, "footnotes": footnotes, "rights_note": rights_note or None}


def flatten_blocks(blocks: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for block in blocks:
        kind = block.get("kind")
        text_value = block.get("text", "")
        if block.get("excluded_for_third_party_rights"):
            lines.append(f"[{kind} omitted due to a third-party rights notice]")
            continue
        if kind == "table":
            if block.get("label"):
                lines.append(block["label"])
            if block.get("caption"):
                lines.append(block["caption"])
            for row in block.get("rows", []):
                lines.append(" | ".join(row))
            for note in block.get("footnotes", []):
                lines.append("Table note: " + note)
        elif kind == "list_item":
            marker = f"{block.get('ordinal')}." if block.get("list_type") in {"order", "ordered"} else "•"
            lines.append(f"{marker} {text_value}")
        elif kind == "figure_caption":
            text_caption = " ".join(v for v in (block.get("label"), text_value) if v)
            if text_caption:
                lines.append(text_caption)
        elif text_value:
            lines.append(text_value)
    return "\n\n".join(lines)


def extract_body(root: ET.Element) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    sections: list[dict[str, Any]] = []
    languages: list[str] = []
    figure_flags: list[dict[str, Any]] = []
    article = next((n for n in root.iter() if xml_local(n.tag) == "article"), root)
    article_lang = article.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") or article.attrib.get("lang")
    if article_lang:
        languages.append(normalize_science_language(article_lang))
    abstract_nodes = [n for n in article.iter() if xml_local(n.tag) == "abstract"]
    for index, abstract in enumerate(abstract_nodes, start=1):
        lang = abstract.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") or abstract.attrib.get("lang")
        normalized_lang = normalize_science_language(lang) if lang else None
        if normalized_lang and normalized_lang not in languages:
            languages.append(normalized_lang)
        title = next((text_content(ch) for ch in list(abstract) if xml_local(ch.tag) == "title"), f"Abstract {index}")
        content = flatten_blocks(section_blocks(abstract)[1])
        if content:
            sections.append({"section_path": [title], "section_type": "abstract", "language": normalized_lang or (normalize_science_language(article_lang) if article_lang else None), "blocks": section_blocks(abstract)[1], "text": content})

    body = next((n for n in article.iter() if xml_local(n.tag) == "body"), None)
    def add_sec(sec: ET.Element, path: list[str], depth: int) -> None:
        title = next((text_content(ch) for ch in list(sec) if xml_local(ch.tag) == "title"), "Untitled section")
        sec_path = path + [title]
        heading, blocks = section_blocks(sec)
        if blocks:
            sections.append({
                "section_path": sec_path,
                "section_type": sec.attrib.get("sec-type"),
                "section_depth": depth,
                "blocks": blocks,
                "text": flatten_blocks(blocks),
            })
            for block in blocks:
                if block.get("kind") == "figure_caption":
                    note = block.get("rights_note") or ""
                    third_party = bool(block.get("excluded_for_third_party_rights") or re.search(r"copyright|reproduced|adapted from|permission|courtesy", note, re.I))
                    figure_flags.append({"label": block.get("label"), "included_caption": not third_party, "media_collected": False, "third_party_rights_signal": third_party})
                elif block.get("kind") == "table":
                    note = block.get("rights_note") or ""
                    third_party = bool(block.get("excluded_for_third_party_rights") or re.search(r"copyright|reproduced|adapted from|permission|courtesy", note, re.I))
                    block["excluded_for_third_party_rights"] = third_party
        for child in list(sec):
            if xml_local(child.tag) == "sec":
                add_sec(child, sec_path, depth + 1)
    def note_figure_flags(blocks: list[dict[str, Any]]) -> None:
        for block in blocks:
            if block.get("kind") == "figure_caption":
                note = block.get("rights_note") or ""
                third_party = bool(block.get("excluded_for_third_party_rights") or re.search(r"copyright|reproduced|adapted from|permission|courtesy", note, re.I))
                figure_flags.append({"label": block.get("label"), "included_caption": not third_party, "media_collected": False, "third_party_rights_signal": third_party})
            elif block.get("kind") == "table":
                note = block.get("rights_note") or ""
                block["excluded_for_third_party_rights"] = bool(block.get("excluded_for_third_party_rights") or re.search(r"copyright|reproduced|adapted from|permission|courtesy", note, re.I))

    if body is not None:
        direct_blocks: list[dict[str, Any]] = []

        def flush_direct_blocks() -> None:
            if not direct_blocks:
                return
            text_value = flatten_blocks(direct_blocks)
            if text_value:
                sections.append({"section_path": ["Body"], "section_type": None, "blocks": list(direct_blocks), "text": text_value})
                note_figure_flags(direct_blocks)
            direct_blocks.clear()

        for child in list(body):
            if xml_local(child.tag) == "sec":
                flush_direct_blocks()
                add_sec(child, [], 1)
            else:
                # A few otherwise structured JATS articles put an introductory
                # paragraph directly under <body> before their titled sections.
                # Preserve it in order instead of treating it as the whole body.
                wrapper = ET.Element("wrapper")
                wrapper.append(child)
                direct_blocks.extend(section_blocks(wrapper)[1])
        flush_direct_blocks()
    back = next((n for n in article.iter() if xml_local(n.tag) == "back"), None)
    if back is not None:
        for child in list(back):
            name = xml_local(child.tag)
            if name == "ref-list":
                # Keep bibliographic context in full: numbered references are often
                # needed to interpret methods, limitations, and quoted evidence.
                refs = [text_content(n) for n in child.iter() if xml_local(n.tag) == "ref" and text_content(n)]
                if refs:
                    blocks = [{"kind": "reference", "ordinal": i, "text": value} for i, value in enumerate(refs, 1)]
                    sections.append({"section_path": ["References"], "section_type": "references", "blocks": blocks, "text": flatten_blocks(blocks)})
            elif name in {"ack", "app-group", "notes", "fn-group", "sec"}:
                title, blocks = section_blocks(child)
                content = flatten_blocks(blocks)
                if content:
                    sections.append({"section_path": [title or name], "section_type": name, "blocks": blocks, "text": content})
    for node in article.iter():
        if xml_local(node.tag) == "table-wrap":
            note = " ".join(text_content(n) for n in node.iter() if xml_local(n.tag) in {"attrib", "permissions", "license"}).strip()
            third_party = bool(re.search(r"copyright|reproduced|adapted from|permission|courtesy", note, re.I))
            for section in sections:
                for block in section.get("blocks", []):
                    if block.get("kind") == "table" and block.get("label") == next((text_content(n) for n in node.iter() if xml_local(n.tag) == "label"), ""):
                        if third_party:
                            block["excluded_for_third_party_rights"] = True
    return sections, languages, figure_flags


def parse_science_document(xml_bytes: bytes, pmcid: str, search_group: str) -> tuple[dict[str, Any] | None, str]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None, "xml_parse_error"
    article = next((n for n in root.iter() if xml_local(n.tag) == "article"), None)
    if article is None:
        return None, "no_article"
    article_meta = next((n for n in article.iter() if xml_local(n.tag) == "article-meta"), None)
    journal_meta = next((n for n in article.iter() if xml_local(n.tag) == "journal-meta"), None)
    title_group = next((n for n in article.iter() if xml_local(n.tag) == "article-title"), None)
    title = text_content(title_group)
    if not title:
        return None, "missing_title"
    licenses, _ = jats_governing_license_evidence(article_meta)
    approved_license = canonical_combined_cc_license(licenses)
    approved: tuple[str, str] | None = None
    if approved_license is not None:
        approved = next((item for item in licenses if canonical_cc_license(item[0], item[1]) == approved_license), None)
    if approved is None:
        return None, "missing_or_unapproved_explicit_item_license"
    doi_nodes = [n for n in article_meta.iter() if xml_local(n.tag) == "article-id" and n.attrib.get("pub-id-type", "").lower() == "doi"] if article_meta is not None else []
    doi = text_content(doi_nodes[0]) if doi_nodes else ""
    pmid_nodes = [n for n in article_meta.iter() if xml_local(n.tag) == "article-id" and n.attrib.get("pub-id-type", "").lower() in {"pmid", "pubmed"}] if article_meta is not None else []
    pmid = text_content(pmid_nodes[0]) if pmid_nodes else None
    article_type = article.attrib.get("article-type", "")
    if article_type in {"correction", "retraction", "expression-of-concern", "editorial", "letter", "news", "obituary"}:
        return None, f"excluded_article_type:{article_type}"
    sections, language_codes, figure_flags = extract_body(root)
    text_parts: list[str] = []
    for section in sections:
        heading = " / ".join(section.get("section_path", []))
        text_parts.append((f"{heading}\n" if heading else "") + section["text"])
    full_text = "\n\n".join(text_parts).strip()
    if len(full_text) < MIN_SCIENCE_CHARS:
        return None, "text_too_short_or_extraction_incomplete"
    if len(full_text) > MAX_SOURCE_CHARS:
        return None, "source_exceeds_100k_chars_hold"
    if len([s for s in sections if s.get("section_type") not in {"abstract", "references"}]) < 2:
        return None, "insufficient_section_structure"
    journal = ""
    if journal_meta is not None:
        journal = next((text_content(n) for n in journal_meta.iter() if xml_local(n.tag) == "journal-title" and text_content(n)), "")
    authors = []
    if article_meta is not None:
        for contrib in (n for n in article_meta.iter() if xml_local(n.tag) == "contrib" and n.attrib.get("contrib-type") == "author"):
            name = next((n for n in contrib.iter() if xml_local(n.tag) == "name"), None)
            if name is not None:
                surname = next((text_content(n) for n in name if xml_local(n.tag) == "surname"), "")
                given = next((text_content(n) for n in name if xml_local(n.tag) == "given-names"), "")
                display = " ".join(v for v in (given, surname) if v)
                if display:
                    authors.append(display)
            else:
                collab = next((text_content(n) for n in contrib.iter() if xml_local(n.tag) == "collab"), "")
                if collab:
                    authors.append(collab)
    if not authors:
        return None, "missing_author_attribution"
    date = parse_date(article_meta)
    language_codes = list(dict.fromkeys(normalize_science_language(code) for code in language_codes))
    lang = language_codes[0] if language_codes else normalize_science_language(article.attrib.get("lang", "und"))
    doi_normalized = normalize_doi(doi) if doi else None
    pmcid = pmcid if pmcid.upper().startswith("PMC") else f"PMC{pmcid}"
    article_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
    publisher = "PLOS" if journal.lower().startswith("plos") else journal or "PMC Open Access Subset"
    license_url, license_text = approved
    license_id, canonical_license_url = approved_license
    retrieved_at = utc_now()
    article_version = next((text_content(n) for n in article.iter() if xml_local(n.tag) == "article-version"), None)
    revision_evidence = {
        "pmcid": pmcid,
        "retrieval_timestamp": retrieved_at,
        "article_version": article_version,
        "source_revision_date": date,
    }
    revision_parts = [f"pmcid:{pmcid}", f"retrieved_at:{retrieved_at}"]
    if article_version:
        revision_parts.append(f"article_version:{article_version}")
    if date:
        revision_parts.append(f"source_revision_date:{date}")
    revision_parts.append(f"raw_jats_sha256:{sha(xml_bytes)}")
    tables = sum(1 for node in article.iter() if xml_local(node.tag) == "table-wrap")
    figure_count = sum(1 for node in article.iter() if xml_local(node.tag) == "fig")
    refs_count = sum(1 for node in article.iter() if xml_local(node.tag) == "ref")
    doc_id = f"science-{pmcid.lower()}"
    record = {
        "id": doc_id,
        "source_family": "scientific",
        "publisher": publisher,
        "journal": journal,
        "title": title,
        "authors": authors,
        "language": lang,
        "languages_present": language_codes or [lang],
        "original_translation": "source_native_paper",
        "url": article_url,
        "source_work_id": doi_normalized or pmcid,
        "pmcid": pmcid,
        "pmid": pmid,
        "doi": doi_normalized,
        "published_at": date,
        "article_type": article_type or None,
        "search_group": search_group,
        "license": {
            "id": license_id,
            "url": canonical_license_url,
            "source_url": license_url,
            "statement": license_text,
            "share_alike": False,
            "commercial_reuse_explicit": True,
        },
        "rights_evidence": {
            "checked_at": utc_now(),
            "policy_url": PMC_OAI_POLICY,
            "dataset_policy_url": PMC_DATA_POLICY,
            "license_filter_method": "exact CC-BY/CC0 machine-readable item license was inspected in PMC OAI-PMH `oai_dc` `dc:rights` metadata before full-text retrieval",
            "item_license_url": canonical_license_url,
            "item_license_source_url": license_url,
            "item_license_statement": license_text,
            "retrieval_route": "PMC OAI-PMH oai_dc item-rights metadata check followed by GetRecord metadataPrefix=pmc",
            "text_license_applies_to_article_text": True,
            "third_party_materials": "figure assets omitted; explicit third-party credits are checked separately and flagged at block level",
        },
        "revision": ";".join(revision_parts),
        "source_revision_evidence": revision_evidence,
        "retrieved_at": retrieved_at,
        "content_sha256": sha(full_text),
        "raw_jats_sha256": sha(xml_bytes),
        "text": full_text,
        "sections": sections,
        "section_count": len(sections),
        "table_count": tables,
        "figure_count": figure_count,
        "reference_count": refs_count,
        "figure_rights": figure_flags,
        "extraction": {
            "format": "JATS XML",
            "sections_preserved": True,
            "nested_sections_preserved": True,
            "procedures_and_lists_preserved": True,
            "tables_preserved_as_rows": True,
            "figure_assets_collected": False,
            "body_chars": len(full_text),
            "changes": "Normalized XML whitespace; section headings, paragraphs, lists, tables, captions, and back matter retained; figure files omitted; third-party flags retained separately.",
        },
        "relation_keys": [f"pmcid:{pmcid}"] + ([f"doi:{doi_normalized}"] if doi_normalized else []),
        "attribution": authors,
    }
    return record, "eligible"


def clean_plos_metadata_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    decoded = html.unescape(str(value or ""))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", decoded)).strip()


def plos_article_slug(journal: str, doi: str) -> str | None:
    by_journal = {
        "plos one": "plosone", "plos biology": "plosbiology", "plos medicine": "plosmedicine",
        "plos computational biology": "ploscompbiol", "plos genetics": "plosgenetics",
        "plos pathogens": "plospathogens", "plos neglected tropical diseases": "plosntds",
        "plos global public health": "globalpublichealth", "plos climate": "climate",
        "plos water": "water", "plos digital health": "digitalhealth",
    }
    normalized = re.sub(r"[^a-z ]", "", journal.casefold()).strip()
    if normalized in by_journal:
        return by_journal[normalized]
    match = re.search(r"10\.1371/journal\.([a-z]+)\.", doi, re.I)
    return {
        "pone": "plosone", "pbio": "plosbiology", "pmed": "plosmedicine",
        "pcbi": "ploscompbiol", "pgen": "plosgenetics", "ppat": "plospathogens",
        "pntd": "plosntds", "pclm": "climate", "pwat": "water",
        "pgph": "globalpublichealth", "pdig": "digitalhealth",
    }.get(match.group(1).lower()) if match else None


def plos_candidate_from_solr(doc: dict[str, Any], *, query_group: str, request: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    doi = normalize_doi(str(doc.get("id") or ""))
    if not doi.startswith("10.1371/"):
        return None, "missing_or_non_plos_doi"
    title = clean_plos_metadata_text(doc.get("title_display"))
    abstract = clean_plos_metadata_text(doc.get("abstract"))
    subjects_value = doc.get("subject") or []
    subjects = " ".join(clean_plos_metadata_text(value) for value in (subjects_value if isinstance(subjects_value, list) else [subjects_value]))
    copyright_statement = clean_plos_metadata_text(doc.get("copyright"))
    metadata_hint = canonical_combined_cc_license([(copyright_statement, copyright_statement)]) if copyright_statement else None
    if metadata_hint is None:
        return None, "copyright_metadata_not_exact_ccby_or_cc0_hint"
    article_type = clean_plos_metadata_text(doc.get("article_type"))
    if article_type.casefold() != "research article":
        return None, "non_research_article_type"
    topic, topic_reason = classify_science_topic(title, abstract, subjects, "und")
    if topic_reason:
        return None, topic_reason
    journal = clean_plos_metadata_text(doc.get("journal"))
    journal_slug = plos_article_slug(journal, doi)
    if not journal_slug:
        return None, "unsupported_plos_journal_route"
    publication = clean_plos_metadata_text(doc.get("publication_date"))
    date_match = re.match(r"(\d{4}(?:-\d{2}(?:-\d{2})?)?)", publication)
    authors_value = doc.get("author_display") or []
    authors = [clean_plos_metadata_text(value) for value in (authors_value if isinstance(authors_value, list) else [authors_value]) if clean_plos_metadata_text(value)]
    doi_suffix = doi.removeprefix("10.1371/journal.").replace(".", "-")
    source_id = f"science-plos-{doi_suffix}"
    publisher_article_url = f"https://journals.plos.org/{journal_slug}/article?id={quote(doi, safe='') }"
    return {
        "id": source_id,
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "subjects": subjects_value,
        "journal": journal,
        "journal_slug": journal_slug,
        "publisher_article_url": publisher_article_url,
        "published_at": date_match.group(1) if date_match else None,
        "article_type": article_type,
        "authors_metadata": authors,
        "language_metadata": None,
        "language_metadata_field_available": False,
        "topic_bucket": topic,
        "search_group": f"plos_{topic}",
        "query_group": query_group,
        "metadata_license_hint": {"id": metadata_hint[0], "url": metadata_hint[1], "statement": copyright_statement},
        "metadata_license_is_not_article_rights_proof": True,
        "solr_metadata_raw_file": request.get("raw_file"),
        "solr_metadata_raw_sha256": request.get("sha256"),
        "solr_request": request,
    }, None


def parse_plos_science_document(xml_bytes: bytes, candidate: dict[str, Any], access: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any], str]:
    identity: dict[str, Any] = {
        "canonical_doi": candidate.get("doi"),
        "canonical_url": f"https://doi.org/{candidate.get('doi')}",
        "solr_metadata_raw_file": candidate.get("solr_metadata_raw_file"),
        "solr_metadata_raw_sha256": candidate.get("solr_metadata_raw_sha256"),
        "plos_jats_raw_file": access.get("raw_file"),
        "plos_jats_raw_sha256": access.get("sha256"),
        "front_article_ids": [],
        "jats_governing_license_evidence": [],
        "same_item_rights_verified": False,
        "metadata_license_is_not_article_rights_proof": True,
        "solr_title": candidate.get("title"),
    }
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return None, identity, "xml_parse_error"
    article = next((node for node in root.iter() if xml_local(node.tag) == "article"), None)
    article_meta = next((node for node in article.iter() if xml_local(node.tag) == "article-meta"), None) if article is not None else None
    title_node = next((node for node in (article_meta.iter() if article_meta is not None else []) if xml_local(node.tag) == "article-title"), None)
    jats_title = text_content(title_node)
    ids = [node for node in (article_meta.iter() if article_meta is not None else []) if xml_local(node.tag) == "article-id"]
    front_ids = [{"type": node.attrib.get("pub-id-type", "").lower(), "value": text_content(node)} for node in ids]
    doi_values = sorted({normalize_doi(item["value"]).rstrip("/") for item in front_ids if item["type"] == "doi" and item.get("value")})
    pmcids = sorted({item["value"].upper() for item in front_ids if item["type"] in {"pmcid", "pmc"} and item.get("value")})
    license_terms, license_evidence = jats_governing_license_evidence(article_meta)
    jats_license = canonical_combined_cc_license(license_terms)
    expected_license = candidate.get("metadata_license_hint") or {}
    expected_doi = normalize_doi(str(candidate.get("doi") or "")).rstrip("/")
    title_match = bool(jats_title and normalized_title_identity(jats_title) == normalized_title_identity(str(candidate.get("title") or "")))
    doi_match = expected_doi in doi_values
    license_match = jats_license == (expected_license.get("id"), expected_license.get("url"))
    identity.update({
        "front_article_ids": front_ids,
        "front_article_title": jats_title,
        "front_title_match": title_match,
        "front_dois": doi_values,
        "front_pmcids": pmcids,
        "jats_governing_license": jats_license,
        "jats_governing_license_evidence": license_evidence,
        "solr_metadata_license_hint": expected_license,
        "same_item_rights_verified": bool(doi_match and title_match and license_match),
        "solr_doi_match": doi_match,
        "solr_jats_license_match": license_match,
    })
    if article is None or article_meta is None:
        return None, identity, "jats_front_article_meta_missing"
    if not doi_match:
        return None, identity, "solr_jats_doi_mismatch"
    if not title_match:
        return None, identity, "solr_jats_title_mismatch"
    if not license_match:
        return None, identity, "solr_jats_item_license_mismatch_or_conflict"
    record, reason = parse_science_document(xml_bytes, f"PLOS-{expected_doi.rsplit('.', 1)[-1]}", str(candidate.get("search_group") or "plos"))
    if record is None:
        return None, identity, reason
    retrieved_at = str(access.get("retrieved_at") or access.get("original_retrieved_at") or utc_now())
    doi = normalize_doi(str(candidate["doi"]))
    pmcid = pmcids[0] if len(pmcids) == 1 else None
    article_version = next((text_content(node) for node in article.iter() if xml_local(node.tag) == "article-version"), None)
    publisher_url = str(candidate.get("publisher_article_url") or f"https://doi.org/{doi}")
    record.update({
        "id": str(candidate["id"]),
        "publisher": "PLOS",
        "url": f"https://doi.org/{doi}",
        "original_url": publisher_url,
        "source_work_id": doi,
        "pmcid": pmcid,
        "doi": doi,
        "published_at": candidate.get("published_at") or record.get("published_at"),
        "search_group": candidate.get("search_group"),
        "retrieved_at": retrieved_at,
        "revision": ";".join(part for part in (
            f"doi:{doi}", f"retrieved_at:{retrieved_at}",
            f"article_version:{article_version}" if article_version else None,
            f"source_revision_date:{record.get('published_at')}" if record.get("published_at") else None,
            f"raw_jats_sha256:{sha(xml_bytes)}",
        ) if part),
        "source_revision_evidence": {
            "doi": doi, "retrieval_timestamp": retrieved_at,
            "article_version": article_version, "source_revision_date": record.get("published_at"),
            "raw_jats_sha256": sha(xml_bytes),
        },
        "rights_evidence": {
            "checked_at": utc_now(),
            "policy_url": PLOS_TDM_POLICY,
            "api_documentation_url": PLOS_TDM_API_DOCS,
            "license_filter_method": "PLOS Solr copyright metadata was only a candidate hint; article-text rights were admitted only after the official JATS front/article-meta permissions and license wording/hrefs matched the same DOI and title.",
            "item_license_url": record["license"]["url"],
            "item_license_source_url": next((href for href, statement in license_terms if canonical_cc_license(href, statement) == jats_license), ""),
            "item_license_statement": " ".join(statement for href, statement in license_terms if statement),
            "retrieval_route": "PLOS Solr metadata discovery followed by official per-article JATS manuscript XML under the published text-and-data-mining route",
            "text_license_applies_to_article_text": True,
            "third_party_materials": "figure assets omitted; article-level third-party credits are flagged separately and not treated as cleared by the article license",
        },
        "source_access": {
            "api_url": access.get("url"), "publisher_article_url": publisher_url,
            "retrieval_timestamp": retrieved_at, "response_sha256": access.get("sha256"),
            "response_bytes_decoded": access.get("response_bytes_decoded"), "raw_file": access.get("raw_file"),
            "http_status": access.get("http_status"),
            "solr_metadata_raw_file": candidate.get("solr_metadata_raw_file"),
            "solr_metadata_raw_sha256": candidate.get("solr_metadata_raw_sha256"),
            "solr_search_url": (candidate.get("solr_request") or {}).get("url"),
            "metadata_license_hint": expected_license,
            "metadata_license_is_not_article_rights_proof": True,
        },
        "relation_keys": [f"doi:{doi}"] + ([f"pmcid:{pmcid}"] if pmcid else []),
    })
    identity["same_item_rights_verified"] = True
    identity["source_json_license"] = record.get("license")
    identity["source_json_sha256"] = None
    return record, identity, "eligible_proof_pending_adapter"


def science_license_filter_query(extra: str = "") -> str:
    base = f"({LICENSE_FILTER})"
    return f"({extra}) AND {base}" if extra else base


def canonical_cc_license(href: str, statement: str) -> tuple[str, str] | None:
    value = (href + " " + statement).lower()
    if re.search(
        r"creativecommons\.org/licenses/by-(?:nc|nd|sa)(?:-(?:nd|sa))?/"
        r"|\b(?:cc[- ]?by)[-_ ]?(?:nc|nd|sa)\b"
        r"|\battribution[- ](?:non[- ]?commercial|no[- ]?derivatives|no[- ]?derivs|share[- ]?alike)\b"
        r"|\bnon[- ]?commercial\b|\bno[- ]?derivatives\b|\bno[- ]?derivs\b|\bshare[- ]?alike\b",
        value,
    ):
        return None
    if "creativecommons.org/publicdomain/zero/" in value or re.search(r"\bcc0(?:[- ]1\.0)?\b|creative commons zero", value):
        return "CC0-1.0", "https://creativecommons.org/publicdomain/zero/1.0/"
    match = re.search(r"creativecommons\.org/licenses/by/(\d+(?:\.\d+)?)/?", value)
    if not match:
        match = re.search(r"(?:cc[- ]?by|creative commons attribution)[- ]+(\d+(?:\.\d+)?)", value)
    if match:
        version = match.group(1)
        return f"CC-BY-{version}", f"https://creativecommons.org/licenses/by/{version}/"
    return None


def canonical_combined_cc_license(evidence: list[tuple[str, str]]) -> tuple[str, str] | None:
    """Accept only one consistent, unrestricted CC license across all item terms."""
    nonempty = [(href, statement) for href, statement in evidence if href.strip() or statement.strip()]
    if not nonempty:
        return None
    combined = " ".join(f"{href} {statement}" for href, statement in nonempty)
    combined_license = canonical_cc_license(combined, combined)
    if combined_license is None:
        return None
    explicit = {license for href, statement in nonempty if (license := canonical_cc_license(href, statement))}
    if len(explicit) > 1 or (explicit and combined_license not in explicit):
        return None
    return combined_license


def normalize_science_language(value: str) -> str:
    code = value.strip().lower().replace("_", "-") or "und"
    aliases = {"eng": "en", "fre": "fr", "fra": "fr", "ger": "de", "deu": "de", "spa": "es", "por": "pt", "ita": "it", "dut": "nl", "nld": "nl", "chi": "zh", "zho": "zh", "jpn": "ja", "kor": "ko", "wel": "cy", "cym": "cy", "und": "und"}
    return aliases.get(code, code)


def normalized_title_identity(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value).casefold()
    return "".join(char for char in decomposed if not unicodedata.combining(char) and char.isalnum())


def oai_dc_fields(record: ET.Element) -> dict[str, list[str]]:
    dc = next((node for node in record.iter() if xml_local(node.tag) == "dc"), None)
    values: dict[str, list[str]] = {}
    for node in (dc.iter() if dc is not None else []):
        key = xml_local(node.tag).lower()
        if key == "dc":
            continue
        value = " ".join(text_content(node).split())
        if value:
            values.setdefault(key, []).append(value)
    return values


SCIENCE_TITLE_TOPIC_PATTERNS = {
    "environment_ecology": re.compile(r"\b(climate(?: change)?|ecolog\w*|biodivers\w*|conservation|ecosystem\w*|wildlife|forest\w*|marine|ocean\w*|marine mammal\w*|cetacean\w*|narwhal\w*|prey selection|wildfire\w*|water resources|pollution|land use|water quality|rainfall|wetland\w*|macroinvertebrate\w*|resource efficiency|material flow|phosphorus budget|plastome|plant species|agro-ecolog\w*|bat fatalities|bird fatalities|wind farm\w*|wind-energy facilit\w*|offshore wind|commercial fishing)\b", re.I),
    "agriculture_food": re.compile(r"\b(agricultur\w*|crop\w*|farming|farmers?|livestock|food systems?|food security|food production|soil health|irrigation|fish(?:ing|ery|eries)|aquaculture)\b", re.I),
    "society_behavior": re.compile(r"\b(social science|social collaboration|social organization\w*|education|student\w*|school\w*|community|communities|service user organizations|local association\w*|inequalit\w*|human migration|migrant communities|governance|public polic\w*|workplace|employment|higher education|household)\b", re.I),
    "public_health": re.compile(r"\b(public health|epidemiolog\w*|health service\w*|healthcare access|health care access|health polic\w*|population health|maternal and child|breastfeeding|nutrition tracker|nutrition epidemiolog\w*|dietary intake|niacin|hearing loss|food marketing|obesity|adolescent health|dietary behavio\w*|sanitation|hygiene|schistosomiasis|triatomines?|kissing bug|vector surveillanc\w*|diabetes.{0,50}communities|mental health service user|health system\w*|disease surveillanc\w*)\b", re.I),
    "physical_science": re.compile(r"\b(material stock|material flow|resource efficiency|energy|renewable|solar|battery|engineering|geolog\w*|geophysic\w*)\b", re.I),
}
SCIENCE_TITLE_BIOLOGICAL_PATTERN = re.compile(
    r"\b(synthetic biolog\w*|biological function|genetic sensor\w*|genotype[- ]phenotype|"
    r"antiviral antibody|chimeric immunoglobulin|immunoglobulin m|"
    r"genetic\w*|genom\w*|gene\w*|chromosom\w*|molecular biolog\w*|cell biolog\w*|"
    r"apoptos\w*|mitos\w*|protein\w*|enzyme\w*|microbi\w*|bacter\w*|phage\w*|"
    r"virolog\w*|viral|host[- ]pathogen|circ\s*rna\w*|micro\s*rna\w*|rna sequencing|"
    r"neuroscien\w*|neuron\w*|brain activation|cerebral cortex|cerebral cortical|"
    r"cortical polarity|interocular disparity|photoreception|circadian|inner clock|"
    r"transferrin|smim1|pik3r1|er stress|endoplasmic reticulum stress|cell migration|"
    r"cell invasion|drug resistance|multidrug resistance|cellular mechanism\w*|"
    r"molecular mechanism\w*|mitotic\w*)\b",
    re.I,
)
SCIENCE_TITLE_CLINICAL_MARKERS = re.compile(
    r"\b(cancer|tumou?r|leukemia|carcinoma|myopia|macular|ophthalmic|retina|visual field|interocular|cortex|biomarker|micro-?rna|gene expression|chromosome|cell[- ]derived|cell migration|mesangial|glomerulonephritis|patient-derived|surgical resection|vascular access|clinical test|in vitro|drug resistance|receptor kinase)\b",
    re.I,
)
SCIENCE_TITLE_PATIENT_CARE_MARKERS = re.compile(
    r"\b(patient(?:s)?|patient-derived|case report|clinical trial|clinical test|"
    r"diagnos\w*|prognos\w*|biomarker|follow[- ]?up|surgical resection|"
    r"screening test|medical advice|treatment guidelines?)\b",
    re.I,
)
SCIENCE_TITLE_BASIC_BIOLOGY_EXCLUSIONS = re.compile(
    r"\b(intermittent energy restriction)\b",
    re.I,
)
SCIENCE_TITLE_COMPUTATIONAL_RESOURCE_EXCLUSIONS = re.compile(
    r"\b(template-based protein structure prediction)\b",
    re.I,
)
SCIENCE_EXTERNAL_EVAL_MARKERS = re.compile(
    r"\b(benchmark(?:ing)?|leaderboard|shared task|external evaluation dataset|evaluation benchmark|MMLU|SuperGLUE|ImageNet|SQuAD|large language model|LLM-based|machine learning benchmark|"
    r"(?:(?:machine learning|deep learning|neural network|classifier|language model|large language model|LLM).{0,120}(?:test set|training dataset|held[- ]out set)|"
    r"(?:test set|training dataset|held[- ]out set).{0,120}(?:machine learning|deep learning|neural network|classifier|language model|large language model|LLM)))\b",
    re.I,
)


SCIENCE_NATIVE_TITLE_TOPIC_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {
    "de": {
        "environment_ecology": re.compile(r"\b(klimawandel|ökolog\w*|biodiversit\w*|naturschutz|ökosystem\w*|wald\w*|meer\w*|verschmutzung|wasserqualität|landnutzung|regenfall)\b", re.I),
        "agriculture_food": re.compile(r"\b(landwirtschaft|anbau|ernte\w*|bauern?|vieh\w*|lebensmittel\w*|ernährungssystem\w*|bewässerung|fischerei|aquakultur)\b", re.I),
        "society_behavior": re.compile(r"\b(sozialwissenschaft\w*|bildung|studierende|schüler\w*|gemeinschaft\w*|ungleichheit\w*|migration|verwaltung|beschäftigung|haushalt\w*)\b", re.I),
        "public_health": re.compile(r"\b(öffentliche gesundheit|epidemiolog\w*|gesundheitsversorgung|gesundheitssystem\w*|gesundheitspolitik|bevölkerungsgesundheit|stillen|krankheitsüberwachung)\b", re.I),
        "physical_science": re.compile(r"\b(materialbestand|materialfluss|ressourceneffizienz|energie|erneuerbar\w*|solar\w*|batterie\w*|ingenieur\w*|geologie|geophysik)\b", re.I),
        "biological_science": re.compile(r"\b(genetik|genomik|molekularbiolog\w*|zellbiolog\w*|apoptose|mitose|protein\w*|enzym\w*|mikrobi\w*|bakter\w*|phage\w*|chromosom\w*|lichtwahrnehmung|innere uhr|biorhythm\w*|circadian\w*|neuro\w*|hirnaktivierung|zellmigration|endoplasmatischer retikulumstress)\b", re.I),
    },
    "fr": {
        "environment_ecology": re.compile(r"\b(climat\w*|écolog\w*|biodiversit\w*|conservation|écosystème\w*|forêt\w*|marin\w*|pollution|qualité de l’eau|qualité de l'eau|utilisation des terres|pluviométrie)\b", re.I),
        "agriculture_food": re.compile(r"\b(agricultur\w*|culture\w*|agriculteur\w*|élevage|systèmes? alimentaires?|sécurité alimentaire|irrigation|pêche|aquaculture)\b", re.I),
        "society_behavior": re.compile(r"\b(sciences sociales|éducation|étudiant\w*|école\w*|communauté\w*|inégalités?|migration|gouvernance|emploi|ménage\w*)\b", re.I),
        "public_health": re.compile(r"\b(santé publique|épidémiolog\w*|services? de santé|accès aux soins|politique de santé|santé de la population|allaitement|surveillance des maladies)\b", re.I),
        "physical_science": re.compile(r"\b(stock de matériaux|flux de matières|efficacité des ressources|énergie|renouvelable|solaire|batterie|ingénierie|géologie|géophysique)\b", re.I),
        "biological_science": re.compile(r"\b(génétique|génomique|biologie moléculaire|biologie cellulaire|apoptose|mitose|protéine\w*|enzyme\w*|microbiolog\w*|bactérie\w*|phage\w*|chromosome\w*|perception de la lumière|horloge circadienne|neuroscien\w*|activation cérébrale|migration cellulaire|stress du réticulum endoplasmique)\b", re.I),
    },
}


def classify_science_topic(title: str, abstract: str, subjects: str, language: str = "en") -> tuple[str | None, str | None]:
    evidence = " ".join((title, abstract, subjects))
    if SCIENCE_EXTERNAL_EVAL_MARKERS.search(evidence):
        return None, "external_benchmark_or_evaluation_topic"
    if SCIENCE_TITLE_BASIC_BIOLOGY_EXCLUSIONS.search(title):
        return None, "narrow_clinical_or_mechanistic_biomedical_title"
    if SCIENCE_TITLE_COMPUTATIONAL_RESOURCE_EXCLUSIONS.search(title):
        return None, "borderline_computational_resource_topic"
    lang = normalize_science_language(language)
    title_matches = [topic for topic, pattern in SCIENCE_TITLE_TOPIC_PATTERNS.items() if pattern.search(title)]
    context_matches = [topic for topic, pattern in SCIENCE_TITLE_TOPIC_PATTERNS.items() if pattern.search(abstract)]
    matches = list(title_matches)
    if not matches:
        matches = context_matches
    elif "physical_science" in title_matches and "environment_ecology" in context_matches:
        matches.append("environment_ecology")
    matches.extend(
        topic for topic, pattern in SCIENCE_NATIVE_TITLE_TOPIC_PATTERNS.get(lang, {}).items()
        if topic not in matches and pattern.search(title)
    )
    biological_match = SCIENCE_TITLE_BIOLOGICAL_PATTERN.search(title) or (
        SCIENCE_NATIVE_TITLE_TOPIC_PATTERNS.get(lang, {}).get("biological_science")
        and SCIENCE_NATIVE_TITLE_TOPIC_PATTERNS[lang]["biological_science"].search(title)
    )
    if biological_match:
        if SCIENCE_TITLE_PATIENT_CARE_MARKERS.search(title):
            return None, "narrow_clinical_or_mechanistic_biomedical_title"
        return "biological_science", None
    if not matches:
        return None, "no_explicit_breadth_topic_in_title"
    population_health_context = any(topic in matches for topic in ("society_behavior", "public_health"))
    if SCIENCE_TITLE_CLINICAL_MARKERS.search(title) and not population_health_context:
        return None, "narrow_clinical_or_mechanistic_biomedical_title"
    priority = ("environment_ecology", "agriculture_food", "society_behavior", "public_health", "physical_science", "biological_science")
    topic = next(value for value in priority if value in matches)
    return topic, None


def candidate_from_oai_dc(record: ET.Element, group: str, page_sha: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    header = next((node for node in list(record) if xml_local(node.tag) == "header"), None)
    if header is None or header.attrib.get("status") == "deleted":
        return None, None, "deleted_or_missing_header"
    identifier = next((text_content(node) for node in header if xml_local(node.tag) == "identifier"), "")
    match = re.search(r"(\d+)$", identifier)
    if not match:
        return None, None, "missing_pmc_identifier"
    pmcid = "PMC" + match.group(1)
    values = oai_dc_fields(record)
    rights = values.get("rights", [])
    license_statement = " ".join(rights)
    license_record = canonical_combined_cc_license([(value, value) for value in rights]) if rights else None
    lang = normalize_science_language((values.get("language") or ["und"])[0])
    if license_record is None:
        return None, pmcid, "missing_or_unapproved_explicit_item_license"
    title = (values.get("title") or [""])[0]
    if not title:
        return None, pmcid, "missing_title"
    abstracts = values.get("description", [])
    subject = " ".join(values.get("subject", []))
    topic, topic_exclusion = classify_science_topic(title, " ".join(abstracts), subject, lang)
    if topic_exclusion:
        return None, pmcid, topic_exclusion
    identifiers = values.get("identifier", [])
    doi = next((normalize_doi(m.group(0).rstrip(".,;:)")) for identifier_value in identifiers if (m := re.search(r"10\.\d{4,9}/[^\s<>\"]+", identifier_value, re.I))), "")
    sources = values.get("source", [])
    journal = sources[0] if sources else ""
    publisher_value = (values.get("publisher") or [""])[0]
    authors = values.get("creator", [])
    date_value = (values.get("date") or [""])[0]
    date_match = re.search(r"\d{4}(?:-\d{2}(?:-\d{2})?)?", date_value)
    date = date_match.group(0) if date_match else None
    is_plos = bool(re.search(r"\bPLOS|\bPLoS", " ".join(sources + [publisher_value]), re.I))
    if is_plos:
        publisher = "PLOS"
    else:
        publisher = publisher_value or journal or "PMC Open Access Subset"
    nonenglish = lang not in {"en", "und"}
    bucket = f"native_{lang}_{topic}" if nonenglish else f"topic_{topic}"
    language_values = [normalize_science_language(value) for value in values.get("language", [])]
    languages = sorted(set(language_values or [lang]))
    return {
        "pmcid": pmcid,
        "doi": doi or None,
        "title": title,
        "authors": authors,
        "journal": journal,
        "publisher": publisher,
        "language": lang,
        "languages_present": languages,
        "published_at": date,
        "license_id": license_record[0],
        "license_url": license_record[1],
        "license_statement": license_statement,
        "topic_bucket": topic,
        "search_group": bucket,
        "candidate_set": group,
        "frontmatter_sha256": page_sha,
    }, pmcid, None


SCIENCE_TOPIC_REVIEW = RUN / "science-topic-review.json"
SCIENCE_CONTRACT_PROJECTIONS = RUN / "science-source-contract-projections.jsonl"
SCIENCE_IDENTIFIER_AUDIT = RUN / "science-source-identity-audit.json"
SCIENCE_TOPIC_REPLACEMENTS = RUN / "science-topic-replacements.json"
SCIENCE_PLOS_MANIFEST = RUN / "science-plos-documents.jsonl"
SCIENCE_PLOS_CANDIDATES = RUN / "science-plos-candidates.json"
SCIENCE_PLOS_ACCESS_AUDIT = RUN / "science-plos-access-audit.json"
SCIENCE_PLOS_IDENTITY_AUDIT = RUN / "science-plos-identity-audit.json"
SCIENCE_PLOS_IDENTITY_AUDIT_REVISIONS = RUN / "science-plos-identity-audit-revisions"
SCIENCE_PLOS_CANDIDATE_REVISIONS = RUN / "science-plos-candidate-revisions"
SCIENCE_PLOS_TOPIC_REVIEW = RUN / "science-plos-topic-review.json"
SCIENCE_PLOS_LANGUAGE_INVENTORY = RUN / "science-plos-language-inventory.json"
SCIENCE_PLOS_TOPIC_REVISION_EVENTS = RUN / "science-plos-topic-revision-events.jsonl"
SCIENCE_PLOS_TOPIC_REVISIONS = RUN / "science-plos-topic-review-revisions"
SCIENCE_PLOS_MANIFEST_REVISIONS = RUN / "science-plos-manifest-revisions"
SCIENCE_PLOS_CANDIDATE_RULE_REVISION_EVENTS = RUN / "science-plos-candidate-rule-revision-events.jsonl"
SCIENCE_PLOS_PENDING_DOCS = DOCS / "plos-proof-pending-adapter"


def refresh_science_review(science: dict[str, Any], manifest_docs: list[dict[str, Any]], persist=None) -> dict[str, Any]:
    review = write_science_topic_review(manifest_docs)
    identity_audit = read_json(SCIENCE_IDENTIFIER_AUDIT, {}) if SCIENCE_IDENTIFIER_AUDIT.exists() else None
    _apply_science_review_counts(science, manifest_docs, review, identity_audit)
    if persist is not None:
        persist()
    return review


def _apply_science_review_counts(science: dict[str, Any], manifest_docs: list[dict[str, Any]], review: dict[str, Any], identity_audit: dict[str, Any] | None = None) -> None:
    topic_eligible_rows = [row for row in review["sources"] if row.get("decision") == "eligible"]
    identity_rows = (identity_audit or {}).get("sources") or {}
    if identity_audit is None:
        eligible_rows = topic_eligible_rows
    else:
        eligible_rows = [
            row for row in topic_eligible_rows
            if (identity := identity_rows.get(row.get("id")))
            and identity.get("decision") == "verified"
            and identity.get("raw_source_json_sha256") == row.get("source_json_sha256")
            and identity.get("topic_review_record_sha256") == canonical_science_review_row_sha256(row)
        ]
    science["retrieved_source_document_count"] = len(manifest_docs)
    science["topic_eligible_source_document_count"] = len(topic_eligible_rows)
    science["identity_verified_eligible_source_document_count"] = len(eligible_rows)
    science["identity_pending_or_held_eligible_source_document_count"] = len(topic_eligible_rows) - len(eligible_rows)
    science["eligible_source_document_count"] = len(eligible_rows)
    science["held_source_document_count"] = review["counts"]["held"]
    science["accepted_work_count"] = len({str(row.get("doi") or row.get("pmcid") or row.get("id")) for row in eligible_rows})
    science["accepted_document_count"] = len(eligible_rows)
    science["eligible_topic_counts"] = review["counts"]["eligible_by_topic"]
    verified_topic_counts: dict[str, int] = {}
    for row in eligible_rows:
        topic = str(row.get("topic") or "unclassified")
        verified_topic_counts[topic] = verified_topic_counts.get(topic, 0) + 1
    science["verified_eligible_topic_counts"] = verified_topic_counts
    science["topic_review_file"] = raw_rel(SCIENCE_TOPIC_REVIEW)
    science["source_contract_projection_file"] = raw_rel(SCIENCE_CONTRACT_PROJECTIONS)
    science["topic_review_sha256"] = sha(SCIENCE_TOPIC_REVIEW.read_bytes())
    science["eligible_sources"] = [
        {"id": row["id"], "pmcid": row.get("pmcid"), "doi": row.get("doi"), "language": row.get("language"),
         "title": row.get("title"), "topic": row.get("topic"), "source_json": row.get("source_json"),
         "source_json_sha256": row.get("source_json_sha256"), "contract_projection_file": row.get("contract_projection_file")}
        for row in eligible_rows
    ]
    non_english = [doc for doc in manifest_docs if str(doc.get("language", "")).lower() not in {"en", "eng", "und", ""}]
    language_inventory = science.setdefault("language_inventory", {})
    language_inventory["non_english_fulltext_papers_found"] = len(non_english)
    language_inventory["non_english_fulltext_languages_found"] = sorted({str(doc.get("language")) for doc in non_english})
    language_inventory["non_english_fulltext_count_scope"] = "full-text source documents actually retrieved and written to the science manifest"


def project_science_source_contract(document: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str], list[str]]:
    projected = copy.deepcopy(document)
    additions: list[str] = []
    issues: list[str] = []
    revision = projected.get("revision")
    if isinstance(revision, dict):
        projected["source_revision_evidence"] = copy.deepcopy(revision)
        parts = []
        for key in ("pmcid", "retrieval_timestamp", "article_version", "source_revision_date"):
            value = revision.get(key)
            if value:
                parts.append(f"{key}:{value}")
        if projected.get("raw_jats_sha256"):
            parts.append(f"raw_jats_sha256:{projected['raw_jats_sha256']}")
        projected["revision"] = ";".join(parts)
        additions.append("converted the structured revision object to the required revision string and preserved it as source_revision_evidence")
    elif not isinstance(revision, str) or not revision.strip():
        issues.append("missing_revision")
    license_info = projected.get("license")
    if isinstance(license_info, dict) and "share_alike" not in license_info:
        license_id = str(license_info.get("id") or "")
        if license_id.startswith("CC-BY-") or license_id == "CC0-1.0":
            license_info["share_alike"] = False
            additions.append("added share_alike=false from the explicit CC-BY/CC0 license identifier")
        else:
            issues.append("share_alike_status_not_derivable_from_explicit_license")
    if not projected.get("attribution"):
        issues.append("missing_attribution")
    if not isinstance(projected.get("relation_keys"), list) or not projected.get("relation_keys"):
        issues.append("missing_relation_keys")
    if not isinstance(projected.get("text"), str) or sha(projected.get("text", "")) != projected.get("content_sha256"):
        issues.append("source_text_hash_mismatch")
    if not projected.get("rights_evidence"):
        issues.append("missing_rights_evidence")
    return (projected if not issues else None), additions, issues


def write_science_topic_review(manifest_docs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Record topic decisions without changing frozen acquisition source JSONs."""
    if manifest_docs is None:
        manifest_docs = existing_records(SCIENCE_MANIFEST)
    previous_review = read_json(SCIENCE_TOPIC_REVIEW, {})
    previous_rows = {
        str(row.get("id")): row
        for row in previous_review.get("sources", [])
        if isinstance(row, dict) and row.get("id")
    }
    replacement_audit = read_json(SCIENCE_TOPIC_REPLACEMENTS, {"sources": []})
    replacement_rows = {
        str(row.get("source_id")): row
        for row in replacement_audit.get("sources", [])
        if isinstance(row, dict) and row.get("source_id")
    }
    previous_projections = {
        str(row.get("source_id")): row
        for row in existing_records(SCIENCE_CONTRACT_PROJECTIONS)
        if row.get("source_id")
    }
    review_rows: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    topic_counts: dict[str, int] = {}
    held_counts: dict[str, int] = {}
    for manifest_row in manifest_docs:
        source_id = str(manifest_row.get("id") or "")
        source_path = DOCS / f"{source_id}.json"
        if not source_path.exists():
            review_rows.append({"id": source_id, "decision": "hold", "reason_codes": ["source_json_missing"]})
            held_counts["source_json_missing"] = held_counts.get("source_json_missing", 0) + 1
            continue
        source_bytes = source_path.read_bytes()
        try:
            source = json.loads(source_bytes)
        except json.JSONDecodeError:
            review_rows.append({"id": source_id, "decision": "hold", "reason_codes": ["source_json_invalid"]})
            held_counts["source_json_invalid"] = held_counts.get("source_json_invalid", 0) + 1
            continue
        replacement = replacement_rows.get(source_id)
        if replacement and replacement.get("source_json_sha256") == sha(source_bytes):
            prior_topic_row = previous_rows.get(source_id) or {}
            reason = "scope_budget_replaced_before_authoring"
            row = {
                "id": source_id,
                "decision": "hold",
                "topic": prior_topic_row.get("topic"),
                "reason_codes": [reason],
                "title": source.get("title"),
                "pmcid": source.get("pmcid"),
                "doi": source.get("doi"),
                "language": source.get("language"),
                "license_id": (source.get("license") or {}).get("id"),
                "license_url": (source.get("license") or {}).get("url"),
                "source_json": raw_rel(source_path),
                "source_json_sha256": sha(source_bytes),
                "source_text_sha256": source.get("content_sha256"),
                "raw_jats_sha256": source.get("raw_jats_sha256"),
                "basis": "eligible source retained but replaced before authoring to keep the biological_science ceiling while admitting a verified native-language paper",
                "replaced_by_source_id": replacement.get("replacement_source_id"),
                "previous_topic_review_record_sha256": replacement.get("previous_topic_review_record_sha256"),
            }
            review_rows.append(row)
            held_counts[reason] = held_counts.get(reason, 0) + 1
            continue
        previous_row = previous_rows.get(source_id)
        if (
            previous_row
            and previous_row.get("decision") == "eligible"
            and previous_row.get("source_json_sha256") == sha(source_bytes)
        ):
            frozen_row = copy.deepcopy(previous_row)
            topic = str(frozen_row.get("topic") or "unclassified")
            topic_counts[topic] = topic_counts.get(topic, 0) + 1
            prior_projection = previous_projections.get(source_id)
            if prior_projection is not None:
                projections.append(prior_projection)
            review_rows.append(frozen_row)
            continue
        abstracts = " ".join(
            str(section.get("text") or "")
            for section in source.get("sections", [])
            if section.get("section_type") == "abstract"
        )
        topic, scope_reason = classify_science_topic(
            str(source.get("title") or ""), abstracts, "", str(source.get("language") or "und")
        )
        projected, additions, contract_issues = project_science_source_contract(source)
        reasons = ([scope_reason] if scope_reason else []) + contract_issues
        license_info = source.get("license") or {}
        rights_evidence = source.get("rights_evidence") or {}
        license_statement = str(rights_evidence.get("item_license_statement") or license_info.get("statement") or "")
        observed_license = canonical_cc_license(str(license_info.get("source_url") or license_info.get("url") or ""), license_statement)
        if observed_license != (license_info.get("id"), license_info.get("url")):
            reasons.append("explicit_item_license_evidence_conflict")
        decision = "eligible" if not reasons else "hold"
        if decision == "eligible" and topic in SCIENCE_TOPIC_LIMITS and topic_counts.get(topic or "", 0) >= SCIENCE_TOPIC_LIMITS[topic]:
            decision = "hold"
            reasons.append(f"topic_limit_{topic}_{SCIENCE_TOPIC_LIMITS[topic]}")
        row = {
            "id": source_id,
            "decision": decision,
            "topic": topic,
            "reason_codes": reasons,
            "title": source.get("title"),
            "pmcid": source.get("pmcid"),
            "doi": source.get("doi"),
            "language": source.get("language"),
            "license_id": (source.get("license") or {}).get("id"),
            "license_url": (source.get("license") or {}).get("url"),
            "source_json": raw_rel(source_path),
            "source_json_sha256": sha(source_bytes),
            "source_text_sha256": source.get("content_sha256"),
            "raw_jats_sha256": source.get("raw_jats_sha256"),
            "basis": "source title and its structured abstract only; full-text body and original source JSON are not edited",
            "mechanical_contract_additions": additions,
        }
        if decision == "eligible" and projected is not None and additions:
            projection_record = {
                "schema_version": "source-contract-projection-v1",
                "source_id": source_id,
                "source_json": raw_rel(source_path),
                "source_json_sha256": sha(source_bytes),
                "source_text_sha256": source.get("content_sha256"),
                "original_license": copy.deepcopy(source.get("license")),
                "mechanical_additions": additions,
                "body_unchanged": projected.get("text") == source.get("text") and projected.get("content_sha256") == source.get("content_sha256"),
                "document": projected,
            }
            projections.append(projection_record)
            row["contract_projection_file"] = raw_rel(SCIENCE_CONTRACT_PROJECTIONS)
        if decision == "eligible":
            topic_counts[topic or "unclassified"] = topic_counts.get(topic or "unclassified", 0) + 1
        else:
            for reason in reasons:
                held_counts[reason] = held_counts.get(reason, 0) + 1
        review_rows.append(row)
    projection_content = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in projections)
    atomic_write(SCIENCE_CONTRACT_PROJECTIONS, projection_content.encode("utf-8"))
    manifest_bytes = SCIENCE_MANIFEST.read_bytes() if SCIENCE_MANIFEST.exists() else b""
    review = {
        "schema_version": "science-topic-review-v1",
        "generated_at": utc_now(),
        "source_manifest": raw_rel(SCIENCE_MANIFEST),
        "source_manifest_sha256": sha(manifest_bytes),
        "projection_file": raw_rel(SCIENCE_CONTRACT_PROJECTIONS),
        "projection_record_count": len(projections),
        "criteria": {
            "allowed_topics": [*SCIENCE_TITLE_TOPIC_PATTERNS, "biological_science"],
            "native_language_title_signals": sorted(SCIENCE_NATIVE_TITLE_TOPIC_PATTERNS),
            "basic_biological_science": "explicit fundamental biology, genetics, microbiology, molecular/cellular mechanisms, or neuroscience; at most 20 authorable works, with no required quota",
            "topic_match_requires_explicit_title_signal": True,
            "incidental_abstract_keyword_is_not_sufficient": True,
            "external_benchmark_or_evaluation_markers": "hold before source eligibility; no benchmark items or results are reproduced in this sidecar",
            "contract_projection": "revision and share-alike additions only; original text, content hash, license terms, rights evidence, and source JSON bytes remain unchanged",
        },
        "counts": {"source_documents_reviewed": len(review_rows), "eligible": sum(row["decision"] == "eligible" for row in review_rows), "held": sum(row["decision"] == "hold" for row in review_rows), "eligible_by_topic": topic_counts, "held_by_reason": held_counts},
        "sources": review_rows,
    }
    write_json(SCIENCE_TOPIC_REVIEW, review)
    return review


def science_identifier_audit(http: SerialHttp, science: dict[str, Any], persist) -> dict[str, Any]:
    """Verify each frozen PMC source against the matching OAI DC item and JATS front matter."""
    manifest_docs = existing_records(SCIENCE_MANIFEST)
    topic_review = write_science_topic_review(manifest_docs)
    topic_eligible_ids = {row["id"] for row in topic_review["sources"] if row.get("decision") == "eligible"}
    superseded_audit_sha = sha(SCIENCE_IDENTIFIER_AUDIT.read_bytes()) if SCIENCE_IDENTIFIER_AUDIT.exists() else None
    all_rows: dict[str, dict[str, Any]] = {}
    audit_dir = RAW_SCIENCE / "identifier-audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    for manifest_row in manifest_docs:
        source_id = str(manifest_row.get("id") or "")
        source_path = DOCS / f"{source_id}.json"
        if not source_path.exists():
            all_rows[source_id] = {"decision": "hold", "reason_codes": ["source_json_missing"]}
            continue
        source_bytes = source_path.read_bytes()
        source = json.loads(source_bytes)
        access = source.get("source_access") or {}
        raw_file = access.get("raw_file")
        full_path = ROOT / str(raw_file) if raw_file else None
        issues: list[str] = []
        full_bytes = full_path.read_bytes() if full_path and full_path.exists() else b""
        full_sha256 = sha(full_bytes) if full_bytes else None
        if not full_bytes:
            issues.append("cached_jats_missing")
        elif source.get("raw_jats_sha256") != full_sha256 or access.get("response_sha256") != full_sha256:
            issues.append("cached_jats_sha_mismatch")

        requested_identifier = f"oai:pubmedcentral.nih.gov:{str(source.get('pmcid') or '').upper().removeprefix('PMC')}"
        original_url = str(access.get("api_url") or "")
        original_identifier = parse_qs(urlsplit(original_url).query).get("identifier", [""])[0]
        preflight_identifier = str(access.get("license_preflight_oai_identifier") or "")
        if preflight_identifier and preflight_identifier != requested_identifier:
            issues.append("license_preflight_oai_identifier_mismatch")
        request_log_row = next((
            row for row in science.get("requests", [])
            if row.get("raw_file") == raw_rel(full_path) and row.get("sha256") == full_sha256
        ), None) if full_path else None
        request_log_identifier = parse_qs(urlsplit(str((request_log_row or {}).get("url") or (request_log_row or {}).get("final_url") or "")).query).get("identifier", [""])[0]
        if original_identifier and original_identifier != requested_identifier:
            issues.append("original_getrecord_identifier_mismatch")
        elif not original_identifier and request_log_identifier != requested_identifier:
            issues.append("original_getrecord_identifier_mismatch")
        if request_log_row and (request_log_row.get("http_status") != 200 or request_log_identifier != requested_identifier):
            issues.append("fulltext_request_log_identity_mismatch")

        dc_path = audit_dir / f"{source_id}-oai_dc.xml"
        try:
            dc_bytes, dc_access = http.get(
                PMC_OAI, dc_path, "science",
                params={"verb": "GetRecord", "identifier": requested_identifier, "metadataPrefix": "oai_dc"},
                max_bytes=1_500_000,
            )
        except RuntimeError as exc:
            all_rows[source_id] = {
                "decision": "hold",
                "reason_codes": issues + ["oai_dc_getrecord_failed"],
                "access_error": str(exc),
                "raw_source_json_sha256": sha(source_bytes),
                "fulltext_raw_file": raw_rel(full_path) if full_path else None,
                "fulltext_raw_sha256": full_sha256,
                "requested_oai_identifier": requested_identifier,
            }
            continue

        try:
            dc_root = ET.fromstring(dc_bytes)
        except ET.ParseError:
            dc_root = None
            issues.append("oai_dc_xml_invalid")
        dc_record = next((node for node in dc_root.iter() if xml_local(node.tag) == "record"), None) if dc_root is not None else None
        dc_header = next((node for node in list(dc_record) if xml_local(node.tag) == "header"), None) if dc_record is not None else None
        dc_header_identifier = next((text_content(node) for node in (list(dc_header) if dc_header is not None else []) if xml_local(node.tag) == "identifier"), "")
        fields = oai_dc_fields(dc_record) if dc_record is not None else {}
        dc_title = (fields.get("title") or [""])[0]
        dc_identifiers = fields.get("identifier", [])
        dc_pmcids = sorted({match.group(1).upper() for value in dc_identifiers if (match := re.search(r"/articles/(PMC\d+)(?:/|$)", value, re.I))})
        dc_dois = sorted({normalize_doi(match.group(0).rstrip(".,;:)/ ")) for value in dc_identifiers if (match := re.search(r"10\.\d{4,9}/[^\s<>\"]+", value, re.I))})
        dc_rights = fields.get("rights", [])
        combined_dc_rights = " ".join(dc_rights)
        dc_license = canonical_combined_cc_license([(value, value) for value in dc_rights]) if dc_rights else None
        expected_pmcid = str(source.get("pmcid") or "").upper()
        expected_doi = normalize_doi(str(source.get("doi") or "")).rstrip("/")
        list_page_file = access.get("license_preflight_metadata_page_file")
        list_page_evidence: dict[str, Any] | None = None
        if list_page_file:
            list_page_path = ROOT / str(list_page_file)
            if not list_page_path.exists():
                issues.append("candidate_oai_dc_list_page_missing")
            else:
                list_page_bytes = list_page_path.read_bytes()
                list_page_sha = sha(list_page_bytes)
                if list_page_sha != access.get("license_preflight_metadata_page_sha256") or list_page_sha != access.get("license_preflight_frontmatter_sha256"):
                    issues.append("candidate_oai_dc_list_page_sha_mismatch")
                try:
                    list_root = ET.fromstring(list_page_bytes)
                    list_record = next((node for node in list_root.iter() if xml_local(node.tag) == "record" and any(text_content(child) == requested_identifier for child in node.iter() if xml_local(child.tag) == "identifier")), None)
                    list_header = next((node for node in list(list_record) if xml_local(node.tag) == "header"), None) if list_record is not None else None
                    list_header_identifier = next((text_content(node) for node in (list(list_header) if list_header is not None else []) if xml_local(node.tag) == "identifier"), "")
                    list_fields = oai_dc_fields(list_record) if list_record is not None else {}
                    list_title = (list_fields.get("title") or [""])[0]
                    list_identifiers = list_fields.get("identifier", [])
                    list_pmcids = sorted({match.group(1).upper() for value in list_identifiers if (match := re.search(r"/articles/(PMC\d+)(?:/|$)", value, re.I))})
                    list_dois = sorted({normalize_doi(match.group(0).rstrip(".,;:)/ ")) for value in list_identifiers if (match := re.search(r"10\.\d{4,9}/[^\s<>\"]+", value, re.I))})
                    list_rights = list_fields.get("rights", [])
                    combined_list_rights = " ".join(list_rights)
                    list_license = canonical_combined_cc_license([(value, value) for value in list_rights]) if list_rights else None
                    if list_record is None or list_header_identifier != requested_identifier:
                        issues.append("candidate_oai_dc_list_record_identifier_mismatch")
                    if normalized_title_identity(list_title) != normalized_title_identity(str(source.get("title") or "")):
                        issues.append("candidate_oai_dc_list_record_title_mismatch")
                    if expected_pmcid not in list_pmcids:
                        issues.append("candidate_oai_dc_list_record_pmcid_mismatch")
                    if expected_doi and expected_doi not in {value.rstrip("/") for value in list_dois}:
                        issues.append("candidate_oai_dc_list_record_doi_mismatch")
                    if list_license != dc_license or list_license != ((source.get("license") or {}).get("id"), (source.get("license") or {}).get("url")):
                        issues.append("candidate_oai_dc_list_record_license_mismatch")
                    list_page_evidence = {
                        "raw_file": str(list_page_file), "raw_sha256": list_page_sha,
                        "header_identifier": list_header_identifier, "canonical_pmcids": list_pmcids,
                        "canonical_dois": list_dois, "title": list_title,
                        "license": list_license,
                    }
                except ET.ParseError:
                    issues.append("candidate_oai_dc_list_page_xml_invalid")
        if dc_header_identifier != requested_identifier:
            issues.append("oai_dc_header_does_not_match_requested_identifier")
        source_title_key = normalized_title_identity(str(source.get("title") or ""))
        dc_title_key = normalized_title_identity(dc_title)
        if not source_title_key or source_title_key != dc_title_key:
            issues.append("oai_dc_title_mismatch")
        if expected_pmcid not in dc_pmcids:
            issues.append("oai_dc_canonical_pmcid_mismatch")
        if expected_doi and expected_doi not in {value.rstrip("/") for value in dc_dois}:
            issues.append("oai_dc_doi_mismatch")

        jats_header_identifier = ""
        jats_ids: list[dict[str, str]] = []
        jats_title = ""
        jats_license_evidence: list[dict[str, Any]] = []
        jats_license_terms: list[tuple[str, str]] = []
        if full_bytes:
            try:
                jats_root = ET.fromstring(full_bytes)
                jats_record = next((node for node in jats_root.iter() if xml_local(node.tag) == "record"), None)
                jats_header = next((node for node in list(jats_record) if xml_local(node.tag) == "header"), None) if jats_record is not None else None
                jats_header_identifier = next((text_content(node) for node in (list(jats_header) if jats_header is not None else []) if xml_local(node.tag) == "identifier"), "")
                article = next((node for node in jats_root.iter() if xml_local(node.tag) == "article"), None)
                front = next((node for node in list(article) if xml_local(node.tag) == "front"), None) if article is not None else None
                article_meta = next((node for node in (list(front) if front is not None else []) if xml_local(node.tag) == "article-meta"), None)
                if article_meta is not None:
                    for node in article_meta.iter():
                        if xml_local(node.tag) == "article-title" and not jats_title:
                            jats_title = text_content(node)
                        if xml_local(node.tag) == "article-id":
                            jats_ids.append({"type": node.attrib.get("pub-id-type", "").lower(), "value": text_content(node)})
                    jats_license_terms, jats_license_evidence = jats_governing_license_evidence(article_meta)
            except ET.ParseError:
                issues.append("cached_jats_xml_invalid")
        front_pmcids = sorted({item["value"].upper() for item in jats_ids if item["type"] == "pmcid"})
        front_dois = sorted({normalize_doi(item["value"]).rstrip("/") for item in jats_ids if item["type"] == "doi"})
        if jats_header_identifier != requested_identifier:
            issues.append("jats_header_does_not_match_requested_identifier")
        if expected_pmcid not in front_pmcids:
            issues.append("front_article_meta_pmcid_mismatch")
        if expected_doi and expected_doi not in front_dois:
            issues.append("front_article_meta_doi_mismatch")
        jats_title_key = normalized_title_identity(jats_title)
        if not source_title_key or source_title_key != jats_title_key:
            issues.append("front_article_meta_title_mismatch")

        document_license = source.get("license") or {}
        expected_license = (document_license.get("id"), document_license.get("url"))
        jats_license = canonical_combined_cc_license(jats_license_terms)
        same_rights = bool(dc_license and expected_license == dc_license and dc_license == jats_license)
        if not same_rights:
            issues.append("same_item_license_not_verified_across_dc_jats_source")
        canonical_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{expected_pmcid}/"
        if norm_url(str(source.get("url") or "")) != norm_url(canonical_url):
            issues.append("source_canonical_url_mismatch")

        all_rows[source_id] = {
            "decision": "verified" if not issues else "hold",
            "reason_codes": sorted(set(issues)),
            "raw_source_json_sha256": sha(source_bytes),
            "canonical_pmcid": expected_pmcid,
            "canonical_doi": expected_doi or None,
            "canonical_url": canonical_url,
            "requested_oai_identifier": requested_identifier,
            "oai_identifier": dc_header_identifier,
            "original_fulltext_request_log_evidence": request_log_row,
            "license_preflight_oai_identifier": preflight_identifier or None,
            "dc_metadata_raw_file": raw_rel(dc_path),
            "dc_metadata_raw_sha256": sha(dc_bytes),
            "dc_list_record_evidence": list_page_evidence,
            "dc_getrecord_request": dc_access,
            "dc_metadata_pmcids": dc_pmcids,
            "dc_metadata_dois": dc_dois,
            "dc_rights_statements": dc_rights,
            "dc_governing_license": dc_license,
            "dc_metadata_title": dc_title,
            "source_title": source.get("title"),
            "dc_metadata_title_match": bool(source_title_key and source_title_key == dc_title_key),
            "dc_metadata_title_exact_match": " ".join(dc_title.split()) == " ".join(str(source.get("title") or "").split()),
            "dc_metadata_title_match_method": "Unicode NFKD, casefold, combining-mark and non-alphanumeric removal; front JATS title remains separately checked",
            "same_item_rights_verified": same_rights,
            "license": {"id": document_license.get("id"), "url": document_license.get("url")},
            "source_json_license_statement": document_license.get("statement"),
            "jats_license_evidence": jats_license_evidence,
            "jats_governing_license": jats_license,
            "fulltext_raw_file": raw_rel(full_path) if full_path else None,
            "fulltext_raw_sha256": full_sha256,
            "front_article_id_evidence": {
                "article_ids": jats_ids,
                "canonical_pmcid": front_pmcids,
                "canonical_doi": front_dois,
                "title": jats_title,
                "title_match": bool(source_title_key and source_title_key == jats_title_key),
                "header_identifier": jats_header_identifier,
                "source_json_sha_matches_raw": bool(full_sha256 and source.get("raw_jats_sha256") == full_sha256 and access.get("response_sha256") == full_sha256),
            },
        }
        persist()

    topic_rows_by_id = {
        str(row.get("id")): row
        for row in topic_review.get("sources", [])
        if isinstance(row, dict) and row.get("id")
    }
    for source_id, identity_row in all_rows.items():
        topic_row = topic_rows_by_id.get(source_id)
        if topic_row is not None:
            identity_row["topic_review_record_sha256"] = canonical_science_review_row_sha256(topic_row)
        identity_row["identity_scope_schema_version"] = "science-identity-scope-v1"
        identity_row["identity_scope_sha256"] = science_identity_scope_sha256(identity_row)

    eligible_sources = {source_id: all_rows[source_id] for source_id in sorted(topic_eligible_ids) if source_id in all_rows}
    held_sources = {source_id: all_rows[source_id] for source_id in sorted(set(all_rows) - topic_eligible_ids)}
    verified_count = sum(row.get("decision") == "verified" for row in all_rows.values())
    archived_audit_sha, archived_audit_file = archive_science_identity_audit(science)
    audit = {
        "schema_version": "science-source-identity-audit-v1",
        "identity_scope_schema_version": "science-identity-scope-v1",
        "generated_at": utc_now(),
        "source_manifest": raw_rel(SCIENCE_MANIFEST),
        "source_manifest_sha256": sha(SCIENCE_MANIFEST.read_bytes() if SCIENCE_MANIFEST.exists() else b""),
        "topic_review": raw_rel(SCIENCE_TOPIC_REVIEW),
        "topic_review_sha256": sha(SCIENCE_TOPIC_REVIEW.read_bytes()),
        "supersedes_previous_audit_sha256": archived_audit_sha or superseded_audit_sha,
        "supersedes_previous_audit_revision_file": archived_audit_file,
        "audit_method": "official PMC OAI-PMH GetRecord metadataPrefix=oai_dc checked serially against the cached PMC OAI-PMH metadataPrefix=pmc response and source JSON; identity evidence comes from front/article-meta only",
        "no_new_fulltext_retrieval_for_audit": True,
        "owner_reported_external_check": {
            "canonical_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC467062/",
            "reported_by": "root acquisition owner",
            "reported_on": utc_now(),
            "verified_fields": ["title", "DOI", "PMCID", "PMID", "CC BY 4.0"],
            "browser_network_bytes": "unmeasured and excluded from the local acquisition network counter",
            "timestamp_note": "This records receipt of the owner report; the browser response timestamp was not supplied separately.",
        },
        "counts": {"source_documents_audited": len(all_rows), "identity_verified": verified_count, "identity_held": len(all_rows) - verified_count,
                   "topic_eligible_sources_audited": len(eligible_sources), "topic_eligible_sources_verified": sum(row.get("decision") == "verified" for row in eligible_sources.values())},
        "sources": eligible_sources,
        "non_authorable_source_identity_checks": held_sources,
    }
    write_json(SCIENCE_IDENTIFIER_AUDIT, audit)
    science["identity_audit_file"] = raw_rel(SCIENCE_IDENTIFIER_AUDIT)
    science["identity_audit_sha256"] = sha(SCIENCE_IDENTIFIER_AUDIT.read_bytes())
    science["identity_audit_counts"] = audit["counts"]
    latest_topic_review = read_json(SCIENCE_TOPIC_REVIEW, topic_review)
    _apply_science_review_counts(science, manifest_docs, latest_topic_review, audit)
    manifest_by_id = {str(row.get("id")): row for row in manifest_docs}
    science["accepted_sources"] = []
    for topic_row in latest_topic_review.get("sources", []):
        source_id = str(topic_row.get("id") or "")
        identity_row = (audit.get("sources") or {}).get(source_id)
        if (
            topic_row.get("decision") != "eligible"
            or not identity_row
            or identity_row.get("decision") != "verified"
            or identity_row.get("raw_source_json_sha256") != topic_row.get("source_json_sha256")
        ):
            continue
        source_doc = read_json(DOCS / f"{source_id}.json", {})
        manifest_row = manifest_by_id.get(source_id, {})
        science["accepted_sources"].append({
            "id": source_id,
            "pmcid": source_doc.get("pmcid"),
            "doi": source_doc.get("doi"),
            "language": source_doc.get("language"),
            "title": source_doc.get("title"),
            "journal": source_doc.get("journal"),
            "group": source_doc.get("search_group"),
            "topic": topic_row.get("topic"),
            "document_file": manifest_row.get("document_file") or raw_rel(DOCS / f"{source_id}.json"),
        })
    eligible_identity_verified = audit["counts"]["topic_eligible_sources_verified"] == len(topic_eligible_ids)
    science["phase"] = "authorable_source_identity_verified_with_quarantined_noneligible_items" if eligible_identity_verified else "source_identity_quarantined"
    persist()
    return audit


def oai_error(root: ET.Element) -> tuple[str, str] | None:
    error = next((node for node in root.iter() if xml_local(node.tag) == "error"), None)
    if error is None:
        return None
    return error.attrib.get("code", "unknown"), text_content(error)


def list_sets_probe(http: SerialHttp, science: dict[str, Any], persist) -> list[str]:
    params = {"verb": "ListSets"}
    path = RAW_SCIENCE / "preflight/oai-listsets.xml"
    body, row = http.get(PMC_OAI, path, "science", params=params, max_bytes=2_000_000)
    root = ET.fromstring(body)
    names = [text_content(node) for node in root.iter() if xml_local(node.tag) == "setSpec" and text_content(node)]
    science["oai_sets_first_page"] = {"set_specs": names, "request": row}
    science["phase"] = "preflight_verified"
    persist()
    return names


def candidate_searches(http: SerialHttp, science: dict[str, Any], persist, *, initial_only: bool,
                       screen_offset: int = 0, screen_limit: int | None = None,
                       blocked_pmids: set[str] | None = None, blocked_dois: set[str] | None = None,
                       date_window: str = "all") -> dict[str, list[dict[str, Any]]]:
    pools: dict[str, list[dict[str, Any]]] = {}
    inventory = science.setdefault("language_inventory", {
        "source": "PMC OAI-PMH `oai_dc` metadata from the bounded article sample",
        "scope": "observed languages of source records scanned; PMC publishes no static all-language inventory through the checked OAI documentation",
        "exhaustive_for_entire_pmc_open_subset": False,
        "observed_record_language_counts": {},
        "eligible_ccby_cc0_language_counts": {},
        "non_english_fulltext_papers_found": 0,
        "note": "OAI Dublin Core exposes article language, not all abstract languages; accepted JATS full texts retain per-abstract language attributes.",
    })
    if inventory.get("source") != "PMC OAI-PMH `oai_dc` metadata from the bounded article sample":
        inventory["previous_frontmatter_inventory"] = {
            "source": inventory.get("source"),
            "observed_record_language_counts": inventory.get("observed_record_language_counts", {}),
            "eligible_ccby_cc0_language_counts": inventory.get("eligible_ccby_cc0_language_counts", {}),
            "note": "Initial `pmc_fm` probe found no machine-readable license nodes and was replaced by the documented `oai_dc` license screen.",
        }
        inventory["source"] = "PMC OAI-PMH `oai_dc` metadata from the bounded article sample"
        inventory["observed_record_language_counts"] = {}
        inventory["eligible_ccby_cc0_language_counts"] = {}
        inventory["non_english_fulltext_papers_found"] = 0
        science["previous_candidate_scan_results_pmc_fm"] = science.pop("candidate_scan_results", {})
        science["candidate_scan_results"] = {}
    inventory.pop("non_english_abstracts_on_other_papers", None)
    inventory.setdefault("accepted_non_english_abstracts_by_language", {})
    inventory["count_semantics"] = {
        "observed_record_language_counts": "distinct OAI item language codes seen in this bounded scan",
        "explicit_ccby_cc0_record_language_counts": "item-license-screened OAI records, before breadth-topic filtering and before full-text retrieval",
        "topic_candidate_language_counts": "records passing explicit item-license and breadth-topic metadata filters",
        "non_english_fulltext_papers_found": "source-language documents actually retrieved and stored in the manifest; does not count metadata-only candidates",
    }
    if not science.get("oai_sets_first_page"):
        list_sets_probe(http, science, persist)

    previous_inventory = {key: inventory.get(key) for key in (
        "available_languages_observed", "eligible_languages_observed", "observed_record_language_counts",
        "eligible_ccby_cc0_language_counts", "candidate_records_scanned", "completed_at",
    ) if inventory.get(key) is not None}
    if previous_inventory:
        science.setdefault("previous_language_inventory_runs", []).append(previous_inventory)
    for key in ("observed_record_language_counts", "eligible_ccby_cc0_language_counts"):
        inventory[key] = {}
    inventory["explicit_ccby_cc0_record_language_counts"] = {}
    inventory["topic_candidate_language_counts"] = {}
    inventory["non_english_item_language_candidates"] = {}
    science["previous_candidate_scan_results"] = science.pop("candidate_scan_results", {})
    science["candidate_scan_results"] = {}
    scan_seen_ids: set[str] = set()
    blocked_pmids = {value.upper() for value in (blocked_pmids or set())}
    seen_dois = set(blocked_dois or set())
    candidate_position = 0
    selected_candidate_count = 0
    stop_after_batch = False
    science["candidate_screening_batch"] = {"offset": screen_offset, "limit": screen_limit, "date_window": date_window, "started_at": utc_now()}
    science["candidate_scan_started_at"] = utc_now()

    # OAI exposes collection/date sets rather than keyword or language search.
    # Use bounded release-date slices, screening only compact Dublin Core
    # records for item license and topic before any full text is requested.
    scans: list[tuple[str, str | None, str | None, int]] = []
    if initial_only:
        scans.append(("pmc-open-initial", None, None, 6))
    else:
        scans.extend([
            ("pmc-open-2025-2026", "2025-01-01", "2026-10-03", 100),
            ("pmc-open-2022-2024", "2022-01-01", "2024-12-31", 100),
            ("pmc-open-2019-2021", "2019-01-01", "2021-12-31", 100),
            ("pmc-open-2016-2018", "2016-01-01", "2018-12-31", 100),
            ("pmc-open-2013-2015", "2013-01-01", "2015-12-31", 100),
            ("pmc-open-2010-2012", "2010-01-01", "2012-12-31", 100),
            ("pmc-open-2007-2009", "2007-01-01", "2009-12-31", 100),
            ("pmc-open-2004-2006", "2004-01-01", "2006-12-31", 100),
            ("pmc-open-2001-2003", "2001-01-01", "2003-12-31", 100),
        ])
    if not initial_only and date_window != "all":
        selected_group = f"pmc-open-{date_window}"
        scans = [scan for scan in scans if scan[0] == selected_group]
        if not scans:
            raise ValueError(f"unknown PMC date window: {date_window}")
    science["candidate_scan_plan"] = [{"set": group if group.startswith("plos") else "pmc-open", "date_from": from_date, "date_until": until, "max_pages": max_pages} for group, from_date, until, max_pages in scans]
    persist()
    for group, from_date, until, max_pages in scans:
        token: str | None = None
        scanned = 0
        for page_index in range(max_pages):
            if token:
                params = {"verb": "ListRecords", "resumptionToken": token}
            else:
                params = {"verb": "ListRecords", "metadataPrefix": "oai_dc", "set": "pmc-open"}
                if from_date:
                    params["from"] = from_date
                if until:
                    params["until"] = until
            name = re.sub(r"[^a-zA-Z0-9_-]+", "_", group)
            body, access = http.get(
                PMC_OAI,
                RAW_SCIENCE / "candidate-frontmatter" / f"{name}-{page_index:03d}.xml",
                "science",
                params=params,
                max_bytes=5_000_000,
                store_raw=True,
            )
            try:
                root = ET.fromstring(body)
            except ET.ParseError as exc:
                raise RuntimeError("PMC OAI-PMH returned invalid XML; source scan stopped") from exc
            error = oai_error(root)
            if error:
                code, detail = error
                science.setdefault("candidate_scan_results", {})[group] = {"pages": page_index + 1, "result": code, "detail": detail, "request": access}
                if code in {"noRecordsMatch", "badArgument"}:
                    break
                raise RuntimeError(f"PMC OAI-PMH returned protocol error {code}; scan stopped")
            records = [node for node in root.iter() if xml_local(node.tag) == "record"]
            scanned += len(records)
            page_eligible = 0
            page_filtered: dict[str, int] = {}
            for record in records:
                fields = oai_dc_fields(record)
                header = next((node for node in list(record) if xml_local(node.tag) == "header"), None)
                identifier = next((text_content(node) for node in (list(header) if header is not None else []) if xml_local(node.tag) == "identifier"), "")
                id_match = re.search(r"(\d+)$", identifier)
                if id_match:
                    record_id = "PMC" + id_match.group(1)
                    if record_id in scan_seen_ids:
                        continue
                    scan_seen_ids.add(record_id)
                lang = normalize_science_language((fields.get("language") or ["und"])[0])
                inventory["observed_record_language_counts"][lang] = int(inventory["observed_record_language_counts"].get(lang, 0)) + 1
                licensed = any(canonical_cc_license(value, value) for value in fields.get("rights", []))
                if licensed:
                    explicit_counts = inventory["explicit_ccby_cc0_record_language_counts"]
                    explicit_counts[lang] = int(explicit_counts.get(lang, 0)) + 1
                summary, pmcid, reason = candidate_from_oai_dc(record, group, access["sha256"])
                if summary is None:
                    filter_key = reason or "candidate_metadata_rejected"
                    page_filtered[filter_key] = int(page_filtered.get(filter_key, 0)) + 1
                    continue
                summary["oai_identifier"] = identifier
                summary["metadata_page_file"] = access.get("raw_file")
                summary["metadata_page_sha256"] = access["sha256"]
                candidate_doi = normalize_doi(str(summary.get("doi") or ""))
                if (pmcid or "").upper() in blocked_pmids or (candidate_doi and candidate_doi in seen_dois):
                    page_filtered["existing_or_duplicate_work"] = int(page_filtered.get("existing_or_duplicate_work", 0)) + 1
                    continue
                if candidate_doi:
                    seen_dois.add(candidate_doi)
                candidate_position += 1
                if candidate_position <= screen_offset:
                    page_filtered["before_candidate_batch_offset"] = int(page_filtered.get("before_candidate_batch_offset", 0)) + 1
                    continue
                page_eligible += 1
                selected_candidate_count += 1
                topic_counts = inventory["topic_candidate_language_counts"]
                topic_counts[summary["language"]] = int(topic_counts.get(summary["language"], 0)) + 1
                eligible_by_lang = inventory["eligible_ccby_cc0_language_counts"]
                eligible_by_lang[summary["language"]] = int(eligible_by_lang.get(summary["language"], 0)) + 1
                if summary["language"] not in {"en", "und"}:
                    native_counts = inventory["non_english_item_language_candidates"]
                    native_counts[summary["language"]] = int(native_counts.get(summary["language"], 0)) + 1
                pools.setdefault(summary["search_group"], []).append(summary)
                if screen_limit is not None and selected_candidate_count >= screen_limit:
                    stop_after_batch = True
                    break
            token_node = next((node for node in root.iter() if xml_local(node.tag) == "resumptionToken"), None)
            token = text_content(token_node) if token_node is not None else None
            science.setdefault("candidate_scan_results", {})[group] = {
                "pages": page_index + 1,
                "records_scanned": scanned,
                "eligible_records_seen": sum(len(values) for values in pools.values()),
                "eligible_in_last_page": page_eligible,
                "metadata_filter_counts": page_filtered,
                "last_page_sha256": access["sha256"],
                "last_request": access,
                "has_resumption_token": bool(token),
            }
            science["language_inventory"]["available_languages_observed"] = sorted(inventory["observed_record_language_counts"])
            science["language_inventory"]["eligible_languages_observed"] = sorted(inventory["eligible_ccby_cc0_language_counts"])
            persist()
            if stop_after_batch:
                break
            if not records or not token:
                break
        if stop_after_batch:
            break
    inventory["completed_at"] = utc_now()
    inventory["candidate_records_scanned"] = len(scan_seen_ids)
    inventory["available_language_count_observed"] = len(inventory["observed_record_language_counts"])
    inventory["eligible_language_count_observed"] = len(inventory["eligible_ccby_cc0_language_counts"])
    inventory["languages_not_observed_are_not_claimed_unavailable"] = True
    science["candidate_screening_batch"].update({
        "finished_at": utc_now(),
        "candidates_selected": selected_candidate_count,
        "next_offset": screen_offset + selected_candidate_count,
        "stopped_at_candidate_limit": stop_after_batch,
    })
    science["candidate_screening_next_offset"] = screen_offset + selected_candidate_count
    persist()
    return pools


def inspect_science_existing(manifest_docs: list[dict[str, Any]], prior: dict[str, set[str]], record: dict[str, Any]) -> str | None:
    pmcid = str(record.get("pmcid", "")).upper()
    doi = record.get("doi")
    url = norm_url(record.get("url", ""))
    relation_keys = {item.casefold() for item in record.get("relation_keys", [])}
    if pmcid in prior["pmcid"]:
        return "prior_pmcid_overlap"
    if doi and normalize_doi(doi) in prior["doi"]:
        return "prior_doi_overlap"
    if url and url in prior["url"]:
        return "prior_canonical_url_overlap"
    if relation_keys & prior["relation"]:
        return "prior_relation_overlap"
    for prior_doc in manifest_docs:
        if str(prior_doc.get("pmcid", "")).upper() == pmcid:
            return "same_run_pmcid_duplicate"
        if doi and normalize_doi(str(prior_doc.get("doi", ""))) == normalize_doi(doi):
            return "same_run_doi_duplicate"
    return None


def add_science_source(http: SerialHttp, science: dict[str, Any], persist, manifest_docs: list[dict[str, Any]], prior: dict[str, set[str]], candidate: dict[str, Any], group: str) -> bool:
    pmcid_id = str(candidate["pmcid"])
    pmcid = pmcid_id if pmcid_id.upper().startswith("PMC") else "PMC" + pmcid_id
    xml_url = PMC_OAI
    params = {"verb": "GetRecord", "identifier": f"oai:pubmedcentral.nih.gov:{pmcid.removeprefix('PMC')}", "metadataPrefix": "pmc"}
    request_hash = sha((pmcid + "\0" + urlencode(params)).encode())[:16]
    raw_path = RAW_SCIENCE / "articles" / f"{pmcid.lower()}-jats.xml"
    try:
        xml_bytes, access = http.get(xml_url, raw_path, "science", params=params, max_bytes=8_000_000)
    except RuntimeError as exc:
        message = str(exc)
        if "returned 404" in message or "HTTP 404" in message:
            science["rejected_counts"]["item_not_found"] = int(science["rejected_counts"].get("item_not_found", 0)) + 1
            return False
        raise
    record, reason = parse_science_document(xml_bytes, pmcid, group)
    if record is None:
        key = "source_exceeds_100k_chars_hold" if reason == "source_exceeds_100k_chars_hold" else reason
        bucket = science["held"] if reason == "source_exceeds_100k_chars_hold" else science["rejected_counts"]
        bucket[key] = int(bucket.get(key, 0)) + 1
        science.setdefault("candidate_decisions", []).append({"pmcid": pmcid, "decision": key, "group": group, "raw_sha256": access["sha256"]})
        persist()
        return False
    if record["license"]["id"] != candidate.get("license_id") or record["license"]["url"] != candidate.get("license_url"):
        science["held"]["frontmatter_fulltext_license_mismatch"] = int(science["held"].get("frontmatter_fulltext_license_mismatch", 0)) + 1
        science.setdefault("candidate_decisions", []).append({"pmcid": pmcid, "decision": "frontmatter_fulltext_license_mismatch", "frontmatter_sha256": candidate.get("frontmatter_sha256"), "fulltext_sha256": access["sha256"]})
        persist()
        return False
    if not record.get("doi") and not record.get("pmcid"):
        science["rejected_counts"]["missing_dedup_identifier"] = int(science["rejected_counts"].get("missing_dedup_identifier", 0)) + 1
        return False
    overlap = inspect_science_existing(manifest_docs, prior, record)
    if overlap:
        science["rejected_counts"][overlap] = int(science["rejected_counts"].get(overlap, 0)) + 1
        science.setdefault("candidate_decisions", []).append({"pmcid": pmcid, "decision": overlap, "group": group})
        persist()
        return False
    record["source_access"] = {
        "api_url": access["url"],
        "retrieval_timestamp": access["retrieved_at"],
        "response_sha256": access["sha256"],
        "response_bytes_decoded": access["response_bytes_decoded"],
        "raw_file": access["raw_file"],
        "request_id": request_hash,
        "http_status": access["http_status"],
        "license_preflight_frontmatter_sha256": candidate.get("frontmatter_sha256"),
        "license_preflight_metadata_prefix": "oai_dc",
        "license_preflight_rights_statement": candidate.get("license_statement"),
        "license_preflight_candidate_set": candidate.get("candidate_set"),
        "license_preflight_oai_identifier": candidate.get("oai_identifier"),
        "license_preflight_metadata_page_file": candidate.get("metadata_page_file"),
        "license_preflight_metadata_page_sha256": candidate.get("metadata_page_sha256"),
    }
    record["candidate_metadata"] = {key: candidate.get(key) for key in ("title", "journal", "language", "languages_present", "published_at", "license_id", "license_url", "license_statement", "topic_bucket", "frontmatter_sha256", "oai_identifier", "metadata_page_file", "metadata_page_sha256")}
    record["search_group"] = candidate.get("search_group") or group
    article_abstract = " ".join(
        str(section.get("text") or "")
        for section in record.get("sections", [])
        if section.get("section_type") == "abstract"
    )
    verified_topic, scope_reason = classify_science_topic(
        str(record.get("title") or ""), article_abstract, "", str(record.get("language") or "und")
    )
    if scope_reason:
        science["held"][scope_reason] = int(science["held"].get(scope_reason, 0)) + 1
        science.setdefault("candidate_decisions", []).append({"pmcid": pmcid, "decision": scope_reason, "group": group, "fulltext_sha256": access["sha256"]})
        persist()
        return False
    record["topic_bucket"] = verified_topic
    try:
        from dataset_factory.document_data import _validate_source
        _validate_source(record)
    except (ImportError, ValueError, TypeError) as exc:
        reason = f"source_contract_invalid:{type(exc).__name__}:{str(exc)}"
        science["held"][reason] = int(science["held"].get(reason, 0)) + 1
        science.setdefault("candidate_decisions", []).append({"pmcid": pmcid, "decision": reason, "group": group, "fulltext_sha256": access["sha256"]})
        persist()
        return False
    if record["language"] in {"en", "und"}:
        for abstract_language in sorted({
            str(section.get("language"))
            for section in record.get("sections", [])
            if section.get("section_type") == "abstract" and section.get("language") not in {None, record["language"], "en", "und"}
        }):
            inventory = science.setdefault("language_inventory", {}).setdefault("accepted_non_english_abstracts_by_language", {})
            inventory[abstract_language] = int(inventory.get(abstract_language, 0)) + 1
    path = DOCS / f"{record['id']}.json"
    write_json(path, record)
    record["document_file"] = raw_rel(path)
    manifest_docs.append(record)
    save_manifest(SCIENCE_MANIFEST, manifest_docs)
    review = refresh_science_review(science, manifest_docs, persist)
    new_review_row = next((row for row in review["sources"] if row.get("id") == record["id"]), None)
    if new_review_row is None or new_review_row.get("decision") != "eligible":
        reasons = (new_review_row or {}).get("reason_codes", ["topic_review_missing_row"])
        held = science.setdefault("held", {})
        held["topic_review_hold"] = int(held.get("topic_review_hold", 0)) + 1
        science.setdefault("candidate_decisions", []).append({"pmcid": pmcid, "decision": "topic_review_hold", "reason_codes": reasons, "group": group, "fulltext_sha256": access["sha256"]})
        persist()
        return True
    identity_audit = science_identifier_audit(http, science, persist)
    identity_row = (identity_audit.get("sources") or {}).get(record["id"])
    if (
        not identity_row
        or identity_row.get("decision") != "verified"
        or identity_row.get("raw_source_json_sha256") != sha(path.read_bytes())
        or not identity_row.get("same_item_rights_verified")
    ):
        reasons = (identity_row or {}).get("reason_codes", ["identity_audit_missing_source_row"])
        held = science.setdefault("held", {})
        held["identity_audit_hold"] = int(held.get("identity_audit_hold", 0)) + 1
        science.setdefault("candidate_decisions", []).append({
            "pmcid": pmcid, "decision": "identity_audit_hold", "reason_codes": reasons,
            "group": group, "source_json_sha256": sha(path.read_bytes()),
        })
        persist()
        return True
    _apply_science_review_counts(science, manifest_docs, review, identity_audit)
    persist()
    print(f"SOURCE_READY science {record['id']} {record['language']} {record['title'][:110]}", flush=True)
    return True


def plos_preflight(http: SerialHttp, science: dict[str, Any], persist) -> dict[str, Any]:
    existing = science.get("plos_preflight")
    if isinstance(existing, dict) and existing.get("completed"):
        return existing

    api_robot_history = next((row for row in reversed(science.get("requests", [])) if row.get("url", "").startswith(PLOS_API_ROBOTS)), None)
    if api_robot_history is not None:
        api_robot = api_robot_history
    else:
        _, api_robot = http.get(PLOS_API_ROBOTS, RAW_SCIENCE / "preflight/plos-api-robots.txt", "science",
                                max_bytes=500_000, allowed_status=(200, 404), min_spacing_seconds=PLOS_SEARCH_SPACING_SECONDS)
    journal_body, journal_robot = http.get(PLOS_JOURNALS_ROBOTS, RAW_SCIENCE / "preflight/plos-journals-robots.txt", "science",
                                          max_bytes=500_000, min_spacing_seconds=PLOS_ARTICLE_SPACING_SECONDS)
    journal_robots = journal_body.decode("utf-8", "replace")
    article_probe = "https://journals.plos.org/plosone/article/file?id=10.1371%2Fjournal.pone.0170929&type=manuscript"
    if journal_robot.get("http_status") != 200 or not robots_allows(journal_robots, article_probe, http.user_agent):
        science["plos_preflight"] = {"completed": False, "decision": "blocked_by_robots", "journal_robots": journal_robot}
        persist()
        raise RuntimeError("PLOS per-article JATS route is disallowed by live journals.plos.org robots.txt")
    crawl_delay = next((float(match.group(1)) for match in [re.search(r"(?im)^\s*crawl-delay\s*:\s*([0-9.]+)", journal_robots)] if match), None)
    if crawl_delay is None or crawl_delay > PLOS_ARTICLE_SPACING_SECONDS:
        science["plos_preflight"] = {"completed": False, "decision": "crawl_delay_unhandled", "crawl_delay_seconds": crawl_delay}
        persist()
        raise RuntimeError("PLOS robots.txt crawl-delay is missing or exceeds the configured serial spacing")

    docs: dict[str, Any] = {}
    for key, url, relpath in (
        ("search_api_docs", PLOS_SEARCH_API_DOCS, "plos-search-api-faq.html"),
        ("search_fields_docs", PLOS_SOLR_FIELDS_DOCS, "plos-solr-search-fields.html"),
        ("tdm_api_docs", PLOS_TDM_API_DOCS, "plos-tdm-api-docs.html"),
        ("tdm_policy", PLOS_TDM_POLICY, "plos-tdm-policy.html"),
    ):
        body, access = http.get(url, RAW_SCIENCE / "preflight" / relpath, "science", max_bytes=2_000_000,
                                min_spacing_seconds=PLOS_SEARCH_SPACING_SECONDS)
        plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body.decode("utf-8", "replace")))).casefold()
        docs[key] = {"url": url, "raw_file": access.get("raw_file"), "sha256": access.get("sha256"), "http_status": access.get("http_status"), "cached": access.get("cached", False)}
        if access.get("http_status") != 200:
            raise RuntimeError(f"official PLOS preflight failed for {key}")
        if key == "search_api_docs" and ("10 per minute" not in plain or "100 rows" not in plain):
            raise RuntimeError("PLOS Search API rate and page limits were not verified from official documentation")
        if key == "tdm_api_docs" and ("jats xml" not in plain or "bulk downloading of xml is discouraged" not in plain):
            raise RuntimeError("PLOS individual JATS route and bounded-use guidance were not verified")
        if key == "tdm_policy" and "may be mined, reused, and shared by anyone, anywhere, for any purpose" not in plain:
            raise RuntimeError("PLOS text-and-data-mining policy could not be verified")

    access_audit = {
        "schema_version": "plos-source-access-audit-v1",
        "checked_at": utc_now(),
        "decision": "bounded_solr_metadata_and_individual_jats_route_permitted",
        "official_evidence": docs,
        "robots": {
            "api": {"url": PLOS_API_ROBOTS, "http_status": api_robot.get("http_status"),
                    "raw_file": api_robot.get("raw_file"), "sha256": api_robot.get("sha256"),
                    "interpretation": "The documented api.plos.org service returned 404 for robots.txt; no rules file was supplied. The documented /search endpoint is used without changing host or route."},
            "journals": {"url": PLOS_JOURNALS_ROBOTS, "http_status": journal_robot.get("http_status"),
                         "raw_file": journal_robot.get("raw_file"), "sha256": journal_robot.get("sha256"),
                         "crawl_delay_seconds": crawl_delay, "article_file_allowed": True,
                         "disallowed_paths": ["*/search", "*/article/metrics", "*/user"]},
        },
        "limits": {"solr_max_requests_per_minute": 10, "solr_max_rows_per_request": 100,
                   "configured_solr_min_spacing_seconds": PLOS_SEARCH_SPACING_SECONDS,
                   "journals_crawl_delay_seconds": crawl_delay, "configured_article_min_spacing_seconds": PLOS_ARTICLE_SPACING_SECONDS,
                   "max_new_network_bytes": MAX_NETWORK_BYTES, "max_source_chars": MAX_SOURCE_CHARS},
        "rights_rule": "Solr copyright text is only a retrieval candidate hint; article eligibility requires matching, explicit CC-BY or CC0 rights in the same DOI's JATS front/article-meta permissions/license text and hrefs.",
        "route_rule": "Solr metadata discovery followed by small-set per-DOI manuscript JATS XML; no allofPLOS archive or full-corpus route used.",
        "user_agent_product": USER_AGENT_PRODUCT,
        "external_benchmark_access": False,
    }
    write_json(SCIENCE_PLOS_ACCESS_AUDIT, access_audit)
    science["plos_preflight"] = {
        "completed": True, "decision": access_audit["decision"], "access_audit_file": raw_rel(SCIENCE_PLOS_ACCESS_AUDIT),
        "access_audit_sha256": sha(SCIENCE_PLOS_ACCESS_AUDIT.read_bytes()), "checked_at": access_audit["checked_at"],
        "api_robot_status": api_robot.get("http_status"), "journals_robot_status": journal_robot.get("http_status"),
        "article_file_allowed": True, "crawl_delay_seconds": crawl_delay,
    }
    persist()
    return science["plos_preflight"]


def plos_candidate_searches(http: SerialHttp, science: dict[str, Any], persist, *, limit: int,
                            blocked_dois: set[str]) -> list[dict[str, Any]]:
    by_topic: dict[str, list[dict[str, Any]]] = {topic: [] for topic in PLOS_QUERY_TOPICS}
    inventory_rows: list[dict[str, Any]] = []
    seen_dois = {normalize_doi(value) for value in blocked_dois if value}
    candidate_decisions: list[dict[str, Any]] = []
    batch_started = utc_now()
    for topic_group, topic_query in PLOS_QUERY_TOPICS.items():
        license_query = '(copyright:"Creative Commons Attribution 4.0" OR copyright:"CC0 public domain dedication")'
        query = f'({topic_query}) AND article_type:"Research Article" AND {license_query}'
        params = {
            "api_key": "DEMO", "q": query,
            "fl": "id,title_display,abstract,publication_date,journal,subject,article_type,copyright,author_display",
            "start": "0", "rows": "100", "wt": "json", "sort": "publication_date desc",
        }
        raw_path = RAW_SCIENCE / "plos" / f"solr-{topic_group}.json"
        body, access = http.get(PLOS_API_SEARCH, raw_path, "science", params=params,
                                max_bytes=3_000_000, min_spacing_seconds=PLOS_SEARCH_SPACING_SECONDS)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("PLOS Search API returned invalid JSON; metadata screening stopped") from exc
        response = payload.get("response") or {}
        docs = response.get("docs") or []
        response_error = payload.get("error") or payload.get("responseHeader", {}).get("error")
        if response_error:
            raise RuntimeError("PLOS Search API returned an error; metadata screening stopped")
        group_decisions: dict[str, int] = {}
        group_eligible = 0
        for raw_doc in docs:
            candidate, reason = plos_candidate_from_solr(raw_doc, query_group=topic_group, request=access)
            if candidate is None:
                key = reason or "metadata_candidate_rejected"
                group_decisions[key] = group_decisions.get(key, 0) + 1
                continue
            doi = candidate["doi"]
            if doi in seen_dois:
                group_decisions["duplicate_doi_or_prior_work"] = group_decisions.get("duplicate_doi_or_prior_work", 0) + 1
                continue
            seen_dois.add(doi)
            candidate["api_raw_metadata_record"] = raw_doc
            by_topic.setdefault(str(candidate["topic_bucket"]), []).append(candidate)
            group_eligible += 1
        inventory_rows.append({
            "query_group": topic_group, "query": query, "request": access,
            "num_found": response.get("numFound"), "rows_requested": 100,
            "metadata_records_returned": len(docs), "metadata_candidates_after_local_screen": group_eligible,
            "rejected_metadata_records": group_decisions,
        })
        science["plos_candidate_search_progress"] = {
            "started_at": batch_started, "last_completed_group": topic_group,
            "completed_groups": [row["query_group"] for row in inventory_rows],
            "candidate_metadata_count": sum(len(values) for values in by_topic.values()),
        }
        persist()

    topic_rows = [candidate for values in by_topic.values() for candidate in values]
    eligible_by_topic_metadata = {topic: len(values) for topic, values in by_topic.items()}
    # Lower represented topics are selected first, with a round-robin tie break.
    current_counts = dict(science.get("verified_eligible_topic_counts", {}))
    for existing in existing_records(SCIENCE_PLOS_MANIFEST):
        if existing.get("authorable_status") != "proof_verified_owner_adapter_pending":
            continue
        topic = str(existing.get("topic") or "unclassified")
        current_counts[topic] = int(current_counts.get(topic, 0)) + 1
    selected_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    order = list(PLOS_QUERY_TOPICS)
    while len(selected) < limit and any(by_topic.values()):
        available = [
            topic for topic in order
            if by_topic.get(topic) and science_topic_capacity_available(topic, current_counts, selected_counts)
        ]
        if not available:
            break
        topic = min(available, key=lambda value: (int(current_counts.get(value, 0)) + selected_counts.get(value, 0), order.index(value)))
        candidate = by_topic[topic].pop(0)
        selected.append(candidate)
        selected_counts[topic] = selected_counts.get(topic, 0) + 1
    topic_cap_skipped = sum(
        len(by_topic.get(topic, []))
        for topic in order
        if topic in SCIENCE_TOPIC_LIMITS
        and int(current_counts.get(topic, 0)) + int(selected_counts.get(topic, 0)) >= SCIENCE_TOPIC_LIMITS[topic]
    )
    source_rows = [{key: value for key, value in candidate.items() if key != "api_raw_metadata_record"} for candidate in topic_rows]
    inventory = {
        "schema_version": "plos-science-candidate-inventory-v1",
        "generated_at": utc_now(), "bounded_search": True,
        "language_inventory": {
            "solr_language_field_available_in_documented_fields": False,
            "fulltext_language_source": "JATS article xml:lang and abstract language attributes, checked for each retrieved paper",
            "metadata_query_languages": "Search terms were English; no language field is documented in the PLOS Solr search-field list, so no claim is made that unseen languages are unavailable.",
        },
        "license_preflight": "PLOS Solr copyright statements are a candidate hint only, never admitted as article-text rights evidence. Final eligibility requires same-item JATS governing rights proof.",
        "counts": {"metadata_candidates": len(source_rows), "selected_for_bounded_fulltext_batch": len(selected),
                   "eligible_by_topic_metadata": eligible_by_topic_metadata,
                   "selected_by_topic": selected_counts, "skipped_by_topic_cap": topic_cap_skipped},
        "searches": inventory_rows,
        "sources": source_rows,
        "selected_source_ids": [row["id"] for row in selected],
    }
    previous_sha, previous_file = archive_plos_json_sidecar(SCIENCE_PLOS_CANDIDATES, SCIENCE_PLOS_CANDIDATE_REVISIONS)
    if previous_sha:
        inventory["supersedes_previous_sha256"] = previous_sha
        inventory["supersedes_previous_revision_file"] = previous_file
    write_json(SCIENCE_PLOS_CANDIDATES, inventory)
    science["plos_candidate_inventory_file"] = raw_rel(SCIENCE_PLOS_CANDIDATES)
    science["plos_candidate_inventory_sha256"] = sha(SCIENCE_PLOS_CANDIDATES.read_bytes())
    science["plos_candidate_inventory_counts"] = inventory["counts"]
    science["plos_metadata_language_inventory"] = inventory["language_inventory"]
    persist()
    return selected


def science_topic_capacity_available(topic: str, current_counts: dict[str, Any], selected_counts: dict[str, int] | None = None) -> bool:
    cap = SCIENCE_TOPIC_LIMITS.get(topic)
    if cap is None:
        return True
    selected = selected_counts or {}
    return int(current_counts.get(topic, 0)) + int(selected.get(topic, 0)) < cap


def plos_authorable_manifest_count(rows: list[dict[str, Any]]) -> int:
    return sum(row.get("authorable_status") == "proof_verified_owner_adapter_pending" for row in rows)


def record_plos_candidate_rule_revision(science: dict[str, Any]) -> None:
    """Record the narrower evaluation-material detector without rewriting old decisions."""
    rule_id = "plos-external-evaluation-marker-v2"
    path = SCIENCE_PLOS_CANDIDATE_RULE_REVISION_EVENTS
    prior = path.read_bytes() if path.exists() else b""
    for line in prior.splitlines():
        if line.strip():
            try:
                if json.loads(line).get("rule_id") == rule_id:
                    return
            except json.JSONDecodeError:
                continue
    old_decisions = [
        row for row in science.get("plos_candidate_decisions", [])
        if row.get("id") == "science-plos-pone-0335349"
        and row.get("reason_code") == "external_benchmark_or_evaluation_topic"
    ]
    if not old_decisions:
        return
    previous = old_decisions[-1]
    event = {
        "rule_id": rule_id,
        "event_type": "candidate_scope_detector_revised",
        "reviewed_at": utc_now(),
        "reviewed_by": "breadth_acquire_science_public",
        "prior_rule": "Generic standalone 'test set' and 'training dataset' mentions triggered a hold.",
        "current_rule": "Require explicit benchmark/evaluation markers, or test/training/held-out set language near explicit ML/model context.",
        "reason": "The prior generic wording can occur in ordinary experimental method validation. The revised rule narrows future holds to externally scoped evaluation material.",
        "historical_decision_preserved": {
            "id": previous.get("id"), "doi": previous.get("doi"), "decision": previous.get("decision"),
            "reason_code": previous.get("reason_code"), "jats_raw_file": previous.get("jats_raw_file"),
            "jats_raw_sha256": previous.get("jats_raw_sha256"),
        },
        "retroactive_admission": False,
        "external_benchmark_items_or_results_inspected": False,
    }
    encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    atomic_write(path, prior + encoded)
    science.setdefault("plos", {})["candidate_rule_revision_events_file"] = raw_rel(path)
    science["plos"]["candidate_rule_revision_events_sha256"] = sha(path.read_bytes())


def archive_plos_identity_audit(science: dict[str, Any]) -> tuple[str | None, str | None]:
    if not SCIENCE_PLOS_IDENTITY_AUDIT.exists():
        return None, None
    prior_bytes = SCIENCE_PLOS_IDENTITY_AUDIT.read_bytes()
    prior_sha = sha(prior_bytes)
    revision = SCIENCE_PLOS_IDENTITY_AUDIT_REVISIONS / f"{prior_sha}.json"
    if revision.exists() and sha(revision.read_bytes()) != prior_sha:
        raise RuntimeError("content-addressed PLOS identity audit revision has a hash mismatch")
    if not revision.exists():
        atomic_write(revision, prior_bytes)
    reference = raw_rel(revision)
    rows = science.setdefault("plos_identity_audit_revisions", [])
    if not any(isinstance(row, dict) and row.get("sha256") == prior_sha for row in rows):
        rows.append({"sha256": prior_sha, "file": reference, "archived_at": utc_now()})
    return prior_sha, reference


def archive_plos_json_sidecar(path: Path, revision_dir: Path) -> tuple[str | None, str | None]:
    if not path.exists():
        return None, None
    prior_bytes = path.read_bytes()
    prior_sha = sha(prior_bytes)
    revision_path = revision_dir / f"{prior_sha}.json"
    if revision_path.exists() and sha(revision_path.read_bytes()) != prior_sha:
        raise RuntimeError(f"content-addressed PLOS sidecar revision hash mismatch: {path.name}")
    if not revision_path.exists():
        atomic_write(revision_path, prior_bytes)
    return prior_sha, raw_rel(revision_path)


class PLOSFieldTable(HTMLParser):
    """Read field names from the official PLOS Solr documentation table."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs):
        if tag == "tr":
            self.row = []
        elif tag in {"td", "th"}:
            self.cell = []

    def handle_endtag(self, tag: str):
        if tag in {"td", "th"} and self.cell is not None:
            value = " ".join("".join(self.cell).split())
            assert self.row is not None
            self.row.append(value)
            self.cell = None
        elif tag == "tr" and self.row:
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data: str):
        if self.cell is not None:
            self.cell.append(data)


def plos_journal_language_policies(http: SerialHttp, science: dict[str, Any], persist,
                                   journal_slugs: set[str]) -> dict[str, dict[str, Any]]:
    field_path = RAW_SCIENCE / "preflight/plos-solr-search-fields.html"
    if field_path.exists():
        field_bytes = field_path.read_bytes()
        field_request = {"raw_file": raw_rel(field_path), "sha256": sha(field_bytes), "cached": True}
    else:
        field_bytes, field_request = http.get(
            PLOS_SOLR_FIELDS_DOCS, field_path, "science", max_bytes=2_000_000,
            min_spacing_seconds=PLOS_SEARCH_SPACING_SECONDS,
        )
    field_parser = PLOSFieldTable()
    field_parser.feed(field_bytes.decode("utf-8", "replace"))
    documented_fields = [row[0] for row in field_parser.rows if row and row[0] and row[0].casefold() != "plos search field name"]
    language_field_available = any(name.casefold() in {"language", "lang", "article_language"} for name in documented_fields)

    robots_path = RAW_SCIENCE / "preflight/plos-journals-robots.txt"
    robots_text = robots_path.read_text(encoding="utf-8", errors="replace") if robots_path.exists() else ""
    policies: dict[str, dict[str, Any]] = {}
    for journal_slug in sorted(journal_slugs):
        url = PLOS_SUBMISSION_GUIDELINES.format(journal_slug=journal_slug)
        raw_path = RAW_SCIENCE / "preflight" / f"plos-{journal_slug}-submission-guidelines.html"
        if not robots_allows(robots_text, url, USER_AGENT_PRODUCT):
            policies[journal_slug] = {
                "journal_slug": journal_slug, "url": url, "decision": "robots_disallowed",
                "language_requirement_verified": False, "translations_as_supporting_information_verified": False,
            }
            continue
        body, request = http.get(url, raw_path, "science", max_bytes=2_000_000,
                                 min_spacing_seconds=PLOS_ARTICLE_SPACING_SECONDS)
        plain = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", body.decode("utf-8", "replace")))).casefold()
        english_required = "manuscripts must be submitted in english" in plain
        translations_supported = (
            "translations of the manuscript or abstract" in plain
            and "supporting information" in plain
        )
        policies[journal_slug] = {
            "journal_slug": journal_slug, "url": url, "http_status": request.get("http_status"),
            "raw_file": request.get("raw_file"), "sha256": request.get("sha256"),
            "language_requirement_verified": english_required,
            "translations_as_supporting_information_verified": translations_supported,
            "decision": "english_main_manuscript_policy_verified" if english_required else "publisher_language_policy_needs_review",
        }

    science["plos_language_policy_inventory"] = {
        "solr_fields_document": {"url": PLOS_SOLR_FIELDS_DOCS, "raw_file": field_request.get("raw_file"),
                                  "sha256": field_request.get("sha256"), "documented_fields": documented_fields,
                                  "language_field_available": language_field_available},
        "journal_policies": policies,
        "checked_at": utc_now(),
    }
    persist()
    return policies


def plos_source_language_evidence(xml_bytes: bytes, manifest_entry: dict[str, Any],
                                  policy: dict[str, Any]) -> dict[str, Any]:
    root = ET.fromstring(xml_bytes)
    article = next((node for node in root.iter() if xml_local(node.tag) == "article"), root)
    body = next((node for node in article.iter() if xml_local(node.tag) == "body"), None)
    xml_lang_key = "{http://www.w3.org/XML/1998/namespace}lang"

    def lang(node: ET.Element | None) -> str | None:
        if node is None:
            return None
        value = node.attrib.get(xml_lang_key) or node.attrib.get("lang")
        return normalize_science_language(value) if value else None

    abstracts = [
        {"element": xml_local(node.tag), "language": lang(node)}
        for node in article.iter() if xml_local(node.tag) in {"abstract", "trans-abstract"}
    ]
    sections = [
        lang(node) for node in article.iter()
        if xml_local(node.tag) == "sec" and lang(node)
    ]
    relevant_tags = []
    for node in article.iter():
        if xml_local(node.tag) in {"article", "body", "sec", "abstract", "trans-abstract"}:
            current_lang = lang(node)
            if current_lang:
                relevant_tags.append({"element": xml_local(node.tag), "language": current_lang})
    supplementaries = []
    language_markers = re.compile(r"\b(translation|translated|english|spanish|french|german|chinese|arabic|portuguese|italian|welsh|deutsch|español|français|中文|العربية)\b", re.I)
    for node in article.iter():
        if xml_local(node.tag) != "supplementary-material":
            continue
        descriptive_text = " ".join(" ".join(node.itertext()).split())
        supplementaries.append({
            "id": node.attrib.get("id"),
            "mimetype": node.attrib.get("mimetype"),
            "language_or_translation_terms": sorted({value.casefold() for value in language_markers.findall(descriptive_text)}),
            "content_retrieved": False,
        })
    article_lang = lang(article)
    body_lang = lang(body)
    abstract_langs = sorted({row["language"] for row in abstracts if row.get("language")})
    explicit_languages = sorted({row["language"] for row in relevant_tags})
    source_language = normalize_science_language(str(manifest_entry.get("language") or "und"))
    language_consistent = (
        source_language == "und" or not explicit_languages or source_language in explicit_languages
    )
    return {
        "source_language": source_language,
        "native_main_text_language_basis": "JATS article/@xml:lang or @lang; body/section language tags are recorded separately where present",
        "article_xml_lang": article_lang,
        "body_xml_lang": body_lang,
        "abstract_language_fields": abstracts,
        "explicit_section_languages": sections,
        "languages_explicitly_tagged_in_article_body": explicit_languages,
        "language_evidence_consistent": language_consistent,
        "jats_raw_file": manifest_entry.get("jats_raw_file"),
        "jats_raw_sha256": manifest_entry.get("jats_raw_sha256"),
        "solr_language_field_available": False,
        "solr_language_field_evidence": science_language_field_summary(),
        "solr_search_fields_document_url": PLOS_SOLR_FIELDS_DOCS,
        "solr_search_fields_document_raw_file": raw_rel(RAW_SCIENCE / "preflight/plos-solr-search-fields.html"),
        "solr_search_fields_document_sha256": sha((RAW_SCIENCE / "preflight/plos-solr-search-fields.html").read_bytes()) if (RAW_SCIENCE / "preflight/plos-solr-search-fields.html").exists() else None,
        "publisher_guidelines_url": policy.get("url"),
        "publisher_guidelines_raw_file": policy.get("raw_file"),
        "publisher_guidelines_sha256": policy.get("sha256"),
        "publisher_language_policy_decision": policy.get("decision"),
        "english_main_manuscript_required": bool(policy.get("language_requirement_verified")),
        "translations_may_be_supporting_information": bool(policy.get("translations_as_supporting_information_verified")),
        "supporting_information": {
            "files_listed_in_article_jats": supplementaries,
            "translation_availability": "not_assessed; linked supporting-information content was not retrieved",
        },
    }


def science_language_field_summary() -> str:
    return "Official PLOS Solr Search Fields documentation was parsed; no language, lang, or article_language search field is listed."


def update_plos_language_and_topic_sidecars(http: SerialHttp, science: dict[str, Any], persist,
                                            policies: dict[str, dict[str, Any]]) -> None:
    manifest = existing_records(SCIENCE_PLOS_MANIFEST)
    if not manifest:
        return
    topic_doc = read_json(SCIENCE_PLOS_TOPIC_REVIEW, {"sources": []})
    topic_rows = {str(row.get("id")): row for row in topic_doc.get("sources", [])}
    audit_doc = read_json(SCIENCE_PLOS_IDENTITY_AUDIT, {"sources": {}})
    identity_rows = audit_doc.get("sources", {})
    events: list[dict[str, Any]] = []
    language_rows: list[dict[str, Any]] = []
    changed_topic = False
    changed_manifest = False
    manifest_by_id = {str(row.get("id")): row for row in manifest}
    correction_specs = {
        "science-plos-pone-0318451": {
            "current_topic": "environment_ecology",
            "reason": "The paper studies bat fatalities and deterrent/curtailment outcomes at a wind-energy facility. Environmental ecology is the research context; the previous physical-science label was triggered by the word energy.",
        },
        "science-plos-pntd-0014005": {
            "current_topic": "public_health",
            "reason": "The paper develops and tests a triatomine vector-surveillance trap. Its central scope is disease-vector surveillance procedure, not physical-science engineering.",
        },
        "science-plos-pgph-0006916": {
            "current_topic": "public_health",
            "reason": "The review examines water, sanitation, and hygiene interventions in schistosomiasis control. Disease prevention and transmission pathways define the scope; agriculture is not the main research topic.",
        },
        "science-plos-pone-0319386": {
            "current_topic": "public_health",
            "reason": "The paper analyzes dietary niacin intake and hearing loss in a population sample. This is nutritional epidemiology, not physical science.",
        },
        "science-plos-pntd-0014642": {
            "current_topic": "biological_science",
            "reason": "The paper engineers and evaluates an antiviral chimeric immunoglobulin. This is molecular/biological research; the biological-science topic cap is already full, so the item remains held from authoring.",
            "authoring_hold_reason": "biological_topic_cap_reached",
        },
        "science-plos-pone-0316466": {
            "current_topic": "biological_science",
            "reason": "The paper characterizes plasmids, virulence potential, and genomic relationships in Salmonella isolates. The agricultural surface-water setting is sampling context; the study's primary subject is microbial biology. The biological-science cap is full, so the source remains held from authoring.",
            "authoring_hold_reason": "biological_topic_cap_reached",
        },
        "science-plos-pone-0268700": {
            "current_topic": "public_health",
            "reason": "The study tests associations between solar/geomagnetic exposure and inflammation/endothelial biomarkers in a human cohort. Health outcomes in population data define the scope; solar terms alone do not make this a physical-science study.",
        },
        "science-plos-ppat-1010564": {
            "current_topic": "biological_science",
            "reason": "The paper engineers viral attachment glycoproteins to change vaccine stability. This is virology/protein research; the biological-science cap is full, so it remains held from authoring.",
            "authoring_hold_reason": "biological_topic_cap_reached",
        },
        "science-plos-pone-0258290": {
            "current_topic": "agriculture_food",
            "reason": "The paper studies dairy farmworker practices, antibiotic stewardship, and infection-prevention behavior on farms. Systems engineering is the analytic framework, not the central topic.",
        },
        "science-plos-pone-0257896": {
            "current_topic": "biological_science",
            "reason": "The paper investigates light-triggered vasodilation through a potassium-channel mechanism. This is experimental physiology; the biological-science cap is full, so it remains held from authoring.",
            "authoring_hold_reason": "biological_topic_cap_reached",
        },
        "science-plos-pone-0215686": {
            "current_topic": "public_health",
            "reason": "The paper studies sugar-sweetened beverage consumption and diet quality in a pregnancy cohort. Nutrition epidemiology is the topic; energy intake does not make it physical science.",
        },
        "science-plos-pone-0204487": {
            "current_topic": "biological_science",
            "reason": "The paper uses CRISPR/Cas9 to engineer a human renal-cell reporter line. This is cellular biology; the biological-science cap is full, so it remains held from authoring.",
            "authoring_hold_reason": "biological_topic_cap_reached",
        },
        "science-plos-pone-0201892": {
            "current_topic": "public_health",
            "reason": "The statistical model addresses measurement error in dietary energy balance and intake using human measurement data. Nutrition methodology defines the application; the word energy is not physical-science scope.",
        },
        "science-plos-pone-0196743": {
            "current_topic": "biological_science",
            "reason": "The study tests dietary challenges and sleep/activity responses in a mouse model. This is experimental animal physiology; the biological-science cap is full, so it remains held from authoring.",
            "authoring_hold_reason": "biological_topic_cap_reached",
        },
        "science-plos-pone-0355827": {
            "current_topic": "society_behavior",
            "reason": "The study analyzes foreign land transactions, spatial proximity to military interests, and policy attention. Agricultural land is the setting; investment and regulatory patterns define the research question.",
        },
        "science-plos-pone-0177953": {
            "current_topic": "biological_science",
            "reason": "The paper profiles energy metabolism in high-fat-diet-fed mice. This is experimental animal metabolism; the biological-science cap is full, so it remains held from authoring.",
            "authoring_hold_reason": "biological_topic_cap_reached",
        },
        "science-plos-pone-0156197": {
            "current_topic": "biological_science",
            "reason": "The paper engineers an alginate lyase enzyme and measures catalytic activity and biofilm disruption. This is molecular biology/biochemistry; the biological-science cap is full, so it remains held from authoring.",
            "authoring_hold_reason": "biological_topic_cap_reached",
        },
        "science-plos-pone-0342552": {
            "current_topic": "society_behavior",
            "reason": "The study analyzes unequal household exposure to livestock-operation pollution by race and income. Its central question is environmental justice and social inequality, not farm production methods.",
        },
        "science-plos-pclm-0000984": {
            "current_topic": "public_health",
            "reason": "The scoping review examines mental-health and sexual/reproductive-health outcomes associated with climate hazards across populations. Climate is the exposure context; public health is the main research scope.",
        },
        "science-plos-pntd-0013648": {
            "current_topic": "public_health",
            "reason": "The study measures Crimean-Congo hemorrhagic fever exposure in livestock and farm workers and calls for integrated One Health surveillance. Zoonotic disease prevention is the research scope; livestock is the sampling context.",
        },
        "science-plos-pone-0147223": {
            "current_topic": "environment_ecology",
            "reason": "The paper measures PCB burdens in wild summer flounder and explains sex differences using growth and bioenergetics. Wildlife contaminant ecology is the scope; energy expenditure is an explanatory mechanism, not physical science.",
        },
        "science-plos-pone-0132157": {
            "current_topic": "public_health",
            "reason": "The paper compares DXA body-composition measures with BMI in current and former U.S. Navy members while framing the question around obesity prevalence and misclassification. Population-health measurement is the scope; the imaging method is not physical-science research.",
        },
    }

    for entry in manifest:
        source_id = str(entry.get("id"))
        topic_row = topic_rows.get(source_id)
        identity_row = identity_rows.get(source_id)
        if topic_row is None or identity_row is None:
            raise RuntimeError(f"PLOS language inventory found a missing topic or identity row for {source_id}")
        source_path = ROOT / str(entry.get("source_json"))
        jats_path = ROOT / str(entry.get("jats_raw_file"))
        if not source_path.exists() or sha(source_path.read_bytes()) != entry.get("source_json_sha256"):
            raise RuntimeError(f"PLOS source JSON hash changed before language audit: {source_id}")
        if not jats_path.exists() or sha(jats_path.read_bytes()) != entry.get("jats_raw_sha256"):
            raise RuntimeError(f"PLOS JATS raw hash changed before language audit: {source_id}")
        source_record = read_json(source_path, {})
        journal_slug = plos_article_slug(str(source_record.get("journal") or ""), str(source_record.get("doi") or ""))
        policy = policies.get(journal_slug or "", {})
        language_evidence = plos_source_language_evidence(jats_path.read_bytes(), entry, policy)
        old_row_sha = plos_topic_review_row_sha256(topic_row)
        old_topic = str(topic_row.get("topic"))
        old_scope_sha = str(identity_row.get("identity_scope_sha256") or "")
        if identity_row.get("raw_source_json_sha256") != entry.get("source_json_sha256"):
            raise RuntimeError(f"PLOS identity row does not bind the current source JSON: {source_id}")
        if identity_row.get("topic_review_record_sha256") != old_row_sha:
            raise RuntimeError(f"PLOS identity row does not bind the current topic row: {source_id}")
        row_changed = topic_row.get("source_language_evidence") != language_evidence
        correction = correction_specs.get(source_id)
        current_topic = str(correction["current_topic"]) if correction else str(topic_row.get("topic"))
        needs_correction = correction is not None and topic_row.get("topic") != current_topic
        if not row_changed and not needs_correction:
            language_rows.append({
                "id": source_id, "doi": entry.get("doi"), "source_json_sha256": entry.get("source_json_sha256"),
                "topic": topic_row.get("topic"), "topic_review_decision": topic_row.get("decision"),
                "authoring_eligible": topic_row.get("decision") == "verified_proof_pending_adapter",
                "topic_review_record_sha256": old_row_sha, **language_evidence,
            })
            continue

        if needs_correction:
            receipt = {
                "previous_topic": topic_row.get("topic"),
                "previous_topic_review_record_sha256": old_row_sha,
                "current_topic": current_topic,
                "reviewed_by": "breadth_acquire_science_public",
                "reason": correction["reason"],
                "source_json_sha256": entry.get("source_json_sha256"),
                "source_text_sha256": entry.get("source_text_sha256"),
                "jats_raw_sha256": entry.get("jats_raw_sha256"),
                "reviewed_at": utc_now(),
            }
            topic_row["topic"] = current_topic
            topic_row["topic_classification_method"] = "source_fulltext_contextual_scope_review"
            topic_row["correction_receipt"] = receipt
            if correction.get("authoring_hold_reason"):
                topic_row["decision"] = "held_not_authorable"
                topic_row["reason_codes"] = [correction["authoring_hold_reason"]]
                topic_row["authoring_eligible"] = False
                entry["authorable_status"] = "held_not_authorable_topic_cap"
                entry["authoring_hold_reason"] = correction["authoring_hold_reason"]
            else:
                topic_row["decision"] = "verified_proof_pending_adapter"
                topic_row["reason_codes"] = ["owner_source_adapter_pending"]
                entry["authorable_status"] = "proof_verified_owner_adapter_pending"
            entry["topic"] = current_topic
            changed_manifest = True
            event_type = "topic_correction_and_language_evidence"
        else:
            event_type = "language_evidence_added"
        topic_row["source_language_evidence"] = language_evidence
        topic_row["source_language_evidence_sha256"] = plos_topic_review_row_sha256(language_evidence)
        topic_row["source_json_sha256"] = entry.get("source_json_sha256")
        new_row_sha = plos_topic_review_row_sha256(topic_row)
        identity_row["topic"] = current_topic
        identity_row["topic_review_record_sha256"] = new_row_sha
        identity_row["identity_scope_sha256"] = plos_identity_scope_sha256(identity_row)
        language_rows.append({
            "id": source_id, "doi": entry.get("doi"), "source_json_sha256": entry.get("source_json_sha256"),
            "topic": topic_row.get("topic"), "topic_review_decision": topic_row.get("decision"),
            "authoring_eligible": topic_row.get("decision") == "verified_proof_pending_adapter",
            "topic_review_record_sha256": new_row_sha, **language_evidence,
        })
        events.append({
            "event_type": event_type, "id": source_id, "doi": entry.get("doi"),
            "previous_topic": old_topic,
            "previous_topic_review_record_sha256": old_row_sha,
            "current_topic": current_topic, "current_topic_review_record_sha256": new_row_sha,
            "reviewed_by": "breadth_acquire_science_public",
            "reason": correction["reason"] if correction else "Added hash-bound publisher/JATS language availability evidence; topic label and source text/license were not changed.",
            "source_json_sha256": entry.get("source_json_sha256"),
            "source_text_sha256": entry.get("source_text_sha256"), "jats_raw_sha256": entry.get("jats_raw_sha256"),
            "previous_identity_scope_sha256": old_scope_sha,
            "current_identity_scope_sha256": identity_row.get("identity_scope_sha256"),
            "reviewed_at": utc_now(),
        })
        topic_rows[source_id] = topic_row
        identity_rows[source_id] = identity_row
        changed_topic = True

    held_rows = []
    candidate_inventory = read_json(SCIENCE_PLOS_CANDIDATES, {"sources": []})
    candidate_by_id = {str(row.get("id")): row for row in candidate_inventory.get("sources", [])}
    for decision in science.get("plos_candidate_decisions", []):
        if decision.get("decision") != "hold" or not decision.get("jats_raw_file"):
            continue
        held_path = ROOT / str(decision["jats_raw_file"])
        if not held_path.exists():
            continue
        try:
            root = ET.fromstring(held_path.read_bytes())
        except ET.ParseError:
            continue
        article = next((node for node in root.iter() if xml_local(node.tag) == "article"), root)
        lang_value = article.attrib.get("{http://www.w3.org/XML/1998/namespace}lang") or article.attrib.get("lang")
        candidate = candidate_by_id.get(str(decision.get("id")), {})
        corrected_topic, topic_reason = classify_science_topic(
            str(candidate.get("title") or ""), str(candidate.get("abstract") or ""),
            " ".join(str(value) for value in candidate.get("subjects", [])),
            normalize_science_language(lang_value or "und"),
        )
        prior_candidate_topic = candidate.get("topic_bucket")
        held_rows.append({
            "id": decision.get("id"), "doi": decision.get("doi"), "decision": "held_not_authorable",
            "reason_code": decision.get("reason_code"), "source_json": None,
            "jats_raw_file": decision.get("jats_raw_file"), "jats_raw_sha256": decision.get("jats_raw_sha256"),
            "article_xml_lang": normalize_science_language(lang_value) if lang_value else None,
            "previous_metadata_topic": prior_candidate_topic, "topic_reviewed_as": corrected_topic,
            "topic_review_reason": topic_reason,
        })
        if corrected_topic and prior_candidate_topic != corrected_topic:
            candidate_sha = plos_topic_review_row_sha256(candidate) if candidate else None
            events.append({
                "event_type": "held_candidate_topic_correction", "id": decision.get("id"), "doi": decision.get("doi"),
                "previous_topic": prior_candidate_topic, "previous_topic_review_record_sha256": candidate_sha,
                "current_topic": corrected_topic, "current_topic_review_record_sha256": None,
                "reviewed_by": "breadth_acquire_science_public",
                "reason": "Metadata topic was corrected from the title/abstract context, but the source remains non-admittable under its full-text size hold.",
                "source_json_sha256": None, "candidate_metadata_sha256": candidate.get("solr_metadata_raw_sha256"),
                "jats_raw_sha256": decision.get("jats_raw_sha256"), "hold_reason": decision.get("reason_code"),
                "authoring_eligible": False, "reviewed_at": utc_now(),
            })

    if changed_topic:
        topic_sha, topic_revision = archive_plos_json_sidecar(SCIENCE_PLOS_TOPIC_REVIEW, SCIENCE_PLOS_TOPIC_REVISIONS)
        audit_sha, audit_revision = archive_plos_identity_audit(science)
        if changed_manifest:
            archive_plos_json_sidecar(SCIENCE_PLOS_MANIFEST, SCIENCE_PLOS_MANIFEST_REVISIONS)
        topic_doc["sources"] = [topic_rows[key] for key in sorted(topic_rows)]
        topic_doc["generated_at"] = utc_now()
        topic_doc["supersedes_previous_sha256"] = topic_sha
        topic_doc["supersedes_previous_revision_file"] = topic_revision
        topic_doc["counts"] = {
            "proof_verified_adapter_pending": sum(row.get("decision") == "verified_proof_pending_adapter" for row in topic_doc["sources"]),
            "held_not_authorable": sum(row.get("decision") != "verified_proof_pending_adapter" for row in topic_doc["sources"]),
            "by_topic": dict(Counter(row.get("topic") for row in topic_doc["sources"] if row.get("decision") == "verified_proof_pending_adapter")),
        }
        audit_doc["sources"] = identity_rows
        audit_doc["generated_at"] = utc_now()
        audit_doc["supersedes_previous_audit_sha256"] = audit_sha
        audit_doc["supersedes_previous_audit_revision_file"] = audit_revision
        audit_doc["counts"] = {
            "proof_verified_adapter_pending": sum(topic_rows.get(source_id, {}).get("decision") == "verified_proof_pending_adapter" for source_id in identity_rows),
            "identity_verified_held_for_topic_scope": sum(
                topic_rows.get(source_id, {}).get("decision") != "verified_proof_pending_adapter"
                and bool(row.get("same_item_rights_verified"))
                for source_id, row in identity_rows.items()
            ),
            "same_item_rights_verified": sum(bool(row.get("same_item_rights_verified")) for row in identity_rows.values()),
        }
        write_json(SCIENCE_PLOS_TOPIC_REVIEW, topic_doc)
        write_json(SCIENCE_PLOS_IDENTITY_AUDIT, audit_doc)
        if changed_manifest:
            save_manifest(SCIENCE_PLOS_MANIFEST, manifest)
        if events:
            prior_log = SCIENCE_PLOS_TOPIC_REVISION_EVENTS.read_bytes() if SCIENCE_PLOS_TOPIC_REVISION_EVENTS.exists() else b""
            prior_event_rows = []
            for line in prior_log.splitlines():
                if line:
                    try:
                        prior_event_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
            new_lines = []
            for event in events:
                if any(
                    prior.get("event_type") == event.get("event_type")
                    and prior.get("id") == event.get("id")
                    and prior.get("current_topic") == event.get("current_topic")
                    and prior.get("source_json_sha256") == event.get("source_json_sha256")
                    for prior in prior_event_rows
                ):
                    continue
                encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
                new_lines.append(encoded + b"\n")
            if new_lines:
                atomic_write(SCIENCE_PLOS_TOPIC_REVISION_EVENTS, prior_log + b"".join(new_lines))

    inventory = {
        "schema_version": "plos-source-language-inventory-v1", "generated_at": utc_now(),
        "scope": "Native-language evidence for bounded PLOS article JATS retrievals; not a census of unobserved supplementary files or all publisher content.",
        "metadata_language_inventory": {
            "official_fields_document_url": PLOS_SOLR_FIELDS_DOCS,
            "documented_fields": science.get("plos_language_policy_inventory", {}).get("solr_fields_document", {}).get("documented_fields", []),
            "language_field_available": science.get("plos_language_policy_inventory", {}).get("solr_fields_document", {}).get("language_field_available"),
            "interpretation": "PLOS Search API exposes no documented language field. English query terms do not prove that all article languages were searched.",
            "raw_file": science.get("plos_language_policy_inventory", {}).get("solr_fields_document", {}).get("raw_file"),
            "sha256": science.get("plos_language_policy_inventory", {}).get("solr_fields_document", {}).get("sha256"),
        },
        "publisher_language_policies": science.get("plos_language_policy_inventory", {}).get("journal_policies", {}),
        "observed_main_text_languages": dict(Counter(row.get("source_language") for row in language_rows)),
        "observed_explicit_jats_languages": dict(Counter(language for row in language_rows for language in row.get("languages_explicitly_tagged_in_article_body", []))),
        "native_non_english_papers": [row["id"] for row in language_rows if row.get("source_language") not in {"en", "eng", "und", None}],
        "supporting_information_note": "PLOS One permits translations as supporting information; linked files were inventoried only from article JATS and not downloaded in this bounded article-text sample. Translation-language availability remains not assessed.",
        "source_count": len(language_rows),
        "authoring_eligible_source_count": sum(bool(row.get("authoring_eligible")) for row in language_rows),
        "held_topic_cap_sources": [
            {"id": row.get("id"), "doi": row.get("doi"), "topic": row.get("topic"),
             "decision": row.get("topic_review_decision"), "source_json_sha256": row.get("source_json_sha256"),
             "topic_review_record_sha256": row.get("topic_review_record_sha256")}
            for row in language_rows if not row.get("authoring_eligible")
        ],
        "held_fulltext_count": len(held_rows),
        "sources": language_rows, "held_fulltexts": held_rows,
    }
    write_json(SCIENCE_PLOS_LANGUAGE_INVENTORY, inventory)
    decision_log = science.setdefault("plos_candidate_decisions", [])
    logged_topic_receipts = {
        (row.get("id"), row.get("topic_review_record_sha256"), row.get("decision"))
        for row in decision_log
        if isinstance(row, dict)
    }
    for source_id, topic_row in topic_rows.items():
        receipt = topic_row.get("correction_receipt")
        if not isinstance(receipt, dict):
            continue
        held = topic_row.get("decision") == "held_not_authorable"
        decision = "hold" if held else "topic_correction_verified"
        topic_hash = plos_topic_review_row_sha256(topic_row)
        key = (source_id, topic_hash, decision)
        if key in logged_topic_receipts:
            continue
        identity_row = identity_rows.get(source_id, {})
        decision_log.append({
            "id": source_id, "doi": topic_row.get("doi"), "decision": decision,
            "reason_code": (topic_row.get("reason_codes") or [None])[0] if held else None,
            "topic": topic_row.get("topic"),
            "source_json_sha256": topic_row.get("source_json_sha256"),
            "topic_review_record_sha256": topic_hash,
            "identity_scope_sha256": identity_row.get("identity_scope_sha256"),
            "supersedes_topic": receipt.get("previous_topic"),
            "decision_recorded_at": utc_now(),
        })
        logged_topic_receipts.add(key)
    science["plos_candidate_decisions_count"] = len(decision_log)
    plos_status = science.setdefault("plos", {})
    plos_status["language_inventory_file"] = raw_rel(SCIENCE_PLOS_LANGUAGE_INVENTORY)
    plos_status["language_inventory_sha256"] = sha(SCIENCE_PLOS_LANGUAGE_INVENTORY.read_bytes())
    plos_status["main_text_language_counts"] = inventory["observed_main_text_languages"]
    plos_status["native_non_english_fulltext_count"] = len(inventory["native_non_english_papers"])
    plos_status["topic_review_file"] = raw_rel(SCIENCE_PLOS_TOPIC_REVIEW)
    plos_status["topic_review_sha256"] = sha(SCIENCE_PLOS_TOPIC_REVIEW.read_bytes())
    plos_status["identity_audit_file"] = raw_rel(SCIENCE_PLOS_IDENTITY_AUDIT)
    plos_status["identity_audit_sha256"] = sha(SCIENCE_PLOS_IDENTITY_AUDIT.read_bytes())
    plos_status["proof_verified_adapter_pending_count"] = plos_authorable_manifest_count(manifest)
    plos_status["held_topic_scope_source_count"] = len(manifest) - plos_authorable_manifest_count(manifest)
    plos_status["proof_verified_topic_counts"] = dict(Counter(
        row.get("topic") for row in manifest
        if row.get("authorable_status") == "proof_verified_owner_adapter_pending"
    ))
    science["plos_proof_verified_adapter_pending_count"] = plos_authorable_manifest_count(manifest)
    if changed_manifest:
        plos_status["topic_correction_revision_events_file"] = raw_rel(SCIENCE_PLOS_TOPIC_REVISION_EVENTS)
    science["plos_authoring_adapter_ready"] = False
    persist()


def science_plos_acquisition(http: SerialHttp, science: dict[str, Any], persist, *, limit: int) -> None:
    plos_preflight(http, science, persist)
    record_plos_candidate_rule_revision(science)
    persist()
    existing_science = existing_records(SCIENCE_MANIFEST)
    plos_records = existing_records(SCIENCE_PLOS_MANIFEST)
    prior = load_prior_keys()
    blocked_dois = set(prior["doi"])
    blocked_dois.update(normalize_doi(str(row.get("doi") or "")) for row in existing_science if row.get("doi"))
    blocked_dois.update(normalize_doi(str(row.get("doi") or "")) for row in plos_records if row.get("doi"))
    for decision in science.get("plos_candidate_decisions", []):
        # A screened DOI needs an explicit correction event before it can be reconsidered.
        terminal_hold = decision.get("decision") in {"hold", "route_or_network_hold", "source_contract_hold"} or decision.get("reason_code") in {
            "source_exceeds_100k_chars_hold", "solr_jats_item_license_mismatch_or_conflict",
            "external_benchmark_or_evaluation_topic", "xml_parse_error", "jats_front_article_meta_missing",
            "solr_jats_doi_mismatch", "solr_jats_title_mismatch",
        }
        if terminal_hold and decision.get("doi"):
            blocked_dois.add(normalize_doi(str(decision["doi"])))
    authorable_plos_count = plos_authorable_manifest_count(plos_records)
    remaining_plos_target = max(0, TARGET_PLOS_PROOF_SOURCES - authorable_plos_count)
    selected = plos_candidate_searches(
        http, science, persist, limit=min(limit, remaining_plos_target), blocked_dois=blocked_dois
    ) if remaining_plos_target else []
    topic_counts = {key: int(value) for key, value in science.get("verified_eligible_topic_counts", {}).items()}
    for existing in plos_records:
        if existing.get("authorable_status") != "proof_verified_owner_adapter_pending":
            continue
        topic = str(existing.get("topic") or "unclassified")
        topic_counts[topic] = int(topic_counts.get(topic, 0)) + 1
    journal_slugs = {str(row.get("journal_slug")) for row in selected if row.get("journal_slug")}
    journal_slugs.update(
        slug for row in plos_records
        if (slug := plos_article_slug(str(row.get("journal") or ""), str(row.get("doi") or "")))
    )
    policies = plos_journal_language_policies(http, science, persist, journal_slugs)
    plos_topic_rows = {str(row.get("id")): row for row in read_json(SCIENCE_PLOS_TOPIC_REVIEW, {"sources": []}).get("sources", [])}
    plos_identity_rows = read_json(SCIENCE_PLOS_IDENTITY_AUDIT, {"sources": {}}).get("sources", {})
    manifest_by_id = {str(row.get("id")): row for row in plos_records}
    decisions = science.setdefault("plos_candidate_decisions", [])
    status_counts = science.setdefault("plos", {})
    status_counts["retrieval_route"] = "PLOS Solr to official per-article JATS; article-text rights verified from governing article-meta only"
    status_counts["fulltext_candidates_selected"] = len(selected)
    persist()

    new_pending_ids: list[str] = []
    for candidate in selected:
        if plos_authorable_manifest_count(plos_records) >= TARGET_PLOS_PROOF_SOURCES:
            break
        source_id = str(candidate["id"])
        document_path = SCIENCE_PLOS_PENDING_DOCS / f"{source_id}.json"
        if source_id in manifest_by_id and document_path.exists():
            continue
        xml_url = f"https://journals.plos.org/{candidate['journal_slug']}/article/file"
        article_params = {"id": candidate["doi"], "type": "manuscript"}
        raw_path = RAW_SCIENCE / "plos" / "articles" / f"{source_id}-jats.xml"
        try:
            xml_bytes, access = http.get(xml_url, raw_path, "science", params=article_params,
                                         max_bytes=10_000_000, min_spacing_seconds=PLOS_ARTICLE_SPACING_SECONDS)
        except RuntimeError as exc:
            decision = {"id": source_id, "doi": candidate["doi"], "decision": "route_or_network_hold", "reason_code": str(exc)[:120],
                        "solr_metadata_raw_file": candidate.get("solr_metadata_raw_file"), "solr_metadata_raw_sha256": candidate.get("solr_metadata_raw_sha256")}
            decisions.append(decision)
            persist()
            continue
        record, identity_row, result = parse_plos_science_document(xml_bytes, candidate, access)
        if record is None:
            decisions.append({"id": source_id, "doi": candidate["doi"], "decision": "hold", "reason_code": result,
                              "jats_raw_file": access.get("raw_file"), "jats_raw_sha256": access.get("sha256"),
                              "same_item_rights_verified": identity_row.get("same_item_rights_verified", False)})
            persist()
            continue
        abstract = " ".join(str(section.get("text") or "") for section in record.get("sections", []) if section.get("section_type") == "abstract")
        topic, scope_reason = classify_science_topic(str(record.get("title") or ""), abstract, "", str(record.get("language") or "und"))
        if scope_reason or topic is None:
            decisions.append({"id": source_id, "doi": candidate["doi"], "decision": "hold", "reason_code": scope_reason or "no_topic",
                              "jats_raw_file": access.get("raw_file"), "jats_raw_sha256": access.get("sha256")})
            persist()
            continue
        if not science_topic_capacity_available(topic, topic_counts):
            decisions.append({
                "id": source_id, "doi": candidate["doi"], "decision": "hold",
                "reason_code": "biological_topic_cap_reached", "topic": topic,
                "jats_raw_file": access.get("raw_file"), "jats_raw_sha256": access.get("sha256"),
                "authoring_eligible": False,
            })
            persist()
            continue
        if SCIENCE_EXTERNAL_EVAL_MARKERS.search(str(record.get("text") or "")):
            decisions.append({"id": source_id, "doi": candidate["doi"], "decision": "hold", "reason_code": "external_benchmark_or_evaluation_topic",
                              "jats_raw_file": access.get("raw_file"), "jats_raw_sha256": access.get("sha256")})
            persist()
            continue
        record["topic_bucket"] = topic
        record["candidate_metadata"] = {key: candidate.get(key) for key in (
            "doi", "title", "journal", "published_at", "article_type", "topic_bucket", "query_group",
            "metadata_license_hint", "metadata_license_is_not_article_rights_proof", "solr_metadata_raw_file", "solr_metadata_raw_sha256",
        )}
        try:
            from dataset_factory.document_data import _validate_source
            _validate_source(record)
        except (ImportError, ValueError, TypeError) as exc:
            decisions.append({"id": source_id, "doi": candidate["doi"], "decision": "source_contract_hold",
                              "reason_code": f"{type(exc).__name__}:{str(exc)[:120]}", "jats_raw_file": access.get("raw_file"),
                              "jats_raw_sha256": access.get("sha256")})
            persist()
            continue

        existing_topic = plos_topic_rows.get(source_id)
        if existing_topic and existing_topic.get("source_json_sha256"):
            continue
        SCIENCE_PLOS_PENDING_DOCS.mkdir(parents=True, exist_ok=True)
        if document_path.exists():
            source_bytes = document_path.read_bytes()
        else:
            write_json(document_path, record)
            source_bytes = document_path.read_bytes()
        source_json_sha = sha(source_bytes)
        topic_row = {
            "id": source_id, "decision": "verified_proof_pending_adapter", "topic": topic,
            "reason_codes": ["owner_source_adapter_pending"], "title": record.get("title"), "doi": record.get("doi"),
            "pmcid": record.get("pmcid"), "language": record.get("language"), "license_id": record["license"]["id"],
            "license_url": record["license"]["url"], "source_json": raw_rel(document_path), "source_json_sha256": source_json_sha,
            "source_text_sha256": record.get("content_sha256"), "raw_jats_sha256": record.get("raw_jats_sha256"),
            "basis": "same DOI/title verified between PLOS Solr metadata and JATS front matter; item-text rights verified from all governing article-meta permissions/license terms; topic checked against native full text",
            "metadata_copyright_is_not_rights_proof": True,
        }
        topic_hash = plos_topic_review_row_sha256(topic_row)
        identity_row.update({
            "decision": "verified_proof_pending_adapter", "reason_codes": [],
            "raw_source_json_sha256": source_json_sha,
            "fulltext_raw_file": raw_rel(raw_path), "fulltext_raw_sha256": access.get("sha256"),
            "source_title": record.get("title"), "source_language": record.get("language"),
            "source_json_license": record.get("license"), "topic": topic,
            "topic_review_record_sha256": topic_hash,
        })
        identity_row["identity_scope_schema_version"] = "plos-identity-scope-v1"
        identity_row["identity_scope_sha256"] = plos_identity_scope_sha256(identity_row)
        identity_row["topic_review_record_sha256"] = topic_hash
        plos_topic_rows[source_id] = topic_row
        plos_identity_rows[source_id] = identity_row
        manifest_entry = {
            "id": source_id, "doi": record.get("doi"), "pmcid": record.get("pmcid"), "title": record.get("title"),
            "language": record.get("language"), "topic": topic, "license_id": record["license"]["id"],
            "license_url": record["license"]["url"], "source_json": raw_rel(document_path),
            "source_json_sha256": source_json_sha, "source_text_sha256": record.get("content_sha256"),
            "jats_raw_file": raw_rel(raw_path), "jats_raw_sha256": access.get("sha256"),
            "authorable_status": "proof_verified_owner_adapter_pending",
        }
        manifest_by_id[source_id] = manifest_entry
        plos_records.append(manifest_entry)
        topic_counts[topic] = int(topic_counts.get(topic, 0)) + 1
        decisions.append({"id": source_id, "doi": candidate["doi"], "decision": "verified_proof_pending_adapter",
                          "topic": topic, "language": record.get("language"), "source_json_sha256": source_json_sha,
                          "jats_raw_sha256": access.get("sha256"), "identity_scope_sha256": identity_row["identity_scope_sha256"]})
        save_manifest(SCIENCE_PLOS_MANIFEST, plos_records)
        write_json(SCIENCE_PLOS_TOPIC_REVIEW, {
            "schema_version": "science-plos-topic-review-v1", "generated_at": utc_now(),
            "counts": {
                "proof_verified_adapter_pending": sum(row.get("decision") == "verified_proof_pending_adapter" for row in plos_topic_rows.values()),
                "held_not_authorable": sum(row.get("decision") != "verified_proof_pending_adapter" for row in plos_topic_rows.values()),
                "by_topic": dict(Counter(row.get("topic") for row in plos_topic_rows.values() if row.get("decision") == "verified_proof_pending_adapter")),
            },
            "sources": [plos_topic_rows[key] for key in sorted(plos_topic_rows)],
        })
        previous_sha, previous_path = archive_plos_identity_audit(science)
        audit_doc = {
            "schema_version": "plos-source-identity-audit-v1", "identity_scope_schema_version": "plos-identity-scope-v1",
            "generated_at": utc_now(), "supersedes_previous_audit_sha256": previous_sha,
            "supersedes_previous_audit_revision_file": previous_path, "source_manifest": raw_rel(SCIENCE_PLOS_MANIFEST),
            "counts": {"proof_verified_adapter_pending": sum(plos_topic_rows.get(source_id, {}).get("decision") == "verified_proof_pending_adapter" for source_id in plos_identity_rows),
                       "identity_verified_held_for_topic_scope": sum(
                           plos_topic_rows.get(source_id, {}).get("decision") != "verified_proof_pending_adapter"
                           and bool(row.get("same_item_rights_verified"))
                           for source_id, row in plos_identity_rows.items()
                       ),
                       "same_item_rights_verified": sum(bool(row.get("same_item_rights_verified")) for row in plos_identity_rows.values())},
            "sources": plos_identity_rows,
        }
        write_json(SCIENCE_PLOS_IDENTITY_AUDIT, audit_doc)
        status_counts["proof_verified_adapter_pending_count"] = plos_authorable_manifest_count(plos_records)
        status_counts["held_topic_scope_source_count"] = len(plos_records) - plos_authorable_manifest_count(plos_records)
        status_counts["proof_verified_topic_counts"] = dict(Counter(
            row.get("topic") for row in plos_records
            if row.get("authorable_status") == "proof_verified_owner_adapter_pending"
        ))
        status_counts["proof_verified_native_language_counts"] = dict(Counter(row.get("language") for row in plos_records))
        status_counts["identity_audit_file"] = raw_rel(SCIENCE_PLOS_IDENTITY_AUDIT)
        status_counts["identity_audit_sha256"] = sha(SCIENCE_PLOS_IDENTITY_AUDIT.read_bytes())
        status_counts["authoring_adapter_ready"] = False
        persist()
        new_pending_ids.append(source_id)

    update_plos_language_and_topic_sidecars(http, science, persist, policies)
    for source_id in new_pending_ids:
        entry = manifest_by_id.get(source_id, {})
        doc = read_json(ROOT / str(entry.get("source_json") or ""), {})
        print(f"PROOF_VERIFIED_ADAPTER_PENDING science PLOS {source_id} {doc.get('language')} {doc.get('title','')[:100]}", flush=True)
    science["plos_proof_verified_adapter_pending_count"] = plos_authorable_manifest_count(plos_records)
    science["plos_candidate_decisions_count"] = len(decisions)
    science["plos_authoring_adapter_ready"] = False
    science["remaining_science_authorable_shortfall_count"] = max(0, TARGET_SCIENCE - int(science.get("eligible_source_document_count", 0)))
    science["phase"] = "plos_proof_pending_owner_adapter" if plos_records else "plos_bounded_candidate_screen_complete"
    persist()


class GOVHtml(HTMLParser):
    BLOCKS = {"p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "td", "th", "blockquote", "pre", "br"}
    SKIP = {"script", "style", "noscript", "svg", "figure", "picture", "img", "iframe", "video", "audio"}
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.lines: list[str] = []
        self.current: list[str] = []
        self.skip_depth = 0
        self.skip_tags: list[str] = []
        self.list_stack: list[tuple[str, int]] = []
        self.table_cell = False
        self.link_stack: list[tuple[str, str]] = []

    def flush(self) -> None:
        value = re.sub(r"\s+", " ", "".join(self.current)).strip()
        if value:
            self.lines.append(value)
        self.current = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attrs_map = dict(attrs)
        if self.skip_depth:
            if tag in self.SKIP:
                self.skip_depth += 1
                self.skip_tags.append(tag)
            return
        if tag in self.SKIP:
            self.flush()
            self.skip_depth = 1
            self.skip_tags.append(tag)
            return
        if tag in {"ul", "ol"}:
            self.flush()
            self.list_stack.append((tag, 0))
        elif tag == "li":
            self.flush()
            if self.list_stack:
                typ, ordinal = self.list_stack[-1]
                self.list_stack[-1] = (typ, ordinal + 1)
                self.current.append(f"{ordinal + 1}. " if typ == "ol" else "• ")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.flush()
            self.current.append("#" * int(tag[1]) + " ")
        elif tag == "a":
            href = attrs_map.get("href") or ""
            self.link_stack.append((href, ""))
        elif tag in {"td", "th"}:
            if self.current and not self.current[-1].endswith(" "):
                self.current.append(" | ")
        elif tag == "br":
            self.flush()
        elif tag in {"p", "tr", "blockquote", "pre"}:
            self.flush()

    def handle_endtag(self, tag: str) -> None:
        if self.skip_depth:
            if self.skip_tags and tag == self.skip_tags[-1]:
                self.skip_tags.pop()
                self.skip_depth -= 1
            return
        if tag == "a" and self.link_stack:
            href, label = self.link_stack.pop()
            if href.startswith("http") and href not in label and label:
                self.current.append(f" ({href})")
        elif tag in {"li", "p", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "blockquote", "pre"}:
            self.flush()
        elif tag in {"td", "th"}:
            self.current.append(" | ")
        elif tag in {"ul", "ol"}:
            self.flush()
            if self.list_stack:
                self.list_stack.pop()

    def handle_data(self, data: str) -> None:
        if self.skip_depth:
            return
        self.current.append(data)
        if self.link_stack:
            href, label = self.link_stack[-1]
            self.link_stack[-1] = (href, label + data)

    def result(self) -> str:
        self.flush()
        return "\n\n".join(line for line in self.lines if line).strip()


def gov_text_for_item(item: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    details = item.get("details") or {}
    parts = details.get("parts") or []
    candidates = []
    if details.get("body"):
        candidates.append({"title": item.get("title") or "", "body": details["body"], "base_path": item.get("base_path")})
    for part in parts:
        if isinstance(part, dict) and part.get("body"):
            candidates.append(part)
    sections: list[dict[str, Any]] = []
    for index, part in enumerate(candidates, 1):
        parser = GOVHtml()
        parser.feed(str(part.get("body", "")))
        text_value = parser.result()
        if text_value:
            heading = str(part.get("title") or part.get("heading") or f"Part {index}").strip()
            sections.append({"section_path": [heading], "section_type": "content_api_part", "text": text_value, "base_path": part.get("base_path")})
    full_text = "\n\n".join(f"## {section['section_path'][0]}\n\n{section['text']}" for section in sections)
    return full_text.strip(), sections


def gov_license_exception(text_value: str) -> str | None:
    patterns = (
        (r"(?:©|copyright\s+)(?!\s*crown\b)[^\n]{0,100}", "non_crown_copyright_notice"),
        (r"reproduced with permission", "reproduced_with_permission"),
        (r"used with permission", "used_with_permission"),
        (r"third[- ]party (?:content|material|rights)", "explicit_third_party_notice"),
    )
    for pattern, label in patterns:
        match = re.search(pattern, text_value, flags=re.I)
        if match:
            return label
    return None


def gov_api_path_for(url_or_path: str) -> str:
    parts = urlsplit(url_or_path)
    path = parts.path.strip("/")
    if path.startswith("api/content/"):
        return "https://www.gov.uk/" + path
    return GOV_API_ROOT + "/" + quote(path, safe="/-")


def discover_gov_paths(http: SerialHttp, public: dict[str, Any], persist) -> list[str]:
    """Use the documented GOV.UK Search API for metadata-only path discovery."""
    robots = (RAW_PUBLIC / "preflight/govuk-robots.txt").read_text(errors="replace")
    if not robots_allows(robots, GOV_SEARCH_API, http.user_agent):
        public["search_discovery"] = {
            "decision": "blocked_by_robots",
            "route": GOV_SEARCH_API,
            "documentation_url": GOV_SEARCH_API_DOCS,
        }
        persist()
        return []
    query_specs = (
        ("transactions", {"filter_content_store_document_type": "transaction"}),
        ("guides", {"filter_format": "guide"}),
        ("detailed_guides", {"filter_format": "detailed_guide"}),
        ("travel_advice", {"filter_format": "travel_advice"}),
    )
    paths: list[str] = []
    seen: set[str] = set()
    observations: list[dict[str, Any]] = []
    for label, filters in query_specs:
        params = {
            **filters,
            "count": "400",
            "fields": "link,title,format,content_store_document_type",
        }
        raw = RAW_PUBLIC / "search" / f"{label}.json"
        body, access = http.get(GOV_SEARCH_API, raw, "public", params=params, max_bytes=8_000_000)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            public["search_discovery"] = {
                "decision": "invalid_search_api_json",
                "documentation_url": GOV_SEARCH_API_DOCS,
                "request": access,
            }
            persist()
            return []
        results = payload.get("results") or []
        added = 0
        for row in results:
            path = str(row.get("link") or "").strip()
            if not path.startswith("/") or path.startswith("//") or "?" in path or "#" in path:
                continue
            if path in seen:
                continue
            seen.add(path)
            paths.append(path.strip("/"))
            added += 1
        observations.append({
            "query_group": label,
            "filters": filters,
            "total_matches": payload.get("total"),
            "returned_metadata_rows": len(results),
            "new_candidate_paths": added,
            "response_sha256": access["sha256"],
            "request": access,
        })
        public["search_discovery"] = {
            "decision": "metadata_only_paths_discovered",
            "documentation_url": GOV_SEARCH_API_DOCS,
            "route": GOV_SEARCH_API,
            "robots_allowed": True,
            "item_bodies_fetched_from_search_api": False,
            "queries_completed": observations,
            "unique_candidate_paths": len(paths),
            "discovered_at": utc_now(),
        }
        persist()
    return paths


def gov_candidates(http: SerialHttp, public: dict[str, Any], persist, *, early: bool = False) -> list[dict[str, Any]]:
    prior_records = existing_records(PUBLIC_MANIFEST)
    prior_keys: set[str] = set()
    for item in prior_records:
        prior_keys.add(norm_url(item.get("url", "")))
        prior_keys.update(str(k).casefold() for k in item.get("relation_keys", []))
    local_seen: set[str] = set()
    accepted_works: set[str] = set(str(doc.get("source_work_id") or doc.get("id")) for doc in prior_records if doc.get("language", "en") == "en")
    all_records = prior_records
    work_docs: list[dict[str, Any]] = []
    candidate_paths = list(GOV_PATHS)
    if not early and len(accepted_works) < TARGET_PUBLIC:
        for path in discover_gov_paths(http, public, persist):
            if path not in candidate_paths:
                candidate_paths.append(path)
    source_languages: dict[str, int] = {}
    advertised_languages: dict[str, int] = {}
    retrieved_languages: dict[str, int] = {}
    for existing in all_records:
        if existing.get("original_translation") == "human_translation":
            language = str(existing.get("language") or "und")
            retrieved_languages[language] = retrieved_languages.get(language, 0) + 1
            continue
        language = str(existing.get("language") or "und")
        source_languages[language] = source_languages.get(language, 0) + 1
        for translation_language in {str(t.get("language")) for t in (existing.get("translations") or []) if t.get("language")}:
            advertised_languages[translation_language] = advertised_languages.get(translation_language, 0) + 1
    public["language_inventory"] = {
        "scope": "available translations advertised by accepted GOV.UK Content API items; page-specific availability, not an all-GOV.UK language claim",
        "accepted_source_languages": source_languages,
        "advertised_translation_languages": advertised_languages,
        "retrieved_native_translation_languages": retrieved_languages,
        "native_work_count": len(accepted_works),
    }
    if advertised_languages and set(advertised_languages) <= {"en"} and len(accepted_works) >= TARGET_PUBLIC:
        public["language_inventory"]["note"] = "No Welsh or other non-English translation was advertised by the accepted works; all accepted native editions are English."
    for path in candidate_paths:
        if len(accepted_works) >= (5 if early else TARGET_PUBLIC):
            break
        api_url = gov_api_path_for(path)
        path_name = urlsplit(api_url).path.rsplit("/", 1)[-1]
        raw_path = RAW_PUBLIC / "content" / f"{re.sub(r'[^a-zA-Z0-9_-]+', '_', path)}.json"
        try:
            body, access = http.get(api_url, raw_path, "public", max_bytes=3_500_000)
        except RuntimeError as exc:
            if "returned 404" in str(exc) or "HTTP 404" in str(exc):
                public["rejected_counts"]["content_not_found"] = int(public["rejected_counts"].get("content_not_found", 0)) + 1
                continue
            raise
        try:
            item = json.loads(body)
        except json.JSONDecodeError:
            public["rejected_counts"]["invalid_content_api_json"] = int(public["rejected_counts"].get("invalid_content_api_json", 0)) + 1
            continue
        public.setdefault("discovered_items", []).append({
            "base_path": item.get("base_path"), "content_id": item.get("content_id"), "locale": item.get("locale"),
            "document_type": item.get("document_type"), "title": item.get("title"), "http_status": access["http_status"],
        })
        if item.get("withdrawn_notice") or item.get("withdrawn"):
            public["rejected_counts"]["withdrawn"] = int(public["rejected_counts"].get("withdrawn", 0)) + 1
            persist()
            continue
        if item.get("phase") != "live":
            public["rejected_counts"]["non_live_phase"] = int(public["rejected_counts"].get("non_live_phase", 0)) + 1
            persist()
            continue
        # Only the declared GOV.UK Content API route carries our automation
        # authority; the API article must also be an OGL-covered text item.
        text_value, sections = gov_text_for_item(item)
        if len(text_value) < MIN_PUBLIC_CHARS:
            public["rejected_counts"]["text_too_short_or_no_body"] = int(public["rejected_counts"].get("text_too_short_or_no_body", 0)) + 1
            persist()
            continue
        if len(text_value) > MAX_SOURCE_CHARS:
            public["held"]["source_exceeds_100k_chars_hold"] = int(public["held"].get("source_exceeds_100k_chars_hold", 0)) + 1
            persist()
            continue
        if item.get("locale") not in {"en", "cy"}:
            public["rejected_counts"]["locale_outside_checked_languages"] = int(public["rejected_counts"].get("locale_outside_checked_languages", 0)) + 1
            persist()
            continue
        exception = gov_license_exception(text_value)
        if exception:
            public["held"][exception] = int(public["held"].get(exception, 0)) + 1
            public.setdefault("candidate_decisions", []).append({"base_path": item.get("base_path"), "decision": exception})
            persist()
            continue
        content_id = str(item.get("content_id") or "")
        base_path = str(item.get("base_path") or path).strip("/")
        relation_keys = [f"govuk-content-id:{content_id}", f"govuk-base-path:/{base_path}"]
        document_url = next((t.get("web_url") for t in (item.get("links", {}).get("available_translations") or []) if t.get("locale") == item.get("locale")), None)
        document_url = document_url or f"https://www.gov.uk/{base_path}"
        if norm_url(document_url) in prior_keys or any(rel.casefold() in prior_keys for rel in relation_keys):
            public["rejected_counts"]["prior_work_overlap"] = int(public["rejected_counts"].get("prior_work_overlap", 0)) + 1
            public.setdefault("candidate_decisions", []).append({"base_path": base_path, "decision": "prior_work_overlap"})
            persist()
            continue
        if base_path.casefold() in local_seen:
            public["rejected_counts"]["same_run_duplicate"] = int(public["rejected_counts"].get("same_run_duplicate", 0)) + 1
            continue
        local_seen.add(base_path.casefold())
        works_id = f"govuk:{base_path.casefold()}"
        if works_id in accepted_works:
            continue
        translations = item.get("links", {}).get("available_translations") or []
        translation_inventory = [{"language": t.get("locale"), "url": t.get("web_url"), "api_url": t.get("api_url"), "content_id": t.get("content_id")} for t in translations]
        source_language = str(item.get("locale") or "und")
        public["language_inventory"]["accepted_source_languages"][source_language] = int(public["language_inventory"]["accepted_source_languages"].get(source_language, 0)) + 1
        for translated_language in sorted({str(t.get("locale")) for t in translations if t.get("locale")}):
            public["language_inventory"]["advertised_translation_languages"][translated_language] = int(public["language_inventory"]["advertised_translation_languages"].get(translated_language, 0)) + 1
        doc_id = "govuk-" + re.sub(r"[^a-z0-9]+", "-", base_path.lower()).strip("-")
        published = item.get("first_published_at") or item.get("public_timestamp")
        updated = item.get("public_updated_at") or item.get("updated_at")
        doc = {
            "id": doc_id,
            "source_family": "public_information",
            "publisher": "GOV.UK",
            "title": item.get("title") or path_name,
            "language": item.get("locale") or "und",
            "original_translation": "source_native_edition",
            "url": document_url,
            "base_path": "/" + base_path,
            "source_work_id": works_id,
            "content_id": content_id or None,
            "document_type": item.get("document_type"),
            "published_at": published,
            "source_updated_at": updated,
            "retrieved_at": access["retrieved_at"],
            "revision": f"content-id:{content_id};updated:{updated};response-sha256:{access['sha256']}",
            "content_sha256": sha(text_value),
            "source_response_sha256": access["sha256"],
            "text": text_value,
            "sections": sections,
            "attribution": ["© Crown copyright; GOV.UK"],
            "license": {"id": "OGL-3.0", "url": "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/", "share_alike": False},
            "rights_evidence": {
                "checked_at": utc_now(),
                "policy_url": GOV_REUSE_POLICY,
                "policy_sha256": public.get("preflight", {}).get("reuse_policy_sha256"),
                "route": "official GOV.UK Content API",
                "item_license_url": "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
                "item_rights_basis": "GOV.UK published content reused under OGL terms; text extracted from live Content API response",
                "excluded_exceptions": ["images, figures, logos, audio/video, and embedded media omitted", "non-Crown or third-party copyright notices held for manual review"],
                "api_response_sha256": access["sha256"],
            },
            "extraction": {
                "method": "HTML block extraction from official Content API body and guide parts",
                "sections_preserved": True,
                "ordered_lists_preserved": True,
                "tables_preserved_as_text_rows": True,
                "media_omitted": True,
                "changes": "HTML wrappers and non-text media omitted; headings, paragraph and list boundaries retained; table rows flattened with cell separators.",
            },
            "translations": translation_inventory,
            "relation_keys": relation_keys,
            "work_independence_key": works_id,
        }
        path_json = DOCS / f"{doc_id}-{doc['language']}.json"
        write_json(path_json, doc)
        doc["document_file"] = raw_rel(path_json)
        all_records.append(doc)
        save_manifest(PUBLIC_MANIFEST, all_records)
        work_docs.append(doc)
        accepted_works.add(works_id)
        public["accepted_work_count"] = len(accepted_works)
        public["accepted_document_count"] = len(all_records)
        public.setdefault("accepted_sources", []).append({"id": doc_id, "language": doc["language"], "title": doc["title"], "url": document_url, "document_file": doc["document_file"], "translation_languages_available": sorted({str(t.get('language')) for t in translations if t.get('language')})})
        persist()
        print(f"SOURCE_READY public {doc_id} {doc['language']} {doc['title'][:110]}", flush=True)
        # Translations are related views of the same work and never increase
        # the independent-work quota. Fetch only language versions advertised
        # by this same official Content API item, currently English and Welsh.
        for translation in translations:
            lang = translation.get("locale")
            translation_api = translation.get("api_url")
            if lang not in {"cy"} or not translation_api or lang == item.get("locale"):
                continue
            if len(accepted_works) >= (5 if early else TARGET_PUBLIC) and len(work_docs) >= (5 if early else TARGET_PUBLIC):
                break
            tr_path = RAW_PUBLIC / "translations" / f"{doc_id}-{lang}.json"
            try:
                tr_bytes, tr_access = http.get(translation_api, tr_path, "public", max_bytes=3_500_000)
            except RuntimeError:
                public.setdefault("translation_results", []).append({"work": works_id, "language": lang, "decision": "blocked_or_missing; no fallback"})
                persist()
                continue
            try:
                tr_item = json.loads(tr_bytes)
            except json.JSONDecodeError:
                public.setdefault("translation_results", []).append({"work": works_id, "language": lang, "decision": "invalid_json"})
                persist()
                continue
            tr_text, tr_sections = gov_text_for_item(tr_item)
            tr_exception = gov_license_exception(tr_text)
            if tr_item.get("locale") != lang or len(tr_text) < MIN_PUBLIC_CHARS or len(tr_text) > MAX_SOURCE_CHARS or tr_exception:
                decision = "rights_or_text_filter_failed" if tr_exception or len(tr_text) > MAX_SOURCE_CHARS else "translation_body_unavailable_or_short"
                public.setdefault("translation_results", []).append({"work": works_id, "language": lang, "decision": decision})
                persist()
                continue
            tr_content_id = str(tr_item.get("content_id") or "")
            tr_url = next((t.get("web_url") for t in (tr_item.get("links", {}).get("available_translations") or []) if t.get("locale") == lang), translation.get("web_url"))
            tr_doc = dict(doc)
            tr_doc.update({
                "id": doc_id + "-" + lang,
                "language": lang,
                "original_translation": "human_translation",
                "original_url": doc["url"],
                "url": tr_url or document_url,
                "content_id": tr_content_id or None,
                "published_at": tr_item.get("first_published_at") or tr_item.get("public_timestamp"),
                "source_updated_at": tr_item.get("public_updated_at") or tr_item.get("updated_at"),
                "retrieved_at": tr_access["retrieved_at"],
                "revision": f"content-id:{tr_content_id};updated:{tr_item.get('public_updated_at')};response-sha256:{tr_access['sha256']}",
                "content_sha256": sha(tr_text),
                "source_response_sha256": tr_access["sha256"],
                "text": tr_text,
                "sections": tr_sections,
                "translations": translation_inventory,
                "rights_evidence": {**doc["rights_evidence"], "api_response_sha256": tr_access["sha256"], "item_rights_basis": "Welsh edition returned by the official Content API and reused under the same OGL policy"},
                "relation_keys": relation_keys + [f"govuk-translation:{lang}:{tr_content_id}"],
            })
            tr_file = DOCS / f"{doc_id}-{lang}.json"
            write_json(tr_file, tr_doc)
            tr_doc["document_file"] = raw_rel(tr_file)
            all_records.append(tr_doc)
            save_manifest(PUBLIC_MANIFEST, all_records)
            public["accepted_document_count"] = len(all_records)
            public.setdefault("translation_results", []).append({"work": works_id, "language": lang, "decision": "included", "document_id": tr_doc["id"], "translation_sha256": tr_access["sha256"]})
            public.setdefault("accepted_languages", {}).setdefault(lang, 0)
            public["accepted_languages"][lang] += 1
            public["language_inventory"]["retrieved_native_translation_languages"][lang] = int(public["language_inventory"]["retrieved_native_translation_languages"].get(lang, 0)) + 1
            persist()
            print(f"SOURCE_READY public {tr_doc['id']} {lang} {tr_doc['title'][:110]}", flush=True)
    return work_docs


def live_preflight(http: SerialHttp, science: dict[str, Any], public: dict[str, Any], persist) -> None:
    if science.get("preflight", {}).get("completed") and public.get("preflight", {}).get("completed"):
        return
    checks: list[tuple[str, str, Path, str]] = [
        ("pmc_oai_robots", PMC_ROBOTS, RAW_SCIENCE / "preflight/pmc-robots.txt", "science"),
        ("ncbi_eutils_robots", NCBI_ROBOTS, RAW_SCIENCE / "preflight/eutils-robots.txt", "science"),
        ("pmc_oai_policy", PMC_OAI_POLICY, RAW_SCIENCE / "preflight/oai-policy.html", "science"),
        ("pmc_data_policy", PMC_DATA_POLICY, RAW_SCIENCE / "preflight/dataset-policy.html", "science"),
        ("pmc_language_guide", PMC_LANGUAGE_GUIDE, RAW_SCIENCE / "preflight/language-guide.html", "science"),
        ("pmc_oai_identify", PMC_OAI + "?verb=Identify", RAW_SCIENCE / "preflight/oai-identify.xml", "science"),
        ("govuk_robots", GOV_ROBOTS, RAW_PUBLIC / "preflight/govuk-robots.txt", "public"),
        ("govuk_content_api_docs", GOV_CONTENT_API_DOCS, RAW_PUBLIC / "preflight/content-api-docs.html", "public"),
        ("govuk_reuse_policy", GOV_REUSE_POLICY, RAW_PUBLIC / "preflight/reuse-policy.html", "public"),
        ("govuk_reuse_policy_api", GOV_REUSE_CONTENT, RAW_PUBLIC / "preflight/reuse-policy-api.json", "public"),
    ]
    results: dict[str, Any] = {}
    for check_id, url, raw_path, owner in checks:
        body, row = http.get(url, raw_path, owner, max_bytes=3_000_000)
        if row["http_status"] != 200:
            raise RuntimeError(f"live official preflight failed for {check_id}")
        results[check_id] = {**row, "url": url}
    pmc_robots = (RAW_SCIENCE / "preflight/pmc-robots.txt").read_text(errors="replace")
    eutils_robots = (RAW_SCIENCE / "preflight/eutils-robots.txt").read_text(errors="replace")
    gov_robots = (RAW_PUBLIC / "preflight/govuk-robots.txt").read_text(errors="replace")
    routes = {
        "pmc_oai": PMC_OAI + "?verb=Identify",
        "eutils_esearch": EUTILS + "esearch.fcgi?db=pmc&term=ecology[la]&retmode=json",
        "gov_content_api": GOV_REUSE_CONTENT,
    }
    robot_checks = {
        "pmc_oai": robots_allows(pmc_robots, routes["pmc_oai"], http.user_agent),
        "eutils_esearch": robots_allows(eutils_robots, routes["eutils_esearch"], http.user_agent),
        "gov_content_api": robots_allows(gov_robots, routes["gov_content_api"], http.user_agent),
    }
    if not robot_checks["pmc_oai"] or not robot_checks["gov_content_api"]:
        science["phase"] = "blocked_by_robots"
        science["stop_reason"] = "official robots.txt disallows a selected PMC OAI or GOV.UK Content API route; no alternate route attempted"
        public["phase"] = "blocked_by_robots"
        public["stop_reason"] = "official robots.txt disallows a selected PMC OAI or GOV.UK Content API route; no alternate route attempted"
        persist()
        raise RuntimeError("robots.txt denied a source route; stopped")
    pmc_policy_text = (RAW_SCIENCE / "preflight/oai-policy.html").read_text(errors="replace")
    data_policy_text = (RAW_SCIENCE / "preflight/dataset-policy.html").read_text(errors="replace")
    gov_policy_text = (RAW_PUBLIC / "preflight/reuse-policy.html").read_text(errors="replace")
    gov_policy_api = json.loads((RAW_PUBLIC / "preflight/reuse-policy-api.json").read_text())
    if "only services that may be used for automated retrieval" not in re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", pmc_policy_text))).lower():
        raise RuntimeError("PMC automated-access authority could not be verified from the live OAI policy")
    if "cc by" not in data_policy_text.lower() or "cc0" not in data_policy_text.lower():
        raise RuntimeError("PMC item-license policy page did not expose CC-BY/CC0 terms")
    gov_body = (((gov_policy_api.get("details") or {}).get("body")) or "")
    if "Open Government Licence" not in gov_body:
        raise RuntimeError("GOV.UK OGL policy text could not be verified in its live Content API response")
    api_doc_text = (RAW_PUBLIC / "preflight/content-api-docs.html").read_text(errors="replace")
    if not api_doc_text:
        raise RuntimeError("GOV.UK Content API documentation returned empty content")
    now = datetime.now(timezone.utc)
    eastern = now.astimezone(ZoneInfo("America/New_York"))
    off_peak = eastern.weekday() >= 5 or not (5 <= eastern.hour < 21)
    preflight = {
        "checked_at": utc_now(),
        "checks": results,
        "robots": robot_checks,
        "disallowed_routes_not_used": [routes["eutils_esearch"]] if not robot_checks["eutils_esearch"] else [],
        "official_automation_authority": {
            "pmc": PMC_OAI_POLICY,
            "pmc_policy_sha256": results["pmc_oai_policy"]["sha256"],
            "pmc_dataset_policy_sha256": results["pmc_data_policy"]["sha256"],
            "eutils": "PMC OAI policy lists E-utilities among permitted services, but its live robots.txt denies `/`; this acquisition does not use E-utilities.",
            "chosen_pmc_route": "PMC OAI-PMH only; the `/api/` route is allowed by PMC robots.txt",
            "govuk": GOV_CONTENT_API_DOCS,
            "govuk_reuse_policy": GOV_REUSE_POLICY,
            "govuk_policy_api_sha256": results["govuk_reuse_policy_api"]["sha256"],
        },
        "request_limits": {"pmc_oai_max_rps": 3, "actual_serial_spacing_seconds": REQUEST_SPACING_SECONDS, "peak_hours_eastern": "Monday-Friday 05:00-21:00 America/New_York", "checked_local_eastern": eastern.isoformat(), "off_peak_for_more_than_100_requests": off_peak},
        "user_agent_product": USER_AGENT_PRODUCT,
        "contact_configured": True,
        "max_new_network_bytes": MAX_NETWORK_BYTES,
        "max_disk_bytes": 1_000_000_000,
    }
    science["preflight"] = {**preflight, "completed": True}
    public["preflight"] = {
        **preflight,
        "completed": True,
        "reuse_policy_sha256": results["govuk_reuse_policy"]["sha256"],
    }
    persist()


def science_acquisition(http: SerialHttp, science: dict[str, Any], persist, *, initial_only: bool,
                        screen_offset: int = 0, screen_limit: int | None = 25,
                        date_window: str = "all") -> None:
    manifest_docs = existing_records(SCIENCE_MANIFEST)
    refresh_science_review(science, manifest_docs, persist)
    prior = load_prior_keys()
    already = {str(doc.get("pmcid", "")).upper() for doc in manifest_docs}
    target = 3 if initial_only else TARGET_SCIENCE
    blocked_pmids = set(already) | set(prior["pmcid"])
    blocked_dois = set(prior["doi"]) | {normalize_doi(str(doc.get("doi"))) for doc in manifest_docs if doc.get("doi")}
    pools = candidate_searches(
        http, science, persist, initial_only=initial_only, screen_offset=screen_offset,
        screen_limit=screen_limit, blocked_pmids=blocked_pmids, blocked_dois=blocked_dois,
        date_window=date_window,
    )
    native_groups = sorted(group for group in pools if group.startswith("native_"))
    plos_groups = sorted(group for group in pools if group.startswith("plos_"))
    other_groups = sorted(group for group in pools if group not in native_groups and group not in plos_groups)

    def interleave(groups: list[str]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for index in range(max((len(pools.get(group, [])) for group in groups), default=0)):
            for group in groups:
                values = pools.get(group, [])
                if index < len(values):
                    result.append(values[index])
        return result

    native_candidates = interleave(native_groups)
    other_candidates = interleave(plos_groups + other_groups)
    # Up to one third of the scientific set comes from native non-English
    # papers when OAI metadata shows that those full texts are available.
    nonenglish_quota = min(30, target)
    flattened = native_candidates[:nonenglish_quota] + other_candidates + native_candidates[nonenglish_quota:]
    seen_ids: set[str] = set()
    accepted_dois = set(prior["doi"])
    accepted_dois.update(normalize_doi(str(doc.get("doi"))) for doc in manifest_docs if doc.get("doi"))
    verified_topic_counts = science.get("verified_eligible_topic_counts", {})
    biological_count = int(verified_topic_counts.get("biological_science", 0)) if isinstance(verified_topic_counts, dict) else 0
    for candidate in flattened:
        pmcid = str(candidate["pmcid"])
        group = str(candidate.get("search_group") or candidate.get("candidate_set") or "pmc-open")
        key = pmcid.upper()
        if key in seen_ids or key in already:
            continue
        seen_ids.add(key)
        if science.get("accepted_work_count", 0) >= target:
            break
        if candidate.get("topic_bucket") == "biological_science" and biological_count >= SCIENCE_TOPIC_LIMITS["biological_science"]:
            science["rejected_counts"]["topic_limit_biological_science_20"] = int(science["rejected_counts"].get("topic_limit_biological_science_20", 0)) + 1
            continue
        candidate_doi = normalize_doi(str(candidate.get("doi") or ""))
        if candidate_doi and candidate_doi in accepted_dois:
            science["rejected_counts"]["same_work_doi_duplicate"] = int(science["rejected_counts"].get("same_work_doi_duplicate", 0)) + 1
            continue
        science["candidates_seen"] = int(science.get("candidates_seen", 0)) + 1
        accepted = add_science_source(http, science, persist, manifest_docs, prior, candidate, group)
        if accepted:
            already.add(key)
            if candidate_doi:
                accepted_dois.add(candidate_doi)
            verified_topic_counts = science.get("verified_eligible_topic_counts", {})
            if isinstance(verified_topic_counts, dict):
                biological_count = int(verified_topic_counts.get("biological_science", biological_count))
        if science.get("network_bytes_decoded", 0) >= MAX_NETWORK_BYTES:
            science["phase"] = "stopped_network_cap"
            science["stop_reason"] = "300 MB per-acquisition network cap reached"
            persist()
            break
    science["candidate_groups_attempted"] = native_groups + plos_groups + other_groups
    review = refresh_science_review(science, manifest_docs)
    if science.get("accepted_work_count", 0) >= target:
        science["phase"] = "target_reached" if not initial_only else "initial_cohort_ready"
    else:
        science["phase"] = "shortfall_after_bounded_candidate_pool" if not initial_only else "initial_cohort_ready_with_shortfall"
    # Eligibility, not merely a successful download, defines admitted science works.
    science["accepted_document_count"] = science.get("eligible_source_document_count", 0)
    science["accepted_native_non_english_fulltext_count"] = sum(1 for row in review["sources"] if row.get("decision") == "eligible" and row.get("language") not in {"eng", "en", "und"})
    science["accepted_native_non_english_languages"] = sorted({str(row.get("language")) for row in review["sources"] if row.get("decision") == "eligible" and row.get("language") not in {"eng", "en", "und"}})
    persist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-eight", action="store_true", help="Write up to 3 science and 5 public works, then stop so cohorts can start early")
    parser.add_argument("--public-only", action="store_true", help="Continue GOV.UK source acquisition without repeating PMC discovery")
    parser.add_argument("--science-only", action="store_true", help="Continue PMC acquisition without changing the completed GOV.UK cohort")
    parser.add_argument("--plos-only", action="store_true", help="Run the separately audited bounded PLOS path without PMC/GOV preflight or public-status writes")
    parser.add_argument("--plos-language-audit-only", action="store_true", help="Record language-policy/JATS-language inventory for existing PLOS proof records without retrieving article bodies")
    parser.add_argument("--plos-fulltext-limit", type=int, default=36, help="Maximum PLOS article bodies retrieved in this bounded batch")
    parser.add_argument("--science-screen-limit", type=int, default=25, help="Maximum new PMC metadata candidates screened before full-text retrieval")
    parser.add_argument("--science-screen-offset", type=int, help="Resume after this many new metadata candidates; defaults to the saved next offset")
    parser.add_argument("--science-date-window", choices=("all", "2025-2026", "2022-2024", "2019-2021", "2016-2018", "2013-2015", "2010-2012", "2007-2009", "2004-2006", "2001-2003"), default="all", help="Bound discovery to a PMC release-date window")
    parser.add_argument("--audit-science-identifiers", action="store_true", help="Verify frozen PMC source identifiers and licenses using OAI DC metadata only")
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    DOCS.mkdir(parents=True, exist_ok=True)
    RAW_SCIENCE.mkdir(parents=True, exist_ok=True)
    RAW_PUBLIC.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "sched_getaffinity"):
        available = sorted(os.sched_getaffinity(0))
        if len(available) > 2:
            os.sched_setaffinity(0, set(available[:2]))
    science_only_mode = args.science_only or args.audit_science_identifiers or args.plos_only or args.plos_language_audit_only
    science, public, persist = process_science_status(write_public=not science_only_mode)
    for state in ((science,) if science_only_mode else (science, public)):
        previous_reason = state.pop("stop_reason", None)
        if previous_reason:
            state.setdefault("previous_stopped_attempts", []).append({
                "at": state.get("updated_at"),
                "reason": previous_reason,
                "note": "Resumed a source-local acquisition after preserving the previous stop reason in status.",
            })
    try:
        http = SerialHttp(science, public, persist)
        if args.audit_science_identifiers:
            audit = science_identifier_audit(http, science, persist)
            print(json.dumps({"identity_audit": raw_rel(SCIENCE_IDENTIFIER_AUDIT), "counts": audit["counts"]}, ensure_ascii=False), flush=True)
            return 0
        if args.plos_language_audit_only:
            plos_preflight(http, science, persist)
            plos_records = existing_records(SCIENCE_PLOS_MANIFEST)
            journal_slugs = {
                slug for row in plos_records
                if (slug := plos_article_slug(str(row.get("journal") or ""), str(row.get("doi") or "")))
            }
            policies = plos_journal_language_policies(http, science, persist, journal_slugs)
            update_plos_language_and_topic_sidecars(http, science, persist, policies)
            science["phase"] = "plos_language_inventory_complete"
            science["finished_at"] = utc_now()
            persist()
            print(json.dumps({
                "phase": science.get("phase"),
                "plos_language_inventory": raw_rel(SCIENCE_PLOS_LANGUAGE_INVENTORY),
                "plos_language_inventory_sha256": sha(SCIENCE_PLOS_LANGUAGE_INVENTORY.read_bytes()),
                "plos_proof_verified_adapter_pending_count": science.get("plos_proof_verified_adapter_pending_count", 0),
                "public_status_written": False,
            }, ensure_ascii=False), flush=True)
            return 0
        if args.plos_only:
            science["phase"] = "acquiring_plos_bounded_batch"
            persist()
            science_plos_acquisition(http, science, persist, limit=min(36, max(1, args.plos_fulltext_limit)))
            science["finished_at"] = utc_now()
            persist()
            print(json.dumps({
                "phase": science.get("phase"),
                "pmc_authorable_work_count": science.get("accepted_work_count", 0),
                "plos_proof_verified_adapter_pending_count": science.get("plos_proof_verified_adapter_pending_count", 0),
                "plos_manifest": raw_rel(SCIENCE_PLOS_MANIFEST),
                "plos_identity_audit": raw_rel(SCIENCE_PLOS_IDENTITY_AUDIT),
                "science_status": raw_rel(SCIENCE_STATUS),
                "public_status_written": False,
            }, ensure_ascii=False), flush=True)
            return 0
        live_preflight(http, science, public, persist)
        if not args.public_only:
            science["phase"] = "acquiring_initial_sources" if args.first_eight else "acquiring"
        if not args.science_only:
            public["phase"] = "acquiring_initial_sources" if args.first_eight else "acquiring"
        persist()
        if not args.public_only:
            screen_offset = args.science_screen_offset if args.science_screen_offset is not None else int(science.get("candidate_screening_next_offset", 0))
            science_acquisition(http, science, persist, initial_only=args.first_eight,
                                screen_offset=max(0, screen_offset), screen_limit=max(1, args.science_screen_limit),
                                date_window=args.science_date_window)
        if not args.science_only:
            public_work_count = public.get("accepted_work_count", 0)
            if args.first_eight:
                gov_candidates(http, public, persist, early=True)
                public["phase"] = "initial_cohort_ready"
            else:
                gov_candidates(http, public, persist, early=False)
                public_work_count = public.get("accepted_work_count", 0)
                if public_work_count >= TARGET_PUBLIC:
                    public["phase"] = "target_reached"
                else:
                    public["phase"] = "shortfall_after_bounded_candidate_pool"
        if not args.public_only:
            science["finished_at"] = utc_now()
        if not args.science_only:
            public["finished_at"] = utc_now()
        persist()
        print(json.dumps({
            "phase": "initial_cohort_ready" if args.first_eight else "acquisition_finished",
            "science_work_count": science.get("accepted_work_count", 0),
            "public_work_count": public.get("accepted_work_count", 0),
            "science_ids": [doc.get("id") for doc in existing_records(SCIENCE_MANIFEST)],
            "public_ids": [doc.get("id") for doc in existing_records(PUBLIC_MANIFEST)],
            "science_status": raw_rel(SCIENCE_STATUS),
            "public_status": raw_rel(PUBLIC_STATUS),
        }, ensure_ascii=False), flush=True)
        return 0
    except Exception as exc:
        message = str(exc)
        if not args.public_only:
            science["phase"] = "stopped_on_access_or_processing_error"
            science["stop_reason"] = message
        if not science_only_mode:
            public["phase"] = "stopped_on_access_or_processing_error"
            public["stop_reason"] = message
        if not args.public_only:
            science["finished_at"] = utc_now()
        if not science_only_mode:
            public["finished_at"] = utc_now()
        persist()
        print(f"ACQUISITION_STOPPED: {message}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
