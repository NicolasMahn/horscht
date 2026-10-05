import hashlib
import json

import pytest

from dataset_factory.document_data import assemble_row, freeze_groups, validate_document_row
from dataset_factory.probability_v2 import shuffle_choices, validate_v2_row
from dataset_factory.document_pilot import export
from dataset_factory.document_acquisition import BoundedClient, clean_article


def source(doc_id, text, relations):
    return {
        "id": doc_id, "source_family": "manual", "language": "en", "topic": "archives",
        "difficulty": "medium", "title": f"Guide {doc_id}", "url": f"https://example.test/{doc_id}",
        "revision": "rev-1", "retrieved_at": "2026-10-02T10:00:00Z",
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text,
        "attribution": ["Archive staff"],
        "license": {"id": "CC-BY-4.0", "url": "https://creativecommons.org/licenses/by/4.0/", "share_alike": False},
        "rights_evidence": {"checked_at": "2026-10-02", "policy_url": "https://example.test/rights",
                            "policy_sha256": "a" * 64, "item_license_url": "https://creativecommons.org/licenses/by/4.0/", "route": "public"},
        "relation_keys": relations,
    }


def draft(text, answer=0):
    quote = text[2:18]
    return {
        "question": "Which statement is supported?", "options": ["Supported claim", "Unsupported claim", "Cannot determine from the supplied document"],
        "answer_index": answer, "abstention_index": 2, "explanation": "The source states this directly.",
        "evidence_spans": [{"start": 2, "end": 2 + len(quote), "quote": quote}], "task_kind": "document_qa",
        "author_name": "luna", "author_model": "gpt-6-luna",
    }


def test_freeze_is_deterministic_transitive_and_keeps_relation_groups_together():
    docs = [source("a", "0123456789abcdef", ["book:a", "series:x"]),
            source("b", "abcdefghijklmnop", ["series:x", "volume:b"]),
            source("c", "qrstuvwxyzabcdef", ["independent:c"])]
    allocation = freeze_groups(docs)
    assert allocation == freeze_groups(list(reversed(docs)))
    assert allocation["a"] == allocation["b"]
    assert set(allocation.values()) <= {"train", "dev"}
    assert "dev" in allocation.values()


def test_hard_row_retains_source_and_passes_pilot_and_probability_contracts():
    doc = source("a", "0123456789abcdef", ["book:a"])
    allocation = {"a": "dev"}
    row = assemble_row(doc, draft(doc["text"]), allocation)
    assert row["context"] == doc["text"]
    assert row["split"] == "dev"
    assert row["abstention_choice_id"] == row["choice_ids"][2]
    assert validate_document_row(row, {"a": doc}, allocation) == []
    # The common v2 contract still reports only its known train/Luna envelope rules.
    assert set(validate_v2_row(row)) == {"split_is_not_train", "provenance_does_not_match_required_label"}


def test_sol_authoring_uses_matching_provenance_and_rejects_model_mismatch():
    doc = source("sol", "0123456789abcdef", ["book:sol"])
    d = draft(doc["text"])
    d["author_model"] = "gpt-6.1-sol"
    row = assemble_row(doc, d, {"sol": "train"})
    assert row["provenance"].startswith("gpt-6.1-sol-generated;")
    assert validate_document_row(row, {"sol": doc}, {"sol": "train"}) == []

    mismatch = {**row, "provenance": row["provenance"].replace("gpt-6.1-sol-generated", "luna-generated", 1)}
    assert "document_provenance_does_not_match_author_model" in validate_document_row(
        mismatch, {"sol": doc}, {"sol": "train"}
    )


def test_invalid_source_hash_evidence_and_allocation_are_rejected():
    doc = source("a", "0123456789abcdef", ["book:a"])
    allocation = {"a": "train"}
    row = assemble_row(doc, draft(doc["text"]), allocation)
    broken = dict(row)
    broken["context"] = "edited source"
    broken["document_metadata"] = {**row["document_metadata"], "evidence_spans": [{"start": 2, "end": 18, "quote": "wrong quote here"}]}
    errors = validate_document_row(broken, {"a": doc}, {"a": "dev"})
    assert "document_context_does_not_match_source_text" in errors
    assert "document_evidence_span_is_not_exact" in errors
    assert "document_split_does_not_match_frozen_allocation" in errors

    altered = {**doc, "content_sha256": "0" * 64}
    assert "document_source_metadata_is_invalid" in validate_document_row(row, {"a": altered}, allocation)
    altered_rights = {**doc, "license": {**doc["license"], "id": "CC-BY-SA-4.0", "share_alike": False}}
    with pytest.raises(ValueError, match="share-alike source"):
        assemble_row(altered_rights, draft(doc["text"]), allocation)


def test_idk_and_soft_distribution_math_are_preserved():
    doc = source("a", "0123456789abcdef", ["book:a"])
    d = draft(doc["text"])
    d.pop("answer_index")
    d["distribution_source_kind"] = "exact_derived"
    d["distribution_evidence"] = {"kind": "exact_derived", "probabilities": [
        {"numerator": 1, "denominator": 2}, {"numerator": 1, "denominator": 2}, {"numerator": 0, "denominator": 1}],
    }
    row = assemble_row(doc, d, {"a": "train"})
    assert row["target_distribution"] == [0.5, 0.5, 0.0]
    assert row["target_distribution"][2] == 0.0
    assert validate_document_row(row, {"a": doc}, {"a": "train"}) == []

    d["distribution_evidence"]["probabilities"][0]["numerator"] = 2
    with pytest.raises(ValueError):
        assemble_row(doc, d, {"a": "train"})


def test_related_document_rows_cannot_cross_frozen_split():
    docs = [source("a", "0123456789abcdef", ["work:x", "series:x"]),
            source("b", "abcdefghijklmnop", ["series:x"]),
            source("c", "qrstuvwxyzabcdef", ["other:c"])]
    allocation = {"a": "train", "b": "dev", "c": "train"}
    row = assemble_row(docs[0], draft(docs[0]["text"]), allocation)
    errors = validate_document_row(row, {d["id"]: d for d in docs}, allocation)
    assert "document_related_sources_span_splits" in errors


def test_permutation_moves_targets_evidence_and_idk_together():
    doc = source("a", "0123456789abcdef", ["book:a"])
    d = draft(doc["text"])
    d.pop("answer_index")
    d["distribution_source_kind"] = "exact_derived"
    d["distribution_evidence"] = {"kind": "exact_derived", "probabilities": [
        {"numerator": 1, "denominator": 3}, {"numerator": 2, "denominator": 3},
        {"numerator": 0, "denominator": 1}],
    }
    row = assemble_row(doc, d, {"a": "dev"})
    shuffled = shuffle_choices(row, [2, 0, 1])
    assert shuffled["id"] == row["id"]
    assert shuffled["target_distribution"] == [0, 1 / 3, 2 / 3]
    assert shuffled["abstention_choice_id"] == row["abstention_choice_id"] == shuffled["choice_ids"][0]
    assert dict(zip(row["choice_ids"], row["target_distribution"])) == dict(zip(shuffled["choice_ids"], shuffled["target_distribution"]))
    assert shuffled["distribution_evidence"]["probabilities"][0] == {"numerator": 0, "denominator": 1}
    assert validate_document_row(shuffled, {"a": doc}, {"a": "dev"}) == []


def test_integration_rows_need_separate_distant_evidence_and_known_generator_authoring():
    text = "A" * 700 + "B" * 700
    doc = source("long", text, ["work:long"])
    d = draft(text)
    d["task_kind"] = "integration"
    d["evidence_spans"] = [
        {"start": 0, "end": 20, "quote": text[:20]},
        {"start": 600, "end": 620, "quote": text[600:620]},
    ]
    row = assemble_row(doc, d, {"long": "train"})
    assert validate_document_row(row, {"long": doc}, {"long": "train"}) == []
    d["evidence_spans"][1] = {"start": 400, "end": 420, "quote": text[400:420]}
    row = assemble_row(doc, d, {"long": "train"})
    assert "document_integration_requires_distant_evidence" in validate_document_row(row, {"long": doc}, {"long": "train"})

    d["author_model"] = "unknown-model"
    row = assemble_row(doc, d, {"long": "train"})
    assert "document_author_metadata_is_missing" in validate_document_row(row, {"long": doc}, {"long": "train"})


def test_export_binds_source_freeze_and_review_to_exact_artifacts(tmp_path):
    doc = source("a", "0123456789abcdef", ["book:a"])
    d = draft(doc["text"])
    source_bytes = (json.dumps(doc) + "\n").encode()
    (tmp_path / "documents.jsonl").write_bytes(source_bytes)
    (tmp_path / "allocation.json").write_text(json.dumps({
        "documents_sha256": hashlib.sha256(source_bytes).hexdigest(), "allocation": {"a": "dev"}}))
    (tmp_path / "drafts").mkdir()
    draft_path = tmp_path / "drafts/a.jsonl"
    draft_path.write_text(json.dumps(d) + "\n")
    (tmp_path / "review.json").write_text(json.dumps({"reviewer_model": "gpt-6-luna", "decisions": [{
        "document_id": "a", "index": 0, "status": "accepted", "reviewed_draft_sha256": hashlib.sha256(
            json.dumps(d, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}]}))
    result = export(tmp_path)
    assert result["accepted_originals"] == 1
    assert result["splits"] == {"dev": 1}
    assert not (tmp_path / "train.jsonl").read_text()
    d["question"] = "A changed question"
    draft_path.write_text(json.dumps(d) + "\n")
    with pytest.raises(ValueError, match="changed after semantic review"):
        export(tmp_path)
    (tmp_path / "documents.jsonl").write_bytes(source_bytes + b"\n")
    with pytest.raises(ValueError, match="changed after allocation freeze"):
        export(tmp_path)


def test_extraction_excludes_third_party_media_and_quoted_blocks():
    text, removed = clean_article('<p>Le reportage.</p><figure><img src="x"><p>Photo: tiers.</p></figure>'
                                  '<blockquote><p>Texte reproduit du tiers.</p></blockquote><h2>Suite</h2><p>Notre prose.</p>')
    assert text == "Le reportage.\n\nSuite\n\nNotre prose."
    assert removed["media"] >= 1
    assert removed["quoted_blocks"] == 1


def test_acquisition_retry_cannot_reset_existing_cache_budget(tmp_path):
    (tmp_path / "previous-response").write_bytes(b"12345678")
    with pytest.raises(ValueError, match="Existing cache exceeds"):
        BoundedClient(tmp_path, maximum=4)
