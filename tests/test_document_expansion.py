import hashlib
import json

import pytest

from dataset_factory.document_expansion import (
    SEED, _build_original_row, _load_frozen, admit, audit_model_routing, audit_review_model_routing,
    correct, export_permutation_augmentations, freeze, quarantine, release,
)
from dataset_factory.document_data import (
    freeze_groups,
    validate_document_augmentation,
)
from dataset_factory.document_model_routing import load_model_routing_audit


def source(doc_id, text, relations, *, language="en"):
    return {
        "id": doc_id,
        "source_family": "reporting" if doc_id.startswith("report") else "encyclopedia",
        "publisher": "Example Archive",
        "language": language,
        "topic": "public records",
        "difficulty": "medium",
        "title": f"Document {doc_id}",
        "url": f"https://example.test/{doc_id}",
        "revision": "rev-1",
        "retrieved_at": "2026-10-02T10:00:00Z",
        "content_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text": text,
        "attribution": ["Example Archive"],
        "license": {
            "id": "CC-BY-4.0",
            "url": "https://creativecommons.org/licenses/by/4.0/",
            "share_alike": False,
        },
        "rights_evidence": {
            "checked_at": "2026-10-02",
            "policy_url": "https://example.test/rights",
            "policy_sha256": "a" * 64,
            "item_license_url": "https://creativecommons.org/licenses/by/4.0/",
            "route": "public archive",
        },
        "relation_keys": relations,
    }


def draft(doc_id, text, *, question="Which statement is supported?", options=None):
    options = options or [
        "The source supports this claim.",
        "The source supports the opposite claim.",
        "The supplied document does not provide enough information.",
    ]
    quote = text[: min(20, len(text))]
    return {
        "document_id": doc_id,
        "question": question,
        "options": options,
        "answer_index": 0,
        "abstention_index": len(options) - 1,
        "abstention_text": options[-1],
        "explanation": "The cited sentence supports the first choice.",
        "evidence_spans": [{"start": 0, "end": len(quote), "quote": quote}],
        "task_kind": "supported_claim",
        "answerability_status": "answerable",
        "unanswerable_reason": None,
        "author_name": "author_luna",
        "author_model": "gpt-6-luna",
    }


def soft_draft(doc_id, text, first=(1, 2), second=(1, 2), *, question="Which outcome is selected?"):
    item = draft(doc_id, text, question=question)
    item.pop("answer_index")
    item["distribution_source_kind"] = "exact_derived"
    item["distribution_evidence"] = {
        "kind": "exact_derived",
        "probabilities": [
            {"numerator": first[0], "denominator": first[1]},
            {"numerator": second[0], "denominator": second[1]},
            {"numerator": 0, "denominator": 1},
        ],
        "target_meaning": "Probability over the two source-supported outcomes.",
        "derivation": f"The mechanism assigns {first[0]}/{first[1]} and {second[0]}/{second[1]}.",
        "mechanism_assumptions": ["The source-defined outcomes are the only selectable outcomes."],
        "outcome_partition": "Two source-supported outcomes and one abstention option.",
        "mechanism_family": "source_count",
    }
    return item


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def draft_hash(item):
    return hashlib.sha256(
        json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def pilot_draft_hash(item):
    return hashlib.sha256(json.dumps(item, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def write_review(path, item, *, status="accepted", reviewer_model="gpt-6-luna", digest=None):
    decision = {
        "document_id": item["document_id"],
        "index": 0,
        "reviewed_draft_sha256": digest or draft_hash(item),
        "status": status,
        "reason": "Independent review confirms this exact draft and target.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "reviewer_model": reviewer_model,
        "reviewed_at": "2026-10-02T12:00:00Z",
        "selected_rows": [decision],
    }, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return decision


def make_batch(tmp_path, documents):
    batch = tmp_path / "batch"
    batch.mkdir()
    wiki = [doc for doc in documents if not doc["id"].startswith("report")]
    reporting = [doc for doc in documents if doc["id"].startswith("report")]
    write_jsonl(batch / "wiki-documents.jsonl", wiki)
    write_jsonl(batch / "reporting-documents.jsonl", reporting)
    return batch


def test_freeze_binds_source_hashes_and_allows_append_only_extension(tmp_path):
    candidates = [source(f"wiki-{index}", f"Document text number {index} has preserved details.", [f"page:{index}"])
                 for index in range(20)]
    initial_allocation = freeze_groups(candidates, seed=SEED)
    train_doc = next(doc for doc in candidates if initial_allocation[doc["id"]] == "train")
    dev_doc = next(doc for doc in candidates if initial_allocation[doc["id"]] == "dev")
    batch = make_batch(tmp_path, [train_doc, dev_doc])

    first = freeze(batch)
    original_allocation = dict(first["allocation"])
    original_hashes = dict(first["document_sha256"])
    assert len(first["allocation_events"]) == 1
    assert len(first["freeze_snapshots"]) == 1

    old_documents_before = (batch / "documents.jsonl").read_bytes()
    old_allocation_before = (batch / "allocation.json").read_bytes()
    write_jsonl(batch / "wiki-documents.jsonl", [{**train_doc, "title": "Changed after freeze"}, dev_doc])
    with pytest.raises(ValueError, match="existing frozen document or its relations changed"):
        freeze(batch)
    assert (batch / "documents.jsonl").read_bytes() == old_documents_before
    assert (batch / "allocation.json").read_bytes() == old_allocation_before

    added = source("report-new", "A new reporting document with its own details.", ["page:new"])
    write_jsonl(batch / "reporting-documents.jsonl", [added])
    # Retain both original source inputs while adding the reporting source.
    write_jsonl(batch / "wiki-documents.jsonl", [train_doc, dev_doc])
    extended = freeze(batch)

    assert {doc_id: extended["allocation"][doc_id] for doc_id in original_allocation} == original_allocation
    assert {doc_id: extended["document_sha256"][doc_id] for doc_id in original_hashes} == original_hashes
    assert len(extended["allocation_events"]) == 2
    assert len(extended["freeze_snapshots"]) == 2
    assert extended["source_origin"]["report-new"] == "reporting"

    before = (batch / "documents.jsonl").read_bytes(), (batch / "allocation.json").read_bytes()
    # A new bridge joining old train and dev sources must reject atomically.
    train_key = train_doc["relation_keys"][0]
    dev_key = dev_doc["relation_keys"][0]
    bridge = source("report-bridge", "A source that relates both frozen groups.", [train_key, dev_key])
    write_jsonl(batch / "reporting-documents.jsonl", [added, bridge])
    with pytest.raises(ValueError, match="join existing train and dev"):
        freeze(batch)
    assert (batch / "documents.jsonl").read_bytes() == before[0]
    assert (batch / "allocation.json").read_bytes() == before[1]


def test_freeze_accepts_optional_public_source_lane(tmp_path):
    text = "A public guidance page identifies steps for preparing for floods."
    batch = make_batch(tmp_path, [source("wiki-one", text, ["page:one"])])
    write_jsonl(batch / "reporting-documents.jsonl", [])
    write_jsonl(batch / "public-documents.jsonl", [source("public-floods", text, ["public:floods"])])
    manifest = freeze(batch)
    assert manifest["source_origin"] == {"wiki-one": "wiki", "public-floods": "public"}


def test_same_split_group_merge_keeps_old_rows_idempotent(tmp_path):
    text = "A source states that the archive retained the dated record for researchers."
    docs = [
        source("wiki-a", text, ["page:zeta"]),
        source("wiki-b", text, ["page:omega"]),
        source("wiki-c", text, ["page:other"]),
    ]
    batch = make_batch(tmp_path, docs)
    first = freeze(batch)
    frozen_documents = (batch / "documents.jsonl").read_bytes()
    old_manifest = json.loads((batch / "allocation.json").read_text())
    old_manifest.pop("authoring_freeze_group_ids")
    (batch / "allocation.json").write_text(json.dumps(old_manifest, ensure_ascii=False) + "\n")
    migrated = freeze(batch)
    assert migrated["authoring_freeze_group_ids"] == first["group_ids"]
    assert (batch / "documents.jsonl").read_bytes() == frozen_documents

    same_split = next(
        (left, right)
        for index, left in enumerate(docs)
        for right in docs[index + 1:]
        if first["allocation"][left["id"]] == first["allocation"][right["id"]]
    )
    left, right = same_split
    previous_group = migrated["authoring_freeze_group_ids"][left["id"]]
    chunk = tmp_path / "initial.jsonl"
    item = draft(left["id"], text)
    write_jsonl(chunk, [item])
    assert admit(batch, chunk)["new_originals"] == 1

    bridge = source(
        "report-bridge",
        "A new source connects two previously independent records.",
        ["a:bridge", *left["relation_keys"], *right["relation_keys"]],
    )
    helper_copy = batch / "documents" / "report-bridge.json"
    helper_copy.parent.mkdir(parents=True, exist_ok=True)
    helper_copy.write_text(json.dumps(bridge, ensure_ascii=False, indent=2) + "\n")
    write_jsonl(batch / "reporting-documents.jsonl", [bridge])
    extended = freeze(batch)
    assert extended["allocation"][left["id"]] == extended["allocation"][right["id"]]
    assert extended["group_ids"][left["id"]] == extended["group_ids"][right["id"]]
    assert extended["group_ids"][left["id"]] != previous_group
    assert extended["authoring_freeze_group_ids"][left["id"]] == previous_group
    assert helper_copy.read_bytes() == (
        json.dumps(bridge, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    retry = admit(batch, chunk)
    row = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])
    assert retry["new_originals"] == 0
    assert row["document_metadata"]["authoring_freeze_group_id"] == previous_group


def test_admission_is_idempotent_and_records_source_and_self_review(tmp_path):
    text = "An archive preserved a dated list of materials for researchers."
    batch = make_batch(tmp_path, [source("wiki-one", text, ["page:one"])])
    freeze(batch)
    chunk = tmp_path / "chunk.jsonl"
    item = draft("wiki-one", text)
    write_jsonl(chunk, [item])

    first = admit(batch, chunk)
    second = admit(batch, chunk)
    rows = [json.loads(line) for line in (batch / "accepted.jsonl").read_text().splitlines()]
    row = rows[0]
    assert first["new_originals"] == 1
    assert second["new_originals"] == 0
    assert second["accepted_originals"] == 1
    assert row["semantic_review_status"] == "author_self_review_only"
    assert row["language_review_status"] == "author_self_review_only"
    assert row["document_metadata"]["source_origin"] == "wiki"
    assert row["document_metadata"]["authoring_freeze_group_id"] == freeze(batch)["authoring_freeze_group_ids"]["wiki-one"]
    assert row["provenance"].startswith("luna-generated; label not independently verified;")
    assert json.loads((batch / "runs" / "document-expansion-2026-10-02" / "status.json").read_text())["counts"]["accepted_originals"] == 1


def test_twenty_original_per_document_cap_and_failed_chunk_leave_ledger_unchanged(tmp_path):
    text = "The archive states that the record remained available for public research."
    batch = make_batch(tmp_path, [source("wiki-one", text, ["page:one"])])
    freeze(batch)
    chunk = tmp_path / "twenty.jsonl"
    rows = [draft("wiki-one", text, question=f"Which supported detail applies in case {index}?")
            for index in range(20)]
    write_jsonl(chunk, rows)
    assert admit(batch, chunk)["accepted_originals"] == 20
    ledger_before = (batch / "accepted.jsonl").read_bytes()
    train_before = (batch / "train.jsonl").read_bytes()
    dev_before = (batch / "dev.jsonl").read_bytes()
    status_before = (batch / "runs" / "document-expansion-2026-10-02" / "status.json").read_bytes()

    overflow = draft("wiki-one", text, question="A twenty-first distinct original?")
    write_jsonl(chunk, [overflow])
    with pytest.raises(ValueError, match="cannot exceed 20"):
        admit(batch, chunk)
    assert (batch / "accepted.jsonl").read_bytes() == ledger_before
    assert (batch / "train.jsonl").read_bytes() == train_before
    assert (batch / "dev.jsonl").read_bytes() == dev_before
    assert (batch / "runs" / "document-expansion-2026-10-02" / "status.json").read_bytes() == status_before


def test_soft_evidence_provenance_and_permutation_keep_targets_aligned(tmp_path):
    text = "Three solar forms are named: collectors, cells, and cookers."
    batch = make_batch(tmp_path, [source("report-solar", text, ["article:solar"])])
    freeze(batch)
    item = draft("report-solar", text, question="Which of these forms would be selected?")
    item.pop("answer_index")
    item["distribution_source_kind"] = "exact_derived"
    item["distribution_evidence"] = {
        "kind": "exact_derived",
        "probabilities": [
            {"numerator": 1, "denominator": 2},
            {"numerator": 1, "denominator": 2},
            {"numerator": 0, "denominator": 1},
        ],
        "target_meaning": "Uniform selection from the two cited solar forms.",
        "derivation": "Each of the two listed forms receives one half.",
        "mechanism_assumptions": ["The source-defined selection is uniform."],
        "outcome_partition": "Two supported forms and one abstention option.",
        "mechanism_family": "source_count",
    }
    chunk = tmp_path / "soft.jsonl"
    write_jsonl(chunk, [item])
    result = admit(batch, chunk)
    row = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])
    aligned = dict(zip(row["options"], row["target_distribution"]))
    assert aligned == {
        "The source supports this claim.": 0.5,
        "The source supports the opposite claim.": 0.5,
        "The supplied document does not provide enough information.": 0.0,
    }
    assert row["distribution_source_kind"] == "exact_derived"
    assert row["distribution_evidence"]["mechanism_family"] == "source_count"
    assert result["counts"]["soft_non_one_hot"] == 1
    assert result["counts"]["soft_multiple_positive_choices"] == 1
    assert result["counts"]["soft_mechanism_families"] == {"source_count": 1}


def test_full_265_choice_soft_original_preserves_choice_evidence_mapping(tmp_path):
    text = "The source identifies two supported outcomes and leaves all other claims outside them."
    related_source = source(
        "wiki-wide-related", "The related archive retained a separate source record.", ["page:wide"]
    )
    batch = make_batch(tmp_path, [source("wiki-wide", text, ["page:wide"]), related_source])
    frozen = freeze(batch)
    frozen_documents = {
        document["id"]: document
        for document in [json.loads(line) for line in (batch / "documents.jsonl").read_text().splitlines()]
    }
    options = ["Supported outcome A", "Supported outcome B"]
    options.extend(f"Authored option {index}" for index in range(262))
    options.append("The supplied document does not provide enough information.")
    assert len(options) == 265
    item = draft("wiki-wide", text, question="Which outcome is selected?", options=options)
    item.pop("answer_index")
    item["distribution_source_kind"] = "explicit_stochastic_process"
    item["distribution_evidence"] = {
        "kind": "explicit_stochastic_process",
        "probabilities": [
            {"numerator": 1, "denominator": 2},
            {"numerator": 1, "denominator": 2},
            *[{"numerator": 0, "denominator": 1} for _ in range(263)],
        ],
        "target_meaning": "An equal random draw between the two source-supported outcomes.",
        "derivation": "The stated two-outcome mechanism assigns one half to each outcome.",
        "mechanism_assumptions": ["Only the first two source-supported outcomes are selectable."],
        "outcome_partition": "Two supported outcomes and 263 zero-mass options.",
        "mechanism_family": "two_stage",
    }
    chunk = tmp_path / "wide.jsonl"
    write_jsonl(chunk, [item])
    admit(batch, chunk)
    row = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])
    assert len(row["options"]) == len(row["choice_ids"]) == len(row["target_distribution"]) == 265
    assert row["abstention_choice_id"] == row["choice_ids"][row["options"].index(options[-1])]
    assert {option: mass for option, mass in zip(row["options"], row["target_distribution"]) if mass > 0} == {
        "Supported outcome A": 0.5,
        "Supported outcome B": 0.5,
    }

    ledger_before = (batch / "accepted.jsonl").read_bytes()
    with pytest.raises(ValueError, match="exceeds max_augmentations"):
        export_permutation_augmentations(batch, [row["id"]], max_augmentations=1)
    assert not (batch / "permutation-augmentations.jsonl").exists()

    exported = export_permutation_augmentations(batch, [row["id"]], max_augmentations=2)
    variants = [json.loads(line) for line in (batch / "augmentations.jsonl").read_text().splitlines()]
    assert exported["new_augmentations"] == 2
    assert exported["accepted_originals"] == 1
    assert exported["augmentation_rows"] == 2
    assert (batch / "accepted.jsonl").read_bytes() == ledger_before
    assert len({variant["id"] for variant in variants}) == 2
    assert {variant["augmentation_type"] for variant in variants} == {
        "highest_substantive_at_head", "abstention_at_head",
    }
    for variant in variants:
        assert variant["augmentation_of"] == row["id"]
        assert variant["problem_group_id"] == row["problem_group_id"]
        assert variant["split"] == row["split"]
        assert variant["document_metadata"]["document_id"] == "wiki-wide"
        assert frozen["group_ids"][variant["document_metadata"]["document_id"]] == frozen["group_ids"][
            related_source["id"]
        ]
        assert variant["distribution_evidence"]["choice_ids"] == variant["choice_ids"]
        assert validate_document_augmentation(variant, row, frozen_documents, frozen["allocation"]) == []

    substantive_mass = max(
        value for choice_id, value in zip(row["choice_ids"], row["target_distribution"])
        if choice_id != row["abstention_choice_id"]
    )
    highest = next(variant for variant in variants if variant["augmentation_type"] == "highest_substantive_at_head")
    abstention = next(variant for variant in variants if variant["augmentation_type"] == "abstention_at_head")
    assert highest["target_distribution"][264] == substantive_mass
    assert highest["choice_ids"][264] != row["abstention_choice_id"]
    assert abstention["choice_ids"][264] == row["abstention_choice_id"]

    misaligned_target = dict(highest)
    misaligned_target["target_distribution"] = list(highest["target_distribution"])
    misaligned_target["target_distribution"][0], misaligned_target["target_distribution"][264] = (
        misaligned_target["target_distribution"][264], misaligned_target["target_distribution"][0]
    )
    assert validate_document_augmentation(misaligned_target, row, frozen_documents, frozen["allocation"])

    misaligned_evidence = dict(highest)
    misaligned_evidence["distribution_evidence"] = dict(highest["distribution_evidence"])
    rational_probabilities = list(highest["distribution_evidence"]["probabilities"])
    rational_probabilities[0], rational_probabilities[-1] = rational_probabilities[-1], rational_probabilities[0]
    misaligned_evidence["distribution_evidence"]["probabilities"] = rational_probabilities
    assert validate_document_augmentation(misaligned_evidence, row, frozen_documents, frozen["allocation"])

    repeat = export_permutation_augmentations(batch, [row["id"]], max_augmentations=2)
    assert repeat["new_augmentations"] == 0
    assert repeat["accepted_originals"] == 1
    assert len((batch / "permutation-augmentations.jsonl").read_text().splitlines()) == 2
    status = json.loads((batch / "runs" / "document-expansion-2026-10-02" / "status.json").read_text())
    assert status["counts"]["accepted_originals"] == 1
    assert status["augmentations"]["rows"] == 2


def test_conflicting_duplicate_id_and_missing_soft_provenance_reject_whole_chunk(tmp_path):
    text = "The archive marks a collection as open to public review."
    batch = make_batch(tmp_path, [source("wiki-one", text, ["page:one"])])
    freeze(batch)
    chunk = tmp_path / "chunk.jsonl"
    item = draft("wiki-one", text)
    write_jsonl(chunk, [item])
    admit(batch, chunk)
    before = (batch / "accepted.jsonl").read_bytes()

    conflict = {**item, "explanation": "A different explanation for the same identity."}
    write_jsonl(chunk, [conflict])
    with pytest.raises(ValueError, match="conflicting content"):
        admit(batch, chunk)

    soft = draft("wiki-one", text, question="How are two outcomes weighted?")
    soft.pop("answer_index")
    soft["distribution_source_kind"] = "exact_derived"
    soft["distribution_evidence"] = {
        "kind": "exact_derived",
        "probabilities": [
            {"numerator": 1, "denominator": 2},
            {"numerator": 1, "denominator": 2},
            {"numerator": 0, "denominator": 1},
        ],
        "target_meaning": "A meaningful distribution.",
        "derivation": "Equal weights.",
        "mechanism_assumptions": ["There are two outcomes."],
        "outcome_partition": "Two outcomes plus abstention.",
        # mechanism_family is deliberately missing.
    }
    write_jsonl(chunk, [item, soft])
    with pytest.raises(ValueError, match="mechanism_family"):
        admit(batch, chunk)
    assert (batch / "accepted.jsonl").read_bytes() == before


def test_correction_overlay_is_idempotent_preserves_raw_and_stays_quarantined(tmp_path):
    text = "The archive defines two routes and describes their relative frequency."
    batch = make_batch(tmp_path, [source("wiki-correct", text, ["page:correct"])])
    freeze(batch)
    chunk = tmp_path / "correction.jsonl"
    original_draft = soft_draft("wiki-correct", text)
    write_jsonl(chunk, [original_draft])
    admit(batch, chunk)
    accepted_path = batch / "accepted.jsonl"
    raw_before = accepted_path.read_bytes()
    original = json.loads(raw_before.decode().splitlines()[0])
    row_id = original["id"]

    corrected_draft = soft_draft("wiki-correct", text, first=(2, 3), second=(1, 3))
    write_jsonl(chunk, [corrected_draft])
    correction = correct(batch, chunk, "Independent review corrected the target ratio.")
    correction_path = batch / "corrections.jsonl"
    correction_bytes = correction_path.read_bytes()
    correction_entry = json.loads(correction_bytes.decode().splitlines()[0])
    assert correction["new_corrections"] == 1
    assert accepted_path.read_bytes() == raw_before
    assert correction_entry["row_id"] == row_id
    assert correction_entry["prior_row_sha256"] == hashlib.sha256(
        json.dumps(original, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    active_path = batch / ("train.jsonl" if original["split"] == "train" else "dev.jsonl")
    active_corrected = json.loads(active_path.read_text().splitlines()[0])
    assert active_corrected["target_distribution"] != original["target_distribution"]
    assert correct(batch, chunk, "Independent review corrected the target ratio.")["new_corrections"] == 0
    assert correction_path.read_bytes() == correction_bytes

    quarantine(batch, [row_id], "Target remains quarantined pending a fresh review decision.")
    next_correction = soft_draft("wiki-correct", text, first=(1, 3), second=(2, 3))
    write_jsonl(chunk, [next_correction])
    assert correct(batch, chunk, "Applied the final reviewed target.")["accepted_originals"] == 0
    assert accepted_path.read_bytes() == raw_before
    assert not (batch / "train.jsonl").read_text()
    assert not (batch / "dev.jsonl").read_text()
    status = json.loads((batch / "runs" / "document-expansion-2026-10-02" / "status.json").read_text())
    assert status["counts"]["accepted_originals"] == 0
    assert status["quarantined_originals"] == 1


def test_quarantine_preserves_raw_rows_and_frees_the_active_document_cap(tmp_path):
    text = "The public archive retained a dated description for consultation."
    batch = make_batch(tmp_path, [source("wiki-cap", text, ["page:cap"])])
    freeze(batch)
    chunk = tmp_path / "cap.jsonl"
    originals = [draft("wiki-cap", text, question=f"Which fact applies in case {index}?") for index in range(20)]
    write_jsonl(chunk, originals)
    admit(batch, chunk)
    accepted_path = batch / "accepted.jsonl"
    raw_before = accepted_path.read_bytes()
    original_rows = [json.loads(line) for line in raw_before.decode().splitlines()]
    quarantined_id = original_rows[0]["id"]
    status_path = batch / "runs" / "document-expansion-2026-10-02" / "status.json"
    prior_status = json.loads(status_path.read_text())
    prior_status["initial_soft_target"] = 0.4
    prior_status["counts"]["initial_target_rows"] = 12
    status_path.write_text(json.dumps(prior_status) + "\n")

    result = quarantine(batch, [quarantined_id], "Targeted independent review rejected this target.")
    quarantine_entry = json.loads((batch / "quarantine.jsonl").read_text().splitlines()[0])
    assert result["accepted_originals"] == 19
    assert result["quarantined_originals"] == 1
    assert accepted_path.read_bytes() == raw_before
    assert quarantine_entry["row_id"] == quarantined_id
    assert quarantine_entry["prior_row_sha256"] == hashlib.sha256(
        json.dumps(original_rows[0], ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    refreshed_status = json.loads(status_path.read_text())
    assert refreshed_status["initial_soft_target"] == 0.4
    assert refreshed_status["counts"]["initial_target_rows"] == 12

    replacement = draft("wiki-cap", text, question="Which new supported fact applies after review?")
    write_jsonl(chunk, [replacement])
    admitted = admit(batch, chunk)
    assert admitted["new_originals"] == 1
    assert admitted["accepted_originals"] == 20
    assert len(accepted_path.read_text().splitlines()) == 21
    assert accepted_path.read_bytes().startswith(raw_before)

    old_draft = originals[0]
    write_jsonl(chunk, [old_draft])
    with pytest.raises(ValueError, match="quarantined original"):
        admit(batch, chunk)


def test_release_requires_current_independent_review_and_releases_exact_overlay(tmp_path):
    text = "The archive defines two routes and describes their relative frequency."
    doc_id = "wiki-release"
    batch = make_batch(tmp_path, [source(doc_id, text, ["page:release"])])
    freeze(batch)
    draft_path = batch / "drafts" / f"{doc_id}.jsonl"
    initial = soft_draft(doc_id, text)
    write_jsonl(draft_path, [initial])
    admit(batch, draft_path)
    row_id = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])["id"]
    raw_before = (batch / "accepted.jsonl").read_bytes()
    quarantine(batch, [row_id], "Hold for targeted review.")

    corrected = soft_draft(doc_id, text, first=(2, 3), second=(1, 3))
    assert draft_hash(corrected) != pilot_draft_hash(corrected)
    write_jsonl(draft_path, [corrected])
    correct(batch, draft_path, "Applied the reviewed probability correction.")
    review_path = batch / "reviews" / "correction-review.json"
    write_review(review_path, corrected)

    release_path = batch / "quarantine-releases.jsonl"
    train_path = batch / "train.jsonl"
    dev_path = batch / "dev.jsonl"
    release_ledger_before = release_path.read_bytes() if release_path.exists() else b""
    exports_before = train_path.read_bytes(), dev_path.read_bytes()

    wrong_model = tmp_path / "wrong-model.json"
    write_review(wrong_model, corrected, reviewer_model="gpt-6.1-sol")
    with pytest.raises(ValueError, match="gpt-6-luna"):
        release(batch, [row_id], wrong_model)

    stale_hash = tmp_path / "stale-hash.json"
    write_review(stale_hash, corrected, digest="0" * 64)
    with pytest.raises(ValueError, match="changed after independent review"):
        release(batch, [row_id], stale_hash)
    assert (release_path.read_bytes() if release_path.exists() else b"") == release_ledger_before
    assert (train_path.read_bytes(), dev_path.read_bytes()) == exports_before

    # A currently hashed draft cannot release a different row version authored
    # through another correction chunk.
    other_target = soft_draft(doc_id, text, first=(1, 3), second=(2, 3))
    external_correction = tmp_path / "external-correction.jsonl"
    write_jsonl(external_correction, [other_target])
    correct(batch, external_correction, "Applied a later correction outside the review draft file.")
    with pytest.raises(ValueError, match="not the latest corrected draft"):
        release(batch, [row_id], review_path)
    assert (release_path.read_bytes() if release_path.exists() else b"") == release_ledger_before
    assert (train_path.read_bytes(), dev_path.read_bytes()) == exports_before

    write_jsonl(draft_path, [other_target])
    write_review(review_path, other_target)

    result = release(batch, [row_id], review_path)
    release_entry = json.loads(release_path.read_text().splitlines()[0])
    assert result["new_releases"] == 1
    assert result["accepted_originals"] == 1
    assert release_entry["row_id"] == row_id
    assert release_entry["reviewer_model"] == "gpt-6-luna"
    assert release_entry["status"] == "accepted"
    assert release_entry["reviewed_draft_sha256"] == draft_hash(other_target)
    correction_entry = json.loads((batch / "corrections.jsonl").read_text().splitlines()[-1])
    assert release_entry["overlay_row_sha256"] == correction_entry["corrected_row_sha256"]
    assert (batch / "accepted.jsonl").read_bytes() == raw_before
    active_path = train_path if json.loads(train_path.read_text().splitlines()[0])["id"] == row_id else dev_path
    active = json.loads(active_path.read_text().splitlines()[0])
    aligned = dict(zip(active["options"], active["target_distribution"]))
    assert aligned["The source supports this claim."] == pytest.approx(1 / 3)
    assert aligned["The source supports the opposite claim."] == pytest.approx(2 / 3)

    assert release(batch, [row_id], review_path)["new_releases"] == 0
    assert len(release_path.read_text().splitlines()) == 1


def test_luna_correction_preserves_primary_sol_author_and_releases_compact_mixed_file_review(tmp_path):
    text = "The archive retained the original register for later research and public reference."
    doc_id = "wiki-revision-author"
    batch = make_batch(tmp_path, [source(doc_id, text, ["page:revision-author"])])
    freeze(batch)
    original = draft(doc_id, text)
    original["author_name"] = "author_sol"
    original["author_model"] = "gpt-6.1-sol"
    draft_path = batch / "drafts" / f"{doc_id}.jsonl"
    write_jsonl(draft_path, [original])
    admit(batch, draft_path)
    raw_accepted = (batch / "accepted.jsonl").read_bytes()
    row_id = json.loads(raw_accepted.decode("utf-8").splitlines()[0])["id"]
    quarantine(batch, [row_id], "Hold for independent review of the evidence spans.")

    repair = dict(original)
    repair["author_name"] = "repair_luna"
    repair["author_model"] = "gpt-6-luna"
    quote = text[26:46]
    repair["evidence_spans"] = [{"start": 26, "end": 46, "quote": quote}]
    correction_chunk = tmp_path / "repair.jsonl"
    write_jsonl(correction_chunk, [repair])
    correction_result = correct(batch, correction_chunk, "Luna supplied the corrected exact evidence span.")
    assert correction_result["new_corrections"] == 1
    assert correct(batch, correction_chunk, "Luna supplied the corrected exact evidence span.")["new_corrections"] == 0
    assert (batch / "accepted.jsonl").read_bytes() == raw_accepted

    correction_event = json.loads((batch / "corrections.jsonl").read_text().splitlines()[0])
    corrected_row = correction_event["row"]
    metadata = corrected_row["document_metadata"]
    assert metadata["author_name"] == "author_sol"
    assert metadata["author_model"] == "gpt-6.1-sol"
    assert corrected_row["provenance"].startswith("gpt-6.1-sol-generated;")
    assert metadata["revision_author"] == {"author_name": "repair_luna", "author_model": "gpt-6-luna"}
    assert correction_event["source_draft_sha256"] == draft_hash(repair)

    filler = [draft(doc_id, text, question=f"Which reviewed fact applies in scenario {index}?") for index in range(3)]
    mixed_file = batch / "drafts" / "reviewed-corrections.jsonl"
    write_jsonl(mixed_file, [*filler, repair])
    review_path = batch / "reviews" / "luna-correction.json"
    decision = {
        "document_id": doc_id,
        "index": 3,
        "draft_file": "drafts/reviewed-corrections.jsonl",
        "reviewed_draft_sha256": draft_hash(repair),
        "status": "accepted",
        "reason": "The corrected evidence quote matches the frozen source exactly.",
    }
    review_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.write_text(json.dumps({
        "reviewer_model": "gpt-6-luna",
        "selected_rows": [decision],
    }, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    released = release(batch, [row_id], review_path)
    release_event = json.loads((batch / "quarantine-releases.jsonl").read_text().splitlines()[0])
    assert released["new_releases"] == 1
    assert release_event["draft_file"] == "drafts/reviewed-corrections.jsonl"
    assert release_event["reviewed_draft_sha256"] == draft_hash(repair)
    assert release_event["review_decision"]["draft_file"] == "drafts/reviewed-corrections.jsonl"
    assert (batch / "accepted.jsonl").read_bytes() == raw_accepted


def test_release_rejects_unsafe_declared_draft_file(tmp_path):
    text = "The archive retained the original register for later research and public reference."
    doc_id = "wiki-unsafe-review-path"
    batch = make_batch(tmp_path, [source(doc_id, text, ["page:unsafe-review-path"])])
    freeze(batch)
    item = draft(doc_id, text)
    draft_path = batch / "drafts" / f"{doc_id}.jsonl"
    write_jsonl(draft_path, [item])
    admit(batch, draft_path)
    row_id = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])["id"]
    quarantine(batch, [row_id], "Hold for an independent decision.")
    review_path = tmp_path / "unsafe.json"
    review_path.write_text(json.dumps({
        "reviewer_model": "gpt-6-luna",
        "selected_rows": [{
            "document_id": doc_id,
            "index": 0,
            "draft_file": "../../outside.jsonl",
            "reviewed_draft_sha256": draft_hash(item),
            "status": "accepted",
        }],
    }) + "\n")
    with pytest.raises(ValueError, match="safe relative JSONL path"):
        release(batch, [row_id], review_path)


def test_release_reconstructs_legacy_correction_without_rewriting_its_receipt(tmp_path):
    text = "The archive retained the original register for later research and public reference."
    doc_id = "wiki-legacy-correction"
    batch = make_batch(tmp_path, [source(doc_id, text, ["page:legacy-correction"])])
    freeze(batch)
    original = draft(doc_id, text)
    draft_path = batch / "drafts" / f"{doc_id}.jsonl"
    write_jsonl(draft_path, [original])
    admit(batch, draft_path)
    row_id = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])["id"]
    quarantine(batch, [row_id], "Hold for a corrected evidence span.")

    corrected = dict(original)
    quote = text[26:46]
    corrected["evidence_spans"] = [{"start": 26, "end": 46, "quote": quote}]
    write_jsonl(draft_path, [corrected])
    correct(batch, draft_path, "Corrected the evidence span.")

    corrections_path = batch / "corrections.jsonl"
    legacy_event = json.loads(corrections_path.read_text().splitlines()[0])
    legacy_event.pop("source_draft_sha256")
    legacy_event.pop("revision_author")
    legacy_event["row"]["document_metadata"].pop("revision_author")
    legacy_event["corrected_row_sha256"] = hashlib.sha256(
        json.dumps(legacy_event["row"], ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()
    write_jsonl(corrections_path, [legacy_event])
    legacy_corrections_before = corrections_path.read_bytes()

    review_path = batch / "reviews" / "legacy-review.json"
    write_review(review_path, corrected)
    result = release(batch, [row_id], review_path)
    assert result["new_releases"] == 1
    assert corrections_path.read_bytes() == legacy_corrections_before


def test_new_correction_invalidates_old_release_even_when_overlay_returns_to_old_digest(tmp_path):
    text = "The archive defines two routes and describes their relative frequency."
    doc_id = "wiki-release-correction"
    batch = make_batch(tmp_path, [source(doc_id, text, ["page:release-correction"])])
    freeze(batch)
    draft_path = batch / "drafts" / f"{doc_id}.jsonl"
    initial = soft_draft(doc_id, text)
    write_jsonl(draft_path, [initial])
    admit(batch, draft_path)
    row_id = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])["id"]
    quarantine(batch, [row_id], "Hold for targeted review.")

    first_correction = soft_draft(doc_id, text, first=(2, 3), second=(1, 3))
    write_jsonl(draft_path, [first_correction])
    correct(batch, draft_path, "Applied correction one.")
    review_path = batch / "reviews" / "first.json"
    write_review(review_path, first_correction)
    release(batch, [row_id], review_path)
    assert len((batch / "train.jsonl").read_text().splitlines() + (batch / "dev.jsonl").read_text().splitlines()) == 1

    # Returning to an earlier distribution still creates a new correction event.
    # The prior release is tied to correction one and must no longer make the row active.
    write_jsonl(draft_path, [initial])
    result = correct(batch, draft_path, "A later correction restores the earlier target.")
    assert result["accepted_originals"] == 0
    assert result["quarantined_originals"] == 1
    assert not (batch / "train.jsonl").read_text()
    assert not (batch / "dev.jsonl").read_text()

    stale_release_ledger = (batch / "quarantine-releases.jsonl").read_bytes()
    with pytest.raises(ValueError, match="draft changed after independent review"):
        release(batch, [row_id], review_path)
    assert (batch / "quarantine-releases.jsonl").read_bytes() == stale_release_ledger


def test_new_quarantine_invalidates_existing_release_binding(tmp_path):
    text = "The archive defines two routes and describes their relative frequency."
    doc_id = "wiki-release-quarantine"
    batch = make_batch(tmp_path, [source(doc_id, text, ["page:release-quarantine"])])
    freeze(batch)
    draft_path = batch / "drafts" / f"{doc_id}.jsonl"
    item = soft_draft(doc_id, text, first=(2, 3), second=(1, 3))
    write_jsonl(draft_path, [item])
    admit(batch, draft_path)
    row_id = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])["id"]
    quarantine(batch, [row_id], "Initial hold.")
    corrected = soft_draft(doc_id, text, first=(1, 3), second=(2, 3))
    write_jsonl(draft_path, [corrected])
    correct(batch, draft_path, "Reviewed target correction.")
    review_path = batch / "reviews" / "current.json"
    write_review(review_path, corrected)
    release(batch, [row_id], review_path)

    quarantine(batch, [row_id], "A later review found a new issue.")
    assert not (batch / "train.jsonl").read_text()
    assert not (batch / "dev.jsonl").read_text()


def test_model_routing_audit_preserves_raw_receipts_and_quarantine(tmp_path):
    text = "The archive records three dated decisions in its public index."
    doc_id = "wiki-routing"
    batch = make_batch(tmp_path, [source(doc_id, text, ["page:routing"])])
    freeze(batch)
    drafts_path = batch / "drafts" / f"{doc_id}.jsonl"
    authored = [
        draft(doc_id, text, question=f"Which archived decision is supported, item {index}?")
        for index in range(3)
    ]
    new_luna_draft = draft(doc_id, text, question="Which newly authored Luna row is supported?")
    new_luna_draft["author_name"] = "new_luna"
    authored.append(new_luna_draft)
    write_jsonl(drafts_path, authored)
    submitted = tmp_path / "submitted.jsonl"
    write_jsonl(submitted, authored[:2])
    admit(batch, submitted)
    accepted_path = batch / "accepted.jsonl"
    accepted_before = accepted_path.read_bytes()
    raw_rows = [json.loads(line) for line in accepted_before.decode("utf-8").splitlines()]
    quarantined_id = raw_rows[0]["id"]
    quarantine(batch, [quarantined_id], "A separate semantic issue remains under review.")
    quarantine_before = (batch / "quarantine.jsonl").read_bytes()
    corrections_before = (batch / "corrections.jsonl").read_bytes() if (batch / "corrections.jsonl").exists() else b""

    result = audit_model_routing(
        batch,
        requested_model="gpt-6-luna",
        recorded_model="gpt-6-luna",
        effective_model="gpt-6.1-sol",
        reason="The author-job session inherited gpt-6.1-sol despite the requested Luna route.",
        author_names=["author_luna"],
    )
    assert result["audited_accepted_rows"] == 2
    assert result["audited_drafts"] == 3
    assert result["raw_accepted_ledger_preserved"] is True
    assert accepted_path.read_bytes() == accepted_before
    assert (batch / "quarantine.jsonl").read_bytes() == quarantine_before
    assert ((batch / "corrections.jsonl").read_bytes() if (batch / "corrections.jsonl").exists() else b"") == corrections_before

    active_paths = [batch / "train.jsonl", batch / "dev.jsonl"]
    active_rows = [json.loads(line) for path in active_paths for line in path.read_text().splitlines() if line]
    assert len(active_rows) == 1
    active = active_rows[0]
    assert active["id"] == raw_rows[1]["id"]
    assert active["document_metadata"]["author_model"] == "gpt-6.1-sol"
    assert active["provenance"].startswith("gpt-6.1-sol-generated;")
    routing = active["document_metadata"]["model_routing_audit"]
    assert routing["requested_model"] == "gpt-6-luna"
    assert routing["recorded_model"] == "gpt-6-luna"
    assert routing["effective_model"] == "gpt-6.1-sol"
    assert routing["reason"]
    assert len(routing["audit_sha256"]) == 64
    assert (active["target_distribution"] if "target_distribution" in active else active["answer_index"]) == (
        raw_rows[1].get("target_distribution", raw_rows[1].get("answer_index"))
    )
    status = json.loads((batch / "runs" / "document-expansion-2026-10-02" / "status.json").read_text())
    assert status["quarantined_originals"] == 1
    assert status["counts"]["accepted_originals"] == 1

    routes = load_model_routing_audit(batch)
    pending_row = _build_original_row(authored[2], {doc_id: json.loads((batch / "documents" / f"{doc_id}.json").read_text())},
                                      json.loads((batch / "allocation.json").read_text()), routes)
    assert pending_row["document_metadata"]["author_model"] == "gpt-6.1-sol"
    assert pending_row["provenance"].startswith("gpt-6.1-sol-generated;")
    assert "model_routing_audit" in pending_row["document_metadata"]
    new_luna_row = _build_original_row(
        authored[3], {doc_id: json.loads((batch / "documents" / f"{doc_id}.json").read_text())},
        json.loads((batch / "allocation.json").read_text()), routes,
    )
    assert new_luna_row["document_metadata"]["author_model"] == "gpt-6-luna"
    assert "model_routing_audit" not in new_luna_row["document_metadata"]

    # Repeating an audit over the same exact artifact hashes is idempotent.
    retry = audit_model_routing(
        batch,
        requested_model="gpt-6-luna",
        recorded_model="gpt-6-luna",
        effective_model="gpt-6.1-sol",
        reason="The author-job session inherited gpt-6.1-sol despite the requested Luna route.",
        author_names=["author_luna"],
    )
    assert retry["audit_events"] == 0
    assert accepted_path.read_bytes() == accepted_before


def test_model_routing_audit_rebuilds_pilot_export_without_changing_reviewed_drafts(tmp_path):
    from dataset_factory.document_pilot import export as export_pilot

    text = "The guide names three steps for preparing a household for flooding."
    doc_id = "wiki-pilot-routing"
    batch = tmp_path / "pilot"
    batch.mkdir()
    document = source(doc_id, text, ["page:pilot-routing"])
    write_jsonl(batch / "documents.jsonl", [document])
    documents_bytes = (batch / "documents.jsonl").read_bytes()
    (batch / "allocation.json").write_text(json.dumps({
        "documents_sha256": hashlib.sha256(documents_bytes).hexdigest(),
        "allocation": {doc_id: "dev"},
    }) + "\n")

    item = draft(doc_id, text)
    item.pop("document_id")  # The pilot binds each draft file to its source document.
    item["author_name"] = "legacy_pilot_author"
    draft_path = batch / "drafts" / f"{doc_id}.jsonl"
    write_jsonl(draft_path, [item])
    draft_bytes = draft_path.read_bytes()
    review_path = batch / "review.json"
    review_path.write_text(json.dumps({
        "reviewer_model": "gpt-6-luna",
        "decisions": [{
            "document_id": doc_id,
            "index": 0,
            "status": "accepted",
            "reviewed_draft_sha256": pilot_draft_hash(item),
        }],
    }, sort_keys=True) + "\n")
    review_bytes = review_path.read_bytes()

    before = export_pilot(batch)
    assert before["accepted_originals"] == 1
    result = audit_model_routing(
        batch,
        requested_model="gpt-6-luna",
        recorded_model="gpt-6-luna",
        effective_model="gpt-6.1-sol",
        reason="The author-job session inherited gpt-6.1-sol despite the requested Luna route.",
        author_names=["legacy_pilot_author"],
    )
    assert result["audited_drafts"] == 1
    assert result["pilot_export"]["accepted_originals"] == 1
    assert draft_path.read_bytes() == draft_bytes
    assert review_path.read_bytes() == review_bytes
    reviewer_result = audit_review_model_routing(
        batch,
        requested_model="gpt-6-luna",
        recorded_model="gpt-6-luna",
        effective_model="gpt-6.1-sol",
        reason="The reviewer session inherited gpt-6.1-sol despite the requested Luna route.",
    )
    assert reviewer_result["new_review_routing_events"] == 1
    assert reviewer_result["raw_review_preserved"] is True
    assert reviewer_result["review_sha256"] == hashlib.sha256(review_bytes).hexdigest()
    assert draft_path.read_bytes() == draft_bytes
    assert review_path.read_bytes() == review_bytes
    exported = json.loads((batch / "dev.jsonl").read_text().splitlines()[0])
    assert exported["document_metadata"]["author_model"] == "gpt-6.1-sol"
    assert exported["provenance"].startswith("gpt-6.1-sol-generated;")
    assert exported["document_metadata"]["model_routing_audit"]["recorded_model"] == "gpt-6-luna"
    assert exported["document_metadata"]["review"]["reviewed_draft_sha256"] == pilot_draft_hash(item)
    assert exported["semantic_review_status"] == "independent_sol_review"
    assert exported["language_review_status"] == "independent_sol_review"
    reviewer_route = exported["document_metadata"]["review_model_routing_audit"]
    assert reviewer_route["requested_model"] == "gpt-6-luna"
    assert reviewer_route["recorded_model"] == "gpt-6-luna"
    assert reviewer_route["effective_model"] == "gpt-6.1-sol"
    assert reviewer_route["reason"]


def test_frozen_source_hold_blocks_admission_and_release_but_allows_append_only_replacements(tmp_path):
    text = "The archive records the public process used to prepare for seasonal flooding."
    held_doc_id = "wiki-biography"
    regular_doc_id = "wiki-regular"
    held_doc = source(held_doc_id, text, ["page:biography"])
    regular_doc = source(regular_doc_id, "The guide lists public preparation steps for floods.", ["page:regular"])
    batch = make_batch(tmp_path, [held_doc, regular_doc])
    initial = freeze(batch)

    draft_path = batch / "drafts" / f"{held_doc_id}.jsonl"
    original_draft = draft(held_doc_id, text)
    write_jsonl(draft_path, [original_draft])
    admit(batch, draft_path)
    row_id = json.loads((batch / "accepted.jsonl").read_text().splitlines()[0])["id"]
    quarantine(batch, [row_id], "Held with its source pending replacement.")

    sidecar_path = batch / "source-screening-limitations.json"
    sidecar = {
        "frozen_source_pool_sha256": initial["documents_sha256"],
        "quarantined_biography_ids": [held_doc_id],
        "replacement_ids_by_language": {},
    }
    sidecar_bytes = (json.dumps(sidecar, sort_keys=True) + "\n").encode("utf-8")
    sidecar_path.write_bytes(sidecar_bytes)

    _, frozen_by_id, frozen_manifest = _load_frozen(batch)
    assert set(frozen_manifest["source_held_document_ids"]) == {held_doc_id}
    assert set(frozen_by_id) == {held_doc_id, regular_doc_id}

    rejected = draft(held_doc_id, text, question="Which newly attempted held-source question applies?")
    attempt = tmp_path / "held-attempt.jsonl"
    write_jsonl(attempt, [rejected])
    accepted_before = (batch / "accepted.jsonl").read_bytes()
    with pytest.raises(ValueError, match="source-held document"):
        admit(batch, attempt)
    assert (batch / "accepted.jsonl").read_bytes() == accepted_before

    review_path = batch / "reviews" / "held-release.json"
    write_review(review_path, original_draft)
    release_before = batch / "quarantine-releases.jsonl"
    with pytest.raises(ValueError, match="source-held document"):
        release(batch, [row_id], review_path)
    assert not release_before.exists()
    assert sidecar_path.read_bytes() == sidecar_bytes

    replacement = source("report-replacement", "A public flood guidance page gives safe preparation advice.",
                         ["page:replacement"])
    write_jsonl(batch / "reporting-documents.jsonl", [replacement])
    extended = freeze(batch)
    assert set(extended["allocation"]) == {held_doc_id, regular_doc_id, replacement["id"]}
    assert extended["document_sha256"][held_doc_id] == initial["document_sha256"][held_doc_id]
    assert sidecar_path.read_bytes() == sidecar_bytes
    _, extended_by_id, extended_manifest = _load_frozen(batch)
    assert set(extended_by_id) == {held_doc_id, regular_doc_id, replacement["id"]}
    assert set(extended_manifest["source_held_document_ids"]) == {held_doc_id}
