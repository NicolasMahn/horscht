import copy
import hashlib
import json

import pytest

from dataset_factory.issue_corrections import (
    MANIFEST_VERSION,
    CorrectionError,
    RecordKey,
    _cli,
    load_withdrawals,
    record_scope,
)


A = "probability-v2-assignment-0000101225"
B = "probability-v2-assignment-0000101226"


def line(record):
    return (json.dumps(record, sort_keys=True) + "\n").encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("field", ["scope_ids", "ids", "question_ids", "exclude_pending_review", "review_recommended"])
def test_list_scope_aliases_are_job_bounded(field):
    assert record_scope({field: ["0000101225", B]}, [A, B]) == {A, B}


@pytest.mark.parametrize("field", ["id", "assignment_id", "question_id"])
def test_scalar_scope_aliases(field):
    assert record_scope({field: "0000101225"}, [A, B]) == {A}


def test_boolean_flags_do_not_supply_scope_or_override_agreeing_fields():
    record = {"exclude_pending_review": True, "review_recommended": True,
              "scope_ids": [A], "question_id": "0000101225"}
    assert record_scope(record, [A, B]) == {A}
    with pytest.raises(CorrectionError, match="record_scope_missing"):
        record_scope({"exclude_pending_review": True, "note": A}, [A, B])


@pytest.mark.parametrize("record,code", [
    ({"ids": ["0000101227"]}, "scope_id_outside_job"),
    ({"ids": ["different-job-0000101225"]}, "scope_id_outside_job"),
    ({"ids": ["101225"]}, "scope_id_outside_job"),
    ({"ids": "0000101225"}, "invalid_exact_affected_ids"),
    ({"ids": []}, "invalid_exact_affected_ids"),
    ({"ids": [A, "0000101225"]}, "duplicate_normalized_scope_id"),
    ({"ids": [A], "scope_ids": [B]}, "conflicting_record_scopes"),
    ({"scope_ids": [A], "review_recommended": [B]}, "conflicting_record_scopes"),
    ({"scope_ids": [A], "exclude_pending_review": "true"}, "invalid_exact_affected_ids"),
])
def test_unsupported_scope_fails_closed(record, code):
    with pytest.raises(CorrectionError, match=code):
        record_scope(record, [A, B])


def test_suffix_must_resolve_uniquely():
    with pytest.raises(CorrectionError, match="scope_alias_ambiguous"):
        record_scope({"ids": ["0000101225"]}, [A, "other-assignment-0000101225"])


@pytest.fixture
def fixture(tmp_path):
    ledger = tmp_path / "author.jsonl"
    target = line({"scope_ids": ["0000101225", "0000101226"],
                   "exclude_pending_review": True, "review_recommended": True})
    receipt = line({"clarification": "Synthetic author clarification; never classified by code."})
    genuine = line({"ids": [A], "exclude_pending_review": True, "concrete_defect": True})
    ledger.write_bytes(target + receipt + genuine)
    entry = {
        "phase": "before_collection",
        "ledger_path": str(ledger), "line": 1, "record_sha256": digest(target),
        "question_ids": [A, B],
        "clarification_receipt": {"path": str(ledger), "line": 2, "sha256": digest(receipt)},
        "decision": "withdraw_exact_record", "replacement_training_status": "review_recommended",
        "approved_by": "central_coordinator", "author_clarification_received": True,
        "concrete_defect_confirmed": False, "author_session_id": "synthetic-author",
        "clarification_id": "synthetic-clarification",
    }
    manifest = tmp_path / "corrections.json"
    def save(entries=None):
        manifest.write_text(json.dumps({"schema_version": MANIFEST_VERSION,
                                        "withdrawals": entries if entries is not None else [entry]}))
    save()
    return ledger, manifest, entry, save


def test_exact_record_filter_preserves_other_flags_and_raw_history(fixture):
    ledger, manifest, _, _ = fixture
    before = ledger.read_bytes()
    verified = load_withdrawals(manifest, {ledger: [A, B]})
    excluded = set()
    for number, raw in enumerate(before.splitlines(keepends=True), 1):
        record = json.loads(raw)
        if record.get("exclude_pending_review") is True:
            scope = record_scope(record, [A, B])
            if not verified.should_withdraw(ledger, number, raw, scope):
                excluded.update(scope)
    assert excluded == {A}
    assert ledger.read_bytes() == before
    assert verified.counts() == {"verified_withdrawals": 1, "unique_affected_ids": 2,
                                 "author_ledgers": 1, "clarification_receipts": 1}
    target = before.splitlines(keepends=True)[0]
    assert not verified.should_withdraw(ledger, 3, target, [A, B])
    assert not verified.should_withdraw(ledger, 1, target + b" ", [A, B])
    assert not verified.should_withdraw(ledger, 1, target, [A])


@pytest.mark.parametrize("mutation,code", [
    (lambda entry: entry.update(question_ids=[A]), "target_affected_ids_mismatch"),
    (lambda entry: entry.update(record_sha256="0" * 64), "target_record_hash_mismatch"),
    (lambda entry: entry["clarification_receipt"].update(sha256="0" * 64), "target_record_hash_mismatch"),
    (lambda entry: entry["clarification_receipt"].update(line=1), "receipt_overlaps_target"),
    (lambda entry: entry.update(approved_by="author"), "invalid_clarification_receipt_binding"),
    (lambda entry: entry.update(concrete_defect_confirmed=True), "invalid_clarification_receipt_binding"),
    (lambda entry: entry.update(phase="after_collection"), "only_precollection_withdrawals_supported"),
])
def test_binding_and_approval_failures(fixture, mutation, code):
    ledger, manifest, entry, save = fixture
    mutation(entry)
    save()
    with pytest.raises(CorrectionError, match=code):
        load_withdrawals(manifest, {ledger: [A, B]})


def test_collected_records_never_revived(fixture):
    ledger, manifest, _, _ = fixture
    with pytest.raises(CorrectionError, match="collected_issue_requires_separate_reversal"):
        load_withdrawals(manifest, {ledger: [A, B]}, already_collected=[RecordKey(ledger, 1, "0" * 64)])


def test_allowlist_and_unscoped_records_fail_closed(fixture):
    ledger, manifest, entry, save = fixture
    with pytest.raises(CorrectionError, match="target_not_in_author_ledger_allowlist"):
        load_withdrawals(manifest, {})
    raw = line({"exclude_pending_review": True, "note": "Generic unreviewed limitation."})
    ledger.write_bytes(raw + ledger.read_bytes().splitlines(keepends=True)[1])
    entry["record_sha256"] = digest(raw)
    save()
    with pytest.raises(CorrectionError, match="record_scope_missing"):
        load_withdrawals(manifest, {ledger: [A, B]})


def test_whole_batch_validates_and_exact_duplicates_do_not_double_credit(fixture):
    ledger, manifest, entry, save = fixture
    save([entry, copy.deepcopy(entry)])
    assert load_withdrawals(manifest, {ledger: [A, B]}).counts()["verified_withdrawals"] == 1
    bad = copy.deepcopy(entry)
    bad["question_ids"] = [A]
    save([entry, bad])
    with pytest.raises(CorrectionError, match="target_affected_ids_mismatch"):
        load_withdrawals(manifest, {ledger: [A, B]})


def test_conflicting_receipts_for_one_record_fail_closed(fixture):
    ledger, manifest, entry, save = fixture
    alternate = line({"clarification": "Another synthetic receipt"})
    ledger.write_bytes(ledger.read_bytes() + alternate)
    other = copy.deepcopy(entry)
    other["clarification_receipt"] = {"path": str(ledger), "line": 4, "sha256": digest(alternate)}
    save([entry, other])
    with pytest.raises(CorrectionError, match="conflicting_duplicate_withdrawal"):
        load_withdrawals(manifest, {ledger: [A, B]})


def test_review_only_record_is_not_an_exclusion(fixture):
    ledger, manifest, entry, save = fixture
    raw = line({"scope_ids": [A, B], "review_recommended": True, "exclude_pending_review": False})
    ledger.write_bytes(raw + b"".join(ledger.read_bytes().splitlines(keepends=True)[1:]))
    entry["record_sha256"] = digest(raw)
    save()
    with pytest.raises(CorrectionError, match="target_is_not_author_exclusion"):
        load_withdrawals(manifest, {ledger: [A, B]})


def test_id_list_exclusion_verifies_and_unrelated_append_is_safe(fixture):
    ledger, manifest, entry, save = fixture
    raw = line({"exclude_pending_review": ["0000101225", "0000101226"]})
    ledger.write_bytes(raw + b"".join(ledger.read_bytes().splitlines(keepends=True)[1:]) + line({"note": "later"}))
    entry["record_sha256"] = digest(raw)
    save()
    assert load_withdrawals(manifest, {ledger: [A, B]}).counts()["verified_withdrawals"] == 1


def test_incomplete_receipt_fails_and_cli_outputs_only_counts_or_codes(fixture, tmp_path, capsys):
    ledger, manifest, entry, save = fixture
    allowlists = tmp_path / "allowlists.json"
    allowlists.write_text(json.dumps({str(ledger): [A, B]}))
    argv = ["--manifest", str(manifest), "--job-allowlists", str(allowlists)]
    assert _cli(argv) == 0
    assert json.loads(capsys.readouterr().out)["verified_withdrawals"] == 1
    receipt = line({"note": "Do not expose this payload"})[:-1]
    ledger.write_bytes(ledger.read_bytes().splitlines(keepends=True)[0] + receipt)
    entry["clarification_receipt"]["sha256"] = digest(receipt)
    save()
    assert _cli(argv) == 2
    assert json.loads(capsys.readouterr().out) == {"error_code": "incomplete_target_record"}
