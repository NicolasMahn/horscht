`dataset_factory.issue_corrections` verifies centrally approved withdrawals of
specific author issue records before collection. It reads only explicit local
metadata paths. It does not modify raw issues, questions, plans, exclusions, or
progress. It does not classify author prose or review semantic correctness.

Normalize author scope using the job's canonical assignment-ID allowlist:

```python
scope = record_scope(record, allowed_job_ids)
```

Recognized list fields are `scope_ids`, `ids`, `question_ids`, and nonboolean
`exclude_pending_review` / `review_recommended`. Recognized scalar fields are
`id`, `assignment_id`, and `question_id`. Exact canonical IDs must belong to the
allowlist. An exact ten-digit suffix must match exactly one allowlisted ID
ending in `-<suffix>`. Multiple explicit fields must normalize to the same set.
Malformed, missing, conflicting, ambiguous, or out-of-job scope raises a
content-free `CorrectionError`. Boolean flags alone provide no scope.

An unsupported or missing scope must remain conservatively excluded and be
reported by the caller. Never treat a normalization failure as an empty scope
or a withdrawal. Existing all-job fallback exclusions already collected into
sidecars remain independent until the coordinator validates a separate narrow
reconciliation. This module provides no reversal of collected exclusions.

The coordinator owns a manifest of this form (example metadata is synthetic):

```json
{
  "schema_version": "author-issue-corrections-v1",
  "withdrawals": [{
    "phase": "before_collection",
    "ledger_path": "local/author-issues.jsonl",
    "line": 16,
    "record_sha256": "<64 lowercase hex digits>",
    "question_ids": ["probability-v2-assignment-0000101225"],
    "clarification_receipt": {
      "path": "local/author-issues.jsonl",
      "line": 17,
      "sha256": "<64 lowercase hex digits>"
    },
    "decision": "withdraw_exact_record",
    "replacement_training_status": "review_recommended",
    "approved_by": "central_coordinator",
    "author_clarification_received": true,
    "concrete_defect_confirmed": false,
    "author_session_id": "author-session-reference",
    "clarification_id": "clarification-reference"
  }]
}
```

Both hashes cover the exact complete JSONL line bytes **including the newline**.
Paths resolve relative to the repository root by default; `root=` can override
that for local fixtures. The clarification may be another line in the same
ledger. Its bytes are checked as a complete JSON object, without interpreting
its prose. The coordinator's attestation authorizes the correction; hashes bind
artifacts and do not prove authorship or semantic truth. Each individual flag
record requires its own exact manifest entry. No per-ID revival list exists.

Integration is a filter before collecting an author record, never subtraction
from an accumulated ID set:

```python
from dataset_factory.issue_corrections import load_withdrawals, record_scope

verified = load_withdrawals(
    manifest_path,
    {author_ledger_path: allowed_job_ids, other_author_flag_path: allowed_job_ids},
    already_collected=collected_record_keys,
)
# During the caller's explicit author-ledger pass:
scope = record_scope(record, allowed_job_ids)
if not verified.should_withdraw(path, line_number, raw_line, scope):
    excluded_ids.update(scope)
```

The caller must invoke this only for exclusion-bearing records, retain all
other genuine flags, and handle scope failures conservatively. Supply any known
collected records as `RecordKey(path, line, sha256)`; matching path/line is enough
to block withdrawal even if its hash changed. Independently retain all existing
known-issue sidecar exclusions. The complete manifest validates before any index
is returned. Repeated identical entries count once; conflicting bindings fail.

For a read-only count check, a dispatcher-local allowlists JSON object maps
each explicit author-ledger path to its canonical job IDs:

```sh
python -m dataset_factory.issue_corrections \
  --manifest local/corrections.json \
  --job-allowlists local/author-ledger-job-ids.json
```

The command emits only counts or `error_code`. It does not discover collected
history; production integration must supply that through the API and retain
existing sidecars. Required inspection is bounded metadata: explicit ledger
paths, target and receipt line numbers/hashes, each job allowlist, normalized
scope counts, and whether the exact records were already collected. Question
payloads and global corpus inspection are unnecessary.
