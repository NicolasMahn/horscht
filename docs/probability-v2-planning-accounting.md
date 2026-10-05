Probability-v2 planning now uses `probability-v2-usable-originals-v1` accounting. Quotas credit unique eligible originals, not raw structurally valid rows. The source targets remain unchanged, and saved rows, plans, IDs, skips, and issue ledgers are preserved.

Each planning snapshot reports separate counters:

| Counter | Meaning |
| --- | --- |
| `structurally_valid_rows_before_identity_checks` | Schema/math-valid rows before duplicate-ID and cross-split group checks. |
| `structural_production_rows` | Canonical accepted rows after identity/group checks and pilot exclusion. |
| `eligible_rows` | Structural production rows after `exclude_pending_review` IDs are removed. |
| `eligible_original_rows` | Eligible rows without explicit child/variant metadata. |
| `unique_eligible_original_rows` | Original candidates after grouped variants and exact-content copies are removed. This supplies quota credit. |

`review_recommended` alone does not remove eligibility; its eligible cohort is counted separately. Pending-review exclusions are unioned by question ID. A saved excluded output still closes its assignment and reserves its ID/group/index. Planning replacements allocate new identities rather than reopening or changing the excluded row.

Explicit child markers are `translation_of`, `parent_question_id`, `source_row_id`, `variant_of`, and `_augmentation`. Their rows remain visible as eligible variants but do not count as original progress. For unmarked candidates, the planner conservatively retains at most one original per `problem_group_id`. Distinct grouped counterfactuals or translations therefore do not independently fill the original target. Pairing metadata without a child marker does not itself remove an entire group.

Exact-content identity uses the existing `training_snapshot.exact_content_fingerprint`: normalized context and question plus option/target pairs, independent of option order and volatile metadata. Files are sorted by path, then read in line order. After exclusion and explicit-child filtering, the first unmarked candidate per group is retained, then the first occurrence of each exact fingerprint receives credit. Cross-source copies credit only the winning row's source. The chosen source is determined by stable input ordering, not by the current quota deficit.

This is deterministic identity accounting, not a semantic review or proof of independent questions. Unmarked paraphrases, translations, and other semantic copies can remain. It does not replace the training snapshot's split/group safeguards. Central progress computed from a frozen baseline plus bounded exclusions is not substituted for a newly verified unique count.

`production_eligible_rows` and `production_eligible_rows_by_source_kind` are compatibility aliases for the unique-original counters in this versioned planner snapshot. Raw `probability_v2.summarize` results lack this accounting version and cannot supply quota budgets. Counter types, per-source sums, stage transitions, and exclusions must agree; inconsistent snapshots fail closed. The planner sums remaining deficits per source. Extra accepted rows in one source cannot satisfy another source's quota or cause an early raw-count completion gate.

Planning performs one schema/fingerprint/accounting pass shared with `PlanningHistory`, using an 8 MiB SQLite cache and disk-backed identity/fingerprint tables. It reuses the resulting summary for every bounded batch shard and does not reparse after each shard. No question bodies are retained in the accounting database. The planner checks for at least 12 GiB available physical RAM before and periodically during scanning; it aborts if that reserve is unavailable.

Live writers may append while the scan runs. Before parsing, the planner captures each initially discovered question/skip file's last complete-line byte boundary and SHA-256 of that prefix. Parsing reads only those bounded bytes and compares their hash with the pinned value. Existing partial final lines are omitted, even if a writer completes them later. Source files may grow beyond the captured prefix. New files created after discovery are omitted from this snapshot. Their assignment IDs remain reserved through immutable plans and are counted as active commitments until a later scan observes their completed output/skip records, rather than receiving both new output credit and active-plan credit.

`question_prefixes` records each file's path, initial device/inode/size/timestamps, complete-line boundary, prefix SHA-256, verified size, growth bytes, and `unchanged` or `append_growth` state. `scan_interval` records ISO 8601 start/finish timestamps, and `new_files_omitted` records paths discovered after the initial snapshot. This is a collection of pinned file prefixes captured during the interval, not an atomic instant shared by every file.

Opaque byte-only hashing pins prefixes before parsing. After accounting, files with observed growth receive another bounded prefix hash check; unchanged files must retain their inode, size, and modification/change timestamps. This adds verification I/O without another semantic parse or classification pass. Replacement, removal, truncation, rewritten pinned bytes, and rewrite-plus-append fail closed. Append bytes beyond the boundary are deferred to a future snapshot and are not validated by this scan. The coordinator-owned exclusion sidecar must remain unchanged for the entire scan; its growth or rewrite still fails closed.

The exclusion sidecar defaults to `known-issues-v1.jsonl` beside the assignment directory. It is required for the canonical production assignment directory. An isolated custom dataset with no local sidecar has no known exclusions; callers can explicitly provide an empty sidecar to pin that fact. An explicitly supplied missing, malformed, incomplete, or changing sidecar always fails closed.

The existing commands accept `--known-issues-file`; Python APIs accept `known_issues_path`. For example, a coordinator can select the sidecar while directing a check report to a separate location:

```bash
python -m dataset_factory.probability_v2_assignments check \
  --known-issues-file data/generated_train/probability-v2/known-issues-v1.jsonl \
  --progress-file /path/to/coordinator-snapshot.json
```

Planning/check commands read the selected corpus and may write their configured progress report; batch planning also creates new plans. The central coordinator owns when these scans run and where reports or plans are written. The correction was verified with synthetic fixtures only; no production recount, quota replacement plan, central progress update, or training change was performed during implementation.
