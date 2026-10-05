`dataset_factory.authoring_writer` builds metadata and validates labels for individually authored drafts under the existing probability-v2 contract. It accepts one job and one complete JSONL draft file, containing 1–8 rows per invocation. Eight rows is the authoring-turn limit; it does not establish semantic quality or language fluency.

```bash
python -m dataset_factory.authoring_writer append --job runs/overnight-2026-09-30/jobs/new-000.json --drafts /path/to/drafts.jsonl
```

Every draft requires these fields:

```json
{
  "assignment_id": "<exact ID from the job's remaining_ids>",
  "context": "<individually authored context>",
  "question": "<individually authored question>",
  "options": ["<first option>", "<second option>", "<localized abstention option>"],
  "explanation": "<individually authored explanation>",
  "abstention_index": 2,
  "abstention_text": "<localized abstention option>",
  "distribution_evidence": {
    "kind": "exact_derived",
    "probabilities": [
      {"numerator": 1, "denominator": 1},
      {"numerator": 0, "denominator": 1},
      {"numerator": 0, "denominator": 1}
    ]
  }
}
```

The example describes the file structure, not reusable question content. Authors supply all prose and evidence. `abstention_index` is zero-based, and `abstention_text` must exactly equal that option. The helper checks the explicit designation and target alignment; it cannot determine whether the text expresses abstention correctly in the assigned language. Options must keep the assigned count and ordering. All label vectors refer to that same order.

- `exact_derived` and `explicit_stochastic_process` require `distribution_evidence.kind` matching the source and a rational `probabilities` vector. The helper derives the target distribution.
- `empirical_frequency` requires evidence with `kind`, `counts`, and `sample_size`. The helper derives the target distribution. It does not verify claims that observations occurred.
- `teacher_estimate` requires evidence with `kind`, `model_id`, `model_version`, and `estimation_method`, plus an explicit `target_distribution` vector. It does not verify calibration.
- `hard_label` requires `answer_index`, and forbids `distribution_evidence` and `target_distribution`.

The helper obtains IDs, language, source, answerability, coverage fields and generator metadata from the exact matching row in the job's local plan. It never pairs drafts with assignments by ordinal position. Drafts cannot override these fields. Each source row retains a deep copy of the plan in `_assignment_metadata`, and sets `semantic_review_status` and `language_review_status` to `author_self_review_only`; these fields do not claim independent correctness. For answerable assignments, abstention must receive zero target mass or remain unselected for hard labels. Unanswerable assignments must target abstention deterministically. No labels or abstention options are supplied automatically to repair a draft.

The job requires `plan`, `output`, `skip_file`, `language`, `source_kind`, and `remaining_ids`. If `plan_sha256` is present, the local plan must match it before writing. Relative job paths resolve against the repository root. The launch allowlist remains fixed. Each invocation checks the job's own output and skip ledgers to find already completed IDs; it does not scan the corpus or update plans, manifests, progress, or issue reports.

All drafts are validated and serialized before any source rows are appended. A shared lock on the output path coordinates append and skip actions. Both ledgers must contain intact, complete JSONL lines and unique IDs from this plan. Appends preserve existing bytes and sync each completed row to disk, including the directory entry after the first row. An identical retry returns a no-op; a retry with different content fails. An interrupted filesystem write may leave a complete prefix or an incomplete final line. The helper refuses incomplete boundaries and does not repair or overwrite raw data.

Explicit skips use a separate JSONL file with only `assignment_id` and a nonempty `reason`:

```bash
python -m dataset_factory.authoring_writer skip --job /path/to/job.json --drafts /path/to/skips.jsonl
```

A skip cannot replace a written source row, and an append cannot revive a skipped assignment. Skip retries with the same reason are no-ops. Changing an existing skip reason fails. New skip records contain `id`, `assignment_id`, `status: "skipped"`, `reason`, `language`, and `source_kind`. Existing skip records may identify an assignment through `id`, `assignment_id`, or `plan_id`; their bytes are preserved.

Success prints only `{"appended": N, "already_present": N}`. Failure exits with code 2 and prints only `{"error_code": "..."}`. Python callers use `append_drafts(Path(job), Path(drafts))` or `skip_assignments(...)`; failures raise `WriterError` with the same content-free `.code`.

Read-only reconciliation uses the same job ownership and optional checksum checks:

```bash
python -m dataset_factory.authoring_writer status --job /path/to/job.json --next-count 8
```

`written`, `skipped`, and `unaccounted` count unique IDs from the job's fixed launch allowlist. Earlier saved rows outside that allowlist do not enter these counts. `next_ids` contains up to eight unaccounted IDs in local plan order. Counts reflect complete saved ledger rows and explicit skips, not semantic quality, distribution correctness, or training eligibility. A job is `complete` only when all allowlisted IDs are accounted for in an intact snapshot.

`output_state` and `skip_state` report `missing`, `empty`, or `complete` boundaries. A writer-held lock or a changing snapshot produces `state: "in_flight"`, `snapshot_available: false`, null counts, no next IDs, and `complete: false`. An incomplete final line similarly reports `state: "incomplete_boundary"` and withholds counts. Duplicate IDs, unknown IDs, conflicting skip identities, and an ID appearing in both source and skip ledgers fail with a content-free error code. Status creates no files, repairs no boundaries, and does not update plans or progress.

A dispatcher keeps a local manifest separating jobs it has launched from jobs still queued:

```json
{
  "launched": ["runs/overnight-2026-09-30/jobs/new-000.json"],
  "queued": ["runs/overnight-2026-09-30/jobs/new-001.json"],
  "lane_limit": 16
}
```

```bash
python -m dataset_factory.authoring_writer cohort --manifest /path/to/dispatcher.json --next-count 8
```

The command reports each job's exact status, `unfinished_launched_jobs`, and `available_slots`. Repeated launched paths, including symlink aliases, count once. A path cannot be both launched and queued. Different jobs cannot own overlapping output/skip paths or assignment IDs. Lane limits are 1–16; the default is 16. Available slots are the limit minus unique unfinished launched jobs, clamped at zero. A completed job appearing twice cannot free two slots. Queued jobs do not occupy a launched slot.

An absent, idle, initializing, or pending-init agent does not complete its job. Any unfinished launched job retains its slot, including jobs with missing runtime information, a held append lock, or incomplete boundaries. These reports do not authorize an overlapping authoring turn: dispatchers must still observe the prior agent turn's actual idle/completed state before sending a next-eight follow-up. The command reads only manifest-listed jobs and their local plans/output/skip ledgers; it does not inspect runtime sessions or scan global history. Repeated reporting never credits completion cumulatively.

Python callers use `job_status(Path(job), next_count=8)` and `cohort_status(Path(manifest), next_count=8)`. Results contain IDs, paths, scalar counts, and boundary states; errors use the same `WriterError.code` contract. Neither command emits question content or performs an LLM review.
