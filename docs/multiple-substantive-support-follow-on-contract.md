`multiple_substantive_support-v1` is an optional requirement for newly planned,
answerable soft tasks. It is a bounded refinement based on internal authoring
diagnostics. It does not imply the earlier v3 adoption window has completed, and
does not modify that window, its artifacts, old plans, source quotas, or frozen
training/evaluation code.

The separate coordinator-owned profile uses the existing artifact schema:

```json
{
  "schema_version": "probability-v2-planning-profile-v1",
  "name": "multiple-substantive-support-v1-pilot",
  "start_index": 128513,
  "max_assignments": 1024,
  "authoring_guidance_version": "support-pilot-v1",
  "bindings": {
    "guidance": {"path": "<separate local guidance path>", "sha256": "<64hex>"},
    "decision": {"path": "<separate local decision path>", "sha256": "<64hex>"}
  }
}
```

The reserved window is 128513–129536 inclusive. First use must match the next
unused historical index. Whole-batch cap checks precede shard writes. Saved
bindings cannot be changed or reanchored, skipped reservations still consume
positions, and profile shards cannot be extended on resume. Guidance and
decision hashes are verified opaquely before commits. A subsequent receipt may
bind the profile; the profile does not bind that receipt.

Explicit follow-on profiles use `multiple-substantive-support-v1-follow-on-<start_index>`
with exactly `max_assignments=4096`. The canonical numeric suffix must equal
`start_index` and be at least 129537. The first name is
`multiple-substantive-support-v1-follow-on-129537`, reserving 129537–133632
inclusive. Each later window has a distinct name and coordinator-owned
guidance/decision/profile artifacts are separate from the original pilot.
The old pilot retains its exact start and limit. Each follow-on requires the
next unused global index and cannot overlap a saved profile window or rebind an
existing name. A subsequent 4,096-row window requires a new hashed artifact and
explicit allocation; there is no automatic continuation or unlimited profile.
The follow-on uses
the same 25% selection, form/structure/source matrix, helper checks, omission
reports, and unchanged source quotas. It does not assert semantic correctness or
completion of earlier generation windows. Saved older plans remain immutable.

Use the existing append-safe, single-snapshot batch API:

```python
write_assignment_batch(
    specs,
    planning_profile_path=Path("runs/overnight-2026-09-30/planning-profile-support-pilot-v1.json"),
    # Existing explicit paths and other planning arguments remain available.
)
```

Or use `plan-batch --spec FILE --planning-profile FILE`. Both actually allocate
plans when invoked. `probability_task_compatibility(catalog, source_kinds)` is a
read-only metadata API for inspecting omissions before any allocation; it
requires only a loaded catalog and explicit source kinds, with no corpus scan.

For this pilot, an independent assignment seed selects 25% of answerable soft
assignments. This is a probability, not an exact finite-batch quota. Marked tasks
use seeded 3 or 4 total choices, including localized IDK, after selecting a
compatible form and structure:

| Source | Form | Operations | Structures |
| --- | --- | --- | --- |
| exact / empirical | finite_population_distribution | taxonomy_outliers, policy_exceptions | inventory_ledger, table_or_grid, taxonomy_key, rule_excerpt |
| stochastic process | stochastic_outcome_distribution | game_sport_rules, physical_mechanisms | state_transition_table, table_or_grid, rule_excerpt |
| teacher estimate | estimated_hypothesis_distribution | argument_evidence, experiment_design | claim_evidence_bundle, measurement_sheet, multi_document_packet |

The selected form asks for a probability distribution over a mutually exclusive,
exhaustive partition of substantive outcomes under stated conditioning and
mechanism assumptions. It must not ask for an unknown realized draw or a uniquely
optimal action. Finite-population probabilities use the specified population
and selection rule; empirical proportions use the described observed sample;
stochastic tasks use the given draw/transition law; teacher tasks use supported
alternative hypotheses without claiming independent calibration. An author
must explicitly skip an impossible assignment rather than pad options or invent
evidence to satisfy its target shape.

An unsupported source/catalog combination is reported as
`unsupported_distribution_task` in the preallocation compatibility report.
Selected unsupported cases remain explicitly unmarked, with that reason.
Other unmarked reasons are `hard_label`, `unanswerable`, and `not_selected`.
No probabilities are fabricated and no task is silently relabeled as a law
question. Unmarked tasks retain default v2 sampling. True unanswerable tasks,
including unknown realized draws, retain the original IDK one-hot requirement.

Marked plans carry `_planner.target_shape_requirement` equal to
`multiple_substantive_support-v1`, a law form with
`question_kind=outcome_distribution`, and the full pinned profile metadata.
Drafts use the existing writer schema. Their `distribution_evidence` additionally
requires:

```json
{
  "question_kind": "outcome_distribution",
  "mechanism_assumptions": ["<explicit assumptions also stated in the context>"],
  "outcome_partition": "<description of the complete substantive partition>"
}
```

These fields accompany the existing exact rational probabilities, empirical
counts/sample size, or teacher metadata/explicit estimate. The helper first uses
the unchanged v2 builder, then requires at least two strictly positive
substantive target entries and exactly zero IDK mass. It never adjusts, smooths,
normalizes, or redistributes an invalid target to make it pass. An unsupported
requirement, marked hard/unanswerable/two-choice plan, missing law metadata,
missing assumptions/partition, or one-hot marked target fails before append.
Entire batches still validate before any write, and retries preserve old bytes.

Structural checks cannot establish that assumptions are complete, the partition
is meaningful, categories are mutually exclusive, or tiny positive masses are
justified. Review statuses remain `author_self_review_only`. This contract is
not independent semantic or language review.

The planner returns `target_shape_planning` with
`preallocation_compatibility`, `planned_marked`, `selection_status_counts`, and
`selection_status_counts_by_source` for batches. These describe plans, not
observed targets. For pilot append calls, the helper additionally emits scalar
`soft_appended`, `answerable_soft_appended`,
`multiple_substantive_support_appended`, and
`required_multiple_support_appended`. Retry calls report zero new contributions.
Old/unmarked non-pilot calls retain the existing two-count result.

The global source allocation remains 80% soft format, including legitimate
one-hot soft rows, and 20% hard labels. A local pilot can contain only soft-source
shards without changing those global targets. Report actual multiple-support
share separately among eligible soft rows and answerable eligible soft rows,
with source breakdowns and explicit before/after-dedup scope. A planned marker
is not a collected nondegenerate original and must never be counted as one.
