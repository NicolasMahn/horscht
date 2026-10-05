The optional `dev-guidance-v3` planning profile applies to new original
assignments within one immutable 4,096-index adoption window. It keeps the v2
schema, source quotas, answerability policy, and default planner unchanged.
Existing plans keep their saved prompts and metadata.

The coordinator owns the profile artifact and its two opaque file bindings:

```json
{
  "schema_version": "probability-v2-planning-profile-v1",
  "name": "dev-guidance-v3",
  "start_index": 124417,
  "max_assignments": 4096,
  "authoring_guidance_version": "dev-guidance-v3",
  "bindings": {
    "guidance": {"path": "docs/probability-v2-authoring-guidance-v3.md", "sha256": "<64hex>"},
    "decision": {"path": "runs/overnight-2026-09-30/generation-decision-v3.json", "sha256": "<64hex>"}
  }
}
```

Paths resolve relative to the repository root. The loader verifies file hashes
without interpreting their content. The subsequent verification receipt can
bind the profile hash; the profile does not bind that receipt, avoiding a hash
cycle. `load_planning_profile(path)` validates these bindings without scanning
question or assignment history.

Use `planning_profile_path=Path(...)` with `write_assignment_batch` or
`write_assignment_shard`, or the additive CLI flag:

```sh
python -m dataset_factory.probability_v2_assignments plan-batch \
  --spec local/new-shards.json --seed 31 \
  --planning-profile runs/overnight-2026-09-30/planning-profile-v3.json
```

As with existing planning commands, this writes plans and progress when actually
run. First adoption requires `start_index` to equal the next unused historical
global index. The example window is 124417–128512 inclusive. The planner rejects
a batch crossing the window before any shard write. Reserved, skipped, failed,
and default assignments within the window consume index positions; they cannot
be reclaimed to extend adoption. Saved profile bindings prevent reanchoring or
editing the artifact for another 4,096 rows. Resume cannot select a profile or
extend an existing profile shard. Default v2 planning remains available after
the bounded window.

Each assignment's derived seed samples a choice band first: 70% for 2–4, 20%
for 5–16, 8% for 17–64, and 2% for 65–265. Every integer within a selected band
is equally reachable, including 265. These are sampling probabilities, not
exact finite-batch counts. The planner then samples among compatible operations
with weight 1.5 for `policy_exceptions`, `graph_kinship`,
`tradeoff_optimization`, and `procedure_recovery`, and 1.0 for all others.
It balances compatible domains within the selected operation.

Compatibility retains the existing v2 operation and input-structure sets. In
particular, `graph_kinship` has no wide-count eligibility; its priority weight
does not invent that support. Operation probabilities are conditional on the
sampled band, so overall operation ratios can differ from the nominal weights.
If the catalog lacks an eligible operation or structure for a band, the planner
explicitly omits that band and normalizes the remaining weights. It records
`effective_band_weights`, `omitted_bands`, `operation_weight`,
`operation_effective_probability`, and `omitted_priority_operations` in each
row's `profile_sampling` metadata. It never truncates a sampled count.

The 65–265 band records `complete_inventory_required=true`. This is an authoring
requirement, not proof that the eventual options form a meaningful complete
inventory. Metadata compatibility cannot establish semantic correctness. The
coordinator owns guidance adoption and its DEV-only decision; the planner does
not promote guidance from held evaluation metrics.

New rows retain the profile artifact hash, guidance/decision hashes, policy
bands and weights, guidance version, and adoption ordinal. The planner rechecks
the artifact and bindings before each commit; the coordinator must keep these
artifacts stable during planning. Source targets remain 300,000 exact, 300,000
stochastic-process, 100,000 empirical, 100,000 teacher, and 200,000 hard-label
originals. A profile changes neither usable-original accounting nor quotas.
