# Horscht probability-v2 continuation handoff

Prepared: 2026-09-29

Superseded status: Nicolas explicitly authorized resumed generation and both larger training arms on 2026-09-30. The old coordinator CLI remains stopped. Current ownership and actual counts are in [the overnight plan](overnight-plan-2026-09-30.md), [live status](../runs/overnight-2026-09-30/status.json), and [production progress](../data/generated_train/probability-v2/progress.md). The paused state below is a preserved historical handoff, not the current launch gate.

## Current state

The project is at `/home/nicolas/Projects/horscht`. Probability-v2 is the active dataset factory. Generation remains paused since 2026-09-29T20:49:21Z; live workers are **0**. The frozen saved-data snapshot captured 2026-09-29T21:02:48Z has **88,355** structurally accepted production rows out of 1,000,000, leaving **911,645**. This is 42,708 soft rows plus 45,647 hard rows (36,177 legacy + 9,470 new). Outstanding saved assignments: 6,849, all corrected-policy; unassigned target balance: 904,796. Saved manifests remain unchanged; outstanding assignments are interrupted/planned work. JSONL integrity: 1,579 saved output, assignment, skip, manifest and issue files checked; 792 question files scanned. Malformed JSON rows: **0**; unterminated final lines: **0**; malformed partial final lines: **0**. There are 54 unfinished output shards: 1 saved file with outstanding assignments and 53 files not yet created. This describes assignment completion, independently of JSONL integrity. Raw rows and assignment/wave manifests were preserved; rejection references are report-only.

Current progress is [data/generated_train/progress.md](../data/generated_train/progress.md), with detail in [data/generated_train/probability-v2/progress.md](../data/generated_train/probability-v2/progress.md) and machine counts in [data/generated_train/probability-v2/progress.json](../data/generated_train/probability-v2/progress.json). The five-row Spanish contract preflight in [data/generated_train/probability-v2-pilot-cardinality-answerability-v1.jsonl](../data/generated_train/probability-v2-pilot-cardinality-answerability-v1.jsonl) passed schema, arithmetic, cardinality, paired-group, and slot-264 checks. It remains excluded from the million-row total; semantic quality and fluency were not independently reviewed. Generation remains paused. Before resuming, obtain explicit authorization after the system update, verify remote child sessions are inactive, and reconcile saved assignment/wave statuses against accepted outputs and skips. Continue only from intact saved line boundaries, keep raw source rows immutable, and omit `exclude_pending_review` IDs from training until resolved.

Schema and planner corrections already landed. `probability_v2` accepts 2–265 choices. New assignments use a versioned low/medium/high cardinality policy and a configurable 80% answerable / 20% unanswerable mix. The localized task-grounded “cannot be determined” choice is included in, not added on top of, the 265 maximum. Answerable rows give it zero target mass; unanswerable rows select it one-hot. Paired context variants must share a problem group and split. Known stochastic outcome distributions remain answerable.

The training runner already supports 265 logits with per-row masks, soft categorical targets, hard one-hot targets, cross-entropy, and Brier score. The 265-option regression checks padding masks, target permutation, gradient/update at slot 264, and checkpoint reload. The full project suite now passes 153 tests.

The illustrative reward policy is +1 for a correct substantive answer, −4 for a wrong answer, and 0 for an external confidence rejection on answerable tasks. The resulting 0.8 threshold assumes calibrated correctness probabilities; it is configurable and not tuned. A task-grounded IDK class is distinct from external confidence rejection. Report proper probability scores separately from coverage, selective risk, IDK recall, false-IDK behavior, expected utility, and always-answer/all-reject baselines.

`dataset_factory.choice_augmentation` is an opt-in, proof-gated helper. It creates a new child row without mutating its source; only exact/stochastic closed-world rows can get exact zero-mass additions, and every result is marked `requires_review`. No existing questions have been augmented. The authoring brief is [probability-v2-authoring-guidance-v2.md](probability-v2-authoring-guidance-v2.md). The current language-centering embedding audit did not justify generation changes.

## Active ownership

The current old session has no remaining bounded workers:

- `pilot_cardinality_answerability` completed `data/generated_train/probability-v2-pilot-cardinality-answerability-v1.jsonl`.
- `decision_reward_metrics` completed `dataset_factory/decision_metrics.py`, `tests/test_decision_metrics.py`, and `docs/probability-v2-decision-metrics.md`.

No bulk writers are active. The last old shard writer stopped at a closed boundary with zero additional rows. This old session advertises 33 agents total. The parent confirmed a fresh Luna CLI runtime capacity of 129 total slots (128 subagents) and started its production coordinator at thread `01a0ede9-90a5-7751-9a7b-de0d5346a308`. Continue there with Luna-only model work.

## Pilot gate and continuation

The pilot gate is complete. It covers low, medium, and exactly 265 choices, with IDK inside the 265; both answerability states; a paired same-group/same-split context; stable choice IDs; target/evidence alignment; and nonzero target mass at slot 264 on the 265-choice answerable example. The deterministic summary accepted 5, rejected 0, and excluded all 5 from production. No independent semantic review was done. Decision metrics are implemented and tested for expected utility, coverage, proper scores, IDK diagnostics, and baselines. The new module does not yet report selective risk or a risk–coverage curve; add those in the fresh session. The old-session marker is `runs/luna-coordinator-2026-09-29/old-workers-complete.md`.

```sh
.venv/bin/pytest -q
.venv/bin/python -m dataset_factory.probability_v2 summary data/generated_train/probability-v2-pilot-cardinality-answerability-v1.jsonl
```

Use a small frozen-backbone training smoke only if the pilot survives schema, arithmetic, and slot checks. Keep pilot data excluded from the million-row count.

For fresh assignments, use `dataset_factory.probability_v2_assignments` with the versioned v2 catalog, cardinality policy, answerability policy, and seed-word pool. Check the supported command syntax with:

```sh
.venv/bin/python -m dataset_factory.probability_v2_assignments --help
.venv/bin/python -m dataset_factory.probability_v2_assignments plan --help
```

Allocate varied task families, domains, language tags, registers, difficulty bands, context lengths, and choice counts. Keep output language explicit and one language per batch; do not send English fallback for a non-English assignment. The current progress snapshot has 144 language tags with valid saved rows or outstanding assignments and 68 tags marked unsupported; do not claim fluency or coverage of all world languages. Seed words in `data/generated_train/seed-word-pool.jsonl.gz` are generator-only cues, not required prompt words or labels. Continue to group translations, paraphrases, and counterfactual variants before splitting. Avoid repeated templates and decorative topic swaps.

Have each Luna writer own a disjoint assignment shard and write durable JSONL directly to its assigned path. Resume only from the last complete line; malformed partial lines go to quarantine and never count as accepted. Keep source rows immutable. Do not count illustrative pilots, shuffles, or translations as independent questions. Track accepted, rejected, skipped, active, and remaining-to-accept counts separately.

Refresh production progress from one frozen snapshot. At this snapshot accepted remaining is 911,645 and unassigned balance is 904,796, because 6,849 saved assignments remain outstanding. The canonical current machine report is `data/generated_train/probability-v2/progress.json`; the top-level progress.json remains historical broad-v1. For hard-label arithmetic, accepted = soft + total hard; new hard = total hard − legacy hard; remaining = 1,000,000 − accepted.

The [known-issues sidecar](../data/generated_train/probability-v2/known-issues-v1.jsonl) is applied by accepted-ID intersection. Known-issue sidecar flags 2,300 unique IDs as `exclude_pending_review`; 1,020 intersect structurally accepted production rows. Training-eligible rows after these exclusions: **87,335**. `review_recommended` covers 10,073 unique IDs, all 10,073 structurally accepted and 10,073 training-eligible; these review flags are separate from exclusions. Neither training eligibility nor worker-reported flags establish semantic correctness or language fluency. Preserve raw rows and omit excluded IDs from future training until resolved.
