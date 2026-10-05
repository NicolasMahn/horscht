# External benchmark isolation

External test sets live under `data/external_evaluation/`. That tree is ignored
by Git and is never a default training input. The snapshot builder rejects
paths from that tree. Raw downloads, canonical questions, gold labels, and
model predictions stay there; docs, tests, reports, and agent-visible output
contain metadata and aggregates only.

Each source release is pinned to an official revision and URL. Its immutable
manifest records the source checksum, byte count, data license, official
metadata link, and any separately licensed code used by an adapter. A
deterministic adapter writes a canonical MCQ file and a second manifest with
the adapter version and aggregate row counts. The adapter must copy only the
canonical ID, context, question, choices, answer index, and optional public
language/category fields. Source-only explanations, worked solutions, and
other metadata must not be copied into model input.

The inference runner uses the same `experiment.format_input` function as
training. It passes only context, question, and choices to the backbone; the
answer index, item ID, language, and category stay out of the prompt. The
fixed external protocol keeps choices in canonical source order, applies no
permutation, refuses truncation above the run's 10,240-token limit, and scores
the selected decision-head checkpoint over the valid masked-choice logits.
It reports accuracy, one-hot negative log likelihood, and multiclass Brier
score overall and by pinned source slice, alongside the expected metrics for a
uniform-choice baseline. No calibration or temperature scaling is applied.
The translated Belebele language slices are reported separately as related
variants.

`evaluation/protocols/blind-mcq-v1.json` is the predeclared protocol. Before
opening the canonical item file, the runner requires the locked system-test
manifest, both completed development-selected checkpoints, and the
coordinator's committed internal-development reflection receipt. Its separate
blind evaluation lock binds the protocol and bundle checksums, receipt,
selected checkpoint manifests and payload hashes, model revision, tokenizer,
and evaluation-code hashes. The runner never publishes per-item predictions,
identifiers, or errors. Cursor files contain only the committed item ordinal
and aggregate sufficient statistics. A graceful stop resumes at the next
committed ordinal; an unclean stop with an in-flight marker fails closed
instead of replaying that item. Final aggregate metrics are written only when
both candidates finish the full locked item set.

After the receipt and GPU release are in place, verify readiness without
starting inference:

```bash
./.venv/bin/python -B -m evaluation.blind_runner --prepare-only
```

The authorized final run is:

```bash
./.venv/bin/python -B -m evaluation.blind_runner
```

A graceful pause is requested with `SIGTERM`; continue using `--resume`. A
later reuse of the external test set must be labeled exploratory.

The overlap audit compares exact normalized context/question/choice
fingerprints and runs a bounded lexical near-duplicate search. It writes only
counts and training IDs: exact matches go to a future-snapshot exclusion
overlay, while near matches are quarantined for review. It never writes
benchmark item IDs or text. The near search is approximate; these checks do not
rule out semantic overlap or pretraining contamination.

Tests use synthetic rows only. Do not add real benchmark examples, previews,
per-item results, or payload-bearing exception text to repository files or
chat output.

Before any external inference, the scoring declaration was extended to include
probability scores and uniform baselines. The superseded protocol hashes were
`94973896924ddcd85af4c0e3a5b43a68d90a10408abc47c455edcf9b41276d79` and
`1fada4cd8312832b02c92da70b4a8a60a5b87b5cee24a35a4b9c316f733f6c34`.
The current protocol SHA-256 is
`ebb801b6b364e6307cc8c34b8ff76b8bcbc686c6cf2b8939f0d7860dbd08e341`.
