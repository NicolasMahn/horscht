---
pretty_name: Horscht decision training data
license: other
license_name: horscht-source-specific-licenses
license_link: LICENSE
task_categories:
  - multiple-choice
  - text-classification
tags:
  - decision-model
  - soft-targets
  - synthetic
  - noncommercial-component
configs:
  - config_name: warmstart
    data_files:
      - split: train
        path: data/warmstart/train.parquet
      - split: validation
        path: data/warmstart/validation.parquet
  - config_name: continuation
    data_files:
      - split: train
        path: data/continuation/train.parquet
      - split: validation
        path: data/continuation/validation.parquet
---

This release contains the training and development inputs for Nicolas Mahn's
Horscht decision-model experiments. It includes both the corpus used to train
the warm-start checkpoint and the later 23,006-row continuation mixture. It is
a source-specific licensed collection; SciQ rows are noncommercial and
Wikipedia-derived rows carry share-alike terms. Read [LICENSE](LICENSE) and
[NOTICE.md](NOTICE.md) before reuse.

| Configuration | Frozen version | Train | Validation/development | Sealed internal test | Sealed calibration |
|---|---|---:|---:|---:|---:|
| warmstart | 2026-10-01-v1 | 128,263 | 7,202 | 7,164 | 0 |
| continuation | 2026-10-08-v4 | 23,006 | 11,339 | 8,333 | 1,045 |

The sealed columns describe original snapshot sizes only; their content and
identities are absent from the public repository. External benchmarks, gold
answers, per-item reports, authoring logs, quarantines and raw source caches
are also absent. Training/dev assignments and related groups are preserved
from the authoritative frozen split registry. Validation has experimental
history and is intended for development, not a fresh independent test claim.

| Continuation source | Train | Validation |
|---|---:|---:|
| Original generated replay | 8,000 | 7,202 |
| Independently reviewed grounded documents | 3,900 | 1,655 |
| Deterministic games | 5,332 | 1,898 |
| Typed Decisions public train | 1,954 | 199 |
| SciQ public train | 1,920 | 190 |
| Banking77 public train | 1,900 | 195 |

Replay deliberately repeats earlier rows. Do not concatenate these
configurations and treat repeated IDs as independent examples. The previous
2026-09-29-v1 snapshot (78,107 train; 4,354 development; 4,380 internal test)
was a historical source/grouping predecessor of the 2026-10-01-v1 corpus.
The model's direct warm-start run used 2026-10-01-v1, starting from
Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, and selected
update 8,017. The subsequent run reset its optimizer/scheduler and trained
on v4. Its selected published candidate is update 1,080. Dataset size is not
the number of row exposures at that checkpoint; the final training run
completed more updates than the selected checkpoint.

The warm-start corpus is original AI-assisted generated reasoning data; its
source manifest describes semantic quality as unreviewed despite structure
and distribution arithmetic checks. It contains 26,011 review-recommended
rows across the original full snapshot. Published availability is not a
claim that every question or target distribution is correct or calibrated.
New document questions passed independent semantic review; game targets use
deterministic engines. Typed Decisions' teacher probabilities are estimates.

Each Parquet row contains `context`, `question`, `options`, `choice_ids`,
`target_kind`, and either `answer_index` (zero-based) or
`target_distribution` (aligned to option order). `abstention_choice_id` is
retained when present. `id`, `problem_group_id` and `snapshot_group_id` preserve
identity and grouping; `source`, license fields and `provenance_json` describe
attribution and transformations. `source_row_sha256` binds the original JSON
record bytes without its final newline. `split` is the authoritative original
`train`/`dev` designation; Hugging Face labels `dev` as `validation`.

Full contexts are retained without cropping. The continuation training
cohort has 148 language tags, 2–265 choices, and 7,471 genuinely non-one-hot
targets. Its measured maximum complete formatted training input is 37,609
Qwen3 tokens; the experiment admits up to 40,960 tokens. These are training
cohort statistics, not guarantees for all formats, tokenizers or languages.
Typed Decisions' six-decimal values can sum to within 0.000002 of one;
raw values remain intact and training normalizes at the tensor boundary.

```python
from datasets import load_dataset

# Replace the repository ID with this dataset's published ID.
data = load_dataset("OWNER/HORSCHT-DATA", "continuation")
```

`manifest.json` records exact counts, source-specific license counts, original
snapshot/artifact hashes and exclusions. `checksums.json` records the SHA-256
of the release files. `public-source-manifest.json` pins native training
sources and acquisition hashes. `document-attribution.jsonl` records every
published document, including each component of composite packets, with
source URLs, pinned revisions, author credits and rights evidence. These
notices accompany the same provenance in exported rows.

To rebuild the public package from the frozen project workspace, install the
project's evaluation dependency group (PyArrow) and run
`uv run --group evaluation python scripts/package_hf_dataset.py --output /tmp/horscht-data-release`.
The packager verifies frozen manifests and row/split hashes, refuses unexpected
sources or group reassignment, and writes only train/development files.
The common schema intentionally excludes generation assignments, wordfreq
frequency lists and runtime/private paths. It preserves training input text,
option order and targets; exact authoring-log reconstruction is unsupported.

For raw public source reconstruction, use the repository, revision and
source_file in `public-source-manifest.json`, verify the raw SHA-256, and
apply `dataset_factory/new_public_data.py`'s converters. Those converters
acquire source-native train only. Frozen selection and option permutation
are recoverable from the released IDs and per-row provenance. For document
sources, use each notice's pinned revision/source URL and extraction record,
verify `content_sha256`, and comply with the source-specific item license.
Live source pages may have changed; a mismatch must not silently substitute
new text. Raw assets/media were not cleared or included. The wordfreq seed
pool remains excluded: reconstruct it only under its upstream source-specific
terms with version 3.1.1 and the original project seed-pool recipe. There is
no released artifact from which sealed calibration/test content can be rebuilt.

The deterministic privacy scan found no recognized credential formats or
Nicolas's private workspace paths in published input/explanation text.
Source document text contains some originally public contact addresses and
names; Typed Decisions contains email-shaped synthetic workflow identifiers.
They remain part of the frozen source text. This is pattern screening and
provenance review, not exhaustive personal-data detection. No private user
documents, mail, chat history or account credentials are included as sources.

No external benchmark content was inspected while preparing this release.
The original aggregate v4 overlap audit found zero exact or near lexical
overlaps against the project's external evaluation suite; it cannot exclude
semantic overlap or base-model pretraining contamination. The mixture includes
SciQ and Banking77 native train data, so evaluation on those dataset families
requires explicit independent split and overlap checks.
