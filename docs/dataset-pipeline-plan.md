# Generated decision data

The factory prepares and tracks Luna-authored, multiple-choice decision questions. Its current target is 1,000,000 training rows. The catalog contains 200 language tags, 80 broad domains, 24 reasoning operations, 20 input structures, and 15 registers, with 231 compatible operation/domain pairs. These are sampling choices, not claims that Luna can write every language well. Workers skip a language when they cannot author it confidently. The language tag and structural checks do not verify fluency.

The word-cue pool contains 420,000 `(word, source_language)` entries from `wordfreq==3.1.1`, with 336,330 distinct words. A reproducible assignment draw chooses 3–8 cues per question independently of its output language. Cues are generator-only inspiration; the output need not contain them. The base seed and per-question derived seed are recorded in each plan. This makes the assignments repeatable, not the model's generated wording.

Each assignment fixes a unique ID and problem-group ID before writing starts, as well as the output language, compatible operation/domain, input structure, register, difficulty, context length, number of choices, answer position, and outcome class. The 24 operations span different forms of reasoning, including conversational intent, coreference, negation scope, word patterns, evidence selection, causal reasoning from stated facts, counterfactuals, debugging, planning, and constrained choices. A one-off question has one problem-group ID. Any future paraphrases or variants of the same semantic problem must share that group and stay in the same split.

All rows produced so far are training rows. Evaluation questions must be assigned to separate problem groups before their text is generated; this factory does not yet create an evaluation set. Image questions and image assets are outside this pipeline.

Luna workers write question shards as JSONL under `data/generated_train/broad-v1/` and record unsupported assignments under `data/generated_train/assignments/`. No unattended generation API or separate model reviewer is used. Every generated `answer_index` is Luna-authored supervision. The current pipeline does not independently verify answers, explanations, or language quality, and a JSON or schema check must never be described as ground-truth verification. The provenance field says `Luna-generated; label not independently verified.`

`dataset_factory.assignments` creates reproducible, stratified plans. A plan file can be created without generating questions:

```sh
uv run python -B dataset_factory/assignments.py --check
uv run python -B dataset_factory/assignments.py --count 128 --seed 20260929
```

The allocator reads existing question and plan shards before assigning new global indices. It balances language, compatible operation/domain cells, and planned outcome classes, while drawing the other task dimensions and cue words per question. Each question gets its own problem-group ID before text generation. Do not reuse an assignment for a variant or put variants of one problem into different splits.

Use `--language-tags` and `--context-length-bands` to focus a shard on languages or context lengths a worker has previously produced. Other task dimensions remain sampled.

`dataset_factory.progress` scans source shards line by line and writes `progress.json` and `progress.md`:

```sh
uv run python -B -m dataset_factory.progress --target 1000000
```

It checks required fields, catalog language tags, choice counts, distinct options, and answer-index bounds. It also detects repeated IDs, normalized question content, problem groups, and identities spanning splits. Rows with invalid structure or duplicate identity/content are reported as quarantined; source files are preserved. Skipped assignments remain in their shard plan and are not counted as questions. The reports count structurally accepted rows only; they do not establish that answers are correct or that text is fluent.

After a worker finishes, `dataset_factory.reconcile` closes assignment rows when it finds a matching output ID or an explicit skip record. Unreported rows stay assigned or in flight for resumption:

```sh
uv run python -B -m dataset_factory.reconcile
```

The JSONL row fields match the decision-example shape used by `experiment.load_examples`, including `context`, `question`, `options`, `answer_index`, and `split`. The source shards are JSONL, while `load_examples` currently reads a JSON array. A versioned bundler that writes that array format and keeps generator metadata/provenance alongside it is still needed before these shards can be passed directly to the training runner. No evaluation export or model training is part of this generation run.
