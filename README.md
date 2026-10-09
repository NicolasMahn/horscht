# Horscht

A research decision model built on **Qwen3-4B**. It reads a context, a question,
and changing answer choices, then returns a probability distribution in one
forward pass. It does not generate a reasoning trace.

Version 0.1 trains the final transformer block and a linear 265-output head.
Earlier transformer blocks remain frozen. This release keeps Qwen3-4B: it
outperformed Qwen3.5-4B in our original small, template-bound comparison.
That early result is not a general ranking of those backbones.

- [Technical paper](docs/paper/paper.md) and [HTML report](docs/paper/paper.html)
- [Model release and inference](release/model/README.md)
- [Dataset release and provenance](release/dataset/README.md)
- [Aggregate evaluation evidence](release/evidence/)
- [Original experiments and setup](docs/original-experiments.md)

| Evaluation | Horscht v0.1 | Original Qwen3-4B immediate-answer reference |
| --- | ---: | ---: |
| MMLU-Pro · 11,874 questions | 38.63% | 39.95% |
| Belebele · five languages, 4,500 questions | 85.40% | 85.89% |
| BBH · six tasks, 1,500 questions | 38.53% | 42.40% |
| Combined · 17,874 questions | 50.40% | 51.72% |

The latest continuation improves the previous last-block checkpoint from
49.14% to 50.40% on the same suite, but remains below the original model's
immediate-answer reference. The latter uses the original generative head and
label likelihood, not thinking-mode generation. These are exploratory results
on a reused external suite, not an untouched final test or a claim of general
reasoning superiority.

The selected checkpoint is update 1,080 of a 1,438-update continuation on
23,006 questions, warm-started from an earlier trained checkpoint. It was
selected using soft-target cross-entropy on 128 development questions.
The additional internal test remains deferred. Training uses soft-target
cross-entropy, deterministic choice permutations and no reinforcement learning.

The model and data have separate release terms. Project code and original
documentation are MIT licensed. Training sources include SciQ's noncommercial
license and source-specific document licenses; see [NOTICE](NOTICE) and the
release cards before reuse. Benchmark content and sealed internal test data
are not included.

For the research environment:

```sh
uv sync --python /usr/bin/python3.12
uv run pytest
```

The default lock uses CPU PyTorch. Training on the project's AMD GPU used a
separate ROCm environment; the release card documents inference dependencies.
Do not interpret the default development environment as the GPU training setup.
