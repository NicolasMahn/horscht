# Horscht

An experiment in direct decisions: context + question + changing choices → choice index.
The first milestone is deliberately memorizing 16 examples with a small head on
a frozen pretrained model. It does not establish reasoning, generalization,
calibrated probabilities, or the ability to adapt the backbone.

All nine checkpoints are downloaded and evaluated on text inputs. The original
16-question smoke experiment completed 90 head fits across seeds 42–51, all
passing overfit and reload checks. The matched 16/64 experiment completed 180
fits; 175 passed the strict overfit criterion. All saved-head and backbone reload
checks passed. See the matched experiment below for the five training failures.

[report.html](report.html), served at http://127.0.0.1:8765/, is the consolidated
selection report. Its current provisional pick is **Qwen3 Post-trained**, with
88.125% mean accuracy on the 64-question development holdout after training on
64 questions. Qwen3.5 follows at 84.6875%, then Qwen3 Base at 82.1875%. These
small, template-bound development results do not establish broad generalization.

Detailed learning curves and individual runs remain available separately:

- [Smoke report](report-smoke.html): http://127.0.0.1:8765/smoke
- [Matched training-size report](report-size.html): http://127.0.0.1:8765/size
- [Exploratory transfer report](report-transfer.html): http://127.0.0.1:8765/transfer

The detailed reports include mean ± sample standard deviation curves. Stopped
runs retain their final value when calculating aggregate curves. The original
one-seed report is retained in `report-single-seed.html`. Transfer measurements
are supplementary and do not select the winner.

To rebuild the reports from saved measurements and serve them locally:

```sh
uv run python build_repeated_report.py
uv run python build_size_report.py
uv run python build_transfer_report.py
uv run python build_summary_report.py
uv run python serve_report.py
```

The server binds only to localhost and serves the report files. Repeated smoke
results live in `results/repeated/`; individual heads and measurements live in
timestamped `runs/*-repeated/` folders. Each head and optimizer is initialized
afresh. Both `compare.py` and `repeat_runs.py` default to all nine candidates.
Use `--models` to restrict either command to a subset; repeated training requires
a successful comparison run with verified cached features for every selected model.
Cached features can be reused without downloading or loading the backbone again.

The first real run passed on 2026-09-28 with Ministral Base. Training accuracy
rose from 31.25% to 100% over 90 head updates; loss fell from 1.15585 to 0.01870.
Encoding 24 examples took 5.73 seconds on the CPU, and head training took 0.13
seconds. The first download took approximately eight minutes. The eight
exploratory probes scored 4/8, which is not a model-selection result. Reloading
the backbone and head in a fresh process reproduced two inputs' features
exactly and preserved their predictions.
The retained [result](results/2026-09-28-ministral-base.json) includes the
manifest, learning curve, individual predictions, and reload verification.

## Run the first experiment

```sh
uv sync --python /usr/bin/python3.12
uv run pytest
uv run python experiment.py run --model ministral-base
uv run python experiment.py verify runs/<run-directory>
```

To encode, train and verify all nine candidates, then rebuild the smoke reports
(weights download automatically if absent from the cache):

```sh
uv run python compare.py
uv run python build_report.py --output report-single-seed.html
uv run python repeat_runs.py
uv run python build_repeated_report.py
```

The single-seed report can also be opened directly in a browser. It includes interactive training and
probe loss/accuracy curves, uniform-choice baselines, timing, results by task,
and every individual prediction. Hover for exact values; use the loss plots'
linear/log buttons to inspect early spikes and late convergence. The file
embeds Plotly and the measurements, so it works offline. It also exports the
raw JSON and individual plots as SVG.

The comparison runner executes one model at a time and retains bundles in
`results/comparison/`. For example, to rerun only the five reasoning candidates while
retaining the four baseline results:

```sh
uv run python compare.py --models deepseek-distill deepscaler phi-reasoning qwen-thinking qwen35-reasoning
uv run python repeat_runs.py --models deepseek-distill deepscaler phi-reasoning qwen-thinking qwen35-reasoning
uv run python build_repeated_report.py
```

Training records metrics after every update, including
update zero. Probe evaluation never changes gradients or the stopping rule.
The original first-run record is preserved separately.

Dependencies are locked in `uv.lock`. PyTorch uses its CPU wheel index. The
selected models download several GB each into the normal Hugging Face cache.
`experiment.py run` loads one model; `compare.py` defaults to all nine. The default uses eight CPU
threads and BF16 backbone weights. Use `--dtype float32` on a CPU without
efficient BF16 support; that doubles weight memory. No local model services are
started or stopped.

The nine candidates and their immutable revisions are in `candidates.json`:

| CLI name | Checkpoint |
| --- | --- |
| `ministral-base` | `mistralai/Ministral-3-3B-Base-2512` |
| `ministral-instruct` | `mistralai/Ministral-3-3B-Instruct-2512-BF16` |
| `qwen-base` | `Qwen/Qwen3-4B-Base` |
| `qwen-instruct` | `Qwen/Qwen3-4B` |
| `deepseek-distill` | `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` |
| `deepscaler` | `agentica-org/DeepScaleR-1.5B-Preview` |
| `phi-reasoning` | `microsoft/Phi-4-mini-reasoning` |
| `qwen-thinking` | `Qwen/Qwen3-4B-Thinking-2507` |
| `qwen35-reasoning` | `Qwen/Qwen3.5-4B` |

`qwen-instruct` names Qwen3's post-trained checkpoint, which supports both
thinking and non-thinking generation. DeepSeek R1 Distill is a distilled
reasoning model; DeepScaleR continues training that checkpoint with GRPO RL.
Their pairing helps examine the effect of that additional training. Phi-4-mini
Reasoning and Qwen3 Thinking add other reasoning-trained checkpoints.

Qwen3.5-4B is a unified vision-and-reasoning checkpoint, not a separate
text-only reasoning release. This comparison uses its language backbone with
text inputs only. For every candidate, the experiment reads hidden states
directly: it does not generate chain-of-thought tokens or exercise a thinking
mode. It tests whether reasoning training makes an immediate choice more
accessible to a small head, not the model's full generated-reasoning ability.

## How a candidate is selected

The consolidated report ranks the **64-training-question condition** by:

1. Highest mean final accuracy on the shared 64-question development holdout,
   across seeds 42–51.
2. Lower mean final development cross-entropy if accuracy ties.
3. All tied candidates if both metrics tie exactly.

All ten seeds must pass the overfit and saved-head reload checks, and the source
features must pass fresh-backbone reload verification. Training accuracy is an
eligibility check, not the ranking objective. DeepScaleR is excluded because
five of its ten 64-question fits missed the strict loss target. Its measurements
remain visible. No training budget or loss target was changed to make it pass.

The older smoke report retains its separate `selection_policy.py` rule on the
eight already inspected probes, fixed after the first four models and before
the five added models. It is not the consolidated report's selection dataset.
Exploratory transfer results also do not choose the winner.

Both development comparisons provide provisional choices, not unbiased estimates
of generalization. Ten seeds measure initialization variability; they do not
supply ten independent question sets. A larger untouched evaluation set is
needed before making a strong model-quality claim.

## What is trained

All candidates receive the same plain-text format, without chat templates.
The input contains all choice descriptions; the answer index and explanation
never reach the backbone. Inputs longer than the token limit fail explicitly.
The final non-padding token supplies one representation of the complete input.
For Ministral and Qwen3.5, the text backbone is used without vision components
and retains its input embeddings. We never compute vocabulary logits.

The frozen backbone runs in evaluation mode once per example. Its FP32 output
features are cached, then a non-affine LayerNorm and trainable linear head are
applied. The head has 265 outputs. Slots beyond the current question's option
count are masked before loss and prediction. The choice count remains
configurable in the data up to that capacity; 265 is the provisional capacity
from our discussion, not a claim about Jev's architecture.

`data/smoke.json` contains 16 training examples and 8 probes with answers computed
from explicit facts. `make_examples.py` reproduces the file. Only slots 0–3 are
exercised. Answer positions vary, but the small set still contains shortcuts;
it is intentionally unsuitable as a benchmark. Probes use new values within
the same families and are exploratory only. They are not a held-out-family
generalization test, and never affect training or stopping.

The run passes when training accuracy reaches 100%, cross-entropy is at most
0.02, head weights have changed, and a fresh head loaded from disk produces
identical logits. It stops after at most 1,000 steps and exits nonzero if those
conditions are not met. `verify` separately reloads the pinned backbone and
head in a new process, recomputes two inputs, and checks feature agreement and
identical predictions. Neither check validates backbone training.

Each ignored `runs/` directory contains the exact data, cached features,
head weights, a manifest with the model revision and execution settings,
and a JSON report with initial/final loss, accuracy, timing, and predictions.
The separate verification writes `reload_verification.json`.

## Matched training-size experiment

The separate experiment completed 180 fits across all nine candidates: 16 versus
64 training questions evaluated on the same new 64-question holdout. All 90 fits
with 16 training questions passed. Five of DeepScaleR's ten 64-question fits
missed the loss target within the fixed 1,000-update limit; the other 85 passed.
Every saved-head and fresh-backbone reload check passed. The deterministic
`make_expanded_examples.py` generator writes 128 questions to `data/expanded.json`:
64 training questions and 64 holdout questions, balanced across the four task
families. The 16-question training condition preserves the original smoke
training questions verbatim, with only subset metadata added. The 64-question
condition contains those same 16 questions plus 48 new training questions.
Both conditions use paired seeds 42–51, giving 180 heads
across nine candidates and two training sizes.

Mean accuracy on the shared 64-question holdout:

| Model | 16 training questions | 64 training questions |
| --- | ---: | ---: |
| Ministral Base | 40.47% | 55.00% |
| Ministral Instruct | 62.81% | 76.88% |
| Qwen3 Base | 82.81% | 82.19% |
| Qwen3 Post-trained | 78.59% | 88.13% |
| DeepSeek R1 Distill | 42.03% | 43.59% |
| DeepScaleR | 30.47% | 38.13% (ineligible) |
| Phi-4-mini Reasoning | 54.84% | 67.50% |
| Qwen3 Thinking | 77.97% | 79.69% |
| Qwen3.5 Reasoning | 79.84% | 84.69% |

The original 16-question training endpoints and stopping steps reproduce exactly
for every seed of the original four models.

The questions, subsets, seeds and stopping rule are fixed before any measurement;
the runner records their shared protocol before extracting features. Neither
condition reuses the original eight probes. The new holdout is shared between
conditions so their paired accuracy differences address the training-size
question on the same examples. Head training and fresh-backbone reload checks
are retained, and probe scores never drive training or stopping.

This is still a template-bound experiment, not evidence of broad generalization.
State questions retain one transition, with varying lamp names to keep prompts
distinct; routing questions vary explicit mappings. Compare the two training
conditions on their shared new holdout. Do not interpret a score change from
the original eight probes as a training-size effect, because the evaluation
questions changed.

To run or resume the matched comparison using the available checkpoints:

```sh
.venv/bin/python size_experiment.py run --runs-root runs/size-comparison-original16
.venv/bin/python build_size_report.py
.venv/bin/python build_summary_report.py
```

The runner also accepts `--models` to select candidates. It keeps results in
`results/size_comparison/` and cached features, heads and the dataset lock in
`runs/size-comparison-original16/`, isolated from the original experiment. The separate
[report-size.html](report-size.html) is available at
http://127.0.0.1:8765/size and shows both conditions and seed-paired changes for
the measured models. Image questions remain deferred.

## Follow-up experiments

Probability-v2 soft targets use a separate frozen-backbone runner. See
[the soft-target experiment notes](docs/probability-softtarget-experiment.md);
the illustrative pilot smoke checks plumbing only.

Current scope is the nine registered candidates on text questions. Image
questions remain a separate, deferred experiment: they require retaining vision
components. Begin that discussion with answer-verifiable drawn scenes, then
decide how to review richer or model-generated questions.

1. Expand data across task families, choice counts, and shuffled option orders.
   Hold out whole scenarios/families before generating paraphrases. Measure
   decision accuracy, ordering robustness, latency, and calibration separately.
2. Verify real backbone adaptation using LoRA plus the head, with uncached
   forward passes and evidence that adapter weights receive gradients and change.
3. Compare native chat formatting and alternative feature pooling as separate
   experiments. Keep training budgets comparable and select on development data
   before evaluating on untouched questions.

The local generation gateway is `/home/nicolas/Projects/local_llm`, with an
OpenAI-compatible endpoint at `http://localhost:8400/v1`. It is not needed for
this answer-verified fixture. On 2026-09-28 its status reported both existing
models running and a GPU busy-without-work warning, so this first environment
uses CPU execution.

Model references: [Ministral report](https://arxiv.org/html/2601.08584v1),
[Ministral Base](https://huggingface.co/mistralai/Ministral-3-3B-Base-2512),
[Qwen3 Base](https://huggingface.co/Qwen/Qwen3-4B-Base),
[Qwen3 post-trained](https://huggingface.co/Qwen/Qwen3-4B),
[DeepSeek R1 Distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B),
[DeepScaleR](https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview),
[Phi-4-mini Reasoning](https://huggingface.co/microsoft/Phi-4-mini-reasoning),
[Qwen3 Thinking](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507),
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-4B).
