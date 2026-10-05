# Published System One comparison, 2026-09-30

Metadata-only research against the Decision Index primary aggregate assets. Live edition: Decision Index 0.2.1, generated 2026-09-28T00:39:36+00:00; Jev version jev-1.13.0. Sources: [aggregate JSON](https://multimodalart-jev-decision-index.static.hf.space/data/index-v0.2.1.json), [methodology JSON](https://multimodalart-jev-decision-index.static.hf.space/data/methodology-v0.2.1.json), [board](https://multimodalart-jev-decision-index.static.hf.space/index.html?bench=57), [reproduction repository](https://github.com/apolinario/decision-index).

| Model | MMLU-Pro accuracy | Requests answered |
| --- | ---: | ---: |
| Jev 1.13.0 | 82.70% | 12032/12032 |
| Decider 4B | 51.65% | 12032/12032 |
| Tev1-4B-experimental | 45.37% | 12032/12032 |
| Decision 1.0 Nox | 36.69% | 12032/12032 |

Our already-completed frozen evaluation reports 22.44% head-only and 37.52% full tuning on 11874 deduplicated MMLU-Pro test rows, removing 158 of the published 12032. These scores are directional comparisons, not identical-harness rankings. Published source revision: TIGER-Lab/MMLU-Pro b189ec765aa7ed75c8acfea42df31fdae71f97be. Decision Index maps published options in order, preserves native entrant inference helpers, prompts, and readout rules, and prohibits prompt tuning, truncation, option pruning, or correctness-based retries. It retains source duplicates. All four listed entrants answered every MMLU-Pro request.

[Decider model card](https://huggingface.co/Mapika/decider-4b) explicitly describes one forward pass with no decoding and letter-logit slot readout. It is a full fine-tune of Qwen3.5-4B-Base, not the same backbone as our Qwen3-based run. [Tev1](https://huggingface.co/togethercomputer/Tev1-4B-experimental) also uses Qwen3.5-4B-Base. The index methodology says its adapter uses vLLM LLM.generate; single-forward or no-CoT details were not independently verified during this research pass. Architecture labels in the board (autoregressive, encoder, diffusion) describe model families and do not establish whether a decision produces a generation sequence.

The publication has BBH aggregate accuracy over 5507 cases across 23 task files (19 families); our selected deduction and object-tracking slices are not comparable to that pooled number. No Belebele entry was found. Our full mixed aggregate is not a Decision Index score. Published latency combines different hardware and inference/network paths and is not a controlled comparison to our local run.

The publication explicitly marks MMLU-Pro training overlap for other entrants (JevK5 and pngwn open-jev), but no MMLU-Pro training-overlap penalty is indicated for Jev, Decider 4B or Tev1 in this snapshot. This is not proof of absence of contamination; their training data and checkpoints differ from ours.

## Exposure record

During post-evaluation metadata research, the methodology added-benchmark framing field unexpectedly contained a small amount of malformed-source item metadata and short answer text. It was accidentally displayed to this isolated reporting agent. No benchmark questions, dataset samples, or per-item results were deliberately accessed. The content is not repeated here and was not forwarded to training or generation agents. This happened after the experiment and evaluation were frozen and completed, so it cannot have affected that completed inference. For future agent-assisted design, MMLU-Pro blindness must no longer be described as untouched in this conversation. Do not use this exposure or these test scores for training, generation, tuning, calibration, prompt selection, or checkpoint selection.
