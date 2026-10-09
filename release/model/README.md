---
license: other
license_name: mixed-source-noncommercial
license_link: LICENSE
base_model: Qwen/Qwen3-4B
base_model_relation: finetune
library_name: transformers
tags:
- decision-model
- multiple-choice
- soft-targets
- qwen3
- noncommercial
---

# Horscht: direct choice probabilities

This is Nicolas's experimental decision model based on **Qwen3-4B**, using a 265-output linear head over the final non-padding token. It assigns probabilities to 2–265 supplied choices in one forward pass. It does not generate an answer or a chain of thought. The final transformer block and the decision head were trained; all other backbone tensors remain unchanged.

The download stores **replacement weights**, not additive deltas: 11 final-block tensors and two head tensors, all in FP32 (`decision.safetensors`, 406,439,524 bytes). Load them over `Qwen/Qwen3-4B` at revision `1cfa9a7208912126459214e8b04321603b3df60c`. The export verified all 387 other backbone tensors exactly equal the upstream tensors cast to FP32. The original private warm-start checkpoint is therefore unnecessary for inference. Downloading upstream weights is still required; FP32 parameters occupy approximately 15 GiB before activations.

## Use

Download this repository locally and install `requirements.txt`. The loader is ordinary Python source you can inspect; Transformers remote code is unnecessary.

```python
import torch
from horscht_model import HorschtModel

model = HorschtModel.from_pretrained('.', device='cuda')  # CPU also works
result = model.predict([{
    'context': 'A box contains two red balls and one blue ball. Draw one ball uniformly.',
    'question': 'Which colour is more likely?',
    'options': ['Red', 'Blue'],
}])[0]
print(result['choice_index'], result['probabilities'])
```

`device='cuda'` also denotes supported ROCm devices in PyTorch. The verified release smoke test used CPU, FP32 stored parameters, and BF16 autocast. GPU/backend numeric agreement has not been established by that smoke test. Parameters stay FP32 and the backbone uses BF16 autocast by default, matching the training precision policy; the head runs in FP32. `autocast=False` requests FP32 computation and may change probabilities. The loader checks packaged artifact SHA-256 hashes before loading. Only PyTorch and safetensors weights from the pinned upstream revision are loaded.

The exact prompt is:

```text
Context: {context}
Question: {question}
Choices:
0: {first option}
1: {second option}
...
Selected choice:
```

Use plain tokenization with special tokens, not a chat template. Inputs include only `context`, `question`, and `options`; supervision, IDs, explanations, and split metadata never enter the model. The default input cap is 32,768 tokens, matching external evaluation. Longer complete inputs fail without truncation. You can explicitly request `max_tokens=40960`, the training budget; quality at that extended budget has not been established by the external evaluation. Process questions individually to bound memory. Long inputs can still exceed available memory.

The head applies non-affine LayerNorm (`eps=1e-5`) to the pooled FP32 features, then a biased linear projection. Positions greater than or equal to the supplied choice count are masked to negative infinity **before** softmax. Returned probabilities follow the supplied option order. There is no special IDK neuron: add a meaningful IDK option only when the task permits it. Choice wording and order affect the model; it is not permutation invariant.

The default is raw softmax (`temperature=1.0`). An exploratory temperature of `0.9174703667456412` was fit on 64 internal calibration questions using full-distribution soft-target cross-entropy. You can pass it to `predict(..., temperature=...)`; it does not alter argmax decisions. It is a small-subset fit, not evidence of universal calibration. External aggregate results below used uncalibrated probabilities.

## Training and selection

The selected artifact is update **1080** of the `2026-10-08` continuation. The run warm-started both backbone and head from the preceding current-corpus run's development-selected update 8017, using a fresh optimizer, schedule, RNG, and cursor. Training adjusted the 100,930,816 parameters of transformer block 35 and 678,665 head parameters. Adafactor used `beta1=None`, `scale_parameter=False`, `relative_step=False`, and zero weight decay. Learning rates were `1e-5` for the block and `1e-4` for the head, with 3% warmup and cosine decay over 1438 updates, accumulating 16 questions per update. The seed was 20261008. Main training completed 1438 updates; the released earlier checkpoint had processed 17,280 questions and 22,532,216 input tokens.

The frozen training snapshot had 23,006 training rows: 8,000 old-data replay, 5,332 engine-verified games, 3,900 document questions, 1,954 typed-decisions, 1,920 SciQ and 1,900 Banking77. The snapshot combined generated problems, grounded documents, public training partitions, hard labels and soft targets. Public-source targets are not all independently verified truth. Training contained 7,471 non-one-hot targets, 148 language tags, and 45 rows with 265 choices; these counts do not demonstrate competence in every language or at maximum cardinality.

Training used cross-entropy against the entire target distribution over available options. Hard labels became one-hot distributions. Stored soft probabilities were checked for finite nonnegative entries and a sum within `2e-6` of one, then normalized at the tensor boundary. Padded target slots were zero. Choice permutations remapped all choice-aligned targets and IDK indices; prompts with ambiguous letter/order references retained their original order. Related documents, translations, game states, and choice variants were grouped before splitting.

The original full-development scoring plan was interrupted under an evaluation-time budget change. Final selection used a frozen **128-question internal development subset**, evaluating updates 0, 360, 720, 1080, and 1438. Their soft cross-entropies were 1.338602, 1.173092, 1.158349, **1.141830**, and 1.144004 nats respectively. Lowest cross-entropy selected 1080; earliest update broke ties. Selection preceded calibration and external evaluation. The frozen subset protocol hash is `09bbc5385c6a3e43d3e4290d06a60454867f1e4e883867c8eb295dc88dd066eb`. The historical checkpoint manifest still records the superseded full-development plan; this card describes the actual final selection.

## Aggregate evaluation and limits

The selected model scored 9,008/17,874 (50.3972%) on the previously observed external MCQ suite, with NLL 1.487872 and multiclass Brier 0.617223. This is **exploratory reuse**, not a fresh untouched benchmark. Scores were obtained after internal selection and were not used to select this checkpoint. No external test examples or gold answers are included in this release.

| Slice | Questions | Accuracy |
|---|---:|---:|
| MMLU-Pro test | 11,874 | 38.6306% |
| Belebele English | 900 | 88.5556% |
| Belebele German | 900 | 84.0000% |
| Belebele French | 900 | 85.7778% |
| Belebele Spanish | 900 | 83.6667% |
| Belebele simplified Chinese | 900 | 85.0000% |
| BBH logical deduction, 3 objects | 250 | 76.8000% |
| BBH logical deduction, 5 objects | 250 | 51.2000% |
| BBH logical deduction, 7 objects | 250 | 47.6000% |
| BBH tracking shuffled objects, 3 objects | 250 | 31.2000% |
| BBH tracking shuffled objects, 5 objects | 250 | 12.0000% |
| BBH tracking shuffled objects, 7 objects | 250 | 12.4000% |

Belebele language slices are translations, not independent semantic problems. The overlap audit found no exact or near lexical overlaps; semantic overlap and pretrained-model exposure remain unknown. External inputs were capped at 32,768 tokens, despite the training budget of 40,960. Shuffled-object results show clear weaknesses, including scores below uniform choice accuracy. Probabilities can be confident and wrong; agreement with teacher distributions is not truth or demonstrated calibration. Use this research artifact to study direct decisions, not as a validated general-purpose decision authority.

## Provenance, reproduction, and rights

`release_manifest.json` records the source checkpoint hash, exact upstream revision, trainable names, frozen-tensor identity check, and packaged weight/tokenizer/config hashes. `smoke_verification.json` records the CPU synthetic inference check. Its output validates the loading/inference pipeline, not statistical calibration or task correctness. The source checkpoint manifest SHA-256 is `0bb4234d8f364c8219e79cee2ca29d8ee41031584bd89dd38830a29a1dac08d7`. The frozen snapshot manifest SHA-256 is `6413df891b6ee10073c8e49f4c3c0b19375c495ff595ab7889a4386e7bb427fd`. The external aggregate protocol SHA-256 is `2b7c8fe68d2345aed5f6125a267868cad226d001bfd35984b6b4a776e9b1eef7`.

In the associated Horscht source tree, export the selected internal checkpoint with:

```sh
python scripts/export_release_model.py \
  --checkpoint runs/new-data-last-block-2026-10-08/main-recovery-002/checkpoints/update-00001080-epoch-000-next-001080 \
  --upstream-snapshot /path/to/huggingface/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --output release/model
```

This command proves that the private full warm-start file is replaceable by upstream for inference, checks source artifact hashes, and serializes exact selected FP32 replacement values. It requires the original internal checkpoint and full warm-start file. Public inference needs neither. Exact retraining also requires the earlier warm-start run, immutable split registry, source snapshot, protocol amendments, and environment; publication of this inference artifact alone is not a claim that the complete two-stage training can be exactly reproduced from the model download.

See `LICENSE`, `LICENSE_CODE`, `LICENSE_QWEN`, and `NOTICE`. Original project software is MIT; original model/data contributions by Nicolas are CC-BY-4.0; Qwen's upstream weights are Apache-2.0. This release carries a **noncommercial-use policy because SciQ training data is CC-BY-NC-3.0**. Document and other source rights remain source-specific; this repository does not blanket-relicense third-party material. Review the associated dataset's attribution and source license records before redistribution or derivative use.
