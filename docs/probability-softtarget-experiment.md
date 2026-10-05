# Probability-v2 soft-target experiment

Run the frozen-backbone decision head directly from validated probability-v2 JSONL:

```sh
uv run python probability_experiment.py run \
  --data data/generated_train/probability-v2-pilot.jsonl \
  --model qwen35-reasoning \
  --steps 100
```

The runner uses the existing input formatter and decision head. It encodes context, question, and choices with a frozen backbone, then fits the head against each soft categorical target. Hard-label rows become one-hot distributions and are also the only rows counted in `hard_accuracy`. Pass `--evaluation-data` to score a separate probability-v2 JSONL file; without it, the report evaluates on the training rows.

The loader checks the v2 contract, target length and normalization, finite values in `[0, 1]`, source kinds, stable choice IDs, and evidence order. Metrics are per-question soft cross-entropy, `KL(p||q)`, the sum of squared probability errors across valid choices, and target argmax agreement. Padded choice positions are excluded from the categorical calculations. Teacher-estimate breakdowns describe agreement with an unverified estimate; they do not measure truth or calibration.

The default model load is local-cache-only. Runs are written under `runs/probability-v2-softtarget-*` and include a manifest, examples, cached training features, head checkpoint, and report. The probability-v2 pilot is illustrative, semantically unreviewed, and excluded from mass training; a run on it checks plumbing only and says nothing about generalization.
