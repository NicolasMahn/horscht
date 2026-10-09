# Aggregate evaluation evidence

These three files compare the selected Horscht v0.1 checkpoint, its previous
last-block warm start, and the original Qwen3-4B immediate-answer reference.
The latter uses original-head label likelihood, not thinking-mode generation.

Every file covers the same 17,874-item suite: 11,874 MMLU-Pro, 4,500 Belebele
(five languages), and 1,500 BBH (six tasks). The suite has been observed in
previous experiments; these results are exploratory. No benchmark questions,
answers, item-level predictions, or sealed internal-test data are included.

The exports preserve aggregate metrics and the SHA-256 of each complete local
aggregate source. They intentionally omit host-specific execution paths.
Regenerate them with `python scripts/export_release_evidence.py` when the local
experiment artifacts are available. This does not rerun inference.
