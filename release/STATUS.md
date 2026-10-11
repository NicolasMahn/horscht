# Publication status

Published and verified on 2026-10-11 for Horscht v0.1, based on Qwen3-4B.

- Website and paper: https://nicolasmahn.github.io/horscht/
- Public source: https://github.com/NicolasMahn/horscht
- Model: https://huggingface.co/NicolasMahn/horscht-qwen3-4b-v0.1
- Dataset: https://huggingface.co/datasets/NicolasMahn/horscht-decisions-v0.1
- Checkpoint: selected update 1080 from the new-data continuation.
- Portable weights: final-block replacement tensors and decision head; all 387
  frozen backbone tensors verified equal to the pinned upstream checkpoint.
- Model package: 14 uploaded files, 417,905,476 bytes.
- Dataset package: 13 uploaded files, 128,389,846 bytes; both training stages,
  169,810 train/dev rows and 154,608 unique IDs because replay repeats rows.
- Dataset verification: exact input/target identity, no train/dev group overlap,
  and no sealed test/calibration ID overlap in either configuration.
- Anonymous verification: all 27 published files downloaded at their recorded
  revisions and checked against SHA-256 hashes. Standalone CPU inference from
  the downloaded model passed on an invented, non-benchmark input.
- Pages: successful deployment, public report URLs return HTTP 200; paper's
  five figures load, with no localhost links or horizontal page overflow.
- Tests: 760 passed locally; fresh source-only environment 751 passed and nine
  explicitly skipped historical checks whose run artifacts are not distributed.

Exact Hugging Face commits and hashes are in [publication-receipt.json](publication-receipt.json).
Download and inference results are in [publication-verification.json](publication-verification.json).
The external benchmark suite was previously observed; results are exploratory,
not an untouched test estimate. Internal test evaluation remains deferred.
No external benchmark items or sealed internal test rows are included.
Model and dataset payloads are hosted on Hugging Face and excluded from Git.
