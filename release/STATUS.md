# Publication status

Prepared and verified on 2026-10-09 for Horscht v0.1, based on Qwen3-4B.

- Checkpoint: selected update 1080 from the new-data continuation.
- Portable weights: final-block replacement tensors and decision head; all 387
  frozen backbone tensors verified equal to the pinned upstream checkpoint.
- Model CPU inference smoke check: passed on an invented, non-benchmark input.
- Model package: 14 files, 417,905,140 bytes.
- Dataset package: 13 files, 128,389,707 bytes. Both training stages included;
  169,810 exported train/dev rows, 154,608 unique IDs because replay repeats rows.
- Dataset verification: exact input/target identity, no train/dev group overlap,
  and no sealed test/calibration ID overlap in either configuration.
- Paper: Markdown and standalone HTML; aggregate evidence verified; public-only
  rebuild and desktop/mobile rendering checked.
- Tests: 760 passed locally; fresh source-only environment 751 passed and nine
  explicitly skipped historical checks whose run artifacts are not distributed.
- GitHub: release/horscht-v0.1 branch prepared for authenticated push to the
  existing private repository. Visibility is not changed.
- Hugging Face: upload pending local authentication and destination namespace.
  Local upload dry-run and all artifact checksums pass; no upload is claimed.

No external benchmark items or sealed internal test rows are included.
Model and dataset payloads are excluded from Git; their cards, manifests,
checksums and supporting metadata are tracked. See README.md for publishing.
