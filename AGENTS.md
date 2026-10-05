# External-session monitoring

- Before or alongside each independent CLI or API agent session, start a visible gpt-6-luna monitor child in the current chat. Give it the external session ID, purpose, log and status paths, expected completion, and stop conditions.
- Monitor actual background work and incremental status or logs. Report meaningful progress, completion, failure, or stalls to the root agent, and end monitoring when the job stops. Do not restart or mutate owner-managed jobs without root direction.
- Prefer visible child agents when practical. Coordinate GPU owners so training and benchmarking do not overlap.

## Blind external benchmarks

Always follow [.cursor/rules/blind-benchmarks.mdc](.cursor/rules/blind-benchmarks.mdc). Benchmark content is evaluation-only and must not be exposed to the user or development agents; deterministic evaluation may process it internally.

In Nicolas's dictated messages, “Jeff” refers to “Jev”.

- When the root starts an independent external session, keep its visible Luna monitor active through verified completion, failure, or a diagnosed stall; give the monitor the session ID, purpose, log/status paths, expected completion, and stop conditions.
