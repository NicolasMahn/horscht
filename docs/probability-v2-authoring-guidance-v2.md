# Probability-v2 semantic-grounding guidance v2

Version: `probability-v2-semantic-grounding-v2`  
Effective: `2026-09-29`  
Scope: new assignments and unfinished rows only. Existing rows and IDs stay unchanged.

Version 1 reduced obvious noun-swaps, but a newly completed shard still used one uniform counting frame while placing domain words around directly listed outcomes. Domain names alone do not create semantic variety.

For each question, the assigned domain must change the facts, mechanism, constraints, or evidence used to derive the target. If removing the domain-specific facts leaves the same generic probability exercise, redesign the scenario or explicitly skip it. Do not use a bare list of equally likely outcomes with topical labels as a shortcut. The domain rules should determine how outcomes arise or how the evidence changes their probabilities.

Across a shard, vary the reasoning operation and the evidence structure: for example, a process description, observations, a table, a conditional mechanism, a sequence of events, or competing explanations. Avoid applying one template to many domains, repeating one question stem, or making unrelated details decorative. Uniform distributions are valid when the stated mechanism makes them uniform, not as a default. A direct finite enumeration is valid when the real assigned task calls for it and the scenario facts matter to the result.

Keep the assigned language, source kind, option count, and target contract. Exact and stochastic targets must follow from stated mechanisms with exhaustive outcomes. Empirical targets must match the stated sample and must not be presented as population truth. Teacher estimates remain explicitly unverified judgments, not calibrated probabilities. Skip rows when the scenario cannot support a coherent target; do not fall back to English or invent certainty.

The v2 choice-count contract allows 2–265 choices. For medium and high assignments, the task and evidence structure must support the full set of meaningful alternatives: use a taxonomy, inventory, evidence bundle, transition table, or similarly explicit structure where appropriate. Do not pad a short decision with arbitrary distractors. High-tier assignments may be simplified to fewer choices only if the assignment is skipped and replaced through the planner; keep the planned count aligned with the authored options.

For answerability assignments, reason about what the supplied context supports. An answerable task must have a supported target choice and give the abstention choice zero target mass. An unanswerable task must target the explicit localized abstention choice deterministically. Use the assigned reason as authoring metadata, not as a prompt hint. Keep probability language separate from answerability: a probability distribution describes outcomes under the stated model, while answerability describes whether the requested answer follows from the available information. Do not substitute a confidence score for either one.

Machine checks validate structure and arithmetic only. Do not claim independent semantic or language review.
