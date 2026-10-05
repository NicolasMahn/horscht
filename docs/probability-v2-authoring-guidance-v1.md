# Probability-v2 semantic-grounding guidance v1

Version: `probability-v2-semantic-grounding-v1`  
Effective: `2026-09-29`  
Scope: new assignments and unfinished rows only. Existing rows and IDs stay unchanged.

The embedding audit of the frozen 41,221-row snapshot found repeated question stems and several near-neighbor examples that shared a reasoning frame while changing only the domain wrapper. Similarity is a screening signal, not proof of duplication; legitimate parameter and answer contrasts remain useful.

For each new row, make the assigned domain matter to the facts or reasoning. If changing the domain details would leave the same generic problem and answer, choose a different scenario. Vary the input structure and question form naturally across assignments; do not fill rows with the same causal-chain, analogy, contradiction, or constraint frame after swapping names and nouns. Avoid paraphrase and translation copies as separate problems. Keep genuinely related counterfactual variants in one problem group when such a group is explicitly planned.

The assigned operation, language, evidence source, choices, and target still govern the row. Idea-seed words are optional inspiration and do not need to appear in the question. Keep contexts self-contained, make the target derivation fit the stated evidence, and never imply that a machine-checked row has independent semantic or language review.
