# Deterministic game data infrastructure

This is an isolated development module. The only generation authorized here is the explicit bounded pilot described below; older generation coordinators remain paused. It neither discovers training data nor writes files, runs engines, starts agent sessions, or uses the GPU.

Run a two-row stdout smoke fixture from the repository root:

```sh
python -m dataset_factory.game_data
python -m pytest tests/test_game_data.py -q
```

Implemented slices:

- Tic-tac-toe: standard 3×3, legal replay, exact minimax within budget.
- Connect Four: standard gravity 6×7, legal replay, depth/node-bounded minimax. Most early positions remain unresolved and cannot be exported. Small endgames or explicitly supplied tactical candidates can be proved.
- Nim: normal play, labelled heaps, exact XOR teacher; losing positions correctly tie all legal moves.

```python
from dataset_factory.game_data import Nim, decision_row, connected_group_ids

row = decision_row(Nim((1, 3, 5)), scenario_id="fixture-1", seed=17)
groups = connected_group_ids([row])
```

The shared state contract exposes rules/version, player, legal actions, transition, terminal outcome and player observation. These games have full public state. No hidden-information adapter or observation omission is implemented; future adapters must derive labels using permitted observations or an explicit belief model. A privileged simulator state must not become model input by accident.

Actions use zero-based indices. Board tuples contain a cell index (tic-tac-toe) or column (Connect Four). Nim tuples contain heap index and amount removed. Player 1 starts the board games; player −1 moves second. Outcomes are winner sign or zero for draw, and action values are minimax −1/0/+1 from the current player's perspective. Terminal states have no decisions. Construct boards through `initial()` and `apply()`; direct construction validates counts, gravity and possible final winning moves but is not a general historical reachability proof for arbitrary Connect Four boards.

Teachers return aligned action/value-bound pairs, optimal ties when proved, label method, budget and visited nonterminal nodes. Terminal checks and cached lookups do not count as expanded nodes. Search never turns a depth cutoff into a draw. Unknown positions retain [−1,+1] bounds. Export requires every candidate still capable of being optimal to have the same proved value; otherwise it raises an error. A bounded search can prove the supplied optimal set without solving dominated candidates, and remains marked `bounded_minimax`. These are mathematical interval bounds, not confidence probabilities.

Targets are a uniform **action policy over proved optimal supplied candidates**, not probabilities of winning and not calibrated correctness probabilities. Multiple optimal actions are explicit ties. `teacher_estimate` describes the chosen policy in the existing probability-v2 taxonomy; `label_method` separately identifies exact analytical, exact minimax, or bounded minimax evidence. No score softmax is used. Targets and candidate IDs reorder together, reproducibly by seed. Candidate permutations preserve the original row ID. At most 264 action candidates are accepted, reserving one of the 265 heads for task IDK. Larger spaces require an explicit caller-supplied subset; provenance declares `supplied_subset` and the question evaluates best among supplied actions. Candidate selection itself is not optimized here.

Full-observation exported rows are answerable and give IDK zero mass. Unresolved teacher search is a compute limitation, so rows are omitted; it is not missing player evidence and must not become task IDK. Future missing-information tasks must use the existing unanswerable one-hot IDK contract and independently establish actual underdetermination.

The default smoke rows retain a development-only envelope and truthful deterministic provenance. The explicitly authorized pilot assigns connected groups to train/development files and validates those splits with an explicit `allowed_splits` argument. The legacy v2 validator currently mandates both `split="train"` and the literal Luna-generated provenance. `validate_game_row` reuses all v2 schema/math/answerability checks while replacing precisely these two envelope requirements. Thus these rows intentionally do **not** pass the legacy training validator, including pilot rows whose train split is assigned truthfully. No existing generation or training contract was changed. Future dataset integration needs an explicitly reviewed envelope update.

All related scenario IDs, trajectory IDs, and canonical state-family IDs are grouping edges. Use `connected_group_ids` before assigning any train/development/evaluation split, rather than splitting rows independently or trusting the display `problem_group_id`. It merges edges transitively and covers board symmetries and Nim heap permutations. Shared states may connect many trajectories into one component; common initial positions can join an entire corpus. Plan episode families or retain only nonoverlapping decision regions rather than overriding those links. Near-identical but unequal trajectories require an additional future overlap/family policy. This helper cannot infer those relations.

Curriculum records a caller-selected easy/medium/hard band, remaining units, search depth and node budget. Difficulty is declared, not calibrated: remaining units alone do not establish reasoning depth. Default search expands at most 20,000 nodes to depth 9; hard caps are 100,000 nodes and depth 42. A solve-local cache dies after the call. Nim is capped at 16 heaps of size 256. The smoke uses only two positions and a 2,000-node budget. The pilot runner has a finite explicit quota, no auto-resume, no API authoring/review cost and no service polling. Keep future CPU jobs serial and separately bounded while GPU training owns shared memory/power.

A later chess adapter should implement this same contract with versioned variant rules and player-visible FEN/history, an externally managed Stockfish UCI binary whose build/hash/options are recorded, and explicit node/depth limits. Start with easier tactical/endgame positions where the strong teacher can establish reliable decisions cheaply. Record whether evidence is exact tablebase, bounded engine evaluation, or an explicit engine policy. Centipawn scores cannot be silently interpreted as outcome probabilities; WDL estimates require engine version, settings and their stated model. Do not build or run Stockfish as part of this module.


## One authorized pilot

The 2026-10-02 pilot is isolated at `data/game_pilot/2026-10-02/`. It is not added to any active training snapshot, discovery path or older coordinator. Its manifest contains explicit train/development file allowlists, SHA-256 hashes, exact accepted/rejected counts, grouping statistics, teacher/rules versions and generator source hashes. `runs/game-pilot-2026-10-02/status.json` records verified completion or a bounded partial/failure state.

The bounded command used for this authorization is:

```sh
python -m dataset_factory.game_data.pilot --output-dir data/game_pilot/2026-10-02 --status runs/game-pilot-2026-10-02/status.json --per-game 64 --seed 20261002
```

The runner refuses existing output/status paths. Do not repeatedly invoke it with new paths without a new generation request. Maximums are 64 originals per game, 512 sampling attempts per game by default, 120 seconds by default, and 500,000 charged search nodes total. Per-position solves use at most 5,000 nodes and depth 9. Rejected unresolved solves conservatively charge the full node budget. Pure-Python generation is serial (one CPU thread); BLAS/OpenMP environment limits are also one. No GPU or external API is imported or used.

Each accepted original is a unique canonical state family sampled from a separate trajectory. There are no translations or candidate-order variants. Board histories replay from the legal initial state and retain one selected decision per trajectory. Nim samples 2–5 heaps sized 1–9. Tic-tac-toe samples 2–7 played moves. Connect Four samples 33–38 played moves while avoiding earlier terminal states; this deliberate endgame sampling bias must be preserved in dataset descriptions.

Before splitting, the pilot additionally joins near positions transitively: board states differing in at most two cells under legal board symmetries, and Nim heap multisets with equal cardinality and total size difference at most two. Twenty percent of connected groups per game, rounded up, are development families selected by a seeded hash. Large components may make row ratios uneven or hold out a whole game. All variants of an original must later inherit its existing `problem_group_id` and split. This grouping policy is conservative, but is not proof against every semantic similarity or pretrained-model overlap.

Future training must read only the manifest's explicit train allowlist, verify file hashes and use `validate_game_row(row, allowed_splits=("train",))`; it must keep development rows excluded. The existing running trainer and its legacy Luna provenance contract remain unchanged. The pilot supplies reliable action-policy labels; it does not establish cross-game or real-world transfer.

## Bounded expansion across five families

The separately authorized expansion uses `dataset_factory.game_data.expansion` and targets at most 10,000 new canonical originals, initially budgeted as 2,500 Connect Four, 2,300 Nim, 2,500 grid navigation, 2,500 Lights Out and 200 tic-tac-toe. The earlier pilot is read through its manifest and checked by SHA-256. Canonical duplicates from that pilot are rejected, as are duplicates within the expansion. Candidate permutations are not counted as originals and no presentation variants are produced.

```sh
python -m dataset_factory.game_data.expansion --output-dir data/game_expansion/2026-10-02 --status runs/game-expansion-2026-10-02/status.json --prior-dir data/game_pilot/2026-10-02 --seed 2026100202
python -m pytest tests/test_game_expansion.py tests/test_grid_navigation.py tests/test_lights_out.py tests/test_game_data.py -q
```

The CLI is serial and caps its process at 1,800 CPU seconds and 2 GiB address space. Generation stops taking candidates at 70% of the 30-minute wall budget, reserving time for grouping, reconstruction and relabel verification. There are at most 30,000 sampling attempts per game. Existing output/status paths are refused. There is no automatic restart, retry wave, GPU inference, training integration or external authoring API. A visible Luna implementation helper contributed the Lights Out adapter; all generated labels and wording are deterministic local code.

Grid navigation uses fully visible 5×5 through 9×9 square maps with static walls, four unit-cost orthogonal actions and distinct player/goal cells. Reverse BFS provides exact shortest distances and all optimal next moves; unreachable goals and singleton-action states are rejected. States on one topology share an episode family regardless of player/goal positions. This abstraction solves observation/perception for the model; it makes no claim about real-world navigation.

Lights Out uses a fully visible 5×5 board and 25 legal press actions. A press toggles itself and orthogonal neighbors. GF(2) elimination enumerates the small nullspace and chooses minimum-weight press solutions, yielding exact shortest-press distances and all valid optimal first presses. The adapter also supports sizes 2–4 for focused verification; exhaustive 3×3 BFS checks independently establish the distance/value/optimal-action semantics. States are reverse-scrambled from the all-off goal using the stated toggle law. This is a generation operation, not a player trajectory after a terminal win. Scramble indices are retained so every observation can be reconstructed. The expansion excludes shortest distances below three.

Connect Four samples early (6–12 played moves), middle (13–28) and late (29–38) nonterminal legal positions. Cheap whole-game proofs use the existing exact teacher. Otherwise the question explicitly defines a finite game of three or five plies: terminal win +1, loss −1, draw 0, and nonterminal-at-horizon reward 0. The finite-horizon minimax is exact for this declared objective. Its zero reward means no terminal reward within that horizon; it is not an unknown full-game result silently labelled a draw. Finite-horizon cases with all action values identical are rejected. The manifest separates finite-horizon objectives from whole-game minimax and reports the achieved phase mix; early-game meaningful tactical states can remain scarce. No teacher score is treated as a calibrated probability.

Nim includes 2–8 heap scenarios, deliberately constructed XOR-zero losing cases with genuine optimal-action ties, and high-cardinality positive partitions. Total objects are capped at 264, so the full legal action set can naturally occupy all 265 heads when IDK is included. Every original has at least two legal actions; no singleton-choice filler is accepted. The state-space cap and wider heap variety prevent exhausting a tiny fixed subset, but Nim remains an analytic introductory game rather than a hard search task.

Grouping runs before any split publication and merges earlier pilot groups with new state relations transitively. Board symmetries and at most two differing board cells are indexed with wildcard signatures. Nim heap multisets use total size difference at most two. Grid families include all goals/player positions on a topology plus at most two wall differences under square symmetry. Lights Out groups states within two observed-light differences or two legal presses under square symmetry. The implementation releases each game's neighbor index before processing another game. Grouping is conservative; it does not prove all semantic overlap has been ruled out.

A component connected to one prior split inherits that split. A component connected to both prior training and development splits puts **every new row in that component into quarantine**, without changing prior records or assignments. New independent components receive a seeded hash split (approximately 80% training, 20% development by group). Only `train.jsonl` appears in `training_allowlist`; `quarantine.jsonl` must never be discovered as training data. Small tic-tac-toe states can join large prior-linked components and therefore produce little or no eligible expansion data; the manifest reports this separately rather than inflating accepted training counts.

The expansion manifest and matching run status distinguish generated, accepted, training, development and quarantined originals; accepted counts exclude quarantine. They include actual teacher/objective/curriculum/candidate/phase distributions, exact file/source hashes, replay/relabel verification and resource accounting. Difficulty uses exact shortest distances for puzzles and declared remaining-unit thresholds for games, so it should not be interpreted as measured reasoning depth. All targets remain uniform optimal-action policies with IDK mass zero under full permitted observations. No hidden-information, positive-IDK or stochastic outcome-probability cases are claimed.

## New finite-law, hidden-inference and irreversible mechanics

Three reusable adapters extend the 2026-10-03 wave without changing earlier module interfaces:

- `DiceDecision`: normalized integer weighted-face law, independent 1–3-die convolutions with exact `Fraction` probabilities, public threshold/payoff offers, exact expected values and ties. The transition requires a reachable observed sum for gamble actions; it never fabricates a stochastic outcome. The task states a one-step risk-neutral objective. This is a small dice game rather than Blackjack.
- `Mastermind`: uniform prior over 2–6 symbols and length 2–4, with repetition; public guesses and exact/misplaced feedback only. The adapter stores no secret. Exact posterior enumeration, symbol/feedback distributions and expected posterior-cardinality guess scores use only permitted evidence. A sampled secret exists transiently only when creating a consistent public transcript and is discarded. Public API histories are capped at eight observations; generation uses one to three.
- `Sokoban`: fully observed push-only boards, bounded one/two-box reverse BFS and exact shortest walk/push solutions. Absent states are deadlocks only after the complete reverse queue is exhausted. `SokobanSearchLimitError` distinguishes incomplete search. Generation uses square sizes 5–7 with border walls, 1–2 boxes, and proven-solvable starting states. The API additionally supports rectangular bounded boards; the cross-version generator uses square terrain grouping.

Run the explicitly authorized finite wave with:

```sh
flock /tmp/horscht-game-breadth-cpu.lock python -m dataset_factory.game_data.breadth --output-dir data/game_breadth/2026-10-03 --status runs/game-breadth-2026-10-03/status.json --per-family 3000 --seed 20261003
flock /tmp/horscht-game-breadth-cpu.lock python -m pytest tests/test_game_breadth.py tests/test_dice_decisions.py tests/test_mastermind.py tests/test_sokoban.py -q
```

The runner is serial, limits address space to 2 GiB and CPU time to 30 minutes, and reserves the final 30% of the wall budget for grouping and verification. It refuses existing outputs and started statuses. The same lock serialized child reference tests so simulator compute stayed within the combined two-thread allowance. No GPU/local model inference, external teacher binary, data-authoring API or active training mutation occurs. Three visible Luna MAX helpers implemented bounded adapters and reference tests; generated questions and labels are deterministic code.

Outcome distributions and chosen optimal-action policies are explicit different `target_semantics`. Exact rational evidence is retained parallel to shuffled choice IDs. Mastermind information guesses may present up to 264 distinct supplied candidates out of up to 1,296 legal codes; their scope is best among supplied candidates. No softmax of engine or confidence scores is used. The generator uses English structured questions; broader natural-language variation is future work.

Known-law forecasts never become IDK just because a random outcome is unobserved. Separate evidence questions ask which actual realized dice sum or Mastermind symbol is **established**, giving IDK only when the supplied observation cannot establish that fact. Sokoban player-location omissions explicitly provide neither a player observation nor a prior. They carry one-hot task IDK after proving multiple compatible player states, rather than pretending an arbitrary uniform prior is truth. Forecast rows keep exact probabilities even when no single realized outcome is known.

Question identity includes the actual query and relevant canonical state, excluding irrelevant presentation changes. Mastermind identities canonicalize public history order, symbol renaming, positional symmetries and candidate order jointly. Dice laws normalize common weight factors and group related offers/roll counts and near-weight laws. Mastermind refinements sharing public feedback patterns are joined transitively before splitting. Sokoban groups full level families and near wall topologies under D4, including links to earlier grid-navigation versions. Exact prior file/manifest hashes are checked. Prior assignments are inherited when consistent; any component touching prior train and development (or prior quarantine) quarantines all new related rows without changing old data.

The final manifest separates train/development/quarantine allowlists and offers equal family-sampling weights for future training. Raw row counts can be uneven across splits because families are indivisible. Every saved record is reconstructed from its permitted observation and relabelled by exact enumeration/solver before publication; there is no hidden code field in model input. Earlier dataset coordinators and active snapshots stay untouched. Self-generated source/rules code has no external source content or engine license to attribute; the manifest records that the repository license is not declared, rather than inventing a license grant.


The verified 2026-10-03 publication contains **8,907 unique originals**: dice 3,000, Mastermind 2,907 and Sokoban 3,000; train 7,674 / development 1,233. An effective-public-history audit excludes 93 redundant-feedback duplicate questions from 9,000 candidates. Repeating identical feedback adds no posterior evidence and therefore cannot create a new original. The exclusions ledger is separate from all training allowlists. `breadth_quality` preserves the assigned split families, normalizes repeated public observations and relabels every retained record. Original generator/quality source snapshots are archived with hashes alongside current source hashes. There are no prior-linked quarantine conflicts.

Coverage includes 1,175 dice-sum distributions, 1,234 expected-payoff policies, 1,223 hidden-symbol posterior distributions, 466 feedback predictions, 521 information-guess policies, 2,038 Sokoban shortest first moves and 661 next-move solvability classifications (52 proved irreversible deadlocks). The achieved IDK count is 1,284. Sokoban includes 524 two-box levels, with exact shortest paths up to 29 moves; only 14 Sokoban development rows remain after conservative world grouping. Choices range from 3 to 265 including task IDK; ten information-guess rows use all 265 heads. The manifest retains distinct semantics and equal family sampling weights. All retained targets and source/prior/output hashes were verified, and 79 focused tests passed. No further generation is running.
