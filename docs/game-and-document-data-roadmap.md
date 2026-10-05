# Game and document training data roadmap

Updated: 2026-10-03

Status: implemented foundations plus the full prospective backlog below. Nicolas has separately authorized bounded game/document generation waves; this roadmap itself does not authorize autonomous generation, simulator installation, or changes to the running training experiment. Older production coordinators remain paused. The current Qwen3-4B model is text-only; all new data belongs in a future frozen dataset version.

## Breadth and depth

Keep the full catalogue below so later work does not repeatedly rediscover or narrow the options. Aim initially for broad coverage with modest samples per implemented game, not exclusively easy questions. Include a range of decision depths, ambiguous cases, distributions and missing-information cases. Invest in reliable reusable simulators and labelers; deepen selected games when that infrastructure makes additional varied examples inexpensive.

Question count, reasoning depth and implementation effort are separate budgets. Many shallow duplicates are not broad coverage. A small collection of expensive deep examples is not broad coverage either. There is no committed per-game quota yet. Do not implement every game at once.

Generalization is a hypothesis to test, not a consequence of collecting many games. Reserve unseen scenario families, rule variants and some whole games for internal development/evaluation before generating their training relatives. External blind benchmarks must not guide authoring. Measure transfer to separately designed non-game tasks, and actual game performance, rather than relying only on shuffled-question accuracy.

## Complete candidate catalogue

These are prospective task ideas, not verified integrations or promises of exact optimal labels. Rules and variant must be explicit. Search-based labels record the engine, version, objective, budget and uncertainty. A legal simulator alone does not provide optimal decisions.

| Board game | Decision emphasis |
|---|---|
| Chess | Threats, exchanges, forced continuations, positional choices |
| Chess960 | Familiar rules with unfamiliar starting arrangements |
| Mini-chess | Smaller search spaces and exact tactical labels |
| Antichess | Reversed objectives and compulsory captures |
| Crazyhouse | Moving versus deploying captured pieces |
| Checkers / draughts | Capture chains, promotions, sacrifices; pin variant |
| Shogi | Promotion and piece drops |
| Xiangqi | Different movement constraints and tactical geometry |
| Small-board Go | Connectivity, capture and territory |
| Othello / Reversi | Mobility versus immediate gain |
| Connect Four | Threats, forced wins and blocking |
| Gomoku | Intersecting threats |
| Hex | Connection planning |
| Havannah | Multiple winning structures |
| Tic-tac-toe | Cheap exact introductory decisions |
| Ultimate tic-tac-toe | Local moves with global consequences |
| Nine Men's Morris | Mills, captures and mobility |
| Mancala / Kalah | Redistribution and extra turns; pin variant |
| Dots and Boxes | Chain sacrifices |
| Amazons | Mobility and territorial blocking |
| Breakthrough | Pawn advance and sacrifice |
| Quoridor | Movement versus barriers |
| Chinese checkers | Jump chains and future mobility |
| Hive | Connectivity and maneuvering |

| Puzzle | Decision emphasis |
|---|---|
| Sokoban | Push dependencies and irreversible deadlocks |
| Sliding-tile puzzles | Reachability and shortest paths |
| Rush Hour | Blocking dependencies |
| Tower of Hanoi | Constrained sequences |
| Peg solitaire | Preserving solvability |
| Lights Out | Interacting switch effects |
| Sudoku | Forced values versus guesses |
| Killer Sudoku | Arithmetic and placement |
| KenKen | Mixed arithmetic constraints |
| Kakuro | Intersecting sums |
| Nonograms | Row/column constraint integration |
| Slitherlink | Local counts and global loop constraints |
| Hashiwokakero | Connectivity and bridge counts |
| Minesweeper | Safe cells and probabilities under stated priors |
| Mastermind | Hidden-code inference and information gathering |
| Wordle-style games | Guess selection with fixed dictionary and prior |
| Word ladders | Valid transformations |
| Boggle | Word paths under a fixed vocabulary |
| Countdown numbers | Arithmetic construction |
| 2048 | Merges and future flexibility |
| Tetris | Reachable placements under explicit objectives |

| Card or dice game | Decision emphasis |
|---|---|
| Blackjack | Expected value under explicit deck/dealer rules |
| Video poker | Card retention and expected payout |
| Kuhn poker | Small mixed-strategy problems |
| Leduc poker | Betting and hidden cards |
| Limit Texas hold'em | Decisions under an opponent model |
| No-limit poker | Supplied bet sizes and risk |
| Liar's Dice | Belief updates and challenges |
| Yahtzee | Rerolls and category allocation |
| Pig | Bank versus continue |
| Farkle | Risk and stopping |
| Backgammon | Planning with dice |
| Cribbage | Discards and sequencing |
| Gin rummy | Retention and disclosure |
| Hearts | Hazardous cards and remaining holdings |
| Spades | Bidding and partnership |
| Bridge | Bidding, inference and play as distinct tasks |
| Euchre | Trump and trick management |
| Durak | Attack, defense and resources |
| Crazy Eights | Disposal versus flexibility |
| Dominoes | Inference from passes and plays |
| Goofspiel | Bidding across successive prizes |

| Hidden-information or cooperative game | Decision emphasis |
|---|---|
| Battleship | Search and ship-location beliefs |
| Hanabi | Hints, knowledge and cooperation |
| Stratego | Hidden identities inferred from actions |
| Clue / Cluedo | Deduction over observation history |
| Memory / Concentration | Recall and exploration |
| Scotland Yard | Hidden-location tracking |
| Codenames | Semantic clues; subjective rubric, not exact truth |
| The Crew | Cooperation with restricted communication |
| Cooperative box-pushing | Roles and sequencing |
| Overcooked-style grids | Dependencies, allocation and congestion |
| Werewolf / Mafia | Claims and uncertain identities; policy-dependent labels |
| The Resistance / Avalon | Teams and uncertain allegiance |

| Resource-management game | Decision emphasis |
|---|---|
| Splendor | Purchases, reservations and resource collection |
| Dominion | Deck improvements and delayed payoff |
| Catan | Placement, trading and resource uncertainty |
| Ticket to Ride | Routes and scarce links |
| Carcassonne | Placement and limited commitments |
| Azul | Drafting and penalties |
| 7 Wonders | Drafting across alternative scoring paths |
| Pandemic | Threat prioritization and coordination |
| Risk | Force allocation under uncertain combat |
| Diplomacy | Simultaneous orders and cooperation |
| Small Civilization-style games | Research, production, expansion and defense |
| Tower defense | Budget allocation and placement |
| Deck-building combat | Cards, energy, enemy intentions and future risk |
| Trading and auctions | Bids under explicit valuations |

| Video-game family | Text-state decisions |
|---|---|
| Snake | Routes and escape preservation |
| Pac-Man-style mazes | Routes, hazards and power-up timing |
| Boulder Dash-style puzzles | Falling objects and safe sequences |
| Roguelike combat | Movement, inventory and combat choices |
| Platformers | Actions from position, velocity and obstacles |
| Racing | Control from speed and track geometry |
| Capture the Flag | Attack, defense and interception |
| Grid navigation | Keys, doors, hazards and resource limits |
| Doom-style scenarios | Navigation, resource management and short action sequences |

The initial candidate subset discussed was chess, Connect Four, Othello, Sokoban, Sudoku, Minesweeper, Mastermind, Tetris, 2048, Blackjack, Liar's Dice and a small poker game. This is a starting proposal, not authorization or a permanent limit.

## Shared training interface

Use versioned rules + allowed observations/history + objective + candidate actions/outcomes -> aligned target distribution. Preserve provenance, scenario/episode IDs, observation restrictions, label method and confidence. Keep all related positions, translations, counterfactuals and permutations within one split; near-identical trajectories must not straddle splits.

Ask about legality, next-state consequences, forced outcomes, best supplied action, outcome probabilities, informative observations and sufficiency of evidence. Softmaxing search scores produces a chosen teacher policy, not calibrated probabilities that moves are correct. Distinguish outcome distributions from action policies. Permit multiple equally good actions and label ties explicitly. Unknown outcomes under a known probability law are not automatically IDK.

The fixed 265-head interface bounds the number of presented choices. Larger legal action spaces require declared candidate selection or hierarchical decisions. Report performance as best among supplied candidates when that is what was trained. Test candidate-order robustness. Never count reordered choices as new original problems.

Use engines, exact solvers and ordinary simulation code for labels wherever feasible. Luna can provide bounded wording and variation; no autonomous bulk rollout or expensive second-model review is approved. Record accepted examples per unit of compute and API-equivalent authoring cost. Running CPU simulations must have resource limits because the active GPU training shares system memory and power.

## Doom-style abstraction and transfer

Nicolas recalls a Jev Doom demo. Its particular implementation and performance have not been verified here; treat it as inspiration, not evidence for our model.

Proposed loop: simulator -> permitted observations -> compact structured state/history -> model choice -> bounded action executor -> next observation and reward. A text model can choose from state descriptions without handling pixels. ViZDoom exposes configurable actions and game variables, and optional screen/depth/label buffers; extracting a suitable observation remains our adapter's responsibility.

Start with small scenarios: reach a destination, acquire a key, find a health item, conserve a limited resource, or choose between routes with different risk. Then combine objectives and require memory across observations. Freeze action durations or terminate actions on defined events, and record them in the task. Benchmark end-to-end decision latency; do not assume one decision per frame is practical.

Separate player-visible information from privileged simulator state. Labels and evaluation must respect the intended observation model. An adapter that supplies hidden positions creates an oracle task, not realistic perception. Keep structured-state and later screenshot tracks explicitly distinct.

Abstract transferable concepts include navigation, prerequisite tracking, resource budgeting, information gathering, uncertainty, recovery and delayed rewards. Re-express these in independently generated non-game settings (document workflows, scheduling, inventory, tool selection) with different surface forms. Test on held-out mechanics and task families. Game competence does not establish real-world competence; better abstractions can also remove the difficulty, so document what the adapter solves for the model.

## Documents remain a parallel source

For every source with multilingual editions or collections, acquire across its eligible languages, including Wikipedia. Inventory coverage and document exclusions rather than silently defaulting to English. Use source-native material, track translations, and keep related versions in the same split. The [source catalogue](article-source-catalogue.md#multilingual-acquisition-rule) defines the shared acquisition rule and current access decisions.

Use reusable articles, papers and wiki revisions with provenance/license metadata. Include bilingual and mixed-language passages, topic classification, evidence quality, stance expressed in text, cross-section inference, contradiction and missing-information questions. Author affiliation must not be inferred from unsupported textual cues. All questions from one document and closely related revisions remain together in splits.

Reporting was approved as a first-class part of the planned article dataset on 2026-10-02, including the initial acquisition pilot after source checks. Include local and international reporting across languages and subjects. Train comprehension, claim attribution, multi-paragraph evidence integration, comparisons between accounts, corrections and uncertainty. Distinguish what a source says from what its evidence establishes; probability estimates need explicit provenance and must not masquerade as measured truth. Group related event coverage, syndication and translations before splitting, preserving publication times and revisions. Source candidates and admission rules are in the [article source catalogue](article-source-catalogue.md#news-organizations-and-archives). This records dataset scope, not a restart of paused bulk generation or a change to the active training snapshot.

Long text alone is not long-context reasoning. Selected tasks must require evidence spread across the document. Keep classification tasks too, but measure their coverage separately. Preserve full inputs under the measured training budget; do not pad or silently truncate.

## Sources and implementation leads

- [OpenSpiel game catalogue](https://github.com/google-deepmind/open_spiel/blob/master/docs/games.md): implementation lead; per-game maturity varies.
- [PettingZoo classic environments](https://pettingzoo.farama.org/environments/classic/): shared interfaces for several games.
- [Gymnasium toy-text environments](https://gymnasium.farama.org/environments/toy_text/): small navigation/decision environments.
- [ViZDoom API](https://vizdoom.farama.org/main/api/python/doom_game/): configurable actions, game variables and observation buffers.
- [Stockfish evaluation explanation](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html): evaluation scores and outcome estimates require careful interpretation.
- [Wikimedia content reuse](https://developer.wikimedia.org/use-content/content/): content access and attribution guidance.

Reviewed metadata/documentation on 2026-10-01. No external benchmark examples are needed for this roadmap.


<!-- game-systems-implemented-2026-10-03 -->
## Implemented game systems and remaining backlog

These are concrete local adapters, not integrations with every catalogue entry. All use self-generated development/training states, preserve explicit observations and rules, and keep external benchmarks isolated.

| Implemented family | Label contract | Current data / limits |
|---|---|---|
| Tic-tac-toe | Exact minimax, ties and D4 state families | Pilot; later related expansion rows quarantined to preserve prior splits |
| Connect Four | Exact whole-game proofs when cheap; exact explicitly declared 3/5-ply terminal-reward objectives elsewhere | 2026-10-02 expansion covers early/middle/late positions; finite horizon is not full-game optimality |
| Normal-play Nim | Exact XOR-optimal action policy | Varied heaps, genuine ties, up to 264 legal actions plus IDK |
| Static grid navigation | Exact BFS shortest-path action policy | Fully observed square maps; shared/near terrain families grouped |
| Lights Out | GF(2) minimum-weight solution policy | Exact 5×5 generation; exhaustive 3×3 reference checks |
| Weighted dice threshold decisions | Exact finite-law outcome distributions and risk-neutral expected-value policies | Independent dice, explicit weights/payoffs; one-step game, not Blackjack |
| Mastermind | Exact uniform-prior posterior/predictive distributions, information-action policies, and logical certainty | Repeats allowed; labels use only public guess/feedback history; selected guess scope declared |
| Sokoban | Exact shortest walk/push solutions and complete-search irreversible-deadlock classification | Bounded square 5–7 boards, one/two boxes; search cutoff is not deadlock |

The 2026-10-03 wave is verified complete: 8,907 unique accepted originals (3,000 dice, 2,907 Mastermind, 3,000 Sokoban), split into 7,674 training and 1,233 development rows. The quality ledger excludes 93 repeated-feedback duplicate questions. There are 1,284 evidence-based IDK rows, separate from exact known-law forecasts. Status: `runs/game-breadth-2026-10-03/status.json`; family-sampling allowlists and source/license/provenance records: `data/game_breadth/2026-10-03/manifest.json`. The generated source run took 24.131 seconds, with a 9.496-second semantic quality refinement; 79 focused tests passed. Nothing is merged into the active training snapshot. Sokoban has only 14 development rows after indivisible terrain grouping, so no stable per-family development accuracy is claimed.

Blackjack, Minesweeper and chess remain unimplemented adapters, alongside the rest of the catalogue. No Stockfish executable was found in PATH or `/usr/games/stockfish` for this wave. A later chess integration still needs official engine documentation, a pinned binary/version/license, explicit CPU/node budgets and honest engine-policy semantics; scores must not be relabelled calibrated probabilities. Mastermind adds real permitted-observation inference now; it does not claim general hidden-information game competence. Sokoban adds irreversible planning now; it does not claim large-level search or visual perception. Generalization and balanced-mixture training remain hypotheses to test on separately grouped internal tasks.
<!-- /game-systems-implemented-2026-10-03 -->
