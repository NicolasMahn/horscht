"""CPU-only game adapters and conservative labels for future dataset versions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import reduce
import hashlib
import json
import operator
import random
from typing import Protocol

from dataset_factory.probability_v2 import validate_v2_row

VERSION = "game-data-v1"
Action = tuple[int, ...]


class GameState(Protocol):
    player: int
    game: str
    rules: str

    def legal_actions(self) -> tuple[Action, ...]: ...
    def apply(self, action: Action) -> GameState: ...
    def outcome(self) -> int | None: ...
    def observation(self) -> dict: ...


@dataclass(frozen=True)
class Board:
    """Construct positions by replay; cells are row-major, top row first."""
    cells: tuple[int, ...]
    player: int = 1
    game: str = "tic_tac_toe"

    def __post_init__(self):
        if self.game not in {"tic_tac_toe", "connect_four"}:
            raise ValueError("unsupported board game")
        size = 9 if self.game == "tic_tac_toe" else 42
        if not isinstance(self.cells, tuple) or len(self.cells) != size or any(type(v) is not int or v not in (-1, 0, 1) for v in self.cells):
            raise ValueError("invalid cells")
        if type(self.player) is not int or self.player not in (-1, 1):
            raise ValueError("invalid player")
        ones, twos = self.cells.count(1), self.cells.count(-1)
        if ones - twos != (0 if self.player == 1 else 1):
            raise ValueError("turn and move counts disagree")
        if self.game == "connect_four":
            for column in range(7):
                seen_piece = False
                for row in range(6):
                    piece = self.cells[row * 7 + column]
                    if piece:
                        seen_piece = True
                    elif seen_piece:
                        raise ValueError("floating piece")
        winners = self._winners()
        if len(winners) > 1 or (winners and next(iter(winners)) != -self.player):
            raise ValueError("invalid winner and turn")
        if winners:
            # A winning position must have a possible final move after which
            # play stopped. This rejects boards filled after an earlier win.
            width = 3 if self.game == "tic_tac_toe" else 7
            predecessors = []
            for index, value in enumerate(self.cells):
                if value != -self.player:
                    continue
                if self.game == "connect_four" and index >= width and self.cells[index - width] != 0:
                    continue
                cells = list(self.cells)
                cells[index] = 0
                try:
                    predecessor = Board(tuple(cells), -self.player, self.game)
                except ValueError:
                    continue
                predecessors.append(predecessor)
            if not any(p.outcome() is None for p in predecessors):
                raise ValueError("position continued after a win")

    @property
    def rules(self):
        return "standard-3x3-v1" if self.game == "tic_tac_toe" else "gravity-6x7-connect4-v1"

    @classmethod
    def initial(cls, game="tic_tac_toe"):
        return cls((0,) * (9 if game == "tic_tac_toe" else 42), game=game)

    def _winners(self):
        height, width, length = (3, 3, 3) if self.game == "tic_tac_toe" else (6, 7, 4)
        winners = set()
        for row in range(height):
            for col in range(width):
                value = self.cells[row * width + col]
                if not value:
                    continue
                for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
                    if 0 <= row + dr * (length - 1) < height and 0 <= col + dc * (length - 1) < width:
                        if all(self.cells[(row + dr * k) * width + col + dc * k] == value for k in range(length)):
                            winners.add(value)
        return winners

    def outcome(self):
        winners = self._winners()
        return next(iter(winners)) if winners else (0 if 0 not in self.cells else None)

    def legal_actions(self):
        if self.outcome() is not None:
            return ()
        if self.game == "tic_tac_toe":
            return tuple((i,) for i, value in enumerate(self.cells) if not value)
        return tuple((col,) for col in range(7) if self.cells[col] == 0)

    def apply(self, action):
        if action not in self.legal_actions():
            raise ValueError("illegal action")
        index = action[0]
        if self.game == "connect_four":
            index = next(row * 7 + index for row in range(5, -1, -1) if self.cells[row * 7 + index] == 0)
        cells = list(self.cells)
        cells[index] = self.player
        return Board(tuple(cells), -self.player, self.game)

    def observation(self):
        return {"game": self.game, "rules": self.rules, "player": self.player,
                "cells": list(self.cells), "visibility": "full_public_state"}


@dataclass(frozen=True)
class Nim:
    heaps: tuple[int, ...]
    player: int = 1
    game: str = "nim"
    rules: str = "normal-play-labelled-heaps-v1"

    def __post_init__(self):
        if not isinstance(self.heaps, tuple) or not self.heaps or len(self.heaps) > 16 or any(type(n) is not int or not 0 <= n <= 256 for n in self.heaps):
            raise ValueError("Nim requires 1..16 heaps of size 0..256")
        if type(self.player) is not int or self.player not in (-1, 1):
            raise ValueError("invalid player")
        if self.game != "nim" or self.rules != "normal-play-labelled-heaps-v1":
            raise ValueError("unsupported Nim rules")

    def outcome(self):
        return -self.player if not any(self.heaps) else None

    def legal_actions(self):
        return tuple((i, count) for i, n in enumerate(self.heaps) for count in range(1, n + 1))

    def apply(self, action):
        if action not in self.legal_actions():
            raise ValueError("illegal action")
        heaps = list(self.heaps)
        heaps[action[0]] -= action[1]
        return Nim(tuple(heaps), -self.player)

    def observation(self):
        return {"game": self.game, "rules": self.rules, "player": self.player,
                "heaps": list(self.heaps), "visibility": "full_public_state"}


@dataclass(frozen=True)
class Budget:
    max_nodes: int = 20_000
    max_depth: int = 9

    def __post_init__(self):
        if type(self.max_nodes) is not int or not 1 <= self.max_nodes <= 100_000:
            raise ValueError("node budget must be 1..100000")
        if type(self.max_depth) is not int or not 1 <= self.max_depth <= 42:
            raise ValueError("depth budget must be 1..42")


@dataclass(frozen=True)
class TeacherResult:
    actions: tuple[Action, ...]
    value_bounds: tuple[tuple[int, int], ...]
    optimal_actions: tuple[Action, ...]
    label_method: str
    nodes: int
    budget: Budget


def teach(state: GameState, candidates: tuple[Action, ...] | None = None, budget: Budget = Budget()) -> TeacherResult:
    """Prove ties using minimax bounds; unresolved search never fabricates targets."""
    legal = state.legal_actions()
    actions = legal if candidates is None else tuple(candidates)
    if not actions or len(set(actions)) != len(actions) or any(action not in legal for action in actions):
        raise ValueError("candidates must be distinct legal actions from a nonterminal state")
    if len(actions) > 264:
        raise ValueError("select at most 264 candidate actions, reserving one head for IDK")
    if isinstance(state, Nim):
        bounds = tuple((1, 1) if reduce(operator.xor, state.apply(a).heaps, 0) == 0 else (-1, -1) for a in actions)
        best = max(v[0] for v in bounds)
        return TeacherResult(actions, bounds, tuple(a for a, v in zip(actions, bounds) if v[0] == best), "exact_nim_xor", 0, budget)

    nodes = 0
    cache = {}

    def search(position, depth):
        nonlocal nodes
        outcome = position.outcome()
        if outcome is not None:
            value = outcome * position.player
            return value, value
        key = (position, depth)
        if key in cache:
            return cache[key]
        if depth == 0 or nodes >= budget.max_nodes:
            return -1, 1
        nodes += 1
        values = []
        for action in position.legal_actions():
            lo, hi = search(position.apply(action), depth - 1)
            values.append((-hi, -lo))
            if values[-1] == (1, 1):
                break
        result = max(v[0] for v in values), max(v[1] for v in values)
        cache[key] = result
        return result

    bounds = []
    for action in actions:
        lo, hi = search(state.apply(action), budget.max_depth - 1)
        bounds.append((-hi, -lo))
    best_lower = max(lo for lo, hi in bounds)
    possible = [i for i, (lo, hi) in enumerate(bounds) if hi >= best_lower]
    proven = all(bounds[i] == (best_lower, best_lower) for i in possible)
    optimal = tuple(actions[i] for i in possible) if proven else ()
    method = "exact_minimax" if all(lo == hi for lo, hi in bounds) else "bounded_minimax"
    return TeacherResult(actions, tuple(bounds), optimal, method, nodes, budget)


def state_id(state: GameState) -> str:
    # Tic-tac-toe rotations/reflections share a split family, even across episodes.
    observation = state.observation()
    if isinstance(state, Board) and state.game == "tic_tac_toe":
        grids = []
        cells = state.cells
        for _ in range(4):
            grids.extend((cells, tuple(cells[r * 3 + (2 - c)] for r in range(3) for c in range(3))))
            cells = tuple(cells[(2 - c) * 3 + r] for r in range(3) for c in range(3))
        observation["cells"] = list(min(grids))
    if isinstance(state, Board) and state.game == "connect_four":
        reflected = tuple(state.cells[r * 7 + (6 - c)] for r in range(6) for c in range(7))
        observation["cells"] = list(min(state.cells, reflected))
    if isinstance(state, Nim):
        observation["heaps"] = sorted(state.heaps)
    return hashlib.sha256(json.dumps(observation, sort_keys=True).encode()).hexdigest()[:24]


def decision_row(state: GameState, *, scenario_id: str, trajectory_id: str | None = None,
                 candidates: tuple[Action, ...] | None = None, seed: int = 0,
                 budget: Budget = Budget(), difficulty: str = "easy") -> dict:
    """Export a full-observation action policy; grouping edges must precede splitting."""
    if not scenario_id.strip() or (trajectory_id is not None and not trajectory_id.strip()):
        raise ValueError("scenario and trajectory IDs must be nonempty")
    if difficulty not in {"easy", "medium", "hard"}:
        raise ValueError("unsupported curriculum difficulty")
    result = teach(state, candidates, budget)
    if not result.optimal_actions:
        raise ValueError("bounded search has not proved the optimal candidate set; omit this row")
    ordered = list(result.actions)
    random.Random(seed).shuffle(ordered)
    ids = ["action:" + ",".join(map(str, a)) for a in ordered] + ["task:idk"]
    targets = [1 / len(result.optimal_actions) if a in result.optimal_actions else 0 for a in ordered] + [0]
    digest = state_id(state)
    group = trajectory_id or scenario_id
    # A permutation changes presentation, but a different candidate set or
    # rotated observation is a separate row within the same split family.
    identity = {"observation": state.observation(), "candidates": sorted(result.actions)}
    row_digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
    row = {
        "id": f"game:{scenario_id}:{row_digest}", "problem_group_id": f"game:{group}",
        "split": "development", "operation_id": "best_supplied_action", "topic": state.game,
        "language": "en", "difficulty": difficulty,
        "context": json.dumps(state.observation(), sort_keys=True),
        "question": "Which supplied action maximizes win/draw/loss under optimal play? Actions use zero-based indices. Board actions place at a cell (tic-tac-toe) or column (Connect Four); Nim actions are (heap, number removed).",
        "options": [str(a) for a in ordered] + ["Cannot determine from the supplied information"],
        "choice_ids": ids, "answerability_status": "answerable", "unanswerable_reason": None,
        "abstention_choice_id": "task:idk",
        "explanation": "Uniform policy over all proved equally optimal supplied actions; this is not a win probability.",
        "provenance": f"Local deterministic {VERSION}; {state.rules}; no benchmark sources.",
        "game_metadata": {
            "version": VERSION, "rules_version": state.rules, "state_family_id": digest,
            "scenario_id": scenario_id, "trajectory_id": trajectory_id,
            "split_group_keys": [f"scenario:{scenario_id}", f"state:{state.game}:{digest}"] + ([f"trajectory:{trajectory_id}"] if trajectory_id else []),
            "candidate_scope": "all_legal" if candidates is None else "supplied_subset",
            "legal_action_count": len(state.legal_actions()), "seed": seed,
            "observation_model": "full_public_state", "objective": "maximize minimax win/draw/loss",
            "curriculum": {"difficulty": difficulty, "remaining_units": sum(state.heaps) if isinstance(state, Nim) else state.cells.count(0)},
            "teacher": {**asdict(result), "distribution_semantics": "uniform_optimal_candidate_action_policy"},
        },
    }
    row.update({"schema_version": "probability-v2", "supervision_kind": "soft_distribution",
                "distribution_source_kind": "teacher_estimate", "target_distribution": targets,
                "distribution_evidence": {"kind": "teacher_estimate", "model_id": "local-game-solver",
                    "model_version": VERSION, "estimation_method": result.label_method,
                    "choice_ids": ids}})
    errors = validate_game_row(row)
    if errors:
        raise ValueError("invalid game row: " + "; ".join(errors))
    return row


def validate_game_row(row: dict, *, allowed_splits: tuple[str, ...] = ("development",)) -> list[str]:
    """Reuse v2 math/IDK rules with an explicitly allowed truthful envelope.

    The old content validator mandates train + Luna provenance. Those two
    constraints cannot describe deterministic solver fixtures.
    """
    errors = [error for error in validate_v2_row(row)
              if error not in {"split_is_not_train", "provenance_does_not_match_required_label"}]
    if row.get("split") not in allowed_splits:
        errors.append("game_split_is_not_explicitly_allowed")
    if not str(row.get("provenance", "")).startswith(f"Local deterministic {VERSION};"):
        errors.append("game_fixture_requires_solver_provenance")
    return errors


def connected_group_ids(rows: list[dict]) -> list[str]:
    """Merge scenario, episode and symmetric-state links before assigning splits.

    A single shared state links whole trajectories, transitively. This can
    intentionally collapse many episodes into one group; never override links
    merely to achieve preferred split sizes.
    """
    parents: dict[str, str] = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    keys_by_row = []
    for row in rows:
        keys = row.get("game_metadata", {}).get("split_group_keys")
        if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or not key for key in keys):
            raise ValueError("each row requires nonempty split_group_keys")
        keys_by_row.append(keys)
        for key in keys[1:]:
            left, right = find(keys[0]), find(key)
            parents[max(left, right)] = min(left, right)
    return ["game-group:" + hashlib.sha256(find(keys[0]).encode()).hexdigest()[:24]
            for keys in keys_by_row]
