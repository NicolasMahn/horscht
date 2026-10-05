"""Finite CPU expansion with prior-version isolation and exact objective labels."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
import resource
import time

from dataset_factory.game_data import VERSION, Board, Budget, Nim, state_id, teach, validate_game_row
from dataset_factory.game_data.grid_navigation import GridNavigation
from dataset_factory.game_data.lights_out import LightsOut

EXPANSION_VERSION = "game-expansion-v1"
QUOTAS = {"connect_four": 2500, "nim": 2300, "grid_navigation": 2500, "lights_out": 2500, "tic_tac_toe": 200}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


@lru_cache(maxsize=8)
def square_maps(size):
    mappings = []
    for reflect in (False, True):
        for rotations in range(4):
            mapping = []
            for position in range(size * size):
                row, col = divmod(position, size)
                if reflect:
                    col = size - 1 - col
                for _ in range(rotations):
                    row, col = col, size - 1 - row
                mapping.append(row * size + col)
            mappings.append(tuple(mapping))
    return tuple(mappings)


def bits(cells):
    return sum(value << i for i, value in enumerate(cells))


def transformed_masks(mask, size):
    return tuple(sum(1 << mapping[i] for i in range(size * size) if mask & (1 << i))
                 for mapping in square_maps(size))


def lights_from_presses(size, presses):
    cells = [0] * (size * size)
    for index in presses:
        row, col = divmod(index, size)
        for nr, nc in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= nr < size and 0 <= nc < size:
                cells[nr * size + nc] ^= 1
    return LightsOut(tuple(cells), size)


def horizon_teacher(state: Board, depth: int):
    """Exact finite game: nonterminal horizon reward is explicitly zero."""
    if state.game != "connect_four" or depth not in (3, 5):
        raise ValueError("declared C4 horizon must be 3 or 5")
    cache = {}
    nodes = 0

    def search(position, remaining):
        nonlocal nodes
        outcome = position.outcome()
        if outcome is not None:
            return outcome * position.player
        if remaining == 0:
            return 0
        key = (position, remaining)
        if key in cache:
            return cache[key]
        nodes += 1
        best = -1
        for action in position.legal_actions():
            best = max(best, -search(position.apply(action), remaining - 1))
            if best == 1:
                break
        cache[key] = best
        return best

    actions = state.legal_actions()
    values = tuple(-search(state.apply(action), depth - 1) for action in actions)
    best = max(values)
    return {"actions": actions, "value_bounds": tuple((v, v) for v in values),
            "optimal_actions": tuple(a for a, v in zip(actions, values) if v == best),
            "label_method": "exact_declared_horizon_minimax", "nodes": nodes,
            "horizon_plies": depth, "nonterminal_horizon_reward": 0}


def export_row(state, teacher, *, scenario, seed, objective, generation):
    actions = list(teacher["actions"])
    if len(actions) < 2 or len(actions) > 264 or not teacher["optimal_actions"]:
        raise ValueError("require 2..264 legal actions and a proved optimal set")
    random.Random(seed).shuffle(actions)
    identifiers = ["action:" + ",".join(map(str, action)) for action in actions] + ["task:idk"]
    optimal = set(teacher["optimal_actions"])
    targets = [1 / len(optimal) if action in optimal else 0 for action in actions] + [0]
    observation = state.observation()
    observation["objective"] = objective
    family = (state.state_family_id() if isinstance(state, GridNavigation) else
              state.family_id() if isinstance(state, LightsOut) else state_id(state))
    identity = hashlib.sha256(json.dumps({"observation": observation, "actions": sorted(actions)}, sort_keys=True).encode()).hexdigest()[:24]
    explanation = "Uniform policy over all equally optimal supplied actions under the stated objective; not outcome probabilities."
    instruction = {
        "tic_tac_toe": "Actions are (zero-based row-major cell index,).",
        "connect_four": "Actions are (zero-based column index,). Gravity drops a piece into the lowest empty cell.",
        "nim": "Actions are (zero-based heap index, number removed). Removing the last object wins.",
        "grid_navigation": "Actions are (direction,): 0=up, 1=right, 2=down, 3=left; walls cannot be crossed.",
        "lights_out": "Actions are (zero-based row-major cell index,). Pressing toggles that cell and orthogonal neighbors; turn all lights off.",
    }[state.game]
    shortest = teacher.get("shortest_distance")
    complexity = shortest if shortest is not None else (sum(state.heaps) if isinstance(state, Nim) else state.cells.count(0))
    difficulty = "easy" if complexity <= 4 else "medium" if complexity <= 9 else "hard"
    row = {
        "id": f"{scenario}:{identity}", "problem_group_id": f"pending:{scenario}", "split": "development",
        "operation_id": "best_supplied_action", "topic": state.game, "language": "en", "difficulty": difficulty,
        "context": json.dumps(observation, sort_keys=True),
        "question": "Which supplied action is optimal under the objective explicitly stated in the observation? " + instruction,
        "options": [str(action) for action in actions] + ["Cannot determine from the supplied information"],
        "choice_ids": identifiers, "answerability_status": "answerable", "unanswerable_reason": None,
        "abstention_choice_id": "task:idk", "explanation": explanation,
        "provenance": f"Local deterministic {VERSION}; {state.rules}; {EXPANSION_VERSION}; self-generated, no benchmark sources.",
        "schema_version": "probability-v2", "supervision_kind": "soft_distribution", "distribution_source_kind": "teacher_estimate",
        "target_distribution": targets,
        "distribution_evidence": {"kind": "teacher_estimate", "model_id": "local-game-solver", "model_version": EXPANSION_VERSION,
                                  "estimation_method": teacher["label_method"], "choice_ids": identifiers},
        "game_metadata": {"version": VERSION, "rules_version": state.rules, "state_family_id": family,
            "scenario_id": scenario, "trajectory_id": scenario, "candidate_scope": "all_legal",
            "legal_action_count": len(actions), "seed": seed, "observation_model": "full_public_state",
            "objective": objective, "teacher": {**teacher, "distribution_semantics": "uniform_optimal_candidate_action_policy"},
            "generation": {**generation, "version": EXPANSION_VERSION, "original_problem_id": f"{scenario}:{identity}", "presentation_variant": False},
            "curriculum": {"difficulty": difficulty, "measure": complexity, "shortest_distance": shortest,
                           "method": "declared_exact_shortest_distance_or_remaining_units_thresholds_4_9_v1"}},
    }
    errors = validate_game_row(row)
    if errors:
        raise ValueError("invalid expansion row: " + "; ".join(errors))
    return row


def sample_board(game, rng, length):
    state = Board.initial(game)
    history = []
    for _ in range(length):
        actions = list(state.legal_actions())
        rng.shuffle(actions)
        for action in actions:
            child = state.apply(action)
            if child.outcome() is None:
                state = child
                history.append(list(action))
                break
        else:
            return None, history
    return state, history


def sample(game, rng, attempt):
    generation = {"attempt_index": attempt}
    if game == "connect_four":
        # Alternate independent state regimes. Early/mid positions use an
        # explicit short objective if whole-game proof is not economical.
        regime = ("early", "middle", "late")[attempt % 3]
        low, high = {"early": (6, 12), "middle": (13, 28), "late": (29, 38)}[regime]
        state, history = sample_board(game, rng, rng.randint(low, high))
        generation.update({"sample_method": "random_nonterminal_legal_replay", "replayed_actions": history, "phase": regime})
        if state is None:
            return None, None, None, generation
        if len(state.legal_actions()) < 2:
            return state, None, None, generation
        if state.cells.count(0) <= 9:
            solved = teach(state, budget=Budget(5000, 9))
            if solved.optimal_actions:
                return state, asdict(solved), {"kind": "full_game_minimax", "description": "Maximize win/draw/loss under optimal play."}, generation
        horizon = 3 if attempt % 2 == 0 else 5
        teacher = horizon_teacher(state, horizon)
        if len(set(teacher["value_bounds"])) == 1:
            return state, None, None, generation
        objective = {"kind": "finite_horizon_terminal_reward", "horizon_plies": horizon,
            "terminal_win_reward": 1, "terminal_loss_reward": -1, "terminal_draw_reward": 0,
            "nonterminal_at_horizon_reward": 0,
            "description": f"Maximize terminal win/loss reward over exactly the next {horizon} plies under adversarial optimal play. A nonterminal state at the horizon receives 0; this is not a draw or a full-game guarantee."}
        return state, teacher, objective, generation
    if game == "tic_tac_toe":
        state, history = sample_board(game, rng, rng.randint(1, 7))
        generation.update({"sample_method": "random_nonterminal_legal_replay", "replayed_actions": history})
        if state is None or len(state.legal_actions()) < 2:
            return state, None, None, generation
        teacher = asdict(teach(state, budget=Budget(5000, 9)))
        return state, teacher, {"kind": "full_game_minimax", "description": "Maximize win/draw/loss under optimal play."}, generation
    if game == "nim":
        if attempt % 5 == 0:
            target = rng.randint(230, 264)
            cuts = sorted(rng.sample(range(1, target), 7))
            heaps = tuple(b - a for a, b in zip([0] + cuts, cuts + [target]))
            kind = "high_cardinality_positive_partition"
        elif attempt % 5 == 1:
            heaps = tuple(rng.randint(1, 33) for _ in range(rng.randint(2, 7)))
            xor = 0
            for value in heaps:
                xor ^= value
            if xor:
                heaps += (xor,)
            kind = "xor_zero_losing_family"
        else:
            heaps = tuple(rng.randint(1, 33) for _ in range(rng.randint(2, 8)))
            kind = "uniform_bounded_heaps"
        if sum(heaps) > 264 or max(heaps) > 256:
            return None, None, None, generation
        state = Nim(heaps)
        generation.update({"sample_method": kind, "initial_heaps": list(heaps)})
        return state, asdict(teach(state)), {"kind": "full_game_minimax", "description": "Maximize win/loss under optimal normal-play Nim."}, generation
    if game == "grid_navigation":
        size = rng.randint(5, 9)
        density = rng.choice((0.15, 0.25, 0.35, 0.45))
        walls = tuple(int(rng.random() < density) for _ in range(size * size))
        free = [i for i, wall in enumerate(walls) if not wall]
        if len(free) < 2:
            return None, None, None, generation
        position, goal = rng.sample(free, 2)
        state = GridNavigation(walls, position, goal, size)
        generation.update({"sample_method": "independent_random_wall_map_and_distinct_endpoints", "obstacle_prior": density})
        if len(state.legal_actions()) < 2:
            return state, None, None, generation
        try:
            teacher = state.teach()
        except ValueError:
            return state, None, None, generation
        if teacher["shortest_distance"] < 2:
            return state, None, None, generation
        return state, teacher, {"kind": "exact_shortest_path", "description": "Choose a move on a shortest path to the stated goal, minimizing unit moves."}, generation
    if game == "lights_out":
        press_count = rng.randint(3, 22)
        presses = rng.sample(range(25), press_count)
        state = lights_from_presses(5, presses)
        generation.update({"sample_method": "reverse_scramble_from_all_off_goal", "scramble_press_indices": presses})
        if state.outcome() is not None:
            return state, None, None, generation
        teacher = state.teach()
        if teacher["shortest_distance"] < 3:
            return state, None, None, generation
        return state, teacher, {"kind": "exact_shortest_presses", "description": "Choose a first press in a shortest sequence that turns all lights off."}, generation
    raise ValueError("unknown game")


class UnionFind:
    def __init__(self, count):
        self.parent = list(range(count))
        self.size = [1] * count

    def find(self, value):
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.size[a] < self.size[b]:
            a, b = b, a
        self.parent[b] = a
        self.size[a] += self.size[b]
        return True


def _board_keys(observation):
    cells = observation["cells"]
    length = len(cells)
    aliases = [cells]
    if length == 9:
        for mapping in square_maps(3):
            transformed = [0] * 9
            for i, target in enumerate(mapping):
                transformed[target] = cells[i]
            aliases.append(transformed)
    else:
        aliases.append([cells[r * 7 + (6 - c)] for r in range(6) for c in range(7)])
    # Wildcarding two positions finds every <=2-cell relation in O(n^2)
    # per board without all-pairs comparison or storing keys in each row.
    for alias in aliases:
        packed = sum((value + 1) << (2 * i) for i, value in enumerate(alias))
        ordinal = 0
        for first in range(length):
            without_first = packed & ~(3 << (2 * first))
            for second in range(first + 1, length):
                erased = without_first & ~(3 << (2 * second))
                yield erased | (ordinal << (length * 2))
                ordinal += 1


def _neighbor_keys(observation):
    game = observation["game"]
    if game in ("tic_tac_toe", "connect_four"):
        yield from _board_keys(observation)
    elif game == "nim":
        heaps = sorted(observation["heaps"])
        yield tuple(heaps)
        for i, value in enumerate(heaps):
            for change in (-1, 1):
                if value + change >= 0:
                    neighbor = list(heaps)
                    neighbor[i] += change
                    yield tuple(sorted(neighbor))
    else:
        size = observation["size"]
        cells = observation["walls"] if game == "grid_navigation" else observation["cells"]
        variants = transformed_masks(bits(cells), size)
        shift = size * size
        yield min(variants) | (size << shift)
        # Shared one-bit neighbors cover <=2 observation-cell differences.
        for cell in range(shift):
            neighbor = min(mask ^ (1 << mapping[cell]) for mask, mapping in zip(variants, square_maps(size)))
            yield neighbor | (size << shift)
        if game == "lights_out":
            # Also group states within two legal presses, whose toggles change
            # several observed lights rather than only one.
            for cell in range(shift):
                row, col = divmod(cell, size)
                affected = [r * size + c for r, c in ((row, col), (row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
                            if 0 <= r < size and 0 <= c < size]
                neighbor = min(mask ^ sum(1 << mapping[i] for i in affected) for mask, mapping in zip(variants, square_maps(size)))
                yield neighbor | (size << shift)


def isolate_and_split(prior, rows, seed, update=None):
    combined = prior + rows
    groups = UnionFind(len(combined))
    prior_group_first = {}
    for index, row in enumerate(prior):
        key = row["problem_group_id"]
        if key in prior_group_first:
            groups.union(index, prior_group_first[key])
        else:
            prior_group_first[key] = index
    unions = Counter()
    for game in QUOTAS:
        neighbors = {}
        indices = [i for i, row in enumerate(combined) if row["topic"] == game]
        for offset, index in enumerate(indices):
            observation = json.loads(combined[index]["context"])
            for key in _neighbor_keys(observation):
                if key in neighbors:
                    if groups.union(index, neighbors[key]):
                        unions[game] += 1
                else:
                    neighbors[key] = index
            if update is not None and offset % 250 == 0:
                update({"phase": "grouping", "grouping_game": game, "grouping_processed": offset,
                        "max_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)})
        # Release the largest C4 wildcard index before indexing another game.
        del neighbors
    prior_splits = {}
    prior_ids = {}
    for index, row in enumerate(prior):
        root = groups.find(index)
        prior_splits.setdefault(root, set()).add(row["split"])
        prior_ids.setdefault(root, set()).add(row["problem_group_id"])
    component_id = {}
    for index, row in enumerate(combined):
        root = groups.find(index)
        key = f"{row['topic']}:{row['game_metadata']['state_family_id']}"
        component_id[root] = min(key, component_id.get(root, key))
    dispositions = Counter()
    for offset, row in enumerate(rows):
        root = groups.find(len(prior) + offset)
        linked = prior_splits.get(root, set())
        if len(linked) > 1:
            split = "quarantine"
            dispositions["prior_split_conflict_quarantined"] += 1
        elif linked:
            split = next(iter(linked))
            dispositions["prior_split_inherited"] += 1
        else:
            split = "development" if int(hashlib.sha256(f"{seed}:{component_id[root]}".encode()).hexdigest(), 16) % 5 == 0 else "train"
            dispositions["new_independent_component"] += 1
        identifiers = sorted(prior_ids.get(root, set()))
        row["split"] = split
        row["problem_group_id"] = (identifiers[0] if identifiers and split != "quarantine" else
                                   "game-expansion-group:" + hashlib.sha256(component_id[root].encode()).hexdigest()[:24])
        row["game_metadata"]["split_policy"] = "preserved_prior_groups_and_transitive_symmetry_neighbors_v1"
        row["game_metadata"]["prior_group_ids"] = identifiers
        row["game_metadata"]["prior_split_disposition"] = "conflict_quarantine" if split == "quarantine" else "inherited" if linked else "new_component"
    split_groups = {split: {row["problem_group_id"] for row in rows if row["split"] == split}
                    for split in ("train", "development", "quarantine")}
    if any(split_groups[a] & split_groups[b] for a, b in itertools.combinations(split_groups, 2)):
        raise ValueError("expansion split-group overlap")
    for row in rows:
        linked_splits = {old["split"] for old in prior if old["problem_group_id"] in row["game_metadata"]["prior_group_ids"]}
        if row["split"] != "quarantine" and linked_splits and linked_splits != {row["split"]}:
            raise ValueError("cross-version split violation")
    return {"components": len({groups.find(i) for i in range(len(prior), len(combined))}),
            "union_counts_by_game": dict(unions), "dispositions": dict(dispositions),
            "prior_rows_unchanged": True, "train_development_group_overlap": 0,
            "policy": "board<=2 cells under symmetries; Nim multiset L1<=2; grid topology goals/positions plus <=2 wall differences; LightsOut<=2 light differences or <=2 presses"}


def reconstruct(row):
    observation = json.loads(row["context"])
    game = row["topic"]
    if game in ("tic_tac_toe", "connect_four"):
        state = Board.initial(game)
        for action in row["game_metadata"]["generation"]["replayed_actions"]:
            state = state.apply(tuple(action))
    elif game == "nim":
        state = Nim(tuple(observation["heaps"]), observation["player"])
    elif game == "grid_navigation":
        state = GridNavigation(tuple(observation["walls"]), observation["position"], observation["goal"], observation["size"])
    else:
        state = lights_from_presses(observation["size"], row["game_metadata"]["generation"]["scramble_press_indices"])
    expected_observation = dict(observation)
    expected_observation.pop("objective")
    if state.observation() != expected_observation:
        raise ValueError("state replay differs from stored observation")
    return state


def verify_row(row):
    errors = validate_game_row(row, allowed_splits=("train", "development", "quarantine"))
    if errors:
        raise ValueError("schema error: " + "; ".join(errors))
    state = reconstruct(row)
    objective = row["game_metadata"]["objective"]
    if objective["kind"] == "finite_horizon_terminal_reward":
        expected = horizon_teacher(state, objective["horizon_plies"])
    elif isinstance(state, (GridNavigation, LightsOut)):
        expected = state.teach()
    else:
        expected = asdict(teach(state, budget=Budget(5000, 9)))
    optimal = {"action:" + ",".join(map(str, action)) for action in expected["optimal_actions"]}
    if not optimal:
        raise ValueError("unresolved saved target")
    if len(row["options"]) != len(state.legal_actions()) + 1:
        raise ValueError("candidate scope is not all legal actions")
    for identifier, target in zip(row["choice_ids"], row["target_distribution"]):
        if target != (1 / len(optimal) if identifier in optimal else 0):
            raise ValueError("target differs from exact recomputed policy")
    stored = row["game_metadata"]["teacher"]
    if [list(a) for a in expected["actions"]] != [list(a) for a in stored["actions"]]:
        raise ValueError("teacher actions changed")
    if [list(v) for v in expected["value_bounds"]] != [list(v) for v in stored["value_bounds"]]:
        raise ValueError("teacher values changed")
    return expected["nodes"]


def load_prior(directory):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = []
    references = []
    for record in manifest["files"]:
        path = Path(record["path"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise ValueError("prior file checksum changed")
        subset = [json.loads(line) for line in content.splitlines()]
        if len(subset) != record["rows"]:
            raise ValueError("prior row count changed")
        rows.extend(subset)
        references.append({"path": str(path), "sha256": record["sha256"], "rows": len(subset)})
    return rows, {"manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "files": references}


def write_jsonl(path, rows):
    digest = hashlib.sha256()
    with path.open("wb") as stream:
        for row in rows:
            line = (json.dumps(row, sort_keys=True) + "\n").encode()
            stream.write(line)
            digest.update(line)
    return {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def generate(output_dir, status_path, prior_dir, *, seed=2026100202, quotas=None, max_seconds=1800):
    quotas = dict(QUOTAS if quotas is None else quotas)
    if set(quotas) - set(QUOTAS) or any(type(n) is not int or not 0 <= n <= QUOTAS[game] for game, n in quotas.items()) or sum(quotas.values()) > 10000:
        raise ValueError("invalid bounded quotas")
    if not 0 < max_seconds <= 1800:
        raise ValueError("wall limit must be >0 and <=1800")
    if output_dir.exists() or status_path.exists():
        raise ValueError("refusing to overwrite existing expansion")
    prior, prior_reference = load_prior(prior_dir)
    output_dir.mkdir(parents=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    status = {"version": EXPANSION_VERSION, "state": "running", "phase": "generation", "started_at": now(),
        "seed": seed, "requested_new_originals": sum(quotas.values()), "quotas": quotas,
        "provisional_originals": 0, "provisional_by_game": {game: 0 for game in quotas},
        "attempts_by_game": {game: 0 for game in quotas}, "rejections": {},
        "limits": {"wall_seconds": max_seconds, "cpu_seconds": 1800, "memory_bytes": 2 * 1024**3,
                   "cpu_threads": 1, "max_candidates": 264, "max_attempts_per_game": 30000},
        "gpu_used": False, "api_calls": 0, "active_training_modified": False, "prior": prior_reference,
        "output_dir": str(output_dir), "variant_rows": 0}

    def update(extra=None):
        status.update(extra or {})
        status["elapsed_seconds"] = round(time.perf_counter() - start, 3)
        status["max_rss_mib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
        atomic_json(status_path, status)

    update()
    rows = []
    prior_families = {(row["topic"], row["game_metadata"]["state_family_id"]) for row in prior}
    seen = set(prior_families)
    rejections = Counter()
    generation_nodes = 0
    try:
        for game_index, (game, quota) in enumerate(quotas.items()):
            rng = random.Random(seed + game_index * 100003)
            for attempt in range(30000):
                if status["provisional_by_game"][game] >= quota or time.perf_counter() - start > max_seconds * 0.7:
                    break
                status["attempts_by_game"][game] += 1
                state, teacher, objective, generation = sample(game, rng, attempt)
                if teacher is not None:
                    generation_nodes += teacher["nodes"]
                if state is None or teacher is None or not teacher["optimal_actions"]:
                    rejections[f"{game}:invalid_uninformative_or_unreachable"] += 1
                    continue
                family = (state.state_family_id() if isinstance(state, GridNavigation) else
                          state.family_id() if isinstance(state, LightsOut) else state_id(state))
                key = (game, family)
                if key in seen:
                    rejections[f"{game}:prior_duplicate" if key in prior_families else f"{game}:new_duplicate"] += 1
                    continue
                scenario = f"{EXPANSION_VERSION}:{seed}:{game}:{attempt:05d}"
                generation["master_seed"] = seed
                row = export_row(state, teacher, scenario=scenario, seed=rng.randrange(2**31), objective=objective, generation=generation)
                rows.append(row)
                seen.add(key)
                status["provisional_by_game"][game] += 1
                status["provisional_originals"] = len(rows)
                if len(rows) % 100 == 0:
                    status["rejections"] = dict(rejections)
                    update({"phase": "generation", "current_game": game})
            update({"phase": "generation", "current_game": game, "rejections": dict(rejections)})
        grouping = isolate_and_split(prior, rows, seed, update)
        verify_nodes = 0
        for index, row in enumerate(rows):
            if time.perf_counter() - start > max_seconds:
                raise TimeoutError("verification wall budget exceeded; no unverified training publication")
            verify_nodes += verify_row(row)
            if index % 100 == 0:
                update({"phase": "verification", "verified_originals": index})
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate original IDs")
        if len({(row["topic"], row["game_metadata"]["state_family_id"]) for row in rows}) != len(rows):
            raise ValueError("duplicate canonical state families")
        if {(row["topic"], row["game_metadata"]["state_family_id"]) for row in rows} & prior_families:
            raise ValueError("prior duplicates escaped")
        files = []
        for split in ("train", "development", "quarantine"):
            subset = [row for row in rows if row["split"] == split]
            files.append(write_jsonl(output_dir / f"{split}.jsonl", subset))
        # Re-read every final allowlisted file and prior checksum before claiming
        # verified completion; quarantined rows are never a training allowlist.
        for record in files + prior_reference["files"]:
            if hashlib.sha256(Path(record["path"]).read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError("final or prior checksum verification failed")
        source_files = [Path(__file__), Path(__file__).with_name("__init__.py"), Path(__file__).with_name("grid_navigation.py"), Path(__file__).with_name("lights_out.py")]
        split_counts = Counter(row["split"] for row in rows)
        valid = [row for row in rows if row["split"] != "quarantine"]
        status.update({"state": "complete" if len(rows) == sum(quotas.values()) else "bounded_partial", "phase": "complete",
            "completed_at": now(), "generated_new_originals": len(rows), "accepted_originals": len(valid),
            "quarantined_originals": split_counts["quarantine"], "split_counts": dict(split_counts),
            "accepted_by_game": dict(Counter(row["topic"] for row in valid)),
            "split_by_game": {game: dict(Counter(row["split"] for row in rows if row["topic"] == game)) for game in quotas},
            "rejections": dict(rejections), "files": files, "grouping": grouping,
            "training_allowlist": [record for record in files if record["path"].endswith("/train.jsonl")],
            "development_allowlist": [record for record in files if record["path"].endswith("/development.jsonl")],
            "quarantine_excluded_from_training": True,
            "diversity": {"teacher_methods": dict(Counter(row["game_metadata"]["teacher"]["label_method"] for row in valid)),
                "objectives": dict(Counter(row["game_metadata"]["objective"]["kind"] for row in valid)),
                "difficulty": dict(Counter(row["difficulty"] for row in valid)),
                "tied_optimal_rows": sum(len(row["game_metadata"]["teacher"]["optimal_actions"]) > 1 for row in valid),
                "candidate_counts": dict(Counter(str(len(row["options"]) - 1) for row in valid)),
                "c4_phases": dict(Counter(row["game_metadata"]["generation"]["phase"] for row in valid if row["topic"] == "connect_four")),
                "grid_shortest_distances": dict(Counter(str(row["game_metadata"]["teacher"]["shortest_distance"]) for row in valid if row["topic"] == "grid_navigation")),
                "lights_shortest_distances": dict(Counter(str(row["game_metadata"]["teacher"]["shortest_distance"]) for row in valid if row["topic"] == "lights_out"))},
            "compute": {"generation_solver_nodes": generation_nodes, "verification_solver_nodes": verify_nodes,
                        "cpu_seconds": round(time.process_time(), 3), "api_equivalent_authoring_cost": 0},
            "verification": {"verified_originals": len(rows), "all_states_reconstructed": True,
                "all_exact_objective_targets_recomputed": True, "data_and_prior_sha256_verified": True,
                "zero_prior_canonical_duplicates": True, "zero_split_group_overlap": True,
                "prior_splits_preserved": True},
            "provenance": {"generator_version": EXPANSION_VERSION, "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files},
                           "source": "self-generated simulator states; no external or benchmark content", "target_semantics": "uniform optimal-action teacher policy, not outcome probability"},
            "limitations": ["No stochastic outcome-probability tasks, hidden information or positive-IDK targets",
                "Early/mid C4 decisions use explicitly declared finite-horizon terminal rewards when full-game proof is unavailable; no whole-game optimality claim",
                "Exact grid/LightsOut solutions teach fully observed abstract states rather than perception",
                "Curriculum uses exact shortest distances for puzzles and declared remaining-unit thresholds for games; no measured transfer claim",
                "Near-neighbor grouping can create large components; conflicting prior-linked components are quarantined, never randomly resplit",
                "Legacy train/Luna-only validator unchanged; future training requires this explicit manifest and game validator"]})
        update()
        atomic_json(output_dir / "manifest.json", status)
        return status
    except BaseException as error:
        status.update({"state": "failed", "phase": "failed", "error_type": type(error).__name__, "completed_at": now()})
        update()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, default=Path("data/game_pilot/2026-10-02"))
    parser.add_argument("--seed", type=int, default=2026100202)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (1800, 1800))
    result = generate(args.output_dir, args.status, args.prior_dir, seed=args.seed)
    print(json.dumps({key: result[key] for key in ("state", "generated_new_originals", "accepted_originals", "quarantined_originals", "split_counts", "elapsed_seconds", "max_rss_mib")}, sort_keys=True))


if __name__ == "__main__":
    main()
