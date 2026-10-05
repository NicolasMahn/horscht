"""One finite, explicitly invoked CPU pilot; no discovery or training integration."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from dataset_factory.game_data import (
    VERSION, Board, Budget, Nim, connected_group_ids, decision_row, state_id,
    validate_game_row,
)

PILOT_VERSION = "game-pilot-v1"
GAMES = ("tic_tac_toe", "connect_four", "nim")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _sample(game: str, rng: random.Random):
    """Keep a single decision per independently sampled legal trajectory."""
    if game == "nim":
        return Nim(tuple(rng.randint(1, 9) for _ in range(rng.randint(2, 5)))), []
    state = Board.initial(game)
    length = rng.randint(2, 7) if game == "tic_tac_toe" else rng.randint(33, 38)
    history = []
    for _ in range(length):
        choices = list(state.legal_actions())
        rng.shuffle(choices)
        # Nonterminal replay deliberately samples decision states, not games
        # won earlier. This sampling law biases C4 toward late survivable states.
        for action in choices:
            child = state.apply(action)
            if child.outcome() is None:
                state = child
                history.append(list(action))
                break
        else:
            return None, history
    return state, history


def _near(state_a: dict, state_b: dict) -> bool:
    if state_a["game"] != state_b["game"]:
        return False
    if state_a["game"] == "nim":
        a, b = sorted(state_a["heaps"]), sorted(state_b["heaps"])
        return len(a) == len(b) and sum(abs(x - y) for x, y in zip(a, b)) <= 2
    a, b = state_a["cells"], state_b["cells"]
    width = 3 if state_a["game"] == "tic_tac_toe" else 7
    variants = [b, [b[r * width + (width - c - 1)] for r in range(len(b) // width) for c in range(width)]]
    if width == 3:
        for _ in range(3):
            b = [b[(2 - c) * 3 + r] for r in range(3) for c in range(3)]
            variants.extend((b, [b[r * 3 + (2 - c)] for r in range(3) for c in range(3)]))
    return any(sum(x != y for x, y in zip(a, variant)) <= 2 for variant in variants)


def assign_splits(rows: list[dict], seed: int) -> dict:
    """Split connected families before any optional downstream variants."""
    states = [json.loads(row["context"]) for row in rows]
    near_edges = 0
    for i, state in enumerate(states):
        for j in range(i):
            if _near(state, states[j]):
                edge = f"near:{rows[j]['id']}:{rows[i]['id']}"
                rows[i]["game_metadata"]["split_group_keys"].append(edge)
                rows[j]["game_metadata"]["split_group_keys"].append(edge)
                near_edges += 1
    groups = connected_group_ids(rows)
    per_game = {game: set() for game in GAMES}
    for row, group in zip(rows, groups):
        per_game[row["topic"]].add(group)
    held_out = set()
    for game, game_groups in per_game.items():
        ordered = sorted(game_groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
        # A whole connected game family may become development-only. Splitting
        # a family to force a ratio would leak near-equivalent decisions.
        held_out.update(ordered[:math.ceil(len(ordered) * 0.2)])
    for row, group in zip(rows, groups):
        row["problem_group_id"] = group
        row["split"] = "development" if group in held_out else "train"
        row["game_metadata"]["split_policy"] = "connected_symmetry_and_near_state_families_v1"
    return {"groups": len(set(groups)), "near_position_edges": near_edges,
            "groups_by_game": {game: len(values) for game, values in per_game.items()},
            "development_group_fraction": 0.2,
            "group_overlap_between_splits": 0}


def _write_jsonl(path: Path, rows: list[dict]) -> dict:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(text.encode()).hexdigest()}


def generate(output_dir: Path, status_path: Path, *, per_game: int = 64,
             seed: int = 20261002, max_attempts_per_game: int = 512,
             max_seconds: float = 120) -> dict:
    if type(per_game) is not int or not 1 <= per_game <= 64:
        raise ValueError("per_game must be 1..64 (192 accepted originals maximum)")
    if type(max_attempts_per_game) is not int or not per_game <= max_attempts_per_game <= 1024:
        raise ValueError("attempt cap must cover quota and be <=1024 per game")
    if not 0 < max_seconds <= 300:
        raise ValueError("wall-clock cap must be >0 and <=300 seconds")
    if output_dir.exists() or status_path.exists():
        raise ValueError("pilot output/status already exists; refusing to overwrite")
    output_dir.mkdir(parents=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    status = {"version": PILOT_VERSION, "state": "running", "started_at": _now(),
              "seed": seed, "requested_per_game": per_game, "accepted_originals": 0,
              "accepted_by_game": {game: 0 for game in GAMES},
              "attempts_by_game": {game: 0 for game in GAMES}, "rejections": {},
              "output_dir": str(output_dir), "cpu_threads": 1, "gpu_used": False,
              "external_agents_used": False, "api_calls": 0,
              "limits": {"max_accepted_originals": per_game * 3,
                         "max_attempts_per_game": max_attempts_per_game,
                         "max_seconds": max_seconds, "max_total_search_nodes": 500_000,
                         "solver_budget": {"max_nodes": 5000, "max_depth": 9}},
              "training_modified": False, "generation_scope": "one_explicit_bounded_pilot"}
    _atomic_json(status_path, status)
    rows, seen = [], set()
    rejections = Counter()
    total_nodes = 0
    try:
        for game_index, game in enumerate(GAMES):
            rng = random.Random(seed + game_index * 100003)
            for attempt in range(max_attempts_per_game):
                if status["accepted_by_game"][game] >= per_game:
                    break
                if time.perf_counter() - started >= max_seconds or total_nodes + 5000 > 500_000:
                    rejections["global_compute_cap"] += 1
                    break
                status["attempts_by_game"][game] += 1
                state, history = _sample(game, rng)
                if state is None:
                    rejections[f"{game}:no_nonterminal_replay"] += 1
                    continue
                family = (game, state_id(state))
                if family in seen:
                    rejections[f"{game}:duplicate_state_family"] += 1
                    continue
                scenario = f"{PILOT_VERSION}-{seed}-{game}-{attempt:04d}"
                remaining = sum(state.heaps) if isinstance(state, Nim) else state.cells.count(0)
                # Explicit declared curriculum, independent from correctness.
                band = "easy" if remaining <= 4 else "medium" if remaining <= 7 else "hard"
                try:
                    row = decision_row(state, scenario_id=scenario, trajectory_id=scenario,
                                       seed=rng.randrange(2**31), budget=Budget(5000, 9), difficulty=band)
                except ValueError as error:
                    if "bounded search has not proved" not in str(error):
                        raise
                    # The node cap applies to rejected searches too.
                    total_nodes += 5000
                    rejections[f"{game}:unresolved_teacher"] += 1
                    continue
                total_nodes += row["game_metadata"]["teacher"]["nodes"]
                row["game_metadata"]["generation"] = {
                    "version": PILOT_VERSION, "master_seed": seed, "attempt_index": attempt,
                    "original_problem_id": row["id"], "presentation_variant": False,
                    "sample_method": "uniform_bounded_heaps" if game == "nim" else "random_nonterminal_legal_replay",
                    "replayed_actions": history, "independent_trajectory": True,
                    "curriculum_method": "declared_remaining_units_thresholds_4_7_v1",
                }
                rows.append(row)
                seen.add(family)
                status["accepted_by_game"][game] += 1
                status["accepted_originals"] = len(rows)
                if len(rows) % 8 == 0:
                    status["elapsed_seconds"] = round(time.perf_counter() - started, 3)
                    status["rejections"] = dict(rejections)
                    _atomic_json(status_path, status)
        split_report = assign_splits(rows, seed)
        for row in rows:
            errors = validate_game_row(row, allowed_splits=("train", "development"))
            if errors:
                raise ValueError("generated pilot failed schema validation: " + "; ".join(errors))
        if len({row["id"] for row in rows}) != len(rows):
            raise ValueError("duplicate original IDs")
        splits = {split: [row for row in rows if row["split"] == split] for split in ("train", "development")}
        train_groups = {row["problem_group_id"] for row in splits["train"]}
        dev_groups = {row["problem_group_id"] for row in splits["development"]}
        if train_groups & dev_groups:
            raise ValueError("split groups overlap")
        files = [_write_jsonl(output_dir / f"{split}.jsonl", subset) for split, subset in splits.items()]
        sources = [Path(__file__), Path(__file__).with_name("__init__.py")]
        status.update({
            "state": "complete" if all(n == per_game for n in status["accepted_by_game"].values()) else "bounded_partial",
            "completed_at": _now(), "elapsed_seconds": round(time.perf_counter() - started, 3),
            "search_nodes_charged": total_nodes, "rejections": dict(rejections),
            "split_counts": {split: len(subset) for split, subset in splits.items()},
            "split_by_game": {game: dict(Counter(row["split"] for row in rows if row["topic"] == game)) for game in GAMES},
            "grouping": split_report, "variant_rows": 0, "files": files,
            "diversity": {"label_methods": dict(Counter(row["game_metadata"]["teacher"]["label_method"] for row in rows)),
                "difficulty": dict(Counter(row["difficulty"] for row in rows)),
                "tied_optimal_rows": sum(len(row["game_metadata"]["teacher"]["optimal_actions"]) > 1 for row in rows),
                "candidate_counts": dict(Counter(str(len(row["options"]) - 1) for row in rows)),
                "candidate_scopes": dict(Counter(row["game_metadata"]["candidate_scope"] for row in rows))},
            "provenance": {"adapter_version": VERSION, "generator_version": PILOT_VERSION,
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
                "rules_by_game": {game: sorted({row["game_metadata"]["rules_version"] for row in rows if row["topic"] == game}) for game in GAMES},
                "source_kind": "self-generated legal states; no external or benchmark content",
                "target_semantics": "uniform_optimal_candidate_action_policy; not outcome probabilities",
                "labels": "deterministic exact or conservative bounded minimax / analytic XOR",
                "authoring_api_equivalent_cost": 0},
            "limitations": ["English full-public-state action decisions only; no hidden information or IDK-positive tasks",
                "Connect Four is biased toward late nonterminal endgames; no early-position mastery claim",
                "Curriculum bands are declared remaining-unit thresholds, not measured reasoning difficulty",
                "Near-state grouping is conservative but does not prove semantic nonoverlap",
                "Legacy train/Luna-only validator intentionally unchanged; future training must use explicit game manifest and validator"],
        })
        _atomic_json(output_dir / "manifest.json", status)
        _atomic_json(status_path, status)
        return status
    except Exception as error:
        status.update({"state": "failed", "completed_at": _now(), "error_type": type(error).__name__,
                       "elapsed_seconds": round(time.perf_counter() - started, 3)})
        _atomic_json(status_path, status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--per-game", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261002)
    args = parser.parse_args()
    # This implementation is serial pure Python. Bound libraries too if imported
    # by a future adapter; no torch or engine is imported here.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    result = generate(args.output_dir, args.status, per_game=args.per_game, seed=args.seed)
    print(json.dumps({key: result[key] for key in ("state", "accepted_originals", "accepted_by_game", "split_counts", "elapsed_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
