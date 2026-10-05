import pytest

from dataset_factory.game_data import Board, Budget, Nim, decision_row, state_id, teach, validate_game_row
from dataset_factory.probability_v2 import validate_v2_row


def replay(game, actions):
    state = Board.initial(game)
    for action in actions:
        state = state.apply((action,))
    return state


def test_tic_tac_toe_win_draw_and_terminal_legality():
    win = replay("tic_tac_toe", [0, 3, 1, 4, 2])
    assert win.outcome() == 1
    assert win.legal_actions() == ()
    with pytest.raises(ValueError, match="illegal"):
        win.apply((8,))
    draw = replay("tic_tac_toe", [0, 1, 2, 4, 3, 5, 7, 6, 8])
    assert draw.outcome() == 0


def test_exact_teacher_proves_immediate_win_and_starting_ties():
    board = replay("tic_tac_toe", [0, 3, 1, 4])
    result = teach(board, budget=Budget(2000, 5))
    assert result.label_method == "exact_minimax"
    assert result.optimal_actions == ((2,),)
    initial = teach(Board.initial(), budget=Budget(20000, 9))
    assert initial.label_method == "exact_minimax"
    assert len(initial.optimal_actions) == 9
    assert set(initial.value_bounds) == {(0, 0)}


def test_bounded_search_cannot_become_idk_or_a_fabricated_policy():
    initial = Board.initial("connect_four")
    result = teach(initial, budget=Budget(1, 1))
    assert result.label_method == "bounded_minimax"
    assert not result.optimal_actions
    assert result.nodes <= 1
    with pytest.raises(ValueError, match="has not proved"):
        decision_row(initial, scenario_id="bounded", budget=Budget(1, 1))
    near_win = replay("connect_four", [0, 1, 0, 1, 0, 2])
    result = teach(near_win, candidates=((0,), (3,)), budget=Budget(1, 1))
    # Winning one candidate does not prove another candidate is not tied.
    assert result.value_bounds[0] == (1, 1)
    assert not result.optimal_actions
    only_win = teach(near_win, candidates=((0,),), budget=Budget(1, 1))
    assert only_win.optimal_actions == ((0,),)


def test_connect_four_gravity_and_winner():
    board = replay("connect_four", [0, 1, 0, 1, 0, 1, 0])
    assert board.cells[5 * 7] == 1
    assert board.cells[4 * 7] == 1
    assert board.outcome() == 1
    with pytest.raises(ValueError, match="floating"):
        Board((1, -1) + (0,) * 40, game="connect_four")


def test_nim_exact_winning_actions_and_losing_ties():
    result = teach(Nim((1, 2, 3)))
    assert len(result.optimal_actions) == 6
    assert set(result.value_bounds) == {(-1, -1)}
    winning = teach(Nim((1, 3, 5)))
    assert winning.optimal_actions == ((2, 3),)
    assert Nim((0,)).outcome() == -1
    assert Nim((0,)).legal_actions() == ()


def test_export_alignment_schema_seed_and_supplied_scope():
    state = Nim((1, 3, 5))
    row = decision_row(state, scenario_id="s", trajectory_id="episode", seed=17)
    assert validate_game_row(row) == []
    assert set(validate_v2_row(row)) == {"split_is_not_train", "provenance_does_not_match_required_label"}
    assert row == decision_row(state, scenario_id="s", trajectory_id="episode", seed=17)
    assert row["target_distribution"][row["choice_ids"].index("action:2,3")] == 1
    assert row["target_distribution"][-1] == 0
    assert row["game_metadata"]["split_group_keys"][-1] == "trajectory:episode"
    reordered = decision_row(state, scenario_id="s", seed=19)
    assert dict(zip(row["choice_ids"], row["target_distribution"])) == dict(zip(reordered["choice_ids"], reordered["target_distribution"]))
    assert row["id"] == reordered["id"]
    subset = decision_row(state, scenario_id="s", candidates=((0, 1), (1, 1)))
    assert subset["id"] != row["id"]
    assert subset["game_metadata"]["candidate_scope"] == "supplied_subset"
    assert subset["target_distribution"] == [0.5, 0.5, 0]
    with pytest.raises(ValueError, match="264"):
        teach(Nim((256, 256)))


def test_family_ids_group_symmetries_and_nim_heap_permutations():
    assert state_id(replay("tic_tac_toe", [0, 4, 1])) == state_id(replay("tic_tac_toe", [2, 4, 5]))
    assert state_id(replay("connect_four", [0, 1])) == state_id(replay("connect_four", [6, 5]))
    assert state_id(Nim((1, 2, 3))) == state_id(Nim((3, 1, 2)))


def test_invalid_states_candidates_and_budgets():
    with pytest.raises(ValueError, match="counts"):
        Board((1,) + (0,) * 8)
    for heaps in ((-1,), (True,), (), (257,)):
        with pytest.raises(ValueError):
            Nim(heaps)
    with pytest.raises(ValueError):
        teach(Nim((1,)), candidates=((0, 2),))
    with pytest.raises(ValueError):
        teach(Nim((1,)), candidates=((0, 1), (0, 1)))
    for budget in ((0, 2), (1, 0), (100001, 2), (True, 2)):
        with pytest.raises(ValueError):
            Budget(*budget)


def test_connected_groups_merge_whole_trajectories_transitively():
    from dataset_factory.game_data import connected_group_ids

    rows = [{"game_metadata": {"split_group_keys": keys}} for keys in (
        ["episode:a", "state:x"], ["episode:a", "state:y"],
        ["episode:b", "state:y"], ["episode:b", "state:z"], ["episode:c", "state:q"])]
    groups = connected_group_ids(rows)
    assert len(set(groups[:4])) == 1
    assert groups[4] != groups[0]
    assert groups == list(reversed(connected_group_ids(list(reversed(rows)))))
    with pytest.raises(ValueError):
        connected_group_ids([{}])


def test_pilot_assigns_close_family_before_variants():
    from dataset_factory.game_data.pilot import assign_splits

    rows = [decision_row(Nim(heaps), scenario_id=f"family-{index}")
            for index, heaps in enumerate(((1, 2), (1, 3), (1, 4), (7, 8, 9)))]
    report = assign_splits(rows, 17)
    assert report["groups"] == 2
    assert len({row["problem_group_id"] for row in rows[:3]}) == 1
    assert len({row["split"] for row in rows[:3]}) == 1
    assert all(validate_game_row(row, allowed_splits=("train", "development")) == [] for row in rows)


def test_finite_pilot_records_checksums_replay_and_repeatability(tmp_path):
    import hashlib
    import json
    from dataset_factory.game_data.pilot import generate

    first = generate(tmp_path / "first", tmp_path / "first-status.json", per_game=3, seed=171,
                     max_attempts_per_game=50, max_seconds=30)
    second = generate(tmp_path / "second", tmp_path / "second-status.json", per_game=3, seed=171,
                      max_attempts_per_game=50, max_seconds=30)
    assert first["state"] == second["state"] == "complete"
    assert first["accepted_originals"] == 9
    assert first["accepted_by_game"] == {game: 3 for game in ("tic_tac_toe", "connect_four", "nim")}
    assert first["cpu_threads"] == 1 and first["api_calls"] == 0 and not first["gpu_used"]
    assert first["variant_rows"] == 0
    rows = []
    for split in ("train", "development"):
        content = (tmp_path / "first" / f"{split}.jsonl").read_bytes()
        assert content == (tmp_path / "second" / f"{split}.jsonl").read_bytes()
        record = next(record for record in first["files"] if record["path"].endswith(f"/{split}.jsonl"))
        assert record["sha256"] == hashlib.sha256(content).hexdigest()
        subset = [json.loads(line) for line in content.splitlines()]
        assert len(subset) == record["rows"]
        assert all(row["split"] == split for row in subset)
        rows.extend(subset)
    assert {row["problem_group_id"] for row in rows if row["split"] == "train"}.isdisjoint(
        {row["problem_group_id"] for row in rows if row["split"] == "development"})
    for row in rows:
        observation = json.loads(row["context"])
        if row["topic"] != "nim":
            state = Board.initial(row["topic"])
            for action in row["game_metadata"]["generation"]["replayed_actions"]:
                state = state.apply(tuple(action))
            assert state.observation() == observation
        else:
            state = Nim(tuple(observation["heaps"]), observation["player"])
        expected = teach(state, budget=Budget(5000, 9))
        optimal_ids = {"action:" + ",".join(map(str, action)) for action in expected.optimal_actions}
        actual_ids = {identifier for identifier, target in zip(row["choice_ids"], row["target_distribution"]) if target > 0}
        assert expected.label_method.startswith("exact")
        assert actual_ids == optimal_ids
    with pytest.raises(ValueError, match="overwrite"):
        generate(tmp_path / "first", tmp_path / "first-status.json", per_game=3)
