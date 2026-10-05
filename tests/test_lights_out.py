from collections import deque
import random

import pytest

from dataset_factory.game_data.lights_out import LightsOut


def _press(mask, index, size):
    row, column = divmod(index, size)
    cells = (index,)
    if row:
        cells += (index - size,)
    if row + 1 < size:
        cells += (index + size,)
    if column:
        cells += (index - 1,)
    if column + 1 < size:
        cells += (index + 1,)
    for cell in cells:
        mask ^= 1 << cell
    return mask


def _state(mask, size):
    cells = tuple((mask >> index) & 1 for index in range(size * size))
    return LightsOut(cells, size)


def _apply_presses(lights, presses, size):
    for index in range(size * size):
        if presses & (1 << index):
            lights = _press(lights, index, size)
    return lights


def test_public_state_and_press_rules():
    state = LightsOut((1, 0, 0, 0), size=2)
    assert state.game == "lights_out"
    assert state.player == 1
    assert state.rules == "orthogonal-self-and-neighbors-toggle-v1"
    assert state.observation() == {
        "game": "lights_out", "rules": state.rules, "player": 1, "size": 2,
        "cells": [1, 0, 0, 0], "visibility": "full_public_state",
    }
    assert state.legal_actions() == ((0,), (1,), (2,), (3,))
    assert state.apply((0,)).cells == (0, 1, 1, 0)
    assert state.apply((3,)).cells == (1, 1, 1, 1)
    assert LightsOut((0, 0, 0, 0), size=2).outcome() == 1
    assert LightsOut((0, 0, 0, 0), size=2).legal_actions() == ()
    with pytest.raises(ValueError, match="terminal"):
        LightsOut((0, 0, 0, 0), size=2).teach()
    for action in ((True,), (4,), [0], (), (0, 1)):
        with pytest.raises(ValueError, match="illegal"):
            state.apply(action)


def test_validation_and_d4_family_id():
    for cells, size in (((0, 0, 0), 2), ((False, 0, 0, 0), 2), ((0, 0, 0, 0), True)):
        with pytest.raises(ValueError):
            LightsOut(cells, size)
    with pytest.raises(ValueError, match="size"):
        LightsOut((0,) * 36, size=6)

    cells = (1, 0, 1, 0, 1, 1, 0, 0, 1)
    rotated = tuple(cells[(2 - column) * 3 + row]
                    for row in range(3) for column in range(3))
    reflected = tuple(cells[row * 3 + (2 - column)]
                       for row in range(3) for column in range(3))
    family = LightsOut(cells, size=3).family_id()
    assert LightsOut(rotated, size=3).family_id() == family
    assert LightsOut(reflected, size=3).family_id() == family
    assert LightsOut((0,) * 16, size=4).family_id() != family


def test_exhaustive_3x3_distances_values_and_optimal_actions_match_bfs():
    # Presses are reversible, so walking outward from all-off gives each
    # reachable board's exact distance back to the goal.
    distances = {0: 0}
    queue = deque([0])
    while queue:
        mask = queue.popleft()
        for index in range(9):
            child = _press(mask, index, 3)
            if child not in distances:
                distances[child] = distances[mask] + 1
                queue.append(child)

    for mask in range(1 << 9):
        state = _state(mask, 3)
        if mask == 0:
            with pytest.raises(ValueError, match="terminal"):
                state.teach()
            assert state.solutions() == (0,)
            continue
        if mask not in distances:
            with pytest.raises(ValueError, match="unsolvable"):
                state.teach()
            with pytest.raises(ValueError, match="unsolvable"):
                state.solutions()
            continue

        result = state.teach()
        distance = distances[mask]
        child_distances = tuple(distances[_press(mask, action[0], 3)]
                                for action in state.legal_actions())
        expected_optimal = tuple((index,) for index, child_distance in enumerate(child_distances)
                                 if child_distance == distance - 1)
        assert result["shortest_distance"] == distance
        assert result["actions"] == state.legal_actions()
        assert result["value_bounds"] == tuple((-child, -child) for child in child_distances)
        assert result["optimal_actions"] == expected_optimal
        solutions = state.solutions()
        assert all(solution.bit_count() == distance for solution in solutions)
        assert all(_apply_presses(mask, solution, 3) == 0 for solution in solutions)


def test_random_reachable_5x5_states_have_exact_shortest_teachers():
    rng = random.Random(519)
    checked = 0
    while checked < 20:
        presses = rng.randrange(1, 1 << 25)
        lights = 0
        for index in range(25):
            if presses & (1 << index):
                lights = _press(lights, index, 5)
        if lights == 0:
            continue

        state = _state(lights, 5)
        result = state.teach()
        shortest = state.solutions()
        distance = shortest[0].bit_count()
        assert result["label_method"] == "exact_lights_out_gf2_minweight"
        assert result["nodes"] == 0
        assert result["shortest_distance"] == distance
        assert all(solution.bit_count() == distance for solution in shortest)
        assert all(_apply_presses(lights, solution, 5) == 0 for solution in shortest)

        child_distances = tuple(child.solutions()[0].bit_count()
                                for child in (state.apply(action) for action in state.legal_actions()))
        assert result["value_bounds"] == tuple((-child, -child) for child in child_distances)
        assert result["optimal_actions"] == tuple(
            (index,) for index in range(25)
            if any(solution & (1 << index) for solution in shortest))
        checked += 1
