from collections import deque

import pytest

from dataset_factory.game_data.sokoban import Sokoban, SokobanSearchLimitError


_DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def _raw_step(cell, direction, width, height):
    row, column = divmod(cell, width)
    dr, dc = _DIRECTIONS[direction]
    row, column = row + dr, column + dc
    if 0 <= row < height and 0 <= column < width:
        return row * width + column
    return None


def _reference_distance(state):
    """Small forward walk-BFS used as an independent Sokoban oracle."""
    walls = state.walls
    goals = tuple(sorted(state.goals))
    width, height = state.width, state.height
    start = (state.player, tuple(sorted(state.boxes)))
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (player, boxes), distance = queue.popleft()
        if boxes == goals:
            return distance
        occupied = set(boxes)
        for direction in range(4):
            adjacent = _raw_step(player, direction, width, height)
            if adjacent is None or walls[adjacent]:
                continue
            next_boxes = boxes
            if adjacent in occupied:
                beyond = _raw_step(adjacent, direction, width, height)
                if (beyond is None or walls[beyond] or beyond in occupied):
                    continue
                next_boxes = tuple(sorted((occupied - {adjacent}) | {beyond}))
            child = (adjacent, next_boxes)
            if child not in seen:
                seen.add(child)
                queue.append((child, distance + 1))
    return None


def _reference_child_distances(state):
    return tuple(_reference_distance(state.apply(action))
                 for action in state.legal_actions())


def test_observation_walk_push_and_exact_shortest_labels_match_forward_bfs():
    walls = (0,) * 16
    state = Sokoban(walls, (14,), (5,), 4, 4, 4)
    observation = state.observation()
    assert observation == {
        "game": "sokoban",
        "rules": state.rules,
        "width": 4,
        "height": 4,
        "walls": list(walls),
        "goals": [14],
        "boxes": [5],
        "player": 4,
        "visibility": "full_public_state",
    }
    assert state.game == "sokoban"
    assert state.legal_actions() == ((0,), (1,), (2,))
    assert state.apply((1,)).player == 5

    result = state.teach()
    child_distances = _reference_child_distances(state)
    expected_distance = _reference_distance(state)
    assert result["shortest_distance"] == expected_distance
    assert result["child_distances"] == child_distances
    assert result["value_bounds"] == tuple(
        None if distance is None else (-distance, -distance)
        for distance in child_distances
    )
    best_child = min(distance for distance in child_distances if distance is not None)
    assert result["optimal_actions"] == tuple(
        action for action, distance in zip(state.legal_actions(), child_distances)
        if distance == best_child
    )
    assert result["label_method"] == "exact_sokoban_bfs"
    assert result["nodes"] > 0


def test_push_into_internal_corner_is_proved_deadlock_but_root_is_solvable():
    # Cell 12 is boxed in by walls above and to its left. A box pushed upward
    # into it cannot leave, while the box has a route along row 3 to goal 14.
    walls = tuple(1 if cell in {7, 11} else 0 for cell in range(25))
    state = Sokoban(walls, (14,), (17,), 22, 5, 5)
    assert _reference_distance(state) is not None

    result = state.teach()
    push_up = result["actions"].index((0,))
    assert result["child_distances"][push_up] is None
    assert result["value_bounds"][push_up] is None
    assert result["shortest_distance"] == _reference_distance(state)
    assert result["optimal_actions"]


def test_two_box_distances_match_independent_walk_bfs():
    state = Sokoban((0,) * 9, (6, 8), (4, 7), 3, 3, 3)
    result = state.teach()
    child_distances = _reference_child_distances(state)
    assert result["shortest_distance"] == _reference_distance(state) == 12
    assert result["child_distances"] == child_distances
    assert result["value_bounds"] == tuple(
        None if distance is None else (-distance, -distance)
        for distance in child_distances
    )
    assert result["optimal_actions"] == ((2,),)


def test_illegal_wall_box_and_terminal_moves_are_rejected():
    walls = tuple(1 if cell in {0, 5} else 0 for cell in range(9))
    state = Sokoban(walls, (8,), (4,), 3, 3, 3)
    assert state.legal_actions() == ((2,),)
    with pytest.raises(ValueError, match="illegal"):
        state.apply((0,))  # player 3 cannot walk upward through the wall at 0
    with pytest.raises(ValueError, match="illegal"):
        state.apply((1,))  # box at 4 cannot be pushed into wall at 5
    with pytest.raises(ValueError, match="illegal"):
        state.apply((True,))

    solved = Sokoban((0,) * 9, (4,), (4,), 3, 3, 3)
    assert solved.outcome() == 1
    assert solved.legal_actions() == ()
    with pytest.raises(ValueError, match="terminal"):
        solved.teach()
    with pytest.raises(ValueError, match="illegal"):
        solved.apply((1,))


def test_search_budget_never_turns_unvisited_states_into_deadlocks():
    state = Sokoban((0,) * 9, (8,), (4,), 3, 3, 3)
    with pytest.raises(SokobanSearchLimitError, match="incomplete"):
        state.teach(max_nodes=1)

    # A complete default search can then classify all child states exactly.
    result = state.teach()
    assert result["child_distances"] == _reference_child_distances(state)


def test_rectangular_d4_families_and_position_independence():
    width, height = 3, 5
    walls = tuple(1 if cell == 1 else 0 for cell in range(width * height))
    state = Sokoban(walls, (13,), (8,), 7, width, height)

    # Rotate clockwise; board dimensions swap from 3x5 to 5x3.
    def rotate(cell):
        row, column = divmod(cell, width)
        next_column, next_row = height - 1 - row, column
        return next_row * height + next_column

    rotated_walls = [0] * (width * height)
    for cell, is_wall in enumerate(walls):
        rotated_walls[rotate(cell)] = is_wall
    rotated = Sokoban(tuple(rotated_walls), (rotate(13),), (rotate(8),),
                      rotate(7), height, width)
    assert state.topology_family_id() == rotated.topology_family_id()
    assert state.state_family_id() == rotated.state_family_id()

    other_position = Sokoban(walls, (13,), (8,), 6, width, height)
    assert state.topology_family_id() == other_position.topology_family_id()
    assert state.state_family_id() != other_position.state_family_id()


def test_validation_rejects_malformed_or_overlarge_boards():
    with pytest.raises(ValueError, match="width"):
        Sokoban((0,) * 4, (3,), (2,), 0, 2, 2)
    with pytest.raises(ValueError, match="row-major"):
        Sokoban((0,) * 9 + (2,), (8,), (4,), 3, 3, 3)
    with pytest.raises(ValueError, match="matching goals"):
        Sokoban((0,) * 9, (8,), (4, 5), 3, 3, 3)
    with pytest.raises(ValueError, match="floor"):
        Sokoban((1,) + (0,) * 8, (8,), (4,), 0, 3, 3)
