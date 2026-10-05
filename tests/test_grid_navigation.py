from collections import deque
import pytest

from dataset_factory.game_data.grid_navigation import GridNavigation


def test_routes_ties_legality_and_goal():
    grid = GridNavigation((0,) * 9, 4, 0, 3)
    teacher = grid.teach()
    assert teacher['shortest_distance'] == 2
    assert teacher['optimal_actions'] == ((0,), (3,))
    assert grid.apply((0,)).position == 1
    goal = grid.apply((0,)).apply((3,))
    assert goal.outcome() == 1 and goal.legal_actions() == ()
    with pytest.raises(ValueError):
        goal.teach()
    with pytest.raises(ValueError):
        goal.apply((1,))


def test_walls_force_detour_and_unreachable_is_not_fake_idk():
    grid = GridNavigation((0, 1, 0, 0, 1, 0, 0, 0, 0), 0, 2, 3)
    assert grid.teach()['shortest_distance'] == 6
    assert grid.teach()['optimal_actions'] == ((2,),)
    blocked = GridNavigation((0, 1, 0, 1, 1, 1, 0, 1, 0), 0, 2, 3)
    with pytest.raises(ValueError, match='unreachable'):
        blocked.teach()


def test_exhaustive_small_grid_distances_match_forward_search():
    for mask in range(512):
        walls = tuple((mask >> i) & 1 for i in range(9))
        free = [i for i, value in enumerate(walls) if not value]
        if len(free) < 2:
            continue
        grid = GridNavigation(walls, free[0], free[-1], 3)
        queue = deque([(grid.position, 0)])
        seen = {grid.position}
        expected = None
        while queue:
            cell, depth = queue.popleft()
            if cell == grid.goal:
                expected = depth
                break
            row, col = divmod(cell, 3)
            for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
                nr, nc = row + dr, col + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    nxt = nr * 3 + nc
                    if not walls[nxt] and nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, depth + 1))
        if expected is None:
            with pytest.raises(ValueError):
                grid.teach()
        else:
            assert grid.teach()['shortest_distance'] == expected


def test_topology_groups_positions_and_goal_choices():
    walls = (0, 1, 0, 0, 0, 0, 0, 0, 0)
    assert GridNavigation(walls, 0, 2, 3).family_id() == GridNavigation(walls, 6, 8, 3).family_id()
    with pytest.raises(ValueError):
        GridNavigation(walls, 1, 2, 3)
