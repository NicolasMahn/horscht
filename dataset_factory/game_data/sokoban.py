"""Exact shortest-path labels for small, fully observed Sokoban boards.

Walls use the same row-major 0/1 bitmap as :mod:`grid_navigation`. Goals and
boxes are row-major cell indices. Reverse BFS gives exact distances for every
state that can reach a solved board; after the queue is exhausted, absent
states are proved deadlocks.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from types import MappingProxyType
from typing import Mapping


Action = tuple[int, ...]
StateKey = tuple[int, tuple[int, ...]]
DistanceTable = Mapping[StateKey, int]
DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


class SokobanSearchLimitError(RuntimeError):
    """Raised when a distance table cannot be proved within its node budget."""


def _step(cell: int, direction: int, width: int, height: int) -> int | None:
    row, column = divmod(cell, width)
    row_delta, column_delta = DIRECTIONS[direction]
    next_row, next_column = row + row_delta, column + column_delta
    if 0 <= next_row < height and 0 <= next_column < width:
        return next_row * width + next_column
    return None


def _add_state(
    distances: dict[StateKey, int],
    queue: deque[StateKey],
    state: StateKey,
    distance: int,
    max_nodes: int,
) -> None:
    if state in distances:
        return
    if len(distances) >= max_nodes:
        raise SokobanSearchLimitError(
            f"Sokoban reverse BFS exceeded max_nodes={max_nodes}; "
            "the distance table is incomplete, so deadlocks cannot be labeled"
        )
    distances[state] = distance
    queue.append(state)


@lru_cache(maxsize=8)
def _reverse_distance_table(
    walls: tuple[int, ...],
    goals: tuple[int, ...],
    width: int,
    height: int,
    box_count: int,
    max_nodes: int,
) -> tuple[DistanceTable, int]:
    """Build the complete reverse-reachable state table for one topology."""
    floor = tuple(cell for cell, is_wall in enumerate(walls) if not is_wall)
    solved_boxes = goals
    distances: dict[StateKey, int] = {}
    queue: deque[StateKey] = deque()

    # A solved state is terminal for every player location that is not occupied
    # by a box. Terminal states have no outgoing walking or pushing actions.
    for player in floor:
        if player not in solved_boxes:
            _add_state(distances, queue, (player, solved_boxes), 0, max_nodes)

    while queue:
        player, boxes = queue.popleft()
        previous_distance = distances[(player, boxes)] + 1
        box_set = set(boxes)

        # Reverse a walk. A predecessor with all boxes on goals would already
        # be terminal and therefore cannot take a walking action.
        if boxes != solved_boxes:
            for direction in range(4):
                previous_player = _step(player, direction, width, height)
                if (previous_player is not None and not walls[previous_player]
                        and previous_player not in box_set):
                    _add_state(distances, queue, (previous_player, boxes),
                               previous_distance, max_nodes)

        # Reverse a push: the current player stood on the old box square, and
        # the old player stood one square behind it before pushing forward.
        for direction in range(4):
            target = _step(player, direction, width, height)
            previous_player = _step(player, (direction + 2) % 4, width, height)
            if (target is None or previous_player is None or walls[target]
                    or target not in box_set or walls[previous_player]
                    or previous_player in box_set):
                continue
            previous_boxes = tuple(sorted((box_set - {target}) | {player}))
            if previous_boxes == solved_boxes:
                continue
            _add_state(distances, queue, (previous_player, previous_boxes),
                       previous_distance, max_nodes)

    # MappingProxyType prevents a caller from corrupting this shared cache.
    return MappingProxyType(distances), len(distances)


def _transformed_cell(cell: int, width: int, height: int,
                      turns: int, reflected: bool) -> tuple[int, int, int]:
    """Return transformed cell and dimensions under one square symmetry."""
    row, column = divmod(cell, width)
    if turns == 0:
        next_column, next_row, next_width, next_height = column, row, width, height
    elif turns == 1:
        next_column, next_row = height - 1 - row, column
        next_width, next_height = height, width
    elif turns == 2:
        next_column, next_row = width - 1 - column, height - 1 - row
        next_width, next_height = width, height
    else:
        next_column, next_row = row, width - 1 - column
        next_width, next_height = height, width
    if reflected:
        next_column = next_width - 1 - next_column
    return next_row * next_width + next_column, next_width, next_height


@dataclass(frozen=True, slots=True)
class Sokoban:
    """A bounded board with one or two boxes and ordinary push-only rules."""

    walls: tuple[int, ...]
    goals: tuple[int, ...]
    boxes: tuple[int, ...]
    player: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if type(self.width) is not int or not 3 <= self.width <= 7:
            raise ValueError("Sokoban width must be an integer from 3 through 7")
        if type(self.height) is not int or not 3 <= self.height <= 7:
            raise ValueError("Sokoban height must be an integer from 3 through 7")
        cell_count = self.width * self.height
        if (not isinstance(self.walls, tuple) or len(self.walls) != cell_count
                or any(type(value) is not int or value not in (0, 1)
                       for value in self.walls)):
            raise ValueError("walls must be a row-major 0/1 tuple matching the board")
        if (not isinstance(self.goals, tuple) or not 1 <= len(self.goals) <= 2
                or any(type(cell) is not int or not 0 <= cell < cell_count
                       for cell in self.goals)
                or len(set(self.goals)) != len(self.goals)):
            raise ValueError("goals must contain one or two distinct board cells")
        if (not isinstance(self.boxes, tuple) or not 1 <= len(self.boxes) <= 2
                or len(self.boxes) != len(self.goals)
                or any(type(cell) is not int or not 0 <= cell < cell_count
                       for cell in self.boxes)
                or len(set(self.boxes)) != len(self.boxes)):
            raise ValueError("boxes must contain one or two distinct cells, matching goals")
        if type(self.player) is not int or not 0 <= self.player < cell_count:
            raise ValueError("player must be a board cell index")

        goals = tuple(sorted(self.goals))
        boxes = tuple(sorted(self.boxes))
        if any(self.walls[cell] for cell in (*goals, *boxes, self.player)):
            raise ValueError("goals, boxes, and player must be on floor cells")
        if self.player in boxes:
            raise ValueError("player and boxes must occupy different cells")
        object.__setattr__(self, "goals", goals)
        object.__setattr__(self, "boxes", boxes)

    @property
    def game(self) -> str:
        return "sokoban"

    @property
    def rules(self) -> str:
        return "orthogonal-walk-push-no-pull-all-boxes-on-goals-v1"

    def outcome(self) -> int | None:
        return 1 if self.boxes == self.goals else None

    def _neighbor(self, cell: int, direction: int) -> int | None:
        return _step(cell, direction, self.width, self.height)

    def legal_actions(self) -> tuple[Action, ...]:
        if self.outcome() is not None:
            return ()
        occupied = set(self.boxes)
        actions = []
        for direction in range(4):
            adjacent = self._neighbor(self.player, direction)
            if adjacent is None or self.walls[adjacent]:
                continue
            if adjacent not in occupied:
                actions.append((direction,))
                continue
            beyond = self._neighbor(adjacent, direction)
            if (beyond is not None and not self.walls[beyond]
                    and beyond not in occupied):
                actions.append((direction,))
        return tuple(actions)

    def apply(self, action: Action) -> Sokoban:
        if (not isinstance(action, tuple) or len(action) != 1
                or type(action[0]) is not int or action not in self.legal_actions()):
            raise ValueError("illegal Sokoban action")
        direction = action[0]
        adjacent = self._neighbor(self.player, direction)
        assert adjacent is not None
        boxes = self.boxes
        if adjacent in boxes:
            beyond = self._neighbor(adjacent, direction)
            assert beyond is not None
            boxes = tuple(sorted((set(boxes) - {adjacent}) | {beyond}))
        return Sokoban(self.walls, self.goals, boxes, adjacent,
                       self.width, self.height)

    def observation(self) -> dict:
        return {
            "game": self.game,
            "rules": self.rules,
            "width": self.width,
            "height": self.height,
            "walls": list(self.walls),
            "goals": list(self.goals),
            "boxes": list(self.boxes),
            "player": self.player,
            "visibility": "full_public_state",
        }

    def topology_family_id(self) -> str:
        """Group every position with this wall-and-goal layout under D4."""
        wall_indices = tuple(index for index, value in enumerate(self.walls) if value)
        variants = []
        for reflected in (False, True):
            for turns in range(4):
                transformed = [_transformed_cell(cell, self.width, self.height,
                                                 turns, reflected)
                               for cell in (*wall_indices, *self.goals)]
                next_width, next_height = transformed[0][1:3]
                walls = tuple(sorted(value[0] for value in transformed[:len(wall_indices)]))
                goals = tuple(sorted(value[0] for value in transformed[len(wall_indices):]))
                variants.append((next_width, next_height, walls, goals))
        payload = {"game": self.game, "rules": self.rules, "topology": min(variants)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                              .encode()).hexdigest()[:24]

    def state_family_id(self) -> str:
        """Group this complete state with its rotations and reflections."""
        wall_indices = tuple(index for index, value in enumerate(self.walls) if value)
        variants = []
        for reflected in (False, True):
            for turns in range(4):
                transformed = [_transformed_cell(cell, self.width, self.height,
                                                 turns, reflected)
                               for cell in (*wall_indices, *self.goals, *self.boxes,
                                            self.player)]
                next_width, next_height = transformed[0][1:3]
                cursor = 0
                walls = tuple(sorted(value[0] for value in transformed[cursor:cursor + len(wall_indices)]))
                cursor += len(wall_indices)
                goals = tuple(sorted(value[0] for value in transformed[cursor:cursor + len(self.goals)]))
                cursor += len(self.goals)
                boxes = tuple(sorted(value[0] for value in transformed[cursor:cursor + len(self.boxes)]))
                cursor += len(self.boxes)
                player = transformed[cursor][0]
                variants.append((next_width, next_height, walls, goals, boxes, player))
        payload = {"game": self.game, "rules": self.rules, "state": min(variants)}
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"))
                              .encode()).hexdigest()[:24]

    def distances(self, max_nodes: int = 20_000) -> DistanceTable:
        """Return exact distances for solvable states as ``(player, boxes)`` keys.

        A completed table omits precisely the states that cannot reach a solved
        board. If the search budget expires, this method raises instead of
        treating unvisited states as deadlocks.
        """
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        table, _nodes = _reverse_distance_table(
            self.walls, self.goals, self.width, self.height, len(self.boxes), max_nodes
        )
        return table

    def teach(self, max_nodes: int = 20_000) -> dict:
        """Return exact action values and shortest-solution first moves."""
        if type(max_nodes) is not int or max_nodes < 1:
            raise ValueError("max_nodes must be a positive integer")
        if self.outcome() is not None:
            raise ValueError("cannot teach a terminal Sokoban state")
        table, nodes = _reverse_distance_table(
            self.walls, self.goals, self.width, self.height, len(self.boxes), max_nodes
        )
        actions = self.legal_actions()
        child_distances = []
        for action in actions:
            child = self.apply(action)
            child_distances.append(table.get((child.player, child.boxes)))

        shortest_distance = table.get((self.player, self.boxes))
        reachable_children = [distance for distance in child_distances if distance is not None]
        if shortest_distance is None:
            optimal_actions: tuple[Action, ...] = ()
        else:
            if not reachable_children:
                raise AssertionError("a reachable nonterminal state needs a solvable child")
            best_child_distance = min(reachable_children)
            optimal_actions = tuple(
                action for action, distance in zip(actions, child_distances)
                if distance == best_child_distance
            )

        value_bounds = tuple(None if distance is None else (-distance, -distance)
                             for distance in child_distances)
        return {
            "actions": actions,
            "child_distances": tuple(child_distances),
            "value_bounds": value_bounds,
            "optimal_actions": optimal_actions,
            "label_method": "exact_sokoban_bfs",
            "nodes": nodes,
            "shortest_distance": shortest_distance,
        }
