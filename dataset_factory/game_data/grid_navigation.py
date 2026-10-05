"""Exact shortest-path decisions in fully observed static square grids."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json

from dataset_factory.game_data.lights_out import _symmetries

DIRECTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))


@dataclass(frozen=True)
class GridNavigation:
    walls: tuple[int, ...]
    position: int
    goal: int
    size: int = 6

    def __post_init__(self):
        if type(self.size) is not int or not 3 <= self.size <= 9:
            raise ValueError("grid size must be 3..9")
        if (not isinstance(self.walls, tuple) or len(self.walls) != self.size * self.size
                or any(type(v) is not int or v not in (0, 1) for v in self.walls)):
            raise ValueError("invalid wall tuple")
        for index in (self.position, self.goal):
            if type(index) is not int or not 0 <= index < len(self.walls) or self.walls[index]:
                raise ValueError("position and goal must be free cells")

    @property
    def game(self):
        return "grid_navigation"

    @property
    def rules(self):
        return "static-square-grid-unit-orthogonal-moves-v1"

    @property
    def player(self):
        return 1

    def outcome(self):
        return 1 if self.position == self.goal else None

    def _neighbors(self, position):
        row, col = divmod(position, self.size)
        for direction, (dr, dc) in enumerate(DIRECTIONS):
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                cell = nr * self.size + nc
                if not self.walls[cell]:
                    yield (direction,), cell

    def legal_actions(self):
        return () if self.outcome() is not None else tuple(action for action, _ in self._neighbors(self.position))

    def apply(self, action):
        if action not in self.legal_actions():
            raise ValueError("illegal action")
        position = next(cell for candidate, cell in self._neighbors(self.position) if candidate == action)
        return GridNavigation(self.walls, position, self.goal, self.size)

    def observation(self):
        return {"game": self.game, "rules": self.rules, "player": 1, "size": self.size,
                "walls": list(self.walls), "position": self.position, "goal": self.goal,
                "visibility": "full_public_state"}

    def family_id(self):
        # A topology is an episode family: all goals/positions and D4 variants
        # of that topology must share a split, not just identical decisions.
        cells = min(_symmetries(self.walls, self.size))
        return hashlib.sha256(json.dumps({"size": self.size, "walls": cells}, sort_keys=True).encode()).hexdigest()[:24]

    def state_family_id(self):
        marked = tuple(1 if v else 3 if i == self.goal else 2 if i == self.position else 0
                       for i, v in enumerate(self.walls))
        return hashlib.sha256(json.dumps({"size": self.size, "cells": min(_symmetries(marked, self.size))}, sort_keys=True).encode()).hexdigest()[:24]

    def distances(self):
        distances = {self.goal: 0}
        queue = deque([self.goal])
        while queue:
            position = queue.popleft()
            for _, child in self._neighbors(position):
                if child not in distances:
                    distances[child] = distances[position] + 1
                    queue.append(child)
        return distances

    def teach(self):
        if self.outcome() is not None:
            raise ValueError("cannot teach a terminal grid state")
        distances = self.distances()
        if self.position not in distances:
            raise ValueError("grid goal is unreachable")
        children = tuple(self._neighbors(self.position))
        values = tuple(-distances[cell] for _, cell in children)
        best = max(values)
        return {"actions": tuple(action for action, _ in children),
                "value_bounds": tuple((value, value) for value in values),
                "optimal_actions": tuple(action for (action, _), value in zip(children, values) if value == best),
                "label_method": "exact_grid_bfs_shortest_path", "nodes": len(distances),
                "shortest_distance": distances[self.position]}
