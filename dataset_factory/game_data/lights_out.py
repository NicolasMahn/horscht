"""Exact CPU-only adapter for small square Lights Out boards."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json

Action = tuple[int, ...]


@dataclass(frozen=True)
class _GF2Factorization:
    """Row-reduced fixed move matrix, with RHS row operations retained."""

    rows: tuple[tuple[int, int], ...]
    pivots: tuple[int, ...]
    nullspace_basis: tuple[int, ...]

    def solutions(self, right_hand_side: int) -> tuple[int, ...] | None:
        variable_count = len(self.pivots) + len(self.nullspace_basis)
        rank = len(self.pivots)
        reduced_rhs = [((source & right_hand_side).bit_count() & 1)
                       for _coefficients, source in self.rows]
        if any(reduced_rhs[row] for row in range(rank, len(self.rows))):
            return None

        particular = 0
        for row, pivot in enumerate(self.pivots):
            if reduced_rhs[row]:
                particular |= 1 << pivot

        solutions = [particular]
        current = particular
        previous_gray = 0
        for ordinal in range(1, 1 << len(self.nullspace_basis)):
            gray = ordinal ^ (ordinal >> 1)
            changed = gray ^ previous_gray
            basis_index = changed.bit_length() - 1
            current ^= self.nullspace_basis[basis_index]
            solutions.append(current)
            previous_gray = gray
        # Keep this assertion close to the bit layout: every solution must be
        # a press mask over exactly the matrix's variables.
        assert all(solution >> variable_count == 0 for solution in solutions)
        return tuple(solutions)


@lru_cache(maxsize=4)
def _factorization(size: int) -> _GF2Factorization:
    """Factor the board's move matrix once for each supported size."""
    cell_count = size * size
    effects = []
    for index in range(cell_count):
        row, column = divmod(index, size)
        affected = (index,)
        if row:
            affected += (index - size,)
        if row + 1 < size:
            affected += (index + size,)
        if column:
            affected += (index - 1,)
        if column + 1 < size:
            affected += (index + 1,)
        effects.append(sum(1 << cell for cell in affected))

    rows = []
    for cell in range(cell_count):
        coefficients = sum(1 << button for button, effect in enumerate(effects)
                           if effect & (1 << cell))
        rows.append((coefficients, 1 << cell))

    pivots = []
    rank = 0
    for column in range(cell_count):
        pivot_row = next((row for row in range(rank, cell_count)
                          if rows[row][0] & (1 << column)), None)
        if pivot_row is None:
            continue
        rows[rank], rows[pivot_row] = rows[pivot_row], rows[rank]
        pivot_coefficients, pivot_source = rows[rank]
        for row in range(cell_count):
            if row != rank and rows[row][0] & (1 << column):
                coefficients, source = rows[row]
                rows[row] = (coefficients ^ pivot_coefficients, source ^ pivot_source)
        pivots.append(column)
        rank += 1

    pivot_set = set(pivots)
    free_columns = [column for column in range(cell_count) if column not in pivot_set]
    basis = []
    for free_column in free_columns:
        vector = 1 << free_column
        for row, pivot in enumerate(pivots):
            if rows[row][0] & (1 << free_column):
                vector |= 1 << pivot
        basis.append(vector)
    return _GF2Factorization(tuple(rows), tuple(pivots), tuple(basis))


@lru_cache(maxsize=2048)
def _all_press_solutions(size: int, lights: int) -> tuple[int, ...] | None:
    return _factorization(size).solutions(lights)


def _mask(cells: tuple[int, ...]) -> int:
    return sum(value << index for index, value in enumerate(cells))


def _symmetries(cells: tuple[int, ...], size: int) -> tuple[tuple[int, ...], ...]:
    variants = []
    current = cells
    for _ in range(4):
        variants.extend((current, tuple(current[row * size + (size - 1 - column)]
                                        for row in range(size) for column in range(size))))
        current = tuple(current[(size - 1 - column) * size + row]
                        for row in range(size) for column in range(size))
    return tuple(variants)


@dataclass(frozen=True)
class LightsOut:
    """A square board; press masks use row-major cell indices as bit positions."""

    cells: tuple[int, ...]
    size: int = 5

    def __post_init__(self):
        if type(self.size) is not int or self.size not in (2, 3, 4, 5):
            raise ValueError("Lights Out size must be an integer from 2 through 5")
        if (not isinstance(self.cells, tuple) or len(self.cells) != self.size * self.size
                or any(type(value) is not int or value not in (0, 1) for value in self.cells)):
            raise ValueError("cells must be a tuple of 0/1 values matching the board size")

    @property
    def game(self) -> str:
        return "lights_out"

    @property
    def rules(self) -> str:
        return "orthogonal-self-and-neighbors-toggle-v1"

    @property
    def player(self) -> int:
        return 1

    def outcome(self) -> int | None:
        return 1 if not any(self.cells) else None

    def legal_actions(self) -> tuple[Action, ...]:
        if self.outcome() is not None:
            return ()
        return tuple((index,) for index in range(self.size * self.size))

    def apply(self, action: Action) -> LightsOut:
        if (not isinstance(action, tuple) or len(action) != 1
                or type(action[0]) is not int or action not in self.legal_actions()):
            raise ValueError("illegal action")
        index = action[0]
        row, column = divmod(index, self.size)
        toggled = (index,)
        if row:
            toggled += (index - self.size,)
        if row + 1 < self.size:
            toggled += (index + self.size,)
        if column:
            toggled += (index - 1,)
        if column + 1 < self.size:
            toggled += (index + 1,)
        cells = list(self.cells)
        for cell in toggled:
            cells[cell] ^= 1
        return LightsOut(tuple(cells), self.size)

    def observation(self) -> dict:
        return {"game": self.game, "rules": self.rules, "player": self.player,
                "size": self.size, "cells": list(self.cells),
                "visibility": "full_public_state"}

    def family_id(self) -> str:
        canonical = min(_symmetries(self.cells, self.size))
        payload = {"game": self.game, "rules": self.rules, "size": self.size,
                   "cells": canonical}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]

    def solutions(self) -> tuple[int, ...]:
        """Return every shortest press sequence as a row-major bit mask."""
        candidates = _all_press_solutions(self.size, _mask(self.cells))
        if candidates is None:
            raise ValueError("Lights Out state is unsolvable")
        distance = min(mask.bit_count() for mask in candidates)
        return tuple(mask for mask in candidates if mask.bit_count() == distance)

    def teach(self) -> dict:
        """Return exact action values and all first presses in shortest solutions."""
        if self.outcome() is not None:
            raise ValueError("cannot teach a terminal Lights Out state")
        candidates = _all_press_solutions(self.size, _mask(self.cells))
        if candidates is None:
            raise ValueError("Lights Out state is unsolvable")

        shortest_distance = min(mask.bit_count() for mask in candidates)
        shortest = tuple(mask for mask in candidates if mask.bit_count() == shortest_distance)
        actions = self.legal_actions()
        child_distances = tuple(min((solution ^ (1 << action[0])).bit_count()
                                    for solution in candidates)
                                for action in actions)
        optimal = tuple(action for action in actions
                        if any(solution & (1 << action[0]) for solution in shortest))
        return {
            "actions": actions,
            "value_bounds": tuple((-distance, -distance) for distance in child_distances),
            "optimal_actions": optimal,
            "label_method": "exact_lights_out_gf2_minweight",
            "nodes": 0,
            "shortest_distance": shortest_distance,
        }
