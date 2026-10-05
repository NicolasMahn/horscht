"""Exact inference for Mastermind with public guess feedback only."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
import hashlib
from itertools import permutations, product
import json
from typing import TypeAlias

Code: TypeAlias = tuple[int, ...]
Feedback: TypeAlias = tuple[int, int]
HistoryItem: TypeAlias = tuple[Code, int, int]

RULES = "exact-and-misplaced-feedback-repeats-allowed-v1"
PRIOR = "uniform_over_all_symbol_sequences_with_replacement_v1"
FEEDBACK_CONVENTION = "exact_positions_then_multiset_matches_minus_exact"


def _feedback_unchecked(secret: Code, guess: Code) -> Feedback:
    """Return exact and misplaced matches, accounting for repeated symbols."""
    exact = sum(secret_symbol == guess_symbol
                for secret_symbol, guess_symbol in zip(secret, guess))
    total_matches = sum(min(secret.count(symbol), guess.count(symbol))
                        for symbol in set(secret))
    return exact, total_matches - exact


@lru_cache(maxsize=8192)
def _posterior(symbols: int, length: int, history: tuple[HistoryItem, ...]) -> tuple[Code, ...]:
    possible = []
    for secret in product(range(symbols), repeat=length):
        if all(_feedback_unchecked(secret, guess) == (exact, misplaced)
               for guess, exact, misplaced in history):
            possible.append(secret)
    return tuple(possible)


def _validate_code(code: object, symbols: int, length: int, name: str) -> Code:
    if (not isinstance(code, tuple) or len(code) != length
            or any(type(symbol) is not int or not 0 <= symbol < symbols for symbol in code)):
        raise ValueError(f"{name} must be a tuple of {length} symbols in 0..{symbols - 1}")
    return code


@dataclass(frozen=True)
class Mastermind:
    """A public Mastermind observation under a uniform code prior.

    Symbols are the integers ``0..symbols-1``. Repeated symbols are allowed,
    and every possible code starts equally likely.
    """

    symbols: int
    length: int
    history: tuple[HistoryItem, ...] = ()

    def __post_init__(self) -> None:
        if type(self.symbols) is not int or not 2 <= self.symbols <= 6:
            raise ValueError("symbols must be an integer from 2 through 6")
        if type(self.length) is not int or not 2 <= self.length <= 4:
            raise ValueError("length must be an integer from 2 through 4")
        if self.symbols ** self.length > 1296:
            raise ValueError("code space exceeds 1296 codes")
        if not isinstance(self.history, tuple):
            raise ValueError("history must be an immutable tuple")
        if len(self.history) > 8:
            raise ValueError("history must contain at most 8 observations")

        normalized: list[HistoryItem] = []
        for item in self.history:
            if not isinstance(item, tuple) or len(item) != 3:
                raise ValueError("history items must be (guess, exact, misplaced) tuples")
            guess = _validate_code(item[0], self.symbols, self.length, "history guess")
            exact, misplaced = item[1], item[2]
            if (type(exact) is not int or type(misplaced) is not int
                    or exact < 0 or misplaced < 0 or exact + misplaced > self.length):
                raise ValueError("feedback counts must be nonnegative and fit the code length")
            normalized.append((guess, exact, misplaced))

        if tuple(normalized) != self.history:
            raise ValueError("history must contain immutable tuple values")
        if not _posterior(self.symbols, self.length, self.history):
            raise ValueError("history is contradictory: no code matches all feedback")

    @property
    def game(self) -> str:
        return "mastermind"

    @property
    def rules(self) -> str:
        return RULES

    def observation(self) -> dict:
        """Expose public code-space settings and feedback history, never a code."""
        return {
            "game": self.game,
            "rules": self.rules,
            "symbols": self.symbols,
            "length": self.length,
            "prior": PRIOR,
            "repeats_allowed": True,
            "feedback_convention": FEEDBACK_CONVENTION,
            "history": [[list(guess), exact, misplaced]
                        for guess, exact, misplaced in self.history],
        }

    @staticmethod
    def feedback(secret: Code, guess: Code) -> Feedback:
        """Count black and white pegs, including multiplicities."""
        if (not isinstance(secret, tuple) or not isinstance(guess, tuple)
                or not secret or len(secret) != len(guess)
                or any(type(symbol) is not int or symbol < 0 for symbol in secret + guess)):
            raise ValueError("secret and guess must be equal-length tuples of nonnegative symbols")
        return _feedback_unchecked(secret, guess)

    def posterior(self) -> tuple[Code, ...]:
        """Return every code consistent with the public history."""
        # __post_init__ guarantees this tuple is nonempty.
        return _posterior(self.symbols, self.length, self.history)

    def symbol_distribution(self, position: int) -> dict[int, Fraction]:
        if type(position) is not int or not 0 <= position < self.length:
            raise ValueError("position must be an integer within the code length")
        codes = self.posterior()
        return {symbol: Fraction(sum(code[position] == symbol for code in codes), len(codes))
                for symbol in range(self.symbols)}

    def feedback_distribution(self, guess: Code) -> dict[Feedback, Fraction]:
        guess = _validate_code(guess, self.symbols, self.length, "guess")
        codes = self.posterior()
        counts: dict[Feedback, int] = {}
        for secret in codes:
            result = _feedback_unchecked(secret, guess)
            counts[result] = counts.get(result, 0) + 1
        return {result: Fraction(count, len(codes)) for result, count in sorted(counts.items())}

    def guess_scores(self, candidates: tuple[Code, ...]) -> dict[Code, Fraction]:
        """Map each proposed guess to its expected remaining posterior size."""
        if not isinstance(candidates, tuple) or not candidates:
            raise ValueError("candidates must be a nonempty tuple of guess codes")
        guesses = tuple(_validate_code(code, self.symbols, self.length, "candidate")
                        for code in candidates)
        if len(set(guesses)) != len(guesses):
            raise ValueError("candidate guesses must be distinct")

        codes = self.posterior()
        denominator = len(codes)
        scores = {}
        for guess in guesses:
            buckets: dict[Feedback, int] = {}
            for secret in codes:
                result = _feedback_unchecked(secret, guess)
                buckets[result] = buckets.get(result, 0) + 1
            scores[guess] = Fraction(sum(count * count for count in buckets.values()), denominator)
        return scores

    def teach(self, candidates: tuple[Code, ...]) -> dict:
        """Rank supplied guesses by exact expected posterior size."""
        scores = self.guess_scores(candidates)
        actions = tuple(scores)
        values = tuple(-scores[action] for action in actions)
        best = max(values)
        return {
            "actions": actions,
            "expected_values": values,
            "optimal_actions": tuple(action for action, value in zip(actions, values)
                                      if value == best),
            "label_method": "exact_uniform_posterior_information",
            # Count each candidate-secret feedback comparison used for the exact values.
            "nodes": len(self.posterior()) * len(actions),
        }

    def family_id(self) -> str:
        """Identify equivalent histories independent of order and symbol names."""
        canonical_history = min(
            tuple(sorted((tuple(mapping[symbol] for symbol in guess), exact, misplaced)
                         for guess, exact, misplaced in self.history))
            for mapping in permutations(range(self.symbols))
        )
        payload = {"game": self.game, "rules": self.rules, "symbols": self.symbols,
                   "length": self.length, "history": canonical_history}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]

    def information_family_id(self) -> str:
        """Group refinements that share their first public observation."""
        if not self.history:
            first_observation = None
        else:
            guess, exact, misplaced = self.history[0]
            first_observation = min(
                (tuple(mapping[symbol] for symbol in guess), exact, misplaced)
                for mapping in permutations(range(self.symbols))
            )
        payload = {"game": self.game, "rules": self.rules, "symbols": self.symbols,
                   "length": self.length, "first_observation": first_observation}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]
