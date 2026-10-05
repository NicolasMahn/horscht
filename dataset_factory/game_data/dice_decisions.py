"""Exact one-step decisions over small weighted dice laws."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import reduce
from math import gcd
import hashlib
import json


@dataclass(frozen=True)
class DiceDecision:
    """Bank or choose one threshold gamble over independent weighted dice.

    Offers are ``(dice_count, threshold, win_payoff, loss_payoff)``. A roll
    wins when its sum is at least the threshold.
    """

    weights: tuple[int, ...]
    offers: tuple[tuple[int, int, int, int], ...]
    bank: int

    def __post_init__(self) -> None:
        if (not isinstance(self.weights, tuple) or not 1 <= len(self.weights) <= 12
                or any(type(weight) is not int or weight < 0 for weight in self.weights)
                or sum(self.weights) <= 0):
            raise ValueError("weights must be 1..12 nonnegative integer face weights with positive total")
        if (not isinstance(self.offers, tuple) or not 1 <= len(self.offers) <= 12
                or any(not isinstance(offer, tuple) or len(offer) != 4
                       or any(type(value) is not int for value in offer)
                       or not 1 <= offer[0] <= 3
                       for offer in self.offers)):
            raise ValueError("offers must be 1..12 (dice_count, threshold, win_payoff, loss_payoff) tuples")
        if type(self.bank) is not int:
            raise ValueError("bank must be an integer")

    @property
    def game(self) -> str:
        return "dice_decisions"

    @property
    def rules(self) -> str:
        return "one-step-weighted-dice-threshold-gamble-v1"

    def observation(self) -> dict:
        return {
            "game": self.game,
            "rules": self.rules,
            "rule_description": (
                "Choose action 0 to receive the bank immediately. Action i, for 1 <= i <= N, "
                "selects offer i and rolls its dice_count independent dice once. Each die shows "
                "face x with probability weights[x-1] / sum(weights). If the face sum is at least "
                "the offer threshold, receive its win_payoff; otherwise receive its loss_payoff. "
                "There is no reroll or later decision."
            ),
            "weights": list(self.weights),
            "weight_total": sum(self.weights),
            "offers": [list(offer) for offer in self.offers],
            "bank": self.bank,
            "action_encoding": {"0": "bank", "1..N": "offer at the matching one-based index"},
            "visibility": "full_public_state",
        }

    def legal_actions(self) -> tuple[tuple[int, ...], ...]:
        return tuple((index,) for index in range(len(self.offers) + 1))

    def outcome_distribution(self, dice_count: int) -> dict[int, Fraction]:
        """Return exact probabilities for reachable face sums of ``dice_count`` dice."""
        if type(dice_count) is not int or not 1 <= dice_count <= 3:
            raise ValueError("dice_count must be an integer from 1 through 3")

        counts = {0: 1}
        for _ in range(dice_count):
            next_counts: dict[int, int] = {}
            for prior_sum, prior_count in counts.items():
                for face, weight in enumerate(self.weights, start=1):
                    if weight:
                        total = prior_sum + face
                        next_counts[total] = next_counts.get(total, 0) + prior_count * weight
            counts = next_counts

        denominator = sum(self.weights) ** dice_count
        return {total: Fraction(count, denominator)
                for total, count in sorted(counts.items()) if count}

    def apply(self, action: tuple[int, ...], observed_sum: int | None = None) -> int:
        """Return the realized payoff; gamble actions require their observed roll sum."""
        if (not isinstance(action, tuple) or len(action) != 1 or type(action[0]) is not int
                or action not in self.legal_actions()):
            raise ValueError("illegal action")
        if action[0] == 0:
            if observed_sum is not None:
                raise ValueError("bank action does not accept an observed sum")
            return self.bank
        if observed_sum is None:
            raise ValueError("gamble action requires an observed sum")
        if type(observed_sum) is not int:
            raise ValueError("observed sum must be an integer")

        dice_count, threshold, win_payoff, loss_payoff = self.offers[action[0] - 1]
        if observed_sum not in self.outcome_distribution(dice_count):
            raise ValueError("observed sum is unreachable for the selected offer")
        return win_payoff if observed_sum >= threshold else loss_payoff

    def expected_values(self) -> tuple[Fraction, ...]:
        """Return each legal action's exact expected payoff, in action order."""
        distributions: dict[int, dict[int, Fraction]] = {}
        values = [Fraction(self.bank)]
        for dice_count, threshold, win_payoff, loss_payoff in self.offers:
            if dice_count not in distributions:
                distributions[dice_count] = self.outcome_distribution(dice_count)
            distribution = distributions[dice_count]
            values.append(sum((probability * (win_payoff if total >= threshold else loss_payoff)
                               for total, probability in distribution.items()), Fraction()))
        return tuple(values)

    def teach(self) -> dict:
        """Return exact EV labels, including every action tied for the best payoff."""
        actions = self.legal_actions()
        values = self.expected_values()
        best = max(values)
        return {
            "actions": actions,
            "optimal_actions": tuple(action for action, value in zip(actions, values) if value == best),
            "expected_values": tuple({"numerator": value.numerator, "denominator": value.denominator}
                                     for value in values),
            "label_method": "exact_finite_dice_ev",
            "nodes": 0,
        }

    def family_id(self) -> str:
        """Identify a decision family independent of offer order and weight scale."""
        payload = {
            "game": self.game,
            "rules": self.rules,
            "weights": self._normalized_weights(),
            "offers": sorted(self.offers),
            "bank": self.bank,
        }
        return self._digest(payload)

    def law_family_id(self) -> str:
        """Group questions that use the same weighted die, regardless of offers."""
        return self._digest({"game": self.game, "rules": self.rules,
                             "weights": self._normalized_weights()})

    def _normalized_weights(self) -> tuple[int, ...]:
        common_divisor = reduce(gcd, self.weights)
        return tuple(weight // common_divisor for weight in self.weights)

    @staticmethod
    def _digest(payload: dict) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:24]
