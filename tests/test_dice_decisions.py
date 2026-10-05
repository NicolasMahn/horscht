from fractions import Fraction
from itertools import product
from math import prod

import pytest

from dataset_factory.game_data.dice_decisions import DiceDecision


def enumerated_distribution(weights, dice_count):
    """Small independent reference implementation by enumerating every roll."""
    total_weight = sum(weights)
    mass_by_sum = {}
    for faces in product(range(1, len(weights) + 1), repeat=dice_count):
        mass = prod(weights[face - 1] for face in faces)
        if mass:
            total = sum(faces)
            mass_by_sum[total] = mass_by_sum.get(total, 0) + mass
    return {total: Fraction(mass, total_weight**dice_count)
            for total, mass in sorted(mass_by_sum.items())}


@pytest.mark.parametrize("weights", [(1,), (1, 2, 0, 3), (0, 0, 5, 0, 2)])
@pytest.mark.parametrize("dice_count", [1, 2, 3])
def test_convolution_matches_independent_enumeration(weights, dice_count):
    game = DiceDecision(weights, ((dice_count, 2, 7, -3),), bank=1)

    actual = game.outcome_distribution(dice_count)

    assert actual == enumerated_distribution(weights, dice_count)
    assert sum(actual.values(), Fraction()) == 1
    assert all(probability > 0 for probability in actual.values())


def test_expected_values_and_teacher_preserve_a_real_exact_tie():
    game = DiceDecision(
        weights=(1, 1, 1),
        offers=((1, 3, 6, 0), (1, 2, 3, 0)),
        bank=2,
    )

    assert game.legal_actions() == ((0,), (1,), (2,))
    assert game.expected_values() == (Fraction(2), Fraction(2), Fraction(2))
    assert game.teach() == {
        "actions": ((0,), (1,), (2,)),
        "optimal_actions": ((0,), (1,), (2,)),
        "expected_values": tuple({"numerator": 2, "denominator": 1} for _ in range(3)),
        "label_method": "exact_finite_dice_ev",
        "nodes": 0,
    }


def test_apply_requires_a_reachable_observation_for_stochastic_gambles():
    game = DiceDecision(weights=(1, 0, 1), offers=((2, 5, 10, -4),), bank=1)

    assert game.apply((0,)) == 1
    assert game.apply((1,), observed_sum=6) == 10
    assert game.apply((1,), observed_sum=4) == -4
    with pytest.raises(ValueError, match="requires an observed sum"):
        game.apply((1,))
    with pytest.raises(ValueError, match="unreachable"):
        game.apply((1,), observed_sum=3)
    with pytest.raises(ValueError, match="does not accept"):
        game.apply((0,), observed_sum=2)
    with pytest.raises(ValueError, match="illegal action"):
        game.apply((True,))


@pytest.mark.parametrize("weights", [
    [1, 1],
    (),
    (0, 0),
    (1, -1),
    (True, 1),
    (1,) * 13,
])
def test_rejects_invalid_dice_laws(weights):
    with pytest.raises(ValueError, match="weights"):
        DiceDecision(weights, ((1, 1, 1, 0),), bank=0)


@pytest.mark.parametrize("offers", [
    [],
    (),
    ((0, 1, 1, 0),),
    ((4, 1, 1, 0),),
    ((True, 1, 1, 0),),
    ((1, False, 1, 0),),
    ([1, 1, 1, 0],),
    ((1, 1, 1),),
    ((1, 1, 1, 0),) * 13,
])
def test_rejects_invalid_offers(offers):
    with pytest.raises(ValueError, match="offers"):
        DiceDecision((1, 1), offers, bank=0)


@pytest.mark.parametrize("bank", [True, 1.0])
def test_rejects_non_integer_bank(bank):
    with pytest.raises(ValueError, match="bank"):
        DiceDecision((1,), ((1, 1, 1, 0),), bank=bank)


@pytest.mark.parametrize("dice_count", [True, 0, 4, 1.0])
def test_outcome_distribution_rejects_invalid_dice_count(dice_count):
    game = DiceDecision((1,), ((1, 1, 1, 0),), bank=0)
    with pytest.raises(ValueError, match="dice_count"):
        game.outcome_distribution(dice_count)


def test_family_ids_canonicalize_offer_order_and_common_weight_scale():
    first = DiceDecision((1, 2, 3), ((1, 2, 8, -1), (3, 7, 20, 0)), bank=4)
    reordered_and_scaled = DiceDecision((2, 4, 6), ((3, 7, 20, 0), (1, 2, 8, -1)), bank=4)
    other_questions_same_law = DiceDecision((5, 10, 15), ((2, 4, 9, -3),), bank=0)

    assert first.family_id() == reordered_and_scaled.family_id()
    assert first.law_family_id() == reordered_and_scaled.law_family_id()
    assert first.family_id() != other_questions_same_law.family_id()
    assert first.law_family_id() == other_questions_same_law.law_family_id()
    assert first.law_family_id() != DiceDecision((3, 2, 1), ((1, 1, 1, 0),), bank=0).law_family_id()


def test_observation_discloses_the_complete_one_step_problem():
    game = DiceDecision((2, 1), ((2, 3, 5, -1),), bank=4)

    observation = game.observation()

    assert observation["game"] == "dice_decisions"
    assert observation["rules"] == game.rules
    assert observation["weights"] == [2, 1]
    assert observation["offers"] == [[2, 3, 5, -1]]
    assert observation["bank"] == 4
    assert "once" in observation["rule_description"]
    assert "There is no reroll" in observation["rule_description"]
    assert observation["visibility"] == "full_public_state"
