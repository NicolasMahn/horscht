from dataclasses import fields
from fractions import Fraction
from itertools import product

import pytest

from dataset_factory.game_data.mastermind import Mastermind


def _independent_feedback(secret, guess):
    exact = sum(left == right for left, right in zip(secret, guess))
    matched = sum(min(secret.count(symbol), guess.count(symbol))
                  for symbol in set(secret + guess))
    return exact, matched - exact


def _brute_posterior(symbols, length, history):
    return tuple(code for code in product(range(symbols), repeat=length)
                 if all(_independent_feedback(code, guess) == (exact, misplaced)
                        for guess, exact, misplaced in history))


def _brute_score(posterior, guess):
    buckets = {}
    for secret in posterior:
        feedback = _independent_feedback(secret, guess)
        buckets[feedback] = buckets.get(feedback, 0) + 1
    return Fraction(sum(count * count for count in buckets.values()), len(posterior))


def test_public_history_is_the_only_state_and_prior_includes_repeats():
    state = Mastermind(symbols=2, length=2)

    assert state.game == "mastermind"
    assert state.rules == "exact-and-misplaced-feedback-repeats-allowed-v1"
    assert state.observation() == {
        "game": "mastermind",
        "rules": state.rules,
        "symbols": 2,
        "length": 2,
        "prior": "uniform_over_all_symbol_sequences_with_replacement_v1",
        "repeats_allowed": True,
        "feedback_convention": "exact_positions_then_multiset_matches_minus_exact",
        "history": [],
    }
    assert len(state.posterior()) == 2 ** 2
    assert (0, 0) in state.posterior() and (1, 1) in state.posterior()
    assert {field.name for field in fields(state)} == {"symbols", "length", "history"}


def test_feedback_counts_duplicate_symbols_by_multiplicity():
    assert Mastermind.feedback((0, 0, 1, 1), (0, 1, 0, 2)) == (1, 2)

    with pytest.raises(ValueError, match="equal-length"):
        Mastermind.feedback((0, 1), (0,))
    with pytest.raises(ValueError, match="nonnegative"):
        Mastermind.feedback((0, -1), (0, 1))


def test_posterior_marginals_and_guess_values_are_exact_rationals():
    state = Mastermind(symbols=2, length=2, history=(((0, 0), 1, 0),))
    assert state.posterior() == ((0, 1), (1, 0))
    assert state.symbol_distribution(0) == {0: Fraction(1, 2), 1: Fraction(1, 2)}
    assert state.symbol_distribution(1) == {0: Fraction(1, 2), 1: Fraction(1, 2)}
    assert state.feedback_distribution((0, 1)) == {
        (0, 2): Fraction(1, 2), (2, 0): Fraction(1, 2),
    }

    candidates = ((0, 0), (0, 1))
    assert state.guess_scores(candidates) == {
        (0, 0): Fraction(2), (0, 1): Fraction(1),
    }
    teaching = state.teach(candidates)
    assert teaching == {
        "actions": candidates,
        "expected_values": (Fraction(-2), Fraction(-1)),
        "optimal_actions": ((0, 1),),
        "label_method": "exact_uniform_posterior_information",
        "nodes": 4,
    }


def test_exact_enumeration_matches_independent_oracle_with_repeated_symbols():
    history = (((0, 1, 1), 2, 0), ((2, 2, 0), 1, 1))
    state = Mastermind(symbols=3, length=3, history=history)
    expected_posterior = _brute_posterior(3, 3, history)
    assert state.posterior() == expected_posterior

    candidates = tuple(product(range(3), repeat=3))
    expected_scores = {guess: _brute_score(expected_posterior, guess)
                       for guess in candidates}
    assert state.guess_scores(candidates) == expected_scores
    assert sum(state.feedback_distribution((0, 1, 1)).values()) == 1

    result = state.teach(candidates)
    best_score = min(expected_scores.values())
    assert result["optimal_actions"] == tuple(
        guess for guess in candidates if expected_scores[guess] == best_score)
    assert result["expected_values"] == tuple(-expected_scores[guess] for guess in candidates)


def test_family_ids_ignore_history_order_and_symbol_renaming():
    history = (((0, 0, 0), 1, 0), ((2, 2, 2), 1, 0))
    state = Mastermind(3, 3, history)
    rename = {0: 2, 1: 0, 2: 1}
    renamed_history = tuple(
        (tuple(rename[symbol] for symbol in guess), exact, misplaced)
        for guess, exact, misplaced in reversed(history)
    )
    renamed = Mastermind(3, 3, renamed_history)
    assert state.family_id() == renamed.family_id()


def test_information_family_groups_refinements_from_same_first_feedback():
    first = ((0, 0, 0), 1, 0)
    left = Mastermind(3, 3, (first, ((2, 2, 2), 1, 0)))
    right = Mastermind(3, 3, (first, ((0, 1, 1), 2, 0)))
    assert left.information_family_id() == right.information_family_id()
    assert left.family_id() != right.family_id()


def test_invalid_parameters_feedback_and_contradictory_histories_are_rejected():
    invalid_states = (
        (True, 2, ()), (1, 2, ()), (2, True, ()), (2, 1, ()),
        (2, 2, []),
        (2, 2, (((0,), 1, 0),)),
        (2, 2, (((2, 0), 1, 0),)),
        (2, 2, (((0, 0), True, 0),)),
        (2, 2, (((0, 0), 2, 1),)),
        (2, 2, (((0, 0), 2, 0),) * 9),
    )
    for arguments in invalid_states:
        with pytest.raises(ValueError):
            Mastermind(*arguments)

    with pytest.raises(ValueError, match="contradictory"):
        Mastermind(2, 2, (((0, 0), 2, 0), ((0, 0), 0, 0)))
    with pytest.raises(ValueError, match="contradictory"):
        Mastermind(2, 2, (((0, 0), 0, 1),))

    state = Mastermind(2, 2)
    for position in (-1, 2, True):
        with pytest.raises(ValueError):
            state.symbol_distribution(position)
    for candidates in ((), ((0,),), ((0, 2),), ((0, 0), (0, 0))):
        with pytest.raises(ValueError):
            state.guess_scores(candidates)
