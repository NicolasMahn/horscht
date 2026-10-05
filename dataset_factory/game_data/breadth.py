"""Bounded exact dice, permitted-history inference and irreversible-puzzle data."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
from itertools import permutations, product
import json
from math import gcd
import os
from pathlib import Path
import random
import resource
import time

from dataset_factory.game_data import VERSION, validate_game_row
from dataset_factory.game_data.dice_decisions import DiceDecision
from dataset_factory.game_data.mastermind import Mastermind
from dataset_factory.game_data.sokoban import Sokoban, SokobanSearchLimitError
from dataset_factory.game_data.expansion import UnionFind, _neighbor_keys, transformed_masks, bits

BREADTH_VERSION = "game-breadth-v1"
FAMILIES = ("dice_decisions", "mastermind", "sokoban")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def exact_json(value):
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, dict):
        return {str(k): exact_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [exact_json(v) for v in value]
    return value


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(exact_json(value), indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def mastermind_key(state, query=None):
    """Canonicalize public history, symbol names and positional symmetries.

    Histories are bounded to three observations in this sampler. First-use
    symbol naming over every history/position ordering avoids a factorial
    enumeration over six unused symbol names.
    """
    variants = []
    query = query or {}
    effective_history = tuple(dict.fromkeys(state.history))
    for order in permutations(range(len(effective_history))):
        for positions in permutations(range(state.length)):
            mapping = {}

            def rename(code):
                result = []
                for old_position in positions:
                    symbol = code[old_position]
                    if symbol not in mapping:
                        mapping[symbol] = len(mapping)
                    result.append(mapping[symbol])
                return tuple(result)

            history = tuple((rename(effective_history[i][0]), effective_history[i][1], effective_history[i][2]) for i in order)
            position = positions.index(query['position']) if 'position' in query else None
            has_codes = 'guess' in query or bool(query.get('candidates'))
            unused = [symbol for symbol in range(state.symbols) if symbol not in mapping]
            unused_orders = permutations(unused) if has_codes else [tuple(unused)]
            for unused_order in unused_orders:
                complete = dict(mapping)
                complete.update({symbol: len(mapping) + i for i, symbol in enumerate(unused_order)})
                def query_code(code):
                    return tuple(complete[code[old_position]] for old_position in positions)
                guesses = tuple(sorted(query_code(tuple(code)) for code in query.get('candidates', [])))
                guess = query_code(tuple(query['guess'])) if 'guess' in query else ()
                variants.append((history, -1 if position is None else position, guesses, guess))
    return digest({'symbols': state.symbols, 'length': state.length, 'canonical': min(variants)})


def make_row(*, family, rules, observation, query, labels, probabilities, semantics,
             method, state_family, group_keys, seed, difficulty, teacher=None, answerability='answerable', reason=None):
    if not labels or len(labels) > 264 or len(set(labels)) != len(labels):
        raise ValueError('require 1..264 unique meaningful supplied choices plus IDK')
    if len(probabilities) != len(labels) or sum(probabilities, Fraction()) != 1:
        raise ValueError('unaligned or nonnormalized exact probabilities')
    identity = digest({'family': family, 'state_family': state_family, 'query': query['identity']})
    choices = list(zip(labels, probabilities))
    random.Random(seed).shuffle(choices)
    identifiers = ['choice:' + digest(label) for label, _ in choices] + ['task:idk']
    fractions = [value for _, value in choices] + [Fraction()]
    if answerability == 'unanswerable':
        fractions = [Fraction()] * len(choices) + [Fraction(1)]
    source = 'explicit_stochastic_process' if semantics == 'dice_sum_outcome_distribution' else 'exact_derived'
    row = {
        'id': f'{BREADTH_VERSION}:{identity}', 'problem_group_id': f'pending:{state_family}', 'split': 'development',
        'operation_id': query['kind'], 'topic': family, 'language': 'en', 'difficulty': difficulty,
        'context': json.dumps(observation, sort_keys=True), 'question': query['question'],
        'options': [label for label, _ in choices] + ['Cannot determine from the supplied information'],
        'choice_ids': identifiers, 'schema_version': 'probability-v2', 'supervision_kind': 'soft_distribution',
        'target_distribution': [float(value) for value in fractions], 'distribution_source_kind': source,
        'distribution_evidence': {'kind': source, 'choice_ids': identifiers, 'probabilities': exact_json(fractions)},
        'answerability_status': answerability, 'unanswerable_reason': reason, 'abstention_choice_id': 'task:idk',
        'explanation': ('The requested realized fact is not established by the permitted observation; select task IDK.' if answerability == 'unanswerable' else
                        'Exact finite-law outcome probabilities.' if semantics.endswith('distribution') else
                        'Uniform policy over all equally best supplied actions under the stated objective.' if 'policy' in semantics else
                        'Exact logical or deterministic classification from permitted observations.'),
        'provenance': f'Local deterministic {VERSION}; {rules}; {BREADTH_VERSION}; self-generated, no benchmark sources.',
        'game_metadata': {'version': BREADTH_VERSION, 'rules_version': rules, 'state_family_id': state_family,
            'split_group_keys': list(group_keys), 'seed': seed, 'query': query,
            'observation_model': 'permitted_public_history' if family == 'mastermind' else 'public_state_with_declared_omission' if answerability == 'unanswerable' else 'full_public_state',
            'target_semantics': semantics, 'teacher': exact_json({'label_method': method, **(teacher or {})}),
            'presentation_variant': False, 'original_problem_id': f'{BREADTH_VERSION}:{identity}',
            'source_license': {'third_party_content': False, 'license_identifier': None,
                'note': 'Self-generated synthetic states and original project code; no third-party source material or external teacher binary.'}},
    }
    errors = validate_game_row(row)
    if errors:
        raise ValueError('invalid breadth row: ' + '; '.join(errors))
    return row


def dice_sample(rng, index):
    weights = tuple(rng.randint(1, 7) for _ in range(rng.randint(2, 12)))
    common = 0
    for weight in weights:
        common = gcd(common, weight)
    weights = tuple(w // common for w in weights)
    count = rng.randint(1, 3)
    law = DiceDecision(weights, ((count, count + 1, 1, 0),), 0)
    kind = index % 5
    distribution = law.outcome_distribution(count)
    if kind in (0, 1):
        observation = {'game': law.game, 'rules': law.rules, 'weights': list(weights), 'faces': '1 through len(weights)',
                       'dice_count': count, 'independence': True, 'sum_observed': False,
                       'probability_law': 'Each face x has probability weights[x-1]/sum(weights).'}
        labels = [f'Sum {total}' for total in distribution]
        query = {'kind': 'dice_sum_distribution', 'dice_count': count, 'identity': {'kind': 'dice_sum_distribution', 'dice_count': count},
                 'question': 'What is the probability distribution of the sum of the stated independent dice? The sum has not been observed.'}
        return make_row(family=law.game, rules=law.rules, observation=observation, query=query, labels=labels,
            probabilities=list(distribution.values()), semantics='dice_sum_outcome_distribution', method='exact_weighted_convolution',
            state_family=law.law_family_id(), group_keys=['dice-law:' + law.law_family_id()], seed=rng.randrange(2**31),
            difficulty='easy' if count == 1 else 'medium' if count == 2 else 'hard')
    if kind == 4:
        observation = {'game': law.game, 'rules': law.rules, 'weights': list(weights), 'dice_count': count,
            'independence': True, 'sum_observed': False,
            'probability_law': 'Each face x has probability weights[x-1]/sum(weights).', 'event': 'A roll was realized, but no observation of its sum is supplied.'}
        labels = [f'Sum {total}' for total in distribution]
        query = {'kind': 'dice_realized_sum_certainty', 'dice_count': count, 'identity': {'kind': 'dice_realized_sum_certainty', 'dice_count': count},
                 'question': 'Which sum is established as the actual realized roll by the supplied evidence? This asks for the realized fact, not a forecast distribution.'}
        return make_row(family=law.game, rules=law.rules, observation=observation, query=query, labels=labels,
            probabilities=list(distribution.values()), semantics='unobserved_realized_fact_idk', method='exact_non_degenerate_support_check',
            state_family=law.law_family_id(), group_keys=['dice-law:' + law.law_family_id()], seed=rng.randrange(2**31), difficulty='medium',
            answerability='unanswerable', reason='realized_random_outcome_unknown', teacher={'support_size': len(distribution)})
    bank = rng.randint(0, 12)
    offers = []
    force_ties = index % 20 == 2
    for _ in range(rng.randint(1, 12)):
        n = rng.randint(1, 3)
        threshold = rng.randint(n + 1, n * len(weights))
        if force_ties:
            p = sum((prob for total, prob in law.outcome_distribution(n).items() if total >= threshold), Fraction())
            scale = rng.randint(1, 3)
            offer = (n, threshold, bank + (p.denominator - p.numerator) * scale, bank - p.numerator * scale)
        else:
            offer = (n, threshold, rng.randint(1, 35), -rng.randint(0, 20))
        if offer not in offers:
            offers.append(offer)
    state = DiceDecision(weights, tuple(offers), bank)
    teacher = state.teach()
    optimal = set(teacher['optimal_actions'])
    labels = ['Bank'] + [f'Offer {i}: {offer}' for i, offer in enumerate(offers, 1)]
    probabilities = [Fraction(1, len(optimal)) if action in optimal else Fraction() for action in state.legal_actions()]
    query = {'kind': 'dice_best_expected_payoff', 'identity': {'kind': 'dice_best_expected_payoff'},
             'question': 'Which supplied action maximizes the exact expected one-step payoff? Risk-neutral objective; action 0 banks, actions 1..N choose the stated offers.',
             'candidate_scope': 'all_legal'}
    return make_row(family=state.game, rules=state.rules, observation=state.observation(), query=query, labels=labels,
        probabilities=probabilities, semantics='uniform_optimal_expected_value_action_policy', method=teacher['label_method'],
        state_family=state.family_id(), group_keys=['dice-law:' + state.law_family_id()], seed=rng.randrange(2**31),
        difficulty='medium' if len(offers) <= 4 else 'hard', teacher=teacher)


def mastermind_sample(rng, index):
    symbols, length = rng.randint(2, 6), rng.randint(2, 4)
    # A secret is used only to produce a consistent public transcript; it is
    # discarded. All subsequent labels enumerate the public-history posterior.
    secret = tuple(rng.randrange(symbols) for _ in range(length))
    history = []
    for _ in range(rng.randint(1, 3)):
        guess = tuple(rng.randrange(symbols) for _ in range(length))
        exact, misplaced = Mastermind.feedback(secret, guess)
        history.append((guess, exact, misplaced))
    state = Mastermind(symbols, length, tuple(dict.fromkeys(history)))
    posterior = state.posterior()
    kind = index % 5
    query = {'kind': ''}
    teacher = {'posterior_count': len(posterior), 'nodes': len(posterior)}
    position = rng.randrange(length)
    if kind in (0, 1):
        probabilities = list(state.symbol_distribution(position).values())
        labels = [f'Symbol {symbol}' for symbol in range(symbols)]
        query = {'kind': 'mastermind_symbol_posterior', 'position': position,
                 'identity': {'kind': 'mastermind_symbol_posterior', 'joint_canonical': mastermind_key(state, {'position': position})},
                 'question': f'What is the posterior probability distribution of the symbol at zero-based position {position}, using only the public feedback and uniform code prior?'}
        semantics, method = 'exact_hidden_symbol_posterior_distribution', 'exact_uniform_prior_conditioning'
    elif kind == 2:
        if len(posterior) < 2:
            return None
        guess = tuple(rng.randrange(symbols) for _ in range(length))
        distribution = state.feedback_distribution(guess)
        if len(distribution) < 2:
            return None
        labels = [f'Feedback exact={black}, misplaced={white}' for black, white in distribution]
        probabilities = list(distribution.values())
        query = {'kind': 'mastermind_feedback_prediction', 'guess': list(guess),
                 'identity': {'kind': 'mastermind_feedback_prediction', 'joint_canonical': mastermind_key(state, {'guess': guess})},
                 'question': f'What is the predictive probability distribution of feedback for guess {guess}, conditioned only on the public history and uniform code prior?'}
        semantics, method = 'exact_posterior_feedback_distribution', 'exact_uniform_prior_feedback_partition'
    elif kind == 3:
        if not 2 <= len(posterior) <= 200:
            return None
        pool = list(product(range(symbols), repeat=length))
        candidate_count = min(len(pool), rng.choice((2, 4, 8, 16, 32, 64, 128, 264)))
        candidates = tuple(rng.sample(pool, candidate_count))
        teacher = state.teach(candidates)
        teacher['posterior_count'] = len(posterior)
        best = set(teacher['optimal_actions'])
        labels = [f'Guess {guess}' for guess in candidates]
        probabilities = [Fraction(1, len(best)) if guess in best else Fraction() for guess in candidates]
        query = {'kind': 'mastermind_best_information_guess', 'candidates': [list(code) for code in candidates],
                 'identity': {'kind': 'mastermind_best_information_guess', 'joint_canonical': mastermind_key(state, {'candidates': candidates})},
                 'candidate_scope': 'all_legal' if len(candidates) == len(pool) else 'supplied_subset', 'legal_guess_count': len(pool),
                 'question': 'Which supplied guess minimizes the exact expected number of remaining possible codes after feedback? Evaluate best among supplied candidates under the uniform posterior, not all guesses unless all are supplied.'}
        semantics, method = 'uniform_optimal_information_action_policy', teacher['label_method']
    else:
        distribution = state.symbol_distribution(position)
        support = [symbol for symbol, p in distribution.items() if p]
        labels = [f'Symbol {symbol}' for symbol in range(symbols)]
        probabilities = list(distribution.values())
        query = {'kind': 'mastermind_established_symbol', 'position': position,
                 'identity': {'kind': 'mastermind_established_symbol', 'joint_canonical': mastermind_key(state, {'position': position})},
                 'question': f'Which symbol at zero-based position {position} is logically established with certainty by the public feedback? This asks for an established fact, not its posterior forecast.'}
        semantics, method = 'logical_certainty_or_idk', 'exact_posterior_support_certainty_check'
        if len(support) > 1:
            return make_row(family=state.game, rules=state.rules, observation=state.observation(), query=query,
                labels=labels, probabilities=probabilities, semantics=semantics, method=method,
                state_family=mastermind_key(state), group_keys=mastermind_group_keys(state), seed=rng.randrange(2**31),
                difficulty='medium', teacher=teacher, answerability='unanswerable', reason='underdetermined')
    return make_row(family=state.game, rules=state.rules, observation=state.observation(), query=query,
        labels=labels, probabilities=probabilities, semantics=semantics, method=method,
        state_family=mastermind_key(state), group_keys=mastermind_group_keys(state), seed=rng.randrange(2**31),
        difficulty='easy' if len(state.history) == 1 else 'medium' if len(state.history) == 2 else 'hard', teacher=teacher)


def mastermind_group_keys(state):
    # Any shared public observation joins refinements; color and position
    # renamings are conservatively represented by guess multiplicities.
    keys = ['mastermind-state:' + mastermind_key(state)]
    for guess, exact, misplaced in state.history:
        pattern = sorted(Counter(guess).values())
        keys.append('mastermind-observation:' + digest((state.symbols, state.length, pattern, exact, misplaced)))
    return keys


def sokoban_sample(rng, index):
    size = rng.randint(5, 7)
    density = rng.choice((0.05, 0.15, 0.25))
    walls = tuple(int(r == 0 or c == 0 or r == size - 1 or c == size - 1 or rng.random() < density)
                  for r in range(size) for c in range(size))
    floor = [i for i, wall in enumerate(walls) if not wall]
    box_count = 2 if index % 4 == 0 else 1
    if len(floor) < box_count * 2 + 3:
        return None
    goals = tuple(rng.sample(floor, box_count))
    boxes = tuple(rng.sample(floor, box_count))
    if set(goals) == set(boxes):
        return None
    player = rng.choice([cell for cell in floor if cell not in boxes])
    state = Sokoban(walls, goals, boxes, player, size, size)
    if len(state.legal_actions()) < 2:
        return None
    teacher = state.teach(max_nodes=20000)
    if teacher['shortest_distance'] is None or teacher['shortest_distance'] < 2:
        return None
    kind = index % 10
    query = {'kind': ''}
    observation = state.observation()
    if kind == 9:
        observation.pop('player')
        observation['visibility'] = 'player_location_omitted; no history or prior over player locations supplied'
        labels = [f'Player cell {cell}' for cell in floor if cell not in state.boxes]
        probabilities = [Fraction(1, len(labels))] * len(labels)
        query = {'kind': 'sokoban_missing_player_fact', 'identity': {'kind': 'sokoban_missing_player_fact'},
                 'question': 'Which current player cell is established by this observation? No player location, observation history or probability law over player locations is supplied.'}
        omitted_variants = []
        from dataset_factory.game_data.expansion import square_maps
        for mapping in square_maps(size):
            mapped_walls = tuple(sorted(mapping[i] for i, wall in enumerate(walls) if wall))
            omitted_variants.append((mapped_walls, tuple(sorted(mapping[i] for i in state.goals)), tuple(sorted(mapping[i] for i in state.boxes))))
        omitted_id = digest({'size': size, 'configuration': min(omitted_variants)})
        return make_row(family=state.game, rules=state.rules, observation=observation, query=query,
            labels=labels, probabilities=probabilities, semantics='missing_player_location_idk', method='exact_multiple_legal_player_states_check',
            state_family=omitted_id, group_keys=['sokoban-topology:' + state.topology_family_id()], seed=rng.randrange(2**31),
            difficulty='medium', teacher={'possible_player_count': len(labels)}, answerability='unanswerable', reason='missing_context')
    if kind in (7, 8):
        deadlocking = [action for action, distance in zip(teacher['actions'], teacher['child_distances']) if distance is None]
        action = rng.choice(deadlocking if deadlocking and kind == 7 else list(state.legal_actions()))
        child = state.apply(action)
        distance = teacher['child_distances'][teacher['actions'].index(action)]
        classification = 'Solved' if child.outcome() is not None else 'Proved unsolvable after this move' if distance is None else 'Still solvable'
        labels = ['Solved', 'Proved unsolvable after this move', 'Still solvable']
        probabilities = [Fraction(int(label == classification)) for label in labels]
        query = {'kind': 'sokoban_next_move_solvability', 'action': list(action),
                 'identity': {'kind': 'sokoban_next_move_solvability', 'action': list(action)},
                 'question': f'After legal direction {action[0]} (0=up,1=right,2=down,3=left), what is the exact solvability status? Pushes cannot be pulled back; a deadlock label requires the complete finite-state proof.'}
        semantics, method = 'deterministic_irreversibility_classification', 'exact_complete_reverse_reachability'
    else:
        best = set(teacher['optimal_actions'])
        labels = [f'Direction {action[0]} (0=up,1=right,2=down,3=left)' for action in state.legal_actions()]
        probabilities = [Fraction(1, len(best)) if action in best else Fraction() for action in state.legal_actions()]
        query = {'kind': 'sokoban_shortest_solution_first_move', 'identity': {'kind': 'sokoban_shortest_solution_first_move'},
                 'candidate_scope': 'all_legal', 'question': 'Which legal direction starts a shortest solution, counting every walk and push as one move? Directions 0=up,1=right,2=down,3=left; pushes cannot be pulled back.'}
        semantics, method = 'uniform_optimal_shortest_solution_action_policy', 'exact_sokoban_bfs'
    return make_row(family=state.game, rules=state.rules, observation=observation, query=query,
        labels=labels, probabilities=probabilities, semantics=semantics, method=method,
        state_family=state.state_family_id(), group_keys=['sokoban-topology:' + state.topology_family_id()], seed=rng.randrange(2**31),
        difficulty='easy' if teacher['shortest_distance'] <= 4 else 'medium' if teacher['shortest_distance'] <= 10 else 'hard', teacher=teacher)


def recompute(row):
    observation = json.loads(row['context'])
    query = row['game_metadata']['query']
    family = row['topic']
    idk = False
    if family == 'dice_decisions':
        if query['kind'] == 'dice_best_expected_payoff':
            state = DiceDecision(tuple(observation['weights']), tuple(tuple(o) for o in observation['offers']), observation['bank'])
            values = state.expected_values(); best = max(values)
            targets = {'Bank' if index == 0 else f'Offer {index}: {state.offers[index-1]}': Fraction(1, values.count(best)) if value == best else Fraction()
                       for index, value in enumerate(values)}
        else:
            state = DiceDecision(tuple(observation['weights']), ((query['dice_count'], 0, 0, 0),), 0)
            dist = state.outcome_distribution(query['dice_count'])
            targets = {f'Sum {total}': p for total, p in dist.items()}
            idk = query['kind'] == 'dice_realized_sum_certainty' and len(dist) > 1
    elif family == 'mastermind':
        state = Mastermind(observation['symbols'], observation['length'], tuple((tuple(g), b, w) for g, b, w in observation['history']))
        if query['kind'] in ('mastermind_symbol_posterior', 'mastermind_established_symbol'):
            dist = state.symbol_distribution(query['position'])
            targets = {f'Symbol {symbol}': p for symbol, p in dist.items()}
            idk = query['kind'] == 'mastermind_established_symbol' and sum(bool(p) for p in dist.values()) > 1
        elif query['kind'] == 'mastermind_feedback_prediction':
            targets = {f'Feedback exact={b}, misplaced={w}': p for (b, w), p in state.feedback_distribution(tuple(query['guess'])).items()}
        else:
            candidates = tuple(tuple(c) for c in query['candidates'])
            teacher = state.teach(candidates); best = set(teacher['optimal_actions'])
            targets = {f'Guess {guess}': Fraction(1, len(best)) if guess in best else Fraction() for guess in candidates}
    else:
        walls, goals, boxes = tuple(observation['walls']), tuple(observation['goals']), tuple(observation['boxes'])
        if query['kind'] == 'sokoban_missing_player_fact':
            allowed = [cell for cell, wall in enumerate(walls) if not wall and cell not in boxes]
            targets = {f'Player cell {cell}': Fraction(1, len(allowed)) for cell in allowed}
            idk = len(allowed) > 1
        else:
            state = Sokoban(walls, goals, boxes, observation['player'], observation['width'], observation['height'])
            teacher = state.teach(20000)
            if query['kind'] == 'sokoban_next_move_solvability':
                action = tuple(query['action']); child = state.apply(action)
                distance = teacher['child_distances'][teacher['actions'].index(action)]
                label = 'Solved' if child.outcome() is not None else 'Proved unsolvable after this move' if distance is None else 'Still solvable'
                targets = {candidate: Fraction(int(candidate == label)) for candidate in ('Solved','Proved unsolvable after this move','Still solvable')}
            else:
                best = set(teacher['optimal_actions'])
                targets = {f'Direction {action[0]} (0=up,1=right,2=down,3=left)': Fraction(1, len(best)) if action in best else Fraction() for action in state.legal_actions()}
    errors = validate_game_row(row, allowed_splits=('train', 'development', 'quarantine'))
    if errors:
        raise ValueError('schema verification failed: ' + '; '.join(errors))
    if (row['answerability_status'] == 'unanswerable') != idk:
        raise ValueError('answerability differs from permitted evidence')
    expected_labels = set(targets) | {'Cannot determine from the supplied information'}
    if set(row['options']) != expected_labels:
        raise ValueError('supplied choice set changed')
    for label, target in zip(row['options'], row['target_distribution']):
        expected = Fraction(int(label == 'Cannot determine from the supplied information')) if idk else targets.get(label, Fraction())
        if target != float(expected):
            raise ValueError('saved target differs from exact reconstruction')
    if family == 'mastermind' and ('secret' in observation or 'hidden_code' in observation):
        raise ValueError('hidden oracle in input')


def prior_rows(directories):
    rows, records = [], []
    for directory in directories:
        manifest_path = directory / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        records.append({'path': str(manifest_path), 'sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()})
        for record in manifest['files']:
            path = Path(record['path']); data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != record['sha256']:
                raise ValueError('prior data checksum mismatch')
            subset = [json.loads(line) for line in data.splitlines()]
            if len(subset) != record['rows']:
                raise ValueError('prior count mismatch')
            rows.extend(subset); records.append({'path': str(path), 'sha256': record['sha256']})
    return rows, records


def split_rows(prior, rows, seed):
    combined = prior + rows; union = UnionFind(len(combined)); seen = {}
    for index, row in enumerate(combined):
        keys = (['prior-group:' + row['problem_group_id']] if index < len(prior) else row['game_metadata']['split_group_keys'])
        for key in keys:
            if key in seen: union.union(index, seen[key])
            else: seen[key] = index
    dice_neighbors = {}
    for index, row in enumerate(combined):
        if row['topic'] != 'dice_decisions':
            continue
        weights = tuple(json.loads(row['context'])['weights'])
        candidates = [weights]
        for position, value in enumerate(weights):
            for delta in (-1, 1):
                if value + delta >= 0:
                    neighbor = list(weights)
                    neighbor[position] += delta
                    candidates.append(tuple(neighbor))
        for candidate in candidates:
            divisor = 0
            for value in candidate:
                divisor = gcd(divisor, value)
            if divisor == 0:
                continue
            key = tuple(value // divisor for value in candidate)
            if key in dice_neighbors:
                union.union(index, dice_neighbors[key])
            else:
                dice_neighbors[key] = index
    # A shared grid topology connects the new pushing task to prior navigation
    # tasks too. Near terrain edits and D4 variants remain one family.
    worlds = {}
    for index, row in enumerate(combined):
        if row['topic'] not in ('grid_navigation', 'sokoban'):
            continue
        observation = json.loads(row['context'])
        size = observation.get('size', observation.get('width'))
        if row['topic'] == 'sokoban' and size != observation['height']:
            raise ValueError('sampler requires square Sokoban cross-version worlds')
        world = {'game': 'grid_navigation', 'size': size, 'walls': observation['walls']}
        for key in _neighbor_keys(world):
            if key in worlds: union.union(index, worlds[key])
            else: worlds[key] = index
    prior_splits, old_ids, minima = {}, {}, {}
    for index, row in enumerate(combined):
        root = union.find(index); value = row['topic'] + ':' + row['game_metadata']['state_family_id']
        minima[root] = min(value, minima.get(root, value))
        if index < len(prior):
            prior_splits.setdefault(root, set()).add(row['split']); old_ids.setdefault(root, set()).add(row['problem_group_id'])
    new_components = {family: set() for family in FAMILIES}
    for index, row in enumerate(rows, len(prior)):
        root = union.find(index)
        if root not in prior_splits: new_components[row['topic']].add(root)
    heldout = set()
    for roots in new_components.values():
        ordered = sorted(roots, key=lambda root: digest((seed, minima[root])))
        count = max(1, (len(ordered) + 4) // 5) if len(ordered) > 1 else 0
        heldout.update(ordered[:count])
    inherited = conflicts = 0
    for index, row in enumerate(rows, len(prior)):
        root = union.find(index); linked = prior_splits.get(root, set())
        if len(linked) > 1 or 'quarantine' in linked:
            row['split'] = 'quarantine'; conflicts += 1
        elif linked:
            row['split'] = next(iter(linked)); inherited += 1
        else:
            row['split'] = 'development' if root in heldout else 'train'
        old = sorted(old_ids.get(root, set()))
        row['problem_group_id'] = old[0] if old and row['split'] != 'quarantine' else 'game-breadth-group:' + digest(minima[root])
        row['game_metadata']['prior_group_ids'] = old
        row['game_metadata']['split_policy'] = 'transitive_permitted_observation_and_world_families_v1'
    train = {r['problem_group_id'] for r in rows if r['split'] == 'train'}
    development = {r['problem_group_id'] for r in rows if r['split'] == 'development'}
    if train & development: raise ValueError('train/development group overlap')
    return {'new_components_by_family': {f: len(v) for f, v in new_components.items()}, 'prior_split_inherited_rows': inherited,
            'prior_conflict_quarantined_rows': conflicts, 'eligible_group_overlap': 0,
            'prior_assignments_preserved': True, 'world_policy': 'same/near wall topology and D4, across navigation and Sokoban',
            'information_policy': 'dice-law groups with near weight laws; Mastermind shared feedback pattern/refinement groups and full state symmetries'}


def generate(output_dir, status_path, *, per_family=3000, seed=20261003, max_seconds=1800,
             prior_dirs=(Path('data/game_pilot/2026-10-02'), Path('data/game_expansion/2026-10-02'))):
    if type(per_family) is not int or not 1 <= per_family <= 3000 or not 0 < max_seconds <= 1800:
        raise ValueError('quota/time exceeds authorized bounds')
    if output_dir.exists(): raise ValueError('refusing to overwrite breadth output')
    if status_path.exists() and json.loads(status_path.read_text()).get('state') != 'implementing':
        raise ValueError('refusing to overwrite started run status')
    prior, references = prior_rows(prior_dirs)
    output_dir.mkdir(parents=True); status_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); cpu_started = time.process_time()
    status = {'version': BREADTH_VERSION, 'state': 'running', 'phase': 'generation', 'started_at': now(),
        'seed': seed, 'quota_per_family': per_family, 'requested_max_originals': per_family * 3,
        'provisional_originals': 0, 'provisional_by_family': {f: 0 for f in FAMILIES}, 'attempts_by_family': {f: 0 for f in FAMILIES},
        'limits': {'cpu_threads': 1, 'memory_bytes': 2 * 1024**3, 'runtime_seconds': max_seconds, 'max_attempts_per_family': 60000},
        'gpu_used': False, 'active_training_modified': False, 'data_authoring_api_calls': 0, 'prior': references}
    def update(extra=None):
        status.update(extra or {}); status['elapsed_seconds'] = round(time.perf_counter() - started, 3)
        status['max_rss_mib'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 2)
        atomic_json(status_path, status)
    update(); rows=[]; seen=set(); rejections=Counter()
    prior_ids = {row['id'] for row in prior}
    try:
        for family_index, (family, sampler) in enumerate(zip(FAMILIES, (dice_sample, mastermind_sample, sokoban_sample))):
            rng = random.Random(seed + family_index * 100003)
            for attempt in range(60000):
                if status['provisional_by_family'][family] >= per_family or time.perf_counter()-started > max_seconds * 0.7: break
                status['attempts_by_family'][family] += 1
                try: row = sampler(rng, attempt)
                except SokobanSearchLimitError:
                    rejections[family + ':unresolved_budget'] += 1; continue
                if row is None:
                    rejections[family + ':trivial_unreachable_or_uninformative'] += 1; continue
                if row['id'] in seen or row['id'] in prior_ids:
                    rejections[family + ':canonical_question_duplicate'] += 1; continue
                seen.add(row['id']); rows.append(row)
                status['provisional_by_family'][family] += 1; status['provisional_originals'] = len(rows)
                if len(rows)%100 == 0: update({'current_family':family,'rejections':dict(rejections)})
            update({'current_family':family,'rejections':dict(rejections)})
        update({'phase':'split_grouping'}); grouping = split_rows(prior, rows, seed)
        for index,row in enumerate(rows):
            if time.perf_counter()-started > max_seconds: raise TimeoutError('no publication before complete verification')
            recompute(row)
            if index%100 == 0: update({'phase':'replay_and_exact_relabel','verified_originals':index})
        files=[]
        for split in ('train','development','quarantine'):
            subset = [r for r in rows if r['split']==split]; path=output_dir/(split+'.jsonl'); checksum=hashlib.sha256()
            with path.open('wb') as stream:
                for row in subset:
                    line=(json.dumps(row,sort_keys=True)+'\n').encode(); stream.write(line); checksum.update(line)
            files.append({'path':str(path),'rows':len(subset),'bytes':path.stat().st_size,'sha256':checksum.hexdigest()})
        for record in files+references:
            if hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()!=record['sha256']: raise ValueError('final/prior checksum mismatch')
        accepted=[r for r in rows if r['split']!='quarantine']
        source_files=[Path(__file__),Path(__file__).with_name('dice_decisions.py'),Path(__file__).with_name('mastermind.py'),Path(__file__).with_name('sokoban.py'),Path(__file__).with_name('expansion.py'),Path(__file__).with_name('__init__.py')]
        status.update({'state':'complete' if len(rows)==per_family*3 else 'bounded_partial','phase':'complete','completed_at':now(),
            'generated_new_originals':len(rows),'accepted_originals':len(accepted),'quarantined_originals':len(rows)-len(accepted),
            'accepted_by_family':dict(Counter(r['topic'] for r in accepted)), 'split_counts':dict(Counter(r['split'] for r in rows)),
            'split_by_family':{f:dict(Counter(r['split'] for r in rows if r['topic']==f)) for f in FAMILIES},
            'files':files,'grouping':grouping,'rejections':dict(rejections),
            'training_allowlist':[f for f in files if f['path'].endswith('/train.jsonl')],
            'development_allowlist':[f for f in files if f['path'].endswith('/development.jsonl')],
            'quarantine_excluded_from_training':True,'balanced_sampling':{'unit':'family','training_family_weights':{f:Fraction(1,3) for f in FAMILIES},'active_training_integration':False},
            'diversity':{'query_types':dict(Counter(r['operation_id'] for r in accepted)),
                'target_semantics':dict(Counter(r['game_metadata']['target_semantics'] for r in accepted)),
                'answerability':dict(Counter(r['answerability_status'] for r in accepted)),
                'candidate_counts':dict(Counter(str(len(r['options'])) for r in accepted)),
                'difficulty':dict(Counter(r['difficulty'] for r in accepted)),
                'sokoban_boxes':dict(Counter(str(len(json.loads(r['context'])['boxes'])) for r in accepted if r['topic']=='sokoban')),
                'sokoban_shortest_distances':dict(Counter(str(r['game_metadata']['teacher'].get('shortest_distance')) for r in accepted if r['topic']=='sokoban'))},
            'verification':{'verified_originals':len(rows),'all_targets_recomputed':True,'all_finite_prior_or_simulator_states_reconstructed':True,
                'prior_and_output_checksums_verified':True,'zero_duplicate_original_ids':len(seen)==len(rows),'zero_eligible_group_overlap':True,
                'no_hidden_secret_in_model_input':True},
            'provenance':{'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                'sources':'self-generated states/public transcripts only; no external benchmarks or game engine',
                'license_note':'No third-party source content. Repository license is not declared by this generator.',
                'label_methods':'Fraction finite-law convolution/EV, exhaustive permitted-history posterior, complete reverse Sokoban BFS'},
            'compute':{'cpu_seconds':round(time.process_time()-cpu_started,3),'data_authoring_api_calls':0,'external_consultations':0},
            'limitations':['Dice is an explicit one-step weighted threshold game, not Blackjack or a multistep policy',
                'Mastermind prior is uniform with replacement; information decisions are best among declared supplied guesses',
                'Sokoban generated levels are bounded square5..7boards with1..2boxes; no large-level search or perception claim',
                'IDK marks unavailable realized/established facts; known-law forecasts retain exact probability targets',
                'Conservative related-family grouping can unevenly allocate rows; conflicts inherit quarantine instead of resplitting prior datasets',
                'Stockfish not installed in PATH or /usr/games; chess remains an unimplemented adapter backlog',
                'No active training integration or external transfer claim']})
        update(); atomic_json(output_dir/'manifest.json',status); return status
    except BaseException as error:
        update({'state':'failed','phase':'failed','error_type':type(error).__name__,'completed_at':now()}); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output-dir',type=Path,required=True); parser.add_argument('--status',type=Path,required=True)
    parser.add_argument('--per-family',type=int,default=3000); parser.add_argument('--seed',type=int,default=20261003); args=parser.parse_args()
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[name]='1'
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3)); resource.setrlimit(resource.RLIMIT_CPU,(1800,1800))
    result=generate(args.output_dir,args.status,per_family=args.per_family,seed=args.seed)
    print(json.dumps({key:result[key] for key in ('state','generated_new_originals','accepted_originals','quarantined_originals','split_counts','elapsed_seconds','max_rss_mib')},sort_keys=True))


if __name__=='__main__': main()
