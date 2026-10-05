from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pytest

from dataset_factory.game_data import Board, Nim, teach
from dataset_factory.game_data.expansion import (
    export_row, horizon_teacher, isolate_and_split, generate, lights_from_presses, verify_row,
)
from dataset_factory.game_data.pilot import generate as generate_pilot


def nim_row(heaps, index, split='development', group=None):
    state = Nim(heaps)
    row = export_row(state, asdict(teach(state)), scenario=f'fixture-{index}', seed=17,
                     objective={'kind': 'full_game_minimax', 'description': 'Maximize win/loss'},
                     generation={'initial_heaps': list(heaps)})
    row['split'] = split
    if group:
        row['problem_group_id'] = group
    return row


def test_finite_horizon_is_exact_for_stated_reward_not_a_whole_game_claim():
    initial = Board.initial('connect_four')
    result = horizon_teacher(initial, 3)
    assert set(result['value_bounds']) == {(0, 0)}
    assert len(result['optimal_actions']) == 7
    state = initial
    history = [0, 1, 0, 1, 0, 1]
    for col in history:
        state = state.apply((col,))
    result = horizon_teacher(state, 3)
    # Column0 wins immediately; other unblocked opponent threats lose.
    assert result['value_bounds'][0] == (1, 1)
    assert result['value_bounds'][2] == (-1, -1)
    assert result['horizon_plies'] == 3 and result['nonterminal_horizon_reward'] == 0

    def reference(position, remaining):
        outcome = position.outcome()
        if outcome is not None:
            return outcome * position.player
        if not remaining:
            return 0
        return max(-reference(position.apply(action), remaining - 1) for action in position.legal_actions())

    assert tuple((-reference(state.apply(action), 2),) * 2 for action in state.legal_actions()) == result['value_bounds']
    with pytest.raises(ValueError):
        horizon_teacher(state, 4)


def test_prior_connected_split_conflict_is_quarantined_without_mutating_prior():
    prior = [nim_row((1, 2), 0, 'train', 'old-train'), nim_row((1, 6), 1, 'development', 'old-dev')]
    original = json.dumps(prior, sort_keys=True)
    rows = [nim_row((1, 4), 2)]
    report = isolate_and_split(prior, rows, 17)
    assert rows[0]['split'] == 'quarantine'
    assert set(rows[0]['game_metadata']['prior_group_ids']) == {'old-train', 'old-dev'}
    assert report['dispositions']['prior_split_conflict_quarantined'] == 1
    assert json.dumps(prior, sort_keys=True) == original


def test_one_prior_split_is_inherited_and_distant_families_remain_separate():
    prior = [nim_row((1, 2), 0, 'train', 'old-train')]
    rows = [nim_row((1, 3), 1), nim_row((14, 16, 18), 2)]
    isolate_and_split(prior, rows, 17)
    assert rows[0]['split'] == 'train' and rows[0]['problem_group_id'] == 'old-train'
    assert rows[1]['game_metadata']['prior_split_disposition'] == 'new_component'
    assert rows[1]['problem_group_id'] != 'old-train'


def test_puzzle_scramble_replays_and_targets_cannot_silently_permute():
    state = lights_from_presses(5, [0, 3, 12, 21])
    row = export_row(state, state.teach(), scenario='lights-fixture', seed=17,
                     objective={'kind': 'exact_shortest_presses', 'description': 'minimize presses'},
                     generation={'scramble_press_indices': [0, 3, 12, 21]})
    verify_row(row)
    # Swap a positive and zero target while keeping option IDs fixed.
    positive = next(i for i, p in enumerate(row['target_distribution']) if p > 0)
    zero = next(i for i, p in enumerate(row['target_distribution']) if p == 0)
    row['target_distribution'][positive], row['target_distribution'][zero] = row['target_distribution'][zero], row['target_distribution'][positive]
    with pytest.raises(ValueError):
        verify_row(row)


def test_small_expansion_has_manifest_verified_new_rows_and_preserved_prior(tmp_path):
    pilot = tmp_path / 'pilot'
    generate_pilot(pilot, tmp_path / 'pilot-status.json', per_game=2, seed=171)
    prior_hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in pilot.iterdir()}
    output = tmp_path / 'expansion'
    quotas = {'connect_four': 3, 'nim': 3, 'grid_navigation': 3, 'lights_out': 3, 'tic_tac_toe': 2}
    report = generate(output, tmp_path / 'status.json', pilot, seed=711, quotas=quotas, max_seconds=30)
    assert report['state'] == 'complete'
    assert report['generated_new_originals'] == 14
    assert report['accepted_originals'] + report['quarantined_originals'] == 14
    assert report['verification']['zero_prior_canonical_duplicates']
    assert report['limits']['cpu_threads'] == 1 and not report['gpu_used']
    assert report['api_calls'] == 0
    for record in report['files']:
        content = Path(record['path']).read_bytes()
        assert hashlib.sha256(content).hexdigest() == record['sha256']
        rows = [json.loads(line) for line in content.splitlines()]
        assert len(rows) == record['rows']
        for row in rows:
            verify_row(row)
    assert all(hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in prior_hashes.items())
    assert all(not item['path'].endswith('quarantine.jsonl') for item in report['training_allowlist'])
    with pytest.raises(ValueError):
        generate(output, tmp_path / 'status.json', pilot, quotas=quotas)
