import json
from pathlib import Path

import pytest
import torch

import experiment as core
from transfer_experiment import measure, validate_source
from size_experiment import SETTINGS, digest


def test_transfer_measurements_mask_choices_and_leave_head_unchanged():
    head = core.DecisionHead(3)
    with torch.no_grad():
        head.score.weight.zero_()
        head.score.bias.zero_()
        head.score.bias[10] = 100  # An unavailable choice must never win.
    before = {key: value.clone() for key, value in head.state_dict().items()}
    examples = [dict(id='a', family='x', options=['a', 'b'], answer_index=0),
                dict(id='b', family='y', options=['a', 'b', 'c', 'd'], answer_index=1)]
    result = measure(head, torch.ones(2, 3), examples)
    assert result['accuracy'] == 0.5
    assert result['families']['x']['accuracy'] == 1
    assert result['families']['y']['accuracy'] == 0
    assert result['predictions'][0]['probabilities'] == [0.5, 0.5]
    assert result['predictions'][1]['probabilities'] == [0.25] * 4
    assert all(torch.equal(value, head.state_dict()[key]) for key, value in before.items())
    assert all(parameter.grad is None for parameter in head.parameters())


def make_underfit_source(tmp_path):
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    examples = tmp_path / 'examples.json'
    write(examples, [])
    metadata = {'data_sha256': digest(examples), 'settings': SETTINGS, 'seeds': [42],
                'subsets': {'16': ['train'], '64': ['train']}}
    write(tmp_path / 'manifest.json', metadata)
    torch.save(torch.ones(2, 3), tmp_path / 'features.pt')
    write(tmp_path / 'cache.json', {'features_sha256': digest(tmp_path / 'features.pt')})
    write(tmp_path / 'reload_verification.json', {'features_close': True, 'predictions_identical': True})
    for size in ('16', '64'):
        folder = tmp_path / size / '42'
        folder.mkdir(parents=True)
        core.save_head(folder / 'head.pt', core.DecisionHead(3))
        write(folder / 'result.json', {'seed': 42, 'training_ids': ['train'],
                                     'head_sha256': digest(folder / 'head.pt'), 'report': {'passed': False, 'reloaded_head_logits_identical': True,
                                                'training': {'weight_change_l2': 1, 'final': {'loss': .2, 'accuracy': .9}}}})


def test_underfit_heads_are_measured_but_tampering_is_rejected(tmp_path):
    make_underfit_source(tmp_path)
    assert len(validate_source(tmp_path)[1]) == 2
    with (tmp_path / '64/42/head.pt').open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        validate_source(tmp_path)


def test_source_pipeline_failure_is_rejected(tmp_path):
    make_underfit_source(tmp_path)
    folder = tmp_path / '64/42'
    core.save_head(folder / 'head.pt', core.DecisionHead(3))
    record_path = folder / 'result.json'
    record = json.loads(record_path.read_text())
    record['head_sha256'] = digest(folder / 'head.pt')
    record['report']['reloaded_head_logits_identical'] = False
    record_path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='pipeline'):
        validate_source(tmp_path)
