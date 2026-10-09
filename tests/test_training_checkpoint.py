import json
import hashlib
import random

import pytest
import torch
from torch.nn import functional as F

from training_checkpoint import load_training_checkpoint, save_training_checkpoint


class ToyConfig:
    def save_pretrained(self, directory):
        directory.mkdir(parents=True)
        (directory / "config.json").write_text('{"hidden_size": 4}\n')


class ToyTokenizer:
    def save_pretrained(self, directory):
        directory.mkdir(parents=True)
        (directory / "tokenizer.json").write_text('{"toy": true}\n')


class ToyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)
        self.dropout = torch.nn.Dropout(0.25)
        self.config = ToyConfig()

    def forward(self, inputs):
        return self.dropout(self.projection(inputs))


class ToyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.score = torch.nn.Linear(4, 2)

    def forward(self, hidden):
        return self.score(hidden)


def make_training_state():
    backbone = ToyBackbone()
    head = ToyHead()
    optimizer = torch.optim.AdamW(
        [*backbone.parameters(), *head.parameters()], lr=0.01, weight_decay=0.02
    )
    return backbone, head, optimizer


def train_update(backbone, head, optimizer, scheduler=None):
    backbone.train()
    head.train()
    inputs = torch.randn(5, 3)
    labels = torch.tensor([0, 1, 1, 0, 1])
    logits = head(backbone(inputs))
    loss = F.cross_entropy(logits, labels) + random.random() * logits.square().mean() * 0.01
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if scheduler is not None:
        scheduler.step()


def save(path, backbone, head, optimizer, **kwargs):
    return save_training_checkpoint(
        path,
        backbone=backbone,
        tokenizer=ToyTokenizer(),
        head=head,
        optimizer=optimizer,
        progress={"epoch": 2, "update": 7, "sampler_seed": 41},
        manifest={"dataset_sha256": "example-hash", "split_sha256": "split-hash"},
        **kwargs,
    )


def test_checkpoint_restores_predictions_and_offline_artifacts(tmp_path):
    torch.manual_seed(13)
    backbone, head, optimizer = make_training_state()
    train_update(backbone, head, optimizer)
    inputs = torch.tensor([[0.25, -1.0, 2.0], [3.0, 0.5, -0.5]])
    backbone.eval()
    head.eval()
    with torch.inference_mode():
        expected = head(backbone(inputs)).clone()

    checkpoint = save(tmp_path / "checkpoint", backbone, head, optimizer)
    assert (checkpoint / "tokenizer" / "tokenizer.json").is_file()
    assert (checkpoint / "backbone_config" / "config.json").is_file()

    restored_backbone, restored_head, _ = make_training_state()
    for parameter in [*restored_backbone.parameters(), *restored_head.parameters()]:
        parameter.data.zero_()
    metadata = load_training_checkpoint(
        checkpoint, backbone=restored_backbone, head=restored_head
    )
    restored_backbone.eval()
    restored_head.eval()
    with torch.inference_mode():
        actual = restored_head(restored_backbone(inputs))

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert metadata == {
        "progress": {"epoch": 2, "update": 7, "sampler_seed": 41},
        "manifest": {"dataset_sha256": "example-hash", "split_sha256": "split-hash"},
    }


def test_resume_matches_the_next_uninterrupted_optimizer_update(tmp_path, monkeypatch):
    torch.manual_seed(404)
    random.seed(123)
    backbone, head, optimizer = make_training_state()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
    train_update(backbone, head, optimizer, scheduler)
    checkpoint = save(tmp_path / "checkpoint", backbone, head, optimizer, scheduler=scheduler)

    train_update(backbone, head, optimizer, scheduler)
    expected_backbone = {key: value.detach().clone() for key, value in backbone.state_dict().items()}
    expected_head = {key: value.detach().clone() for key, value in head.state_dict().items()}

    resumed_backbone, resumed_head, resumed_optimizer = make_training_state()
    resumed_scheduler = torch.optim.lr_scheduler.StepLR(
        resumed_optimizer, step_size=1, gamma=0.7
    )
    real_load = torch.load
    loads = []

    def record_load(path, **kwargs):
        loads.append((path.name, kwargs.get("mmap", False)))
        return real_load(path, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    load_training_checkpoint(
        checkpoint,
        backbone=resumed_backbone,
        head=resumed_head,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )
    train_update(resumed_backbone, resumed_head, resumed_optimizer, resumed_scheduler)
    assert {name for name, mapped in loads if mapped} == {"backbone.pt", "head.pt", "optimizer.pt"}
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    assert hashlib.sha256((checkpoint / "optimizer.pt").read_bytes()).hexdigest() == metadata["files"]["optimizer.pt"]

    for key, expected in expected_backbone.items():
        torch.testing.assert_close(resumed_backbone.state_dict()[key], expected, rtol=0, atol=0)
    for key, expected in expected_head.items():
        torch.testing.assert_close(resumed_head.state_dict()[key], expected, rtol=0, atol=0)
    assert resumed_scheduler.state_dict() == scheduler.state_dict()


def test_checkpoint_rejects_corrupted_artifact_before_loading(tmp_path):
    backbone, head, optimizer = make_training_state()
    checkpoint = save(tmp_path / "checkpoint", backbone, head, optimizer)
    artifact = checkpoint / "head.pt"
    contents = bytearray(artifact.read_bytes())
    contents[-1] ^= 1
    artifact.write_bytes(contents)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_training_checkpoint(checkpoint, backbone=ToyBackbone(), head=ToyHead())


def test_checkpoint_does_not_overwrite_an_existing_directory(tmp_path):
    backbone, head, optimizer = make_training_state()
    checkpoint = save(tmp_path / "checkpoint", backbone, head, optimizer)

    with pytest.raises(FileExistsError, match="already exists"):
        save(checkpoint, backbone, head, optimizer)

    assert json.loads((checkpoint / "checkpoint.json").read_text())["progress"]["update"] == 7


def test_weights_only_continuation_keeps_fresh_rng_and_optimizer(tmp_path):
    torch.manual_seed(27)
    backbone, head, optimizer = make_training_state()
    train_update(backbone, head, optimizer)
    checkpoint = save(tmp_path / 'weights-source', backbone, head, optimizer)
    fresh_backbone, fresh_head, fresh_optimizer = make_training_state()
    torch.manual_seed(913)
    random.seed(914)
    torch_state, python_state = torch.get_rng_state().clone(), random.getstate()
    load_training_checkpoint(checkpoint, backbone=fresh_backbone, head=fresh_head, restore_rng=False)
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert random.getstate() == python_state
    assert fresh_optimizer.state == {}
    for expected, actual in zip([*backbone.parameters(), *head.parameters()],
                               [*fresh_backbone.parameters(), *fresh_head.parameters()]):
        assert torch.equal(expected, actual)


def test_compact_checkpoint_reconstructs_full_base_in_frozen_inference_and_resumes(tmp_path):
    from collections import OrderedDict
    from training_loop import warmup_cosine_scheduler
    backbone = torch.nn.Sequential(OrderedDict([
        ('prefix', torch.nn.Linear(3, 3)), ('last', torch.nn.Linear(3, 4))]))
    head = torch.nn.Linear(4, 2)
    names = ['last.weight', 'last.bias']
    backbone.prefix.requires_grad_(False)
    optimizer = torch.optim.AdamW([*backbone.last.parameters(), *head.parameters()], lr=.01)
    manifest = {'trainable_backbone_parameter_names':names}
    base = save_training_checkpoint(tmp_path / 'base', backbone=backbone, head=head, optimizer=optimizer,
                                    progress={}, manifest=manifest)
    prefix = {name:value.clone() for name,value in backbone.prefix.state_dict().items()}
    scheduler = warmup_cosine_scheduler(optimizer, total_updates=4, warmup_fraction=.25)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        head(backbone(torch.ones(2, 3))).square().sum().backward()
        optimizer.step()
        scheduler.step()
    compact = save_training_checkpoint(tmp_path / 'compact', backbone=backbone, head=head,
        optimizer=optimizer, scheduler=scheduler, progress={'update':2}, manifest=manifest,
        backbone_parameter_names=names, base_checkpoint=base)
    metadata = json.loads((compact / 'checkpoint.json').read_text())
    assert metadata['backbone_state_kind'] == 'trainable-overlay-v1'
    assert set(torch.load(compact / 'backbone.pt', weights_only=True)) == set(names)
    expected = {name:value.clone() for name,value in backbone.state_dict().items()}
    expected_head = {name:value.clone() for name,value in head.state_dict().items()}
    for parameter in [*backbone.parameters(), *head.parameters()]:
        parameter.data.zero_()
    backbone.requires_grad_(False)
    load_training_checkpoint(compact, backbone=backbone, head=head, restore_rng=False)
    for name,value in backbone.state_dict().items():
        assert torch.equal(value, expected[name])
    for name,value in head.state_dict().items():
        assert torch.equal(value, expected_head[name])
    for name,value in backbone.prefix.state_dict().items():
        assert torch.equal(value, prefix[name])
    load_training_checkpoint(compact, backbone=backbone, head=head, optimizer=optimizer, scheduler=scheduler)
    assert scheduler.last_epoch == 2
    assert optimizer.state
    wrong_manifest = {'trainable_backbone_parameter_names':['prefix.weight']}
    with pytest.raises(ValueError, match='base training scope'):
        save_training_checkpoint(tmp_path / 'wrong',backbone=backbone,head=head,optimizer=optimizer,
            progress={},manifest=wrong_manifest,backbone_parameter_names=['prefix.weight'],base_checkpoint=base)
    with pytest.raises(ValueError, match='full, noncompact'):
        save_training_checkpoint(tmp_path / 'nested',backbone=backbone,head=head,optimizer=optimizer,
            progress={},manifest=manifest,backbone_parameter_names=names,base_checkpoint=compact)
    (base / 'checkpoint.json').write_text('{}\n')
    with pytest.raises(ValueError, match='base manifest checksum'):
        load_training_checkpoint(compact,backbone=backbone,head=head)
