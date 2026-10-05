import json
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


def test_resume_matches_the_next_uninterrupted_optimizer_update(tmp_path):
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
    load_training_checkpoint(
        checkpoint,
        backbone=resumed_backbone,
        head=resumed_head,
        optimizer=resumed_optimizer,
        scheduler=resumed_scheduler,
    )
    train_update(resumed_backbone, resumed_head, resumed_optimizer, resumed_scheduler)

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
