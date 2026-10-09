import copy

import pytest
import torch
from torch.nn import functional as F

from training_checkpoint import load_training_checkpoint, save_training_checkpoint
from training_loop import FeatureCache, StatusPublisher, StopController, TrainingCursor, build_epoch_plan, run_epoch_updates


def test_epoch_plan_visits_questions_once_and_bounds_actual_padded_batches():
    lengths = {f"question-{index}": 10 + index * 13 for index in range(37)}
    kwargs = {"seed": 23, "microbatch": 4, "accumulation_questions": 7,
              "max_batch_tokens": 900, "bucket_questions": 12}
    first = build_epoch_plan(lengths, epoch=0, **kwargs)
    assert first == build_epoch_plan(lengths, epoch=0, **kwargs)
    second = build_epoch_plan(lengths, epoch=1, **kwargs)
    assert first.sha256 != second.sha256
    identifiers = [identifier for update in first.updates for batch in update for identifier in batch]
    assert sorted(identifiers) == sorted(lengths)
    assert all(sum(map(len, update)) == 7 for update in first.updates[:-1])
    for update in first.updates:
        for batch in update:
            assert len(batch) <= 4
            assert len(batch) * max(lengths[identifier] for identifier in batch) <= 900
    with pytest.raises(ValueError, match="refusing truncation"):
        build_epoch_plan({"long": 901}, epoch=0, **kwargs)


def toy_state():
    backbone = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.2))
    head = torch.nn.Linear(4, 3)
    optimizer = torch.optim.AdamW([*backbone.parameters(), *head.parameters()], lr=.01)
    return backbone, head, optimizer


def toy_step(backbone, head, optimizer, stop=None):
    calls = 0

    def step(batches):
        nonlocal calls
        calls += 1
        optimizer.zero_grad(set_to_none=False)
        total = sum(map(len, batches))
        loss_sum = 0
        for index, batch in enumerate(batches):
            values = torch.tensor([[int(identifier) / 10, 1, -int(identifier) / 20] for identifier in batch])
            labels = torch.tensor([int(identifier) % 3 for identifier in batch])
            loss = F.cross_entropy(head(backbone(values)), labels)
            (loss * len(batch) / total).backward()
            loss_sum += loss.item() * len(batch)
            if stop is not None and calls == 3 and index == 0:
                stop.request("test stop during a microbatch")
        optimizer.step()
        return {"train_loss": loss_sum / total}

    return step


def test_stop_finishes_update_and_multiepoch_resume_matches_uninterrupted(tmp_path):
    lengths = {str(index): 10 + index for index in range(23)}
    plans = [build_epoch_plan(lengths, seed=31, epoch=epoch, microbatch=3,
                              accumulation_questions=7, max_batch_tokens=200) for epoch in range(2)]
    torch.manual_seed(42)
    expected_backbone, expected_head, expected_optimizer = toy_state()
    expected = TrainingCursor()
    expected_step = toy_step(expected_backbone, expected_head, expected_optimizer)
    for plan in plans:
        assert run_epoch_updates(plan, expected, stop=StopController(), step=expected_step,
                                 observe=lambda cursor, metrics: None)

    torch.manual_seed(42)
    backbone, head, optimizer = toy_state()
    cursor = TrainingCursor()
    stop = StopController()
    assert not run_epoch_updates(plans[0], cursor, stop=stop,
                                 step=toy_step(backbone, head, optimizer, stop),
                                 observe=lambda cursor, metrics: None)
    assert cursor.next_update == 3
    assert cursor.questions_seen == 21
    path = save_training_checkpoint(tmp_path / "interrupted", backbone=backbone, head=head,
                                    optimizer=optimizer, progress=cursor.state_dict(),
                                    manifest={"epoch_plan_sha256": plans[0].sha256})
    restored_backbone, restored_head, restored_optimizer = toy_state()
    restored = load_training_checkpoint(path, backbone=restored_backbone, head=restored_head,
                                        optimizer=restored_optimizer)
    resumed = TrainingCursor.from_state_dict(restored["progress"])
    restored_step = toy_step(restored_backbone, restored_head, restored_optimizer)
    for plan in plans[resumed.epoch:]:
        assert run_epoch_updates(plan, resumed, stop=StopController(), step=restored_step,
                                 observe=lambda cursor, metrics: None)
    assert resumed == expected
    assert resumed.questions_seen == 46
    for expected_value, actual_value in zip([*expected_backbone.parameters(), *expected_head.parameters()],
                                          [*restored_backbone.parameters(), *restored_head.parameters()]):
        assert torch.equal(expected_value, actual_value)


def test_stop_before_update_does_not_advance_exposure():
    plan = build_epoch_plan({"one": 10}, seed=1, epoch=0, microbatch=1,
                             accumulation_questions=1, max_batch_tokens=10)
    cursor = TrainingCursor()
    before = copy.deepcopy(cursor)
    stop = StopController()
    stop.request()
    assert not run_epoch_updates(plan, cursor, stop=stop,
                                 step=lambda batches: pytest.fail("step must not run"),
                                 observe=lambda cursor, metrics: None)
    assert cursor == before


def test_feature_cache_keys_order_and_batch_layout_and_bounds_cpu_storage():
    cache = FeatureCache("model-revision-and-format", 16)
    key = cache.key(["choices: first, second"], 10)
    assert key != cache.key(["choices: second, first"], 10)
    assert key != cache.key(["choices: first, second"], 11)
    assert key != cache.key(["choices: first, second", "another prompt"], 10)
    cache.put(key, torch.tensor([[1., 2.]], requires_grad=True))
    cached = cache.get(key, "cpu")
    assert cached is not None and not cached.requires_grad
    cached[0, 0] = 99
    assert cache.get(key, "cpu")[0, 0] == 1
    cache.put("second", torch.ones(2))
    cache.put("third", torch.ones(2))
    assert cache.bytes == 16
    assert cache.get(key, "cpu") is None


def test_status_distinguishes_preparation_running_and_interruption(tmp_path):
    import json

    global_path, run_path = tmp_path / "global.json", tmp_path / "run.json"
    publisher = StatusPublisher(global_path, run_path, {"launch_gate": "held", "arm": "full"})
    publisher.publish("preparing", message="No training launch is authorized")
    assert json.loads(global_path.read_text())["training_running"] is False
    publisher.publish("running", TrainingCursor(global_update=3, questions_seen=12), launch_gate="fulfilled")
    assert json.loads(global_path.read_text())["training_running"] is True
    publisher.publish("interrupted", TrainingCursor(global_update=3, questions_seen=12))
    published = json.loads(run_path.read_text())
    assert published["status"] == "interrupted" and not published["training_running"]
    assert published["questions_seen"] == 12
    assert json.loads(global_path.read_text()) == published


def test_development_plateau_requires_consecutive_meaningful_stagnation():
    from training_loop import development_plateau
    assert not development_plateau([1., .9, .89999], patience=2, min_delta=.0001)
    assert development_plateau([1., .9, .89999, .89998], patience=2, min_delta=.0001)
    assert not development_plateau([1., .9, .89999, .8], patience=2, min_delta=.0001)
    assert not development_plateau([1., 2., 3.], patience=0, min_delta=.0001)


def test_warmup_cosine_scales_groups_and_resume_preserves_next_update(tmp_path):
    from training_loop import warmup_cosine_scheduler
    backbone, head, optimizer = toy_state()
    optimizer.param_groups[0]['lr'] = .1
    optimizer.add_param_group({'params': [torch.nn.Parameter(torch.ones(1))], 'lr': .01})
    scheduler = warmup_cosine_scheduler(optimizer, total_updates=10, warmup_fraction=.3)
    assert optimizer.param_groups[0]['lr'] == pytest.approx(.1 / 3)
    rates = []
    for _ in range(4):
        rates.append([group['lr'] for group in optimizer.param_groups])
        optimizer.step()
        scheduler.step()
    assert rates[2][0] == pytest.approx(.1)
    assert all(first == pytest.approx(second * 10) for first, second in rates)
    path = save_training_checkpoint(tmp_path / 'scheduled', backbone=backbone, head=head,
                                    optimizer=optimizer, scheduler=scheduler, progress={}, manifest={})
    expected = [group['lr'] for group in optimizer.param_groups]
    load_training_checkpoint(path, backbone=backbone, head=head, optimizer=optimizer, scheduler=scheduler)
    assert [group['lr'] for group in optimizer.param_groups] == expected
    for _ in range(6):
        optimizer.step()
        scheduler.step()
    assert all(group['lr'] == 0 for group in optimizer.param_groups)
