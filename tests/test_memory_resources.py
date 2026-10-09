import random
import signal
import threading
from collections import deque

import pytest
import torch

import paired_training
import training_loop


GIB = 1024**3


def snapshot(available):
    return {
        "host_available_bytes": available,
        "driver_free_bytes": None,
        "device_total_bytes": None,
        "device_allocated_bytes": 0,
        "device_reserved_bytes": 0,
        "reusable_device_bytes": 0,
        "effective_available_bytes": available,
        "process_cap_bytes": None,
        "process_cap_headroom_bytes": None,
    }


def checks_for(monkeypatch, snapshots, *, additional=0, reserve_gib=1):
    pending = deque(snapshots)

    def read_snapshot(*args, **kwargs):
        assert pending, "Unexpected extra resource check"
        return pending.popleft()

    monkeypatch.setattr(paired_training, "memory_snapshot", read_snapshot)
    return lambda: paired_training.require_memory(additional, reserve_gib)


class ScriptedStop(training_loop.StopController):
    def __init__(self, *, stop_on_wait=False):
        super().__init__()
        self.waits = []
        self.stop_on_wait = stop_on_wait

    def wait(self, seconds):
        self.waits.append(seconds)
        if self.stop_on_wait:
            self.request("test stop while waiting")
        return self.requested


def test_cpu_snapshot_uses_host_memory_without_querying_cuda(monkeypatch):
    monkeypatch.setattr(paired_training, "available_memory_bytes", lambda: 20 * GIB)

    def unexpected_cuda(*args, **kwargs):
        pytest.fail("CPU resource checks must not query CUDA")

    for name in ("mem_get_info", "memory_allocated", "memory_reserved", "get_per_process_memory_fraction"):
        monkeypatch.setattr(torch.cuda, name, unexpected_cuda)

    assert paired_training.memory_snapshot("cpu") == snapshot(20 * GIB)


@pytest.mark.parametrize(
    "host,driver_free,allocated,reserved,effective,reusable",
    [
        (20, 12, 2, 8, 18, 6),
        (14, 12, 2, 8, 14, 6),
        (20, 12, 8, 2, 12, 0),
    ],
)
def test_cuda_reuses_own_cache_without_adding_it_to_host_memory(
    monkeypatch, host, driver_free, allocated, reserved, effective, reusable
):
    monkeypatch.setattr(paired_training, "available_memory_bytes", lambda: host * GIB)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (driver_free * GIB, 64 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: allocated * GIB)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args, **kwargs: reserved * GIB)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda *args, **kwargs: 1.0)

    actual = paired_training.memory_snapshot("cuda")

    assert actual == {
        "host_available_bytes": host * GIB,
        "driver_free_bytes": driver_free * GIB,
        "device_total_bytes": 64 * GIB,
        "device_allocated_bytes": allocated * GIB,
        "device_reserved_bytes": reserved * GIB,
        "reusable_device_bytes": reusable * GIB,
        "effective_available_bytes": effective * GIB,
        "process_cap_bytes": 64 * GIB,
        "process_cap_headroom_bytes": max(64 - allocated, 0) * GIB,
    }


@pytest.mark.parametrize("additional,passes", [(4, True), (5, False)])
def test_private_cap_limits_new_allocations_without_charging_global_reserve_to_it(
    monkeypatch, additional, passes
):
    monkeypatch.setattr(paired_training, "available_memory_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (32 * GIB, 64 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 12 * GIB)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args, **kwargs: 14 * GIB)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda *args, **kwargs: .25)

    actual = paired_training.memory_snapshot("cuda")
    assert actual["process_cap_bytes"] == 16 * GIB
    assert actual["process_cap_headroom_bytes"] == 4 * GIB
    assert actual["effective_available_bytes"] == 34 * GIB
    if passes:
        assert paired_training.require_memory(additional * GIB, 12, "cuda") == actual
    else:
        with pytest.raises(paired_training.TrainingMemoryPressure) as caught:
            paired_training.require_memory(additional * GIB, 12, "cuda")
        assert caught.value.snapshot == actual


def test_cache_reserved_beyond_cap_does_not_increase_private_headroom(monkeypatch):
    monkeypatch.setattr(paired_training, "available_memory_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda *args, **kwargs: (32 * GIB, 64 * GIB))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 12 * GIB)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *args, **kwargs: 20 * GIB)
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda *args, **kwargs: .25)

    actual = paired_training.memory_snapshot("cuda")
    assert actual["reusable_device_bytes"] == 8 * GIB
    assert actual["process_cap_headroom_bytes"] == 4 * GIB
    assert paired_training.require_memory(4 * GIB, 12, "cuda") == actual
    with pytest.raises(paired_training.TrainingMemoryPressure):
        paired_training.require_memory(5 * GIB, 12, "cuda")


def test_cached_memory_cannot_replace_physical_reserve_and_is_released_before_recheck(monkeypatch):
    released = []
    monkeypatch.setattr(paired_training, "available_memory_bytes", lambda: 40 * GIB)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda *args, **kwargs: ((24 if released else 8) * GIB, 64 * GIB)
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *args, **kwargs: 2 * GIB)
    monkeypatch.setattr(
        torch.cuda, "memory_reserved", lambda *args, **kwargs: (2 if released else 18) * GIB
    )
    monkeypatch.setattr(torch.cuda, "get_per_process_memory_fraction", lambda *args, **kwargs: 1.0)

    before = paired_training.memory_snapshot("cuda")
    assert before["effective_available_bytes"] == 24 * GIB
    with pytest.raises(paired_training.TrainingMemoryPressure):
        paired_training.require_memory(8 * GIB, 12, "cuda")

    stop = ScriptedStop()
    checkpoints = []
    assert training_loop.wait_for_resources(
        lambda: paired_training.require_memory(8 * GIB, 12, "cuda"),
        stop=stop, release_cache=lambda: released.append(True), before_wait=checkpoints.append,
    )
    assert released == [True]
    assert checkpoints == []
    assert stop.waits == []


def test_memory_threshold_is_inclusive_and_pressure_retains_byte_requirements(monkeypatch):
    enough = snapshot(3 * GIB)
    monkeypatch.setattr(paired_training, "memory_snapshot", lambda *args, **kwargs: enough)
    assert paired_training.require_memory(2 * GIB, 1) == enough

    low = snapshot(3 * GIB - 1)
    monkeypatch.setattr(paired_training, "memory_snapshot", lambda *args, **kwargs: low)
    with pytest.raises(paired_training.TrainingMemoryPressure) as caught:
        paired_training.require_memory(2 * GIB, 1)
    assert caught.value.snapshot == low
    assert caught.value.additional_bytes == 2 * GIB
    assert caught.value.reserve_bytes == GIB


def test_releasing_cache_rechecks_before_checkpointing_or_waiting(monkeypatch):
    check = checks_for(monkeypatch, [snapshot(GIB - 1), snapshot(GIB)])
    stop = ScriptedStop()
    releases, checkpoints, publications = [], [], []

    assert training_loop.wait_for_resources(
        check,
        stop=stop,
        release_cache=lambda: releases.append(True),
        before_wait=checkpoints.append,
        publish=lambda state, waiting: publications.append((state, waiting)),
        poll_seconds=17,
    )

    assert releases == [True]
    assert checkpoints == []
    assert stop.waits == []
    assert publications == []


def test_pressure_checkpoints_once_then_polls_until_resources_return(monkeypatch):
    low, lower, enough = snapshot(GIB - 1), snapshot(GIB - 2), snapshot(GIB)
    check = checks_for(monkeypatch, [low, low, lower, enough])
    stop = ScriptedStop()
    releases, checkpoints, publications = [], [], []
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()

    assert training_loop.wait_for_resources(
        check,
        stop=stop,
        release_cache=lambda: releases.append(True),
        before_wait=checkpoints.append,
        publish=lambda state, waiting: publications.append((state, waiting)),
        poll_seconds=17,
    )

    assert releases == [True]
    assert checkpoints == [low]
    assert stop.waits == [17, 17]
    assert publications == [(low, True), (lower, True), (enough, False)]
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_stop_during_resource_wait_preserves_cursor_and_exposure(monkeypatch):
    low = snapshot(GIB - 1)
    check = checks_for(monkeypatch, [low, low])
    stop = ScriptedStop(stop_on_wait=True)
    plan = training_loop.build_epoch_plan(
        {"one": 10, "two": 20}, seed=1, epoch=0, microbatch=1,
        accumulation_questions=1, max_batch_tokens=20,
    )
    cursor = training_loop.TrainingCursor(global_update=1024, questions_seen=16384, tokens_seen=90000)
    before = cursor.state_dict()
    checkpoints = []
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()

    assert not training_loop.run_epoch_updates(
        plan,
        cursor,
        stop=stop,
        before_step=lambda: training_loop.wait_for_resources(
            check, stop=stop, release_cache=lambda: None,
            before_wait=lambda state: checkpoints.append(cursor.state_dict()), poll_seconds=17,
        ),
        step=lambda batches: pytest.fail("A stopped resource wait must not execute an update"),
        observe=lambda cursor, metrics: pytest.fail("A stopped resource wait must not publish an update"),
    )

    assert cursor.state_dict() == before
    assert checkpoints == [before]
    assert stop.waits == [17]
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)


def test_gate_runs_before_each_step_and_rejection_keeps_next_update_pending():
    plan = training_loop.build_epoch_plan(
        {"one": 10, "two": 20}, seed=1, epoch=0, microbatch=1,
        accumulation_questions=1, max_batch_tokens=20,
    )
    cursor = training_loop.TrainingCursor()
    events, gates = [], deque([True, False])

    def gate():
        events.append("gate")
        return gates.popleft()

    def step(batches):
        events.append("step")
        return {}

    assert not training_loop.run_epoch_updates(
        plan, cursor, stop=training_loop.StopController(), before_step=gate,
        step=step, observe=lambda cursor, metrics: events.append("observe"),
    )
    assert events == ["gate", "step", "observe", "gate"]
    assert cursor.epoch == 0
    assert cursor.next_update == cursor.global_update == cursor.questions_seen == 1
    completed = plan.updates[0][0][0]
    assert cursor.tokens_seen == plan.token_lengths[completed]


@pytest.mark.parametrize("via_signal", [False, True])
def test_stop_wait_is_interruptible_by_request_or_signal(via_signal):
    stop = training_loop.StopController()
    ready = threading.Event()
    completed = threading.Event()
    python_rng, torch_rng = random.getstate(), torch.get_rng_state().clone()

    def wait():
        ready.set()
        stop.wait(10)
        completed.set()

    worker = threading.Thread(target=wait, daemon=True)
    worker.start()
    assert ready.wait(1)
    if via_signal:
        stop._signal(signal.SIGTERM, None)
    else:
        stop.request("test requested stop")
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert completed.is_set()
    assert stop.requested
    assert stop.reason == ("SIGTERM" if via_signal else "test requested stop")
    assert random.getstate() == python_rng
    assert torch.equal(torch.get_rng_state(), torch_rng)
