"""Deterministic minibatch plans and update-boundary training lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import tempfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@dataclass
class TrainingCursor:
    epoch: int = 0
    next_update: int = 0
    global_update: int = 0
    questions_seen: int = 0
    tokens_seen: int = 0

    def state_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> TrainingCursor:
        values = {name: state[name] for name in cls.__dataclass_fields__}
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("Training cursor values must be non-negative integers")
        return cls(**values)


@dataclass(frozen=True)
class EpochPlan:
    epoch: int
    updates: list[list[list[str]]]
    token_lengths: dict[str, int]
    sha256: str


def build_epoch_plan(
    token_lengths: Mapping[str, int], *, seed: int, epoch: int,
    microbatch: int, accumulation_questions: int, max_batch_tokens: int,
    bucket_questions: int = 256,
) -> EpochPlan:
    """Shuffle questions, bucket similar lengths, and cap padded batch exposure."""
    if not token_lengths or min(microbatch, accumulation_questions, max_batch_tokens, bucket_questions) < 1:
        raise ValueError("A nonempty corpus and positive batch limits are required")
    if any(type(length) is not int or not 1 <= length <= max_batch_tokens for length in token_lengths.values()):
        raise ValueError("Every complete input must fit the batch token budget; refusing truncation")
    material = json.dumps([seed, epoch, "length-bucket-shuffle-v1"]).encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))
    identifiers = sorted(token_lengths)
    rng.shuffle(identifiers)
    batches: list[list[str]] = []
    for offset in range(0, len(identifiers), bucket_questions):
        bucket = sorted(identifiers[offset:offset + bucket_questions], key=token_lengths.__getitem__)
        batch: list[str] = []
        for identifier in bucket:
            if batch and (len(batch) == microbatch or
                          (len(batch) + 1) * token_lengths[identifier] > max_batch_tokens):
                batches.append(batch)
                batch = []
            batch.append(identifier)
        if batch:
            batches.append(batch)
    rng.shuffle(batches)
    updates: list[list[list[str]]] = []
    update: list[list[str]] = []
    questions = 0
    for original_batch in batches:
        batch = original_batch
        while batch:
            take = min(len(batch), accumulation_questions - questions)
            update.append(batch[:take])
            batch = batch[take:]
            questions += take
            if questions == accumulation_questions:
                updates.append(update)
                update, questions = [], 0
    if update:
        updates.append(update)
    canonical = json.dumps({"epoch": epoch, "updates": updates, "token_lengths": dict(token_lengths)},
                           sort_keys=True, separators=(",", ":")).encode()
    return EpochPlan(epoch, updates, dict(token_lengths), hashlib.sha256(canonical).hexdigest())


class StopController:
    """Finish the current optimizer update before responding to a normal stop."""

    def __init__(self):
        self.requested = False
        self.reason: str | None = None
        self._handlers: dict[int, Any] = {}

    def request(self, reason: str = "requested") -> None:
        self.requested = True
        self.reason = reason

    def _signal(self, number, frame) -> None:
        self.request(signal.Signals(number).name)

    def __enter__(self):
        for number in [signal.SIGINT, signal.SIGTERM]:
            self._handlers[number] = signal.signal(number, self._signal)
        return self

    def __exit__(self, exc_type, exc, traceback):
        for number, handler in self._handlers.items():
            signal.signal(number, handler)


def run_epoch_updates(
    plan: EpochPlan, cursor: TrainingCursor, *, stop: StopController,
    step: Callable[[list[list[str]]], dict[str, Any]],
    observe: Callable[[TrainingCursor, dict[str, Any]], None],
) -> bool:
    """Advance only completed optimizer updates; return false after a stop."""
    if cursor.epoch != plan.epoch or not 0 <= cursor.next_update <= len(plan.updates):
        raise ValueError("Resume cursor does not belong to the epoch plan")
    while cursor.next_update < len(plan.updates):
        if stop.requested:
            return False
        groups = plan.updates[cursor.next_update]
        metrics = step(groups)
        identifiers = [identifier for batch in groups for identifier in batch]
        cursor.next_update += 1
        cursor.global_update += 1
        cursor.questions_seen += len(identifiers)
        cursor.tokens_seen += sum(plan.token_lengths[identifier] for identifier in identifiers)
        observe(cursor, metrics)
        if stop.requested:
            return False
    cursor.epoch += 1
    cursor.next_update = 0
    return True


class FeatureCache:
    """Bounded CPU cache keyed by exact rendered batches and model semantics."""

    def __init__(self, namespace: str, max_bytes: int):
        if max_bytes < 0:
            raise ValueError("Feature cache limit must be non-negative")
        self.namespace = namespace
        self.max_bytes = max_bytes
        self.bytes = 0
        self.values: OrderedDict[str, torch.Tensor] = OrderedDict()

    def key(self, ordered_prompts: Sequence[str], padded_tokens: int) -> str:
        material = json.dumps([self.namespace, list(ordered_prompts), padded_tokens],
                              ensure_ascii=False, separators=(",", ":")).encode()
        return hashlib.sha256(material).hexdigest()

    def get(self, key: str, device: str) -> torch.Tensor | None:
        value = self.values.get(key)
        if value is not None:
            self.values.move_to_end(key)
            return value.to(device).clone()
        return None

    def put(self, key: str, features: torch.Tensor) -> None:
        value = features.detach().to("cpu").float().clone()
        size = value.numel() * value.element_size()
        if size > self.max_bytes:
            return
        previous = self.values.pop(key, None)
        if previous is not None:
            self.bytes -= previous.numel() * previous.element_size()
        while self.values and self.bytes + size > self.max_bytes:
            _, oldest = self.values.popitem(last=False)
            self.bytes -= oldest.numel() * oldest.element_size()
        self.values[key] = value
        self.bytes += size


class StatusPublisher:
    def __init__(self, global_path: Path, run_path: Path, metadata: Mapping[str, Any]):
        self.global_path = global_path
        self.run_path = run_path
        self.metadata = dict(metadata)

    def publish(self, status: str, cursor: TrainingCursor | None = None, **fields: Any) -> None:
        if status not in {"preparing", "running", "interrupted", "completed", "failed"}:
            raise ValueError("Unsupported training lifecycle status")
        value = {"schema_version": 1, **self.metadata, **fields, "status": status,
                 "training_running": status == "running",
                 "updated_at": datetime.now(timezone.utc).isoformat()}
        if cursor is not None:
            value.update(cursor.state_dict())
            value["update"] = cursor.global_update
        atomic_json(self.run_path, value)
        atomic_json(self.global_path, value)


def development_plateau(scores: Sequence[float], *, patience: int, min_delta: float) -> bool:
    """Count evaluations since meaningful improvement, independently of checkpoint ties."""
    if patience < 0 or min_delta < 0:
        raise ValueError("Early stopping limits must be nonnegative")
    if not patience or not scores:
        return False
    best, stale = scores[0], 0
    for score in scores[1:]:
        if score < best - min_delta:
            best, stale = score, 0
        else:
            stale += 1
    return stale >= patience
