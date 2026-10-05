"""Atomic, checksummed checkpoints for fine-tuning runs."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import torch


SCHEMA_VERSION = 1
CHECKPOINT_FILE = "checkpoint.json"
_CHUNK_SIZE = 1024 * 1024


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _python_rng_state() -> list[Any]:
    # JSON turns tuples into arrays; load converts them back before random.setstate.
    return list(random.getstate())


def _as_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_as_tuples(item) for item in value)
    return value


def _artifact_hashes(directory: Path) -> dict[str, str]:
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Checkpoint artifacts cannot be symlinks: {path.relative_to(directory)}")
        if path.is_file() and path.relative_to(directory).as_posix() != CHECKPOINT_FILE:
            result[path.relative_to(directory).as_posix()] = _sha256(path)
    return result


def _verify_artifacts(directory: Path, hashes: Any) -> None:
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Checkpoint has no artifact hashes")

    actual = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Checkpoint artifacts cannot be symlinks: {path.relative_to(directory)}")
        if path.is_file() and path.relative_to(directory).as_posix() != CHECKPOINT_FILE:
            actual.add(path.relative_to(directory).as_posix())
    if actual != set(hashes):
        raise ValueError("Checkpoint artifact list does not match its manifest")

    for relative_path, expected in hashes.items():
        candidate_name = PurePosixPath(relative_path)
        if candidate_name.is_absolute() or ".." in candidate_name.parts:
            raise ValueError(f"Invalid checkpoint artifact path: {relative_path}")
        candidate = directory.joinpath(*candidate_name.parts)
        if not candidate.is_file() or _sha256(candidate) != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {relative_path}")


def _capture_rng_state() -> dict[str, Any]:
    cuda_states = None
    if torch.cuda.is_initialized():
        cuda_states = torch.cuda.get_rng_state_all()
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": cuda_states}


def save_training_checkpoint(
    output_dir: str | Path,
    *,
    backbone: torch.nn.Module,
    tokenizer: Any = None,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any = None,
    progress: dict[str, Any],
    manifest: dict[str, Any],
    include_backbone: bool = True,
) -> Path:
    """Write a complete checkpoint to a new directory and return its path."""
    destination = Path(output_dir)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Checkpoint already exists: {destination}")
    if not isinstance(progress, dict) or not isinstance(manifest, dict):
        raise TypeError("progress and manifest must be dictionaries")
    if type(include_backbone) is not bool:
        raise TypeError("include_backbone must be a boolean")
    # Validate metadata before writing large model files.
    json.dumps(progress, allow_nan=False)
    json.dumps(manifest, allow_nan=False)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        if include_backbone:
            torch.save(backbone.state_dict(), temporary / "backbone.pt")
        torch.save(head.state_dict(), temporary / "head.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizer.pt")
        if scheduler is not None:
            torch.save(scheduler.state_dict(), temporary / "scheduler.pt")

        if tokenizer is not None:
            tokenizer.save_pretrained(temporary / "tokenizer")
        config = getattr(backbone, "config", None)
        if config is not None and hasattr(config, "save_pretrained"):
            config.save_pretrained(temporary / "backbone_config")

        torch.save(_capture_rng_state(), temporary / "torch_rng.pt")
        _write_json(temporary / "python_rng.json", _python_rng_state())
        checkpoint = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "include_backbone": include_backbone,
            "has_scheduler": scheduler is not None,
            "progress": progress,
            "manifest": manifest,
            "files": _artifact_hashes(temporary),
        }
        _write_json(temporary / CHECKPOINT_FILE, checkpoint)

        # A same-filesystem rename makes the finished checkpoint visible at once.
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Checkpoint already exists: {destination}")
        os.rename(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return destination


def load_training_checkpoint(
    path: str | Path,
    *,
    backbone: torch.nn.Module | None,
    head: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any = None,
) -> dict[str, Any]:
    """Verify and restore checkpoint state, returning saved progress and metadata."""
    directory = Path(path)
    checkpoint_path = directory / CHECKPOINT_FILE
    if not directory.is_dir() or not checkpoint_path.is_file():
        raise FileNotFoundError(f"No training checkpoint found at {directory}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported training checkpoint schema")
    _verify_artifacts(directory, checkpoint.get("files"))

    has_scheduler = checkpoint.get("has_scheduler") is True
    if optimizer is not None and has_scheduler and scheduler is None:
        raise ValueError("This training checkpoint has a scheduler; pass it to resume training")
    if scheduler is not None and not has_scheduler:
        raise ValueError("A scheduler was provided, but the checkpoint has no scheduler state")
    if scheduler is not None and optimizer is None:
        raise ValueError("Pass the optimizer when restoring a scheduler")

    device = "cpu"
    if checkpoint.get("include_backbone"):
        if backbone is None:
            raise ValueError("This checkpoint contains a backbone; pass a backbone to restore it")
        backbone.load_state_dict(torch.load(directory / "backbone.pt", map_location=device, weights_only=True))
    head.load_state_dict(torch.load(directory / "head.pt", map_location=device, weights_only=True))

    if optimizer is not None:
        optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location=device, weights_only=True))
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(directory / "scheduler.pt", map_location=device, weights_only=True))

    rng_state = torch.load(directory / "torch_rng.pt", map_location="cpu", weights_only=True)
    torch.set_rng_state(rng_state["torch_cpu"])
    if rng_state["torch_cuda"] is not None and torch.cuda.is_initialized():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
    random.setstate(_as_tuples(json.loads((directory / "python_rng.json").read_text(encoding="utf-8"))))

    return {"progress": checkpoint["progress"], "manifest": checkpoint["manifest"]}
