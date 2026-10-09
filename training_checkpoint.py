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
    backbone_parameter_names: list[str] | None = None,
    base_checkpoint: str | Path | None = None,
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
    base_reference = None
    if backbone_parameter_names is not None:
        if not include_backbone or not backbone_parameter_names or base_checkpoint is None:
            raise ValueError("Compact backbone checkpoints require trainable names and an immutable base checkpoint")
        if len(set(backbone_parameter_names)) != len(backbone_parameter_names):
            raise ValueError("Compact backbone parameter names must be unique")
        base_path = Path(base_checkpoint)
        if any(candidate.is_symlink() for candidate in [base_path, *base_path.parents]):
            raise ValueError("Compact checkpoint bases cannot use symlink paths")
        base = base_path.resolve()
        base_manifest = base / CHECKPOINT_FILE
        if not base_manifest.is_file() or base == destination.resolve():
            raise ValueError("A separate immutable base checkpoint is required")
        base_record = json.loads(base_manifest.read_text(encoding="utf-8"))
        if (base_record.get("include_backbone") is not True or base_record.get("backbone_state_kind", "full") != "full"
                or base_record.get("base_checkpoint") is not None):
            raise ValueError("Compact checkpoints require a pinned full, noncompact base")
        expected_names = base_record.get("manifest", {}).get("trainable_backbone_parameter_names")
        if expected_names != backbone_parameter_names or manifest.get("trainable_backbone_parameter_names") != expected_names:
            raise ValueError("Compact checkpoint parameter names differ from the immutable base training scope")
        base_reference = {"path": os.path.relpath(base, destination.resolve()),
                          "manifest_sha256": _sha256(base_manifest)}
        warm_start = manifest.get("warm_start")
        if isinstance(warm_start, dict) and warm_start.get("checkpoint_manifest_sha256") != base_reference["manifest_sha256"]:
            raise ValueError("Compact checkpoint base differs from its pinned warm start")
    elif base_checkpoint is not None:
        raise ValueError("An immutable base reference is only needed for a compact backbone checkpoint")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    try:
        if include_backbone:
            state = backbone.state_dict()
            if backbone_parameter_names is not None:
                if not set(backbone_parameter_names) <= set(dict(backbone.named_parameters())):
                    raise ValueError("Compact state must contain existing backbone parameters")
                state = {name:state[name] for name in backbone_parameter_names}
            torch.save(state, temporary / "backbone.pt")
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
            "backbone_state_kind": "trainable-overlay-v1" if backbone_parameter_names is not None else "full",
            "backbone_parameter_names": backbone_parameter_names,
            "base_checkpoint": base_reference,
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
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Verify and restore checkpoint state, returning saved progress and metadata."""
    directory = Path(path)
    checkpoint_path = directory / CHECKPOINT_FILE
    if directory.is_symlink() or not directory.is_dir() or not checkpoint_path.is_file():
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
        # File-backed CPU storage avoids materializing a second full model on
        # shared-memory accelerators while copying the selected state to GPU.
        kind = checkpoint.get("backbone_state_kind", "full")
        if kind == "trainable-overlay-v1":
            reference = checkpoint.get("base_checkpoint")
            names = checkpoint.get("backbone_parameter_names")
            if (not isinstance(reference, dict) or not isinstance(reference.get("path"), str)
                    or not isinstance(names, list) or not names or len(set(names)) != len(names)):
                raise ValueError("Compact checkpoint has no valid immutable base or parameter allowlist")
            if Path(reference["path"]).is_absolute():
                raise ValueError("Compact checkpoint base references must be relative")
            base_path = directory / reference["path"]
            if any(candidate.is_symlink() for candidate in [base_path, *base_path.parents]):
                raise ValueError("Compact checkpoint bases cannot use symlink paths")
            base = base_path.resolve()
            if _sha256(base / CHECKPOINT_FILE) != reference.get("manifest_sha256"):
                raise ValueError("Compact checkpoint base manifest checksum mismatch")
            warm_start = checkpoint.get("manifest", {}).get("warm_start")
            if isinstance(warm_start, dict) and warm_start.get("checkpoint_manifest_sha256") != reference["manifest_sha256"]:
                raise ValueError("Compact checkpoint base differs from its pinned warm start")
            base_record = json.loads((base / CHECKPOINT_FILE).read_text(encoding="utf-8"))
            if (base_record.get("include_backbone") is not True or base_record.get("backbone_state_kind", "full") != "full"
                    or base_record.get("base_checkpoint") is not None):
                raise ValueError("Compact checkpoints require a pinned full, noncompact base")
            expected_names = base_record.get("manifest", {}).get("trainable_backbone_parameter_names")
            if names != expected_names or checkpoint.get("manifest", {}).get("trainable_backbone_parameter_names") != expected_names:
                raise ValueError("Compact checkpoint parameter names differ from its base training scope")
            load_training_checkpoint(base, backbone=backbone, head=head, restore_rng=False)
            state = torch.load(directory / "backbone.pt", map_location=device, weights_only=True, mmap=True)
            if set(state) != set(names) or not set(names) <= set(dict(backbone.named_parameters())):
                raise ValueError("Compact checkpoint does not match its backbone parameter allowlist")
            backbone.load_state_dict(state, strict=False)
        elif kind == "full":
            backbone.load_state_dict(torch.load(directory / "backbone.pt", map_location=device, weights_only=True, mmap=True))
        else:
            raise ValueError("Unsupported backbone checkpoint state kind")
    head.load_state_dict(torch.load(directory / "head.pt", map_location=device, weights_only=True, mmap=True))

    if optimizer is not None:
        optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location=device, weights_only=True, mmap=True))
    if scheduler is not None:
        scheduler.load_state_dict(torch.load(directory / "scheduler.pt", map_location=device, weights_only=True))

    if restore_rng:
        rng_state = torch.load(directory / "torch_rng.pt", map_location="cpu", weights_only=True)
        torch.set_rng_state(rng_state["torch_cpu"])
        if rng_state["torch_cuda"] is not None and torch.cuda.is_initialized():
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
        random.setstate(_as_tuples(json.loads((directory / "python_rng.json").read_text(encoding="utf-8"))))

    return {"progress": checkpoint["progress"], "manifest": checkpoint["manifest"]}
