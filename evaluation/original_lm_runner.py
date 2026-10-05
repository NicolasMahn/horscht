"""Frozen aggregate-only baseline using the original Qwen causal LM head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from evaluation import blind_runner
from evaluation.isolation import EXTERNAL_DATA_ROOT


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "evaluation" / "protocols" / "original-qwen3-label-likelihood-v1.json"
DEFAULT_OUTPUT = EXTERNAL_DATA_ROOT / "results" / "original-qwen-label-likelihood-2026-10-04"
_STOP_REQUESTED = False
_CHOICE_BIN_NAMES = ("2-4", "5-8", "9-16", "17+")
_TOKEN_BIN_NAMES = ("<=256", "257-512", "513-1024", "1025-2048", ">2048")
_SCORE_SUM_KEYS = blind_runner._SCORE_SUM_KEYS


class OriginalEvaluationError(RuntimeError):
    """An evaluation failure that carries only a fixed, safe code."""

    def __init__(self, safe_code: str):
        super().__init__(safe_code)
        self.safe_code = safe_code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _now() -> str:
    return blind_runner._now()


def _safe_component_path(path: Path, root: Path) -> Path:
    resolved_root = root.resolve()
    if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
        raise OriginalEvaluationError("evaluation_path_invalid")
    return path


def _read_json(path: Path, safe_code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise OriginalEvaluationError(safe_code) from None
    if not isinstance(value, dict):
        raise OriginalEvaluationError(safe_code)
    return value


def _write_json(path: Path, value: Mapping[str, Any], *, private: bool = True) -> None:
    blind_runner._atomic_json(path, value, private=private)


def _load_protocol(path: Path = PROTOCOL_PATH) -> tuple[dict[str, Any], str]:
    _safe_component_path(path, ROOT)
    protocol = _read_json(path, "protocol_invalid")
    expected = {
        "schema_version": "original-lm-blind-protocol-v1",
        "protocol_id": "original-qwen3-4b-label-likelihood-2026-10-v1",
        "canonical_item_count": 17874,
        "canonical_items_path": "data/external_evaluation/canonical/bundle/items.jsonl",
        "canonical_items_sha256": "3e21e20a79831a99508c7ed8d4f471cf40bb24c4b9c1259e776b217986155fe4",
        "system_test_lock_path": "data/external_evaluation/system-test-lock-current-corpus-2026-10-01.json",
        "system_test_lock_sha256": "23d7e9a733cf6662da37ebaece9feb8836026dc96209d3e67ad647ffd294b705",
        "bundle_manifest_path": "data/external_evaluation/bundle-manifest.json",
        "bundle_manifest_sha256": "023e948c34198c8fc5231e8109a5ca4f0d8f263d10871328ad3809665eb30343",
        "model_id": "Qwen/Qwen3-4B",
        "model_revision": "1cfa9a7208912126459214e8b04321603b3df60c",
        "candidate": "unedited_original_lm_head",
    }
    if any(protocol.get(key) != value for key, value in expected.items()):
        raise OriginalEvaluationError("protocol_identity_mismatch")
    if protocol.get("prompt", {}).get("enable_thinking") is not False:
        raise OriginalEvaluationError("thinking_must_be_disabled")
    if protocol.get("inference", {}).get("generated_tokens") != 0:
        raise OriginalEvaluationError("text_generation_forbidden")
    if protocol.get("scoring", {}).get("temperature_scaling") is not False:
        raise OriginalEvaluationError("calibration_forbidden")
    diagnostics = protocol.get("scoring", {}).get("exploratory_diagnostics", {})
    if (
        diagnostics.get("choice_count_bins") != list(_CHOICE_BIN_NAMES)
        or diagnostics.get("input_token_bins") != list(_TOKEN_BIN_NAMES)
        or diagnostics.get("cross_tabs")
        != "source_slice x choice_count_bin x input_token_bin; source_slice x exact_choice_count x input_token_bin"
    ):
        raise OriginalEvaluationError("diagnostic_protocol_mismatch")
    return protocol, _sha256(path)


def _choice_bin(choice_count: int) -> str:
    if 2 <= choice_count <= 4:
        return "2-4"
    if choice_count <= 8:
        return "5-8"
    if choice_count <= 16:
        return "9-16"
    if choice_count >= 17:
        return "17+"
    raise OriginalEvaluationError("choice_count_invalid")


def _token_bin(token_count: int) -> str:
    if token_count <= 256:
        return "<=256"
    if token_count <= 512:
        return "257-512"
    if token_count <= 1024:
        return "513-1024"
    if token_count <= 2048:
        return "1025-2048"
    return ">2048"


def _user_text(model_row: Mapping[str, Any], labels: list[str]) -> str:
    if set(model_row) != {"id", "context", "question", "options"}:
        raise OriginalEvaluationError("model_input_field_allowlist_failed")
    context = model_row["context"]
    question = model_row["question"]
    options = model_row["options"]
    if not isinstance(context, str) or not isinstance(question, str) or not isinstance(options, list):
        raise OriginalEvaluationError("model_input_contract_invalid")
    if len(options) != len(labels):
        raise OriginalEvaluationError("choice_label_count_invalid")
    choices = "\n".join(f"{label}. {option}" for label, option in zip(labels, options))
    return f"Context:\n{context}\n\nQuestion:\n{question}\n\nOptions:\n{choices}\n\nReply with one option letter only."


def _chat_prompt(tokenizer: Any, model_row: Mapping[str, Any], protocol: Mapping[str, Any]) -> str:
    prompt_spec = protocol["prompt"]
    labels = [chr(ord("A") + index) for index in range(len(model_row["options"]))]
    messages = [
        {"role": "system", "content": prompt_spec["system_message"]},
        {"role": "user", "content": _user_text(model_row, labels)},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        raise OriginalEvaluationError("no_thinking_chat_template_unavailable") from None
    if not isinstance(prompt, str) or not prompt:
        raise OriginalEvaluationError("chat_template_return_invalid")
    return prompt


def _as_id_list(value: Any) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        raise OriginalEvaluationError("tokenizer_contract_invalid")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise OriginalEvaluationError("tokenizer_contract_invalid")
        value = value[0]
    if any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise OriginalEvaluationError("tokenizer_contract_invalid")
    return value


def _encode_batch(tokenizer: Any, texts: list[str]) -> list[list[int]]:
    try:
        encoded = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)
        rows = encoded["input_ids"]
    except Exception:
        raise OriginalEvaluationError("tokenizer_encoding_failed") from None
    if hasattr(rows, "tolist"):
        rows = rows.tolist()
    if not isinstance(rows, list) or len(rows) != len(texts):
        raise OriginalEvaluationError("tokenizer_contract_invalid")
    normalized = []
    for row in rows:
        normalized.append(_as_id_list(row))
    return normalized


def _prompt_and_label_ids(tokenizer: Any, prompt: str, choice_count: int) -> tuple[list[int], list[int]]:
    if not 2 <= choice_count <= 26:
        raise OriginalEvaluationError("choice_label_capacity_exceeded")
    labels = [chr(ord("A") + index) for index in range(choice_count)]
    encoded = _encode_batch(tokenizer, [prompt, *(prompt + label for label in labels)])
    base = encoded[0]
    label_ids = []
    for extension in encoded[1:]:
        if len(extension) != len(base) + 1 or extension[:len(base)] != base:
            raise OriginalEvaluationError("choice_label_not_single_prefix_stable_token")
        label_ids.append(extension[-1])
    if len(set(label_ids)) != choice_count:
        raise OriginalEvaluationError("choice_label_token_ids_not_unique")
    return base, label_ids


def _restricted_choice_logits(vocabulary_logits: Any, label_ids: list[int], choice_count: int) -> list[float]:
    """Keep only the original LM-head logits for the valid single-token labels."""
    if not 2 <= choice_count <= len(label_ids):
        raise OriginalEvaluationError("choice_label_count_invalid")
    try:
        if hasattr(vocabulary_logits, "index_select"):
            import torch
            indices = torch.tensor(label_ids[:choice_count], dtype=torch.long, device=vocabulary_logits.device)
            selected = vocabulary_logits.index_select(0, indices)
        else:
            selected = [vocabulary_logits[index] for index in label_ids[:choice_count]]
        if hasattr(selected, "detach"):
            selected = selected.detach().float().cpu().tolist()
        values = [float(value) for value in selected]
    except Exception:
        raise OriginalEvaluationError("model_output_invalid") from None
    if len(values) != choice_count or any(not math.isfinite(value) for value in values):
        raise OriginalEvaluationError("model_output_invalid")
    return values


def _tokenizer_hashes(tokenizer: Any) -> dict[str, str]:
    try:
        backend = tokenizer.backend_tokenizer.to_str().encode("utf-8")
        template = tokenizer.chat_template
        if not isinstance(template, str) or not template:
            raise ValueError
        template_bytes = template.encode("utf-8")
    except Exception:
        raise OriginalEvaluationError("tokenizer_artifact_fingerprint_failed") from None
    return {
        "backend_sha256": hashlib.sha256(backend).hexdigest(),
        "chat_template_sha256": hashlib.sha256(template_bytes).hexdigest(),
    }


def _load_runtime(protocol: Mapping[str, Any]):
    try:
        import torch
        import transformers
        from transformers import AutoTokenizer
    except Exception:
        raise OriginalEvaluationError("pinned_runtime_import_failed") from None
    expected = protocol["runtime"]
    if torch.__version__ != expected["torch_version"] or transformers.__version__ != expected["transformers_version"]:
        raise OriginalEvaluationError("pinned_runtime_version_mismatch")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            protocol["model_id"],
            revision=protocol["model_revision"],
            local_files_only=True,
            trust_remote_code=False,
        )
    except Exception:
        raise OriginalEvaluationError("pinned_base_tokenizer_load_failed") from None
    return torch, transformers, tokenizer


def _preflight_inputs(
    items_path: Path,
    tokenizer: Any,
    protocol: Mapping[str, Any],
    *,
    batch_rows: int = 64,
) -> dict[str, Any]:
    from evaluation.blind_runner import _group_name, _iter_items, _validate_item

    expected_groups = protocol["expected_group_counts"]
    expected_count = protocol["canonical_item_count"]
    max_tokens = protocol["inference"]["max_input_tokens"]
    counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    rows = 0
    over_limit = 0
    maximum = 0
    label_contexts_checked = 0
    batch: list[tuple[dict[str, Any], str, int, str]] = []

    def consume(records: list[tuple[dict[str, Any], str, int, str]]) -> None:
        nonlocal over_limit, maximum, label_contexts_checked
        encoded_texts: list[str] = []
        layout: list[tuple[int, int, list[str]]] = []
        for model_row, prompt, _answer, _group in records:
            options = model_row["options"]
            if len(options) > 26:
                raise OriginalEvaluationError("choice_label_capacity_exceeded")
            labels = [chr(ord("A") + index) for index in range(len(options))]
            start = len(encoded_texts)
            encoded_texts.extend([prompt, *(prompt + label for label in labels)])
            layout.append((start, len(options), labels))
        encoded = _encode_batch(tokenizer, encoded_texts)
        for record, (start, option_count, _labels) in zip(records, layout):
            base = encoded[start]
            if len(base) < 1:
                raise OriginalEvaluationError("tokenizer_returned_empty_input")
            maximum = max(maximum, len(base))
            over_limit += len(base) > max_tokens
            label_ids = []
            for extension in encoded[start + 1:start + option_count + 1]:
                if len(extension) != len(base) + 1 or extension[:len(base)] != base:
                    raise OriginalEvaluationError("choice_label_not_single_prefix_stable_token")
                label_ids.append(extension[-1])
            if len(set(label_ids)) != option_count:
                raise OriginalEvaluationError("choice_label_token_ids_not_unique")
            label_contexts_checked += 1

    for raw in _iter_items(items_path):
        item_id, model_row, answer, group = _validate_item(raw, expected_groups)
        if item_id in seen_ids:
            raise OriginalEvaluationError("canonical_item_identity_duplicate")
        seen_ids.add(item_id)
        rows += 1
        counts[group] = counts.get(group, 0) + 1
        prompt = _chat_prompt(tokenizer, model_row, protocol)
        batch.append((model_row, prompt, answer, group))
        if len(batch) >= batch_rows:
            consume(batch)
            batch.clear()
    if batch:
        consume(batch)
    if rows != expected_count or counts != dict(expected_groups):
        raise OriginalEvaluationError("canonical_item_aggregate_contract_invalid")
    return {
        "items_checked": rows,
        "groups_checked": len(counts),
        "max_input_tokens": maximum,
        "over_limit_items": over_limit,
        "label_contexts_checked": label_contexts_checked,
    }


def _load_lock_artifacts(protocol_path: Path, protocol: Mapping[str, Any], protocol_sha: str) -> tuple[dict[str, Any], Path]:
    system_lock, items_path = blind_runner._validate_system_bundle(protocol)
    blind_runner._verify_exclusion_rebind(protocol, system_lock)
    return system_lock, items_path


def _code_hashes(protocol_path: Path) -> dict[str, str]:
    paths = (
        Path("evaluation/original_lm_runner.py"),
        protocol_path.relative_to(ROOT),
        Path("evaluation/blind_runner.py"),
        Path("evaluation/isolation.py"),
    )
    hashes = {}
    for relative in paths:
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise OriginalEvaluationError("evaluation_code_artifact_missing")
        hashes[relative.as_posix()] = _sha256(path)
    return hashes


def _make_or_validate_lock(
    output: Path,
    protocol_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    system_lock: Mapping[str, Any],
    tokenizer: Any,
    torch: Any,
    transformers: Any,
) -> tuple[Path, dict[str, Any], str]:
    if not output.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
        raise OriginalEvaluationError("evaluation_output_outside_external_root")
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "blind-evaluation-lock.json"
    proposed = {
        "schema_version": "original-lm-blind-evaluation-lock-v1",
        "status": "locked",
        "locked_at_utc": _now(),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "system_test_lock_sha256": protocol["system_test_lock_sha256"],
        "bundle_manifest_sha256": protocol["bundle_manifest_sha256"],
        "canonical_items_sha256": protocol["canonical_items_sha256"],
        "training_snapshot_manifest_sha256": protocol["training_snapshot_manifest_sha256"],
        "model_id": protocol["model_id"],
        "model_revision": protocol["model_revision"],
        "candidate": protocol["candidate"],
        "model_artifact_binding": "immutable Hugging Face commit revision; local_files_only enforced",
        "tokenizer_artifact_hashes": _tokenizer_hashes(tokenizer),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "evaluation_code_sha256": _code_hashes(protocol_path),
        "prompt": protocol["prompt"],
        "inference": protocol["inference"],
        "scoring": protocol["scoring"],
        "expected_group_counts": protocol["expected_group_counts"],
        "suite_reuse": protocol["suite_reuse"],
    }
    if lock_path.exists():
        existing = _read_json(lock_path, "evaluation_lock_invalid")
        comparable = dict(existing)
        comparable.pop("locked_at_utc", None)
        intended = dict(proposed)
        intended.pop("locked_at_utc", None)
        if comparable != intended:
            raise OriginalEvaluationError("existing_evaluation_lock_does_not_match_frozen_protocol")
        locked = existing
    else:
        _write_json(lock_path, proposed)
        locked = proposed
    lock_sha = _sha256(lock_path)
    if system_lock.get("status") != "locked":
        raise OriginalEvaluationError("system_test_lock_not_locked")
    return lock_path, locked, lock_sha


def _empty_stats() -> dict[str, Any]:
    return {"examples": 0, "correct": 0, **{key: 0.0 for key in _SCORE_SUM_KEYS}}


def _empty_cursor(protocol_sha: str, lock_sha: str, item_count: int) -> dict[str, Any]:
    return {
        "schema_version": "original-lm-blind-cursor-v1",
        "protocol_sha256": protocol_sha,
        "evaluation_lock_sha256": lock_sha,
        "candidate": "unedited_original_lm_head",
        "item_count": item_count,
        "next_item_index": 0,
        "inflight_item_index": None,
        "status": "running",
        "updated_at_utc": _now(),
        "aggregate": _empty_stats(),
        "by_source_slice": {},
        "by_choice_count_bin": {},
        "by_input_token_bin": {},
        "by_source_choice_token_bins": {},
        "by_source_exact_choice_token_bins": {},
    }


def _write_cursor(path: Path, state: Mapping[str, Any]) -> None:
    value = dict(state)
    value.pop("cursor_sha256", None)
    value["cursor_sha256"] = blind_runner._mapping_sha256(value)
    _write_json(path, value)


def _validate_cursor(state: Mapping[str, Any], protocol_sha: str, lock_sha: str, count: int) -> None:
    payload = dict(state)
    checksum = payload.pop("cursor_sha256", None)
    if checksum != blind_runner._mapping_sha256(payload):
        raise OriginalEvaluationError("cursor_checksum_invalid")
    if (
        state.get("schema_version") != "original-lm-blind-cursor-v1"
        or state.get("protocol_sha256") != protocol_sha
        or state.get("evaluation_lock_sha256") != lock_sha
        or state.get("candidate") != "unedited_original_lm_head"
        or state.get("item_count") != count
        or state.get("status") not in {"paused", "running", "complete"}
        or not isinstance(state.get("next_item_index"), int)
        or not 0 <= state["next_item_index"] <= count
        or state.get("inflight_item_index") is not None
    ):
        raise OriginalEvaluationError("cursor_contract_invalid_or_inflight")


def _add_stratum(container: dict[str, Any], key: str, outcome: Mapping[str, Any], correct: bool) -> None:
    stats = container.setdefault(key, _empty_stats())
    blind_runner._add_outcome(stats, outcome, correct)


def _add_nested(container: dict[str, Any], first: str, second: str, third: str, outcome: Mapping[str, Any], correct: bool) -> None:
    child = container.setdefault(first, {}).setdefault(second, {})
    stats = child.setdefault(third, _empty_stats())
    blind_runner._add_outcome(stats, outcome, correct)


def _process_items(
    items_path: Path,
    state_path: Path,
    status_path: Path,
    protocol: Mapping[str, Any],
    protocol_sha: str,
    lock_sha: str,
    tokenizer: Any,
    model: Any,
    torch: Any,
    *,
    resume: bool,
    stop_after_items: int | None = None,
) -> dict[str, Any]:
    from evaluation.blind_runner import _group_name, _iter_items, _validate_item

    expected_count = protocol["canonical_item_count"]
    expected_groups = protocol["expected_group_counts"]
    if state_path.exists():
        if not resume:
            raise OriginalEvaluationError("cursor_exists_requires_resume")
        state = _read_json(state_path, "cursor_invalid")
        _validate_cursor(state, protocol_sha, lock_sha, expected_count)
        if state["status"] == "complete":
            return state
        if state["status"] not in {"paused", "running"}:
            raise OriginalEvaluationError("cursor_not_safely_resumable")
    else:
        if resume:
            raise OriginalEvaluationError("cursor_missing_for_resume")
        state = _empty_cursor(protocol_sha, lock_sha, expected_count)
    state["status"] = "running"
    _write_cursor(state_path, state)

    seen_ids: set[str] = set()
    group_counts: dict[str, int] = {}
    seen_rows = 0
    for raw in _iter_items(items_path):
        item_id, model_row, answer, group = _validate_item(raw, expected_groups)
        if item_id in seen_ids:
            raise OriginalEvaluationError("canonical_item_identity_duplicate")
        seen_ids.add(item_id)
        group_counts[group] = group_counts.get(group, 0) + 1
        ordinal = seen_rows
        seen_rows += 1
        if ordinal < state["next_item_index"]:
            continue
        if ordinal != state["next_item_index"]:
            raise OriginalEvaluationError("cursor_order_invalid")
        if _STOP_REQUESTED:
            state["status"] = "paused"
            state["updated_at_utc"] = _now()
            _write_cursor(state_path, state)
            blind_runner._progress_status(
                status_path, phase="paused", arm="original", processed=ordinal, total=expected_count,
            )
            return state

        prompt = _chat_prompt(tokenizer, model_row, protocol)
        input_ids, label_ids = _prompt_and_label_ids(tokenizer, prompt, len(model_row["options"]))
        token_count = len(input_ids)
        if not input_ids or token_count > protocol["inference"]["max_input_tokens"]:
            raise OriginalEvaluationError("complete_input_exceeds_locked_token_limit")
        choice_bin = _choice_bin(len(model_row["options"]))
        token_bin = _token_bin(token_count)
        state["inflight_item_index"] = ordinal
        state["updated_at_utc"] = _now()
        _write_cursor(state_path, state)
        try:
            ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
            mask = torch.ones_like(ids)
            with torch.inference_mode():
                output = model(input_ids=ids, attention_mask=mask, use_cache=False)
            vocabulary_logits = output.logits[0, -1]
            logits = _restricted_choice_logits(vocabulary_logits, label_ids, len(model_row["options"]))
        except OriginalEvaluationError:
            raise
        except Exception:
            raise OriginalEvaluationError("model_inference_failed") from None

        outcome = blind_runner._score_outcome(logits, answer, len(model_row["options"]))
        correct = outcome["prediction"] == answer
        blind_runner._add_outcome(state["aggregate"], outcome, correct)
        _add_stratum(state["by_source_slice"], group, outcome, correct)
        _add_stratum(state["by_choice_count_bin"], choice_bin, outcome, correct)
        _add_stratum(state["by_input_token_bin"], token_bin, outcome, correct)
        _add_nested(state["by_source_choice_token_bins"], group, choice_bin, token_bin, outcome, correct)
        _add_nested(state["by_source_exact_choice_token_bins"], group, str(len(model_row["options"])), token_bin, outcome, correct)
        state["next_item_index"] = ordinal + 1
        state["inflight_item_index"] = None
        state["status"] = "running"
        state["updated_at_utc"] = _now()
        _write_cursor(state_path, state)
        blind_runner._progress_status(
            status_path, phase="candidate_running", arm="original",
            processed=state["next_item_index"], total=expected_count,
        )
        if _STOP_REQUESTED or (stop_after_items is not None and state["next_item_index"] >= stop_after_items):
            state["status"] = "paused"
            state["updated_at_utc"] = _now()
            _write_cursor(state_path, state)
            blind_runner._progress_status(
                status_path, phase="paused", arm="original",
                processed=state["next_item_index"], total=expected_count,
            )
            return state

    if seen_rows != expected_count or state["next_item_index"] != expected_count:
        raise OriginalEvaluationError("canonical_item_count_changed_during_inference")
    if group_counts != dict(expected_groups):
        raise OriginalEvaluationError("canonical_group_counts_changed_during_inference")
    state["status"] = "complete"
    state["updated_at_utc"] = _now()
    _write_cursor(state_path, state)
    return state


def _one_metrics(stats: Mapping[str, Any]) -> dict[str, Any]:
    examples = stats.get("examples")
    if not isinstance(examples, int) or examples < 1:
        raise OriginalEvaluationError("aggregate_stratum_invalid")
    blind_runner._validate_sufficient_stats(stats, examples, "aggregate_stratum_invalid")
    return {
        "examples": examples,
        "correct": stats["correct"],
        "accuracy": stats["correct"] / examples,
        "negative_log_likelihood": stats["negative_log_likelihood_sum"] / examples,
        "multiclass_brier_score": stats["multiclass_brier_score_sum"] / examples,
        "uniform_baseline": {
            "accuracy": stats["uniform_accuracy_sum"] / examples,
            "negative_log_likelihood": stats["uniform_negative_log_likelihood_sum"] / examples,
            "multiclass_brier_score": stats["uniform_multiclass_brier_score_sum"] / examples,
        },
    }


def _metric_map(stats: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _one_metrics(value) for key, value in sorted(stats.items()) if value["examples"]}


def _metric_nested(stats: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for first, second_map in sorted(stats.items()):
        result[first] = {}
        for second, third_map in sorted(second_map.items()):
            metrics = _metric_map(third_map)
            if metrics:
                result[first][second] = metrics
        if not result[first]:
            result.pop(first)
    return result


def _aggregate_result(state: Mapping[str, Any], protocol: Mapping[str, Any], protocol_sha: str, lock_sha: str) -> dict[str, Any]:
    if state.get("status") != "complete" or state["aggregate"]["examples"] != protocol["canonical_item_count"]:
        raise OriginalEvaluationError("aggregate_requires_complete_candidate")
    if {key: value["examples"] for key, value in state["by_source_slice"].items()} != protocol["expected_group_counts"]:
        raise OriginalEvaluationError("aggregate_source_slice_coverage_invalid")
    diagnostics = {
        "exploratory_reuse": True,
        "choice_count_bin": _metric_map(state["by_choice_count_bin"]),
        "input_token_bin": _metric_map(state["by_input_token_bin"]),
        "source_choice_count_bin_x_input_token_bin": _metric_nested(state["by_source_choice_token_bins"]),
        "source_exact_choice_count_x_input_token_bin": _metric_nested(state["by_source_exact_choice_token_bins"]),
        "interpretation_note": "Post hoc descriptive strata on a reused test suite; not used for model, prompt, checkpoint, or training selection.",
    }
    return {
        "schema_version": "original-lm-blind-aggregate-result-v1",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "evaluation_lock_sha256": lock_sha,
        "system_test_lock_sha256": protocol["system_test_lock_sha256"],
        "evaluated_at_utc": _now(),
        "candidate": "unedited_original_lm_head",
        "model_id": protocol["model_id"],
        "model_revision": protocol["model_revision"],
        "overall": _one_metrics(state["aggregate"]),
        "by_source_slice": _metric_map(state["by_source_slice"]),
        "exploratory_diagnostics": diagnostics,
        "comparison_limit": protocol["comparison_limit"],
        "suite_reuse": protocol["suite_reuse"],
    }


def _make_or_validate_manifest(run_directory: Path, protocol: Mapping[str, Any], protocol_sha: str, lock_sha: str) -> Path:
    if not run_directory.resolve().is_relative_to((ROOT / "runs").resolve()):
        raise OriginalEvaluationError("run_manifest_outside_runs_root")
    run_directory.mkdir(parents=True, exist_ok=True)
    path = run_directory / "manifest.json"
    manifest = {
        "schema_version": "original-lm-blind-run-manifest-v1",
        "status": "locked",
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "evaluation_lock_sha256": lock_sha,
        "canonical_item_count": protocol["canonical_item_count"],
        "model_id": protocol["model_id"],
        "model_revision": protocol["model_revision"],
        "candidate": "unedited_original_lm_head",
        "output_directory": DEFAULT_OUTPUT.relative_to(ROOT).as_posix(),
    }
    if path.exists():
        if _read_json(path, "run_manifest_invalid") != manifest:
            raise OriginalEvaluationError("run_manifest_does_not_match_locked_protocol")
    else:
        _write_json(path, manifest)
    return path


def _load_original_model(torch: Any, transformers: Any, protocol: Mapping[str, Any]):
    if not torch.cuda.is_available():
        raise OriginalEvaluationError("locked_accelerator_unavailable")
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        cap_bytes = int(protocol["inference"]["gpu_allocation_cap_gib"] * 1024**3)
        if total_bytes <= cap_bytes:
            raise OriginalEvaluationError("locked_gpu_memory_cap_invalid")
        torch.cuda.set_per_process_memory_fraction(cap_bytes / total_bytes)
        torch.set_num_threads(protocol["inference"]["torch_num_threads"])
        from paired_training import require_memory
        require_memory(int(protocol["inference"]["minimum_model_memory_gib"] * 1024**3),
                       protocol["inference"]["reserve_memory_gib"], "cuda")
    except OriginalEvaluationError:
        raise
    except Exception:
        raise OriginalEvaluationError("locked_gpu_memory_guard_failed") from None
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            protocol["model_id"],
            revision=protocol["model_revision"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map="cuda",
        )
        if model.get_output_embeddings() is None:
            raise OriginalEvaluationError("original_lm_output_head_missing")
        model.eval()
        model.requires_grad_(False)
        return model, {"gpu_free_gib_before_load": round(free_bytes / 1024**3, 3),
                       "gpu_total_gib": round(total_bytes / 1024**3, 3)}
    except OriginalEvaluationError:
        raise
    except Exception:
        raise OriginalEvaluationError("pinned_original_model_load_failed") from None


def run_evaluation(
    *,
    output_directory: Path = DEFAULT_OUTPUT,
    run_directory: Path = ROOT / "runs" / "original-qwen-label-likelihood-2026-10-04",
    protocol_path: Path = PROTOCOL_PATH,
    resume: bool = False,
    stop_after_items: int | None = None,
    prepare_only: bool = False,
) -> dict[str, Any]:
    def rooted(path: Path) -> Path:
        path = Path(path)
        return path if path.is_absolute() else ROOT / path

    output_directory = rooted(output_directory)
    run_directory = rooted(run_directory)
    protocol_path = rooted(protocol_path)
    protocol, protocol_sha = _load_protocol(protocol_path)
    system_lock, items_path = _load_lock_artifacts(protocol_path, protocol, protocol_sha)
    torch, transformers, tokenizer = _load_runtime(protocol)
    lock_path, _lock, lock_sha = _make_or_validate_lock(
        output_directory, protocol_path, protocol, protocol_sha, system_lock, tokenizer, torch, transformers,
    )
    manifest_path = _make_or_validate_manifest(run_directory, protocol, protocol_sha, lock_sha)
    status_path = output_directory / "run-status.json"
    base = {
        "protocol_sha256": protocol_sha,
        "evaluation_lock_sha256": lock_sha,
        "canonical_item_count": protocol["canonical_item_count"],
        "manifest_path": manifest_path.relative_to(ROOT).as_posix(),
    }

    preflight = _preflight_inputs(items_path, tokenizer, protocol)
    if preflight["over_limit_items"]:
        raise OriginalEvaluationError("complete_input_exceeds_locked_token_limit")
    if prepare_only:
        blind_runner._progress_status(
            status_path, phase="prepared", processed=preflight["items_checked"],
            total=protocol["canonical_item_count"], max_input_tokens=preflight["max_input_tokens"],
        )
        return {"phase": "prepared", "max_input_tokens": preflight["max_input_tokens"],
                "label_contexts_checked": preflight["label_contexts_checked"], **base}

    result_path = output_directory / "blind-aggregate-result.json"
    if result_path.exists():
        if resume:
            return {"phase": "complete", "result_path": result_path.relative_to(ROOT).as_posix(), **base}
        raise OriginalEvaluationError("final_external_result_already_exists")
    state_path = output_directory / "cursor-original.json"
    if resume and not state_path.is_file():
        raise OriginalEvaluationError("cursor_missing_for_resume")
    if state_path.exists() and not resume:
        raise OriginalEvaluationError("cursor_exists_requires_resume")

    _progress = blind_runner._progress_status
    _progress(status_path, phase="input_preflight_complete", processed=preflight["items_checked"],
              total=protocol["canonical_item_count"], max_input_tokens=preflight["max_input_tokens"])
    model, resources = _load_original_model(torch, transformers, protocol)
    _write_json(run_directory / "runtime-metadata.json", {
        "phase": "model_loaded",
        "updated_at_utc": _now(),
        **resources,
        "host_pid": os.getpid(),
    })
    state = _process_items(
        items_path, state_path, status_path, protocol, protocol_sha, lock_sha,
        tokenizer, model, torch, resume=resume, stop_after_items=stop_after_items,
    )
    if state["status"] == "paused":
        return {
            "phase": "paused",
            "processed_items": state["next_item_index"],
            "total_items": state["item_count"],
            **base,
        }
    result = _aggregate_result(state, protocol, protocol_sha, lock_sha)
    if result_path.exists():
        raise OriginalEvaluationError("final_external_result_already_exists")
    _write_json(result_path, result)
    _progress(status_path, phase="complete", processed=protocol["canonical_item_count"],
              total=protocol["canonical_item_count"])
    return {"phase": "complete", "processed_items": protocol["canonical_item_count"],
            "result_path": result_path.relative_to(ROOT).as_posix(), **base}


def _request_stop(_signal: int, _frame: Any) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen original-Qwen blind label-likelihood evaluation")
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-directory", type=Path, default=ROOT / "runs" / "original-qwen-label-likelihood-2026-10-04")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-items", type=int)
    return parser.parse_args()


def main() -> int:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    args = _parse_args()
    try:
        if args.stop_after_items is not None and args.stop_after_items < 1:
            raise OriginalEvaluationError("stop_after_items_invalid")
        result = run_evaluation(
            output_directory=args.output_directory,
            run_directory=args.run_directory,
            protocol_path=args.protocol,
            resume=args.resume,
            stop_after_items=args.stop_after_items,
            prepare_only=args.prepare_only,
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except OriginalEvaluationError as error:
        try:
            protocol, _ = _load_protocol(args.protocol)
            output = args.output_directory if args.output_directory.is_absolute() else ROOT / args.output_directory
            if output.resolve().is_relative_to(EXTERNAL_DATA_ROOT.resolve()):
                blind_runner._progress_status(output / "run-status.json", phase="failed", safe_error_code=error.safe_code)
        except Exception:
            pass
        print(json.dumps({"phase": "failed", "safe_error_code": error.safe_code}, sort_keys=True), flush=True)
        return 2
    except Exception:
        print(json.dumps({"phase": "failed", "safe_error_code": "unexpected_safe_failure"}, sort_keys=True), flush=True)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
