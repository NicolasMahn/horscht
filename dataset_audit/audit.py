#!/usr/bin/env python3
"""Freeze and inspect a stratified sample of accepted training questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_factory import probability_v2


MODEL_ID = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
SAMPLE_SIZE = 8_000
SAMPLE_SEED = 20260929
TSNE_RUNS = ((30, 42), (20, 13))
OVERLAP_TOKENS = 64
BATCH_SIZE = 16
CPU_THREADS = 2
EMBEDDING_PIPELINE_REVISION = "masked-mean-overlap-v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def hash_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def source_files(root: Path) -> list[Path]:
    broad = sorted((root / "data/generated_train/broad-v1").glob("*.jsonl"))
    v2 = sorted((root / "data/generated_train/probability-v2/questions").rglob("shard-*.jsonl"))
    return [path for path in broad + v2 if path.is_file() and path.suffix == ".jsonl"]


def _field(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        for key in ("id", "tag", "name", "value"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _generator_metadata(row: dict[str, Any]) -> dict[str, Any]:
    for key in ("_generator", "generation_metadata"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return {}


def metadata_for(row: dict[str, Any], path: str) -> dict[str, str]:
    generator = _generator_metadata(row)
    operation = next(
        (_field(row.get(key)) for key in ("operation", "operation_id", "family", "task_id") if _field(row.get(key))),
        "not_recorded",
    )
    domain = next(
        (_field(row.get(key)) for key in ("domain", "domain_id", "topic") if _field(row.get(key))),
        "not_recorded",
    )
    recorded_batch = next(
        (_field(source.get(key)) for source in (generator, row) for key in ("batch_id", "batch", "shard_id") if _field(source.get(key))),
        None,
    )
    batch = f"metadata:{recorded_batch}" if recorded_batch else f"file:{path.removesuffix('.jsonl')}"
    language = _field(row.get("language")) or _field(row.get("language_profile")) or "not_recorded"
    if row.get("schema_version") == probability_v2.SCHEMA_VERSION:
        source_kind = _field(row.get("distribution_source_kind")) or _field(row.get("supervision_kind")) or "unknown"
    else:
        source_kind = "hard_label"
    return {
        "language": language,
        "source_kind": source_kind,
        "operation": operation,
        "domain": domain,
        "batch": batch,
    }


def render_embedding_input(row: dict[str, Any]) -> str:
    """Allowlist only context, question and ordered answer-choice text."""
    context = row["context"]
    question = row["question"]
    options = row["options"]
    if not isinstance(context, str) or not isinstance(question, str):
        raise ValueError("accepted row is missing text fields")
    if not isinstance(options, list) or not all(isinstance(option, str) for option in options):
        raise ValueError("accepted row has invalid choices")
    choice_lines = "\n".join(f"- {option}" for option in options)
    return f"query: {context}\nQuestion: {question}\nOptions:\n{choice_lines}"


def _accepted_snapshot(root: Path, destination: Path) -> dict[str, Any]:
    started = utc_now()
    files = source_files(root)
    if not files:
        raise RuntimeError("no accepted-production source files found")

    valid_rows: list[dict[str, Any]] = []
    group_splits: dict[str, set[str]] = defaultdict(set)
    source_manifest: list[dict[str, Any]] = []
    malformed = 0
    incomplete_final_lines = 0
    rejected = Counter()
    seen_ids: set[str] = set()
    duplicate_ids = 0
    content_counts: Counter[str] = Counter()
    first_content_row: dict[str, dict[str, Any]] = {}
    duplicate_content_examples: list[dict[str, Any]] = []
    question_counts: Counter[str] = Counter()

    for file_index, path in enumerate(files, 1):
        data = path.read_bytes()
        relative = path.relative_to(root).as_posix()
        lines = data.splitlines()
        if data and not data.endswith((b"\n", b"\r")):
            lines = lines[:-1]
            incomplete_final_lines += 1
        source_manifest.append({
            "path": relative,
            "sha256": hash_bytes(data),
            "size_bytes": len(data),
            "line_count": len(lines),
        })
        for line_number, raw_line in enumerate(lines, 1):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except (UnicodeDecodeError, json.JSONDecodeError):
                malformed += 1
                continue
            if isinstance(row, dict):
                group = row.get("problem_group_id")
                split = row.get("split")
                if isinstance(group, str) and group.strip() and isinstance(split, str) and split.strip():
                    group_splits[group.strip()].add(split.strip())

            source, errors, schema = probability_v2.classify_row(row)
            if errors or source is None or not isinstance(row, dict):
                rejected[errors[0] if errors else "invalid_row"] += 1
                continue

            identifier = row["id"].strip()
            if identifier in seen_ids:
                duplicate_ids += 1
                continue
            seen_ids.add(identifier)
            meta = metadata_for(row, relative)
            text = render_embedding_input(row)
            canonical_content = json.dumps(
                [normalize(row["context"]), normalize(row["question"]), [normalize(x) for x in row["options"]]],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            content_key = hash_bytes(canonical_content.encode("utf-8"))
            content_counts[content_key] += 1
            safe_content = {
                "id": identifier,
                **meta,
                "context": row["context"],
                "question": row["question"],
                "options": row["options"],
                "source_file": relative,
                "source_line": line_number,
            }
            if content_key not in first_content_row:
                first_content_row[content_key] = safe_content
            else:
                duplicate_content_examples.append({
                    "content_sha256": content_key,
                    "first": first_content_row[content_key],
                    "duplicate": safe_content,
                })
            question_counts[normalize(row["question"])] += 1
            valid_rows.append({
                "id": identifier,
                "problem_group_id": row["problem_group_id"].strip(),
                "split": row["split"].strip(),
                "schema": schema,
                **meta,
                "context": row["context"],
                "question": row["question"],
                "options": row["options"],
                "source_file": relative,
                "source_line": line_number,
                "row_sha256": hash_bytes(raw_line),
                "text_sha256": hash_bytes(text.encode("utf-8")),
            })
        if file_index % 50 == 0:
            print(f"snapshotted {file_index}/{len(files)} source files", flush=True)

    conflicting_groups = {group for group, splits in group_splits.items() if len(splits) > 1}
    accepted = [row for row in valid_rows if row["problem_group_id"] not in conflicting_groups]
    if not accepted:
        raise RuntimeError("accepted snapshot is empty")

    selected = select_sample(accepted, min(SAMPLE_SIZE, len(accepted)), SAMPLE_SEED)
    sample_jsonl = destination / "sample.jsonl"
    sample_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with sample_jsonl.open("w", encoding="utf-8") as stream:
        for row in selected:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    index_path = destination / "accepted-index.jsonl"
    with index_path.open("w", encoding="utf-8") as stream:
        for row in accepted:
            slim = {key: row[key] for key in ("id", "source_file", "source_line", "row_sha256", "language", "source_kind", "operation", "domain", "batch")}
            stream.write(json.dumps(slim, ensure_ascii=False, separators=(",", ":")) + "\n")
    (destination / "full-snapshot-duplicate-examples.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n" for item in duplicate_content_examples),
        encoding="utf-8",
    )

    accepted_sources = Counter(row["source_kind"] for row in accepted)
    accepted_languages = Counter(row["language"] for row in accepted)
    accepted_operations = Counter(row["operation"] for row in accepted)
    duplicate_content_rows = sum(count - 1 for count in content_counts.values() if count > 1)
    duplicate_question_rows = sum(count - 1 for count in question_counts.values() if count > 1)
    duplicate_question_groups = sum(count > 1 for count in question_counts.values())
    sample_source_counts = Counter(row["source_kind"] for row in selected)
    sample_language_counts = Counter(row["language"] for row in selected)
    sample_operation_counts = Counter(row["operation"] for row in selected)
    started_at = started
    finished_at = utc_now()

    fingerprint = hash_bytes(json.dumps(
        [{"path": entry["path"], "sha256": entry["sha256"]} for entry in source_manifest],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8"))
    manifest = {
        "schema_version": "question-diversity-audit-v1",
        "snapshot_started_at": started_at,
        "snapshot_finished_at": finished_at,
        "source_files": source_manifest,
        "source_snapshot_sha256": fingerprint,
        "snapshot_scope": "canonical broad-v1 legacy rows and probability-v2/questions shards; excludes pilot and non-JSONL/partial files",
        "accepted_rows": len(accepted),
        "accepted_by_source_kind": dict(sorted(accepted_sources.items())),
        "accepted_language_count": len(accepted_languages),
        "accepted_by_language": dict(sorted(accepted_languages.items())),
        "accepted_operation_count": len(accepted_operations),
        "accepted_by_operation": dict(sorted(accepted_operations.items())),
        "rejected_schema_rows": sum(rejected.values()),
        "rejection_reasons": dict(sorted(rejected.items())),
        "malformed_lines": malformed,
        "incomplete_final_lines_excluded": incomplete_final_lines,
        "duplicate_ids_skipped": duplicate_ids,
        "problem_group_split_conflicts": len(conflicting_groups),
        "full_snapshot_exact_content_duplicate_rows": duplicate_content_rows,
        "full_snapshot_exact_question_duplicate_rows": duplicate_question_rows,
        "full_snapshot_repeated_exact_question_groups": duplicate_question_groups,
        "sample_size": len(selected),
        "sample_seed": SAMPLE_SEED,
        "stratification": "minimum 4 sampled rows per language where available, then minimum one per observed language/source-kind/operation cell, then population-proportional allocation; inverse cell inclusion weights accompany each sampled row",
        "sample_by_source_kind": dict(sorted(sample_source_counts.items())),
        "sample_by_language": dict(sorted(sample_language_counts.items())),
        "sample_by_operation": dict(sorted(sample_operation_counts.items())),
        "sample_ids_and_source_hashes": [
            {"id": row["id"], "source_file": row["source_file"], "source_line": row["source_line"], "row_sha256": row["row_sha256"], "text_sha256": row["text_sha256"], "sample_weight": row["sample_weight"]}
            for row in selected
        ],
        "progress_file_at_snapshot": None,
        "embedding": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "text_fields": ["context", "question", "ordered options"],
            "excluded_fields": ["answer_index", "explanation", "target_distribution", "distribution_evidence", "provenance", "generator metadata"],
            "prefix": "query: ",
            "max_sequence_tokens": 512,
            "overlap_tokens": OVERLAP_TOKENS,
            "pooling": "masked mean per chunk; mean chunk vectors weighted by new token coverage; final L2 normalization",
        },
    }
    progress_path = root / "data/generated_train/probability-v2/progress.json"
    if progress_path.is_file():
        try:
            progress_data = json.loads(progress_path.read_text(encoding="utf-8"))
            manifest["progress_file_at_snapshot"] = {
                "updated_at": progress_data.get("updated_at"),
                "accepted_production_rows": progress_data.get("accepted_production_rows"),
            }
        except (OSError, json.JSONDecodeError):
            pass
    (destination / "snapshot.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        f"snapshot complete: {len(accepted):,} accepted rows from {len(source_manifest)} files; "
        f"sample={len(selected):,}, languages={len(accepted_languages)}, operations={len(accepted_operations)}",
        flush=True,
    )
    return manifest


def select_sample(rows: list[dict[str, Any]], size: int, seed: int) -> list[dict[str, Any]]:
    """Stratify across languages and observed source-kind/operation cells."""
    if not rows or size <= 0:
        raise ValueError("sample size and population must both be non-empty")
    rng = random.Random(seed)
    by_language: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_cell: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_language[row["language"]].append(row)
        by_cell[(row["language"], row["source_kind"], row["operation"])].append(row)
    selected_ids: set[str] = set()
    selected: list[dict[str, Any]] = []

    def choose(pool: list[dict[str, Any]], count: int) -> None:
        available = [row for row in pool if row["id"] not in selected_ids]
        count = min(count, len(available))
        if count <= 0:
            return
        chosen = rng.sample(available, count)
        selected.extend(chosen)
        selected_ids.update(row["id"] for row in chosen)

    for language in sorted(by_language):
        choose(by_language[language], min(4, len(by_language[language])))
    for key in sorted(by_cell):
        if len(selected) >= size:
            break
        if not any(row["id"] in selected_ids for row in by_cell[key]):
            choose(by_cell[key], 1)
    if len(selected) > size:
        selected.sort(key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest())
        selected = selected[:size]
        selected_ids = {row["id"] for row in selected}

    remaining_slots = size - len(selected)
    remaining = [row for row in rows if row["id"] not in selected_ids]
    if remaining_slots > 0 and remaining:
        cells: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in remaining:
            cells[(row["language"], row["source_kind"], row["operation"])].append(row)
        quota_remaining = remaining_slots
        quota_total = len(remaining)
        allocation: dict[tuple[str, str, str], int] = {}
        remainders: list[tuple[float, tuple[str, str, str]]] = []
        for key, pool in cells.items():
            exact = quota_remaining * len(pool) / quota_total
            allocation[key] = min(len(pool), math.floor(exact))
            remainders.append((exact - math.floor(exact), key))
        slots = quota_remaining - sum(allocation.values())
        for _, key in sorted(remainders, key=lambda value: (-value[0], value[1])):
            if slots <= 0:
                break
            if allocation[key] < len(cells[key]):
                allocation[key] += 1
                slots -= 1
        for key in sorted(cells):
            choose(cells[key], allocation[key])

    counts = Counter((row["language"], row["source_kind"], row["operation"]) for row in rows)
    sampled_counts = Counter((row["language"], row["source_kind"], row["operation"]) for row in selected)
    for row in selected:
        cell = (row["language"], row["source_kind"], row["operation"])
        row["sample_weight"] = counts[cell] / sampled_counts[cell]
        row["sample_cell_population"] = counts[cell]
        row["sample_cell_n"] = sampled_counts[cell]
    return sorted(selected, key=lambda row: (row["language"], row["source_kind"], row["operation"], row["id"]))


def _chunks(tokenizer: Any, text: str, maximum: int, overlap: int) -> tuple[list[list[int]], list[int], int]:
    content_ids = tokenizer.encode(text, add_special_tokens=False)
    content_limit = maximum - tokenizer.num_special_tokens_to_add(pair=False)
    if content_limit <= overlap:
        raise RuntimeError("tokenizer limit is too short for configured overlap")
    start_token = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else tokenizer.bos_token_id
    end_token = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else tokenizer.eos_token_id
    if start_token is None or end_token is None:
        raise RuntimeError("tokenizer does not expose start/end special token IDs")

    def add_special_tokens(window: list[int]) -> list[int]:
        # XLM-R uses <s> and </s>; Transformers 5 removed the old builder method.
        return [start_token, *window, end_token]

    step = content_limit - overlap
    windows: list[list[int]] = []
    weights: list[int] = []
    start = 0
    covered_until = 0
    while start < len(content_ids):
        window = content_ids[start:start + content_limit]
        if not window:
            break
        end = start + len(window)
        windows.append(add_special_tokens(window))
        weights.append(max(1, end - max(start, covered_until)))
        covered_until = max(covered_until, end)
        if end >= len(content_ids):
            break
        start += step
    if not windows:
        windows = [add_special_tokens([])]
        weights = [1]
    if covered_until < len(content_ids):
        raise RuntimeError("token chunking dropped part of an input")
    return windows, weights, len(content_ids)


def embedding_cache_identity(manifest: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pipeline_revision": EMBEDDING_PIPELINE_REVISION,
        "source_snapshot_sha256": manifest["source_snapshot_sha256"],
        "sample_ids_and_row_text_hashes": [[row["id"], row["row_sha256"], row["text_sha256"]] for row in rows],
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "prefix": "query: ",
        "max_sequence_tokens": 512,
        "overlap_tokens": OVERLAP_TOKENS,
        "pooling": "masked-mean per chunk, new-token weighted mean, final L2 normalization",
    }


def embedding_cache_key(identity: dict[str, Any]) -> str:
    return hash_bytes(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def embedding_cache_matches(cache: dict[str, Any], identity: dict[str, Any], rows: int, dimensions: int = 384) -> bool:
    return (
        cache.get("cache_key") == embedding_cache_key(identity)
        and cache.get("rows") == rows
        and cache.get("dimensions") == dimensions
    )


def _cached_embeddings(destination: Path, identity: dict[str, Any], rows: int) -> dict[str, Any] | None:
    cache_path = destination / "embedding-cache.json"
    array_path = destination / "embeddings.npy"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if not embedding_cache_matches(cache, identity, rows):
            return None
        array = np.load(array_path, mmap_mode="r")
        if array.ndim != 2 or array.shape != (rows, 384):
            return None
        return cache
    except (OSError, ValueError, json.JSONDecodeError):
        return None


def analysis_cache_matches(metrics: dict[str, Any], cache_key: str) -> bool:
    return metrics.get("embedding_cache_key") == cache_key


def _embed(destination: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoModel, AutoTokenizer

    snapshot_path = destination / "snapshot.json"
    if not snapshot_path.is_file() or not (destination / "sample.jsonl").is_file():
        raise RuntimeError("run the snapshot stage before embedding")
    manifest = json.loads(snapshot_path.read_text(encoding="utf-8"))
    embedding_meta = manifest["embedding"]
    if embedding_meta["model_id"] != MODEL_ID or embedding_meta["revision"] != MODEL_REVISION:
        raise RuntimeError("model identity does not match the frozen snapshot")

    rows = [json.loads(line) for line in (destination / "sample.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise RuntimeError("sample is empty")
    identity = embedding_cache_identity(manifest, rows)
    cached = _cached_embeddings(destination, identity, len(rows))
    if cached is not None:
        print(f"embedding cache valid: {cached['cache_key'][:12]}", flush=True)
        return cached
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(1)
    local_model = Path.home() / ".cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots" / MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(local_model, local_files_only=True)
    model = AutoModel.from_pretrained(local_model, local_files_only=True)
    model.eval()
    maximum = min(int(model.config.max_position_embeddings), int(tokenizer.model_max_length))
    if maximum != 512:
        raise RuntimeError(f"expected a 512-token model limit; found {maximum}")
    # Inputs are windowed below; keep the tokenizer from warning that the original
    # unwindowed text exceeds the model limit before those windows are built.
    tokenizer.model_max_length = 1_000_000

    flat_tokens: list[list[int]] = []
    window_weights: list[float] = []
    owner: list[int] = []
    input_tokens: list[int] = []
    window_counts = [0] * len(rows)
    for index, row in enumerate(rows):
        text = render_embedding_input(row)
        if hash_bytes(text.encode("utf-8")) != row["text_sha256"]:
            raise RuntimeError(f"sample text fingerprint mismatch for {row['id']}")
        windows, weights, token_count = _chunks(tokenizer, text, maximum, OVERLAP_TOKENS)
        input_tokens.append(token_count)
        window_counts[index] = len(windows)
        flat_tokens.extend(windows)
        window_weights.extend(float(weight) for weight in weights)
        owner.extend([index] * len(windows))

    print(f"encoding {len(rows):,} rows as {len(flat_tokens):,} chunks with {CPU_THREADS} CPU threads", flush=True)
    total = np.zeros((len(rows), int(model.config.hidden_size)), dtype=np.float32)
    weights_by_row = np.zeros((len(rows),), dtype=np.float64)
    for start in range(0, len(flat_tokens), BATCH_SIZE):
        batch_tokens = flat_tokens[start:start + BATCH_SIZE]
        features = [{"input_ids": tokens, "attention_mask": [1] * len(tokens)} for tokens in batch_tokens]
        batch = tokenizer.pad(features, padding=True, return_tensors="pt")
        with torch.inference_mode():
            hidden = model(**batch).last_hidden_state
            mask = batch["attention_mask"].unsqueeze(-1)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors = pooled.cpu().numpy().astype(np.float32, copy=False)
        for offset, vector in enumerate(vectors):
            owner_index = owner[start + offset]
            weight = window_weights[start + offset]
            total[owner_index] += vector * weight
            weights_by_row[owner_index] += weight
        if (start // BATCH_SIZE) % 100 == 0:
            print(f"encoded {min(start + BATCH_SIZE, len(flat_tokens)):,}/{len(flat_tokens):,} chunks", flush=True)

    if not np.all(weights_by_row > 0):
        raise RuntimeError("one or more rows received no token chunks")
    embeddings = total / weights_by_row[:, None].astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.maximum(norms, 1e-12)
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) < 0.99):
        raise RuntimeError("embeddings are empty or non-finite")
    np.save(destination / "embeddings.npy", embeddings)

    if maximum != identity["max_sequence_tokens"]:
        raise RuntimeError("model sequence length changed; update cache identity and rerun")
    cache_key = embedding_cache_key(identity)
    cache = {
        **identity,
        "cache_key": cache_key,
        "rows": len(rows),
        "dimensions": int(embeddings.shape[1]),
        "input_token_count_min": min(input_tokens),
        "input_token_count_median": float(np.median(input_tokens)),
        "input_token_count_max": max(input_tokens),
        "chunked_rows": sum(count > 1 for count in window_counts),
        "total_chunks": len(flat_tokens),
        "truncated_rows": 0,
        "truncated_tokens": 0,
        "cpu_threads": CPU_THREADS,
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "embedded_at": utc_now(),
    }
    (destination / "embedding-cache.json").write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    manifest["embedding_run"] = {key: cache[key] for key in ("cache_key", "rows", "dimensions", "input_token_count_min", "input_token_count_median", "input_token_count_max", "chunked_rows", "total_chunks", "truncated_rows", "truncated_tokens", "cpu_threads", "embedded_at")}
    snapshot_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"embedding complete: shape={embeddings.shape}; chunked={cache['chunked_rows']}; truncated=0", flush=True)
    return cache


def _token_words(text: str) -> set[str]:
    return {token for token in re.findall(r"[^\W_]+", normalize(text), flags=re.UNICODE) if len(token) > 1}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    index = np.searchsorted(np.cumsum(weights), quantile * weights.sum(), side="left")
    return float(values[min(index, len(values) - 1)])


def _nearest_neighbors(rows: list[dict[str, Any]], embeddings: np.ndarray, block_size: int = 256, top_k: int = 10) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    n = len(rows)
    all_top_ids = np.empty((n, top_k), dtype=np.int32)
    all_top_sims = np.empty((n, top_k), dtype=np.float32)
    same_language = np.full((n,), -1.0, dtype=np.float32)
    languages = np.array([row["language"] for row in rows], dtype=object)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        scores = embeddings[start:end] @ embeddings.T
        for local, row_index in enumerate(range(start, end)):
            scores[local, row_index] = -np.inf
            row_scores = scores[local]
            candidate_ids = np.argpartition(row_scores, -top_k)[-top_k:]
            candidate_ids = candidate_ids[np.argsort(row_scores[candidate_ids])[::-1]]
            all_top_ids[row_index] = candidate_ids
            all_top_sims[row_index] = row_scores[candidate_ids]
            mask = languages == languages[row_index]
            mask[row_index] = False
            if mask.any():
                same_language[row_index] = float(np.max(row_scores[mask]))
        print(f"cosine neighbors {end:,}/{n:,}", flush=True)
    neighbors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        for rank, neighbor_index in enumerate(all_top_ids[index], 1):
            other = rows[int(neighbor_index)]
            neighbors.append({
                "id": row["id"],
                "neighbor_id": other["id"],
                "rank": rank,
                "cosine": float(all_top_sims[index, rank - 1]),
                "same_language": row["language"] == other["language"],
                "question_jaccard": _jaccard(_token_words(row["question"]), _token_words(other["question"])),
                "operation": row["operation"],
                "neighbor_operation": other["operation"],
                "domain": row["domain"],
                "neighbor_domain": other["domain"],
                "language": row["language"],
                "neighbor_language": other["language"],
            })
    return neighbors, all_top_sims[:, 0], same_language, all_top_ids


def _coordinate_neighbors(coordinates: np.ndarray, block_size: int = 256, top_k: int = 10) -> np.ndarray:
    n = len(coordinates)
    neighbors = np.empty((n, top_k), dtype=np.int32)
    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        delta = coordinates[start:end, None, :] - coordinates[None, :, :]
        distances = np.einsum("bnd,bnd->bn", delta, delta)
        for local, row_index in enumerate(range(start, end)):
            distances[local, row_index] = np.inf
            candidate_ids = np.argpartition(distances[local], top_k)[:top_k]
            candidate_ids = candidate_ids[np.argsort(distances[local, candidate_ids])]
            neighbors[row_index] = candidate_ids
    return neighbors


def _option_pattern(options: list[str]) -> str:
    kinds: list[str] = []
    for value in options:
        value = value.strip()
        if re.fullmatch(r"[-+]?\d+(?:[.,]\d+)?(?:\s*[%\w]+)?", value, flags=re.UNICODE):
            kind = "numeric"
        elif value.casefold() in {"yes", "no", "true", "false", "ja", "nein", "sí", "non", "oui", "да", "нет"}:
            kind = "boolean"
        elif len(value.split()) <= 2:
            kind = "short"
        elif len(value.split()) <= 8:
            kind = "phrase"
        else:
            kind = "sentence"
        kinds.append(kind)
    return f"{len(options)} choices · " + "/".join(kinds)


def _run_tsne(embeddings: np.ndarray, destination: Path, prefix: str = "") -> list[dict[str, Any]]:
    from sklearn.manifold import TSNE

    # PCA to 50 dimensions improves Barnes-Hut t-SNE speed and stability.
    from sklearn.decomposition import PCA

    reduction = PCA(n_components=min(50, embeddings.shape[1]), svd_solver="randomized", random_state=0)
    reduced = reduction.fit_transform(embeddings)
    outputs = []
    for perplexity, seed in TSNE_RUNS:
        model = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            max_iter=900,
            random_state=seed,
            method="barnes_hut",
            angle=0.5,
            n_jobs=2,
        )
        coordinates = model.fit_transform(reduced).astype(np.float32)
        np.save(destination / f"tsne-{prefix}p{perplexity}-seed{seed}.npy", coordinates)
        outputs.append({
            "perplexity": perplexity,
            "seed": seed,
            "kl_divergence": float(model.kl_divergence_),
            "iterations": int(model.n_iter_),
        })
    return outputs


def _existing_tsne(destination: Path) -> list[dict[str, Any]] | None:
    expected = [
        {"perplexity": perplexity, "seed": seed}
        for perplexity, seed in TSNE_RUNS
    ]
    coordinate_paths = [destination / f"tsne-p{item['perplexity']}-seed{item['seed']}.npy" for item in expected]
    metrics_path = destination / "metrics.json"
    embedding_path = destination / "embeddings.npy"
    if not all(path.is_file() for path in coordinate_paths) or not metrics_path.is_file() or not embedding_path.is_file():
        return None
    try:
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs = previous["tsne_runs"]
        if [(run["perplexity"], run["seed"]) for run in runs] != [(item["perplexity"], item["seed"]) for item in expected]:
            return None
        if any(path.stat().st_mtime_ns < embedding_path.stat().st_mtime_ns for path in coordinate_paths):
            return None
        return runs
    except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _hsl_hex(index: int) -> str:
    hue = (index * 0.61803398875) % 1.0
    saturation, lightness = 0.65, 0.48
    chroma = (1 - abs(2 * lightness - 1)) * saturation
    x = chroma * (1 - abs((hue * 6) % 2 - 1))
    m = lightness - chroma / 2
    sector = int(hue * 6)
    rgb = ((chroma, x, 0), (x, chroma, 0), (0, chroma, x), (0, x, chroma), (x, 0, chroma), (chroma, 0, x))[sector]
    return "#" + "".join(f"{round((value + m) * 255):02x}" for value in rgb)


def _write_html(
    rows: list[dict[str, Any]],
    coords: list[np.ndarray],
    run_meta: list[dict[str, Any]],
    destination: Path,
    centered_coords: list[np.ndarray] | None = None,
    centering_summary: dict[str, Any] | None = None,
) -> None:
    import html as html_module
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    weights = [float(row["sample_weight"]) for row in rows]
    for row in rows:
        row["choice_pattern"] = _option_pattern(row["options"])
        if not row["batch"].startswith("metadata:"):
            row["batch"] = f"file:{row['source_file'].removesuffix('.jsonl')}"
    dimensions = [
        ("source kind", "source_kind", 6),
        ("language", "language", 200),
        ("operation", "operation", 30),
        ("domain", "domain", 16),
        ("batch / source shard", "batch", 16),
        ("choice pattern", "choice_pattern", 16),
    ]

    def dimension_config(field: str, limit: int) -> tuple[list[str], list[int], list[list[Any]]]:
        totals: Counter[str] = Counter()
        for row, weight in zip(rows, weights):
            totals[str(row[field])] += weight
        categories = [value for value, _ in totals.most_common(limit)]
        has_other = any(value not in categories for value in totals)
        labels = categories + (["Other / low frequency"] if has_other else [])
        code_by_value = {value: index for index, value in enumerate(categories)}
        other_index = len(labels) - 1
        codes = [code_by_value.get(str(row[field]), other_index) for row in rows]
        count = max(1, len(labels))
        colors = [_hsl_hex(index) for index in range(count)]
        scale: list[list[Any]] = []
        if count == 1:
            scale = [[0, colors[0]], [1, colors[0]]]
        else:
            for index, color in enumerate(colors):
                start = max(0.0, (index - 0.5) / (count - 1))
                end = min(1.0, (index + 0.5) / (count - 1))
                scale.extend([[start, color], [end, color]])
        return labels, codes, scale

    configs = {field: dimension_config(field, limit) for _, field, limit in dimensions}
    fig = make_subplots(rows=1, cols=2, subplot_titles=[
        f"Perplexity {run_meta[0]['perplexity']} · seed {run_meta[0]['seed']}",
        f"Perplexity {run_meta[1]['perplexity']} · seed {run_meta[1]['seed']}",
    ], horizontal_spacing=0.08)
    for column, coordinate in enumerate(coords, 1):
        categories, codes, scale = configs["language"]
        tick_indices = list(range(min(12, len(categories)))) if column == 1 else []
        colorbar = {
            "title": "language (hover for all)",
            "tickmode": "array",
            "tickvals": tick_indices,
            "ticktext": [categories[index] for index in tick_indices],
            "tickfont": {"size": 9},
            "len": 0.82,
        }
        fig.add_trace(go.Scattergl(
            x=coordinate[:, 0],
            y=coordinate[:, 1],
            mode="markers",
            name="questions",
            showlegend=False,
            marker={
                "size": 5,
                "opacity": 0.62,
                "color": codes,
                "cmin": 0,
                "cmax": max(1, len(categories) - 1),
                "colorscale": scale,
                "showscale": False,
                "colorbar": colorbar,
            },
            customdata=[[
                row["id"], row["language"], row["source_kind"], row["operation"], row["domain"], row["batch"], row["choice_pattern"],
            ] for row in rows],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>language: %{customdata[1]}<br>source: %{customdata[2]}<br>"
                "operation: %{customdata[3]}<br>domain: %{customdata[4]}<br>"
                "batch / shard: %{customdata[5]}<br>choices: %{customdata[6]}<extra></extra>"
            ),
        ), row=1, col=column)
    fig.update_layout(
        title="Question diversity · color by language",
        height=720,
        template="plotly_white",
        margin={"l": 35, "r": 25, "t": 105, "b": 35},
    )
    fig.update_xaxes(title_text="t-SNE 1", showgrid=False, zeroline=False)
    fig.update_yaxes(title_text="t-SNE 2", showgrid=False, zeroline=False)
    plot = fig.to_html(full_html=False, include_plotlyjs=True, config={"displaylogo": False, "scrollZoom": True}, div_id="tsne-plots")
    color_options = "".join(
        f'<option value="{html_module.escape(field)}"{" selected" if field == "language" else ""}>{html_module.escape(label)}</option>'
        for label, field, _limit in dimensions
    )
    color_configs = {}
    for label, field, _limit in dimensions:
        categories, codes, scale = configs[field]
        color_configs[field] = {
            "label": label,
            "codes": codes,
            "scale": scale,
            "cmin": 0,
            "cmax": max(1, len(categories) - 1),
            "showscale": field != "language",
            "tickvals": list(range(min(12, len(categories)))) if field == "language" else list(range(len(categories))),
            "ticktext": categories[:12] if field == "language" else categories,
        }
    serialized_color_configs = json.dumps(color_configs, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    projection_configs = None
    language_options = ""
    if centered_coords is not None:
        if centering_summary is None:
            raise ValueError("centering metrics are required when centered coordinates are provided")
        projection_configs = {
            "original": [{"x": np.round(coordinates[:, 0].astype(np.float64), 4).tolist(), "y": np.round(coordinates[:, 1].astype(np.float64), 4).tolist()} for coordinates in coords],
            "centered": [{"x": np.round(coordinates[:, 0].astype(np.float64), 4).tolist(), "y": np.round(coordinates[:, 1].astype(np.float64), 4).tolist()} for coordinates in centered_coords],
        }
        language_weights = Counter()
        for row in rows:
            language_weights[row["language"]] += float(row["sample_weight"])
        language_options = '<option value="all">All languages (joint layout)</option>' + "".join(
            f'<option value="{html_module.escape(lang)}">{html_module.escape(lang)} ({int(count):,} weighted rows)</option>'
            for lang, count in language_weights.most_common()
        )
        serialized_projection_configs = json.dumps(projection_configs, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
        serialized_languages = json.dumps([row["language"] for row in rows], ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    else:
        serialized_projection_configs = "null"
        serialized_languages = "null"
    centering_summary_html = ""
    if centered_coords is not None and centering_summary is not None:
        original_neighbor = centering_summary["nearest_neighbors"]["original"]
        centered_neighbor = centering_summary["nearest_neighbors"]["centered"]
        original_between = centering_summary["language_effect"]["original"]["between_language_fraction_of_total"]
        centered_between = centering_summary["language_effect"]["centered"]["between_language_fraction_of_total"]
        within = centering_summary["neighbor_retention"]["within_language"]
        operation = centering_summary["nearest_neighbors"]["weighted_top10_same_operation_change"]
        domain = centering_summary["nearest_neighbors"]["weighted_top10_same_domain_change"]
        centering_summary_html = (
            "<p><strong>Centering check:</strong> weighted top-1 neighbor same-language share "
            f"{original_neighbor['top1_same_language_fraction_weighted']:.1%} → {centered_neighbor['top1_same_language_fraction_weighted']:.1%}; "
            f"between-language centroid variance share {original_between:.1%} → {centered_between:.1%}. "
            f"Within-language top-10 neighbors retained {within['weighted_overlap']:.1%} across "
            f"{within['sample_rows_included']:,} rows in {within['languages_included']} languages. "
            f"Top-10 same-operation/domain metadata agreement changed {operation:+.1%}/{domain:+.1%}; "
            "these metadata matches are only rough semantic proxies.</p>"
        )
    weighted_languages: Counter[str] = Counter()
    for row in rows:
        weighted_languages[row["language"]] += float(row["sample_weight"])
    supported_preview = ", ".join(f"{lang} ({int(n)})" for lang, n in weighted_languages.most_common(16))
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Question diversity audit</title>
<style>
body{{font:15px/1.55 system-ui,sans-serif;max-width:1200px;margin:28px auto;padding:0 18px;color:#1e293b}}
h1{{font-size:1.55rem;margin-bottom:.25rem}} p{{max-width:95ch}} .note{{color:#53627a}} a{{color:#3157c8}}
code{{background:#f2f4f8;padding:.1rem .25rem;border-radius:4px}}
</style></head><body>
<h1>Question diversity audit</h1>
<p>Each point is one question represented by context, question and ordered choices. Answers, explanations, teacher targets and generation tags are excluded from the embedding. Hover shows sample metadata only.</p>
<p>Color by <select id="color-by">{color_options}</select> to compare language, source kind, operation, domain, batch / shard, and answer-choice format. Language labels appear on hover; use the language filter to isolate a language. The two plots use different t-SNE settings on the same embeddings.</p>
{"" if centered_coords is None else f'<p>Projection <select id="projection-by"><option value="original">Original</option><option value="centered">Language-centered</option></select> · Language <select id="language-filter">{language_options}</select></p><p class="note">The language filter hides other sample points; it keeps the joint 8,000-row t-SNE coordinates and does not refit a within-language projection. The centered view is a separately fit t-SNE of the centered vectors using the same settings.</p>'}
{centering_summary_html}
<p class="note">t-SNE is a neighborhood visualization: distances between distant clusters and cluster sizes are distorted. Use the reported cosine-neighbor statistics for conclusions. The model card says multilingual-e5-small supports 100 languages and cautions that low-resource languages can be less reliable; the full dataset has more catalog languages than that.</p>
{plot}
<script>
const colorConfigs = {serialized_color_configs};
const projectionConfigs = {serialized_projection_configs};
const pointLanguages = {serialized_languages};
function updateReportPlot() {{
  const config = colorConfigs[document.getElementById('color-by').value];
  const projectionSelect = document.getElementById('projection-by');
  const languageSelect = document.getElementById('language-filter');
  const updates = {{
    'marker.color': [config.codes, config.codes],
    'marker.colorscale': [config.scale, config.scale],
    'marker.cmin': [config.cmin, config.cmin],
    'marker.cmax': [config.cmax, config.cmax],
    'marker.showscale': [config.showscale, config.showscale],
    'marker.colorbar.title.text': [config.label, config.label],
    'marker.colorbar.tickvals': [config.tickvals, config.tickvals],
    'marker.colorbar.ticktext': [config.ticktext, config.ticktext]
  }};
  if (projectionConfigs !== null) {{
    const view = projectionConfigs[projectionSelect.value];
    const selectedLanguage = languageSelect.value;
    for (const key of ['x', 'y']) {{
      updates[key] = view.map(panel => panel[key].map((coordinate, index) =>
        selectedLanguage === 'all' || pointLanguages[index] === selectedLanguage ? coordinate : null
      ));
    }}
  }}
  Plotly.restyle('tsne-plots', updates, [0, 1]);
  Plotly.relayout('tsne-plots', {{'title.text': 'Question diversity · color by ' + config.label}});
}}
document.getElementById('color-by').addEventListener('change', updateReportPlot);
if (projectionConfigs !== null) {{
  document.getElementById('projection-by').addEventListener('change', updateReportPlot);
  document.getElementById('language-filter').addEventListener('change', updateReportPlot);
}}
</script>
<p class="note">Most represented sample languages: {supported_preview}. The sample oversamples some language/task cells intentionally; use each point's saved sample weight for population estimates.</p>
<p>Full results: <a href="audit.md">audit.md</a>. Reproducibility inputs: <a href="snapshot.json">snapshot.json</a>, <a href="embedding-cache.json">embedding-cache.json</a>{"" if centered_coords is None else ', <a href="language-centering-v1.json">language-centering-v1.json</a>'}.</p>
</body></html>"""
    (destination / "report.html").write_text(html, encoding="utf-8")


def _analyze(destination: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in (destination / "sample.jsonl").read_text(encoding="utf-8").splitlines() if line]
    embeddings = np.load(destination / "embeddings.npy", mmap_mode="r")
    if len(rows) != len(embeddings) or len(rows) == 0:
        raise RuntimeError("sample and embedding counts do not match")
    neighbors, nearest, same_language_nearest, original_knn = _nearest_neighbors(rows, np.asarray(embeddings))
    weights = np.array([float(row["sample_weight"]) for row in rows], dtype=np.float64)
    valid_same = same_language_nearest >= 0
    neighbor_metrics = {
        "nearest_cosine_mean_weighted": float(np.average(nearest, weights=weights)),
        "nearest_cosine_p50_weighted": _weighted_quantile(nearest, weights, 0.5),
        "nearest_cosine_p90_weighted": _weighted_quantile(nearest, weights, 0.9),
        "nearest_cosine_gt_0_90_sample_rows": int(np.count_nonzero(nearest > 0.90)),
        "nearest_cosine_sample_rows": len(rows),
        "nearest_cosine_gt_0_90_sample_fraction": float(np.mean(nearest > 0.90)),
        "nearest_cosine_gt_0_90_weighted_rows": float(weights[nearest > 0.90].sum()),
        "nearest_cosine_gt_0_90_weighted_fraction": float(weights[nearest > 0.90].sum() / weights.sum()),
        "nearest_cosine_gt_0_95_weighted_rows": float(weights[nearest > 0.95].sum()),
        "nearest_cosine_gt_0_95_weighted_fraction": float(weights[nearest > 0.95].sum() / weights.sum()),
        "nearest_same_language_mean_weighted": float(np.average(same_language_nearest[valid_same], weights=weights[valid_same])),
        "sample_weight_sum": float(weights.sum()),
    }
    duplicate_candidates = []
    seen_candidate_pairs: set[tuple[str, str]] = set()
    for pair in neighbors:
        if pair["rank"] != 1 or not pair["same_language"] or pair["cosine"] < 0.90 or pair["question_jaccard"] < 0.5:
            continue
        pair_key = tuple(sorted((pair["id"], pair["neighbor_id"])))
        if pair_key not in seen_candidate_pairs:
            seen_candidate_pairs.add(pair_key)
            duplicate_candidates.append(pair)
    duplicate_candidates.sort(key=lambda pair: (pair["cosine"], pair["question_jaccard"]), reverse=True)
    rows_by_id = {row["id"]: row for row in rows}
    candidate_rows: set[str] = set()
    candidate_pair_counts = Counter()
    for pair in duplicate_candidates:
        left = rows_by_id[pair["id"]]
        right = rows_by_id[pair["neighbor_id"]]
        candidate_rows.update((left["id"], right["id"]))
        candidate_pair_counts["same_question"] += normalize(left["question"]) == normalize(right["question"])
        candidate_pair_counts["same_context"] += normalize(left["context"]) == normalize(right["context"])
        candidate_pair_counts["same_operation"] += left["operation"] == right["operation"]
        candidate_pair_counts["same_domain"] += left["domain"] == right["domain"]
        candidate_pair_counts["same_ordered_options"] += left["options"] == right["options"]
        candidate_pair_counts["same_option_set"] += set(left["options"]) == set(right["options"])
    (destination / "nearest-neighbors.jsonl").write_text(
        "".join(json.dumps(pair, ensure_ascii=False, separators=(",", ":")) + "\n" for pair in neighbors),
        encoding="utf-8",
    )
    lang_stats = []
    by_lang: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(rows):
        by_lang[row["language"]].append(i)
    snapshot = json.loads((destination / "snapshot.json").read_text(encoding="utf-8"))
    accepted_langs = snapshot["accepted_by_language"]
    for lang, indexes in sorted(by_lang.items()):
        if len(indexes) < 30:
            continue
        idx = np.array(indexes)
        lang_stats.append({
            "language": lang,
            "population": accepted_langs.get(lang, 0),
            "sample_n": len(indexes),
            "top1_same_language_mean": float(np.average(same_language_nearest[idx][same_language_nearest[idx] >= 0], weights=weights[idx][same_language_nearest[idx] >= 0])),
            "top1_same_language_p90": _weighted_quantile(same_language_nearest[idx][same_language_nearest[idx] >= 0], weights[idx][same_language_nearest[idx] >= 0], 0.9),
        })
    choice_patterns = Counter(_option_pattern(row["options"]) for row in rows)
    t_sne_meta = _existing_tsne(destination)
    if t_sne_meta is None:
        t_sne_meta = _run_tsne(np.asarray(embeddings), destination)
    coords = [np.load(destination / f"tsne-p{item['perplexity']}-seed{item['seed']}.npy") for item in t_sne_meta]
    layout_knn = [_coordinate_neighbors(coordinate) for coordinate in coords]
    k = original_knn.shape[1]
    neighborhood_checks = []
    for meta, neighbors_in_layout in zip(t_sne_meta, layout_knn):
        overlap = np.fromiter(
            (len(set(original) & set(projected)) / k for original, projected in zip(original_knn, neighbors_in_layout)),
            dtype=np.float32,
            count=len(rows),
        )
        neighborhood_checks.append({
            "perplexity": meta["perplexity"],
            "seed": meta["seed"],
            "mean_cosine_neighbor_overlap_at_10": float(np.average(overlap, weights=weights)),
        })
    between_layout = np.fromiter(
        (len(set(first) & set(second)) / k for first, second in zip(layout_knn[0], layout_knn[1])),
        dtype=np.float32,
        count=len(rows),
    )

    question_groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        question_groups[normalize(row["question"])].append(index)
    repeated_questions = []
    for key, indexes in question_groups.items():
        if len(indexes) < 2:
            continue
        repeated_questions.append({
            "question": rows[indexes[0]]["question"],
            "sample_rows": len(indexes),
            "weighted_population_rows": float(weights[indexes].sum()),
            "languages": dict(sorted(Counter(rows[index]["language"] for index in indexes).items())),
            "operations": dict(sorted(Counter(rows[index]["operation"] for index in indexes).items())),
        })
    repeated_questions.sort(key=lambda row: (-row["weighted_population_rows"], row["question"]))
    _write_html(rows, coords, t_sne_meta, destination)

    cache = json.loads((destination / "embedding-cache.json").read_text(encoding="utf-8"))
    results = {
        "analyzed_at": utc_now(),
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "embedding_cache_key": cache["cache_key"],
        "sample_size": len(rows),
        "sample_population_weight_sum": neighbor_metrics["sample_weight_sum"],
        "cosine_neighbor_metrics": neighbor_metrics,
        "languages_with_at_least_30_sample_rows": lang_stats,
        "nearest_neighbor_candidate_count": len(duplicate_candidates),
        "nearest_neighbor_candidate_sample_summary": {
            "pair_count": len(duplicate_candidates),
            "sample_rows_touched": len(candidate_rows),
            "sample_row_fraction_touched": len(candidate_rows) / len(rows),
            "pair_counts": dict(sorted(candidate_pair_counts.items())),
            "rule": "unique unordered pair; rank-1 neighbor; same language; cosine >= 0.90; question token Jaccard >= 0.50",
        },
        "nearest_neighbor_candidates": duplicate_candidates[:50],
        "repeated_question_frames_in_sample": repeated_questions[:20],
        "choice_pattern_top_20": [{"pattern": key, "rows": value} for key, value in choice_patterns.most_common(20)],
        "tsne_runs": t_sne_meta,
        "tsne_local_neighborhood_checks": neighborhood_checks,
        "tsne_between_run_neighbor_overlap_at_10_weighted": float(np.average(between_layout, weights=weights)),
        "interpretation_notes": [
            "The t-SNE views are exploratory and do not preserve global distance or cluster size.",
            "The cosine and lexical neighbor checks use the original embedding space.",
            "Candidate near-duplicates retain their IDs and require human review; no row was removed or relabeled.",
            "The sample intentionally oversamples small language/task cells; use saved inverse cell-sampling weights for population-level metrics.",
        ],
    }
    (destination / "metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "audit-2026-09-29")
    parser.add_argument("--stage", choices=("snapshot", "embed", "analyze", "all"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage in {"snapshot", "all"} and not (args.output / "snapshot.json").exists():
        _accepted_snapshot(args.root.resolve(), args.output.resolve())
    elif args.stage in {"snapshot", "all"}:
        print("using existing frozen snapshot", flush=True)
    cache: dict[str, Any] | None = None
    if args.stage in {"embed", "all"}:
        cache = _embed(args.output.resolve())
    if args.stage in {"analyze", "all"}:
        cache_path = args.output / "embedding-cache.json"
        if cache is None and cache_path.is_file():
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        metrics_path = args.output / "metrics.json"
        metrics_valid = False
        if metrics_path.is_file() and cache:
            try:
                metrics_valid = analysis_cache_matches(json.loads(metrics_path.read_text(encoding="utf-8")), cache["cache_key"])
            except (OSError, ValueError, json.JSONDecodeError, KeyError):
                metrics_valid = False
        if args.stage == "analyze" or not metrics_valid:
            _analyze(args.output.resolve())
        else:
            print("using matching audit metrics", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
