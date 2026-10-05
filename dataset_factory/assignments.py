#!/usr/bin/env python3
"""Balance deterministic train-question assignments without writing questions."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CATALOG = Path("data/generated_train/coverage-catalog-v1.json")
WORD_POOL = Path("data/generated_train/seed-word-pool.jsonl.gz")
ASSIGNMENT_DIR = Path("data/generated_train/assignments")
BATCH_DIR = Path("data/generated_train/batches")
BROAD_V1_DIR = Path("data/generated_train/broad-v1")
MULTILINGUAL_DIR = Path("data/generated_train/multilingual")
MAX_BATCH = 10_000
PROVENANCE = "Luna-generated; label not independently verified"


class ContractError(ValueError):
    pass


def _json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ContractError(f"required input file is missing: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read valid JSON from {path}: {exc}") from exc


def _items(value: Any, label: str) -> list[Any]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or any(item is None for item in value):
        raise ContractError(f"{label} must be a non-empty list")
    return value


def _text(obj: dict[str, Any], key: str, label: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{label}.{key} must be a non-empty string")
    return value.strip()


def load_catalog(path: Path = CATALOG, *, max_choice_count: int = 4) -> dict[str, Any]:
    """Validate the catalog fields and build its compatible operation/domain cells."""
    data = _json(path)
    if not isinstance(data, dict):
        raise ContractError("catalog root must be an object")

    for group, id_key, required in (
        ("languages", "tag", ("name", "family", "script")),
        ("domains", "id", ("name", "region")),
        ("operations", "id", ("name",)),
    ):
        entries = data.get(group)
        if not isinstance(entries, list) or not entries:
            raise ContractError(f"catalog.{group} must be a non-empty list")
        seen: set[str] = set()
        for i, entry in enumerate(entries):
            label = f"catalog.{group}[{i}]"
            if not isinstance(entry, dict):
                raise ContractError(f"{label} must be an object")
            identifier = _text(entry, id_key, label)
            for key in required:
                _text(entry, key, label)
            if identifier in seen:
                raise ContractError(f"duplicate {group} identifier: {identifier}")
            seen.add(identifier)

    domains = {domain["id"]: domain for domain in data["domains"]}
    cells: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for operation in data["operations"]:
        compatible = operation.get("compatible_domains")
        forms = _items(operation.get("question_forms"), f"operation {operation['id']}.question_forms")
        if not isinstance(compatible, list) or not compatible or any(not isinstance(x, str) for x in compatible):
            raise ContractError(f"operation {operation['id']}.compatible_domains must be a list of domain ids")
        if len(set(compatible)) != len(compatible):
            raise ContractError(f"operation {operation['id']} repeats a compatible domain")
        if any(not isinstance(form, (str, dict)) for form in forms):
            raise ContractError(f"operation {operation['id']}.question_forms entries must be strings or objects")
        for domain_id in compatible:
            if domain_id not in domains:
                raise ContractError(f"operation {operation['id']} references unknown domain {domain_id!r}")
            cells.append((operation, domains[domain_id]))

    for key in ("input_structures", "registers", "difficulty_bands", "context_length_bands"):
        _items(data.get(key), f"catalog.{key}")

    choice_counts: list[int] = []
    for entry in _items(data.get("choice_counts"), "catalog.choice_counts"):
        value = entry
        if isinstance(entry, dict):
            value = next((entry[k] for k in ("count", "value", "choices") if k in entry), None)
        if isinstance(value, str) and value.isdecimal():
            value = int(value)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 2 <= value <= max_choice_count
        ):
            raise ContractError(f"choice_counts entries must resolve to 2–{max_choice_count}")
        if value not in choice_counts:
            choice_counts.append(value)
    if not cells:
        raise ContractError("catalog has no compatible operation/domain cells")
    data["_cells"] = cells
    data["_choice_counts"] = choice_counts
    return data


def _pool_defaults(path: Path) -> dict[str, str]:
    if not path.name.endswith(".jsonl.gz"):
        return {}
    manifest_path = path.with_name(path.name.removesuffix(".jsonl.gz") + "-manifest.json")
    if not manifest_path.exists():
        return {}
    manifest = _json(manifest_path)
    source = manifest.get("source") if isinstance(manifest, dict) else None
    defaults = {"pool_id": manifest.get("pool_id")} if isinstance(manifest, dict) else {}
    if isinstance(source, dict):
        defaults["source_version"] = source.get("source_version")
    for key, value in list(defaults.items()):
        if not isinstance(value, str) or not value.strip():
            defaults.pop(key)
        else:
            defaults[key] = value.strip()
    return defaults


def load_word_pool(path: Path = WORD_POOL) -> list[dict[str, str]]:
    """Read JSONL word entries, using the adjacent manifest for pool provenance when needed."""
    defaults = _pool_defaults(path)
    rows: list[dict[str, str]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_no, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    source = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ContractError(f"invalid JSON in {path} line {line_no}: {exc}") from exc
                if not isinstance(source, dict):
                    raise ContractError(f"{path} line {line_no} must be an object")
                row: dict[str, str] = {}
                for key in ("word", "source_language", "pool_id", "source_version"):
                    value = source.get(key, defaults.get(key))
                    if not isinstance(value, str) or not value.strip():
                        raise ContractError(f"{path} line {line_no} needs {key!r} or manifest metadata")
                    row[key] = value.strip()
                rows.append(row)
    except FileNotFoundError as exc:
        raise ContractError(f"required input file is missing: {path}") from exc
    except (OSError, EOFError, UnicodeError) as exc:
        raise ContractError(f"cannot read compressed word pool {path}: {exc}") from exc
    if len({row["word"].casefold() for row in rows}) < 3:
        raise ContractError("word pool must contain at least 3 distinct words")
    return rows


def _cell_key(operation_id: str, domain_id: str) -> str:
    return f"{operation_id}|{domain_id}"


def _language(row: dict[str, Any]) -> str | None:
    value = row.get("language")
    if isinstance(value, dict):
        value = value.get("tag")
    if isinstance(value, str):
        return value
    profile = row.get("language_profile")
    return profile.get("tag") if isinstance(profile, dict) else None


def _identifier(row: dict[str, Any], field: str) -> str | None:
    direct = row.get(f"{field}_id")
    if isinstance(direct, str):
        return direct
    nested = row.get(field)
    if isinstance(nested, dict) and isinstance(nested.get("id"), str):
        return nested["id"]
    return nested if isinstance(nested, str) else None


def _meta(row: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in ("generation_metadata", "_generator"):
        value = row.get(key)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _operation_id(row: dict[str, Any], known: set[str]) -> str | None:
    direct_id = row.get("operation_id")
    if isinstance(direct_id, str):
        return direct_id
    operation = row.get("operation")
    if isinstance(operation, dict) and isinstance(operation.get("id"), str):
        return operation["id"]
    if isinstance(operation, str) and operation in known:
        return operation
    meta = _meta(row)
    for candidate in (meta.get("operation_id"), row.get("family"), row.get("task_id")):
        if isinstance(candidate, str) and candidate in known:
            return candidate
    return None


def _domain_id(row: dict[str, Any], known: set[str]) -> str | None:
    direct_id = row.get("domain_id")
    if isinstance(direct_id, str):
        return direct_id
    domain = row.get("domain")
    if isinstance(domain, dict) and isinstance(domain.get("id"), str):
        return domain["id"]
    if isinstance(domain, str) and domain in known:
        return domain
    meta = _meta(row)
    candidate = meta.get("coverage_domain_id") or meta.get("domain_id")
    if isinstance(candidate, str):
        return candidate
    topic = row.get("topic")
    return topic if isinstance(topic, str) and topic in known else None


def _is_assignment_plan(row: Any, path: Path) -> bool:
    return (
        isinstance(row, dict)
        and isinstance(row.get("_generator"), dict)
        and ("assignments" in path.parts or str(row.get("id", "")).startswith(("broad-v1-assignment-", "assignment-")))
    )


def _plan_counts_as_active(row: dict[str, Any]) -> bool:
    """Only uncovered work should influence selection after its shard lands."""
    if "status" not in row or row["status"] is None:
        return True
    status = row["status"]
    if not isinstance(status, str):
        return False
    normalized = status.strip().casefold().replace("-", "_").replace(" ", "_")
    return normalized in {"assigned", "in_flight"}


class Coverage:
    def __init__(self, catalog: dict[str, Any] | None = None) -> None:
        self.known_languages = {x["tag"] for x in catalog["languages"]} if catalog else set()
        self.known_domains = {x["id"] for x in catalog["domains"]} if catalog else set()
        self.known_operations = {x["id"] for x in catalog["operations"]} if catalog else set()
        self.known_cells = {
            _cell_key(operation["id"], domain["id"])
            for operation, domain in catalog["_cells"]
        } if catalog else set()
        self.languages: Counter[str] = Counter()
        self.domains: Counter[str] = Counter()
        self.operations: Counter[str] = Counter()
        self.cells: Counter[str] = Counter()
        self.outcomes: Counter[tuple[str, str]] = Counter()
        self.seed_sources: Counter[str] = Counter()
        self.groups: set[str] = set()
        self.assignment_ids: set[str] = set()
        self.indexes: set[int] = set()
        self.train_rows = 0
        self.question_rows = 0
        self.assignment_plans = 0
        self.active_assignment_plans = 0
        self.quarantined = 0
        self.quarantine_reasons: Counter[str] = Counter()
        self.quarantine_files: Counter[str] = Counter()

    def add(self, row: dict[str, Any], catalog: dict[str, Any], is_plan: bool = False) -> None:
        if row.get("split") != "train":
            return
        self.train_rows += 1
        if is_plan:
            self.assignment_plans += 1
            active_plan = _plan_counts_as_active(row)
            if active_plan:
                self.active_assignment_plans += 1
        else:
            self.question_rows += 1
        if isinstance(row.get("problem_group_id"), str):
            self.groups.add(row["problem_group_id"])
        if is_plan and isinstance(row.get("id"), str):
            self.assignment_ids.add(row["id"])
        meta = _meta(row)
        index = meta.get("global_index")
        if isinstance(index, int) and not isinstance(index, bool):
            self.indexes.add(index)
        if is_plan and not active_plan:
            return

        lang = _language(row)
        domain = _domain_id(row, self.known_domains)
        operation = _operation_id(row, self.known_operations)
        if lang in self.known_languages:
            self.languages[lang] += 1
        if domain in self.known_domains:
            self.domains[domain] += 1
        if operation in self.known_operations:
            self.operations[operation] += 1
        if domain in self.known_domains and operation in self.known_operations:
            key = _cell_key(operation, domain)
            if key in self.known_cells:
                self.cells[key] += 1
                outcome = _meta(row).get("outcome_class") or row.get("semantic_outcome")
                if isinstance(outcome, str):
                    self.outcomes[(key, outcome)] += 1
        seed_words = meta.get("seed_words")
        if isinstance(seed_words, list):
            for word in seed_words:
                if isinstance(word, dict) and isinstance(word.get("source_language"), str):
                    self.seed_sources[word["source_language"]] += 1
        source_tags = meta.get("seed_word_source_languages")
        if not seed_words and isinstance(source_tags, list):
            self.seed_sources.update(x for x in source_tags if isinstance(x, str))


def _reasons(row: Any, path: Path, languages: set[str]) -> list[str]:
    if not isinstance(row, dict):
        return ["row_is_not_an_object"]
    errors = [f"missing_or_empty_{k}" for k in ("id", "problem_group_id") if not isinstance(row.get(k), str) or not row[k].strip()]
    if row.get("split") != "train":
        errors.append("split_is_not_train")
    lang = _language(row)
    if not lang:
        errors.append("missing_or_empty_language")
    elif lang not in languages:
        errors.append("language_not_in_catalog")
    if _is_assignment_plan(row, path):
        return errors + [f"missing_or_empty_{k}" for k in ("domain_id", "operation_id") if not isinstance(row.get(k), str)]

    for key in ("difficulty", "context", "question", "explanation", "provenance"):
        if not isinstance(row.get(key), str) or not row[key].strip():
            errors.append(f"missing_or_empty_{key}")
    has_operation = any(
        isinstance(row.get(key), str) and row[key].strip()
        for key in ("family", "task_id", "operation_id")
    )
    operation = row.get("operation")
    if isinstance(operation, str) and operation.strip():
        has_operation = True
    elif isinstance(operation, dict) and isinstance(operation.get("id"), str) and operation["id"].strip():
        has_operation = True
    if not has_operation:
        errors.append("missing_or_empty_operation")
    if not any(isinstance(row.get(k), str) and row[k].strip() for k in ("topic", "domain", "domain_id")):
        errors.append("missing_or_empty_topic_or_domain")
    if row.get("difficulty") not in {"easy", "medium", "hard"}:
        errors.append("unsupported_difficulty")
    options = row.get("options")
    if not isinstance(options, list) or not 2 <= len(options) <= 4 or any(not isinstance(x, str) or not x.strip() for x in options):
        errors.append("options_must_contain_two_to_four_strings")
    elif len({" ".join(unicodedata.normalize("NFKC", x).casefold().split()) for x in options}) != len(options):
        errors.append("options_are_not_distinct")
    answer = row.get("answer_index")
    if not isinstance(answer, int) or isinstance(answer, bool):
        errors.append("answer_index_is_not_an_integer")
    elif isinstance(options, list) and not 0 <= answer < len(options):
        errors.append("answer_index_is_out_of_range")
    provenance = row.get("provenance")
    if isinstance(provenance, str) and provenance.strip():
        label = " ".join(unicodedata.normalize("NFKC", provenance).casefold().split()).rstrip(".")
        if label != PROVENANCE.casefold():
            errors.append("provenance_does_not_match_required_label")
    return errors


def load_history(directories: Iterable[Path], catalog: dict[str, Any]) -> Coverage:
    languages = {x["tag"] for x in catalog["languages"]}
    seen: set[Path] = set()
    unique_dirs = {directory.resolve() for directory in directories}
    for directory in unique_dirs:
        if directory.exists():
            if directory.name in {"assignments", "batches"}:
                pattern = "shard-*.jsonl"
            elif directory.name == "broad-v1":
                pattern = "*.jsonl"
            else:
                pattern = "*.jsonl"
            seen.update(p.resolve() for p in directory.glob(pattern) if p.is_file())
    coverage = Coverage(catalog)
    for path in sorted(seen):
        with path.open("rb") as stream:
            for raw in stream:
                try:
                    line = raw.decode("utf-8")
                except UnicodeDecodeError:
                    row, errors = None, ["invalid_utf8"]
                else:
                    try:
                        row = json.loads(line)
                        errors = ["invalid_json"] if not line.strip() else _reasons(row, path, languages)
                    except json.JSONDecodeError:
                        row, errors = None, ["invalid_json"]
                if errors:
                    coverage.quarantined += 1
                    coverage.quarantine_reasons.update(set(errors))
                    coverage.quarantine_files[path.as_posix()] += 1
                else:
                    coverage.add(row, catalog, _is_assignment_plan(row, path))
    return coverage


def _derived_seed(seed: int, index: int) -> int:
    value = f"horscht-assignment-v1:{seed}:{index}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _pick_min(items: list[Any], key: Any, counts: Counter[Any], rng: random.Random) -> Any:
    minimum = min(counts[key(item)] for item in items)
    tied = sorted((item for item in items if counts[key(item)] == minimum), key=key)
    return rng.choice(tied)


def _outcomes(operation: dict[str, Any]) -> tuple[str, tuple[str, ...]]:
    text = " ".join((operation["id"], operation["name"], json.dumps(operation["question_forms"]))).casefold()
    groups = (
        (("sort", "order", "sequence", "rank", "priority"), "produce the required order", ("ascending_order", "descending_order", "stable_tie_order", "dependency_order")),
        (("classif", "categor", "status", "label", "grade"), "assign the category that matches the stated rule", ("positive_category", "negative_category", "boundary_category", "defer_category")),
        (("count", "calcul", "total", "average", "ratio", "convert", "measure"), "compute the requested value from the supplied data", ("exact_total", "difference", "weighted_value", "minimum_required_count")),
        (("filter", "merge", "dedup", "reconcil", "retain", "record"), "retain records allowed by the stated rule", ("retain_eligible", "keep_first_occurrence", "keep_latest_record", "exclude_duplicate")),
        (("valid", "verif", "check", "eligib", "audit", "satisf"), "determine whether the stated requirement is satisfied", ("requirement_met", "requirement_not_met", "evidence_incomplete", "boundary_satisfied")),
        (("select", "choos", "allocat", "assign", "schedul", "route", "plan", "decision", "resolv", "optim"), "choose an option that best meets the stated constraints", ("feasible_choice", "lowest_cost_choice", "earliest_valid_choice", "highest_value_choice")),
    )
    for needles, goal, outcomes in groups:
        if any(term in text for term in needles):
            return goal, outcomes
    return "identify the action supported by the supplied rules", ("supported_action", "blocked_action", "deferred_action", "rule_satisfied")


def _word_groups(pool: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in pool:
        groups[row["word"].casefold()].append(row)
    return [groups[key] for key in sorted(groups)]


def _one(
    catalog: dict[str, Any],
    word_groups: list[list[dict[str, str]]],
    coverage: Coverage,
    seed: int,
    index: int,
    context_length_options: list[Any],
) -> dict[str, Any]:
    derived = _derived_seed(seed, index)
    rng = random.Random(derived)
    language = _pick_min(catalog["languages"], lambda x: x["tag"], coverage.languages, rng)
    operation, domain = _pick_min(catalog["_cells"], lambda x: _cell_key(x[0]["id"], x[1]["id"]), coverage.cells, rng)
    form = rng.choice(_items(operation["question_forms"], "question_forms"))
    choices = rng.choice(catalog["_choice_counts"])
    goal, outcomes = _outcomes(operation)
    cell = _cell_key(operation["id"], domain["id"])
    least_used = min(coverage.outcomes[(cell, outcome)] for outcome in outcomes)
    outcome = rng.choice([x for x in outcomes if coverage.outcomes[(cell, x)] == least_used])
    words = [rng.choice(group).copy() for group in rng.sample(word_groups, rng.randint(3, min(8, len(word_groups))))]
    row = {
        "id": f"broad-v1-assignment-{index:010d}",
        "problem_group_id": f"broad-v1-group-{index:010d}",
        "split": "train",
        "language": language["tag"],
        "language_profile": language,
        "domain_id": domain["id"],
        "domain": domain,
        "operation_id": operation["id"],
        "operation": operation,
        "question_form": form,
        "input_structure": rng.choice(_items(catalog["input_structures"], "input_structures")),
        "register": rng.choice(_items(catalog["registers"], "registers")),
        "difficulty_band": rng.choice(_items(catalog["difficulty_bands"], "difficulty_bands")),
        "context_length_band": rng.choice(context_length_options),
        "choice_count": choices,
        "_generator": {
            "base_seed": seed,
            "derived_seed": derived,
            "global_index": index,
            "seed_words": words,
            "outcome_goal": goal,
            "outcome_class": outcome,
            "answer_position": rng.randrange(choices),
        },
        "status": "assigned",
    }
    coverage.add(row, catalog, is_plan=True)
    return row


def _sampling_catalog(catalog: dict[str, Any], language_tags: Iterable[str] | None) -> dict[str, Any]:
    if language_tags is None:
        return catalog
    requested = list(language_tags)
    if not requested:
        raise ContractError("language_tags must contain at least one catalog tag")
    if len(set(requested)) != len(requested):
        raise ContractError("language_tags must not contain duplicates")
    languages = {language["tag"]: language for language in catalog["languages"]}
    unknown = sorted(set(requested) - languages.keys())
    if unknown:
        raise ContractError(f"language_tags contain unknown catalog tags: {', '.join(unknown)}")
    selected = dict(catalog)
    selected["languages"] = [languages[tag] for tag in requested]
    return selected


def _context_length_options(catalog: dict[str, Any], bands: Iterable[str] | None) -> list[Any]:
    available = _items(catalog.get("context_length_bands"), "catalog.context_length_bands")
    if bands is None:
        return available
    requested = list(bands)
    if not requested or any(not isinstance(tag, str) or not tag.strip() for tag in requested):
        raise ContractError("context_length_bands must contain non-empty catalog ids")
    requested = [tag.strip() for tag in requested]
    if len(set(requested)) != len(requested):
        raise ContractError("context_length_bands must not contain duplicates")
    by_id: dict[str, Any] = {}
    for item in available:
        identifier = item.get("id") if isinstance(item, dict) else item
        if isinstance(identifier, str) and identifier.strip():
            by_id[identifier.strip()] = item
    unknown = sorted(set(requested) - by_id.keys())
    if unknown:
        raise ContractError(f"context_length_bands contain unknown catalog ids: {', '.join(unknown)}")
    return [by_id[tag] for tag in requested]


def create_assignments(
    catalog: dict[str, Any],
    pool: list[dict[str, str]],
    count: int,
    seed: int,
    start_index: int,
    coverage: Coverage | None = None,
    language_tags: Iterable[str] | None = None,
    context_length_bands: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if not 1 <= count <= MAX_BATCH:
        raise ContractError(f"count must be 1–{MAX_BATCH}")
    if start_index < 0:
        raise ContractError("start_index must be zero or greater")
    groups = _word_groups(pool)
    if len(groups) < 3:
        raise ContractError("word pool must have at least 3 distinct words")
    current = coverage or Coverage(catalog)
    sampling_catalog = _sampling_catalog(catalog, language_tags)
    context_length_options = _context_length_options(catalog, context_length_bands)
    return [
        _one(sampling_catalog, groups, current, seed, i, context_length_options)
        for i in range(start_index, start_index + count)
    ]


def _index(row: dict[str, Any]) -> int | None:
    meta = row.get("_generator")
    value = meta.get("global_index") if isinstance(meta, dict) else None
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _read_shard(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot resume {path}: {exc}") from exc
    if any(not isinstance(row, dict) or row.get("split") != "train" for row in rows):
        raise ContractError(f"cannot resume {path}: expected train assignment rows")
    indexes = [_index(row) for row in rows]
    if not rows or any(i is None for i in indexes):
        raise ContractError(f"cannot resume {path}: every row needs _generator.global_index")
    concrete = [i for i in indexes if i is not None]
    if concrete != list(range(concrete[0], concrete[0] + len(concrete))):
        raise ContractError(f"cannot resume {path}: global_index values must be contiguous and ordered")
    groups = [row.get("problem_group_id") for row in rows]
    if any(not isinstance(x, str) or not x for x in groups) or len(set(groups)) != len(groups):
        raise ContractError(f"cannot resume {path}: problem_group_id values must be non-empty and unique")
    return rows


def _counts(rows: list[dict[str, Any]], catalog: dict[str, Any]) -> dict[str, Any]:
    langs, domains, operations, cells, sources = (Counter() for _ in range(5))
    known_domains = {x["id"] for x in catalog["domains"]}
    known_operations = {x["id"] for x in catalog["operations"]}
    for row in rows:
        lang = _language(row)
        domain = _domain_id(row, known_domains)
        operation = _operation_id(row, known_operations)
        if lang:
            langs[lang] += 1
        if domain:
            domains[domain] += 1
        if operation:
            operations[operation] += 1
        if domain and operation:
            cells[_cell_key(operation, domain)] += 1
        meta = _meta(row)
        for word in meta.get("seed_words", []):
            if isinstance(word, dict) and isinstance(word.get("source_language"), str):
                sources[word["source_language"]] += 1
        if not meta.get("seed_words") and isinstance(meta.get("seed_word_source_languages"), list):
            sources.update(x for x in meta["seed_word_source_languages"] if isinstance(x, str))
    return {
        "assignments": len(rows),
        "languages": {x["tag"]: langs[x["tag"]] for x in catalog["languages"]},
        "domains": {x["id"]: domains[x["id"]] for x in catalog["domains"]},
        "operations": {x["id"]: operations[x["id"]] for x in catalog["operations"]},
        "operation_domain_cells": {_cell_key(op["id"], domain["id"]): cells[_cell_key(op["id"], domain["id"])] for op, domain in catalog["_cells"]},
        "seed_source_languages": dict(sorted(sources.items())),
    }


def _coverage(coverage: Coverage, catalog: dict[str, Any]) -> dict[str, Any]:
    result = _counts_from_coverage(coverage, catalog)
    result.update({
        "train_rows_seen": coverage.train_rows,
        "question_rows_seen": coverage.question_rows,
        "assignment_plans_seen": coverage.assignment_plans,
        "active_assignment_plans": coverage.active_assignment_plans,
        "quarantined_lines": coverage.quarantined,
        "quarantine_reasons": dict(sorted(coverage.quarantine_reasons.items())),
        "quarantine_files": dict(sorted(coverage.quarantine_files.items())),
        "seed_source_languages": dict(sorted(coverage.seed_sources.items())),
    })
    return result


def _counts_from_coverage(c: Coverage, catalog: dict[str, Any]) -> dict[str, Any]:
    return {
        "languages": {x["tag"]: c.languages[x["tag"]] for x in catalog["languages"]},
        "domains": {x["id"]: c.domains[x["id"]] for x in catalog["domains"]},
        "operations": {x["id"]: c.operations[x["id"]] for x in catalog["operations"]},
        "operation_domain_cells": {_cell_key(op["id"], domain["id"]): c.cells[_cell_key(op["id"], domain["id"])] for op, domain in catalog["_cells"]},
    }


def _next_shard(directory: Path) -> int:
    indexes = [int(p.stem[6:]) for p in directory.glob("shard-*.jsonl") if p.stem[6:].isdecimal()] if directory.exists() else []
    return max(indexes, default=0) + 1


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]], existing: Path | None = None) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            if existing is not None:
                with existing.open("rb") as source:
                    shutil.copyfileobj(source, stream)
            for row in rows:
                line = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                stream.write(line.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_assignment_batch(
    *,
    count: int,
    seed: int = 0,
    start_index: int | None = None,
    shard_index: int | None = None,
    resume: bool = False,
    language_tags: Iterable[str] | None = None,
    context_length_bands: Iterable[str] | None = None,
    catalog_path: Path = CATALOG,
    word_pool_path: Path = WORD_POOL,
    assignment_dir: Path = ASSIGNMENT_DIR,
    batch_history_dir: Path = BATCH_DIR,
) -> dict[str, Any]:
    if not 1 <= count <= MAX_BATCH:
        raise ContractError(f"count must be 1–{MAX_BATCH}")
    catalog = load_catalog(catalog_path)
    pool = load_word_pool(word_pool_path)
    assignment_dir.mkdir(parents=True, exist_ok=True)
    shard = shard_index or _next_shard(assignment_dir)
    if shard < 1:
        raise ContractError("shard_index must be one or greater")
    target = assignment_dir / f"shard-{shard:03d}.jsonl"
    if target.exists() != resume:
        raise ContractError(f"{target} {'already exists; use --resume' if target.exists() else 'does not exist for --resume'}")

    history = load_history((assignment_dir, batch_history_dir, BROAD_V1_DIR, MULTILINGUAL_DIR), catalog)
    prior_groups = set(history.groups)
    prior_assignment_ids = set(history.assignment_ids)
    old = _read_shard(target) if resume else []
    if resume:
        start = _index(old[0])
        indexes = [_index(row) for row in old]
        if start_index is not None and start_index != start:
            raise ContractError(f"start_index must match existing shard start {start}")
        if count < len(old):
            raise ContractError(f"count {count} is less than existing shard size {len(old)}")
        next_index = indexes[-1] + 1
        later = [i for i in history.indexes if i >= next_index and i not in indexes]
        if count > len(old) and later:
            raise ContractError(f"cannot resume before later assignment offset {max(later)}")
        add_count = count - len(old)
    else:
        start = start_index
        if start is None:
            start = max(max(history.indexes, default=0) + 1, history.train_rows + 1)
        next_index = start
        overlaps = [i for i in history.indexes if start <= i < start + count]
        if overlaps:
            raise ContractError(f"global index range overlaps existing assignment at {min(overlaps)}")
        add_count = count

    new = (
        create_assignments(
            catalog, pool, add_count, seed, next_index, history,
            language_tags=language_tags, context_length_bands=context_length_bands,
        )
        if add_count
        else []
    )
    old_groups = {r.get("problem_group_id") for r in old}
    if len(old_groups | {r["problem_group_id"] for r in new}) != len(old) + len(new):
        raise ContractError("generated problem_group_id values are not unique")
    if any(r["problem_group_id"] in prior_groups for r in new):
        raise ContractError("generated problem_group_id conflicts with historical data")
    if any(r["id"] in prior_assignment_ids for r in new):
        raise ContractError("generated assignment id conflicts with history")

    try:
        if new:
            _write_jsonl_atomic(target, new, existing=target if resume else None)
    except OSError as exc:
        raise ContractError(f"cannot write {target}: {exc}") from exc
    return {
        "output": target.as_posix(),
        "shard_index": shard,
        "start_index": start,
        "new_assignments": len(new),
        "shard_assignments": len(old) + len(new),
        "batch_counts": _counts(new, catalog),
        "shard_counts": _counts(old + new, catalog),
        "coverage_after": _coverage(history, catalog),
    }


def _check(
    catalog_path: Path,
    word_pool_path: Path,
    assignment_dir: Path,
    batch_dir: Path,
) -> dict[str, Any]:
    catalog = load_catalog(catalog_path)
    pool = load_word_pool(word_pool_path)
    history = load_history((assignment_dir, batch_dir, BROAD_V1_DIR, MULTILINGUAL_DIR), catalog)
    return {
        "status": "ok",
        "languages": len(catalog["languages"]),
        "domains": len(catalog["domains"]),
        "operations": len(catalog["operations"]),
        "compatible_operation_domain_cells": len(catalog["_cells"]),
        "distinct_seed_words": len({x["word"].casefold() for x in pool}),
        "historical_train_rows": history.train_rows,
        "historical_question_rows": history.question_rows,
        "historical_assignment_plans": history.assignment_plans,
        "historical_active_assignment_plans": history.active_assignment_plans,
        "quarantined_historical_lines": history.quarantined,
        "quarantine_reasons": dict(sorted(history.quarantine_reasons.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--word-pool", type=Path, default=WORD_POOL)
    parser.add_argument("--assignment-dir", type=Path, default=ASSIGNMENT_DIR)
    parser.add_argument("--batch-history-dir", type=Path, default=BATCH_DIR)
    parser.add_argument("--count", type=int, help=f"batch size (1–{MAX_BATCH}); with --resume, final shard size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument(
        "--language-tags", nargs="+",
        help="restrict output-language sampling to these catalog tags",
    )
    parser.add_argument(
        "--context-length-bands", nargs="+",
        help="restrict context-length sampling to these catalog ids",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--check", action="store_true", help="validate catalog, pool, and history without writing assignments")
    args = parser.parse_args(argv)
    try:
        if args.check:
            result = _check(args.catalog, args.word_pool, args.assignment_dir, args.batch_history_dir)
        else:
            if args.count is None:
                parser.error("--count is required unless --check is used")
            result = write_assignment_batch(
                count=args.count, seed=args.seed, start_index=args.start_index,
                shard_index=args.shard_index, resume=args.resume, language_tags=args.language_tags,
                context_length_bands=args.context_length_bands,
                catalog_path=args.catalog,
                word_pool_path=args.word_pool, assignment_dir=args.assignment_dir,
                batch_history_dir=args.batch_history_dir,
            )
    except ContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
