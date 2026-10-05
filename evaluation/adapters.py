"""Pinned deterministic transforms for the selected external MCQ slices."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping

from evaluation.isolation import BenchmarkIsolationError, iter_json_records


class MMLUProAdapter:
    """Apply the official placeholder-choice rule without carrying CoT fields."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    @property
    def aggregate_stats(self) -> dict[str, int]:
        return dict(self._counts)

    def __call__(self, row: Mapping[str, Any], row_number: int) -> dict[str, Any]:
        self._counts["source_rows_seen"] += 1
        item_id = row.get("question_id")
        question = row.get("question")
        options = row.get("options")
        if item_id is None or not isinstance(question, str) or not question.strip():
            raise BenchmarkIsolationError("mmlu_required_field_invalid")
        if not isinstance(options, list) or any(not isinstance(option, str) for option in options):
            raise BenchmarkIsolationError("mmlu_options_invalid")

        original_index = _choice_index(row.get("answer_index"), len(options))
        if original_index is None:
            raise BenchmarkIsolationError("mmlu_answer_index_invalid")
        retained_indices = [index for index, option in enumerate(options) if option != "N/A"]
        removed = len(options) - len(retained_indices)
        self._counts["placeholder_choices_removed"] += removed
        self._counts["rows_with_placeholder_choices"] += removed > 0
        if original_index not in retained_indices:
            raise BenchmarkIsolationError("mmlu_answer_points_to_placeholder")
        remapped_index = retained_indices.index(original_index)
        retained_options = [options[index] for index in retained_indices]
        category = row.get("category")
        result = {
            "item_id": str(item_id),
            "context": "",
            "question": question,
            "options": retained_options,
            "answer_index": remapped_index,
        }
        if isinstance(category, str) and category.strip():
            result["category"] = category.strip()
        return result


class BelebeleAdapter:
    """Map one complete source-language file and retain its parallel group key."""

    def __init__(self, language_tag: str) -> None:
        self.language_tag = language_tag
        self._counts: Counter[str] = Counter()

    @property
    def aggregate_stats(self) -> dict[str, int]:
        return dict(self._counts)

    def __call__(self, row: Mapping[str, Any], row_number: int) -> dict[str, Any]:
        self._counts["source_rows_seen"] += 1
        context = row.get("flores_passage")
        question = row.get("question")
        options = [row.get(f"mc_answer{index}") for index in range(1, 5)]
        answer_index = _choice_index(row.get("correct_answer_num"), 4, one_based=True)
        if (
            not isinstance(context, str)
            or not isinstance(question, str)
            or not question.strip()
            or any(not isinstance(option, str) for option in options)
        ):
            raise BenchmarkIsolationError("belebele_required_field_invalid")
        if answer_index is None:
            raise BenchmarkIsolationError("belebele_answer_index_invalid")
        self._counts["numeric_answer_labels"] += 1
        passage_id = row.get("flores_passage_id")
        question_number = row.get("question_number")
        group_id = (
            f"{passage_id}:{question_number}"
            if passage_id is not None and question_number is not None
            else f"{self.language_tag}:{row_number}"
        )
        return {
            "item_id": f"{self.language_tag}-{row_number}",
            "context": context,
            "question": question,
            "options": options,
            "answer_index": answer_index,
            "language": self.language_tag,
            "group_id": group_id,
        }


class BBHMultipleChoiceAdapter:
    """Parse only the explicit choice labels present in each selected prompt."""

    _MARKER = re.compile(r"(?<!\w)\(([A-J])\)\s*")

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._counts: Counter[str] = Counter()

    @property
    def aggregate_stats(self) -> dict[str, int]:
        return dict(self._counts)

    def __call__(self, row: Mapping[str, Any], row_number: int) -> dict[str, Any]:
        self._counts["source_rows_seen"] += 1
        prompt = row.get("input")
        target = row.get("target")
        if not isinstance(prompt, str) or not isinstance(target, str):
            raise BenchmarkIsolationError("bbh_required_field_invalid")
        markers = list(self._MARKER.finditer(prompt))
        labels = [match.group(1) for match in markers]
        expected_labels = [chr(ord("A") + index) for index in range(len(labels))]
        if len(labels) < 2 or labels != expected_labels:
            raise BenchmarkIsolationError("bbh_explicit_choice_parse_failed")
        options = [
            prompt[match.end():markers[index + 1].start() if index + 1 < len(markers) else len(prompt)].strip()
            for index, match in enumerate(markers)
        ]
        question = prompt[:markers[0].start()].strip()
        if not question or any(not option for option in options):
            raise BenchmarkIsolationError("bbh_explicit_choice_parse_failed")
        answer_index = _bbh_target_index(target, labels, options)
        if answer_index is None:
            raise BenchmarkIsolationError("bbh_target_mapping_failed")
        self._counts["explicit_choice_rows"] += 1
        self._counts[f"choice_count_{len(options)}"] += 1
        return {
            "item_id": f"{self.task_id}-{row_number}",
            "context": "",
            "question": question,
            "options": options,
            "answer_index": answer_index,
            "category": self.task_id,
        }


def iter_parquet_records(path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream Parquet rows; parser failures are reduced to a safe code."""
    try:
        import pyarrow.parquet as parquet

        reader = parquet.ParquetFile(path)
        for batch in reader.iter_batches(batch_size=256):
            for record in batch.to_pylist():
                if not isinstance(record, dict):
                    raise BenchmarkIsolationError("mmlu_parquet_record_not_an_object")
                yield record
    except BenchmarkIsolationError:
        raise
    except Exception:
        raise BenchmarkIsolationError("mmlu_parquet_read_failed") from None


def iter_bbh_records(path: Path) -> Iterator[Mapping[str, Any]]:
    """Yield only the selected task's evaluation examples from its pinned file."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise BenchmarkIsolationError("bbh_source_json_invalid") from None
    if not isinstance(value, dict) or not isinstance(value.get("examples"), list):
        raise BenchmarkIsolationError("bbh_source_schema_invalid")
    for row_number, record in enumerate(value["examples"], 1):
        if not isinstance(record, dict):
            raise BenchmarkIsolationError("bbh_record_not_an_object")
        yield record


def _choice_index(value: Any, option_count: int, *, one_based: bool = False) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        result = value - int(one_based)
    elif isinstance(value, str):
        normalized = value.strip()
        if normalized.isdigit():
            result = int(normalized) - int(one_based)
        elif len(normalized) == 1 and normalized.upper().isalpha():
            result = ord(normalized.upper()) - ord("A")
        else:
            return None
    else:
        return None
    return result if 0 <= result < option_count else None


def _bbh_target_index(target: str, labels: list[str], options: list[str]) -> int | None:
    normalized_target = target.strip()
    label_match = re.fullmatch(r"[\(\[]?([A-J])[\)\].:]?", normalized_target, flags=re.IGNORECASE)
    if label_match is not None:
        label = label_match.group(1).upper()
        return labels.index(label) if label in labels else None
    normalized = " ".join(normalized_target.split()).casefold()
    matches = [index for index, option in enumerate(options) if " ".join(option.split()).casefold() == normalized]
    return matches[0] if len(matches) == 1 else None
