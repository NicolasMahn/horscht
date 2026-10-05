"""Create reviewed high-cardinality variants without changing source rows.

This opt-in utility only adds choices when each added option has an explicit
impossibility reason. It does not discover or verify semantic impossibility.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
from collections.abc import Mapping, Sequence
from typing import Any

from dataset_factory import probability_v2


MAX_CHOICES = probability_v2.MAX_CHOICES
SOFT_AUGMENTABLE_SOURCES = frozenset({"exact_derived", "explicit_stochastic_process"})


class AugmentationError(ValueError):
    """Raised when a choice extension cannot preserve its target contract."""


class BalancedTargetSlotSchedule:
    """Cycle through output slots evenly, with a seeded phase per cardinality.

    The schedule only depends on the option count and seed, so topic and
    language allocations cannot systematically determine the target position.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self._positions: dict[int, int] = {}
        self._phases: dict[int, tuple[int, int]] = {}

    def next_slot(self, choice_count: int) -> int:
        if isinstance(choice_count, bool) or not isinstance(choice_count, int):
            raise AugmentationError("choice_count must be an integer")
        if not 2 <= choice_count <= MAX_CHOICES:
            raise AugmentationError(f"choice_count must be between 2 and {MAX_CHOICES}")
        if choice_count not in self._phases:
            rng = random.Random(f"{self.seed}:{choice_count}")
            offset = rng.randrange(choice_count)
            strides = [step for step in range(1, choice_count) if math.gcd(step, choice_count) == 1]
            stride = rng.choice(strides)
            self._phases[choice_count] = (offset, stride)
            self._positions[choice_count] = 0
        offset, stride = self._phases[choice_count]
        position = self._positions[choice_count]
        self._positions[choice_count] = position + 1
        return (offset + position * stride) % choice_count

    def state_dict(self) -> dict[str, Any]:
        """Return enough state to resume each cardinality's balanced cycle."""
        return {
            "schema_version": "balanced-target-slot-schedule-v1",
            "seed": self.seed,
            "positions": {str(key): value for key, value in sorted(self._positions.items())},
            "phases": {
                str(key): {"offset": value[0], "stride": value[1]}
                for key, value in sorted(self._phases.items())
            },
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "BalancedTargetSlotSchedule":
        if state.get("schema_version") != "balanced-target-slot-schedule-v1":
            raise AugmentationError("unsupported target-slot schedule state")
        schedule = cls(seed=state.get("seed", 0))
        positions = state.get("positions", {})
        phases = state.get("phases", {})
        if not isinstance(positions, Mapping) or not isinstance(phases, Mapping):
            raise AugmentationError("target-slot schedule state is malformed")
        for raw_count, raw_position in positions.items():
            count = int(raw_count)
            if isinstance(raw_position, bool) or not isinstance(raw_position, int) or raw_position < 0:
                raise AugmentationError("target-slot position must be a non-negative integer")
            phase = phases.get(str(count))
            if not isinstance(phase, Mapping):
                raise AugmentationError("target-slot schedule phase is missing")
            offset, stride = phase.get("offset"), phase.get("stride")
            if (
                not isinstance(offset, int)
                or isinstance(offset, bool)
                or not 0 <= offset < count
                or not isinstance(stride, int)
                or isinstance(stride, bool)
                or not 1 <= stride < count
                or math.gcd(stride, count) != 1
            ):
                raise AugmentationError("target-slot phase is invalid")
            schedule._positions[count] = raw_position
            schedule._phases[count] = (offset, stride)
        return schedule


def augmentation_id(source_id: str, variant_key: str) -> str:
    """Derive a stable child ID while keeping related rows in one problem group."""
    if not isinstance(source_id, str) or not source_id.strip():
        raise AugmentationError("source_id must be a non-empty string")
    if not isinstance(variant_key, str) or not variant_key.strip():
        raise AugmentationError("variant_key must be a non-empty string")
    digest = hashlib.sha256(f"{source_id}\0{variant_key}".encode("utf-8")).hexdigest()[:20]
    return f"probability-v2-augmentation-{digest}"


def _validate_review(proof: Mapping[str, Any], *, allow_fixture_proof: bool) -> None:
    reviewer = proof.get("reviewer")
    status = proof.get("review_status")
    accepted = {"human_verified"}
    if allow_fixture_proof:
        accepted.add("deterministic_fixture")
    if status not in accepted or not isinstance(reviewer, str) or not reviewer.strip():
        raise AugmentationError("impossibility proof must name a reviewer or an allowed test oracle")
    quote = proof.get("context_quote")
    if not isinstance(quote, str) or not quote.strip():
        raise AugmentationError("impossibility proof must quote a context statement")


def _validate_added_options(
    source_options: Sequence[str], additions: Sequence[Mapping[str, Any]]
) -> list[str]:
    if not isinstance(additions, Sequence) or isinstance(additions, (str, bytes)) or not additions:
        raise AugmentationError("at least one impossible option is required")
    texts: list[str] = []
    for item in additions:
        if not isinstance(item, Mapping):
            raise AugmentationError("each added option needs text and an impossibility reason")
        text = item.get("text")
        reason = item.get("reason")
        if not isinstance(text, str) or not text.strip():
            raise AugmentationError("added option text must be non-empty")
        if not isinstance(reason, str) or not reason.strip():
            raise AugmentationError("each added option needs an explicit impossibility reason")
        texts.append(text)
    normalized = [probability_v2.progress.normalize(text) for text in [*source_options, *texts]]
    if len(normalized) != len(set(normalized)):
        raise AugmentationError("added options must remain distinct from all existing choices")
    if len(normalized) > MAX_CHOICES:
        raise AugmentationError(f"augmented row exceeds the {MAX_CHOICES}-choice head capacity")
    return texts


def _targeted_permutation(
    probabilities: Sequence[float], choice_count: int, target_slot: int, seed: int
) -> list[int]:
    if isinstance(target_slot, bool) or not isinstance(target_slot, int) or not 0 <= target_slot < choice_count:
        raise AugmentationError("target_slot must refer to an available output position")
    positive = [index for index, value in enumerate(probabilities) if value > 0]
    if not positive:
        raise AugmentationError("a target distribution must have positive mass")
    selected = positive[seed % len(positive)]
    remaining = [index for index in range(choice_count) if index != selected]
    random.Random(seed).shuffle(remaining)
    remaining.insert(target_slot, selected)
    return remaining


def augment_with_impossible_options(
    source_row: Mapping[str, Any],
    additions: Sequence[Mapping[str, Any]],
    *,
    proof: Mapping[str, Any],
    target_slot: int,
    seed: int,
    variant_key: str,
    allow_fixture_proof: bool = False,
) -> dict[str, Any]:
    """Return a high-cardinality child row; never mutate or write the source.

    Soft rows are limited to exact/stochastic targets with an explicit,
    reviewed closed-world outcome set. Empirical and teacher vectors are
    rejected because an unobserved or low-estimate choice is not proof of
    impossibility. Hard rows require a reviewed impossibility proof per option.
    """
    if not isinstance(source_row, Mapping):
        raise AugmentationError("source_row must be a mapping")
    original_errors = probability_v2.validate_v2_row(dict(source_row))
    if original_errors:
        raise AugmentationError(f"source row fails the v2 contract: {original_errors[:3]}")
    if not isinstance(proof, Mapping):
        raise AugmentationError("proof must be a mapping")
    _validate_review(proof, allow_fixture_proof=allow_fixture_proof)
    if proof["context_quote"] not in source_row.get("context", ""):
        raise AugmentationError("proof quote must appear verbatim in the source context")

    source_id = source_row["id"]
    child_id = augmentation_id(source_id, variant_key)
    if child_id == source_id:
        raise AugmentationError("augmentation ID must differ from the source ID")
    source_options = source_row["options"]
    added_texts = _validate_added_options(source_options, additions)
    choice_count = len(source_options) + len(added_texts)
    source_kind = source_row.get("distribution_source_kind")
    if source_kind is not None and source_kind not in SOFT_AUGMENTABLE_SOURCES:
        raise AugmentationError("only exact/stochastic soft targets can receive added zero-mass choices")

    child = copy.deepcopy(dict(source_row))
    child["id"] = child_id
    child["choice_count"] = choice_count
    child["options"] = [*source_options, *added_texts]
    child["choice_ids"] = probability_v2.stable_choice_ids(child_id, child["options"])
    old_choice_ids = list(source_row["choice_ids"])
    new_choice_ids = list(child["choice_ids"])
    choice_id_map = dict(zip(old_choice_ids, new_choice_ids[: len(old_choice_ids)]))
    abstention_id = source_row.get("abstention_choice_id")
    if abstention_id is not None:
        child["abstention_choice_id"] = choice_id_map.get(abstention_id)
        if child["abstention_choice_id"] is None:
            raise AugmentationError("source abstention choice does not map to a child option")

    if source_kind is not None:
        old_ids = old_choice_ids
        old_probabilities = list(source_row["target_distribution"])
        possible_ids = [choice_id for choice_id, value in zip(old_ids, old_probabilities) if value > 0]
        declared_possible = proof.get("possible_choice_ids")
        if proof.get("kind") != "explicit_closed_world_outcomes" or declared_possible != possible_ids:
            raise AugmentationError("soft augmentation needs the exact positive support listed in a closed-world proof")
        added_rational = {"numerator": 0, "denominator": 1}
        evidence = child.get("distribution_evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("probabilities"), list):
            raise AugmentationError("exact/stochastic evidence must contain rational probabilities")
        if len(evidence["probabilities"]) != len(source_options):
            raise AugmentationError("source rational probabilities are not parallel to its options")
        evidence["probabilities"] = [*evidence["probabilities"], *[dict(added_rational) for _ in added_texts]]
        for key in ("counts", "favorable_counts"):
            vector = evidence.get(key)
            if vector is not None:
                if not isinstance(vector, list) or len(vector) != len(source_options):
                    raise AugmentationError(f"source evidence {key} is not parallel to its options")
                evidence[key] = [*vector, *([0] * len(added_texts))]
        child["target_distribution"] = [*old_probabilities, *([0.0] * len(added_texts))]
        child["distribution_evidence"] = evidence
        permutation = _targeted_permutation(child["target_distribution"], choice_count, target_slot, seed)
    else:
        if proof.get("kind") != "reviewed_impossible_distractors":
            raise AugmentationError("hard-label augmentation needs a reviewed impossible-distractor proof")
        answer_index = source_row.get("answer_index")
        correct_choice_id = source_row["choice_ids"][answer_index]
        if proof.get("correct_choice_id") != correct_choice_id:
            raise AugmentationError("hard-label proof does not identify the source answer")
        if len(additions) != len(proof.get("distractor_proofs", [])):
            raise AugmentationError("hard-label proof must cover every added distractor")
        if any(
            item.get("text") != record.get("text")
            or item.get("reason") != record.get("reason")
            for item, record in zip(additions, proof["distractor_proofs"])
        ):
            raise AugmentationError("hard-label proof entries must match every added distractor")
        one_hot = [0.0] * len(source_options)
        one_hot[answer_index] = 1.0
        permutation = _targeted_permutation([*one_hot, *([0.0] * len(added_texts))], choice_count, target_slot, seed)

    if source_kind is not None:
        child["distribution_evidence"]["choice_ids"] = list(child["choice_ids"])
    else:
        child.pop("distribution_evidence", None)
    child["_augmentation"] = {
        "schema_version": "choice-augmentation-v1",
        "source_question_id": source_id,
        "method": "reviewed_closed_world_impossibles",
        "source_choice_count": len(source_options),
        "added_option_count": len(added_texts),
        "target_slot": target_slot,
        "proof": copy.deepcopy(dict(proof)),
        "added_options": copy.deepcopy(list(additions)),
        "mass_training_status": "requires_review",
    }
    result = probability_v2.shuffle_choices(child, permutation)
    errors = probability_v2.validate_v2_row(result)
    if errors:
        raise AugmentationError(f"augmented row fails the v2 contract: {errors[:3]}")
    return result
