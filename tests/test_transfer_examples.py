"""Check answers independently, including open-world entailment semantics."""

import hashlib
import itertools
import json

import pytest
from collections import Counter, defaultdict

from experiment import format_input
from make_transfer_examples import ROOT, make_transfer_examples


def test_frozen_dataset_reproducible_and_manifest_matches():
    if not (ROOT / "data/transfer.json").is_file():
        pytest.skip("Historical frozen transfer dataset is not installed")
    payload = (ROOT / "data/transfer.json").read_bytes()
    examples = json.loads(payload)
    manifest = json.loads((ROOT / "data/transfer-manifest.json").read_text())
    assert examples == make_transfer_examples() == make_transfer_examples()
    assert manifest["data_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["frozen_before_measurement"] is True
    assert len(examples) == len({row["id"] for row in examples}) == 64
    assert set(Counter(row["family"] for row in examples).values()) == {16}
    assert len({format_input(row) for row in examples}) == 64
    original = json.loads((ROOT / "data/expanded.json").read_text())
    assert {row["family"] for row in examples}.isdisjoint(row["family"] for row in original)
    assert {format_input(row) for row in examples}.isdisjoint(format_input(row) for row in original)


def test_choice_positions_balanced_and_metadata_not_in_prompt():
    positions = defaultdict(Counter)
    for row in make_transfer_examples():
        assert row["split"] == "probe"
        assert 2 <= len(row["options"]) <= 4
        assert len(set(row["options"])) == len(row["options"])
        positions[row["family"], len(row["options"])][row["answer_index"]] += 1
        prompt = format_input(row)
        changed = dict(row, answer_index=999, facts={"secret": "LABEL_LEAK"}, provenance="LABEL_LEAK")
        assert format_input(changed) == prompt
        assert len(prompt.split()) < 100
    for (_, count), frequencies in positions.items():
        assert set(frequencies) == set(range(count))
        assert max(frequencies.values()) - min(frequencies.values()) <= 1


def test_answers_from_structured_facts():
    for row in make_transfer_examples():
        facts = row["facts"]
        family = row["family"]
        actual = row["options"][row["answer_index"]]
        if family == "spatial_composition":
            # Each pair has one horizontal and one vertical displacement.
            directions = {facts["first_direction"], facts["second_direction"]}
            vertical = "North" if "north" in directions else "South"
            horizontal = "east" if "east" in directions else "west"
            assert actual == vertical + horizontal
            assert all(direction in row["context"] for direction in directions)
        elif family == "string_reversal":
            assert list(actual) == list(reversed(facts["input"]))
            assert facts["input"] in row["context"]
        elif family == "set_operations":
            chosen = set(actual.removeprefix("{").removesuffix("}").split(", "))
            expected = {item for item in facts["left"]
                        if (item in facts["right"]) == (facts["operation"] == "intersection")}
            assert chosen == expected
        else:
            assert family == "propositional_entailment"
            # Evaluate the claim across every world consistent with the facts.
            possibilities = []
            for active, warm in itertools.product((False, True), repeat=2):
                if facts["active"] is not None and active != facts["active"]:
                    continue
                if facts["warm"] is not None and warm != facts["warm"]:
                    continue
                possibilities.append(all((active, warm)) if facts["operator"] == "and"
                                     else any((active, warm)))
            expected = "True" if all(possibilities) else "False" if not any(possibilities) else "Unknown"
            assert actual == expected


def test_set_operation_does_not_predict_answer_position():
    positions = defaultdict(Counter)
    for row in make_transfer_examples():
        if row["family"] == "set_operations":
            positions[row["facts"]["operation"]][row["answer_index"]] += 1
    assert all(counts == Counter({0: 2, 1: 2, 2: 2, 3: 2}) for counts in positions.values())
