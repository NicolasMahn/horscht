import copy
import json
import re
from collections import Counter
from pathlib import Path

from experiment import format_input, load_examples
from make_examples import make_examples
from make_expanded_examples import make_expanded_examples


def test_expanded_dataset_is_reproducible_and_nested_with_balanced_families(tmp_path):
    examples = make_expanded_examples()
    assert examples == make_expanded_examples()
    path = tmp_path / "expanded.json"
    path.write_text(json.dumps(examples))
    assert load_examples(path) == examples
    assert len(examples) == 128
    assert len({example["id"] for example in examples}) == 128
    families = {"comparison", "routing", "addition", "state"}
    for split in ("train", "probe"):
        selected = [example for example in examples if example["split"] == split]
        assert Counter(example["family"] for example in selected) == dict.fromkeys(families, 16)
        for family in families:
            rows = [example for example in selected if example["family"] == family]
            counts = Counter(example["answer_index"] for example in rows)
            assert set(counts) == set(range(len(rows[0]["options"])))
            assert max(counts.values()) - min(counts.values()) <= 1
    subset = [example for example in examples if example["train_subset_16"]]
    assert Counter(example["family"] for example in subset) == dict.fromkeys(families, 4)
    assert all(example["split"] == "train" for example in subset)
    for family in families:
        counts = Counter(example["answer_index"] for example in subset if example["family"] == family)
        assert max(counts.values()) - min(counts.values()) <= 1


def test_expanded_questions_are_disjoint_without_relying_on_choice_order():
    examples = make_expanded_examples()
    prompts = {(example["context"], example["question"]) for example in examples}
    original = {(example["context"], example["question"]) for example in make_examples()}
    assert len(prompts) == len(examples)
    original_train = {(example["context"], example["question"]) for example in make_examples()
                      if example["split"] == "train"}
    assert prompts & original == original_train
    new_prompts = {(example["context"], example["question"]) for example in examples
                   if not example["train_subset_16"]}
    assert new_prompts.isdisjoint(original)
    assert len({format_input(example) for example in examples}) == len(examples)
    for family in ("comparison", "addition"):
        pairs = [tuple(map(int, re.findall(r"\d+", example["context"])))
                 for example in examples if example["family"] == family]
        assert len(set(pairs)) == 32


def test_expanded_answers_follow_only_the_stated_facts():
    for example in make_expanded_examples():
        context = example["context"]
        match example["family"]:
            case "comparison":
                a, b = map(int, re.findall(r"\d+", context))
                expected = "Yes" if a > b else "No"
            case "addition":
                initial, added = map(int, re.findall(r"\d+", context))
                expected = str(initial + added)
            case "routing":
                rules = dict(re.findall(r"(\w+) goes to (\w+)", context))
                keyword = re.search(r"ticket keyword is (\w+)\.", context).group(1)
                assert len(rules) == 3
                assert set(rules.values()) == set(example["options"])
                expected = rules[keyword]
            case "state":
                transitions = re.findall(r"changes to (\w+)\.", context)
                assert len(transitions) == 1
                expected = transitions[-1].title()
            case _:
                raise AssertionError("Unverified family")
        assert example["options"][example["answer_index"]] == expected
        assert len(set(example["options"])) == len(example["options"])


def test_subset_annotation_is_not_exposed_to_model():
    example = make_expanded_examples()[0]
    changed = copy.deepcopy(example)
    changed["train_subset_16"] = not changed["train_subset_16"]
    assert format_input(example) == format_input(changed)


def test_nested_subset_preserves_original_training_examples_verbatim():
    root = Path(__file__).resolve().parents[1]
    original = [example for example in make_examples()
                if example["split"] == "train"]
    subset = [{key: value for key, value in example.items() if key != "train_subset_16"}
              for example in make_expanded_examples() if example["train_subset_16"]]
    assert subset == original
