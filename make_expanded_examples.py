"""Deterministic template-transfer fixtures, fixed before measuring any model.

The nested 16/64 training comparison shares all 64 held-out questions. This
expands coverage inside four templates; it is not an independent benchmark.
"""

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def make_expanded_examples():
    original = json.loads((ROOT / "data" / "smoke.json").read_text())
    examples = [dict(example, train_subset_16=True) for example in original
                if example["split"] == "train"]
    rng = random.Random(20260928)

    def add(family, index, context, question, options, answer, explanation, position=None):
        if position is None:
            position = index % len(options)
        distractors = [option for option in options if option != answer]
        rng.shuffle(distractors)
        choices = distractors[:]
        choices.insert(position, answer)
        examples.append({
            "id": f"expanded-{family}-{index:02d}",
            "split": "train" if index < 16 else "probe",
            "train_subset_16": False,
            "family": family,
            "context": context,
            "question": question,
            "options": choices,
            "answer_index": position,
            "explanation": explanation,
            "provenance": "deterministic expanded fixture; answer computed from stated facts",
        })

    def new_numeric_pairs(family):
        old_contexts = {example["context"] for example in original if example["family"] == family}
        template = ("Box A contains {} balls. Box B contains {} balls." if family == "comparison"
                    else "A tray has {} coins. Someone adds {} coins.")
        pairs = [(a, b) for a in range(2, 18) for b in range(2, 18)
                 if (family != "comparison" or a != b)
                 and template.format(a, b) not in old_contexts]
        rng.shuffle(pairs)
        return pairs

    pairs = new_numeric_pairs("comparison")
    less = iter(pair for pair in pairs if pair[0] < pair[1])
    greater = iter(pair for pair in pairs if pair[0] > pair[1])
    for i in range(4, 32):
        # Truth and answer position vary independently in each group of four.
        a, b = next(greater if (i // 2) % 2 else less)
        answer = "Yes" if a > b else "No"
        add("comparison", i,
            f"Box A contains {a} balls. Box B contains {b} balls.",
            "Does box A contain more balls than box B?", ["Yes", "No"], answer,
            f"{a} > {b} is {a > b}.")

    keywords = ["refund", "login", "delivery", "renewal", "password", "booking"]
    teams = ["Support", "Accounts", "Operations", "Dispatch", "Security", "Reservations"]
    seen_rules = set()
    for i in range(4, 32):
        while True:
            selected_keywords = rng.sample(keywords, 3)
            selected_teams = rng.sample(teams, 3)
            mapping = tuple(zip(selected_keywords, selected_teams))
            if mapping not in seen_rules:
                seen_rules.add(mapping)
                break
        keyword, answer = mapping[(i // 3 + i) % 3]
        rules = "; ".join(f"{key} goes to {team}" for key, team in mapping)
        add("routing", i, f"Rule: {rules}. The ticket keyword is {keyword}.",
            "Which team should receive the ticket?", selected_teams, answer,
            f"The stated rule maps {keyword} to {answer}.")

    for i, (initial, added) in enumerate(new_numeric_pairs("addition")[:28], start=4):
        result = initial + added
        add("addition", i,
            f"A tray has {initial} coins. Someone adds {added} coins.",
            "How many coins are now on the tray?",
            [str(result + offset) for offset in (0, -1, 1, 2)], str(result),
            f"{initial} + {added} = {result}.")

    colors = ["Red", "Blue", "Green", "Yellow"]
    lamp_names = ["Ash", "Birch", "Cedar", "Elm", "Maple", "Oak", "Pine", "Willow"]
    for i in range(4, 32):
        # Names distinguish questions while preserving the original single transition.
        finish = i % 4
        start = (finish + 1 + (i // 4) % 3) % 4
        position = (i + i // 4) % 4
        add("state", i,
            f"A lamp named {lamp_names[i // 4]} starts {colors[start].lower()}. "
            f"It changes to {colors[finish].lower()}.",
            "What color is the lamp now?", colors, colors[finish],
            f"The last stated color is {colors[finish].lower()}.", position=position)

    return examples


if __name__ == "__main__":
    path = ROOT / "data" / "expanded.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(make_expanded_examples(), indent=2) + "\n")
    print(path)
