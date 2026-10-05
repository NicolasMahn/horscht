"""Build answer-verified fixtures; this is a plumbing dataset, not a benchmark."""

import json
from pathlib import Path


def make_examples():
    examples = []

    def add(family, index, context, question, options, answer, explanation):
        # Rotate positions so the task cannot be solved by always picking slot zero.
        position = index % len(options)
        options = [answer] + [option for option in options if option != answer]
        options = options[-position:] + options[:-position] if position else options
        examples.append({
            "id": f"{family}-{index}",
            "split": "train" if index < 4 else "probe",
            "family": family,
            "context": context,
            "question": question,
            "options": options,
            "answer_index": options.index(answer),
            "explanation": explanation,
            "provenance": "deterministic fixture, answer computed from stated facts",
        })

    for i, (a, b) in enumerate([(3, 7), (9, 4), (2, 8), (11, 6), (13, 5), (4, 12)]):
        answer = "Yes" if a > b else "No"
        add("comparison", i, f"Box A contains {a} balls. Box B contains {b} balls.",
            "Does box A contain more balls than box B?", ["Yes", "No"], answer,
            f"{a} > {b} is {a > b}.")

    routes = {"invoice": "Billing", "crash": "Engineering", "quote": "Sales"}
    for i, keyword in enumerate(["invoice", "crash", "quote", "invoice", "quote", "crash"]):
        add("routing", i,
            "Rule: invoice goes to Billing; crash goes to Engineering; quote goes to Sales. "
            f"The ticket keyword is {keyword}.",
            "Which team should receive the ticket?" if i != 3 else "Apply the rule: which team receives this ticket?",
            list(routes.values()), routes[keyword],
            f"The stated rule maps {keyword} to {routes[keyword]}.")

    for i, (initial, added) in enumerate([(3, 4), (5, 8), (9, 6), (12, 5), (7, 11), (14, 6)]):
        result = initial + added
        add("addition", i, f"A tray has {initial} coins. Someone adds {added} coins.",
            "How many coins are now on the tray?",
            [str(result + offset) for offset in [0, -1, 1, 2]], str(result),
            f"{initial} + {added} = {result}.")

    colors = ["Red", "Blue", "Green", "Yellow"]
    for i, (start, finish) in enumerate([(0, 2), (2, 1), (1, 3), (3, 0), (2, 0), (0, 1)]):
        add("state", i,
            f"A lamp starts {colors[start].lower()}. It changes to {colors[finish].lower()}.",
            "What color is the lamp now?", colors, colors[finish],
            f"The last stated color is {colors[finish].lower()}.")
    return examples


if __name__ == "__main__":
    path = Path(__file__).parent / "data" / "smoke.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(make_examples(), indent=2) + "\n")
    print(path)
