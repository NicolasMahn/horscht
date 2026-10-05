"""Freeze novel-operation probes before scoring; these are not pretraining holdouts."""

import hashlib
import itertools
import json
import random
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
SEED = 20260929


def make_transfer_examples():
    rng = random.Random(SEED)
    examples = []

    def add(family, index, context, question, options, answer, facts, position=None):
        position = index % len(options) if position is None else position
        distractors = [option for option in options if option != answer]
        rng.shuffle(distractors)
        distractors.insert(position, answer)
        examples.append({
            "id": f"transfer-{family}-{index:02d}", "split": "probe",
            "family": family, "context": context, "question": question,
            "options": distractors, "answer_index": position, "facts": facts,
            "provenance": "deterministic novel-operation fixture; mechanically computed answer",
        })

    directions = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
    diagonals = {(1, 1): "Northeast", (1, -1): "Southeast",
                 (-1, -1): "Southwest", (-1, 1): "Northwest"}
    turns = [(a, b) for a, va in directions.items() for b, vb in directions.items()
             if va[0] * vb[0] + va[1] * vb[1] == 0]
    rng.shuffle(turns)
    for i in range(16):
        first, second = turns[i % len(turns)]
        a, b, c = (f"{name}{i}" for name in ("A", "B", "C"))
        dx = directions[first][0] + directions[second][0]
        dy = directions[first][1] + directions[second][1]
        add("spatial_composition", i,
            f"On a map, {b} is one unit {first} of {a}. {c} is one unit {second} of {b}.",
            f"In which direction is {c} from {a}?", list(diagonals.values()), diagonals[dx, dy],
            {"first_direction": first, "second_direction": second})

    for i in range(16):
        word = "".join(rng.sample("abcdefghjkmnpqrstuvwxyz", 6))
        answer = word[::-1]
        options = [answer, word, word[1:] + word[0], word[-1] + word[:-1]]
        if i < 8:
            options = options[:2]
        add("string_reversal", i, f"The string is {word}. Treat each letter as one character.",
            "Which string has the characters in reverse order?", options, answer,
            {"input": word, "operation": "reverse"})

    def render_set(values):
        return "{" + ", ".join(sorted(values)) + "}"

    for i in range(16):
        universe = rng.sample(["ash", "birch", "cedar", "elm", "fir", "oak", "pine", "yew"], 5)
        left, right = set(universe[:3]), set(universe[2:])
        operation = "intersection" if i % 2 == 0 else "difference"
        answer = left & right if operation == "intersection" else left - right
        option_sets = [left & right, left - right, right - left, left | right]
        question = ("Which set contains exactly the items in both A and B?" if operation == "intersection"
                    else "Which set contains exactly the items in A but not in B?")
        add("set_operations", i, f"Set A = {render_set(left)}. Set B = {render_set(right)}.",
            question, [render_set(s) for s in option_sets], render_set(answer),
            {"left": sorted(left), "right": sorted(right), "operation": operation}, position=(i // 2) % 4)

    # Balance semantic labels separately from rotating answer slots. Unstated facts
    # remain unknown: absence never silently becomes a negative assertion.
    cases = {label: [] for label in ("True", "False", "Unknown")}
    for p, q, op in itertools.product((True, False, None), (True, False, None), ("and", "or")):
        outcomes = {((a and b) if op == "and" else (a or b))
                    for a in ((True, False) if p is None else (p,))
                    for b in ((True, False) if q is None else (q,))}
        label = "Unknown" if len(outcomes) == 2 else ("True" if True in outcomes else "False")
        cases[label].append((p, q, op))
    used = {label: 0 for label in cases}
    labels = ["True", "False", "Unknown"] * 5 + ["True"]
    rng.shuffle(labels)
    for i, label in enumerate(labels):
        p, q, op = cases[label][used[label] % len(cases[label])]
        used[label] += 1
        subject = f"Device D{i}"
        statements = [f"{subject} is {'not ' if not value else ''}{property_}."
                      for value, property_ in ((p, "active"), (q, "warm")) if value is not None]
        context = " ".join(statements) or f"No facts about {subject.lower()} are given."
        context += " Unstated properties are unknown. Use inclusive OR."
        add("propositional_entailment", i, context,
            f"Is the claim '{subject} is active {op.upper()} warm' true, false, or unknown?",
            ["True", "False", "Unknown"], label,
            {"active": p, "warm": q, "operator": op})
    return examples


def write_dataset():
    examples = make_transfer_examples()
    path = ROOT / "data" / "transfer.json"
    path.parent.mkdir(exist_ok=True)
    payload = (json.dumps(examples, indent=2) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    manifest_path = path.parent / "transfer-manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    previous_digest = previous.get("data_sha256", previous.get("sha256"))
    if previous_digest is not None and previous_digest != digest:
        raise ValueError("Transfer dataset is locked; create a versioned dataset for changes.")
    path.write_bytes(payload)
    manifest = {
        "schema": 1, "dataset": "transfer.json", "data_sha256": digest,
        "locked_at": previous.get("locked_at", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        "generator_seed": SEED, "examples": len(examples), "split": "probe",
        "families": {family: sum(row["family"] == family for row in examples)
                     for family in sorted({row["family"] for row in examples})},
        "purpose": "Novel operations relative to the four training templates; not a pretraining holdout.",
        "selection_policy": "Diagnostic only; do not select seeds, checkpoints, or models using these scores.",
        "frozen_before_measurement": True,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path


if __name__ == "__main__":
    print(write_dataset())
