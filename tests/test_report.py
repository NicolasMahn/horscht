import copy
import json
import math
import re

import pytest

import build_report
from make_examples import make_examples


@pytest.fixture
def record():
    examples = make_examples()
    history = [
        {"step": 0, "loss": 1.2, "accuracy": 0.25, "probe_loss": 1.3, "probe_accuracy": 0.25},
        {"step": 1, "loss": 0.01, "accuracy": 1.0, "probe_loss": 1.1, "probe_accuracy": 0.5},
    ]
    return {
        "manifest": {
            "candidate": "ministral-base", "data_sha256": "fixture", "seed": 42,
            "learning_rate": 0.01, "max_steps": 1000, "format": "fixture",
            "dtype": "bfloat16", "threads": 1, "repo": "fixture/model",
            "revision": "fixture-revision", "started_at": "2026-09-28T00:00:00+00:00",
            "capacity": 265, "device": "cpu", "head": "non-affine LayerNorm followed by linear projection",
            "pooling": "last non-padding token", "max_tokens": 256,
            "optimizer": "AdamW, weight_decay=0", "chat_template": False,
            "feature_dtype": "float32", "backbone_trainable": False,
        },
        "examples": examples,
        "report": {
            "training": {"initial": {"loss": 1.2, "accuracy": 0.25},
                         "final": {"loss": 0.01, "accuracy": 1.0}, "history": history, "steps": 1},
            "probe": {"loss": 1.1, "accuracy": 0.5}, "training_examples": 16,
            "probe_examples": 8, "passed": True, "feature_extraction_seconds": 1.0,
            "head_training_seconds": 0.1, "total_seconds": 2.0,
            "exercised_output_positions": [0, 1, 2, 3],
            "predictions": [
                {"id": ex["id"], "expected": ex["answer_index"],
                 "predicted": ex["answer_index"],
                 "probabilities": [1 / len(ex["options"])] * len(ex["options"])}
                for ex in examples
            ],
        },
        "reload_verification": {"features_close": True, "predictions_identical": True},
    }


@pytest.mark.parametrize("change", ["examples", "seed", "learning_rate", "dtype", "missing_probe",
                                    "capacity", "pooling", "optimizer", "chat_template", "sparse_history"])
def test_report_rejects_incomparable_or_unmeasured_data(record, change):
    other = copy.deepcopy(record)
    if change == "examples":
        other["examples"][0]["question"] = "Different question"
    elif change == "missing_probe":
        del other["report"]["training"]["history"][0]["probe_loss"]
    elif change == "sparse_history":
        other["report"]["training"]["history"][1]["step"] = 2
        other["report"]["training"]["steps"] = 2
    else:
        other["manifest"][change] = "different"
    with pytest.raises(ValueError):
        build_report.build([record, other])


def test_report_escapes_visible_data_and_embedded_json_without_template_expansion(record, monkeypatch):
    payload = '</script><script>alert("test")</script> @@PLOTLY@@ <b>literal</b>'
    record["examples"][0]["context"] = payload
    monkeypatch.setattr(build_report, "get_plotlyjs", lambda: "/* offline plotly stub */")
    monkeypatch.setattr(build_report, "make_curve", lambda *args: "<div>chart</div>")
    rendered = build_report.build([record])
    assert payload not in rendered
    assert build_report.escape(payload) in rendered
    embedded = re.search(r'<script id="measurements" type="application/json">(.*?)</script>', rendered, re.S)
    assert embedded is not None
    assert "<" not in embedded.group(1)
    decoded = json.loads(embedded.group(1))
    assert decoded[0]["examples"][0]["context"] == payload
    assert "No observed training loss exceeded its initial value" in rendered
    assert "Endpoints alone hide this instability" not in rendered

    record["manifest"].update(dtype="float32", capacity=300, max_steps=500, started_at="2027-01-02T00:00:00+00:00")
    changed = build_report.build([record])
    assert "FP32 backbone / FP32 head" in changed
    assert "<strong>300</strong> output slots" in changed
    assert "or at 500 updates" in changed
    assert "measured candidates · 2027-01-02" in changed

    record["report"]["training"]["history"][1]["loss"] = 5.0
    spiked = build_report.build([record])
    assert "training-loss spike is 5.00 at update 1" in spiked


def test_curves_use_measured_points_correct_uniform_baselines_and_probe_family_counts(record, monkeypatch):
    figures = []
    monkeypatch.setattr(build_report.pio, "to_html", lambda figure, **kwargs: figures.append(figure) or "chart")
    for split in ("train", "probe"):
        for metric in ("loss", "accuracy"):
            build_report.make_curve([record], metric, split)
            figure = figures[-1]
            points = record["report"]["training"]["history"]
            key = metric if split == "train" else f"probe_{metric}"
            assert list(figure.data[0].x) == [p["step"] for p in points]
            assert list(figure.data[0].y) == [p[key] for p in points]
            examples = [ex for ex in record["examples"] if ex["split"] == split]
            expected = sum(math.log(len(ex["options"])) if metric == "loss" else 1 / len(ex["options"])
                           for ex in examples) / len(examples)
            assert figure.layout.shapes[0].y0 == pytest.approx(expected)
    probe_example = next(ex for ex in record["examples"] if ex["split"] == "probe")
    for prediction in record["report"]["predictions"]:
        if prediction["id"] == probe_example["id"]:
            prediction["predicted"] = (prediction["expected"] + 1) % len(probe_example["options"])
    family_rows = build_report.family_table([record])
    assert family_rows.count("<td>1/2</td>") == 1
    assert family_rows.count("<td>2/2</td>") == 3
