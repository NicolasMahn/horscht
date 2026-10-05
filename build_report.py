"""Create an offline HTML report from measured comparison records."""

import argparse
import html
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from report_theme import MODELS, PALETTE

ROOT = Path(__file__).parent


def escape(value):
    return html.escape(str(value), quote=True)


def chart_base(y_title):
    fig = go.Figure()
    fig.update_layout(
        template="none", height=420, margin=dict(l=66, r=25, t=50, b=65),
        paper_bgcolor=PALETTE["paper"], plot_bgcolor=PALETTE["paper"],
        font=dict(family="system-ui, sans-serif", size=14, color=PALETTE["ink"]),
        hovermode="x unified",
        xaxis=dict(title="Head update", gridcolor=PALETTE["line"], zeroline=False),
        yaxis=dict(title=y_title, gridcolor=PALETTE["line"], zeroline=False),
        legend=dict(orientation="h", y=-0.4, x=0, font=dict(size=13)),
    )
    return fig


def make_curve(records, metric, split):
    is_loss = metric == "loss"
    fig = chart_base("Cross-entropy (nats)" if is_loss else "Accuracy")
    key = metric if split == "train" else f"probe_{metric}"
    for record in records:
        name = record["manifest"]["candidate"]
        style = MODELS[name]
        points = record["report"]["training"]["history"]
        fig.add_trace(go.Scatter(
            x=[p["step"] for p in points], y=[p[key] for p in points],
            name=style["label"], mode="lines", line=dict(color=PALETTE[style["color"]], width=2.5),
            hovertemplate=("%{y:.4f}<extra>%{fullData.name}</extra>" if is_loss
                           else "%{y:.1%}<extra>%{fullData.name}</extra>"),
        ))
    example_split = [ex for ex in records[0]["examples"] if ex["split"] == split]
    uniform = sum(math.log(len(ex["options"])) if is_loss else 1 / len(ex["options"])
                  for ex in example_split) / len(example_split)
    fig.add_hline(y=uniform, line_dash="dot", line_color=PALETTE["muted"],
                  annotation_text=f"Uniform choices: {uniform:.3f}" if is_loss else f"Uniform choices: {uniform:.1%}",
                  annotation_position="top right", annotation_font_color=PALETTE["muted"])
    if is_loss:
        fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0, y=1.15, xanchor="left",
                                           buttons=[dict(label="Linear", method="relayout", args=[{"yaxis.type": "linear", "yaxis.autorange": True}]),
                                                    dict(label="Log", method="relayout", args=[{"yaxis.type": "log", "yaxis.autorange": True}])])])
    else:
        fig.update_yaxes(range=[-0.03, 1.05], tickformat=".0%", dtick=0.25)
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id=f"{split}-{metric}", config={"responsive": True, "displaylogo": False,
                       "modeBarButtonsToRemove": ["select2d", "lasso2d"],
                       "toImageButtonOptions": {"format": "svg", "filename": f"{split}-{metric}"}})


def summary_table(records):
    rows = []
    for record in records:
        report = record["report"]
        model = record["manifest"]["candidate"]
        training = report["training"]
        probe = report["probe"]
        first_perfect = next((p["step"] for p in training["history"] if p["accuracy"] == 1), None)
        verification = record.get("reload_verification") or {}
        verified = verification.get("features_close") and verification.get("predictions_identical")
        rows.append(f'''<tr><th scope="row" class="model-{model}">{escape(MODELS[model]['label'])}</th>
          <td>{training['initial']['loss']:.3f} → {training['final']['loss']:.3f}</td>
          <td>{round(training['final']['accuracy'] * report['training_examples'])}/{report['training_examples']}</td>
          <td>{first_perfect if first_perfect is not None else '—'} / {training['steps']}</td>
          <td>{round(probe['accuracy'] * report['probe_examples'])}/{report['probe_examples']}</td>
          <td>{probe['loss']:.3f}</td>
          <td>{'Pass' if report['passed'] and verified else 'Failed check'}</td></tr>''')
    return "".join(rows)


def timing_table(records):
    return "".join(f'''<tr><th scope="row" class="model-{r['manifest']['candidate']}">{escape(MODELS[r['manifest']['candidate']]['label'])}</th>
        <td>{r['report']['feature_extraction_seconds']:.2f} s</td>
        <td>{r['report']['head_training_seconds']:.3f} s</td>
        <td>{r['report']['total_seconds']:.1f} s</td></tr>''' for r in records)


def family_table(records):
    families = sorted({ex["family"] for ex in records[0]["examples"]})
    rows = []
    for family in families:
        cells = []
        for record in records:
            ids = {ex["id"] for ex in record["examples"] if ex["family"] == family and ex["split"] == "probe"}
            predictions = [p for p in record["report"]["predictions"] if p["id"] in ids]
            correct = sum(p["expected"] == p["predicted"] for p in predictions)
            cells.append(f"<td>{correct}/{len(predictions)}</td>")
        rows.append(f"<tr><th scope='row'>{escape(family.title())}</th>{''.join(cells)}</tr>")
    return "".join(rows)


def example_rows(records):
    rows = []
    for example in sorted(records[0]["examples"], key=lambda ex: (ex["split"] != "probe", ex["id"])):
        cells = []
        any_wrong = False
        for record in records:
            prediction = next(p for p in record["report"]["predictions"] if p["id"] == example["id"])
            chosen = prediction["predicted"]
            correct = chosen == example["answer_index"]
            any_wrong |= not correct
            options = "".join(f"<li>{escape(option)}: {prob:.1%}</li>" for option, prob in zip(example["options"], prediction["probabilities"]))
            cells.append(f'''<td><details class="{'correct' if correct else 'incorrect'}"><summary>
                {'✓' if correct else '✕'} {escape(example['options'][chosen])}<br>
                <span class="probability">{prediction['probabilities'][chosen]:.1%}</span></summary>
                <ul>{options}</ul></details></td>''')
        options = " · ".join(f"{i}: {escape(option)}" for i, option in enumerate(example["options"]))
        rows.append(f'''<tr data-split="{example['split']}" data-error="{str(any_wrong).lower()}">
          <th scope="row"><span class="small">{escape(example['id'])} · {example['split']}</span>
          <p>{escape(example['context'])}</p><p>{escape(example['question'])}</p>
          <span class="small">{options}</span></th>
          <td>{escape(example['options'][example['answer_index']])}</td>{''.join(cells)}</tr>''')
    return "".join(rows)


def build(records):
    if not records:
        raise ValueError("No measured comparison records found")
    if any(record.get("status") == "failed" for record in records):
        raise ValueError("A candidate failed; inspect its run log before rebuilding the comparison")
    reference = records[0]
    for record in records:
        if record["examples"] != reference["examples"]:
            raise ValueError("Cannot compare records from different examples")
        for key in ("data_sha256", "seed", "learning_rate", "max_steps", "format", "dtype", "threads", "capacity", "device",
                    "head", "pooling", "max_tokens", "optimizer", "chat_template", "feature_dtype", "backbone_trainable"):
            if record["manifest"][key] != reference["manifest"][key]:
                raise ValueError(f"Comparison setting differs: {key}")
        history = record["report"]["training"]["history"]
        if not history or any("probe_loss" not in p or "probe_accuracy" not in p for p in history):
            raise ValueError("Report requires measured probe curves; rerun with dense monitoring")
        if [point["step"] for point in history] != list(range(record["report"]["training"]["steps"] + 1)):
            raise ValueError("History must include every measured update from zero through the final step")
    model_headers = "".join(f"<th scope='col' class='model-{r['manifest']['candidate']}'>{escape(MODELS[r['manifest']['candidate']]['label'])}</th>" for r in records)
    theme = ":root{" + "".join(f"--{name}:{value};" for name, value in PALETTE.items()) + "}"
    theme += "".join(f".model-{name}{{color:var(--{style['color']})}}" for name, style in MODELS.items())
    peaks = [(r, max(r["report"]["training"]["history"], key=lambda p: p["loss"])) for r in records]
    spikes = [(r, p) for r, p in peaks if p["step"] > 0 and p["loss"] > r["report"]["training"]["initial"]["loss"]]
    if spikes:
        peak_record, peak = max(spikes, key=lambda pair: pair[1]["loss"])
        peak_name = MODELS[peak_record["manifest"]["candidate"]]["label"]
        peak_note = f"The largest observed training-loss spike is {peak['loss']:.2f} at update {peak['step']} ({escape(peak_name)}). Endpoints alone hide this instability. The fixed learning rate may contribute; no learning-rate sweep was performed."
    else:
        peak_note = "No observed training loss exceeded its initial value. Inspect the full curves for plateaus, reversals and differences in learning speed."
    manifest = reference["manifest"]
    dtype = {"bfloat16": "BF16", "float32": "FP32"}[manifest["dtype"]]
    settings = f"{manifest['device'].upper()} · {dtype} backbone / FP32 head · {manifest['threads']} threads · AdamW · learning rate {manifest['learning_rate']} · seed {manifest['seed']}"
    train_count = sum(ex["split"] == "train" for ex in reference["examples"])
    probe_count = sum(ex["split"] == "probe" for ex in reference["examples"])
    records_json = json.dumps(records, allow_nan=False).replace("<", "\\u003c")
    replacements = {
        "THEME": theme,
        "PLOTLY": get_plotlyjs(),
        "GENERATED": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "MODEL_COUNT": str(len(records)),
        "DATE": escape(reference["manifest"]["started_at"][:10]),
        "TRAIN_COUNT": str(train_count),
        "PROBE_COUNT": str(probe_count),
        "EXAMPLE_COUNT": str(train_count + probe_count),
        "TRAIN_STEP": f"{100 / train_count:g}",
        "PROBE_STEP": f"{100 / probe_count:g}",
        "CAPACITY": str(manifest["capacity"]),
        "SLOTS": ", ".join(str(slot) for slot in reference["report"]["exercised_output_positions"]),
        "MAX_STEPS": f"{manifest['max_steps']:,}",
        "SUMMARY_ROWS": summary_table(records),
        "TRAIN_LOSS": make_curve(records, "loss", "train"),
        "PROBE_LOSS": make_curve(records, "loss", "probe"),
        "TRAIN_ACCURACY": make_curve(records, "accuracy", "train"),
        "PROBE_ACCURACY": make_curve(records, "accuracy", "probe"),
        "MODEL_HEADERS": model_headers,
        "FAMILY_ROWS": family_table(records),
        "TIMING_ROWS": timing_table(records),
        "EXAMPLE_ROWS": example_rows(records),
        "SETTINGS": escape(settings),
        "PEAK_NOTE": peak_note,
        "RECORDS_JSON": records_json,
        "REVISIONS": "".join(f"<li><strong>{escape(MODELS[r['manifest']['candidate']]['label'])}</strong><br>{escape(r['manifest']['repo'])}<br><code>{escape(r['manifest']['revision'])}</code><br>{escape(r['manifest']['started_at'])}</li>" for r in records),
    }
    template = (ROOT / "report_template.html").read_text()
    # Replace the template in one pass so data cannot be interpreted as template syntax.
    import re
    return re.sub(r"@@([A-Z_]+)@@", lambda match: replacements[match.group(1)], template)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "comparison")
    parser.add_argument("--output", type=Path, default=ROOT / "report.html")
    args = parser.parse_args()
    records = [json.loads((args.results / f"{name}.json").read_text()) for name in MODELS
               if (args.results / f"{name}.json").exists()]
    args.output.write_text(build(records))
    print(f"Wrote {args.output} ({len(records)} measured models)")


if __name__ == "__main__":
    main()
