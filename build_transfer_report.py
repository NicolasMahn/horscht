"""Build a standalone exploratory report; transfer scores never select models."""

import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import mean, stdev

from report_theme import MODELS, PALETTE

ROOT = Path(__file__).parent
CONDITIONS = {'initial': 'Random initial head', '16': '16 training questions', '64': '64 training questions'}


def summary(values, percent=True):
    scale = 100 if percent else 1
    precision = 1 if percent else 3
    return f'{mean(values) * scale:.{precision}f} ± {stdev(values) * scale:.{precision}f}'


def build(records):
    if not records:
        raise ValueError('No transfer results')
    reference = records[0]
    for record in records:
        if record['status'] != 'complete' or record['experiment'] != 'novel-task-transfer':
            raise ValueError('Incomplete transfer results')
        for key in ('dataset_manifest', 'source_data_sha256', 'settings', 'seeds'):
            if record['manifest'][key] != reference['manifest'][key]:
                raise ValueError(f'Incompatible transfer results: {key}')
        if record['examples'] != reference['examples']:
            raise ValueError('Transfer examples differ')
        if [r['seed'] for r in record['runs']] != list(range(42, 52)):
            raise ValueError('Expected ten matched seeds')
    css = ''.join(f'--{key}:{value};' for key, value in PALETTE.items())
    parts = [f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Novel-task transfer · decision heads</title><style>:root{{{css}}}
body{{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.55 system-ui,sans-serif}}main{{max-width:1250px;margin:auto;padding:32px 24px}}a{{color:var(--blue)}}section{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:24px;margin:24px 0}}h1{{font-size:30px}}h2{{font-size:21px}}p{{max-width:1000px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}th,td{{padding:11px 14px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}thead th{{font-size:14px;color:var(--muted)}}tbody th{{white-space:nowrap}}summary{{cursor:pointer}}.muted{{color:var(--muted)}}code{{overflow-wrap:anywhere}}select,button{{font:inherit;padding:6px 12px}}.chart{{display:grid;grid-template-columns:190px 1fr;gap:8px 16px;align-items:center}}.track{{background:var(--canvas);position:relative;height:26px}}.bar{{height:100%;background:var(--blue)}}.chance{{position:absolute;top:0;bottom:0;border-left:2px dashed var(--ink)}}.bar-label{{position:absolute;left:8px;top:0;color:var(--ink);font-size:14px;background:var(--paper)}}@media(max-width:600px){{main{{padding:16px 10px}}section{{padding:16px}}.chart{{grid-template-columns:1fr}}}}</style><main>
<nav><a href="/">Original experiment</a> · <a href="/size">16 versus 64 questions</a> · Novel-task transfer</nav>
<h1>Does the learned head transfer to new tasks?</h1>
<p>64 untouched questions across four task families absent from our training fixtures. Existing saved heads are evaluated once at their endpoints: no transfer gradients, stopping decisions, or model selection. {len(records)} models × 10 paired seeds × two training sizes, plus each seed’s random initial head.</p>
<p><strong>This tests transfer of our small classification head, conditional on a pretrained backbone.</strong> The operations may be familiar from pretraining. Success would not show that tiny training created new reasoning abilities; failure would not show the backbone cannot solve these questions through generation.</p>
<section><h2>Accuracy across seeds</h2><p>Percent correct, mean ± sample standard deviation. Each seed sees the same 64 questions. The spread describes initialization variation, not uncertainty over new questions.</p>
<p>Uniform random choice: <strong>{reference['uniform']['accuracy'] * 100:.1f}%</strong>. The randomized head uses the same initialization seed as its trained counterparts; it is a stronger control for feature and answer-position biases than chance alone.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Initial head</th><th>16 trained</th><th>64 trained</th><th>Initial → 16 (pp)</th><th>Initial → 64 (pp)</th><th>16 → 64 (pp)</th></tr></thead><tbody>''']
    for record in records:
        label = MODELS[record['manifest']['candidate']]['label']
        cells = [summary([run['conditions'][key]['accuracy'] for run in record['runs']]) for key in CONDITIONS]
        cells += [summary([run[key] for run in record['runs']]) for key in ('accuracy_delta_initial_to_16', 'accuracy_delta_initial_to_64', 'accuracy_delta_16_to_64')]
        parts.append(f'<tr><th>{escape(label)}</th>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    parts.append('</tbody></table></div><p class="muted">pp = percentage points. Paired differences are computed per seed before aggregation. These exploratory scores are excluded from the existing development-probe selection policy.</p></section>')
    parts.append('<section><h2>Did the source heads finish memorizing?</h2><p>The fixed stopping target is 100% training accuracy and loss ≤ 0.02 within 1,000 updates. Transfer tables retain all ten seeds, including heads that missed this target. A lower transfer score can therefore reflect incomplete fitting as well as limited transfer. These diagnostics do not change selection eligibility.</p><div class="scroll"><table><thead><tr><th>Model</th><th>16-question target reached</th><th>64-question target reached</th></tr></thead><tbody>')
    for record in records:
        checks = record.get('source_training_checks', {})
        cells = [f'{checks[c]["passed"]} / {checks[c]["total"]}' if c in checks else 'Not recorded' for c in ('16', '64')]
        parts.append('<tr><th>' + escape(MODELS[record['manifest']['candidate']]['label']) + '</th>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    parts.append('</tbody></table></div></section>')

    parts.append('<section><h2>Cross-entropy loss</h2><p>Lower is better. Loss penalizes confident wrong choices, which accuracy alone hides. Softmax probabilities are not calibrated.</p><div class="scroll"><table><thead><tr><th>Model</th><th>Initial head</th><th>16 trained</th><th>64 trained</th></tr></thead><tbody>')
    for record in records:
        parts.append('<tr><th>' + escape(MODELS[record['manifest']['candidate']]['label']) + '</th>' + ''.join('<td>' + summary([r['conditions'][c]['loss'] for r in record['runs']], False) + '</td>' for c in CONDITIONS) + '</tr>')
    parts.append(f'</tbody></table></div><p>Uniform-choice loss: {reference["uniform"]["loss"]:.3f}.</p></section>')
    families = sorted({ex['family'] for ex in reference['examples']})
    parts.append('<section><h2>Results by new task family</h2><p>16 questions per family. Values are mean accuracy across the ten seeds.</p><div class="scroll"><table><thead><tr><th>Model / head</th>' + ''.join(f'<th>{escape(f.replace('_', ' '))}</th>' for f in families) + '</tr></thead><tbody>')
    for record in records:
        for condition, label in CONDITIONS.items():
            parts.append('<tr><th>' + escape(MODELS[record['manifest']['candidate']]['label'] + ' / ' + label) + '</th>' + ''.join(f'<td>{mean(r["conditions"][condition]["families"][family]["accuracy"] for r in record["runs"]) * 100:.1f}%</td>' for family in families) + '</tr>')
    parts.append('</tbody></table></div></section><section><h2>Inspect every question</h2><p>Prompts use the identical Context / Question / numbered Choices / Selected choice format as training. Ground truth and explanations stay outside the model input. Expand a question to compare every seed’s prediction and probabilities.</p>')
    for i, ex in enumerate(reference['examples']):
        parts.append(f'<details><summary>{escape(ex["id"])} · {escape(ex["question"])} </summary><p>{escape(ex["context"])}</p><ol start="0">' + ''.join(f'<li>{escape(option)}{" ✓" if index == ex["answer_index"] else ""}</li>' for index, option in enumerate(ex['options'])) + f'</ol><p>{escape(json.dumps(ex["facts"], ensure_ascii=False))}</p><div class="scroll"><table><thead><tr><th>Model</th><th>Seed</th><th>Initial</th><th>16 trained</th><th>64 trained</th></tr></thead><tbody>')
        for record in records:
            for run in record['runs']:
                parts.append(f'<tr><th>{escape(MODELS[record["manifest"]["candidate"]]["label"])}</th><td>{run["seed"]}</td>')
                for condition in CONDITIONS:
                    prediction = run['conditions'][condition]['predictions'][i]
                    if prediction['id'] != ex['id']:
                        raise ValueError('Prediction order differs from examples')
                    picked = prediction['predicted']
                    probs = ', '.join(f'{j}: {p:.3f}' for j, p in enumerate(prediction['probabilities']))
                    parts.append(f'<td>{picked} {"✓" if picked == ex["answer_index"] else "✗"}<br><small>{probs}</small></td>')
                parts.append('</tr>')
        parts.append('</tbody></table></div></details>')
    parts.append('</section><section><h2>Locked protocol and provenance</h2><p>All model features are extracted once and reused for every head. Existing head checksums are verified. Freshly recomputed source features must agree with the original cache. Inputs exceeding 256 tokens are rejected, never truncated. Backbone weights and the original heads remain unchanged.</p><p>Endpoint evaluation only: these are not transfer learning curves. The four families are new to our fixtures, but share the multiple-choice interface and provided-context format. Templates and examples are a limited, exploratory sample.</p>')
    parts.append('<div class="scroll"><table><thead><tr><th>Model</th><th>Longest input (tokens)</th><th>Source feature max difference</th></tr></thead><tbody>')
    for record in records:
        parts.append(f'<tr><th>{escape(MODELS[record["manifest"]["candidate"]]["label"])}</th><td>{max(record["feature_cache"]["token_lengths"])}</td><td>{record["feature_cache"]["source_feature_max_difference"]:.6g}</td></tr>')
    parts.append('</tbody></table></div>')
    parts.append(f'<p>Locked at {reference["manifest"]["dataset_manifest"]["locked_at"]}. Dataset SHA-256: <code>{reference["manifest"]["dataset_manifest"]["data_sha256"]}</code></p><button id="download">Download all measurements</button></section>')
    parts.append('<script type="application/json" id="measurements">' + json.dumps(records).replace('<', '\\u003c') + '</script><script>document.getElementById("download").onclick=()=>{const url=URL.createObjectURL(new Blob([document.getElementById("measurements").textContent],{type:"application/json"}));const a=document.createElement("a");a.href=url;a.download="transfer-measurements.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};</script>')
    parts.append(f'<p class="muted">Generated {datetime.now(timezone.utc).isoformat()}. Fully self-contained; no external requests.</p></main></html>')
    return '\n'.join(parts)


def main():
    records = [json.loads(path.read_text()) for alias in MODELS if (path := ROOT / 'results/transfer' / f'{alias}.json').exists()]
    (ROOT / 'report-transfer.html').write_text(build(records))
    print(ROOT / 'report-transfer.html')


if __name__ == '__main__':
    main()
