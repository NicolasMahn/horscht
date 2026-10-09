"""Render the completed experiment using only allowlisted aggregate results."""
import hashlib
import json
from html import escape
from pathlib import Path

from report_theme import PALETTE

ROOT = Path(__file__).resolve().parent
RUN = ROOT / 'runs/new-data-last-block-2026-10-08/fast-evaluation-003'
RESULTS = ROOT / 'data/external_evaluation/results'
PRESENTATION = {'background': 'canvas', 'surface': 'paper', 'text': 'ink',
                'secondary': 'muted', 'border': 'line', 'accent': 'blue',
                'improvement': 'good', 'warning': 'orange'}


def read(path):
    return json.loads(path.read_text())


def grouped(result, prefix):
    rows = [v for k, v in result['by_source_slice'].items() if k.startswith(prefix)]
    count = sum(r['examples'] for r in rows)
    return {'examples': count, 'accuracy': sum(r['correct'] for r in rows) / count}


def comparison_row(label, previous, current):
    delta = 100 * (current['accuracy'] - previous['accuracy'])
    assert previous['examples'] == current['examples']
    return (f'<tr><th scope="row">{escape(label)}<small>{current["examples"]:,} questions</small></th>'
            f'<td>{previous["accuracy"]:.2%}</td><td class="current">{current["accuracy"]:.2%}</td>'
            f'<td class="{"gain" if delta >= 0 else "loss"}">{delta:+.2f} pp</td></tr>')


def comparison_table(rows):
    return ('<div class="scroll"><table><thead><tr><th scope="col">Benchmark</th>'
            '<th scope="col">Before new training</th><th scope="col">New checkpoint</th>'
            '<th scope="col">Change</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>')


def build():
    receipt = read(RUN / 'benchmark-report.json')
    aggregate_path = RESULTS / 'new-data-last-block-fast-2026-10-09/blind-aggregate-result.json'
    assert hashlib.sha256(aggregate_path.read_bytes()).hexdigest() == receipt['aggregate_sha256']
    current = read(aggregate_path)['candidate_results']['last-block']
    previous = read(RESULTS / 'current-corpus-last-block-2026-10-05/blind-aggregate-result.json')['candidate_results']['last-block']
    original = read(RESULTS / 'original-qwen-label-likelihood-2026-10-04/blind-aggregate-result.json')
    overall, old = current['overall'], previous['overall']
    assert overall['examples'] == old['examples'] == 17874
    assert sum(r['correct'] for r in current['by_source_slice'].values()) == overall['correct']
    change = 100 * (overall['accuracy'] - old['accuracy'])
    extra = overall['correct'] - old['correct']
    rows = [comparison_row(name, grouped(previous, prefix), grouped(current, prefix))
            for name, prefix in [('MMLU-Pro', 'mmlu-'), ('Belebele', 'belebele-'), ('BBH', 'bbh-')]]
    rows.append(comparison_row('Overall', old, overall))
    languages = {'deu_Latn': 'German', 'eng_Latn': 'English', 'fra_Latn': 'French',
                 'spa_Latn': 'Spanish', 'zho_Hans': 'Chinese'}
    language_rows = [comparison_row(label, previous['by_source_slice']['belebele-' + key],
                                    current['by_source_slice']['belebele-' + key]) for key, label in languages.items()]
    bbh_rows = []
    for key, value in current['by_source_slice'].items():
        if key.startswith('bbh-'):
            label = key.removeprefix('bbh-').replace('-', ' ').capitalize()
            bbh_rows.append(comparison_row(label, previous['by_source_slice'][key], value))
    tracking = grouped(current, 'bbh-tracking-')['accuracy']
    tracking_rows = [v for k, v in current['by_source_slice'].items() if k.startswith('bbh-tracking-')]
    chance = sum(v['uniform_baseline']['accuracy'] * v['examples'] for v in tracking_rows) / sum(v['examples'] for v in tracking_rows)
    selection = read(RUN / 'development-selection.json')
    development_rows = ''.join(
        f'<tr><th scope="row">{int(key):,}{" · selected" if int(key) == selection["selected_update"] else ""}</th>'
        f'<td>{v["metrics"]["soft_cross_entropy"]:.4f}</td><td>{v["metrics"]["argmax_agreement"]:.1%}</td></tr>'
        for key, v in sorted(selection['candidates'].items(), key=lambda item: int(item[0])))
    tokens = ';'.join(f'--{role}:{PALETTE[color]}' for role, color in PRESENTATION.items())
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Benchmark comparison · Qwen3-4B</title>
<style>
:root{{{tokens};font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:var(--background);font-synthesis:none}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1080px;margin:auto;padding:32px 24px 60px}}
a{{color:var(--accent);text-underline-offset:3px}}nav{{display:flex;justify-content:space-between;gap:16px;font-size:14px;margin-bottom:36px}}
h1{{font-size:clamp(28px,4vw,44px);line-height:1.15;letter-spacing:-.035em;max-width:800px;margin:12px 0 16px}}
p{{line-height:1.65;margin:12px 0}}.muted,small{{color:var(--secondary)}}.intro{{max-width:780px}}
.eyebrow{{font-size:13px;letter-spacing:.07em;text-transform:uppercase;color:var(--secondary)}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}}
.card,.panel,details{{background:var(--surface);border:1px solid var(--border);border-radius:16px}}
.card{{padding:22px}}.card .number{{display:block;font-size:38px;font-weight:720;letter-spacing:-.04em;margin:6px 0}}
.card p{{font-size:13px;margin:0;color:var(--secondary)}}.gain{{color:var(--improvement)}}.loss{{color:var(--warning)}}
.panel{{padding:8px 24px 20px;margin:22px 0}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
th,td{{padding:18px 12px;border-bottom:1px solid var(--border);text-align:right;white-space:nowrap}}
th:first-child{{text-align:left;padding-left:0}}thead th{{font-size:12px;font-weight:600;color:var(--secondary)}}
tbody th{{font-size:15px;font-weight:600}}small{{display:block;font-size:12px;font-weight:400;margin-top:4px}}
.current{{font-weight:750}}tbody tr:last-child th,tbody tr:last-child td{{border-bottom:0}}
.note{{font-size:14px;color:var(--secondary)}}details{{margin:12px 0;padding:0 22px}}summary{{padding:20px 0;cursor:pointer;font-weight:600}}
details p{{font-size:14px}}.interpretation{{border-left:3px solid var(--accent);padding:0 20px;margin:28px 0}}
footer{{font-size:12px;color:var(--secondary);line-height:1.7;margin-top:30px}}
@media(max-width:650px){{main{{padding:22px 16px}}.cards{{grid-template-columns:1fr;gap:10px}}.card{{padding:16px}}.card .number{{font-size:32px}}nav{{flex-wrap:wrap;margin-bottom:24px}}.panel{{padding:4px 14px 16px}}th,td{{padding:16px 10px}}}}
</style><main>
<nav><a href="/runs/project-record-2026-10-04.html">← All experiment results</a><span class="muted">Completed {escape(receipt['completed_at'].split('.')[0])}Z</span></nav>
<div class="eyebrow">Qwen3-4B · last transformer block + linear head</div>
<h1>A modest gain across all three benchmarks.</h1>
<p><a href="/runs/decision-model-comparison-2026-10-09.html">Compare with other 4B decision models →</a></p>
<p class="intro">The new-data continuation answered <strong>{extra:,} more questions correctly</strong> than the previous last-block checkpoint on the same full suite.</p>
<div class="cards">
<div class="card">Overall accuracy<span class="number">{overall['accuracy']:.2%}</span><p>{overall['correct']:,} / {overall['examples']:,} correct</p></div>
<div class="card">Change from previous<span class="number gain">+{change:.2f} pp</span><p>{old['accuracy']:.2%} → {overall['accuracy']:.2%} · percentage points</p></div>
<div class="card">Brier score ↓<span class="number">{overall['multiclass_brier_score']:.4f}</span><p>Previously {old['multiclass_brier_score']:.4f} · lower is better</p></div>
</div>
<section class="panel" aria-label="Full benchmark comparison">{comparison_table(rows)}
<p class="note">Same 17,874 questions and 32,768-token evaluation limit. “Before” is the previous trained last-block checkpoint, before this new-data continuation. Negative log-likelihood also improved: {old['negative_log_likelihood']:.4f} → {overall['negative_log_likelihood']:.4f} (lower is better).</p></section>
<div class="interpretation"><p>The original Qwen immediate-answer reference scored <strong>{original['overall']['accuracy']:.2%}</strong> overall, so the new decision model remains <strong>{100*(original['overall']['accuracy']-overall['accuracy']):.2f} percentage points behind</strong>. That reference uses the original output head and a different scoring method; it is not a thinking-mode result.</p>
<p>Object tracking remains a weakness: <strong>{tracking:.1%}</strong> across the three tracking tasks, versus <strong>{chance:.1%}</strong> for uniform guessing.</p></div>
<details><summary>Language breakdown · Belebele</summary>{comparison_table(language_rows)}<p class="note">The language slices translate the same underlying questions; they are not independent semantic problems.</p></details>
<details><summary>Reasoning task breakdown · BBH</summary>{comparison_table(bbh_rows)}</details>
<details><summary>Checkpoint selection · separate development data</summary>
<p>Update <strong>{selection['selected_update']:,}</strong> was selected by lowest soft-target cross-entropy on a fixed 128-question panel, before this external evaluation. The final checkpoint was nearly tied; this small panel does not establish a meaningful difference between the two.</p>
<div class="scroll"><table><thead><tr><th>Update</th><th>Cross-entropy ↓</th><th>Top-choice agreement ↑</th></tr></thead><tbody>{development_rows}</tbody></table></div>
<p>All 1,438 training updates completed on 23,006 questions. A separate 64-question calibration pass completed. The additional internal test remains deferred. The benchmark scores above use the standard benchmark runner probabilities.</p></details>
<footer>Exploratory reuse of a previously observed benchmark suite, not an untouched final test. Aggregate results only; benchmark questions and answers are not displayed.<br>
<a href="benchmark-report.json">Download result JSON</a> · <a href="/runs/project-record-2026-10-04.html">Project comparison</a></footer>
</main></html>'''
    output = RUN / 'benchmark-report.html'
    output.write_text(html)
    return output


if __name__ == '__main__':
    print(build())
