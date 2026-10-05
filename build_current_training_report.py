"""Render aggregate internal training results without reading evaluation content."""

import hashlib
import json
import math
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from report_theme import PALETTE

ROOT = Path(__file__).parent
RUN = ROOT / 'runs/current-corpus-2026-10-01'
BENCHMARK_REPORT = ROOT / 'runs/current-corpus-benchmark-2026-10-04/report.json'
BENCHMARK_REPORT_SHA256 = 'd845337ff15a3e4295c2a3111e53bd9a5aa31da805e2bfe843c335c0eab66b66'
ORIGINAL_REPORT = ROOT / 'data/external_evaluation/results/original-qwen-label-likelihood-2026-10-04/blind-aggregate-result.json'
ORIGINAL_REPORT_SHA256 = '5b31638b803f6deeaf8fca368a76d8191f22b4dbe90dfd11b723072b841e0046'
COLORS = {'development': 'blue', 'training': 'teal', 'selected': 'good', 'recovery': 'orange'}


def merge_history(original, recovery, resume_update):
    """Discard superseded work; coalesce checkpoint and evaluation records by update."""
    merged = {}
    for row in [r for r in original if r['update'] <= resume_update] + recovery:
        update = row['update']
        merged[update] = {**merged.get(update, {}), **row}
    return [merged[update] for update in sorted(merged)]


def load(run=RUN):
    read = lambda path: json.loads((run / path).read_text())
    report = read('recovery-2026-10-04/full/report.json')
    status = read('recovery-2026-10-04/full/status.json')
    exited = read('recovery-2026-10-04/process-exit.json')
    config = read('run-config.json')
    validation = read('recovery-2026-10-04/validation.json')
    resume = validation['progress']['cursor']['global_update']
    history = merge_history(read('full/history.json'), read('recovery-2026-10-04/full/history.json'), resume)
    last_update = report['cursor']['global_update']
    if exited['exit_code'] != 0 or report['status'] != 'completed' or status['training_running']:
        raise ValueError('Training has not verifiably completed')
    if [p['update'] for p in history] != list(range(last_update + 1)):
        raise ValueError('Missing or duplicated update history')
    if report['training_questions'] != history[-1]['questions_seen'] or report['training_questions'] != config['training_questions_per_epoch']:
        raise ValueError('Completed training cardinality differs from one epoch')
    dev = [p for p in history if 'dev' in p]
    for point in dev:
        metrics = point['dev']
        if metrics['questions'] != config['dev_questions'] or metrics['hard_label_questions'] != 2740:
            raise ValueError('Development denominators changed')
        if sum(v['questions'] for v in metrics['by_source_kind'].values()) != metrics['questions']:
            raise ValueError('Source counts do not sum to development cardinality')
        if not math.isclose(metrics['soft_cross_entropy'] - metrics['kl_pq'], .227466804, abs_tol=1e-7):
            raise ValueError('Target entropy changed across evaluations')
    selected = report['selected_checkpoint_development']
    best = next(p for p in dev if p['update'] == report['best_update'])
    if selected['soft_cross_entropy'] != best['dev']['soft_cross_entropy'] or report['best_dev_soft_cross_entropy'] != min(p['dev']['soft_cross_entropy'] for p in dev):
        raise ValueError('Selected checkpoint does not match development minimum')
    if not report['selected_checkpoint_reload_verified']:
        raise ValueError('Selected checkpoint reload was not verified')
    return history, report, status, config, resume


def chart(series, best, resume, *, ylabel, percent=False, zoom=False):
    width, height = 1000, 330
    left, right, top, bottom = 65, 20, 22, 48
    values = [value for _, points in series for _, value in points]
    ymax = max(values) * 1.08
    ymin = 0
    if percent:
        ymax = 1
    if zoom:
        ymin, ymax = .40, .55
    xmax = max(update for _, points in series for update, _ in points)
    x = lambda value: left + value / xmax * (width - left - right)
    y = lambda value: top + (ymax - value) / (ymax - ymin) * (height - top - bottom)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(ylabel)} by training update">']
    for i in range(6):
        value = ymin + (ymax - ymin) * i / 5
        label = f'{value:.0%}' if percent else f'{value:.2f}'
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y(value):.2f}" y2="{y(value):.2f}"/><text x="{left-12}" y="{y(value)+5:.2f}" text-anchor="end">{label}</text>')
    for update in [0, 2000, 4000, 6000, xmax]:
        parts.append(f'<text x="{x(update):.2f}" y="{height-21}" text-anchor="middle">{update:,}</text>')
    parts.append(f'<text x="{width/2}" y="{height-1}" text-anchor="middle">Training update</text>')
    for role, update in [('selected', best), ('recovery', resume)]:
        parts.append(f'<line class="marker {role}" x1="{x(update):.2f}" x2="{x(update):.2f}" y1="{top}" y2="{height-bottom}"/>')
    for role, points in series:
        # Zoom excludes out-of-range samples instead of silently clipping the initial drop.
        segments, current = [], []
        for update, value in points:
            if ymin <= value <= ymax:
                current.append(f'{x(update):.2f},{y(value):.2f}')
            elif current:
                segments.append(current)
                current = []
        if current:
            segments.append(current)
        for points_string in segments:
            parts.append(f'<polyline class="trace {role}" points="{" ".join(points_string)}"/>')
        if role == 'development':
            for update, value in points:
                if ymin <= value <= ymax:
                    label = f'{value:.2%}' if percent else f'{value:.5f}'
                    parts.append(f'<circle class="point {role}" cx="{x(update):.2f}" cy="{y(value):.2f}" r="3.5"><title>Update {update:,}: {label}</title></circle>')
    parts.append('</svg>')
    return ''.join(parts)


def load_benchmark(path=BENCHMARK_REPORT):
    """Read only the owner's verified aggregate report, never evaluation items."""
    if not path.exists():
        return None
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != BENCHMARK_REPORT_SHA256:
        raise ValueError('Aggregate benchmark report differs from verified artifact')
    report = json.loads(content)
    validate_benchmark(report)
    return report


def validate_benchmark(report):
    if report['status'] != 'completed':
        raise ValueError('Benchmark evaluation is incomplete')
    if Path(report['selected_checkpoint']['path']).name != 'update-00004608-epoch-000-next-004608':
        raise ValueError('Benchmark checkpoint differs from selected training checkpoint')
    for metrics in [report['metrics'], report['comparison_to_previous_full']['metrics']]:
        overall = metrics['overall']
        slices = list(metrics['by_source_slice'].values())
        for key in ('correct', 'examples'):
            if sum(part[key] for part in slices) != overall[key]:
                raise ValueError('Benchmark slice totals differ from aggregate')
        for part in [overall, *slices]:
            if not math.isclose(part['correct'] / part['examples'], part['accuracy'], abs_tol=1e-12):
                raise ValueError('Benchmark accuracy differs from count ratio')
        for key in ('negative_log_likelihood', 'multiclass_brier_score'):
            weighted = sum(part[key] * part['examples'] for part in slices) / overall['examples']
            if not math.isclose(weighted, overall[key], abs_tol=1e-10):
                raise ValueError('Benchmark weighted metric differs from aggregate')
    previous = report['comparison_to_previous_full']['metrics']['overall']
    current = report['metrics']['overall']
    if previous['examples'] != current['examples']:
        raise ValueError('Benchmark comparison denominators differ')
    delta = report['comparison_to_previous_full']['deltas_current_minus_previous']['overall']
    if not math.isclose(100 * (current['accuracy'] - previous['accuracy']), delta['accuracy_percentage_points'], abs_tol=1e-10):
        raise ValueError('Benchmark accuracy delta differs from measured endpoints')


def benchmark_html(report):
    if report is None:
        return ''
    validate_benchmark(report)
    current = report['metrics']['overall']
    previous = report['comparison_to_previous_full']['metrics']['overall']
    delta = 100 * (current['accuracy'] - previous['accuracy'])
    labels = {
        'mmlu-pro-test': 'MMLU-Pro',
        'belebele-deu_Latn': 'Belebele · German',
        'belebele-eng_Latn': 'Belebele · English',
        'belebele-fra_Latn': 'Belebele · French',
        'belebele-spa_Latn': 'Belebele · Spanish',
        'belebele-zho_Hans': 'Belebele · Chinese',
    }
    rows = []
    for key, now in report['metrics']['by_source_slice'].items():
        before = report['comparison_to_previous_full']['metrics']['by_source_slice'][key]
        label = labels.get(key, key.replace('bbh-', 'BBH · ').replace('-', ' '))
        rows.append(f'<tr><th>{escape(label)}<small>{now["examples"]:,} items</small></th><td>{before["accuracy"]:.2%}</td><td>{now["accuracy"]:.2%}</td><td>{100 * (now["accuracy"] - before["accuracy"]):+.2f} pp</td></tr>')
    return f'''<section class="surface benchmark"><p><strong>External evaluation · reused suite</strong></p>
<p class="benchmark-result">{current['accuracy']:.2%} <span class="muted">versus {previous['accuracy']:.2%} previously</span></p>
<p><strong>{delta:+.2f} percentage points</strong> on the same {current['examples']:,} items: {current['correct']:,} correct now versus {previous['correct']:,} for the prior FULL checkpoint. The internal development gains did not translate into a better overall score on this suite.</p>
<div class="scroll"><table><thead><tr><th>Overall benchmark metric</th><th>Prior FULL</th><th>Current FULL · 4,608</th></tr></thead><tbody>
<tr><th>Accuracy ↑</th><td>{previous['accuracy']:.2%}</td><td>{current['accuracy']:.2%}</td></tr>
<tr><th>Negative log-likelihood ↓</th><td>{previous['negative_log_likelihood']:.4f}</td><td>{current['negative_log_likelihood']:.4f}</td></tr>
<tr><th>Distribution Brier score ↓</th><td>{previous['multiclass_brier_score']:.4f}</td><td>{current['multiclass_brier_score']:.4f}</td></tr></tbody></table></div>
<p>MMLU-Pro fell from 37.52% to 28.37%; all five Belebele language slices improved, while the BBH changes were mixed. The overall average is weighted by item count, with MMLU-Pro contributing 11,874 of 17,874 items.</p>
<p class="muted">Same external suite and inference protocol; different training snapshots and checkpoints. This is a descriptive comparison, not a causal test. The suite was reused, not an untouched new test. The current checkpoint was selected using development results before evaluation.</p>
<details><summary>Benchmark breakdown and limitations</summary><div class="scroll"><table><thead><tr><th>Slice</th><th>Prior FULL</th><th>Current FULL</th><th>Change</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>Belebele language slices are translations, not independent semantic problems. Uniform random-choice accuracy across this mixed suite is {current['uniform_baseline']['accuracy']:.2%}.</p>
<p>{escape(report['overlap_limit'])}</p><p>The prior checkpoint was selected in a different training run. These two endpoints do not isolate the effect of more data, more updates, or any single training choice. No item-level outputs or benchmark content appear in this report.</p>
<p>Evaluated {escape(report['evaluated_at_utc'])}. Aggregate report SHA-256: <code>{BENCHMARK_REPORT_SHA256}</code><br>Verified result SHA-256: <code>{escape(report['result_artifact']['sha256'])}</code></p>
<p><a href="../current-corpus-benchmark-2026-10-04/report.json">Verified aggregate benchmark report</a></p></details></section>'''


def load_original_benchmark(path=ORIGINAL_REPORT):
    if not path.exists():
        return None
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != ORIGINAL_REPORT_SHA256:
        raise ValueError('Original-model aggregate differs from verified artifact')
    return json.loads(content)


def validate_original_benchmark(original, current):
    if current is None:
        raise ValueError('Original-model comparison requires the trained-model aggregate')
    if original['suite_reuse']['canonical_items_sha256'] != current['protocol']['canonical_items_sha256']:
        raise ValueError('Original-model comparison uses a different suite')
    if original['suite_reuse']['current_trained_head_result_sha256'] != current['result_artifact']['sha256']:
        raise ValueError('Original-model comparison references a different trained result')
    if original['overall']['examples'] != current['metrics']['overall']['examples']:
        raise ValueError('Original-model comparison denominators differ')
    overall = original['overall']
    slices = list(original['by_source_slice'].values())
    for key in ('correct', 'examples'):
        if sum(part[key] for part in slices) != overall[key]:
            raise ValueError('Original-model slice totals differ from aggregate')
    for part in [overall, *slices]:
        if not math.isclose(part['correct'] / part['examples'], part['accuracy'], abs_tol=1e-12):
            raise ValueError('Original-model accuracy differs from count ratio')
    for key in ('negative_log_likelihood', 'multiclass_brier_score'):
        weighted = sum(part[key] * part['examples'] for part in slices) / overall['examples']
        if not math.isclose(weighted, overall[key], abs_tol=1e-10):
            raise ValueError('Original-model weighted metric differs from aggregate')


def original_benchmark_html(original, current):
    if original is None:
        return ''
    validate_original_benchmark(original, current)
    measured = original['overall']
    trained = current['metrics']['overall']
    delta = 100 * (measured['accuracy'] - trained['accuracy'])
    rows = []
    for key, now in original['by_source_slice'].items():
        trained_slice = current['metrics']['by_source_slice'][key]
        label = key.replace('bbh-', 'BBH · ').replace('belebele-', 'Belebele · ').replace('mmlu-pro-test', 'MMLU-Pro').replace('-', ' ')
        rows.append(f'<tr><th>{escape(label)}<small>{now["examples"]:,} items</small></th><td>{trained_slice["accuracy"]:.2%}</td><td>{now["accuracy"]:.2%}</td><td>{100 * (now["accuracy"] - trained_slice["accuracy"]):+.2f} pp</td></tr>')
    diagnostic_tables = []
    for key, label in [('choice_count_bin', 'Number of choices'), ('input_token_bin', 'Input tokens')]:
        parts = original['exploratory_diagnostics'][key]
        if sum(part['examples'] for part in parts.values()) != measured['examples']:
            raise ValueError('Original-model diagnostic strata do not cover all items')
        body = ''.join(f'<tr><th>{escape(name)}</th><td>{part["examples"]:,}</td><td>{part["accuracy"]:.2%}</td></tr>' for name,part in parts.items())
        diagnostic_tables.append(f'<div class="scroll"><table><thead><tr><th>{label}</th><th>Items</th><th>Original Qwen accuracy</th></tr></thead><tbody>{body}</tbody></table></div>')
    return f'''<section class="surface"><p><strong>Original Qwen · exploratory end-to-end comparison</strong></p>
<p class="benchmark-result">{measured['accuracy']:.2%} <span class="muted">with the original language-model head</span></p>
<p>{measured['correct']:,} / {measured['examples']:,} correct: <strong>{delta:+.2f} percentage points</strong> above the current trained decision model on this reused suite.</p>
<p class="muted">This uses a different prompt wrapper, choice labels and readout: label likelihood from the unedited Qwen3-4B language-model head. It compares complete systems; it does not isolate the effect of backbone training or measure the original model through our decision head. It is not a generated reasoning-chain evaluation.</p>
<div class="scroll"><table><thead><tr><th>Aggregate metric</th><th>Current decision head</th><th>Original Qwen · label likelihood</th></tr></thead><tbody>
<tr><th>Accuracy ↑</th><td>{trained['accuracy']:.2%}</td><td>{measured['accuracy']:.2%}</td></tr>
<tr><th>Negative log-likelihood ↓</th><td>{trained['negative_log_likelihood']:.4f}</td><td>{measured['negative_log_likelihood']:.4f}</td></tr>
<tr><th>Distribution Brier score ↓</th><td>{trained['multiclass_brier_score']:.4f}</td><td>{measured['multiclass_brier_score']:.4f}</td></tr></tbody></table></div>
<p>The original method chooses more correct answers but has worse probability losses under this scoring method. These probabilities come from different readouts; higher accuracy does not imply better-calibrated confidence.</p>
<details><summary>Original-model slices and exploratory diagnostics</summary><div class="scroll"><table><thead><tr><th>Slice</th><th>Current decision head</th><th>Original Qwen</th><th>Original minus current</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>The following strata describe only the original model. Choice count, task family and input length are confounded; they do not establish a causal explanation for the result. These are post hoc diagnostics on a reused suite.</p>{''.join(diagnostic_tables)}
<p>{escape(original['comparison_limit'])} Belebele translations are not independent semantic questions. The existing overlap limitations apply; no new contamination claim is made.</p>
<p>Evaluated {escape(original['evaluated_at_utc'])}. Aggregate SHA-256: <code>{ORIGINAL_REPORT_SHA256}</code><br>Protocol SHA-256: <code>{escape(original['protocol_sha256'])}</code><br>Evaluation lock SHA-256: <code>{escape(original['evaluation_lock_sha256'])}</code></p>
<p><a href="results-before-original-qwen-2026-10-04.html">Preserved report before the original-model evaluation</a></p></details></section>'''


def build(history, report, status, config, resume, benchmark=None, original_benchmark=None):
    dev = [p for p in history if 'dev' in p]
    baseline = dev[0]['dev']
    selected = report['selected_checkpoint_development']
    final = dev[-1]['dev']
    best = report['best_update']
    last = report['cursor']['global_update']
    pct = lambda value: f'{value:.1%}'
    ce = [(p['update'], p['dev']['soft_cross_entropy']) for p in dev]
    training = [(p['update'], p['train_loss']) for p in history if 'train_loss' in p]
    accuracy = [(p['update'], p['dev']['hard_accuracy']) for p in dev]
    metrics = [('Development cross-entropy ↓', 'soft_cross_entropy', False),
               ('Excess loss above target entropy (KL) ↓', 'kl_pq', False),
               ('Hard-label accuracy ↑ · 2,740 questions', 'hard_accuracy', True),
               ('Target argmax agreement ↑ · 7,202 questions', 'argmax_agreement', True),
               ('Distribution Brier score ↓', 'categorical_brier_sum', False)]
    rows = ''.join('<tr><th>' + label + '</th>' + ''.join('<td>' + (pct(m[key]) if percent else f'{m[key]:.4f}') + '</td>' for m in (baseline, selected, final)) + '</tr>' for label, key, percent in metrics)
    idk_rows = ''
    for kind, label in [('unanswerable', 'Unanswerable: chose “I don’t know” ↑'), ('answerable', 'Answerable: chose “I don’t know” ↓')]:
        count = selected['task_grounded_idk'][kind]['questions']
        idk_rows += f'<tr><th>{label}<small>{count:,} tagged development questions</small></th>' + ''.join(f'<td>{pct(m["task_grounded_idk"][kind]["predicted_idk_fraction"])}</td>' for m in (baseline, selected, final)) + '</tr>'
    theme = ''.join(f'--{key}:{value};' for key, value in PALETTE.items())
    theme += ''.join(f'--{role}:var(--{token});' for role, token in COLORS.items())
    plots = [
        ('development', 'Development loss · full range', chart([('development', ce)], best, resume, ylabel='Development cross-entropy'), 'Every recorded full-development evaluation, including the initial baseline. Lower is better.'),
        ('zoom', 'Development loss · after the initial drop', chart([('development', ce)], best, resume, ylabel='Development cross-entropy, zoomed', zoom=True), 'Same evaluations, vertical range 0.40–0.55. The initial baseline is outside this view; no samples are smoothed.'),
        ('training', 'Training and development loss · every update', chart([('training', training), ('development', ce)], best, resume, ylabel='Training and development cross-entropy'), f'All {len(training):,} raw training losses. Each is a different minibatch; development evaluates the same 7,202 questions. The curves therefore measure different samples.'),
        ('accuracy', 'Development hard-label accuracy', chart([('development', accuracy)], best, resume, ylabel='Hard-label development accuracy', percent=True), 'Accuracy on the same 2,740 hard-label questions. This is not accuracy across the full mixed-target development set.'),
    ]
    charts = ''.join(f'<details class="plot" id="{key}" {"open" if i == 0 else ""}><summary>{label}</summary>{svg}<p class="muted">{note}</p></details>' for i, (key, label, svg, note) in enumerate(plots))
    source_paths = ['full/history.json', 'recovery-2026-10-04/full/history.json', 'recovery-2026-10-04/full/report.json', 'recovery-2026-10-04/full/status.json', 'recovery-2026-10-04/process-exit.json', 'recovery-2026-10-04/validation.json', 'run-config.json']
    sources = ''.join(f'<li><code>{escape(path)}</code></li>' for path in source_paths)
    measurements = [{'update': p['update'], 'questions_seen': p['questions_seen'], **({'train_loss': p['train_loss']} if 'train_loss' in p else {}), **({'dev': {k:v for k,v in p['dev'].items() if not k.startswith('by_')}} if 'dev' in p else {})} for p in history]
    payload = json.dumps({'history': measurements, 'best_update': best, 'resume_update': resume, 'selected': {k:v for k,v in selected.items() if not k.startswith('by_')}}, allow_nan=False).replace('<', '\\u003c')
    benchmark_section = benchmark_html(benchmark) + original_benchmark_html(original_benchmark, benchmark)
    benchmark_status = ('The external evaluation below is complete. The separate internal held-out test has not been run.'
                        if benchmark else 'Independent benchmarks and the held-out test have <strong>not been run</strong> for this checkpoint.')
    external_provenance = ('The external comparison uses a previously evaluated suite and is descriptive, not an untouched-test claim.'
                           if benchmark else 'No prior external benchmark result is reused here.')
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Completed training · current corpus</title><style>
:root{{{theme}}}*{{box-sizing:border-box}}body{{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:32px 26px}}.benchmark-result{{font-size:28px;font-weight:650}}.benchmark-result span{{font-size:17px;font-weight:400}}h1{{font-size:30px;line-height:1.25;letter-spacing:-.02em;margin:10px 0}}p{{margin:12px 0}}.muted,small{{color:var(--muted)}}.status{{color:var(--good);font-size:14px;font-weight:650}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:24px 0}}.card,.surface{{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px}}.card strong{{font-size:30px;display:block;margin:4px 0}}small{{display:block;font-size:13px}}.surface{{margin:20px 0}}details{{border-top:1px solid var(--line);padding:14px 0}}summary{{cursor:pointer;font-weight:600}}.plot:first-of-type{{border-top:0}}svg{{width:100%;height:auto;display:block;margin-top:16px;overflow:visible}}svg text{{fill:var(--muted);font:14px system-ui}}.grid{{stroke:var(--line);stroke-width:1}}.trace{{fill:none;stroke-width:2;stroke-linejoin:round}}.trace.training{{stroke:var(--training);stroke-width:1;opacity:.5}}.trace.development{{stroke:var(--development);stroke-width:2.5}}.point{{fill:var(--development)}}.marker{{stroke-width:1.5;stroke-dasharray:5 4}}.marker.selected{{stroke:var(--selected)}}.marker.recovery{{stroke:var(--recovery)}}.legend{{display:flex;flex-wrap:wrap;gap:12px 24px;font-size:13px}}.legend span::before{{content:"";display:inline-block;width:16px;height:3px;vertical-align:middle;background:var(--development);margin-right:7px}}.legend .train::before{{background:var(--training)}}.legend .best::before{{background:var(--selected)}}.legend .resume::before{{background:var(--recovery)}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{border-bottom:1px solid var(--line);padding:12px;text-align:left}}thead th{{font-size:14px;color:var(--muted)}}tbody th{{font-weight:500}}td{{white-space:nowrap}}.scroll{{overflow-x:auto}}code{{font-size:12px;overflow-wrap:anywhere}}a{{color:var(--blue)}}button,select{{font:inherit;border:1px solid var(--line);border-radius:6px;background:var(--paper);color:var(--ink);padding:7px 12px}}.control{{margin:14px 0}}footer{{font-size:12px;color:var(--muted);margin:24px 0}}@media(max-width:650px){{main{{padding:20px 14px}}h1{{font-size:25px}}.cards{{grid-template-columns:1fr;gap:10px}}.card{{padding:14px 18px}}.card strong{{font-size:26px}}.surface{{padding:14px}}td,th{{padding:10px 8px}}table{{min-width:580px}}svg text{{font-size:16px}}}}
</style></head><body><main>
<div class="status">Completed · checkpoint reload verified · exit 0</div>
<h1>Training finished. Selected update {best:,}.</h1>
<p>Qwen3-4B with a 265-slot decision head, trained across the full backbone. One pass through {report['training_questions']:,} questions finished at update {last:,}; recovery was capped at one epoch for assessment, so no second epoch ran.</p>
{benchmark_section}
<p><strong>Internal development · selected checkpoint</strong></p>
<div class="cards"><div class="card">Hard-label accuracy<strong>{pct(selected['hard_accuracy'])}</strong><small>from {pct(baseline['hard_accuracy'])} · 2,740 development questions</small></div><div class="card">Development loss<strong>{selected['soft_cross_entropy']:.3f}</strong><small>from {baseline['soft_cross_entropy']:.3f} · selected by lowest cross-entropy</small></div><div class="card">IDK on unanswerable questions<strong>{pct(selected['task_grounded_idk']['unanswerable']['predicted_idk_fraction'])}</strong><small>from {pct(baseline['task_grounded_idk']['unanswerable']['predicted_idk_fraction'])} · 748 tagged questions</small></div></div>
<p class="muted">These are internal development results used for checkpoint selection. {benchmark_status}</p>
<section class="surface"><div class="legend"><span>Development</span><span class="train">Training</span><span class="best">Selected · {best:,}</span><span class="resume">Recovery · {resume:,}</span></div>
<div class="control" id="plot-control" hidden><label for="view">Curve view </label><select id="view">{''.join(f'<option value="{key}">{label}</option>' for key,label,_,_ in plots)}</select></div>
{charts}
<p class="muted">Connected points show recorded evaluations, not intermediate measurements. Hover over development points for exact values. All charts are embedded SVG and remain usable without JavaScript.</p></section>
<section class="surface"><p><strong>Selected versus final checkpoint</strong></p><p>The final update's loss was {final['soft_cross_entropy']:.4f}, above the selected {selected['soft_cross_entropy']:.4f}. Keep the earlier minimum; the last saved checkpoint is a recovery endpoint, not the chosen result.</p>
<div class="scroll"><table><thead><tr><th>Development metric</th><th>Initial · 0</th><th>Selected · {best:,}</th><th>Final · {last:,}</th></tr></thead><tbody>{rows}</tbody></table></div>
<details><summary>Uncertainty and “I don’t know”</summary><div class="scroll"><table><thead><tr><th>Tagged subset</th><th>Initial</th><th>Selected</th><th>Final</th></tr></thead><tbody>{idk_rows}</tbody></table></div><p>Answerability tags cover 3,850 of 7,202 development questions; the remaining 3,352 have no recorded tag. A low answerable-IDK rate means fewer abstentions, not necessarily more correct answers. These rates do not establish calibrated confidence.</p></details>
<details><summary>How to read the loss</summary><p>The full development set mixes hard labels and target distributions. Cross-entropy includes their intrinsic target entropy, approximately 0.22747 nats. KL subtracts that constant and measures excess loss above matching those targets exactly. Neither lower loss nor agreement with teacher estimates proves factual correctness.</p><p>Hard-label accuracy uses only 2,740 questions. Argmax agreement uses all 7,202 targets and is a different measure. The Brier score sums squared probability errors across choices; lower is better.</p></details></section>
<details><summary>Run details, recovery and source files</summary><p>Seed {config['seed']}; fixed backbone learning rate {config['backbone_learning_rate']:g}, head learning rate {config['head_learning_rate']:g}; gradient accumulation {config['accumulation']}. The starting backbone was {escape(config['repo'])}, revision <code>{escape(config['revision'])}</code>, with a newly seeded random head. This is full-backbone training, not the earlier frozen-feature experiment.</p><p>The original plan allowed up to {config['epochs']} epochs. Recovery was capped at one epoch for assessment and completed that epoch. The completion artifact also records a development-plateau stop reason; the run does not establish what would happen in a second epoch. History is merged by update, with recovery taking precedence and superseded work after update {resume:,} excluded. The recovery evaluation at {resume:,} is merged with its existing training update, not counted twice. The report contains {len(training):,} unique training points and {len(dev)} development evaluations; no smoothing or downsampling.</p><p>Selected checkpoint: <code>{escape(str(Path('full/checkpoints') / Path(report['best_checkpoint_relative']).name))}</code><br>Final checkpoint: <code>recovery-2026-10-04/full/{escape(report['last_checkpoint_relative'])}</code></p><p>Development questions influenced stopping and selection. This is one run, without repeated-seed uncertainty estimates or an untouched-test generalization claim. {external_provenance}</p><p>Completion: {escape(status['completed_at'])}. Sources are aggregate local run artifacts:</p><ul>{sources}</ul><p><a href="recovery-2026-10-04/full/curves.html">Recovery's original curve report</a></p><button id="download" hidden>Download aggregate curve data</button></details>
<footer>Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} · Self-contained report; no network dependencies.</footer>
<script type="application/json" id="measurements">{payload}</script><script>
document.getElementById('plot-control').hidden=false;
document.getElementById('view').addEventListener('change',event=>{{document.querySelectorAll('.plot').forEach(plot=>{{plot.open=plot.id===event.target.value;}});}});
const download=document.getElementById('download');download.hidden=false;download.onclick=()=>{{const url=URL.createObjectURL(new Blob([document.getElementById('measurements').textContent],{{type:'application/json'}}));const a=document.createElement('a');a.href=url;a.download='current-training-aggregate-curves.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}};
</script></main></body></html>'''


def main():
    output = RUN / 'results.html'
    output.write_text(build(*load(), benchmark=load_benchmark(), original_benchmark=load_original_benchmark()))
    print(output)


if __name__ == '__main__':
    main()
