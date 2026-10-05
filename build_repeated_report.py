"""Build the repeated-seed evaluation report from measured runs."""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from build_report import chart_base, escape
from report_theme import MODELS, PALETTE
from selection_policy import POLICY, select

ROOT = Path(__file__).parent


def validate(records):
    if not records:
        raise ValueError('No repeated results')
    reference = records[0]
    seeds = [run['seed'] for run in reference['runs']]
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError('Expected at least two distinct seeds')
    for record in records:
        if record['examples'] != reference['examples']:
            raise ValueError('Examples differ')
        if [run['seed'] for run in record['runs']] != seeds:
            raise ValueError('Seed sets differ')
        for key in ('data_sha256', 'learning_rate', 'max_steps', 'format', 'dtype', 'threads',
                    'capacity', 'device', 'head', 'pooling', 'max_tokens', 'optimizer',
                    'chat_template', 'feature_dtype', 'backbone_trainable'):
            if record['manifest'][key] != reference['manifest'][key]:
                raise ValueError(f'Comparison setting differs: {key}')
        for run in record['runs']:
            history = run['training']['history']
            if [p['step'] for p in history] != list(range(run['training']['steps'] + 1)):
                raise ValueError('Incomplete learning curve')
            if any(not math.isfinite(p[key]) for p in history
                   for key in ('loss', 'accuracy', 'probe_loss', 'probe_accuracy')):
                raise ValueError('Non-finite measurement')


def curve_statistics(runs, key):
    """Hold a stopped checkpoint fixed so later averages retain every seed."""
    histories = [run['training']['history'] for run in runs]
    steps = range(max(len(history) for history in histories))
    values = [[history[min(step, len(history) - 1)][key] for history in histories] for step in steps]
    return list(steps), [mean(v) for v in values], [stdev(v) for v in values]


def curve(records, key):
    loss = 'loss' in key
    fig = chart_base('Cross-entropy (nats)' if loss else 'Accuracy')
    averages, individual = [], []
    for record in records:
        style = MODELS[record['manifest']['candidate']]
        color = PALETTE[style['color']]
        red, green, blue = (int(color[i:i + 2], 16) for i in (1, 3, 5))
        x, y, sd = curve_statistics(record['runs'], key)
        lower = [max(1e-8 if loss else 0, v - s) for v, s in zip(y, sd)]
        upper = [v + s if loss else min(1, v + s) for v, s in zip(y, sd)]
        for boundary in (lower, upper):
            is_upper = boundary is upper
            fig.add_trace(go.Scatter(x=x, y=boundary, mode='lines', line=dict(width=0),
                                    fill='tonexty' if is_upper else None,
                                    fillcolor=f'rgba({red},{green},{blue},0.12)',
                                    showlegend=False, hoverinfo='skip', legendgroup=style['label']))
            averages.append(True); individual.append(False)
        fig.add_trace(go.Scatter(x=x, y=y, name=style['label'], mode='lines',
                                legendgroup=style['label'], line=dict(color=color, width=2.5),
                                customdata=sd,
                                hovertemplate='%{y:.4f} ± %{customdata:.4f}<extra>%{fullData.name}</extra>'))
        averages.append(True); individual.append(False)
        for run in record['runs']:
            points = run['training']['history']
            fig.add_trace(go.Scatter(x=[p['step'] for p in points], y=[p[key] for p in points],
                                    name=f"{style['label']} · seed {run['seed']}", mode='lines',
                                    line=dict(color=color, width=1), opacity=.65, visible=False,
                                    showlegend=False,
                                    hovertemplate='%{y:.4f}<extra>%{fullData.name}</extra>'))
            averages.append(False); individual.append(True)
    buttons = [dict(label='All models · mean ± SD', method='update', args=[{'visible': averages}])]
    offset = 0
    for record in records:
        label = MODELS[record['manifest']['candidate']]['label']
        width = 3 + len(record['runs'])
        for name, mask in [('mean ± SD', averages), ('individual runs', individual)]:
            visible = [value and offset <= i < offset + width for i, value in enumerate(mask)]
            buttons.append(dict(label=f'{label} · {name}', method='update', args=[{'visible': visible}]))
        offset += width
    menus = [dict(type='dropdown', direction='down', x=0, xanchor='left', y=1.18,
                  buttons=buttons)]
    if loss:
        menus.append(dict(type='buttons', direction='right', x=0, xanchor='left', y=1.4,
                          buttons=[dict(label='Linear', method='relayout', args=[{'yaxis.type': 'linear', 'yaxis.autorange': True}]),
                                   dict(label='Log', method='relayout', args=[{'yaxis.type': 'log', 'yaxis.autorange': True}])]))
    else:
        fig.update_yaxes(range=[-.03, 1.05], tickformat='.0%')
    split = 'probe' if key.startswith('probe') else 'train'
    examples = [ex for ex in records[0]['examples'] if ex['split'] == split]
    baseline = mean(math.log(len(ex['options'])) if loss else 1 / len(ex['options']) for ex in examples)
    fig.add_hline(y=baseline, line_dash='dot', line_color=PALETTE['muted'])
    fig.update_layout(height=710, margin=dict(t=110, b=280), updatemenus=menus,
                      legend=dict(y=-.3, font=dict(size=11), entrywidth=220, tracegroupgap=0))
    return pio.to_html(fig, full_html=False, include_plotlyjs=False, div_id=key,
                       config={'responsive': True, 'displaylogo': False})


def spread(values, percent=False):
    if percent:
        return f'{mean(values):.1%} ± {stdev(values) * 100:.1f} pp'
    return f'{mean(values):.3f} ± {stdev(values):.3f}'


def summary_rows(records):
    rows = []
    for record in records:
        runs = record['runs']
        candidate = record['manifest']['candidate']
        accuracy = [r['probe']['accuracy'] for r in runs]
        rows.append(f'''<tr><th class="model-{candidate}">{escape(MODELS[candidate]['label'])}</th>
        <td>{sum(r['passed'] for r in runs)}/{len(runs)}</td>
        <td>{spread([r['training']['final']['loss'] for r in runs])}</td>
        <td>{mean(r['training']['steps'] for r in runs):.1f} ({min(r['training']['steps'] for r in runs)}–{max(r['training']['steps'] for r in runs)})</td>
        <td>{spread(accuracy, True)}</td><td>{min(accuracy):.1%}–{max(accuracy):.1%}</td>
        <td>{spread([r['probe']['loss'] for r in runs])}</td></tr>''')
    return ''.join(rows)


def run_tables(records):
    sections = []
    for record in records:
        rows = []
        for run in record['runs']:
            training = run['training']
            first = next((p['step'] for p in training['history'] if p['accuracy'] == 1), '—')
            rows.append(f'''<tr><th>{run['seed']}</th><td>{training['initial']['loss']:.4f} → {training['final']['loss']:.4f}</td>
            <td>{training['final']['accuracy']:.1%}</td><td>{first} / {training['steps']}</td>
            <td>{run['probe']['accuracy']:.1%}</td><td>{run['probe']['loss']:.4f}</td>
            <td>{'Pass' if run['passed'] else 'Failed'}</td><td>{run['seconds']:.3f} s</td></tr>''')
        sections.append(f'''<details class="surface"><summary>{escape(MODELS[record['manifest']['candidate']]['label'])} · every run</summary>
        <div class="scroll"><table><thead><tr><th>Seed</th><th>Training loss</th><th>Training correct</th><th>First 100% / stop</th><th>Probe correct</th><th>Probe loss</th><th>Overfit + head reload</th><th>Fit + save + check</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table></div></details>''')
    return ''.join(sections)


def decisions(records):
    rows = []
    for ex in sorted(records[0]['examples'], key=lambda ex: (ex['split'] != 'probe', ex['id'])):
        cells = []
        for record in records:
            predictions = [(run['seed'], next(p for p in run['predictions'] if p['id'] == ex['id']))
                           for run in record['runs']]
            correct = sum(p['predicted'] == ex['answer_index'] for _, p in predictions)
            details = ''.join(f"<li>Seed {seed}: {escape(ex['options'][p['predicted']])}</li>" for seed, p in predictions)
            cells.append(f'<td><details><summary>{correct}/{len(predictions)} correct</summary><ul>{details}</ul></details></td>')
        choices = ' · '.join(f'{i}: {escape(option)}' for i, option in enumerate(ex['options']))
        rows.append(f'''<tr><th><span class="small">{escape(ex['id'])} · {ex['split']}</span><p>{escape(ex['context'])}</p>
        <p>{escape(ex['question'])}</p><span class="small">{choices}</span><p>Answer: {escape(ex['options'][ex['answer_index']])}</p></th>{''.join(cells)}</tr>''')
    return ''.join(rows)


def selection_summary(records, unavailable=()):
    result = select([*records, *unavailable], MODELS)
    if result['leaders']:
        labels = ', '.join(MODELS[name]['label'] for name in result['leaders'])
        status = 'Provisional selection' if result['complete'] else 'Leading available results · comparison incomplete'
        heading = f"<p><strong>{escape(status)}: {escape(labels)}</strong></p>"
        if len(result['leaders']) > 1:
            heading += '<p>Exact tie on both selection metrics; no single winner.</p>'
    else:
        heading = '<p>No eligible candidate yet.</p>'
    rows = []
    previous, rank = None, 0
    for index, row in enumerate(result['ranked'], 1):
        score = (row['accuracy'], row['loss'])
        if score != previous:
            rank = index
        previous = score
        rows.append(f"<tr><td>{rank}</td><th>{escape(MODELS[row['candidate']]['label'])}</th>"
                    f"<td>{row['accuracy']:.2%}</td><td>{row['loss']:.6f}</td><td>Eligible</td></tr>")
    for row in result['excluded']:
        rows.append(f"<tr><td>—</td><th>{escape(MODELS[row['candidate']]['label'])}</th>"
                    f"<td>—</td><td>—</td><td>{escape(row['reason'])}</td></tr>")
    return heading + '<div class="scroll"><table class="summary"><thead><tr><th>Rank</th><th>Model</th><th>Mean probe accuracy ↑</th><th>Mean probe loss ↓</th><th>Eligibility</th></tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


def build(records, unavailable=()):
    validate(records)
    template = (ROOT / 'repeated_report_template.html').read_text()
    seeds = [run['seed'] for run in records[0]['runs']]
    theme = ':root{' + ''.join(f'--{k}:{v};' for k, v in PALETTE.items()) + '}'
    theme += ''.join(f".model-{name}{{color:var(--{style['color']})}}" for name, style in MODELS.items())
    examples = records[0]['examples']
    manifest = records[0]['manifest']
    replacements = {
        'THEME': theme, 'PLOTLY': get_plotlyjs(), 'COUNT': str(len(seeds)),
        'MODELS': str(len(records)), 'TOTAL': str(sum(len(r['runs']) for r in records)),
        'TRAIN': str(sum(e['split'] == 'train' for e in examples)),
        'PROBE': str(sum(e['split'] == 'probe' for e in examples)),
        'SEEDS': ', '.join(map(str, seeds)), 'SUMMARY': summary_rows(records),
        'SELECTION': selection_summary(records, unavailable), 'POLICY_LOCKED': POLICY['locked_at'],
        'RUN_TABLES': run_tables(records), 'DECISIONS': decisions(records),
        'MODEL_HEADERS': ''.join(f"<th>{escape(MODELS[r['manifest']['candidate']]['label'])}</th>" for r in records),
        'RAW': json.dumps(records, allow_nan=False).replace('<', '\\u003c'),
        'GENERATED': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'SETTINGS': escape(f"{manifest['device'].upper()} · {manifest['dtype']} backbone · {manifest['feature_dtype']} head · "
                           f"{manifest['optimizer']} · learning rate {manifest['learning_rate']} · "
                           f"maximum {manifest['max_steps']} updates · {manifest['threads']} threads"),
        'SOURCES': ''.join(f"<li>{escape(MODELS[r['manifest']['candidate']]['label'])}: "
                           f"{escape(r['manifest']['repo'])}@<code>{escape(r['manifest']['revision'])}</code>" 
                           f"<br>Cached features: <code>{escape(r['feature_source'])}</code></li>" for r in records),
    }
    for key in ('loss', 'accuracy', 'probe_loss', 'probe_accuracy'):
        replacements[key.upper()] = curve(records, key)
    return re.sub(r'@@([A-Z_]+)@@', lambda match: replacements[match[1]], template)


def load_records(root=ROOT):
    records, unavailable = [], []
    for model in MODELS:
        repeated_path = root / 'results' / 'repeated' / f'{model}.json'
        source_path = root / 'results' / 'comparison' / f'{model}.json'
        source = json.loads(source_path.read_text()) if source_path.exists() else None
        record = json.loads(repeated_path.read_text()) if repeated_path.exists() else None
        reason = None
        if source and source.get('status') == 'failed':
            reason = 'Latest backbone run failed; previous results are excluded'
        elif record and source and record['feature_source'] != source['run_directory']:
            reason = 'Latest backbone run has not completed repeated-seed evaluation'
        if reason:
            unavailable.append({'manifest': {'candidate': model}, 'status': 'failed',
                                'failure_reason': reason})
        elif record is not None:
            records.append(record)
    return records, unavailable


if __name__ == '__main__':
    records, unavailable = load_records()
    output = ROOT / 'report-smoke.html'
    temporary = output.with_suffix('.html.tmp')
    temporary.write_text(build(records, unavailable))
    temporary.replace(output)
    print(f'Wrote {output}')
