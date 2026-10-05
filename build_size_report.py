"""Build the separate, matched 16-versus-64 training-question report."""

import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

from build_report import escape
from build_repeated_report import curve
from report_theme import MODELS, PALETTE
from selection_policy import EXPECTED_SEEDS

ROOT = Path(__file__).parent
SIZES = ('16', '64')


def validate(records):
    if not records:
        raise ValueError('No measured size-comparison results')
    reference = records[0]
    candidates = [r['manifest']['candidate'] for r in records]
    if len(set(candidates)) != len(candidates):
        raise ValueError('Duplicate candidate')
    for record in records:
        manifest = record['manifest']
        if record['schema_version'] != 1 or record['experiment'] != 'matched-training-size':
            raise ValueError('Unsupported size-comparison schema')
        if manifest['candidate'] not in MODELS:
            raise ValueError('Unknown candidate')
        for key in ('data_sha256', 'dataset_locked_at', 'settings', 'seeds', 'subsets', 'probe_ids'):
            if manifest[key] != reference['manifest'][key]:
                raise ValueError(f'Comparison setting differs: {key}')
        if record['examples'] != reference['examples']:
            raise ValueError('Comparison examples differ')
        if manifest['seeds'] != list(EXPECTED_SEEDS):
            raise ValueError('Expected seeds 42–51')
        examples = {ex['id']: ex for ex in record['examples']}
        if len(examples) != len(record['examples']):
            raise ValueError('Duplicate example ids')
        probes = manifest['probe_ids']
        if len(probes) != 64 or len(set(probes)) != 64:
            raise ValueError('Expected 64 distinct shared probes')
        if any(examples[i]['split'] != 'probe' for i in probes):
            raise ValueError('Probe ids must refer to probe examples')
        if not set(manifest['subsets']['16']) <= set(manifest['subsets']['64']):
            raise ValueError('Training subsets must be nested')
        for size in SIZES:
            condition = record['conditions'][size]
            ids = condition['training_ids']
            if ids != manifest['subsets'][size] or len(ids) != int(size) or len(set(ids)) != int(size):
                raise ValueError('Training subset size or ids differ')
            if set(ids) & set(probes) or any(examples[i]['split'] != 'train' for i in ids):
                raise ValueError('Training and probes must be disjoint')
            runs = condition['runs']
            if [run['seed'] for run in runs] != list(EXPECTED_SEEDS):
                raise ValueError('Incomplete or mismatched paired seeds')
            for run in runs:
                report = run['report']
                training = report['training']
                history = training['history']
                if [p['step'] for p in history] != list(range(training['steps'] + 1)):
                    raise ValueError('Incomplete learning curve')
                for point in history:
                    for key in ('loss', 'accuracy', 'probe_loss', 'probe_accuracy'):
                        if not math.isfinite(point[key]):
                            raise ValueError('Non-finite curve measurement')
                if not (0 <= report['probe']['accuracy'] <= 1 and 0 <= report['probe']['loss'] < math.inf):
                    raise ValueError('Invalid final probe metric')
                if (history[-1]['probe_accuracy'] != report['probe']['accuracy']
                        or history[-1]['probe_loss'] != report['probe']['loss']):
                    raise ValueError('Final probe metrics differ from curve endpoint')


def paired_changes(record):
    smaller = {r['seed']: r['report']['probe']['accuracy'] for r in record['conditions']['16']['runs']}
    larger = {r['seed']: r['report']['probe']['accuracy'] for r in record['conditions']['64']['runs']}
    if set(smaller) != set(larger):
        raise ValueError('Paired seeds differ')
    return [larger[seed] - smaller[seed] for seed in EXPECTED_SEEDS]


def eligibility_reason(record, size):
    verification = record.get('cache_verification', {})
    if not (verification.get('features_close') is True and verification.get('predictions_identical') is True):
        return 'Fresh feature / prediction verification did not pass'
    for run in record['conditions'][size]['runs']:
        report = run['report']
        training = report['training']
        final = training['final']
        if report.get('reloaded_head_logits_identical') is not True:
            return f"{size} questions, seed {run['seed']}: saved-head reload did not pass"
        if not (final['accuracy'] == 1 and 0 <= final['loss'] <= .02 and training['weight_change_l2'] > 0):
            return f"{size} questions, seed {run['seed']}: overfitting target not reached"
        if report.get('passed') is not True:
            return f"{size} questions, seed {run['seed']}: run checks did not pass"
    if record.get('status') != 'complete':
        return 'Paired experiment has not finished'
    return None


def eligible(record, size):
    return eligibility_reason(record, size) is None


def selection(records, size):
    scores = [(mean(r['report']['probe']['accuracy'] for r in record['conditions'][size]['runs']),
               mean(r['report']['probe']['loss'] for r in record['conditions'][size]['runs']),
               record['manifest']['candidate']) for record in records if eligible(record, size)]
    scores.sort(key=lambda row: (-row[0], row[1]))
    leaders = [name for accuracy, loss, name in scores if (accuracy, loss) == scores[0][:2]]
    complete = {r['manifest']['candidate'] for r in records} == set(MODELS) and len(scores) == len(MODELS)
    return {'leaders': leaders, 'complete': complete}


def selection_text(records):
    paragraphs = []
    for size in SIZES:
        result = selection(records, size)
        labels = ', '.join(MODELS[name]['label'] for name in result['leaders']) or 'No eligible candidate'
        status = 'provisional selection' if result['complete'] else 'leading available results; comparison incomplete'
        tie = ' · exact tie on both metrics' if len(result['leaders']) > 1 else ''
        paragraphs.append(f'<p><strong>{size} training questions: {escape(labels)}</strong> · {status}{tie}</p>')
    return ''.join(paragraphs)


def summary_rows(records):
    rows = []
    for record in records:
        cells = []
        for size in SIZES:
            runs = record['conditions'][size]['runs']
            accuracy = [r['report']['probe']['accuracy'] for r in runs]
            losses = [r['report']['probe']['loss'] for r in runs]
            cells.extend([f'{mean(accuracy):.2%} ± {100 * stdev(accuracy):.2f} pp',
                          f'{mean(losses):.4f} ± {stdev(losses):.4f}'])
        delta = paired_changes(record)
        cells.extend([f'{100 * mean(delta):+.2f} ± {100 * stdev(delta):.2f} pp',
                      f'{100 * min(delta):+.2f} to {100 * max(delta):+.2f} pp',
                      escape('; '.join(eligibility_reason(record, size) or f'{size} questions: eligible' for size in SIZES))])
        label = MODELS[record['manifest']['candidate']]['label']
        rows.append(f'<tr><th>{escape(label)}</th>' + ''.join(f'<td>{cell}</td>' for cell in cells) + '</tr>')
    return ''.join(rows)


def distributions(records, differences=False):
    fig = go.Figure()
    for record in records:
        style = MODELS[record['manifest']['candidate']]
        if differences:
            values = [100 * value for value in paired_changes(record)]
            fig.add_trace(go.Box(x=values, name=style['label'], orientation='h', boxpoints='all',
                                jitter=0, pointpos=0, marker_color=PALETTE[style['color']],
                                customdata=list(EXPECTED_SEEDS), showlegend=False))
        else:
            for size, symbol in [('16', 'circle-open'), ('64', 'diamond')]:
                values = [r['report']['probe']['accuracy'] for r in record['conditions'][size]['runs']]
                fig.add_trace(go.Scatter(x=values, y=[style['label']] * len(values), mode='markers',
                                        name=f"{style['label']} · {size}", showlegend=False,
                                        marker=dict(color=PALETTE[style['color']], symbol=symbol, size=10),
                                        customdata=list(EXPECTED_SEEDS),
                                        hovertemplate=f'{size} questions · seed %{{customdata}}: %{{x:.2%}}<extra>%{{y}}</extra>'))
    fig.update_layout(height=max(420, 45 * len(records) + 100), template='plotly_white',
                      margin=dict(l=200, r=25, t=20, b=60),
                      xaxis_title='Paired accuracy change · 64 minus 16 (percentage points)' if differences else 'Final probe accuracy',
                      paper_bgcolor=PALETTE['paper'], plot_bgcolor=PALETTE['paper'], font_color=PALETTE['ink'])
    if differences:
        fig.add_vline(x=0, line_dash='dot', line_color=PALETTE['muted'])
    else:
        fig.update_xaxes(range=[0, 1], tickformat='.0%')
    return pio.to_html(fig, full_html=False, include_plotlyjs=False,
                       div_id='paired-changes' if differences else 'final-distribution',
                       config={'responsive': True, 'displaylogo': False})


def condition_records(records, size):
    """Adapt one size to the existing descriptive learning-curve renderer."""
    return [{'manifest': record['manifest'],
             'examples': [ex for ex in record['examples'] if ex['id'] in
                          set(record['manifest']['subsets'][size] + record['manifest']['probe_ids'])],
             'runs': [dict(run['report'], seed=run['seed']) for run in record['conditions'][size]['runs']]}
            for record in records]


def build(records, unavailable=()):
    if records:
        validate(records)
    theme = ':root{' + ''.join(f'--{key}:{value};' for key, value in PALETTE.items()) + '}'
    curves = []
    for size in SIZES if records else ():
        charts = []
        for key, title in [('loss', 'Training loss'), ('probe_loss', 'Probe loss'),
                           ('accuracy', 'Training accuracy'), ('probe_accuracy', 'Probe accuracy')]:
            # Plot ids must remain unique when reusing the same metric for two conditions.
            chart = curve(condition_records(records, size), key).replace(f'"{key}"', f'"size-{size}-{key}"')
            charts.append(f'<section class="surface chart"><h3>{title}</h3>{chart}</section>')
        curves.append(f'<h2>{size} training questions</h2><div class="grid">' + ''.join(charts) + '</div>')
    replacements = {
        'THEME': theme, 'PLOTLY': get_plotlyjs(), 'MODELS': str(len(records)),
        'TOTAL': str(sum(len(r['conditions'][size]['runs']) for r in records for size in SIZES)),
        'SUMMARY': summary_rows(records), 'SELECTION': selection_text(records),
        'UNAVAILABLE': ''.join(f'<p>{escape(MODELS[name]["label"])}: {escape(reason)}</p>' for name, reason in unavailable),
        'DISTRIBUTIONS': distributions(records) if records else '<p>No complete measurements yet.</p>',
        'CHANGES': distributions(records, True) if records else '<p>No complete measurements yet.</p>',
        'CURVES': ''.join(curves),
        'SETTINGS': escape(json.dumps(records[0]['manifest']['settings'], sort_keys=True)) if records else 'No completed measurement bundle yet.',
        'LOCKED': escape(records[0]['manifest']['dataset_locked_at']) if records else 'not available in completed results',
        'RAW': json.dumps(records, allow_nan=False).replace('<', '\\u003c'),
        'GENERATED': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    template = (ROOT / 'size_report_template.html').read_text()
    return re.sub(r'@@([A-Z_]+)@@', lambda match: replacements[match[1]], template)


def load_records(root=ROOT):
    records, unavailable = [], []
    for candidate in MODELS:
        path = root / 'results' / 'size_comparison' / f'{candidate}.json'
        if not path.exists():
            unavailable.append((candidate, 'Measurements not available'))
            continue
        record = json.loads(path.read_text())
        conditions = record.get('conditions', {})
        complete = all(len(conditions.get(size, {}).get('runs', [])) == len(EXPECTED_SEEDS)
                       and all('report' in run for run in conditions[size]['runs']) for size in SIZES)
        if not record.get('manifest') or not complete:
            unavailable.append((candidate, str(record.get('error', 'Partial measurements; paired evaluation incomplete'))))
            continue
        records.append(record)
    return records, unavailable


if __name__ == '__main__':
    records, unavailable = load_records()
    output = ROOT / 'report-size.html'
    temporary = output.with_suffix('.html.tmp')
    temporary.write_text(build(records, unavailable))
    temporary.replace(output)
    print(f'Wrote {output}')
