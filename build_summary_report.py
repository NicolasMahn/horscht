"""One compact comparison of the matched 64-question experiment."""

import json
import re
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import mean

from build_size_report import eligibility_reason, load_records, selection, validate
from report_theme import MODELS, PALETTE
from selection_policy import EXPECTED_SEEDS

ROOT = Path(__file__).parent


def transfer_scores(records, sizes):
    scores = {}
    reference = None
    for record in records:
        if record.get('status') != 'complete':
            continue
        candidate = record['manifest']['candidate']
        runs = record.get('runs', [])
        if sorted(r['seed'] for r in runs) != list(EXPECTED_SEEDS):
            continue
        if len(record['examples']) != 64:
            raise ValueError('Transfer comparison requires 64 questions')
        signature = {key: record['manifest'][key] for key in
                     ('dataset_manifest', 'source_data_sha256', 'settings', 'seeds')}
        if reference is not None and signature != reference:
            raise ValueError('Transfer comparison protocols differ')
        reference = signature
        source = sizes.get(candidate)
        if source and record['manifest']['source_data_sha256'] != source['manifest']['data_sha256']:
            raise ValueError('Transfer head source differs from main experiment')
        values = [r['conditions']['64']['accuracy'] for r in runs]
        if not all(0 <= value <= 1 for value in values):
            raise ValueError('Invalid transfer accuracy')
        scores[candidate] = mean(values)
    return scores


def build(records, transfers=(), unavailable=(), generated=None):
    if records:
        validate(records)
    by_name = {r['manifest']['candidate']: r for r in records}
    selection_result = selection(records, '64')
    transferred = transfer_scores(transfers, by_name)
    missing = dict(unavailable)
    ranked = sorted(records, key=lambda r: (
        eligibility_reason(r, '64') is not None,
        -mean(run['report']['probe']['accuracy'] for run in r['conditions']['64']['runs']),
        mean(run['report']['probe']['loss'] for run in r['conditions']['64']['runs'])))
    order = [r['manifest']['candidate'] for r in ranked]
    order += [name for name in MODELS if name not in by_name]
    leaders = selection_result['leaders']
    labels = ', '.join(MODELS[name]['label'] for name in leaders)
    if leaders:
        score = mean(r['report']['probe']['accuracy'] for r in by_name[leaders[0]]['conditions']['64']['runs'])
        recommendation = f'Current pick: {escape(labels)}'
        rationale = f'{score:.1%} on familiar task families after training on 64 questions.'
        if len(leaders) > 1:
            rationale += ' These models tie on accuracy and loss.'
    else:
        recommendation = 'No eligible pick yet'
        rationale = 'A model must finish all ten runs and pass the training and reload checks.'
    eligible_count = sum(eligibility_reason(r, '64') is None for r in records)
    rows = []
    for name in order:
        record = by_name.get(name)
        reason = eligibility_reason(record, '64') if record else missing.get(name, 'Measurements not available')
        score = '—'
        if record:
            values = [run['report']['probe']['accuracy'] for run in record['conditions']['64']['runs']]
            score = f'<span title="Ten-seed range: {min(values):.1%}–{max(values):.1%}">{mean(values):.1%}</span>'
        novel = f'{transferred[name]:.1%}' if name in transferred else '—'
        if not record:
            status = 'Pending' if reason == 'Measurements not available' else 'Unavailable'
        elif reason:
            status = 'Not eligible'
        elif name not in transferred:
            status = 'Ready · new tasks pending'
        else:
            status = 'Ready'
        detail = f' title="{escape(reason, quote=True)}"' if reason else ''
        state = 'excluded' if record and reason else 'ready' if record else 'pending'
        pick = ' <span class="pick">Current pick</span>' if name in leaders else ''
        rows.append(f'<tr class="{state}"><th scope="row">{escape(MODELS[name]["label"])}{pick}</th>'
                    f'<td>{score}</td><td>{novel}</td><td{detail}>{status}</td></tr>')
    exclusions = ''.join(f'<li>{escape(MODELS[r["manifest"]["candidate"]]["label"])}: '
                         f'{escape(eligibility_reason(r, "64"))}.</li>'
                         for r in records if eligibility_reason(r, '64'))
    transfer_note = ''
    eligible_transfer = {name: value for name, value in transferred.items()
                         if name in by_name and eligibility_reason(by_name[name], '64') is None}
    if leaders and leaders[0] in eligible_transfer and eligible_transfer:
        best = max(eligible_transfer, key=eligible_transfer.get)
        if eligible_transfer[best] > eligible_transfer[leaders[0]]:
            transfer_note = (f'{escape(MODELS[best]["label"])} scores higher on new tasks '
                             f'({eligible_transfer[best]:.1%}). Those exploratory scores do not choose the winner.')
    replacements = {
        'THEME': ''.join(f'--{key}:{value};' for key, value in PALETTE.items()),
        'RECOMMENDATION': recommendation, 'RATIONALE': rationale,
        'PROGRESS': f'{len(records)} / {len(MODELS)} models measured · {eligible_count} eligible · {len(transferred)} with new-task results',
        'STATUS': ('Provisional development comparison. All models measured; exclusions are shown below.'
                   if len(records) == len(MODELS) and eligible_count < len(MODELS)
                   else 'Provisional development comparison. All models measured.'
                   if len(records) == len(MODELS)
                   else 'Provisional: the comparison is still incomplete.'),
        'ROWS': ''.join(rows), 'TRANSFER_NOTE': transfer_note,
        'EXCLUSIONS': f'<ul>{exclusions}</ul>' if exclusions else '',
        'GENERATED': escape(generated or datetime.now(timezone.utc).isoformat(timespec='seconds')),
    }
    template = (ROOT / 'summary_report_template.html').read_text()
    return re.sub(r'@@([A-Z_]+)@@', lambda match: replacements[match[1]], template)


def main():
    records, unavailable = load_records(ROOT)
    transfers = [json.loads(path.read_text()) for candidate in MODELS
                 if (path := ROOT / 'results' / 'transfer' / f'{candidate}.json').exists()]
    output = ROOT / 'report.html'
    temporary = output.with_suffix('.html.tmp')
    temporary.write_text(build(records, transfers, unavailable))
    temporary.replace(output)
    print(output)


if __name__ == '__main__':
    main()
