"""Development-probe selection rule fixed before the added candidates run."""

from statistics import mean

EXPECTED_SEEDS = tuple(range(42, 52))
POLICY = {
    'version': 1,
    'locked_at': '2026-09-28T21:24:12Z',
    'timing': 'chosen after inspecting the first four models, before results from the five added candidates',
    'primary': 'maximize mean final development-probe choice accuracy',
    'tie_break': 'minimize mean final development-probe cross-entropy',
    'exact_tie': 'report all tied candidates',
    'seeds': list(EXPECTED_SEEDS),
    'scope': 'provisional development selection; these eight probes have already been inspected',
}


def ineligibility(record):
    if record.get('status') == 'failed':
        return record.get('failure_reason', 'Candidate run failed')
    runs = record.get('runs', [])
    if sorted(r.get('seed', -1) for r in runs) != list(EXPECTED_SEEDS):
        return 'Requires all ten seeds (42–51), once each'
    verification = record.get('source_reload_verification', {})
    if not (verification.get('features_close') is True and verification.get('predictions_identical') is True):
        return 'Backbone reload verification did not pass'
    for run in runs:
        training = run.get('training', {})
        final = training.get('final', {})
        if not (run.get('passed') is True and run.get('reloaded_head_logits_identical') is True
                and final.get('accuracy') == 1 and 0 <= final.get('loss', float('inf')) <= .02
                and training.get('weight_change_l2', 0) > 0):
            return 'Every seed must pass overfitting and saved-head reload checks'
        probe = run.get('probe', {})
        if not (0 <= probe.get('accuracy', -1) <= 1
                and 0 <= probe.get('loss', -1) < float('inf')):
            return 'Missing or invalid final probe measurements'
    return None


def select(records, expected_candidates):
    """Never declare the comparison complete while a candidate is absent or failed."""
    by_candidate = {r['manifest']['candidate']: r for r in records}
    if len(by_candidate) != len(records):
        raise ValueError('Duplicate candidate records')
    ranked, excluded = [], []
    for candidate in expected_candidates:
        record = by_candidate.get(candidate)
        reason = 'Results not available' if record is None else ineligibility(record)
        if reason:
            excluded.append({'candidate': candidate, 'reason': reason})
            continue
        ranked.append({'candidate': candidate,
                       'accuracy': mean(r['probe']['accuracy'] for r in record['runs']),
                       'loss': mean(r['probe']['loss'] for r in record['runs'])})
    ranked.sort(key=lambda item: (-item['accuracy'], item['loss']))
    leaders = [r['candidate'] for r in ranked
               if (r['accuracy'], r['loss']) == (ranked[0]['accuracy'], ranked[0]['loss'])]
    return {'complete': not excluded, 'leaders': leaders, 'ranked': ranked, 'excluded': excluded,
            'policy': POLICY}
