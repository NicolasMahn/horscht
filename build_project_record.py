"""Build a single project record from explicitly allowed aggregate artifacts.

External benchmark access is confined to explicitly named aggregate result files.
Discussion summaries are editorial records, not executable experiment plans.
"""

import hashlib
import json
import math
import subprocess
from collections import Counter
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from build_current_training_report import chart, load as load_training
from report_theme import PALETTE

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "runs/project-record-2026-10-04.html"
HEAD_RUN = "runs/current-corpus-head-only-2026-10-04"
BENCHMARK_ROOT = "data/external_evaluation/results"
BENCHMARKS = {
    "original": f"{BENCHMARK_ROOT}/original-qwen-label-likelihood-2026-10-04/blind-aggregate-result.json",
    "paired": f"{BENCHMARK_ROOT}/paired-overnight-2026-09-30/blind-aggregate-result.json",
    "current": f"{BENCHMARK_ROOT}/current-corpus-benchmark-2026-10-04-final/blind-aggregate-result.json",
}
PRESENTATION = {
    "original": "purple", "previous": "teal", "current": "blue",
    "development": "blue", "training": "teal", "selected": "good",
    "recovery": "orange", "completed": "good", "proposed": "orange",
    "hypothesis": "purple",
}
SOURCES = {}


def read(path):
    source = ROOT / path
    content = source.read_bytes()
    SOURCES[path] = hashlib.sha256(content).hexdigest()
    return json.loads(content)


def link(path, label):
    return f'<a href="../{escape(path, quote=True)}">{escape(label)}</a>'


def table(headers, rows, caption):
    head = ''.join(f'<th scope="col">{escape(value)}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(
        f'<th scope="row">{value}</th>' if i == 0 else f'<td>{value}</td>'
        for i, value in enumerate(row)) + '</tr>' for row in rows)
    return f'<div class="scroll"><table><caption>{escape(caption)}</caption><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def detail(title, body, ident=None):
    anchor = f' id="{ident}"' if ident else ''
    return f'<details{anchor}><summary>{escape(title)}</summary><div class="detail-body">{body}</div></details>'


def current_head_state():
    if not (ROOT / HEAD_RUN / 'process.json').exists():
        return 'Not started', 'Fresh decision head on original backbone.', 'No current head-only job recorded.', ''
    process = read(f'{HEAD_RUN}/process.json')
    declaration = read(f'{HEAD_RUN}/declaration.json')
    status_path = f'{HEAD_RUN}/progress.json'
    progress = read(status_path) if (ROOT / status_path).exists() else {}
    state = 'Running · ' + progress.get('phase', 'starting')
    unit = subprocess.run(['systemctl', '--user', 'is-active', process['service']],
                          capture_output=True, text=True).stdout.strip()
    if unit != 'active':
        state = 'Stopped · verify saved status'
    updates = progress.get('global_update', 0)
    meaning = f'{updates:,} / 8,017 updates; same 128,263 training questions, fresh matched head and original frozen Qwen backbone.'
    body = f'<p>Launched {escape(process["start_time"])} as a durable user job. The initial head weights and frozen dataset selection match the completed FULL run. The backbone stays in evaluation mode with gradients disabled; only the 678,665 head parameters update.</p>'
    latest_dev = progress.get('latest_development')
    if latest_dev and not updates:
        body += f'<p>Initial development baseline: {latest_dev["questions_done"]:,} / {latest_dev["questions"]:,} questions in the current pass. No optimizer update has occurred yet.</p>'
    if 'measured_update_seconds' in progress:
        body += f'<p>Measured recent throughput: {progress["measured_questions_per_second"]:.2f} questions/second; {progress["measured_update_seconds"]:.2f} seconds/update. Estimated training completion {escape(progress["estimated_training_completion_at"])}; includes repeated development scoring and checkpoint verification, and can change with input lengths.</p>'
    report_path = f'{HEAD_RUN}/head/report.json'
    if (ROOT / report_path).exists():
        report = read(report_path)
        if report['status'] == 'completed' and report['selected_checkpoint_reload_verified']:
            state = 'Training completed'
            meaning = f'8,017 updates; development selected update {report["best_update"]:,}.'
            body += f'<p>Selected development soft cross-entropy {report["best_dev_soft_cross_entropy"]:.5f}; checkpoint reload verified. {link(f"{HEAD_RUN}/comparison/comparison.html", "Matched internal comparison")}</p>'
    benchmark_path = f'{HEAD_RUN}/benchmark-report.json'
    if (ROOT / benchmark_path).exists():
        benchmark = read(benchmark_path)
        aggregate_path = 'data/external_evaluation/results/current-corpus-head-only-2026-10-04/blind-aggregate-result.json'
        aggregate = read(aggregate_path)
        if SOURCES[aggregate_path] != benchmark['aggregate_sha256'] or aggregate['candidate_results']['head'] != benchmark['metrics']:
            raise ValueError('Head-only aggregate differs from its report binding')
        validate_metrics(benchmark['metrics'])
        metrics = benchmark['metrics']['overall']
        state = 'Training and external evaluation completed'
        body += f'<p>Exploratory external suite: {metrics["accuracy"]:.2%} accuracy ({metrics["correct"]:,} / {metrics["examples"]:,}), NLL {metrics["negative_log_likelihood"]:.4f}, Brier {metrics["multiclass_brier_score"]:.4f}. Accuracy change versus matched current FULL: {benchmark["accuracy_delta_percentage_points"]:+.2f} percentage points. Each checkpoint was frozen using development before external scoring.</p>'
    body += f'<p>{link(status_path, "Latest saved progress")} · {link(f"{HEAD_RUN}/declaration.json", "Matched experiment declaration")} · {link(f"{HEAD_RUN}/completion-status.json", "Completion and evaluation status")}</p>'
    gpu = 'GPU owned by frozen-backbone comparison' if unit == 'active' else 'Training GPU job ended; completion status records external evaluation.'
    return state, meaning, gpu, body


def validate_metrics(metrics):
    overall = metrics['overall']
    slices = list(metrics['by_source_slice'].values())
    for key in ('examples', 'correct'):
        if sum(part[key] for part in slices) != overall[key]:
            raise ValueError(f'Benchmark slice {key} differs from aggregate')
    for part in [overall, *slices]:
        if not math.isclose(part['correct'] / part['examples'], part['accuracy'], abs_tol=1e-12):
            raise ValueError('Benchmark accuracy differs from count ratio')
    for key in ('negative_log_likelihood', 'multiclass_brier_score'):
        weighted = sum(part[key] * part['examples'] for part in slices) / overall['examples']
        if not math.isclose(weighted, overall[key], abs_tol=1e-10):
            raise ValueError(f'Benchmark weighted {key} differs from aggregate')


def corpus_counts():
    path = 'data/training_snapshots/2026-10-01-v1/manifest.json'
    manifest = read(path)
    source = ROOT / 'data/training_snapshots/2026-10-01-v1/rows.jsonl'
    digest = hashlib.sha256()
    splits, languages, kinds, choices = Counter(), Counter(), Counter(), Counter()
    non_one_hot, review = 0, 0
    with source.open('rb') as stream:
        for line in stream:
            digest.update(line)
            row = json.loads(line)
            snapshot = row['_training_snapshot']
            splits[snapshot['split']] += 1
            if snapshot['split'] != 'train':
                continue
            languages[row['language']] += 1
            kinds[snapshot['source_kind']] += 1
            choices[row.get('choice_count', len(row['options']))] += 1
            non_one_hot += sum(p > 0 for p in row.get('target_distribution', [])) > 1
            review += 'review_recommended' in snapshot['cohorts']
    expected = manifest['artifacts']['rows.jsonl']
    if digest.hexdigest() != expected['sha256'] or sum(splits.values()) != expected['records']:
        raise ValueError('Frozen corpus does not match its manifest')
    if dict(splits) != {'train': 128263, 'dev': 7202, 'test': 7164}:
        raise ValueError('Frozen split cardinalities changed')
    if non_one_hot != 35069 or review != 23368:
        raise ValueError('Frozen target or review accounting changed')
    SOURCES['data/training_snapshots/2026-10-01-v1/rows.jsonl'] = digest.hexdigest()
    return splits, languages, kinds, choices, non_one_hot, review


def benchmark_chart(systems):
    parts = ['<svg viewBox="0 0 900 230" role="img" aria-label="External suite accuracy: original Qwen 51.72 percent, earlier full tune 45.57 percent, current full tune 40.19 percent. These are different systems, not a trajectory within one run.">']
    for i, (role, label, metrics) in enumerate(systems):
        y = 22 + 66 * i
        parts.append(f'<text x="0" y="{y+23}">{escape(label)}</text><rect class="bar {role}" x="260" y="{y}" width="{metrics["accuracy"]*950:.2f}" height="36" rx="5"/><text x="{274+metrics["accuracy"]*950:.2f}" y="{y+24}">{metrics["accuracy"]:.2%}</text>')
    parts.append('<text x="260" y="225">Same 17,874 items · different training snapshots/readouts</text></svg>')
    return ''.join(parts)


def build():
    head_state, head_meaning, head_gpu, head_body = current_head_state()
    observed = datetime.now(timezone.utc).isoformat(timespec='seconds')
    history, training, status, config, resume = load_training()
    for path in ('run-config.json', 'recovery-2026-10-04/full/report.json',
                 'recovery-2026-10-04/full/status.json', 'recovery-2026-10-04/process-exit.json',
                 'full/history.json', 'recovery-2026-10-04/full/history.json'):
        read(f'runs/current-corpus-2026-10-01/{path}')
    aggregates = {name: read(path) for name, path in BENCHMARKS.items()}
    original = aggregates['original']
    earlier = aggregates['paired']['candidate_results']['full']
    frozen = aggregates['paired']['candidate_results']['head']
    current = aggregates['current']['candidate_results']['full']
    for metrics in [original, earlier, frozen, current]:
        validate_metrics(metrics)
    if {m['overall']['examples'] for m in [original, earlier, frozen, current]} != {17874}:
        raise ValueError('Benchmark comparison denominators differ')
    if original['suite_reuse']['canonical_items_sha256'] != aggregates['current']['suite_reuse']['canonical_items_sha256']:
        raise ValueError('Original and current systems used different canonical suites')
    if aggregates['current']['suite_reuse']['previous_result_sha256'] != SOURCES[BENCHMARKS['paired']]:
        raise ValueError('Previous benchmark binding differs')
    attribution = read('runs/attribution-diagnostic-2026-10-04/summary.json')
    attr_status = read('runs/attribution-diagnostic-2026-10-04/status.json')
    read('runs/attribution-diagnostic-2026-10-04/protocol.json')
    if attr_status['phase'] != 'completed' or attribution['total_fixtures'] != 8 or not attribution['synthetic_only']:
        raise ValueError('Attribution diagnostic is not the completed synthetic suite')
    splits, languages, kinds, choices, soft, review = corpus_counts()
    breadth = read('runs/document-breadth-2026-10-03/results.json')
    luna = read('runs/document-luna-max-2026-10-02/summary.json')
    games = read('runs/game-breadth-2026-10-03/status.json')
    game_pilot = read('data/game_pilot/2026-10-02/manifest.json')
    game_expansion = read('data/game_expansion/2026-10-02/manifest.json')
    doc_pilot = read('data/document_pilot/2026-10-02/summary.json')
    dev_points = [point for point in history if 'dev' in point]
    baseline, selected, final = dev_points[0]['dev'], training['selected_checkpoint_development'], dev_points[-1]['dev']
    systems = [('original', 'Original Qwen · native LM head', original['overall']),
               ('previous', 'Earlier full tune', earlier['overall']),
               ('current', 'Current full tune · update 4,608', current['overall'])]
    benchmark_rows = [(label, f'{metrics["correct"]:,} / {metrics["examples"]:,}',
                       f'{metrics["accuracy"]:.2%}', f'{metrics["negative_log_likelihood"]:.4f}',
                       f'{metrics["multiclass_brier_score"]:.4f}') for _, label, metrics in systems]
    benchmark_rows.append(('Earlier frozen-backbone head', f'{frozen["overall"]["correct"]:,} / 17,874',
                           f'{frozen["overall"]["accuracy"]:.2%}', f'{frozen["overall"]["negative_log_likelihood"]:.4f}',
                           f'{frozen["overall"]["multiclass_brier_score"]:.4f}'))
    slices = []
    for key, metrics in current['by_source_slice'].items():
        label = key.replace('mmlu-pro-test', 'MMLU-Pro').replace('belebele-', 'Belebele · ').replace('bbh-', 'BBH · ').replace('-', ' ')
        slices.append((escape(label), f'{metrics["examples"]:,}',
                       f'{original["by_source_slice"][key]["accuracy"]:.2%}',
                       f'{earlier["by_source_slice"][key]["accuracy"]:.2%}', f'{metrics["accuracy"]:.2%}'))
    diagnostics = ''
    for key, title in [('choice_count_bin', 'Original Qwen by number of choices'), ('input_token_bin', 'Original Qwen by input length')]:
        rows = original['exploratory_diagnostics'][key]
        if sum(v['examples'] for v in rows.values()) != 17874:
            raise ValueError('Exploratory diagnostic denominator differs')
        diagnostics += table(['Stratum', 'Items', 'Original accuracy'],
                             [(escape(k), f'{v["examples"]:,}', f'{v["accuracy"]:.2%}') for k,v in rows.items()], title)
    internal_rows = []
    for label, key, percent in [('Cross-entropy ↓', 'soft_cross_entropy', False), ('KL ↓', 'kl_pq', False),
                                ('Brier ↓', 'categorical_brier_sum', False), ('Hard-label accuracy ↑ · 2,740', 'hard_accuracy', True),
                                ('Target argmax agreement ↑ · 7,202', 'argmax_agreement', True)]:
        internal_rows.append((label, *[(f'{m[key]:.2%}' if percent else f'{m[key]:.4f}') for m in [baseline, selected, final]]))
    attr_rows = []
    for key, label in [('original', 'Original native Qwen'), ('frozen_head', 'Earlier frozen head'),
                       ('early_full', 'Earlier full tune'), ('current_full', 'Current full tune')]:
        weight = attribution['parameter_changes'].get(key)
        activation = attribution['mean_final_activations'].get(key)
        attr_rows.append((label, f'{attribution["correct_counts"][key]} / 8',
                          f'{weight["relative_l2"]:.3%}' if weight else 'Unchanged reference backbone',
                          f'{activation["mean_token_cosine"]:.3f}' if activation else 'Reference / not measured'))
    choice_rows = []
    for lo, hi, label in [(2,4,'2–4'),(5,16,'5–16'),(17,64,'17–64'),(65,264,'65–264'),(265,265,'265')]:
        count = sum(v for k,v in choices.items() if lo <= k <= hi)
        choice_rows.append((label, f'{count:,}', f'{count/splits["train"]:.2%}'))
    overview = f'''
<header><p class="muted">Horscht · project record · 2026-10-04</p><h1>What we built, what we measured, and what is still open.</h1>
<p class="lead">The current full tune learned the internal corpus well. It trails original Qwen and the earlier full tune on the reused external suite. The completed attribution diagnostic shows substantial representation changes, but does not prove that knowledge was erased.</p>
<p class="muted">Snapshot generated {observed}. This is a saved record, not a live activity dashboard. Discussion summaries consolidate the saved project evidence and the coordinating conversation; they are not a verbatim transcript.</p></header>
<nav aria-label="Record contents"><a href="#results">Results</a><a href="#diagnostic">Attribution</a><a href="#corpus">Training data</a><a href="#discussion">Discussion and decisions</a><a href="#history">Earlier work</a><a href="#sources">Sources</a></nav>
<section class="surface"><p><strong>Current state</strong></p>
{table(['Work', 'State at this record', 'Meaning'], [
('Current full-backbone training', '<span class="completed">Completed</span>', 'One epoch; 8,017 updates. Selected checkpoint: 4,608.'),
('External evaluation and original comparison', '<span class="completed">Completed</span>', '17,874 reused items; aggregate-only reporting.'),
('Synthetic attribution diagnostic', '<span class="completed">Completed</span>', 'Eight probes, four systems; about 97 seconds.'),
('New full-dataset frozen-backbone experiment', escape(head_state), escape(head_meaning)),
('Choice-count and length comparison', 'Partial · not running', 'Original-model strata available; matched trained-model diagnostics pending.'),
('New game and document batches', 'Generated · not trained', 'Separate future-data artifacts; absent from the completed frozen run.'),
('GPU / background owners', escape(head_gpu), 'Durable training and completion jobs publish saved status.')], 'Work and status')}
<p>The earlier frozen-head control already exists. It used an earlier dataset and is <strong>not</strong> the newly discussed head-only experiment on the full current corpus.</p>
{head_body}</section>
<section class="surface" id="results"><p><strong>External results</strong></p>{benchmark_chart(systems)}
{table(['System', 'Correct / items', 'Accuracy ↑', 'NLL ↓', 'Brier ↓'], benchmark_rows, 'Same reused external suite')}
<p>Original Qwen exceeds current FULL by {100*(original['overall']['accuracy']-current['overall']['accuracy']):.2f} percentage points. Earlier FULL exceeds current FULL by {100*(earlier['overall']['accuracy']-current['overall']['accuracy']):.2f} points. These compare different training runs and readouts; they do not show monotonic benchmark decline across checkpoints of one run.</p>
<p>Original Qwen chooses more correct answers but has worse probability losses. Its native language-model label probabilities are often extremely confident. The current decision-head distribution has lower NLL and Brier here. Neither accuracy alone nor these scores establish calibrated confidence.</p>
<p class="muted">All three comparisons are now exploratory evidence from an observed suite. MMLU-Pro contributes 11,874 of 17,874 items; five Belebele languages contribute 4,500, and six BBH slices contribute 1,500. The mixed average is weighted by item count. Belebele translations are related semantic problems.</p>
{detail('Slice results, prompt differences, and evaluation boundary', table(['Slice','Items','Original Qwen','Earlier FULL','Current FULL'], slices, 'Aggregate accuracy by source slice') + '''
<p>MMLU-Pro moved from 37.52% in earlier FULL to 28.37% in current FULL; original Qwen scored 39.95%. All five Belebele language slices improved from earlier FULL to current FULL. BBH changes were mixed. A single weighted score hides these differences.</p>
<p>The two trained decision systems use the same external inference protocol but different training snapshots and checkpoints. Original Qwen uses native no-thinking chat, uppercase choice labels and restricted original LM-head label likelihood. The decision systems use plain numeric prompts and their learned decision heads. This is an end-to-end comparison; prompt, labels and readout confound any explanation in terms of backbone forgetting. It is not a generated chain-of-thought comparison.</p>
<p>The selected current checkpoint was fixed from development before evaluation. Exact grouping, deduplication and overlap checks do not establish freedom from semantic overlap or pretrained contamination. The suite has been reused and discussed, so further adaptive work must not call it an untouched new test. Benchmark items, answers and per-item errors are absent from this report.</p>''' + f'<p>Evaluation times: earlier paired {escape(aggregates["paired"]["evaluated_at_utc"])}; current {escape(aggregates["current"]["evaluated_at_utc"])}; original {escape(original["evaluated_at_utc"])}.</p>')}
{detail('Choice count and input length: available evidence and pending work', diagnostics + '''
<p>These strata exist only for the original-model method. Comparable trained-model strata have not been run. Choice cardinality, source family, difficulty and prompt length vary together; the tables do not identify a causal choice-count or length penalty. The longest stratum contains only 14 items. Comparing a long original chat prompt with a shorter decision prompt also requires a declared length definition.</p>
<p>Proposed follow-up: compute the same fixed cardinality/length strata for each system and inspect within source family. Treat this as exploratory descriptive analysis, keep content hidden, and do not use it to select a prompt or checkpoint on the external test.</p>''')}
</section>
<section class="surface"><p><strong>What the current training run actually completed</strong></p>
<p>Original pinned Qwen3-4B plus a fresh random decision head; all backbone weights were trainable. It consumed 128,263 training questions and 37,756,818 input tokens in one epoch. The original four-epoch plan was interrupted by user-manager shutdown. Verified recovery resumed optimizer and RNG state from update 7,865 in a separate output directory and was capped at one epoch for assessment.</p>
{table(['Development metric','Initial · 0','Selected · 4,608','Final · 8,017'], internal_rows, 'Internal development: selection evidence, not an independent benchmark')}
{chart([('development', [(p['update'],p['dev']['soft_cross_entropy']) for p in dev_points])], training['best_update'], resume, ylabel='Internal development cross-entropy')}
<p class="muted">Blue: every recorded full-development evaluation. Green marker: selected update 4,608. Orange marker: recovery at 7,865. Connected points are measured evaluations; intermediate values were not measured.</p>
<p>Checkpoint selection used the minimum development cross-entropy, {selected['soft_cross_entropy']:.4f}. The final checkpoint’s development loss was {final['soft_cross_entropy']:.4f}; completion does not make the final checkpoint the selected one. The selected checkpoint was reloaded and verified. Training completed {escape(status['completed_at'])}, with exit 0.</p>
<p>{link('runs/current-corpus-2026-10-01/results.html','Detailed training report, all loss curves and selected/final metrics')}</p>
{detail('Probability targets, IDK, hyperparameters, and remaining held-out tests', f'''
<p>Development has 7,202 questions, including 2,740 hard-label questions. Hard accuracy {selected['hard_accuracy']:.2%} and target argmax agreement {selected['argmax_agreement']:.2%} use different denominators. Cross-entropy contains target entropy, about 0.22747 nats; KL subtracts that constant. Agreement with a teacher estimate measures agreement, not truth or calibration.</p>
<p>Of the tagged development subsets, the selected model chose task-grounded IDK on {selected['task_grounded_idk']['unanswerable']['predicted_idk_fraction']:.2%} of 748 unanswerable questions and {selected['task_grounded_idk']['answerable']['predicted_idk_fraction']:.2%} of 3,102 answerable questions. Another 3,352 development questions lack answerability tags. A low false-IDK rate does not itself establish that the substantive answers are correct.</p>
<p>Backbone learning rate {config['backbone_learning_rate']:g}; head learning rate {config['head_learning_rate']:g}; seed {config['seed']}; microbatch one; accumulation 16; complete-input auditing with no silent truncation. Choice options, stable IDs, target distributions and evidence are permuted together. The full checkpoint stores backbone and head, optimizer, RNG and cursor state.</p>
<p>The 7,164-question internal source-held-out test has not been evaluated for this current checkpoint. The report’s null final-test field must not be mistaken for the completed external evaluation. No second epoch ran, and no claim is made about what later epochs would achieve. Superseded pre-recovery work is excluded from merged history.</p>''')}
</section>
<section class="surface" id="diagnostic"><p><strong>Attribution, backbone movement, and the forgetting question</strong></p>
{table(['System','Correct synthetic probes','Backbone relative L2 change','Final-layer mean token cosine'],attr_rows,'Eight fixed synthetic probes; comparison to original backbone')}
<p>The current full tune answered all eight probes correctly. The earlier frozen head answered five, even though its backbone was unchanged; original native Qwen answered seven. Therefore a drop in a complete system’s score can occur without a changed backbone. This makes the head/readout/prompt explanation plausible, but does not establish that it explains the external gap.</p>
<p>Relative backbone weight change was 0.697% for earlier FULL and 2.048% for current FULL. Mean aligned-token final-layer cosine was 0.722 and 0.428 respectively. The current model moved further. Distance is evidence of change, not a measurement of lost knowledge, preserved reasoning, or task capability.</p>
<p>{link('runs/attribution-diagnostic-2026-10-04/report.html','Full attribution report: masks, per-layer movement and exact group Shapley results')}</p>
{detail('What this diagnostic explains, and what it cannot explain', f'''
<p>The diagnostic was completed at {escape(attr_status['updated_at'])}; runtime {attribution['runtime']['elapsed_seconds']:.2f} seconds; peak GPU allocation {attribution['runtime']['peak_gpu_allocated_gib']:.2f} GiB. Four native systems were evaluated on eight fixed synthetic tasks, outside external benchmark content. No training or prompt/checkpoint selection occurred.</p>
<p>Three predetermined context groups yield eight deletion masks per probe. Exact group Shapley values average each group’s marginal effect on the chosen target-probability contrast over all subsets, relative to empty context. Question, options, order and full target stay fixed. This is exact finite group attribution under that background; it is not token SHAP, KernelSHAP, a reasoning trace, or a unique account of the computation.</p>
<p>Deleting facts changes length and can make inputs unnatural or underdetermined. Confidence, calibration, interactions and native wrapper differences affect attribution. Equal answers need not produce equal attributions, and changed attributions do not establish a new reasoning algorithm. The maximum efficiency residual was {attribution['max_shapley_efficiency_residual']:.2e}, confirming arithmetic reconstruction under the specified attribution game, not explanation validity.</p>
<p>Activation comparisons use identical complete numeric-prompt token IDs across original and fully tuned backbones, without fitted alignment. They measure aligned token cosine and relative L2 layer by layer, plus last-token values. Native original attribution separately uses its chat wrapper. The earlier frozen-head control is not matched to the current full corpus. Eight probes cannot settle broad catastrophic forgetting or retention.</p>
<p>The discussed alternatives included SHAP-style input sensitivity, layer activation comparisons and direct weight comparisons. The completed bounded diagnostic implements all three limited views. The matched frozen-backbone head experiment adds a controlled comparison using the same current corpus.</p>''')}
</section>
<section class="surface" id="corpus"><p><strong>The data that was actually trained</strong></p>
<p>The frozen 2026-10-01 corpus contains <strong>142,629 eligible unique originals</strong>: 128,263 train, 7,202 development and 7,164 held-out test. Later game/document data is absent. The previous 86,841-row snapshot and old progress entries describe earlier experiments.</p>
{table(['Presented choices','Train questions','Share'],choice_rows,'Actual frozen training choice cardinalities')}
<p>95.95% of training questions have two to four choices. Only 36 train questions use all 265 slots. Supporting 265 outputs in code is different from having broad training coverage of high-cardinality decisions.</p>
{detail('Languages, supervision, quality flags, and the million-question goal', table(['Recorded target source','Train questions'],[(escape(k.replace('_',' ')),f'{v:,}') for k,v in sorted(kinds.items())],'Actual frozen training supervision') + f'''
<p>The train split has {len(languages)} language codes; {sum(v <= 10 for v in languages.values())} codes have ten or fewer rows. A code count is not balanced language coverage or verified native fluency. Whitespace context-length counts (reported median around 55 and 90th percentile around 187) are a rough corpus description, especially weak across scripts; tokenizer-based complete-input audits are the training resource measure.</p>
<p>{soft:,} train targets are non-one-hot distributions ({soft/splits['train']:.2%}). Probability-source membership does not guarantee a non-degenerate target. {review:,} train rows carry review-recommended cohort flags; eligibility and structural validity do not prove semantic correctness. The corpus was frozen after explicit exclusions and exact duplicate removal; unknown semantic duplicates remain possible.</p>
<p>Answerability accounting reported by the coordinating audit: 68,226 rows contain an abstention choice; 54,582 are tagged answerable, 13,644 unanswerable, and 60,037 train rows are untagged. Task-grounded IDK means the supplied information cannot determine the exact target asked about. It is different from calibrated model-confidence rejection.</p>
<p>The target remains one million varied multilingual original questions. The saved eligible-original corpus represents 14.26% of that target, leaving 857,371 against that particular baseline. Newly isolated wave totals are not automatically added to the million-row production total; admission, deduplication, grouping and a new freeze are required first. Shuffles and translations do not create independent original problems.</p>
<p>Broad generic decision transfer is the goal: diverse operations, domains, difficulty, long-context integration, known stochastic distributions and genuinely missing-information questions. Collecting many surface variants or many games does not itself demonstrate generalization. Preserve reliable independent labels, group related tasks before splitting, and test transfer on independently designed tasks.</p>
<p>{link('data/training_snapshots/2026-10-01-v1/manifest.json','Frozen corpus manifest')} · {link('docs/probability-dataset-v2.md','Probability-v2 dataset contract')} · {link('docs/cross-context-authoring-contract-v1.md','Cross-context authoring contract')}</p>''')}
</section>
<section class="surface" id="discussion"><p><strong>Discussion and decisions to carry forward</strong></p>
{table(['Topic','Evidence / agreed interpretation','State'],[
('Single-pass decision system','Text context + question + changing supplied choices → masked categorical distribution across up to 265 slots. Stable choice IDs preserve target alignment.','Implemented'),
('Soft probabilities','Known laws, exact derivations and empirical frequencies are separate from teacher estimates. Proper scores and answerability require separate reporting.','Implemented contract; truth limits remain'),
('Option order','Permute options, labels and evidence together; report aligned order consistency. New choice-padding variants need reviewed impossibility proofs.','Implemented; future matched audit proposed'),
('Forgetting','FULL changed weights/activations; external gap is observed; prompt/head/readout differences remain confounded.','Hypothesis, not established finding'),
('Full-corpus frozen backbone','Original pinned backbone + fresh random head, same dataset and protocol as current FULL.',escape(head_state)),
('Data strategy','Improve trustworthy breadth rather than simply increasing template volume; exact game solvers and source-grounded documents supply different evidence.','Foundations implemented; training integration pending'),
('Qwen3.5 and images','Potential new backbone / multimodal route; current Qwen3-4B path is text-only. Images require new data/interface/resource work.','Deferred; no experiment launched'),
('Jev / published System One','82.70% published Jev MMLU-Pro snapshot; different harness, source deduplication and entrant setup.','Research recorded; no direct ranking claim')],'Implemented facts, hypotheses, proposals and deferred ideas')}
{detail('Why the matched control freezes the backbone', '''
<p>The question is whether preserving pretrained representations while teaching the decision interface can retain more useful general capability than full tuning. A fresh head on the original frozen backbone separates one major difference from the current FULL arm. It must use the same current train/dev splits, choice/permutation contract, complete-input policy and metric code. Select and freeze the head by internal development before scoring.</p>
<p>This control still compares entire systems and does not prove all representations remain useful to the head. Head optimization, prompt/readout mismatch, dataset bias and limited high-cardinality coverage can constrain performance. The earlier frozen-feature result is useful history but uses an earlier cohort and is not this matched control.</p>
<p>The external suite is already observed; the matched comparison on it is exploratory. An independently frozen fresh system test provides stronger generalization evidence. Training was explicitly authorized and launched separately; this document records its saved state.</p>''')}
{detail('265 changing choices, honest distributions, and “I don’t know”', '''
<p>The output slots represent the positions of the supplied options, not 265 permanently named concepts. Per-row masking excludes unused slots. Option identities, text, target mass, hard indexes and evidence must move together under augmentation. The IDK choice is part of the maximum 265, not an extra 266th class. Slot-264 gradient/update/reload checks exist; purposeful high-cardinality data remains sparse in the completed corpus.</p>
<p>Known stochastic uncertainty is answerable as a distribution. Missing information about a realized fact can justify IDK. A question about which missing measurement is needed may be answerable even though that measurement is unavailable. Do not turn arbitrary uncertainty into IDK or give IDK credit when a substantive choice already answers the asked question.</p>
<p>Adding impossible choices is a proof-gated child augmentation, preserving source group and split. Exact zero mass is allowed only when closed-world support and impossibility are justified. Unobserved empirical outcomes or low teacher estimates do not prove impossibility. Do not mechanically pad every task to 265, count shuffles as new originals, or equate a softmaxed engine score with calibrated real-world probability.</p>
<p>A separate reject policy can use calibrated correctness confidence and explicit utility. The discussed +1 correct / −4 incorrect / 0 reject example yields a theoretical 0.8 threshold, conditional on calibration; it is not a fitted optimum. Tune any threshold on development and report coverage, selective risk, utility and answerability separately.</p>''' + f'<p>{link("docs/probability-v2-choice-augmentation.md","Choice augmentation contract")} · {link("docs/idk-task-target-clarification-v1.md","IDK target clarification")} · {link("docs/probability-v2-decision-metrics.md","Decision metrics")}</p>')}
{detail('Games, document access, diversity, images, and model choice', '''
<p>Games were discussed as reusable sources of decisions with simulators and exact or carefully qualified labels: action legality, consequences, planning, irreversible moves, inference from public observations, probability laws and missing information. A working simulator is not an optimal labeler. Label finite-law outcomes, optimal supplied-action policies and IDK distinctly; keep rule versions, observations, objectives, search budgets and related episodes together.</p>
<p>The broad catalogue covers board games, puzzles, card/dice games, hidden-information and cooperative games, resource management and text-state video games. Only the adapters listed in the generated-data record below are implemented. Chess/Stockfish and the rest of the catalogue remain backlog. Start broadly with reliable modest samples, then deepen valuable families; question count, decision depth and engineering effort are separate budgets.</p>
<p>Documents add knowledge-grounded decisions and evidence integration across context. Source candidates included multilingual reporting, Wikipedia, public information and scientific text. Access and rights must be checked per item/edition; preserve revision, content hash, attribution, license, exact evidence and related translation/event groups. Acquisition is bounded and cache-first. The implemented pipeline uses cleared Global Voices, Wikipedia, GOV.UK and selected PLOS sources; this does not authorize indiscriminate scraping or a blanket dataset license.</p>
<p>Multilingual diversity means meaningful native-language tasks with review, not English content under another language tag. Independent model review improves gates but can share mistakes. Rights and exact source spans do not prove semantic necessity or linguistic quality. Distribution tasks use explicit source counts or stated finite sampling, not invented confidence about an uncertain news event.</p>
<p>Images and a Qwen3.5 route were discussed but deferred. The completed Qwen3 model is text-only. Multimodal support would require a compatible backbone, image-aware inputs/choices, provenance and labels, and a separately measured resource/evaluation protocol. No claimed image capability or model replacement exists.</p>''' + f'<p>{link("docs/game-and-document-data-roadmap.md","Full prospective game/document catalogue")} · {link("docs/article-source-catalogue.md","Document source catalogue")} · {link("runs/source-access-audit-2026-10-02/summary.json","Saved access audit")}</p>')}
{detail('Published Jev and other System One results', '''
<p>The saved metadata-only Decision Index 0.2.1 snapshot (generated 2026-09-28; Jev 1.13.0) reports MMLU-Pro: Jev 82.70%, Decider 4B 51.65%, Tev1-4B experimental 45.37%, Decision 1.0 Nox 36.69%, each on 12,032 requests. Our suite retains 11,874 deduplicated rows. These are directional comparisons across different harnesses, prompts, readouts and training data, not an identical-harness leaderboard.</p>
<p>Decider’s card describes one forward pass and letter-logit readout on a full tune of Qwen3.5-4B-Base. Tev1 also uses that backbone; its single-forward/no-CoT behavior was not independently verified in the saved research. Board architecture labels do not establish whether a decision uses generation. Published latency is across different hardware and inference/network paths. The pooled published BBH score is not our selected BBH-slice aggregate.</p>
<p>A metadata research page accidentally exposed a small malformed item-bearing field to an isolated reporting agent after the original experiment was frozen and completed. Content is not repeated or forwarded here. This conversation’s MMLU-Pro boundary must not be described as untouched for future design. No absence-of-contamination claim follows from an index penalty field.</p>
<p>“Jeff” in dictated messages refers to Jev.</p>''' + f'<p>{link("docs/system-one-comparison.md","Saved primary-source research, links, harness details and exposure record")}</p>')}
</section>
'''
    waves = [
        ('Game pilot · 2026-10-02', f'{game_pilot["accepted_originals"]:,}', '158 / 34', 'Nim, tic-tac-toe, Connect Four; exact reference mechanics', link('data/game_pilot/2026-10-02/manifest.json','Manifest')),
        ('Game expansion · 2026-10-02', f'{game_expansion["accepted_originals"]:,}', '8,137 / 1,527', 'Connect Four, Lights Out, grid navigation, Nim; 336 quarantined', link('data/game_expansion/2026-10-02/manifest.json','Manifest')),
        ('Game breadth · 2026-10-03', f'{games["accepted_originals"]:,}', '7,674 / 1,233', '3,000 dice/EV; 2,907 Mastermind; 3,000 Sokoban', link('runs/game-breadth-2026-10-03/status.json','Verified status')),
        ('Document pilot · 2026-10-02', f'{doc_pilot["accepted_originals"]:,}', '40 / 8', 'Six documents / six languages; corrected Sol routing provenance', link('docs/document-pilot-2026-10-02.md','Record')),
        ('Document expansion · 2026-10-02', '299', '205 / 94', '22 accepted documents / 14 languages; stopped on quality blocker', link('docs/document-expansion-2026-10-02.md','Record')),
        ('Document Luna MAX · 2026-10-02', f'{luna["status"]["accepted_new_originals"]:,}', '1,791 / 488', '145 documents / 25 languages; every accepted row independently model-reviewed', link('docs/document-luna-max-2026-10-02.md','Record')),
        ('Document breadth · 2026-10-03', f'{breadth["targets_and_outcome"]["accepted_originals"]:,}', '3,090 / 807', '429 documents / 44 source-native languages; 1,065 non-one-hot targets', link('runs/document-breadth-2026-10-03/results.md','Record')),
    ]
    extra = f'''
<section class="surface"><p><strong>New data: separate artifacts, absent from current training</strong></p>
{table(['Batch','Accepted originals','Train / dev','Coverage / qualifications','Evidence'],waves,'Bounded game and document waves')}
<p>These are per-batch counts, not a new unified training manifest or a revised million-question total. Keep related worlds, episodes, sources and translations in one split, retain quarantines, screen overlaps, then admit through explicit allowlists in a future frozen dataset.</p>
{detail('Game breadth: exact labels and bounded capabilities', '''
<p>The latest game breadth wave produced 9,000 candidates and excluded 93 redundant-feedback duplicates, leaving 8,907 unique accepted originals. Exact finite-law dice/expected value, public-history Mastermind inference and bounded Sokoban planning are distinct label mechanisms. Every final record was reconstructed and relabelled; archived generator, output and prior hashes were verified. The owner reports 79 focused tests.</p>
<p>Generation took 24.131 seconds and 468.04 MiB peak RSS on serial CPU, with no GPU or data-authoring API calls. Implementation assistance is separately recorded. Ten new originals have 265 choices. All wording in this wave is English; no multilingual coverage is claimed. Family weights are recorded for future sampling but have not affected active training.</p>
<p>Mastermind inputs contain public feedback, not hidden secrets. Information choices are optimal among supplied guesses under a stated uniform prior. Sokoban uses small boards with one or two boxes and has only 14 development rows after conservative terrain grouping; no stable family performance claim is justified. Dice is an explicit one-step finite-law game, not a complete Blackjack or multistep policy. Chess remains unimplemented.</p>''')}
{detail('Document author comparison, quality gates, costs and production accounting', '''
<p>The corrected author A/B v2 used six identical frozen documents and 24 matched assignments per arm. Luna MEDIUM accepted 16/24; Luna MAX, Sol HIGH and Sol LOW each accepted 23/24. All 96 submissions passed the shared structural contract. Primary blind review, balanced secondary review and adjudication were recorded. One session per arm and a tiny source set do not establish a general model ranking.</p>
<p>V2 shared deterministic schema/offset/arithmetic assembly so authors wrote task content directly. V1 remains preserved and is not pooled. Across arms, 85 matched accepted records contain 84 unique exact question/options pairs. Duplicate submissions remain valid matched comparison records but do not add original training-question credit. Conditional Standard-short author-only cost per accepted was $0.00073 Luna MEDIUM, $0.00060 Luna MAX, $0.01147 Sol HIGH and $0.00955 Sol LOW. Total observed review/adjudication/coordination scenario was $1.61749, or $0.01926 per unique pair. These are API-equivalent scenarios, not actual billed costs.</p>
<p>The earlier routing error used full-history forks that inherited Sol despite a requested Luna override. Original receipts are preserved, with exact-hash provenance overlays and corrected exports. The later Luna MAX wave used verified model/effort metadata and independent semantic gates. The 2,279 accepted rows include 606 non-one-hot distributions and 300 whole-context integration tasks; the aspirational 40% soft-target share was not forced.</p>
<p>The later document breadth wave closed with 3,897 accepted originals from 5,164 reviewed candidates. It covered 429 accepted source documents and 44 native language tags, with 1,065 meaningful non-one-hot targets, 361 IDK tasks, 411 integration tasks and 242 multi-document comparisons. Capacity and source-mix hold release accounting, science source identity repairs and completed lifecycle receipts are retained. Review ratings remain model judgments, not human ground truth.</p>
<p>Auxiliary 265-choice revisions and aligned permutations validate the late head and target alignment; they receive zero new-original credit and remain separate from ordinary accepted original exports. Mixed licenses retain per-item attribution/share-alike obligations; no single blanket license replaces them.</p>
<p>Historical saved production accounting at 2026-10-01 reported 149,626 structurally valid rows, 6,503 known-issue exclusions, 494 exact duplicates removed, and 142,629 eligible originals. Review-recommended saved production counts (26,446) differ from the train-only frozen cohort (23,368). Keep the denominator explicit. The historical overnight API-equivalent estimate was $197.00; whole audited project $418.38. Those snapshots are not final actual charges and do not automatically include later waves.</p>''' + f'<p>{link("runs/document-author-ab-v2-2026-10-02/results.md","Corrected A/B v2 results")} · {link("runs/document-breadth-2026-10-03/results.json","Document breadth aggregate")} · {link("runs/usage-audit/api-equivalent-2026-10-01.json","Historical usage audit")}</p>')}
</section>
<section class="surface" id="history"><p><strong>Chronology and earlier experiments</strong></p>
{detail('2026-09-28 to 2026-09-29 · small decision-head experiments and corpus foundations', '''
<p>The early work established a decision-head training/readout/checkpoint contract and compared small backbones, base/post-trained variants and reasoning variants. The matched 64-question comparison used ten seeds and a separate new-task transfer view. Qwen3 post-trained averaged 88.1% on familiar tasks and 71.4% on new tasks; Qwen3 Base averaged 82.2% and 77.2%. Familiar development results chose the candidate; exploratory transfer did not retroactively change selection. DeepScaleR was excluded because the overfitting/reload eligibility contract was not met for one required run.</p>
<p>Size comparisons, repeated seeds, transfer probes and pilot GPU arms remain preserved as distinct experiments. Small familiar-family success cannot be substituted for broad generic reasoning or external benchmark performance. The 32-train/16-dev/16-test paired GPU pilot was exploratory, used only locked train IDs for updates, and selected by development. Random tensor and synthetic throughput/resource smokes are not task benchmarks.</p>
<p>Probability-v2 introduced explicit full target distributions, rational/exact evidence, grouped partitions, task-grounded answerability, changing cardinalities through 265 and sidecar exclusions. Production authoring guidance was revised as internal authoring/DEV defects appeared. Legacy raw rows and fixed assignment manifests were preserved. Generation paused, then separately authorized later; old handoff pause gates remain historical.</p>''' + f'<p>{link("report.html","Small-model summary")} · {link("report-size.html","Size comparison")} · {link("report-transfer.html","Transfer comparison")} · {link("docs/probability-dataset-v2.md","Dataset design")} · {link("docs/training-evaluation-protocol.md","Pilot chronology and evaluation protocol")}</p>')}
{detail('2026-09-30 to 2026-10-01 · larger paired run, overnight corpus and full-current freeze', '''
<p>Generation and two larger training arms were separately authorized on 2026-09-30. The older frozen snapshot had 86,841 total rows. The earlier paired head-only and full-backbone experiments were scored on the locked external suite; head-only reached 37.33% mixed accuracy and FULL 45.57%. Their MMLU-Pro results were 22.44% and 37.52%. These are completed historical arms, not the later proposed full-current frozen-backbone control.</p>
<p>The overnight saved corpus reached 142,629 eligible unique originals after explicit invalid-row and exact-content filtering. Generation was paused by user instruction; unfinished ledger assignments are saved work, not live authors. Cross-context plans, reviewed exclusion sidecars, allocation corrections and conditional usage accounting were retained.</p>
<p>A fresh 2026-10-01 snapshot preserved train/dev/test assignments. Current FULL started from original pinned Qwen and a new random decision head, not the earlier learned checkpoint. It trained 128,263 rows with complete-input audits, deterministic aligned choice shuffles, full-development checkpoints and resource caps. The original plan allowed up to four epochs.</p>''' + f'<p>{link("docs/overnight-plan-2026-09-30.md","Overnight plan and ownership")} · {link("data/generated_train/probability-v2/progress.md","Saved production snapshot")} · {link("runs/current-corpus-2026-10-01/run-config.json","Current run configuration")}</p>')}
{detail('2026-10-02 to 2026-10-03 · bounded data pilots, authoring quality and broader exact tasks', '''
<p>The discussion shifted from sheer row volume to reliable diversity and meaningful context. Bounded document/game pilots established acquisition, provenance, license, evidence, simulator, label and split infrastructure. The document expansion stopped after a real author-quality blocker rather than forcing its target. Provenance was corrected after model routing was diagnosed. The supported author comparison informed the bounded Luna MAX wave.</p>
<p>Independent model gates and deterministic source/target checks produced separate document artifacts. Game adapters used ordinary solver/simulation code for exact outcomes and stated policies. Later breadth waves added dice/EV, Mastermind, Sokoban and more multilingual publishers/science/public-information documents. Their completed counts are above; none was added to the running frozen model. Older bulk coordinators stayed paused.</p>
<p>The broad game and document roadmap remains prospective beyond these implementations. Images, a new backbone and larger native-language/general transfer claims were discussed but deferred. Source access and rights audits are evidence for selected acquisition, not unlimited permission or a correctness certificate.</p>''')}
{detail('2026-10-04 · verified completion, external comparisons and attribution assessment', f'''
<p>The user-manager shutdown interrupted original current-corpus training. Hash/state validation supported a separate recovery from update 7,865 with the same frozen data and code. Recovery finished the first epoch at 8,017, reloaded selected update 4,608 and stopped with exit 0 at {escape(status['completed_at'])}.</p>
<p>The current selected checkpoint’s external evaluation completed at {escape(aggregates['current']['evaluated_at_utc'])}. It was below earlier FULL overall, despite improved internal development and improved Belebele slices. Original native Qwen’s same-suite evaluation completed at {escape(original['evaluated_at_utc'])}, with higher accuracy and worse probability losses. The different original prompt/head means this cannot isolate backbone damage.</p>
<p>Questions about forgetting, choice count, context length, SHAP/input sensitivity and layer movement led to the completed synthetic attribution assessment at {escape(attr_status['updated_at'])}. It showed current FULL 8/8 probes with larger weight/activation movement, and the earlier frozen head 5/8 despite unchanged backbone. These observations narrowed the interpretation but did not resolve general capability retention.</p>
<p>The full-current frozen-backbone/fresh-head experiment was explicitly authorized and launched at 2026-10-04T16:12:54+00:00. Its saved state is {escape(head_state)}. Original choice/length strata are available; trained comparison strata are pending. Current training and completion receipts distinguish this job from historical owners.</p>''')}
</section>
<section class="surface"><p><strong>What remains unresolved</strong></p>
<p>We do not yet know how much of the external gap comes from head/readout alignment, prompts, training distribution, choice cardinality, model adaptation, forgetting, or the differing old/current cohorts. We also do not know whether newly generated games/documents improve non-game transfer, long-context integration, or multilingual reliability: they have not been trained or measured in this run.</p>
<p>The useful next evidence is a matched fresh head on the frozen original backbone, internal held-out evaluation under a declared boundary, and a separately designed frozen system test. Keep the observed external suite exploratory, preserve raw artifacts, and change one major experimental difference at a time. This follows the Pragmatic Programmer principle of deliberate experiments and avoiding accidental complexity.</p></section>
'''
    source_rows = [(link(path,path),f'<code>{digest}</code>') for path,digest in sorted(SOURCES.items()) if not path.endswith('rows.jsonl')]
    source_section = '<section class="surface" id="sources"><p><strong>Evidence and reproducibility</strong></p><p>The builder reads only explicitly named external aggregate files. All core tables and curves are embedded; no CDN, external script or network dependency is needed to read this record. Linked detailed reports and source files are optional drill-downs and require the repository to remain beside this file.</p>'
    source_section += detail('Exact source files and SHA-256 bindings', table(['Source artifact','SHA-256'],source_rows,'Input bindings at record generation') + f'<p>Frozen corpus rows SHA-256: <code>{SOURCES["data/training_snapshots/2026-10-01-v1/rows.jsonl"]}</code>. Read internally for aggregate train accounting; corpus text is not embedded.</p><p>The current training history merge/selection/reload validation and the benchmark count/weighted-metric/suite-binding validations run before rendering. Static conversation summaries distinguish observations, interpretation, proposals and implementation; they do not authorize jobs.</p>')
    source_section += f'<p>{link("build_project_record.py","Rebuild script")} · {link(".cursor/rules/blind-benchmarks.mdc","Blind benchmark rule")} · {link("AGENTS.md","Project coordination instructions")}</p></section>'
    theme = ''.join(f'--{key}:{value};' for key,value in PALETTE.items())
    theme += ''.join(f'--{role}:var(--{token});' for role,token in PRESENTATION.items())
    style = '''
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:36px 26px}h1{font-size:34px;line-height:1.2;letter-spacing:-.025em;max-width:880px;margin:12px 0 20px}p{margin:12px 0}.lead{font-size:19px;max-width:970px}.muted,small{color:var(--muted)}nav{display:flex;gap:10px 24px;flex-wrap:wrap;margin:26px 0}a{color:var(--blue);text-underline-offset:3px}a:focus-visible,summary:focus-visible,button:focus-visible{outline:3px solid var(--blue);outline-offset:4px}.surface{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:24px;margin:22px 0}.surface>p:first-child{font-size:19px;margin-top:0}details{border-top:1px solid var(--line);padding:16px 0}summary{cursor:pointer;font-weight:600;padding:2px 0}.detail-body{padding:8px 0 0}details:last-child{padding-bottom:0}.scroll{overflow-x:auto;margin:20px 0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}caption{text-align:left;color:var(--muted);padding-bottom:8px;font-size:13px}td,th{text-align:left;border-bottom:1px solid var(--line);padding:12px 10px;vertical-align:top}thead th{color:var(--muted);font-size:13px}tbody th{font-weight:500}td{min-width:76px}code{font-size:12px;overflow-wrap:anywhere}.completed{color:var(--completed);font-weight:650}.proposed{color:var(--proposed);font-weight:650}svg{width:100%;height:auto;display:block;margin:18px 0}svg text{font:15px system-ui;fill:var(--muted)}.bar.original{fill:var(--original)}.bar.previous{fill:var(--previous)}.bar.current{fill:var(--current)}.grid{stroke:var(--line);stroke-width:1}.trace{fill:none;stroke-width:2.5;stroke:var(--development)}.point{fill:var(--development)}.marker{stroke-width:1.5;stroke-dasharray:5 4}.marker.selected{stroke:var(--selected)}.marker.recovery{stroke:var(--recovery)}button{font:inherit;border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:6px;padding:7px 12px;cursor:pointer}.toolbar{display:flex;gap:10px;margin-top:20px}footer{color:var(--muted);font-size:13px;margin:24px 0}@media(max-width:650px){main{padding:20px 14px}h1{font-size:28px}.lead{font-size:17px}.surface{padding:17px}td,th{padding:10px 8px}table{min-width:530px}nav{gap:8px 18px}svg text{font-size:16px}}@media print{body{background:var(--paper)}main{max-width:none;padding:0}.surface{border-radius:0;break-inside:avoid}.toolbar,nav{display:none}summary{font-size:18px}a{color:var(--ink)}.scroll{overflow:visible}table{font-size:11px}footer{font-size:10px}}
'''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Horscht · consolidated project record · 2026-10-04</title><style>:root{{{theme}}}{style}</style></head><body><main>{overview}{extra}{source_section}<div class="toolbar" hidden id="controls"><button id="expand">Expand all details</button><button id="collapse">Collapse all details</button><button id="print">Print / save PDF</button></div><footer>Saved project record · {observed} · All core results embedded; historical artifacts preserved.</footer></main><script>document.getElementById('controls').hidden=false;document.getElementById('expand').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=true);document.getElementById('collapse').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=false);document.getElementById('print').onclick=()=>{{document.querySelectorAll('details').forEach(d=>d.open=true);window.print()}};</script></body></html>'''


def main():
    output = build()
    temporary = OUTPUT.with_suffix('.html.tmp')
    temporary.write_text(output)
    temporary.replace(OUTPUT)
    print(OUTPUT)


if __name__ == '__main__':
    main()
