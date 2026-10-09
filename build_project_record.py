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
# The familiar URL shows the short dashboard; the full record is one click away.
DASHBOARD = ROOT / "runs/project-record-2026-10-04.html"
RECORD = ROOT / "runs/project-record-2026-10-04-details.html"
HEAD_RUN = "runs/current-corpus-head-only-2026-10-04"
LAST_BLOCK_RUN = "runs/current-corpus-last-block-2026-10-05"
ATTENTION_RUN = "runs/current-corpus-option-attention-2026-10-06"
NEW_DATA_RUN = "runs/new-data-last-block-2026-10-08"
BENCHMARK_ROOT = "data/external_evaluation/results"
BENCHMARKS = {
    "original": f"{BENCHMARK_ROOT}/original-qwen-label-likelihood-2026-10-04/blind-aggregate-result.json",
    "paired": f"{BENCHMARK_ROOT}/paired-overnight-2026-09-30/blind-aggregate-result.json",
    "current": f"{BENCHMARK_ROOT}/current-corpus-benchmark-2026-10-04-final/blind-aggregate-result.json",
}
PRESENTATION = {
    "matched": "blue", "reference": "orange",
    "head-only": "blue", "last-block": "green", "option-attention": "magenta", "full": "orange", "original": "muted",
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
    if unit == 'active' and 'measured_update_seconds' in progress:
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


def current_last_block_state():
    if not (ROOT / LAST_BLOCK_RUN / 'process.json').exists():
        return 'Not started', 'Fresh head and final transformer block.', '', ''
    process = read(f'{LAST_BLOCK_RUN}/process.json')
    declaration = read(f'{LAST_BLOCK_RUN}/declaration.json')
    status_path = f'{LAST_BLOCK_RUN}/progress.json'
    if not (ROOT / status_path).exists():
        status_path = f'{LAST_BLOCK_RUN}/training-status.json'
    progress = read(status_path) if (ROOT / status_path).exists() else {}
    unit = subprocess.run(['systemctl', '--user', 'is-active', process['service']], capture_output=True, text=True).stdout.strip()
    state = 'Running · ' + progress.get('phase', 'starting') if unit == 'active' else 'Stopped · verify saved status'
    if unit == 'active' and progress.get('global_update') == 8017:
        state = 'Training exposure completed · checkpoint verification running'
    meaning = f'{progress.get("global_update", 0):,} / 8,017 updates; matched frozen corpus, fresh head and original Qwen.'
    body = f'<p>Last-block arm launched {escape(process["start_time"])}. {escape(declaration["training_scope"])} Same immutable selection, seed, fresh head initialization, choice permutations and one-epoch budget as current FULL and current head-only: 128,263 questions and 37,756,818 tokens. Adafactor uses head learning rate 0.001 and last-block rate 0.00001.</p>'
    durability_path = f'{LAST_BLOCK_RUN}/durability-proof.json'
    if (ROOT / durability_path).exists():
        durability = read(durability_path)
        if durability['linger_after'] is not True:
            raise ValueError('Last-block user-service persistence is unverified')
        body += f'<p>User-service persistence across logout was enabled and verified {escape(durability["verified_at"])}. {link(durability_path, "Service durability receipt")}</p>'
    manifest_path = f'{LAST_BLOCK_RUN}/last-block/manifest.json'
    if (ROOT / manifest_path).exists():
        manifest = read(manifest_path)
        body += f'<p>Trainable backbone parameters: {manifest["trainable_backbone_parameter_count"]:,}; head: {manifest["head_parameter_count"]:,}. Frozen prefix builds no gradient graph; the frozen final normalization propagates gradients into the final block. Initial head SHA-256 matches the earlier arms. Complete full-format checkpoints preserve optimizer, cursor and RNG state.</p>'
    initial_path = f'{LAST_BLOCK_RUN}/initial-comparison-proof.json'
    if (ROOT / initial_path).exists():
        initial = read(initial_path)
        if not initial['same_initial_head'] or not initial['runtime_baseline_match']:
            raise ValueError('Last-block initialization is not matched')
        body += f'<p>The initial 7,202-question runtime baseline matches both reference arms exactly. {link(initial_path, "Initialization comparison proof")}</p>'
    gradient_path = f'{LAST_BLOCK_RUN}/last-block/first-gradient-proof.json'
    if (ROOT / gradient_path).exists():
        gradient = read(gradient_path)
        if not all(gradient[key] for key in ('all_trainable_backbone_gradients_present', 'frozen_backbone_gradients_absent', 'head_gradients_present')):
            raise ValueError('Last-block gradient proof failed')
        body += f'<p>The first optimizer update reached the allowed final-block and head parameters; frozen backbone parameters received no gradients. {link(gradient_path, "First-gradient proof")}</p>'
    early_frozen_path = f'{LAST_BLOCK_RUN}/early-frozen-weights-proof.json'
    if (ROOT / early_frozen_path).exists():
        frozen = read(early_frozen_path)
        if frozen['all_frozen_weights_bitwise_equal_to_original'] is not True:
            raise ValueError('Last-block early frozen-weight proof failed')
        body += f'<p>At saved update {frozen["checkpoint_update"]:,}, all {frozen["frozen_parameters_checked"]:,} frozen parameters remained byte-identical to the pinned original; {frozen["changed_trainable_tensors"]} final-block tensors changed. {link(early_frozen_path, "Learned-checkpoint frozen-weight verification")}</p>'
    exposure_path = f'{LAST_BLOCK_RUN}/first-learned-updates-proof.json'
    if (ROOT / exposure_path).exists():
        exposure = read(exposure_path)
        if not exposure['matched_update_exposure'] or not exposure['matched_epoch_plan']:
            raise ValueError('Last-block first-update exposure differs')
        body += f'<p>{link(exposure_path, "First 20 updates: matched exposure, measured speed and memory")}</p>'
    if unit == 'active' and 'measured_update_seconds' in progress:
        body += f'<p>Measured recent throughput: {progress["measured_questions_per_second"]:.2f} questions/second; {progress["measured_update_seconds"]:.2f} seconds/update. Estimated training completion {escape(progress["estimated_training_completion_at"])}; repeated development evaluations and selected-checkpoint verification are included.</p>'
    best_path = f'{LAST_BLOCK_RUN}/last-block/best-checkpoint.json'
    if (ROOT / best_path).exists():
        best = read(best_path)
        body += f'<p>Saved development minimum: cross-entropy {best["loss"]:.5f} at update {best["update"]:,}. {link(best_path, "Development checkpoint selection")}</p>'
    early_dev_path = f'{LAST_BLOCK_RUN}/development-update-00000512-comparison.json'
    if (ROOT / early_dev_path).exists():
        early_dev = read(early_dev_path)
        early_metrics = early_dev['metrics']
        body += f'<p>At matched update 512, development cross-entropy was {early_metrics["last-block"]["soft_cross_entropy"]:.5f} for last-block, {early_metrics["head-only"]["soft_cross_entropy"]:.5f} for head-only and {early_metrics["FULL"]["soft_cross_entropy"]:.5f} for FULL. These are early development results; the one-epoch schedule remains unchanged. {link(early_dev_path, "Matched-update development comparison")}</p>'
    report_path = f'{LAST_BLOCK_RUN}/last-block/report.json'
    if (ROOT / report_path).exists():
        report = read(report_path)
        if report['status'] == 'completed' and report['selected_checkpoint_reload_verified']:
            state = 'Training completed'
            meaning = f'8,017 updates; development selected update {report["best_update"]:,}.'
            body += f'<p>Selected development cross-entropy {report["best_dev_soft_cross_entropy"]:.5f}; reload verified. {link(f"{LAST_BLOCK_RUN}/comparison.json", "Matched exposure and selection comparison")}</p>'
    external_running = False
    external_status_path = f'{BENCHMARK_ROOT}/current-corpus-last-block-2026-10-05/run-status.json'
    if (ROOT / external_status_path).exists():
        external_status = read(external_status_path)
        external_running = external_status['phase'] == 'candidate_running'
        if external_running:
            state = 'Training completed · external evaluation running'
            body += f'<p>Serial external evaluation has processed {external_status["processed_items"]:,} / {external_status["total_items"]:,} items as of {escape(external_status["updated_at_utc"])}. {link(external_status_path, "Safe aggregate evaluation progress")}</p>'
    benchmark_path = f'{LAST_BLOCK_RUN}/benchmark-report.json'
    if (ROOT / benchmark_path).exists():
        benchmark = read(benchmark_path)
        aggregate_path = f'{BENCHMARK_ROOT}/current-corpus-last-block-2026-10-05/blind-aggregate-result.json'
        aggregate = read(aggregate_path)
        if SOURCES[aggregate_path] != benchmark['aggregate_sha256'] or aggregate['candidate_results']['last-block'] != benchmark['metrics']:
            raise ValueError('Last-block aggregate differs from its report binding')
        frozen = read(f'{LAST_BLOCK_RUN}/selected-frozen-weights-proof.json')
        if frozen['all_frozen_weights_bitwise_equal_to_original'] is not True:
            raise ValueError('Last-block frozen-weight proof failed')
        validate_metrics(benchmark['metrics'])
        state = 'Training and external evaluation completed'
        rows = []
        for label, metrics in [('Current head-only', benchmark['comparison_head_only']), ('Current last-block', benchmark['metrics']), ('Current FULL', benchmark['comparison_current_full'])]:
            validate_metrics(metrics)
            metric = metrics['overall']
            mmlu = metrics['by_source_slice']['mmlu-pro-test']
            rows.append((label, f'{metric["correct"]:,} / {metric["examples"]:,}', f'{metric["accuracy"]:.2%}', f'{mmlu["accuracy"]:.2%}', f'{metric["negative_log_likelihood"]:.4f}', f'{metric["multiclass_brier_score"]:.4f}'))
        body += table(['Matched arm', 'Correct / items', 'Accuracy', 'MMLU-Pro', 'NLL', 'Brier'], rows, 'Same observed 17,874-item suite; exploratory reuse; development-selected checkpoints')
        body += f'<p>External evaluation completed {escape(benchmark["evaluated_at_utc"])}. All frozen backbone weights were checked against the original pinned weights before scoring. {link(f"{LAST_BLOCK_RUN}/selected-frozen-weights-proof.json", "Frozen-weight verification")}</p>'
        verification_path = f'{LAST_BLOCK_RUN}/final-verification.json'
        if (ROOT / verification_path).exists():
            verification = read(verification_path)
            if verification['aggregate_sha256'] != benchmark['aggregate_sha256'] or not verification['selection_hash_bindings_verified'] or not verification['aggregate_totals_verified']:
                raise ValueError('Last-block final verification differs')
            body += f'<p>Final aggregate, selected-checkpoint bindings, matched reference metrics and successful service exits verified {escape(verification["verified_at"])}. {link(verification_path, "Final completion verification")}</p>'
    body += f'<p>{link(status_path, "Last-block saved progress")} · {link(f"{LAST_BLOCK_RUN}/declaration.json", "Last-block experiment declaration")} · {link(f"{LAST_BLOCK_RUN}/completion-status.json", "Last-block completion and evaluation status")}</p>'
    gpu = 'GPU owned by last-block external evaluation' if external_running else 'GPU owned by last-block comparison' if unit == 'active' else 'Last-block training GPU job ended; completion receipt records evaluation.'
    return state, meaning, gpu, body


def attention_training_output(process=None):
    """Resolve the saved attention output pointer without allowing it to leave its run."""
    if process is None:
        process_path = ROOT / ATTENTION_RUN / 'process.json'
        process = read(f'{ATTENTION_RUN}/process.json') if process_path.exists() else {}
    run_root = (ROOT / ATTENTION_RUN).resolve()
    declared = process.get('training_output')
    if declared is None:
        candidate = (run_root / 'attention-last-block').resolve()
    elif not isinstance(declared, str) or not declared.strip():
        raise ValueError('Option-attention training output pointer is invalid')
    else:
        pointer = Path(declared)
        candidate = (pointer if pointer.is_absolute() else ROOT / pointer).resolve()
    if not candidate.is_relative_to(run_root):
        raise ValueError('Option-attention training output must remain inside its run directory')
    if candidate.exists() and not candidate.is_dir():
        raise ValueError('Option-attention training output is not a directory')
    return candidate


def current_attention_state():
    """Describe the final old-corpus architecture run from its saved receipts."""
    progress_path = f'{ATTENTION_RUN}/progress.json'
    if not (ROOT / progress_path).exists():
        progress_path = f'{ATTENTION_RUN}/training-status.json'
    process_path = f'{ATTENTION_RUN}/process.json'
    completion_path = f'{ATTENTION_RUN}/completion-status.json'
    process = read(process_path) if (ROOT / process_path).exists() else {}
    output_directory = attention_training_output(process)
    output_relative = output_directory.relative_to(ROOT.resolve()).as_posix()
    report_path = f'{output_relative}/report.json'
    benchmark_path = f'{ATTENTION_RUN}/benchmark-report.json'
    has_receipt = any((ROOT / path).exists() for path in (progress_path, process_path, completion_path, report_path, benchmark_path))
    if not has_receipt:
        return 'Not started', 'Option-conditioned attention head plus final transformer block.', '', ''

    progress = read(progress_path) if (ROOT / progress_path).exists() else {}
    completion = read(completion_path) if (ROOT / completion_path).exists() else {}
    training_report = read(report_path) if (ROOT / report_path).exists() else {}
    phase = progress.get('phase', 'starting')
    status = progress.get('status')
    unit = 'unknown'
    if process.get('service'):
        unit = subprocess.run(['systemctl', '--user', 'is-active', process['service']],
                              capture_output=True, text=True).stdout.strip()

    verified_training = (training_report.get('status') == 'completed'
                         and training_report.get('selected_checkpoint_reload_verified') is True)
    benchmark = attention_benchmark_report() if (ROOT / benchmark_path).exists() else None
    if benchmark is not None and not verified_training:
        raise ValueError('Option-attention external report exists without verified training completion')

    if benchmark is not None:
        # The finisher rebuilds this report before marking itself complete.
        # Verified training and the hash-bound aggregate establish evaluation completion.
        state = 'Training and external evaluation completed'
        metric = benchmark['metrics']['overall']
        meaning = (f'{metric["correct"]:,} / {metric["examples"]:,} aggregate items; '
                   f'accuracy {metric["accuracy"]:.2%}; final planned old-corpus evaluation.')
        body = (f'<p>The option-conditioned attention head (dimension 256) and final transformer block were trained on the frozen current corpus. '
                f'The final development-selected checkpoint was verified before aggregate-only external scoring. '
                f'This is the last planned external test for the old training corpus; further training and evaluation wait for newly admitted data.</p>')
        body += (f'<p>Development selected update {training_report["best_update"]:,} with cross-entropy '
                 f'{training_report["best_dev_soft_cross_entropy"]:.5f}. {link(report_path, "Selected development report")}</p>')
        body += f'<p>Exploratory external suite: {metric["accuracy"]:.2%} accuracy ({metric["correct"]:,} / {metric["examples"]:,}), NLL {metric["negative_log_likelihood"]:.4f}, Brier {metric["multiclass_brier_score"]:.4f}. {link(benchmark_path, "Aggregate-only benchmark report")}</p>'
    else:
        if status == 'preparing':
            state = f'Preparing · {phase}'
        elif status == 'running':
            state = f'Training · {phase}' if unit == 'active' or unit == 'unknown' else 'Training stopped · verify saved status'
        elif status == 'interrupted':
            state = 'Training interrupted'
        elif status == 'failed':
            state = 'Training failed'
        elif verified_training or status == 'completed':
            selection = completion.get('phase')
            if selection == 'starting blind':
                state = 'Training completed · external evaluation starting'
            elif selection == 'frozen':
                state = 'Checkpoint frozen · external evaluation pending'
            else:
                state = 'Training completed · external evaluation pending'
        elif completion.get('phase') == 'frozen':
            state = 'Prepared · checkpoint frozen, waiting for training'
        elif completion.get('phase') == 'starting blind':
            state = 'External evaluation starting'
        elif completion.get('phase') == 'waiting':
            state = 'Preparing · waiting for training'
        elif process:
            state = 'Service launched · awaiting saved training status' if unit == 'active' else 'Stopped · verify saved status'
        else:
            state = 'Prepared · awaiting saved training status'
        updates = progress.get('global_update', 0)
        total = progress.get('total_questions', 128263)
        latest_development = progress.get('latest_development')
        if latest_development and not updates:
            meaning = (f'Initial development baseline: {latest_development["questions_done"]:,} / '
                       f'{latest_development["questions"]:,}; no optimizer update yet.')
        else:
            meaning = f'{updates:,} / 8,017 updates; {progress.get("questions_seen", 0):,} / {total:,} training questions seen.'
        body = ('<p>The final architecture arm pairs an option-conditioned attention head with Qwen’s final transformer block (block 35); '
                'the embedding and preceding 35 blocks stay frozen. It uses the same frozen corpus, prompt, loss and one-epoch budget as the matched arms. '
                'This is the last planned old-corpus test. Further training and evaluation wait for newly admitted game and document data.</p>')
        if progress:
            observed = progress.get('observed_at') or progress.get('updated_at_utc')
            if latest_development and not updates:
                detail = (f'Initial development baseline: {latest_development["questions_done"]:,} / '
                          f'{latest_development["questions"]:,} questions in the current pass. No optimizer update has occurred yet.')
                if observed:
                    detail += f' Saved as of {escape(observed)}.'
                body += f'<p>{detail}</p>'
            elif observed:
                body += f'<p>Saved training status: {escape(status or "unknown")} · {escape(phase)}; {updates:,} / 8,017 updates and {progress.get("questions_seen", 0):,} training questions seen as of {escape(observed)}.</p>'
            if status == 'running' and 'measured_update_seconds' in progress:
                body += (f'<p>Measured throughput: {progress["measured_questions_per_second"]:.2f} questions/second; '
                         f'{progress["measured_update_seconds"]:.2f} seconds/update. Estimated completion '
                         f'{escape(progress["estimated_training_completion_at"])}.</p>')
        if verified_training:
            body += (f'<p>Selected development cross-entropy {training_report["best_dev_soft_cross_entropy"]:.5f} at update '
                     f'{training_report["best_update"]:,}; checkpoint reload verified. No external result is recorded yet. '
                     f'{link(report_path, "Selected development report")}</p>')
        if completion:
            body += f'<p>{link(completion_path, "Completion and evaluation status")}</p>'
    if (ROOT / process_path).exists():
        body += f'<p>{link(progress_path, "Latest saved progress")} · {link(process_path, "Training process receipt")}</p>'
    gpu = ('GPU owned by option-attention training or evaluation' if unit == 'active'
           else 'Option-attention job ended; saved receipts determine completion.')
    return state, meaning, gpu, body


def attention_benchmark_report():
    """Load only the fixed aggregate result bound to the final architecture report."""
    report_path = f'{ATTENTION_RUN}/benchmark-report.json'
    if not (ROOT / report_path).exists():
        return None
    report = read(report_path)
    aggregate_path = f'{BENCHMARK_ROOT}/current-corpus-option-attention-2026-10-06/blind-aggregate-result.json'
    aggregate = read(aggregate_path)
    if report.get('aggregate_path') != aggregate_path:
        raise ValueError('Option-attention aggregate path differs from its allowlist')
    if SOURCES[aggregate_path] != report.get('aggregate_sha256'):
        raise ValueError('Option-attention aggregate differs from its report binding')
    metrics = report['metrics']
    if aggregate['candidate_results']['last-block'] != metrics:
        raise ValueError('Option-attention metrics differ from the bound aggregate')
    validate_metrics(metrics)
    return report


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


def matched_arms():
    """External metrics for completed arms trained on the same frozen corpus."""
    path = f'{LAST_BLOCK_RUN}/benchmark-report.json'
    arms = {}
    if (ROOT / path).exists():
        report = read(path)
        arms.update({'head-only': report['comparison_head_only'], 'last-block': report['metrics'],
                     'full': report['comparison_current_full']})
    attention = attention_benchmark_report()
    if attention is not None:
        arms['option-attention'] = attention['metrics']
    for metrics in arms.values():
        validate_metrics(metrics)
    return arms


def bar(x, y, width, height, radius=4):
    """Bar rounded at the data end and square at the baseline."""
    radius = min(radius, width)
    return (f'M{x:.2f},{y:.2f}h{width-radius:.2f}a{radius},{radius} 0 0 1 {radius},{radius}'
            f'v{height-2*radius:.2f}a{radius},{radius} 0 0 1 -{radius},{radius}h-{width-radius:.2f}z')


def benchmark_chart(groups):
    """Horizontal accuracy bars, one band per group of (role, label, metrics) rows."""
    left, right, plot_right, row, thickness = 300, 900, 840, 34, 22
    rows = [entry for _, entries in groups for entry in entries]
    top = max(math.ceil(max(m['accuracy'] for _, _, m in rows) * 10) / 10, .1)
    x = lambda value: left + value / top * (plot_right - left)
    summary = '; '.join(f'{label} {m["accuracy"]:.2%}' for _, label, m in rows)
    height = 40 + row * len(rows) + 30 * len(groups)
    parts = [f'<svg viewBox="0 0 {right} {height}" role="img" aria-label="External suite accuracy. {escape(summary, quote=True)}">']
    for step in range(int(round(top * 10)) + 1):
        value = step / 10
        parts.append(f'<line class="grid" x1="{x(value):.2f}" x2="{x(value):.2f}" y1="8" y2="{height-28}"/>'
                     f'<text class="tick" x="{x(value):.2f}" y="{height-8}" text-anchor="middle">{value:.0%}</text>')
    y = 8
    for title, entries in groups:
        y += 26
        parts.append(f'<text class="group" x="0" y="{y-6}">{escape(title)}</text>')
        for role, label, metrics in entries:
            width = x(metrics['accuracy']) - left
            tip = (f'{label}: {metrics["accuracy"]:.2%} ({metrics["correct"]:,} / {metrics["examples"]:,}); '
                   f'NLL {metrics["negative_log_likelihood"]:.4f}; Brier {metrics["multiclass_brier_score"]:.4f}')
            parts.append(f'<g class="row"><title>{escape(tip)}</title><rect class="hit" x="0" y="{y}" width="{right}" height="{row}"/>'
                         f'<text class="label" x="0" y="{y+row/2+5}">{escape(label)}</text>'
                         f'<path class="bar {role}" d="{bar(left, y + (row-thickness)/2, width, thickness)}"/>'
                         f'<text class="value" x="{left+width+8:.2f}" y="{y+row/2+5}">{metrics["accuracy"]:.2%}</text></g>')
            y += row
    parts.append('</svg>')
    return ''.join(parts)


def theme_css():
    return (''.join(f'--{key}:{value};' for key, value in PALETTE.items())
            + ''.join(f'--{role}:var(--{token});' for role, token in PRESENTATION.items()))


TOTAL_UPDATES = 8017
DASHBOARD_RUNS = [
    # key, name, what was trained, run directory, training output directory
    ('head-only', 'Answer layer only', 'Qwen frozen. Only a new answer layer was trained.', HEAD_RUN, 'head'),
    ('last-block', 'Last block + answer layer', 'Qwen’s final block and a new answer layer were trained.', LAST_BLOCK_RUN, 'last-block'),
    ('option-attention', 'Last block + attention head', 'Qwen’s final block and a new answer head that compares the options were trained.', ATTENTION_RUN, 'attention-last-block'),
    ('full', 'Whole model', 'Every Qwen weight and a new answer layer were trained.', 'runs/current-corpus-2026-10-01', None),
]


def dashboard_runs():
    """Development history, kept checkpoint and outside result for each run on the current corpus."""
    arms = matched_arms()
    full_history, full_report, *_ = load_training()
    runs = []
    for key, name, description, run, output in DASHBOARD_RUNS:
        progress = read(f'{run}/progress.json') if (ROOT / run / 'progress.json').exists() else {}
        if output is None:
            history, report = full_history, full_report
        else:
            if key == 'option-attention':
                output_directory = attention_training_output()
                history_path = (output_directory / 'history.json').relative_to(ROOT.resolve()).as_posix()
                report_path = (output_directory / 'report.json').relative_to(ROOT.resolve()).as_posix()
            else:
                history_path = f'{run}/{output}/history.json'
                report_path = f'{run}/{output}/report.json'
            if not (ROOT / history_path).exists():
                continue
            history = read(history_path)
            report = read(report_path) if (ROOT / report_path).exists() else None
        # Update 0 is the untrained head, so it is not a result of training.
        dev = [(point['update'], point['dev']) for point in history if 'dev' in point and point['update'] > 0]
        finished = bool(report and report['status'] == 'completed' and report['selected_checkpoint_reload_verified'])
        kept = (report['best_update'], report['selected_checkpoint_development']) if finished else None
        runs.append({'key': key, 'name': name, 'description': description, 'dev': dev,
                     'kept': kept, 'outside': arms.get(key, {}).get('overall'), 'progress': progress})
    return runs


def nice_range(values, step):
    return math.floor(min(values) / step) * step, math.ceil(max(values) / step) * step


def line_chart(runs, metric, percent, title):
    """Development metric over training, one line per run, on a shared update axis."""
    width, height, left, right, top, bottom = 440, 250, 48, 14, 14, 36
    series = [(run, [(update, dev[metric]) for update, dev in run['dev']]) for run in runs]
    values = [value for _, points in series for _, value in points]
    if not values:
        return ''
    step = .05 if percent else .1
    low, high = nice_range(values, step)
    while (high - low) / step > 6:
        step *= 2
        low, high = nice_range(values, step)
    x = lambda update: left + update / TOTAL_UPDATES * (width - left - right)
    y = lambda value: top + (high - value) / (high - low) * (height - top - bottom)
    fmt = (lambda value: f'{value:.0%}') if percent else (lambda value: f'{value:.1f}')
    parts = [f'<figure><figcaption>{escape(title)}</figcaption><svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(title)} by training update">']
    ticks = round((high - low) / step)
    for i in range(ticks + 1):
        value = low + step * i
        parts.append(f'<line class="grid" x1="{left}" x2="{width-right}" y1="{y(value):.1f}" y2="{y(value):.1f}"/>'
                     f'<text x="{left-8}" y="{y(value)+4:.1f}" text-anchor="end">{fmt(value)}</text>')
    for update in range(0, TOTAL_UPDATES, 2000):
        parts.append(f'<text x="{x(update):.1f}" y="{height-16}" text-anchor="middle">{update:,}</text>')
    parts.append(f'<text x="{(left+width-right)/2}" y="{height-1}" text-anchor="middle">Training update</text>')
    for run, points in series:
        if len(points) > 1:
            parts.append(f'<polyline class="trace s-{run["key"]}" points="{" ".join(f"{x(u):.1f},{y(v):.1f}" for u, v in points)}"/>')
        kept = run['kept'][0] if run['kept'] else None
        for update, value in points:
            label = f'{run["name"]} · update {update:,}: {value:.1%}' if percent else f'{run["name"]} · update {update:,}: {value:.3f}'
            label += ' · kept checkpoint' if update == kept else ''
            parts.append(f'<g><title>{escape(label)}</title><circle class="hit" cx="{x(update):.1f}" cy="{y(value):.1f}" r="9"/>'
                         f'<circle class="dot s-{run["key"]}" cx="{x(update):.1f}" cy="{y(value):.1f}" r="{6 if update == kept else 3.5}"/></g>')
    parts.append('</svg></figure>')
    return ''.join(parts)


def new_data_panel():
    """Keep the different dataset separate from the matched old-corpus comparison."""
    directory = ROOT / NEW_DATA_RUN
    if not directory.exists():
        return ''
    status = read(f'{NEW_DATA_RUN}/experiment-status.json') if (directory / 'experiment-status.json').exists() else {}
    phase = str(status.get('phase', status.get('stage', 'dataset and protocol preparation')))
    state = str(status.get('status', 'preparing'))
    observed = status.get('updated_at', status.get('completed_at', ''))
    fast_directory = directory / 'fast-evaluation-003'
    fast_active = (fast_directory / 'launch-attempt.json').exists()
    if fast_active:
        fast_status = read(f'{NEW_DATA_RUN}/fast-evaluation-003/status.json') if (fast_directory / 'status.json').exists() else {}
        terminal = read(f'{NEW_DATA_RUN}/fast-evaluation-003/driver-exit.json') if (fast_directory / 'driver-exit.json').exists() else None
        state = ('completed' if terminal.get('exit_code') == 0 else 'failed') if terminal else 'running'
        if (fast_directory / 'service-watch-incident.json').exists() and not terminal:
            state = 'failed'
        phase = str(fast_status.get('phase', 'bounded checkpoint selection, then full benchmark'))
        observed = fast_status.get('updated_at_utc', '')
        retry_pointer = fast_directory / 'retry-active.json'
        if retry_pointer.exists():
            retry = read(f'{NEW_DATA_RUN}/fast-evaluation-003/retry-active.json')
            retry_exit = (fast_directory / retry['driver_exit']).resolve()
            if not retry_exit.is_relative_to(fast_directory.resolve()):
                raise ValueError('Evaluation retry resolves outside its run directory')
            terminal = json.loads(retry_exit.read_text()) if retry_exit.exists() else None
            state = ('completed' if terminal.get('exit_code') == 0 else 'failed') if terminal else str(retry['status'])
            if state == 'prepared':
                phase = 'full benchmark handoff recovery; development selection complete'
    body = (f'<p><strong>New-data continuation</strong> · saved state: {escape(state)} · {escape(phase)}</p>'
            '<p>Continue the best last-block + linear-head model on reviewed documents, games, public training sources and old-data replay. '
            'This uses a different development set from the comparison above.</p>')
    if observed:
        body += f'<p class="muted">Status observed {escape(str(observed))}. Refresh after the next report rebuild.</p>'
    if status.get('message'):
        if not fast_active:
            body += f'<p>{escape(str(status["message"]))}</p>'
    if fast_active:
        body += ('<p>All 1,438 training updates are saved. The revised evaluation compares five checkpoints on the same '
                 '128 development questions, then runs the full 17,874-question external suite on the selected checkpoint. '
                 'Calibration uses 64 separate questions; the extra internal test is deferred. '
                 'Selection on this small development panel is exploratory.</p>')
        if (fast_directory / 'benchmark-report.json').exists():
            body += '<p>' + link(f'{NEW_DATA_RUN}/fast-evaluation-003/benchmark-report.html', 'Full benchmark comparison — new data versus previous checkpoint') + '</p>'
            body += '<p>' + link('runs/decision-model-comparison-2026-10-09.html', 'Comparison with other 4B decision models') + '</p>'
        selection_path = fast_directory / 'development-selection.json'
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            rows = []
            for update, result in sorted(selection['candidates'].items(), key=lambda item: int(item[0])):
                metrics = result['metrics']
                label = f'Update {update}' + (' (selected)' if int(update) == selection['selected_update'] else '')
                rows.append((label, f'{metrics["soft_cross_entropy"]:.4f}', f'{metrics["argmax_agreement"]:.1%}'))
            body += table(['Checkpoint', 'Development cross-entropy ↓', 'Top-choice agreement ↑'], rows,
                          'Same 128-question development panel; these are not external benchmark scores')
    if (directory / 'protocol.json').exists():
        protocol = read(f'{NEW_DATA_RUN}/protocol.json')
        body += (f'<p>{protocol["training_questions"]:,} training questions; {protocol["development_questions"]:,} development questions. '
                 f'Two {protocol["pilot"]["updates"]:,}-update learning-rate pilots, then one {protocol["main"]["updates"]:,}-update epoch. '
                 'Separate calibration and internal test follow checkpoint selection.</p>')
    if (directory / 'pilot-choice.json').exists():
        choice = read(f'{NEW_DATA_RUN}/pilot-choice.json')
        body += f'<p>Development selected head learning rate: {choice["selected_head_learning_rate"]:g}. Last-block rate: 1e-5.</p>'
    if (directory / 'full-context-smoke.json').exists():
        smoke = read(f'{NEW_DATA_RUN}/full-context-smoke.json')
        if smoke.get('backward_completed') and smoke.get('compact_checkpoint_reload_verified'):
            body += (f'<p>GPU check passed on a complete {smoke["complete_input_tokens"]:,}-token input, including backward computation '
                     f'and exact compact checkpoint restoration. Peak allocation: {smoke["peak_device_allocated_gib"]:.1f} GiB.</p>')
    if (directory / 'training-status.json').exists():
        progress = read(f'{NEW_DATA_RUN}/training-status.json')
        body += (f'<p>Saved training progress: {escape(str(progress.get("phase", progress.get("status", "unknown"))))}; '
                 f'{progress.get("global_update", 0):,} optimizer updates, {progress.get("questions_seen", 0):,} training questions. '
                 f'<span class="muted">Observed {escape(str(progress.get("updated_at", "")))}.</span></p>')
    main_directory = directory / 'main'
    if (directory / 'resolved-main.json').exists():
        resolved = read(f'{NEW_DATA_RUN}/resolved-main.json')
        main_directory = (ROOT / resolved['output']).resolve()
        if not main_directory.is_relative_to(directory.resolve()):
            raise ValueError('New-data output resolves outside its run directory')
    main_relative = main_directory.relative_to(ROOT).as_posix()
    if (main_directory / 'report.json').exists():
        report = read(f'{main_relative}/report.json')
        body += (f'<p>Main training: {escape(report["status"])}. Selected update {report["best_update"]:,}; '
                 f'development cross-entropy {report["best_dev_soft_cross_entropy"]:.4f}.</p>')
    test_relative = f'{main_relative}/calibration-evaluation/final-test.json'
    if (ROOT / test_relative).exists():
        result = read(test_relative)
        cohorts = result['cohort_metrics']
        comparison = []
        for key, label in [('fresh', 'Fresh internal test'), ('original_replay', 'Historical internal test')]:
            cohort = cohorts[key]
            before, after = cohort['uncalibrated'], cohort['calibrated']
            comparison.append((label, f'{cohort["questions"]:,}',
                               f'{before["soft_target_cross_entropy_nats"]:.4f} → {after["soft_target_cross_entropy_nats"]:.4f}',
                               f'{before["categorical_brier_sum"]:.4f} → {after["categorical_brier_sum"]:.4f}'))
        body += table(['Test cohort', 'Questions', 'Cross-entropy ↓', 'Brier ↓'], comparison,
                      'Before → after scalar temperature calibration; same selected checkpoint and test inference')
        body += '<p class="muted">The historical internal test has been reused. Fresh internal questions are separate from calibration and development.</p>'
    if (directory / 'benchmark-report.json').exists():
        external = read(f'{NEW_DATA_RUN}/benchmark-report.json')
        aggregate_path = external['aggregate_path']
        if not (ROOT / aggregate_path).resolve().is_relative_to((ROOT / BENCHMARK_ROOT).resolve()):
            raise ValueError('New-data benchmark aggregate resolves outside evaluation results')
        aggregate = read(aggregate_path)
        if SOURCES[aggregate_path] != external['aggregate_sha256'] or aggregate['candidate_results']['last-block'] != external['metrics']:
            raise ValueError('New-data benchmark aggregate differs from its report binding')
        overall = external['metrics']['overall']
        body += (f'<p>Exploratory external accuracy: <strong>{overall["accuracy"]:.2%}</strong> '
                 f'on {overall["examples"]:,} items. Prior last-block model: 49.14%. '
                 'The suite has been reused; the new run also changes the training data and schedule.</p>')
    links = [(f'{NEW_DATA_RUN}/protocol.json', 'Frozen plan'), (f'{NEW_DATA_RUN}/pilot-choice.json', 'Pilot comparison'),
             (f'{main_relative}/curves.html', 'New-data learning curves'), (test_relative, 'Internal test and calibration'),
             (f'{NEW_DATA_RUN}/benchmark-report.json', 'Exploratory external evaluation'), (f'{NEW_DATA_RUN}/experiment-status.json', 'Saved execution status'),
             (f'{NEW_DATA_RUN}/training-status.json', 'Latest training progress')]
    body += '<p>' + ' · '.join(link(path, title) for path, title in links if (ROOT / path).exists()) + '</p>'
    return f'<section class="surface">{body}</section>'


def build_dashboard():
    """One screen: what each run is, how they compare on loss and accuracy, and what is running."""
    original = read(BENCHMARKS['original'])['overall']
    runs = dashboard_runs()
    for run in runs:
        run['shown'] = run['kept'][1] if run['kept'] else (run['dev'][-1][1] if run['dev'] else None)

    def bar_cell(value, role):
        if value is None:
            return '<td class="muted">—</td>'
        return (f'<td><span class="meter"><span class="fill s-{role}" style="width:{value:.1%}"></span></span>'
                f'<span class="num">{value:.1%}</span></td>')

    def number_cell(value, best):
        if value is None:
            return '<td class="muted">—</td>'
        strong = ' class="best"' if best is not None and math.isclose(value, best) else ''
        return f'<td><span{strong}>{value:.3f}</span></td>'

    best_dev_loss = min((run['shown']['soft_cross_entropy'] for run in runs if run['kept']), default=None)
    best_outside_loss = min((run['outside']['negative_log_likelihood'] for run in runs if run['outside']), default=None)
    rows = ''
    for run in runs:
        dev, outside = run['shown'], run['outside']
        if run['kept']:
            dev_note = f'Kept checkpoint: update {run["kept"][0]:,}.'
        elif run['dev']:
            dev_note = f'Training · latest check at update {run["dev"][-1][0]:,} of {TOTAL_UPDATES:,}.'
        else:
            dev_note = 'Training · first development check after update 512.'
        rows += (f'<tr><th scope="row"><span class="key s-{run["key"]}"></span><strong>{escape(run["name"])}</strong>'
                 f'<small>{escape(run["description"])} {escape(dev_note)}</small></th>'
                 + bar_cell(dev and dev['hard_accuracy'], run['key'])
                 + number_cell(dev and dev['soft_cross_entropy'], best_dev_loss)
                 + bar_cell(outside and outside['accuracy'], run['key'])
                 + number_cell(outside and outside['negative_log_likelihood'], best_outside_loss) + '</tr>')
    rows += ('<tr><th scope="row"><span class="key s-original"></span><strong>Original Qwen</strong>'
             '<small>Untouched. Answers through its own text output, not a trained answer layer, so its loss is not comparable.</small></th>'
             '<td class="muted">—</td><td class="muted">—</td>' + bar_cell(original['accuracy'], 'original')
             + f'<td class="muted">{original["negative_log_likelihood"]:.3f}</td></tr>')

    scored = [run for run in runs if run['kept'] and run['outside']]
    if len(scored) > 1:
        dev_best = max(scored, key=lambda run: run['kept'][1]['hard_accuracy'])
        out_best = max(scored, key=lambda run: run['outside']['accuracy'])
        out_worst = min(scored, key=lambda run: run['outside']['accuracy'])
        finding = (f'<strong>{escape(dev_best["name"])}</strong> learned our own questions best '
                   f'({dev_best["kept"][1]["hard_accuracy"]:.1%} correct)')
        if dev_best is out_worst:
            finding += f', but did worst on outside questions ({dev_best["outside"]["accuracy"]:.1%}).'
        else:
            finding += f' and scored {dev_best["outside"]["accuracy"]:.1%} on outside questions.'
        if out_best is not dev_best:
            finding += (f' <strong>{escape(out_best["name"])}</strong> did best outside '
                        f'({out_best["outside"]["accuracy"]:.1%}).')
        finding += f' The untouched original still scores highest outside ({original["accuracy"]:.1%}).'
    else:
        finding = f'The untouched original scores {original["accuracy"]:.1%} on outside questions.'

    training = []
    for run in runs:
        progress = run['progress']
        if not run['kept'] and progress.get('status') == 'running':
            text = f'{run["name"]} is training: update {progress.get("global_update", 0):,} of {TOTAL_UPDATES:,}'
            if progress.get('estimated_training_completion_at'):
                text += f', estimated to finish {progress["estimated_training_completion_at"][:16]}Z'
            training.append(text + '.')
    running = ' '.join(training) if training else 'The old-corpus experiments are complete.'
    charts = line_chart(runs, 'soft_cross_entropy', False, 'Development loss · lower is better') + \
        line_chart(runs, 'hard_accuracy', True, 'Development accuracy · higher is better')
    style = (
        '*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.55 system-ui,sans-serif}'
        'main{max-width:1000px;margin:auto;padding:40px 24px}p{margin:12px 0}.muted,small{color:var(--muted)}a{color:var(--blue)}'
        '.finding{font-size:21px;line-height:1.45;margin:8px 0 26px;max-width:880px}'
        '.surface{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:22px 24px;margin:20px 0}'
        '.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}'
        'th,td{text-align:left;vertical-align:top;padding:12px 10px;border-bottom:1px solid var(--line)}tbody tr:last-child>*{border-bottom:0}'
        'thead th{font-size:13px;font-weight:600;color:var(--muted);padding-top:4px}thead tr:first-child th{padding-bottom:0;color:var(--ink)}'
        'tbody th{font-weight:400;min-width:260px}tbody th strong{font-weight:600}tbody th small{display:block;font-size:13px;line-height:1.4;margin-top:2px}'
        '.group{border-left:1px solid var(--line)}td:nth-child(2),td:nth-child(4){border-left:1px solid var(--line)}'
        '.key{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px;background:var(--series)}'
        '.meter{display:block;width:110px;height:8px;background:var(--canvas);border-radius:4px;margin:8px 0 4px}'
        '.fill{display:block;height:100%;border-radius:0 4px 4px 0;background:var(--series)}.num{font-weight:600}.best{font-weight:700}'
        '.charts{display:grid;grid-template-columns:1fr 1fr;gap:24px}figure{margin:0}figcaption{font-weight:600;margin-bottom:6px}'
        'svg{width:100%;height:auto;display:block}svg text{font:11px system-ui;fill:var(--muted)}.grid{stroke:var(--line);stroke-width:1}'
        '.trace{fill:none;stroke:var(--series);stroke-width:2;stroke-linejoin:round;stroke-linecap:round}'
        '.dot{fill:var(--series);stroke:var(--paper);stroke-width:2}.hit{fill:transparent}'
        + ''.join(f'.s-{key}{{--series:var(--{key})}}' for key in ('head-only', 'last-block', 'option-attention', 'full', 'original'))
        + '@media(max-width:700px){main{padding:24px 14px}.charts{grid-template-columns:1fr}tbody th{min-width:170px}tbody th small{display:none}.meter{width:64px}th,td{padding:10px 6px}.finding{font-size:18px}.surface{padding:16px}}'
    )
    observed = datetime.now(timezone.utc).isoformat(timespec='seconds')
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Horscht · dashboard</title><style>:root{{{theme_css()}}}{style}</style></head><body><main>
<p class="muted">Horscht · {observed[:10]}</p>
<p class="finding">{finding}</p>
{new_data_panel()}
<section class="surface"><div class="scroll"><table>
<thead><tr><th></th><th colspan="2" class="group">Our development questions</th><th colspan="2" class="group">Outside questions</th></tr>
<tr><th scope="col">Run</th><th scope="col" class="group">Accuracy ↑</th><th scope="col">Loss ↓</th><th scope="col" class="group">Accuracy ↑</th><th scope="col">Loss ↓</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p class="muted">All runs trained on the same 128,263 questions with the same settings, for one pass. Development: 7,202 held-back questions from our own data (accuracy uses the 2,740 with one right answer). Outside: 17,874 multiple-choice questions from MMLU-Pro, Belebele and BBH, never trained on. Loss is cross-entropy; compare it down a column, not across, because the two question sets differ. We have looked at the outside questions before, so treat them as a guide, not proof.</p></section>
<section class="surface"><div class="charts">{charts}</div>
<p class="muted">Each dot is one development check, every 512 updates. The large dot is the checkpoint we kept (lowest development loss); that checkpoint is the one scored on outside questions.</p></section>
<p>{escape(running)} Still open: is the original ahead because of what it knows, or because of how it answers?</p>
<p><a href="{RECORD.name}">Full record: every number, check, decision and earlier experiment →</a></p>
<p class="muted">Generated {observed}.</p>
</main></body></html>"""


def build():
    head_state, head_meaning, head_gpu, head_body = current_head_state()
    last_state, last_meaning, last_gpu, last_body = current_last_block_state()
    attention_state, attention_meaning, attention_gpu, attention_body = current_attention_state()
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
    arms = matched_arms()
    if arms and arms['full'] != current:
        raise ValueError('Matched FULL metrics differ from the current FULL aggregate')
    matched_labels = [('head-only', 'Head only · backbone frozen'),
                      ('last-block', 'Last block + head'),
                      ('option-attention', 'Option attention + final block'),
                      ('full', 'Full backbone + head')]
    matched = [(label, arms[key]) for key, label in matched_labels if key in arms]
    if not any(key == 'full' for key, _ in matched_labels if key in arms):
        matched.append(('Full backbone + head', current))
    references = [('Original Qwen · native LM head', original), ('Earlier full tune', earlier),
                  ('Earlier frozen head', frozen)]
    groups = [('Same current corpus · trainable scope grows downward', [('matched', l, m['overall']) for l, m in matched]),
              ('Reference systems · different data or readout', [('reference', l, m['overall']) for l, m in references])]
    benchmark_rows = [(label, f'{m["overall"]["correct"]:,} / {m["overall"]["examples"]:,}',
                       f'{m["overall"]["accuracy"]:.2%}', f'{m["overall"]["negative_log_likelihood"]:.4f}',
                       f'{m["overall"]["multiclass_brier_score"]:.4f}') for label, m in matched + references]
    slice_systems = [m for _, m in matched] + [original, earlier]
    slices = []
    for key, metrics in current['by_source_slice'].items():
        label = key.replace('mmlu-pro-test', 'MMLU-Pro').replace('belebele-', 'Belebele · ').replace('bbh-', 'BBH · ').replace('-', ' ')
        slices.append((escape(label), f'{metrics["examples"]:,}',
                       *[f'{m["by_source_slice"][key]["accuracy"]:.2%}' for m in slice_systems]))
    best_arm = max(matched, key=lambda item: item[1]['overall']['accuracy'])
    if arms:
        scores = ', '.join(f'{label.lower()} {metrics["overall"]["accuracy"]:.2%}' for label, metrics in matched)
        lead = (f'Matched runs on the same frozen corpus scored {scores} on the reused external suite. '
                f'Original Qwen scored {original["overall"]["accuracy"]:.2%} with a different prompt and readout. '
                f'The suite is reused, so this is exploratory evidence.')
    else:
        lead = (f'The current full tune scored {current["overall"]["accuracy"]:.2%} on the reused external suite, below original Qwen '
                f'({original["overall"]["accuracy"]:.2%}) and the earlier full tune ({earlier["overall"]["accuracy"]:.2%}).')
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
<header><p class="muted"><a href="project-record-2026-10-04.html">← Dashboard</a> · Horscht · full project record · 2026-10-04</p><h1>What we built, what we measured, and what is still open.</h1>
<p class="lead">{escape(lead)}</p>
<p class="muted">Snapshot generated {observed}. This is a saved record, not a live activity dashboard. Discussion summaries consolidate the saved project evidence and the coordinating conversation; they are not a verbatim transcript.</p></header>
<nav aria-label="Record contents"><a href="#status">Status</a><a href="#diagnostic">Attribution</a><a href="#corpus">Training data</a><a href="#discussion">Discussion and decisions</a><a href="#history">Earlier work</a><a href="#sources">Sources</a></nav>
<section class="surface" id="results"><p><strong>External suite accuracy · 17,874 items</strong></p>{benchmark_chart(groups)}
{table(['System', 'Correct / items', 'Accuracy ↑', 'NLL ↓', 'Brier ↓'], benchmark_rows, 'Same reused external suite')}
<p>Best matched arm: {escape(best_arm[0].lower())}. Each matched checkpoint was frozen by internal development before external scoring. Original Qwen exceeds current FULL by {100*(original['overall']['accuracy']-current['overall']['accuracy']):.2f} percentage points. Earlier FULL exceeds current FULL by {100*(earlier['overall']['accuracy']-current['overall']['accuracy']):.2f} points. These compare different training runs and readouts; they do not show monotonic benchmark decline across checkpoints of one run.</p>
<p>Original Qwen chooses more correct answers but has worse probability losses. Its native language-model label probabilities are often extremely confident. The current decision-head distribution has lower NLL and Brier here. Neither accuracy alone nor these scores establish calibrated confidence.</p>
<p class="muted">All of these comparisons are exploratory evidence from an observed suite. MMLU-Pro contributes 11,874 of 17,874 items; five Belebele languages contribute 4,500, and six BBH slices contribute 1,500. The mixed average is weighted by item count. Belebele translations are related semantic problems.</p>
{detail('Slice results, prompt differences, and evaluation boundary', table(['Slice','Items',*[label for label, _ in matched],'Original Qwen','Earlier FULL'], slices, 'Aggregate accuracy by source slice') + '''
<p>MMLU-Pro moved from 37.52% in earlier FULL to 28.37% in current FULL; original Qwen scored 39.95%. All five Belebele language slices improved from earlier FULL to current FULL. BBH changes were mixed. A single weighted score hides these differences.</p>
<p>The two trained decision systems use the same external inference protocol but different training snapshots and checkpoints. Original Qwen uses native no-thinking chat, uppercase choice labels and restricted original LM-head label likelihood. The decision systems use plain numeric prompts and their learned decision heads. This is an end-to-end comparison; prompt, labels and readout confound any explanation in terms of backbone forgetting. It is not a generated chain-of-thought comparison.</p>
<p>The selected current checkpoint was fixed from development before evaluation. Exact grouping, deduplication and overlap checks do not establish freedom from semantic overlap or pretrained contamination. The suite has been reused and discussed, so further adaptive work must not call it an untouched new test. Benchmark items, answers and per-item errors are absent from this report.</p>''' + f'<p>Evaluation times: earlier paired {escape(aggregates["paired"]["evaluated_at_utc"])}; current {escape(aggregates["current"]["evaluated_at_utc"])}; original {escape(original["evaluated_at_utc"])}.</p>')}
{detail('Choice count and input length: available evidence and pending work', diagnostics + '''
<p>These strata exist only for the original-model method. Comparable trained-model strata have not been run. Choice cardinality, source family, difficulty and prompt length vary together; the tables do not identify a causal choice-count or length penalty. The longest stratum contains only 14 items. Comparing a long original chat prompt with a shorter decision prompt also requires a declared length definition.</p>
<p>Proposed follow-up: compute the same fixed cardinality/length strata for each system and inspect within source family. Treat this as exploratory descriptive analysis, keep content hidden, and do not use it to select a prompt or checkpoint on the external test.</p>''')}
</section>
<section class="surface" id="status"><p><strong>Current state</strong></p>
{table(['Work', 'State at this record', 'Meaning'], [
('Current full-backbone training', '<span class="completed">Completed</span>', 'One epoch; 8,017 updates. Selected checkpoint: 4,608.'),
('External evaluation and original comparison', '<span class="completed">Completed</span>', '17,874 reused items; aggregate-only reporting.'),
('Synthetic attribution diagnostic', '<span class="completed">Completed</span>', 'Eight probes, four systems; about 97 seconds.'),
('New full-dataset frozen-backbone experiment', escape(head_state), escape(head_meaning)),
('Matched last-block experiment', escape(last_state), escape(last_meaning)),
('Final option-attention + last-block arm', escape(attention_state), escape(attention_meaning)),
('Choice-count and length comparison', 'Partial · not running', 'Original-model strata available; matched trained-model diagnostics pending.'),
('New game and document batches', 'Generated · not trained', 'Separate future-data artifacts; absent from the completed frozen run.'),
('GPU / background owners', escape(attention_gpu if attention_gpu.startswith('GPU owned') else last_gpu or head_gpu), 'Durable training and completion jobs publish saved status.')], 'Work and status')}
<p>The earlier frozen-head control already exists. It used an earlier dataset and is <strong>not</strong> the newly discussed head-only experiment on the full current corpus.</p>
{detail('Frozen-backbone run: evidence and receipts', head_body)}{detail('Last-block run: evidence and receipts', last_body)}{detail('Option-attention run: evidence and receipts', attention_body)}</section>
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
('Full-corpus last block','Original pinned backbone; final block plus fresh head train; embeddings, preceding blocks and final norm frozen.',escape(last_state)),
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
{detail('2026-10-05 · matched final-block arm', f'''
<p>The user authorized the final transformer block plus a fresh 265-way head on the original pinned Qwen3-4B. The embedding, preceding 35 blocks and final backbone normalization remain frozen. The arm uses the same immutable dataset, seed, fresh head initialization, deterministic epoch plan and one-epoch exposure as current FULL and current head-only. The last-block learning rate is the previously used full-backbone rate, 0.00001; no hyperparameter search or new data was introduced.</p>
<p>Saved state at this record: {escape(last_state)}. The durable completion job verifies the selected development checkpoint, exact matched exposure and unchanged frozen weights before serial evaluation on the same 17,874-item suite. That evaluation is exploratory reuse, with no external checkpoint selection or tuning.</p>''')}
{detail('2026-10-06 · final old-corpus architecture test', f'''
<p>The option-attention arm pairs a 256-wide option-conditioned attention head with Qwen’s final transformer block (block 35). It keeps the embedding, preceding 35 blocks and final backbone normalization frozen. The authorized comparison uses the same frozen training corpus, prompt, loss and one-epoch budget as the matched arms.</p>
<p>Saved state at this record: {escape(attention_state)}. This is the last planned external test for the old training corpus. After its aggregate-only evaluation, further training and external comparison wait for newly admitted game and document data. The observed suite has been reused and remains exploratory.</p>''')}
</section>
<section class="surface"><p><strong>What remains unresolved</strong></p>
<p>We do not yet know how much of the external gap comes from head/readout alignment, prompts, training distribution, choice cardinality, model adaptation, forgetting, or the differing old/current cohorts. We also do not know whether newly generated games/documents improve non-game transfer, long-context integration, or multilingual reliability: they have not been trained or measured in this run.</p>
<p>The matched fresh-head, final-block and option-attention arms provide controlled evidence with the same current corpus. Further training and external evaluation on this old corpus stop after the final architecture test. Internal held-out evaluation under a declared boundary and a separately designed frozen system test remain future evidence. Keep the observed external suite exploratory, preserve raw artifacts, and change one major experimental difference at a time. This follows the Pragmatic Programmer principle of deliberate experiments and avoiding accidental complexity.</p></section>
'''
    source_rows = [(link(path,path),f'<code>{digest}</code>') for path,digest in sorted(SOURCES.items()) if not path.endswith('rows.jsonl')]
    source_section = '<section class="surface" id="sources"><p><strong>Evidence and reproducibility</strong></p><p>The builder reads only explicitly named external aggregate files. All core tables and curves are embedded; no CDN, external script or network dependency is needed to read this record. Linked detailed reports and source files are optional drill-downs and require the repository to remain beside this file.</p>'
    source_section += detail('Exact source files and SHA-256 bindings', table(['Source artifact','SHA-256'],source_rows,'Input bindings at record generation') + f'<p>Frozen corpus rows SHA-256: <code>{SOURCES["data/training_snapshots/2026-10-01-v1/rows.jsonl"]}</code>. Read internally for aggregate train accounting; corpus text is not embedded.</p><p>The current training history merge/selection/reload validation and the benchmark count/weighted-metric/suite-binding validations run before rendering. Static conversation summaries distinguish observations, interpretation, proposals and implementation; they do not authorize jobs.</p>')
    source_section += f'<p>{link("build_project_record.py","Rebuild script")} · {link(".cursor/rules/blind-benchmarks.mdc","Blind benchmark rule")} · {link("AGENTS.md","Project coordination instructions")}</p></section>'
    theme = theme_css()
    style = '''
*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);font:16px/1.6 system-ui,sans-serif}main{max-width:1120px;margin:auto;padding:36px 26px}h1{font-size:34px;line-height:1.2;letter-spacing:-.025em;max-width:880px;margin:12px 0 20px}p{margin:12px 0}.lead{font-size:19px;max-width:970px}.muted,small{color:var(--muted)}nav{display:flex;gap:10px 24px;flex-wrap:wrap;margin:26px 0}a{color:var(--blue);text-underline-offset:3px}a:focus-visible,summary:focus-visible,button:focus-visible{outline:3px solid var(--blue);outline-offset:4px}.surface{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:24px;margin:22px 0}.surface>p:first-child{font-size:19px;margin-top:0}details{border-top:1px solid var(--line);padding:16px 0}summary{cursor:pointer;font-weight:600;padding:2px 0}.detail-body{padding:8px 0 0}details:last-child{padding-bottom:0}.scroll{overflow-x:auto;margin:20px 0}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums;font-size:14px}caption{text-align:left;color:var(--muted);padding-bottom:8px;font-size:13px}td,th{text-align:left;border-bottom:1px solid var(--line);padding:12px 10px;vertical-align:top}thead th{color:var(--muted);font-size:13px}tbody th{font-weight:500}td{min-width:76px}code{font-size:12px;overflow-wrap:anywhere}.completed{color:var(--completed);font-weight:650}.proposed{color:var(--proposed);font-weight:650}svg{width:100%;height:auto;display:block;margin:18px 0}svg text{font:15px system-ui;fill:var(--muted)}svg .label,svg .value{fill:var(--ink)}svg .tick{font-size:13px}svg .group{font-size:13px;font-weight:600}.hit{fill:transparent}.row:hover .hit{fill:var(--canvas)}.bar.matched{fill:var(--matched)}.bar.reference{fill:var(--reference)}.grid{stroke:var(--line);stroke-width:1}.trace{fill:none;stroke-width:2.5;stroke:var(--development)}.point{fill:var(--development)}.marker{stroke-width:1.5;stroke-dasharray:5 4}.marker.selected{stroke:var(--selected)}.marker.recovery{stroke:var(--recovery)}button{font:inherit;border:1px solid var(--line);background:var(--paper);color:var(--ink);border-radius:6px;padding:7px 12px;cursor:pointer}.toolbar{display:flex;gap:10px;margin-top:20px}footer{color:var(--muted);font-size:13px;margin:24px 0}@media(max-width:650px){main{padding:20px 14px}h1{font-size:28px}.lead{font-size:17px}.surface{padding:17px}td,th{padding:10px 8px}table{min-width:530px}nav{gap:8px 18px}svg text{font-size:16px}}@media print{body{background:var(--paper)}main{max-width:none;padding:0}.surface{border-radius:0;break-inside:avoid}.toolbar,nav{display:none}summary{font-size:18px}a{color:var(--ink)}.scroll{overflow:visible}table{font-size:11px}footer{font-size:10px}}
'''
    return f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Horscht · consolidated project record · 2026-10-04</title><style>:root{{{theme}}}{style}</style></head><body><main>{overview}{extra}{source_section}<div class="toolbar" hidden id="controls"><button id="expand">Expand all details</button><button id="collapse">Collapse all details</button><button id="print">Print / save PDF</button></div><footer>Saved project record · {observed} · All core results embedded; historical artifacts preserved.</footer></main><script>document.getElementById('controls').hidden=false;document.getElementById('expand').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=true);document.getElementById('collapse').onclick=()=>document.querySelectorAll('details').forEach(d=>d.open=false);document.getElementById('print').onclick=()=>{{document.querySelectorAll('details').forEach(d=>d.open=true);window.print()}};</script></body></html>'''


def write(path, content):
    temporary = path.with_suffix('.html.tmp')
    temporary.write_text(content)
    temporary.replace(path)
    print(path)


def main():
    if (ROOT / NEW_DATA_RUN / 'fast-evaluation-003/benchmark-report.json').exists():
        from build_benchmark_report import build as build_benchmark
        build_benchmark()
        from build_peer_comparison_report import build as build_peers
        build_peers()
    write(RECORD, build().replace('<section class="surface" id="status">', new_data_panel() + '<section class="surface" id="status">', 1))
    write(DASHBOARD, build_dashboard())


if __name__ == '__main__':
    main()
