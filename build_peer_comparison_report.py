"""Build a dated, source-linked peer review from aggregate evidence only."""
from html import escape
from pathlib import Path
from build_benchmark_report import RESULTS, RUN, grouped, read
from report_theme import PALETTE

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'runs/decision-model-comparison-2026-10-09.html'
PRESENTATION = {'background': 'canvas', 'surface': 'paper', 'text': 'ink', 'muted': 'muted',
                'border': 'line', 'local': 'blue', 'published': 'teal', 'caution': 'orange'}
SOURCES = {
    'kev': ('Kev-4B model card', 'https://huggingface.co/jaredpalmer/kev-4b'),
    'mica': ('Mica model card and evaluation', 'https://huggingface.co/sky7350/Mica-v0.1-4B'),
    'ima': ('ImaJev-4B model card', 'https://huggingface.co/mohit67890/imajev-4b'),
    'inner': ('InnerJev-4B-Full model card', 'https://huggingface.co/jylin001206/InnerJev-4B-Full'),
    'llm': ('LLM2Jev research paper', 'https://arxiv.org/abs/2610.02076'),
    'clef': ('Clef model card and Decision Index results', 'https://huggingface.co/Cloudflare/clef'),
    'flash': ('Clef-Flash model card', 'https://huggingface.co/Cloudflare/clef-flash'),
}


def cite(key):
    title, url = SOURCES[key]
    return f'<a href="{escape(url)}">{escape(title)}</a>'


def build():
    current = read(RESULTS / 'new-data-last-block-fast-2026-10-09/blind-aggregate-result.json')['candidate_results']['last-block']
    original = read(RESULTS / 'original-qwen-label-likelihood-2026-10-04/blind-aggregate-result.json')
    mmlu = grouped(current, 'mmlu-')
    rows = ''.join(f'<tr><th>{name}</th><td>{grouped(current, prefix)["accuracy"]:.2%}</td><td>{grouped(original, prefix)["accuracy"]:.2%}</td><td>{grouped(current, prefix)["examples"]:,}</td></tr>'
                   for name, prefix in [('MMLU-Pro', 'mmlu-'), ('Belebele · five languages', 'belebele-'), ('BBH · six tasks', 'bbh-')])
    benchmark_matrix = ''.join(
        '<tr><th>' + label + '</th>' + ''.join(f'<td>{grouped(result, prefix)["accuracy"]:.2%}</td>' for prefix in ('mmlu-', 'belebele-', 'bbh-')) + '<td>Local matched suite</td></tr>'
        for label, result in [('Horscht · current', current), ('Original Qwen3-4B · immediate answer', original)])
    tokens = ';'.join(f'--{role}:{PALETTE[color]}' for role, color in PRESENTATION.items())
    html = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Horscht versus 4B decision models</title><style>
:root{{{tokens};font-family:system-ui,-apple-system,sans-serif;color:var(--text);background:var(--background)}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1120px;margin:auto;padding:30px 28px 64px}}a{{color:var(--local);text-underline-offset:3px}}
nav{{display:flex;justify-content:space-between;gap:16px;font-size:14px;color:var(--muted);margin-bottom:40px}}h1{{font-size:clamp(30px,4vw,46px);line-height:1.12;letter-spacing:-.035em;max-width:870px;margin:12px 0 20px}}h2{{font-size:23px;letter-spacing:-.02em;margin:0 0 14px}}p,li{{line-height:1.65}}.intro{{max-width:880px;font-size:18px}}.muted,small{{color:var(--muted)}}.kicker{{font-size:13px;color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin:28px 0}}.card,section,details{{border:1px solid var(--border);background:var(--surface);border-radius:16px}}.card{{padding:22px}}.number{{display:block;font-weight:750;font-size:38px;letter-spacing:-.04em;margin:8px 0;color:var(--local)}}.card p{{font-size:13px;color:var(--muted);margin:0}}section{{padding:26px;margin:20px 0}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{text-align:left;vertical-align:top;border-bottom:1px solid var(--border);padding:16px 12px;line-height:1.5}}thead th{{font-size:12px;color:var(--muted)}}tbody tr:last-child>*{{border-bottom:0}}th:first-child,td:first-child{{padding-left:0}}.numeric td{{white-space:nowrap}}small{{display:block;font-size:12px;margin-top:6px;font-weight:400}}.note{{font-size:13px;color:var(--muted)}}.callout{{border-left:3px solid var(--caution);padding:0 18px;margin:24px 0}}.tag{{font-size:12px;color:var(--published);font-weight:650}}details{{padding:0 24px;margin:14px 0}}summary{{padding:21px 0;font-weight:650;cursor:pointer}}details p,details li{{font-size:14px}}footer{{font-size:12px;color:var(--muted);line-height:1.7;margin-top:28px}}li{{margin:10px 0}}
@media(max-width:700px){{main{{padding:24px 16px}}nav{{flex-wrap:wrap}}.cards{{grid-template-columns:1fr}}section{{padding:18px}}.peers{{min-width:820px}}.numeric{{min-width:550px}}.card{{padding:16px}}}}
</style><main><nav><a href="/runs/new-data-last-block-2026-10-08/fast-evaluation-003/benchmark-report.html">← Our benchmark results</a><span>Research snapshot · 2026-10-09</span></nav>
<div class="kicker">Horscht · Qwen3-4B · selected update 1080</div>
<h1>The 4B field sets a higher target.</h1>
<p class="intro">Our latest model improves on our previous checkpoint, but published results suggest there is substantial room left at this size. <strong>Mica and Kev are the most useful immediate comparisons.</strong> Their higher reported scores are a reason to test them in our harness, not a measured head-to-head defeat.</p>
<div class="cards"><div class="card">Our MMLU-Pro<span class="number">{mmlu['accuracy']:.2%}</span><p>{mmlu['examples']:,} questions · measured locally</p></div><div class="card">Mica’s published MMLU-Pro<span class="number">53.0%</span><p>10,032 questions · different harness · {cite('mica')}</p></div><div class="card">Peer group<span class="number">4B class</span><p>Four trained peers below; Clef’s 9B and 27B models are separate references.</p></div></div>
<section><h2>What the numbers actually establish</h2><div class="scroll"><table class="numeric"><thead><tr><th>Model / evaluation</th><th>MMLU-Pro</th><th>Evidence and scope</th></tr></thead><tbody>
<tr><th>Horscht · current</th><td><strong>{mmlu['accuracy']:.2%}</strong></td><td>Local full test: 11,874 questions; decision head, no generated reasoning.</td></tr>
<tr><th>Original Qwen3-4B</th><td>{grouped(original, 'mmlu-')['accuracy']:.2%}</td><td>Same local questions; original head with immediate-answer label likelihood.</td></tr>
<tr><th>Mica v0.1 4B</th><td><strong>53.0%</strong></td><td>Author’s 10,032-question evaluation; BF16 PyTorch. {cite('mica')}</td></tr>
<tr><th>Kev-4B · in Mica’s evaluation</th><td>49.7%</td><td>Same published panel as Mica; not necessarily the latest Kev release. {cite('mica')}</td></tr>
<tr><th>Kev-4B · own model card</th><td>56.5%</td><td>Transfer-v9 development panel; different subset and release context. {cite('kev')}</td></tr>
</tbody></table></div><p class="note">These are accuracy percentages, not one shared leaderboard. Mica’s panel also reports stock Qwen3.5-4B at 45.2%, versus Mica at 53.0% within that author’s evaluation. That supports the possibility of a useful decision-specific adaptation; it does not isolate which training change caused the gain. {cite('mica')}</p></section>
<div class="callout"><p><strong>Backbone choice:</strong> our original 64-question training comparison favored Qwen3 post-trained (88.125% mean development accuracy) over Qwen3.5 (84.6875%). Those were small, template-bound tests. We are retaining Qwen3-4B for this release; the peers’ newer backbone does not establish that switching would improve our model.</p></div>
<div class="callout"><p><strong>My reading:</strong> our linear head is not, by itself, the obvious mistake. ImaJev also uses a linear decision readout. The stronger questions are whether our older backbone and training targets preserve enough useful reasoning, and whether the data teaches the actual decisions we care about. This is a hypothesis, not a causal result from these cross-project scores.</p></div>
<section><h2>All three benchmarks · peer coverage</h2><div class="scroll"><table class="numeric"><thead><tr><th>Model</th><th>MMLU-Pro</th><th>Belebele</th><th>BBH</th><th>Evidence</th></tr></thead><tbody>
{benchmark_matrix}
<tr><th>Mica v0.1 4B</th><td>53.0%</td><td>Not verified</td><td>Not verified</td><td>{cite('mica')}</td></tr>
<tr><th>Kev-4B · own card</th><td>56.5%</td><td>Not verified</td><td>Not verified</td><td>{cite('kev')}</td></tr>
<tr><th>ImaJev-4B</th><td>Not verified</td><td>Not verified</td><td>Not verified</td><td>{cite('ima')}</td></tr>
<tr><th>InnerJev-4B-Full</th><td>Not verified</td><td>Not verified</td><td>Not verified</td><td>{cite('inner')}</td></tr>
</tbody></table></div><p class="note">“Not verified” means no aggregate for that benchmark was verified in the reviewed primary sources; it does not mean zero or prove the model was never evaluated. Our Belebele score covers German, English, French, Spanish and Chinese. Our BBH score covers logical deduction and shuffled-object tracking at three sizes. Cross-model BBH comparisons require the same task mix.</p></section>
<section><h2>BBH · larger reference models</h2><div class="scroll"><table class="numeric"><thead><tr><th>Model</th><th>Size</th><th>Published BBH</th><th>Evaluation</th></tr></thead><tbody>
<tr><th>Clef-Flash</th><td>9B</td><td>68.9%</td><td>Cloudflare Decision Index</td></tr>
<tr><th>Clef</th><td>27B</td><td>73.7%</td><td>Cloudflare Decision Index</td></tr>
<tr><th>Jev</th><td>Not verified</td><td>92.9%</td><td>Cloudflare Decision Index</td></tr>
</tbody></table></div><p class="note">Published aggregate results from {cite('clef')}. These are not verified to use our six-task BBH subset, so subtracting our 38.53% would not give a matched performance gap. No Belebele result was found in this model card.</p></section>
<section><h2>Four relevant models around our size</h2><div class="scroll"><table class="peers"><thead><tr><th>Model</th><th>Backbone and adaptation</th><th>Interface / limits</th><th>Why it matters here</th></tr></thead><tbody>
<tr><th>Horscht<small>Our checkpoint</small></th><td>Qwen3-4B. Last transformer block + linear 265-output head; earlier backbone frozen.</td><td>Text; changing options; soft-target probabilities. 32,768-token benchmark cap.</td><td>Small adaptation surface. Our measured baseline; current image capability is not demonstrated.</td></tr>
<tr><th>Kev-4B<small>{cite('kev')}</small></th><td>Qwen3.5-4B-Base with LoRA and a pointer head over supplied options.</td><td>Text decisions; choice, binary and ordered scores; one forward pass.</td><td>Best head-design comparator: scores options through a pointer rather than our fixed output slots.</td></tr>
<tr><th>Mica v0.1 4B<small>{cite('mica')}</small></th><td>Qwen3.5-4B; merged rank-16 LoRA across all 32 layers.</td><td>2–255 options; 8,192-token request cap; English and Korean training.</td><td>Useful first external baseline with an explicitly sized, substantial MMLU-Pro panel.</td></tr>
<tr><th>ImaJev-4B<small>{cite('ima')}</small></th><td>Qwen3.5-4B; rank-64 language LoRA + linear 256-code readout. Vision encoder frozen.</td><td>Text + up to two images; 255 option codes + unknown; 4,096-token request cap; English.</td><td>Closest readout idea, with explicit abstention and vision. Card reports 83.9% on its own ImajevBench; not comparable to our suite.</td></tr>
<tr><th>InnerJev-4B-Full<small>{cite('inner')}</small></th><td>Qwen3.5-4B; full-rank language-linear updates, distilled from the same model’s reasoning readouts.</td><td>Option distributions with thinking disabled; 29,326 training questions.</td><td>Most relevant reasoning-distillation recipe. No matching local-suite score verified for this report.</td></tr>
</tbody></table></div><p class="note">“4B” means the advertised backbone class, not identical deployed parameter counts or memory use. Vision modules, adapters, output heads and quantization differ. Repository parameter badges can count only adapter or packed tensors; they are not a reliable size comparison. The four peer cards list Apache-2.0 for their models; upstream training-data terms still need separate review.</p></section>
<section><h2>Our strongest comparison is still the local one</h2><div class="scroll"><table class="numeric"><thead><tr><th>Benchmark</th><th>Horscht</th><th>Original Qwen3-4B</th><th>Questions</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>Our combined result is <strong>{current['overall']['accuracy']:.2%}</strong>, versus <strong>{original['overall']['accuracy']:.2%}</strong> for the original immediate-answer reference. We have not yet beaten that reference across this suite. Our 85.40% Belebele result is encouraging, but does not establish a lead over peers we have not tested here.</p><p class="note">The original reference uses a different readout. Neither local column generates a thinking trace. BBH covers six tasks, not the complete benchmark; Belebele covers five languages. The combined score is weighted by our question counts and must not be compared with Decision Index or JEVal composites.</p></section>
<section><h2>What I would compare next</h2><ol><li><strong>Mica and Kev, against this frozen checkpoint.</strong> Evaluate the same questions, preserve each model’s documented decision interface, and record rejected inputs separately. Start with MMLU-Pro, then expand if runtime is reasonable.</li><li><strong>A no-training Qwen3.5-4B decision readout.</strong> The LLM2Jev paper reports that an untrained 4B readout can match community decision models. This is the necessary control before crediting a newer backbone’s gains to training. {cite('llm')}</li><li><strong>ImaJev when image decisions become a priority; InnerJev when testing reasoning distillation.</strong> Both are useful recipe comparisons, but changing backbone, data and head together would not tell us which change helped.</li></ol><p class="note">Recommendation only: this report has not started model downloads, training or new benchmark jobs. A fair latency comparison also needs the same hardware, precision, input lengths, batching and warm-up policy; published milliseconds are not ranked against our AMD machine.</p></section>
<details><summary>Where Clef and Jev fit</summary><p>Clef-Flash is <strong>9B</strong> on Qwen3.5; Clef is <strong>27B</strong> on Qwen3.8. Both use a joint transformer schema head and support multimodal inputs. They are larger reference systems, not our same-size peer group. {cite('flash')} · {cite('clef')}</p><p>Cloudflare reports MMLU-Pro accuracy of 65.3% for Clef-Flash, 65.9% for Clef and 82.7% for Jev in its Decision Index evaluation. Those are externally reported results under another protocol. Jev’s parameter count was not verified here, so it is not classified as a 4B peer. {cite('clef')}</p></details>
<details><summary>Evidence, limitations and reproduction</summary><p>Research checked on 2026-10-09. External claims come from developer model cards or the linked research paper; none of the peer models was run locally for this report. Cards and releases can change. Kev’s own result and its older result in Mica’s table are deliberately kept separate.</p><p>Our current result is selected update 1080, following a 23,006-question continuation of an already trained checkpoint. This is not its complete lifetime training count. Selection used 128 development questions. The 17,874-question external suite has been observed before and is exploratory; the additional internal test remains deferred.</p><p>Only aggregate scores and public model metadata were inspected. Benchmark items remain evaluation-only. Our training mixture includes Banking77 and SciQ, so a future comparison using those sources needs explicit split and overlap checks. There is no claim of pretraining contamination being ruled out.</p><p>The existing local report verifies the aggregate file against its completion receipt. This report reads those same aggregate files directly. It contains no external scripts or chart dependencies.</p><p><a href="/runs/new-data-last-block-2026-10-08/fast-evaluation-003/benchmark-report.html">Local results and checkpoint details</a> · <a href="/runs/new-data-last-block-2026-10-08/fast-evaluation-003/benchmark-report.json">Completion receipt</a></p></details>
<footer>Prepared for the Horscht experiment · 2026-10-09 · Source links accompany the relevant claims.<br><a href="/docs/paper/paper.html">Model paper</a> · <a href="/runs/project-record-2026-10-04.html">All experiment results</a></footer></main></html>'''
    OUTPUT.write_text(html)
    return OUTPUT


if __name__ == '__main__':
    print(build())
