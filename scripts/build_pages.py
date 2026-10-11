"""Build and validate the public site from an explicit publication allowlist.

No training rows, benchmark items, local runs, or model weights enter this build.
The checked-in paper already embeds its figures and needs no rendering packages.
"""

import argparse
import json
import re
import shutil
import sys
from html import escape
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_peer_comparison_report import build as build_peer_comparison
from report_theme import PALETTE

REPOSITORY = "https://github.com/nicolasmahn/horscht"
MODEL = "https://huggingface.co/NicolasMahn/horscht-qwen3-4b-v0.1"
DATASET = "https://huggingface.co/datasets/NicolasMahn/horscht-decisions-v0.1"
PAPER = ROOT / "docs/paper"
DEFAULT_OUTPUT = ROOT / "publicsite/_site"
FIGURES = (
    "benchmark-comparison",
    "continuation-sources",
    "development-selection",
    "adaptation-arms",
    "bbh-task-results",
)
AGGREGATES = (
    "horscht-v0.1.json",
    "previous-last-block.json",
    "original-qwen3-4b-immediate-answer.json",
)
PRESENTATION = {
    "background": "canvas", "surface": "paper", "text": "ink",
    "secondary": "muted", "border": "line", "accent": "blue",
}


def read_aggregate(name):
    result = json.loads((ROOT / "release/evidence" / name).read_text())
    total = result["overall"]
    slices = result["by_source_slice"].values()
    if (total["examples"] != 17874
            or sum(row["examples"] for row in slices) != total["examples"]
            or sum(row["correct"] for row in slices) != total["correct"]):
        raise ValueError(f"Invalid public aggregate counts: {name}")
    return result


def public_paper_link(url):
    """Keep downloadable paper assets local; resolve source references on GitHub."""
    parsed = urlsplit(url)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return url
    if parsed.path in {"paper.html", "paper.md", "evidence.json"}:
        return url
    if parsed.path.startswith("figures/"):
        return url
    target = (PAPER / unquote(parsed.path)).resolve()
    relative = target.relative_to(ROOT)
    if not target.is_file():
        raise ValueError(f"Missing repository reference: {relative}")
    suffix = ("?" + parsed.query if parsed.query else "")
    suffix += "#" + parsed.fragment if parsed.fragment else ""
    return f"{REPOSITORY}/blob/main/{relative.as_posix()}{suffix}"


def rewrite_paper_html(source):
    return re.sub(
        r'(\b(?:href|src)=)([\"\'])(.*?)(\2)',
        lambda match: match[1] + match[2] + escape(public_paper_link(match[3]), quote=True) + match[4],
        source,
    )


def rewrite_paper_markdown(source):
    return re.sub(
        r'(\]\()([^\s)]+)',
        lambda match: match[1] + public_paper_link(match[2]),
        source,
    )


def page(title, content):
    tokens = ";".join(f"--{role}:{PALETTE[color]}" for role, color in PRESENTATION.items())
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Horscht: an open Qwen3-4B decision-model experiment with model weights, training data and aggregate evidence.">
<title>{escape(title)}</title><style>
:root{{{tokens}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--background);color:var(--text);font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{max-width:1060px;margin:auto;padding:36px 26px 64px}}nav{{display:flex;justify-content:space-between;gap:20px;font-size:13px;margin-bottom:68px}}a{{color:var(--accent);text-underline-offset:4px}}.brand{{color:var(--text);font-weight:750;letter-spacing:.06em;text-decoration:none}}.version,footer,.note{{color:var(--secondary);font-size:13px;line-height:1.7}}h1{{font-family:Georgia,"Times New Roman",serif;font-size:clamp(40px,7vw,72px);font-weight:500;line-height:1.08;letter-spacing:-.04em;max-width:850px;margin:0 0 24px}}.intro{{font-size:19px;line-height:1.7;max-width:740px}}p{{line-height:1.7}}.actions{{display:flex;flex-wrap:wrap;gap:12px;margin:30px 0 42px}}.button{{border:1px solid var(--border);background:var(--surface);padding:12px 20px;border-radius:30px;text-decoration:none;font-weight:600}}.button.primary{{background:var(--accent);border-color:var(--accent);color:var(--surface)}}.panel{{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:26px;margin:22px 0}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}}.number{{display:block;font-size:32px;font-weight:700;letter-spacing:-.03em;margin:8px 0}}.label{{font-size:14px;color:var(--secondary)}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{text-align:right;padding:15px 12px;border-bottom:1px solid var(--border);white-space:nowrap}}th:first-child,td:first-child{{text-align:left;padding-left:0}}thead th{{font-size:12px;color:var(--secondary)}}tr:last-child td,tr:last-child th{{border-bottom:0}}.links{{display:flex;flex-wrap:wrap;gap:14px 24px}}footer{{margin-top:42px}}.chart{{display:block;width:100%;height:auto}}h2{{font-size:22px;letter-spacing:-.02em;margin-top:32px}}
@media(max-width:650px){{main{{padding:26px 18px 44px}}nav{{margin-bottom:42px;flex-wrap:wrap}}.intro{{font-size:17px}}.panel{{padding:20px}}.metrics{{grid-template-columns:1fr;gap:18px}}.number{{font-size:28px}}}}
</style></head><body><main><nav><a class="brand" href="index.html">HORSCHT</a><span class="version">Qwen3-4B · v0.1</span></nav>
{content}
<footer>By Nicolas Mahn · Report snapshot 2026-10-09<br><a href="{REPOSITORY}">Source and reproduction instructions</a> · <a href="paper.html">Technical report</a></footer>
</main></body></html>'''


def landing_page(current):
    return page("Horscht · A decision model experiment", f'''
<h1>A small update.<br>A model that chooses.</h1>
<p class="intro">Horscht adapts Qwen3-4B to return probabilities over supplied choices in one forward pass. This release trains the final transformer block and a linear decision head, and publishes the model, training mixture and evidence.</p>
<div class="actions"><a class="button primary" href="paper.html">Read the paper →</a><a class="button" href="{MODEL}">Get the model</a><a class="button" href="{DATASET}">Explore the dataset</a></div>
<div class="panel metrics"><div><span class="label">External suite accuracy</span><span class="number">{current['overall']['accuracy']:.2%}</span><span class="note">17,874 questions · exploratory</span></div><div><span class="label">Trainable parameters</span><span class="number">101.6M</span><span class="note">Final block + 265-output head</span></div><div><span class="label">Continuation training rows</span><span class="number">23,006</span><span class="note">An addition to earlier training</span></div></div>
<p>The continuation improves on its previous checkpoint across MMLU-Pro, Belebele and six BBH tasks. The original Qwen3-4B immediate-answer reference still leads on overall accuracy. The external suite has been reused; these results describe an experiment, rather than an untouched final test.</p>
<div class="links"><a href="benchmark.html">Benchmark results →</a><a href="comparison.html">Compare with other decision models →</a><a href="{REPOSITORY}">GitHub source →</a></div>
<p class="note">The published mixture includes source-specific licenses, including noncommercial material. Read the dataset card before reuse.</p>
<p class="note">Horscht v0.1 carries a non-commercial-use policy because its training included SciQ data licensed CC-BY-NC-3.0. A potential future version would replace non-commercial training sources and retrain from a checkpoint that predates them, with the aim of releasing our original contributions under MIT for commercial as well as non-commercial use. Qwen3-4B's Apache-2.0 license and other applicable third-party obligations would remain in place. This is a future direction, not a license change or a promise of a new release.</p>''')


def benchmark_page(results):
    rows = []
    for label, prefix in (("MMLU-Pro", "mmlu-"), ("Belebele · five languages", "belebele-"), ("BBH · six tasks", "bbh-"), ("Overall", "")):
        columns = []
        for result in results:
            slices = [row for key, row in result["by_source_slice"].items() if key.startswith(prefix)]
            count = sum(row["examples"] for row in slices)
            accuracy = sum(row["correct"] for row in slices) / count
            columns.append(f"<td>{accuracy:.2%}</td>")
        rows.append(f"<tr><th scope=\"row\">{escape(label)}</th><td>{count:,}</td>{''.join(columns)}</tr>")
    evidence_links = " · ".join(f'<a href="evidence/{name}">{escape(label)}</a>' for name, label in zip(AGGREGATES, ("Horscht v0.1", "Previous checkpoint", "Original Qwen3-4B")))
    return page("Horscht · Benchmark results", f'''
<h1>A modest gain across three benchmarks.</h1>
<p class="intro">The selected checkpoint improves on its warm start, while the original Qwen3-4B immediate-answer reference remains ahead overall.</p>
<div class="panel scroll"><table><thead><tr><th scope="col">Benchmark</th><th scope="col">Questions</th><th scope="col">Previous</th><th scope="col">Horscht v0.1</th><th scope="col">Original Qwen3-4B</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<p>All columns cover the same 17,874 questions. Horscht and the previous checkpoint use a learned decision head; the original reference uses original-head label likelihood. No column generates a thinking trace. Belebele covers German, English, French, Spanish and Chinese. BBH covers logical deduction and shuffled-object tracking at three sizes.</p>
<p>The checkpoint was selected using 128 development questions. External probabilities use uncalibrated masked softmax, with a 32,768-token cap and no truncation. This suite was observed in earlier experiments, so these results are exploratory. The additional internal test remains deferred.</p>
<div class="panel"><img class="chart" src="figures/benchmark-comparison.svg" alt="Accuracy comparison of the previous checkpoint, Horscht v0.1 and original Qwen3-4B across the three benchmark groups."></div>
<div class="links"><a href="comparison.html">Peer comparison →</a><a href="paper.html#5-aggregate-results">Probability metrics and task results →</a></div>
<p class="note">Download aggregate evidence: {evidence_links}. The exports preserve source SHA-256 digests and include no benchmark items or predictions.</p>''')


class DocumentLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.anchors = set()

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if attributes.get("id"):
            self.anchors.add(attributes["id"])
        for key in ("href", "src"):
            if attributes.get(key):
                self.links.append(attributes[key])


def validate_site(output, expected):
    actual = {path.relative_to(output).as_posix() for path in output.rglob("*") if path.is_file()}
    if actual != expected:
        raise ValueError(f"Publication allowlist mismatch: {actual ^ expected}")
    documents = {}
    for path in output.rglob("*.html"):
        source = path.read_text()
        if any(marker in source for marker in ("localhost", "file://", "/home/", 'href="/runs/', 'href="/data/')):
            raise ValueError(f"Private or invalid link in {path.name}")
        parser = DocumentLinks()
        parser.feed(source)
        documents[path.resolve()] = parser
    for path, document in documents.items():
        for link in document.links:
            parsed = urlsplit(link)
            if parsed.scheme in {"https", "http", "data", "mailto"}:
                continue
            if parsed.scheme or parsed.netloc or parsed.path.startswith("/"):
                raise ValueError(f"Unsupported site link: {link}")
            target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
            if not target.is_relative_to(output.resolve()) or not target.is_file():
                raise ValueError(f"Broken site link in {path.name}: {link}")
            if parsed.fragment and target in documents and unquote(parsed.fragment) not in documents[target].anchors:
                raise ValueError(f"Broken anchor in {path.name}: {link}")


def build(output=DEFAULT_OUTPUT):
    output = Path(output).resolve()
    # Restrict cleanup to this dedicated build directory, never an arbitrary path.
    if output.name != "_site" or output == ROOT:
        raise ValueError("Output must be a dedicated directory named _site")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    expected = {"index.html", "paper.html", "paper.md", "evidence.json", "benchmark.html", "comparison.html", ".nojekyll"}
    (output / "paper.html").write_text(rewrite_paper_html((PAPER / "paper.html").read_text()))
    (output / "paper.md").write_text(rewrite_paper_markdown((PAPER / "paper.md").read_text()))
    shutil.copyfile(PAPER / "evidence.json", output / "evidence.json")
    for figure in FIGURES:
        for extension in ("svg", "png"):
            name = f"figures/{figure}.{extension}"
            (output / "figures").mkdir(exist_ok=True)
            shutil.copyfile(PAPER / name, output / name)
            expected.add(name)
    for name in AGGREGATES:
        (output / "evidence").mkdir(exist_ok=True)
        shutil.copyfile(ROOT / "release/evidence" / name, output / "evidence" / name)
        expected.add(f"evidence/{name}")
    current, previous, original = [read_aggregate(name) for name in AGGREGATES]
    (output / "index.html").write_text(landing_page(current))
    (output / "benchmark.html").write_text(benchmark_page((previous, current, original)))
    build_peer_comparison(output / "comparison.html", public_links=True)
    (output / ".nojekyll").touch()
    validate_site(output, expected)
    print(f"Validated {len(expected)} public files in {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    build(parser.parse_args().output)
