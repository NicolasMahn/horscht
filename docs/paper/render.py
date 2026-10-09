"""Verify the report's aggregate evidence and render its standalone HTML."""
from __future__ import annotations

import hashlib
import base64
import json
from html import escape
from pathlib import Path
import re
import sys

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[2]
PAPER = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from report_theme import PALETTE

PRESENTATION = {
    "background": "canvas", "surface": "paper", "text": "ink",
    "secondary": "muted", "border": "line", "accent": "blue",
    "evidence": "teal", "caution": "orange",
}
RESULTS = "data/external_evaluation/results"
RUN = "runs/new-data-last-block-2026-10-08/fast-evaluation-003"
AGGREGATES = {
    "current": f"{RESULTS}/new-data-last-block-fast-2026-10-09/blind-aggregate-result.json",
    "previous": f"{RESULTS}/current-corpus-last-block-2026-10-05/blind-aggregate-result.json",
    "head_only": f"{RESULTS}/current-corpus-head-only-2026-10-04/blind-aggregate-result.json",
    "full": f"{RESULTS}/current-corpus-benchmark-2026-10-04-final/blind-aggregate-result.json",
    "attention": f"{RESULTS}/current-corpus-option-attention-2026-10-06/blind-aggregate-result.json",
    "original": f"{RESULTS}/original-qwen-label-likelihood-2026-10-04/blind-aggregate-result.json",
}
RECEIPTS = {
    "completion": f"{RUN}/benchmark-report.json",
    "selection": f"{RUN}/development-selection.json",
    "calibration": f"{RUN}/calibration.json",
    "checkpoint": f"{RUN}/selected-checkpoint.json",
    "protocol": f"{RUN}/blind-protocol.json",
    "training_manifest": "runs/new-data-last-block-2026-10-08/main-recovery-002/manifest.json",
    "snapshot_manifest": "data/training_snapshots/2026-10-08-v4/manifest.json",
}


def read(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def digest(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def evidence() -> dict:
    required = [*AGGREGATES.values(), *RECEIPTS.values()]
    if not all((ROOT / path).exists() for path in required):
        saved = json.loads((PAPER / "evidence.json").read_text())
        if saved["schema_version"] != "horscht-paper-evidence-v1":
            raise ValueError("Unexpected public evidence schema")
        for key, result in saved["aggregate_results"].items():
            overall, slices = result["overall"], result["by_source_slice"]
            if (overall["examples"] != 17874
                    or sum(row["examples"] for row in slices.values()) != 17874
                    or sum(row["correct"] for row in slices.values()) != overall["correct"]
                    or abs(overall["accuracy"] - overall["correct"] / 17874) > 1e-12):
                raise ValueError(f"{key}: inconsistent public aggregate")
        scores = saved["development_soft_cross_entropy"]
        if int(min(scores, key=lambda key: (scores[key], int(key)))) != saved["selected_update"]:
            raise ValueError("Public selection does not follow the declared criterion")
        return saved
    records = {key: read(path) for key, path in RECEIPTS.items()}
    completion = records["completion"]
    if digest(AGGREGATES["current"]) != completion["aggregate_sha256"]:
        raise ValueError("Current aggregate does not match the completion receipt")
    if completion["selection"] != records["checkpoint"]:
        raise ValueError("Checkpoint selection receipt differs from completion binding")
    if records["selection"]["selected_update"] != 1080:
        raise ValueError("Paper snapshot requires selected update 1080")
    if records["training_manifest"]["repo"] != "Qwen/Qwen3-4B":
        raise ValueError("Paper snapshot requires Qwen3-4B")
    if records["protocol"]["scoring"]["no_temperature_scaling_or_calibration"] is not True:
        raise ValueError("External probability protocol changed")
    summaries = {}
    for key, path in AGGREGATES.items():
        raw = read(path)
        result = next(iter(raw["candidate_results"].values())) if "candidate_results" in raw else raw
        overall = result["overall"]
        slices = result["by_source_slice"]
        if overall["examples"] != 17874:
            raise ValueError(f"{key}: unexpected suite count")
        if sum(row["examples"] for row in slices.values()) != overall["examples"]:
            raise ValueError(f"{key}: slice counts disagree")
        if sum(row["correct"] for row in slices.values()) != overall["correct"]:
            raise ValueError(f"{key}: slice successes disagree")
        if abs(overall["accuracy"] - overall["correct"] / overall["examples"]) > 1e-12:
            raise ValueError(f"{key}: accuracy inconsistent")
        summaries[key] = {"overall": overall, "by_source_slice": slices}
    if summaries["current"]["overall"]["correct"] != 9008 or summaries["previous"]["overall"]["correct"] != 8783:
        raise ValueError("Paper accuracy counts differ from source snapshot")
    candidate_scores = {key: value["metrics"]["soft_cross_entropy"] for key, value in records["selection"]["candidates"].items()}
    selected = min(candidate_scores, key=lambda key: (candidate_scores[key], int(key)))
    if int(selected) != 1080:
        raise ValueError("Selected candidate does not minimize declared criterion")
    return {
        "schema_version": "horscht-paper-evidence-v1",
        "report_date": "2026-10-09",
        "claim_scope": "descriptive exploratory reused external suite; internal test deferred",
        "sources": {key: {"path": path, "sha256": digest(path)} for key, path in {**AGGREGATES, **RECEIPTS}.items()},
        "aggregate_results": summaries,
        "selected_update": 1080,
        "development_questions": 128,
        "development_soft_cross_entropy": candidate_scores,
        "calibration_questions": 64,
        "temperature": records["calibration"]["temperature"],
        "external_probabilities": "uncalibrated masked softmax",
        "completed_at": completion["completed_at"],
        "authoring_content_audit": {
            "external_benchmark_item_files_read": False,
            "unsolicited_web_search_diagnostic_snippet": True,
            "handling": "not opened, reproduced, or used; recorded as accidental exposure",
        },
    }


def build() -> None:
    verified = evidence()
    from figures import build_figures
    build_figures(verified)
    (PAPER / "evidence.json").write_text(json.dumps(verified, indent=2, sort_keys=True) + "\n")
    source = (PAPER / "paper.md").read_text()
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    renderer.renderer.rules["table_open"] = lambda tokens, index, options, env: '<div class="table-scroll"><table>\n'
    renderer.renderer.rules["table_close"] = lambda tokens, index, options, env: '</table></div>\n'
    def figure(tokens, index, options, env):
        token = tokens[index]
        relative = Path(token.attrGet("src"))
        if relative.parent != Path("figures") or relative.suffix != ".svg":
            raise ValueError("Paper figures must be local generated SVGs")
        encoded = base64.b64encode((PAPER / relative).read_bytes()).decode()
        caption = escape(token.attrGet("title") or "")
        return (f'<figure><div class="figure-plot" tabindex="0" aria-label="Scrollable chart"><img src="data:image/svg+xml;base64,{encoded}" alt="{escape(token.content, quote=True)}"></div>'
                f'<figcaption>{caption}<span class="figure-downloads">'
                f'<a href="{relative.as_posix()}" download>SVG</a> · '
                f'<a href="{relative.with_suffix(".png").as_posix()}" download>PNG</a></span>'
                '</figcaption></figure>')
    renderer.renderer.rules["image"] = figure
    tokens = renderer.parse(source)
    navigation = []
    for index, token in enumerate(tokens):
        if (token.type == "paragraph_open" and index + 2 < len(tokens)
                and tokens[index + 1].type == "inline"
                and len(tokens[index + 1].children or []) == 1
                and tokens[index + 1].children[0].type == "image"):
            token.hidden = tokens[index + 2].hidden = True
        if token.type == "heading_open":
            title = tokens[index + 1].content
            anchor = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            token.attrSet("id", anchor)
            if token.tag == "h2":
                navigation.append(f'<a href="#{anchor}">{escape(title)}</a>')
    content = renderer.renderer.render(tokens, renderer.options, {})
    table_titles = iter(("Exact dataset split counts", "Checkpoint selection measurements",
                         "Exact benchmark scores", "Probability scores", "Exact adaptation scores"))
    content = re.sub(r'<div class="table-scroll"><table>.*?</table></div>',
                     lambda match: '<details class="measurements"><summary>' + next(table_titles)
                     + '</summary>' + match.group(0) + '</details>', content, flags=re.S)
    colors = ";".join(f"--{role}:{PALETTE[value]}" for role, value in PRESENTATION.items())
    styles = """
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;color:var(--text);background:var(--background);font-synthesis:none}
a{color:var(--accent);text-underline-offset:3px}a:hover{text-decoration-thickness:2px}.frame{max-width:1260px;margin:auto;padding:32px 30px 64px}.masthead{display:flex;justify-content:space-between;gap:20px;align-items:center;font-size:13px;color:var(--secondary);padding:0 0 26px}.brand{font-weight:750;letter-spacing:.05em;color:var(--text);text-decoration:none}.masthead span{font-variant-numeric:tabular-nums}.layout{display:grid;grid-template-columns:210px minmax(0,1fr);gap:44px;align-items:start}.toc{position:sticky;top:28px;font-size:12px;line-height:1.5;border-top:1px solid var(--border);padding-top:12px}.toc a{display:block;margin:0 0 13px;color:var(--secondary);text-decoration:none}.toc a:hover{color:var(--accent)}article{background:var(--surface);border:1px solid var(--border);border-radius:18px;padding:42px 48px;min-width:0;box-shadow:0 12px 36px color-mix(in srgb,var(--text) 4%,transparent)}h1{font-family:Georgia,"Times New Roman",serif;font-size:clamp(32px,3.6vw,49px);font-weight:500;line-height:1.12;letter-spacing:-.035em;margin:0 0 20px;max-width:820px}h1+p{font-size:12px;color:var(--secondary);margin-bottom:36px}h2{font-family:Georgia,"Times New Roman",serif;font-size:26px;font-weight:500;letter-spacing:-.015em;line-height:1.2;margin:38px 0 17px;padding-top:24px;border-top:1px solid var(--border);scroll-margin-top:24px}p,li{font-family:Georgia,"Times New Roman",serif;font-size:16px;line-height:1.78;margin:14px 0}strong{font-weight:700}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.81em;overflow-wrap:anywhere;background:var(--background);border-radius:4px;padding:2px 4px}.table-scroll{overflow-x:auto}table{display:table;width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums;margin:24px 0;line-height:1.55}th,td{padding:12px 9px;border-bottom:1px solid var(--border);vertical-align:top}th{background:var(--background);font-weight:650;font-size:11px}th:first-child,td:first-child{text-align:left!important}table a{overflow-wrap:anywhere}footer{font-size:12px;line-height:1.7;color:var(--secondary);margin-top:24px}.actions{display:flex;gap:18px;flex-wrap:wrap}.pill{border:1px solid var(--border);border-radius:30px;padding:7px 13px;color:var(--secondary);text-decoration:none}.pill:hover{color:var(--accent)}
@media(max-width:1050px){.layout{grid-template-columns:1fr}.toc{position:static;display:flex;flex-wrap:wrap;gap:12px 20px;padding-bottom:6px}.toc a{margin:0}article{padding:34px}}
@media(max-width:650px){.frame{padding:20px 14px 36px}.masthead{flex-wrap:wrap}.layout{gap:20px}article{padding:24px 20px;border-radius:12px}p,li{font-size:15px}table{font-size:11px}th,td{padding:10px 5px}h2{font-size:24px}}
@media print{@page{size:A4;margin:19mm 17mm}body{background:var(--surface)}.frame{padding:0;max-width:none}.masthead,.toc,.actions,footer{display:none}.layout{display:block}article{border:0;box-shadow:none;padding:0;border-radius:0}h1{font-size:31pt}h2{font-size:17pt;break-after:avoid;padding-top:14px;margin-top:22px}p,li{font-size:10.5pt;line-height:1.55}h1+p{font-size:9pt}.table-scroll{overflow:visible}table{font-size:8pt;break-inside:avoid}tr{break-inside:avoid}th{font-size:7.5pt}a{color:var(--text);text-decoration:none}code{font-size:8pt}p{orphans:3;widows:3}}
"""
    styles += """
figure{margin:30px -12px 24px;padding:18px 10px 15px;border:1px solid var(--border);border-radius:12px;background:var(--surface);break-inside:avoid}
.figure-plot{overflow-x:auto}figure img{display:block;width:100%;height:auto}figcaption{font-family:system-ui,-apple-system,sans-serif;font-size:12px;line-height:1.65;color:var(--secondary);padding:12px 14px 0}.figure-downloads{display:block;font-size:11px;margin-top:7px}
.measurements{border-top:1px solid var(--border);border-bottom:1px solid var(--border);margin:20px 0}.measurements summary{cursor:pointer;color:var(--accent);font-size:12px;font-weight:600;padding:13px 0}.measurements table{margin:0 0 12px}
@media(max-width:650px){figure{margin:25px -12px;padding:8px 2px 12px}figure img{min-width:650px}figcaption{font-size:11px;padding:8px 10px 0}}
@media print{figure{margin:18px 0;padding:8px;border-radius:0}figcaption{font-size:8pt;line-height:1.4}.figure-downloads{display:none}.measurements{display:block}.measurements>summary{font-size:8pt;color:var(--text)}}
"""
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="Horscht Qwen3-4B technical report: a 265-choice decision head, frozen-prefix adaptation, data lineage and exploratory aggregate results."><title>Horscht · Qwen3-4B technical report</title><style>:root{{{colors}}}{styles}</style></head><body><div class="frame"><header class="masthead"><a class="brand" href="../../README.md">HORSCHT</a><span>TECHNICAL REPORT · 2026-10-09</span><div class="actions"><a class="pill" href="paper.md">Markdown source</a><a class="pill" href="evidence.json">Verified evidence</a></div></header><div class="layout"><nav class="toc" aria-label="Report sections">{''.join(navigation)}</nav><article>{content}</article></div><footer>Repository evidence snapshot · Qwen3-4B · Aggregate evaluation only · No remote scripts, fonts, or images.</footer></div><script>
const printOpened = [];
addEventListener('beforeprint', () => {{
  document.querySelectorAll('details.measurements:not([open])').forEach(node => {{node.open = true; printOpened.push(node);}});
}});
addEventListener('afterprint', () => {{printOpened.splice(0).forEach(node => node.open = false);}});
</script></body></html>'''
    (PAPER / "paper.html").write_text(html)
    print(json.dumps({"rendered": str(PAPER / "paper.html"), "verified_external_rows": 17874, "selected_update": 1080, "remote_rendering_dependencies": 0}))


if __name__ == "__main__":
    build()
