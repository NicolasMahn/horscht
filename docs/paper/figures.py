"""Publication figures drawn only from public aggregates and source counts."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from report_theme import PALETTE

# Report concepts have one color assignment across every exported figure.
PRESENTATION = {
    "background": "paper",
    "text": "ink",
    "secondary": "muted",
    "grid": "line",
    "current": "blue",
    "previous": "teal",
    "original": "orange",
    "head_only": "purple",
    "full": "rose",
    "selected": "teal",
    "chance": "orange",
    "source": "blue",
}


def color(role: str) -> str:
    return PALETTE[PRESENTATION[role]]


def _canvas(title: str, note: str, *, height: float = 4.9) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(9, height))
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.23, top=0.82)
    fig.text(0.035, 0.94, title, fontsize=15, fontweight="bold", va="top")
    fig.text(0.035, 0.04, note, fontsize=12, color=color("secondary"), va="bottom", linespacing=1.45)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=0, pad=7)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(color("grid"))
    return fig, ax


def _accuracy_axis(ax: Axes, *, horizontal: bool = True) -> None:
    if horizontal:
        ax.set_xlim(0, 100)
        ax.set_xticks(range(0, 101, 20))
        ax.set_xlabel("Accuracy (%)", labelpad=10)
        ax.grid(axis="x", color=color("grid"), linewidth=0.8)
    else:
        ax.set_ylim(0, 100)
        ax.set_yticks(range(0, 101, 20))
        ax.set_ylabel("Accuracy (%)", labelpad=10)
        ax.grid(axis="y", color=color("grid"), linewidth=0.8)


def _benchmark_summary(result: dict[str, Any], prefix: str) -> tuple[float, int]:
    rows = [row for name, row in result["by_source_slice"].items() if name.startswith(prefix)]
    if not rows:
        raise ValueError(f"No aggregate slices for {prefix}")
    total = sum(row["examples"] for row in rows)
    correct = sum(row["correct"] for row in rows)
    return 100 * correct / total, total


def _benchmark_comparison(evidence: dict[str, Any]) -> Figure:
    fig, ax = _canvas(
        "External benchmark accuracy",
        "Same evaluated items; descriptive results from a reused external suite.\n"
        "Original Qwen3-4B uses label likelihood; Horscht uses its decision head.",
        height=5.3,
    )
    ax.set_position([0.105, 0.26, 0.865, 0.49])
    benchmarks = [("MMLU-Pro", "mmlu-pro"), ("Belebele", "belebele-"), ("BBH", "bbh-")]
    models = [("previous", "Previous Horscht"), ("current", "Current Horscht"), ("original", "Original Qwen3-4B")]
    width = 0.24
    counts = []
    for series, (key, label) in enumerate(models):
        summaries = [_benchmark_summary(evidence["aggregate_results"][key], prefix) for _, prefix in benchmarks]
        values = [value for value, _ in summaries]
        if counts and counts != [count for _, count in summaries]:
            raise ValueError("Benchmark comparison counts differ between models")
        counts = [count for _, count in summaries]
        bars = ax.bar([i + (series - 1) * width for i in range(3)], values, width,
                      color=color(key), label=label,
                      hatch="//" if key == "original" else None,
                      edgecolor=color("background"), linewidth=0.6)
        ax.bar_label(bars, labels=[f"{value:.1f}" for value in values], padding=4, fontsize=12)
    ax.set_xticks(range(3), [f"{name}\nn = {count:,}" for (name, _), count in zip(benchmarks, counts)])
    _accuracy_axis(ax, horizontal=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.04), ncol=3, frameon=False,
              fontsize=12, handlelength=1.4, columnspacing=1.3)
    return fig


def _adaptation_arms(evidence: dict[str, Any]) -> Figure:
    results = evidence["aggregate_results"]
    attention = results["attention"]["overall"]["accuracy"] * 100
    fig, ax = _canvas(
        "Historical adaptation arms",
        "Item-weighted accuracy across 17,874 external items; historical runs on the earlier corpus.\n"
        f"Later attention readout: {attention:.2f}%; a separate experiment, not a matched fourth arm.",
    )
    ax.set_position([0.20, 0.30, 0.76, 0.49])
    models = [("head_only", "Head only"), ("previous", "Last block"), ("full", "Full backbone")]
    values = [results[key]["overall"]["accuracy"] * 100 for key, _ in models]
    bars = ax.barh(range(3), values, height=0.52, color=[color(key) for key, _ in models])
    ax.bar_label(bars, labels=[f"{value:.2f}%" for value in values], padding=7, fontsize=12)
    ax.set_yticks(range(3), [label for _, label in models])
    ax.set_ylim(2.65, -0.65)
    _accuracy_axis(ax)
    return fig


def _development_selection(evidence: dict[str, Any]) -> Figure:
    scores = sorted((int(update), float(value)) for update, value in evidence["development_soft_cross_entropy"].items())
    selected = evidence["selected_update"]
    fig, ax = _canvas(
        "Checkpoint selection on development data",
        f"Five measured checkpoints on {evidence['development_questions']} development questions; lower is better.\n"
        "Points show measurements only. The vertical axis is zoomed and does not start at zero.",
        height=5.3,
    )
    ax.set_position([0.13, 0.27, 0.82, 0.52])
    updates, values = zip(*scores)
    ax.scatter(updates, values, s=65, color=color("previous"), zorder=3)
    selected_value = dict(scores)[selected]
    ax.scatter([selected], [selected_value], s=150, marker="D", color=color("selected"),
               edgecolor=color("background"), linewidth=1.3, zorder=4)
    for update, value in scores:
        label = f"{value:.4f}"
        offset = (0, 13)
        if update == selected:
            label = f"Selected · {value:.4f}"
            offset = (-32, 20)
        elif update == max(updates):
            offset = (0, -24)
        ax.annotate(label, (update, value), textcoords="offset points", xytext=offset,
                    ha="center", fontsize=12,
                    fontweight="bold" if update == selected else "normal")
    ax.set_xlim(-100, max(updates) + 100)
    ax.set_ylim(min(values) - 0.035, max(values) + 0.028)
    ax.set_xticks(updates)
    ax.set_xlabel("Training update", labelpad=10)
    ax.set_ylabel("Soft cross-entropy (nats; zoomed axis)", labelpad=10)
    ax.grid(axis="y", color=color("grid"), linewidth=0.8)
    return fig


def _continuation_sources() -> Figure:
    manifest = json.loads((ROOT / "release/dataset/manifest.json").read_text())
    continuation = manifest["configs"]["continuation"]
    counts = {key.removeprefix("train/"): value for key, value in continuation["by_source"].items()
              if key.startswith("train/")}
    total = continuation["exported"]["train"]
    if len(counts) != 6 or sum(counts.values()) != total:
        raise ValueError("Continuation source counts do not match the six-source release")
    names = {"original_replay": "Original replay", "games": "Games", "documents": "Documents",
             "typed-decisions": "Typed decisions", "sciq": "SciQ", "banking77": "BANKING77"}
    sources = sorted(counts, key=lambda key: (-counts[key], key))
    values = [counts[key] for key in sources]
    fig, ax = _canvas(
        "Continuation training data",
        f"{total:,} training rows in the 2026-10-08-v4 release; row counts, not sampling weights.\n"
        "Source: public release manifest. Development, calibration, and test rows are excluded.",
        height=5.5,
    )
    ax.set_position([0.23, 0.29, 0.70, 0.52])
    bars = ax.barh(range(6), values, height=0.59, color=color("source"))
    ax.bar_label(bars, labels=[f"{value:,}  ({100 * value / total:.1f}%)" for value in values],
                 padding=7, fontsize=12)
    ax.set_yticks(range(6), [names[key] for key in sources])
    ax.set_ylim(5.65, -0.65)
    ax.set_xlim(0, max(values) * 1.26)
    ax.set_xticks([0, 2000, 4000, 6000, 8000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, position: f"{value:,.0f}"))
    ax.set_xlabel("Training rows", labelpad=10)
    ax.grid(axis="x", color=color("grid"), linewidth=0.8)
    return fig


def _bbh_tasks(evidence: dict[str, Any]) -> Figure:
    slices = evidence["aggregate_results"]["current"]["by_source_slice"]
    tasks = []
    for family, label in [("logical-deduction", "Logical deduction"), ("tracking-shuffled-objects", "Shuffled tracking")]:
        for count, word in [(3, "three"), (5, "five"), (7, "seven")]:
            row = slices[f"bbh-{family}-{word}-objects"]
            tasks.append((f"{label} · {count} choices", row))
    counts = {row["examples"] for _, row in tasks}
    if counts != {250}:
        raise ValueError("BBH task figure requires 250 items per task")
    fig, ax = _canvas(
        "Current Horscht across six BBH tasks",
        "250 evaluated items per task; descriptive results from a reused external suite.\n"
        "Diamonds show the uniform-choice baseline (1 / number of choices).",
        height=5.9,
    )
    ax.set_position([0.36, 0.29, 0.60, 0.49])
    values = [row["accuracy"] * 100 for _, row in tasks]
    chance = [row["uniform_baseline"]["accuracy"] * 100 for _, row in tasks]
    bars = ax.barh(range(6), values, height=0.55, color=color("current"), label="Current Horscht")
    ax.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=6, fontsize=12)
    # Place the baseline beside each bar so its marker never obscures a score label.
    ax.scatter(chance, [i + 0.34 for i in range(6)], s=42, marker="D", color=color("chance"),
               label="Uniform chance", zorder=3)
    ax.set_yticks(range(6), [label for label, _ in tasks])
    ax.set_ylim(5.7, -0.7)
    _accuracy_axis(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.05), ncol=2, frameon=False,
              fontsize=12, handlelength=1.4)
    return fig


def build_figures(evidence: dict[str, Any], output_dir: Path | None = None) -> dict[str, dict[str, Path]]:
    """Export SVG with editable text and 200-dpi PNG from verified public evidence."""
    destination = Path(output_dir) if output_dir is not None else Path(__file__).with_suffix("")
    destination.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, dict[str, Path]] = {}
    style = {
        "font.family": "DejaVu Sans", "font.size": 12,
        "text.color": color("text"), "axes.labelcolor": color("text"),
        "xtick.color": color("secondary"), "ytick.color": color("text"),
        "figure.facecolor": color("background"), "axes.facecolor": color("background"),
        "savefig.facecolor": color("background"), "svg.fonttype": "none",
        "svg.hashsalt": "horscht-paper", "axes.unicode_minus": False,
    }
    builders = {
        "benchmark-comparison": lambda: _benchmark_comparison(evidence),
        "adaptation-arms": lambda: _adaptation_arms(evidence),
        "development-selection": lambda: _development_selection(evidence),
        "continuation-sources": _continuation_sources,
        "bbh-task-results": lambda: _bbh_tasks(evidence),
    }
    with plt.rc_context(style):
        for name, build in builders.items():
            fig = build()
            try:
                paths = {extension: destination / f"{name}.{extension}" for extension in ("svg", "png")}
                fig.savefig(paths["svg"], metadata={"Date": None, "Creator": "Horscht public aggregate figure builder"})
                fig.savefig(paths["png"], dpi=200, metadata={"Software": "Horscht public aggregate figure builder"})
                artifacts[name] = paths
            finally:
                plt.close(fig)
    return artifacts


if __name__ == "__main__":
    public_evidence = json.loads((Path(__file__).parent / "evidence.json").read_text())
    generated = build_figures(public_evidence)
    print(json.dumps({name: {kind: str(path) for kind, path in paths.items()} for name, paths in generated.items()}))
