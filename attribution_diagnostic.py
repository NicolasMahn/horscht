"""Exact fact-group Shapley and backbone comparisons on new synthetic probes.

This program never opens benchmark items or training examples. It compares each
system's native readout, and separately compares backbones on identical tokens.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from report_theme import PALETTE

ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "runs/attribution-diagnostic-2026-10-04"
MODEL_ID = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
EARLY = ROOT / "runs/paired-overnight-2026-09-30/full/checkpoints/update-00000512-epoch-001-next-000256"
CURRENT = ROOT / "runs/current-corpus-2026-10-01/full/checkpoints/update-00004608-epoch-000-next-004608"
FROZEN = ROOT / "runs/paired-overnight-2026-09-30/head/checkpoints/update-00000128-epoch-000-next-000128"
PRESENTATION = {
    "original": {"label": "Original Qwen3-4B · LM head", "color": "purple"},
    "frozen_head": {"label": "Earlier frozen backbone · decision head", "color": "teal"},
    "early_full": {"label": "Earlier full training · update 512", "color": "orange"},
    "current_full": {"label": "Current full training · update 4608", "color": "blue"},
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixtures() -> list[dict[str, Any]]:
    def item(identifier, groups, question, options, answer):
        return {"id": identifier, "groups": [{"name": name, "text": text} for name, text in groups],
                "question": question, "options": options, "gold": options.index(answer)}

    neutral = " ".join(f"Archive entry {i} records a routine inspection of the west corridor; no parcels were handled." for i in range(1, 25))
    return [
        item("parcel-route-4", [("initial location", "Mina placed the brass token in the amber box."),
                               ("move", "Leo moved the token from the amber box into the desk drawer."),
                               ("irrelevant note", "The silver coin stayed on the windowsill.")],
             "Where is the brass token now?", ["Amber box", "Desk drawer", "Windowsill", "Coat pocket"], "Desk drawer"),
        item("inventory-2", [("initial count", "A tray initially contained 7 screws."),
                             ("removal", "A mechanic removed 2 screws and added none."),
                             ("irrelevant note", "A second tray contains 11 washers.")],
             "How many screws remain in the first tray?", ["5", "7"], str(7 - 2)),
        item("weekday-6", [("starting day", "The shipment was logged on Tuesday."),
                            ("elapsed time", "Delivery occurred exactly 3 days after it was logged."),
                            ("irrelevant note", "A separate invoice was signed on Monday.")],
             "On which weekday was the shipment delivered?", ["Sunday", "Monday", "Wednesday", "Thursday", "Friday", "Saturday"], "Friday"),
        item("lookup-10", [("lookup key", "The requested entry has tag LUMA."),
                            ("mapping", "Tag LUMA maps to code 7; tag NERO maps to code 2."),
                            ("irrelevant note", "Code 9 is used for maintenance records.")],
             "What code belongs to the requested entry?", [str(i) for i in range(10)], str(7)),
        item("transitive-4", [("first comparison", "Ada's cable is longer than Bea's cable."),
                               ("second comparison", "Bea's cable is longer than Cal's cable."),
                               ("irrelevant note", "All three cables have identical connectors.")],
             "Whose cable is longest?", ["Cal", "Bea", "Ada", "All are equally long"], "Ada"),
        item("rule-2", [("rule", "A gate opens if and only if its status lamp is green."),
                        ("observation", "The gate's status lamp is red and is not green."),
                        ("irrelevant note", "The nearby wall lamp is green.")],
             "Is the gate open according to this rule?", ["Yes", "No"], "No"),
        item("german-route-6", [("Ausgangsort", "Der blaue Stift liegt zuerst in der Schublade."),
                                 ("Bewegung", "Nora legt den blauen Stift aus der Schublade in die Tasche."),
                                 ("Nebeninformation", "Das rote Buch bleibt auf dem Tisch.")],
             "Wo liegt der blaue Stift zuletzt?", ["Im Regal", "Auf dem Tisch", "In der Schublade", "In der Tasche", "Am Fenster", "Auf dem Boden"], "In der Tasche"),
        item("long-parcel-10", [("initial location", "A sealed teal parcel starts in locker 3."),
                                ("move", "The courier moves the teal parcel from locker 3 to locker 8, with no later moves."),
                                ("irrelevant archive", neutral)],
             "Which locker contains the sealed teal parcel now?", [f"Locker {i}" for i in range(10)], "Locker 8"),
    ]


def masked_row(fixture: dict[str, Any], mask: int) -> dict[str, Any]:
    groups = fixture["groups"]
    if not 0 <= mask < 2 ** len(groups):
        raise ValueError("Mask is outside the fixture's group set")
    return {"context": "\n".join(group["text"] for index, group in enumerate(groups) if mask & (1 << index)),
            "question": fixture["question"], "options": list(fixture["options"])}


def exact_shapley(values: list[float], groups: int) -> list[float]:
    if len(values) != 2 ** groups or groups < 1:
        raise ValueError("Exactly one value per coalition is required")
    attribution = []
    for index in range(groups):
        total = 0.0
        for mask in range(2 ** groups):
            if mask & (1 << index):
                continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(groups - size - 1) / math.factorial(groups)
            total += weight * (values[mask | (1 << index)] - values[mask])
        attribution.append(total)
    return attribution


def target_contrast(probabilities: list[float], target: int) -> float:
    if not 0 <= target < len(probabilities) or len(probabilities) < 2:
        raise ValueError("Invalid fixed answer target")
    return probabilities[target] - 1 / len(probabilities)


def summarize_scores(fixture: dict[str, Any], scores: list[list[float]]) -> dict[str, Any]:
    count, target = len(fixture["options"]), fixture["gold"]
    if len(scores) != 2 ** len(fixture["groups"]) or any(len(score) != count for score in scores):
        raise ValueError("Coalition scoring changed the option set")
    values = [target_contrast(probabilities, target) for probabilities in scores]
    phi = exact_shapley(values, len(fixture["groups"]))
    return {"probabilities_by_mask": scores, "target_contrast_by_mask": values,
            "full_target_probability": scores[-1][target], "empty_target_probability": scores[0][target],
            "full_prediction": max(range(count), key=lambda index: scores[-1][index]),
            "correct": max(range(count), key=lambda index: scores[-1][index]) == target,
            "shapley": phi, "efficiency_residual": sum(phi) - (values[-1] - values[0])}


def chat_prompt(tokenizer, row):
    choices = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(row["options"]))
    messages = [{"role": "system", "content": "Select the best answer. Reply with exactly one uppercase option letter and no explanation."},
                {"role": "user", "content": f"Context:\n{row['context']}\n\nQuestion:\n{row['question']}\n\nOptions:\n{choices}\n\nReply with one option letter only."}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def label_ids(tokenizer, prompt, count):
    base = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    identifiers = []
    for index in range(count):
        extended = tokenizer(prompt + chr(65 + index), add_special_tokens=False)["input_ids"]
        if extended[:-1] != base or len(extended) != len(base) + 1:
            raise ValueError("Candidate label must be one prefix-stable token")
        identifiers.append(extended[-1])
    if len(set(identifiers)) != count:
        raise ValueError("Candidate label token IDs must be unique")
    return base, identifiers


def activation_distance(reference, candidate) -> dict[str, float]:
    import torch
    from torch.nn.functional import cosine_similarity
    reference, candidate = reference.float(), candidate.float()
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError("Activation comparison requires aligned tokens and features")
    epsilon = torch.finfo(torch.float32).eps
    return {"mean_token_cosine": cosine_similarity(reference, candidate, dim=-1).mean().item(),
            "relative_l2": ((candidate - reference).norm() / reference.norm().clamp_min(epsilon)).item(),
            "last_token_cosine": cosine_similarity(reference[-1], candidate[-1], dim=0).item(),
            "last_token_relative_l2": ((candidate[-1] - reference[-1]).norm() / reference[-1].norm().clamp_min(epsilon)).item()}


def parameter_changes(reference, candidate):
    import torch
    if set(reference) != set(candidate):
        raise ValueError("Backbone parameter names do not match")
    groups = {}
    for name, initial in reference.items():
        trained = candidate[name]
        if initial.shape != trained.shape:
            raise ValueError("Backbone parameter shapes do not match")
        group = ".".join(name.split(".")[:2]) if name.startswith("layers.") else name.split(".")[0]
        summary = groups.setdefault(group, {"parameters": 0, "changed": 0, "initial_squared_norm": 0.0, "delta_squared_norm": 0.0, "max_abs_change": 0.0})
        # Chunking avoids a second model-sized temporary allocation on this UMA PC.
        initial, trained = initial.reshape(-1), trained.reshape(-1)
        for start in range(0, initial.numel(), 1024**2):
            a, b = initial[start:start + 1024**2].float(), trained[start:start + 1024**2].float()
            delta = b - a
            summary["parameters"] += a.numel()
            summary["changed"] += torch.count_nonzero(delta).item()
            summary["initial_squared_norm"] += torch.sum(a.double().square()).item()
            summary["delta_squared_norm"] += torch.sum(delta.double().square()).item()
            summary["max_abs_change"] = max(summary["max_abs_change"], delta.abs().max().item())
    total = {key: sum(group[key] for group in groups.values()) for key in ("parameters", "changed", "initial_squared_norm", "delta_squared_norm")}
    total["max_abs_change"] = max(group["max_abs_change"] for group in groups.values())
    for summary in [*groups.values(), total]:
        summary["changed_fraction"] = summary["changed"] / summary["parameters"]
        summary["relative_l2"] = math.sqrt(summary.pop("delta_squared_norm") / max(summary.pop("initial_squared_norm"), 1e-30))
    return {"overall": total, "groups": groups}


def protocol(selected):
    return {"schema_version": "synthetic-fact-group-attribution-v1", "created_at": now(),
            "model_id": MODEL_ID, "revision": REVISION, "fixtures": selected,
            "fixture_source": "new handwritten synthetic development diagnostics; no benchmark or training-data input",
            "checkpoints": {"early_full": str(EARLY.relative_to(ROOT)), "current_full": str(CURRENT.relative_to(ROOT)), "frozen_head": str(FROZEN.relative_to(ROOT))},
            "value": "P(fixed gold choice | coalition) - 1/K; gold index is identical for every coalition and model",
            "attribution": "exact Shapley over all 2^3 fact-group coalitions; not token SHAP, KernelSHAP, or a reasoning trace",
            "background": "empty context; absent groups are deleted; question, options, order, and full target remain fixed",
            "readouts": {"original": "native no-thinking chat with uppercase labels; restricted original LM-head softmax",
                         "frozen_head": "plain numeric prompt; earlier trained decision head on unchanged original backbone",
                         "early_full": "plain numeric prompt; checkpoint decision head", "current_full": "plain numeric prompt; checkpoint decision head"},
            "activation_inputs": "identical complete plain numeric prompts and token IDs across the original and fully trained backbones",
            "activation_metrics": "aligned-token average cosine and relative L2, plus last-token versions; no fit or alignment",
            "runtime": "float32 weights, common bf16 autocast backbone, sdpa, batch one, no cache, no truncation; decision heads float32",
            "not_allowed": ["training", "calibration", "prompt selection", "checkpoint selection", "benchmark item reads", "original LM-head transplantation onto trained backbones"],
            "caveats": ["Native prompts and readouts differ; attribution changes cannot isolate forgetting.",
                        "Deleting facts changes input length and may create unnatural or underdetermined masked inputs.",
                        "Shapley values explain changes in one output under this background; they do not expose internal reasoning.",
                        "Equal reasoning does not imply equal attributions after training; confidence and calibration matter.",
                        "Activation and parameter changes prove change, not lost or retained capabilities.",
                        "The small fixed probe suite is exploratory and cannot establish general capability retention.",
                        "The head-only control was trained in the earlier paired run; it is not a matched head-only control for the current corpus.",
                        "Shared-token activation prompts match the decision systems; original native attribution uses its chat wrapper."]}


def run(output: Path, limit: int, max_seconds: int):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiment import DecisionHead, format_input

    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        raise FileExistsError("Completed diagnostic exists; use a fresh output directory")
    selected = fixtures()[:limit]
    started = time.monotonic()
    def status(phase, **extra):
        write_json(output / "status.json", {"phase": phase, "pid": os.getpid(), "updated_at": now(), "elapsed_seconds": round(time.monotonic() - started, 1), **extra})
        print(json.dumps({"phase": phase, **extra}), flush=True)
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Diagnostic exceeded its fixed wall-clock budget")

    try:
        status("preflight")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU unavailable")
        torch.set_num_threads(4)
        free, total = torch.cuda.mem_get_info()
        if free < 24 * 1024**3:
            raise RuntimeError("GPU free-memory guard requires at least 24 GiB")
        torch.cuda.set_per_process_memory_fraction(min(32 * 1024**3 / total, 0.85))
        spec = protocol(selected)
        spec["code_sha256"] = sha256(Path(__file__))
        for name, directory in (("early_full", EARLY), ("current_full", CURRENT), ("frozen_head", FROZEN)):
            checkpoint = json.loads((directory / "checkpoint.json").read_text())
            if checkpoint["include_backbone"] != (name != "frozen_head") or checkpoint["manifest"]["revision"] != REVISION:
                raise ValueError("Pinned checkpoint contract failed")
            if checkpoint["manifest"]["backbone_trainable"] != (name != "frozen_head"):
                raise ValueError("Training manifest backbone contract failed")
            spec.setdefault("checkpoint_bindings", {})[name] = {"checkpoint_json_sha256": sha256(directory / "checkpoint.json"),
                "files": {filename: checkpoint["files"][filename] for filename in ("backbone.pt", "head.pt") if filename in checkpoint["files"]},
                "backbone_trainable": checkpoint["manifest"]["backbone_trainable"]}
        write_json(output / "protocol.json", spec)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
        # Check the complete tokenization, rather than assuming matching artifact names imply identical tokens.
        for directory in (EARLY, CURRENT, FROZEN):
            checkpoint_tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True)
            for fixture in selected:
                for mask in range(8):
                    text = format_input(masked_row(fixture, mask))
                    if tokenizer(text)["input_ids"] != checkpoint_tokenizer(text)["input_ids"]:
                        raise ValueError("Checkpoint tokenization differs on diagnostic inputs")
            del checkpoint_tokenizer
        status("load_original")
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True,
            trust_remote_code=False, dtype=torch.float32, attn_implementation="sdpa", device_map="cuda")
        model.eval().requires_grad_(False)
        backbone = model.model
        frozen = DecisionHead(backbone.config.hidden_size).to("cuda").eval().requires_grad_(False)
        frozen.load_state_dict(torch.load(FROZEN / "head.pt", map_location="cpu", weights_only=True))
        if sha256(FROZEN / "head.pt") != spec["checkpoint_bindings"]["frozen_head"]["files"]["head.pt"]:
            raise ValueError("Frozen head hash mismatch")
        result = {"protocol": spec, "started_at": now(), "models": {}, "parameters": {}, "activations": {},
                  "runtime": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "free_gib_before_load": free / 1024**3}}
        reference_parameters = {name: value.detach().cpu().clone() for name, value in backbone.state_dict().items()}
        reference_activations = {}

        def forward(text, hidden=False):
            tokens = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
            if tokens["input_ids"].shape[-1] > 4096:
                raise ValueError("Synthetic diagnostic input exceeded fixed token budget")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                value = backbone(**tokens, use_cache=False, output_hidden_states=hidden)
            return value, tokens["input_ids"][0].tolist()

        def head_probabilities(value, head, count):
            with torch.inference_mode():
                logits = head(value.last_hidden_state[:, -1].float(), torch.tensor([count], device="cuda"))
                return torch.softmax(logits[0, :count], -1).cpu().tolist()

        original_scores, frozen_scores = {}, {}
        for fixture in selected:
            native, frozen_native = [], []
            for mask in range(8):
                status("score_original", fixture=fixture["id"], mask=mask, completed=len(original_scores), total=len(selected))
                row = masked_row(fixture, mask)
                prompt = chat_prompt(tokenizer, row)
                _, ids = label_ids(tokenizer, prompt, len(row["options"]))
                value, _ = forward(prompt)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                    # Project only valid labels at the answer position, not the entire vocabulary at every token.
                    logits = torch.nn.functional.linear(value.last_hidden_state[:, -1], model.get_output_embeddings().weight[ids])
                native.append(torch.softmax(logits[0].float(), -1).cpu().tolist())
                value, token_ids = forward(format_input(row), hidden=mask == 7)
                frozen_native.append(head_probabilities(value, frozen, len(row["options"])))
                if mask == 7:
                    reference_activations[fixture["id"]] = {"token_ids": token_ids,
                        "states": [state[0].detach().float().cpu() for state in value.hidden_states]}
            original_scores[fixture["id"]] = summarize_scores(fixture, native)
            frozen_scores[fixture["id"]] = summarize_scores(fixture, frozen_native)
        result["models"]["original"], result["models"]["frozen_head"] = original_scores, frozen_scores
        # Remove the LM readout before replacing backbone weights; it is never used on a tuned backbone.
        model.lm_head = None
        del frozen
        gc.collect()
        torch.cuda.empty_cache()
        write_json(output / "partial-result.json", result)

        for model_name, directory in (("early_full", EARLY), ("current_full", CURRENT)):
            status("verify_checkpoint", model=model_name)
            for filename, expected in spec["checkpoint_bindings"][model_name]["files"].items():
                if sha256(directory / filename) != expected:
                    raise ValueError("Pinned checkpoint artifact hash mismatch")
            state = torch.load(directory / "backbone.pt", map_location="cpu", weights_only=True, mmap=True)
            status("compare_parameters", model=model_name)
            result["parameters"][model_name] = parameter_changes(reference_parameters, state)
            status("load_tuned", model=model_name)
            backbone.load_state_dict(state, strict=True)
            del state
            head = DecisionHead(backbone.config.hidden_size).to("cuda").eval().requires_grad_(False)
            head.load_state_dict(torch.load(directory / "head.pt", map_location="cpu", weights_only=True))
            scores, distances = {}, {}
            for fixture in selected:
                native = []
                for mask in range(8):
                    status("score_tuned", model=model_name, fixture=fixture["id"], mask=mask, completed=len(scores), total=len(selected))
                    value, token_ids = forward(format_input(masked_row(fixture, mask)), hidden=mask == 7)
                    native.append(head_probabilities(value, head, len(fixture["options"])))
                    if mask == 7:
                        reference = reference_activations[fixture["id"]]
                        if token_ids != reference["token_ids"] or len(value.hidden_states) != len(reference["states"]):
                            raise ValueError("Activation inputs or layer counts are not identical")
                        distances[fixture["id"]] = [activation_distance(a, b[0].detach().cpu())
                            for a, b in zip(reference["states"], value.hidden_states)]
                scores[fixture["id"]] = summarize_scores(fixture, native)
            result["models"][model_name], result["activations"][model_name] = scores, distances
            write_json(output / "partial-result.json", result)
            del head
            gc.collect()
            torch.cuda.empty_cache()
        result["completed_at"] = now()
        result["runtime"]["elapsed_seconds"] = time.monotonic() - started
        result["runtime"]["peak_gpu_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
        result["input_tokens"] = {key: len(value["token_ids"]) for key, value in reference_activations.items()}
        write_json(output / "result.json", result)
        build_report(result, output / "report.html")
        status("completed", result=str(output / "result.json"), report=str(output / "report.html"))
    except BaseException as error:
        write_json(output / "status.json", {"phase": "failed", "pid": os.getpid(), "updated_at": now(),
            "elapsed_seconds": round(time.monotonic() - started, 1), "error_type": type(error).__name__, "error": str(error)})
        raise


def build_report(result, destination):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    selected = result["protocol"]["fixtures"]
    sections = []
    sections.append("<p>Exploratory diagnostic on newly written synthetic examples. Exact fact-group Shapley explains changes in a fixed answer probability when facts are removed. It does not reveal a model's reasoning.</p>")
    overview = go.Figure()
    for name, values in result["models"].items():
        overview.add_bar(name=PRESENTATION[name]["label"], x=[item["id"] for item in selected],
            y=[values[item["id"]]["full_target_probability"] for item in selected], marker_color=PALETTE[PRESENTATION[name]["color"]])
    overview.update_layout(barmode="group", yaxis_title="Probability of fixed correct choice", yaxis_range=[0, 1], height=430)
    sections.append(overview.to_html(full_html=False, include_plotlyjs=True))
    rows = []
    for name, values in result["models"].items():
        correct = sum(value["correct"] for value in values.values())
        confidence = sum(value["full_target_probability"] for value in values.values()) / len(values)
        rows.append(f"<tr><td>{PRESENTATION[name]['label']}</td><td>{correct}/{len(values)}</td><td>{confidence:.3f}</td></tr>")
    sections.append("<table><tr><th>System</th><th>Correct synthetic answers</th><th>Mean correct-choice probability</th></tr>" + "".join(rows) + "</table>")
    sections.append("<p>Each system uses its native answer interface. The original uses a no-thinking chat prompt and its original LM head; trained systems use the existing plain numeric prompt and decision heads. Their probability scales and prompts differ. The frozen-head system uses the unchanged original backbone with the earlier trained decision head.</p>")
    if result["parameters"]:
        sections.append("<p>The stored backbone weights were compared directly with the pinned original weights, before loading each trained backbone. These differences verify that full training changed the backbone.</p>")
        rows = []
        for name, values in result["parameters"].items():
            overall = values["overall"]
            rows.append(f"<tr><td>{PRESENTATION[name]['label']}</td><td>{overall['changed_fraction']:.2%}</td><td>{overall['relative_l2']:.6f}</td><td>{overall['max_abs_change']:.6f}</td></tr>")
        sections.append("<table><tr><th>Backbone</th><th>Changed parameter elements</th><th>Relative L2 change</th><th>Largest absolute change</th></tr>" + "".join(rows) + "</table>")
        activation = make_subplots(rows=1, cols=2, subplot_titles=["Aligned tokens · mean cosine", "Aligned tokens · relative L2 change"])
        for name, probes in result["activations"].items():
            length = len(next(iter(probes.values())))
            for column, metric in ((1, "mean_token_cosine"), (2, "relative_l2")):
                mean = [sum(probe[layer][metric] for probe in probes.values()) / len(probes) for layer in range(length)]
                activation.add_scatter(x=list(range(length)), y=mean, name=PRESENTATION[name]["label"],
                    line_color=PALETTE[PRESENTATION[name]["color"]], showlegend=column == 1, row=1, col=column)
        activation.update_layout(height=430)
        activation.update_xaxes(title_text="Hidden-state index (0 = embeddings; last = normalized final output)")
        sections.append(activation.to_html(full_html=False, include_plotlyjs=False))
        sections.append("<p>Activation comparison uses identical full plain prompts and exactly identical token IDs. Each point averages over the synthetic fixtures. State 0 is the embedding output; the final state includes final normalization. Cosine and relative L2 show representation change, not whether a capability survived. No LM head is attached to either trained backbone.</p>")
    for fixture in selected:
        identifier = fixture["id"]
        parts = [f"<details><summary>{html.escape(identifier)} · {len(fixture['options'])} choices</summary>",
                 "<p>" + html.escape(fixture["question"]) + "</p>",
                 "<ol>" + "".join(f"<li>{html.escape(group['name'])}: {html.escape(group['text'])}</li>" for group in fixture["groups"]) + "</ol>",
                 "<p>Options: " + html.escape("; ".join(f"{index}: {option}" for index, option in enumerate(fixture["options"]))) + ". Fixed target: " + html.escape(fixture["options"][fixture["gold"]]) + ".</p>"]
        attribution = go.Figure()
        for name, values in result["models"].items():
            attribution.add_bar(name=PRESENTATION[name]["label"], x=[group["name"] for group in fixture["groups"]],
                y=values[identifier]["shapley"], marker_color=PALETTE[PRESENTATION[name]["color"]])
        attribution.update_layout(barmode="group", yaxis_title="Shapley contribution (probability)", height=340)
        parts.append(attribution.to_html(full_html=False, include_plotlyjs=False))
        parts.append("<table><tr><th>System</th><th>Empty-context probability</th><th>Full-context probability</th><th>Chosen option</th><th>Shapley efficiency residual</th></tr>")
        for name, values in result["models"].items():
            value = values[identifier]
            parts.append(f"<tr><td>{PRESENTATION[name]['label']}</td><td>{value['empty_target_probability']:.3f}</td><td>{value['full_target_probability']:.3f}</td><td>{html.escape(fixture['options'][value['full_prediction']])}</td><td>{value['efficiency_residual']:.2e}</td></tr>")
        parts.append("</table></details>")
        sections.append("".join(parts))
    sections.append("<ul>" + "".join("<li>" + html.escape(caveat) + "</li>" for caveat in result["protocol"]["caveats"]) + "</ul>")
    palette = PALETTE
    from string import Template
    css = Template("body{background:$canvas;color:$ink;font:16px/1.6 system-ui;margin:24px auto;max-width:1200px;padding:0 20px} table{border-collapse:collapse;width:100%;background:$paper} th,td{text-align:left;padding:8px;border-bottom:1px solid $line} details{background:$paper;margin:16px 0;padding:12px} summary{cursor:pointer;font-weight:600}").substitute(palette)
    # Plotly cannot measure a collapsed detail's width until the reader opens it.
    resize_details = "<script>document.querySelectorAll('details').forEach(detail => detail.addEventListener('toggle', () => { if (detail.open) detail.querySelectorAll('.plotly-graph-div').forEach(plot => Plotly.Plots.resize(plot)); }));</script>"
    destination.write_text("<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'><title>Synthetic attribution diagnostic</title><style>" + css + "</style></head><body>" + "".join(sections) + resize_details + "</body></html>")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--max-seconds", type=int, default=1200)
    parser.add_argument("--report-only", type=Path)
    args = parser.parse_args()
    if args.report_only:
        build_report(json.loads(args.report_only.read_text()), args.output / "report.html")
    else:
        run(args.output, args.limit, args.max_seconds)


if __name__ == "__main__":
    main()
