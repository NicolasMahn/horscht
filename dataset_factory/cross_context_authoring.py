"""Bounded author-claim checks outside learner input; no semantic-gold claim."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile

from dataset_factory import authoring_writer as writer


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = "cross-context-dependency-v1-137729"
START = 137729
LIMIT = 4096
BANDS = [[1000, 1999], [2000, 3999], [4000, 8000]]
TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
RESERVE = 12 * 1024**3


class ContractError(ValueError):
    pass


def require(condition, code):
    if not condition:
        raise ContractError(code)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def path_at(value):
    require(isinstance(value, (str, Path)) and bool(str(value).strip()), "invalid_local_path")
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def document_pin(binding):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "invalid_document_binding")
    path = path_at(binding["path"])
    require(path.is_relative_to(ROOT / "docs"), "document_outside_authoring_scope")
    digest = binding["sha256"]
    require(isinstance(digest, str) and len(digest) == 64
            and all(char in "0123456789abcdef" for char in digest), "invalid_document_pin")
    require(sha(path.read_bytes()) == digest, "document_pin_mismatch")
    return path, digest


def load_profile(path, expected_sha=None):
    data = Path(path).read_bytes()
    require(expected_sha is None or sha(data) == expected_sha, "profile_pin_mismatch")
    profile = json.loads(data)
    require(isinstance(profile, dict), "invalid_dependency_profile")
    required = {"name": CONTRACT, "start_index": START, "max_assignments": LIMIT,
                "selection_probability": 0.2, "prompt_token_bands": BANDS, "max_prompt_tokens": 10240,
                "context_length_policy": "selected_full_prompt_band_overrides_legacy_context_length_target"}
    require(all(profile.get(key) == value and not isinstance(profile.get(key), bool)
                for key, value in required.items()), "invalid_dependency_profile_bounds")
    require(isinstance(profile.get("seed"), int) and not isinstance(profile["seed"], bool), "invalid_selection_seed")
    tokenizer = profile.get("tokenizer")
    require(isinstance(tokenizer, dict) and isinstance(tokenizer.get("files"), dict)
            and set(tokenizer["files"]) == TOKENIZER_FILES, "invalid_tokenizer_pins")
    path_at(tokenizer.get("path"))
    for digest in [profile.get("formatter_sha256"), profile.get("helper_sha256"), *tokenizer["files"].values()]:
        require(isinstance(digest, str) and len(digest) == 64
                and all(char in "0123456789abcdef" for char in digest), "invalid_artifact_pin")
    document_pin(profile.get("contract"))
    return profile


def assignment_selection(profile, assignment_id, global_index):
    """Pure preallocation API: every index in the cohort remains in its denominator."""
    require(isinstance(assignment_id, str) and assignment_id, "invalid_assignment_id")
    require(isinstance(global_index, int) and not isinstance(global_index, bool)
            and START <= global_index < START + LIMIT, "assignment_outside_dependency_window")
    def draw(purpose):
        data = json.dumps([CONTRACT, profile["seed"], assignment_id, purpose], separators=(",", ":")).encode()
        return int.from_bytes(hashlib.sha256(data).digest()[:8], "big")
    selected = draw("selection") < 2**64 // 5
    return {"id": assignment_id, "required_cross_context_dependency": selected,
            "prompt_token_band": BANDS[draw("length_band") % len(BANDS)] if selected else None}


def load_job(job_path):
    path = path_at(job_path)
    require(path.parent.name == "jobs" and path.parent.parent.parent == ROOT / "runs"
            and path.suffix == ".json", "job_path_outside_runs")
    job_bytes = path.read_bytes()
    raw = json.loads(job_bytes)
    require(isinstance(raw, dict), "invalid_job_json")
    dataset = ROOT / "data/generated_train/probability-v2"
    for field, directory in (("plan", "assignments"), ("output", "questions"), ("skip_file", "questions")):
        require(path_at(raw.get(field)).is_relative_to(dataset / directory), "job_artifact_outside_training_scope")
    require(isinstance(raw.get("plan_sha256"), str), "plan_pin_required")
    binding = raw.get("cross_context_profile")
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "missing_dependency_profile_binding")
    profile_path = path_at(binding["path"])
    run = path.parent.parent
    require(profile_path.parent == run, "profile_outside_own_run")
    profile = load_profile(profile_path, binding["sha256"])
    guidance_pin = document_pin({"path": raw.get("guidance_path"), "sha256": raw.get("guidance_sha256")})
    require(raw.get("cross_context_checker_sha256") == sha(Path(__file__).read_bytes()), "checker_pin_mismatch")
    sidecar = path_at(raw.get("dependency_evidence_file"))
    require(sidecar == run / "dependency-evidence" / f"{path.stem}.jsonl", "evidence_ledger_outside_own_run")
    require(path_at(raw.get("scratch_prefix")) == run / "work" / path.stem, "scratch_outside_own_job")
    require(len({path, profile_path, sidecar, *(path_at(raw[field]) for field in ("plan", "output", "skip_file"))}) == 6,
            "dependency_paths_overlap")
    job, plans = writer._load_job(path)
    selections = {}
    for identifier in job["remaining_ids"]:
        plan = plans[identifier]
        require(plan.get("distribution_source_kind") in (writer.v2.SOFT_SOURCE_KINDS | {writer.v2.HARD_LABEL})
                and plan.get("answerability_status") in writer.v2.ANSWERABILITY_STATUSES
                and isinstance(plan.get("choice_count"), int) and not isinstance(plan["choice_count"], bool)
                and 2 <= plan["choice_count"] <= 265
                and isinstance(plan.get("operation_id"), str) and bool(plan["operation_id"]), "unsupported_assignment_metadata")
        generator = plan.get("_generator", {})
        require(isinstance(generator, dict), "invalid_assignment_generator")
        selections[identifier] = assignment_selection(profile, identifier, generator.get("global_index"))
    pins = {path: sha(job_bytes), profile_path: binding["sha256"], job["plan"]: raw["plan_sha256"],
            Path(__file__): raw["cross_context_checker_sha256"], Path(writer.__file__): profile["helper_sha256"],
            ROOT / "experiment.py": profile["formatter_sha256"]}
    pins.update([document_pin(profile["contract"]), guidance_pin])
    verify_pins(pins)
    return path, job, plans, profile, selections, sidecar, pins


def verify_pins(pins):
    require(all(sha(path.read_bytes()) == expected for path, expected in pins.items()), "dependency_metadata_changed")


def inspect_job(job_path, ids=None):
    _, job, plans, _, selections, _, pins = load_job(job_path)
    if ids is None:
        status = writer._job_status(job, plans, writer.MAX_DRAFTS)
        require(status["snapshot_available"], "fresh_status_unavailable")
        ids = status["next_ids"]
    else:
        require(isinstance(ids, list) and 1 <= len(ids) <= 8 and len(set(ids)) == len(ids)
                and all(identifier in job["remaining_ids"] for identifier in ids), "invalid_inspection_ids")
    verify_pins(pins)
    return {"assignments": [selections[identifier] for identifier in ids]}


def validate_evidence(context, evidence, answerability):
    require(isinstance(context, str) and bool(context.strip()) and isinstance(evidence, dict), "missing_dependency_evidence")
    require(set(evidence) == {"blocks", "combined_dependency_rationale", "no_summary_shortcut", "no_padding", "review_status"},
            "invalid_dependency_evidence_fields")
    require(evidence["review_status"] == "author_self_review_only" and evidence["no_summary_shortcut"] is True
            and evidence["no_padding"] is True and isinstance(evidence["combined_dependency_rationale"], str)
            and bool(evidence["combined_dependency_rationale"].strip()), "missing_combined_dependency_claim")
    blocks = evidence["blocks"]
    require(isinstance(blocks, list) and len(blocks) >= 3, "insufficient_essential_blocks")
    regions, identifiers, spans = set(), set(), []
    bounds = {"beginning": (0, len(context) // 3), "middle": (len(context) // 3, 2 * len(context) // 3),
              "end": (2 * len(context) // 3, len(context))}
    for block in blocks:
        require(isinstance(block, dict) and set(block) == {"block_id", "start", "end", "region", "role", "essential",
                "dependency_effect", "ablation_rationale"}, "invalid_dependency_block_fields")
        identifier, start, end, region = (block[key] for key in ("block_id", "start", "end", "region"))
        require(isinstance(identifier, str) and bool(identifier.strip()) and identifier not in identifiers,
                "duplicate_or_invalid_block_id")
        require(isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool)
                and region in bounds and bounds[region][0] <= start < end <= bounds[region][1]
                and bool(context[start:end].strip()), "invalid_dependency_block_position")
        require(block["role"] == "substantive" and block["essential"] is True, "nonessential_dependency_block")
        effect = block["dependency_effect"]
        require(effect in {"changes_target", "makes_undetermined"}
                or (effect == "changes_missing_information_analysis" and answerability == "unanswerable"),
                "invalid_dependency_effect")
        require(isinstance(block["ablation_rationale"], str) and bool(block["ablation_rationale"].strip()),
                "missing_ablation_rationale")
        identifiers.add(identifier)
        regions.add(region)
        spans.append((start, end))
    require(regions == set(bounds), "dependency_does_not_span_context")
    spans.sort()
    require(all(left[1] <= right[0] for left, right in zip(spans, spans[1:])), "overlapping_dependency_blocks")


def available_memory():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise ContractError("memory_reserve_unavailable")


def load_tokenizer(profile):
    require(available_memory() >= RESERVE + 1024**3, "insufficient_tokenizer_memory_reserve")
    source = path_at(profile["tokenizer"]["path"])
    # Give AutoTokenizer only the pinned tokenizer files, never model weights,
    # optional unpinned configuration, remote code, or a GPU model.
    with tempfile.TemporaryDirectory(prefix="cross-context-tokenizer-") as directory:
        target = Path(directory)
        for name, expected in profile["tokenizer"]["files"].items():
            data = (source / name).read_bytes()
            require(sha(data) == expected, "tokenizer_pin_mismatch")
            (target / name).write_bytes(data)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(target, local_files_only=True, trust_remote_code=False)
    require(available_memory() >= RESERVE, "insufficient_tokenizer_memory_reserve")
    return tokenizer


def prompt_length(row, tokenizer, profile):
    require(sha((ROOT / "experiment.py").read_bytes()) == profile["formatter_sha256"], "formatter_pin_mismatch")
    from experiment import format_input
    prompt = format_input(row)
    encoded = tokenizer(prompt, add_special_tokens=True, truncation=False)
    return len(encoded["input_ids"])


def check_prompt_lengths(staged, claims, profile, tokenizer=None):
    lock_path = ROOT / "runs/.cross-context-tokenizer.lock"
    with lock_path.open("a") as lock:
        # Acquire before importing either the tokenizer or canonical formatter.
        # One local process owns the expensive CPU check at a time.
        fcntl.flock(lock, fcntl.LOCK_EX)
        require(available_memory() >= RESERVE + 1024**3, "insufficient_tokenizer_memory_reserve")
        tokenizer = load_tokenizer(profile) if tokenizer is None else tokenizer
        for claim in claims:
            length = prompt_length(staged[claim["assignment_id"]], tokenizer, profile)
            require(length <= 10240, "prompt_exceeds_verified_capacity")
            band = claim["prompt_token_band"]
            require(band is None or band[0] <= length <= band[1], "prompt_outside_assigned_band")
            claim["full_prompt_tokens"] = length
        require(available_memory() >= RESERVE, "insufficient_tokenizer_memory_reserve")
        source = path_at(profile["tokenizer"]["path"])
        require(all(sha((source / name).read_bytes()) == digest
                    for name, digest in profile["tokenizer"]["files"].items()), "tokenizer_pin_mismatch")


def append(job_path, drafts_path, evidence_path, *, tokenizer=None):
    path, job, plans, profile, selections, sidecar, pins = load_job(job_path)
    drafts_path, evidence_path = path_at(drafts_path), path_at(evidence_path)
    owned = {path, job["plan"], job["output"], job["skip_file"], sidecar, *pins}
    require(drafts_path not in owned and evidence_path not in owned and drafts_path != evidence_path, "input_path_overlaps_ledger")
    scratch = path_at(job["scratch_prefix"])
    require(drafts_path.parent == scratch and evidence_path.parent == scratch,
            "draft_or_evidence_outside_own_scratch")
    draft_bytes, evidence_bytes = drafts_path.read_bytes(), evidence_path.read_bytes()
    drafts, evidence_rows = writer._parse_jsonl(draft_bytes), writer._parse_jsonl(evidence_bytes)
    pins.update({drafts_path: sha(draft_bytes), evidence_path: sha(evidence_bytes)})
    require(1 <= len(drafts) <= 8 and len(evidence_rows) <= 8, "batch_must_be_1_to_8")
    evidence_by_id = {}
    for record in evidence_rows:
        require(isinstance(record, dict) and set(record) == {"assignment_id", "author_dependency_evidence"}, "invalid_evidence_record")
        identifier = record["assignment_id"]
        require(isinstance(identifier, str) and identifier in job["remaining_ids"] and identifier not in evidence_by_id,
                "duplicate_or_outside_evidence_id")
        evidence_by_id[identifier] = record["author_dependency_evidence"]
    staged, claims = {}, []
    for draft in drafts:
        identifier = draft.get("assignment_id")
        require(isinstance(identifier, str) and identifier in job["remaining_ids"] and identifier not in staged,
                "duplicate_or_outside_draft_id")
        row = writer._build_row(plans[identifier], draft)
        selection = selections[identifier]
        selected = selection["required_cross_context_dependency"]
        evidence = evidence_by_id.get(identifier)
        if selected:
            validate_evidence(draft["context"], evidence, plans[identifier]["answerability_status"])
        elif identifier in evidence_by_id:
            validate_evidence(draft["context"], evidence, plans[identifier]["answerability_status"])
        staged[identifier] = row
        claims.append({"assignment_id": identifier, "status": "preappend_author_claim",
                       "review_status": "author_self_review_only", "job_sha256": pins[path],
                       "profile_sha256": job["cross_context_profile"]["sha256"], "checker_sha256": sha(Path(__file__).read_bytes()),
                       "draft_sha256": sha(json.dumps(draft, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()),
                       "built_row_sha256": sha(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()),
                       "legacy_context_length_target_overridden": selected,
                       **selection, "author_dependency_evidence": evidence})
    require(set(evidence_by_id) <= set(staged), "evidence_id_not_in_draft_batch")
    check_prompt_lengths(staged, claims, profile, tokenizer)
    verify_pins(pins)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    with sidecar.with_name(f".{sidecar.name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ContractError("dependency_evidence_locked") from None
        old = {}
        for claim in writer._jsonl(sidecar, missing_ok=True):
            identifier = claim.get("assignment_id")
            require(isinstance(identifier, str) and identifier in job["remaining_ids"] and identifier not in old,
                    "invalid_existing_evidence_ledger")
            old[identifier] = claim
        for claim in claims:
            identifier = claim["assignment_id"]
            require(identifier not in old or old[identifier] == claim, "dependency_evidence_retry_conflict")
        new = [claim for claim in claims if claim["assignment_id"] not in old]
        verify_pins(pins)
        writer._durable_append(sidecar, new)
        # Claims precede source durability. A crash here leaves a claim, never a
        # completion receipt; acceptance is always reconciled from source IDs.
        # Delegate the exact bytes checked, even if the author's scratch file
        # changes after the last pin verification.
        with tempfile.NamedTemporaryFile(dir=scratch, prefix=".checked-drafts-", suffix=".jsonl") as frozen:
            frozen.write(draft_bytes)
            frozen.flush()
            result = writer.append_drafts(path, Path(frozen.name))
    return {**result, "author_claims_appended": len(new), "selected_checked": sum(
            claim["required_cross_context_dependency"] for claim in claims), "checked": len(claims)}


def audit_job(job_path):
    job_path, job, plans, _, selections, sidecar, pins = load_job(job_path)
    files = (job["output"], job["skip_file"], sidecar)
    before = tuple(writer._file_stamp(path) for path in files)
    status = writer._job_status(job, plans, writer.MAX_DRAFTS)
    require(status["snapshot_available"], "fresh_status_unavailable")
    written = writer._ledger(job["output"], plans)
    skipped = writer._ledger(job["skip_file"], plans, skip=True)
    require(not written.keys() & skipped.keys(), "source_skip_ledger_conflict")
    claims = {}
    for claim in writer._jsonl(sidecar, missing_ok=True):
        identifier = claim.get("assignment_id")
        require(identifier in job["remaining_ids"] and identifier not in claims, "invalid_existing_evidence_ledger")
        require(claim.get("required_cross_context_dependency") == selections[identifier]["required_cross_context_dependency"]
                and claim.get("prompt_token_band") == selections[identifier]["prompt_token_band"]
                and claim.get("profile_sha256") == job["cross_context_profile"]["sha256"], "evidence_selection_mismatch")
        require(claim.get("status") == "preappend_author_claim" and claim.get("review_status") == "author_self_review_only"
                and claim.get("job_sha256") == pins[job_path]
                and claim.get("checker_sha256") == pins[Path(__file__)], "evidence_receipt_pin_mismatch")
        length, band = claim.get("full_prompt_tokens"), claim.get("prompt_token_band")
        require(isinstance(length, int) and not isinstance(length, bool) and 0 < length <= 10240
                and (band is None or band[0] <= length <= band[1]), "invalid_receipt_prompt_length")
        claims[identifier] = claim
    require(before == tuple(writer._file_stamp(path) for path in files), "audit_boundary_in_flight")
    verify_pins(pins)
    allowed = job["remaining_ids"]
    accepted = allowed & written.keys() & claims.keys()
    for identifier in accepted:
        require(claims[identifier].get("built_row_sha256") == sha(json.dumps(
                written[identifier], sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()), "accepted_claim_row_mismatch")
        if selections[identifier]["required_cross_context_dependency"]:
            validate_evidence(written[identifier]["context"], claims[identifier].get("author_dependency_evidence"),
                              plans[identifier]["answerability_status"])
    selected = {identifier for identifier, selection in selections.items() if selection["required_cross_context_dependency"]}
    return {"planned": len(allowed), "selected": len(selected), "written": len(allowed & written.keys()),
            "skipped": len(allowed & skipped.keys()), "selected_skipped": len(selected & skipped.keys()),
            "author_claims": len(claims), "accepted_claims": len(accepted),
            "selected_accepted_claims": len(selected & accepted),
            "claimed_not_written": len(claims.keys() - written.keys()),
            "written_missing_claims": len((allowed & written.keys()) - claims.keys()),
            "unaccounted": len(allowed - written.keys() - skipped.keys()),
            "bands": [{"band": band, "planned_selected": sum(selections[identifier]["prompt_token_band"] == band for identifier in selected),
                       "accepted_selected": sum(selections[identifier]["prompt_token_band"] == band for identifier in selected & accepted)}
                      for band in BANDS]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--job", type=Path, required=True)
    inspect.add_argument("--ids", nargs="+")
    audit = commands.add_parser("audit")
    audit.add_argument("--job", type=Path, required=True)
    write = commands.add_parser("append")
    write.add_argument("--job", type=Path, required=True)
    write.add_argument("--drafts", type=Path, required=True)
    write.add_argument("--evidence", type=Path, required=True)
    select = commands.add_parser("selection")
    select.add_argument("--profile", type=Path, required=True)
    select.add_argument("--id", required=True)
    select.add_argument("--index", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            result = inspect_job(args.job, args.ids)
        elif args.command == "audit":
            result = audit_job(args.job)
        elif args.command == "selection":
            result = assignment_selection(load_profile(args.profile), args.id, args.index)
        else:
            result = append(args.job, args.drafts, args.evidence)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ContractError, writer.WriterError) as exc:
        print(json.dumps({"error_code": str(exc)}))
    except Exception:
        print(json.dumps({"error_code": "cross_context_contract_failed"}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
