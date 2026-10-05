import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import pytest

from dataset_factory import probability_v2_assignments as planner


def inputs(root, *, wide=True):
    root.mkdir(parents=True, exist_ok=True)
    operations = ["policy_exceptions", "graph_kinship", "tradeoff_optimization",
                  "procedure_recovery", "taxonomy_outliers", "evidence_review"] if wide else ["evidence_review", "sequence_reasoning"]
    catalog = {
        "languages": [{"tag": "en", "name": "English", "family": "test", "script": "Latin"}],
        "domains": [{"id": name, "name": name, "region": "Global"} for name in ("archive", "textile")],
        "operations": [{"id": name, "name": name, "compatible_domains": ["archive", "textile"],
                        "question_forms": ["choose the supported option", "identify the next step"]} for name in operations],
        "input_structures": [{"id": name} for name in (["prose", "table_or_grid"] if wide else ["prose", "table"])],
        "registers": [{"id": "plain"}, {"id": "formal"}],
        "difficulty_bands": [{"id": name} for name in ("easy", "medium", "hard")],
        "context_length_bands": [{"id": name} for name in ("short", "medium", "long")],
        "choice_counts": [2, 3, 4, 265] if wide else [2, 3, 4],
    }
    catalog_path = root / "catalog.json"
    catalog_path.write_text(json.dumps(catalog))
    pool_path = root / "pool.jsonl.gz"
    with gzip.open(pool_path, "wt") as stream:
        for language in ("en", "fr", "ja", "ar"):
            for index in range(4):
                stream.write(json.dumps({"word": f"{language}-seed-{index}", "source_language": language,
                                         "pool_id": "test-pool", "source_version": "fixture-1"}) + "\n")
    return dict(catalog_path=catalog_path, word_pool_path=pool_path,
                assignment_dir=root / "assignments", question_dir=root / "questions",
                legacy_dir=root / "legacy", pilot_file=root / "missing-pilot.jsonl")


def artifact(root, start=1):
    bindings = {}
    for name in ("guidance", "decision"):
        path = root / f"{name}.md"
        path.write_text(f"Synthetic {name} artifact.\n")
        bindings[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    path = root / "profile.json"
    config = {"schema_version": "probability-v2-planning-profile-v1", "name": "dev-guidance-v3",
              "start_index": start, "max_assignments": 4096,
              "authoring_guidance_version": "synthetic-guidance-v3", "bindings": bindings}
    path.write_text(json.dumps(config))
    return path


def rows(result):
    return [json.loads(line) for shard in result["shards"] for line in Path(shard["output"]).read_text().splitlines()]


def signature(row):
    return (row["operation_id"], row["domain_id"], row["choice_count"], row["input_structure"]["id"],
            row["answerability_status"], row["_generator"]["derived_seed"])


def test_seeded_profile_repeats_across_shards_and_keeps_source_quotas(tmp_path):
    kwargs = inputs(tmp_path)
    profile = artifact(tmp_path)
    specs = [{"language_tag": "en", "source_kind": "exact_derived", "count": 32},
             {"language_tag": "en", "source_kind": "hard_label", "count": 32, "seed": 29}]
    first = planner.write_assignment_batch(specs, seed=17, planning_profile_path=profile, **kwargs)
    second_kwargs = inputs(tmp_path / "repeat")
    second = planner.write_assignment_batch(specs, seed=17, planning_profile_path=profile, **second_kwargs)
    assert [signature(row) for row in rows(first)] == [signature(row) for row in rows(second)]
    assert first["budget_before"]["source_targets"] == {
        "exact_derived": 300000, "explicit_stochastic_process": 300000, "empirical_frequency": 100000,
        "teacher_estimate": 100000, "hard_label": 200000,
    }
    assert [row["_planner"]["profile_assignment_number"] for row in rows(first)] == list(range(1, 65))
    assert all(row["_planner"]["authoring_guidance_version"] == "synthetic-guidance-v3" for row in rows(first))
    assert all(row["_planner"]["planning_profile"] == first["planning_profile"] for row in rows(first))
    later = planner.write_assignment_batch(specs[:1], seed=17, planning_profile_path=profile, **kwargs)
    assert rows(later)[0]["_planner"]["profile_assignment_number"] == 65


def test_default_seed_contract_is_unchanged(tmp_path):
    kwargs = inputs(tmp_path, wide=False)
    result = planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 4}], seed=17, **kwargs)
    assert [signature(row) for row in rows(result)] == [
        ("evidence_review", "archive", 3, "prose", "answerable", 12346574278444935172),
        ("sequence_reasoning", "textile", 2, "table", "answerable", 9108016424101870212),
        ("sequence_reasoning", "archive", 3, "prose", "answerable", 3898631356375708527),
        ("evidence_review", "textile", 2, "table", "answerable", 10835476322837576057),
    ]
    assert all("planning_profile" not in row["_planner"] for row in rows(result))


def test_band_mix_and_weighted_operations_are_conditioned_on_compatibility(tmp_path):
    kwargs = inputs(tmp_path)
    catalog = planner.assignments.load_catalog(kwargs["catalog_path"], max_choice_count=265)
    bands, low_operations = Counter(), Counter()
    for seed in range(10000):
        operation, _, count, structure, metadata = planner._pick_profile_task(catalog, Counter(), random.Random(seed))
        bands[metadata["choice_band"]] += 1
        if count <= 4:
            low_operations[operation["id"]] += 1
        elif count <= 16:
            assert operation["id"] in planner.MEDIUM_CARDINALITY_OPERATIONS
            assert structure["id"] in planner.MEDIUM_CARDINALITY_STRUCTURES
        else:
            assert operation["id"] in planner.HIGH_CARDINALITY_OPERATIONS
            assert structure["id"] in planner.HIGH_CARDINALITY_STRUCTURES
            assert "graph_kinship" in metadata["omitted_priority_operations"]
        assert metadata["complete_inventory_required"] == (count >= 65)
    for band, target in {"2-4": .70, "5-16": .20, "17-64": .08, "65-265": .02}.items():
        assert abs(bands[band] / 10000 - target) < .015
    assert 1.3 < low_operations["graph_kinship"] / low_operations["evidence_review"] < 1.7


def test_every_count_from_2_through_265_is_reachable_without_truncation(tmp_path):
    kwargs = inputs(tmp_path)
    catalog = planner.assignments.load_catalog(kwargs["catalog_path"], max_choice_count=265)
    class ForcedCount(random.Random):
        def __init__(self, target):
            super().__init__(17)
            self.target = target
        def choices(self, population, weights=None, *, cum_weights=None, k=1):
            if population[0] == "2-4":
                return [next(name for name, (low, high, _) in planner.DEV_PROFILE_BANDS.items() if low <= self.target <= high)]
            return super().choices(population, weights, cum_weights=cum_weights, k=k)
        def randint(self, low, high):
            assert low <= self.target <= high
            return self.target
    for target in range(2, 266):
        assert planner._pick_profile_task(catalog, Counter(), ForcedCount(target))[2] == target


def test_unsupported_wide_bands_are_explicitly_omitted(tmp_path):
    kwargs = inputs(tmp_path, wide=False)
    result = planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 8}],
                                            planning_profile_path=artifact(tmp_path), **kwargs)
    for row in rows(result):
        assert 2 <= row["choice_count"] <= 4
        assert row["_planner"]["profile_sampling"]["omitted_bands"] == ["5-16", "17-64", "65-265"]
        assert row["_planner"]["profile_sampling"]["effective_band_weights"] == {"2-4": 1.0}


def test_window_crossing_preflights_entire_batch_before_writes(tmp_path):
    kwargs = inputs(tmp_path)
    profile = artifact(tmp_path)
    with pytest.raises(planner.ContractError, match="4,096-assignment"):
        planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 4097}],
                                       planning_profile_path=profile, **kwargs)
    assert not list(kwargs["assignment_dir"].rglob("shard-*.jsonl"))
    loaded = planner.load_planning_profile(profile)
    catalog = planner.assignments.load_catalog(kwargs["catalog_path"], max_choice_count=265)
    history = planner.PlanningHistory(catalog)
    history.max_index = 4095
    history.planning_profiles[planner.DEV_PROFILE_NAME] = loaded
    planner._profile_preflight(loaded, history, 4096, 1)
    with pytest.raises(planner.ContractError, match="4,096-assignment"):
        planner._profile_preflight(loaded, history, 4097, 1)


def test_profile_cannot_reanchor_or_change_bindings_after_first_plan(tmp_path):
    kwargs = inputs(tmp_path)
    path = artifact(tmp_path)
    spec = [{"language_tag": "en", "source_kind": "exact_derived", "count": 2}]
    first = planner.write_assignment_batch(spec, planning_profile_path=path, **kwargs)
    preserved = Path(first["shards"][0]["output"]).read_bytes()
    config = json.loads(path.read_text())
    config["start_index"] = 3
    path.write_text(json.dumps(config))
    with pytest.raises(planner.ContractError, match="immutable saved plans"):
        planner.write_assignment_batch(spec, planning_profile_path=path, **kwargs)
    assert Path(first["shards"][0]["output"]).read_bytes() == preserved
    assert len(list(kwargs["assignment_dir"].rglob("shard-*.jsonl"))) == 1


def test_bound_file_changes_and_resume_profile_use_fail_closed(tmp_path):
    kwargs = inputs(tmp_path)
    path = artifact(tmp_path)
    first = planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 2}],
                                            planning_profile_path=path, **kwargs)
    with pytest.raises(planner.ContractError, match="only to new plans"):
        planner.write_assignment_shard(count=2, source_kind="exact_derived", language_tag="en", shard_index=1,
                                       resume=True, planning_profile_path=path, **kwargs)
    with pytest.raises(planner.ContractError, match="cannot be extended"):
        planner.write_assignment_shard(count=3, source_kind="exact_derived", language_tag="en", shard_index=1,
                                       resume=True, **kwargs)
    (tmp_path / "decision.md").write_text("Changed synthetic decision")
    with pytest.raises(planner.ContractError, match="binding hash mismatch"):
        planner.load_planning_profile(path)
    assert len(first["shards"]) == 1


def test_changed_binding_before_commit_leaves_no_plan(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    path = artifact(tmp_path)
    original = planner.create_plan_rows
    def changed_after_allocation(**arguments):
        result = original(**arguments)
        (tmp_path / "guidance.md").write_text("Changed synthetic guidance")
        return result
    monkeypatch.setattr(planner, "create_plan_rows", changed_after_allocation)
    with pytest.raises(planner.ContractError, match="binding hash mismatch"):
        planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 2}],
                                       planning_profile_path=path, **kwargs)
    assert not list(kwargs["assignment_dir"].rglob("shard-*.jsonl"))


def test_cli_passes_pinned_profile_to_new_batch(tmp_path, capsys):
    kwargs = inputs(tmp_path)
    path = artifact(tmp_path)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([{"language_tag": "en", "source_kind": "exact_derived", "count": 2}]))
    argv = ["plan-batch", "--spec", str(spec), "--planning-profile", str(path), "--seed", "17"]
    for flag, key in [("--catalog", "catalog_path"), ("--word-pool", "word_pool_path"),
                      ("--assignment-dir", "assignment_dir"), ("--question-dir", "question_dir"),
                      ("--legacy-dir", "legacy_dir"), ("--pilot-file", "pilot_file")]:
        argv.extend([flag, str(kwargs[key])])
    assert planner._cli(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["planning_profile"]["name"] == "dev-guidance-v3"
    assert len(rows(result)) == 2


def pilot_inputs(root, *, compatible=True):
    kwargs = inputs(root, wide=compatible)
    catalog = json.loads(kwargs["catalog_path"].read_text())
    if compatible:
        template = catalog["operations"][0]
        catalog["operations"] = [{**template, "id": name, "name": name} for name in (
            "taxonomy_outliers", "policy_exceptions", "game_sport_rules", "physical_mechanisms",
            "argument_evidence", "experiment_design", "graph_kinship",
        )]
        catalog["input_structures"] = [{"id": name} for name in (
            "table_or_grid", "rule_excerpt", "claim_evidence_bundle", "measurement_sheet",
        )]
    catalog["languages"] = [{"tag": tag, "name": tag, "family": "test", "script": "Latin"}
                            for tag in ("en", "de", "fr", "es", "pt", "it", "nl", "id")]
    kwargs["catalog_path"].write_text(json.dumps(catalog))
    baseline = planner.write_assignment_shard(count=1, source_kind="hard_label", language_tag="en",
                                              start_index=128512, **kwargs)
    profile = artifact(root)
    config = json.loads(profile.read_text())
    config.update(name=planner.MULTIPLE_SUPPORT_PROFILE, start_index=128513, max_assignments=1024,
                  authoring_guidance_version="synthetic-support-pilot-v1")
    profile.write_text(json.dumps(config))
    return kwargs, profile, Path(baseline["output"])


def test_pilot_seeded_selection_is_bounded_compatible_and_keeps_old_bytes(tmp_path):
    kwargs, profile, baseline = pilot_inputs(tmp_path)
    preserved = baseline.read_bytes()
    sources = ["exact_derived"] * 3 + ["explicit_stochastic_process"] * 3 + ["empirical_frequency", "teacher_estimate"]
    specs = [{"language_tag": language, "source_kind": source, "count": 128, "seed": 2026093000 + index}
             for index, (language, source) in enumerate(zip(("en", "de", "fr", "es", "pt", "it", "nl", "id"), sources))]
    first = planner.write_assignment_batch(specs, planning_profile_path=profile, **kwargs)
    repeat_kwargs, repeat_profile, _ = pilot_inputs(tmp_path / "repeat")
    second = planner.write_assignment_batch(specs, planning_profile_path=repeat_profile, **repeat_kwargs)
    first_rows = rows(first)
    assert [signature(row) for row in first_rows] == [signature(row) for row in rows(second)]
    assert baseline.read_bytes() == preserved
    assert [row["_generator"]["global_index"] for row in first_rows] == list(range(128513, 129537))
    answerable = [row for row in first_rows if row["answerability_status"] == "answerable"]
    marked = [row for row in first_rows if row["_planner"].get("target_shape_requirement")]
    assert .20 < len(marked) / len(answerable) < .30
    assert first["target_shape_planning"]["planned_marked"] == len(marked)
    assert first["budget_before"]["source_targets"] == planner.DEFAULT_SOURCE_TARGETS
    for row in marked:
        assert row["answerability_status"] == "answerable" and row["unanswerable_reason"] is None
        assert row["choice_count"] in (3, 4)
        assert row["question_form"]["question_kind"] == "outcome_distribution"
        compatibility = planner.DISTRIBUTION_TASK_FORMS[row["question_form"]["id"]]
        assert row["operation_id"] in compatibility["operations"]
        assert row["input_structure"]["id"] in compatibility["input_structures"]
        assert row["distribution_source_kind"] in compatibility["source_kinds"]
    with pytest.raises(planner.ContractError, match="1,024-assignment"):
        planner.write_assignment_batch(specs[:1], planning_profile_path=profile, **kwargs)


def test_pilot_reports_unsupported_omission_without_fabricating_a_form(tmp_path):
    kwargs, profile, _ = pilot_inputs(tmp_path, compatible=False)
    catalog = planner.assignments.load_catalog(kwargs["catalog_path"], max_choice_count=265)
    before = planner.probability_task_compatibility(catalog, ["exact_derived", "hard_label"])
    assert before["exact_derived"]["omission_reason"] == "unsupported_distribution_task"
    assert before["hard_label"]["omission_reason"] == "hard_label"
    result = planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 128}],
                                            seed=17, planning_profile_path=profile, **kwargs)
    assert result["target_shape_planning"]["planned_marked"] == 0
    assert result["target_shape_planning"]["selection_status_counts"]["unsupported_distribution_task"] > 0
    assert all("target_shape_requirement" not in row["_planner"] for row in rows(result))
    assert all(isinstance(row["question_form"], str) for row in rows(result))


def test_pilot_never_marks_hard_labels_or_unknown_realized_draws(tmp_path):
    kwargs, profile, _ = pilot_inputs(tmp_path)
    result = planner.write_assignment_batch([
        {"language_tag": "en", "source_kind": "hard_label", "count": 64},
        {"language_tag": "en", "source_kind": "explicit_stochastic_process", "count": 256},
    ], seed=17, planning_profile_path=profile, **kwargs)
    hard = [row for row in rows(result) if row["distribution_source_kind"] == "hard_label"]
    unknown = [row for row in rows(result) if row["unanswerable_reason"] == "realized_random_outcome_unknown"]
    assert unknown
    assert all(row["_planner"]["target_shape_selection"]["status"] == "hard_label" for row in hard)
    assert all("target_shape_requirement" not in row["_planner"] for row in hard + unknown)


def test_pilot_invalid_window_preflights_before_any_new_plan(tmp_path):
    kwargs, profile, baseline = pilot_inputs(tmp_path)
    with pytest.raises(planner.ContractError, match="1,024-assignment"):
        planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 1025}],
                                       planning_profile_path=profile, **kwargs)
    assert list(kwargs["assignment_dir"].rglob("shard-*.jsonl")) == [baseline]
    config = json.loads(profile.read_text())
    config["start_index"] = 128514
    profile.write_text(json.dumps(config))
    with pytest.raises(planner.ContractError, match="must begin at index 128513"):
        planner.load_planning_profile(profile)


def follow_on_inputs(root):
    kwargs, pilot_profile, baseline = pilot_inputs(root)
    old_pilot = planner.write_assignment_shard(
        count=1, source_kind="exact_derived", language_tag="en", start_index=129536,
        planning_profile_path=pilot_profile, **kwargs,
    )
    directory = root / "follow-on-artifacts"
    directory.mkdir()
    profile = artifact(directory)
    config = json.loads(profile.read_text())
    config.update(name=planner.MULTIPLE_SUPPORT_FOLLOW_ON_PREFIX + "129537", start_index=129537, max_assignments=4096,
                  authoring_guidance_version="synthetic-support-follow-on")
    profile.write_text(json.dumps(config))
    return kwargs, profile, [baseline, Path(old_pilot["output"])], pilot_profile


def test_follow_on_4096_window_is_deterministic_and_preserves_original_pilot(tmp_path):
    kwargs, profile, old_plans, pilot_profile = follow_on_inputs(tmp_path)
    saved = {path: path.read_bytes() for path in old_plans + [pilot_profile]}
    specs = [{"language_tag": "en", "source_kind": source, "count": 128, "seed": 37 + index}
             for index, source in enumerate(["exact_derived", "explicit_stochastic_process",
                                            "empirical_frequency", "teacher_estimate"] * 8)]
    result = planner.write_assignment_batch(specs, planning_profile_path=profile, **kwargs)
    repeated_kwargs, repeated_profile, _, _ = follow_on_inputs(tmp_path / "repeat")
    repeated = planner.write_assignment_batch(specs, planning_profile_path=repeated_profile, **repeated_kwargs)
    planned = rows(result)
    assert [signature(row) for row in planned] == [signature(row) for row in rows(repeated)]
    assert [row["_generator"]["global_index"] for row in planned] == list(range(129537, 133633))
    assert [row["_planner"]["profile_assignment_number"] for row in planned] == list(range(1, 4097))
    assert result["target_shape_planning"]["planned_marked"] > 0
    assert sum(result["target_shape_planning"]["selection_status_counts"].values()) == 4096
    assert result["planning_profile"]["target_shape_requirement"] == "multiple_substantive_support-v1"
    assert all(path.read_bytes() == raw for path, raw in saved.items())
    with pytest.raises(planner.ContractError, match="4,096-assignment"):
        planner.write_assignment_batch(specs[:1], planning_profile_path=profile, **kwargs)
    # A distinct declared window starts exactly after the previous reservations.
    directory = tmp_path / "next-window-artifacts"
    directory.mkdir()
    next_profile = artifact(directory)
    config = json.loads(next_profile.read_text())
    config.update(name=planner.MULTIPLE_SUPPORT_FOLLOW_ON_PREFIX + "133633", start_index=133633,
                  max_assignments=4096, authoring_guidance_version="synthetic-next-window")
    next_profile.write_text(json.dumps(config))
    next_result = planner.write_assignment_batch(
        [{"language_tag": "en", "source_kind": "exact_derived", "count": 2}],
        planning_profile_path=next_profile, **kwargs,
    )
    assert [row["_generator"]["global_index"] for row in rows(next_result)] == [133633, 133634]
    assert all(path.read_bytes() == raw for path, raw in saved.items())


@pytest.mark.parametrize("name,start,limit", [
    ("multiple-substantive-support-v1-pilot", 128513, 1024),
    ("multiple-substantive-support-v1-follow-on-129537", 129537, 4096),
])
def test_each_support_profile_requires_its_exact_start_limit_and_name(tmp_path, name, start, limit):
    path = artifact(tmp_path)
    config = json.loads(path.read_text())
    config.update(name=name, start_index=start, max_assignments=limit)
    path.write_text(json.dumps(config))
    assert planner.load_planning_profile(path)["max_assignments"] == limit
    for change in ({"start_index": start + 1}, {"max_assignments": limit + 1},
                   {"name": "multiple-substantive-support-v1-unlimited"},
                   {"name": "multiple-substantive-support-v1-follow-on"},
                   {"name": "multiple-substantive-support-v1-follow-on-0129537"},
                   {"name": "multiple-substantive-support-v1-follow-on-128513"}):
        path.write_text(json.dumps({**config, **change}))
        with pytest.raises(planner.ContractError):
            planner.load_planning_profile(path)


def test_follow_on_window_overrun_and_reanchor_leave_all_old_plans_untouched(tmp_path):
    kwargs, profile, old_plans, _ = follow_on_inputs(tmp_path)
    saved = {path: path.read_bytes() for path in old_plans}
    with pytest.raises(planner.ContractError, match="4,096-assignment"):
        planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 4097}],
                                       planning_profile_path=profile, **kwargs)
    assert set(kwargs["assignment_dir"].rglob("shard-*.jsonl")) == set(old_plans)
    result = planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 2}],
                                            planning_profile_path=profile, **kwargs)
    config = json.loads(profile.read_text())
    config["authoring_guidance_version"] = "changed-follow-on-version"
    profile.write_text(json.dumps(config))
    with pytest.raises(planner.ContractError, match="immutable saved plans"):
        planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 2}],
                                       planning_profile_path=profile, **kwargs)
    assert all(path.read_bytes() == raw for path, raw in saved.items())
    assert len(rows(result)) == 2


def test_distinct_follow_on_names_cannot_overlap_a_partially_allocated_window(tmp_path):
    kwargs, profile, _, _ = follow_on_inputs(tmp_path)
    first = planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 2}],
                                           planning_profile_path=profile, **kwargs)
    directory = tmp_path / "overlap-artifacts"
    directory.mkdir()
    overlap = artifact(directory)
    config = json.loads(overlap.read_text())
    config.update(name=planner.MULTIPLE_SUPPORT_FOLLOW_ON_PREFIX + "129539", start_index=129539,
                  max_assignments=4096)
    overlap.write_text(json.dumps(config))
    with pytest.raises(planner.ContractError, match="windows overlap"):
        planner.write_assignment_batch([{"language_tag": "en", "source_kind": "exact_derived", "count": 2}],
                                       planning_profile_path=overlap, **kwargs)
    assert len(rows(first)) == 2
