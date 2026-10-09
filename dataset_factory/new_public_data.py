"""Pinned, train-only public acquisition; never fetch or preview test releases."""

from __future__ import annotations

import csv
import hashlib
import io
import math
import json
import shutil
import urllib.request
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from dataset_factory import probability_v2

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_ROOT = ROOT / "data/newtraining/public/2026-10-08"
BANKING_REVISION = "57ec275d8078af65b7731c2a98be812d844a6d6b"
SOURCES = {
    "typed-decisions": (
        "LocalLLaMA/typed-decisions",
        "e039ebffcc280174dd354227424fb2b249f191de",
        "all/train-00000-of-00001.parquet",
        "apache-2.0",
    ),
    "sciq": (
        "allenai/sciq",
        "2c94ad3e1aafab77146f384e23536f97a4849815",
        "data/train-00000-of-00001.parquet",
        "cc-by-nc-3.0",
    ),
}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def envelope(
    source: str,
    source_row: str,
    context: str,
    question: str,
    options: list[str],
    metadata: dict,
) -> dict:
    return {
        "id": f"public:{source}:{source_row}",
        "problem_group_id": f"public:{source}:{source_row.split(':question:')[0]}",
        "language": "en",
        "split": "train",
        "context": context,
        "question": question,
        "options": options,
        "choice_ids": [f"label:{i}" for i in range(len(options))],
        "difficulty": "medium",
        "topic": source,
        "operation_id": source,
        "provenance": f"Public source-native training split: {source}; see public_metadata",
        "explanation": "Source training supervision retained without semantic relabeling.",
        "public_metadata": metadata,
    }


def convert_typed(records: list[dict], metadata: dict) -> list[dict]:
    rows = []
    for record in records:
        if record["split"] != "train":
            raise ValueError("public_nontraining_record")
        questions, gold = json.loads(record["questions"]), json.loads(record["gold"])
        for task, spec in questions.items():
            probabilities = gold[task]["probabilities"]
            labels = list(probabilities)
            criteria = spec.get("criteria")
            options = [
                f"{label}: {criteria[label]}"
                if isinstance(criteria, dict)
                else f"{label}: {criteria[int(label)]}"
                if isinstance(criteria, list)
                else label
                for label in labels
            ]
            row = envelope(
                "typed-decisions",
                f"{record['id']}:question:{task}",
                record["state"],
                spec["instructions"],
                options,
                {
                    **metadata,
                    "source_row_id": record["id"],
                    "workflow": record["workflow"],
                    "decision_kind": task,
                    "label_agreement": json.loads(record["label_agreement"])[task],
                    "gold_type": gold[task]["type"],
                },
            )
            row["operation_id"] = f"typed:{record['workflow']}:{task}"
            row.update(
                schema_version="probability-v2",
                supervision_kind="soft_distribution",
                distribution_source_kind="teacher_estimate",
                target_distribution=[probabilities[label] for label in labels],
                distribution_evidence={
                    "kind": "teacher_estimate",
                    "choice_ids": row["choice_ids"],
                    "model_id": "source-published-teacher-consensus",
                    "model_version": metadata["revision"],
                    "estimation_method": "published_label_probabilities_preserved; see pinned raw source for annotator details",
                },
            )
            rows.append(row)
    return rows


def convert_sciq(records: list[dict], metadata: dict) -> list[dict]:
    rows = []
    for index, record in enumerate(records):
        if not record["support"].strip():
            continue
        options = [
            record["correct_answer"],
            *(record[f"distractor{i}"] for i in range(1, 4)),
        ]
        row = envelope(
            "sciq",
            str(index),
            record["support"],
            record["question"],
            options,
            {**metadata, "source_row_id": index},
        )
        row.update(
            schema_version="probability-v2",
            supervision_kind="hard_label",
            answer_index=0,
        )
        rows.append(row)
    return rows


def convert_banking(records: list[dict], metadata: dict) -> list[dict]:
    categories = sorted({row["category"] for row in records})
    if len(categories) != 77:
        raise ValueError("banking_category_count_not_77")
    options = [category.replace("_", " ") for category in categories]
    rows = []
    for index, record in enumerate(records):
        row = envelope(
            "banking77",
            str(index),
            record["text"],
            "Which of the listed banking intents describes this customer message?",
            options,
            {**metadata, "source_row_id": index, "category": record["category"]},
        )
        row.update(
            schema_version="probability-v2",
            supervision_kind="hard_label",
            answer_index=categories.index(record["category"]),
        )
        rows.append(row)
    return rows


def validate_public_row(row: dict) -> list[str]:
    errors = [
        error
        for error in probability_v2.validate_v2_row(row)
        if error
        not in {"provenance_does_not_match_required_label", "split_is_not_train"}
    ]
    metadata = row.get("public_metadata", {})
    if metadata.get("split") != "train" or not str(
        row.get("provenance", "")
    ).startswith("Public source-native training split:"):
        errors.append("public_train_provenance_invalid")
    if metadata.get("repository") == "LocalLLaMA/typed-decisions":
        # The published six-decimal probabilities carry rounding, not missing mass.
        total = sum(row.get("target_distribution", []))
        if math.isfinite(total) and abs(total - 1.0) <= 2e-6:
            errors = [
                error
                for error in errors
                if error != "target_distribution_does_not_sum_to_one"
            ]
    return errors


def acquire(output: Path = PUBLIC_ROOT) -> dict:
    import pyarrow.parquet as pq

    if (output / "manifest.json").exists():
        raise ValueError("public_acquisition_directory_is_immutable")
    output.mkdir(parents=True, exist_ok=True)
    sources, totals = [], {}
    for source, (repo, revision, filename, license_id) in SOURCES.items():
        cached = Path(
            hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
        )
        raw = output / f"{source}-train.parquet"
        shutil.copyfile(cached, raw)
        metadata = {
            "license_url": "https://www.apache.org/licenses/LICENSE-2.0"
            if license_id == "apache-2.0"
            else "https://creativecommons.org/licenses/by-nc/3.0/",
            "attribution": repo,
            "designation": "training_source_only; never an independent benchmark",
            "repository": repo,
            "revision": revision,
            "source_file": filename,
            "split": "train",
            "license": license_id,
            "sha256": digest(raw.read_bytes()),
            "source_url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
        }
        records = pq.read_table(raw).to_pylist()
        rows = (
            convert_typed(records, metadata)
            if source == "typed-decisions"
            else convert_sciq(records, metadata)
        )
        sources.append(
            {
                "source_id": source,
                **metadata,
                "raw_rows": len(records),
                "converted_rows": len(rows),
            }
        )
        write_rows(output / f"{source}.jsonl", rows)
        totals[source] = len(rows)
    revision = BANKING_REVISION
    url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/banking_data/train.csv"
    raw = urllib.request.urlopen(url, timeout=30).read()
    (output / "banking77-train.csv").write_bytes(raw)
    metadata = {
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "attribution": "PolyAI-LDN/task-specific-datasets; Banking77",
        "designation": "training_source_only",
        "repository": "PolyAI-LDN/task-specific-datasets",
        "revision": revision,
        "source_file": "banking_data/train.csv",
        "split": "train",
        "license": "cc-by-4.0",
        "license_metadata_repository": "PolyAI/banking77",
        "license_metadata_revision": "90d4e2ee5521c04fc1488f065b8b083658768c57",
        "sha256": digest(raw),
        "source_url": url,
    }
    records = list(csv.DictReader(io.StringIO(raw.decode())))
    rows = convert_banking(records, metadata)
    write_rows(output / "banking77.jsonl", rows)
    sources.append(
        {
            "source_id": "banking77",
            **metadata,
            "raw_rows": len(records),
            "converted_rows": len(rows),
        }
    )
    totals["banking77"] = len(rows)
    invalid = 0
    exclusions = {}
    for filename in output.glob("*.jsonl"):
        accepted, rejected = [], 0
        for line in filename.open():
            row = json.loads(line)
            errors = validate_public_row(row)
            if errors:
                rejected += 1
            else:
                accepted.append(row)
        write_rows(filename, accepted)
        exclusions[filename.stem] = rejected
        totals[filename.stem] = len(accepted)
        invalid += sum(bool(validate_public_row(row)) for row in accepted)
    for source_record in sources:
        source_record["admitted_converted_rows"] = totals[source_record["source_id"]]
        source_record["excluded_invalid_rows"] = exclusions[source_record["source_id"]]
        source_record["converted_sha256"] = digest(
            (output / f"{source_record['source_id']}.jsonl").read_bytes()
        )
        repo = source_record.get(
            "license_metadata_repository", source_record["repository"]
        )
        revision = source_record.get(
            "license_metadata_revision", source_record["revision"]
        )
        card = HfApi().dataset_info(repo, revision=revision).card_data.to_dict()
        evidence = {
            "repository": repo,
            "revision": revision,
            "license": card.get("license"),
            "language": card.get("language"),
            "task_categories": card.get("task_categories"),
        }
        encoded = (json.dumps(evidence, sort_keys=True) + "\n").encode()
        evidence_path = output / f"{source_record['source_id']}-license-metadata.json"
        evidence_path.write_bytes(encoded)
        source_record["license_evidence"] = {
            "url": f"https://huggingface.co/datasets/{repo}/tree/{revision}",
            "metadata_sha256": digest(encoded),
            "path": evidence_path.name,
        }
    report = {
        "schema_version": "public-train-acquisition-v1",
        "sources": sources,
        "counts": totals,
        "invalid_rows": invalid,
        "excluded_invalid_rows": exclusions,
        "policy": "Only source-native train artifacts downloaded. Raw supervision retained. SciQ is noncommercial; private research use only.",
        "checkpoint_restriction": "Contains SciQ CC-BY-NC-3.0 training data: checkpoint redistribution/commercial use needs separate rights review; current run is private noncommercial research.",
        "rejected_candidates": {
            "fancyzhx/ag_news": "unknown license",
            "MMLU and derivatives": "evaluation-only lineage",
        },
    }
    if invalid:
        raise ValueError("public_conversion_invalid_rows")
    (output / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


if __name__ == "__main__":
    report = acquire()
    print(
        json.dumps({"counts": report["counts"], "invalid_rows": report["invalid_rows"]})
    )
