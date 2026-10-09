#!/usr/bin/env python3
"""Export only explicit frozen training/development allowlists to Hugging Face.

The split sidecar is authoritative: historical rows retain their source's
`split` field even when the snapshot assigned them to a held-out group.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = {
    "warmstart": "2026-10-01-v1",
    "continuation": "2026-10-08-v4",
}
HASHES = {
    "2026-10-01-v1": "eac9afe4f6ec25e0ea27237fcd3b59e656e6cab426d3642082aae5c99e429f57",
    "2026-10-08-v4": "6413df891b6ee10073c8e49f4c3c0b19375c495ff595ab7889a4386e7bb427fd",
}
DOCUMENT_ROOTS = [
    ROOT / "data/document_breadth/2026-10-03/frozen_documents",
    ROOT / "data/document_luna_max/2026-10-02/documents",
]
LICENSE_URLS = {
    "CC-BY-4.0": "https://creativecommons.org/licenses/by/4.0/",
    "CC-BY-3.0": "https://creativecommons.org/licenses/by/3.0/",
    "CC-BY-2.0": "https://creativecommons.org/licenses/by/2.0/",
    "CC-BY-SA-4.0": "https://creativecommons.org/licenses/by-sa/4.0/",
    "CC-BY-NC-3.0": "https://creativecommons.org/licenses/by-nc/3.0/",
    "CC0-1.0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "OGL-3.0": "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/",
    "Apache-2.0": "https://www.apache.org/licenses/LICENSE-2.0",
}
PUBLIC_LICENSES = {
    "banking77": "CC-BY-4.0", "sciq": "CC-BY-NC-3.0",
    "typed-decisions": "Apache-2.0",
}
SECRET_PATTERN = re.compile(
    r"sk-(?:proj-)?[A-Za-z0-9_-]{24,}|gh[pousr]_[A-Za-z0-9]{30,}|"
    r"AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----"
)
EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}")
SCHEMA = pa.schema([
    ("id", pa.string()), ("split", pa.string()), ("snapshot", pa.string()),
    ("source", pa.string()), ("source_license", pa.string()),
    ("contribution_license", pa.string()), ("language", pa.string()),
    ("context", pa.string()), ("question", pa.string()),
    ("options", pa.list_(pa.string())), ("choice_ids", pa.list_(pa.string())),
    ("target_kind", pa.string()), ("answer_index", pa.int64()),
    ("target_distribution", pa.list_(pa.float64())),
    ("abstention_choice_id", pa.string()), ("answerability_status", pa.string()),
    ("explanation", pa.string()), ("problem_group_id", pa.string()),
    ("snapshot_group_id", pa.string()), ("source_row_sha256", pa.string()),
    ("provenance_json", pa.string()),
])


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def input_digest(row, fields) -> str:
    value = {k: row.get(k) for k in fields}
    if value.get("target_distribution") is not None:
        value["target_distribution"] = [float(p) for p in value["target_distribution"]]
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def public_metadata(value):
    """Retain evidence hashes while omitting owner-local file references."""
    if isinstance(value, dict):
        return {
            k: public_metadata(v) for k, v in value.items()
            if not (k.endswith("_file") or k in {"path", "source_json", "raw_file", "output", "log_path"})
        }
    if isinstance(value, list):
        return [public_metadata(v) for v in value]
    if isinstance(value, str) and value.startswith(("/home/", "/workspace/", "runs/", "data/")):
        return None
    return value


def document_notice(metadata, context: str):
    doc_id = metadata["document_id"]
    paths = [base / f"{doc_id}.json" for base in DOCUMENT_ROOTS]
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise ValueError("A document attribution record is missing; refusing export")
    doc = json.loads(path.read_text())
    expected = hashlib.sha256(context.encode()).hexdigest()
    if expected != metadata["content_sha256"] or doc["content_sha256"] != expected:
        raise ValueError("Frozen document context hash mismatch; refusing export")
    keys = ["id", "title", "url", "publisher", "attribution", "license", "revision",
            "retrieved_at", "content_sha256", "rights_evidence", "extraction", "changes"]
    notice = {k: public_metadata(doc[k]) for k in keys if k in doc}
    components = doc.get("component_documents", [])
    if components:
        if any(c["license"]["id"] != doc["license"]["id"] for c in components):
            raise ValueError("Composite document license conflict; refusing export")
        notice["component_documents"] = [
            {k: public_metadata(c[k]) for k in keys if k in c} for c in components
        ]
    if not notice.get("attribution") or not notice.get("rights_evidence"):
        raise ValueError("Document rights evidence incomplete; refusing export")
    return notice


def package(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    if (output / "manifest.json").exists():
        raise ValueError("Output already packaged; choose a fresh output directory")
    for name in ["README.md", "LICENSE", "NOTICE.md", "LICENSES/Apache-2.0.txt"]:
        source = ROOT / "release/dataset" / name
        target = output / name
        if source != target:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    manifests = {}; notices = {}; inventory = []; counts = {}; privacy = {}
    all_groups = {}; warmstart_ids = {}; exposed_ids = set()
    for config, version in SNAPSHOTS.items():
        snapshot = ROOT / "data/training_snapshots" / version
        manifest_path = snapshot / "manifest.json"
        if digest(manifest_path) != HASHES[version]:
            raise ValueError("Snapshot manifest differs from frozen release contract")
        manifest = json.loads(manifest_path.read_text())
        for name in ["rows.jsonl", "splits.jsonl"]:
            if digest(snapshot / name) != manifest["artifacts"][name]["sha256"]:
                raise ValueError("Snapshot artifact checksum mismatch")
        splits = {}
        for line in (snapshot / "splits.jsonl").open():
            assignment = json.loads(line)
            if assignment["id"] in splits:
                raise ValueError("Duplicate split identity")
            splits[assignment["id"]] = assignment
            group = assignment["snapshot_group_id"]
            key = (version, group)
            if all_groups.setdefault(key, assignment["split"]) != assignment["split"]:
                raise ValueError("A group crosses snapshot splits")
        buffers = {"train": [], "dev": []}; source_counts = Counter(); lic_counts = Counter()
        email_counts = Counter(); seen = set(); excluded = Counter()
        for raw_line in (snapshot / "rows.jsonl").open("rb"):
            row = json.loads(raw_line)
            assignment = splits[row["id"]]
            seen.add(row["id"])
            split = assignment["split"]
            if config == "warmstart":
                warmstart_ids[row["id"]] = split
            elif row["id"] in warmstart_ids and split != warmstart_ids[row["id"]]:
                raise ValueError("Historical split reassignment found")
            if split not in buffers:
                excluded[split] += 1
                continue
            source = row.get("_new_dataset", {}).get("source_id", "original_generated")
            if source not in {"original_generated", "original_replay", "documents", "games", *PUBLIC_LICENSES}:
                raise ValueError("Unexpected source outside release allowlist")
            content = "\n".join([row["context"], row["question"], *row["options"], row.get("explanation", "")])
            if SECRET_PATTERN.search(content) or "/home/nicolas/" in content:
                raise ValueError("Potential credential or private workspace input; refusing export")
            if EMAIL_PATTERN.search(content):
                email_counts[source] += 1
            source_license = PUBLIC_LICENSES.get(source, "CC-BY-4.0")
            provenance = {
                "snapshot_row_source_file": row["_training_snapshot"]["source_file"],
                "snapshot_row_source_line": row["_training_snapshot"]["source_line"],
                "source_kind": row["_training_snapshot"].get("source_kind"),
                "changes": "Exported a common schema; authoritative snapshot split overrides source row split; input text, option order and raw targets unchanged.",
            }
            if source == "documents":
                metadata = row["document_metadata"]
                source_license = metadata["license"]["id"]
                notice = document_notice(metadata, row["context"])
                notices[metadata["document_id"]] = notice
                provenance["document"] = notice
                provenance["document"]["synthetic_question_contributor"] = "Nicolas Mahn; AI-assisted authoring"
            if source in PUBLIC_LICENSES:
                provenance["public_source"] = public_metadata(row["public_metadata"])
                provenance["canonical_choice_permutation"] = row.get("_new_dataset", {}).get("canonical_choice_permutation")
            if source == "games":
                provenance["game_metadata"] = public_metadata(row.get("game_metadata", {}))
            if source_license not in LICENSE_URLS:
                raise ValueError("Unknown source license; refusing export")
            contribution_license = "CC-BY-SA-4.0" if source_license == "CC-BY-SA-4.0" else "CC-BY-4.0"
            if source == "sciq":
                contribution_license = "CC-BY-NC-3.0"
            # The upstream Apache grant is retained for the converted source;
            # the project grant covers only original wrapper contributions.
            target = row.get("target_distribution")
            answer = row.get("answer_index")
            if target is None and not isinstance(answer, int):
                raise ValueError("Missing target")
            buffers[split].append({
                "id": row["id"], "split": split, "snapshot": version,
                "source": source, "source_license": source_license,
                "contribution_license": contribution_license, "language": row["language"],
                "context": row["context"], "question": row["question"],
                "options": row["options"], "choice_ids": row["choice_ids"],
                "target_kind": "soft_full_distribution" if target is not None else "hard_onehot",
                "answer_index": answer, "target_distribution": target,
                "abstention_choice_id": row.get("abstention_choice_id"),
                "answerability_status": row.get("answerability_status"),
                "explanation": row.get("explanation"),
                "problem_group_id": row["problem_group_id"],
                "snapshot_group_id": assignment["snapshot_group_id"],
                "source_row_sha256": hashlib.sha256(raw_line.rstrip(b"\r\n")).hexdigest(),
                "provenance_json": canonical(provenance),
            })
            source_counts[(split, source)] += 1; lic_counts[(split, source_license)] += 1
            exposed_ids.add(row["id"])
        if len(seen) != len(splits):
            raise ValueError("Snapshot rows and registry differ")
        config_dir = output / "data" / config
        config_dir.mkdir(parents=True, exist_ok=True)
        for split, rows in buffers.items():
            name = "validation" if split == "dev" else "train"
            path = config_dir / f"{name}.parquet"
            pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")
            inventory.append({"path": str(path.relative_to(output)), "rows": len(rows), "sha256": digest(path), "bytes": path.stat().st_size})
        counts[config] = {
            "snapshot": version, "exported": {s: len(r) for s, r in buffers.items()},
            "sealed_not_published": dict(excluded),
            "by_source": {f"{s}/{k}": v for (s, k), v in sorted(source_counts.items())},
            "by_license": {f"{s}/{k}": v for (s, k), v in sorted(lic_counts.items())},
        }
        privacy[config] = {"potential_credentials": 0, "private_workspace_inputs": 0,
                           "rows_with_email_patterns": dict(email_counts)}
        manifests[version] = {
            "manifest_sha256": HASHES[version],
            "artifacts": manifest["artifacts"], "split_counts": manifest["rows"]["by_split"],
            "source_files": manifest["source_files"],
        }
    notices_path = output / "document-attribution.jsonl"
    notices_path.write_text("".join(canonical(notices[k]) + "\n" for k in sorted(notices)))
    inventory.append({"path": notices_path.name, "records": len(notices), "sha256": digest(notices_path), "bytes": notices_path.stat().st_size})
    public = json.loads((ROOT / "data/newtraining/public/2026-10-08/manifest.json").read_text())
    write_json(output / "public-source-manifest.json", public_metadata(public))
    manifest = {
        "schema_version": "horscht-hf-dataset-release-v1", "release_date": "2026-10-09",
        "creator": "Nicolas Mahn", "license": "source-specific mixed collection; see LICENSE and NOTICE.md",
        "configs": counts, "frozen_inputs": manifests, "artifacts": inventory,
        "privacy_scan": privacy, "unique_published_ids": len(exposed_ids),
        "input_policy": "Complete context/question/options unchanged; no truncation; raw target probabilities unchanged.",
        "split_policy": "Use splits.jsonl exclusively; dev is exported as HF validation; train/dev groups preserved; no calibration or test content/IDs exported.",
        "exclusions": ["All external evaluation data and per-item results", "All calibration and internal test rows and identities", "Quarantines and rejected snapshots", "Raw scraped assets, source caches and author/reviewer logs", "wordfreq seed pool and seed assignment metadata", "Owner-local runtime paths and unrelated repository artifacts"],
        "reproduction_limit": "Input/target training reproduction is supported; exact raw authoring metadata and sealed evaluation/calibration content are not public.",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "checksums.json", {str(p.relative_to(output)): digest(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    print(json.dumps({"configs": counts, "document_notices": len(notices), "output": str(output)}, indent=2))


def verify(output: Path):
    """Prove release identity and sealed-row exclusion without sample output."""
    checks = {}; overlap = set()
    fields = ["context", "question", "options", "choice_ids", "answer_index",
              "target_distribution", "abstention_choice_id", "answerability_status", "explanation"]
    for config, version in SNAPSHOTS.items():
        snapshot = ROOT / "data/training_snapshots" / version
        splits = {r["id"]: r["split"] for line in (snapshot / "splits.jsonl").open()
                  if (r := json.loads(line))}
        expected = {}
        for line in (snapshot / "rows.jsonl").open():
            row = json.loads(line)
            if splits[row["id"]] in {"train", "dev"}:
                expected[row["id"]] = input_digest(row, fields)
        actual = set(); group_splits = {}; counts = Counter()
        for name, split in [("train", "train"), ("validation", "dev")]:
            parquet = pq.ParquetFile(output / "data" / config / f"{name}.parquet")
            for batch in parquet.iter_batches(batch_size=128):
                for row in batch.to_pylist():
                    row_id = row["id"]
                    if row_id in actual or splits.get(row_id) != split or row["split"] != split:
                        raise ValueError("Release split identity verification failed")
                    actual.add(row_id); counts[split] += 1
                    checksum = input_digest(row, fields)
                    if checksum != expected.get(row_id):
                        raise ValueError("Release changed input, option order, target or explanation")
                    group = row["snapshot_group_id"]
                    if group_splits.setdefault(group, split) != split:
                        raise ValueError("Release group crosses splits")
        if actual != set(expected):
            raise ValueError("Release identity allowlist mismatch")
        sealed = {k for k, v in splits.items() if v not in {"train", "dev"}}
        if actual & sealed:
            raise ValueError("Sealed identities leaked")
        checks[config] = {"rows": dict(counts), "identical_input_target_and_explanation": True,
                          "identity_allowlist_exact": True, "sealed_id_overlap": 0,
                          "train_dev_group_overlap": 0, "groups": len(group_splits)}
        overlap.update(actual)
    receipt = {"schema_version": "horscht-dataset-verification-v1",
               "verified_at": "2026-10-09", "checks": checks,
               "unique_published_ids": len(overlap), "packager_sha256": digest(Path(__file__))}
    write_json(output / "verification.json", receipt)
    write_json(output / "checksums.json", {str(p.relative_to(output)): digest(p)
               for p in sorted(output.rglob("*")) if p.is_file() and p.name != "checksums.json"})
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release/dataset")
    parser.add_argument("--verify", action="store_true", help="Compare a packaged release with frozen source rows")
    args = parser.parse_args()
    if args.verify:
        verify(args.output.resolve())
    else:
        package(args.output.resolve())
