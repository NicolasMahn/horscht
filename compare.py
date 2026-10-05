"""Run selected candidates sequentially and retain self-contained result records."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent
CANDIDATES = json.loads((ROOT / "candidates.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=CANDIDATES, default=list(CANDIDATES))
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    batch = ROOT / "runs" / f"{stamp}-comparison"
    batch.mkdir(parents=True)
    results = ROOT / "results" / "comparison"
    results.mkdir(parents=True, exist_ok=True)
    failures = []
    for candidate in args.models:
        run_dir = batch / candidate
        print(f"Starting {candidate}; log: {batch / (candidate + '.log')}", flush=True)
        with (batch / f"{candidate}.log").open("w") as log:
            trained = subprocess.run([sys.executable, "-u", str(ROOT / "experiment.py"), "run",
                                      "--model", candidate, "--output", str(run_dir)], stdout=log, stderr=subprocess.STDOUT)
            if not (run_dir / "report.json").exists():
                (results / f"{candidate}.json").write_text(json.dumps({
                    "candidate": candidate, "status": "failed", "training_exit_code": trained.returncode,
                    "run_directory": str(run_dir.relative_to(ROOT)),
                }, indent=2) + "\n")
                failures.append(candidate)
                print(f"{candidate}: training failed; see log", flush=True)
                continue
            verified = subprocess.run([sys.executable, "-u", str(ROOT / "experiment.py"), "verify", str(run_dir)],
                                      stdout=log, stderr=subprocess.STDOUT)
        record = {name: json.loads((run_dir / filename).read_text()) for name, filename in [
            ("manifest", "manifest.json"), ("report", "report.json"), ("examples", "examples.json")
        ]}
        record["reload_verification"] = (json.loads((run_dir / "reload_verification.json").read_text())
                                         if (run_dir / "reload_verification.json").exists() else None)
        record["run_directory"] = str(run_dir.relative_to(ROOT))
        record["training_exit_code"] = trained.returncode
        record["verification_exit_code"] = verified.returncode
        (results / f"{candidate}.json").write_text(json.dumps(record, indent=2) + "\n")
        report = record["report"]
        print(f"{candidate}: train {report['training']['final']}, probe {report['probe']}, "
              f"{report['training']['steps']} steps, {report['feature_extraction_seconds']:.2f}s encoding", flush=True)
        if trained.returncode or verified.returncode:
            failures.append(candidate)
    if failures:
        print(f"Failed checks: {', '.join(failures)}", flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
