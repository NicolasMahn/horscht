#!/usr/bin/env python3
"""Watch one explicitly identified training service without controlling it."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping


SYSTEMD_PROPERTIES = (
    "ActiveState,SubState,InvocationID,MainPID,Result,ExecMainStatus,ExecStart"
)
TERMINAL_STATES = {"inactive", "failed"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def write_json(path: Path, value: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _inside_root(root: Path, value: str | Path) -> Path:
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("watcher paths must stay inside the repository") from error
    return resolved


def normalized_output(root: Path, value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    if value.startswith("/workspace/"):
        path = _inside_root(root, value.removeprefix("/workspace/"))
    else:
        path = _inside_root(root, value)
    return path.relative_to(root.resolve()).as_posix()


def systemd_show(unit: str) -> dict[str, str] | None:
    result = subprocess.run(
        ["systemctl", "--user", "show", "--no-pager", f"--property={SYSTEMD_PROPERTIES}", unit],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    values: dict[str, str] = {}
    last_key = None
    for line in result.stdout.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
            last_key = key
        elif last_key == "ExecStart":
            values[last_key] += "\n" + line
    return values


@dataclass(frozen=True)
class WatchConfig:
    root: Path
    service_name: str
    invocation_id: str
    run_generation: str
    launch_attempt: Path
    active_run: Path
    training_status: Path
    experiment_status: Path
    worker_exit: Path
    driver_exit: Path
    incident: Path
    heartbeat: Path
    report_command: tuple[str, ...]
    interval_seconds: int = 30

    @classmethod
    def load(cls, path: Path) -> "WatchConfig":
        raw = read_json(path)
        if raw is None or raw.get("schema_version") != "training-service-watch-config-v1":
            raise ValueError("invalid training service watch configuration")
        root = Path(raw["repository_root"]).resolve()
        service = raw["service_name"]
        generation = raw["run_generation"]
        invocation = raw["systemd_invocation_id"]
        if not isinstance(service, str) or not re.fullmatch(r"[A-Za-z0-9_.@:-]+\.service", service):
            raise ValueError("service_name must be one explicit systemd service unit")
        if not isinstance(generation, str) or not generation:
            raise ValueError("run_generation is required")
        if not isinstance(invocation, str) or not re.fullmatch(r"[0-9a-fA-F]{32}", invocation):
            raise ValueError("systemd_invocation_id must be a 32-character systemd ID")
        report_command = raw.get("report_command")
        if not isinstance(report_command, list) or not report_command or not all(
            isinstance(part, str) and part for part in report_command
        ):
            raise ValueError("report_command must be a nonempty argument list")
        interval = raw.get("interval_seconds", 30)
        if not isinstance(interval, int) or interval < 1:
            raise ValueError("interval_seconds must be a positive integer")

        def repo_path(name: str) -> Path:
            return _inside_root(root, raw[name])

        return cls(
            root=root,
            service_name=service,
            invocation_id=invocation.lower(),
            run_generation=generation,
            launch_attempt=repo_path("launch_attempt"),
            active_run=repo_path("active_run"),
            training_status=repo_path("training_status"),
            experiment_status=repo_path("experiment_status"),
            worker_exit=repo_path("worker_exit"),
            driver_exit=repo_path("driver_exit"),
            incident=repo_path("incident"),
            heartbeat=repo_path("heartbeat"),
            report_command=tuple(report_command),
            interval_seconds=interval,
        )


def _same_identity(record: dict | None, generation: str, invocation: str) -> bool:
    return bool(
        record
        and record.get("run_generation") == generation
        and str(record.get("systemd_invocation_id", "")).lower() == invocation
    )


def _receipt(path: Path, config: WatchConfig) -> dict | None:
    record = read_json(path)
    return record if _same_identity(record, config.run_generation, config.invocation_id) else None


def _write_heartbeat(config: WatchConfig, state: str, detail: str, **extra: object) -> dict:
    result = {
        "schema_version": "training-service-watch-heartbeat-v1",
        "service_name": config.service_name,
        "run_generation": config.run_generation,
        "systemd_invocation_id": config.invocation_id,
        "state": state,
        "detail": detail,
        "checked_at": utc_now(),
        **extra,
    }
    write_json(config.heartbeat, result)
    return result


def _refresh_report(config: WatchConfig) -> int:
    result = subprocess.run(
        list(config.report_command),
        cwd=config.root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return int(result.returncode)


def _resume_incident(
    config: WatchConfig,
    incident: dict,
    refresh_report: Callable[[WatchConfig], int],
) -> dict:
    current = read_json(config.training_status)
    expected_output = incident["expected_output"]
    shared_status_matches = normalized_output(
        config.root, current.get("output") if current else None
    ) == expected_output

    if not incident.get("status_update_done"):
        if not shared_status_matches:
            incident["training_status_update_error"] = "shared training status output does not match this run"
            incident["training_status_preserved_completed"] = False
        elif current and current.get("status") == "completed":
            incident["training_status_preserved_completed"] = True
        elif current is not None:
            current["status"] = "failed"
            current["training_running"] = False
            current["phase"] = "Training service terminated before pipeline completion"
            current["message"] = "The matching training service exited unsuccessfully. See the service-watch incident receipt."
            current["updated_at"] = utc_now()
            write_json(config.training_status, current)
            incident["training_status_preserved_completed"] = False
        else:
            incident["training_status_update_error"] = "shared training status is unavailable"

        experiment = read_json(config.experiment_status)
        if _same_identity(experiment, config.run_generation, config.invocation_id):
            if experiment.get("status") == "completed":
                incident["experiment_status_preserved_completed"] = True
            else:
                experiment["status"] = "failed"
                experiment["message"] = "The matching training service exited unsuccessfully. See the service-watch incident receipt."
                experiment["updated_at"] = utc_now()
                write_json(config.experiment_status, experiment)
                incident["experiment_status_updated"] = True
        incident["status_update_done"] = True
        incident["updated_at"] = utc_now()
        write_json(config.incident, incident)

    if not incident.get("report_refresh_attempted"):
        incident["report_refresh_attempted"] = True
        incident["report_refresh_started_at"] = utc_now()
        write_json(config.incident, incident)
        try:
            code = refresh_report(config)
        except Exception as error:  # record only exception type; stderr may contain evaluation data
            incident["report_refresh_exit_code"] = None
            incident["report_refresh_error_type"] = type(error).__name__
        else:
            incident["report_refresh_exit_code"] = code
        incident["report_refresh_completed_at"] = utc_now()
        write_json(config.incident, incident)
    return _write_heartbeat(config, "failure_recorded", "matching terminal failure recorded")


def monitor_once(
    config: WatchConfig,
    *,
    show_service: Callable[[str], dict[str, str] | None] = systemd_show,
    refresh_report: Callable[[WatchConfig], int] = _refresh_report,
) -> dict:
    manifest = read_json(config.launch_attempt)
    if manifest is None:
        return _write_heartbeat(config, "waiting", "launch attempt manifest not available")
    expected_output = normalized_output(config.root, manifest.get("expected_output"))
    manifest_ok = (
        manifest.get("schema_version") == "training-launch-attempt-v1"
        and manifest.get("run_generation") == config.run_generation
        and manifest.get("service_name") == config.service_name
        and expected_output is not None
    )
    if not manifest_ok:
        return _write_heartbeat(config, "identity_mismatch", "launch manifest does not match watcher configuration")

    service = show_service(config.service_name)
    if service is None or not service.get("InvocationID"):
        return _write_heartbeat(config, "waiting", "matching service invocation not visible")
    observed_invocation = service.get("InvocationID", "").lower()
    if observed_invocation != config.invocation_id:
        return _write_heartbeat(config, "waiting", "service invocation differs from configured invocation")

    active = read_json(config.active_run)
    if not _same_identity(active, config.run_generation, config.invocation_id):
        return _write_heartbeat(config, "waiting", "active run generation or invocation not bound yet")
    if normalized_output(config.root, active.get("output")) != expected_output:
        return _write_heartbeat(config, "identity_mismatch", "active run output differs from launch manifest")
    command_marker = str(config.launch_attempt.parent)
    exec_start = service.get("ExecStart", "")
    if exec_start and command_marker not in exec_start:
        return _write_heartbeat(config, "identity_mismatch", "service command does not point at this recovery attempt")

    if config.incident.exists():
        incident = read_json(config.incident)
        if incident and incident.get("run_generation") == config.run_generation and incident.get("systemd_invocation_id") == config.invocation_id:
            return _resume_incident(config, incident, refresh_report)

    state = service.get("ActiveState", "unknown")
    if state not in TERMINAL_STATES:
        status_output_matches = normalized_output(
            config.root, (read_json(config.training_status) or {}).get("output")
        ) == expected_output
        return _write_heartbeat(
            config,
            "watching" if status_output_matches else "waiting",
            "matching training service is active" if status_output_matches else "shared training status has not switched to this output",
            active_state=state,
        )

    driver_exit = _receipt(config.driver_exit, config)
    worker_exit = _receipt(config.worker_exit, config)
    experiment = read_json(config.experiment_status)
    experiment_matches = _same_identity(experiment, config.run_generation, config.invocation_id)
    result = service.get("Result", "unknown")
    try:
        main_status = int(service.get("ExecMainStatus", "0"))
    except ValueError:
        main_status = -1

    successful = bool(
        state == "inactive"
        and result == "success"
        and main_status == 0
        and driver_exit
        and driver_exit.get("exit_code") == 0
        and experiment_matches
        and experiment.get("status") == "completed"
        and normalized_output(config.root, (read_json(config.training_status) or {}).get("output")) == expected_output
    )
    if successful:
        return _write_heartbeat(config, "completed", "matching training service and pipeline completed")

    if state == "inactive" and result == "success" and main_status == 0 and not driver_exit:
        return _write_heartbeat(config, "waiting", "service stopped cleanly; awaiting driver exit receipt")
    if state == "inactive" and result == "success" and main_status == 0 and driver_exit and driver_exit.get("exit_code") == 0 and not experiment_matches:
        return _write_heartbeat(config, "waiting", "awaiting matching experiment completion receipt")

    incident = {
        "schema_version": "training-service-watch-incident-v1",
        "service_name": config.service_name,
        "run_generation": config.run_generation,
        "systemd_invocation_id": config.invocation_id,
        "expected_output": expected_output,
        "terminal_state": state,
        "systemd_result": result,
        "systemd_exit_code": main_status,
        "driver_exit_code": driver_exit.get("exit_code") if driver_exit else None,
        "worker_exit_code": worker_exit.get("exit_code") if worker_exit else None,
        "experiment_status": experiment.get("status") if experiment_matches else None,
        "detected_at": utc_now(),
        "status_update_done": False,
        "report_refresh_attempted": False,
    }
    write_json(config.incident, incident)
    return _resume_incident(config, incident, refresh_report)


def run(config: WatchConfig) -> int:
    lock_path = config.heartbeat.with_suffix(config.heartbeat.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 0
        while True:
            result = monitor_once(config)
            if result["state"] in {"completed", "failure_recorded", "identity_mismatch"}:
                return 0
            time.sleep(config.interval_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config = WatchConfig.load(args.config)
    return run(config)


if __name__ == "__main__":
    sys.exit(main())
