import json
from pathlib import Path

from scripts.watch_training_service import WatchConfig, monitor_once


GENERATION = "new-data-recovery-003-dc1fc3f13f6ddc24"
INVOCATION = "0123456789abcdef0123456789abcdef"
SERVICE = "horscht-newdata-deferred-20261008.service"
OUTPUT = "runs/new-data-last-block-2026-10-08/main-recovery-002"


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _config(tmp_path: Path) -> WatchConfig:
    root = tmp_path
    attempt_dir = root / "runs/new-data-last-block-2026-10-08/recovery-003"
    launch = attempt_dir / "launch-attempt.json"
    active = root / "runs/new-data-last-block-2026-10-08/main-active.json"
    status = root / "runs/new-data-last-block-2026-10-08/training-status.json"
    experiment = root / "runs/new-data-last-block-2026-10-08/experiment-status.json"
    worker_exit = root / "runs/new-data-last-block-2026-10-08/main-recovery-003-process-exit.json"
    driver_exit = attempt_dir / "driver-process-exit.json"
    incident = attempt_dir / "service-watch-incident.json"
    heartbeat = attempt_dir / "service-watch-heartbeat.json"
    _write(launch, {
        "schema_version": "training-launch-attempt-v1",
        "run_generation": GENERATION,
        "service_name": SERVICE,
        "expected_output": OUTPUT,
    })
    _write(active, {
        "run_generation": GENERATION,
        "systemd_invocation_id": INVOCATION,
        "output": OUTPUT,
    })
    _write(status, {"output": "/workspace/" + OUTPUT, "status": "running", "training_running": True})
    _write(experiment, {
        "run_generation": GENERATION,
        "systemd_invocation_id": INVOCATION,
        "status": "running",
    })
    return WatchConfig(
        root=root,
        service_name=SERVICE,
        invocation_id=INVOCATION,
        run_generation=GENERATION,
        launch_attempt=launch,
        active_run=active,
        training_status=status,
        experiment_status=experiment,
        worker_exit=worker_exit,
        driver_exit=driver_exit,
        incident=incident,
        heartbeat=heartbeat,
        report_command=("unused",),
        interval_seconds=30,
    )


def _service(config: WatchConfig, *, state="active", invocation=INVOCATION, result="success", exit_code=0):
    return {
        "ActiveState": state,
        "SubState": state,
        "InvocationID": invocation,
        "MainPID": "42" if state == "active" else "0",
        "Result": result,
        "ExecMainStatus": str(exit_code),
        "ExecStart": f"{{ path=/usr/bin/bash ; argv[]=/usr/bin/bash {config.launch_attempt.parent}/run_pipeline.sh }}",
    }


def _status(config: WatchConfig) -> dict:
    return json.loads(config.training_status.read_text(encoding="utf-8"))


def test_active_service_only_writes_nonterminal_heartbeat(tmp_path):
    config = _config(tmp_path)
    refreshed = []
    result = monitor_once(
        config,
        show_service=lambda _unit: _service(config),
        refresh_report=lambda _config: refreshed.append(True) or 0,
    )
    assert result["state"] == "watching"
    assert _status(config)["training_running"] is True
    assert not config.incident.exists()
    assert refreshed == []


def test_wrong_systemd_invocation_never_mutates_training_status(tmp_path):
    config = _config(tmp_path)
    before = config.training_status.read_bytes()
    refreshed = []
    result = monitor_once(
        config,
        show_service=lambda _unit: _service(config, state="failed", invocation="f" * 32, result="exit-code", exit_code=139),
        refresh_report=lambda _config: refreshed.append(True) or 0,
    )
    assert result["state"] == "waiting"
    assert config.training_status.read_bytes() == before
    assert not config.incident.exists()
    assert refreshed == []


def test_terminal_failure_updates_matching_status_and_refreshes_report_once(tmp_path):
    config = _config(tmp_path)
    _write(config.driver_exit, {
        "run_generation": GENERATION,
        "systemd_invocation_id": INVOCATION,
        "exit_code": 139,
    })
    refreshed = []
    show = lambda _unit: _service(config, state="failed", result="exit-code", exit_code=1)
    refresh = lambda _config: refreshed.append(True) or 0

    first = monitor_once(config, show_service=show, refresh_report=refresh)
    second = monitor_once(config, show_service=show, refresh_report=refresh)

    assert first["state"] == second["state"] == "failure_recorded"
    assert _status(config)["status"] == "failed"
    assert _status(config)["training_running"] is False
    incident = json.loads(config.incident.read_text(encoding="utf-8"))
    assert incident["run_generation"] == GENERATION
    assert incident["driver_exit_code"] == 139
    assert incident["report_refresh_exit_code"] == 0
    assert refreshed == [True]


def test_failure_before_shared_output_switch_records_incident_without_overwriting_old_run(tmp_path):
    config = _config(tmp_path)
    old = {"output": "/workspace/runs/new-data-last-block-2026-10-08/main-recovery-001", "status": "failed", "training_running": False}
    _write(config.training_status, old)
    refreshed = []

    result = monitor_once(
        config,
        show_service=lambda _unit: _service(config, state="failed", result="exit-code", exit_code=1),
        refresh_report=lambda _config: refreshed.append(True) or 0,
    )

    assert result["state"] == "failure_recorded"
    assert _status(config) == old
    assert config.incident.exists()
    assert "does not match" in json.loads(config.incident.read_text(encoding="utf-8"))["training_status_update_error"]
    assert refreshed == [True]


def test_completed_status_is_never_overwritten_on_service_failure(tmp_path):
    config = _config(tmp_path)
    completed = {"output": "/workspace/" + OUTPUT, "status": "completed", "training_running": False}
    _write(config.training_status, completed)
    refreshed = []

    result = monitor_once(
        config,
        show_service=lambda _unit: _service(config, state="failed", result="signal", exit_code=139),
        refresh_report=lambda _config: refreshed.append(True) or 0,
    )

    assert result["state"] == "failure_recorded"
    assert _status(config) == completed
    incident = json.loads(config.incident.read_text(encoding="utf-8"))
    assert incident["training_status_preserved_completed"] is True
    assert refreshed == [True]


def test_clean_pipeline_completion_requires_matching_driver_and_experiment_receipts(tmp_path):
    config = _config(tmp_path)
    _write(config.driver_exit, {
        "run_generation": GENERATION,
        "systemd_invocation_id": INVOCATION,
        "exit_code": 0,
    })
    _write(config.experiment_status, {
        "run_generation": GENERATION,
        "systemd_invocation_id": INVOCATION,
        "status": "completed",
    })
    _write(config.training_status, {"output": "/workspace/" + OUTPUT, "status": "completed", "training_running": False})

    result = monitor_once(
        config,
        show_service=lambda _unit: _service(config, state="inactive", result="success", exit_code=0),
    )

    assert result["state"] == "completed"
    assert not config.incident.exists()
