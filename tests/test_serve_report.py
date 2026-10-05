import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

import serve_report


@contextmanager
def report_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve_report.ReportHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def write_status(path: Path, curves_path: str) -> None:
    path.write_text(json.dumps({"curves_path": curves_path}), encoding="utf-8")


def test_comparison_routes_serve_overnight_and_keep_pilot_access(tmp_path, monkeypatch):
    overnight = tmp_path / "overnight.html"
    pilot = tmp_path / "pilot.html"
    overnight.write_text("overnight comparison", encoding="utf-8")
    pilot.write_text("bounded pilot comparison", encoding="utf-8")
    monkeypatch.setattr(serve_report, "TRAINING_COMPARISON", overnight)
    monkeypatch.setattr(serve_report, "TRAINING_PILOT_COMPARISON", pilot)

    with report_server() as base_url:
        for path, expected in (
            ("/training/comparison", b"overnight comparison"),
            ("/training/pilot-comparison", b"bounded pilot comparison"),
        ):
            with urllib.request.urlopen(f"{base_url}{path}") as response:
                assert response.read() == expected
            request = urllib.request.Request(f"{base_url}{path}", method="HEAD")
            with urllib.request.urlopen(request) as response:
                assert response.status == 200
                assert response.read() == b""


@pytest.mark.parametrize("container_path", [False, True])
def test_curves_route_serves_only_paths_under_runs(tmp_path, monkeypatch, container_path):
    project_root = tmp_path / "project"
    runs_root = project_root / "runs"
    curve = runs_root / "run-1" / "head" / "curves.html"
    curve.parent.mkdir(parents=True)
    curve.write_text("<html>curves</html>", encoding="utf-8")
    status_path = tmp_path / "status.json"
    monkeypatch.setattr(serve_report, "PROJECT_ROOT", project_root)
    monkeypatch.setattr(serve_report, "TRAINING_STATUS", status_path)

    if container_path:
        requested_path = "/workspace/runs/" + curve.relative_to(runs_root).as_posix()
    else:
        requested_path = str(curve)
    write_status(status_path, requested_path)

    with report_server() as base_url:
        with urllib.request.urlopen(f"{base_url}/training/curves") as response:
            assert response.status == 200
            assert response.headers.get_content_type() == "text/html"
            assert response.read() == b"<html>curves</html>"
        request = urllib.request.Request(f"{base_url}/training/curves", method="HEAD")
        with urllib.request.urlopen(request) as response:
            assert response.status == 200
            assert response.read() == b""

    outside = project_root / "report.html"
    outside.parent.mkdir(parents=True, exist_ok=True)
    outside.write_text("must not be served", encoding="utf-8")
    write_status(status_path, str(outside))
    with report_server() as base_url:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(f"{base_url}/training/curves")
        assert error.value.code == 404
