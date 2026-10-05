"""Serve the generated report on localhost without exposing the project directory."""

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

PROJECT_ROOT = Path(__file__).resolve().parent
DIVERSITY_ROOT = PROJECT_ROOT / "dataset_audit" / "audit-2026-09-29"
TRAINING_STATUS = PROJECT_ROOT / "runs" / "training-status.json"
TRAINING_COMPARISON = (
    PROJECT_ROOT / "runs" / "paired-overnight-2026-09-30" / "comparison" / "comparison.html"
)
TRAINING_PILOT_COMPARISON = PROJECT_ROOT / "runs" / "paired-pilot-2026-09-29" / "comparison.html"


def current_training_curves_file():
    try:
        status = json.loads(TRAINING_STATUS.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    raw_path = status.get("curves_path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None

    if raw_path == "/workspace/runs":
        candidate = PROJECT_ROOT / "runs"
    elif raw_path.startswith("/workspace/runs/"):
        candidate = PROJECT_ROOT / "runs" / raw_path.removeprefix("/workspace/runs/")
    else:
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate

    try:
        resolved = candidate.resolve(strict=True)
        runs_root = (PROJECT_ROOT / "runs").resolve(strict=True)
        if resolved.suffix != ".html" or not resolved.is_file():
            return None
        if not resolved.is_relative_to(runs_root):
            return None
    except (OSError, RuntimeError, ValueError):
        return None
    return resolved


TRAINING_PAGE = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light dark">
  <title>Training status</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; padding: 1.5rem; font: 16px/1.5 system-ui, sans-serif; }
    main { max-width: 64rem; margin: 0 auto; }
    nav { margin-bottom: 1rem; }
    dl { display: grid; grid-template-columns: minmax(9rem, 13rem) 1fr; gap: .35rem 1rem; }
    dt { font-weight: 600; }
    dd { margin: 0; overflow-wrap: anywhere; }
    @media (max-width: 36rem) { dl { grid-template-columns: 1fr; gap: 0; } dd { margin-bottom: .7rem; } }
  </style>
</head>
<body>
  <main>
    <nav aria-label="Training views">
      <a href="/training/comparison">Completed overnight comparison</a>
      <a href="/training/pilot-comparison">Bounded GPU pilot comparison</a>
      <a id="curves-link" href="/training/curves" hidden>Live training curves</a>
    </nav>
    <p role="status" aria-live="polite"><strong id="state">Loading current run status</strong></p>
    <p id="message">Loading the latest training and evaluation state.</p>
    <p>The retained bounded pilot compared two arms on the same 32 questions and 9,212 tokens. Its same-generator holdout was exploratory and is not an independent system test.</p>
    <dl id="details" hidden></dl>
    <ul id="metrics" aria-label="Latest training metrics" hidden></ul>
  </main>
  <script>
    const stateNode = document.getElementById('state');
    const messageNode = document.getElementById('message');
    const detailsNode = document.getElementById('details');
    const metricsNode = document.getElementById('metrics');
    const curvesLink = document.getElementById('curves-link');
    const labels = {
      arm: 'Arm', phase: 'Phase', epoch: 'Epoch', total_epochs: 'Total epochs',
      update: 'Update', questions_seen: 'Questions seen', total_questions: 'Total questions',
      started_at: 'Started', updated_at: 'Updated', completed_at: 'Completed',
      training_running: 'Training running', evaluation_running: 'Evaluation running',
      launch_gate: 'Launch gate'
    };
    const states = {
      preparing: 'Preparing the larger matched run',
      running: 'Training is running',
      interrupted: 'Training is interrupted',
      completed: 'Training completed',
      failed: 'Training failed'
    };
    function statusLabel(status) {
      if (status.training_running === true && status.evaluation_running === true) {
        return 'Training and evaluation are running';
      }
      if (status.evaluation_running === true) return 'Evaluation is running';
      if (status.training_running === true) return 'Training is running';
      return states[status.status] || `Training status: ${status.status || 'unknown'}`;
    }
    function displayValue(value) {
      return typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean'
        ? String(value)
        : JSON.stringify(value);
    }
    function addPair(parent, label, value) {
      const term = document.createElement('dt');
      const description = document.createElement('dd');
      term.textContent = label;
      description.textContent = displayValue(value);
      parent.append(term, description);
    }
    async function updateCurvesLink(status) {
      curvesLink.hidden = true;
      if (typeof status.curves_path !== 'string' || !status.curves_path.trim()) return;
      try {
        const response = await fetch('/training/curves', { method: 'HEAD', cache: 'no-store' });
        curvesLink.hidden = !response.ok;
      } catch (_) {
        curvesLink.hidden = true;
      }
    }
    function render(status) {
      stateNode.textContent = statusLabel(status);
      messageNode.textContent = status.message || '';
      updateCurvesLink(status);
      detailsNode.replaceChildren();
      for (const [key, label] of Object.entries(labels)) {
        if (status[key] !== undefined && status[key] !== null && status[key] !== '') {
          addPair(detailsNode, label, status[key]);
        }
      }
      detailsNode.hidden = detailsNode.childElementCount === 0;
      metricsNode.replaceChildren();
      const metrics = status.latest_metrics || {};
      for (const [key, value] of Object.entries(metrics)) {
        const item = document.createElement('li');
        item.textContent = `${key}: ${displayValue(value)}`;
        metricsNode.append(item);
      }
      metricsNode.hidden = metricsNode.childElementCount === 0;
    }
    async function refresh() {
      try {
        const response = await fetch('/training-status.json', { cache: 'no-store' });
        if (!response.ok) return;
        render(await response.json());
      } catch (_) {
        // Keep the last published status visible while the runner or server is unavailable.
      }
    }
    refresh();
    window.setInterval(refresh, 3000);
  </script>
</body>
</html>"""

REPORTS = {
    "/": Path(__file__).with_name("report.html"),
    "/report.html": Path(__file__).with_name("report.html"),
    "/smoke": Path(__file__).with_name("report-smoke.html"),
    "/report-smoke.html": Path(__file__).with_name("report-smoke.html"),
    "/size": Path(__file__).with_name("report-size.html"),
    "/report-size.html": Path(__file__).with_name("report-size.html"),
    "/transfer": Path(__file__).with_name("report-transfer.html"),
    "/report-transfer.html": Path(__file__).with_name("report-transfer.html"),
}


class ReportHandler(BaseHTTPRequestHandler):
    def do_HEAD(self):
        path = urlsplit(self.path).path
        if path == "/training/curves":
            report = current_training_curves_file()
        elif path == "/training/comparison":
            report = TRAINING_COMPARISON
        elif path == "/training/pilot-comparison":
            report = TRAINING_PILOT_COMPARISON
        else:
            report = None
        if report is None:
            self.send_error(404)
            return
        try:
            size = report.stat().st_size
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/training/":
            self.send_response(302)
            self.send_header("Location", "/training")
            self.end_headers()
            return
        if path == "/training":
            content = TRAINING_PAGE
            content_type = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if path == "/training-status.json":
            if not TRAINING_STATUS.is_file():
                self.send_error(404, "Training status has not been published yet")
                return
            content = TRAINING_STATUS.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return
        if path == "/training/curves":
            report = current_training_curves_file()
            if report is None:
                self.send_error(404, "Training curves are not available")
                return
            try:
                content = report.read_bytes()
            except OSError:
                self.send_error(404, "Training curves are not available")
                return
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(content)
            return

        if path == "/diversity":
            self.send_response(302)
            self.send_header("Location", "/diversity/")
            self.end_headers()
            return

        diversity_files = {
            "": ("report.html", "text/html; charset=utf-8"),
            "report.html": ("report.html", "text/html; charset=utf-8"),
            "audit.md": ("audit.md", "text/markdown; charset=utf-8"),
            "snapshot.json": ("snapshot.json", "application/json; charset=utf-8"),
            "embedding-cache.json": ("embedding-cache.json", "application/json; charset=utf-8"),
            "language-centering-v1.json": ("language-centering-v1.json", "application/json; charset=utf-8"),
            "metrics.json": ("metrics.json", "application/json; charset=utf-8"),
            "nearest-neighbors.jsonl": ("nearest-neighbors.jsonl", "application/x-ndjson; charset=utf-8"),
            "sample.jsonl": ("sample.jsonl", "application/x-ndjson; charset=utf-8"),
            "accepted-index.jsonl": ("accepted-index.jsonl", "application/x-ndjson; charset=utf-8"),
        }
        if path == "/diversity/" or path.startswith("/diversity/"):
            relative = path.removeprefix("/diversity/")
            entry = diversity_files.get(relative)
            if entry is None:
                self.send_error(404)
                return
            filename, content_type = entry
            report = DIVERSITY_ROOT / filename
        elif path == "/training/comparison":
            report = TRAINING_COMPARISON
            content_type = "text/html; charset=utf-8"
        elif path == "/training/pilot-comparison":
            report = TRAINING_PILOT_COMPARISON
            content_type = "text/html; charset=utf-8"
        else:
            report = REPORTS.get(path)
            content_type = "text/html; charset=utf-8"

        if report is None or not report.is_file():
            self.send_error(404)
            return
        content = report.read_bytes()
        if path in {"/", "/report.html"}:
            content = content.replace(
                b"</nav>",
                b'<a href="/training">Live training status</a></nav>',
                1,
            )
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8765), ReportHandler)
    print("Report: http://127.0.0.1:8765/", flush=True)
    server.serve_forever()
