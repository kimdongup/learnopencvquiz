#!/usr/bin/env python3
"""Local web dashboard for LearnOpenCV progression tasks.

Run this script and open http://127.0.0.1:8000 in your browser.
It shows the current task list, lets you view code, execute progression actions,
verifies cell runs, and provides Excel report downloads.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import secrets
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
GAME_PAGE = ROOT / "learnopencv_game.html"
QUEST_MAP_FILE = ROOT / ".cvquest" / "quest_notebooks.json"
EXCEL_OUTPUT = ROOT / ".cvquest" / "learnopencv_quest_report.xlsx"
JUPYTER_EXECUTABLE = ROOT / ".venv-jupyter" / "bin" / "jupyter"
JUPYTER_HOST = "127.0.0.1"
JUPYTER_PORT = 8888
JUPYTER_TOKEN = secrets.token_urlsafe(24)
JUPYTER_PROCESS: Optional[subprocess.Popen] = None
JUPYTER_LOCK = threading.Lock()

_STATE_CACHE: Optional[Dict[str, Any]] = None
_CACHE_TIME: float = 0.0
CACHE_TTL_SECONDS: float = 30.0

try:
    import learnopencv_progression as prog
except ImportError as exc:
    raise SystemExit("Failed to import learnopencv_progression.py: " + str(exc))

FALLBACK_HTML = """<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>LearnOpenCV Progression Dashboard</title>
  <style>
    body { font-family: system-ui, sans-serif; background: #111827; color: #f8fafc; padding: 20px; }
    h1 { color: #60a5fa; }
  </style>
</head>
<body>
  <h1>LearnOpenCV Progression Dashboard</h1>
  <p>learnopencv_game.html 파일을 찾을 수 없어 기본 모드로 실행합니다.</p>
</body>
</html>
"""


def load_html_page() -> str:
    """Load UI HTML template."""
    if GAME_PAGE.exists():
        return GAME_PAGE.read_text(encoding="utf-8")
    return FALLBACK_HTML


def safe_read_file(requested_path: str) -> str:
    """Safely read workspace text file with path traversal protection."""
    path = (ROOT / requested_path).resolve()
    try:
        path.relative_to(ROOT)
    except ValueError:
        raise FileNotFoundError("Unsafe file path")

    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {requested_path}")

    allowed_exts = {".py", ".json", ".md", ".txt", ".html", ".css", ".js", ".yml", ".yaml", ".ipynb"}
    if path.suffix.lower() not in allowed_exts:
        raise PermissionError(f"Access to {path.suffix} files is not permitted.")

    return path.read_text(encoding="utf-8", errors="replace")


def capture_output(func, *args, **kwargs) -> str:
    """Safely capture stdout from function calls."""
    stream = StringIO()
    with contextlib.redirect_stdout(stream):
        try:
            func(*args, **kwargs)
        except Exception as exc:
            print(f"Error: {exc}")
    return stream.getvalue()


def load_quest_map() -> Dict[str, dict]:
    if not QUEST_MAP_FILE.exists():
        return {}
    try:
        data = json.loads(QUEST_MAP_FILE.read_text(encoding="utf-8"))
        return {item["project"]: item for item in data.get("quests", [])}
    except Exception:
        return {}


def invalidate_cache() -> None:
    global _STATE_CACHE, _CACHE_TIME
    _STATE_CACHE = None
    _CACHE_TIME = 0.0


def get_state(force_refresh: bool = False) -> Dict[str, Any]:
    global _STATE_CACHE, _CACHE_TIME
    now = time.time()
    if not force_refresh and _STATE_CACHE is not None and (now - _CACHE_TIME) < CACHE_TTL_SECONDS:
        return _STATE_CACHE

    projects = prog.scan_projects()
    projects = prog.ensure_state(projects)
    notebook_map = load_quest_map()
    tasks = [
        {
            "id": p.id,
            "name": p.name,
            "status": p.status,
            "classification": p.classification,
            "recommendation": p.recommendation,
            "reason": p.reason,
            "notebook": notebook_map.get(p.name),
        }
        for p in projects
    ]
    counts = {"completed": 0, "unlocked": 0, "locked": 0, "failed": 0, "unavailable": 0}
    for p in projects:
        counts[p.status] = counts.get(p.status, 0) + 1

    state = {
        "total": len(projects),
        "completed": counts["completed"],
        "unlocked": counts["unlocked"],
        "locked": counts["locked"],
        "failed": counts["failed"],
        "unavailable": counts["unavailable"],
        "tasks": tasks,
        "files": ["learnopencv_progression.py", "progress_state.json", "progress_manifest.json"],
        "excel_available": EXCEL_OUTPUT.exists(),
    }
    _STATE_CACHE = state
    _CACHE_TIME = now
    return state


def jupyter_ready() -> bool:
    try:
        with urlopen(
            f"http://{JUPYTER_HOST}:{JUPYTER_PORT}/api/status?token={JUPYTER_TOKEN}",
            timeout=0.5,
        ) as response:
            return response.status == 200
    except Exception:
        return False


def ensure_jupyter() -> None:
    global JUPYTER_PROCESS
    with JUPYTER_LOCK:
        if jupyter_ready():
            return
        if not JUPYTER_EXECUTABLE.exists():
            raise RuntimeError(".venv-jupyter에 JupyterLab이 설치되어 있지 않습니다.")
        if JUPYTER_PROCESS is None or JUPYTER_PROCESS.poll() is not None:
            log_dir = ROOT / ".cvquest"
            log_dir.mkdir(parents=True, exist_ok=True)
            log = (log_dir / "jupyter.log").open("a", encoding="utf-8")
            JUPYTER_PROCESS = subprocess.Popen(
                [
                    str(JUPYTER_EXECUTABLE),
                    "lab",
                    "--no-browser",
                    f"--ServerApp.ip={JUPYTER_HOST}",
                    f"--ServerApp.port={JUPYTER_PORT}",
                    "--ServerApp.port_retries=0",
                    f"--ServerApp.root_dir={ROOT}",
                    f"--IdentityProvider.token={JUPYTER_TOKEN}",
                    "--PasswordIdentityProvider.hashed_password=",
                ],
                cwd=str(ROOT),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        for _ in range(40):
            if jupyter_ready():
                return
            if JUPYTER_PROCESS.poll() is not None:
                break
            time.sleep(0.25)
        raise RuntimeError("JupyterLab을 시작하지 못했습니다. .cvquest/jupyter.log를 확인하세요.")


def launch_notebook(task_id: int) -> Dict[str, str]:
    projects = prog.ensure_state(prog.scan_projects())
    project = next((p for p in projects if p.id == task_id), None)
    if project is None:
        raise ValueError(f"Task {task_id} not found.")
    mapping = load_quest_map().get(project.name)
    if not mapping:
        raise RuntimeError("이 퀘스트의 노트북 매핑이 없습니다. 생성기를 다시 실행하세요.")
    launch_path = mapping["launch_path"]
    path = (ROOT / launch_path).resolve()
    path.relative_to(ROOT)
    if not path.is_file() or path.suffix.lower() != ".ipynb":
        raise RuntimeError("연결된 노트북 파일을 찾을 수 없습니다.")
    ensure_jupyter()
    url = f"http://{JUPYTER_HOST}:{JUPYTER_PORT}/lab/tree/{quote(launch_path)}?token={quote(JUPYTER_TOKEN)}"
    return {"url": url, "path": launch_path, "kind": mapping["kind"]}


def stop_jupyter() -> None:
    global JUPYTER_PROCESS
    with JUPYTER_LOCK:
        if JUPYTER_PROCESS is not None and JUPYTER_PROCESS.poll() is None:
            JUPYTER_PROCESS.terminate()
            try:
                JUPYTER_PROCESS.wait(timeout=5)
            except subprocess.TimeoutExpired:
                JUPYTER_PROCESS.kill()
                JUPYTER_PROCESS.wait(timeout=2)
        JUPYTER_PROCESS = None


atexit.register(stop_jupyter)


def handle_command(command: str, task_id: Optional[int]) -> str:
    invalidate_cache()
    projects = prog.scan_projects()
    projects = prog.ensure_state(projects)

    if command == "list":
        return capture_output(prog.list_tasks, projects)
    if command == "info":
        if task_id is None:
            return "task_id is required for info"
        return capture_output(prog.show_info, projects, task_id)
    if command == "verify":
        if task_id is None:
            return "task_id is required for verify"
        res = prog.verify_task_execution(projects, task_id)
        return f"Task {task_id} Execution Result: Status={res.get('status')}, Details={res.get('reason')}"
    if command == "complete":
        if task_id is None:
            return "task_id is required for complete"
        output = capture_output(prog.complete_task, projects, task_id)
        return output
    if command == "reset":
        output = capture_output(prog.reset_state, projects)
        return output
    if command == "scan":
        output = capture_output(prog.update_and_reload)
        return output
    if command == "summary":
        return capture_output(prog.print_summary, projects)
    if command == "export":
        prog.export_manifest(projects)
        return "Exported progress_manifest.json"
    return f"Unknown command: {command}"


class DashboardHandler(BaseHTTPRequestHandler):
    def _set_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _set_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._set_html(load_html_page())
            return
        if parsed.path == "/api/tasks":
            self._set_json(get_state())
            return
        if parsed.path == "/api/jupyter":
            self._set_json({"running": jupyter_ready(), "port": JUPYTER_PORT})
            return
        if parsed.path == "/api/export-excel":
            if not EXCEL_OUTPUT.exists():
                # Generate excel on the fly if missing
                try:
                    import generate_excel_report
                    generate_excel_report.main()
                except Exception as exc:
                    self._set_json({"error": f"Failed to generate Excel report: {exc}"}, status=500)
                    return

            try:
                excel_bytes = EXCEL_OUTPUT.read_bytes()
                self.send_response(200)
                self.send_header(
                    "Content-Type",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                self.send_header(
                    "Content-Disposition", 'attachment; filename="learnopencv_quest_report.xlsx"'
                )
                self.send_header("Content-Length", str(len(excel_bytes)))
                self.end_headers()
                self.wfile.write(excel_bytes)
            except Exception as exc:
                self._set_json({"error": str(exc)}, status=500)
            return

        if parsed.path == "/api/file":
            params = parse_qs(parsed.query)
            path = params.get("path", ["learnopencv_progression.py"])[0]
            try:
                content = safe_read_file(path)
                self._set_json({"content": content})
            except (FileNotFoundError, PermissionError) as exc:
                self._set_json({"error": str(exc)}, status=404)
            return
        self.send_error(404, "Not found")

    def do_POST(self):
        if self.path not in {"/api/command", "/api/launch", "/api/jupyter", "/api/verify-run"}:
            self.send_error(404, "Not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body)
            if self.path == "/api/jupyter":
                action = data.get("action")
                if action == "start":
                    ensure_jupyter()
                elif action == "stop":
                    stop_jupyter()
                else:
                    raise ValueError("action must be start or stop")
                self._set_json({"running": jupyter_ready(), "port": JUPYTER_PORT})
                return

            if self.path == "/api/launch":
                self._set_json(launch_notebook(int(data.get("task_id", 0))))
                return

            if self.path == "/api/verify-run":
                task_id = int(data.get("task_id", 0))
                projects = prog.scan_projects()
                projects = prog.ensure_state(projects)
                res = prog.verify_task_execution(projects, task_id)
                invalidate_cache()
                self._set_json({"result": res, "tasks": get_state(force_refresh=True)["tasks"]})
                return

            command = data.get("command")
            task_id = data.get("task_id")
            output = handle_command(command, task_id)
            self._set_json({"output": output, "tasks": get_state(force_refresh=True)["tasks"]})
        except Exception as exc:
            self._set_json({"output": f"Error: {exc}"}, status=500)


def run_server(host: str = "127.0.0.1", port: int = 8000):
    server = ThreadingHTTPServer((host, port), DashboardHandler)
    print(f"LearnOpenCV dashboard running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    run_server()
