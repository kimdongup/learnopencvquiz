#!/usr/bin/env python3
"""CLI dashboard and progression engine for LearnOpenCV project progression.

This script scans repository folders, classifies projects (notebook, script, docs-only, mixed,
or unavailable), manages progression state, and provides execution verification.

Usage:
  python learnopencv_progression.py list
  python learnopencv_progression.py info 3
  python learnopencv_progression.py complete 3
  python learnopencv_progression.py verify 3
  python learnopencv_progression.py reset
  python learnopencv_progression.py scan
  python learnopencv_progression.py summary
  python learnopencv_progression.py export
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "progress_state.json"
MAP_FILE = ROOT / ".cvquest" / "quest_notebooks.json"
EXCLUDE_DIRS: Set[str] = {
    ".git",
    ".venv",
    ".venv-jupyter",
    ".cvquest",
    "__pycache__",
    ".ipynb_checkpoints",
    "build",
    "dist",
    "node_modules",
    ".idea",
    ".vscode",
}
KNOWN_NOTEBOOK_EXTS: Set[str] = {".ipynb"}
KNOWN_SCRIPT_EXTS: Set[str] = {".py", ".sh", ".cpp", ".c", ".cc", ".cxx"}
KNOWN_ENV_FILES: Set[str] = {
    "requirements.txt",
    "requirements-dev.txt",
    "requirements.in",
    "environment.yml",
    "env.yml",
    "pyproject.toml",
    "setup.py",
    "conda.yml",
}

UNAVAILABLE_KEYWORDS: Dict[str, str] = {
    "oak-d": "OAK-D / DepthAI Spatial Camera hardware required.",
    "depthai": "OAK-D / DepthAI Spatial Camera hardware required.",
    "oakd": "OAK-D / DepthAI Spatial Camera hardware required.",
    "jetson": "NVIDIA Jetson Edge Hardware required.",
    "arduino": "Arduino microcontroller hardware required.",
    "drone": "Tello / PX4 Drone hardware required.",
    "ros2": "ROS2 (Robot Operating System 2) environment required.",
    "carla": "CARLA Autonomous Driving Simulator required.",
    "moondream-cloud-api": "External Moondream Cloud API Key required.",
    "speech-to-speech": "External OpenAI Realtime API Key required.",
    "install-opencv-windows-exe": "Documentation-only guide for Windows EXE installation.",
    "install-opencv-5-on-linux": "Documentation-only guide for Linux build installation.",
    "opencv-dnn-gpu-support-linux": "Documentation-only guide for OpenCV GPU compilation.",
    "opencv-dnn-gpu-support-windows": "Documentation-only guide for OpenCV GPU compilation.",
    "ci": "GitHub Actions CI workflow documentation.",
    "docs": "Documentation repository index.",
}


@dataclass
class ProjectEntry:
    id: int
    name: str
    path: str
    status: str  # "locked" | "unlocked" | "completed" | "failed" | "unavailable"
    classification: str
    runnable: bool
    recommendation: str
    has_readme: bool
    has_requirements: bool
    has_env: bool
    file_count: int
    extensions: Dict[str, int]
    reason: str = ""

    def lock_reason(self, previous: Optional["ProjectEntry"]) -> str:
        if self.status == "locked":
            if previous is None:
                return "First task is locked until you run scan or reset."
            if previous.status not in {"completed", "unavailable"}:
                return f"Unlocks after task {previous.id} is completed or marked unavailable."
            return "Locked."
        if self.status == "unavailable":
            return f"Unavailable: {self.reason}"
        return ""


def get_project_files(project_dir: Path) -> List[Path]:
    """Retrieve all relevant files inside project_dir, pruning excluded directories."""
    files: List[Path] = []
    for root, dirs, filenames in os.walk(project_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fn in filenames:
            if not fn.startswith("."):
                files.append(Path(root) / fn)
    return files


def classify_unavailability(project_name: str, classification: str, project_dir: Path) -> Tuple[bool, str]:
    """Check if a project is inherently Unavailable."""
    if classification == "docs-only":
        return True, "Documentation-only project (No executable code script or notebook)."

    if classification == "cpp-wrapper" and not (project_dir / "CMakeLists.txt").exists():
        return True, "C++ project requiring manual CMake & OpenCV C++ SDK compilation."

    name_lower = project_name.lower()
    for key, reason in UNAVAILABLE_KEYWORDS.items():
        if key in name_lower:
            return True, reason

    return False, ""


def scan_projects() -> List[ProjectEntry]:
    """Scan all project directories in the workspace root and return project entries."""
    project_dirs = sorted(
        [p for p in ROOT.iterdir() if p.is_dir() and p.name not in EXCLUDE_DIRS and not p.name.startswith(".")]
    )
    projects: List[ProjectEntry] = []

    for idx, project_dir in enumerate(project_dirs, start=1):
        all_files = get_project_files(project_dir)
        file_names = [str(p.relative_to(project_dir)) for p in all_files]
        extensions: Dict[str, int] = {}
        run_files: List[str] = []

        for path in all_files:
            ext = path.suffix.lower()
            if ext:
                extensions[ext] = extensions.get(ext, 0) + 1
            if ext in KNOWN_NOTEBOOK_EXTS or ext in KNOWN_SCRIPT_EXTS:
                run_files.append(str(path.relative_to(project_dir)))

        has_readme = any(name.lower() == "readme.md" or name.lower().startswith("readme") for name in file_names)
        has_requirements = any(Path(name).name.lower() in KNOWN_ENV_FILES for name in file_names)
        has_env = has_requirements
        has_notebook = any(ext in KNOWN_NOTEBOOK_EXTS for ext in extensions)
        has_script = any(ext in KNOWN_SCRIPT_EXTS for ext in extensions)

        if has_notebook and has_script:
            classification = "mixed"
            runnable = True
        elif has_notebook:
            classification = "notebook"
            runnable = True
        elif has_script:
            classification = "script"
            runnable = True
        else:
            classification = "docs-only"
            runnable = False

        is_unavail, unavail_reason = classify_unavailability(project_dir.name, classification, project_dir)
        initial_status = "unavailable" if is_unavail else "locked"

        recommendation = generate_recommendation(project_dir, file_names, classification, run_files)

        projects.append(
            ProjectEntry(
                id=idx,
                name=project_dir.name,
                path=str(project_dir.relative_to(ROOT)),
                status=initial_status,
                classification=classification,
                runnable=runnable and not is_unavail,
                recommendation=recommendation,
                has_readme=has_readme,
                has_requirements=has_requirements,
                has_env=has_env,
                file_count=len(all_files),
                extensions=extensions,
                reason=unavail_reason if is_unavail else "",
            )
        )

    return projects


def generate_recommendation(
    project_dir: Path, files: List[str], classification: str, run_files: List[str]
) -> str:
    if classification == "notebook":
        notebook_files = [f for f in run_files if f.lower().endswith(".ipynb")]
        return (
            f"Open {notebook_files[0]} in Jupyter and run all cells."
            if notebook_files
            else "Open a notebook in Jupyter and run all cells."
        )

    if classification == "script":
        if any(f.lower().endswith("main.py") for f in run_files):
            return "Run `python main.py` from the project folder."
        if any(Path(f).parts[0].lower() == "run" for f in run_files):
            return "Inspect the `run/` folder and execute the main script."
        if run_files:
            return f"Run {run_files[0]} from the project folder."
        return "Inspect the README and run the available script."

    if classification == "mixed":
        if run_files:
            return f"This project has notebooks and scripts. Run {run_files[0]} or launch Jupyter."
        return "This project contains mixed runnable content."

    if classification == "docs-only":
        return "Read README.md for documentation guidance (Unavailable for code execution)."

    return "Review the project folder for runnable content."


def load_state(projects: List[ProjectEntry]) -> List[ProjectEntry]:
    """Apply saved progression state to scanned projects based on relative path matching."""
    if not STATE_FILE.exists():
        return initialize_state(projects)

    try:
        with STATE_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        old_state = {
            item.get("path"): (item.get("status"), item.get("reason", ""))
            for item in data.get("tasks", [])
            if item.get("path")
        }
    except Exception:
        return initialize_state(projects)

    for project in projects:
        if project.status == "unavailable":
            continue

        saved_info = old_state.get(project.path)
        if saved_info:
            saved_status, saved_reason = saved_info
            if saved_status in {"completed", "failed", "unavailable"}:
                project.status = saved_status
                project.reason = saved_reason

    # Unlock first runnable incomplete task
    first_incomplete = next(
        (p for p in projects if p.status not in {"completed", "unavailable"}), None
    )
    for project in projects:
        if project.status not in {"completed", "failed", "unavailable"}:
            project.status = "unlocked" if project is first_incomplete else "locked"

    return projects


def initialize_state(projects: List[ProjectEntry]) -> List[ProjectEntry]:
    """Initialize state from scratch, unlocking the first available project."""
    if not projects:
        return []

    first_runnable = next((p for p in projects if p.status != "unavailable"), None)
    for project in projects:
        if project.status != "unavailable":
            project.status = "unlocked" if project is first_runnable else "locked"
            project.reason = ""

    save_state(projects)
    return projects


def save_state(projects: List[ProjectEntry]) -> None:
    """Save progression state to STATE_FILE."""
    serialized = {"tasks": [asdict(project) for project in projects]}
    with STATE_FILE.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, ensure_ascii=False)


def ensure_state(projects: List[ProjectEntry]) -> List[ProjectEntry]:
    """Ensure state is loaded and synchronized."""
    state = load_state(projects)
    save_state(state)
    return state


def list_tasks(projects: List[ProjectEntry]) -> None:
    print("LearnOpenCV Progression Dashboard")
    print("--------------------------------")
    print("ID  Status       Type        Name")
    print("--- ------------ ---------- ----------------------------------------")
    for project in projects:
        print(f"{project.id:03d}  {project.status:<12} {project.classification:<10} {project.name}")
    print("\nUse `info <id>` for details, `verify <id>` to run/test, or `complete <id>`.")


def show_info(projects: List[ProjectEntry], task_id: int) -> None:
    project = next((p for p in projects if p.id == task_id), None)
    if project is None:
        raise ValueError(f"Task {task_id} not found.")

    previous = next((p for p in projects if p.id == task_id - 1), None)
    print(f"Task {project.id}: {project.name}")
    print(f"  Path: {project.path}")
    print(f"  Status: {project.status}")
    if project.status in {"locked", "unavailable"}:
        print(f"  Reason: {project.lock_reason(previous) or project.reason}")
    elif project.reason:
        print(f"  Execution info: {project.reason}")
    print(f"  Classification: {project.classification}")
    print(f"  Runnable: {project.runnable}")
    print(f"  Recommendation: {project.recommendation}")
    print(f"  Has README: {project.has_readme}")
    print(f"  Has requirements/env: {project.has_requirements or project.has_env}")
    print(f"  File count: {project.file_count}")
    print(f"  Extensions: {sorted(project.extensions.items())}")


def verify_task_execution(projects: List[ProjectEntry], task_id: int) -> Dict[str, Any]:
    """Run execution test for specified task and update progression state."""
    project = next((p for p in projects if p.id == task_id), None)
    if project is None:
        raise ValueError(f"Task {task_id} not found.")

    if project.status == "unavailable":
        return {"task_id": task_id, "status": "unavailable", "reason": project.reason}

    if project.status == "locked":
        raise ValueError(f"Task {task_id} is locked. Unlock previous tasks first.")

    # Load mapping
    notebook_map = {}
    if MAP_FILE.exists():
        try:
            data = json.loads(MAP_FILE.read_text(encoding="utf-8"))
            notebook_map = {item["project"]: item for item in data.get("quests", [])}
        except Exception:
            pass

    quest_info = notebook_map.get(project.name)
    launch_path_str = quest_info.get("launch_path") if quest_info else None
    if not launch_path_str:
        project.status = "unavailable"
        project.reason = "No launch notebook mapped."
        save_state(projects)
        return {"task_id": task_id, "status": "unavailable", "reason": project.reason}

    target_file = ROOT / launch_path_str
    if not target_file.exists():
        project.status = "failed"
        project.reason = f"Launch file not found: {launch_path_str}"
        save_state(projects)
        return {"task_id": task_id, "status": "failed", "reason": project.reason}

    # Execute in headless mode
    from generate_excel_report import test_headless_execution

    print(f"Testing execution for Task {task_id} ({project.name})...")
    success, err_msg = test_headless_execution(target_file, ROOT / project.path, project.classification)

    if success:
        project.status = "completed"
        project.reason = "All cells executed to completion."
        # Unlock next runnable project
        next_proj = next(
            (p for p in projects if p.id > task_id and p.status not in {"completed", "unavailable"}),
            None,
        )
        if next_proj and next_proj.status == "locked":
            next_proj.status = "unlocked"
        save_state(projects)
        print(f"Task {task_id} PASSED and marked completed!")
        return {"task_id": task_id, "status": "completed", "reason": project.reason}
    else:
        project.status = "failed"
        project.reason = err_msg
        save_state(projects)
        print(f"Task {task_id} FAILED: {err_msg}")
        return {"task_id": task_id, "status": "failed", "reason": project.reason}


def complete_task(projects: List[ProjectEntry], task_id: int) -> None:
    project = next((p for p in projects if p.id == task_id), None)
    if project is None:
        raise ValueError(f"Task {task_id} not found.")
    if project.status == "locked":
        raise ValueError(f"Task {task_id} is locked.")
    if project.status == "unavailable":
        raise ValueError(f"Task {task_id} is unavailable for execution.")

    project.status = "completed"
    project.reason = "Manually marked complete."
    next_project = next(
        (p for p in projects if p.id > task_id and p.status not in {"completed", "unavailable"}),
        None,
    )
    if next_project and next_project.status == "locked":
        next_project.status = "unlocked"
    save_state(projects)
    print(f"Marked task {task_id} as completed.")


def reset_state(projects: List[ProjectEntry]) -> None:
    initialize_state(projects)
    print("Progression state reset. First available task is unlocked.")


def update_and_reload() -> List[ProjectEntry]:
    projects = scan_projects()
    state = ensure_state(projects)
    print("Scanned projects and preserved the current progression state.")
    return state


def print_summary(projects: List[ProjectEntry]) -> None:
    total = len(projects)
    completed = sum(1 for p in projects if p.status == "completed")
    unlocked = sum(1 for p in projects if p.status == "unlocked")
    locked = sum(1 for p in projects if p.status == "locked")
    failed = sum(1 for p in projects if p.status == "failed")
    unavailable = sum(1 for p in projects if p.status == "unavailable")

    print("Progression Summary")
    print("-------------------")
    print(f"Total tasks: {total}")
    print(f"Completed (Passed): {completed}")
    print(f"Unlocked: {unlocked}")
    print(f"Locked: {locked}")
    print(f"Failed: {failed}")
    print(f"Unavailable: {unavailable}")
    print("\nRun `list` to see all tasks or `verify <id>` to run and test a task.")


def export_manifest(projects: List[ProjectEntry]) -> None:
    manifest_file = ROOT / "progress_manifest.json"
    serialized = [asdict(project) for project in projects]
    with manifest_file.open("w", encoding="utf-8") as handle:
        json.dump(serialized, handle, indent=2, ensure_ascii=False)
    print(f"Exported progression manifest to {manifest_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LearnOpenCV progression dashboard")
    parser.add_argument(
        "command",
        nargs="?",
        default="list",
        help="Command: list, info, complete, verify, reset, scan, summary, export",
    )
    parser.add_argument("task_id", nargs="?", type=int, help="Task ID for info, complete, or verify")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projects = scan_projects()
    projects = ensure_state(projects)

    try:
        if args.command == "list":
            list_tasks(projects)
        elif args.command == "info":
            if args.task_id is None:
                raise ValueError("Please specify a task ID for info.")
            show_info(projects, args.task_id)
        elif args.command == "verify":
            if args.task_id is None:
                raise ValueError("Please specify a task ID to verify.")
            verify_task_execution(projects, args.task_id)
        elif args.command == "complete":
            if args.task_id is None:
                raise ValueError("Please specify a task ID to complete.")
            complete_task(projects, args.task_id)
        elif args.command == "reset":
            reset_state(projects)
        elif args.command == "scan":
            update_and_reload()
        elif args.command == "summary":
            print_summary(projects)
        elif args.command == "export":
            export_manifest(projects)
        else:
            raise ValueError(f"Unknown command: {args.command}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
