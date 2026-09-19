#!/usr/bin/env python3
"""Build a deterministic Jupyter launch map for every LearnOpenCV project."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nbformat

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / ".cvquest" / "notebooks"
MAP_FILE = ROOT / ".cvquest" / "quest_notebooks.json"
EXCLUDED = {
    ".git",
    ".venv",
    ".venv-jupyter",
    ".cvquest",
    "__pycache__",
    ".ipynb_checkpoints",
    "build",
    "dist",
}

OFFICIAL_SOURCES: Dict[str, Tuple[str, str]] = {
    "2D_Gaussian_Splatting_Geometrically_Accurate_Radiance_Field_Reconstruction": (
        "Official 2D Gaussian Splatting implementation",
        "https://github.com/hbb1/2d-gaussian-splatting",
    ),
    "DUSt3R-Dense-3D-Reconstruction": (
        "Official DUSt3R implementation",
        "https://github.com/naver/dust3r",
    ),
    "MASt3R-SLAM": (
        "Official MASt3R-SLAM implementation",
        "https://github.com/rmurai0610/MASt3R-SLAM",
    ),
    "Feature-Matching-Using-Neural-Networks": (
        "LearnOpenCV feature matching tutorial",
        "https://learnopencv.com/feature-matching/introduction-to-feature-matching-in-neural-networks/",
    ),
    "ComfyUI": ("Official ComfyUI documentation", "https://docs.comfy.org/"),
    "How_to_Build_a_GitHub_Code_Analyser_Agent_for_Developer_Productivity": (
        "LangChain code agent documentation",
        "https://docs.langchain.com/oss/python/deepagents/code/overview",
    ),
    "Install-OpenCV-Windows-exe": (
        "Official OpenCV installation overview",
        "https://docs.opencv.org/5.0/tutorials/introduction/general_install/general_install.html",
    ),
    "CI": ("GitHub Actions documentation", "https://docs.github.com/actions"),
    "docs": ("JupyterLab documentation", "https://jupyterlab.readthedocs.io/en/stable/"),
    "speech-to-speech": ("OpenAI Realtime API documentation", "https://platform.openai.com/docs/guides/realtime"),
    "industrial_cv_TensorRT_python": (
        "NVIDIA TensorRT Python documentation",
        "https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html",
    ),
}


def projects() -> List[Path]:
    """Retrieve all non-excluded project directories in repository root."""
    return sorted(
        p for p in ROOT.iterdir()
        if p.is_dir() and p.name not in EXCLUDED and not p.name.startswith(".")
    )


def representative(project: Path) -> Tuple[str, Optional[Path]]:
    """Determine the representative notebook/source file for a project."""
    notebooks = sorted(
        [p for p in project.rglob("*.ipynb") if ".ipynb_checkpoints" not in p.parts],
        key=lambda p: (len(p.relative_to(project).parts), str(p).lower()),
    )
    if notebooks:
        return "existing-notebook", notebooks[0]

    python_files = [
        p for p in project.rglob("*.py")
        if "__pycache__" not in p.parts and not p.name.startswith(".")
    ]
    preferred = {"main.py", "demo.py", "example.py", "app.py", "inference.py", "test.py"}
    python_files.sort(
        key=lambda p: (0 if p.name.lower() in preferred else 1, len(p.relative_to(project).parts), str(p).lower())
    )
    if python_files:
        return "python-wrapper", python_files[0]

    cpp_files = sorted(
        project.rglob("*.cpp"), key=lambda p: (len(p.relative_to(project).parts), str(p).lower())
    )
    if cpp_files:
        return "cpp-wrapper", cpp_files[0]
    return "starter", None


def readme_excerpt(project: Path, limit: int = 5000) -> str:
    """Extract clean text preview from README files in project directory."""
    readmes = sorted(project.glob("README*"))
    if not readmes:
        return "이 프로젝트에는 README가 없습니다. 아래 준비 셀부터 실행하세요."
    text = readmes[0].read_text(encoding="utf-8", errors="replace")
    text = re.sub(r"<[^>]+>", "", text).strip()
    return text[:limit]


def setup_cell(project_name: str, source_path: Optional[str]) -> str:
    source_line = f"SOURCE = PROJECT / {source_path!r}" if source_path else "SOURCE = None"
    return f"""from pathlib import Path

here = Path.cwd().resolve()
ROOT = next(p for p in [here, *here.parents] if (p / "learnopencv_progression.py").exists())
PROJECT = ROOT / {project_name!r}
{source_line}
print("Repository:", ROOT)
print("Project:", PROJECT)
print("Source:", SOURCE if SOURCE else "starter example")"""


def starter_code(project_name: str) -> str:
    clean_name = project_name[:32].replace('"', '\\"')
    return f"""import cv2
import numpy as np
import matplotlib.pyplot as plt

# 모든 보충 퀘스트가 로컬에서 바로 실행되는지 확인하는 작은 CV 예제입니다.
canvas = np.zeros((320, 640, 3), dtype=np.uint8)
cv2.rectangle(canvas, (30, 45), (610, 275), (36, 92, 69), -1)
cv2.circle(canvas, (150, 160), 72, (117, 239, 216), -1)
cv2.putText(canvas, "{clean_name}", (245, 170),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
edges = cv2.Canny(canvas, 80, 160)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
axes[0].set_title("Synthetic input")
axes[1].imshow(edges, cmap="gray")
axes[1].set_title("Canny edges")
for axis in axes:
    axis.axis("off")
plt.show()
print("OpenCV", cv2.__version__, "— starter execution succeeded")"""


def make_wrapper(index: int, project: Path, kind: str, source: Optional[Path]) -> Path:
    """Generate a Jupyter notebook wrapper for projects lacking a primary notebook."""
    relative_source = str(source.relative_to(project)) if source else None
    # Strip any BOM from project name for source lookup
    clean_proj_name = project.name.strip("\ufeff")
    source_info = OFFICIAL_SOURCES.get(clean_proj_name)
    attribution = (
        f"\n\n## 참고 출처\n[{source_info[0]}]({source_info[1]})"
        if source_info
        else "\n\n## 출처\n이 저장소의 프로젝트 README와 원본 소스 코드"
    )
    intro = (
        f"# Quest {index:03d} · {project.name}\n\n"
        "이 노트북은 CV Quest가 저장소의 기존 학습 자료를 실행 가능하게 연결한 래퍼입니다. "
        "위에서 아래 순서로 셀을 실행하세요. 모델·데이터·GPU가 필요한 프로젝트는 README 요구사항을 먼저 확인해야 합니다."
        f"{attribution}\n\n## 프로젝트 안내\n{readme_excerpt(project)}"
    )
    cells = [
        nbformat.v4.new_markdown_cell(intro),
        nbformat.v4.new_code_cell(setup_cell(project.name, relative_source)),
    ]

    if kind == "python-wrapper":
        cells.extend([
            nbformat.v4.new_markdown_cell("## 대표 Python 소스 미리보기"),
            nbformat.v4.new_code_cell(
                "code = SOURCE.read_text(encoding='utf-8', errors='replace')\n"
                "print(code[:12000])\n"
                "print(f'\\n--- {len(code)} characters total ---')"
            ),
            nbformat.v4.new_markdown_cell(
                "## 실행\n아래 셀은 원본 스크립트를 프로젝트 폴더에서 실행합니다. "
                "웹캠·GUI·모델 다운로드를 사용하는 예제는 중지 버튼으로 종료할 수 있습니다."
            ),
            nbformat.v4.new_code_cell(
                "import os, runpy\n"
                "previous = Path.cwd()\n"
                "os.chdir(PROJECT)\n"
                "try:\n"
                "    runpy.run_path(str(SOURCE), run_name='__main__')\n"
                "finally:\n"
                "    os.chdir(previous)"
            ),
        ])
    elif kind == "cpp-wrapper":
        cells.extend([
            nbformat.v4.new_markdown_cell("## 대표 C++ 소스"),
            nbformat.v4.new_code_cell("print(SOURCE.read_text(encoding='utf-8', errors='replace')[:16000])"),
            nbformat.v4.new_markdown_cell(
                "## 빌드 안내\n프로젝트의 CMakeLists.txt가 있으면 아래 셀로 구성과 빌드를 시도합니다. "
                "OpenCV C++ 개발 라이브러리가 시스템에 설치되어 있어야 합니다."
            ),
            nbformat.v4.new_code_cell(
                "import subprocess\n"
                "build = ROOT / '.cvquest' / 'build' / PROJECT.name\n"
                "build.mkdir(parents=True, exist_ok=True)\n"
                "if (PROJECT / 'CMakeLists.txt').exists():\n"
                "    subprocess.run(['cmake', '-S', str(PROJECT), '-B', str(build)], check=True)\n"
                "    subprocess.run(['cmake', '--build', str(build)], check=True)\n"
                "else:\n"
                "    print('CMakeLists.txt가 없습니다. README의 컴파일 명령을 사용하세요.')"
            ),
        ])
    else:
        cells.extend([
            nbformat.v4.new_markdown_cell(
                "## 로컬 실행 스모크 테스트\n원 프로젝트가 문서 중심이므로 공통 OpenCV 예제로 커널과 시각화 환경을 검증합니다. "
                "상단 공식 출처에서 전체 구현 요구사항을 확인하세요."
            ),
            nbformat.v4.new_code_cell(starter_code(project.name)),
        ])

    notebook = nbformat.v4.new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python (LearnOpenCV Quest)", "language": "python", "name": "learnopencv-quest"},
            "language_info": {"name": "python", "version": "3.11"},
            "cvquest": {
                "project": project.name,
                "kind": kind,
                "source": relative_source,
                "official_source": source_info[1] if source_info else None,
            },
        },
    )
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", project.name).strip("-")[:90]
    target = OUT_DIR / f"{index:03d}-{safe_name}.ipynb"
    nbformat.write(notebook, target)
    return target


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    entries: List[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    for index, project in enumerate(projects(), start=1):
        kind, source = representative(project)
        launch = source if kind == "existing-notebook" else make_wrapper(index, project, kind, source)
        counts[kind] = counts.get(kind, 0) + 1
        clean_proj_name = project.name.strip("\ufeff")
        entries.append({
            "id": index,
            "project": project.name,
            "kind": kind,
            "launch_path": str(launch.relative_to(ROOT)),
            "source_path": str(source.relative_to(ROOT)) if source else None,
            "official_source": OFFICIAL_SOURCES.get(clean_proj_name, (None, None))[1],
        })
    MAP_FILE.parent.mkdir(parents=True, exist_ok=True)
    MAP_FILE.write_text(
        json.dumps({"version": 1, "counts": counts, "quests": entries}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"total": len(entries), "counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
