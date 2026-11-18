# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_local_tools_package_wins_over_later_regular_package(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    competing_root = tmp_path / "dependency"
    competing_package = competing_root / "tools"
    competing_package.mkdir(parents=True)
    (competing_package / "__init__.py").write_text("SOURCE = 'dependency'\n", encoding="utf-8")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join((str(repo_root), str(competing_root)))
    program = "from pathlib import Path; import tools; print(Path(tools.__file__).resolve())"
    completed = subprocess.run(
        [sys.executable, "-c", program],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )

    assert Path(completed.stdout.strip()) == (repo_root / "tools/__init__.py").resolve()
