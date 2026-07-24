# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Guard package discovery outside repo-root ``sys.path[0]``.

Covers the same failure class as:
- ``vllm ... --omni`` (upstream ``find_spec("vllm_omni")``) — see #5364
- ``python examples/.../end2end.py`` / multimodal client subprocesses, where
  ``sys.path[0]`` is the script directory (not the repo root), so a cwd-only
  layout can start OmniServer via ``python -m vllm_omni...`` and still fail
  example scripts with ``ModuleNotFoundError: No module named 'vllm_omni'``.
"""

from __future__ import annotations

import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Modules imported at the top of CI-failing example entrypoints.
_EXAMPLE_IMPORT_PROBES = (
    "vllm_omni",
    "vllm_omni.entrypoints.cli.main",
    "vllm_omni.utils.tracking_parser",
    "vllm_omni.diffusion.models.hunyuan_image3.prompt_utils",
)


def _assert_subprocess_ok(proc: subprocess.CompletedProcess[str], *, detail: str) -> None:
    assert proc.returncode == 0, f"{detail} (exit={proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"


def _write_import_probe_script(path: Path, modules: tuple[str, ...] = _EXAMPLE_IMPORT_PROBES) -> None:
    """Write a tiny script that imports the same modules example entrypoints need."""
    # Build at column 0 — do not mix indented triple-quotes with unindented
    # ``import_lines`` under textwrap.dedent (common indent becomes 0 and the
    # surrounding lines keep leading spaces → IndentationError).
    lines = [
        "from importlib.util import find_spec",
        "import sys",
        'assert find_spec("vllm_omni") is not None, (sys.path[0], sys.path[:5])',
        *(f"import {name}" for name in modules),
        'print("ok", sys.path[0])',
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def test_find_spec_vllm_omni() -> None:
    """Same probe as upstream ``vllm`` CLI before delegating to omni_main."""
    assert find_spec("vllm_omni") is not None
    assert find_spec("vllm_omni.entrypoints.cli.main") is not None


def test_find_spec_vllm_omni_from_non_repo_cwd(tmp_path: Path) -> None:
    """Console-script style: discovery must work when cwd is not the repo root."""
    code = (
        "from importlib.util import find_spec; "
        "assert find_spec('vllm_omni') is not None; "
        "assert find_spec('vllm_omni.entrypoints.cli.main') is not None"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_subprocess_ok(proc, detail=f"find_spec failed from cwd={tmp_path}")


def test_python_script_sys_path0_is_script_dir_not_repo(tmp_path: Path) -> None:
    """``python path/to/script.py`` puts the script dir on ``sys.path[0]``, not repo root."""
    script = tmp_path / "examples" / "offline_inference" / "hunyuan_image3" / "end2end_probe.py"
    _write_import_probe_script(script)
    # cwd intentionally elsewhere so neither cwd nor sys.path[0] is the repo.
    cwd = tmp_path / "run_cwd"
    cwd.mkdir()
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_subprocess_ok(
        proc,
        detail=f"import failed for python {script} with cwd={cwd} (example end2end-style)",
    )
    assert str(script.parent) in proc.stdout


def test_python_online_serving_client_style_script(tmp_path: Path) -> None:
    """Online example client layout: ``python examples/online_serving/<client>.py``."""
    script = (
        tmp_path / "examples" / "online_serving" / "openai_chat_completion_client_for_multimodal_generation_probe.py"
    )
    _write_import_probe_script(script)
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path / "examples" / "online_serving",
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_subprocess_ok(
        proc,
        detail=f"import failed for online-serving client-style script {script}",
    )


@pytest.mark.parametrize(
    "module",
    [
        "vllm_omni.utils.tracking_parser",
        "vllm_omni.diffusion.models.hunyuan_image3.prompt_utils",
    ],
)
def test_example_entrypoint_modules_importable_from_script_dir(tmp_path: Path, module: str) -> None:
    """Modules from CI-failing example top-level imports must resolve off-repo."""
    script = tmp_path / "orphan_dir" / "probe.py"
    _write_import_probe_script(script, modules=(module,))
    proc = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path / "orphan_dir",
        capture_output=True,
        text=True,
        check=False,
    )
    _assert_subprocess_ok(proc, detail=f"failed to import {module!r} from script dir {script.parent}")
