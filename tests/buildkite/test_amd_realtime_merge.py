# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for selection and fail-closed execution, not GPU qualification."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-merge.yml")
RUNNER = Path(".buildkite/amd/scripts/run-qwen3-omni-rocm-nightly.sh")
MANIFEST = Path(".buildkite/amd/collections/qwen3-omni-realtime.txt")
TARGET = Path("tests/entrypoints/openai_api/test_qwen3_omni_realtime_websocket.py")
LABEL = "MERGE · Qwen3-Omni Realtime WebSocket · 2-GPU"


def _step() -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))
    return next(step for group in pipeline["steps"] for step in group.get("steps", []) if step.get("label") == LABEL)


def test_realtime_merge_job_contract() -> None:
    step = _step()
    assert step["grade"] == "Blocking"
    assert step["agent_pool"] == "mi300_2"
    assert step["depends_on"] == "amd-build"
    assert step["mirror_hardwares"] == ["amdproduction"]
    assert step["timeout_in_minutes"] == 90
    assert step["commands"] == [f"bash {RUNNER.as_posix()}"]
    assert step["artifact_paths"] == ["artifacts/rocm-merge-qwen3-omni-realtime/**/*"]
    env = step["env"]
    assert env["QWEN3_OMNI_EXPECTED_GPUS"] == "2"
    assert env["QWEN3_OMNI_EXPECTED_TESTS"] == "8"
    assert env["QWEN3_OMNI_EXPECTED_RESULTS"] == "0"
    assert env["QWEN3_OMNI_RUN_LEVEL"] == "advanced_model"
    assert env["QWEN3_OMNI_TEST_TARGET"] == TARGET.as_posix()
    assert env["QWEN3_OMNI_EXPECTED_NODES"] == MANIFEST.as_posix()
    assert env["QWEN3_OMNI_MARK_EXPRESSION"] == "advanced_model and omni and rocm and MI325 and cards_2"
    assert env["QWEN3_OMNI_TIMEOUT"] == "75m"
    assert {"entrypoints_function", "omni_qwen3_omni_function", MANIFEST.as_posix()} <= set(
        step["source_file_dependencies"]
    )


def test_realtime_selection_matches_existing_test_decorators(tmp_path: Path) -> None:
    """Collect real decorators/parameter IDs without importing GPU serving fixtures."""
    tree = ast.parse(TARGET.read_text(encoding="utf-8"))
    projected = [
        "import pytest",
        "from types import ModuleType",
        "from tests.helpers.mark import hardware_test",
        "class CPUPlatform:",
        "    def is_cuda(self): return False",
        "    def is_rocm(self): return False",
        "platforms = ModuleType('vllm.platforms')",
        "platforms.current_platform = CPUPlatform()",
        # Limit the stub to GPU availability; use the real hardware-mark helper.
        "patcher = pytest.MonkeyPatch()",
        "patcher.setitem(__import__('sys').modules, 'vllm.platforms', platforms)",
    ]
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name.startswith("realtime_") and name.endswith("_server_params"):
                assert isinstance(node.value, ast.List)
                params = []
                for param in node.value.elts:
                    assert isinstance(param, ast.Call)
                    param_id = next(keyword.value for keyword in param.keywords if keyword.arg == "id")
                    params.append(f"pytest.param(None, id={ast.literal_eval(param_id)!r})")
                projected.append(f"{name} = [{', '.join(params)}]")
        if isinstance(node, ast.ClassDef) and node.name == "TestQwen3OmniRealtimeWebSocket":
            functions = [method for method in node.body if isinstance(method, ast.FunctionDef)]
            for method in functions:
                if method.name.startswith("test_"):
                    method.body = [ast.Pass()]
            node.body = [method for method in functions if method.name.startswith("test_")]
            projected.append(ast.unparse(node))

    collection_target = tmp_path / "test_realtime_contract.py"
    collection_target.write_text("\n".join(projected) + "\n", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--noconftest",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "--collect-only",
            "-q",
            "--color=no",
            "-m",
            _step()["env"]["QWEN3_OMNI_MARK_EXPRESSION"],
            str(collection_target),
        ],
        env={
            **os.environ,
            "PYTHONPATH": os.pathsep.join(filter(None, (str(Path.cwd()), os.environ.get("PYTHONPATH")))),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    selected = [
        TARGET.as_posix() + "::" + line.split("::", 1)[1]
        for line in result.stdout.splitlines()
        if "::TestQwen3OmniRealtimeWebSocket::" in line
    ]
    expected = MANIFEST.read_text(encoding="utf-8").splitlines()
    assert selected == expected
    assert len(selected) == len(set(selected)) == 8
    assert all("without_runtime_jit" not in node for node in selected)


@pytest.mark.parametrize(
    "overrides, expected_status, executes_tests",
    [
        ({}, 0, True),
        ({"QWEN3_OMNI_RUN_LEVEL": "", "QWEN3_OMNI_EXPECTED_NODES": ""}, 0, True),
        ({"QWEN3_OMNI_RUN_LEVEL": "invalid"}, 2, False),
        ({"COLLECTION_MODE": "missing"}, 1, False),
        ({"COLLECTION_MODE": "replaced"}, 1, False),
        ({"COLLECTION_MODE": "duplicate"}, 1, False),
        ({"COLLECTION_STATUS": "2"}, 2, False),
        ({"JUNIT_TESTS": "7"}, 5, True),
        ({"JUNIT_SKIPPED": "1"}, 5, True),
        ({"JUNIT_FAILURES": "1"}, 5, True),
        ({"JUNIT_ERRORS": "1"}, 5, True),
        ({"MISSING_XML": "1"}, 5, True),
        ({"PYTEST_STATUS": "1"}, 1, True),
        ({"TIMEOUT_STATUS": "124"}, 124, False),
        ({"LEAK": "1"}, 1, True),
        ({"LEAK": "1", "PYTEST_STATUS": "2"}, 2, True),
        ({"PS_FAILURE": "1"}, 1, True),
    ],
    ids=[
        "pass",
        "nightly-defaults",
        "invalid-level",
        "missing-node",
        "wrong-node-same-count",
        "duplicate-node-same-count",
        "collection-error",
        "missing-result",
        "skip",
        "junit-failure",
        "junit-error",
        "missing-xml",
        "pytest-failure",
        "timeout",
        "leaked-server",
        "preserve-pytest-error",
        "process-inspection-error",
    ],
)
def test_realtime_runner_fails_closed(
    tmp_path: Path, overrides: dict[str, str], expected_status: int, executes_tests: bool
) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    def executable(name: str, body: str) -> None:
        path = fake_bin / name
        path.write_text("#!/bin/bash\nset -euo pipefail\n" + body, encoding="utf-8", newline="\n")
        path.chmod(0o755)

    for name in ("rocminfo", "rocm-smi", "sleep"):
        executable(name, "exit 0\n")
    executable("git", "echo fixture-head\n")
    executable(
        "python3",
        'if [[ "$#" == 1 && "$1" == - ]]; then cat >/dev/null; exit 0; fi\n'
        'if [[ "${1:-}" == -m ]]; then exit 0; fi\n'
        'exec "$REAL_PYTHON" "$@"\n',
    )
    executable(
        "ps",
        'if [[ -f "$STATE/ps-called" && "$PS_FAILURE" == 1 ]]; then exit 1; fi\n'
        'if [[ -f "$STATE/ps-called" && "$LEAK" == 1 ]]; then\n'
        '  echo "200 1 Mon Nov 2 10:00:01 2026 S python leaked-server"\n'
        'fi\necho "100 1 Mon Nov 2 10:00:00 2026 S bash baseline"\n'
        'touch "$STATE/ps-called"\n',
    )
    executable(
        "timeout",
        'shift 3\nif [[ "$TIMEOUT_STATUS" != 0 ]]; then exit "$TIMEOUT_STATUS"; fi\nexec "$@"\n',
    )
    executable(
        "pytest",
        'printf "%s\\n" "$@" >> "$STATE/pytest-args"\n'
        'if [[ "$1" == --collect-only ]]; then\n'
        '  case "$COLLECTION_MODE" in\n'
        '    missing) head -n 7 "$EXPECTED_NODES" ;;\n'
        '    replaced) head -n 7 "$EXPECTED_NODES"; echo "tests/other.py::test_unexpected" ;;\n'
        '    duplicate) head -n 7 "$EXPECTED_NODES"; head -n 1 "$EXPECTED_NODES" ;;\n'
        '    *) cat "$EXPECTED_NODES" ;;\n'
        '  esac\n  exit "$COLLECTION_STATUS"\nfi\n'
        'touch "$STATE/executed"\n'
        'for arg in "$@"; do\n'
        '  if [[ "$arg" == --junitxml=* && "$MISSING_XML" == 0 ]]; then\n'
        '    printf \'<testsuite tests="%s" skipped="%s" failures="%s" errors="%s"/>\' \\\n'
        '      "$JUNIT_TESTS" "$JUNIT_SKIPPED" "$JUNIT_FAILURES" "$JUNIT_ERRORS" > "${arg#*=}"\n'
        '  fi\ndone\nexit "$PYTEST_STATUS"\n',
    )
    env = {
        **os.environ,
        **_step()["env"],
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "FAKE_BIN": fake_bin.as_posix(),
        "REAL_PYTHON": Path(sys.executable).as_posix(),
        "BUILDKITE_BUILD_CHECKOUT_PATH": tmp_path.as_posix(),
        "QWEN3_OMNI_ARTIFACT_DIR": (tmp_path / "artifacts/realtime").as_posix(),
        "STATE": tmp_path.as_posix(),
        "EXPECTED_NODES": MANIFEST.as_posix(),
        "COLLECTION_MODE": "pass",
        "COLLECTION_STATUS": "0",
        "JUNIT_TESTS": "8",
        "JUNIT_SKIPPED": "0",
        "JUNIT_FAILURES": "0",
        "JUNIT_ERRORS": "0",
        "MISSING_XML": "0",
        "PYTEST_STATUS": "0",
        "TIMEOUT_STATUS": "0",
        "LEAK": "0",
        "PS_FAILURE": "0",
        **overrides,
    }
    command = ["bash", RUNNER.as_posix()]
    if os.name == "nt":
        # Git Bash imports Windows PATH differently; prepend the POSIX path inside Bash.
        command = [
            "bash",
            "-c",
            'export PATH="$(cygpath -u "$FAKE_BIN"):$PATH"; exec bash "$1"',
            "--",
            RUNNER.as_posix(),
        ]
    result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == expected_status, result.stdout + result.stderr
    assert (tmp_path / "executed").exists() == executes_tests
    if overrides.get("QWEN3_OMNI_RUN_LEVEL") != "invalid":
        args = (tmp_path / "pytest-args").read_text()
        assert f"--run-level\n{env['QWEN3_OMNI_RUN_LEVEL'] or 'full_model'}\n" in args
        assert (tmp_path / "artifacts/realtime/process-cleanup.txt").exists()
