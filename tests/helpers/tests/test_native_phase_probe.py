# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import ast
import json
import os
import shlex
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
_ROOT = Path(__file__).resolve().parents[3]
_PLUGIN = "tests.helpers.native_phase_probe"
_CASES = """
import gc
import pytest

@pytest.fixture
def lifecycle():
    gc.collect()
    yield
    gc.collect()

def test_pass(lifecycle):
    assert True

@pytest.mark.parametrize("value", ["PARAMETER_PAYLOAD_DO_NOT_RECORD"])
def test_skip(value):
    pytest.skip("SKIP_PAYLOAD_DO_NOT_RECORD")

def test_fail():
    raise AssertionError("FAILURE_PAYLOAD_DO_NOT_RECORD")
"""
_AUDIT = """
import atexit
import gc
import json
import os
from pathlib import Path

data = {"collection": [], "reports": []}
def existing_callback(phase, info):
    pass
gc.callbacks.append(existing_callback)
initial = tuple(gc.callbacks)
def pytest_collection_modifyitems(items):
    data["collection"] = [item.nodeid for item in items]
def pytest_runtest_logreport(report):
    data["reports"].append([report.nodeid, report.when, report.outcome])
@atexit.register
def save():
    data["callbacks_restored"] = tuple(gc.callbacks) == initial
    Path(os.environ["PROBE_TEST_AUDIT_OUTPUT"]).write_text(json.dumps(data))
"""


@pytest.fixture
def project(tmp_path: Path):
    root = tmp_path / "project"
    helpers = root / "tests" / "helpers"
    helpers.mkdir(parents=True)
    for path in (root / "tests" / "__init__.py", helpers / "__init__.py"):
        path.write_text("")
    (helpers / "native_phase_probe.py").write_text((_ROOT / "tests/helpers/native_phase_probe.py").read_text())
    (root / "tests/test_cases.py").write_text(textwrap.dedent(_CASES))
    (root / "probe_audit.py").write_text(textwrap.dedent(_AUDIT))
    (root / "pytest.ini").write_text("[pytest]\n")
    output = tmp_path / "output"
    output.mkdir(mode=0o700)
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("NATIVE_PHASE_PROBE", "VLLM_OMNI_NATIVE_PHASE_PROBE"))
    }
    environment.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=str(root),
        NATIVE_PHASE_PROBE_OUTPUT_DIR=str(output),
        NATIVE_PHASE_PROBE_SLOW_SECONDS="0",
    )
    return root, output, environment


def _run(project, *, observed: bool, extra: tuple[str, ...] = ()):
    root, output, environment = project
    audit = root.parent / ("observed-audit.json" if observed else "baseline-audit.json")
    args = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "probe_audit", "-q"]
    if observed:
        args += ["-p", _PLUGIN]
    result = subprocess.run(
        args + list(extra),
        cwd=root,
        env={**environment, "PROBE_TEST_AUDIT_OUTPUT": str(audit)},
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert audit.exists(), result.stdout + result.stderr
    return result, json.loads(audit.read_text())


@pytest.mark.parametrize("collect_only", [False, True])
def test_probe_preserves_collection_reports_exit_and_gc_cleanup(project, collect_only):
    extra = ("--collect-only",) if collect_only else ()
    baseline, baseline_audit = _run(project, observed=False, extra=extra)
    assert list(project[1].iterdir()) == []
    observed, observed_audit = _run(project, observed=True, extra=extra)
    assert baseline.returncode == observed.returncode == (0 if collect_only else 1)
    assert baseline_audit == observed_audit
    assert observed_audit["callbacks_restored"]
    (directory,) = project[1].iterdir()
    summary = json.loads((directory / "summary.json").read_text())
    assert summary["collected"] == len(observed_audit["collection"]) == 3
    assert summary["exitstatus"] == observed.returncode
    assert summary["gc_callback_removed"]
    assert summary["observer_errors"] == 0
    assert sum(stats["reports"] for stats in summary["phases"].values()) == len(observed_audit["reports"])
    if not collect_only:
        assert summary["gc_by_generation"]["2"]["count"] >= 2
        assert summary["phases"]["call"]["outcomes"] == {"passed": 1, "failed": 1, "skipped": 1}
    for path in directory.iterdir():
        assert "PAYLOAD_DO_NOT_RECORD" not in path.read_text()


@pytest.mark.parametrize("mode", ["off", "wrong_parent", "missing_parent", "missing_output", "invalid_output", "on"])
def test_registration_token_fences_nested_pytest_sessions(project, mode):
    root, output, environment = project
    # Exercise the real loader hook without importing unrelated runtime fixtures.
    source = (_ROOT / "tests/conftest.py").read_text()
    hook = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "pytest_configure"
    )
    hook_source = ast.get_source_segment(source, hook)
    assert hook_source is not None
    (root / "tests/conftest.py").write_text("import os\n" + hook_source + "\n")
    (root / "nested_gate.py").write_text(
        "def pytest_sessionstart(session):\n"
        "    assert not session.config.pluginmanager.hasplugin('native-phase-probe')\n"
    )
    (root / "tests/test_cases.py").write_text(
        textwrap.dedent("""
        import subprocess
        import sys
        import pytest
        def test_nested():
            args = ["-p", "nested_gate", "--collect-only", "-q"]
            child = subprocess.run([sys.executable, "-m", "pytest", *args], timeout=15)
            assert child.returncode == 0
            assert pytest.main(args) == 0
    """)
    )
    if mode != "off":
        environment["VLLM_OMNI_NATIVE_PHASE_PROBE"] = "1"
    environment["NATIVE_PHASE_PROBE_PARENT_PID"] = str(os.getpid()) if mode == "on" else "-1"
    if mode == "missing_parent":
        environment.pop("NATIVE_PHASE_PROBE_PARENT_PID")
    if mode in {"missing_output", "invalid_output"}:
        environment["NATIVE_PHASE_PROBE_PARENT_PID"] = str(os.getpid())
        if mode == "missing_output":
            environment.pop("NATIVE_PHASE_PROBE_OUTPUT_DIR")
        else:
            environment["NATIVE_PHASE_PROBE_OUTPUT_DIR"] = str(output / "absent")
    result, _ = _run(project, observed=False)
    assert result.returncode == 0, result.stdout + result.stderr
    log = result.stdout + result.stderr
    if mode == "off":
        assert "native-phase-probe:" not in log
    else:
        assert "native-phase-probe: gate pid=" in log
        assert f"token_present={mode != 'missing_parent'}" in log
        assert f"token_matches={mode in {'on', 'missing_output', 'invalid_output'}}" in log
        assert f"output_configured={mode != 'missing_output'}" in log
        assert ("native-phase-probe: started pid=" in log) == (mode == "on")
        assert ("disabled; output unavailable or invalid" in log) == (mode == "invalid_output")
    directories = list(output.iterdir())
    assert len(directories) == (1 if mode == "on" else 0)
    if directories:
        summary = json.loads((directories[0] / "summary.json").read_text())
        assert summary["collected"] == 1
        assert summary["phases"]["call"]["outcomes"]["passed"] == 1


@pytest.mark.skipif(os.name != "posix", reason="SIGKILL requires POSIX")
def test_sigkill_retains_last_complete_checkpoint_within_budget(project):
    root, output, environment = project
    (root / "tests/test_cases.py").write_text(
        textwrap.dedent("""
        import gc
        import pytest
        @pytest.mark.parametrize("index", range(20))
        def test_prefix(index):
            gc.collect()
    """)
    )
    ready = root.parent / "checkpoint-replacement-ready"
    (root / "delay_checkpoint.py").write_text(
        textwrap.dedent("""
        import os
        import time
        from pathlib import Path
        real_replace = os.replace
        replacements = 0
        def delay_replace(source, destination):
            global replacements
            if Path(destination).name == "checkpoint.json":
                replacements += 1
                if replacements == 4:
                    Path(os.environ["PROBE_TEST_READY"]).write_text("ready")
                    while True:
                        time.sleep(0.01)
            return real_replace(source, destination)
        os.replace = delay_replace
    """)
    )
    environment.update(
        NATIVE_PHASE_PROBE_MAX_EVENTS="2000",
        NATIVE_PHASE_PROBE_MAX_BYTES="16384",
        NATIVE_PHASE_PROBE_CHECKPOINT_REPORTS="10",
        PROBE_TEST_READY=str(ready),
    )
    process = subprocess.Popen(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-p", "delay_checkpoint", "-p", _PLUGIN, "-q"],
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 20
        while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "Child did not reach the controlled checkpoint replacement"
        (directory,) = output.iterdir()
        complete = (directory / "checkpoint.json").read_bytes()
        checkpoint = json.loads(complete)
        pending = json.loads((directory / "checkpoint.tmp").read_text())
        assert checkpoint["observed_prefix_reports"] == 30
        assert checkpoint["checkpoint_sequence"] == 3
        assert pending["observed_prefix_reports"] == 40
        assert checkpoint["omissions"]["byte_limit"] > 0
        assert checkpoint["eligible_phase_events_omitted"] > 0
        assert not checkpoint["final"]
        assert all(checkpoint["phases"][phase]["reports"] == 10 for phase in ("setup", "call", "teardown"))
        assert checkpoint["gc_by_generation"]["2"]["count"] >= 10
        assert sum(path.stat().st_size for path in directory.iterdir()) <= 16384
        process.kill()
        stdout, stderr = process.communicate(timeout=10)
        assert process.returncode == -signal.SIGKILL
        assert "native-phase-probe: started pid=" in stdout + stderr
        assert (directory / "checkpoint.json").read_bytes() == complete
        assert not (directory / "summary.json").exists()
        assert sum(path.stat().st_size for path in directory.iterdir()) <= 16384
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)


@pytest.mark.skipif(os.name != "posix", reason="The native CPU agent requires POSIX shells")
@pytest.mark.parametrize("shell", ["sh", "bash"])
@pytest.mark.parametrize(("enabled", "exit_status"), [(False, 0), (True, 0), (True, 7)])
def test_native_pipeline_script_preserves_parent_token_and_pytest_status(tmp_path: Path, shell, enabled, exit_status):
    source = (_ROOT / ".buildkite/cuda/test-nightly.yml").read_text()
    step = source.split("key: nightly-simple-cpu-coverage\n", 1)[1].split("mirror_hardwares:", 1)[0]
    (command,) = [line.strip()[2:] for line in step.splitlines() if line.strip().startswith("- bash ")]
    # The uploaded command contains no shell variables for Kubernetes to expand.
    assert "$" not in command
    args = shlex.split(command)
    assert args[0] == "bash"
    args[1] = str(_ROOT / args[1])
    receipt = tmp_path / "pytest-receipt.json"
    output = tmp_path / "observer-output"
    output.mkdir()
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    pytest_stub = bin_dir / "pytest"
    pytest_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "report = next(arg.removeprefix('--cov-report=xml:') for arg in args if arg.startswith('--cov-report=xml:'))\n"
        "Path(report).write_text('<testsuites/>')\n"
        "token = os.environ.get('NATIVE_PHASE_PROBE_PARENT_PID')\n"
        "Path(os.environ['PROBE_TEST_RECEIPT']).write_text(json.dumps(dict(\n"
        "    args=args, token=token, parent=os.getppid(), output=os.environ.get('NATIVE_PHASE_PROBE_OUTPUT_DIR'))))\n"
        "sys.exit(int(os.environ['PROBE_TEST_EXIT']))\n"
    )
    pytest_stub.chmod(0o755)
    mktemp_stub = bin_dir / "mktemp"
    mktemp_stub.write_text(f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(str(output))}\n")
    mktemp_stub.chmod(0o755)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("NATIVE_PHASE_PROBE_")}
    environment.update(
        PATH=str(bin_dir) + os.pathsep + environment["PATH"],
        VLLM_OMNI_NATIVE_PHASE_PROBE=str(int(enabled)),
        BUILDKITE_STEP_ID="shell-regression",
        PROBE_TEST_RECEIPT=str(receipt),
        PROBE_TEST_EXIT=str(exit_status),
    )
    result = subprocess.run(
        [shell, "-e", "-c", shlex.join(args)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == exit_status, result.stdout + result.stderr
    observed = json.loads(receipt.read_text())
    assert observed["args"] == [
        "-sv",
        "tests/",
        "-m",
        "core_model and cpu",
        "--cov=vllm_omni",
        "--cov-report=term-missing:skip-covered",
        "--cov-report=xml:coverage-core-model-cpu-shell-regression.xml",
    ]
    assert (tmp_path / "coverage-core-model-cpu-shell-regression.xml.gz").is_file()
    if enabled:
        assert observed["token"] == str(observed["parent"])
        assert observed["output"] == str(output)
        assert f"shell enabled=1 pid={observed['parent']}" in result.stderr
    else:
        assert observed["token"] is None
        assert observed["output"] is None
        assert "shell enabled=0" in result.stderr
