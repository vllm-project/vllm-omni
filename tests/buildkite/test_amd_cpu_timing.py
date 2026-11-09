# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Exercise the opt-in runner without importing vLLM or requiring GPU hardware."""

import json
import os
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / ".buildkite/amd/scripts/run_cpu_suite.py"
SPEC = spec_from_file_location("cpu_suite_timing", RUNNER)
assert SPEC is not None and SPEC.loader is not None
TIMING = module_from_spec(SPEC)
SPEC.loader.exec_module(TIMING)


@pytest.fixture
def suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "")
    monkeypatch.setenv("BUILDKITE_BUILD_CHECKOUT_PATH", str(tmp_path))
    monkeypatch.setenv("BUILDKITE_BUILD_NUMBER", "123")
    monkeypatch.setenv("BUILDKITE_JOB_ID", "job-one")
    monkeypatch.setenv("BUILDKITE_PARALLEL_JOB", "0")
    monkeypatch.setenv("BUILDKITE_COMMIT", "abc123")
    (tmp_path / "pytest.ini").write_text("[pytest]\nmarkers =\n    core_model\n    cpu\n", encoding="utf-8")
    return tmp_path


def run_suite(suite: Path, source: str, *args: str) -> tuple[subprocess.CompletedProcess, Path]:
    (suite / "test_sample.py").write_text(source, encoding="utf-8")
    before = set((suite / "artifacts/amd-cpu-suites").glob("*"))
    result = subprocess.run(
        [sys.executable, str(RUNNER), "other", "-q", "test_sample.py", *args],
        cwd=suite,
        capture_output=True,
        text=True,
        timeout=30,
    )
    created = set((suite / "artifacts/amd-cpu-suites").glob("*")) - before
    assert len(created) == 1, result.stdout + result.stderr
    return result, created.pop()


def read_json(directory: Path, name: str) -> dict:
    return json.loads((directory / name).read_text(encoding="utf-8"))


def test_phase_totals_outcomes_and_context(suite: Path) -> None:
    source = """
import pytest
@pytest.fixture
def bad_setup():
    raise RuntimeError('setup failure')
@pytest.fixture
def bad_teardown():
    yield
    raise RuntimeError('teardown failure')
def test_pass(): pass
def test_fail(): assert False
@pytest.mark.skip(reason='intentional')
def test_skip(): pass
def test_setup_error(bad_setup): pass
def test_teardown_error(bad_teardown): pass
"""
    result, directory = run_suite(suite, source)
    assert result.returncode == 1, result.stdout + result.stderr
    summary = read_json(directory, "summary.json")
    assert summary["counts"] == {
        "collected": 5,
        "selected": 5,
        "not_selected": 0,
        "deselected": 0,
        "started": 5,
        "executed": 3,
        "collection_errors": 0,
        "passed": 1,
        "failed": 3,
        "skipped": 1,
        "incomplete": 0,
    }
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
    reports = [event for event in events if event["event"] == "test_report"]
    for phase in ("setup", "call", "teardown"):
        assert summary["seconds"][phase] == pytest.approx(sum(r["seconds"] for r in reports if r["phase"] == phase))
    assert summary["seconds"]["collection"] >= 0
    assert summary["seconds"]["runner_wall"] >= summary["seconds"]["collection"]
    assert summary["pytest_exit_code"] == 1
    assert summary["collection_finished"]
    assert summary["artifact_errors"] == []
    assert len(ET.parse(directory / "pytest.xml").findall(".//testcase")) == 5
    assert "cpu_max" in read_json(directory, "cpu-before.json")
    assert "thread_environment" in read_json(directory, "cpu-after.json")
    invocation = read_json(directory, "invocation.json")
    assert invocation["buildkite"]["COMMIT"] == "abc123"
    assert invocation["image_digest"] is None


@pytest.mark.parametrize("shard", range(3))
def test_selection_matches_pytest_shard(suite: Path, shard: int) -> None:
    source = (
        "import pytest\n@pytest.mark.cpu\n@pytest.mark.parametrize('number', range(12))\ndef test_case(number): pass\n"
    )
    args = (
        "-p",
        "pytest_shard.pytest_shard",
        "--num-shards=3",
        f"--shard-id={shard}",
        "-m",
        "cpu",
        "--deselect=test_sample.py::test_case[0]",
    )
    result, directory = run_suite(suite, source, *args)
    assert result.returncode == 0, result.stdout + result.stderr
    baseline = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "test_sample.py", *args],
        cwd=suite,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    nodes = [line for line in baseline.stdout.splitlines() if line.startswith("test_sample.py::")]
    selection = read_json(directory, "selection.json")
    assert selection["selected"] == nodes
    assert len(selection["collected"]) == 12
    assert set(selection["collected"]) == set(selection["selected"]) | set(selection["not_selected"])
    assert read_json(directory, "summary.json")["counts"]["executed"] == len(nodes)
    assert read_json(directory, "invocation.json")["shard_id"] == shard


@pytest.mark.parametrize(
    ("source", "args", "status"),
    [
        ("def test_ok(): pass\n", (), 0),
        ("def test_bad(): assert False\n", (), 1),
        ("raise RuntimeError('collection failure')\n", (), 2),
        ("def test_ok(): pass\n", ("-k", "absent"), 5),
    ],
)
def test_pytest_exit_codes(suite: Path, source: str, args: tuple[str, ...], status: int) -> None:
    result, directory = run_suite(suite, source, *args)
    assert result.returncode == status, result.stdout + result.stderr
    summary = read_json(directory, "summary.json")
    assert summary["pytest_exit_code"] == status
    if status in (2, 5):
        assert summary["counts"]["executed"] == 0
    if status == 2:
        assert summary["counts"]["collection_errors"] == 1


def test_repeated_invocations_do_not_overwrite_artifacts(suite: Path) -> None:
    first, first_dir = run_suite(suite, "def test_first(): pass\n")
    second, second_dir = run_suite(suite, "def test_second(): pass\n")
    assert first.returncode == second.returncode == 0
    assert first_dir != second_dir
    assert first_dir.name.startswith("other-123-job-one-0-")
    assert read_json(first_dir, "selection.json")["selected"] == ["test_sample.py::test_first"]
    assert read_json(second_dir, "selection.json")["selected"] == ["test_sample.py::test_second"]


def test_sessionfinish_persists_artifacts_before_pytest_unconfigure_exit(suite: Path) -> None:
    (suite / "conftest.py").write_text(
        "def pytest_unconfigure(config):\n    raise SystemExit(0)\n",
        encoding="utf-8",
    )
    result, directory = run_suite(suite, "def test_ok(): pass\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert read_json(directory, "summary.json")["pytest_exit_code"] == 0
    assert read_json(directory, "summary.json")["counts"]["passed"] == 1
    assert "thread_environment" in read_json(directory, "cpu-after.json")


def test_artifact_directory_failure_stops_before_execution(suite: Path) -> None:
    (suite / "artifacts").write_text("not a directory", encoding="utf-8")
    (suite / "test_sample.py").write_text("raise AssertionError('must not collect')", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(RUNNER), "other", "-q", "test_sample.py"],
        cwd=suite,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1
    assert "Cannot create CPU timing artifacts" in result.stderr
    assert "must not collect" not in result.stdout + result.stderr


@pytest.mark.parametrize("status", [0, 7])
def test_summary_write_failure_preserves_test_failure(suite: Path, status: int) -> None:
    source = f"""
from pathlib import Path
import pytest
def test_artifact_error():
    directory = next(Path('artifacts/amd-cpu-suites').iterdir())
    (directory / 'summary.json').mkdir()
    if {status}:
        pytest.exit('intentional exit', returncode={status})
"""
    result, _ = run_suite(suite, source)
    assert result.returncode == (status or 1), result.stdout + result.stderr
    assert "CPU timing artifact error: summary.json" in result.stderr


def test_interrupted_test_is_incomplete(tmp_path: Path) -> None:
    recorder = TIMING.CpuSuiteArtifacts(tmp_path)
    recorder.pytest_runtest_logstart("test_sample.py::test_interrupted")
    (tmp_path / "pytest.xml").write_text("<testsuites/>", encoding="utf-8")
    assert recorder.finish(2) == 2
    counts = read_json(tmp_path, "summary.json")["counts"]
    assert counts["started"] == counts["incomplete"] == 1
    assert counts["executed"] == counts["passed"] == 0


@pytest.mark.parametrize("xml", [None, "<broken", "<invalid/>"])
@pytest.mark.parametrize("status", [0, 2])
def test_missing_or_invalid_junit_fails_closed(tmp_path: Path, xml: str | None, status: int) -> None:
    recorder = TIMING.CpuSuiteArtifacts(tmp_path)
    if xml is not None:
        (tmp_path / "pytest.xml").write_text(xml, encoding="utf-8")
    assert recorder.finish(status) == (status or 1)
    assert read_json(tmp_path, "summary.json")["artifact_errors"]


@pytest.mark.skipif(os.name != "posix", reason="CI sends POSIX signals")
def test_sigterm_leaves_incremental_diagnostics(suite: Path) -> None:
    (suite / "test_sample.py").write_text(
        "import time\ndef test_wait():\n    while True: time.sleep(1)\n", encoding="utf-8"
    )
    with subprocess.Popen(
        [sys.executable, str(RUNNER), "other", "-q", "test_sample.py"],
        cwd=suite,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ) as process:
        try:
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                paths = list((suite / "artifacts/amd-cpu-suites").glob("*/events.jsonl"))
                if paths and '"phase": "setup"' in paths[0].read_text():
                    break
                assert process.poll() is None
                time.sleep(0.05)
            else:
                pytest.fail("runner did not reach test setup")
            process.terminate()
            assert process.wait(timeout=10) == -signal.SIGTERM
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    directory = paths[0].parent
    assert read_json(directory, "selection.json")["selected"] == ["test_sample.py::test_wait"]
    events = [json.loads(line) for line in paths[0].read_text().splitlines()]
    assert any(event["event"] == "test_start" for event in events)
    assert not (directory / "summary.json").exists()


def test_ready_merge_timing_jobs_stay_identical() -> None:
    jobs = []
    for tier in ("ready", "merge"):
        pipeline = yaml.safe_load((ROOT / f".buildkite/amd/test-amd-{tier}.yml").read_text(encoding="utf-8"))
        jobs.append(
            [
                step
                for group in pipeline["steps"]
                for step in group.get("steps", [])
                if any("run_cpu_suite.py" in command for command in step.get("commands", []))
            ]
        )
    assert jobs[0] == jobs[1]  # The combiner still deduplicates READY + MERGE.
    assert len(jobs[0]) == 2
    for step in jobs[0]:
        assert step["grade"] == "Blocking"
        assert step["artifact_paths"] == ["artifacts/amd-cpu-suites/**/*"]
    other = next(step for step in jobs[0] if step["label"] == "Simple · Other Test")
    assert "--ignore=tests/model_executor" in other["commands"][-1]
