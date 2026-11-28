# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ci_case_coverage import compare

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
ROOT = Path(__file__).resolve().parents[3]
PLUGIN = "tests.helpers.fixtures.ci_case_coverage"
COMMIT = "a" * 40


def scope() -> dict:
    return {
        "schema_version": 1,
        "commit": COMMIT,
        "jobs": [{"platform": platform, "tier": "ready", "job_id": platform} for platform in ("cuda", "amd")],
    }


def run_pytest(
    tmp_path: Path, platform: str, source: str, *args: str
) -> tuple[subprocess.CompletedProcess, list[list]]:
    suite = tmp_path / platform
    suite.mkdir(exist_ok=True)
    (suite / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (suite / "test_sample.py").write_text(source, encoding="utf-8")
    artifacts = tmp_path / "artifacts"
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT),
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
        "PYTEST_ADDOPTS": "",
        "BUILDKITE_COMMIT": COMMIT,
        "BUILDKITE_JOB_ID": platform,
        "VLLM_OMNI_CI_CASE_DIR": str(artifacts),
        "VLLM_OMNI_CI_CASE_PLATFORM": platform,
        "VLLM_OMNI_CI_CASE_TIER": "ready",
        "VLLM_OMNI_CI_CASE_BLOCKING": "true",
    }
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--noconftest", "-p", PLUGIN, "-q", *args],
        cwd=suite,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    records = [[json.loads(line) for line in path.read_text().splitlines()] for path in artifacts.glob("*.jsonl")]
    return result, records


def test_real_selection_and_execution_are_separate(tmp_path: Path) -> None:
    source = "import pytest\n@pytest.mark.parametrize('x', [1, 2])\ndef test_value(x):\n assert x > 0\n"
    for platform in ("cuda", "amd"):
        result, _ = run_pytest(tmp_path, platform, source, "-k", "1")
        assert result.returncode == 0, result.stdout + result.stderr
    report = compare(scope(), tmp_path / "artifacts")
    assert report["valid"], report["problems"]
    assert len(report["cases"]) == 1
    assert report["cases"][0]["nodeid"] == "test_sample.py::test_value[1]"
    assert report["cases"][0]["amd"]["status"] == "already-covered"


@pytest.mark.parametrize("phase", ["setup", "call", "teardown"])
def test_failures_cannot_be_covered(tmp_path: Path, phase: str) -> None:
    source = {
        "setup": "import pytest\n@pytest.fixture\ndef f():\n assert False\ndef test_value(f): pass\n",
        "call": "def test_value(): assert False\n",
        "teardown": "import pytest\n@pytest.fixture\ndef f():\n yield\n assert False\ndef test_value(f): pass\n",
    }[phase]
    run_pytest(tmp_path, "cuda", "def test_value(): pass\n")
    result, _ = run_pytest(tmp_path, "amd", source)
    assert result.returncode == 1
    report = compare(scope(), tmp_path / "artifacts")
    assert not report["valid"]
    assert report["cases"][0]["amd"]["status"] == "unverified"
    assert report["cases"][0]["amd"]["phase_results"]["failed"] == 1


def test_unexpected_skip_and_explicit_policy(tmp_path: Path) -> None:
    run_pytest(tmp_path, "cuda", "def test_value(): pass\n")
    run_pytest(tmp_path, "amd", "import pytest\ndef test_value(): pytest.skip('unsupported driver')\n")
    config = scope()
    report = compare(config, tmp_path / "artifacts")
    assert any("Unexpected skip" in problem for problem in report["problems"])
    assert "unsupported driver" in report["cases"][0]["amd"]["skips"][0]["reason"]
    config["allowed_skips"] = [
        {"platform": "amd", "tier": "ready", "nodeid": "test_sample.py::test_value", "reason": "Reviewed driver gap"}
    ]
    report = compare(config, tmp_path / "artifacts")
    assert report["valid"]
    assert report["cases"][0]["amd"]["status"] == "unverified"


def test_collect_only_and_empty_selection_do_not_prove_execution(tmp_path: Path) -> None:
    run_pytest(tmp_path, "cuda", "def test_value(): pass\n")
    run_pytest(tmp_path, "amd", "def test_value(): pass\n", "--collect-only")
    report = compare(scope(), tmp_path / "artifacts")
    assert any("Missing execution artifact" in problem for problem in report["problems"])
    result, _ = run_pytest(tmp_path, "amd", "def test_value(): pass\n", "-k", "nonexistent")
    assert result.returncode == 5
    report = compare(scope(), tmp_path / "artifacts")
    assert any("Empty selection" in problem for problem in report["problems"])


def test_revision_mismatch_is_rejected(tmp_path: Path) -> None:
    run_pytest(tmp_path, "cuda", "def test_value(): pass\n")
    config = scope()
    config["commit"] = "b" * 40
    with pytest.raises(ValueError, match="revision/schema mismatch"):
        compare(config, tmp_path / "artifacts")


def test_missing_shard_and_incomplete_session_are_reported(tmp_path: Path) -> None:
    for platform in ("cuda", "amd"):
        run_pytest(tmp_path, platform, "def test_value(): pass\n")
    artifact = next((tmp_path / "artifacts").glob("*.jsonl"))
    lines = artifact.read_text().splitlines()
    artifact.write_text("\n".join(lines[:-1]) + "\n")
    config = scope()
    config["jobs"].append({"platform": "amd", "tier": "ready", "job_id": "missing-shard"})
    report = compare(config, tmp_path / "artifacts")
    assert any("Incomplete session" in problem for problem in report["problems"])
    assert any("missing-shard" in problem for problem in report["problems"])


def test_nonblocking_and_support_exclusions_are_not_passes(tmp_path: Path) -> None:
    run_pytest(tmp_path, "cuda", "def test_value(): pass\ndef test_extra(): pass\n")
    run_pytest(tmp_path, "amd", "def test_value(): pass\n")
    for artifact in (tmp_path / "artifacts").glob("*.jsonl"):
        events = [json.loads(line) for line in artifact.read_text().splitlines()]
        if events[0]["platform"] == "amd":
            events[0]["blocking"] = "false"
            artifact.write_text("".join(json.dumps(event) + "\n" for event in events))
    config = scope()
    report = compare(config, tmp_path / "artifacts")
    rows = {row["nodeid"]: row for row in report["cases"]}
    assert rows["test_sample.py::test_value"]["amd"]["status"] == "nonblocking"
    assert rows["test_sample.py::test_extra"]["amd"]["status"] == "absent"
    config["unsupported"] = [
        {"platform": "amd", "tier": "ready", "nodeid": "test_sample.py::test_extra", "reason": "NVIDIA-only contract"}
    ]
    report = compare(config, tmp_path / "artifacts")
    assert report["valid"]
    assert report["cases"][0]["amd"]["status"] == "unsupported"


def test_xdist_groups_and_worker_selection_are_combined(tmp_path: Path) -> None:
    source = (
        "import pytest\n@pytest.mark.xdist_group('file:/different/checkout')\n"
        "@pytest.mark.parametrize('x', ['user@host', 'second'])\ndef test_value(x): pass\n"
    )
    for platform in ("cuda", "amd"):
        result, _ = run_pytest(tmp_path, platform, source, "-p", "xdist.plugin", "-n", "2", "--dist", "loadgroup")
        assert result.returncode == 0, result.stdout + result.stderr
    report = compare(scope(), tmp_path / "artifacts")
    assert report["valid"], report["problems"]
    assert len(report["cases"]) == 2
    assert all("@file:" not in row["nodeid"] for row in report["cases"])
    assert any("user@host" in row["nodeid"] for row in report["cases"])
    assert all(row["amd"]["phase_results"] == {"passed": 3} for row in report["cases"])


def test_collection_failure_is_an_evidence_failure(tmp_path: Path) -> None:
    run_pytest(tmp_path, "cuda", "def test_value(): pass\n")
    result, _ = run_pytest(tmp_path, "amd", "raise RuntimeError('collection failed')\n")
    assert result.returncode == 2
    report = compare(scope(), tmp_path / "artifacts")
    assert not report["valid"]
    assert any("Collection failed" in problem for problem in report["problems"])


@pytest.mark.parametrize("split", [False, True])
def test_incomplete_attempts_cannot_be_stitched_into_a_pass(tmp_path: Path, split: bool) -> None:
    for platform in ("cuda", "amd"):
        run_pytest(tmp_path, platform, "def test_value(): pass\n")
    for artifact in (tmp_path / "artifacts").glob("*.jsonl"):
        events = [json.loads(line) for line in artifact.read_text().splitlines()]
        if events[0]["platform"] != "amd":
            continue
        incomplete = [event for event in events if event.get("phase") not in {"setup", "call"}]
        artifact.write_text("".join(json.dumps(event) + "\n" for event in incomplete))
        if split:
            second = [event for event in events if event.get("phase") != "teardown"]
            artifact.with_name("second.jsonl").write_text("".join(json.dumps(event) + "\n" for event in second))
    report = compare(scope(), tmp_path / "artifacts")
    assert not report["valid"]
    assert report["cases"][0]["amd"]["status"] == "unverified"
    assert any("Incomplete execution" in problem for problem in report["problems"])


def test_unwritable_reports_do_not_change_pytest_result(tmp_path: Path) -> None:
    (tmp_path / "artifacts").write_text("not a directory")
    result, records = run_pytest(tmp_path, "amd", "def test_value(): pass\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "CI case reporting unavailable" in result.stderr
    assert records == []
