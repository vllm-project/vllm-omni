# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import importlib.util
from pathlib import Path
from shlex import split
from xml.etree import ElementTree

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-nightly.yml")
EVIDENCE_SCRIPT = Path(".buildkite/amd/scripts/rocm_ci_evidence.py")
JOBS = {
    "ROCm · Tiny Diffusion Multi-GPU · 2-GPU": ("mi300_2", 40, "artifacts/rocm-tiny-diffusion-2gpu"),
    "ROCm · Tiny Diffusion Multi-GPU · 4-GPU": ("mi300_4", 40, "artifacts/rocm-tiny-diffusion-4gpu"),
    "ROCm · Diffusion Quantization · H100-class Scope": ("mi300_1", 100, "artifacts/rocm-diffusion-quant-h100"),
    "ROCm · Diffusion Quantization · Legacy L4 Scope": ("mi300_1", 70, "artifacts/rocm-diffusion-quant-l4"),
}


def _find_step(label: str) -> dict:
    pipeline = yaml.safe_load(PIPELINE.read_text(encoding="utf-8"))

    def walk(steps: list[dict]) -> dict | None:
        for step in steps:
            if step.get("label") == label:
                return step
            if nested := walk(step.get("steps", [])):
                return nested
        return None

    step = walk(pipeline["steps"])
    assert step is not None, f"missing AMD pipeline step: {label}"
    return step


@pytest.mark.parametrize(("label", "contract"), JOBS.items())
def test_jobs_are_nonblocking_and_retain_evidence(label: str, contract: tuple[str, int, str]) -> None:
    queue, timeout, artifact_dir = contract
    step = _find_step(label)
    commands = step["commands"]
    pytest_commands = [command for command in commands if "pytest " in command]

    assert step["agent_pool"] == queue
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == timeout
    assert step["artifact_paths"] == [f"{artifact_dir}/**/*"]
    assert len(pytest_commands) == 1
    assert "--collect-only" not in "\n".join(commands)
    assert "VLLM_CI_ALLOW_NO_TESTS" not in "\n".join(commands)
    for evidence in (
        "environment.txt",
        "pytest.log",
        "pytest.xml",
        "pytest-summary.txt",
        "pytest-result.txt",
        "process-cleanup.txt",
    ):
        assert evidence in "\n".join(commands)
    result_command = next(command for command in commands if "rocm_ci_evidence.py pytest-result" in command)
    assert '--xml "$$ROCM_CI_ARTIFACT_DIR/pytest.xml"' in result_command
    assert '--log "$$ROCM_CI_ARTIFACT_DIR/pytest.log"' in result_command


@pytest.mark.parametrize(
    ("label", "markers", "run_level"),
    [
        ("ROCm · Tiny Diffusion Multi-GPU · 2-GPU", "full_model and L4 and B200 and cuda and cards_2", "core_model"),
        ("ROCm · Tiny Diffusion Multi-GPU · 4-GPU", "full_model and L4 and B200 and cuda and cards_4", "core_model"),
        (
            "ROCm · Diffusion Quantization · H100-class Scope",
            "full_model and cuda and H100 and B200 and cards_1",
            "full_model",
        ),
        (
            "ROCm · Diffusion Quantization · Legacy L4 Scope",
            "full_model and cuda and L4 and B200 and cards_1",
            "full_model",
        ),
    ],
)
def test_jobs_match_current_cuda_marker_scopes(label: str, markers: str, run_level: str) -> None:
    step = _find_step(label)
    command = next(command for command in step["commands"] if "pytest " in command)
    argv = split(command)
    assert argv[argv.index("-m") + 1] == markers
    assert argv[argv.index("--run-level") + 1] == run_level
    if "Multi-GPU" in label:
        assert "export VLLM_OMNI_TEST_INIT_TIMEOUT=900" in step["commands"]
        assert "export VLLM_OMNI_TEST_STAGE_INIT_TIMEOUT=600" in step["commands"]


def test_h100_quantization_uses_rocm_stable_attention_backend_and_budget() -> None:
    commands = _find_step("ROCm · Diffusion Quantization · H100-class Scope")["commands"]
    command_text = "\n".join(commands)

    assert "export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in commands
    assert "export VLLM_OMNI_TEST_INIT_TIMEOUT=1800" in commands
    assert "export VLLM_OMNI_TEST_STAGE_INIT_TIMEOUT=1200" in commands
    assert "diffusion_attention_backend=$$DIFFUSION_ATTENTION_BACKEND" in command_text
    assert "timeout --signal=TERM --kill-after=1m 90m" in command_text


def test_legacy_l4_scope_installs_pinned_gguf_plugin() -> None:
    commands = "\n".join(_find_step("ROCm · Diffusion Quantization · Legacy L4 Scope")["commands"])
    assert 'python3 -m pip install "vllm-gguf-plugin==0.0.4"' in commands


def test_evidence_process_identity_includes_start_time() -> None:
    spec = importlib.util.spec_from_file_location("rocm_ci_evidence", EVIDENCE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    process_table = "12 1 Mon Sep 15 10:00:00 2026 S python3\n13 1 Mon Sep 15 10:00:01 2026 S python3\n"
    rows = module._process_identities(process_table, ignored_pids={13})
    assert rows == {"12|Mon|Sep|15|10:00:00|2026": "12 1 Mon Sep 15 10:00:00 2026 S python3"}


def _evidence_module():
    spec = importlib.util.spec_from_file_location("rocm_ci_evidence", EVIDENCE_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_evidence_helper_reports_executed_and_deselected_counts(tmp_path: Path) -> None:
    xml = tmp_path / "pytest.xml"
    log = tmp_path / "pytest.log"
    output = tmp_path / "pytest-result.txt"
    xml.write_text(
        '<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1"/></testsuites>', encoding="utf-8"
    )
    log.write_text("2 passed, 1 skipped, 12 deselected\n", encoding="utf-8")

    _evidence_module().summarize_pytest([xml], log, output)

    assert output.read_text(encoding="utf-8") == (
        "collected=15 selected=3 passed=2 failed=0 skipped=1 deselected=12 errors=0 executed=2\n"
    )


@pytest.mark.parametrize(
    "report",
    [
        '<testsuite tests="0" failures="0" errors="0" skipped="0"/>',
        '<testsuite tests="2" failures="0" errors="0" skipped="2"/>',
        '<testsuite tests="2" failures="1" errors="0" skipped="0"/>',
        '<testsuite tests="2" failures="0" errors="1" skipped="0"/>',
        '<testsuite tests="1" failures="0" errors="0" skipped="2"/>',
        '<testsuite tests="-1" failures="0" errors="0" skipped="0"/>',
        "<unrecognized/>",
    ],
)
def test_evidence_helper_rejects_unqualified_junit(tmp_path: Path, report: str) -> None:
    xml = tmp_path / "pytest.xml"
    log = tmp_path / "pytest.log"
    xml.write_text(report, encoding="utf-8")
    log.write_text("12 deselected\n", encoding="utf-8")

    with pytest.raises(RuntimeError):
        _evidence_module().summarize_pytest([xml], log, tmp_path / "pytest-result.txt")


@pytest.mark.parametrize("report", [None, "<testsuite"])
def test_evidence_helper_rejects_missing_or_malformed_junit(tmp_path: Path, report: str | None) -> None:
    xml = tmp_path / "pytest.xml"
    log = tmp_path / "pytest.log"
    if report is not None:
        xml.write_text(report, encoding="utf-8")
    log.write_text("1 passed\n", encoding="utf-8")

    with pytest.raises((FileNotFoundError, ElementTree.ParseError)):
        _evidence_module().summarize_pytest([xml], log, tmp_path / "pytest-result.txt")
