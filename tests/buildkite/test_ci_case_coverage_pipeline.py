# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Render real pipeline entrypoints to verify case-report metadata reaches jobs."""

import copy
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".buildkite/common/scripts"))

from upload_pipeline import _render_test_pipeline  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPORT_GLOB = "artifacts/ci-case-coverage/*.jsonl"
PREFIX = "VLLM_OMNI_CI_CASE_"
TIERS = ("ready", "merge", "nightly", "weekly")
AMD_TEMPLATE = Path(".buildkite/amd/test-template-amd-omni.j2")


@pytest.fixture(autouse=True)
def _offline_render(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIRROR_HW", "")
    monkeypatch.setenv("VLLM_OMNI_SKIP_SHARD_TIMINGS_DOWNLOAD", "1")
    monkeypatch.setenv("VLLM_OMNI_SKIP_SHARD_FILE_WRITE", "1")


def _cuda_source() -> dict:
    jobs: list[dict] = [
        {"label": "default blocking", "commands": ["pytest -q tests/example.py"], "artifact_paths": "logs/*.txt"},
        {"label": "explicit blocking", "command": "pytest -q tests/example.py", "soft_fail": False},
        {"label": "nonblocking", "commands": ["echo prepare", "pytest -q tests/example.py"], "soft_fail": True},
        {"label": "conditional", "commands": ["pytest -q tests/example.py"], "soft_fail": [{"exit_status": 1}]},
    ]
    for job in jobs:
        job["agents"] = {"queue": "medium_cpu_queue_premerge"}
        job["env"] = {"KEEP_ME": "unchanged"}
    jobs[-1]["artifact_paths"] = ["results/*.xml", REPORT_GLOB]
    return {"env": {"PIPELINE_ENV": "retained"}, "steps": [{"group": "Nested tests", "steps": jobs}, "wait"]}


@pytest.mark.parametrize("tier", TIERS)
def test_cuda_render_preserves_jobs_and_adds_tier_and_blocking_metadata(tier: str) -> None:
    source = _cuda_source()
    original = copy.deepcopy(source)
    rendered = _render_test_pipeline(source, None, pipeline_path=Path(f".buildkite/cuda/test-{tier}.yml"))

    assert source == original
    assert rendered["env"] == original["env"]
    assert rendered["steps"][-1] == "wait"
    group = rendered["steps"][0]
    assert "env" not in group
    expected_grades = ("true", "true", "false", "conditional")
    for before, after, grade in zip(original["steps"][0]["steps"], group["steps"], expected_grades, strict=True):
        assert after["env"] == {
            "KEEP_ME": "unchanged",
            PREFIX + "DIR": "artifacts/ci-case-coverage",
            PREFIX + "PLATFORM": "cuda",
            PREFIX + "TIER": tier,
            PREFIX + "BLOCKING": grade,
        }
        assert after.get("command") == before.get("command")
        assert after.get("commands") == before.get("commands")
        assert after.get("soft_fail") == before.get("soft_fail")
        assert ("soft_fail" in after) == ("soft_fail" in before)
        assert after["agents"] == before["agents"]
        old_artifacts = before.get("artifact_paths", [])
        if isinstance(old_artifacts, str):
            old_artifacts = [old_artifacts]
        assert after["artifact_paths"] == list(dict.fromkeys([*old_artifacts, REPORT_GLOB]))


@pytest.mark.parametrize(
    "pipeline_path",
    [
        None,
        ".buildkite/npu/test-ready.yml",
        ".buildkite/amd/test-amd-ready.yml",
        ".buildkite/cuda/bootstrap-upload-steps.yml",
    ],
)
def test_case_metadata_is_not_added_outside_cuda_test_tiers(pipeline_path: str | None) -> None:
    source = _cuda_source()
    rendered = _render_test_pipeline(source, None, pipeline_path=Path(pipeline_path) if pipeline_path else None)
    assert rendered == source


@pytest.mark.parametrize("tier", TIERS)
def test_amd_render_preserves_commands_and_artifacts_with_tier_metadata(tier: str) -> None:
    minijinja = pytest.importorskip("minijinja", reason="AMD template render requires the MiniJinja Python binding")
    jobs: list[dict] = [
        {
            "label": grade,
            "agent_pool": "mi300_2",
            "mirror_hardwares": ["amdproduction"],
            "grade": grade,
            "commands": ["echo prepare", "pytest -q tests/example.py"],
            "env": {"KEEP_ME": "unchanged"},
            "artifact_paths": [f"artifacts/{grade}/*.xml"],
            "timeout_in_minutes": 30,
        }
        for grade in ("Blocking", "NonBlocking")
    ]
    group_name = f"{tier.upper()}_TESTS"
    rendered = minijinja.Environment().render_str(
        AMD_TEMPLATE.read_text(encoding="utf-8"),
        steps=[{"group": group_name, "steps": jobs}],
        env={"PIPELINE_ENV": "retained"},
        mirror_hw="amdproduction",
        vllm_aiter_mha_trusted_commit="fixture-commit",
    )
    pipeline = yaml.safe_load(rendered)
    group = next(step for step in pipeline["steps"] if step.get("group") == group_name)
    assert pipeline["env"] == {"PIPELINE_ENV": "retained"}
    assert len(group["steps"]) == 2
    for before, after in zip(jobs, group["steps"], strict=True):
        is_blocking = before["grade"] == "Blocking"
        assert after["soft_fail"] is not is_blocking
        assert after["command"] == "bash .buildkite/amd/scripts/run-amd-test.sh"
        assert after["env"]["KEEP_ME"] == "unchanged"
        assert {name: value for name, value in after["env"].items() if name.startswith(PREFIX)} == {
            PREFIX + "DIR": "artifacts/ci-case-coverage",
            PREFIX + "PLATFORM": "amd",
            PREFIX + "TIER": tier,
            PREFIX + "BLOCKING": "true" if is_blocking else "false",
        }
        assert after["env"]["TEST_COMMANDS"].endswith("\n".join(before["commands"]))
        assert 'export KEEP_ME="unchanged"' in after["env"]["TEST_COMMANDS"]
        assert after["artifact_paths"] == [REPORT_GLOB, *before["artifact_paths"]]
        assert after["timeout_in_minutes"] == before["timeout_in_minutes"]
        container = after["plugins"][0]["kubernetes"]["podSpecPatch"]["containers"][0]
        assert container["resources"]["limits"]["amd.com/gpu"] == "2"
