# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selection and failure propagation for the AMD bagel-shm-connectors MERGE job."""

from pathlib import Path

import pytest

from tests.buildkite.test_amd_pipeline import AMD_MERGE_PIPELINE, _assert_nightly_junit_guard, _find_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LABEL = "MERGE · Bagel Shared-Memory Connectors"
BAGEL_CONNECTOR_TEST = Path("tests/distributed/omni_connectors/test_bagel_shared_memory_connector.py")
NODES = (
    "tests/distributed/omni_connectors/test_bagel_shared_memory_connector.py::test_bagel_img2img_shared_memory_connector",
    "tests/distributed/omni_connectors/test_bagel_shared_memory_connector.py::test_bagel_text2img_shared_memory_connector",
)
GROUPS = (("connectors", 2),)


def test_gap_07_limits_relaxed_pixel_tolerance_to_rocm_img2img() -> None:
    source = BAGEL_CONNECTOR_TEST.read_text(encoding="utf-8")

    assert "PIXEL_TOLERANCE = 10" in source
    assert "ROCM_PIXEL_TOLERANCE = 25" in source
    assert "TEXT2IMG_PIXEL_TOLERANCE = 5" in source
    assert source.count("current_omni_platform.is_rocm()") == 1
    assert "return ROCM_PIXEL_TOLERANCE" in source
    assert "return PIXEL_TOLERANCE" in source
    assert "_validate_pixels(generated_image, tolerance=_img2img_pixel_tolerance())" in source
    assert "tolerance=TEXT2IMG_PIXEL_TOLERANCE" in source


def test_gap_07_selects_bounded_advanced_rocm_cases() -> None:
    step = _find_step(LABEL, AMD_MERGE_PIPELINE)
    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 75
    assert step["source_file_dependencies"]
    assert step["artifact_paths"] == ["artifacts/rocm-merge-bagel-shm-connectors/**/*"]
    commands = step["commands"]
    collection = next(command for command in step["commands"] if "--collect-only" in command)
    executions = [command for command in step["commands"] if "--junitxml=" in command]
    before = next(command for command in commands if "processes-before.txt" in command)
    cleanup = next(command for command in commands if "leaked_processes_detected=1" in command)
    assert len(executions) == len(GROUPS)
    assert "diff -u" in collection
    assert "timeout --signal=TERM --kill-after=2m 65m" in executions[0]
    assert "processes-after.txt" in cleanup
    assert "process-cleanup.txt" in cleanup
    assert commands.index(before) < commands.index(executions[0]) < commands.index(cleanup)
    for command in [collection, *executions]:
        assert "-m 'advanced_model and diffusion and rocm and MI325 and cards_1'" in command
        assert "--run-level advanced_model" in command
        assert "--shard-id=0 --num-shards=1" in command
    for node in NODES:
        assert collection.count(node) == 1
        assert sum(command.count(node) for command in executions) == 1
    assert "--deselect" not in "\n".join(executions)


@pytest.mark.parametrize("group, expected_count", GROUPS)
@pytest.mark.parametrize(
    "pytest_status, timeout_status, count_delta, skipped, failures, errors, expected_status",
    [
        (0, None, 0, 0, 0, 0, 0),
        (0, None, -1, 0, 0, 0, 5),
        (0, None, 1, 0, 0, 0, 5),
        (0, None, 0, 1, 0, 0, 5),
        (0, None, 0, 0, 1, 0, 5),
        (0, None, 0, 0, 0, 1, 5),
        (1, None, 0, 0, 0, 0, 1),
        (5, None, 0, 0, 0, 0, 5),
        (0, 124, 0, 0, 0, 0, 124),
        (0, 137, 0, 0, 0, 0, 137),
    ],
    ids=[
        "pass",
        "missing-case",
        "extra-case",
        "skip",
        "junit-failure",
        "junit-error",
        "assertion",
        "empty",
        "deadline",
        "killed",
    ],
)
def test_gap_07_execution_guard(
    tmp_path: Path,
    group: str,
    expected_count: int,
    pytest_status: int,
    timeout_status: int | None,
    count_delta: int,
    skipped: int,
    failures: int,
    errors: int,
    expected_status: int,
) -> None:
    _assert_nightly_junit_guard(
        tmp_path,
        pipeline_path=AMD_MERGE_PIPELINE,
        step_label=LABEL,
        command_fragment=f"/{group}.xml",
        expected_message=f"Expected exactly {expected_count} executed bagel-shm-connectors {group} tests",
        artifact_env="ROCM_MERGE_ARTIFACT_DIR",
        job_id_prefix="gap-07",
        pytest_status=pytest_status,
        tests=expected_count + count_delta,
        skipped=skipped,
        failures=failures,
        errors=errors,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )
