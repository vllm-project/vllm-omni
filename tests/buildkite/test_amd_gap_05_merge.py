# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selection and failure propagation for the AMD minicpmo-duplex-window MERGE job."""

from pathlib import Path

import pytest

from tests.buildkite.test_amd_pipeline import AMD_MERGE_PIPELINE, _assert_nightly_junit_guard, _find_step

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LABEL = "MERGE · MiniCPM-o 4.5 Advanced Duplex/Window"
NODES = (
    "tests/e2e/online_serving/test_minicpmo_4_5_duplex.py::test_duplex_two_sessions_resume_and_takeover[three-stage-single-gpu]",
    "tests/e2e/online_serving/test_minicpmo_4_5_window.py::test_window_rebuild_and_next_session[basic-False-three-stage-single-gpu]",
    "tests/e2e/online_serving/test_minicpmo_4_5_window.py::test_window_rebuild_and_next_session[context-False-three-stage-single-gpu]",
    "tests/e2e/online_serving/test_minicpmo_4_5_window.py::test_window_rebuild_and_next_session[context-True-three-stage-single-gpu]",
)
GROUPS = (("duplex-window", 4),)


def test_gap_05_selects_bounded_advanced_rocm_cases() -> None:
    step = _find_step(LABEL, AMD_MERGE_PIPELINE)
    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    assert step["timeout_in_minutes"] == 90
    assert step["source_file_dependencies"]
    assert step["artifact_paths"] == ["artifacts/rocm-merge-minicpmo-duplex-window/**/*"]
    commands = step["commands"]
    collection = next(command for command in step["commands"] if "--collect-only" in command)
    executions = [command for command in step["commands"] if "--junitxml=" in command]
    before = next(command for command in commands if "processes-before.txt" in command)
    cleanup = next(command for command in commands if "leaked_processes_detected=1" in command)
    assert len(executions) == len(GROUPS)
    assert "diff -u" in collection
    assert "timeout --signal=TERM --kill-after=2m 80m" in executions[0]
    assert "processes-after.txt" in cleanup
    assert "process-cleanup.txt" in cleanup
    assert commands.index(before) < commands.index(executions[0]) < commands.index(cleanup)
    for command in [collection, *executions]:
        assert "-m 'advanced_model and omni and rocm and MI325 and cards_1'" in command
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
def test_gap_05_execution_guard(
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
        expected_message=f"Expected exactly {expected_count} executed minicpmo-duplex-window {group} tests",
        artifact_env="ROCM_MERGE_ARTIFACT_DIR",
        job_id_prefix="gap-05",
        pytest_status=pytest_status,
        tests=expected_count + count_delta,
        skipped=skipped,
        failures=failures,
        errors=errors,
        timeout_status=timeout_status,
        expected_status=expected_status,
    )
