# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_SCRIPT = REPO_ROOT / ".buildkite/amd/scripts/build-ci-image.sh"


@pytest.fixture
def docker_environment(tmp_path: Path) -> dict[str, str]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    source = r"""#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_LOG"
case "$*" in
    "buildx create "*) exit "$FAKE_CREATE_STATUS" ;;
    "buildx inspect "*) exit "$FAKE_BOOTSTRAP_STATUS" ;;
    "buildx imagetools inspect "*) exit "$FAKE_CACHE_STATUS" ;;
    "buildx rm "*) exit "$FAKE_CLEANUP_STATUS" ;;
    "buildx use "*) echo "Shared builder selection was changed" >&2; exit 91 ;;
esac
if [[ "$*" == "buildx build "* ]]; then
    [[ " $* " == *" --builder vllm-omni-rocm-${BUILDKITE_JOB_ID} "* ]] || exit 92
    if [[ "$*" == *"--target test"* ]]; then
        if [[ -n "$FAKE_PEER" ]]; then
            touch "$FAKE_READY"
            for ((attempt=0; attempt<100; attempt++)); do
                [[ -f "$FAKE_PEER" ]] && break
                sleep 0.05
            done
            [[ -f "$FAKE_PEER" ]] || exit 93
        fi
        exit "$FAKE_IMAGE_STATUS"
    fi
fi
"""
    docker = fake_bin / "docker"
    docker.write_text(source, encoding="utf-8", newline="\n")
    docker.chmod(0o755)
    return {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "BUILDKITE_COMMIT": "0123456789abcdef0123456789abcdef01234567",
        "BUILDKITE_JOB_ID": "job-a",
        "ROCM_CI_IMAGE_REGISTRY": "example/vllm-omni",
        "AMD_IMAGE_PUSH_ATTEMPTS": "1",
        "FAKE_LOG": (tmp_path / "docker.log").as_posix(),
        "FAKE_CREATE_STATUS": "0",
        "FAKE_BOOTSTRAP_STATUS": "0",
        "FAKE_CACHE_STATUS": "1",
        "FAKE_CLEANUP_STATUS": "0",
        "FAKE_IMAGE_STATUS": "0",
        "FAKE_PEER": "",
        "FAKE_READY": "",
    }


@pytest.mark.parametrize(
    ("failures", "expected_status", "cleaned_up"),
    [
        ((), 0, True),
        (("FAKE_CREATE_STATUS",), 17, False),
        (("FAKE_BOOTSTRAP_STATUS",), 17, True),
        (("FAKE_IMAGE_STATUS",), 17, True),
        (("FAKE_CLEANUP_STATUS",), 0, True),
        (("FAKE_IMAGE_STATUS", "FAKE_CLEANUP_STATUS"), 17, True),
    ],
)
def test_builder_ownership_and_failure_cleanup(
    docker_environment: dict[str, str], failures: tuple[str, ...], expected_status: int, cleaned_up: bool
) -> None:
    for failure in failures:
        docker_environment[failure] = "17"
    result = subprocess.run(
        ["bash", BUILD_SCRIPT.as_posix()],
        cwd=REPO_ROOT,
        env=docker_environment,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == expected_status, result.stderr
    calls = Path(docker_environment["FAKE_LOG"]).read_text().splitlines()
    assert calls[0] == "buildx create --name vllm-omni-rocm-job-a --driver docker-container"
    assert ("buildx rm --force vllm-omni-rocm-job-a" in calls) == cleaned_up
    if not failures:
        assert "buildx inspect vllm-omni-rocm-job-a --bootstrap" in calls
        builds = [call for call in calls if call.startswith("buildx build ")]
        assert len(builds) == 2  # Both the dependency export and image push are bound.
        assert all("--builder vllm-omni-rocm-job-a" in call for call in builds)


def test_overlapping_image_jobs_use_independent_builders(docker_environment: dict[str, str], tmp_path: Path) -> None:
    # A barrier in the fake Docker command proves both image publishers overlap.
    # Each command rejects implicit selection or a builder belonging to its peer.
    environments = []
    for own, peer in (("job-a", "job-b"), ("job-b", "job-a")):
        environments.append(
            {
                **docker_environment,
                "BUILDKITE_JOB_ID": own,
                "FAKE_CACHE_STATUS": "0",
                "FAKE_LOG": (tmp_path / f"{own}.log").as_posix(),
                "FAKE_READY": (tmp_path / f"{own}.ready").as_posix(),
                "FAKE_PEER": (tmp_path / f"{peer}.ready").as_posix(),
            }
        )
    with (
        subprocess.Popen(
            ["bash", BUILD_SCRIPT.as_posix()],
            cwd=REPO_ROOT,
            env=environments[0],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as first,
        subprocess.Popen(
            ["bash", BUILD_SCRIPT.as_posix()],
            cwd=REPO_ROOT,
            env=environments[1],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as second,
    ):
        try:
            for process in (first, second):
                _, stderr = process.communicate(timeout=20)
                assert process.returncode == 0, stderr
        finally:
            for process in (first, second):
                if process.poll() is None:
                    process.kill()
    for env in environments:
        calls = Path(env["FAKE_LOG"]).read_text().splitlines()
        assert calls[-1] == f"buildx rm --force vllm-omni-rocm-{env['BUILDKITE_JOB_ID']}"
