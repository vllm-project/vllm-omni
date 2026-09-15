# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
AMD_TEMPLATE = REPO_ROOT / ".buildkite/amd/test-template-amd-omni.j2"
CI_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.ci"
ROCM_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.rocm"


def _docker_arg(path: Path, name: str) -> str:
    prefix = f"ARG {name}="
    matches = [
        line.removeprefix(prefix) for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(prefix)
    ]
    assert len(matches) == 1, f"expected one {name} declaration in {path}, found {len(matches)}"
    return matches[0]


def _line_index(lines: list[str], prefix: str) -> int:
    matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
    assert len(matches) == 1, f"expected one line starting with {prefix!r}, found {len(matches)}"
    return matches[0]


def test_rocm_base_tracks_ci_vllm_release() -> None:
    ci_release = _docker_arg(CI_DOCKERFILE, "VLLM_BASE_TAG")
    rocm_base = _docker_arg(ROCM_DOCKERFILE, "BASE_IMAGE")

    image_ref, separator, image_tag = rocm_base.rpartition(":")
    assert separator, f"expected a tagged ROCm base image, got {rocm_base}"
    assert image_ref.rsplit("/", 1)[-1] == "vllm-openai-rocm", image_ref
    assert image_tag == ci_release


def test_rocm_defaults_to_prebuilt_base_image() -> None:
    assert _docker_arg(ROCM_DOCKERFILE, "USE_NIGHTLY_BUILD") == "0"


def test_amd_build_uses_rocm_dockerfile_defaults() -> None:
    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    build_commands = [line.strip() for line in template.splitlines() if '"docker build ' in line]

    assert len(build_commands) == 1, f"expected one AMD image build command, found {len(build_commands)}"
    build_command = build_commands[0]
    assert "-f docker/Dockerfile.rocm" in build_command
    for arg_name in ("BASE_IMAGE", "USE_NIGHTLY_BUILD"):
        assert f"--build-arg {arg_name}" not in build_command
        assert f"--build-arg={arg_name}" not in build_command


def test_rocm_source_ref_tracks_ci_vllm_release() -> None:
    ci_release = _docker_arg(CI_DOCKERFILE, "VLLM_BASE_TAG")
    rocm_source_ref = _docker_arg(ROCM_DOCKERFILE, "VLLM_VERSION_OR_COMMIT_HASH")

    assert rocm_source_ref == ci_release


def test_rocm_dockerfile_contains_vllm_api_canary() -> None:
    dockerfile = ROCM_DOCKERFILE.read_text(encoding="utf-8")
    lines = dockerfile.splitlines()
    nightly_start = _line_index(lines, 'RUN if [ "${USE_NIGHTLY_BUILD}" = "1" ]; then')
    canary = (
        'RUN python3 -c "import vllm; '
        "from vllm.v1.kv_cache_interface import compute_layout_strides; "
        "from vllm.v1.kv_cache_layout import KVCacheLayout;"
    )
    canary_index = _line_index(lines, canary)

    assert any(line.strip() == "fi" for line in lines[nightly_start + 1 : canary_index]), (
        "vLLM API canary must follow the optional nightly source reinstall block"
    )
