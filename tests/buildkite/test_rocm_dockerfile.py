# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os
import subprocess
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

REPO_ROOT = Path(__file__).resolve().parents[2]
AMD_TEMPLATE = REPO_ROOT / ".buildkite/amd/test-template-amd-omni.j2"
AMD_BUILD_SCRIPT = REPO_ROOT / ".buildkite/amd/scripts/build-ci-image.sh"
CUDA_RELEASE_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.cuda"
CUDA_CI_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.ci"
ROCM_DOCKERFILE = REPO_ROOT / "docker/Dockerfile.rocm"
ROCM_DOCKERIGNORE = REPO_ROOT / "docker/Dockerfile.rocm.dockerignore"


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


def test_rocm_base_tracks_cuda_release() -> None:
    # Every platform defaults to the matching published release image.
    assert _docker_arg(CUDA_CI_DOCKERFILE, "VLLM_BASE_IMAGE") == "vllm/vllm-openai"
    assert "@sha256:" in _docker_arg(CUDA_CI_DOCKERFILE, "VLLM_BASE_TAG")
    assert len(_docker_arg(CUDA_CI_DOCKERFILE, "VLLM_PRECOMPILED_WHEEL_COMMIT")) == 40

    cuda_base = _docker_arg(CUDA_RELEASE_DOCKERFILE, "BASE_IMAGE").partition("@")[0]
    cuda_image, _, cuda_release_tag = cuda_base.rpartition(":")
    rocm_base = _docker_arg(ROCM_DOCKERFILE, "BASE_IMAGE")

    image_ref, separator, image_tag = rocm_base.partition("@")[0].rpartition(":")
    assert separator, f"expected a tagged ROCm base image, got {rocm_base}"
    assert cuda_image == "vllm/vllm-openai"
    assert image_ref == "vllm/vllm-openai-rocm"
    cuda_ci_tag = _docker_arg(CUDA_CI_DOCKERFILE, "VLLM_BASE_TAG").replace(
        "${VLLM_VERSION}", _docker_arg(CUDA_CI_DOCKERFILE, "VLLM_VERSION")
    )
    assert image_tag == cuda_release_tag == cuda_ci_tag.split("@")[0]


def test_rocm_defaults_to_the_published_runtime() -> None:
    assert _docker_arg(ROCM_DOCKERFILE, "USE_NIGHTLY_BUILD") == "0"


def test_amd_build_uses_registry_cache() -> None:
    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    cache_command = '"bash .buildkite/amd/scripts/build-ci-image.sh cache"'
    image_command = '"bash .buildkite/amd/scripts/build-ci-image.sh image"'
    script = AMD_BUILD_SCRIPT.read_text(encoding="utf-8")
    cache_export = script.index("--target deps")
    image_push = script.index("--target test")

    assert template.count(cache_command) == 1
    assert template.count(image_command) == 1
    assert "docker buildx build --push" in script
    assert 'docker buildx imagetools inspect "$CACHE_REF"' in script
    assert '--cache-from "type=registry,ref=${CACHE_REF}"' in script
    assert script.count('--cache-to "type=registry,ref=${CACHE_REF}') == 1
    assert "mode=max" in script
    assert cache_export < image_push
    assert "--cache-to" in script[cache_export:image_push]
    assert "--cache-to" not in script[image_push:]


def test_amd_cache_writer_is_serialized_and_image_publishers_are_bounded() -> None:
    template = AMD_TEMPLATE.read_text(encoding="utf-8")
    script = AMD_BUILD_SCRIPT.read_text(encoding="utf-8")
    fixed_steps = template.split("steps:\n", 1)[1].split("{% for step_group", 1)[0]
    image_stages = yaml.safe_load("steps:\n" + fixed_steps)["steps"][0]["steps"]

    assert len(image_stages) == 2
    cache_step, image_step = image_stages
    assert cache_step["key"] == "amd-cache"
    assert cache_step["depends_on"] is None
    assert cache_step["commands"] == ["bash .buildkite/amd/scripts/build-ci-image.sh cache"]
    assert cache_step["concurrency"] == 1
    assert cache_step["concurrency_group"] == "vllm-omni-amd-cache-publish"
    assert image_step["key"] == "amd-build"
    assert image_step["depends_on"] == "amd-cache"
    assert image_step["commands"] == ["bash .buildkite/amd/scripts/build-ci-image.sh image"]
    assert image_step["concurrency"] == 2
    assert image_step["concurrency_group"] == "vllm-omni-amd-image-publish"
    assert 'IMAGE_PUSH_TIMEOUT="${AMD_IMAGE_PUSH_TIMEOUT:-15m}"' in script
    assert 'IMAGE_PUSH_ATTEMPTS="${AMD_IMAGE_PUSH_ATTEMPTS:-2}"' in script
    assert 'run_with_retry "ROCm CI image push"' in script


def _write_executable(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)


def _run_amd_build_script(
    tmp_path: Path,
    *arguments: str,
    cache_exists: bool = True,
    fail_cache_export: bool = False,
    image_failures: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker_log = tmp_path / "docker.log"
    image_attempts = tmp_path / "image-attempts"
    _write_executable(
        fake_bin / "timeout",
        """#!/usr/bin/env bash
set -euo pipefail
while [[ $# -gt 0 && "$1" == --* ]]; do
    shift
done
shift
exec "$@"
""",
    )
    _write_executable(fake_bin / "sleep", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(
        fake_bin / "docker",
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$*" >> "$FAKE_DOCKER_LOG"
if [[ "$*" == "buildx imagetools inspect "* ]]; then
    exit "$FAKE_CACHE_INSPECT_STATUS"
fi
if [[ "$*" == *"buildx build"* && "$*" == *"--target deps"* ]]; then
    exit "$FAKE_CACHE_EXPORT_STATUS"
fi
if [[ "$*" == *"buildx build"* && "$*" == *"--target test"* ]]; then
    attempts=0
    if [[ -f "$FAKE_IMAGE_ATTEMPTS" ]]; then
        attempts=$(<"$FAKE_IMAGE_ATTEMPTS")
    fi
    attempts=$((attempts + 1))
    printf '%s\n' "$attempts" > "$FAKE_IMAGE_ATTEMPTS"
    if ((attempts <= FAKE_IMAGE_FAILURES)); then
        exit 1
    fi
fi
exit 0
""",
    )

    env = {
        **os.environ,
        "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}",
        "BUILDKITE_COMMIT": "0123456789abcdef0123456789abcdef01234567",
        "ROCM_CI_IMAGE_REGISTRY": "example/vllm-omni",
        "FAKE_DOCKER_LOG": str(docker_log),
        "FAKE_IMAGE_ATTEMPTS": str(image_attempts),
        "FAKE_CACHE_INSPECT_STATUS": "0" if cache_exists else "1",
        "FAKE_CACHE_EXPORT_STATUS": "1" if fail_cache_export else "0",
        "FAKE_IMAGE_FAILURES": str(image_failures),
    }
    result = subprocess.run(
        ["bash", str(AMD_BUILD_SCRIPT), *arguments],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    docker_calls = docker_log.read_text(encoding="utf-8").splitlines() if docker_log.exists() else []
    return result, docker_calls


def _build_calls(docker_calls: list[str]) -> list[str]:
    return [call for call in docker_calls if "buildx build" in call]


def _build_targets(docker_calls: list[str]) -> list[str]:
    return [call.split("--target ", 1)[1].split()[0] for call in _build_calls(docker_calls)]


def test_amd_cache_phase_skips_export_on_hit(tmp_path: Path) -> None:
    result, docker_calls = _run_amd_build_script(tmp_path, "cache")

    assert result.returncode == 0, result.stderr
    assert any("buildx imagetools inspect" in call for call in docker_calls)
    assert _build_calls(docker_calls) == []
    assert "dependency cache hit; skipping cache export" in result.stdout


def test_amd_cache_phase_exports_only_dependency_target_on_miss(tmp_path: Path) -> None:
    result, docker_calls = _run_amd_build_script(tmp_path, "cache", cache_exists=False)

    assert result.returncode == 0, result.stderr
    assert _build_targets(docker_calls) == ["deps"]
    assert "--cache-to" in _build_calls(docker_calls)[0]
    assert "dependency cache miss; exporting dependency layers" in result.stdout


def test_amd_cache_export_failure_is_nonblocking(tmp_path: Path) -> None:
    result, docker_calls = _run_amd_build_script(
        tmp_path,
        "cache",
        cache_exists=False,
        fail_cache_export=True,
    )

    assert result.returncode == 0
    assert _build_targets(docker_calls) == ["deps"]
    assert "image build may rebuild missing layers" in result.stderr


def test_amd_image_phase_skips_cache_publication_and_retries_push(tmp_path: Path) -> None:
    result, docker_calls = _run_amd_build_script(tmp_path, "image", image_failures=1)

    assert result.returncode == 0, result.stderr
    assert not any("buildx imagetools inspect" in call for call in docker_calls)
    assert _build_targets(docker_calls) == ["test", "test"]
    assert all("--cache-to" not in call for call in _build_calls(docker_calls))
    assert "retrying in 5 seconds" in result.stderr


def test_amd_image_phase_fails_after_bounded_attempts(tmp_path: Path) -> None:
    result, docker_calls = _run_amd_build_script(tmp_path, "image", image_failures=2)

    assert result.returncode == 1
    assert _build_targets(docker_calls) == ["test", "test"]
    assert "failed after 2 attempts" in result.stderr


@pytest.mark.parametrize("arguments", [(), ("all",)])
def test_amd_combined_mode_preserves_cache_then_image_flow(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    result, docker_calls = _run_amd_build_script(tmp_path, *arguments, cache_exists=False)

    assert result.returncode == 0, result.stderr
    assert any("buildx imagetools inspect" in call for call in docker_calls)
    assert _build_targets(docker_calls) == ["deps", "test"]
    assert "--cache-to" in _build_calls(docker_calls)[0]
    assert "--cache-to" not in _build_calls(docker_calls)[1]


@pytest.mark.parametrize("arguments", [("invalid",), ("cache", "extra")])
def test_amd_build_rejects_invalid_phase_before_docker(
    tmp_path: Path,
    arguments: tuple[str, ...],
) -> None:
    result, docker_calls = _run_amd_build_script(tmp_path, *arguments)

    assert result.returncode == 2
    assert docker_calls == []
    assert "Usage:" in result.stderr


def test_amd_cache_key_tracks_dependency_inputs_and_architecture() -> None:
    script = AMD_BUILD_SCRIPT.read_text(encoding="utf-8")

    for cache_input in (
        "docker/Dockerfile.rocm",
        "${DOCKERFILE}.dockerignore",
        "pyproject.toml",
        "setup.py",
        "requirements",
        "tools/install_torchcodec_rocm.sh",
        "tools/build_gradio_pillow13.py",
        "tools/gradio_pillow13/*",
        "ROCM_ARCH",
    ):
        assert cache_input in script


def test_amd_build_uses_rocm_dockerfile_defaults() -> None:
    build_script = AMD_BUILD_SCRIPT.read_text(encoding="utf-8")

    assert 'DOCKERFILE="docker/Dockerfile.rocm"' in build_script
    for arg_name in ("BASE_IMAGE", "USE_NIGHTLY_BUILD"):
        assert f"--build-arg {arg_name}" not in build_script
        assert f"--build-arg={arg_name}" not in build_script


def test_rocm_source_copy_preserves_dependency_cache() -> None:
    dockerfile = ROCM_DOCKERFILE.read_text(encoding="utf-8")
    lines = dockerfile.splitlines()
    source_copy_index = _line_index(lines, "COPY --link . .")
    test_stage_index = _line_index(lines, "FROM base AS test")

    assert "FROM deps AS base" in lines[:source_copy_index]
    assert any('-e ".[dev]"' in line for line in lines[:source_copy_index])
    assert not any(
        line.startswith(("RUN ", "COPY ", "ADD ")) for line in lines[source_copy_index + 1 : test_stage_index]
    )


def test_rocm_build_context_excludes_git_history() -> None:
    patterns = {
        line.strip()
        for line in ROCM_DOCKERIGNORE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }

    assert ".git" in patterns


def test_rocm_source_ref_tracks_cuda_vllm_release() -> None:
    # The optional ROCm source rebuild must track the same release commit
    # as the published CUDA runtime.
    cuda_version = _docker_arg(CUDA_RELEASE_DOCKERFILE, "VLLM_VERSION")
    rocm_source_ref = _docker_arg(ROCM_DOCKERFILE, "VLLM_VERSION_OR_COMMIT_HASH")

    assert cuda_version == _docker_arg(CUDA_CI_DOCKERFILE, "VLLM_VERSION")
    assert rocm_source_ref == _docker_arg(CUDA_CI_DOCKERFILE, "VLLM_PRECOMPILED_WHEEL_COMMIT")
    assert _docker_arg(CUDA_RELEASE_DOCKERFILE, "VLLM_PRECOMPILED_WHEEL_COMMIT") == _docker_arg(
        CUDA_CI_DOCKERFILE, "VLLM_PRECOMPILED_WHEEL_COMMIT"
    )


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
