# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path
from shlex import split

import pytest
import yaml

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PIPELINE = Path(".buildkite/amd/test-amd-ready.yml")
DISTRIBUTED_LABEL = "ROCm · Distributed Core GPU Coverage"
TINY_BASE_LABEL = "ROCm · Tiny Diffusion Base GPU Coverage"


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


@pytest.mark.parametrize(
    ("label", "artifact_dir"),
    [
        (DISTRIBUTED_LABEL, "artifacts/rocm-distributed-core"),
        (TINY_BASE_LABEL, "artifacts/rocm-tiny-diffusion-base"),
    ],
)
def test_new_ready_jobs_follow_rocm_evidence_contract(label: str, artifact_dir: str) -> None:
    step = _find_step(label)
    commands = step["commands"]
    pytest_commands = [command for command in commands if "--junitxml=" in command]

    assert step["agent_pool"] == "mi300_1"
    assert step["grade"] == "NonBlocking"
    expected_timeout = 60 if label == TINY_BASE_LABEL else 30
    assert step["timeout_in_minutes"] == expected_timeout
    assert step["artifact_paths"] == [f"{artifact_dir}/**/*"]
    assert len(pytest_commands) == (4 if label == TINY_BASE_LABEL else 1)
    assert "--collect-only" not in "\n".join(commands)
    assert "VLLM_CI_ALLOW_NO_TESTS" not in "\n".join(commands)
    assert "environment.txt" in "\n".join(commands)
    assert "torch.accelerator.device_count()" in "\n".join(commands)
    assert "expected {expected_gpus} ROCm GPU(s)" in "\n".join(commands)
    assert "process-cleanup.txt" in "\n".join(commands)
    assert "pytest-summary.txt" in "\n".join(commands)
    assert "pytest-result.txt" in "\n".join(commands)
    assert "executed = tests - skipped" in "\n".join(commands)
    assert "tests == 0 or executed == 0 or failures or errors" in "\n".join(commands)
    assert "--junitxml=" in pytest_commands[0]


def test_distributed_job_matches_current_cuda_scope() -> None:
    command = next(command for command in _find_step(DISTRIBUTED_LABEL)["commands"] if "--junitxml=" in command)
    argv = split(command)
    assert "tests/distributed/" in argv
    assert argv[argv.index("-m") + 1] == "core_model and cuda and L4"
    assert argv[argv.index("--run-level") + 1] == "core_model"


def test_distributed_cuda_scope_contains_rocm_runnable_gpu_smokes() -> None:
    source = Path("tests/distributed/omni_connectors/test_mooncake_transfer_engine_buffer.py").read_text(
        encoding="utf-8"
    )
    assert source.count('@hardware_test(res={"cuda": "L4"}, num_cards=1)') == 2


def test_tiny_base_job_matches_current_cuda_scope() -> None:
    step = _find_step(TINY_BASE_LABEL)
    pytest_commands = [command for command in step["commands"] if "--junitxml=" in command]
    default_command, native_conv_command, ltx2_command, qwen_command = pytest_commands
    default_argv = split(default_command)
    native_conv_argv = split(native_conv_command)
    ltx2_argv = split(ltx2_command)
    qwen_argv = split(qwen_command)
    commands = "\n".join(step["commands"])
    assert "export VLLM_ROCM_USE_AITER=0" in step["commands"]
    assert "export DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA" in step["commands"]
    assert "export MIOPEN_FIND_MODE=3" in step["commands"]
    assert "export MIOPEN_FIND_ENFORCE=3" in step["commands"]
    assert "MIOPEN_DEBUG_DISABLE_FIND_DB" not in commands
    assert "MIOPEN_USER_DB_PATH=" in commands
    assert "MIOPEN_CUSTOM_CACHE_DIR=" in commands
    assert "export VLLM_OMNI_ROCM_CI_DISABLE_MIOPEN=0" in step["commands"]
    assert "export VLLM_OMNI_ROCM_CI_DISABLE_MIOPEN=1" in step["commands"]
    assert "export VLLM_OMNI_ROCM_CI_FORCE_MATH_SDPA=0" in step["commands"]
    assert "export VLLM_OMNI_ROCM_CI_FORCE_MATH_SDPA=1" in step["commands"]
    assert "rocm-ci-sitecustomize" in commands
    assert "diffusion_attention_backend=" in commands
    assert "miopen_find_mode=" in commands
    assert "miopen_find_enforce=" in commands
    assert "miopen_user_db_path=" in commands
    assert "miopen_custom_cache_dir=" in commands
    assert "default_cudnn_enabled=" in commands
    assert "native_conv_cudnn_enabled=" in commands
    assert "default_flash_sdp_enabled=" in commands
    assert "default_mem_efficient_sdp_enabled=" in commands
    assert "default_math_sdp_enabled=" in commands
    assert "qwen_flash_sdp_enabled=" in commands
    assert "qwen_mem_efficient_sdp_enabled=" in commands
    assert "qwen_math_sdp_enabled=" in commands
    for argv in (default_argv, native_conv_argv, ltx2_argv, qwen_argv):
        assert "tests/model_tests/diffusion/" in argv
        assert argv[argv.index("-m") + 1] == "core_model and cuda"
        assert argv[argv.index("--run-level") + 1] == "core_model"
        assert argv[argv.index("-n") + 1] == "1"
    qwen_models = "QwenImagePipeline or QwenImageEditPipeline or QwenImageEditPlusPipeline"
    native_conv_models = "LongCatImageEditPipeline or FluxKontextPipeline"
    special_models = f"LTX2Pipeline or {native_conv_models} or {qwen_models}"
    assert default_argv[default_argv.index("-k") + 1] == f"not ({special_models})"
    assert native_conv_argv[native_conv_argv.index("-k") + 1] == native_conv_models
    assert ltx2_argv[ltx2_argv.index("-k") + 1] == "LTX2Pipeline"
    assert qwen_argv[qwen_argv.index("-k") + 1] == qwen_models
    assert "timeout --signal=TERM --kill-after=1m 30m" in default_command
    assert "timeout --signal=TERM --kill-after=1m 15m" in native_conv_command
    assert "timeout --signal=TERM --kill-after=1m 10m" in ltx2_command
    assert "timeout --signal=TERM --kill-after=1m 10m" in qwen_command
    assert "pytest-default.xml" in commands
    assert "pytest-native-conv.xml" in commands
    assert "pytest-ltx2.xml" in commands
    assert "pytest-qwen-math.xml" in commands


def test_rocm_sitecustomize_applies_only_requested_kernel_controls() -> None:
    source = Path(".buildkite/amd/rocm-ci-sitecustomize/sitecustomize.py").read_text(encoding="utf-8")
    assert 'os.environ.get("VLLM_OMNI_ROCM_CI_FORCE_MATH_SDPA")' in source
    assert 'os.environ.get("VLLM_OMNI_ROCM_CI_DISABLE_MIOPEN")' in source
    assert "torch.backends.cudnn.enabled = False" in source
    assert "torch.backends.cudnn.deterministic" not in source
    assert "torch.backends.cuda.enable_flash_sdp(False)" in source
    assert "torch.backends.cuda.enable_mem_efficient_sdp(False)" in source
    assert "torch.backends.cuda.enable_math_sdp(True)" in source
    assert "torch.use_deterministic_algorithms" not in source
