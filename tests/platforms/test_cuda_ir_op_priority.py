# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm.config import CompilationConfig, CompilationMode, DeviceConfig, VllmConfig

from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _vllm_config(*, backend: str = "inductor", mode: CompilationMode) -> VllmConfig:
    return VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(backend=backend, mode=mode),
    )


@pytest.mark.parametrize(
    "mode",
    [CompilationMode.NONE, CompilationMode.VLLM_COMPILE, CompilationMode.STOCK_TORCH_COMPILE],
)
def test_cuda_default_ir_op_priority_prefers_vllm_c_when_inductor_backend(mode: CompilationMode) -> None:
    """Regression for #4964: inductor-active configs must not switch to native-only."""
    priority = CudaOmniPlatform.get_default_ir_op_priority(_vllm_config(mode=mode))
    assert priority.rms_norm == ["vllm_c", "native"]
    assert priority.fused_add_rms_norm == ["vllm_c", "native"]


def test_cuda_default_ir_op_priority_with_oink(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_USE_OINK_OPS", True)
    priority = CudaOmniPlatform.get_default_ir_op_priority(
        _vllm_config(mode=CompilationMode.VLLM_COMPILE),
    )
    assert priority.rms_norm == ["oink", "vllm_c", "native"]
    assert priority.fused_add_rms_norm == ["oink", "vllm_c", "native"]
