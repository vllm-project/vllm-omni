# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ROCm IR-priority contract for the Omni custom-op AITER gate."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from vllm.config import CompilationConfig, CompilationMode, DeviceConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    "gfx90a,rdna4,aiter,mode,custom_ops,expected",
    [
        (True, False, True, CompilationMode.NONE, ["all"], ["vllm_c", "native"]),
        (False, True, True, CompilationMode.NONE, ["all"], ["aiter", "vllm_c", "native"]),
        (False, False, True, CompilationMode.VLLM_COMPILE, ["all"], ["aiter", "native"]),
        (False, False, False, CompilationMode.NONE, ["all"], ["vllm_c", "native"]),
        (False, False, True, CompilationMode.NONE, ["none"], ["vllm_c", "native"]),
    ],
    ids=["gfx90a", "rdna4-no-cudagraph", "inductor", "aiter-disabled", "custom-op-disabled"],
)
def test_rocm_ir_op_priority(
    monkeypatch: pytest.MonkeyPatch,
    gfx90a: bool,
    rdna4: bool,
    aiter: bool,
    mode: CompilationMode,
    custom_ops: list[str],
    expected: list[str],
) -> None:
    # ROCm import reads the architecture; these policy tests need no device.
    with patch("torch.cuda.get_device_properties", return_value=SimpleNamespace(gcnArchName="gfx942")):
        import vllm.platforms.rocm as rocm

        from vllm_omni.platforms.rocm import platform as rocm_platform_mod
        from vllm_omni.platforms.rocm.platform import RocmOmniPlatform

    monkeypatch.setattr(rocm, "on_gfx90a", lambda: gfx90a)
    # A copied upstream RDNA4 gate must not replace Omni's custom-op policy.
    monkeypatch.setattr(rocm, "on_rdna4", lambda: rdna4)
    monkeypatch.setattr(rocm_platform_mod.envs, "VLLM_ROCM_USE_AITER", aiter)
    monkeypatch.setattr(rocm_platform_mod.envs, "VLLM_ROCM_USE_AITER_RMSNORM", aiter)
    config = VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(
            backend="inductor",
            mode=mode,
            custom_ops=custom_ops,
            # The Omni AITER gate must work without CUDA graph capture.
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )

    priority = RocmOmniPlatform.get_default_ir_op_priority(config)

    assert priority.rms_norm == expected
    assert priority.fused_add_rms_norm == expected
    assert priority.gelu_and_mul_sparse == ["native"]
