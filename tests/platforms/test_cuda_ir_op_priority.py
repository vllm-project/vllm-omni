# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
from vllm.config import CompilationConfig, CompilationMode, DeviceConfig, VllmConfig
from vllm.platforms import current_platform

# Importing CudaOmniPlatform pulls in vllm.platforms.cuda, which imports the
# CUDA-only ``vllm._C_stable_libtorch`` extension at module top level. That
# extension is absent on non-CUDA builds (e.g. XPU), so skip the whole module
# there before the import can crash collection. ``is_cuda()`` resolves the
# platform without importing cuda.py.
if not current_platform.is_cuda():
    pytest.skip("CUDA-only IR op priority tests", allow_module_level=True)

from vllm_omni.platforms.cuda.platform import CudaOmniPlatform  # noqa: E402

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


@pytest.mark.parametrize(
    "mode",
    [CompilationMode.NONE, CompilationMode.VLLM_COMPILE],
)
def test_cuda_default_ir_op_priority_names_only_registered_providers(mode: CompilationMode) -> None:
    """Every field must name providers that are actually registered.

    ``IrOpPriorityConfig.with_default`` fills any field it is not given with the
    generic ``default`` list, so an override that does not name a field silently
    inherits every provider in that list. Upstream added ``gelu_and_mul_sparse``
    (implemented by ``triton``/``native`` only), which therefore must not inherit
    CUDA's ``["vllm_c", "native"]`` default: ``IrOp._filter_priority_impls``
    asserts on unregistered providers and is reached from ``WorkerBase.__init__``
    and the omni diffusion forward context.
    """
    priority = CudaOmniPlatform.get_default_ir_op_priority(_vllm_config(mode=mode))
    assert priority.gelu_and_mul_sparse == ["triton", "native"]
    # Scoped context manager: runs the real upstream provider validation for
    # every field and restores the previous priorities on exit.
    with priority.set_priority():
        pass
