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


def _registered(op_name: str, providers: list[str]) -> list[str]:
    """What ``get_default_ir_op_priority`` must keep, per the real registry.

    Mirrors the filtering done by ``CudaOmniPlatform._registered_only`` so the
    expected values track what is actually registered on the test device
    (``vllm_c`` is absent on e.g. sm70/V100).
    """
    from vllm.ir.op import IrOp
    from vllm.platforms import current_platform

    current_platform.import_ir_kernels()
    registered = set(IrOp.registry[op_name].impls)
    return [p for p in providers if p in registered]


@pytest.mark.parametrize(
    "mode",
    [CompilationMode.NONE, CompilationMode.VLLM_COMPILE, CompilationMode.STOCK_TORCH_COMPILE],
)
def test_cuda_default_ir_op_priority_prefers_vllm_c_when_inductor_backend(mode: CompilationMode) -> None:
    """Regression for #4964: inductor-active configs must not switch to native-only.

    On devices where ``vllm_c`` is registered it must stay first in the
    priority lists.
    """
    priority = CudaOmniPlatform.get_default_ir_op_priority(_vllm_config(mode=mode))
    assert priority.rms_norm == _registered("rms_norm", ["vllm_c", "native"])
    assert priority.fused_add_rms_norm == _registered("fused_add_rms_norm", ["vllm_c", "native"])
    if "vllm_c" in priority.rms_norm:
        assert priority.rms_norm[0] == "vllm_c"


def test_cuda_default_ir_op_priority_with_oink(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_USE_OINK_OPS", True)
    priority = CudaOmniPlatform.get_default_ir_op_priority(
        _vllm_config(mode=CompilationMode.VLLM_COMPILE),
    )
    assert priority.rms_norm == _registered("rms_norm", ["oink", "vllm_c", "native"])
    assert priority.fused_add_rms_norm == _registered("fused_add_rms_norm", ["oink", "vllm_c", "native"])


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
    assert priority.gelu_and_mul_sparse == _registered("gelu_and_mul_sparse", ["triton", "native"])
    # Scoped context manager: runs the real upstream provider validation for
    # every field and restores the previous priorities on exit.
    with priority.set_priority():
        pass


def test_cuda_default_ir_op_priority_drops_unregistered_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    """sm70 (e.g. V100) regression: ``vllm_c`` kernels are not registered there.

    Before per-op filtering, diffusion worker startup crashed in
    ``IrOp._filter_priority_impls`` with "All providers in priority must be
    registered implementations." Hide ``vllm_c`` from the registry to
    reproduce an sm70-like registry on any device.
    """
    from vllm.ir.op import IrOp
    from vllm.platforms import current_platform

    current_platform.import_ir_kernels()
    for op_name in ("rms_norm", "fused_add_rms_norm", "gelu_and_mul_sparse"):
        op = IrOp.registry[op_name]
        monkeypatch.setattr(op, "impls", {p: impl for p, impl in op.impls.items() if p != "vllm_c"})

    priority = CudaOmniPlatform.get_default_ir_op_priority(
        _vllm_config(mode=CompilationMode.VLLM_COMPILE),
    )
    assert priority.rms_norm == ["native"]
    assert priority.fused_add_rms_norm == ["native"]
    assert priority.gelu_and_mul_sparse == ["triton", "native"]
    # Scoped context manager: runs the real upstream provider validation for
    # every field against the patched (vllm_c-less) registry.
    with priority.set_priority():
        pass


def test_registered_only_unknown_op_falls_back_to_native(monkeypatch: pytest.MonkeyPatch) -> None:
    """An op missing from the registry must not pass providers through.

    Returning the unfiltered list for an unknown op would reintroduce the
    unregistered-provider assert in ``IrOp._filter_priority_impls``; the
    fallback must be ``native`` only.
    """
    from vllm.ir.op import IrOp

    monkeypatch.setattr(IrOp, "registry", {k: v for k, v in IrOp.registry.items() if k != "rms_norm"})
    assert CudaOmniPlatform._registered_only(["vllm_c", "native"], "rms_norm") == ["native"]
