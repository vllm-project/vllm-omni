# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""ROCm IR-priority contract for the Omni custom-op AITER gate."""

import pytest
from vllm.config import CompilationConfig, CompilationMode, DeviceConfig, VllmConfig
from vllm.config.compilation import CUDAGraphMode

from vllm_omni.platforms.rocm import platform as rocm_platform_mod
from vllm_omni.platforms.rocm.platform import RocmOmniPlatform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _vllm_config(*, mode: CompilationMode, custom_ops: list[str]) -> VllmConfig:
    return VllmConfig(
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(
            backend="inductor",
            mode=mode,
            custom_ops=custom_ops,
            # NONE makes an upstream `cudagraph_mode != NONE` gate drop aiter.
            cudagraph_mode=CUDAGraphMode.NONE,
        ),
    )


def _patch_arch(monkeypatch: pytest.MonkeyPatch, *, gfx90a: bool, rdna4: bool) -> None:
    import vllm.platforms.rocm as rocm

    # Bound inside get_default_ir_op_priority via `from vllm.platforms.rocm import on_gfx90a`.
    monkeypatch.setattr(rocm, "on_gfx90a", lambda: gfx90a)
    # Not consulted today. A copied `not on_rdna4()` gate would see this patch.
    monkeypatch.setattr(rocm, "on_rdna4", lambda: rdna4)


def _patch_aiter_flags(monkeypatch: pytest.MonkeyPatch, *, enabled: bool) -> None:
    # The method reads these on the module-level `envs` binding, not a local import.
    monkeypatch.setattr(rocm_platform_mod.envs, "VLLM_ROCM_USE_AITER", enabled)
    monkeypatch.setattr(rocm_platform_mod.envs, "VLLM_ROCM_USE_AITER_RMSNORM", enabled)


def _priority(monkeypatch: pytest.MonkeyPatch, **kwargs: object):
    gfx90a = bool(kwargs.pop("gfx90a"))
    rdna4 = bool(kwargs.pop("rdna4"))
    aiter = bool(kwargs.pop("aiter"))
    mode = kwargs.pop("mode")
    custom_ops = kwargs.pop("custom_ops")
    assert not kwargs
    _patch_arch(monkeypatch, gfx90a=gfx90a, rdna4=rdna4)
    _patch_aiter_flags(monkeypatch, enabled=aiter)
    assert isinstance(mode, CompilationMode)
    assert isinstance(custom_ops, list)
    return RocmOmniPlatform.get_default_ir_op_priority(_vllm_config(mode=mode, custom_ops=custom_ops))


def test_gfx90a_does_not_prepend_aiter(monkeypatch: pytest.MonkeyPatch) -> None:
    priority = _priority(
        monkeypatch,
        gfx90a=True,
        rdna4=False,
        aiter=True,
        mode=CompilationMode.NONE,
        custom_ops=["all"],
    )
    assert priority.rms_norm == ["vllm_c", "native"]
    assert priority.fused_add_rms_norm == ["vllm_c", "native"]
    assert priority.gelu_and_mul_sparse == ["native"]


def test_custom_op_gate_prepends_aiter_with_rdna4_and_no_cudagraph(monkeypatch: pytest.MonkeyPatch) -> None:
    priority = _priority(
        monkeypatch,
        gfx90a=False,
        rdna4=True,
        aiter=True,
        mode=CompilationMode.NONE,
        custom_ops=["all"],
    )
    assert priority.rms_norm == ["aiter", "vllm_c", "native"]
    assert priority.fused_add_rms_norm == ["aiter", "vllm_c", "native"]
    assert priority.gelu_and_mul_sparse == ["native"]


def test_inductor_compile_uses_native_default(monkeypatch: pytest.MonkeyPatch) -> None:
    priority = _priority(
        monkeypatch,
        gfx90a=False,
        rdna4=False,
        aiter=True,
        mode=CompilationMode.VLLM_COMPILE,
        custom_ops=["all"],
    )
    assert priority.rms_norm == ["aiter", "native"]
    assert priority.fused_add_rms_norm == ["aiter", "native"]
    assert priority.gelu_and_mul_sparse == ["native"]


def test_false_aiter_flags_do_not_prepend(monkeypatch: pytest.MonkeyPatch) -> None:
    priority = _priority(
        monkeypatch,
        gfx90a=False,
        rdna4=False,
        aiter=False,
        mode=CompilationMode.NONE,
        custom_ops=["all"],
    )
    assert priority.rms_norm == ["vllm_c", "native"]
    assert priority.fused_add_rms_norm == ["vllm_c", "native"]
    assert priority.gelu_and_mul_sparse == ["native"]


def test_disabled_custom_op_does_not_prepend_aiter(monkeypatch: pytest.MonkeyPatch) -> None:
    priority = _priority(
        monkeypatch,
        gfx90a=False,
        rdna4=False,
        aiter=True,
        mode=CompilationMode.NONE,
        custom_ops=["none"],
    )
    assert priority.rms_norm == ["vllm_c", "native"]
    assert priority.fused_add_rms_norm == ["vllm_c", "native"]
    assert priority.gelu_and_mul_sparse == ["native"]
