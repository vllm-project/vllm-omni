# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest
import torch
from vllm.platforms.interface import DeviceCapability

from vllm_omni.diffusion.models.ltx2 import ops as ltx2_platform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass(frozen=True)
class _TensorProbe:
    is_cuda: bool
    device: torch.device


@pytest.fixture(autouse=True)
def clear_platform_cache():
    ltx2_platform._cuda_compute_capability.cache_clear()
    try:
        yield
    finally:
        ltx2_platform._cuda_compute_capability.cache_clear()


@pytest.mark.parametrize(
    ("capability", "triton", "tilelang", "fusion_expected", "fna_expected"),
    [
        (80, True, True, False, False),
        (90, True, True, True, True),
        (90, True, False, True, False),
        (90, False, True, False, False),
        (100, True, True, True, False),
        (100, True, False, True, False),
        (103, True, True, True, False),
        (103, True, False, True, False),
        (110, True, True, False, False),
        (120, True, True, False, False),
        (121, True, True, False, False),
    ],
)
def test_production_resolver_and_tensor_eligibility_agree(
    monkeypatch, capability, triton, tilelang, fusion_expected, fna_expected
):
    monkeypatch.setattr(ltx2_platform, "HAS_TRITON", triton)
    monkeypatch.setattr(ltx2_platform, "has_tilelang", lambda: tilelang)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(
        ltx2_platform.current_omni_platform,
        "get_device_capability",
        lambda device_id=0: DeviceCapability(major=capability // 10, minor=capability % 10),
    )
    # Resolution must be lazy even when FNA is selected.
    monkeypatch.setitem(sys.modules, "vllm_omni.diffusion.models.ltx2.ops.fna", None)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: False)
    device = torch.device("cuda:0")
    operators = ltx2_platform.resolve_ltx2_vae_operators(device)
    assert (operators is not None) is fusion_expected
    assert (operators is not None and operators.fna is not None) is fna_expected
    if operators is not None:
        from vllm_omni.diffusion.models.ltx2.ops import qk_rms_norm, residual_adaln, swiglu

        assert operators.qk_norm_rope is qk_rms_norm.try_qk_rms_norm_scale_rope_3d_exact
        assert operators.residual_norm is residual_adaln.try_residual_rms_norm_modulate_exact
        assert operators.residual_add is residual_adaln.try_residual_add3_exact
        assert operators.swiglu is swiglu.try_swiglu_tiled_exact

    tensor = _TensorProbe(is_cuda=True, device=device)
    with torch.inference_mode():
        assert ltx2_platform.is_ltx2_fusion_eligible(tensor) is fusion_expected
        assert ltx2_platform.is_ltx2_fna_eligible(tensor) is fna_expected


@pytest.mark.parametrize(
    ("device", "is_cuda", "available"),
    [("cpu", True, True), ("cuda:0", False, True), ("cuda:0", True, False)],
)
def test_resolver_rejects_unsupported_platform_before_capability_lookup(monkeypatch, device, is_cuda, available):
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_available", lambda: available)

    def unexpected_lookup(_index):
        raise AssertionError("Unsupported platforms must not query CUDA capability")

    monkeypatch.setattr(ltx2_platform, "_cuda_compute_capability", unexpected_lookup)
    assert ltx2_platform.resolve_ltx2_vae_operators(torch.device(device)) is None


@pytest.mark.parametrize(("compiling", "grad_enabled"), [(True, False), (False, True)])
def test_runtime_guards_bypass_operator_resolution(monkeypatch, compiling, grad_enabled):
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(torch, "is_grad_enabled", lambda: grad_enabled)

    def unexpected_resolve(_device):
        raise AssertionError("Ineligible runtime must not select eager operators")

    monkeypatch.setattr(ltx2_platform, "resolve_ltx2_vae_operators", unexpected_resolve)
    tensor = _TensorProbe(is_cuda=True, device=torch.device("cuda:0"))
    assert not ltx2_platform.is_ltx2_fusion_eligible(tensor)
    assert not ltx2_platform.is_ltx2_fna_eligible(tensor)


def test_device_capability_is_cached(monkeypatch):
    calls = 0
    monkeypatch.setattr(ltx2_platform, "HAS_TRITON", True)

    def get_device_capability(device_id=0):
        nonlocal calls
        calls += 1
        return DeviceCapability(major=9, minor=0)

    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "is_available", lambda: True)
    monkeypatch.setattr(ltx2_platform.current_omni_platform, "get_device_capability", get_device_capability)
    assert ltx2_platform._cuda_compute_capability(0) == 90
    assert ltx2_platform._cuda_compute_capability(0) == 90
    assert calls == 1
