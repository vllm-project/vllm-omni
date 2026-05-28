# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest
import torch
from vllm.platforms.interface import DeviceCapability

from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
from vllm_omni.diffusion.envs import PACKAGES_CHECKER
from vllm_omni.platforms.cuda import platform as cuda_platform_module
from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

SAGE_ATTN3_MODULE = "vllm_omni.diffusion.attention.backends.sage_attn3"


def load_sage_attn3_module(monkeypatch: pytest.MonkeyPatch, kernel_impl):
    fake_module = types.ModuleType("sageattn3")
    fake_module.sageattn3_blackwell = kernel_impl
    monkeypatch.setitem(sys.modules, "sageattn3", fake_module)
    sys.modules.pop(SAGE_ATTN3_MODULE, None)
    return importlib.import_module(SAGE_ATTN3_MODULE)


def test_sage_attn3_forward_uses_blackwell_layout(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_kernel(query, key, value, is_causal=False):
        calls["query_shape"] = query.shape
        calls["is_causal"] = is_causal
        return query + key + value

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    impl = backend_module.SageAttention3Impl(
        num_heads=4,
        head_size=64,
        softmax_scale=1.0 / 8.0,
        causal=False,
    )

    query = torch.randn(2, 8, 4, 64)
    key = torch.randn(2, 8, 4, 64)
    value = torch.randn(2, 8, 4, 64)

    output = impl.forward_cuda(query, key, value)

    assert calls["query_shape"] == (2, 4, 8, 64)
    assert calls["is_causal"] is False
    expected = (query.transpose(1, 2) + key.transpose(1, 2) + value.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(output, expected)


def test_sage_attn3_falls_back_to_sdpa_for_gqa(monkeypatch: pytest.MonkeyPatch):
    def fake_kernel(*args, **kwargs):
        raise AssertionError("sageattn3_blackwell should not be used for GQA")

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    sdpa_calls = {}

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_calls["query_shape"] = query.shape
        sdpa_calls["key_shape"] = key.shape
        sdpa_calls["enable_gqa"] = kwargs["enable_gqa"]
        return query + 1

    monkeypatch.setattr(backend_module.F, "scaled_dot_product_attention", fake_sdpa)

    impl = backend_module.SageAttention3Impl(
        num_heads=4,
        head_size=64,
        softmax_scale=1.0 / 8.0,
        causal=False,
    )

    query = torch.randn(2, 8, 4, 64)
    key = torch.randn(2, 8, 2, 64)
    value = torch.randn(2, 8, 2, 64)

    output = impl.forward_cuda(query, key, value)

    assert sdpa_calls["query_shape"] == (2, 4, 8, 64)
    assert sdpa_calls["key_shape"] == (2, 2, 8, 64)
    assert sdpa_calls["enable_gqa"] is True
    expected = (query.permute(0, 2, 1, 3) + 1).permute(0, 2, 1, 3)
    assert torch.allclose(output, expected)


def test_cuda_platform_selects_sage_attn3_alias(monkeypatch: pytest.MonkeyPatch):
    original_import_module = importlib.import_module

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    monkeypatch.setattr(
        cuda_platform_module.importlib,
        "import_module",
        lambda module_name: object() if module_name == "sageattn3" else original_import_module(module_name),
    )

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)

    assert backend_path == DiffusionAttentionBackendEnum.SAGE_ATTN_3.get_path()


def test_cuda_platform_falls_back_when_sage_attn3_gpu_is_unsupported(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)

    assert backend_path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
