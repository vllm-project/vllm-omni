# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for tower dispatch and tensor plumbing, not NPU kernel accuracy."""

import importlib.util
import sys
from enum import Enum
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.fish_omni.kernels import (
    eager_qk_rmsnorm_rope_gate,
    gemma_rms_norm,
    tower_mix_norm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

EPS = 1e-6


class AscendDeviceType(Enum):
    A2 = "A2"
    A3 = "A3"
    A5 = "A5"
    _310P = "310P"


def _gemma_op(x, weight, eps):
    return gemma_rms_norm(x, weight, eps), torch.empty(0)


def _rotary_mul(x, cos, sin, unsqueeze_dim):
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos.unsqueeze(unsqueeze_dim) + rotated * sin.unsqueeze(unsqueeze_dim)


def _load_tower_kernels():
    spec = importlib.util.find_spec("vllm_omni.platforms.npu.models.fish_omni_tower")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def kernels(monkeypatch):
    # Stub device detection and external kernels; run the real adapter and eager references.
    ascend_utils = ModuleType("vllm_ascend.utils")
    monkeypatch.setattr(ascend_utils, "AscendDeviceType", AscendDeviceType, raising=False)
    monkeypatch.setattr(ascend_utils, "get_ascend_device_type", lambda: AscendDeviceType.A2, raising=False)
    monkeypatch.setitem(sys.modules, ascend_utils.__name__, ascend_utils)
    extension_name = "vllm_ascend.vllm_ascend_C"
    monkeypatch.setitem(sys.modules, extension_name, ModuleType(extension_name))
    rotary_name = "vllm_omni.platforms.npu.layers.rotary_embedding"
    rotary = ModuleType(rotary_name)
    monkeypatch.setattr(rotary, "npu_rotary_mul_with_bsnd_fallback", Mock(side_effect=_rotary_mul), raising=False)
    monkeypatch.setitem(sys.modules, rotary_name, rotary)
    monkeypatch.setattr(torch.ops._C_ascend, "npu_gemma_rms_norm", Mock(side_effect=_gemma_op), raising=False)
    return _load_tower_kernels()


@pytest.mark.parametrize(
    "device_type,uses_fused",
    [
        (AscendDeviceType.A2, True),
        (AscendDeviceType.A3, True),
        (AscendDeviceType.A5, False),
        (AscendDeviceType._310P, False),
    ],
)
def test_tower_fuses_only_on_a2_and_a3(kernels, monkeypatch, device_type, uses_fused):
    gemma_op = kernels._gemma_op
    monkeypatch.setattr(sys.modules["vllm_ascend.utils"], "get_ascend_device_type", lambda: device_type)
    kernels = _load_tower_kernels()
    assert kernels._gemma_op is (gemma_op if uses_fused else None)

    q_gate = torch.arange(32, dtype=torch.bfloat16).view(1, 32)
    key = torch.arange(16, dtype=torch.bfloat16).view(1, 16)
    weight = torch.zeros(16, dtype=torch.bfloat16)
    cache = torch.tensor([[0.0] * 4 + [1.0] * 4])
    positions = torch.tensor([0])
    args = (q_gate, key, weight, weight, cache, positions, EPS, 1, 1, 16, 8)
    expected = eager_qk_rmsnorm_rope_gate(*args, norm_beta=1.0)
    actual = kernels.qk_rmsnorm_rope_gate(*args, norm_beta=1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert gemma_op.call_count == (2 if uses_fused else 0)
    assert kernels.npu_rotary_mul_with_bsnd_fallback.call_count == (2 if uses_fused else 0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("with_residual", [False, True])
@pytest.mark.parametrize("rows", [0, 3])
def test_norm_matches_reference(kernels, dtype, with_residual, rows):
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(rows, 16, generator=generator, dtype=dtype)
    weight = torch.randn(16, generator=generator, dtype=dtype)
    residual = torch.randn(rows, 16, generator=generator, dtype=dtype) if with_residual else None
    expected = gemma_rms_norm(x, weight, EPS, residual)
    actual = kernels.gemma_rms_norm(x, weight, EPS, residual)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert kernels._gemma_op.call_count == int(rows > 0 and dtype == torch.bfloat16 and not with_residual)


@pytest.mark.parametrize("with_extra", [False, True])
@pytest.mark.parametrize("with_fused_norm", [False, True])
def test_mix_matches_reference(kernels, with_extra, with_fused_norm):
    if not with_fused_norm:
        kernels._gemma_op = None
    generator = torch.Generator().manual_seed(42)
    hidden = torch.randn(3, 16, generator=generator, dtype=torch.bfloat16)
    table = torch.randn(4, 16, generator=generator, dtype=torch.bfloat16)
    extra = torch.randn(3, 16, generator=generator, dtype=torch.bfloat16) if with_extra else None
    enorm = torch.randn(16, generator=generator, dtype=torch.bfloat16)
    hnorm = torch.randn(16, generator=generator, dtype=torch.bfloat16)
    code = torch.tensor([-2, 2, 9])
    mask = torch.tensor([True, False, True])
    args = (hidden, code, mask, table, extra, enorm, hnorm, EPS)
    torch.testing.assert_close(kernels.tower_mix_norm(*args), tower_mix_norm(*args), rtol=0, atol=0)


@pytest.mark.parametrize("tokens", [0, 1, 9])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("with_fused_norm", [False, True])
def test_qk_matches_reference_without_mutating_inputs(kernels, tokens, dtype, with_fused_norm):
    if not with_fused_norm:
        kernels._gemma_op = None
    generator = torch.Generator().manual_seed(42)
    heads, kv_heads, head_dim, rotary_dim = 3, 1, 16, 8
    q_size, kv_size = heads * head_dim, kv_heads * head_dim
    qkv = torch.randn(tokens, 2 * q_size + 2 * kv_size, generator=generator, dtype=dtype)
    original = qkv.clone()
    q_gate, key, _ = qkv.split([2 * q_size, kv_size, kv_size], dim=-1)
    q_weight = torch.randn(head_dim, generator=generator, dtype=dtype)
    k_weight = torch.randn(head_dim, generator=generator, dtype=dtype)
    angles = torch.randn(32, rotary_dim // 2, generator=generator)
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1)
    positions = torch.arange(tokens, dtype=torch.long) * 2
    args = (q_gate, key, q_weight, k_weight, cache, positions, EPS, heads, kv_heads, head_dim, rotary_dim)
    expected = eager_qk_rmsnorm_rope_gate(*args, norm_beta=1.0)
    actual = kernels.qk_rmsnorm_rope_gate(*args, norm_beta=1.0)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(qkv, original, rtol=0, atol=0)
    assert kernels.npu_rotary_mul_with_bsnd_fallback.call_count == (2 if tokens > 0 and dtype == torch.bfloat16 else 0)


@pytest.mark.parametrize("norm_beta,mrope_section", [(0.0, None), (1.0, (1, 1, 0))])
def test_qk_unsupported_geometry_uses_reference(kernels, norm_beta, mrope_section):
    q_gate = torch.arange(32, dtype=torch.bfloat16).view(1, 32)
    key = torch.arange(16, dtype=torch.bfloat16).view(1, 16)
    weight = torch.zeros(16, dtype=torch.bfloat16)
    cache = torch.tensor([[1.0] * 4 + [0.0] * 4])
    positions = torch.tensor([0])
    args = (q_gate, key, weight, weight, cache, positions, EPS, 1, 1, 16, 8, mrope_section, norm_beta)
    torch.testing.assert_close(kernels.qk_rmsnorm_rope_gate(*args), eager_qk_rmsnorm_rope_gate(*args), rtol=0, atol=0)
    kernels._gemma_op.assert_not_called()
    kernels.npu_rotary_mul_with_bsnd_fallback.assert_not_called()


def test_missing_operator_uses_reference(kernels, monkeypatch):
    monkeypatch.delattr(torch.ops._C_ascend, "npu_gemma_rms_norm")
    kernels = _load_tower_kernels()
    assert kernels._gemma_op is None
    x = torch.ones(1, 16, dtype=torch.bfloat16)
    weight = torch.zeros(16, dtype=torch.bfloat16)
    torch.testing.assert_close(kernels.gemma_rms_norm(x, weight, EPS), gemma_rms_norm(x, weight, EPS))


def test_resolved_operator_is_reused_and_runtime_errors_propagate(kernels, monkeypatch):
    monkeypatch.delattr(torch.ops._C_ascend, "npu_gemma_rms_norm")
    kernels._gemma_op.side_effect = RuntimeError("device failure")
    with pytest.raises(RuntimeError, match="device failure"):
        kernels.gemma_rms_norm(torch.ones(1, 16, dtype=torch.bfloat16), torch.zeros(16, dtype=torch.bfloat16), EPS)
