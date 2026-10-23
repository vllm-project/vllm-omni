# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Independent numeric checks for the SenseNova temporal/spatial QK split."""

import builtins
from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.sensenova_u1.sensenova_u1_transformer import (
    Qwen3RMSNorm,
    SenseNovaU1Attention,
    apply_rotary_pos_emb,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _norm_reference(x, weight, eps):
    x = x.double()
    return x / (x.square().mean(dim=-1, keepdim=True) + eps).sqrt() * weight.double()


def _rotation_reference(x, cos, sin):
    """Matrix form independently defines the half rotation, including its sign."""
    width = x.shape[-1]
    rotation = torch.zeros(width, width, dtype=torch.float64)
    eye = torch.eye(width // 2, dtype=torch.float64)
    rotation[: width // 2, width // 2 :] = eye
    rotation[width // 2 :, : width // 2] = -eye
    return x.double() * cos.double().unsqueeze(1) + (x.double() @ rotation) * sin.double().unsqueeze(1)


def _segmented_reference(x, time_weight, spatial_weight, tables):
    width = x.shape[-1]
    # The entire spatial half shares one RMS denominator: normalizing the h/w
    # quarters separately is incorrect, even though RoPE rotates them separately.
    time = _norm_reference(x[..., : width // 2], time_weight, 1e-6)
    spatial = _norm_reference(x[..., width // 2 :], spatial_weight, 1e-6)
    sections = (time, spatial[..., : width // 4], spatial[..., width // 4 :])
    return torch.cat(
        [_rotation_reference(section.transpose(1, 2), *table) for section, table in zip(sections, tables)],
        dim=-1,
    )


def _inputs(batch, seq, head_dim, dtype):
    generator = torch.Generator().manual_seed(151)
    # Deliberately different spatial-quarter scales detect an incorrect norm axis.
    q = torch.randn(batch, seq, 4, head_dim, generator=generator, dtype=dtype)
    k = torch.randn(batch, seq, 1, head_dim, generator=generator, dtype=dtype)
    q[..., head_dim // 2 : head_dim * 3 // 4] *= 5
    k[..., head_dim * 3 // 4 :] *= 4
    v = torch.randn(batch, seq, 1, head_dim, generator=generator, dtype=dtype)
    tables = []
    for axis, width in enumerate((head_dim // 2, head_dim // 4, head_dim // 4)):
        angles = torch.arange(seq).double()[:, None] * (axis + 1.25)
        angles = angles * torch.linspace(0.1, 0.9, width // 2)[None, :]
        angles = angles.repeat(1, 2).unsqueeze(0)
        tables.append((angles.cos().to(dtype), angles.sin().to(dtype)))
    weights = [torch.rand(head_dim // 2, generator=generator, dtype=dtype) + 0.5 for _ in range(4)]
    return q, k, v, tables, weights


def _project(q, k, v, tables, weights, device):
    width = q.shape[-1]
    host = SimpleNamespace(
        num_heads=q.shape[2],
        num_kv_heads=k.shape[2],
        head_dim=width,
        q_size=q.shape[2] * width,
        kv_size=k.shape[2] * width,
        config=SimpleNamespace(rms_norm_eps=1e-6),
    )
    packed = torch.cat([tensor.flatten(2) for tensor in (q, k, v)], dim=-1).to(device)
    norms = [Qwen3RMSNorm(width // 2).to(device=device, dtype=q.dtype) for _ in weights]
    for norm, weight in zip(norms, weights):
        with torch.no_grad():
            norm.weight.copy_(weight.to(device))
    device_tables = tuple(tuple(table.to(device) for table in pair) for pair in tables)
    return SenseNovaU1Attention._project_and_rope(host, packed, device_tables, lambda x: (x, None), *norms)


def _check_project(device, batch, seq, head_dim, dtype):
    q, k, v, tables, weights = _inputs(batch, seq, head_dim, dtype)
    got_q, got_k, got_v = _project(q, k, v, tables, weights, device)
    want_q = _segmented_reference(q, weights[0], weights[2], tables)
    want_k = _segmented_reference(k, weights[1], weights[3], tables)
    tolerance = 0.04 if dtype == torch.bfloat16 else 2e-5
    for actual, expected in ((got_q, want_q), (got_k, want_k)):
        assert actual.dtype == dtype
        torch.testing.assert_close(actual.cpu().double(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(got_v.cpu(), v.transpose(1, 2), atol=0, rtol=0)


@pytest.mark.cpu
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float64])
@pytest.mark.parametrize("batch,seq,head_dim", [(1, 1, 128), (2, 5, 128), (2, 3, 24)])
def test_segmented_qk_norm_rope_reference(dtype, batch, seq, head_dim):
    _check_project("cpu", batch, seq, head_dim, dtype)


@pytest.mark.cpu
def test_cpu_does_not_import_accelerator_kernels(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_npu" or "fused_rmsnorm_rope" in name:
            raise AssertionError(f"CPU execution imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _check_project("cpu", 1, 2, 128, torch.bfloat16)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("batch,seq,head_dim", [(1, 1, 128), (2, 5, 128), (2, 3, 24)])
def test_npu_segmented_qk_norm_rope_reference(batch, seq, head_dim):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    _check_project("npu", batch, seq, head_dim, torch.bfloat16)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("width", [64, 4096])
def test_npu_rmsnorm_reference(width):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    torch.manual_seed(89)
    x = torch.randn(2, 3, width, dtype=torch.bfloat16)
    norm = Qwen3RMSNorm(width).to(device="npu", dtype=x.dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 1.5, width, dtype=x.dtype).to("npu"))
    expected = _norm_reference(x, norm.weight.cpu(), 1e-6)
    actual = norm(x.to("npu")).cpu()
    torch.testing.assert_close(actual.double(), expected, atol=0.03, rtol=0.03)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_npu_kernel_execution_errors_propagate(monkeypatch):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU unavailable")

    def fail(*args, **kwargs):
        raise RuntimeError("kernel execution failed")

    x = torch.ones(1, 1, 1, 64, device="npu", dtype=torch.bfloat16)
    norm = Qwen3RMSNorm(64).to(device="npu", dtype=x.dtype)
    monkeypatch.setattr(torch_npu, "npu_rms_norm", fail)
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        norm(x)
    monkeypatch.setattr(torch_npu.npu, "is_jit_compile_false", lambda: True)
    monkeypatch.setattr(torch_npu, "npu_rotary_mul", fail)
    with pytest.raises(RuntimeError, match="kernel execution failed"):
        apply_rotary_pos_emb(x, x, x[:, 0], x[:, 0])


@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("batch,seq,head_dim", [(1, 5, 128), (2, 5, 128), (1, 3, 24)])
def test_cuda_segmented_rope_keeps_npu_dependency_optional(monkeypatch, batch, seq, head_dim):
    from vllm_omni.platforms import current_omni_platform

    if not current_omni_platform.is_cuda():
        pytest.skip("CUDA unavailable")
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch_npu":
            raise AssertionError("CUDA execution imported torch_npu")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    _check_project("cuda", batch, seq, head_dim, torch.bfloat16)


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_npu_rope_jit_mode_uses_numeric_fallback(monkeypatch):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    q, k, _, tables, _ = _inputs(2, 3, 128, torch.bfloat16)
    q = q[..., :64].transpose(1, 2)
    k = k[..., :64].transpose(1, 2)
    cos, sin = tables[0]
    monkeypatch.setattr(torch_npu.npu, "is_jit_compile_false", lambda: False)

    def fail(*args, **kwargs):
        raise AssertionError("unsupported compile mode selected fused RoPE")

    monkeypatch.setattr(torch_npu, "npu_rotary_mul", fail)
    actual = apply_rotary_pos_emb(q.to("npu"), k.to("npu"), cos.to("npu"), sin.to("npu"))
    for result, source in zip(actual, (q, k)):
        expected = _rotation_reference(source, cos, sin)
        torch.testing.assert_close(result.cpu().double(), expected, atol=0.04, rtol=0.04)


@hardware_test(res={"npu": "A3"}, num_cards=1)
@pytest.mark.parametrize("ndim", [1, 9])
def test_npu_rmsnorm_unqualified_rank_uses_fallback(monkeypatch, ndim):
    torch_npu = pytest.importorskip("torch_npu")
    if not torch_npu.npu.is_available():
        pytest.skip("Ascend NPU unavailable")
    x = torch.linspace(-2, 2, 64, dtype=torch.bfloat16).reshape(*([1] * (ndim - 1)), 64)
    norm = Qwen3RMSNorm(64).to(device="npu", dtype=x.dtype)

    def fail(*args, **kwargs):
        raise AssertionError("unsupported input rank selected fused RMSNorm")

    monkeypatch.setattr(torch_npu, "npu_rms_norm", fail)
    expected = _norm_reference(x, norm.weight.cpu(), 1e-6)
    actual = norm(x.to("npu")).cpu()
    torch.testing.assert_close(actual.double(), expected, atol=0.03, rtol=0.03)
