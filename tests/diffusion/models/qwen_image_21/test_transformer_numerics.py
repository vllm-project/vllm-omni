# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_marks, hardware_test
from vllm_omni.diffusion.models.qwen_image_21 import qwen_image_21_transformer as transformer

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cpu
@pytest.mark.parametrize("invalid_norm", ["query_bias", "key_bias", "different_eps"])
def test_qkv_dispatch_rejects_unsupported_norms(monkeypatch, invalid_norm):
    attention = transformer.QwenImage21Attention.__new__(transformer.QwenImage21Attention)
    torch.nn.Module.__init__(attention)
    attention.norm_q = torch.nn.RMSNorm(64, eps=1e-5)
    attention.norm_k = torch.nn.RMSNorm(64, eps=1e-5)
    attention.norm_q.bias = attention.norm_k.bias = None
    if invalid_norm == "query_bias":
        attention.norm_q.bias = torch.nn.Parameter(torch.ones(64))
    elif invalid_norm == "key_bias":
        attention.norm_k.bias = torch.nn.Parameter(torch.ones(64))
    else:
        attention.norm_k.eps = 1e-6

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported norm must not enter either fused path")

    monkeypatch.setattr(transformer, "try_fused_qkv_norm_rope", unexpected)
    monkeypatch.setattr(transformer, "qk_rotary", unexpected)
    q = torch.ones(1, 3, 2, 64)
    tables = (torch.ones(3, 64), torch.zeros(3, 64))
    assert attention._try_fused_qkv_norm_rope(q, q, q, torch.empty(3, 32), tables) is None


@pytest.mark.cpu
@pytest.mark.parametrize(
    "route", ["qkv", "qk_with_tables", "qk_without_tables", "none_with_tables", "none_without_tables"]
)
def test_qkv_dispatch_uses_operator_tables_only_when_present(monkeypatch, route):
    attention = transformer.QwenImage21Attention.__new__(transformer.QwenImage21Attention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 64
    attention.norm_q = torch.nn.RMSNorm(64, eps=1e-5)
    attention.norm_k = torch.nn.RMSNorm(64, eps=1e-5)
    attention.norm_q.bias = attention.norm_k.bias = None
    q, k, v = (torch.full((1, 3, 2, 64), value) for value in (1.0, 2.0, 3.0))
    freqs = torch.empty(3, 32, dtype=torch.complex64)
    tables = None if route.endswith("without_tables") else (torch.ones(3, 64), torch.zeros(3, 64))
    qkv_outputs = (q + 10, k + 10, v.clone())
    qk_outputs = (q + 20, k + 20)
    calls = []

    def fused_qkv(query, key, value, q_weight, k_weight, operator_tables, eps, **kwargs):
        calls.append("qkv")
        assert operator_tables is tables
        return qkv_outputs if route == "qkv" else None

    def fused_qk(query, key, q_weight, k_weight, rotary_freqs, eps):
        calls.append("qk")
        assert rotary_freqs is freqs
        return qk_outputs if route.startswith("qk_") else None

    monkeypatch.setattr(transformer, "try_fused_qkv_norm_rope", fused_qkv)
    monkeypatch.setattr(transformer, "qk_rotary", fused_qk)
    result = attention._try_fused_qkv_norm_rope(q, k, v, freqs, tables)
    if route == "qkv":
        assert result is qkv_outputs
        assert calls == ["qkv"]
    else:
        assert calls == (["qk"] if tables is None else ["qkv", "qk"])
        if route.startswith("qk_"):
            assert result[0] is qk_outputs[0] and result[1] is qk_outputs[1]
            assert result[2] is v
        else:
            assert result is None


@pytest.mark.cpu
@pytest.mark.parametrize("with_operator_tables", [False, True])
def test_qkv_dispatch_propagates_execution_errors(monkeypatch, with_operator_tables):
    attention = transformer.QwenImage21Attention.__new__(transformer.QwenImage21Attention)
    torch.nn.Module.__init__(attention)
    attention.head_dim = 64
    attention.norm_q = torch.nn.RMSNorm(64, eps=1e-5)
    attention.norm_k = torch.nn.RMSNorm(64, eps=1e-5)
    attention.norm_q.bias = attention.norm_k.bias = None

    def fail(*args, **kwargs):
        raise RuntimeError("fusion execution failed")

    monkeypatch.setattr(transformer, "try_fused_qkv_norm_rope", fail)
    monkeypatch.setattr(transformer, "qk_rotary", fail)
    q = torch.ones(1, 3, 2, 64)
    tables = (torch.ones(3, 64), torch.zeros(3, 64)) if with_operator_tables else None
    with pytest.raises(RuntimeError, match="fusion execution failed"):
        attention._try_fused_qkv_norm_rope(q, q, q, torch.empty(3, 32), tables)


@pytest.mark.cpu
@pytest.mark.parametrize("cache_dtype", [None, "fp8_e4m3", "fp8_e4m3_v"])
@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("heads", [1, 2])
def test_qkv_fusion_preserves_prefix_cache_and_sp_joint(monkeypatch, cache_dtype, sp, heads):
    """Exercise routing with local TP heads, batch two and both CFG caches.

    The CPU fusion stand-in checks data flow; the shared-layer NPU test checks
    the real kernel. Cached K must never be normalized or rotated a second time.
    """
    attention = transformer.QwenImage21Attention.__new__(transformer.QwenImage21Attention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = heads
    attention.head_dim = 64
    attention.prefix_kv_cache_dtype = cache_dtype
    attention.norm_q = torch.nn.RMSNorm(64)
    attention.norm_k = torch.nn.RMSNorm(64)
    # Match the Diffusers norm interface used by the production module.
    attention.norm_q.bias = attention.norm_k.bias = None
    attention.norm_q.eps = attention.norm_k.eps = 1e-5
    attention.to_out = torch.nn.Identity()

    class Projection(torch.nn.Module):
        def forward(self, x):
            return torch.cat((x, x * 2, x * 3), -1), None

    received = []

    class Capture(torch.nn.Module):
        def forward(self, q, k, v, metadata):
            received.append((q.clone(), k.clone(), v.clone(), metadata))
            return q

    attention.to_qkv = Projection()
    attention.attn = Capture()
    calls = []

    def fused(q, k, v, *args, **kwargs):
        calls.append(q.shape)
        assert not q.is_contiguous()
        # Make accidental reuse/double-processing of cached K observable.
        return q + 1, k + 7, v

    monkeypatch.setattr(transformer, "try_fused_qkv_norm_rope", fused)
    cache: dict[str, dict[str, torch.Tensor]] = {}
    tables = (torch.ones(5, 64), torch.zeros(5, 64))
    with torch.inference_mode():
        for branch, offset in (("cond", 0.0), ("uncond", 1.0)):
            x = torch.randn(2, 5, heads * 64) + offset
            attention(
                x,
                torch.empty(5, 32),
                kv_cache=cache,
                cache_branch=branch,
                cache_write_len=2,
                sp_prefix_len=2 if sp else 0,
                norm_rope_concat_tables=tables,
            )
            prefix_key = (x * 2).unflatten(-1, (heads, 64))[:, :2] + 7
            prefix_value = (x * 3).unflatten(-1, (heads, 64))[:, :2]
            if cache_dtype == "fp8_e4m3":
                key, scale = transformer._quantize_prefix_kv_fp8(prefix_key)
                prefix_key = transformer._dequantize_prefix_kv_fp8(key, scale, x.dtype)
            if cache_dtype is not None:
                value, scale = transformer._quantize_prefix_kv_fp8(prefix_value)
                prefix_value = transformer._dequantize_prefix_kv_fp8(value, scale, x.dtype)
            if sp:
                metadata = received[-1][3]
                torch.testing.assert_close(metadata.joint_key, (x * 2).unflatten(-1, (heads, 64))[:, :2] + 7)
                assert metadata.joint_strategy == "front"
            decode = x[:, 2:]
            attention(
                decode,
                torch.empty(3, 32),
                kv_cache=cache,
                cache_branch=branch,
                sp_decode=sp,
                norm_rope_concat_tables=tuple(t[2:] for t in tables),
            )
            q, k, v, metadata = received[-1]
            current_key = (decode * 2).unflatten(-1, (heads, 64)) + 7
            if sp:
                assert metadata.joint_query.shape == (2, 0, heads, 64)
                torch.testing.assert_close(metadata.joint_key, prefix_key)
                torch.testing.assert_close(metadata.joint_value, prefix_value)
                torch.testing.assert_close(k, current_key)
            else:
                torch.testing.assert_close(k[:, :2], prefix_key)
                torch.testing.assert_close(v[:, :2], prefix_value)
                torch.testing.assert_close(k[:, 2:], current_key)
        assert [shape[1] for shape in calls] == [5, 3, 5, 3]


@pytest.mark.cpu
@pytest.mark.parametrize("fused_qk", [False, True])
def test_qkv_unavailable_preserves_existing_norm_rope(monkeypatch, fused_qk):
    attention = transformer.QwenImage21Attention.__new__(transformer.QwenImage21Attention)
    torch.nn.Module.__init__(attention)
    attention.num_heads = attention.num_kv_heads = 2
    attention.head_dim = 64
    # Keep this CPU routing test independent of Diffusers' process-wide NPU
    # dispatch, which selects npu_rms_norm even for CPU inputs on an NPU host.
    attention.norm_q = torch.nn.RMSNorm(64, eps=1e-5)
    attention.norm_k = torch.nn.RMSNorm(64, eps=1e-5)
    attention.norm_q.bias = attention.norm_k.bias = None
    attention.to_out = torch.nn.Identity()
    attention._apply_rotary_emb = transformer._apply_qwen_image21_rotary_emb_native

    class Projection(torch.nn.Module):
        def forward(self, x):
            return torch.cat([x] * 3, -1), None

    class Capture(torch.nn.Module):
        def forward(self, q, k, v, metadata):
            return q

    attention.to_qkv = Projection()
    attention.attn = Capture()
    monkeypatch.setattr(transformer, "try_fused_qkv_norm_rope", lambda *a, **kw: None)
    monkeypatch.setattr(transformer, "qk_rotary", lambda q, k, *a: (q + 9, k + 9) if fused_qk else None)
    x = torch.randn(2, 3, 128)
    freqs = torch.polar(torch.ones(3, 32), torch.randn(3, 32))
    with torch.inference_mode():
        actual = attention(x, freqs, norm_rope_concat_tables=(torch.ones(3, 64), torch.zeros(3, 64)))
        q = x.unflatten(-1, (2, 64))
        expected = q + 9 if fused_qk else attention._apply_rotary_emb(attention.norm_q(q), freqs)
    torch.testing.assert_close(actual, expected.flatten(2, 3), rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("cpu", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_qk_norm_matches_diffusers_rounding(monkeypatch, device, dtype):
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )
    monkeypatch.setattr(transformer, "Attention", lambda **kwargs: torch.nn.Identity())
    attention = transformer.QwenImage21Attention(dim=256, heads=2, dim_head=128).to(device=device, dtype=dtype)
    values = torch.randn(2, 17, 2, 128, generator=torch.Generator().manual_seed(42)).to(device=device, dtype=dtype)
    for norm in (attention.norm_q, attention.norm_k):
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 128, device=device, dtype=dtype))
        variance = values.float().square().mean(-1, keepdim=True)
        expected = (values * torch.rsqrt(variance + norm.eps)).to(dtype) * norm.weight
        torch.testing.assert_close(norm(values), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "device",
    [
        pytest.param("meta", marks=pytest.mark.cpu),
        pytest.param("cuda", marks=hardware_marks(res={"cuda": "L4"}, num_cards=1)),
    ],
)
def test_rope_frequencies_do_not_depend_on_loader_device(device):
    reference = transformer.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    with torch.device(device):
        loaded = transformer.QwenImage21Rope(theta=10000, axes_dim=[16, 56, 56])
    for expected, actual in zip(reference.freqs, loaded.freqs):
        assert actual.device.type == "cpu"
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_scheduler_matches_diffusers_on_latent_device():
    import numpy as np
    from diffusers import FlowMatchEulerDiscreteScheduler

    from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline

    pipeline = object.__new__(QwenImage21Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cuda", 0)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True)
    timesteps, _ = pipeline.prepare_timesteps(50, None, 4096)
    reference = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    reference.set_timesteps(sigmas=np.linspace(1.0, 1 / 50, 50), device=pipeline.device, mu=1.15)
    sample = torch.randn(1, 16, 64, generator=torch.Generator().manual_seed(42)).to(
        device=pipeline.device, dtype=torch.bfloat16
    )
    noise = sample.cos()
    actual = pipeline.scheduler.step(noise, timesteps[0], sample, return_dict=False)[0]
    expected = reference.step(noise, reference.timesteps[0], sample, return_dict=False)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert pipeline.scheduler.sigmas.device == sample.device
    assert timesteps.device == sample.device


@hardware_test(res={"cuda": "L4"}, num_cards=1)
def test_timestep_frequencies_match_cpu_initialization():
    reference = transformer.QwenImage21TemporalTimesteps(timestep_dim=256)
    with torch.device("cuda"):
        loaded = transformer.QwenImage21TemporalTimesteps(timestep_dim=256)
    assert loaded.freqs.device.type == "cuda"
    torch.testing.assert_close(loaded.freqs.cpu(), reference.freqs, rtol=0, atol=0)
    timesteps = torch.tensor([0.0, 0.984375, 1.0], device="cuda")
    torch.testing.assert_close(loaded(timesteps), reference.to("cuda")(timesteps), rtol=0, atol=0)
