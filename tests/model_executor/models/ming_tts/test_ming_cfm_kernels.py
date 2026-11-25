# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Equivalence, dispatch, compile-count, and CUDA-graph tests for Ming CFM kernels."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from tests.helpers.mark import hardware_test
from vllm_omni.model_executor.models.common.ming import cfm_kernels
from vllm_omni.model_executor.models.common.ming.aggregator import Aggregator
from vllm_omni.model_executor.models.common.ming.cfm_cudagraph import CFMGraphExecutor, CFMSampler
from vllm_omni.model_executor.models.common.ming.cfm_dit import DiT
from vllm_omni.model_executor.models.common.ming.cfm_kernels import (
    MAX_FUSED_ATTN_SEQ_LEN,
    cfm_qkv_rope_attn_supported,
    cfm_triton_supported,
    fused_cfm_euler_sde_step,
)
from vllm_omni.model_executor.models.common.ming.cfm_solver import build_timesteps
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.tts]
nvidia_cuda = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is not None,
    reason="NVIDIA CUDA required",
)


def _tiny_dit(*, depth: int = 2, hidden_size: int = 128, num_heads: int = 2, native_attention: bool = False) -> DiT:
    dit = DiT(
        in_channels=16,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=2.0,
        llm_cond_dim=32,
        dropout=0.0,
        attn_mask_enabled=False,
        native_attention=native_attention,
    ).eval()
    for block in dit.blocks:
        block.attn.short_attention = True
        block.mlp.fused_gelu = True
    return dit


@pytest.mark.cpu
def test_cfm_triton_gating_respects_platform_and_cpu(monkeypatch: pytest.MonkeyPatch):
    cpu_tensor = torch.zeros(2, 3, 64)
    assert not cfm_triton_supported(cpu_tensor)
    assert not cfm_qkv_rope_attn_supported(cpu_tensor, heads=2, head_dim=64, seq_len=3)

    # Platform gate must disable Triton on non-CUDA platforms (e.g., ROCm/HIP).
    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: False)
    assert not cfm_triton_supported(cpu_tensor)


@pytest.mark.cpu
def test_cfm_sampler_falls_back_to_native_on_cpu():
    torch.manual_seed(0)
    dit = _tiny_dit()
    sampler = CFMSampler(dit, steps=3, include_final_noise=False)
    cond = torch.randn(2, 1, 32)
    hist = torch.randn(2, 2, 16)
    y0 = torch.randn(2, 1, 16)
    t = build_timesteps(3, device=torch.device("cpu"), dtype=torch.float32)
    sde_args = torch.tensor([2.0, 0.25, 0.5])
    sde_rnd = torch.randn(3, 2, 1, 16)

    assert not sampler._can_use_fast_path(cond, hist, y0)
    out = sampler(cond, hist, y0, t, sde_args, sde_rnd)
    ref = sampler.forward_native(cond, hist, y0, t, sde_args, sde_rnd)
    assert torch.equal(out, ref)


@nvidia_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("seq_len", [3, 5, 8, 37])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_fused_qkv_rope_attn_matches_native_and_respects_fallbacks(
    seq_len: int,
    dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
):
    torch.manual_seed(0)
    dit = _tiny_dit(depth=1).to(device="cuda", dtype=dtype)
    attn = dit.blocks[0].attn
    x = torch.randn(4, seq_len, 128, device="cuda", dtype=dtype)
    rope = dit.rotary_embed.forward_from_seq_len(seq_len)

    # Reference without fused Triton kernel.
    monkeypatch.setattr(cfm_kernels, "HAS_TRITON", False)
    ref = attn(x, mask=None, rope=rope)
    monkeypatch.undo()

    # Verify the fused kernel is actually invoked on the fast path.
    calls = 0
    real_fused = cfm_kernels.fused_qkv_rope_attn

    def _counting_fused(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_fused(*args, **kwargs)

    monkeypatch.setattr("vllm_omni.model_executor.models.common.ming.dit_blocks.fused_qkv_rope_attn", _counting_fused)
    got = attn(x, mask=None, rope=rope)
    assert calls == 1

    cos_sim = F.cosine_similarity(ref.float().reshape(-1), got.float().reshape(-1), dim=0).item()
    assert cos_sim > 0.9999
    torch.testing.assert_close(got, ref, atol=2e-2, rtol=2e-2)

    # native_attention=True and seq_len > MAX_FUSED_ATTN_SEQ_LEN must bypass fused kernel.
    calls = 0
    attn.native_attention = True
    _ = attn(x, mask=None, rope=rope)
    assert calls == 0
    attn.native_attention = False

    long_len = MAX_FUSED_ATTN_SEQ_LEN + 1
    x_long = torch.randn(2, long_len, 128, device="cuda", dtype=dtype)
    rope_long = dit.rotary_embed.forward_from_seq_len(long_len)
    _ = attn(x_long, mask=None, rope=rope_long)
    assert calls == 0


@nvidia_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize("patch_size", [1, 4])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("add_noise", [False, True])
@torch.inference_mode()
def test_fused_cfm_euler_sde_step_matches_native(
    patch_size: int,
    state_dtype: torch.dtype,
    add_noise: bool,
):
    torch.manual_seed(1)
    batch, dim = 4, 64
    pred_cfg = torch.randn(2 * batch, patch_size, dim, device="cuda", dtype=torch.bfloat16)
    y0 = torch.randn(batch, patch_size, dim, device="cuda", dtype=state_dtype)
    noise = torch.randn(batch, patch_size, dim, device="cuda", dtype=state_dtype)
    t = build_timesteps(10, device=torch.device("cuda"), dtype=state_dtype)
    sde_args = torch.tensor([2.0, 0.25, 0.5], device="cuda", dtype=state_dtype)

    dt = t[4] - t[3]
    pred, null_pred = torch.chunk(pred_cfg, 2, dim=0)
    flow = (pred + (pred - null_pred) * sde_args[0]).to(state_dtype)
    ref = y0 + flow * dt
    if add_noise:
        ref = ref + sde_args[1] * (sde_args[2] ** 0.5) * (dt.abs() ** 0.5) * noise

    got, got_cast = fused_cfm_euler_sde_step(
        pred_cfg, y0, noise, t, sde_args, 3, add_noise=add_noise, model_dtype=torch.bfloat16
    )
    assert got.dtype == state_dtype
    assert got_cast.dtype == torch.bfloat16
    atol = 2e-2 if state_dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(got, ref, atol=atol, rtol=atol)
    torch.testing.assert_close(got_cast, got.to(torch.bfloat16), atol=0.0, rtol=0.0)


@nvidia_cuda
@hardware_test(res={"cuda": "L4"}, num_cards=1)
@pytest.mark.parametrize(
    ("patch_size", "history_len", "noise_dtype", "include_final_noise"),
    [
        (1, 1, torch.bfloat16, True),
        (4, 32, torch.float32, False),
    ],
)
@torch.inference_mode()
def test_cfm_sampler_fast_matches_native_and_replays_in_cudagraph(
    patch_size: int,
    history_len: int,
    noise_dtype: torch.dtype,
    include_final_noise: bool,
    monkeypatch: pytest.MonkeyPatch,
):
    from vllm.triton_utils import triton

    torch.manual_seed(2)
    dit = _tiny_dit(depth=2).to(device="cuda", dtype=torch.bfloat16)
    agg = (
        Aggregator(
            in_channels=16, hidden_size=64, depth=1, num_heads=2, llm_input_dim=32, dropout=0.0, attn_mask_enabled=False
        )
        .to(device="cuda", dtype=torch.bfloat16)
        .eval()
    )
    stop_head = torch.nn.Linear(32, 2).to(device="cuda", dtype=torch.bfloat16).eval()
    sampler = CFMSampler(dit, steps=4, include_final_noise=include_final_noise)

    # Verify fast path matches forward_native numerically.
    cond = torch.randn(3, 1, 32, device="cuda", dtype=torch.bfloat16)
    hist = torch.randn(3, history_len, 16, device="cuda", dtype=torch.bfloat16)
    y0 = torch.randn(3, patch_size, 16, device="cuda", dtype=noise_dtype)
    sde_rnd = torch.randn(4, 3, patch_size, 16, device="cuda", dtype=noise_dtype)
    t = build_timesteps(4, device=torch.device("cuda"), dtype=noise_dtype)
    sde_args = torch.tensor([2.0, 0.25, 0.5], device="cuda", dtype=noise_dtype)

    ref = sampler.forward_native(cond, hist, y0, t, sde_args, sde_rnd)
    got = sampler(cond, hist, y0, t, sde_args, sde_rnd)
    cos_sim = F.cosine_similarity(ref.float().reshape(-1), got.float().reshape(-1), dim=0).item()
    assert cos_sim > 0.9995

    # Verify CUDA graph capture & replay across padded live batch (3 on bucket 4)
    # and multiple buckets without recompiling Triton kernels per batch size.
    cfm_kernels._qkv_rope_attn_kernel.device_caches.clear()
    cfm_kernels._cfm_euler_sde_step_kernel.device_caches.clear()
    exec_b1 = CFMGraphExecutor(sampler, agg, stop_head, patch_size=patch_size, latent_dim=16, batch_size=1, steps=4)
    exec_b4 = CFMGraphExecutor(sampler, agg, stop_head, patch_size=patch_size, latent_dim=16, batch_size=4, steps=4)
    exec_b8 = CFMGraphExecutor(sampler, agg, stop_head, patch_size=patch_size, latent_dim=16, batch_size=8, steps=4)

    exec_b1.execute(cond[:1], hist[:1], 2.0, 0.25, 0.5, initial_noise=y0[:1], solver_noise=sde_rnd[:, :1])
    compiles: list[str] = []
    monkeypatch.setattr(triton.knobs.runtime, "jit_post_compile_hook", lambda **kw: compiles.append(kw["fn"].name))

    graph_out = exec_b4.execute(cond, hist, 2.0, 0.25, 0.5, initial_noise=y0, solver_noise=sde_rnd)
    exec_b8.execute(
        torch.randn(8, 1, 32, device="cuda", dtype=torch.bfloat16),
        torch.randn(8, history_len, 16, device="cuda", dtype=torch.bfloat16),
        2.0,
        0.25,
        0.5,
        initial_noise=torch.randn(8, patch_size, 16, device="cuda", dtype=noise_dtype),
        solver_noise=torch.randn(4, 8, patch_size, 16, device="cuda", dtype=noise_dtype),
    )

    # Batch size and step index are runtime args: buckets 4 and 8 must reuse the bucket-1 compiles.
    assert compiles == []
    assert graph_out[0].shape == (3, patch_size, 16)
    torch.testing.assert_close(graph_out[0], got.to(torch.bfloat16), atol=1e-2, rtol=1e-2)
