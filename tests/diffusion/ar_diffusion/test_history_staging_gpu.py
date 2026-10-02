# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA regressions for contiguous AR-Diffusion history staging."""

import pytest
import torch

from tests.diffusion.ar_diffusion.test_paged_attention import (
    BLOCK,
    HEAD_DIM,
    N_HEADS,
    POS,
    _commit_video_span,
    _gpu_flash_attn_usable,
    make_state,
)
from tests.helpers.mark import hardware_test
from vllm_omni.experimental.ar_diffusion.kv_cache import paged_write_attn
from vllm_omni.experimental.ar_diffusion.kv_cache.config import KV_GATHER_ENV
from vllm_omni.platforms import current_omni_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.skipif(
    torch.version.hip is not None or not _gpu_flash_attn_usable(), reason="usable CUDA FlashAttention is required"
)
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.parametrize("window_chunks", [2, 4])
@torch.inference_mode()
def test_staged_attention_updates_only_the_active_window_gpu(monkeypatch, window_chunks):
    """Full and reused gathers match fresh attention without touching spare capacity or stale history."""
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    device, dtype = torch.device("cuda"), torch.bfloat16
    kv, st = make_state(device=device, dtype=dtype, window_chunks=window_chunks, reuse_history_staging=True)
    _commit_video_span(kv, st, kv_branch=POS, n_chunks=1, dtype=dtype, device=device)
    stage_key, stage_value = kv.history_staging[0]
    stage_key.fill_(-7)
    stage_value.fill_(-7)
    query = torch.randn(BLOCK, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
    for step in range(2):
        ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
        ctx.max_video_tokens = 2 * BLOCK
        ctx.prepare(device, action_len=0, query_len=BLOCK)
        inputs = ctx.layer_inputs(0)
        assert inputs.reuse_history is (step > 0)
        key = torch.randn_like(query)
        value = torch.randn_like(query)
        actual = paged_write_attn(inputs, query, key, value, None, None, HEAD_DIM**-0.5)
        expected = paged_write_attn(
            inputs._replace(stage_key=None, stage_value=None, reuse_history=False),
            query,
            key,
            value,
            None,
            None,
            HEAD_DIM**-0.5,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ids = inputs.block_table[0].long()
        for staged, cache in ((stage_key, kv.key_cache(0)), (stage_value, kv.value_cache(0))):
            torch.testing.assert_close(
                staged[: inputs.max_seq_len], cache.index_select(0, ids).flatten(0, 1), rtol=0, atol=0
            )
            assert (staged[inputs.max_seq_len :] == -7).all()


@pytest.mark.skipif(
    torch.version.hip is not None or not _gpu_flash_attn_usable(), reason="usable CUDA FlashAttention is required"
)
@hardware_test(res={"cuda": ["L4", "H100"]}, num_cards=1)
@pytest.mark.parametrize("window_chunks", [2, 4])
@pytest.mark.parametrize("reuse_history", [False, True])
@torch.inference_mode()
def test_staged_attention_replays_inductor_cudagraph(monkeypatch, window_chunks, reuse_history):
    """Inductor must capture the real mutable op and replay with fresh current K/V."""
    monkeypatch.setenv(KV_GATHER_ENV, "1")
    device, dtype = torch.device("cuda"), torch.bfloat16
    kv, st = make_state(device=device, dtype=dtype, window_chunks=window_chunks, reuse_history_staging=True)
    _commit_video_span(kv, st, kv_branch=POS, n_chunks=1, dtype=dtype, device=device)
    stage_key, stage_value = kv.history_staging[0]
    stage_key.fill_(-7)
    stage_value.fill_(-7)
    query = torch.randn(BLOCK, N_HEADS, HEAD_DIM, device=device, dtype=dtype)
    key, value = torch.randn_like(query), torch.randn_like(query)

    ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
    ctx.max_video_tokens = 2 * BLOCK
    ctx.prepare(device, action_len=0, query_len=BLOCK)
    inputs = ctx.layer_inputs(0)
    # Populate history before testing the reuse variant, as the first denoising probe does.
    paged_write_attn(inputs, query, key, value, None, None, HEAD_DIM**-0.5)
    if reuse_history:
        ctx = st.get_kv_caches(POS, seq_len=BLOCK, commit_current=False)[0].forward_ctx
        ctx.max_video_tokens = 2 * BLOCK
        ctx.prepare(device, action_len=0, query_len=BLOCK)
        inputs = ctx.layer_inputs(0)
    assert inputs.reuse_history is reuse_history

    torch._dynamo.reset()
    try:
        compiled = torch.compile(paged_write_attn, mode="reduce-overhead", fullgraph=True)
        for _ in range(3):
            torch.compiler.cudagraph_mark_step_begin()
            compiled(inputs, query, key, value, None, None, HEAD_DIM**-0.5)
        current_omni_platform.synchronize()

        previous = None
        for step in range(2):
            key.normal_()
            # Distinct value ranges ensure a stale replay cannot match the reference.
            value.fill_(2.0 if step == 0 else -2.0)
            torch.compiler.cudagraph_mark_step_begin()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            ) as prof:
                actual = compiled(inputs, query, key, value, None, None, HEAD_DIM**-0.5)
                current_omni_platform.synchronize()
            assert any("cudaGraphLaunch" in event.key for event in prof.key_averages()), (
                "Inductor did not replay a CUDA graph; mutable staging inputs may have lost their static annotation"
            )
            expected = paged_write_attn(
                inputs._replace(stage_key=None, stage_value=None, reuse_history=False),
                query,
                key,
                value,
                None,
                None,
                HEAD_DIM**-0.5,
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if previous is not None:
                assert not torch.equal(actual, previous)
            # Copy outside capture: graph-owned output storage is reused on the next step.
            previous = actual.clone()
            del actual
            ids = inputs.block_table[0].long()
            for staged, cache in ((stage_key, kv.key_cache(0)), (stage_value, kv.value_cache(0))):
                torch.testing.assert_close(
                    staged[: inputs.max_seq_len], cache.index_select(0, ids).flatten(0, 1), rtol=0, atol=0
                )
                assert (staged[inputs.max_seq_len :] == -7).all()
    finally:
        torch._dynamo.reset()
