# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real-manager visibility, default-off, and graph regressions for superpages."""

import json

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig, paged_write_attn
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import _resolve_fa_version
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState
from vllm_omni.experimental.ar_diffusion.runner import paging_block_size

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def same_bits(left, right):
    return torch.equal(left.contiguous().view(torch.uint8), right.contiguous().view(torch.uint8))


def make_cache(enabled, *, frame=1560, layers=2, heads=10, dim=128, sink=9, **overrides):
    page = 128 if _resolve_fa_version(dim) == 4 else 160
    config = ARDiffusionKVConfig(
        enable=True,
        chunk_size=frame,
        window_chunks=9,
        sink_chunks=sink,
        gpu_memory_fraction=0.8,
        superpage_tokens=page if enabled else 0,
    )
    unit = paging_block_size(frame)
    cache = ARDiffusionKVCache(
        config,
        num_layers=layers,
        num_kv_heads=heads,
        head_size=dim,
        dtype=torch.bfloat16,
        block_size=page if enabled else unit,
        visibility_block_size=unit,
        max_model_len=1000000,
        available_bytes=1 << 30,
        kv_branches=(ARDiffusionKVBranchSpec("positive", 0),),
        session_capacity=1,
        frames_per_block=3,
        device=torch.device("cuda"),
        **overrides,
    )
    adapter = cache.begin_request("test")
    return cache, ARDiffusionKVState(cache, "test", {"positive": adapter}, num_layers=layers)


def visible_rows(cache, ctx, layer):
    slots = (
        ctx.block_table[0].long()[:, None] * cache.block_size + torch.arange(cache.block_size, device=cache.device)
    ).flatten()[: ctx.kv_len]
    return cache._k_pools[layer][slots], cache._v_pools[layer][slots]


def token_ids(key):
    return key[:, 0, 0].long() + key[:, 0, 1].long() * 128 + key[:, 0, 2].long() * 128**2


def tag_keys(key, start):
    ids = torch.arange(start, start + key.shape[0], device=key.device)
    for j in range(3):
        key[:, 0, j] = (ids // 128**j) % 128


@hardware_test(res={"cuda": ["H100", "H200", "B200"]}, num_cards=1)
@pytest.mark.parametrize("frame,sink,heads", [(1560, 9, 10), (168, 9, 2), (168, 0, 2)])
@torch.inference_mode()
def test_superpage_visible_multiset_and_dense_order(frame, sink, heads):
    from vllm.vllm_flash_attn import flash_attn_varlen_func as fa

    torch.manual_seed(42)
    baseline, bs = make_cache(False, frame=frame, sink=sink, heads=heads)
    cache, ss = make_cache(True, frame=frame, sink=sink, heads=heads)
    length = 3 * frame
    query = torch.randn(length, heads, 128, device="cuda", dtype=cache.dtype)
    checked = []
    for chunk in range(24):
        previous_probe = None
        for commit in (False, False, True):
            contexts = [
                state.get_kv_caches("positive", seq_len=length, commit_current=commit)[0].forward_ctx
                for state in (bs, ss)
            ]
            for ctx in contexts:
                ctx.prepare(cache.device, 0, length)
            if not commit:
                if previous_probe is not None:
                    assert contexts[1].private_slots is previous_probe.private_slots
                    assert contexts[1].block_table is previous_probe.block_table
                previous_probe = contexts[1]
            for layer in range(cache.num_layers):
                key, value = torch.randn_like(query), torch.randn_like(query)
                tag_keys(key, chunk * length)
                for kv, ctx in zip((baseline, cache), contexts):
                    actual = paged_write_attn(ctx.layer_inputs(layer), query, key, value, None, None, 128**-0.5)
                expected_k, expected_v = visible_rows(baseline, contexts[0], layer)
                actual_k, actual_v = visible_rows(cache, contexts[1], layer)
                expected_order, actual_order = token_ids(expected_k).argsort(), token_ids(actual_k).argsort()
                assert same_bits(actual_k[actual_order], expected_k[expected_order]), (chunk, commit, "keys")
                assert same_bits(actual_v[actual_order], expected_v[expected_order]), (chunk, commit, "values")
                dense = fa(
                    query,
                    actual_k,
                    actual_v,
                    max_seqlen_q=length,
                    cu_seqlens_q=contexts[1].query_start_loc,
                    max_seqlen_k=contexts[1].max_seq_len,
                    cu_seqlens_k=torch.tensor([0, contexts[1].kv_len], device="cuda", dtype=torch.int32),
                    causal=False,
                    fa_version=_resolve_fa_version(128),
                )
                assert same_bits(actual, dense), (chunk, commit, layer, "same-order dense")
            checked.append(
                dict(
                    chunk=chunk,
                    commit=commit,
                    length=contexts[1].kv_len,
                    fragments=contexts[1].superpage_fragment_tokens,
                    baseline_compacted=contexts[0].adapter.compacted_tokens,
                    v2_compacted=contexts[1].adapter.compacted_tokens,
                    absolute_position=contexts[1].adapter.absolute_num_computed_tokens,
                )
            )
            assert contexts[0].adapter.absolute_num_computed_tokens == contexts[1].adapter.absolute_num_computed_tokens
            if commit:
                bs.commit_paged_context("positive")
                ss.commit_paged_context("positive")
    print("SUPERPAGE_VISIBLE_CASES", json.dumps(dict(frame=frame, sink=sink, rows=checked)))
    bs.close()
    ss.close()


@hardware_test(res={"cuda": ["H100", "H200", "B200"]}, num_cards=1)
@torch.inference_mode()
def test_superpage_compiled_graph_changes_layers_and_current_kv():
    cache, state = make_cache(True)
    length = cache.frames_per_block * cache.spec.chunk_size
    ctx = state.get_kv_caches("positive", seq_len=length, commit_current=False)[0].forward_ctx
    ctx.prepare(cache.device, 0, length)
    query = torch.randn(length, cache.num_kv_heads, cache.head_size, dtype=cache.dtype, device=cache.device)
    key, value = torch.randn_like(query), torch.randn_like(query)
    torch._dynamo.reset()
    compiled = torch.compile(paged_write_attn, mode="reduce-overhead", fullgraph=True)
    try:
        for _ in range(3):
            for layer in range(cache.num_layers):
                torch.compiler.cudagraph_mark_step_begin()
                compiled(ctx.layer_inputs(layer), query, key, value, None, None, cache.head_size**-0.5)
        for step in range(3):
            for layer in reversed(range(cache.num_layers)):
                key.normal_()
                value.fill_(step + layer + 1)
                inputs = ctx.layer_inputs(layer)
                torch.compiler.cudagraph_mark_step_begin()
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
                ) as prof:
                    actual = compiled(inputs, query, key, value, None, None, cache.head_size**-0.5)
                assert any("cudaGraphLaunch" in e.key for e in prof.key_averages())
                expected = paged_write_attn(inputs, query, key, value, None, None, cache.head_size**-0.5)
                assert same_bits(actual, expected)
                del actual
    finally:
        torch._dynamo.reset()
        state.close()


@hardware_test(res={"cuda": ["H100", "H200", "B200"]}, num_cards=1)
@torch.inference_mode()
def test_superpage_action_and_capacity_errors(monkeypatch):
    with pytest.raises(ValueError, match="action/state"):
        make_cache(True, max_scratch_tokens_per_branch=1)
    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", "1")
    with pytest.raises(ValueError, match="incompatible"):
        make_cache(True)
    monkeypatch.delenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER")
    cache, state = make_cache(True)
    ctx = state.get_kv_caches("positive", seq_len=4680, commit_current=False)[0].forward_ctx
    with pytest.raises(ValueError, match="action/state"):
        ctx.prepare(cache.device, 1, 4680)
    cache.superpage_private_pages = 0
    with pytest.raises(RuntimeError, match="boundary bound"):
        ctx.prepare(cache.device, 0, 4680)
    state.close()


@pytest.mark.cpu
@pytest.mark.parametrize(
    "page,version,expected", [(160, 3, 160), (128, 4, 128), (161, 3, 16), (160, 2, 16), (0, 3, 16)]
)
def test_runner_superpage_selection(monkeypatch, page, version, expected):
    from dataclasses import replace

    from tests.diffusion.ar_diffusion.test_capability_runner import CapablePipeline, lingbot_like_spec, make_runner
    from vllm_omni.experimental.ar_diffusion.kv_cache import paged_attention

    monkeypatch.setattr(paged_attention, "_resolve_fa_version", lambda _: version)
    pipeline = CapablePipeline(replace(lingbot_like_spec(), tokens_per_frame=1560))
    runner = make_runner(pipeline, preallocate=False)
    runner.ar_diffusion_kv_config.superpage_tokens = page
    runner._preallocate_kv_cache(available_bytes=1 << 30)
    assert runner.kv_cache.block_size == expected
    assert runner.kv_cache.visibility_block_size == 16
    assert runner.kv_cache.superpage_size == (page if expected == page else 0)
