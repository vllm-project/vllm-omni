# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real frame-sized Transfer pages against contiguous FlashAttention."""

import pytest
import torch

from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig, paged_write_attn
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import _resolve_fa_version
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.gpu]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    ("chunk_size", "execution"),
    [(chunk, "eager") for chunk in (1, 2, 3, 4, 8)] + [(chunk, mode) for chunk in (1, 4) for mode in ("compile", "cg")],
)
def test_transfer_frame_pages_match_contiguous_attention(chunk_size, execution, monkeypatch):
    from vllm.vllm_flash_attn import flash_attn_varlen_func

    device = torch.device("cuda")
    dtype = torch.bfloat16
    tokens, heads, dim = 390, 8, 128  # 832x480 Transfer, vision-only pages
    window, sinks = 3, 1
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(
            enable=True,
            chunk_size=tokens,
            window_chunks=3 if chunk_size == 1 else 96,
            sink_chunks=2 if chunk_size == 1 else 0,
        ),
        num_layers=1,
        num_kv_heads=heads,
        head_size=dim,
        dtype=dtype,
        block_size=tokens,
        max_model_len=1 << 20,
        available_bytes=1 << 28,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        frames_per_block=chunk_size,
        max_scratch_frames_per_branch=chunk_size,
        max_scratch_tokens_per_branch=517,
        cross_attention_lengths={"text": 517},
        eviction_group_frames=2 if chunk_size == 1 else 1,
        device=device,
    )
    state = ARDiffusionKVState(cache, "test", {"main": cache.begin_request("test")}, num_layers=1)
    generator = torch.Generator(device=device).manual_seed(42)

    def rand(count, nheads=heads):
        return torch.randn(count, nheads, dim, generator=generator, device=device, dtype=dtype)

    text_k, text_v = rand(517), rand(517)
    forward = paged_write_attn
    replays = 0
    if execution != "eager":
        torch._dynamo.reset()
        forward = torch.compile(
            paged_write_attn,
            fullgraph=True,
            dynamic=True,
            mode="reduce-overhead" if execution == "cg" else "default",
        )
        original_replay = torch.cuda.CUDAGraph.replay

        def record_replay(graph):
            nonlocal replays
            replays += 1
            return original_replay(graph)

        monkeypatch.setattr(torch.cuda.CUDAGraph, "replay", record_replay)
    history = []
    cursor = 0
    try:
        spans = [1] * 10 if chunk_size == 1 else [1, chunk_size, chunk_size, chunk_size - 1]
        for span in spans:
            if chunk_size == 1:
                history = [entry for entry in history if entry[0] < sinks or entry[0] >= cursor - (window - sinks - 1)]
            phases = [("control", cursor, span)] + [("denoise", cursor, span)] * 4
            phases += [("clean", frame, 1) for frame in range(cursor, cursor + span)]
            for phase, first, count in phases:
                commit = phase in ("control", "clean")
                length = count * tokens
                layer = state.get_kv_caches(
                    "main",
                    seq_len=length,
                    commit_current=commit,
                    extra_visible_tokens=length,
                )[0]
                layer.forward_ctx.prepare(device=device, action_len=517, query_len=length)
                q, k, v = rand(length, 32), rand(length), rand(length)
                if execution != "eager":
                    torch.compiler.cudagraph_mark_step_begin()
                actual = forward(
                    layer.to_layer_inputs(),
                    q,
                    k,
                    v,
                    text_k,
                    text_v,
                    dim**-0.5,
                ).clone()
                dense_k = torch.cat([text_k, *(item[1] for item in history), k])
                dense_v = torch.cat([text_v, *(item[2] for item in history), v])
                expected = flash_attn_varlen_func(
                    q=q,
                    k=dense_k,
                    v=dense_v,
                    cu_seqlens_q=torch.tensor([0, length], dtype=torch.int32, device=device),
                    cu_seqlens_k=torch.tensor([0, len(dense_k)], dtype=torch.int32, device=device),
                    max_seqlen_q=length,
                    max_seqlen_k=len(dense_k),
                    softmax_scale=dim**-0.5,
                    causal=False,
                    fa_version=_resolve_fa_version(dim),
                )
                torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-3)
                state.commit_paged_context("main")
                if commit:
                    history.extend(
                        (first + i, ki, vi) for i, (ki, vi) in enumerate(zip(k.split(tokens), v.split(tokens)))
                    )
            cursor += span
            if cursor == 1:
                # Same-shaped replacement must update the pages read by an
                # already compiled/captured executable, not its old text.
                text_k, text_v = rand(517), rand(517)
        assert (state.adapter("main").compacted_tokens > 0) == (chunk_size == 1)
        if execution == "cg":
            assert replays > 0, "The graph test must exercise replay, not silently run eagerly"
    finally:
        state.close()
        if execution != "eager":
            torch._dynamo.reset()
