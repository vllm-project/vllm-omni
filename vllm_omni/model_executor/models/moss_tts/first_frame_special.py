# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Experimental empty-history decoder; owns no persistent streaming state."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from vllm.triton_utils import tl, triton

from .audio_tokenizer_v2 import MossAudioTokenizerMultiheadAttention
from .slot_attention import _rotate


@triton.jit
def _empty_attention(
    projected_ptr,
    output_ptr,
    stride_b: tl.constexpr,
    stride_t: tl.constexpr,
    heads: tl.constexpr,
    frames: tl.constexpr,
    head_dim: tl.constexpr,
    rope_scale: tl.constexpr,
    block_m: tl.constexpr,
):
    rows = tl.program_id(0) * block_m + tl.arange(0, block_m)
    bh = tl.program_id(1)
    b, h = bh // heads, bh % heads
    dims = tl.arange(0, head_dim)
    cols = tl.arange(0, 64)
    qb = projected_ptr + b * stride_b + rows[:, None] * stride_t + h * head_dim
    q = tl.load(qb + dims[None, :], rows[:, None] < frames, 0)
    qp = tl.load(qb + (dims[None, :] ^ 1), rows[:, None] < frames, 0)
    q = _rotate(q, qp, dims[None, :], rows[:, None], rope_scale)
    kb = projected_ptr + b * stride_b + cols[None, :] * stride_t + heads * head_dim + h * head_dim
    k = tl.load(kb + dims[:, None], cols[None, :] < frames, 0)
    kp = tl.load(kb + (dims[:, None] ^ 1), cols[None, :] < frames, 0)
    k = _rotate(k, kp, dims[:, None], cols[None, :], rope_scale)
    score = tl.dot(q, k).to(tl.float32) * (head_dim**-0.5)
    allowed = (cols[None, :] <= rows[:, None]) & (cols[None, :] < frames)
    score = tl.where(allowed, score, -float("inf"))
    maximum = tl.max(score, 1)
    probability = tl.exp(score - maximum[:, None])
    denominator = tl.sum(probability, 1)
    vb = projected_ptr + b * stride_b + cols[:, None] * stride_t + 2 * heads * head_dim + h * head_dim
    v = tl.load(vb + dims[None, :], cols[:, None] < frames, 0)
    result = tl.dot(probability.to(v.dtype), v) / denominator[:, None]
    dst = output_ptr + ((b * frames + rows[:, None]) * heads + h) * head_dim + dims[None, :]
    tl.store(dst, result, rows[:, None] < frames)


@torch.library.custom_op("vllm_omni::moss_empty_first_attention", mutates_args=())
def empty_attention(projected: torch.Tensor, heads: int, max_period: float) -> torch.Tensor:
    b, t, width = projected.shape
    d = width // (3 * heads)
    assert 1 < t <= 32 and d == 64 and projected.dtype == torch.bfloat16
    output = torch.empty((b, t, heads * d), device=projected.device, dtype=projected.dtype)
    _empty_attention[(triton.cdiv(t, 16), b * heads)](
        projected,
        output,
        projected.stride(0),
        projected.stride(1),
        heads,
        t,
        d,
        -math.log(max_period) * 2 / d,
        16,
        num_warps=8,
        num_stages=2,
    )
    return output


@empty_attention.register_fake
def _(projected, heads, max_period):
    b, t, width = projected.shape
    return torch.empty((b, t, width // 3), device=projected.device, dtype=projected.dtype)


class EmptyHistoryAttention(nn.Module):
    def __init__(self, original):
        super().__init__()
        assert original.causal and original.weights_per_step == 0
        assert original.context is None or original.context >= 32
        assert original.rope is not None
        self.in_proj = original.in_projs[0]
        self.out_proj = original.out_projs[0]
        self.dim = original.embed_dim
        self.heads = original.num_heads
        self.max_period = original.rope.max_period

    def forward(self, query, key, value, execution_context=None):
        if query.shape[1] == 1:
            # The sole causal key has probability one. Its state is discarded.
            attended = F.linear(query, self.in_proj.weight[2 * self.dim :])
        else:
            attended = empty_attention(self.in_proj(query), self.heads, self.max_period)
        return self.out_proj(attended)


def specialize(codec):
    """Only call on a private, first-frame-only decoder before compiling it."""
    assert not codec._streaming_modules, "First-only codec must not own live stream states"
    count = 0
    for module in codec.decoder.modules():
        if hasattr(module, "self_attn") and isinstance(module.self_attn, MossAudioTokenizerMultiheadAttention):
            module.self_attn = EmptyHistoryAttention(module.self_attn)
            count += 1
    assert count == 92, count
    return codec


class StatelessFirstGraphs:
    """Fixed shapes and private graph memory; every returned PCM owns storage."""

    @torch.inference_mode()
    def __init__(self, codec):
        self.codec = codec
        self.entries = {}

        def decode(codes, lengths):
            return codec._decode_frame_tensors(codes, lengths)[0].float()

        self.compiled = torch.compile(
            decode,
            dynamic=False,
            fullgraph=True,
            options={"triton.cudagraphs": False, "epilogue_fusion": False, "emulate_precision_casts": True},
        )
        device = next(codec.parameters()).device
        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for batch in (8, 4, 2, 1):
                codes = torch.zeros(12, batch, 1, dtype=torch.long, device=device)
                lengths = torch.ones(batch, dtype=torch.long, device=device)
                for _ in range(3):
                    self.compiled(codes, lengths)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream, pool=torch.cuda.graph_pool_handle()):
                    pcm = self.compiled(codes, lengths)
                self.entries[batch] = graph, codes, pcm
        torch.cuda.current_stream(device).wait_stream(stream)
        stream.synchronize()

    @torch.inference_mode()
    def __call__(self, codes):
        parts = []
        for start in range(0, len(codes), 8):
            block = codes[start : start + 8]
            count = len(block)
            bucket = next(b for b in (1, 2, 4, 8) if b >= count)
            graph, inputs, output = self.entries[bucket]
            if count != bucket:
                inputs.zero_()
            inputs[:, :count, 0].copy_(block.transpose(0, 1))
            graph.replay()
            parts.append(output[:count].clone())
        return parts[0] if len(parts) == 1 else torch.cat(parts)
