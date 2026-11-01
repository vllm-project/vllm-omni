# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packing tests for the MixFusion chunked attention path (CPU).

The FlashAttention kernel needs an accelerator, so these tests drive the real
packing logic — the standard ``QwenImageCrossAttention.forward`` reading the
plan from an armed ``MixFusionStep`` on the forward context, routed through
the attention layer's packed entry — with
stub projections and a capturing ``forward_packed`` stub: mixed-length
requests (different text lengths and image chunk counts) must be flattened
into one packed sequence with per-request ``cu_seqlens``, in each request's
``text + image`` token order, with RoPE cos/sin cast to the activation dtype
(the eager-path contract, #7494). The packed entry delegates to
``Attention.forward_packed`` once, so the backends that route to a varlen
kernel and those served by the layer's bucketed dense fallback are covered
through the same entry point.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    MixFusionStep,
    override_forward_context,
)
from vllm_omni.diffusion.mixfusion.packed_attention import (
    MIXFUSION_PLAN_KEY,
    run_mixfusion_packed_attention,
)
from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageCrossAttention,
    build_mixfusion_plan,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

HEADS, KV_HEADS, HEAD_DIM = 2, 1, 4
Q_SIZE = HEADS * HEAD_DIM
QKV_DIM = Q_SIZE + 2 * KV_HEADS * HEAD_DIM
ROT_DIM = HEAD_DIM // 2

# Two requests with different text lengths and chunk counts:
#   req0: 3 text tokens + 2 image chunks (8 tokens)  -> seq 11
#   req1: 5 text tokens + 3 image chunks (12 tokens) -> seq 17
SEQ_LEN_TXT = 7  # padded text length shared by both requests
REAL_TXT_LENS = [3, 5]
REQUEST_CHUNK_RANGES = [(0, 2), (2, 5)]
CHUNK_SIZE = 4
NUM_CHUNKS = 5
REQ_LENS = [11, 17]
CU_SEQLENS = [0, 11, 28]


class _CapturingRope:
    """Stand-in for RotaryEmbedding that records the cos/sin dtype."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, tensor, cos, sin):
        self.calls.append(
            {
                "tensor_dtype": tensor.dtype,
                "cos_dtype": cos.dtype,
                "sin_dtype": sin.dtype,
            }
        )
        return tensor


def _segment_mean_flat(query, key, value, cu_seqlens):
    """Per-document semantics: each document's rows gain its own K/V means.

    A cross-document leak changes the means, so routing this through both the
    varlen route and the bucketed dense fallback makes the two observable
    parity checks of the packed layout, not just of shape plumbing.
    """
    out = torch.empty_like(query)
    lengths = cu_seqlens.diff().tolist()
    offset = 0
    for length in lengths:
        if length:
            out[offset : offset + length] = (
                query[offset : offset + length]
                + key[offset : offset + length].mean(dim=0, keepdim=True)
                + value[offset : offset + length].mean(dim=0, keepdim=True)
            )
        offset += length
    return out


class _PackedAttentionStub:
    """Attention stand-in exposing the ``forward_packed`` entry point.

    ``varlen`` applies the per-document semantics directly (the single-kernel
    route); ``bucketed`` delegates to the production dense fallback so the
    model's packing works unchanged on backends without packed varlen support.
    """

    def __init__(self, mode: str):
        assert mode in ("varlen", "bucketed")
        self.mode = mode
        self.calls: list[dict] = []
        self.dense_shapes: list[tuple] = []

    def __call__(self, query, key, value, attn_metadata):
        # Mirror the metadata-keyed dispatch at the top of ``Attention.forward``
        # (the real layer routes the same way; this stub replaces the layer so
        # the test stays on CPU).
        if attn_metadata.extra.get(MIXFUSION_PLAN_KEY) is not None:
            return run_mixfusion_packed_attention(self, query, key, value, attn_metadata)
        raise AssertionError("stub reached without a mixfusion plan")

    def forward_packed(self, query, key, value, *, cu_seqlens, max_seqlen):
        self.calls.append(
            {
                "query": query,
                "key": key,
                "value": value,
                "cu_seqlens": cu_seqlens,
                "max_seqlen": max_seqlen,
            }
        )
        if self.mode == "varlen":
            return _segment_mean_flat(query, key, value, cu_seqlens)

        def dense_forward(dense_query, dense_key, dense_value, attn_metadata):
            self.dense_shapes.append(tuple(dense_query.shape))
            return dense_query + dense_key.mean(dim=1, keepdim=True) + dense_value.mean(dim=1, keepdim=True)

        fake_layer = SimpleNamespace(forward=dense_forward)
        return Attention._forward_packed_dense_buckets(fake_layer, query, key, value, cu_seqlens)


def _mixfusion_module(*, attn: _PackedAttentionStub, rope: _CapturingRope):
    """A QwenImageCrossAttention with only the packed-path dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed:
    identity projections keep q/k/v rows equal to the input rows, which makes
    the packed token order directly assertable. Only non-Module attributes
    are set, so ``object.__setattr__`` bypasses the nn.Module machinery.
    """

    module = QwenImageCrossAttention.__new__(QwenImageCrossAttention)
    attrs = {
        "parallel_config": None,
        "query_num_heads": HEADS,
        "kv_num_heads": KV_HEADS,
        "add_query_num_heads": HEADS,
        "add_kv_num_heads": KV_HEADS,
        "head_dim": HEAD_DIM,
        "to_qkv": lambda x: (x, None),
        "add_kv_proj": lambda x: (x, None),
        "to_out": lambda x: x,
        "to_add_out": lambda x: x,
        "norm_q": lambda t: t,
        "norm_k": lambda t: t,
        "norm_added_q": lambda t: t,
        "norm_added_k": lambda t: t,
        "qk_norm": True,
        "eps": 1e-6,
        "rope": rope,
        "attn": attn,
    }
    for name, value in attrs.items():
        object.__setattr__(module, name, value)
    return module


def _mixfusion_inputs():
    gen = torch.Generator().manual_seed(2026)
    image_chunks = torch.randn(NUM_CHUNKS, CHUNK_SIZE, QKV_DIM, generator=gen).to(torch.bfloat16)
    encoder_hidden_states = torch.randn(2, SEQ_LEN_TXT, QKV_DIM, generator=gen).to(torch.bfloat16)
    image_freq_chunks = torch.randn(NUM_CHUNKS, CHUNK_SIZE, ROT_DIM, generator=gen, dtype=torch.complex64)
    text_freqs = torch.randn(2, SEQ_LEN_TXT, ROT_DIM, generator=gen, dtype=torch.complex64)
    chunk_to_request = torch.tensor([0, 0, 1, 1, 1])
    return image_chunks, encoder_hidden_states, image_freq_chunks, text_freqs, chunk_to_request


def _forward(module):
    image_chunks, encoder_hidden_states, image_freq_chunks, text_freqs, chunk_to_request = _mixfusion_inputs()
    plan = build_mixfusion_plan(
        seq_len_txt=SEQ_LEN_TXT,
        chunk_size=CHUNK_SIZE,
        request_chunk_ranges=REQUEST_CHUNK_RANGES,
        real_txt_lens=REAL_TXT_LENS,
        chunk_to_request=chunk_to_request,
        device=image_chunks.device,
    )
    # The attention reads the plan from the armed step, mirroring the pipeline.
    with override_forward_context(ForwardContext(mixfusion_step=MixFusionStep(plan=plan))):
        return module.forward(
            hidden_states=image_chunks,
            encoder_hidden_states=encoder_hidden_states,
            vid_freqs=image_freq_chunks,
            txt_freqs=text_freqs,
        )


def _assert_rope_matches_eager_dtype_contract(rope: _CapturingRope):
    # The eager path (_qwen_image_qk_norm_rope) casts cos/sin to the activation
    # dtype; FP32 rotary numerics drop Omni-vs-Diffusers PSNR (#7494), so the
    # mixfusion path must not diverge from the default forward.
    assert rope.calls, "RoPE was never applied"
    for call in rope.calls:
        assert call["tensor_dtype"] == torch.bfloat16
        assert call["cos_dtype"] == torch.bfloat16
        assert call["sin_dtype"] == torch.bfloat16


def _assert_output_shapes(img_out, txt_out):
    assert img_out.shape == (NUM_CHUNKS, CHUNK_SIZE, Q_SIZE)
    assert txt_out.shape == (len(REQ_LENS), SEQ_LEN_TXT, Q_SIZE)
    # Text rows beyond a request's real length are zero-padded, matching the
    # mask-unpad path used by the non-mixfusion forward.
    assert torch.all(txt_out[0, REAL_TXT_LENS[0] :] == 0)
    assert torch.all(txt_out[1, REAL_TXT_LENS[1] :] == 0)


def test_packed_attention_packs_mixed_requests_into_one_packed_call():
    attn = _PackedAttentionStub(mode="varlen")
    rope = _CapturingRope()
    module = _mixfusion_module(attn=attn, rope=rope)

    img_out, txt_out = _forward(module)

    # One forward_packed call with per-request boundaries.
    assert len(attn.calls) == 1
    call = attn.calls[0]
    query, key, value = call["query"], call["key"], call["value"]
    total_tokens = sum(REQ_LENS)
    assert query.shape == (total_tokens, HEADS, HEAD_DIM)
    assert key.shape == (total_tokens, KV_HEADS, HEAD_DIM)
    assert value.shape == (total_tokens, KV_HEADS, HEAD_DIM)
    assert torch.equal(call["cu_seqlens"], torch.tensor(CU_SEQLENS, dtype=torch.int32))
    assert call["max_seqlen"] == max(REQ_LENS)

    # Each request's tokens are contiguous in "text, then image chunks" order.
    # Identity projections make the packed rows equal to the input rows.
    image_chunks, encoder_hidden_states, *_ = _mixfusion_inputs()

    def _txt_rows(req_idx):
        return encoder_hidden_states[req_idx, : REAL_TXT_LENS[req_idx], :Q_SIZE].unflatten(-1, (HEADS, HEAD_DIM))

    def _img_rows(chunk_start, chunk_end):
        return image_chunks[chunk_start:chunk_end].reshape(-1, QKV_DIM)[:, :Q_SIZE].unflatten(-1, (HEADS, HEAD_DIM))

    offset = 0
    for req_idx, (chunk_start, chunk_end) in enumerate(REQUEST_CHUNK_RANGES):
        txt_len = REAL_TXT_LENS[req_idx]
        img_len = (chunk_end - chunk_start) * CHUNK_SIZE
        assert torch.equal(query[offset : offset + txt_len], _txt_rows(req_idx))
        assert torch.equal(query[offset + txt_len : offset + txt_len + img_len], _img_rows(chunk_start, chunk_end))
        offset += txt_len + img_len
    assert offset == total_tokens

    _assert_rope_matches_eager_dtype_contract(rope)
    _assert_output_shapes(img_out, txt_out)
    # The recovered rows come from the packed result: the varlen stub writes
    # q + per-document K/V means, so check against the same reference.
    flat = _segment_mean_flat(query, call["key"], call["value"], call["cu_seqlens"])
    # Text rows are recovered in request order: req0's text opens the packed
    # layout, req1's text opens at its own request offset.
    req0_txt = flat[: REAL_TXT_LENS[0]]
    req1_txt = flat[REQ_LENS[0] : REQ_LENS[0] + REAL_TXT_LENS[1]]
    assert torch.equal(txt_out[0, : REAL_TXT_LENS[0]], req0_txt.flatten(1, 2))
    assert torch.equal(txt_out[1, : REAL_TXT_LENS[1]], req1_txt.flatten(1, 2))
    # req0's image chunks sit right after its text rows in the packed layout.
    assert torch.equal(img_out[0], flat[REAL_TXT_LENS[0] : REAL_TXT_LENS[0] + CHUNK_SIZE].flatten(1, 2))


def test_forward_mixfusion_bucketed_fallback_matches_the_varlen_result():
    # The model no longer picks a route itself: the same forward_packed call
    # must serve backends without packed varlen support through the layer's
    # bucketed dense fallback, and the recovered outputs must match what a
    # single varlen kernel over the same packing would produce.
    varlen_attn = _PackedAttentionStub(mode="varlen")
    varlen_module = _mixfusion_module(attn=varlen_attn, rope=_CapturingRope())
    varlen_img_out, varlen_txt_out = _forward(varlen_module)

    bucketed_attn = _PackedAttentionStub(mode="bucketed")
    rope = _CapturingRope()
    bucketed_module = _mixfusion_module(attn=bucketed_attn, rope=rope)
    img_out, txt_out = _forward(bucketed_module)

    # Same single packed call; equal-length documents would share a bucket,
    # but 11 and 17 differ so each dense forward holds one request.
    assert len(bucketed_attn.calls) == 1
    assert sorted(bucketed_attn.dense_shapes) == [
        (1, REQ_LENS[0], HEADS, HEAD_DIM),
        (1, REQ_LENS[1], HEADS, HEAD_DIM),
    ]
    assert torch.equal(img_out, varlen_img_out)
    assert torch.equal(txt_out, varlen_txt_out)

    _assert_rope_matches_eager_dtype_contract(rope)
    _assert_output_shapes(img_out, txt_out)
