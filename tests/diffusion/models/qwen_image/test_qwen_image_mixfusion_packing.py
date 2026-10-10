# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packing tests for ``QwenImageCrossAttention.forward_mixfusion`` (CPU).

The FlashAttention kernel needs an accelerator, so these tests drive the real
``forward_mixfusion`` packing logic with stub projections and a capturing
attention stub: mixed-length requests (different text lengths and image chunk
counts) must be flattened into one packed sequence with per-request
``cu_seqlens``, in each request's ``text + image`` token order, with RoPE
cos/sin cast to the activation dtype (the eager-path contract, #7494). Both
the flat varlen branch and the per-request dense fallback are covered.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenImageCrossAttention,
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


def _mixfusion_module(*, flat_varlen: bool, rope: _CapturingRope):
    """A QwenImageCrossAttention with only forward_mixfusion's dependencies.

    Built via ``__new__`` so the vLLM parallel layers are never constructed:
    identity projections keep q/k/v rows equal to the input rows, which makes
    the packed token order directly assertable. Only non-Module attributes
    are set, so ``object.__setattr__`` bypasses the nn.Module machinery.
    """

    attn_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]] = []

    def fake_attn(query, key, value, attn_metadata):
        attn_calls.append((query, key, value, attn_metadata))
        return query

    module = QwenImageCrossAttention.__new__(QwenImageCrossAttention)
    attrs = {
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
        "rope": rope,
        "attn": fake_attn,
        "_supports_flat_varlen_attention": lambda: flat_varlen,
    }
    for name, value in attrs.items():
        object.__setattr__(module, name, value)
    return module, attn_calls


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
    return module.forward_mixfusion(
        image_chunks=image_chunks,
        encoder_hidden_states=encoder_hidden_states,
        image_freq_chunks=image_freq_chunks,
        text_freqs=text_freqs,
        chunk_to_request=chunk_to_request,
        request_chunk_ranges=REQUEST_CHUNK_RANGES,
        real_txt_lens=REAL_TXT_LENS,
    )


def _assert_rope_matches_eager_dtype_contract(rope: _CapturingRope):
    # The eager path (_qwen_image_qk_norm_rope) casts cos/sin to the activation
    # dtype; FP32 rotary numerics drop Omni-vs-Diffusers PSNR (#7494), so the
    # mixfusion branches must not diverge from the default forward.
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


def test_forward_mixfusion_packs_mixed_requests_into_one_flat_sequence():
    rope = _CapturingRope()
    module, attn_calls = _mixfusion_module(flat_varlen=True, rope=rope)

    img_out, txt_out = _forward(module)

    # One flat varlen call with per-request boundaries.
    assert len(attn_calls) == 1
    query, key, value, metadata = attn_calls[0]
    total_tokens = sum(REQ_LENS)
    assert query.shape == (total_tokens, HEADS, HEAD_DIM)
    assert key.shape == (total_tokens, KV_HEADS, HEAD_DIM)
    assert value.shape == (total_tokens, KV_HEADS, HEAD_DIM)
    assert metadata.is_varlen
    assert torch.equal(metadata.q_cu_seqlens, torch.tensor(CU_SEQLENS, dtype=torch.int32))
    assert torch.equal(metadata.kv_cu_seqlens, torch.tensor(CU_SEQLENS, dtype=torch.int32))
    assert metadata.max_q_len == max(REQ_LENS)
    assert metadata.max_kv_len == max(REQ_LENS)
    assert metadata.padded_tokens == 0

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
    # Identity out-projections make the outputs slices of the flat result.
    assert torch.equal(txt_out[0, : REAL_TXT_LENS[0]], query[: REAL_TXT_LENS[0]].flatten(1, 2))
    assert torch.equal(img_out[0], query[REAL_TXT_LENS[0] : REAL_TXT_LENS[0] + CHUNK_SIZE].flatten(1, 2))


def test_forward_mixfusion_dense_fallback_runs_each_request_independently():
    rope = _CapturingRope()
    module, attn_calls = _mixfusion_module(flat_varlen=False, rope=rope)

    img_out, txt_out = _forward(module)

    # Backends without flat varlen support still get one dense call per
    # request, with that request's own text + image sequence.
    assert len(attn_calls) == len(REQ_LENS)
    for req_idx, (query, key, value, metadata) in enumerate(attn_calls):
        assert query.shape == (1, REQ_LENS[req_idx], HEADS, HEAD_DIM)
        assert key.shape == (1, REQ_LENS[req_idx], KV_HEADS, HEAD_DIM)
        assert value.shape == (1, REQ_LENS[req_idx], KV_HEADS, HEAD_DIM)
        assert not metadata.is_varlen

    _assert_rope_matches_eager_dtype_contract(rope)
    _assert_output_shapes(img_out, txt_out)
