# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests that DreamZero's cross-attention KV cache is actually init'd and reused.

DreamZero's I2V cross-attention caches the text k/v and image k_img/v_img in a
per-layer ``crossattn_cache`` dict: they are computed on the first forward of a
session (``is_init`` False -> True) and reused on every later step, because the
text/image context is session-invariant. Only the query (which depends on the
per-step hidden state) is recomputed.

These tests lock in two things:

1. ``WanI2VCrossAttention.forward`` initialises the cache on the first call and
   reuses it (not recompute) on later calls -- and recomputes every call when no
   cache is provided.
2. ``CausalWanModel._forward_blocks`` actually threads ``crossattn_cache`` down
   to each per-layer block. This is a regression guard: the cache was previously
   dead code because the dict never reached the block from ``_forward_inference``.

CPU-only: ``WanI2VCrossAttention.__init__`` needs a tensor-parallel group (it
builds ColumnParallel/RowParallel linears), so we bypass ``__init__`` and stub
the projections/attention to count calls and exercise only the caching logic.
"""

from unittest.mock import MagicMock

import pytest
import torch

from vllm_omni.diffusion.models.dreamzero.causal_wan_model import (
    CausalWanModel,
    WanI2VCrossAttention,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_CLIP_LEN = 257  # image tokens the I2V cross-attn splits off the front of context


def _make_cross_attn(n_heads: int = 2, head_dim: int = 4) -> WanI2VCrossAttention:
    """A WanI2VCrossAttention whose submodules are call-counting stubs.

    ``__init__`` is bypassed (it needs a TP group); we only wire what
    ``forward`` touches. Each projection returns a [B, S, n*head_dim] tensor so
    the subsequent ``unflatten(2, (n, head_dim))`` works, and is a MagicMock so
    we can assert how many times it was invoked.
    """
    inner = n_heads * head_dim
    module = WanI2VCrossAttention.__new__(WanI2VCrossAttention)
    module.tp_num_heads = n_heads
    module.head_dim = head_dim

    def _proj() -> MagicMock:
        return MagicMock(side_effect=lambda t: torch.zeros(t.shape[0], t.shape[1], inner))

    module.q = _proj()
    module.k = _proj()
    module.v = _proj()
    module.k_img = _proj()
    module.v_img = _proj()
    module.norm_q = lambda t: t
    module.norm_k = lambda t: t
    module.norm_k_img = lambda t: t
    # attn returns the query unchanged ([B, S, n, d]); flatten(2) downstream is happy.
    module.attn = lambda q, k, v: q
    module.o = lambda t: t
    return module


def _fresh_cache() -> dict:
    return {"is_init": False, "k": None, "v": None, "k_img": None, "v_img": None}


def _inputs(inner: int = 8):
    x = torch.randn(1, 3, inner)  # B=1, query_len=3
    context = torch.randn(1, _CLIP_LEN + 5, inner)  # 257 image tokens + 5 text tokens
    return x, context


def test_cache_inits_on_first_call_and_reuses_on_second() -> None:
    module = _make_cross_attn()
    cache = _fresh_cache()
    x, context = _inputs()

    # First call: cache is initialised and every k/v projection runs once.
    module.forward(x, context, crossattn_cache=cache)
    assert cache["is_init"] is True
    for key in ("k", "v", "k_img", "v_img"):
        assert cache[key] is not None, f"{key} should be cached after first call"
    assert module.k.call_count == 1
    assert module.v.call_count == 1
    assert module.k_img.call_count == 1
    assert module.v_img.call_count == 1

    cached = {key: cache[key] for key in ("k", "v", "k_img", "v_img")}

    # Second call: k/v/k_img/v_img are read from cache, NOT recomputed.
    module.forward(x, context, crossattn_cache=cache)
    assert module.k.call_count == 1
    assert module.v.call_count == 1
    assert module.k_img.call_count == 1
    assert module.v_img.call_count == 1
    for key, tensor in cached.items():
        assert cache[key] is tensor, f"{key} should be the same cached object"

    # The query always depends on the per-step hidden state, so it is recomputed.
    assert module.q.call_count == 2


def test_no_cache_recomputes_every_call() -> None:
    module = _make_cross_attn()
    x, context = _inputs()

    module.forward(x, context, crossattn_cache=None)
    module.forward(x, context, crossattn_cache=None)

    # Without a cache, the image/text k/v are recomputed on every call.
    assert module.k_img.call_count == 2
    assert module.v_img.call_count == 2
    assert module.k.call_count == 2
    assert module.v.call_count == 2


class _RecordingBlock(torch.nn.Module):
    """A stand-in transformer block that records the crossattn_cache it receives."""

    def __init__(self) -> None:
        super().__init__()
        self.received: list = []

    def forward(self, *, x, crossattn_cache, **kwargs):
        self.received.append(crossattn_cache)
        return x, None


def _make_model_with_recording_block(block: _RecordingBlock, dim: int = 16, freq_dim: int = 8) -> CausalWanModel:
    """A CausalWanModel stubbed down to just the path _forward_blocks needs.

    __init__ is bypassed. With action=None and clip_feature=None, _forward_blocks
    skips the action/state encoders and img_emb, so we only need the time/text
    embedding stubs and a single recording block.
    """
    model = CausalWanModel.__new__(CausalWanModel)
    model.dim = dim
    model.freq_dim = freq_dim
    model.time_embedding = lambda t: torch.zeros(t.shape[0], dim)
    model.time_projection = lambda t: torch.zeros(t.shape[0], t.shape[1], 6 * dim)
    model.text_embedding = lambda t: t
    model.freqs_action = torch.zeros(1)
    model.freqs_state = torch.zeros(1)
    model.blocks = torch.nn.ModuleList([block])
    # _forward_blocks applies the output head to the video tokens before returning.
    model.head = lambda x_video, e: x_video
    return model


def test_forward_blocks_threads_crossattn_cache_to_each_block() -> None:
    """Regression guard: crossattn_cache must reach the per-layer block.

    Previously _forward_blocks did not accept/forward crossattn_cache, so every
    block (and thus cross-attn) saw None and the cache was never used.
    """
    dim = 16
    block = _RecordingBlock()
    model = _make_model_with_recording_block(block, dim=dim)

    # x is [B, dim, T, H, W]; _forward_blocks flattens spatial -> [B, seq_len, dim].
    x = torch.randn(1, dim, 1, 2, 2)
    seq_len = 1 * 2 * 2
    timestep = torch.zeros(1, 1, dtype=torch.long)  # [B, F]
    context = torch.randn(1, 4, dim)
    sentinel_cache = [_fresh_cache()]

    model._forward_blocks(
        x=x,
        seq_len=seq_len,
        freqs=torch.zeros(1),
        timestep=timestep,
        context=context,
        clip_feature=None,
        embodiment_id=None,
        action=None,
        timestep_action=None,
        state=None,
        kv_cache=None,
        crossattn_cache=sentinel_cache,
        current_start_frame=0,
    )

    assert len(block.received) == 1
    # The block must receive this layer's cache dict, not None.
    assert block.received[0] is sentinel_cache[0]
