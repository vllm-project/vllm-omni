# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Streaming-state tests for the cross-layer RoPE cos/sin sharing path.

These run on CPU and cover the streaming bookkeeping that feeds the shared
cos/sin tables (offset advance, slot reset, ring-buffer wraparound) plus the
guarantee that the eager path produces identical results when handed a
precomputed ``table`` instead of recomputing per layer. The shared-vs-
per-layer parity across an entire transformer block is covered on NPU in
``test_rope_cos_sin_sharing_npu.py``.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerTransformer,
    StreamingExecutionContext,
    apply_rope,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

D_MODEL = 64
NUM_HEADS = 4
D_HEAD = D_MODEL // NUM_HEADS


def _build_transformer(
    num_layers: int = 3,
    causal: bool = True,
    context: int | None = 32,
) -> MossAudioTokenizerTransformer:
    return MossAudioTokenizerTransformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=num_layers,
        dim_feedforward=128,
        causal=causal,
        context=context,
        positional_embedding="rope",
        max_period=10000.0,
        layer_scale=0.01,
        device="cpu",
        dtype=torch.float32,
    )


def _reset_streaming_slots(module: torch.nn.Module, slot_ids: list[int]) -> None:
    """Reset slots through the module's streaming states.

    Drives each ``StreamingModule``'s ``reset_slots`` so that child attention
    offsets and KV caches are reset alongside the parent transformer offsets.
    """
    slots = torch.as_tensor(slot_ids, dtype=torch.long)
    for submodule in module.modules():
        state = getattr(submodule, "_streaming_state", None)
        if state is not None:
            state.reset_slots(slots)


def test_eager_reuses_precomputed_table():
    """The eager path must produce identical results with a ``table`` that
    matches the values it would recompute internally.

    Production builds one table per transformer block (``rope_angles``) and
    passes it to every layer, so reusing it must be numerically identical to
    the per-layer recomputation.
    """
    torch.manual_seed(42)
    B, H, T, D = 2, NUM_HEADS, 5, D_HEAD
    max_period = 10_000.0
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)
    offset = torch.tensor([3, 7])

    ds = torch.arange(D // 2, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float().view(-1, 1) + torch.arange(T, dtype=torch.float32)
    angles = ts.view(B, 1, T, 1) * freqs.view(1, 1, 1, D // 2)
    table = (torch.cos(angles), torch.sin(angles))

    q_ref, k_ref = apply_rope(q, k, offset, max_period=max_period)
    q_t, k_t = apply_rope(q, k, offset, max_period=max_period, table=table)

    torch.testing.assert_close(q_ref, q_t, atol=0, rtol=0)
    torch.testing.assert_close(k_ref, k_t, atol=0, rtol=0)


def test_streaming_multi_step_offset_advance():
    """Streaming mode: the transformer offsets that build the shared cos/sin
    tables advance by T on every decode step."""
    torch.manual_seed(42)
    tr = _build_transformer(num_layers=3)
    tr.eval()

    batch_size = 2
    with tr.streaming(batch_size):
        state = tr._streaming_state
        assert state is not None, "streaming() must install a TransformerState"

        offsets_seen = []
        for step in range(4):
            T = 3
            x = torch.randn(batch_size, T, D_MODEL)
            with torch.no_grad():
                y = tr(x)
            assert y.shape == (batch_size, T, D_MODEL)
            offsets_seen.append(state.offsets.clone())
            if step > 0:
                advance = offsets_seen[step] - offsets_seen[step - 1]
                assert torch.equal(advance, torch.full_like(advance, T))


def test_streaming_slot_reset_reuses_reset_slot():
    """After a slot reset, that slot must attend from offset 0 again.

    Resets both the transformer offsets and the child attention offsets/KV
    cache, then executes the reset slot and compares it against a fresh
    single-slot reference.
    """
    torch.manual_seed(42)
    tr = _build_transformer(num_layers=3)
    tr.eval()

    reference = _build_transformer(num_layers=3)
    reference.load_state_dict(tr.state_dict())
    reference.eval()

    T = 3
    x_both = torch.randn(2, T, D_MODEL)
    x_slot1 = torch.randn(1, T, D_MODEL)

    # Reference: slot 0 at offset 0 for a single slot.
    with reference.streaming(1):
        with torch.no_grad():
            y_reference = reference(x_slot1)

    with tr.streaming(2):
        with torch.no_grad():
            tr(x_both)
        assert int(tr._streaming_state.offsets[1]) == T

        _reset_streaming_slots(tr, [1])
        assert int(tr._streaming_state.offsets[1]) == 0

        ctx = StreamingExecutionContext(
            state_slot_ids=torch.tensor([1]),
            valid_rows=torch.tensor([True]),
        )
        with torch.no_grad():
            y = tr(x_slot1, execution_context=ctx)

    assert y.shape == (1, T, D_MODEL)
    torch.testing.assert_close(y, y_reference, atol=1e-6, rtol=1e-5)


def test_streaming_wraparound():
    """Streaming mode: offsets keep advancing past the ring-buffer capacity."""
    torch.manual_seed(42)
    tr = _build_transformer(num_layers=2, context=4)
    tr.eval()

    with tr.streaming(1):
        state = tr._streaming_state
        for _ in range(6):
            T = 2
            x = torch.randn(1, T, D_MODEL)
            with torch.no_grad():
                y = tr(x)
            assert y.shape == (1, T, D_MODEL)
        # After 6 steps of T=2, offset = 12, capacity = 4 -> wrapped 3 times.
        assert int(state.offsets[0]) == 12
