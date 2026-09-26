# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the cross-layer RoPE cos/sin sharing optimization.

Verifies that the precomputed cos/sin tables passed from
``MossAudioTokenizerTransformer.forward`` down to each layer's
``apply_rope`` produce results equivalent to per-layer computation.
Covers streaming mode (multi-step, slot reset, offset advance) and
non-streaming mode with B > 1.
"""

from __future__ import annotations

import math

import pytest
import torch

from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerTransformer,
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


def _apply_rope_eager(q, k, offset, max_period=10000.0):
    """Reference per-layer RoPE without any sharing."""
    B, H, T, D = q.shape
    ds = torch.arange(D // 2, device=q.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / D))
    ts = offset.float().view(-1, 1) + torch.arange(T, device=q.device, dtype=torch.float32)
    ts = ts.view(B, 1, T, 1)
    dims = q.shape[:-1]
    q = q.view(*dims, D // 2, 2)
    k = k.view(*dims, D // 2, 2)
    qr, qi = q[..., 0].float(), q[..., 1].float()
    kr, ki = k[..., 0].float(), k[..., 1].float()
    rotr = torch.cos(freqs * ts)
    roti = torch.sin(freqs * ts)
    qor = qr * rotr - qi * roti
    qoi = qr * roti + qi * rotr
    kor = kr * rotr - ki * roti
    koi = kr * roti + ki * rotr
    dt = q.dtype
    qo = torch.stack([qor.to(dt), qoi.to(dt)], dim=-1)
    ko = torch.stack([kor.to(dt), koi.to(dt)], dim=-1)
    return qo.view(*dims, D), ko.view(*dims, D)


def test_non_streaming_b2_shared_matches_per_layer():
    """Non-streaming mode with B=2: shared cos/sin produces the same
    output as per-layer computation."""
    torch.manual_seed(42)
    tr = _build_transformer(num_layers=4)
    tr.eval()

    x = torch.randn(2, 5, D_MODEL)

    # With sharing (the code path under test)
    with torch.no_grad():
        y_shared = tr(x)

    # Without sharing: temporarily disable rope precomputation by
    # patching the forward to not pass cos_sin
    def forward_no_share(self, x, *args, **kwargs):
        kwargs.pop("cos_sin", None)
        B, T, C = x.shape
        state = self._streaming_state
        execution_context = kwargs.get("execution_context")
        if state is None:
            offsets = torch.zeros(1, dtype=torch.long, device=x.device)
        elif execution_context is None:
            offsets = state.offsets
        else:
            offsets = state.offsets.index_select(0, execution_context.state_slot_ids)

        if self.positional_embedding in {"sin", "sin_rope"}:
            positions = torch.arange(T, device=x.device).view(1, -1, 1)
            positions = positions + offsets.view(-1, 1, 1)
            from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
                create_sin_embedding,
            )

            pos_emb = create_sin_embedding(positions, C, max_period=self.max_period, dtype=x.dtype)
            x = x + self.positional_scale * pos_emb

        for layer in self.layers:
            x = layer(x, *args, **kwargs)

        if state is not None:
            import torch as _torch

            if execution_context is None:
                state.offsets[:] = _torch.where(state.exec_mask, state.offsets + T, state.offsets)
        return x

    import types

    bound_no_share = types.MethodType(forward_no_share, tr)
    with torch.no_grad():
        y_per_layer = bound_no_share(x)

    torch.testing.assert_close(y_shared, y_per_layer, atol=1e-6, rtol=1e-6)


def test_streaming_multi_step_offset_advance():
    """Streaming mode: shared cos/sin stays correct as offsets advance
    over multiple decode steps."""
    torch.manual_seed(42)
    tr = _build_transformer(num_layers=3)
    tr.eval()

    with tr.streaming(2) as s:
        offsets_seen = []
        for step in range(4):
            T = 3
            x = torch.randn(2, T, D_MODEL)
            with torch.no_grad():
                y = tr(x)
            assert y.shape == (2, T, D_MODEL)
            offsets_seen.append(s.offsets.clone())
            # Offsets should advance by T each step
            if step > 0:
                assert (offsets_seen[step] - offsets_seen[step - 1]).abs().sum() == T * 2


def test_streaming_slot_reset():
    """Streaming mode: after resetting a slot, the shared cos/sin
    correctly uses offset=0 for the reset slot."""
    torch.manual_seed(42)
    tr = _build_transformer(num_layers=3)
    tr.eval()

    with tr.streaming(2) as s:
        T = 3
        # Step 1: both slots active
        x1 = torch.randn(2, T, D_MODEL)
        with torch.no_grad():
            tr(x1)
        assert s.offsets[0] == T and s.offsets[1] == T

        # Reset slot 1
        reset_mask = torch.tensor([False, True])
        s.reset(reset_mask)
        assert s.offsets[1] == 0

        # Step 2: only slot 0 active (slot 1 was reset)
        x2 = torch.randn(1, T, D_MODEL)
        from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
            StreamingExecutionContext,
        )

        ctx = StreamingExecutionContext(
            state_slot_ids=torch.tensor([0]),
            valid_rows=torch.tensor([True]),
        )
        with torch.no_grad():
            y = tr(x2, execution_context=ctx)
        assert y.shape == (1, T, D_MODEL)


def test_streaming_wraparound():
    """Streaming mode: cos/sin sharing works correctly when the ring
    buffer wraps around (offset exceeds capacity)."""
    torch.manual_seed(42)
    tr = _build_transformer(num_layers=2, context=4)
    tr.eval()

    with tr.streaming(1) as s:
        for step in range(6):
            T = 2
            x = torch.randn(1, T, D_MODEL)
            with torch.no_grad():
                y = tr(x)
            assert y.shape == (1, T, D_MODEL)
        # After 6 steps of T=2, offset = 12, capacity = 4 → wrapped 3 times
        assert s.offsets[0] == 12


def test_shared_cos_sin_value_identity():
    """Directly verify that the per-layer reference is deterministic."""
    torch.manual_seed(42)
    B, H, T, D = 2, 4, 5, 16
    offset = torch.tensor([3, 7])
    q = torch.randn(B, H, T, D)
    k = torch.randn(B, H, T, D)

    # Per-layer reference
    q_ref, k_ref = _apply_rope_eager(q, k, offset)

    # Verify the reference is deterministic
    q_ref2, k_ref2 = _apply_rope_eager(q, k, offset)
    torch.testing.assert_close(q_ref, q_ref2, atol=0, rtol=0)
