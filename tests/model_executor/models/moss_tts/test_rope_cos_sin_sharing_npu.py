# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""NPU parity tests for the cross-layer RoPE cos/sin sharing optimization.

Production hoists a single ``rope_angles`` table per transformer block (for
both CUDA and NPU, gated on ``_shared_rope`` and an explicit execution
context) and consumes it in the fused ``torch_npu.npu_rotary_mul`` path. These
tests run on NPU and compare the shared-table path against per-layer
computation with identical weights and inputs, and assert the tables are built
once per transformer forward and broadcast to every layer.

The head dimension is 64 (``d_model=256``, ``num_heads=4``) because the fused
rotary kernel rejects small/misaligned head dimensions.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

import vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 as at
from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
    MossAudioTokenizerTransformer,
    StreamingExecutionContext,
)

pytestmark = [pytest.mark.core_model, pytest.mark.npu]

DEVICE = "npu"
DTYPE = torch.bfloat16
D_MODEL = 256  # head dim = 64, required by the fused rotary kernel
NUM_HEADS = 4
D_HEAD = D_MODEL // NUM_HEADS
ATOL = 2e-2
RTOL = 2e-2


def _build_transformer(num_layers: int = 3, context: int = 64) -> MossAudioTokenizerTransformer:
    tr = MossAudioTokenizerTransformer(
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=num_layers,
        dim_feedforward=256,
        causal=True,
        context=context,
        positional_embedding="rope",
        max_period=10000.0,
        layer_scale=0.01,
        device=DEVICE,
        dtype=DTYPE,
    )
    tr._shared_rope = True
    return tr.to(device=DEVICE, dtype=DTYPE).eval()


def _ctx_all(batch_size: int) -> StreamingExecutionContext:
    """Full-batch execution context in natural slot order."""
    return StreamingExecutionContext(
        state_slot_ids=torch.arange(batch_size, device=DEVICE),
        valid_rows=torch.ones(batch_size, dtype=torch.bool, device=DEVICE),
    )


@contextlib.contextmanager
def _per_layer_rope(tr: MossAudioTokenizerTransformer):
    """Disable the transformer's table hoist so each layer computes its own.

    The layer submodules keep their own reference to the RoPE module, so
    clearing ``tr.rope`` only skips the shared-table precomputation.
    """
    rope = tr.rope
    tr.rope = None
    try:
        yield
    finally:
        tr.rope = rope


def _reset_streaming_slots(module: torch.nn.Module, slot_ids: list[int]) -> None:
    """Reset slots through every live ``StreamingModule`` state.

    Resets the transformer offsets as well as child attention offsets and KV
    caches.
    """
    slots = torch.as_tensor(slot_ids, dtype=torch.long)
    for submodule in module.modules():
        state = getattr(submodule, "_streaming_state", None)
        if state is not None:
            state.reset_slots(slots)


def _assert_close(a: torch.Tensor, b: torch.Tensor) -> None:
    torch.testing.assert_close(a, b, atol=ATOL, rtol=RTOL)


def test_b2_first_step_matches_per_layer():
    """B>1 single execution step: the hoisted table (all slots at offset 0)
    matches the per-layer computation in the fused NPU path."""
    torch.manual_seed(0)
    tr = _build_transformer()
    x = torch.randn(2, 5, D_MODEL, device=DEVICE, dtype=DTYPE)

    with tr.streaming(2):
        with torch.no_grad():
            y_shared = tr(x, execution_context=_ctx_all(2))
    with tr.streaming(2):
        with torch.no_grad(), _per_layer_rope(tr):
            y_per_layer = tr(x, execution_context=_ctx_all(2))

    _assert_close(y_shared, y_per_layer)


def test_tables_built_once_and_shared_across_layers(monkeypatch):
    """The shared tables are constructed once per forward and passed to every
    layer (same tuple/objects)."""
    torch.manual_seed(0)
    tr = _build_transformer(num_layers=4)
    x = torch.randn(2, 4, D_MODEL, device=DEVICE, dtype=DTYPE)

    seen: list[tuple[torch.Tensor, torch.Tensor] | None] = []
    original = at.apply_rope

    def spy(q, k, offset, max_period=10000.0, time_before_heads=False, freqs_cache=None, table=None):
        seen.append(table)
        return original(q, k, offset, max_period, time_before_heads, freqs_cache=freqs_cache, table=table)

    monkeypatch.setattr(at, "apply_rope", spy)

    with tr.streaming(2), torch.no_grad():
        tr(x, execution_context=_ctx_all(2))

    assert len(seen) == 4, "apply_rope must run once per layer"
    assert all(t is not None for t in seen), "the hoist must supply a table to every layer"
    assert len({id(t) for t in seen}) == 1, "every layer must receive the same table"
    assert len({id(v) for t in seen for v in t}) == 2, "exactly one cos and one sin tensor"


def test_streaming_multi_step_matches_per_layer():
    """Multi-step streaming with B=2: shared tables stay identical to per-layer
    computation as offsets advance."""
    torch.manual_seed(0)
    tr = _build_transformer(num_layers=3)
    steps = [torch.randn(2, 3, D_MODEL, device=DEVICE, dtype=DTYPE) for _ in range(4)]

    with tr.streaming(2):
        with torch.no_grad():
            ctx = _ctx_all(2)
            ys_shared = [tr(x, execution_context=ctx) for x in steps]

    with tr.streaming(2):
        with torch.no_grad(), _per_layer_rope(tr):
            ctx = _ctx_all(2)
            ys_per_layer = [tr(x, execution_context=ctx) for x in steps]

    for shared, per_layer in zip(ys_shared, ys_per_layer):
        _assert_close(shared, per_layer)


def test_streaming_reordered_slots_matches_per_layer():
    """Reordered/subset slot execution: shared tables use the same
    ``index_select`` offsets as attention."""
    torch.manual_seed(0)
    tr = _build_transformer(num_layers=3)
    batch = 3

    x_full = torch.randn(batch, 2, D_MODEL, device=DEVICE, dtype=DTYPE)
    x_c = torch.randn(1, 2, D_MODEL, device=DEVICE, dtype=DTYPE)
    x_ba = torch.randn(2, 2, D_MODEL, device=DEVICE, dtype=DTYPE)

    ctx_c = StreamingExecutionContext(
        state_slot_ids=torch.tensor([2], device=DEVICE),
        valid_rows=torch.tensor([True], device=DEVICE),
    )
    ctx_ba = StreamingExecutionContext(
        state_slot_ids=torch.tensor([1, 0], device=DEVICE),
        valid_rows=torch.tensor([True, True], device=DEVICE),
    )

    def run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with tr.streaming(batch):
            with torch.no_grad():
                y0 = tr(x_full, execution_context=_ctx_all(batch))
                y1 = tr(x_c, execution_context=ctx_c)
                y2 = tr(x_ba, execution_context=ctx_ba)
        return y0, y1, y2

    with _per_layer_rope(tr):
        per_layer = run()
    shared = run()

    for a, b in zip(shared, per_layer):
        _assert_close(a, b)


def test_streaming_reset_and_reuse_matches_per_layer():
    """Reset-and-reuse: after resetting slot 1 and re-running it, the shared
    path matches per-layer computation and the reset slot starts at 0."""
    torch.manual_seed(0)
    tr = _build_transformer(num_layers=3)
    x_both = torch.randn(2, 3, D_MODEL, device=DEVICE, dtype=DTYPE)
    x_slot1 = torch.randn(1, 3, D_MODEL, device=DEVICE, dtype=DTYPE)
    ctx1 = StreamingExecutionContext(
        state_slot_ids=torch.tensor([1], device=DEVICE),
        valid_rows=torch.tensor([True], device=DEVICE),
    )

    def run() -> tuple[int, torch.Tensor, int]:
        with tr.streaming(2):
            with torch.no_grad():
                tr(x_both, execution_context=_ctx_all(2))
                _reset_streaming_slots(tr, [1])
                offset_before = int(tr._streaming_state.offsets[1])
                y = tr(x_slot1, execution_context=ctx1)
                offset_after = int(tr._streaming_state.offsets[1])
        return offset_before, y, offset_after

    with _per_layer_rope(tr):
        before_pl, y_pl, after_pl = run()
    before_sh, y_sh, after_sh = run()

    assert before_sh == before_pl == 0
    assert after_sh == after_pl == 3
    _assert_close(y_sh, y_pl)


def test_streaming_wraparound_matches_per_layer():
    """Ring-buffer wraparound: shared tables remain correct after the offset
    exceeds the ring capacity."""
    torch.manual_seed(0)
    tr = _build_transformer(num_layers=2, context=4)
    steps = [torch.randn(1, 2, D_MODEL, device=DEVICE, dtype=DTYPE) for _ in range(6)]

    def run() -> tuple[list[torch.Tensor], int]:
        with tr.streaming(1):
            with torch.no_grad():
                ctx = _ctx_all(1)
                ys = [tr(x, execution_context=ctx) for x in steps]
                offset = int(tr._streaming_state.offsets[0])
        return ys, offset

    with _per_layer_rope(tr):
        ys_pl, off_pl = run()
    ys_sh, off_sh = run()

    assert off_sh == off_pl == 12
    for shared, per_layer in zip(ys_sh, ys_pl):
        _assert_close(shared, per_layer)
