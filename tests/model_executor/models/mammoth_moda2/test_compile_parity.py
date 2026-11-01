# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""torch.compile parity for MammothModa2 AR helpers.

These tests target the two functions inside `mammoth_moda2.py` that the AR
`@support_torch_compile` decoration relies on being graph-break free:

* `moe_forward` — MoE routing between understanding and generation experts.
* `MammothModa2Qwen2ForCausalLM.get_input_embeddings` — mixed base+gen vocab
  embedding lookup.

We compile each in isolation with `fullgraph=True, dynamic=True` and verify:

1. The compiled call succeeds (no graph break under fullgraph=True).
2. Output is elementwise bitwise-identical to the *pre-refactor* branchy
   implementations (kept here as local oracles) across the mask regimes that
   used to hit the data-dependent branches (`.any()`/`.all()`/`.item()`/
   `numel()`): all-und, all-gen, mixed.
3. Re-invoking with a different token count does NOT trigger a shape-driven
   recompile (Dynamo cache-hit under `dynamic=True`).

The embedding tests bind the **production** method onto a minimal shim
(just the attributes `get_input_embeddings` reads) instead of copying its
body, so a regression in the real class fails here; `MammothModa2Qwen3ForCausalLM`
inherits the same method via MRO and gets a dispatch smoke test too.

These are pure-tensor unit tests — they do not touch the full vLLM engine, so
they run in CPU CI and don't need the model weights.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import (
    MammothModa2Qwen2ForCausalLM,
    MammothModa2Qwen3ForCausalLM,
    moe_forward,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# ---------------------------------------------------------------------------
# moe_forward
# ---------------------------------------------------------------------------


class _LinearExpert(nn.Module):
    """Stand-in for the real MLP experts: deterministic + differentiable."""

    def __init__(self, d: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.proj = nn.Linear(d, d, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(d, d, generator=gen))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _legacy_moe_reference(
    hidden_states: torch.Tensor,
    und_expert: nn.Module,
    gen_expert: nn.Module,
    gen_token_mask: torch.Tensor,
) -> torch.Tensor:
    """The pre-refactor branchy routing (as of bad50980), kept as the oracle.

    This is the `.any()`/`.all()`/`int(sum().item())`/`argsort` implementation
    that the shape-static `nonzero` + `index_copy` rewrite replaced.
    """
    if gen_expert is None:
        return und_expert(hidden_states)
    if not gen_token_mask.any():
        return und_expert(hidden_states)
    if gen_token_mask.all():
        return gen_expert(hidden_states)

    d_model = hidden_states.shape[-1]
    flat_hid = hidden_states.reshape(-1, d_model)
    total_tokens = flat_hid.shape[0]
    if gen_token_mask.numel() != total_tokens:
        raise ValueError(
            "gen_token_mask shape mismatch: "
            f"mask={tuple(gen_token_mask.shape)}, hidden_states={tuple(hidden_states.shape)}"
        )
    flat_mask = gen_token_mask.reshape(-1)
    gen_pos = torch.where(flat_mask)[0]
    und_pos = torch.where(~flat_mask)[0]
    permute_order = torch.cat([gen_pos, und_pos], dim=0)
    inverse_order = torch.argsort(permute_order)
    gen_token_num = int(flat_mask.sum().item())
    gen_hid, und_hid = flat_hid[permute_order].split([gen_token_num, total_tokens - gen_token_num], dim=0)

    merged = torch.cat([gen_expert(gen_hid), und_expert(und_hid)], dim=0)
    merged = merged[inverse_order]
    return merged.view(*hidden_states.shape[:-1], -1)


def _make_mask(mask_kind: str, n: int) -> torch.Tensor:
    if mask_kind == "all_und":
        return torch.zeros(n, dtype=torch.bool)
    if mask_kind == "all_gen":
        return torch.ones(n, dtype=torch.bool)
    if mask_kind == "mixed":
        return torch.tensor([True, False, True, True, False, False])
    if mask_kind == "single_gen":
        mask = torch.zeros(n, dtype=torch.bool)
        mask[3] = True
        return mask
    assert mask_kind == "single_und"
    mask = torch.ones(n, dtype=torch.bool)
    mask[3] = False
    return mask


@pytest.mark.parametrize(
    "mask_kind",
    ["all_und", "all_gen", "mixed", "single_gen", "single_und"],
    ids=lambda k: f"mask={k}",
)
def test_moe_forward_compile_matches_eager(mask_kind: str) -> None:
    torch.manual_seed(0)
    d = 8
    n = 6

    und = _LinearExpert(d, seed=1)
    gen = _LinearExpert(d, seed=2)
    hidden = torch.randn(n, d)
    mask = _make_mask(mask_kind, n)

    eager_out = moe_forward(hidden, und, gen, mask)

    # Parity against the pre-refactor branchy routing — bitwise.
    legacy_out = _legacy_moe_reference(hidden, und, gen, mask)
    torch.testing.assert_close(eager_out, legacy_out, rtol=0, atol=0)

    # `fullgraph=True` is the contract we care about — a compile-time graph
    # break would raise here.
    compiled = torch.compile(moe_forward, fullgraph=True, dynamic=True)
    compiled_out = compiled(hidden, und, gen, mask)

    # Elementwise identity is the promise made in the docstring; test with
    # exact equality (no tolerance) — the compiled path executes the same
    # ops on the same inputs.
    torch.testing.assert_close(compiled_out, eager_out, rtol=0, atol=0)
    torch.testing.assert_close(compiled_out, legacy_out, rtol=0, atol=0)


def test_moe_forward_gen_expert_none_stays_und_only() -> None:
    """Static per-layer branch: `gen_expert is None` must short-circuit."""
    d = 4
    n = 3
    und = _LinearExpert(d, seed=42)
    hidden = torch.randn(n, d)

    eager = moe_forward(hidden, und, None, None)
    compiled = torch.compile(moe_forward, fullgraph=True, dynamic=True)(hidden, und, None, None)
    torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
    torch.testing.assert_close(compiled, und(hidden), rtol=0, atol=0)


def test_moe_forward_mask_shape_mismatch_raises() -> None:
    """An under-sized mask must raise, not return uninitialized rows."""
    d = 8
    und = _LinearExpert(d, seed=7)
    gen = _LinearExpert(d, seed=8)
    hidden = torch.randn(6, d)
    short_mask = torch.zeros(4, dtype=torch.bool)

    with pytest.raises(RuntimeError):
        moe_forward(hidden, und, gen, short_mask)

    compiled = torch.compile(moe_forward, fullgraph=True, dynamic=True)
    with pytest.raises(RuntimeError):
        compiled(hidden, und, gen, short_mask)


def test_moe_forward_no_recompile_across_shapes() -> None:
    """`dynamic=True` should reuse one graph across different token counts."""
    d = 8
    und = _LinearExpert(d, seed=3)
    gen = _LinearExpert(d, seed=4)

    from torch._dynamo.utils import counters

    counters.clear()
    compiled = torch.compile(moe_forward, fullgraph=True, dynamic=True)

    for n in (4, 6, 10, 12):
        hidden = torch.randn(n, d)
        mask = torch.arange(n) % 2 == 0
        _ = compiled(hidden, und, gen, mask)

    # `frames` counts unique frame compilations. dynamic=True means one
    # symbolic frame covers all sizes.
    frame_compiles = sum(counters["frames"].values()) if counters.get("frames") else 0
    # We accept 1 (ideal) or 2 (one specialization + one dynamic) — anything
    # more means we regressed to shape-driven recompiles.
    assert frame_compiles <= 2, f"unexpected recompiles: {frame_compiles} (Dynamo counters={dict(counters)})"


# ---------------------------------------------------------------------------
# get_input_embeddings — mixed base+gen vocab
# ---------------------------------------------------------------------------


class _EmbedShim:
    """Carries just the attributes `get_input_embeddings` reads; the
    production method is bound as-is so a regression in the real class fails
    these tests."""

    extra_gen_vocab = True
    get_input_embeddings = MammothModa2Qwen2ForCausalLM.get_input_embeddings

    def __init__(self, base_vocab: int, gen_vocab: int, hidden: int, gen_start: int) -> None:
        torch.manual_seed(0)
        self.embed_tokens = nn.Embedding(base_vocab, hidden)
        self.gen_embed_tokens = nn.Embedding(gen_vocab, hidden)
        self.gen_vocab_start_index = gen_start

    def eager_reference(self, input_ids: torch.Tensor) -> torch.Tensor:
        """The old branchy implementation, kept locally to check parity."""
        gen_start = int(self.gen_vocab_start_index)
        gen_mask = input_ids >= gen_start
        if not gen_mask.any():
            return self.embed_tokens(input_ids)
        if gen_mask.all():
            return self.gen_embed_tokens(input_ids - gen_start)
        flat_ids = input_ids.reshape(-1)
        flat_mask = gen_mask.reshape(-1)
        out = torch.empty(
            (flat_ids.shape[0], self.embed_tokens.weight.shape[-1]),
            dtype=self.embed_tokens.weight.dtype,
            device=flat_ids.device,
        )
        base_pos = torch.where(~flat_mask)[0]
        gen_pos = torch.where(flat_mask)[0]
        if base_pos.numel() > 0:
            out[base_pos] = self.embed_tokens(flat_ids[base_pos])
        if gen_pos.numel() > 0:
            out[gen_pos] = self.gen_embed_tokens(flat_ids[gen_pos] - gen_start)
        return out.view(*input_ids.shape, -1).contiguous()


class _NoGenVocabEmbedShim(_EmbedShim):
    """Short-circuit regime: text-only model without the gen vocab table."""

    extra_gen_vocab = False

    def __init__(self, base_vocab: int, hidden: int) -> None:
        torch.manual_seed(0)
        self.embed_tokens = nn.Embedding(base_vocab, hidden)
        self.gen_embed_tokens = None
        self.gen_vocab_start_index = None


def test_get_input_embeddings_handles_async_sentinel() -> None:
    """`async_scheduling=True` may feed `-1` for not-yet-written-back slots.

    The clamp must keep both gathers in-bounds on the dual-vocab path and on
    the short-circuit path, and stay compile-friendly.
    """
    base_vocab = 32
    gen_vocab = 16
    hidden = 12
    gen_start = base_vocab
    m = _EmbedShim(base_vocab, gen_vocab, hidden, gen_start)

    # bs=8, three `-1` sentinels interleaved with base + gen ids.
    ids = torch.tensor([1, -1, gen_start + 2, -1, 5, gen_start + 9, -1, 7])

    out = m.get_input_embeddings(ids)
    assert out.shape == (8, hidden)
    # No NaN/Inf — a stale-memory read on CUDA would frequently surface here.
    assert torch.isfinite(out).all()

    # Compile path must also succeed under fullgraph=True and match eager.
    compiled = torch.compile(m.get_input_embeddings, fullgraph=True, dynamic=True)
    torch.testing.assert_close(compiled(ids), out, rtol=0, atol=0)

    # Short-circuit branch (no gen vocab table) must also survive -1. A
    # text-only backbone never sees gen-range ids, so its input mixes base
    # ids with the async sentinels only.
    text_only = _NoGenVocabEmbedShim(base_vocab, hidden)
    text_ids = torch.tensor([1, -1, 5, -1, 0, 7, -1, 3])
    text_out = text_only.get_input_embeddings(text_ids)
    assert text_out.shape == (8, hidden)
    assert torch.isfinite(text_out).all()
    text_compiled = torch.compile(text_only.get_input_embeddings, fullgraph=True, dynamic=True)
    torch.testing.assert_close(text_compiled(text_ids), text_out, rtol=0, atol=0)


def test_qwen3_backbone_inherits_embedding_dispatch() -> None:
    """Qwen3 dispatches to the same compiled embedding method via MRO.

    `MammothModa2Qwen3ForCausalLM` must not shadow the Qwen2 embedding (the
    `@support_torch_compile` decoration patches `__bases__` in place, so the
    subclass inherits it); calling through the Qwen3 class must compile
    fullgraph and match the Qwen2 result bitwise.
    """
    assert MammothModa2Qwen3ForCausalLM.get_input_embeddings is MammothModa2Qwen2ForCausalLM.get_input_embeddings

    base_vocab = 32
    gen_vocab = 16
    hidden = 12
    gen_start = base_vocab
    m = _EmbedShim(base_vocab, gen_vocab, hidden, gen_start)
    ids = torch.tensor([1, gen_start + 2, 4, gen_start + 9, 0, gen_start + 15])

    q2_out = MammothModa2Qwen2ForCausalLM.get_input_embeddings(m, ids)
    q3_out = MammothModa2Qwen3ForCausalLM.get_input_embeddings(m, ids)
    torch.testing.assert_close(q3_out, q2_out, rtol=0, atol=0)

    q3_compiled = torch.compile(MammothModa2Qwen3ForCausalLM.get_input_embeddings, fullgraph=True, dynamic=True)
    torch.testing.assert_close(q3_compiled(m, ids), q3_out, rtol=0, atol=0)


@pytest.mark.parametrize("regime", ["all_base", "all_gen", "mixed"])
def test_get_input_embeddings_parity(regime: str) -> None:
    base_vocab = 32
    gen_vocab = 16
    hidden = 12
    gen_start = base_vocab

    m = _EmbedShim(base_vocab, gen_vocab, hidden, gen_start)

    if regime == "all_base":
        ids = torch.tensor([0, 5, 12, 7])
    elif regime == "all_gen":
        ids = torch.tensor([gen_start, gen_start + 3, gen_start + 15, gen_start + 8])
    else:
        ids = torch.tensor([1, gen_start + 2, 4, gen_start + 9, 0, gen_start + 15])

    ref = m.eager_reference(ids)
    new = m.get_input_embeddings(ids)
    # The masked-out positions differ in `base_emb` / `gen_emb` before the
    # `torch.where` selects — but the selection is bitwise identical to the
    # reference on the surviving positions.
    torch.testing.assert_close(new, ref, rtol=0, atol=0)

    compiled = torch.compile(m.get_input_embeddings, fullgraph=True, dynamic=True)
    out = compiled(ids)
    torch.testing.assert_close(out, new, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# CompileCounter guards — hard-fail on graph break / shape-driven recompile
# ---------------------------------------------------------------------------
#
# Same pattern as `tests/diffusion/ar_diffusion/test_paged_attention.py:402`:
# wrap the function with `fullgraph=True` (any graph break → raise) and pin
# `counter.frame_count == 1` (any shape-driven recompile → assertion error).
# `torch._dynamo.reset()` in try/finally keeps state clean for other suites.


def test_moe_forward_single_frame_across_shapes() -> None:
    d = 8
    und = _LinearExpert(d, seed=5)
    gen = _LinearExpert(d, seed=6)

    torch._dynamo.reset()
    try:
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()
        compiled = torch.compile(moe_forward, backend=counter, fullgraph=True, dynamic=True)

        for n in (4, 6, 10, 12, 40):
            hidden = torch.randn(n, d)
            mask = torch.arange(n) % 2 == 0
            compiled(hidden, und, gen, mask)

        assert counter.frame_count == 1, f"moe_forward recompiled across shapes: frame_count={counter.frame_count}"
    finally:
        torch._dynamo.reset()


def test_get_input_embeddings_single_frame_across_shapes() -> None:
    base_vocab = 32
    gen_vocab = 16
    hidden = 12
    gen_start = base_vocab
    m = _EmbedShim(base_vocab, gen_vocab, hidden, gen_start)

    torch._dynamo.reset()
    try:
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()
        compiled = torch.compile(m.get_input_embeddings, backend=counter, fullgraph=True, dynamic=True)

        # Mixed regime at multiple token counts — this is the code path that
        # went through `.any()`/`.all()` in the old implementation.
        for n in (4, 8, 16, 32):
            torch.manual_seed(n)
            ids = torch.randint(0, gen_start + gen_vocab, (n,))
            compiled(ids)

        assert counter.frame_count == 1, (
            f"get_input_embeddings recompiled across shapes: frame_count={counter.frame_count}"
        )
    finally:
        torch._dynamo.reset()
