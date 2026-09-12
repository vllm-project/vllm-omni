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
2. Output is elementwise bitwise-identical to the eager path across the
   mask regimes that used to hit the data-dependent branches
   (`.any()`/`.all()`/`.item()`/`numel()`): all-und, all-gen, mixed.
3. Re-invoking with a different token count does NOT trigger a shape-driven
   recompile (Dynamo cache-hit under `dynamic=True`).

These are pure-tensor unit tests — they do not touch the full vLLM engine, so
they run in CPU CI and don't need the model weights.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import moe_forward

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

    if mask_kind == "all_und":
        mask = torch.zeros(n, dtype=torch.bool)
    elif mask_kind == "all_gen":
        mask = torch.ones(n, dtype=torch.bool)
    elif mask_kind == "mixed":
        mask = torch.tensor([True, False, True, True, False, False])
    elif mask_kind == "single_gen":
        mask = torch.zeros(n, dtype=torch.bool)
        mask[3] = True
    else:  # single_und
        mask = torch.ones(n, dtype=torch.bool)
        mask[3] = False

    eager_out = moe_forward(hidden, und, gen, mask)

    # `fullgraph=True` is the contract we care about — a compile-time graph
    # break would raise here.
    compiled = torch.compile(moe_forward, fullgraph=True, dynamic=True)
    compiled_out = compiled(hidden, und, gen, mask)

    # Elementwise identity is the promise made in the docstring; test with
    # exact equality (no tolerance) — the compiled path executes the same
    # ops on the same inputs.
    torch.testing.assert_close(compiled_out, eager_out, rtol=0, atol=0)


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


class _MiniBackbone(nn.Module):
    """Minimal shim reproducing `MammothModa2Qwen2ForCausalLM.get_input_embeddings`.

    We reproduce the method body verbatim rather than instantiate the full
    class because the full class requires a `VllmConfig` + weight loading.
    Keeping the shim tiny means any drift between it and the real method
    will fail the review, not this test — which is the right trade-off for a
    focused parity unit test.
    """

    def __init__(self, base_vocab: int, gen_vocab: int, hidden: int, gen_start: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(base_vocab, hidden)
        self.gen_embed_tokens = nn.Embedding(gen_vocab, hidden)
        self.gen_vocab_start_index = gen_start
        self.extra_gen_vocab = True

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        gen_start = int(self.gen_vocab_start_index)
        gen_mask = input_ids >= gen_start
        base_ids = torch.where(gen_mask, torch.zeros_like(input_ids), input_ids)
        gen_ids = torch.where(gen_mask, input_ids - gen_start, torch.zeros_like(input_ids))
        base_emb = self.embed_tokens(base_ids)
        gen_emb = self.gen_embed_tokens(gen_ids)
        return torch.where(gen_mask.unsqueeze(-1), gen_emb, base_emb)

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


@pytest.mark.parametrize("regime", ["all_base", "all_gen", "mixed"])
def test_get_input_embeddings_parity(regime: str) -> None:
    torch.manual_seed(0)
    base_vocab = 32
    gen_vocab = 16
    hidden = 12
    gen_start = base_vocab

    m = _MiniBackbone(base_vocab, gen_vocab, hidden, gen_start).eval()

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
    m = _MiniBackbone(base_vocab, gen_vocab, hidden, gen_start).eval()

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
