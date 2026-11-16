# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Framework-owned chunked packing from a declarative ``_mixfusion_plan`` (CPU).

The fake DiT here deliberately uses non-Qwen argument and output names, so the
declarative path is shown to depend only on the plan:

- the chunked adapter builds the plan from the root declaration, reuses it
  while the layout is unchanged and validates the root's input requirements;
- the chunk-outputs boundary runs the model's own module once per request and
  chunk-stacks / per-request-stacks its outputs, matching individual calls;
- a packed forward reassembles one output per request equal to running each
  request alone;
- an explicit ``mixfusion_adapter`` still overrides the framework adapter.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    MixFusionStep,
    get_forward_context,
    override_forward_context,
)
from vllm_omni.diffusion.hooks.mixfusion import apply_mixfusion
from vllm_omni.diffusion.mixfusion.chunked import ChunkedJointAdapter, resolve_mixfusion_adapter
from vllm_omni.diffusion.mixfusion.plan import (
    ChunkedOutput,
    MixFusionChunkOutputs,
    MixFusionExpandRows,
    MixFusionRoot,
    PerRequestOutput,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

DIM = 4


class _Embed(nn.Module):
    """Per-request geometry-dependent embedding: position = f(grid shape)."""

    def forward(self, x, grids, text_width):
        grid = grids[0]
        positions = torch.arange(x.shape[1], dtype=x.dtype) * (grid[0] + 10 * grid[1])
        return x + 1, positions.unsqueeze(-1).expand(-1, 2), torch.full((text_width + 3, 2), float(grid[1]))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mod = nn.Linear(1, 1)

    def forward(self, x, cond, pos):
        shift = self.mod(cond)
        assert shift.shape[0] == x.shape[0], "modulation rows must match hidden rows"
        return x * 2 + shift.unsqueeze(1) + pos[..., :1]


class _FakeDiT(nn.Module):
    _mixfusion_plan = {
        "": MixFusionRoot(
            ragged_arg="x", text_arg="ctx", text_mask_arg="ctx_mask", output_attr="pred", required_inputs=("grids",)
        ),
        "embed": MixFusionChunkOutputs(
            per_request_args=("x", "grids"),
            outputs=(ChunkedOutput(seq_dim=1), ChunkedOutput(seq_dim=0), PerRequestOutput()),
        ),
        "blocks.*.mod": MixFusionExpandRows(),
    }

    def __init__(self) -> None:
        super().__init__()
        self.embed = _Embed()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.text_tables: list[torch.Tensor] = []

    def forward(self, x, ctx, ctx_mask=None, grids=None, cond=None):
        x, pos, text_table = self.embed(x, grids, ctx.shape[1])
        self.text_tables.append(text_table)
        for block in self.blocks:
            x = block(x, cond, pos)
        return SimpleNamespace(pred=x)


def _requests():
    torch.manual_seed(0)
    grids = [(1, 2), (2, 1), (2, 2)]
    latents = [torch.randn(1, 2, DIM), torch.randn(2, 2, DIM), torch.randn(1, 4, DIM)]
    # One context row per image.
    ctx = torch.zeros(4, 3, DIM)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=torch.bool)
    cond = torch.arange(4, dtype=torch.float32).reshape(4, 1)
    return grids, latents, ctx, mask, cond


def _packed(model, grids, latents, ctx, mask, cond):
    with override_forward_context(ForwardContext(mixfusion_step=MixFusionStep())):
        out = model(x=latents, ctx=ctx, ctx_mask=mask, grids=grids, cond=cond)
        return out, get_forward_context().mixfusion_step.plan


def test_resolve_prefers_an_explicit_adapter():
    explicit = object()
    model = _FakeDiT()
    assert isinstance(resolve_mixfusion_adapter(model, model._mixfusion_plan), ChunkedJointAdapter)
    model.mixfusion_adapter = explicit
    assert resolve_mixfusion_adapter(model, model._mixfusion_plan) is explicit
    assert resolve_mixfusion_adapter(nn.Module(), {"blocks.*.mod": MixFusionExpandRows()}) is None


def test_validation_uses_declared_input_requirements():
    adapter = ChunkedJointAdapter(_FakeDiT._mixfusion_plan[""])
    adapter.validate_inputs(_FakeDiT(), SimpleNamespace(grids=[(1, 1)]))
    with pytest.raises(ValueError, match="missing grids"):
        adapter.validate_inputs(_FakeDiT(), SimpleNamespace(grids=None))


def test_packed_forward_matches_each_request_alone():
    grids, latents, ctx, mask, cond = _requests()
    reference = _FakeDiT()
    expected = []
    row = 0
    for grid, latent in zip(grids, latents):
        count = latent.shape[0]
        expected.append(
            reference(x=latent, ctx=ctx[row : row + count], grids=[grid], cond=cond[row : row + count]).pred
        )
        row += count

    model = _FakeDiT()
    model.load_state_dict(reference.state_dict())
    apply_mixfusion(model, model._mixfusion_plan)
    out, plan = _packed(model, grids, latents, ctx, mask, cond)

    # gcd(2, 2, 4) = 2 -> 1 + 2 + 2 chunks; text lengths come from the mask.
    assert plan.chunk_size == 2 and plan.num_chunks == 5
    assert plan.image_row_counts == (1, 2, 1)
    assert plan.request_stream_lens == {"text": (2, 1, 1, 3), "image": (2, 2, 2, 4)}
    assert isinstance(out, list) and len(out) == len(latents)
    for got, exp in zip(out, expected):
        assert torch.equal(got, exp)
    # Per-request tables are trimmed to the text width, one per image row.
    table = model.text_tables[-1]
    assert table.shape == (4, 3, 2)
    assert table[:, 0, 0].tolist() == [2.0, 1.0, 1.0, 2.0]


def test_plan_is_reused_until_the_layout_changes():
    grids, latents, ctx, mask, cond = _requests()
    model = _FakeDiT()
    apply_mixfusion(model, model._mixfusion_plan)
    _, first = _packed(model, grids, latents, ctx, mask, cond)
    _, second = _packed(model, grids, [latent + 1 for latent in latents], ctx, mask, cond)
    assert second is first
    # A different text mask is a different layout.
    _, third = _packed(model, grids, latents, ctx, torch.ones_like(mask), cond)
    assert third is not first and third.request_stream_lens["text"] == (3, 3, 3, 3)


def test_mismatched_text_rows_are_rejected():
    grids, latents, ctx, mask, cond = _requests()
    model = _FakeDiT()
    apply_mixfusion(model, model._mixfusion_plan)
    with pytest.raises(ValueError, match="text rows"):
        _packed(model, grids, latents, ctx[:3], mask[:3], cond)


def test_serial_forward_is_untouched_without_an_armed_step():
    grids, latents, ctx, _, cond = _requests()
    model = _FakeDiT()
    expected = model(x=latents[0], ctx=ctx[:1], grids=[grids[0]], cond=cond[:1]).pred
    apply_mixfusion(model, model._mixfusion_plan)
    assert torch.equal(model(x=latents[0], ctx=ctx[:1], grids=[grids[0]], cond=cond[:1]).pred, expected)


def test_dynamic_batch_recomputes_gcd_and_restores_each_request():
    model = _FakeDiT()
    reference = _FakeDiT()
    reference.load_state_dict(model.state_dict())
    apply_mixfusion(model, model._mixfusion_plan)
    latents = [torch.randn(1, length, DIM) for length in (8, 16, 12)]
    grids = [(1, length) for length in (8, 16, 12)]
    plans = []
    # Join, leave, and reorder. The same model/root hook retains its cache.
    for indices, expected_chunk, expected_count in [
        ([0, 1], 8, 3),
        ([0, 1, 2], 4, 9),
        ([0, 1], 8, 3),
        ([1, 0], 8, 3),
    ]:
        ctx = torch.zeros(len(indices), 3, DIM)
        mask = torch.ones(len(indices), 3, dtype=torch.bool)
        cond = torch.tensor(indices, dtype=torch.float32).reshape(-1, 1)
        out, plan = _packed(model, [grids[i] for i in indices], [latents[i] for i in indices], ctx, mask, cond)
        assert (plan.chunk_size, plan.num_chunks) == (expected_chunk, expected_count)
        for row, index in enumerate(indices):
            expected = reference(
                x=latents[index],
                ctx=ctx[row : row + 1],
                grids=[grids[index]],
                cond=cond[row : row + 1],
            ).pred
            torch.testing.assert_close(out[row], expected)
        plans.append(plan)
    assert all(left is not right for left, right in zip(plans, plans[1:]))
