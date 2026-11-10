# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MixFusion module-boundary hooks (CPU).

The hooks implement the packed chunk-stacked layout purely at declared module
boundaries (see ``mixfusion/plan.py``), mirroring the SP hook mechanism. The
model-specific half lives behind a ``MixFusionAdapter``; these tests use a
minimal stand-in transformer plus a fake adapter that reuses the real
chunk layout, and verify:

- hooks are inert without an armed ``mixfusion_step`` (serial path unchanged);
- the root hook binds the forward call, delegates packing to the adapter,
  caches its bundle per CFG branch and restores the per-request output list;
- the replace-output hook returns the adapter-computed boundary value instead
  of running the (layout-incompatible) original forward;
- the expand hooks map per-request rows to per-chunk rows;
- ``apply_mixfusion`` / ``remove_mixfusion`` round-trip, including the
  ``transformer_blocks.*`` wildcard, idempotence and unknown-spec refusal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    MixFusionStep,
    override_forward_context,
)
from vllm_omni.diffusion.hooks.mixfusion import apply_mixfusion, remove_mixfusion
from vllm_omni.diffusion.mixfusion.adapter import MixFusionAdapter, PackedInputs
from vllm_omni.diffusion.mixfusion.execution_plan import (
    expand_per_chunk,
    reassemble_chunk_rows,
)
from vllm_omni.diffusion.mixfusion.plan import (
    MixFusionExpandArg,
    MixFusionExpandRows,
    MixFusionReplaceOutput,
    MixFusionRoot,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

SEQ_LEN = 4
CHUNK_SIZE = 2
NUM_REQUESTS = 2
NUM_CHUNKS = SEQ_LEN // CHUNK_SIZE * NUM_REQUESTS


class _RopePrepare(nn.Module):
    """Stand-in for ``ImageRopePrepare``: wrong for the packed layout.

    Returns a marker string so any test that accidentally runs it with the
    packed batch fails loudly instead of silently computing garbage.
    """

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict] = []

    def forward(self, hidden_states, img_shapes, txt_seq_lens):
        self.calls.append({"hidden_states": hidden_states})
        return hidden_states, "SERIAL_VID_FREQS", "SERIAL_TXT_FREQS"


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # AdaLN-style modulation params: per-request [B, 1] in, expanded to
        # per-chunk rows by the hook.
        self.img_mod = nn.Linear(1, 1)

    def forward(self, x, temb):
        mod = self.img_mod(temb)
        # Requires mod rows to match x rows ([num_chunks, chunk, C]); a
        # per-request [B, 1] tensor would broadcast-fail under torch >= 2.1
        # strictness or silently misalign, so assert the row count.
        assert mod.shape[0] == x.shape[0], f"img_mod rows {mod.shape[0]} != hidden rows {x.shape[0]}"
        return x + mod.unsqueeze(1)


class _NormOut(nn.Module):
    def forward(self, x, temb):
        assert temb.shape[0] == x.shape[0], f"temb rows {temb.shape[0]} != hidden rows {x.shape[0]}"
        return x


class _DummyAdapter:
    """Packs a list of ``[1, SEQ_LEN, C]`` latents into ``CHUNK_SIZE`` chunks."""

    def __init__(self) -> None:
        self.pack_calls: list[dict] = []

    def validate_inputs(self, module, input_batch) -> None:
        pass

    def pack_inputs(self, module, bound, *, cache: PackedInputs | None) -> PackedInputs:
        rows = [h.float() for h in bound.arguments["hidden_states"]]
        row_freqs = [torch.arange(SEQ_LEN, dtype=torch.float32).reshape(SEQ_LEN, 1)] * len(rows)
        chunks = [chunk for row in rows for chunk in row.split(CHUNK_SIZE, dim=1)]
        freq_chunks = [chunk for freqs in row_freqs for chunk in freqs.split(CHUNK_SIZE, dim=0)]
        per_row = SEQ_LEN // CHUNK_SIZE
        ranges = [(idx * per_row, (idx + 1) * per_row) for idx in range(len(rows))]
        chunk_to_request = [idx for idx in range(len(rows)) for _ in range(per_row)]
        packed_hidden = torch.cat(chunks, dim=0)
        bound.arguments["hidden_states"] = packed_hidden
        packed = PackedInputs(
            args=tuple(bound.args),
            kwargs=dict(bound.kwargs),
            plan=SimpleNamespace(
                request_chunk_ranges=ranges,
                chunk_to_request=torch.tensor(chunk_to_request, dtype=torch.long),
                image_row_counts=(1,) * len(rows),
            ),
            boundary_outputs={"rope": (packed_hidden, torch.cat(freq_chunks, dim=0), torch.zeros(SEQ_LEN, 1))},
        )
        self.pack_calls.append({"cache": cache, "produced": packed})
        return packed

    def restore_output(self, module, output, plan):
        return reassemble_chunk_rows(output.sample, plan.request_chunk_ranges, plan.image_row_counts)


class _DummyTransformer(nn.Module):
    """Minimal transformer with the four MixFusion boundaries."""

    def __init__(self, num_blocks: int = 2) -> None:
        super().__init__()
        self.mixfusion_adapter = _DummyAdapter()
        self.image_rope_prepare = _RopePrepare()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(num_blocks)])
        self.norm_out = _NormOut()

    @property
    def pack_calls(self) -> list[dict]:
        return self.mixfusion_adapter.pack_calls

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        img_shapes=None,
        return_dict=True,
    ):
        hidden_states, vid_freqs, txt_freqs = self.image_rope_prepare(hidden_states, img_shapes, None)
        # Keep what the rope boundary produced so tests can tell the packed
        # triple from the stand-in's serial markers apart.
        self.rope_outputs = (vid_freqs, txt_freqs)
        # Per-request conditioning rows: derived from the request list, NOT
        # from the hidden-state row count (which is num_chunks when packed).
        num_requests = len(img_shapes) if img_shapes is not None else hidden_states.shape[0]
        temb = torch.arange(num_requests, dtype=torch.float32).reshape(-1, 1)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, temb)
        sample = self.norm_out(hidden_states, temb)
        return SimpleNamespace(sample=sample)


def _plan() -> dict:
    return {
        "": MixFusionRoot(),
        "image_rope_prepare": MixFusionReplaceOutput(boundary="rope"),
        "transformer_blocks.*.img_mod": MixFusionExpandRows(),
        "norm_out": MixFusionExpandArg(arg_name="temb", expected_dims=2),
    }


def _latents() -> list[torch.Tensor]:
    torch.manual_seed(0)
    return [torch.randn(1, SEQ_LEN, 3) for _ in range(NUM_REQUESTS)]


def _armed_ctx() -> ForwardContext:
    return ForwardContext(mixfusion_step=MixFusionStep())


def _packed_kwargs(latents):
    return dict(
        hidden_states=latents,
        encoder_hidden_states=torch.zeros(NUM_REQUESTS, 5, 2),
        img_shapes=[(1, 4, 4)] * NUM_REQUESTS,
    )


def _serial_forward(model, batch_hidden: torch.Tensor):
    """Reference: run the dummy forward without any hooks installed."""
    return model(
        hidden_states=batch_hidden,
        encoder_hidden_states=torch.zeros(batch_hidden.shape[0], 5, 2),
    ).sample


class TestInertWithoutArmedStep:
    @pytest.mark.parametrize("ctx", [None, ForwardContext()], ids=["no-context", "no-step"])
    def test_serial_forward_unchanged(self, ctx):
        model = _DummyTransformer()
        batch = torch.cat(_latents(), dim=0)
        expected = _serial_forward(model, batch)
        apply_mixfusion(model, _plan())
        if ctx is None:
            out = _serial_forward(model, batch)
        else:
            with override_forward_context(ctx):
                out = _serial_forward(model, batch)
        assert torch.equal(out, expected)
        assert model.pack_calls == []


class TestPackedForward:
    def test_dummy_adapter_satisfies_the_protocol(self):
        assert isinstance(_DummyAdapter(), MixFusionAdapter)

    def test_root_packs_and_reassembles(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        latents = _latents()
        ctx = _armed_ctx()
        with override_forward_context(ctx):
            out = model(**_packed_kwargs(latents))

        assert isinstance(out, list) and len(out) == NUM_REQUESTS
        # Bit-exact against a hand-rolled reference on the packed layout.
        chunks = [latent.split(CHUNK_SIZE, dim=1) for latent in latents]
        expected = torch.cat([c for req in chunks for c in req], dim=0)
        for block in model.transformer_blocks:
            expected = expected + block.img_mod(
                expand_per_chunk(
                    torch.arange(NUM_REQUESTS, dtype=torch.float32).reshape(-1, 1), torch.tensor([0, 0, 1, 1])
                )
            ).unsqueeze(1)
        expected = reassemble_chunk_rows(expected, [(0, 2), (2, 4)], [1, 1])
        for got, exp, latent in zip(out, expected, latents):
            assert got.shape == latent.shape
            assert torch.equal(got, exp)
        # The step advertises what the downstream hooks and attention read.
        assert ctx.mixfusion_step.plan is ctx.mixfusion_step.packed.plan

    def test_replace_output_returns_packed_triple(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        latents = _latents()
        with override_forward_context(_armed_ctx()):
            model(**_packed_kwargs(latents))
        # The rope boundary was replaced by the packed triple: the stand-in
        # forward never ran and the freq tensors are the per-chunk ones.
        assert model.image_rope_prepare.calls == []
        vid_freqs, txt_freqs = model.rope_outputs
        assert isinstance(txt_freqs, torch.Tensor)
        assert vid_freqs.shape[0] == NUM_CHUNKS * CHUNK_SIZE
        # A serial call (no armed step) does run the original forward.
        _serial_forward(model, torch.cat(latents, dim=0))
        assert len(model.image_rope_prepare.calls) == 1
        assert model.rope_outputs == ("SERIAL_VID_FREQS", "SERIAL_TXT_FREQS")

    def test_cache_is_scoped_per_cfg_branch(self):
        # Sequential true-CFG alternates positive and negative calls within a
        # step; each branch must get back its own bundle. ``None`` is the
        # no-CFG branch.
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        kwargs = _packed_kwargs(_latents())
        branches = (None, None, "positive", "negative", "positive", "negative")
        for branch in branches:
            model.cfg_branch = branch
            with override_forward_context(_armed_ctx()):
                model(**kwargs)

        calls = model.pack_calls
        assert [call["cache"] is None for call in calls] == [True, False, True, True, False, False]
        assert calls[1]["cache"] is calls[0]["produced"]
        assert calls[4]["cache"] is calls[2]["produced"]
        assert calls[5]["cache"] is calls[3]["produced"]


class TestBoundaryErrors:
    def test_replace_output_missing_boundary_raises(self):
        model = _DummyTransformer()
        apply_mixfusion(model, {"image_rope_prepare": MixFusionReplaceOutput(boundary="missing")})
        step = MixFusionStep(packed=PackedInputs(args=(), kwargs={}, plan=SimpleNamespace()))
        with override_forward_context(ForwardContext(mixfusion_step=step)):
            with pytest.raises(KeyError, match="boundary output 'missing' was not produced"):
                model.image_rope_prepare(torch.zeros(1, 2), None, None)

    def test_expand_arg_unknown_name_raises(self):
        model = _DummyTransformer()
        apply_mixfusion(model, {"norm_out": MixFusionExpandArg(arg_name="missing")})
        step = MixFusionStep(plan=SimpleNamespace(chunk_to_request=torch.tensor([0])))
        with override_forward_context(ForwardContext(mixfusion_step=step)):
            with pytest.raises(TypeError, match="expects a forward argument named 'missing'"):
                model.norm_out(torch.zeros(1, 2), torch.zeros(1, 1))

    def test_expand_arg_wrong_dims_raises(self):
        model = _DummyTransformer()
        apply_mixfusion(model, {"norm_out": MixFusionExpandArg(arg_name="temb", expected_dims=3)})
        step = MixFusionStep(plan=SimpleNamespace(chunk_to_request=torch.tensor([0])))
        with override_forward_context(ForwardContext(mixfusion_step=step)):
            with pytest.raises(ValueError, match="must be 3-D"):
                model.norm_out(torch.zeros(1, 2), torch.zeros(1, 1))


class TestApplyRemove:
    def test_apply_registers_on_all_boundaries(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        assert model._mixfusion_hooks_applied is True
        assert "mixfusion_pack" in model._hook_registry._hooks
        assert "mixfusion_replace_output" in model.image_rope_prepare._hook_registry._hooks
        for block in model.transformer_blocks:
            assert "mixfusion_expand_rows" in block.img_mod._hook_registry._hooks
        assert "mixfusion_expand_arg" in model.norm_out._hook_registry._hooks

    def test_apply_is_idempotent(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        n_root_hooks = len(model._hook_registry._hooks)
        apply_mixfusion(model, _plan())
        assert len(model._hook_registry._hooks) == n_root_hooks

    def test_remove_round_trip(self):
        model = _DummyTransformer()
        batch = torch.cat(_latents(), dim=0)
        serial = _serial_forward(model, batch)
        apply_mixfusion(model, _plan())
        remove_mixfusion(model, _plan())
        assert model._mixfusion_hooks_applied is False
        assert not model._hook_registry._hooks
        assert not model.image_rope_prepare._hook_registry._hooks
        for block in model.transformer_blocks:
            assert not block.img_mod._hook_registry._hooks
        assert not model.norm_out._hook_registry._hooks
        assert torch.equal(_serial_forward(model, batch), serial)

    def test_unknown_spec_raises(self):
        model = _DummyTransformer()
        with pytest.raises(TypeError, match="Unsupported MixFusion plan entry"):
            apply_mixfusion(model, {"norm_out": object()})
