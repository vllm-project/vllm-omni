# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MixFusion module-boundary hooks (CPU).

The hooks implement the packed chunk-stacked layout purely at declared module
boundaries (see ``distributed/mixfusion_plan.py``), mirroring the SP hook
mechanism. These tests use a minimal stand-in transformer whose pack method
reuses the real ``split_sequence_chunks`` layout, and verify:

- hooks are inert without an armed ``mixfusion_step`` (serial path unchanged);
- the root hook packs, rewrites ``hidden_states`` and reassembles the
  per-request output list;
- the rope hook returns the root-computed packed triple instead of running
  the (layout-incompatible) original forward;
- the expand hooks map per-request rows to per-chunk rows;
- ``apply_mixfusion`` / ``remove_mixfusion`` round-trip, including the
  ``transformer_blocks.*`` wildcard, idempotence and unknown-spec refusal.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.distributed.mixfusion_plan import (
    MixFusionExpandArg,
    MixFusionExpandRows,
    MixFusionRoot,
    MixFusionRopePrepare,
)
from vllm_omni.diffusion.forward_context import (
    ForwardContext,
    MixFusionStep,
    override_forward_context,
)
from vllm_omni.diffusion.hooks import module_has_active_hooks
from vllm_omni.diffusion.hooks.mixfusion import (
    apply_mixfusion,
    remove_mixfusion,
)
from vllm_omni.diffusion.mixfusion.layout import (
    expand_per_chunk,
    reassemble_chunk_rows,
    split_sequence_chunks,
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


class _DummyTransformer(nn.Module):
    """Minimal transformer with the four MixFusion boundaries."""

    def __init__(self, num_blocks: int = 2) -> None:
        super().__init__()
        self.image_rope_prepare = _RopePrepare()
        self.transformer_blocks = nn.ModuleList([_Block() for _ in range(num_blocks)])
        self.norm_out = _NormOut()
        self.pack_calls: list[dict] = []

    def prepare_mixfusion_inputs(
        self,
        hidden_states,
        encoder_hidden_states,
        encoder_hidden_states_mask=None,
        img_shapes=None,
        txt_seq_lens=None,
        attention_kwargs=None,
        cached=None,
    ):
        """Stand-in pack method with the real model's contract."""
        self.pack_calls.append({"cached": cached})
        rows = [h.float() for h in hidden_states]
        row_freqs = [torch.arange(SEQ_LEN, dtype=torch.float32).reshape(SEQ_LEN, 1)] * len(rows)
        chunks, freq_chunks, request_chunk_ranges, chunk_to_request = split_sequence_chunks(rows, row_freqs, CHUNK_SIZE)
        packed_hidden = torch.cat(chunks, dim=0)
        return SimpleNamespace(
            hidden_states=packed_hidden,
            encoder_hidden_states_mask=encoder_hidden_states_mask,
            image_rotary_emb=(
                torch.cat(freq_chunks, dim=0),
                torch.zeros(SEQ_LEN, 1),
            ),
            plan=SimpleNamespace(
                request_chunk_ranges=request_chunk_ranges,
                chunk_to_request=torch.tensor(chunk_to_request, dtype=torch.long),
            ),
            image_row_counts=[1] * len(rows),
        )

    def forward(
        self,
        hidden_states,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        timestep=None,
        img_shapes=None,
        txt_seq_lens=None,
        attention_kwargs=None,
        return_dict=True,
    ):
        hidden_states, vid_freqs, txt_freqs = self.image_rope_prepare(hidden_states, img_shapes, txt_seq_lens)
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
        "": MixFusionRoot(pack_method="prepare_mixfusion_inputs"),
        "image_rope_prepare": MixFusionRopePrepare(),
        "transformer_blocks.*.img_mod": MixFusionExpandRows(),
        "norm_out": MixFusionExpandArg(arg_index=1, expected_dims=2),
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
        txt_seq_lens=[5] * NUM_REQUESTS,
    )


def _serial_forward(model, batch_hidden: torch.Tensor):
    """Reference: run the dummy forward without any hooks installed."""
    return model(
        hidden_states=batch_hidden,
        encoder_hidden_states=torch.zeros(batch_hidden.shape[0], 5, 2),
    ).sample


class TestInertWithoutArmedStep:
    def test_no_forward_context(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        batch = torch.cat(_latents(), dim=0)
        out = model(hidden_states=batch, encoder_hidden_states=torch.zeros(2, 5, 2))
        assert torch.equal(out.sample, _serial_forward(model, batch))
        assert model.pack_calls == []

    def test_context_without_step(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        batch = torch.cat(_latents(), dim=0)
        with override_forward_context(ForwardContext()):
            out = model(hidden_states=batch, encoder_hidden_states=torch.zeros(2, 5, 2))
        assert torch.equal(out.sample, _serial_forward(model, batch))
        assert model.pack_calls == []

    def test_armed_but_tensor_hidden_states_stays_serial(self):
        # Memory profiling style: a batched tensor call while the step is
        # armed must not attempt packing.
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        batch = torch.cat(_latents(), dim=0)
        with override_forward_context(_armed_ctx()):
            out = model(hidden_states=batch, encoder_hidden_states=torch.zeros(2, 5, 2))
        assert torch.equal(out.sample, _serial_forward(model, batch))
        assert model.pack_calls == []


class TestPackedForward:
    def test_root_packs_and_reassembles(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        latents = _latents()
        with override_forward_context(_armed_ctx()):
            out = model(**_packed_kwargs(latents))

        assert isinstance(out, list) and len(out) == NUM_REQUESTS
        for request_sample, latent in zip(out, latents):
            assert request_sample.shape == latent.shape
        # Bit-exact against a hand-rolled reference on the packed layout.
        chunks = [latent.split(CHUNK_SIZE, dim=1) for latent in latents]
        packed = torch.cat([c for req in chunks for c in req], dim=0)
        expected = packed
        for block in model.transformer_blocks:
            expected = expected + block.img_mod(
                expand_per_chunk(
                    torch.arange(NUM_REQUESTS, dtype=torch.float32).reshape(-1, 1), torch.tensor([0, 0, 1, 1])
                )
            ).unsqueeze(1)
        expected = reassemble_chunk_rows(expected, [(0, 2), (2, 4)], [1, 1])
        for got, exp in zip(out, expected):
            assert torch.equal(got, exp)

    def test_rope_hook_returns_packed_triple(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        latents = _latents()
        with override_forward_context(_armed_ctx()):
            model(**_packed_kwargs(latents))
        # The rope boundary was replaced by the packed triple: the stand-in
        # forward never ran, and the freq tensors are the packed ones (not the
        # "SERIAL_*" string markers, not the per-request tables).
        assert model.image_rope_prepare.calls == []
        vid_freqs, txt_freqs = model.rope_outputs
        assert isinstance(vid_freqs, torch.Tensor)
        assert isinstance(txt_freqs, torch.Tensor)
        # Per-chunk stacked rows: one block of CHUNK_SIZE freq rows per chunk.
        assert vid_freqs.shape[0] == NUM_CHUNKS * CHUNK_SIZE
        # A serial call (no armed step) does run the original forward.
        model.image_rope_prepare.calls.clear()
        batch = torch.cat(latents, dim=0)
        model(hidden_states=batch, encoder_hidden_states=torch.zeros(2, 5, 2))
        assert len(model.image_rope_prepare.calls) == 1
        assert model.rope_outputs == ("SERIAL_VID_FREQS", "SERIAL_TXT_FREQS")

    def test_cache_threaded_across_calls(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        kwargs = _packed_kwargs(_latents())
        with override_forward_context(_armed_ctx()):
            model(**kwargs)
        with override_forward_context(_armed_ctx()):
            model(**kwargs)
        assert len(model.pack_calls) == 2
        # Second call received the first bundle as ``cached``.
        assert model.pack_calls[0]["cached"] is None
        assert model.pack_calls[1]["cached"] is not None


class TestApplyRemove:
    def test_apply_registers_on_all_boundaries(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        assert model._mixfusion_hooks_applied is True
        assert "mixfusion_pack" in model._hook_registry._hooks
        assert "mixfusion_rope_prepare" in model.image_rope_prepare._hook_registry._hooks
        for block in model.transformer_blocks:
            assert "mixfusion_expand_rows" in block.img_mod._hook_registry._hooks
        assert "mixfusion_expand_arg" in model.norm_out._hook_registry._hooks
        assert module_has_active_hooks(model)

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
        assert not module_has_active_hooks(model)
        assert not module_has_active_hooks(model.image_rope_prepare)
        for block in model.transformer_blocks:
            assert not module_has_active_hooks(block.img_mod)
        assert not module_has_active_hooks(model.norm_out)
        assert torch.equal(_serial_forward(model, batch), serial)

    def test_unknown_spec_raises(self):
        model = _DummyTransformer()
        with pytest.raises(TypeError, match="Unsupported MixFusion plan entry"):
            apply_mixfusion(model, {"norm_out": object()})
