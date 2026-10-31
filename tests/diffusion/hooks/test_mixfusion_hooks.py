# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MixFusion module-boundary hooks (CPU).

The hooks implement the packed chunk-stacked layout purely at declared module
boundaries (see ``mixfusion/plan.py``), mirroring the SP hook mechanism. The
model-specific half lives behind a ``MixFusionAdapter``; these tests use a
minimal stand-in transformer plus a fake adapter that reuses the real
``split_sequence_chunks`` layout, and verify:

- hooks are inert without an armed ``mixfusion_step`` (serial path unchanged);
- the root hook binds the forward call, delegates packing to the adapter and
  restores the per-request output list;
- the replace-output hook returns the adapter-computed boundary value instead
  of running the (layout-incompatible) original forward;
- the expand hooks map per-request rows to per-chunk rows;
- adapter refusals (``can_pack`` false / ``pack_inputs`` returning ``None``)
  and a missing adapter fail or fall back as documented;
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
from vllm_omni.diffusion.hooks import module_has_active_hooks
from vllm_omni.diffusion.hooks.mixfusion import (
    apply_mixfusion,
    remove_mixfusion,
)
from vllm_omni.diffusion.mixfusion.adapter import (
    AttentionLayout,
    MixFusionAdapter,
    PackedInputs,
    StreamSpec,
)
from vllm_omni.diffusion.mixfusion.execution_plan import (
    expand_per_chunk,
    reassemble_chunk_rows,
    split_sequence_chunks,
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
    """Fake adapter with the Qwen adapter's shape (list in, list out)."""

    def can_pack(self, module, bound) -> tuple[bool, str]:
        if "hidden_states" not in bound.arguments:
            raise TypeError("MixFusion packing requires a `hidden_states` argument.")
        if not isinstance(bound.arguments["hidden_states"], (list, tuple)):
            return False, "hidden_states is a batched tensor (serial forward)"
        return True, ""

    def build_attention_layout(self, plan=None) -> AttentionLayout:
        return AttentionLayout((StreamSpec("text", "request"), StreamSpec("image", "chunked")))

    def pack_inputs(self, module, bound, *, step, cache: PackedInputs | None) -> PackedInputs:
        cached = None if cache is None or cache.adapter_state is None else cache.adapter_state["prepare_inputs"]
        packed = module.prepare_mixfusion_inputs(
            hidden_states=bound.arguments["hidden_states"],
            encoder_hidden_states=bound.arguments.get("encoder_hidden_states"),
            encoder_hidden_states_mask=bound.arguments.get("encoder_hidden_states_mask"),
            img_shapes=bound.arguments.get("img_shapes"),
            txt_seq_lens=bound.arguments.get("txt_seq_lens"),
            attention_kwargs=bound.arguments.get("attention_kwargs"),
            cached=cached,
        )
        bound.arguments["hidden_states"] = packed.hidden_states
        bound.arguments["encoder_hidden_states_mask"] = packed.encoder_hidden_states_mask
        return PackedInputs(
            args=tuple(bound.args),
            kwargs=dict(bound.kwargs),
            plan=packed.plan,
            boundary_outputs={"rope": (packed.hidden_states, *packed.image_rotary_emb)},
            cache_key=packed.key,
            adapter_state={"prepare_inputs": packed},
        )

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
                image_row_counts=tuple([1] * len(rows)),
            ),
            image_row_counts=[1] * len(rows),
            key=("dummy", tuple(h.shape for h in hidden_states)),
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

    def test_step_carries_adapter_plan_and_packed(self):
        model = _DummyTransformer()
        apply_mixfusion(model, _plan())
        ctx = _armed_ctx()
        with override_forward_context(ctx):
            model(**_packed_kwargs(_latents()))
        step = ctx.mixfusion_step
        assert step.adapter is model.mixfusion_adapter
        assert step.plan.image_row_counts == (1, 1)
        assert step.packed.boundary_outputs["rope"][0].shape[0] == NUM_CHUNKS

    def test_replace_output_returns_packed_triple(self):
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


class _FakeModel(nn.Module):
    def __init__(self, adapter) -> None:
        super().__init__()
        if adapter is not None:
            self.mixfusion_adapter = adapter
        self.seen = None

    def forward(self, hidden_states, shift=0):
        self.seen = (hidden_states, shift)
        return SimpleNamespace(sample=hidden_states + shift)


class _RecordingAdapter:
    """Fake adapter that records the framework's calls."""

    def __init__(self, *, can_pack: bool = True, returns_none: bool = False) -> None:
        self._can_pack = can_pack
        self._returns_none = returns_none
        self.calls: list[tuple] = []

    def build_packed_call(self, input_batch):
        self.calls.append(("build_packed_call", input_batch))
        return {"hidden_states": input_batch.latents}

    def can_pack(self, module, bound):
        self.calls.append(("can_pack", dict(bound.arguments)))
        return self._can_pack, "" if self._can_pack else "refused"

    def build_attention_layout(self, plan=None):
        self.calls.append(("build_attention_layout", plan))
        return AttentionLayout((StreamSpec("image", "chunked"),))

    def pack_inputs(self, module, bound, *, step, cache):
        self.calls.append(("pack_inputs", dict(bound.arguments), cache))
        if self._returns_none:
            return None
        bound.arguments["hidden_states"] = bound.arguments["hidden_states"] * 10
        return PackedInputs(args=tuple(bound.args), kwargs=dict(bound.kwargs), plan="PLAN")

    def prepare_position_embeddings(self, module, plan, cache=None):
        return None

    def restore_output(self, module, output, plan):
        self.calls.append(("restore_output", plan))
        return ["restored"]


class TestAdapterProtocol:
    def test_is_runtime_checkable_protocol(self):
        assert isinstance(_RecordingAdapter(), MixFusionAdapter)

    def test_root_hook_delegates_all_three_methods(self):
        adapter = _RecordingAdapter()
        model = _FakeModel(adapter)
        apply_mixfusion(model, {"": MixFusionRoot()})
        x = torch.zeros(1, 2)
        ctx = _armed_ctx()
        with override_forward_context(ctx):
            out = model(hidden_states=x)
        assert out == ["restored"]
        assert [call[0] for call in adapter.calls] == [
            "can_pack",
            "pack_inputs",
            "build_attention_layout",
            "restore_output",
        ]
        # The adapter sees the original forward signature (no ``self``), with
        # a positional argument bound by name; defaults stay unbound.
        assert torch.equal(adapter.calls[0][1]["hidden_states"], x)
        assert "shift" not in adapter.calls[0][1]
        # The rewritten call reached the model.
        assert torch.equal(model.seen[0], x * 10)
        # The step advertises what the hooks downstream need.
        assert ctx.mixfusion_step.plan == "PLAN"
        assert ctx.mixfusion_step.layout is not None
        assert ctx.mixfusion_step.adapter is adapter
        assert ctx.mixfusion_step.packed is not None

    def test_can_pack_false_keeps_serial(self):
        adapter = _RecordingAdapter(can_pack=False)
        model = _FakeModel(adapter)
        apply_mixfusion(model, {"": MixFusionRoot()})
        x = torch.zeros(1, 2)
        ctx = _armed_ctx()
        with override_forward_context(ctx):
            out = model(hidden_states=x)
        assert torch.equal(out.sample, x)
        assert [call[0] for call in adapter.calls] == ["can_pack"]
        assert ctx.mixfusion_step.plan is None
        assert ctx.mixfusion_step.packed is None

    def test_pack_inputs_none_keeps_serial(self):
        adapter = _RecordingAdapter(returns_none=True)
        model = _FakeModel(adapter)
        apply_mixfusion(model, {"": MixFusionRoot()})
        x = torch.zeros(1, 2)
        with override_forward_context(_armed_ctx()):
            out = model(hidden_states=x)
        assert torch.equal(out.sample, x)
        assert [call[0] for call in adapter.calls] == ["can_pack", "pack_inputs"]

    def test_missing_adapter_raises_actionable_error(self):
        model = _FakeModel(adapter=None)
        apply_mixfusion(model, {"": MixFusionRoot()})
        with override_forward_context(_armed_ctx()):
            with pytest.raises(TypeError, match="declares no `mixfusion_adapter`"):
                model(hidden_states=torch.zeros(1, 2))


class TestBoundaryErrors:
    def test_replace_output_missing_boundary_raises(self):
        model = _DummyTransformer()
        apply_mixfusion(model, {"image_rope_prepare": MixFusionReplaceOutput(boundary="missing")})
        step = MixFusionStep(
            adapter=_DummyAdapter(),
            packed=PackedInputs(args=(), kwargs={}, plan=SimpleNamespace()),
        )
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
