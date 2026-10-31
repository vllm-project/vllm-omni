# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-side MixFusion contract tests (CPU).

Pins the public surface a new model plugs into: the ``AttentionLayout`` /
``StreamSpec`` declarations (with their validation), the ``PackedInputs``
bundle the framework moves between hooks, and structural conformance with the
``MixFusionAdapter`` protocol. Everything runs against a fake model so the
framework layer stays independent of any concrete DiT.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.mixfusion.adapter import (
    AttentionLayout,
    MixFusionAdapter,
    PackedInputs,
    StreamSpec,
)
from vllm_omni.diffusion.mixfusion.execution_plan import PackedExecutionPlan

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


class _FakeAdapter:
    """A complete protocol implementation that touches no model code."""

    def build_packed_call(self, input_batch) -> dict:
        return {"hidden_states": input_batch.latents}

    def can_pack(self, module, bound):
        return True, "ok"

    def build_attention_layout(self, plan=None) -> AttentionLayout:
        return AttentionLayout((StreamSpec("text", "request"), StreamSpec("image", "chunked")))

    def pack_inputs(self, module, bound, *, step, cache):
        return None

    def prepare_position_embeddings(self, module, plan, cache=None):
        return None

    def restore_output(self, module, output, plan) -> list:
        return list(output)


class _IncompleteAdapter:
    """Missing ``restore_output``; must not satisfy the protocol."""

    def build_packed_call(self, input_batch) -> dict:
        return {}

    def can_pack(self, module, bound):
        return True, "ok"

    def build_attention_layout(self, plan=None) -> AttentionLayout:
        return AttentionLayout((StreamSpec("image", "chunked"),))

    def pack_inputs(self, module, bound, *, step, cache):
        return None


# ── StreamSpec ──


def test_stream_spec_records_name_and_kind():
    spec = StreamSpec("image", "chunked")

    assert (spec.name, spec.kind) == ("image", "chunked")


@pytest.mark.parametrize("name", ["", None])
def test_stream_spec_rejects_an_empty_name(name):
    with pytest.raises(ValueError, match="StreamSpec.name must be non-empty"):
        StreamSpec(name, "chunked")


def test_stream_spec_rejects_an_unknown_kind():
    with pytest.raises(ValueError, match=r"StreamSpec 'image' has unknown kind 'batched'"):
        StreamSpec("image", "batched")


# ── AttentionLayout ──


def test_attention_layout_preserves_declaration_order():
    layout = AttentionLayout((StreamSpec("text", "request"), StreamSpec("image", "chunked")))

    assert layout.names == ("text", "image")
    assert layout.streams[1].kind == "chunked"


def test_attention_layout_rejects_no_streams():
    with pytest.raises(ValueError, match="AttentionLayout needs at least one stream"):
        AttentionLayout(())


def test_attention_layout_rejects_duplicate_names():
    with pytest.raises(ValueError, match=r"stream names must be unique, got \['image', 'image'\]"):
        AttentionLayout((StreamSpec("image", "chunked"), StreamSpec("image", "request")))


# ── PackedInputs ──


def test_packed_inputs_defaults_are_per_instance():
    plan = PackedExecutionPlan(
        chunk_size=4,
        num_chunks=1,
        request_chunk_ranges=[(0, 1)],
        image_row_counts=(1,),
        chunk_to_request=torch.zeros(1, dtype=torch.long),
        request_stream_lens={"image": (4,)},
        stream_padded_lens={"image": 4},
        cu_seqlens=[0, 4],
        cu_seqlens_tensor=torch.tensor([0, 4], dtype=torch.int32),
        max_seq_len=4,
        output_indices={"image": (torch.arange(4), None)},
    )
    first = PackedInputs(args=(), kwargs={}, plan=plan)
    second = PackedInputs(args=(), kwargs={}, plan=plan)

    assert first.boundary_outputs == {}
    assert first.cache_key is None
    assert first.adapter_state is None
    # Reused plans must not share hook payload storage across calls.
    first.boundary_outputs["rope"] = "value"
    assert second.boundary_outputs == {}


# ── protocol conformance ──


def test_a_complete_adapter_satisfies_the_protocol():
    assert isinstance(_FakeAdapter(), MixFusionAdapter)


def test_an_adapter_missing_a_method_does_not_satisfy_the_protocol():
    assert not isinstance(_IncompleteAdapter(), MixFusionAdapter)


def test_protocol_checks_are_structural_not_nominal():
    # The fake adapter declares no base class: models plug in by shape only.
    adapter = _FakeAdapter()
    assert not isinstance(adapter, nn.Module)
    assert type(adapter).__bases__ == (object,)
