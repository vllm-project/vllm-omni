# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MixFusion adapter contract and prefix-KV packing.

The root hook packs and restores a forward call through an object
implementing :class:`MixFusionAdapter`. Models with a declarative
``_mixfusion_plan`` select the framework's chunked joint or prefix-KV adapter
through the root's request declaration. An explicit ``mixfusion_adapter`` is
the escape hatch for a layout family the declarative plan cannot express.

The hooks never interpret the payloads in :class:`PackedInputs`
(``boundary_outputs`` and ``adapter_state``); they only move them between the
boundaries the plan declares.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import BoundArguments
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import torch
from torch import nn

from vllm_omni.diffusion.mixfusion.execution_plan import (
    PackedDocumentGeometry,
    PackedExecutionPlan,
    build_document_plan,
    pack_document_queries,
)
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionSequenceInfo

if TYPE_CHECKING:
    from vllm_omni.diffusion.worker.input_batch import InputBatch

SEQUENCE_DOCUMENTS_KEY = "mixfusion_documents"

#: How a stream's tokens are laid out across the packed rows. ``"chunked"``
#: streams arrive as one row per (request, chunk) and carry one chunk per
#: position in the second dimension; ``"request"`` streams arrive as one row
#: per request with the real token count in ``request_stream_lens``.
StreamKind = Literal["chunked", "request"]


@dataclass(frozen=True)
class StreamSpec:
    """One attention stream of a packed model call.

    Stream order in the adapter's :class:`AttentionLayout` is the token order
    inside every packed request.
    """

    name: str
    kind: StreamKind


@dataclass(frozen=True)
class AttentionLayout:
    """The model's attention streams, in per-request token order.

    Exactly one stream is ``"chunked"``: it is the positional Q/K/V of the
    packed attention call. The adapter stores the layout on the plan.
    """

    streams: tuple[StreamSpec, ...]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.streams)


@dataclass
class PackedInputs:
    """Rewritten forward call produced by :meth:`MixFusionAdapter.pack_inputs`.

    ``boundary_outputs`` replaces the outputs of boundaries only the adapter
    can compute; ``adapter_state`` carries adapter-private reuse data for the
    next step (step-invariant layout only — it must not reference the input
    latents, text embeddings or masks, which live on the active step). Both
    are opaque to the framework.
    """

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    plan: PackedExecutionPlan
    boundary_outputs: dict[str, Any] = field(default_factory=dict)
    adapter_state: Any = None


@runtime_checkable
class MixFusionAdapter(Protocol):
    """Model-side contract for the MixFusion packed path.

    Scheduler admission precedes packing. Worker validation checks only that
    the actual inputs satisfy the model contract; it never chooses a fallback.
    """

    def validate_inputs(self, module: nn.Module, input_batch: Any) -> None:
        """Raise when runtime inputs disagree with the scheduled packed contract."""
        ...

    def pack_inputs(self, module: nn.Module, bound: Any, *, adapter_state: Any | None) -> PackedInputs:
        """Rewrite the bound forward call into the packed layout.

        ``bound`` is an :class:`inspect.BoundArguments` over the model's
        forward. ``adapter_state`` is this branch's value from the previous
        step's ``PackedInputs`` (or ``None``); it holds only step-invariant
        layout data, so reusing it must never keep inputs alive.
        """
        ...

    def restore_output(self, module: nn.Module, output: Any, plan: PackedExecutionPlan) -> list[Any]:
        """Turn the packed forward output back into one output per request."""
        ...


@dataclass
class PackedDocument:
    geometry: PackedDocumentGeometry
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    prefix_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
    captured_kv: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


class PrefixKVAdapter:
    """Root adapter for decoder-only, token-local blocks with prefix KV."""

    def validate_inputs(self, module: nn.Module, input_batch: InputBatch) -> None:
        if not isinstance(input_batch.latents, list):
            raise ValueError("Packed sequences require per-request latents")
        for latent, state in zip(input_batch.latents, input_batch.states, strict=True):
            info = state.mixfusion_info
            if not isinstance(info, MixFusionSequenceInfo) or tuple(latent.shape) != info.latent_shape:
                raise ValueError(
                    f"MixFusion request {state.request_id}: actual spatial latents disagree with admission"
                )
            if state.attention_schedule or state.attention_sigma_schedule:
                raise ValueError("Packed prefix-KV sequences do not support request attention schedules")

    def pack_inputs(self, module: nn.Module, bound: BoundArguments, *, adapter_state=None) -> PackedInputs:
        documents = bound.arguments.get(SEQUENCE_DOCUMENTS_KEY)
        if not documents or not all(isinstance(doc, PackedDocument) for doc in documents):
            raise ValueError("Packed decoder input requires prepared PackedDocuments")
        geometry = tuple(doc.geometry for doc in documents)
        device = documents[0].hidden_states.device
        key = (geometry, device)
        plan = (
            adapter_state[1]
            if adapter_state is not None and adapter_state[0] == key
            else build_document_plan(geometry, device)
        )

        def pack(name: str) -> torch.Tensor:
            return pack_document_queries([getattr(doc, name)[0] for doc in documents], plan).unsqueeze(0)

        bound.arguments["inputs_embeds"] = pack("hidden_states")
        bound.arguments["position_ids"] = pack("position_ids")
        bound.arguments["custom_pos_emb"] = (pack("cos"), pack("sin"))
        length = bound.arguments["inputs_embeds"].shape[1]
        bound.arguments["query_lens"] = [length]
        bound.arguments["seq_lens"] = [length]
        bound.arguments[SEQUENCE_DOCUMENTS_KEY] = None
        return PackedInputs(
            tuple(bound.args),
            dict(bound.kwargs),
            plan,
            boundary_outputs={SEQUENCE_DOCUMENTS_KEY: documents},
            adapter_state=(key, plan),
        )

    def restore_output(self, module: nn.Module, output, plan: PackedExecutionPlan) -> list[torch.Tensor]:
        hidden = output[0]
        restored = []
        for doc, indices in zip(plan.documents, plan.document_indices, strict=True):
            value = hidden.index_select(1, indices)
            if doc.padded_length != doc.query_length:
                value = torch.nn.functional.pad(value, (0, 0, 0, doc.padded_length - doc.query_length))
            restored.append(value)
        return restored
