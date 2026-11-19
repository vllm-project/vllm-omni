# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Flat decoder sequences with request-owned prefix KV and piecewise attention.

The model prepares embeddings and document semantics. This layout family owns
packing, attention segments and output restoration, independently of image
projection, tokenization or diffusion scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from inspect import BoundArguments
from typing import TYPE_CHECKING

import torch
from torch import nn

from vllm_omni.diffusion.attention.backends.utils.piecewise_attn import build_segments
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionSequenceInfo

SEQUENCE_DOCUMENTS_KEY = "mixfusion_documents"

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.mixfusion.adapter import PackedInputs
    from vllm_omni.diffusion.mixfusion.execution_plan import PackedExecutionPlan
    from vllm_omni.diffusion.worker.input_batch import InputBatch


@dataclass(frozen=True)
class SequenceGeometry:
    request_id: str
    branch: int
    query_length: int
    padded_length: int
    prefix_length: int
    full_spans: tuple[tuple[int, int], ...]
    # Capture the invariant prefix on the first denoising step only.
    capture_length: int | None = None


@dataclass
class SequenceDocument:
    geometry: SequenceGeometry
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    prefix_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
    captured_kv: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


@dataclass
class SequenceSegmentBatch:
    causal: bool
    query_indices: torch.Tensor
    kv_indices: torch.Tensor
    cu_query: torch.Tensor
    cu_key: torch.Tensor
    query_lengths: tuple[int, ...]
    key_lengths: tuple[int, ...]


@dataclass
class PackedSequencePlan:
    documents: tuple[SequenceGeometry, ...]
    query_offsets: tuple[int, ...]
    segments: tuple[SequenceSegmentBatch, ...]


def build_sequence_plan(documents: tuple[SequenceGeometry, ...], device: torch.device) -> PackedSequencePlan:
    """Batch causal/full segments, without requiring aligned document phases."""
    query_offsets = [0]
    kv_offset = 0
    grouped: dict[bool, list[tuple[list[int], list[int]]]] = {}
    for doc in documents:
        qlen, plen = doc.query_length, doc.prefix_length
        if not 0 < qlen <= doc.padded_length or plen < 0:
            raise ValueError("Invalid packed sequence query/prefix length")
        end = 0
        for start, stop in doc.full_spans:
            if not end <= start < stop <= plen + qlen:
                raise ValueError("Packed full-attention spans must be ordered and within the document")
            end = stop
        if doc.capture_length is not None and not plen <= doc.capture_length <= plen + qlen:
            raise ValueError("Captured prefix must include imported KV and fit the first step")
        for segment in build_segments(doc.full_spans, plen, qlen):
            query = list(range(query_offsets[-1] + segment.q_start - plen, query_offsets[-1] + segment.q_end - plen))
            keys = list(range(kv_offset, kv_offset + segment.kv_end))
            grouped.setdefault(segment.mode == "causal", []).append((query, keys))
        query_offsets.append(query_offsets[-1] + qlen)
        kv_offset += plen + qlen
    batches = []
    for causal, segments in grouped.items():
        qi, ki, cq, ck = [], [], [0], [0]
        for query, keys in segments:
            qi.extend(query)
            ki.extend(keys)
            cq.append(len(qi))
            ck.append(len(ki))
        batches.append(
            SequenceSegmentBatch(
                causal,
                torch.tensor(qi, device=device, dtype=torch.long),
                torch.tensor(ki, device=device, dtype=torch.long),
                torch.tensor(cq, device=device, dtype=torch.int32),
                torch.tensor(ck, device=device, dtype=torch.int32),
                tuple(len(q) for q, _ in segments),
                tuple(len(k) for _, k in segments),
            )
        )
    return PackedSequencePlan(documents, tuple(query_offsets), tuple(batches))


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
        from vllm_omni.diffusion.mixfusion.adapter import PackedInputs

        documents = bound.arguments.get(SEQUENCE_DOCUMENTS_KEY)
        if not documents or not all(isinstance(doc, SequenceDocument) for doc in documents):
            raise ValueError("Packed decoder input requires prepared SequenceDocuments")
        geometry = tuple(doc.geometry for doc in documents)
        device = documents[0].hidden_states.device
        key = (geometry, device)
        plan = (
            adapter_state[1]
            if adapter_state is not None and adapter_state[0] == key
            else build_sequence_plan(geometry, device)
        )
        bound.arguments["inputs_embeds"] = torch.cat(
            [doc.hidden_states[:, : doc.geometry.query_length] for doc in documents], dim=1
        )
        bound.arguments["position_ids"] = torch.cat(
            [doc.position_ids[:, : doc.geometry.query_length] for doc in documents], dim=1
        )
        bound.arguments["custom_pos_emb"] = tuple(
            torch.cat([getattr(doc, name)[:, : doc.geometry.query_length] for doc in documents], dim=1)
            for name in ("cos", "sin")
        )
        bound.arguments["query_lens"] = [plan.query_offsets[-1]]
        bound.arguments["seq_lens"] = [plan.query_offsets[-1]]
        bound.arguments[SEQUENCE_DOCUMENTS_KEY] = None
        return PackedInputs(
            tuple(bound.args),
            dict(bound.kwargs),
            plan,
            boundary_outputs={SEQUENCE_DOCUMENTS_KEY: documents},
            adapter_state=(key, plan),
        )

    def restore_output(
        self, module: nn.Module, output, plan: PackedSequencePlan | PackedExecutionPlan
    ) -> list[torch.Tensor]:
        if not isinstance(plan, PackedSequencePlan):
            raise TypeError("Prefix-KV output requires its matching execution plan")
        hidden = output[0]
        restored = []
        for doc, start, end in zip(plan.documents, plan.query_offsets[:-1], plan.query_offsets[1:], strict=True):
            value = hidden[:, start:end]
            if doc.padded_length != doc.query_length:
                value = torch.nn.functional.pad(value, (0, 0, 0, doc.padded_length - doc.query_length))
            restored.append(value)
        return restored


def run_prefix_attention(
    attn: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: PackedSequencePlan,
    documents: list[SequenceDocument],
    layer_index: int,
) -> torch.Tensor:
    """Assemble each document's KV, run segment batches, restore query order."""
    keys, values = [], []
    for doc, start, end in zip(documents, plan.query_offsets[:-1], plan.query_offsets[1:], strict=True):
        k, v = key[start:end], value[start:end]
        if doc.geometry.prefix_length:
            pk, pv = doc.prefix_kv[layer_index]
            if pk.shape[0] != doc.geometry.prefix_length or pv.shape != pk.shape:
                raise ValueError("Packed prefix KV does not match the document's declared prefix length")
            k, v = torch.cat((pk, k)), torch.cat((pv, v))
        capture = doc.geometry.capture_length
        if capture is not None:
            # The request outlives this step. Copy only its invariant prefix;
            # a view would pin every current image token until completion.
            doc.captured_kv.append((k[:capture].detach().clone(), v[:capture].detach().clone()))
        keys.append(k)
        values.append(v)
    flat_key, flat_value = torch.cat(keys), torch.cat(values)
    result = torch.empty_like(query)
    for segment in plan.segments:
        output = attn.forward_packed(
            query.index_select(0, segment.query_indices),
            flat_key.index_select(0, segment.kv_indices),
            flat_value.index_select(0, segment.kv_indices),
            cu_seqlens=segment.cu_query,
            max_seqlen=max(segment.query_lengths),
            cu_seqlens_k=segment.cu_key,
            max_seqlen_k=max(segment.key_lengths),
            causal=segment.causal,
            sequence_lengths=(segment.query_lengths, segment.key_lengths),
        )
        result.index_copy_(0, segment.query_indices, output)
    return result
