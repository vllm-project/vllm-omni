# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Chunked decoder inputs with request-owned prefix KV and piecewise attention.

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
from vllm_omni.diffusion.mixfusion.execution_plan import PackedAttentionSegments, build_chunk_plan
from vllm_omni.diffusion.mixfusion.scheduling import MixFusionSequenceInfo

SEQUENCE_DOCUMENTS_KEY = "mixfusion_documents"

if TYPE_CHECKING:
    from vllm_omni.diffusion.attention.layer import Attention
    from vllm_omni.diffusion.mixfusion.adapter import PackedInputs
    from vllm_omni.diffusion.mixfusion.execution_plan import PackedExecutionPlan
    from vllm_omni.diffusion.worker.input_batch import InputBatch


@dataclass(frozen=True)
class PackedDocumentGeometry:
    request_id: str
    branch: int
    query_length: int
    padded_length: int
    prefix_length: int
    full_spans: tuple[tuple[int, int], ...]
    # Half-open image range in the logical Q+KV sequence.
    image_span: tuple[int, int]
    # Capture the invariant prefix on the first denoising step only.
    capture_length: int | None = None


@dataclass
class PackedDocument:
    geometry: PackedDocumentGeometry
    hidden_states: torch.Tensor
    position_ids: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    prefix_kv: tuple[tuple[torch.Tensor, torch.Tensor], ...] = ()
    captured_kv: list[tuple[torch.Tensor, torch.Tensor]] = field(default_factory=list)


def build_document_plan(documents: tuple[PackedDocumentGeometry, ...], device: torch.device) -> PackedExecutionPlan:
    """Address each request's queries in image-chunk then non-image storage."""
    image_lengths = [doc.image_span[1] - doc.image_span[0] for doc in documents]
    # Each request/CFG branch supplies one image row.
    plan = build_chunk_plan(image_lengths, [1] * len(documents), device)
    other_offset = plan.num_chunks * plan.chunk_size
    kv_offset = sum(doc.query_length for doc in documents)
    document_indices = []
    grouped: dict[bool, list[tuple[list[int], list[int]]]] = {}
    for doc, (chunk_start, chunk_end) in zip(documents, plan.request_chunk_ranges, strict=True):
        qlen, plen = doc.query_length, doc.prefix_length
        if not 0 < qlen <= doc.padded_length or plen < 0:
            raise ValueError("Invalid packed sequence query/prefix length")
        start, stop = doc.image_span
        if not plen <= start < stop <= plen + qlen:
            raise ValueError("Image chunks must cover current queries")
        before, after = start - plen, plen + qlen - stop
        indices = [
            *range(other_offset, other_offset + before),
            *range(chunk_start * plan.chunk_size, chunk_end * plan.chunk_size),
            *range(other_offset + before, other_offset + before + after),
        ]
        document_indices.append(torch.tensor(indices, device=device, dtype=torch.long))
        other_offset += before + after
        end = 0
        for start, stop in doc.full_spans:
            if not end <= start < stop <= plen + qlen:
                raise ValueError("Packed full-attention spans must be ordered and within the document")
            end = stop
        if doc.capture_length is not None and not plen <= doc.capture_length <= plen + qlen:
            raise ValueError("Captured prefix must include imported KV and fit the first step")
        keys_in_order = [*range(kv_offset, kv_offset + plen), *indices]
        for segment in build_segments(doc.full_spans, plen, qlen):
            query = indices[segment.q_start - plen : segment.q_end - plen]
            keys = keys_in_order[: segment.kv_end]
            grouped.setdefault(segment.mode == "causal", []).append((query, keys))
        kv_offset += plen
    batches = []
    for causal, segments in grouped.items():
        qi, ki, cq, ck = [], [], [0], [0]
        for query, keys in segments:
            qi.extend(query)
            ki.extend(keys)
            cq.append(len(qi))
            ck.append(len(ki))
        batches.append(
            PackedAttentionSegments(
                causal,
                torch.tensor(qi, device=device, dtype=torch.long),
                torch.tensor(ki, device=device, dtype=torch.long),
                torch.tensor(cq, device=device, dtype=torch.int32),
                torch.tensor(ck, device=device, dtype=torch.int32),
                tuple(len(q) for q, _ in segments),
                tuple(len(k) for _, k in segments),
            )
        )
    plan.documents = documents
    plan.document_indices = tuple(document_indices)
    plan.segments = tuple(batches)
    return plan


def pack_document_queries(rows: list[torch.Tensor], plan: PackedExecutionPlan) -> torch.Tensor:
    """Pack contiguous image rows followed by the remaining queries."""
    images, others = [], []
    for row, doc in zip(rows, plan.documents, strict=True):
        start, stop = (pos - doc.prefix_length for pos in doc.image_span)
        images.append(row[start:stop])
        if start:
            others.append(row[:start])
        if stop < doc.query_length:
            others.append(row[stop : doc.query_length])
    return torch.cat(images + others)


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


def run_prefix_attention(
    attn: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    plan: PackedExecutionPlan,
    documents: list[PackedDocument],
    layer_index: int,
) -> torch.Tensor:
    """Gather legal KV directly from chunk storage or each request's cache."""
    cached_keys, cached_values = [], []
    for doc, indices in zip(documents, plan.document_indices, strict=True):
        plen = doc.geometry.prefix_length
        if plen:
            pk, pv = doc.prefix_kv[layer_index]
            if pk.shape[0] != plen or pv.shape != pk.shape:
                raise ValueError("Packed prefix KV does not match the document's declared prefix length")
            cached_keys.append(pk)
            cached_values.append(pv)
        capture = doc.geometry.capture_length
        if capture is not None:
            # index_select owns the captured storage. A view could pin all image tokens.
            rows = indices[: capture - plen]
            ck, cv = key.index_select(0, rows), value.index_select(0, rows)
            if plen:
                ck, cv = torch.cat((pk, ck)), torch.cat((pv, cv))
            doc.captured_kv.append((ck.detach(), cv.detach()))
    flat_key = torch.cat((key, *cached_keys)) if cached_keys else key
    flat_value = torch.cat((value, *cached_values)) if cached_values else value
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
