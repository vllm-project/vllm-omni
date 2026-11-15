# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed PixelUMM local attention on vLLM-Omni's packed varlen attention path.

After the decoder's tile-major permutation every query tile is a contiguous
token range, so one varlen call covers all tiles instead of one call per tile.
Each tile's 3x3 spatial neighbourhood is gathered once into contiguous K/V, in
the order of the eager concatenation, and the layer's own attention backend
runs the call from ``AttentionMetadata.extra`` cu_seqlens. Only backends that
keep N-document packed boundaries (FlashAttention on CUDA/ROCm/MUSA) qualify;
others keep the per-tile path.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata

Neighborhoods = Sequence[tuple[tuple[int, int], Sequence[tuple[int, int]]]]


@dataclass(frozen=True)
class PackedLocalAttention:
    """Packed varlen plan for every query tile of a ``[batch, tokens]`` sequence.

    Segment ``s`` is one query tile. It reads the gathered tokens
    ``kv_index[cu_seqlens_k[s] : cu_seqlens_k[s + 1]]`` of the flattened batch.
    """

    kv_index: torch.Tensor
    metadata: AttentionMetadata

    def __call__(
        self, attention: torch.nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """``[B, N, heads, D]`` queries over ``[B, N, kv_heads, D]`` K/V -> ``[B, N, heads, D]``."""
        batch, count, heads, dim = query.shape
        kv_heads = key.shape[2]
        # The packed path takes one sequence; batch elements are separate segments.
        query = query.reshape(1, batch * count, heads, dim)
        key = key.reshape(batch * count, kv_heads, dim)
        value = value.reshape(batch * count, kv_heads, dim)
        key = key.index_select(0, self.kv_index).unsqueeze(0)
        value = value.index_select(0, self.kv_index).unsqueeze(0)
        output = attention(query, key, value, self.metadata)
        return output.view(batch, count, heads, dim)


def plan_local_attention(
    neighborhoods: Neighborhoods, batch: int, tokens: int, device: torch.device
) -> PackedLocalAttention:
    """Build the packed plan for the tiles of ``local_attention_layout``.

    Tiles are consecutive ranges covering ``[0, tokens)``; batch ``b`` is
    offset by ``b * tokens`` in the packed sequence.
    """
    query_ends = [0]
    query_lengths = []
    kv_ends = [0]
    kv_lengths = []
    run_firsts = []
    run_lengths = []
    for index in range(batch):
        offset = index * tokens
        for (start, end), neighbors in neighborhoods:
            query_ends.append(offset + end)
            query_lengths.append(end - start)
            kv_length = 0
            for first, last in neighbors:
                run_firsts.append(offset + first)
                run_lengths.append(last - first)
                kv_length += last - first
            kv_ends.append(kv_ends[-1] + kv_length)
            kv_lengths.append(kv_length)
    # Consecutive runs expand to the tokens each segment reads, in neighbour order.
    total = kv_ends[-1]
    lengths = torch.tensor(run_lengths, device=device)
    run_starts = torch.cumsum(lengths, 0) - lengths
    shift = torch.tensor(run_firsts, device=device) - run_starts
    kv_index = torch.arange(total, device=device) + shift.repeat_interleave(lengths, output_size=total)
    extra = {
        "cu_seqlens_q": torch.tensor(query_ends, dtype=torch.int32, device=device),
        "cu_seqlens_k": torch.tensor(kv_ends, dtype=torch.int32, device=device),
        "max_seqlen_q": max(query_lengths),
        "max_seqlen_k": max(kv_lengths),
    }
    return PackedLocalAttention(kv_index=kv_index, metadata=AttentionMetadata(extra=extra))


def packed_local_attention(
    attention: torch.nn.Module, neighborhoods: Neighborhoods, hidden_states: torch.Tensor
) -> PackedLocalAttention | None:
    """Plan one varlen call for all local-attention tiles, or decline when the backend cannot isolate tiles."""
    backend = attention.attn_backend
    if (
        backend is None
        or not backend.supports_multi_doc_packed_varlen()
        or hidden_states.device.type == "cpu"
        or hidden_states.dtype not in (torch.float16, torch.bfloat16)
        or torch.is_grad_enabled()
    ):
        return None
    batch, tokens = hidden_states.shape[:2]
    return plan_local_attention(neighborhoods, batch, tokens, hidden_states.device)
