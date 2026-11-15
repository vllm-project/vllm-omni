# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Complete ZAC decode graphs: RVQ → neural decoder → iSTFT/FIR → resampling.

Buckets are keyed by (num_seqs, total RVQ tokens), with one global slab tail.
Per-request lengths govern state commits and audio coverage; final remains a
device input. Eager and graph execution share the packed neural/synthesis core.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

if TYPE_CHECKING:
    from .codec import ZACDecoderNetwork

logger = init_logger(__name__)


class _ReplayGraph(Protocol):
    def replay(self) -> None: ...


@dataclass
class _PackedBucket:
    graph: _ReplayGraph
    codes: torch.Tensor
    slots: torch.Tensor
    offsets: torch.Tensor
    final: torch.Tensor
    output: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


class ZACCodecGraphWrapper:
    """Complete pack→pack capture, keyed only by (B, sum RVQ T).

    All buckets share Codec persistent state. Requests absent from a bucket use
    distinct non-leaseable slots, zero offsets and false final. Real tokens are
    contiguous at the start of codes; only the slab tail is padded. Returned
    output buffers alias graph storage; the caller must copy before reuse.
    """

    def __init__(
        self,
        network: ZACDecoderNetwork,
        capacity: int,
        max_chunk: int,
        capture_keys: Sequence[tuple[int, int]],
    ) -> None:
        from .codec import ZACPackedCodec

        self.capacity = capacity
        self.capture_keys = sorted(set(capture_keys))
        self.batch_sizes = sorted({b for b, _ in self.capture_keys}) or [capacity]
        scratch = max(self.batch_sizes) if self.capture_keys else 0
        self.codec = ZACPackedCodec(network, capacity + scratch, max_chunk, capacity * max_chunk)
        self.buckets: dict[tuple[int, int], _PackedBucket] = {}
        self.replay_count = 0
        self.eager_count = 0

    @torch.inference_mode()
    def capture(self) -> None:
        device = self.codec.state.frames.device
        for batch, tokens in sorted(self.capture_keys, reverse=True):
            codes = torch.zeros(tokens, 10, dtype=torch.long, device=device)
            slots = torch.arange(self.capacity, self.capacity + batch, device=device)
            offsets = torch.zeros(batch + 1, dtype=torch.long, device=device)
            final = torch.zeros(batch, dtype=torch.bool, device=device)
            result = None

            def execute():
                nonlocal result
                result = self.codec(codes, slots, offsets, final)

            for _ in range(3):
                execute()
            current_omni_platform.synchronize()
            graph = current_omni_platform.capture_graph(execute)
            if graph is None:
                raise RuntimeError("Packed Codec requires a real device graph")
            assert result is not None
            self.buckets[batch, tokens] = _PackedBucket(graph, codes, slots, offsets, final, result)
            logger.info("Captured packed Codec graph num_seqs=%d sigma_T=%d", batch, tokens)

    @torch.inference_mode()
    def __call__(
        self, codes: torch.Tensor, slots: torch.Tensor, lengths: torch.Tensor, final: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = next(n for n in self.batch_sizes if n >= slots.numel())
        tokens = next((t for b, t in sorted(self.buckets) if b == batch and t >= max(1, codes.shape[0])), None)
        bucket = self.buckets.get((batch, tokens))
        if bucket is None:
            self.eager_count += 1
            offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
            return self.codec(codes, slots, offsets, final)
        n = slots.numel()
        bucket.codes.zero_()
        bucket.codes[: codes.shape[0]].copy_(codes)
        bucket.slots.copy_(torch.arange(self.capacity, self.capacity + batch, device=slots.device))
        bucket.slots[:n].copy_(slots)
        bucket.offsets.zero_()
        bucket.offsets[1 : n + 1].copy_(lengths.cumsum(0))
        bucket.offsets[n + 1 :].copy_(bucket.offsets[n].expand(batch - n))
        bucket.final.zero_()
        bucket.final[:n].copy_(final)
        bucket.graph.replay()
        self.replay_count += 1
        if self.replay_count == 1:
            logger.info("Packed Codec replay active num_seqs=%d sigma_T=%d", batch, tokens)
        audio, offsets, valid = bucket.output
        return audio, offsets[: n + 1], valid[:n]
