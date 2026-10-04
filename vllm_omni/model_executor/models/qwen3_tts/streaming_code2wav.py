# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-owned streaming codec state for an independent Code2Wav stage."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

import torch

from vllm_omni.utils.device_copy import index_to_device

from .tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Decoder
from .tokenizer_12hz.streaming_decoder import StreamingCodecDecoder, StreamingDecodeGraphs


@dataclass
class _Stream:
    cache: dict[str, Any]
    slot: int
    position: int = 0


class StreamingCode2Wav:
    """Decode new frames without recomputing the suffix's left context.

    Slots belong to scheduler request IDs across chunk scheduling/preemption.
    ICL requests retain the original anchored-reference decoder. New requests
    beyond the slot budget also retain that decoder for their entire lifetime.
    """

    def __init__(
        self,
        decoder: Qwen3TTSTokenizerV2Decoder,
        *,
        num_slots: int,
        max_batch_size: int,
        batch_sizes: list[int],
        frame_sizes: list[int],
        capture: bool,
    ) -> None:
        if num_slots <= 0:
            raise ValueError("decode_streaming_num_slots must be positive")
        self.decoder = StreamingCodecDecoder(decoder, num_slots)
        self.max_batch_size = max_batch_size or num_slots
        self.free_slots = deque(range(num_slots))
        self.streams: dict[str, _Stream] = {}
        self.fallback_ids: set[str] = set()
        self.graphs: dict[int, StreamingDecodeGraphs] = {}
        if capture:
            sizes = sorted({size for size in batch_sizes if 0 < size <= min(num_slots, self.max_batch_size)})
            for frames in sorted({min(size, self.decoder.max_frames) for size in frame_sizes if size > 0}):
                self.graphs[frames] = StreamingDecodeGraphs(self.decoder, sizes, frames=frames)

    def release(self, request_ids: set[str] | list[str]) -> None:
        for request_id in request_ids:
            stream = self.streams.pop(request_id, None)
            if stream is not None:
                self.free_slots.append(stream.slot)
            self.fallback_ids.discard(request_id)

    @torch.no_grad()
    def decode(
        self,
        codes: torch.Tensor,
        lengths: list[int],
        *,
        request_ids: list[str],
        caches: list[dict[str, Any]],
        terminal: list[bool],
        legacy_decoder: Qwen3TTSTokenizerV2Decoder,
        chunk_size: int,
        left_context_size: int,
    ) -> list[torch.Tensor]:
        groups: dict[int, list[int]] = defaultdict(list)
        fallback_rows: list[int] = []
        outputs = [codes.new_empty((1, 0), dtype=torch.float32) for _ in lengths]
        for row, (request_id, cache, length) in enumerate(zip(request_ids, caches, lengths, strict=True)):
            stream = self.streams.get(request_id)
            if stream is not None and stream.cache is not cache:
                self.release([request_id])
                stream = None
            # ICL keeps a reference anchor outside the rolling attention
            # window. Ordinary sliding-window state cannot replace it.
            if (
                cache.get("_is_dummy_run", False)
                or int(cache.get("prefix_frames", 0)) > 0
                or request_id in self.fallback_ids
            ):
                fallback_rows.append(row)
                continue
            if stream is None:
                if not self.free_slots:
                    self.fallback_ids.add(request_id)
                    fallback_rows.append(row)
                    continue
                stream = _Stream(cache, self.free_slots.popleft())
                self.streams[request_id] = stream
            frames = length
            # Padding is safe only on terminal chunks: no later call may
            # consume the resulting state. Trim PCM back to the real length.
            if terminal[row]:
                frames = next((size for size in sorted(self.graphs) if size >= length), length)
            groups[frames].append(row)

        if fallback_rows:
            selected = index_to_device(fallback_rows, codes.device)
            fallback = legacy_decoder.batched_chunked_decode(
                codes.index_select(0, selected),
                [lengths[row] for row in fallback_rows],
                caches=[caches[row] for row in fallback_rows],
                chunk_size=chunk_size,
                left_context_size=left_context_size,
                max_batch_size=self.max_batch_size,
            )
            for row, wav in zip(fallback_rows, fallback, strict=True):
                outputs[row] = wav

        sd = self.decoder
        for frames, rows in groups.items():
            for offset in range(0, len(rows), self.max_batch_size):
                part = rows[offset : offset + self.max_batch_size]
                records = [self.streams[request_ids[row]] for row in part]
                selected = index_to_device(part, codes.device)
                value = codes.index_select(0, selected)[:, :, :frames].transpose(1, 2).to(torch.int32).contiguous()
                if value.shape[1] < frames:
                    value = torch.nn.functional.pad(value, (0, 0, 0, frames - value.shape[1]))
                slots = index_to_device([record.slot for record in records], codes.device, dtype=torch.int32)
                positions = index_to_device([record.position for record in records], codes.device, dtype=torch.int32)
                # Own the output before another batch replays a graph with
                # static PCM storage. Copy slices directly into this buffer.
                pcm = torch.empty(len(part), frames * sd.spf, device=codes.device, dtype=torch.float32)
                for start in range(0, frames, sd.max_frames):
                    end = min(frames, start + sd.max_frames)
                    fn = self.graphs.get(end - start, sd)
                    pcm[:, start * sd.spf : end * sd.spf].copy_(fn(value[:, start:end], slots, positions + start))
                for index, (row, record) in enumerate(zip(part, records, strict=True)):
                    skip = int(bool(caches[row].get("skip_first_audio", False))) if record.position == 0 else 0
                    outputs[row] = pcm[index : index + 1, skip * sd.spf : lengths[row] * sd.spf]
                    record.position += lengths[row]
        return outputs
