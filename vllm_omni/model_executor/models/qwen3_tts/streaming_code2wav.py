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
class _RequestDecodeState:
    slot: int
    position: int = 0
    prefix_frames: int = 0


class StreamingCode2Wav:
    """Decode new frames without recomputing the suffix's left context.

    Slots belong to scheduler request IDs across chunk scheduling/preemption.
    ICL primes reference codes once, then appends generated frames just like
    xvec. Requests beyond the slot budget use the legacy decoder for their
    entire lifetime.
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
        decode_modes: tuple[str, ...] = ("icl", "xvec"),
    ) -> None:
        if not decode_modes or set(decode_modes) - {"icl", "xvec"}:
            raise ValueError("decode_modes must contain icl and/or xvec")
        self.decode_modes = decode_modes
        if num_slots <= 0:
            raise ValueError("decode_streaming_num_slots must be positive")
        self.decoder = StreamingCodecDecoder(decoder, num_slots)
        self.max_batch_size = max_batch_size or num_slots
        self.free_slots = deque(range(num_slots))
        self.request_states: dict[str, _RequestDecodeState] = {}
        self.fallback_ids: set[str] = set()
        self.graphs: dict[int, StreamingDecodeGraphs] = {}
        self.prefix_graphs: dict[int, StreamingDecodeGraphs] = {}
        self.graph_pool = torch.cuda.graph_pool_handle() if capture else None
        self.capture_batch_sizes: list[int] = []
        if capture:
            sizes = sorted({size for size in batch_sizes if 0 < size <= min(num_slots, self.max_batch_size)})
            self.capture_batch_sizes = sizes
            captured_frames = {min(size, self.decoder.max_frames) for size in frame_sizes if size > 0}
            for frames in sorted(captured_frames):
                self.graphs[frames] = StreamingDecodeGraphs(self.decoder, sizes, frames=frames, pool=self.graph_pool)
            if "icl" in self.decode_modes:
                frames = min(self.decoder.window, self.decoder.RING)
                self.prefix_graphs[frames] = StreamingDecodeGraphs(
                    self.decoder, sizes, frames=frames, pool=self.graph_pool, initial=True
                )

    def release(self, request_ids: set[str] | list[str]) -> None:
        for request_id in request_ids:
            state = self.request_states.pop(request_id, None)
            if state is not None:
                self.free_slots.append(state.slot)
            self.fallback_ids.discard(request_id)

    def _run_batch(
        self,
        codes: torch.Tensor,
        slots: list[int],
        positions: list[int],
        *,
        prefix: bool = False,
    ) -> torch.Tensor | None:
        """Run requests with the same frame count together, without prefix padding."""
        sd = self.decoder
        frames_total = codes.shape[-1]
        device = codes.device
        slot_tensor = index_to_device(slots, device, dtype=torch.int32)
        position_tensor = index_to_device(positions, device, dtype=torch.int32)
        pcm = None if prefix else torch.empty(len(codes), frames_total * sd.spf, device=device, dtype=torch.float32)
        values = codes
        offset = 0
        while offset < frames_total:
            remaining = frames_total - offset
            initial = prefix and offset == 0
            frames = min(remaining, sd.RING if initial else sd.max_frames)
            value = values[:, :, offset : offset + frames].transpose(1, 2).to(torch.int32).contiguous()
            graphs = self.prefix_graphs if initial else self.graphs
            if initial and frames not in self.prefix_graphs and self.capture_batch_sizes:
                self.prefix_graphs[frames] = StreamingDecodeGraphs(
                    sd, self.capture_batch_sizes, frames=frames, pool=self.graph_pool, initial=True
                )
            fn = graphs.get(frames)
            if fn is not None:
                wav = fn(value, slot_tensor, position_tensor + offset)
            elif initial:
                wav = sd(value, slot_tensor, position_tensor + offset, initial=True)
            else:
                wav = sd(value, slot_tensor, position_tensor + offset)
            if pcm is not None:
                pcm[:, offset * sd.spf : (offset + frames) * sd.spf].copy_(wav)
            offset += frames
        return pcm

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
        prefix_groups: dict[int, list[int]] = defaultdict(list)
        suffix_starts: dict[int, int] = {}
        fallback_rows: list[int] = []
        outputs = [codes.new_empty((1, 0), dtype=torch.float32) for _ in lengths]
        for row, (request_id, cache, length) in enumerate(zip(request_ids, caches, lengths, strict=True)):
            state = self.request_states.get(request_id)
            mode = "icl" if int(cache.get("prefix_frames", 0)) > 0 else "xvec"
            if mode not in self.decode_modes:
                fallback_rows.append(row)
                continue
            if cache.get("_is_dummy_run", False) or request_id in self.fallback_ids:
                fallback_rows.append(row)
                continue
            prefix = 0
            if state is None:
                if not self.free_slots:
                    self.fallback_ids.add(request_id)
                    fallback_rows.append(row)
                    continue
                prefix = int(cache.get("prefix_frames", 0))
                if prefix and not 0 < prefix < length:
                    raise ValueError("ICL first chunk must contain reference and generated frames")
                state = _RequestDecodeState(self.free_slots.popleft(), prefix_frames=prefix)
                self.request_states[request_id] = state
                if prefix:
                    prefix_groups[prefix].append(row)
            suffix_starts[row] = prefix
            length -= prefix
            frames = length
            # Padding is safe only on terminal chunks: no later call may
            # consume the resulting state. Trim PCM back to the real length.
            if terminal[row]:
                frames = next((size for size in sorted(self.graphs) if size >= length), length)
            groups[frames].append(row)

        for frames, rows in prefix_groups.items():
            for offset in range(0, len(rows), self.max_batch_size):
                part = rows[offset : offset + self.max_batch_size]
                records = [self.request_states[request_ids[row]] for row in part]
                self._run_batch(
                    torch.cat([codes[row : row + 1, :, :frames] for row in part], dim=0),
                    [record.slot for record in records],
                    [0] * len(part),
                    prefix=True,
                )
                for record in records:
                    record.position += frames

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
                records = [self.request_states[request_ids[row]] for row in part]
                has_prefix = any(suffix_starts[row] > 0 for row in part)
                if has_prefix:
                    pieces = []
                    for row in part:
                        start = suffix_starts[row]
                        value = codes[row : row + 1, :, start : start + frames]
                        if value.shape[-1] < frames:
                            value = torch.nn.functional.pad(value, (0, frames - value.shape[-1]))
                        pieces.append(value)
                    values = torch.cat(pieces, dim=0)
                else:
                    selected = index_to_device(part, codes.device)
                    values = codes.index_select(0, selected)[:, :, :frames]
                    if values.shape[-1] < frames:
                        values = torch.nn.functional.pad(values, (0, frames - values.shape[-1]))
                pcm = self._run_batch(
                    values,
                    [record.slot for record in records],
                    [record.position for record in records],
                )
                assert pcm is not None
                for index, (row, record) in enumerate(zip(part, records, strict=True)):
                    new_frames = lengths[row] - suffix_starts[row]
                    skip = (
                        int(bool(caches[row].get("skip_first_audio", False)))
                        if record.position == record.prefix_frames
                        else 0
                    )
                    outputs[row] = pcm[index : index + 1, skip * sd.spf : new_frames * sd.spf]
                    record.position += new_frames
        return outputs
