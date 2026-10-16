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
    cache: dict[str, Any]
    slot: int
    position: int = 0
    prefix_frames: int = 0


@dataclass
class _IclWork:
    row: int
    state: _RequestDecodeState
    cache: dict[str, Any]
    generated: torch.Tensor
    prefix_codes: torch.Tensor | None = None


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
        if capture:
            sizes = sorted({size for size in batch_sizes if 0 < size <= min(num_slots, self.max_batch_size)})
            captured_frames = {min(size, self.decoder.max_frames) for size in frame_sizes if size > 0}
            if "icl" in self.decode_modes:
                captured_frames.update((1, 2, 4, 8, 16, 32, self.decoder.max_frames))
            for frames in sorted(captured_frames):
                self.graphs[frames] = StreamingDecodeGraphs(self.decoder, sizes, frames=frames)

    def release(self, request_ids: set[str] | list[str]) -> None:
        for request_id in request_ids:
            state = self.request_states.pop(request_id, None)
            if state is not None:
                self.free_slots.append(state.slot)
            self.fallback_ids.discard(request_id)

    def _run_batch(
        self,
        codes: list[torch.Tensor],
        slots: list[int],
        positions: list[int],
        *,
        prefix: bool = False,
    ) -> torch.Tensor | None:
        """Run requests with the same frame count together, without prefix padding."""
        sd = self.decoder
        frames_total = codes[0].shape[-1]
        if any(value.shape[-1] != frames_total for value in codes):
            raise ValueError("Streaming batch requires equal frame counts")
        device = codes[0].device
        slot_tensor = index_to_device(slots, device, dtype=torch.int32)
        position_tensor = index_to_device(positions, device, dtype=torch.int32)
        pcm = None if prefix else torch.empty(len(codes), frames_total * sd.spf, device=device, dtype=torch.float32)
        values = torch.cat(codes, dim=0)
        offset = 0
        while offset < frames_total:
            remaining = frames_total - offset
            if prefix and self.graphs:
                frames = next((size for size in sorted(self.graphs, reverse=True) if size <= remaining), 1)
            else:
                frames = min(remaining, sd.max_frames)
            value = values[:, :, offset : offset + frames].transpose(1, 2).to(torch.int32).contiguous()
            fn = self.graphs.get(frames, sd)
            wav = fn(value, slot_tensor, position_tensor + offset)
            if pcm is not None:
                pcm[:, offset * sd.spf : (offset + frames) * sd.spf].copy_(wav)
            offset += frames
        return pcm

    def _decode_icl_batch(self, work: list[_IclWork], outputs: list[torch.Tensor]) -> None:
        """Batch prefix initialization and equal-length suffix work separately."""
        sd = self.decoder
        prefix_groups: dict[int, list[_IclWork]] = defaultdict(list)
        for item in work:
            if item.prefix_codes is not None:
                prefix_groups[item.state.prefix_frames].append(item)
        for frames, items in prefix_groups.items():
            for start in range(0, len(items), self.max_batch_size):
                part = items[start : start + self.max_batch_size]
                self._run_batch(
                    [item.prefix_codes for item in part if item.prefix_codes is not None],
                    [item.state.slot for item in part],
                    [0] * len(part),
                    prefix=True,
                )

        suffix_groups: dict[int, list[tuple[_IclWork, torch.Tensor, int]]] = defaultdict(list)
        for item in work:
            state = item.state
            to_decode = item.generated
            position = state.prefix_frames + state.position
            suffix_groups[to_decode.shape[-1]].append((item, to_decode, position))

        for frames, items in suffix_groups.items():
            for start in range(0, len(items), self.max_batch_size):
                part = items[start : start + self.max_batch_size]
                pcm = self._run_batch(
                    [codes for _, codes, _ in part],
                    [item.state.slot for item, _, _ in part],
                    [position for _, _, position in part],
                )
                assert pcm is not None
                for index, (item, _, _) in enumerate(part):
                    state = item.state
                    new_frames = item.generated.shape[-1]
                    result = pcm[index : index + 1, -new_frames * sd.spf :]
                    if state.position == 0 and item.cache.get("skip_first_audio", False):
                        result = result[:, sd.spf :]
                    outputs[item.row] = result
                    state.position += new_frames

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
        icl_work: list[_IclWork] = []
        fallback_rows: list[int] = []
        outputs = [codes.new_empty((1, 0), dtype=torch.float32) for _ in lengths]
        for row, (request_id, cache, length) in enumerate(zip(request_ids, caches, lengths, strict=True)):
            state = self.request_states.get(request_id)
            if state is not None and state.cache is not cache:
                self.release([request_id])
                state = None
            mode = "icl" if int(cache.get("prefix_frames", 0)) > 0 else "xvec"
            if mode not in self.decode_modes:
                fallback_rows.append(row)
                continue
            # Prime reference codes once, then append generated frames to the same state.
            if (
                int(cache.get("prefix_frames", 0)) > 0
                and not cache.get("_is_dummy_run", False)
                and request_id not in self.fallback_ids
            ):
                row_codes = codes[row : row + 1, :, :length]
                if state is None:
                    if not self.free_slots:
                        self.fallback_ids.add(request_id)
                        fallback_rows.append(row)
                        continue
                    prefix_frames = int(cache["prefix_frames"])
                    if not 0 < prefix_frames < length:
                        raise ValueError("ICL first chunk must contain reference and generated frames")
                    state = _RequestDecodeState(cache, self.free_slots.popleft(), prefix_frames=prefix_frames)
                    self.request_states[request_id] = state
                    icl_work.append(
                        _IclWork(row, state, cache, row_codes[:, :, prefix_frames:], row_codes[:, :, :prefix_frames])
                    )
                else:
                    icl_work.append(_IclWork(row, state, cache, row_codes))
                continue
            if (
                cache.get("_is_dummy_run", False)
                or int(cache.get("prefix_frames", 0)) > 0
                or request_id in self.fallback_ids
            ):
                fallback_rows.append(row)
                continue
            if state is None:
                if not self.free_slots:
                    self.fallback_ids.add(request_id)
                    fallback_rows.append(row)
                    continue
                state = _RequestDecodeState(cache, self.free_slots.popleft())
                self.request_states[request_id] = state
            frames = length
            # Padding is safe only on terminal chunks: no later call may
            # consume the resulting state. Trim PCM back to the real length.
            if terminal[row]:
                frames = next((size for size in sorted(self.graphs) if size >= length), length)
            groups[frames].append(row)

        if icl_work:
            self._decode_icl_batch(icl_work, outputs)

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
                selected = index_to_device(part, codes.device)
                value = codes.index_select(0, selected)[:, :, :frames]
                if value.shape[-1] < frames:
                    value = torch.nn.functional.pad(value, (0, frames - value.shape[-1]))
                pcm = self._run_batch(
                    [value[index : index + 1] for index in range(len(part))],
                    [record.slot for record in records],
                    [record.position for record in records],
                )
                assert pcm is not None
                for index, (row, record) in enumerate(zip(part, records, strict=True)):
                    skip = int(bool(caches[row].get("skip_first_audio", False))) if record.position == 0 else 0
                    outputs[row] = pcm[index : index + 1, skip * sd.spf : lengths[row] * sd.spf]
                    record.position += lengths[row]
        return outputs
