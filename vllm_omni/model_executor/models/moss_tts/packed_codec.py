# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Packed streaming decode for MOSS-Audio-Tokenizer-v2.

A batched streaming decode pads every stream to one chunk length, so a call
mixing first chunks (1 frame), steady chunks and short tails either pads the
short rows or runs once per length. The decoder is a stack of causal
transformers and patch upsamplers whose only cross-token operation is
attention, so streams can instead be packed along time: one (1, N) sequence of
per-stream segments. Upsampling keeps a segment contiguous (its tokens at a
stage are its frames' tokens, in order); attention is per segment
(``slot_ring_flash_attention_packed``); every other operator is per token.
One graph replay then decodes any mix of chunk lengths for the cost of the
frames it holds.

The packed sequence of a graph has a fixed frame count; the frames behind the
real segments form a padding segment on a scratch state slot whose state is
never advanced. Output is gathered per segment, channel-major: segment ``s``
is ``out[2 * 3840 * start[s] :][: 2 * 3840 * frames[s]].view(2, -1)`` for a
stereo codec.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import CUDAGraph
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu import buffer_utils

logger = init_logger(__name__)


@triton.jit
def _gather_segments(
    src_ptr,
    seg_ptr,
    start_ptr,
    out_ptr,
    channels: tl.constexpr,
    samples_per_frame: tl.constexpr,
    block: tl.constexpr,
):
    """Frame ``f`` of segment ``s``: interleaved ``src[f]`` -> channel-major ``out`` of ``s``."""
    frame = tl.program_id(0)
    channel = tl.program_id(1)
    segment = tl.load(seg_ptr + frame)
    first = tl.load(start_ptr + segment)
    seg_frames = tl.load(start_ptr + segment + 1) - first
    local = frame - first
    dst = (first * channels + channel * seg_frames + local).to(tl.int64) * samples_per_frame
    src = frame.to(tl.int64) * samples_per_frame * channels + channel
    for i in range(0, samples_per_frame, block):
        idx = i + tl.arange(0, block)
        mask = idx < samples_per_frame
        value = tl.load(src_ptr + src + idx * channels, mask)
        tl.store(out_ptr + dst + idx, value.to(tl.float32), mask)


@torch.library.custom_op("vllm_omni::moss_codec_gather_segments", mutates_args=())
def gather_segments(
    audio: torch.Tensor, frame_segments: torch.Tensor, bounds: torch.Tensor, channels: int, samples_per_frame: int
) -> torch.Tensor:
    """fp32 per-segment channel-major copy of interleaved ``audio`` (N * spf * channels,).

    ``bounds`` is (S + 1,) segment frame boundaries ``[0, cumsum(frames)]``.
    """
    frames = frame_segments.shape[0]
    out = torch.empty(frames * samples_per_frame * channels, device=audio.device, dtype=torch.float32)
    _gather_segments[(frames, channels)](
        audio, frame_segments, bounds, out, channels, samples_per_frame, 1024, num_warps=4
    )
    return out


@gather_segments.register_fake
def _(
    audio: torch.Tensor, frame_segments: torch.Tensor, bounds: torch.Tensor, channels: int, samples_per_frame: int
) -> torch.Tensor:
    return audio.new_empty((frame_segments.shape[0] * samples_per_frame * channels,), dtype=torch.float32)


def decode_packed_tensors(codec: nn.Module, codes: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
    """Decode packed segments; returns the fp32 per-segment output (see module docstring).

    ``codes`` is (n_vq, 1, N). ``meta`` is (3, S) int64: frames, state slot and
    validity of each segment (frames sum to N; the slots are distinct).
    """
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
        StreamingExecutionContext,
        StreamingModule,
    )

    frames, slots, valid = meta[0], meta[1], meta[2] != 0
    bounds = torch.nn.functional.pad(torch.cumsum(frames, 0), (1, 0))
    n = codes.shape[2]
    frame_segments = torch.searchsorted(bounds[1:], torch.arange(n, device=codes.device), right=True).clamp_(
        max=frames.shape[0] - 1
    )
    context = StreamingExecutionContext(
        state_slot_ids=slots,
        valid_rows=valid,
        segment_frames=frames,
        segment_starts=bounds[:-1],
        frame_segments=frame_segments,
    )
    lengths = torch.full((1,), n, device=codes.device, dtype=torch.long)
    with torch.autocast(device_type=codes.device.type, dtype=torch.bfloat16):
        d = codec.quantizer.decode_codes(codes)
        for module in codec.decoder:
            if isinstance(module, StreamingModule):
                d, lengths = module(d, lengths, execution_context=context)
            else:
                d, lengths = module(d, lengths)
    channels = int(codec.number_channels) if codec.enable_channel_interleave else 1
    return gather_segments(d.reshape(-1), frame_segments, bounds, channels, int(codec.downsample_rate))


# Packed attention launch (block_m, block_n, num_warps, num_stages) per tokens
# per code frame, tuned on H200 for 15-frame chunks (12.5 Hz stage 39.6 ->
# 33.3 us, 25 Hz stage 50.0 -> 39.9 us per launch; faster stages already at
# their best tile). Rounding-level differences from the batched tile.
_TUNED_PACKED_LAUNCH = {1: (16, 32, 2, 2), 2: (32, 64, 4, 2)}


def enable_packed_decode(codec: nn.Module, chunk_frames: int) -> int:
    """Configure the decoder transformers for packed decode; returns their number.

    Each transformer gets its tokens per code frame and the attention tile a
    steady ``chunk_frames`` batched chunk uses at that rate (the per-row
    arithmetic of packed attention then matches the batched launch).
    Known transformer rates use the tuned launch geometry.
    """
    from vllm_omni.model_executor.models.moss_tts.audio_tokenizer_v2 import (
        MossAudioTokenizerMultiheadAttention,
        MossAudioTokenizerProjectedTransformer,
    )

    count = 0
    for module in codec.decoder:
        if not isinstance(module, MossAudioTokenizerProjectedTransformer):
            continue
        factor = int(getattr(module, "tokens_per_input_frame", 1))
        block_m = 16 if chunk_frames * factor <= 32 else 64
        launch = (0, 0, 0)
        if factor in _TUNED_PACKED_LAUNCH:
            block_m, *launch = _TUNED_PACKED_LAUNCH[factor]
        module.transformer._packed_factor = factor
        module.transformer._packed_block_m = block_m
        for attention in module.modules():
            if isinstance(attention, MossAudioTokenizerMultiheadAttention):
                attention._packed_block_m = block_m
                attention._packed_launch = tuple(launch)
        count += 1
    return count


@dataclass
class _PackedGraph:
    graph: CUDAGraph
    codes: torch.Tensor  # (n_vq, 1, N)
    meta: torch.Tensor  # (3, S)
    audio: torch.Tensor  # (N * samples_per_frame * channels,)


class CUDAGraphPackedDecoderWrapper:
    """Packed decode graphs keyed by total frame count N (padded up to a bucket).

    A call holds at most ``max_segments - 1`` streams: segment slots past the
    real streams (including the padding segment) are distinct scratch slots
    ``[scratch_base, scratch_base + max_segments)``, never advanced.
    """

    def __init__(
        self,
        codec: nn.Module,
        *,
        frame_sizes: list[int],
        max_segments: int,
        scratch_base: int,
        num_quantizers: int,
        vllm_config: VllmConfig,
    ) -> None:
        self.codec = codec
        self._warmups = max(1, vllm_config.compilation_config.cudagraph_num_of_warmups)
        # No one-frame bucket: a one-frame call replays the two-frame graph.
        self.frame_sizes = sorted({int(size) for size in frame_sizes if int(size) >= 2})
        self.max_segments = int(max_segments)
        self.scratch_base = int(scratch_base)
        self.num_quantizers = int(num_quantizers)
        self.channels = int(codec.number_channels) if codec.enable_channel_interleave else 1
        self.samples_per_frame = int(codec.downsample_rate)
        self.graphs: dict[int, _PackedGraph] = {}
        self._pool = None
        self._stream: torch.cuda.Stream | None = None
        self._host: list[tuple[torch.Tensor, np.ndarray, torch.cuda.Event, bool]] = []
        self._host_index = 0
        # This wrapper owns CUDA Graph capture/replay; the decode runs uncompiled.
        self._decode = functools.partial(decode_packed_tensors, codec)

    @property
    def max_frames(self) -> int:
        return max(self.frame_sizes, default=0)

    def _scratch_meta(self, frames: int, device: torch.device) -> torch.Tensor:
        meta = torch.zeros((3, self.max_segments), dtype=torch.long, device=device)
        meta[0, 0] = frames
        meta[1] = self.scratch_base + torch.arange(self.max_segments, device=device)
        return meta

    @torch.no_grad()
    def warmup(self, device: torch.device) -> None:
        if device.type != "cuda" or not self.frame_sizes:
            return
        start_s = time.perf_counter()
        self._stream = torch.cuda.Stream(device=device)
        self._pool = current_platform.get_global_graph_pool()
        self._capture(device)
        logger.info(
            "MOSS-TTS packed codec CUDA graphs: %d frame buckets %s, %d segments, captured in %.1f s",  # noqa: E501
            len(self.graphs),
            sorted(self.graphs),
            self.max_segments,
            time.perf_counter() - start_s,
        )

    def _capture(self, device: torch.device) -> None:
        for _ in range(buffer_utils._DEFAULT_MAX_CONCURRENCY):
            host = torch.empty((3, self.max_segments), dtype=torch.long, pin_memory=True)
            self._host.append((host, host.numpy(), torch.cuda.Event(), False))
        for frames in sorted(self.frame_sizes, reverse=True):
            codes = torch.zeros((self.num_quantizers, 1, frames), dtype=torch.long, device=device)
            meta = self._scratch_meta(frames, device)
            self._stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self._stream):
                for _ in range(self._warmups):
                    self._decode(codes, meta)
            torch.cuda.current_stream().wait_stream(self._stream)
            torch.accelerator.synchronize(device)
            graph = CUDAGraph()
            with torch.cuda.graph(graph, pool=self._pool, stream=self._stream, capture_error_mode="thread_local"):
                audio = self._decode(codes, meta)
            self.graphs[frames] = _PackedGraph(graph, codes, meta, audio)
        self.codec.reset_decoder_state_slots(
            self.scratch_base + torch.arange(self.max_segments, dtype=torch.long, device=device)
        )

    def decode(self, slots: list[int], codes: list[torch.Tensor]) -> tuple[torch.Tensor, list[int], list[int]]:
        """Decode one packed call; returns (output, segment starts, frames) of the real streams.

        ``codes[i]`` is (n_vq, frames_i) on the device; ``len(slots) < max_segments``.
        """
        frames = [int(c.shape[1]) for c in codes]
        total = sum(frames)
        bucket = next(size for size in self.frame_sizes if size >= total)
        entry = self.graphs[bucket]
        host, view, event, recorded = self._host[self._host_index]
        if recorded and not event.query():
            event.synchronize()
        n = len(slots)
        view[0, :n] = frames
        view[0, n] = bucket - total
        view[0, n + 1 :] = 0
        view[1, :n] = slots
        view[1, n:] = self.scratch_base + np.arange(n, self.max_segments)
        view[2, :n] = 1
        view[2, n:] = 0
        entry.meta.copy_(host, non_blocking=True)
        event.record(torch.cuda.current_stream())
        self._host[self._host_index] = (host, view, event, True)
        self._host_index = (self._host_index + 1) % len(self._host)
        # Frames past ``total`` keep earlier valid codes: they feed only the padding segment.
        entry.codes[:, 0, :total].copy_(torch.cat(codes, dim=1) if len(codes) > 1 else codes[0])
        entry.graph.replay()
        starts = np.concatenate(([0], np.cumsum(frames)[:-1])).tolist()
        return entry.audio[: total * self.samples_per_frame * self.channels], starts, frames


__all__ = [
    "CUDAGraphPackedDecoderWrapper",
    "decode_packed_tensors",
    "enable_packed_decode",
    "gather_segments",
]
