# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import time
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.model_executor.models.minicpmo_4_5.whole_euler_ops import (
    euler_cfg_step,
    fused_euler_supported,
    stage_estimator_input,
)
from vllm_omni.platforms import current_omni_platform
from vllm_omni.utils.device_copy import index_to_device, to_device_nonblocking

logger = init_logger(__name__)


def _euler_step(
    cur_x: torch.Tensor,
    estimate: torch.Tensor,
    dt: float,
    inference_cfg_rate: float,
    batch_size: int,
) -> torch.Tensor:
    """One Euler step of the CFG-guided flow: ``cur_x + dt * ((1 + cfg) * cond - cfg * uncond)``."""
    conditional, unconditional = estimate.split(batch_size, dim=0)
    velocity = (1.0 + inference_cfg_rate) * conditional - inference_cfg_rate * unconditional
    return cur_x + dt * velocity


def _euler_timeline(n_timesteps: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, list[float]]:
    """The cosine Euler timeline and its step sizes, read to the host once."""
    t = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)
    timeline = 1 - torch.cos(t * 0.5 * torch.pi)
    return timeline, (timeline[1:] - timeline[:-1]).tolist()


def codec_frame_range(value, *, name: str) -> range:
    """An inclusive ``[first, last]`` codec-frame range from the connector extra (empty: off)."""
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return range(0)
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"MiniCPM-o {name} must be [first, last] codec frames, got {value!r}")
    first, last = (int(v) for v in value)
    if first < 1 or last < first:
        raise ValueError(f"MiniCPM-o {name} must satisfy 1 <= first <= last, got {value!r}")
    return range(first, last + 1)


def empty_hift_outputs(speech_feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """HiFT's ``(speech, source)`` for a zero-width mel, without touching the device.

    HiFT emits a fixed number of samples per mel frame (480 at 24 kHz), so no
    frames means no samples. Running it instead fails in its first convolution,
    which cannot pad an empty input up to its kernel. Zero-element tensors
    allocate nothing and launch no kernel.
    """
    batch_size = int(speech_feat.shape[0])
    return speech_feat.new_empty((batch_size, 0)), speech_feat.new_empty((batch_size, 1, 0))


class HiFTGraphWrapper:
    def __init__(self, token2wav, connector_config, capture_batch_sizes, max_serial_batch: int | None = None):
        self.decode_fn = token2wav.hift.inference
        self.graph_fn = token2wav.hift._inference_pre_istft
        self.finalize_fn = token2wav.hift._finalize_decode
        self.codec_chunk_frames = connector_config["codec_chunk_frames"]
        self.initial_codec_chunk_frames = int(connector_config.get("initial_codec_chunk_frames", 0))
        self.codec_left_context_frames = connector_config["codec_left_context_frames"]
        lookahead_layer = getattr(token2wav.flow.encoder, "pre_lookahead_layer", None)
        pre_lookahead_len = getattr(lookahead_layer, "pre_lookahead_len", None)
        self.pre_lookahead_len = int(pre_lookahead_len) if pre_lookahead_len is not None else 3
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.mel_frames = int(token2wav.hift.conv_pre.in_channels)
        self.flow_upsample_rate = int(getattr(token2wav.flow, "token_mel_ratio", 2))
        self.extra_codec_chunk_frames = [int(c) for c in connector_config.get("hift_graph_codec_chunk_frames") or ()]
        self.capture_bucket_size, self.capture_source_cache_len = self.derive_capture_bucket_size()
        # The only (mel_frames, cache_source_len) shapes a graph is ever
        # captured for. Duplex first chunks carry arbitrary mel widths that are
        # seen once, so capturing one would stall every stream on the device
        # for a graph that pays for itself once; they run eager (``replay``).
        self._legit_shapes = set(zip(self.capture_bucket_size, self.capture_source_cache_len, strict=True))
        # Exact-shape graphs (``hift_graph_first_chunk_frames`` /
        # ``hift_graph_continuation_frames``, default off): the vocoder shapes
        # outside the buckets -- a stream's short first chunk, a merged backlog
        # continuation -- each get a graph of their own at
        # ``hift_graph_exact_batch_sizes``, captured with the buckets and
        # replayed without padding. The replay runs the kernels eager HiFT runs
        # for that shape, so it changes no output; other batch sizes of these
        # shapes still run eager (never padded, never captured lazily).
        self.first_chunk_frames = codec_frame_range(
            connector_config.get("hift_graph_first_chunk_frames"), name="hift_graph_first_chunk_frames"
        )
        self.continuation_frames = codec_frame_range(
            connector_config.get("hift_graph_continuation_frames"), name="hift_graph_continuation_frames"
        )
        self.exact_batch_sizes = sorted(
            {int(b) for b in connector_config.get("hift_graph_exact_batch_sizes") or (1,) if int(b) > 0}
        )
        self.exact_shapes = self.derive_exact_shapes()
        self._exact_keys = frozenset(
            (batch_size, mel_frames, cache_len)
            for batch_size in self.exact_batch_sizes
            for mel_frames, cache_len in self.exact_shapes
        )
        self.capture_batch_sizes = capture_batch_sizes
        self.graph: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.static_speech_inputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_magnitude_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_phase_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_cache_source_inputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_cache_source_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        parameter = next(token2wav.hift.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.max_lazy_graphs = int(connector_config.get("hift_max_lazy_graphs", 8))
        if self.max_lazy_graphs < 0:
            raise ValueError("MiniCPM-o hift_max_lazy_graphs must be >= 0")
        self.lazy_graph_count = 0
        self.max_serial_batch = 4 if max_serial_batch is None else int(max_serial_batch)

    def _chunk_mel_frames(self, codec_frames: int) -> int:
        return (int(codec_frames) + self.codec_left_context_frames - self.pre_lookahead_len) * self.flow_upsample_rate

    def derive_capture_bucket_size(self):
        chunk_mel_frames = self._chunk_mel_frames(self.codec_chunk_frames)

        frames = [chunk_mel_frames, chunk_mel_frames + self.mel_cache_len]
        cache_lengths = [0, self.source_cache_len]
        if self.initial_codec_chunk_frames and self.initial_codec_chunk_frames != self.codec_chunk_frames:
            first_frames = self._chunk_mel_frames(self.initial_codec_chunk_frames)
            if first_frames <= self.mel_cache_len:
                raise ValueError("MiniCPM-o initial codec chunk must emit audio beyond the HiFT mel cache")
            frames.append(first_frames)
            cache_lengths.append(0)
        # Extra codec chunk sizes that also stream as continuations. A
        # full-duplex unit is one ``initial_codec_chunk_frames``-sized chunk
        # after another, so its steady vocoder call (mel cache + source
        # cache) is outside the buckets above and would run eager every unit.
        shapes = list(zip(frames, cache_lengths, strict=True))
        for codec_frames in self.extra_codec_chunk_frames:
            mel_frames = self._chunk_mel_frames(codec_frames)
            if mel_frames <= self.mel_cache_len:
                raise ValueError("MiniCPM-o hift_graph_codec_chunk_frames must emit audio beyond the HiFT mel cache")
            for shape in ((mel_frames, 0), (mel_frames + self.mel_cache_len, self.source_cache_len)):
                if shape not in shapes:
                    shapes.append(shape)
        return [shape[0] for shape in shapes], [shape[1] for shape in shapes]

    def derive_exact_shapes(self) -> list[tuple[int, int]]:
        """``(mel_frames, cache_source_len)`` of the exact-shape graphs, outside the buckets.

        A first chunk of ``f`` codec frames vocodes ``_chunk_mel_frames(f)``
        mel frames with no source cache; a continuation also carries the mel
        cache in front and the source cache.
        """
        buckets = set(zip(self.capture_bucket_size, self.capture_source_cache_len, strict=True))
        shapes: list[tuple[int, int]] = []
        candidates = [(self._chunk_mel_frames(f), 0) for f in getattr(self, "first_chunk_frames", ())]
        candidates += [
            (self._chunk_mel_frames(f) + self.mel_cache_len, self.source_cache_len)
            for f in getattr(self, "continuation_frames", ())
        ]
        for mel_frames, cache_len in candidates:
            shape = (int(mel_frames), int(cache_len))
            if shape[0] > 0 and shape not in buckets and shape not in shapes:
                shapes.append(shape)
        return shapes

    def capture(self):
        for batch_size in self.capture_batch_sizes:
            for mel_frames, source_cache_len in zip(
                self.capture_bucket_size,
                self.capture_source_cache_len,
                strict=True,
            ):
                self._capture(batch_size, mel_frames, source_cache_len)
        self.capture_exact()

    def capture_exact(self) -> int:
        """Capture the exact-shape graphs (``derive_exact_shapes``); returns how many are new."""
        shapes = getattr(self, "exact_shapes", ())
        batch_sizes = getattr(self, "exact_batch_sizes", ())
        if not shapes or not batch_sizes:
            return 0
        before = len(self.graph)
        started = time.perf_counter()
        memory_before = _memory_snapshot(self.device)
        used_before = _device_used_bytes(self.device)
        for batch_size in batch_sizes:
            for mel_frames, source_cache_len in shapes:
                key = (batch_size, mel_frames, source_cache_len)
                self._capture(*key)
                if key in self.graph:
                    # Build this frame count's ISTFT envelope now: the first build
                    # reads a value back to the host (``_istft_without_host_sync``).
                    self.finalize_fn(self.static_magnitude_outputs[key][:1], self.static_phase_outputs[key][:1])
        captured = len(self.graph) - before
        used_after = _device_used_bytes(self.device)
        logger.info(
            "Captured %d exact-shape HiFT CUDA Graphs (batch sizes %s, %d shapes) in %.1f s%s, device memory %s",
            captured,
            list(batch_sizes),
            len(shapes),
            time.perf_counter() - started,
            _format_memory_delta(memory_before, _memory_snapshot(self.device)),
            "n/a" if used_before is None or used_after is None else f"+{(used_after - used_before) / 2**20:.1f} MiB",
        )
        return captured

    def _capture(
        self,
        batch_size: int,
        mel_frames: int,
        source_cache_len: int,
    ):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Cannot capture HiFT graph during an active stream capture")

        key = (batch_size, mel_frames, source_cache_len)

        if key in self.graph:
            return

        static_mel = torch.zeros(batch_size, self.mel_frames, mel_frames, device=self.device, dtype=self.dtype)
        static_source_cache = torch.zeros(batch_size, 1, source_cache_len, device=self.device, dtype=self.dtype)
        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(3):
                warmup_outputs = self.graph_fn(static_mel, static_source_cache)
        current_stream.wait_stream(warmup_stream)
        del warmup_outputs

        graph = CUDAGraph()
        with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
            static_magnitude_output, static_phase_output, static_cache_source_output = self.graph_fn(
                static_mel,
                static_source_cache,
            )

        self.graph[key] = graph
        self.static_speech_inputs[key] = static_mel
        self.static_cache_source_inputs[key] = static_source_cache

        self.static_magnitude_outputs[key] = static_magnitude_output
        self.static_phase_outputs[key] = static_phase_output
        self.static_cache_source_outputs[key] = static_cache_source_output
        logger.info("Captured HiFT CUDA Graph for shape %s", key)

    def replay(self, speech_feat, cache_source):
        if int(speech_feat.shape[2]) == 0:
            return empty_hift_outputs(speech_feat)
        if torch.cuda.is_current_stream_capturing():
            logger.info("Falling back to eager HiFT inference during an active stream capture")
            return self.decode_fn(speech_feat, cache_source)

        batch_size = speech_feat.shape[0]
        num_frames = speech_feat.shape[2]
        cache_source_len = cache_source.shape[2]

        exact_key = (batch_size, num_frames, cache_source_len)
        if exact_key in getattr(self, "_exact_keys", ()) and exact_key in self.graph:
            # An exact-shape graph: same shape as the eager call, no padding.
            return self._replay_key(exact_key, speech_feat, cache_source)

        if (num_frames, cache_source_len) not in self._legit_shapes:
            # Never lazily captured (see ``_legit_shapes``); one eager call over
            # the whole batch, since none of its rows can graph anyway.
            logger.info(
                "Falling back to eager HiFT inference for shape (%d, %d, %d) outside the capture buckets",
                batch_size,
                num_frames,
                cache_source_len,
            )
            return self.decode_fn(speech_feat, cache_source)

        target_b = next((b for b in sorted(self.capture_batch_sizes) if b >= batch_size), None)

        if target_b is None:
            if 1 in self.capture_batch_sizes and 1 < batch_size <= self.max_serial_batch:
                speeches = []
                sources = []
                for b in range(batch_size):
                    sp, src = self.replay(speech_feat[b : b + 1], cache_source[b : b + 1])
                    speeches.append(sp)
                    sources.append(src)
                return torch.cat(speeches, dim=0), torch.cat(sources, dim=0)
            logger.info("Falling back to eager HiFT inference for unsupported batch size %d", batch_size)
            return self.decode_fn(speech_feat, cache_source)

        key = (target_b, num_frames, cache_source_len)

        if key not in self.graph:
            if self.lazy_graph_count >= self.max_lazy_graphs:
                logger.info("Falling back to eager HiFT inference after reaching the lazy Graph limit")
                return self.decode_fn(speech_feat, cache_source)
            logger.info("Lazily capturing HiFT CUDA Graph for shape %s", key)
            self._capture(*key)
            self.lazy_graph_count += 1

        return self._replay_key(key, speech_feat, cache_source)

    def _replay_key(self, key, speech_feat, cache_source):
        """Replay the captured graph ``key`` on a batch of at most ``key[0]`` rows."""
        batch_size = speech_feat.shape[0]
        static_speech_inputs = self.static_speech_inputs[key].zero_()
        static_speech_inputs[:batch_size].copy_(speech_feat)
        static_cache_sources = self.static_cache_source_inputs[key].zero_()
        static_cache_sources[:batch_size].copy_(cache_source)

        self.graph[key].replay()
        static_magnitude_output = self.static_magnitude_outputs[key]
        static_phase_output = self.static_phase_outputs[key]
        static_cache_source_output = self.static_cache_source_outputs[key]
        cache_source = static_cache_source_output[:batch_size].clone()
        speech = self.finalize_fn(static_magnitude_output[:batch_size], static_phase_output[:batch_size]).clone()
        return speech, cache_source


def _tensor_signature(value: torch.Tensor | None) -> tuple:
    if value is None:
        return (None,)
    return tuple(value.shape), str(value.dtype), str(value.device)


_DTYPE_MAP = {str(dtype): dtype for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64, torch.bool)}


def _memory_snapshot(device: torch.device) -> tuple[int, int] | None:
    """(allocated, reserved) bytes, or None when the device cannot report them.

    Capture draws on the caching allocator, so these are the numbers that say
    what a capture cost. Free device memory is not: the allocator serves a
    capture out of memory it has already reserved, which is most of the device
    on a normally configured worker.
    """
    if device.type != "cuda":
        return None
    try:
        return int(torch.accelerator.memory_allocated(device)), int(torch.accelerator.memory_reserved(device))
    except Exception:
        return None


def _device_used_bytes(device: torch.device) -> int | None:
    """Device memory in use (``cudaMemGetInfo``), including what the caching allocator never sees.

    CUDA graphs also hold driver-side memory per captured graph, which the
    allocator's ``memory_reserved`` does not count.
    """
    if device.type != "cuda":
        return None
    try:
        free, total = current_omni_platform.get_device_memory(device)
    except Exception:
        return None
    return int(total - free)


def _format_memory_delta(before: tuple[int, int] | None, after: tuple[int, int] | None) -> str:
    if before is None or after is None:
        return ""
    mib = 1024 * 1024
    return (
        f" [allocated {after[0] / mib:.1f} MiB (+{(after[0] - before[0]) / mib:.1f}), "
        f"reserved {after[1] / mib:.1f} MiB (+{(after[1] - before[1]) / mib:.1f})]"
    )


# Frame granularity of the shared attention cache storage.
_ATT_FRAME_ALIGN = 16


def _minus_ones(shape, *, device, dtype) -> torch.Tensor:
    return torch.full(shape, -1, device=device, dtype=dtype)


def _align_up(n: int, bucket: int) -> int:
    if n <= 0 or bucket <= 1:
        return max(0, n)
    return ((n + bucket - 1) // bucket) * bucket


def _capture_query_width(mel_width: int, bucket: int | tuple[int, ...]) -> int:
    """Snap the query axis onto the Whole-Euler capture grid.

    ``_cfm_pad_frames`` already aligns onto ``bucket_frames=16``, which is why
    12+4 and 10+6 share a graph (``test_whole_euler_same_bucket_different_padding_hits_cache``)
    but 16/32/48/64 stay distinct (``test_varied_chunk_lengths_collapse_onto_few_widths``).
    The capture bucket reuses that same pad-and-mask scheme so those
    decode widths collapse onto one CUDA graph; first-chunk widths above the
    bucket (e.g. 304) align up separately so streaming replay stays on the
    small decode graph.

    A tuple of widths snaps to the narrowest one that holds the chunk (a
    25-token duplex unit and a 75-token turn chunk each keep their own width);
    past the widest, the chunk aligns up on it.
    """
    widths = sorted(int(w) for w in (bucket if isinstance(bucket, tuple) else (bucket,)) if int(w) > 1)
    if not widths:
        return int(mel_width)
    for width in widths:
        if mel_width <= width:
            return width
    return _align_up(int(mel_width), widths[-1])


def _capture_offset(offset: int, bucket: int, steady: int) -> int:
    """Snap an estimator-cache length onto the Whole-Euler offset grid.

    Exact lengths made one graph per first-chunk size: under full duplex a
    capture (1-2 s, stalling every stream) on almost every response. The grid
    is anchored at the steady ``prompt + 100`` length so the steady solve is
    never padded; shorter caches round up to it in ``bucket``-frame steps. The
    padded frames are masked out of attention (``_build_capture_mask``), the
    only place the DiT reads its cache, so no row's result changes.
    """
    offset = int(offset)
    if bucket <= 1 or offset <= 0:
        return offset
    if 0 < steady and offset <= steady:
        return steady - ((steady - offset) // bucket) * bucket
    return _align_up(offset, bucket)


def _pad_query_for_capture(
    x: torch.Tensor,
    mu_cfg: torch.Tensor,
    cond_cfg: torch.Tensor,
    query_cap: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad query tensors to ``query_cap`` the same way ``_decode_cfm`` pads to 16.

    ``mu``/``cond`` replicate the last frame (closer continuation than silence);
    ``x`` is zeroed on the pad so the Euler state does not integrate noise there.
    """
    extra = int(query_cap) - int(mu_cfg.shape[2])
    if extra <= 0:
        return x, mu_cfg, cond_cfg
    return (
        F.pad(x, (0, extra)),
        F.pad(mu_cfg, (0, extra), mode="replicate"),
        F.pad(cond_cfg, (0, extra), mode="replicate"),
    )


def _build_capture_mask(
    *,
    attn_mask: torch.Tensor | None,
    batch_size: int,
    query_cap: int,
    offset: int,
    mel_width: int,
    mel_frames: int,
    device: torch.device,
    offset_cap: int | None = None,
) -> torch.Tensor:
    """Always-on attention mask in the capture layout, so presence-of-mask is not a graph key.

    The DiT puts the current chunk ahead of its cache (stepaudio2
    ``Attention.forward_chunk``: ``torch.cat([k, k_cache], dim=2)``). The
    caller's keys are therefore ``[current(mel_width) | cache(offset)]`` while
    the captured graph sees ``[current(query_cap) | cache(offset)]``: the cache
    block moves to ``query_cap`` and the capture padding in between is never
    attended. A missing caller mask means "attend to the valid prefix".

    Capture-padding query rows reuse the first row's keys. Their output is
    dropped, but an all-False row makes SDPA emit NaN there, and a NaN value in
    a masked key still poisons real rows through ``0 * NaN`` in ``P @ V``.
    """
    query_cap = int(query_cap)
    offset = int(offset)
    mel_width = int(mel_width)
    # ``offset_cap`` columns of cache follow the query block; the ones past the
    # request's own ``offset`` are ``_capture_offset`` padding and stay False.
    offset_cap = offset if offset_cap is None else int(offset_cap)
    mask = torch.zeros(
        2 * int(batch_size),
        query_cap,
        query_cap + offset_cap,
        dtype=torch.bool,
        device=device,
    )
    if attn_mask is not None:
        mask[:, :mel_width, :mel_width] = attn_mask[:, :mel_width, :mel_width]
        if offset > 0:
            mask[:, :mel_width, query_cap : query_cap + offset] = attn_mask[
                :, :mel_width, mel_width : mel_width + offset
            ]
    else:
        mask[:, :mel_width, : int(mel_frames)] = True
        if offset > 0:
            mask[:, :mel_width, query_cap : query_cap + offset] = True
    if query_cap > mel_width:
        mask[:, mel_width:] = mask[:, :1]
    return mask


def _pad_invalid_queries(mask: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """``mask`` with the queries past each CFG row's ``lengths`` taking that row's first query's keys.

    Their output is dropped, but no attention row may be empty (see
    ``_build_capture_mask``).
    """
    invalid = torch.arange(int(mask.shape[1]), device=mask.device).unsqueeze(0) >= lengths.unsqueeze(1)
    return torch.where(invalid.unsqueeze(2), mask[:, :1], mask)


def _att_keep_ranges(total: int, keep: tuple[int, int] | None) -> list[tuple[int, int]]:
    """``(start, length)`` frame ranges of a new estimator cache that survive the streaming trim.

    Mirrors stepaudio2 ``Token2wav.stream``: once the cache outgrows
    ``prompt_len + 100`` frames it keeps the first ``prompt_len`` and the last
    100. ``keep`` is ``(prompt_len, 100)``; ``None`` keeps everything.
    """
    total = int(total)
    if keep is not None:
        prefix, suffix = int(keep[0]), int(keep[1])
        if total > prefix + suffix:
            return [(0, prefix), (total - suffix, suffix)]
    return [(0, total)]


def _whole_euler_att_segments(
    *,
    mel_width: int,
    query_cap: int,
    offset: int,
    keep: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    """Frame segments of a Whole-Euler cache output, in caller layout and trimmed.

    The graph writes ``[current(query_cap) | cache(offset)]`` (current chunk
    first, see ``_build_capture_mask``); callers expect
    ``[current(mel_width) | cache(offset)]``. The segments skip the capture
    padding in between, then apply ``keep`` (``_att_keep_ranges``).
    """
    segments: list[tuple[int, int]] = []
    for start, length in _att_keep_ranges(mel_width + offset, keep):
        end = start + length
        if start < mel_width:
            segments.append((start, min(end, mel_width) - start))
        if end > mel_width:
            low = max(start, mel_width)
            segments.append((query_cap + low - mel_width, end - low))
    merged: list[tuple[int, int]] = []
    for start, length in segments:
        if merged and sum(merged[-1]) == start:
            merged[-1] = (merged[-1][0], merged[-1][1] + length)
        elif length > 0:
            merged.append((start, length))
    return merged


def _resized_frame_view(cache: torch.Tensor, frames: int) -> torch.Tensor | None:
    """``cache`` (..., L, width) as ``frames`` frames of its own allocation, or ``None`` when it has no room.

    Only a view of the first frames of a whole contiguous allocation
    qualifies, as ``WholeEulerCFMGraphWrapper.replay`` hands out: a cache
    allocated for its steady length then grows into it without a second copy.
    """
    width = int(cache.shape[-1])
    room = cache.stride(-3) // max(width, 1)
    full = (*cache.shape[:-2], room, width)
    strides, step = [], 1
    for size in reversed(full):
        strides.append(step)
        step *= size
    if (
        room < frames
        or cache.stride() != tuple(reversed(strides))
        or cache.storage_offset() != 0
        or cache.untyped_storage().nbytes() != step * cache.element_size()
    ):
        return None
    return cache.as_strided((*cache.shape[:-2], frames, width), cache.stride())


def _copy_frame_segments(dst: torch.Tensor, src: torch.Tensor, segments: list[tuple[int, int]]) -> None:
    """Write ``src[..., start:start + length, :]`` segments back to back into ``dst``'s frame axis."""
    position = 0
    for start, length in segments:
        dst[..., position : position + length, :].copy_(src[..., start : start + length, :])
        position += length


class SharedPromptAttCache:
    """A request-owned cache prefix plus a read-only prompt suffix.

    The logical cache remains ``(..., frames, width)`` so callers and graph
    bucketing keep their existing shape contracts.  Only the tail is retained
    per request; graph staging can copy a range directly without materializing
    the shared suffix. The current-first cache layout means the immutable
    voice segment is at the logical tail after streaming trim.
    """

    __slots__ = ("prefix", "prompt")

    def __init__(self, prefix: torch.Tensor, prompt: torch.Tensor) -> None:
        if prefix.ndim != 6 or prompt.ndim != 6 or tuple(prefix.shape[:4]) != tuple(prompt.shape[:4]):
            raise ValueError("shared prompt attention cache tensors have incompatible shapes")
        if prefix.shape[5] != prompt.shape[5] or prefix.device != prompt.device or prefix.dtype != prompt.dtype:
            raise ValueError("shared prompt attention cache tensors must have the same device, dtype and width")
        self.prefix = prefix
        self.prompt = prompt

    @property
    def shape(self) -> torch.Size:
        return torch.Size((*self.prefix.shape[:4], self.prefix.shape[4] + self.prompt.shape[4], self.prefix.shape[5]))

    @property
    def dtype(self) -> torch.dtype:
        return self.prefix.dtype

    @property
    def device(self) -> torch.device:
        return self.prefix.device

    def data_ptr(self) -> int:
        """Expose a stable pointer for cache-owner accounting without a full materialization."""
        return self.prefix.data_ptr()

    def copy_range_to(self, dst: torch.Tensor, start: int, length: int) -> None:
        """Copy a logical frame range into ``dst`` without joining prompt and tail."""
        start, length = int(start), int(length)
        if start < 0 or length < 0 or start + length > int(self.shape[4]):
            raise ValueError(f"attention-cache range ({start}, {length}) is outside {tuple(self.shape)}")
        prefix_len = int(self.prefix.shape[4])
        position = 0
        if start < prefix_len:
            count = min(length, prefix_len - start)
            dst[..., position : position + count, :].copy_(self.prefix[..., start : start + count, :])
            position += count
            start += count
            length -= count
        if length:
            prompt_start = start - prefix_len
            dst[..., position : position + length, :].copy_(self.prompt[..., prompt_start : prompt_start + length, :])

    def materialize(self) -> torch.Tensor:
        """Materialize the logical cache for eager or unsupported graph paths."""
        return torch.cat((self.prefix, self.prompt), dim=4)


def _copy_att_cache_range(dst: torch.Tensor, src, start: int, length: int) -> None:
    if isinstance(src, SharedPromptAttCache):
        src.copy_range_to(dst, start, length)
    elif isinstance(src, ResidentAttCache):
        materialized = src.materialize()
        dst.copy_(materialized[..., int(start) : int(start) + int(length), :])
    else:
        dst.copy_(src[..., int(start) : int(start) + int(length), :])


def _tensors_from_key(key: tuple) -> tuple[torch.Tensor, ...]:
    """Rebuild zero tensors from a cache key (shape, dtype, device tuples).

    Raises ``KeyError`` for a dtype the key cannot round-trip, so the caller
    can fall back to eager instead of capturing wrong-dtype static buffers.
    """
    tensors = []
    for shape, dtype_str, device_str in key[1:]:
        dtype = _DTYPE_MAP[dtype_str]
        device = torch.device(device_str)
        tensors.append(torch.zeros(shape, dtype=dtype, device=device))
    return tuple(tensors)


class CFMGraphWrapper:
    """Per-shape CUDA graph capture/replay for the CFM DiT estimator.

    Captures one blocks_forward_chunk call (in_proj -> DiT blocks -> final_layer)
    as the graph target. The 10-step Euler loop stays in Python, replaying
    the graph 10 times per decode.

    Graphs are retired a whole generation at a time rather than one at a time.
    Every capture shares one private memory pool, so a retired graph's blocks
    return to that pool while its live peers still hold those addresses in
    their recorded kernel arguments, and the next replay reads memory that now
    belongs to something else. Retiring the whole generation bounds the cache
    without ever leaving a live graph behind a freed one.

    Cache misses capture. A capture failure disables the wrapper for the rest
    of the process, while a key whose static buffers cannot be rebuilt sends
    only that one shape eager. Outputs are cloned after replay to prevent
    streaming cache corruption.
    """

    def __init__(
        self,
        graph_fn,
        *,
        max_graphs: int = 32,
    ) -> None:
        self.graph_fn = graph_fn
        self.max_graphs = int(max_graphs)
        self.device = next(graph_fn.__self__.parameters()).device
        # A non-positive budget means "no graphs", the same as
        # `enable_cfm_graph: false`. Clamping to 1 would instead build a
        # one-entry cache that flushes on every new shape.
        self.enabled = self.max_graphs > 0
        self._cache: dict[tuple, tuple] = {}
        # Shapes whose key cannot round-trip: eager for those, keep the rest.
        self._unsupported: set[tuple] = set()
        self._stats = {
            "calls": 0,
            "hits": 0,
            "captures": 0,
            "flushes": 0,
            "eager": 0,
        }

    def stats_snapshot(self) -> dict[str, int]:
        """Bounded cumulative telemetry for the graph cache."""
        return {**self._stats, "cache_size": len(self._cache)}

    def _call_graph_fn(self, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.graph_fn(args[0], args[1], args[6], args[2], args[3], args[4], args[5])

    def _eager(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._stats["eager"] += 1
        with torch.no_grad():
            result = self._call_graph_fn(inputs)
        return result, inputs[4], inputs[5]

    def _flush(self) -> None:
        """Retire every captured graph at once.

        The sync keeps a replay from being in flight when the graphs go, and
        the explicit ``reset()`` tears each one down here rather than whenever
        Python drops the last reference. Nothing process-wide is touched: the
        cuBLAS workspace is shared with graphs this wrapper does not own.
        """
        if not self._cache:
            return
        torch.accelerator.synchronize(self.device)
        for entry in self._cache.values():
            entry[2].reset()
        self._cache.clear()
        self._stats["flushes"] += 1
        logger.info("CFM graph cache flushed; stats=%s", self.stats_snapshot())

    def _disable(self, reason: str, key: tuple) -> None:
        logger.warning("Disabling CFM CUDA graphs (%s) for shape=%s; using eager", reason, key, exc_info=True)
        self.enabled = False
        self._flush()

    def _capture(self, key: tuple, inputs: tuple | None = None) -> tuple | None:
        """Capture a CUDA graph for the given key. Returns None on failure.

        ``inputs`` should be the real tensors for this key whenever they are
        available: the static buffers must be built from the values the graph
        will actually run with. Zero-filled placeholders are only a fallback --
        capturing with them bakes the placeholder branch into the graph (e.g. an
        attention mask that looks like "nothing is masked").
        """
        try:
            static_inputs = (
                tuple(None if tensor is None else tensor.detach().clone() for tensor in inputs)
                if inputs is not None
                else _tensors_from_key(key)
            )
        except KeyError:
            # An unsupported dtype is a property of this shape alone, so run it
            # eager and keep the other shapes on graphs.
            logger.warning("CFM graph key carries an unsupported dtype: %s; using eager", key)
            self._unsupported.add(key)
            return None

        memory_before = _memory_snapshot(self.device)
        try:
            # Warmup runs the same kernels as the capture, so a fault here
            # leaves the same dirty capture-stream and pool state, and must be
            # handled the same way.
            current_stream = torch.cuda.current_stream(self.device)
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream), torch.no_grad():
                for _ in range(3):
                    warmup_output = self._call_graph_fn(static_inputs)
            current_stream.wait_stream(warmup_stream)
            del warmup_output

            graph = CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self._call_graph_fn(static_inputs)
        except Exception:
            # A failed capture can leave the capture stream current and the
            # allocator still routing into the graph pool, so there is no safe
            # way to keep capturing afterwards.
            self._disable("capture failed", key)
            return None

        self._stats["captures"] += 1
        logger.info(
            "Captured CFM CUDA Graph for shape %s (cache=%d/%d, stats=%s)%s",
            key,
            len(self._cache) + 1,
            self.max_graphs,
            self.stats_snapshot(),
            _format_memory_delta(memory_before, _memory_snapshot(self.device)),
        )
        return (static_inputs, static_output, graph)

    def replay(
        self,
        estimator_input: torch.Tensor,
        time_emb: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_out: torch.Tensor,
        att_out: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = (estimator_input, time_emb, cnn_cache, att_cache, cnn_out, att_out, attn_mask)
        self._stats["calls"] += 1

        if not self.enabled or torch.cuda.is_current_stream_capturing() or estimator_input.device.type != "cuda":
            return self._eager(inputs)

        key = ("estimator_step",) + tuple(_tensor_signature(v) for v in inputs)
        if key in self._unsupported:
            return self._eager(inputs)
        entry = self._cache.get(key)

        if entry is None:
            if len(self._cache) >= self.max_graphs:
                self._flush()
            entry = self._capture(key, inputs)
            if entry is None:
                return self._eager(inputs)
            self._cache[key] = entry
        else:
            self._stats["hits"] += 1

        static_inputs, static_output, graph = entry
        for static, current in zip(static_inputs, inputs, strict=True):
            if static is not None:
                static.copy_(current)
        graph.replay()
        return (
            static_output.detach().clone(),
            static_inputs[4].detach().clone(),
            static_inputs[5].detach().clone(),
        )


def _zero_padded_cnn_cache(
    cnn_cache: torch.Tensor,
    estimator: torch.nn.Module,
    pad_frames: int,
) -> None:
    """Clear the cache positions that come from padded frames, in place.

    Each block's CNN cache holds the tail of its convolution output
    (``new_cnn_cache = x[..., -causal_padding[0]:]``, inside ``stepaudio2``),
    so when the padding sits at the chunk tail those positions are
    padding-derived and would otherwise become the next chunk's left context.
    Padding is at most ``pad_frames`` wide, so only the trailing positions that
    can come from it are cleared; the valid part of the window is kept.
    ``cnn_cache`` is indexed by block on its first axis.
    """
    blocks = estimator.blocks
    if pad_frames > 0 and blocks and len(blocks) == cnn_cache.shape[0]:
        width = int(cnn_cache.shape[-1])
        if width > 0 and all(int(block.conv.block[1].causal_padding[0]) == width for block in blocks):
            # _estimator_buffers packs equal-width blocks into one tensor.
            # Clear their shared tail with one write instead of one per block.
            cnn_cache[..., max(0, width - pad_frames) :] = 0.0
            return

    for index, block in enumerate(blocks):
        width = int(block.conv.block[1].causal_padding[0])
        if width <= 0:
            continue
        zero_from = max(0, width - pad_frames)
        if zero_from < width:
            cnn_cache[index][..., zero_from:] = 0.0


class WholeEulerExecutionArena:
    """Static staging buffers for Whole-Euler graphs, shared across graph entries.

    Time embeddings, CNN caches and speakers are allocated once per graph batch
    size. The attention caches dominate the arena (timesteps x depth x CFG x
    heads x frames, ~1.25 MiB per frame per request in fp32 on the shipped
    DiT), so there is exactly one attention cache storage: every graph, of
    every batch size and offset, takes a ``[:2B, ..., :frames]`` view of it.
    A graph's output cache is ``[current(query_cap) | cache(offset)]`` and its
    input cache is the ``cache`` part of that same view, so a block that
    attends over its output in place (``_attend_in_place``) finds the cache
    already there; upstream's block concatenates the cache it reads
    (``torch.cat([k, k_cache])``) before writing it back, which is safe on
    that aliasing too. Graphs replay one at a time on one stream and refill
    their inputs first, so the sharing across graphs is safe. Separate
    per-offset, per-batch or input/output storages each held another full copy.
    Small shape-keyed buffers (x, mu, cond, mask) stay per query width.
    """

    def __init__(
        self,
        estimator: torch.nn.Module,
        *,
        n_timesteps: int = 10,
        device: torch.device,
        dtype: torch.dtype,
        att_cache_dtype: torch.dtype = torch.float32,
        timeline: torch.Tensor,
    ) -> None:
        self.estimator = estimator
        self.n_timesteps = int(n_timesteps)
        self.device = device
        self.dtype = dtype
        self.att_cache_dtype = att_cache_dtype
        self.timeline = timeline

        blocks = estimator.blocks
        self.depth = len(blocks)
        block0 = blocks[0]
        self.cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        self.cnn_width = int(block0.conv.block[1].causal_padding[0])
        self.heads = int(block0.attn.num_heads)
        self.att_width = int(block0.attn.head_dim * 2)

        # Static staging buffers, shared by every graph that stages the same shape.
        self._buffers: dict[tuple, torch.Tensor] = {}
        # The one attention cache storage; every graph takes a prefix view.
        self._att: torch.Tensor | None = None
        # Per-timestep modulation table for a fused body (``get_modulation``).
        self._modulation: torch.Tensor | None = None

    def _buffer(self, name: str, shape: tuple[int, ...], new=torch.zeros, dtype: torch.dtype | None = None):
        key = (name, shape)
        buf = self._buffers.get(key)
        if buf is None:
            buf = new(shape, device=self.device, dtype=self.dtype if dtype is None else dtype)
            self._buffers[key] = buf
        return buf

    def get_time_embeddings(self, batch_size: int) -> torch.Tensor:
        key = ("time_embeddings", batch_size)
        t_emb = self._buffers.get(key)
        if t_emb is None:
            t_emb = torch.stack(
                [
                    self.estimator.t_embedder(self.timeline[s].expand(2 * batch_size)).unsqueeze(1)
                    for s in range(self.n_timesteps)
                ],
                dim=0,
            )
            self._buffers[key] = t_emb
        return t_emb

    def get_modulation(self, modulation_fn: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        """``modulation_fn`` of each timestep's (one-row) time embedding, stacked on dim 0.

        Every row of a solve shares the timestep, so this is computed once,
        outside any capture, and shared by every graph.
        """
        if self._modulation is None:
            self._modulation = torch.stack(
                [
                    modulation_fn(self.estimator.t_embedder(self.timeline[s].expand(1)).unsqueeze(1))
                    for s in range(self.n_timesteps)
                ],
                dim=0,
            )
        return self._modulation

    def _cnn_cache_shape(self, batch_size: int) -> tuple[int, ...]:
        return (self.n_timesteps, self.depth, 2 * batch_size, self.cnn_channels, self.cnn_width)

    def get_cnn_cache_in(self, batch_size: int) -> torch.Tensor:
        return self._buffer("cnn_in", self._cnn_cache_shape(batch_size))

    def get_cnn_cache_out(self, batch_size: int) -> torch.Tensor:
        return self._buffer("cnn_out", self._cnn_cache_shape(batch_size), new=torch.empty)

    def get_speakers_staging(self, batch_size: int, spk_dim: int) -> torch.Tensor:
        return self._buffer("speakers", (2 * batch_size, spk_dim))

    def get_x_staging(self, batch_size: int, channels: int, mel_width: int) -> torch.Tensor:
        return self._buffer("x", (batch_size, channels, mel_width))

    def get_mu_staging(self, batch_size: int, channels: int, mel_width: int) -> torch.Tensor:
        return self._buffer("mu", (2 * batch_size, channels, mel_width))

    def get_cond_staging(self, batch_size: int, channels: int, mel_width: int) -> torch.Tensor:
        return self._buffer("cond", (2 * batch_size, channels, mel_width))

    def att_cache_fits(self, batch_size: int, frames: int) -> bool:
        """Whether a view of ``batch_size`` requests ending at frame ``frames`` fits the current storage."""
        storage = self._att
        return storage is not None and int(storage.shape[2]) >= 2 * batch_size and int(storage.shape[4]) >= frames

    def att_cache_view(
        self, batch_size: int, frames: int, *, start: int = 0, capacity: int = 0, rows: int = 0
    ) -> torch.Tensor:
        """``(n_t, depth, 2B, heads, frames, width)`` view of the shared cache storage from frame ``start``.

        ``capacity`` frames and ``rows`` requests are reserved up front. A
        larger request replaces the storage. Graphs captured on the old one
        would keep it alive beside the new one, so
        ``WholeEulerCFMGraphWrapper._capture`` flushes them first.
        """
        storage = self._att
        frames = int(frames)
        end = int(start) + frames
        if not self.att_cache_fits(batch_size, end):
            old_rows = 0 if storage is None else int(storage.shape[2]) // 2
            old_frames = 0 if storage is None else int(storage.shape[4])
            storage = torch.zeros(
                self.n_timesteps,
                self.depth,
                2 * max(int(batch_size), int(rows), old_rows),
                self.heads,
                _align_up(max(end, int(capacity), old_frames, 1), _ATT_FRAME_ALIGN),
                self.att_width,
                device=self.device,
                dtype=self.att_cache_dtype,
            )
            self._att = storage
        return storage[:, :, : 2 * batch_size, :, start:end, :]

    def get_mask_staging(self, batch_size: int, mel_width: int, total_len: int) -> torch.Tensor:
        return self._buffer("mask", (2 * batch_size, mel_width, total_len), new=torch.ones, dtype=torch.bool)

    def get_lengths_staging(self, batch_size: int) -> torch.Tensor:
        """Per-CFG-row valid query lengths read by a ragged graph body."""
        return self._buffer("lengths", (2 * batch_size,), dtype=torch.long)

    def att_bytes(self) -> int:
        """Device bytes of the shared attention cache storage."""
        storage = self._att
        return 0 if storage is None else storage.numel() * storage.element_size()

    def clear(self) -> None:
        self._buffers.clear()
        self._att = None
        self._modulation = None


class AttSlotPool:
    """Resident estimator attention caches, one slot per request, for the Whole-Euler graphs.

    The arena path copies each request's cache (~0.5 GiB in fp32) into the
    shared arena and the new one back out on every replay; here the cache
    stays in its slot and the graph attends to it in place.

    The streaming trim (``_att_keep_ranges``) keeps a cache's first ``prefix``
    and last ``suffix`` frames, and chunks are prepended, so the last
    ``suffix`` frames never change once present. They sit at physical frames
    ``0 .. suffix - 1`` and the streaming frames anywhere behind them (a
    ``ResidentAttCache`` lists where). The graph writes a chunk's keys/values
    to free frames and a trim only releases frames, so nothing moves and the
    cache offset is no graph key: attention runs over the whole slot under a
    mask of the request's frames (no positional term reads the key order).

    Storage is ``(n_t, depth, slots, 2, heads, frames, width)``; request
    ``s``'s CFG rows are rows ``2s`` and ``2s + 1`` of ``rows()``.
    """

    def __init__(
        self,
        *,
        n_timesteps: int,
        depth: int,
        heads: int,
        width: int,
        slots: int,
        frames: int,
        suffix: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.n_timesteps, self.depth, self.heads, self.width = n_timesteps, depth, heads, width
        self.slots, self.frames, self.suffix = int(slots), int(frames), int(suffix)
        self.device, self.dtype = device, dtype
        self.storage = torch.zeros(
            (n_timesteps, depth, self.slots, 2, heads, self.frames, width), device=device, dtype=dtype
        )
        # Lowest slot first, so a lightly loaded pool keeps touching the same pages.
        self._free = list(range(self.slots - 1, -1, -1))

    def rows(self) -> torch.Tensor:
        """``(n_t, depth, 2 * slots, heads, frames, width)``: one row per request CFG half."""
        return self.storage.flatten(2, 3)

    @property
    def free_slots(self) -> int:
        return len(self._free)

    def bytes(self) -> int:
        return self.storage.numel() * self.storage.element_size()

    def fits(self, frames: int) -> bool:
        """Whether a logical cache of ``frames`` frames (after a chunk) fits one slot."""
        return self.suffix <= frames <= self.frames

    def adopt(self, cache: torch.Tensor) -> "ResidentAttCache | None":
        """A slot holding the logical ``(n_t, depth, 2, heads, L, width)`` ``cache``; ``None`` when full.

        The last ``suffix`` frames go to the fixed frames, the rest in order
        behind them. ``cache`` itself is not written, so a shared one (the
        prompt state) can be adopted by every request that starts from it.
        """
        frames = int(cache.shape[4])
        if not self._free or not self.fits(frames):
            return None
        slot = self._free.pop()
        streaming = frames - self.suffix
        _copy_frame_segments(self.storage[:, :, slot], cache, [(streaming, self.suffix), (0, streaming)])
        return ResidentAttCache(self, slot, list(range(self.suffix, self.suffix + streaming)))


class ResidentAttCache:
    """A request's estimator attention cache resident in an ``AttSlotPool`` slot.

    Stands in for the ``(n_t, depth, 2, heads, L, width)`` tensor the
    Whole-Euler path otherwise hands back per request: ``shape``, ``dtype``
    and ``device`` describe that logical tensor, and ``materialize()`` builds
    it for the paths that need one. ``layout`` is the physical frame of each
    streaming frame, in logical order. The slot returns to the pool when the
    handle is collected; work queued on it before then runs first, since all
    of it is on one stream.
    """

    __slots__ = ("pool", "slot", "layout", "__weakref__")

    def __init__(self, pool: AttSlotPool, slot: int, layout: list[int]) -> None:
        self.pool = pool
        self.slot = slot
        self.layout = layout
        weakref.finalize(self, pool._free.append, slot)

    @property
    def shape(self) -> torch.Size:
        pool = self.pool
        return torch.Size((pool.n_timesteps, pool.depth, 2, pool.heads, len(self.layout) + pool.suffix, pool.width))

    @property
    def dtype(self) -> torch.dtype:
        return self.pool.dtype

    @property
    def device(self) -> torch.device:
        return self.pool.device

    def physical_frames(self) -> list[int]:
        """The physical frame of every logical frame, in logical order."""
        return [*self.layout, *range(self.pool.suffix)]

    def materialize(self) -> torch.Tensor:
        """The logical cache as a tensor of its own."""
        index = index_to_device(self.physical_frames(), self.pool.device)
        return self.pool.storage[:, :, self.slot].index_select(4, index)


def _materialize_att_rows(rows: list) -> list[torch.Tensor]:
    return [row.materialize() if isinstance(row, (ResidentAttCache, SharedPromptAttCache)) else row for row in rows]


class _WholeEulerStatics(NamedTuple):
    """A Whole-Euler graph's static inputs (``lengths`` only with a ragged body)."""

    x: torch.Tensor
    mu_cfg: torch.Tensor
    speakers_cfg: torch.Tensor
    cond_cfg: torch.Tensor
    cnn_cache: torch.Tensor
    att_cache: torch.Tensor
    attn_mask: torch.Tensor
    time_embeddings: torch.Tensor
    lengths: torch.Tensor | None
    # (n_timesteps, ...) per-timestep modulation for a body that takes it.
    modulation: torch.Tensor | None = None
    # Slot-pool graphs (``AttSlotPool``): each row's pool row, and the pool
    # frame each query frame's key/value is written to (-1: not written).
    slot_rows: torch.Tensor | None = None
    slot_positions: torch.Tensor | None = None


class WholeEulerCFMGraphWrapper:
    """Per-shape CUDA graph capture/replay for the entire 10-step CFM Euler ODE solver.

    Captures all 10 iterations of:
    (time embedding -> in_proj -> DiT blocks -> final layer -> CFG combination -> Euler step)
    into a single CUDA Graph replay.

    Replaces 10 separate step-level graph replays and 30 Python clones per chunk
    with a single graph replay, reducing CPU launch overhead and eliminating host-device
    synchronization bubbles during high concurrency.

    A graph must hold all 10 timesteps' attention caches in static buffers,
    ten times what one step-level graph holds, so memory is the constraint:
    graphs exist for the powers of two up to ``micro_batch_size``, all of them
    share the one attention cache storage of ``WholeEulerExecutionArena``
    (sized for ``micro_batch_size`` requests by default), and larger batches
    run as a sequence of graph-tier replays that copy each request's cache in
    and out of that storage directly. An opt-in grid-sized arena can use fewer
    rows while retaining the scheduler's larger admission limit.
    """

    def __init__(
        self,
        estimator: torch.nn.Module,
        *,
        n_timesteps: int = 10,
        inference_cfg_rate: float = 0.7,
        att_cache_dtype: torch.dtype = torch.float32,
        max_graphs: int = 32,
        max_serial_batch: int | None = None,
        max_graph_batch: int | None = None,
        micro_batch_size: int = 4,
        query_bucket_frames: int | tuple[int, ...] | None = None,
        pad_max_rows: int | None = None,
        ragged_body: Callable[..., torch.Tensor] | None = None,
        offset_bucket_frames: int = 0,
        modulation_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        att_slots: int = 0,
        arena_rows_from_graph_grid: bool = False,
        row_offsets: bool = False,
        graph_grid: str | Sequence[int] | None = None,
        fused_euler_step: bool = False,
    ) -> None:
        """``ragged_body(estimator, input, t_emb, mask, cnn, att, cnn_out, att_out, lengths)``
        replaces ``estimator.blocks_forward_chunk`` in the captured solve when
        given: it takes each row's valid query length as a ``(2B,)`` tensor and
        writes that row's exact CNN cache (``_blocks_forward_chunk_ragged``).
        It lets one graph solve rows of different lengths, and lets a padded
        query axis keep the CNN cache the unpadded solve would produce instead
        of zeroing it.

        ``query_bucket_frames`` is one capture width or a tuple of them
        (``_capture_query_width``); ``offset_bucket_frames`` > 1 snaps cache
        lengths onto a grid (``_capture_offset``). ``modulation_fn`` maps a
        one-row time embedding to the per-timestep table a fused body takes as
        ``modulation=``. ``att_slots`` > 0 keeps request caches resident in an
        ``AttSlotPool`` (fused body only): streaming chunks then replay graphs
        keyed by batch and width alone, with no cache copies; the arena serves
        the prompt solve and any chunk the pool cannot hold. ``row_offsets``
        lets such a replay take per-request caches of different lengths (a
        stream's second chunk next to steady ones): each row attends its own
        slot frames, so one graph solves them all.

        ``fused_euler_step`` stages the estimator input once per solve and
        runs each step's CFG combination and update as one pass that also
        writes the next step's input (``whole_euler_ops.py``; same values).
        """
        self.estimator = estimator
        self.fused_euler_step = bool(fused_euler_step)
        self.ragged_body = ragged_body
        self.modulation_fn = modulation_fn if ragged_body is not None else None
        self.n_timesteps = int(n_timesteps)
        self.inference_cfg_rate = float(inference_cfg_rate)
        self.att_cache_dtype = att_cache_dtype
        self.max_graphs = int(max_graphs)
        # Lazy captures may grow ``max_graphs`` up to 4x this (``_entry``).
        self._configured_max_graphs = int(max_graphs)
        if isinstance(query_bucket_frames, (tuple, list)):
            widths = tuple(sorted({int(w) for w in query_bucket_frames if int(w) > 1}))
            self.query_bucket_frames = widths[-1] if widths else 0
        else:
            self.query_bucket_frames = 0 if query_bucket_frames is None else int(query_bucket_frames)
            widths = (self.query_bucket_frames,) if self.query_bucket_frames > 1 else ()
        # The capture widths, narrowest first (``_capture_query_width``).
        self.query_widths: tuple[int, ...] = widths
        self.offset_bucket_frames = int(offset_bucket_frames)
        self.micro_batch_size = 4 if micro_batch_size is None else int(micro_batch_size)
        # ``max_serial_batch`` only caps the default ``max_graph_batch``.
        if max_graph_batch is None:
            if max_serial_batch is not None and max_serial_batch < self.micro_batch_size:
                self.max_graph_batch = int(max_serial_batch)
            else:
                self.max_graph_batch = 16
        else:
            self.max_graph_batch = int(max_graph_batch)
        if pad_max_rows is None:
            pad_max_rows = self.micro_batch_size // 4
        self.pad_max_rows = max(0, int(pad_max_rows))
        self.graph_grid = graph_grid
        self.arena_rows_from_graph_grid = bool(arena_rows_from_graph_grid)
        self._arena_rows = max(self._graph_batches()) if self.arena_rows_from_graph_grid else self.micro_batch_size
        # Largest steady cache length a caller announced; sizes the arena storage.
        self._att_capacity = 0
        parameter = next(estimator.parameters(), None)
        if parameter is not None:
            self.device = parameter.device
            self.dtype = parameter.dtype
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.dtype = torch.float32

        self.enabled = self.max_graphs > 0 and self.device.type == "cuda"
        self._cache: dict[tuple, tuple] = {}
        self._unsupported: set[tuple] = set()
        self._stats = {
            "calls": 0,
            "hits": 0,
            "captures": 0,
            "flushes": 0,
            "eager": 0,
            "slot_replays": 0,
        }

        self.timeline, self.dt_steps = _euler_timeline(self.n_timesteps, self.device, self.dtype)
        arena_args = {
            "estimator": self.estimator,
            "n_timesteps": self.n_timesteps,
            "device": self.device,
            "dtype": self.dtype,
            "att_cache_dtype": self.att_cache_dtype,
            "timeline": self.timeline,
        }
        self.arena = WholeEulerExecutionArena(**arena_args)
        # Resident request caches (``AttSlotPool``), sized once the prompt
        # length is known; their graphs and staging buffers are kept apart
        # from the arena's so an arena flush never retires them.
        self.att_slots = int(att_slots) if ragged_body is not None and modulation_fn is not None else 0
        self.slot_pool: AttSlotPool | None = None
        self.row_offsets = bool(row_offsets) and self.att_slots > 0
        self._slot_arena = WholeEulerExecutionArena(**arena_args)
        self._slot_graphs: dict[tuple, tuple] = {}

    def stats_snapshot(self) -> dict[str, int]:
        """Bounded cumulative telemetry for the graph cache."""
        return {**self._stats, "cache_size": len(self._cache) + len(self._slot_graphs)}

    @staticmethod
    def _retire(graphs: dict[tuple, tuple]) -> None:
        for entry in graphs.values():
            entry[4].reset()
        graphs.clear()

    def _flush(self) -> None:
        """Retire every captured graph at once."""
        torch.accelerator.synchronize(self.device)
        if self._cache:
            self._retire(self._cache)
            self._stats["flushes"] += 1
            logger.info("Whole-Euler CFM graph cache flushed; stats=%s", self.stats_snapshot())
        self.arena.clear()

    def _disable(self, reason: str, key: tuple) -> None:
        logger.warning(
            "Disabling Whole-Euler CFM CUDA graphs (%s) for shape=%s; using step/eager fallback",
            reason,
            key,
            exc_info=True,
        )
        self.enabled = False
        self._flush()
        self._retire(self._slot_graphs)

    def _run_euler_loop(
        self,
        statics: _WholeEulerStatics,
        out_cnn_cache: torch.Tensor,
        out_att_cache: torch.Tensor,
        *,
        batch_size: int,
    ) -> torch.Tensor:
        cur_x = statics.x
        width = int(statics.mu_cfg.shape[2])
        # A slot-pool graph attends to the pool in place: no cache goes in.
        slotted = statics.slot_rows is not None
        no_cache = [None] * len(self.estimator.blocks)
        fused_step = self.fused_euler_step and fused_euler_supported(cur_x)
        if fused_step:
            # Frames-major estimator input, staged once; each fused step rewrites only x.
            in_channels = sum(int(t.shape[1]) for t in (cur_x, statics.mu_cfg, statics.speakers_cfg, statics.cond_cfg))
            staged = cur_x.new_empty((2 * batch_size, width, in_channels))
            stage_estimator_input(staged, cur_x, statics.mu_cfg, statics.speakers_cfg, statics.cond_cfg)
            # Out of place: padded rows of ``statics.x`` keep their values across replays.
            next_x = torch.empty_like(cur_x)

        for step in range(self.n_timesteps):
            dt = self.dt_steps[step]
            if fused_step:
                estimator_input = staged.transpose(1, 2)
            else:
                x_cfg = torch.cat((cur_x, cur_x), dim=0)
                speaker_features = statics.speakers_cfg.unsqueeze(-1).expand(-1, -1, width)
                estimator_input = torch.cat((x_cfg, statics.mu_cfg, speaker_features, statics.cond_cfg), dim=1)
            args = (
                estimator_input,
                statics.time_embeddings[step],
                statics.attn_mask,
                statics.cnn_cache[step],
                no_cache if slotted else statics.att_cache[step],
                out_cnn_cache[step],
                out_att_cache[step],
            )
            if statics.lengths is not None:
                assert self.ragged_body is not None
                kwargs = {}
                if statics.modulation is not None:
                    kwargs["modulation"] = statics.modulation[step]
                if slotted:
                    kwargs["slots"] = (statics.slot_rows, statics.slot_positions)
                estimate = self.ragged_body(self.estimator, *args, statics.lengths, **kwargs)
            else:
                estimate = self.estimator.blocks_forward_chunk(*args)

            if fused_step:
                last = step + 1 == self.n_timesteps
                cur_x = euler_cfg_step(
                    next_x,
                    cur_x,
                    estimate,
                    dt,
                    self.inference_cfg_rate,
                    None if last else staged,
                )
            else:
                cur_x = _euler_step(cur_x, estimate, dt, self.inference_cfg_rate, batch_size)

        return cur_x

    def _graph_batches(self) -> list[int]:
        """Native graph batch sizes: the powers of two below ``micro_batch_size``, and it.

        ``graph_grid`` pins them instead; sizes above ``micro_batch_size`` are dropped.
        """
        micro = self.micro_batch_size
        raw = getattr(self, "graph_grid", None)
        if raw:
            if isinstance(raw, str):
                sizes = sorted({int(x) for x in raw.split(",") if x.strip()})
            else:
                sizes = sorted({int(x) for x in raw if int(x) > 0})
            return [b for b in sizes if 0 < b <= micro] or [micro]
        sizes = []
        size = 1
        while size < micro:
            sizes.append(size)
            size *= 2
        sizes.append(micro)
        return sizes

    def _plan_groups(self, batch_size: int) -> list[tuple[int, int]]:
        """``(graph_batch, rows)`` replays that cover ``batch_size`` requests.

        Larger batches run whole micro-batches. The remainder takes the smallest
        native size that holds it if that pads at most ``pad_max_rows`` rows
        (the padded rows are ignored), and otherwise the largest native size
        below it, repeating on what is left. On a saturated device a small
        graph's replay costs well over its share of a larger one, so a padded
        row is cheaper than a further replay.
        """
        sizes = self._graph_batches()
        largest = max(sizes)
        groups: list[tuple[int, int]] = []
        remainder = batch_size
        while remainder >= largest:
            groups.append((largest, largest))
            remainder -= largest
        while remainder > 0:
            up = next(size for size in sizes if size >= remainder)
            if up - remainder <= self.pad_max_rows:
                groups.append((up, remainder))
                break
            down = max(size for size in sizes if size <= remainder)
            groups.append((down, down))
            remainder -= down
        return groups

    @staticmethod
    def _fill_static_inputs(
        statics: _WholeEulerStatics,
        *,
        graph_batch: int,
        start: int,
        stop: int,
        x: torch.Tensor,
        mu_cfg: torch.Tensor,
        speakers_cfg: torch.Tensor,
        cond_cfg: torch.Tensor,
        attn_mask: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        att_rows: list[torch.Tensor] | None,
        offset: int,
        lengths: torch.Tensor | int | None = None,
    ) -> None:
        """Copy requests ``start:stop`` into a graph's static inputs.

        CFG inputs are stacked ``[cond x B | uncond x B]``; ``unflatten`` into
        ``(2, B)`` moves both halves of a request range with one copy. Rows past
        ``stop - start`` belong to a padded replay and keep whatever finite
        values they held: DiT rows never mix, so their output is simply dropped.
        ``lengths`` is the ragged body's per-row lengths, or one for every row.
        Padded rows get length 0: every query width shares the buffer, and a
        wider replay's length would index past this one's causal history.
        """
        batch_size = int(x.shape[0])
        rows = stop - start
        statics.x[:rows].copy_(x[start:stop])
        pairs = [
            (statics.mu_cfg, mu_cfg),
            (statics.speakers_cfg, speakers_cfg),
            (statics.cond_cfg, cond_cfg),
            (statics.attn_mask, attn_mask),
        ]
        if statics.lengths is not None:
            assert lengths is not None
            if isinstance(lengths, int):
                statics.lengths.fill_(lengths)
            else:
                if rows < graph_batch:
                    statics.lengths.zero_()
                pairs.append((statics.lengths, lengths))
        for static, value in pairs:
            static.unflatten(0, (2, graph_batch))[:, :rows].copy_(value.unflatten(0, (2, batch_size))[:, start:stop])
        if cnn_cache is None:
            statics.cnn_cache.zero_()
        else:
            statics.cnn_cache.unflatten(2, (2, graph_batch))[:, :, :, :rows].copy_(
                cnn_cache.unflatten(2, (2, batch_size))[:, :, :, start:stop]
            )
        if offset <= 0:
            return
        # A graph on the ``_capture_offset`` grid holds at least ``offset``
        # cache frames; the rest are masked out and keep the finite values the
        # zero-initialized storage last held, like padded rows do.
        static_att_rows = statics.att_cache.unflatten(2, (2, graph_batch))[..., :offset, :]
        if att_rows is None:
            assert att_cache is not None
            static_att_rows[:, :, :, :rows].copy_(att_cache.unflatten(2, (2, batch_size))[:, :, :, start:stop])
        else:
            for row in range(rows):
                _copy_att_cache_range(static_att_rows[:, :, :, row], att_rows[start + row], 0, offset)

    def _group_fills(self, groups: list[tuple[int, int]], after=None, **inputs) -> list[Callable]:
        """Per replay group, a ``fill(statics)`` that copies the group's requests in, then runs ``after``."""
        fills = []
        start = 0
        for graph_batch, rows in groups:

            def fill(statics, *, _graph_batch=graph_batch, _start=start, _stop=start + rows):
                self._fill_static_inputs(statics, graph_batch=_graph_batch, start=_start, stop=_stop, **inputs)
                if after is not None:
                    after(statics, _graph_batch, _start, _stop)

            fills.append(fill)
            start += rows
        return fills

    def _statics(
        self,
        arena: WholeEulerExecutionArena,
        *,
        graph_batch: int,
        channels: int,
        query_cap: int,
        spk_dim: int,
        att_cache: torch.Tensor,
        key_frames: int,
        **slots: torch.Tensor,
    ) -> _WholeEulerStatics:
        return _WholeEulerStatics(
            x=arena.get_x_staging(graph_batch, channels, query_cap),
            mu_cfg=arena.get_mu_staging(graph_batch, channels, query_cap),
            speakers_cfg=arena.get_speakers_staging(graph_batch, spk_dim),
            cond_cfg=arena.get_cond_staging(graph_batch, channels, query_cap),
            cnn_cache=arena.get_cnn_cache_in(graph_batch),
            att_cache=att_cache,
            attn_mask=arena.get_mask_staging(graph_batch, query_cap, key_frames),
            time_embeddings=arena.get_time_embeddings(graph_batch),
            lengths=arena.get_lengths_staging(graph_batch) if self.ragged_body is not None else None,
            modulation=arena.get_modulation(self.modulation_fn) if self.modulation_fn is not None else None,
            **slots,
        )

    def _capture(
        self,
        key: tuple,
        *,
        graph_batch: int,
        channels: int,
        query_cap: int,
        spk_dim: int,
        offset: int,
        fill,
    ) -> tuple | None:
        arena = self.arena
        # Reserve the steady offset plus two query buckets, and the largest
        # graph batch, up front, so every offset on the way there
        # (304 -> 368 -> 400) and every batch size share one storage. Two
        # buckets because a final chunk carries the lookahead tail past one
        # (50 + 6 mel frames). The wide offset-0 prompt solve fits within that;
        # adding its width on top of the steady offset would hold ~300 frames
        # per row that nothing uses.
        capacity = max(offset + query_cap, max(offset, self._att_capacity) + 2 * self.query_bucket_frames)
        rows = self._arena_rows
        if self.att_slots:
            # Streaming chunks run from the slot pool; the arena only holds
            # what reaches it (the prompt solve), not a steady batch.
            capacity, rows = offset + query_cap, graph_batch
        if self._cache and not arena.att_cache_fits(max(graph_batch, rows), offset + query_cap):
            # The captured graphs hold views of the current storage: retire
            # them, or they keep it alive beside the larger replacement.
            self._flush()
        try:
            statics = self._statics(
                arena,
                graph_batch=graph_batch,
                channels=channels,
                query_cap=query_cap,
                spk_dim=spk_dim,
                # Right behind the current chunk in the output view below.
                att_cache=arena.att_cache_view(graph_batch, offset, start=query_cap, capacity=capacity, rows=rows),
                key_frames=offset + query_cap,
            )
            out_cnn_cache = arena.get_cnn_cache_out(graph_batch)
            # The output view aliases the input view; see WholeEulerExecutionArena.
            out_att_cache = arena.att_cache_view(graph_batch, offset + query_cap, capacity=capacity, rows=rows)
            fill(statics)
        except Exception:
            logger.warning("Failed to allocate static buffers for Whole-Euler graph key: %s", key, exc_info=True)
            self._unsupported.add(key)
            return None
        storage = arena._att
        return self._record(
            key,
            statics,
            out_cnn_cache,
            out_att_cache,
            graph_batch=graph_batch,
            where=f"att storage rows/frames={int(storage.shape[2]) // 2}/{int(storage.shape[4])}",
        )

    def _record(
        self,
        key: tuple,
        statics: _WholeEulerStatics,
        out_cnn_cache: torch.Tensor,
        out_att_cache: torch.Tensor,
        *,
        graph_batch: int,
        where: str,
    ) -> tuple | None:
        """Warm up and capture one graph over filled ``statics``; ``None`` (and disabled) on failure."""
        static_x = statics.x
        # The fused Euler step integrates static_x in place, so every warmup
        # and the capture itself start from the same initial state.
        initial_x = static_x.clone()
        loop_args = (statics, out_cnn_cache, out_att_cache)
        memory_before = _memory_snapshot(self.device)
        try:
            current_stream = torch.cuda.current_stream(self.device)
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream), torch.no_grad():
                for _ in range(3):
                    static_x.copy_(initial_x)
                    self._run_euler_loop(*loop_args, batch_size=graph_batch)
            current_stream.wait_stream(warmup_stream)
            static_x.copy_(initial_x)
            graph = CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_final_x = self._run_euler_loop(*loop_args, batch_size=graph_batch)
        except Exception:
            self._disable("capture failed", key)
            return None

        self._stats["captures"] += 1
        logger.info(
            "Captured Whole-Euler CFM CUDA Graph for shape %s (cache=%d/%d, %s, stats=%s)%s",
            key,
            len(self._cache) + len(self._slot_graphs) + 1,
            self.max_graphs,
            where,
            self.stats_snapshot(),
            _format_memory_delta(memory_before, _memory_snapshot(self.device)),
        )
        return (statics, static_final_x, out_cnn_cache, out_att_cache, graph)

    def _entry(
        self,
        *,
        graph_batch: int,
        query_cap: int,
        offset: int,
        x: torch.Tensor,
        spk_dim: int,
        fill,
    ) -> tuple | None:
        # One graph per (batch, capture query width, cache offset).
        # Presence of cnn/att/mask is data copied into static buffers, not a key.
        key = ("whole_euler", graph_batch, query_cap, offset, str(x.dtype), str(x.device))
        if key in self._unsupported:
            return None
        entry = self._cache.get(key)
        if entry is not None:
            self._stats["hits"] += 1
            return entry
        if len(self._cache) >= self.max_graphs:
            # Serving lazily captures keys precapture cannot enumerate (the
            # prompt solve). A flush would retire every graph and the arena,
            # so grow the budget instead, up to 4x the configured one.
            if len(self._cache) < 4 * self._configured_max_graphs:
                self.max_graphs = len(self._cache) + 1
                logger.warning(
                    "Whole-Euler lazy capture grew max_graphs to %d (cache=%d)",
                    self.max_graphs,
                    len(self._cache),
                )
            else:
                self._unsupported.add(key)
                return None
        entry = self._capture(
            key,
            graph_batch=graph_batch,
            channels=int(x.shape[1]),
            query_cap=query_cap,
            spk_dim=spk_dim,
            offset=offset,
            fill=fill,
        )
        if entry is not None:
            self._cache[key] = entry
        return entry

    def _group_entries(
        self, entry_fn: Callable[..., tuple | None], groups: list[tuple[int, int]], fills: list, **entry_args
    ) -> list[tuple] | None:
        """Every group's graph, acquired before any of them replays.

        A replay may update request caches in place, so no group may fall back
        to eager after an earlier one ran. Capturing a later group can flush the
        cache (``max_graphs``, arena growth) and retire an earlier group's
        graph; the groups are then acquired once more, and fit this time.
        """
        for _ in range(2):
            flushes = self._stats["flushes"]
            entries = []
            for (graph_batch, _), fill in zip(groups, fills, strict=True):
                entry = entry_fn(graph_batch=graph_batch, fill=fill, **entry_args)
                if entry is None:
                    return None
                entries.append(entry)
            if self._stats["flushes"] == flushes:
                return entries
        return None

    def _ensure_slot_pool(self, keep: tuple[int, int]) -> AttSlotPool | None:
        """The slot pool for streaming trim ``keep = (prefix, suffix)``, created on first use.

        A slot holds a steady cache (``prefix + suffix`` frames) plus the
        widest capture width, the largest chunk a steady cache takes.
        """
        if not self.att_slots or not self.enabled:
            return None
        prefix, suffix = int(keep[0]), int(keep[1])
        pool = self.slot_pool
        if pool is None:
            if prefix < 0 or suffix <= 0 or self.query_bucket_frames <= 1:
                return None
            arena = self.arena
            frames = _align_up(prefix + suffix + self.query_bucket_frames, _ATT_FRAME_ALIGN)
            try:
                pool = AttSlotPool(
                    n_timesteps=self.n_timesteps,
                    depth=arena.depth,
                    heads=arena.heads,
                    width=arena.att_width,
                    slots=self.att_slots,
                    frames=frames,
                    suffix=suffix,
                    device=self.device,
                    dtype=self.att_cache_dtype,
                )
            except torch.OutOfMemoryError:
                logger.warning(
                    "Whole-Euler slot pool (%d slots x %d frames) does not fit; using the arena", self.att_slots, frames
                )
                self.att_slots = 0
                return None
            self.slot_pool = pool
            logger.info(
                "Whole-Euler slot pool: %d slots x %d frames (%.2f GiB)", pool.slots, pool.frames, pool.bytes() / 2**30
            )
        return pool if pool.suffix == suffix else None

    def _slot_entry(self, *, graph_batch: int, query_cap: int, channels: int, spk_dim: int, fill) -> tuple | None:
        """The slot-pool graph of ``graph_batch`` rows at width ``query_cap``, captured on first use."""
        key = ("whole_euler_slots", graph_batch, query_cap)
        entry = self._slot_graphs.get(key)
        if entry is not None:
            self._stats["hits"] += 1
            return entry
        if key in self._unsupported:
            return None
        pool = self.slot_pool
        assert pool is not None
        arena = self._slot_arena
        try:
            statics = self._statics(
                arena,
                graph_batch=graph_batch,
                channels=channels,
                query_cap=query_cap,
                spk_dim=spk_dim,
                att_cache=pool.rows(),
                key_frames=pool.frames,
                slot_rows=arena._buffer("slot_rows", (2 * graph_batch,), dtype=torch.long),
                # Nothing is written until a replay says where.
                slot_positions=arena._buffer(
                    "slot_positions", (2 * graph_batch, query_cap), new=_minus_ones, dtype=torch.int32
                ),
            )
            out_cnn_cache = arena.get_cnn_cache_out(graph_batch)
            fill(statics)
        except Exception:
            logger.warning("Failed to allocate static buffers for Whole-Euler graph key: %s", key, exc_info=True)
            self._unsupported.add(key)
            return None
        entry = self._record(
            key,
            statics,
            out_cnn_cache,
            pool.rows(),
            graph_batch=graph_batch,
            where=f"slot pool {pool.slots} x {pool.frames} frames",
        )
        if entry is not None:
            self._slot_graphs[key] = entry
        return entry

    def _replay_slots(
        self,
        *,
        x_cap: torch.Tensor,
        mu_cap: torch.Tensor,
        cond_cap: torch.Tensor,
        speakers_cfg: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_rows: list,
        attn_mask: torch.Tensor | None,
        mel_width: int,
        query_cap: int,
        row_lengths: list[int],
        lengths: torch.Tensor | int,
        att_keep: tuple[int, int],
        groups: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[ResidentAttCache]] | None:
        """``replay`` on resident caches (``AttSlotPool``); ``None`` leaves the batch to the arena.

        Each row's new frames take free frames of its slot, its logical
        ``[current | cache]`` mask is scattered onto the slot's frames, the
        graph writes the new keys/values there and attends in place, and the
        streaming trim drops frames from the row's layout afterwards. Rows may
        hold caches of different lengths (``row_offsets``): the mask's cache
        columns then span the longest one, and a shorter row's extra columns
        map past its slot and are dropped.
        """
        pool = self._ensure_slot_pool(att_keep)
        if pool is None:
            return None
        if attn_mask is None and any(length != mel_width for length in row_lengths):
            # Without a mask the arena path lets every row attend all current
            # frames, a row's padding included; a slot never holds padding.
            return None
        batch_size = len(att_rows)
        offsets = [int(row.shape[4]) for row in att_rows]
        # Everything that can refuse the pool is checked before a slot changes.
        if min(offsets) < pool.suffix or any(
            offset + length > pool.frames for offset, length in zip(offsets, row_lengths, strict=True)
        ):
            return None
        seen: set[int] = set()
        adopt = []
        for row in att_rows:
            resident = isinstance(row, ResidentAttCache) and row.pool is pool and id(row) not in seen
            if resident:
                seen.add(id(row))
            elif row.dtype != pool.dtype:
                return None
            adopt.append(not resident)
        if sum(adopt) > pool.free_slots:
            return None
        handles: list[ResidentAttCache] = []
        for row, needs in zip(att_rows, adopt, strict=True):
            handle = pool.adopt(row.materialize() if isinstance(row, ResidentAttCache) else row) if needs else row
            assert handle is not None
            handles.append(handle)

        # Per CFG row, in one host buffer (one copy to the device): its pool
        # row, the slot frame each query writes (-1: none), and the slot frame
        # of each logical key (a row's current padding maps past the slot and
        # is dropped).
        rows_cfg, keys = 2 * batch_size, mel_width + max(offsets)
        host = np.full((rows_cfg, 1 + query_cap + keys), -1, dtype=np.int64)
        host[:, 1 + query_cap :] = pool.frames
        new_frames: list[np.ndarray] = []
        for row, (handle, length) in enumerate(zip(handles, row_lengths, strict=True)):
            free = np.ones(pool.frames, dtype=bool)
            free[: pool.suffix] = False
            free[handle.layout] = False
            frames = np.flatnonzero(free)[:length]
            new_frames.append(frames)
            halves = [row, row + batch_size]
            host[halves, 0] = [2 * handle.slot, 2 * handle.slot + 1]
            host[halves, 1 : 1 + length] = frames
            host[halves, 1 + query_cap : 1 + query_cap + length] = frames
            cached = handle.physical_frames()
            host[halves, 1 + query_cap + mel_width : 1 + query_cap + mel_width + len(cached)] = cached
        device = x_cap.device
        index = to_device_nonblocking(torch.from_numpy(host), device)
        slot_rows, positions, columns = index.split((1, query_cap, keys), dim=1)
        columns = columns.unsqueeze(1).expand(-1, mel_width, -1)
        slot_mask = torch.zeros((rows_cfg, query_cap, pool.frames + 1), dtype=torch.bool, device=device)
        if attn_mask is None:
            slot_mask[:, :mel_width].scatter_(2, columns, True)
        else:
            slot_mask[:, :mel_width].scatter_(2, columns, attn_mask[:, :mel_width, :keys])
        cfg_lengths = lengths if isinstance(lengths, torch.Tensor) else index_to_device(row_lengths * 2, device)
        slot_mask = _pad_invalid_queries(slot_mask[..., : pool.frames], cfg_lengths)
        request_rows = slot_rows.squeeze(1).unflatten(0, (2, batch_size))
        request_positions = positions.unflatten(0, (2, batch_size))

        def copy_slots(statics: _WholeEulerStatics, graph_batch: int, start: int, stop: int) -> None:
            count = stop - start
            static_rows = statics.slot_rows.unflatten(0, (2, graph_batch))
            static_positions = statics.slot_positions.unflatten(0, (2, graph_batch))
            static_rows[:, :count].copy_(request_rows[:, start:stop])
            static_positions[:, :count].copy_(request_positions[:, start:stop])
            if count < graph_batch:
                # Padded rows only read (the first row's slot) and never write.
                static_rows[:, count:].copy_(request_rows[:, start : start + 1].expand(-1, graph_batch - count))
                static_positions[:, count:].fill_(-1)

        fills = self._group_fills(
            groups,
            copy_slots,
            x=x_cap,
            mu_cfg=mu_cap,
            speakers_cfg=speakers_cfg,
            cond_cfg=cond_cap,
            attn_mask=slot_mask,
            cnn_cache=cnn_cache,
            att_cache=None,
            att_rows=None,
            offset=0,
            lengths=lengths,
        )
        entries = self._group_entries(
            self._slot_entry,
            groups,
            fills,
            query_cap=query_cap,
            channels=int(x_cap.shape[1]),
            spk_dim=int(speakers_cfg.shape[1]),
        )
        if entries is None:
            return None

        chunk_mel = x_cap.new_empty((batch_size, int(x_cap.shape[1]), mel_width))
        out_cnn = torch.empty(self._slot_arena._cnn_cache_shape(batch_size), device=device, dtype=self.dtype)
        out_cnn_rows = out_cnn.unflatten(2, (2, batch_size))
        start = 0
        for (graph_batch, rows), fill, entry in zip(groups, fills, entries, strict=True):
            stop = start + rows
            statics, static_final_x, out_cnn_cache, _, graph = entry
            fill(statics)
            graph.replay()
            chunk_mel[start:stop].copy_(static_final_x[:rows, :, :mel_width])
            out_cnn_rows[:, :, :, start:stop].copy_(out_cnn_cache.unflatten(2, (2, graph_batch))[:, :, :, :rows])
            start = stop

        for handle, frames in zip(handles, new_frames, strict=True):
            layout = [*frames.tolist(), *handle.layout]
            # The streaming frames the trim keeps are its first range.
            _, kept = _att_keep_ranges(len(layout) + pool.suffix, att_keep)[0]
            handle.layout = layout[:kept]
        self._stats["slot_replays"] += 1
        return chunk_mel, out_cnn, handles

    def _precapture_slots(self, *, channels: int, spk_dim: int) -> int:
        """Capture every slot-pool graph (batch grid x capture widths); the cache offset is not a key."""
        before = self._stats["captures"]
        for graph_batch in sorted(self._graph_batches(), reverse=True):
            for query_cap in self.query_widths:
                entry = self._slot_entry(
                    graph_batch=graph_batch,
                    query_cap=query_cap,
                    channels=channels,
                    spk_dim=spk_dim,
                    fill=self._precapture_fill,
                )
                if entry is None:
                    return self._stats["captures"] - before
        return self._stats["captures"] - before

    @staticmethod
    def _precapture_fill(statics: _WholeEulerStatics) -> None:
        # The zero-initialized staging buffers are finite; a ragged body also
        # needs lengths within its query width for the causal cache gather.
        if statics.lengths is not None:
            statics.lengths.fill_(int(statics.x.shape[2]))

    def precapture(
        self,
        *,
        offsets: Iterable[int],
        steady: int,
        channels: int,
        spk_dim: int,
        max_graph_bytes: int = 1 << 30,
        keep: tuple[int, int] | None = None,
    ) -> int:
        """Capture the graphs streams of one prompt need, before serving them.

        A capture while serving stalls every stream on the device (warmup
        solves plus the capture, 1-2 s), so every graph batch, capture width
        and cache offset (``offsets``, snapped onto the ``_capture_offset``
        grid anchored at ``steady``) is captured here instead. The cross
        product is bounded by ``max_graphs`` so no capture flushes another,
        and it stops once one graph holds more than ``max_graph_bytes`` of
        device memory beyond the shared arena (a graph normally holds a few
        MiB; the rest then capture on first use). Returns the number of
        graphs captured.

        With a slot pool (``att_slots``) and the streaming trim ``keep``,
        the streaming graphs are the slot-pool ones, one per batch and width.
        """
        if not self.enabled:
            return 0
        if self.att_slots and keep is not None and self._ensure_slot_pool(keep) is not None:
            return self._precapture_slots(channels=int(channels), spk_dim=int(spk_dim))
        self._att_capacity = max(self._att_capacity, int(steady))
        grid = sorted({_capture_offset(int(o), self.offset_bucket_frames, int(steady)) for o in offsets}, reverse=True)
        widths = self.query_widths
        if not widths:
            return 0
        keys = [(b, w, o) for b in sorted(self._graph_batches(), reverse=True) for w in widths for o in grid]
        room = self.max_graphs - len(self._cache)
        if len(keys) > room:
            # Grow the budget rather than truncate the sweep (max_graph_bytes
            # still bounds it), with a slot per width for the offset-0 prompt
            # solve the sweep cannot cover, so its lazy capture flushes nothing.
            self.max_graphs = len(self._cache) + len(keys) + len(widths)
            logger.warning(
                "Whole-Euler precapture needs %d graphs; raising max_graphs to %d",
                len(keys),
                self.max_graphs,
            )
        x = torch.empty((1, int(channels), 1), device=self.device, dtype=self.dtype)
        before = self._stats["captures"]
        for graph_batch, query_cap, offset in keys:
            memory_before, arena_before = _memory_snapshot(self.device), self.arena.att_bytes()
            entry = self._entry(
                graph_batch=graph_batch,
                query_cap=query_cap,
                offset=offset,
                x=x,
                spk_dim=int(spk_dim),
                fill=self._precapture_fill,
            )
            if entry is None:
                break
            memory_after = _memory_snapshot(self.device)
            if memory_before is not None and memory_after is not None:
                # What the graph itself holds, apart from the shared arena.
                held = memory_after[1] - memory_before[1] - (self.arena.att_bytes() - arena_before)
                if held > max_graph_bytes:
                    # A capture failure disables every graph (see _capture),
                    # so stop well before one could run the device dry.
                    logger.warning(
                        "Whole-Euler precapture stopped at %s: the graph reserved %.2f GiB (limit %.2f GiB); "
                        "the remaining graphs capture on first use",
                        (graph_batch, query_cap, offset),
                        held / 2**30,
                        max_graph_bytes / 2**30,
                    )
                    break
        return self._stats["captures"] - before

    def replay(
        self,
        *,
        x: torch.Tensor,
        mu_cfg: torch.Tensor,
        speakers_cfg: torch.Tensor,
        cond_cfg: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | list[torch.Tensor] | None,
        attn_mask: torch.Tensor | None = None,
        mel_frames: int | None = None,
        pad_frames: int = 0,
        valid_lengths: list[int] | None = None,
        att_keep: tuple[int, int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | list[torch.Tensor]] | None:
        """Solve the chunk through captured graphs; ``None`` sends it to the step/eager path.

        ``att_cache`` is the stacked ``(n_t, depth, 2B, heads, L, 2*head_dim)``
        estimator cache, or one ``(n_t, depth, 2, heads, L, 2*head_dim)``
        tensor per request; the new cache comes back in the same form, trimmed
        by ``att_keep`` (``_att_keep_ranges``) on the way out. Per request,
        caches move straight into and out of the arena and no stacked copy of
        the batch's cache is ever built. ``sum(att_keep)`` is also the steady
        cache length the arena reserves.

        Request caches are updated in place: holding the old and new cache of
        every request at once was the largest block of Stage-2 memory at high
        concurrency (~7 GiB at 16 requests). A new one is allocated with room
        for the steady length ``sum(att_keep)`` and returned as a view of its
        first frames, so it also grows in place (``_resized_frame_view``). The
        caller hands over those caches; a tensor shared by several rows (the
        prompt state) is never written.

        ``valid_lengths`` (one per request; needs ``ragged_body``) solves rows
        of different lengths in one graph, as ``_decode_cfm`` does eagerly:
        ``mu_cfg`` is padded to the longest row and ``attn_mask`` masks each
        row's padding. The new cache then always comes back per request, each
        row keeping only its own current frames.
        """
        self._stats["calls"] += 1

        if not self.enabled or torch.cuda.is_current_stream_capturing() or x.device.type != "cuda":
            return None

        batch_size = int(x.shape[0])
        if batch_size > self.max_graph_batch:
            return None
        groups = self._plan_groups(batch_size)

        att_rows = None if att_cache is None or isinstance(att_cache, torch.Tensor) else list(att_cache)
        # Per-request caches of different lengths only replay from the slot pool (``row_offsets``).
        mixed_offsets = att_rows is not None and len({int(row.shape[4]) for row in att_rows}) != 1
        if att_rows is not None:
            if len(att_rows) != batch_size or (mixed_offsets and not self.row_offsets):
                return None
            offset = int(att_rows[0].shape[4])
        else:
            offset = int(att_cache.shape[4]) if att_cache is not None else 0
        mel_width = int(mu_cfg.shape[2])
        att_rows_out = att_rows is not None or valid_lengths is not None
        if valid_lengths is not None:
            if self.ragged_body is None or len(valid_lengths) != batch_size:
                return None
            pad_frames = 0
            mel_frames = mel_width
        elif mel_frames is None:
            mel_frames = mel_width - pad_frames
        if att_keep is not None:
            self._att_capacity = max(self._att_capacity, sum(att_keep))

        query_cap = _capture_query_width(mel_width, self.query_widths)
        x_cap, mu_cap, cond_cap = _pad_query_for_capture(x, mu_cfg, cond_cfg, query_cap)
        row_lengths = [mel_width] * batch_size if valid_lengths is None else [int(n) for n in valid_lengths]
        lengths: torch.Tensor | int | None = None
        if self.ragged_body is not None:
            lengths = mel_width
            if valid_lengths is not None:
                lengths = torch.tensor((*row_lengths, *row_lengths), dtype=torch.long, device=x.device)
        if self.att_slots and att_rows is not None and pad_frames == 0 and att_keep is not None:
            slotted = self._replay_slots(
                x_cap=x_cap,
                mu_cap=mu_cap,
                cond_cap=cond_cap,
                speakers_cfg=speakers_cfg,
                cnn_cache=cnn_cache,
                att_rows=att_rows,
                attn_mask=attn_mask,
                mel_width=mel_width,
                query_cap=query_cap,
                row_lengths=row_lengths,
                lengths=lengths,
                att_keep=att_keep,
                groups=groups,
            )
            if slotted is not None:
                return slotted
        if mixed_offsets:
            return None
        offset_cap = _capture_offset(offset, self.offset_bucket_frames, sum(att_keep) if att_keep is not None else 0)
        mask_cap = _build_capture_mask(
            attn_mask=attn_mask,
            batch_size=batch_size,
            query_cap=query_cap,
            offset=offset,
            mel_width=mel_width,
            mel_frames=mel_frames,
            device=x.device,
            offset_cap=offset_cap,
        )
        if isinstance(lengths, torch.Tensor):
            # Queries past a row's length take its first query's mask, as
            # capture padding does.
            mask_cap = _pad_invalid_queries(mask_cap, lengths)
        segments_by_length = {
            length: _whole_euler_att_segments(mel_width=length, query_cap=query_cap, offset=offset, keep=att_keep)
            for length in set(row_lengths)
        }
        segments = segments_by_length[row_lengths[0]]
        keep_len = sum(length for _, length in segments)

        arena = self.arena
        chunk_mel = x.new_empty((batch_size, int(x.shape[1]), mel_frames))
        out_cnn = torch.empty(arena._cnn_cache_shape(batch_size), device=x.device, dtype=self.dtype)
        out_cnn_rows = out_cnn.unflatten(2, (2, batch_size))
        out_att: torch.Tensor | None = None
        out_att_rows = None
        request_att: list[torch.Tensor] = []
        if not att_rows_out:
            out_att = torch.empty(
                (self.n_timesteps, arena.depth, 2 * batch_size, arena.heads, keep_len, arena.att_width),
                device=x.device,
                dtype=self.att_cache_dtype,
            )
            out_att_rows = out_att.unflatten(2, (2, batch_size))
        # A ragged body writes each row's exact CNN cache; only the bucket
        # padding (``pad_frames``) is then cleared. Otherwise every padded position is.
        cnn_pad = pad_frames if self.ragged_body is not None else query_cap - mel_frames

        fills = self._group_fills(
            groups,
            x=x_cap,
            mu_cfg=mu_cap,
            speakers_cfg=speakers_cfg,
            cond_cfg=cond_cap,
            attn_mask=mask_cap,
            cnn_cache=cnn_cache,
            att_cache=att_cache if att_rows is None else None,
            att_rows=att_rows,
            offset=offset,
            lengths=lengths,
        )
        entries = self._group_entries(
            self._entry,
            groups,
            fills,
            query_cap=query_cap,
            offset=offset_cap,
            x=x,
            spk_dim=int(speakers_cfg.shape[1]),
        )
        if entries is None:
            return None
        owners = Counter(row.data_ptr() for row in att_rows) if att_rows is not None else Counter()
        steady_frames = sum(att_keep) if att_keep is not None else 0

        start = 0
        for (graph_batch, rows), fill, entry in zip(groups, fills, entries, strict=True):
            stop = start + rows
            statics, static_final_x, out_cnn_cache, out_att_cache, graph = entry
            fill(statics)
            graph.replay()

            chunk_mel[start:stop].copy_(static_final_x[:rows, :, :mel_frames])
            if cnn_pad > 0:
                # (n_timesteps, depth, ...): put the block axis first for the helper.
                _zero_padded_cnn_cache(out_cnn_cache.transpose(0, 1), self.estimator, cnn_pad)
            out_cnn_rows[:, :, :, start:stop].copy_(out_cnn_cache.unflatten(2, (2, graph_batch))[:, :, :, :rows])
            att_src = out_att_cache.unflatten(2, (2, graph_batch))[:, :, :, :rows]
            if pad_frames > 0:
                # The static output is scratch until the next replay, so the
                # padded frames are cleared once here rather than per copy.
                att_src[..., mel_frames:mel_width, :] = 0.0
            if out_att_rows is not None:
                _copy_frame_segments(out_att_rows[:, :, :, start:stop], att_src, segments)
            else:
                for row in range(rows):
                    row_segments = segments_by_length[row_lengths[start + row]]
                    shape = (
                        self.n_timesteps,
                        arena.depth,
                        2,
                        arena.heads,
                        sum(length for _, length in row_segments),
                        arena.att_width,
                    )
                    # The row's old cache was copied into the arena by ``fill``.
                    old = att_rows[start + row] if att_rows is not None else None
                    request = None
                    if (
                        isinstance(old, torch.Tensor)
                        and old.dtype == self.att_cache_dtype
                        and owners[old.data_ptr()] == 1
                    ):
                        request = _resized_frame_view(old, shape[4])
                    if request is None:
                        # Room for the steady length, so the cache grows into it in place.
                        room = max(shape[4], steady_frames)
                        request = torch.empty(
                            (*shape[:4], room, shape[5]), device=x.device, dtype=self.att_cache_dtype
                        )[..., : shape[4], :]
                    _copy_frame_segments(request, att_src[:, :, :, row], row_segments)
                    request_att.append(request)
            start = stop

        return chunk_mel, out_cnn, (request_att if att_rows_out else out_att)
