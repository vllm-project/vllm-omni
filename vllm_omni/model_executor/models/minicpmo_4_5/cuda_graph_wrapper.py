# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from collections import Counter
from collections.abc import Callable
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

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


class HiFTGraphWrapper:
    def __init__(self, token2wav, connector_config, capture_batch_sizes, max_serial_batch: int | None = None):
        self.decode_fn = token2wav.hift.inference
        self.graph_fn = token2wav.hift._inference_pre_istft
        self.finalize_fn = token2wav.hift._finalize_decode
        self.codec_chunk_frames = connector_config["codec_chunk_frames"]
        self.codec_left_context_frames = connector_config["codec_left_context_frames"]
        lookahead_layer = getattr(token2wav.flow.encoder, "pre_lookahead_layer", None)
        pre_lookahead_len = getattr(lookahead_layer, "pre_lookahead_len", None)
        self.pre_lookahead_len = int(pre_lookahead_len) if pre_lookahead_len is not None else 3
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.mel_frames = int(token2wav.hift.conv_pre.in_channels)
        self.flow_upsample_rate = int(getattr(token2wav.flow, "token_mel_ratio", 2))
        self.capture_bucket_size, self.capture_source_cache_len = self.derive_capture_bucket_size()
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
        self.max_lazy_graphs = 8
        self.lazy_graph_count = 0
        self.max_serial_batch = 4 if max_serial_batch is None else int(max_serial_batch)

    def derive_capture_bucket_size(self):
        chunk_mel_frames = (
            self.codec_chunk_frames + self.codec_left_context_frames - self.pre_lookahead_len
        ) * self.flow_upsample_rate

        return [chunk_mel_frames, chunk_mel_frames + self.mel_cache_len], [
            0,
            self.source_cache_len,
        ]

    def capture(self):
        for batch_size in self.capture_batch_sizes:
            for mel_frames, source_cache_len in zip(
                self.capture_bucket_size,
                self.capture_source_cache_len,
                strict=True,
            ):
                self._capture(batch_size, mel_frames, source_cache_len)

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
        if torch.cuda.is_current_stream_capturing():
            logger.info("Falling back to eager HiFT inference during an active stream capture")
            return self.decode_fn(speech_feat, cache_source)

        batch_size = speech_feat.shape[0]
        num_frames = speech_feat.shape[2]
        cache_source_len = cache_source.shape[2]
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


def _align_up(n: int, bucket: int) -> int:
    if n <= 0 or bucket <= 1:
        return max(0, n)
    return ((n + bucket - 1) // bucket) * bucket


def _capture_query_width(mel_width: int, bucket: int) -> int:
    """Snap the query axis onto the Whole-Euler capture grid.

    ``_cfm_pad_frames`` already aligns onto ``bucket_frames=16``, which is why
    12+4 and 10+6 share a graph (``test_whole_euler_same_bucket_different_padding_hits_cache``)
    but 16/32/48/64 stay distinct (``test_varied_chunk_lengths_collapse_onto_few_widths``).
    The capture bucket reuses that same pad-and-mask scheme so those
    decode widths collapse onto one CUDA graph; first-chunk widths above the
    bucket (e.g. 304) align up separately so streaming replay stays on the
    small decode graph.
    """
    if bucket <= 1:
        return int(mel_width)
    if mel_width <= bucket:
        return int(bucket)
    return _align_up(int(mel_width), int(bucket))


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
    mask = torch.zeros(
        2 * int(batch_size),
        query_cap,
        query_cap + offset,
        dtype=torch.bool,
        device=device,
    )
    if attn_mask is not None:
        mask[:, :mel_width, :mel_width] = attn_mask[:, :mel_width, :mel_width]
        if offset > 0:
            mask[:, :mel_width, query_cap:] = attn_mask[:, :mel_width, mel_width : mel_width + offset]
    else:
        mask[:, :mel_width, : int(mel_frames)] = True
        if offset > 0:
            mask[:, :mel_width, query_cap:] = True
    if query_cap > mel_width:
        mask[:, mel_width:] = mask[:, :1]
    return mask


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
    blocks = getattr(estimator, "blocks", None)
    if pad_frames > 0 and blocks and len(blocks) == cnn_cache.shape[0]:
        width = int(cnn_cache.shape[-1])
        if width > 0 and all(
            hasattr(block, "conv")
            and hasattr(block.conv, "block")
            and len(block.conv.block) > 1
            and hasattr(block.conv.block[1], "causal_padding")
            and int(block.conv.block[1].causal_padding[0]) == width
            for block in blocks
        ):
            # _estimator_buffers packs equal-width blocks into one tensor.
            # Clear their shared tail with one write instead of one per block.
            cnn_cache[..., max(0, width - pad_frames) :] = 0.0
            return

    if blocks is not None:
        for index, block in enumerate(blocks):
            if index >= len(cnn_cache):
                break
            conv = getattr(block, "conv", None)
            conv_block = getattr(conv, "block", None) if conv is not None else None
            if conv_block is None or len(conv_block) <= 1 or not hasattr(conv_block[1], "causal_padding"):
                continue
            width = int(conv_block[1].causal_padding[0])
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
        time_steps: list[torch.Tensor] | None = None,
    ) -> None:
        self.estimator = estimator
        self.n_timesteps = int(n_timesteps)
        self.device = device
        self.dtype = dtype
        self.att_cache_dtype = att_cache_dtype
        self.time_steps = list(time_steps) if time_steps is not None else []

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
                    self.estimator.t_embedder(self.time_steps[s].expand(2 * batch_size)).unsqueeze(1)
                    for s in range(self.n_timesteps)
                ],
                dim=0,
            )
            self._buffers[key] = t_emb
        return t_emb

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

    def clear(self) -> None:
        self._buffers.clear()
        self._att = None


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
    (sized for ``micro_batch_size`` requests), and larger batches run as a
    sequence of micro-batch replays that copy each request's cache in and out
    of that storage directly.
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
        query_bucket_frames: int | None = None,
        pad_max_rows: int | None = None,
        ragged_body: Callable[..., torch.Tensor] | None = None,
    ) -> None:
        """``ragged_body(estimator, input, t_emb, mask, cnn, att, cnn_out, att_out, lengths)``
        replaces ``estimator.blocks_forward_chunk`` in the captured solve when
        given: it takes each row's valid query length as a ``(2B,)`` tensor and
        writes that row's exact CNN cache (``_blocks_forward_chunk_ragged``).
        It lets one graph solve rows of different lengths, and lets a padded
        query axis keep the CNN cache the unpadded solve would produce instead
        of zeroing it.
        """
        self.estimator = estimator
        self.ragged_body = ragged_body
        self.n_timesteps = int(n_timesteps)
        self.inference_cfg_rate = float(inference_cfg_rate)
        self.att_cache_dtype = att_cache_dtype
        self.max_graphs = int(max_graphs)
        self.query_bucket_frames = 0 if query_bucket_frames is None else int(query_bucket_frames)
        self.max_serial_batch = 4 if max_serial_batch is None else int(max_serial_batch)
        self.micro_batch_size = 4 if micro_batch_size is None else int(micro_batch_size)
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
        }

        self.timeline, self.dt_steps = _euler_timeline(self.n_timesteps, self.device, self.dtype)
        self.time_steps = [self.timeline[i] for i in range(self.n_timesteps)]
        self.arena = WholeEulerExecutionArena(
            estimator=self.estimator,
            n_timesteps=self.n_timesteps,
            device=self.device,
            dtype=self.dtype,
            att_cache_dtype=self.att_cache_dtype,
            time_steps=self.time_steps,
        )

    def stats_snapshot(self) -> dict[str, int]:
        """Bounded cumulative telemetry for the graph cache."""
        return {**self._stats, "cache_size": len(self._cache)}

    def _flush(self) -> None:
        """Retire every captured graph at once."""
        torch.accelerator.synchronize(self.device)
        if self._cache:
            for entry in self._cache.values():
                entry[4].reset()
            self._cache.clear()
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
        speaker_features = statics.speakers_cfg.unsqueeze(-1).expand(-1, -1, width)

        for step in range(self.n_timesteps):
            dt = self.dt_steps[step]
            x_cfg = torch.cat((cur_x, cur_x), dim=0)
            estimator_input = torch.cat((x_cfg, statics.mu_cfg, speaker_features, statics.cond_cfg), dim=1)
            args = (
                estimator_input,
                statics.time_embeddings[step],
                statics.attn_mask,
                statics.cnn_cache[step],
                statics.att_cache[step],
                out_cnn_cache[step],
                out_att_cache[step],
            )
            if statics.lengths is not None:
                assert self.ragged_body is not None
                estimate = self.ragged_body(self.estimator, *args, statics.lengths)
            else:
                estimate = self.estimator.blocks_forward_chunk(*args)

            cur_x = _euler_step(cur_x, estimate, dt, self.inference_cfg_rate, batch_size)

        return cur_x

    def _graph_batches(self) -> list[int]:
        """Native graph batch sizes: the powers of two below ``micro_batch_size``, and it."""
        sizes = []
        size = 1
        while size < self.micro_batch_size:
            sizes.append(size)
            size *= 2
        sizes.append(self.micro_batch_size)
        return sizes

    def _plan_groups(self, batch_size: int) -> list[tuple[int, int]] | None:
        """``(graph_batch, rows)`` replays that cover ``batch_size`` requests.

        Larger batches run whole micro-batches. The remainder takes the smallest
        native size that holds it if that pads at most ``pad_max_rows`` rows
        (the padded rows are ignored), and otherwise the largest native size
        below it, repeating on what is left. On a saturated device a small
        graph's replay costs well over its share of a larger one, so a padded
        row is cheaper than a further replay. Returns ``None`` when the plan
        needs more replays than allowed.
        """
        micro = self.micro_batch_size
        sizes = self._graph_batches()
        groups: list[tuple[int, int]] = []
        remainder = batch_size
        while remainder >= micro:
            groups.append((micro, micro))
            remainder -= micro
        while remainder > 0:
            up = next(size for size in sizes if size >= remainder)
            if up - remainder <= self.pad_max_rows:
                groups.append((up, remainder))
                break
            down = max(size for size in sizes if size <= remainder)
            groups.append((down, down))
            remainder -= down
        max_allowed = max(
            self.max_serial_batch,
            (self.max_graph_batch + micro - 1) // micro + (micro - 1),
        )
        if len(groups) > max_allowed:
            return None
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
        static_att_rows = statics.att_cache.unflatten(2, (2, graph_batch))
        if att_rows is None:
            assert att_cache is not None
            static_att_rows[:, :, :, :rows].copy_(att_cache.unflatten(2, (2, batch_size))[:, :, :, start:stop])
        else:
            for row in range(rows):
                static_att_rows[:, :, :, row].copy_(att_rows[start + row])

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
        rows = self.micro_batch_size
        if self._cache and not arena.att_cache_fits(graph_batch, offset + query_cap):
            # The captured graphs hold views of the current storage: retire
            # them, or they keep it alive beside the larger replacement.
            self._flush()
        try:
            statics = _WholeEulerStatics(
                x=arena.get_x_staging(graph_batch, channels, query_cap),
                mu_cfg=arena.get_mu_staging(graph_batch, channels, query_cap),
                speakers_cfg=arena.get_speakers_staging(graph_batch, spk_dim),
                cond_cfg=arena.get_cond_staging(graph_batch, channels, query_cap),
                cnn_cache=arena.get_cnn_cache_in(graph_batch),
                # Right behind the current chunk in the output view below.
                att_cache=arena.att_cache_view(graph_batch, offset, start=query_cap, capacity=capacity, rows=rows),
                attn_mask=arena.get_mask_staging(graph_batch, query_cap, offset + query_cap),
                time_embeddings=arena.get_time_embeddings(graph_batch),
                lengths=arena.get_lengths_staging(graph_batch) if self.ragged_body is not None else None,
            )
            out_cnn_cache = arena.get_cnn_cache_out(graph_batch)
            # The output view aliases the input view; see WholeEulerExecutionArena.
            out_att_cache = arena.att_cache_view(graph_batch, offset + query_cap, capacity=capacity, rows=rows)
            fill(statics)
        except Exception:
            logger.warning("Failed to allocate static buffers for Whole-Euler graph key: %s", key, exc_info=True)
            self._unsupported.add(key)
            return None

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
            "Captured Whole-Euler CFM CUDA Graph for shape %s (cache=%d/%d, att storage rows/frames=%d/%d, stats=%s)%s",
            key,
            len(self._cache) + 1,
            self.max_graphs,
            int(arena._att.shape[2]) // 2,
            int(arena._att.shape[4]),
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
            self._flush()
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

    def _group_entries(self, groups: list[tuple[int, int]], fills: list, **entry_args) -> list[tuple] | None:
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
                entry = self._entry(graph_batch=graph_batch, fill=fill, **entry_args)
                if entry is None:
                    return None
                entries.append(entry)
            if self._stats["flushes"] == flushes:
                return entries
        return None

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
        if groups is None:
            return None

        att_rows = None if att_cache is None or isinstance(att_cache, torch.Tensor) else list(att_cache)
        if att_rows is not None:
            if len(att_rows) != batch_size or len({int(row.shape[4]) for row in att_rows}) != 1:
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

        query_cap = _capture_query_width(mel_width, self.query_bucket_frames)
        x_cap, mu_cap, cond_cap = _pad_query_for_capture(x, mu_cfg, cond_cfg, query_cap)
        mask_cap = _build_capture_mask(
            attn_mask=attn_mask,
            batch_size=batch_size,
            query_cap=query_cap,
            offset=offset,
            mel_width=mel_width,
            mel_frames=mel_frames,
            device=x.device,
        )
        row_lengths = [mel_width] * batch_size if valid_lengths is None else [int(n) for n in valid_lengths]
        lengths: torch.Tensor | int | None = None
        if self.ragged_body is not None:
            lengths = mel_width
            if valid_lengths is not None:
                lengths = torch.tensor((*row_lengths, *row_lengths), dtype=torch.long, device=x.device)
                # Queries past a row's length take its first query's mask, as
                # capture padding does, so no attention row is left empty.
                invalid = torch.arange(query_cap, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
                mask_cap = torch.where(invalid.unsqueeze(2), mask_cap[:, :1, :], mask_cap)
        segments_by_length = {
            length: _whole_euler_att_segments(mel_width=length, query_cap=query_cap, offset=offset, keep=att_keep)
            for length in set(row_lengths)
        }
        segments = segments_by_length[row_lengths[0]]
        keep_len = sum(length for _, length in segments)

        arena = self.arena
        chunk_mel = x.new_empty((batch_size, int(x.shape[1]), mel_frames))
        out_cnn = torch.empty(
            (self.n_timesteps, arena.depth, 2 * batch_size, arena.cnn_channels, arena.cnn_width),
            device=x.device,
            dtype=self.dtype,
        )
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

        fills = []
        start = 0
        for graph_batch, rows in groups:
            stop = start + rows

            def fill(statics, *, _graph_batch=graph_batch, _start=start, _stop=stop):
                self._fill_static_inputs(
                    statics,
                    graph_batch=_graph_batch,
                    start=_start,
                    stop=_stop,
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

            fills.append(fill)
            start = stop
        entries = self._group_entries(
            groups,
            fills,
            query_cap=query_cap,
            offset=offset,
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
                    if old is not None and old.dtype == self.att_cache_dtype and owners[old.data_ptr()] == 1:
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
