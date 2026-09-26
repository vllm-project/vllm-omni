# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import os

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:
    triton = None
    tl = None
    _HAS_TRITON = False

if _HAS_TRITON:

    @triton.jit
    def _fused_euler_step_kernel(
        x_ptr,
        estimate_ptr,
        dt,
        cfg_rate,
        n_elements,
        block_size: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        cond = tl.load(estimate_ptr + offsets, mask=mask)
        uncond = tl.load(estimate_ptr + n_elements + offsets, mask=mask)

        v = (1.0 + cfg_rate) * cond - cfg_rate * uncond
        new_x = x + dt * v

        tl.store(x_ptr + offsets, new_x.to(x.dtype), mask=mask)


def _fused_euler_step(
    cur_x: torch.Tensor,
    estimate: torch.Tensor,
    dt: float,
    inference_cfg_rate: float,
    batch_size: int,
) -> torch.Tensor:
    """In-place Euler ODE update: cur_x = cur_x + dt * ((1+cfg)*cond - cfg*uncond).

    Uses Triton kernel when available on CUDA for a single fused launch without DRAM roundtrips.
    Falls back to standard PyTorch ops when Triton is unavailable or tensors are non-contiguous.
    """
    if _HAS_TRITON and cur_x.is_cuda and cur_x.is_contiguous() and estimate.is_contiguous():
        n_elements = cur_x.numel()
        block_size = 1024
        grid = (triton.cdiv(n_elements, block_size),)
        _fused_euler_step_kernel[grid](
            cur_x,
            estimate,
            float(dt),
            float(inference_cfg_rate),
            n_elements,
            block_size=block_size,
        )
        return cur_x

    conditional, unconditional = estimate.split(batch_size, dim=0)
    velocity = (1.0 + inference_cfg_rate) * conditional - inference_cfg_rate * unconditional
    return cur_x + dt * velocity


class HiFTGraphWrapper:
    max_serial_batch: int = 4

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
        if max_serial_batch is None:
            self.max_serial_batch = int(os.getenv("VLLM_OMNI_MAX_GRAPH_SERIAL_BATCH", "4"))
        else:
            self.max_serial_batch = int(max_serial_batch)

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

    max_serial_batch: int = 4

    def __init__(
        self,
        graph_fn,
        *,
        max_graphs: int = 32,
        max_serial_batch: int | None = None,
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
        if max_serial_batch is None:
            self.max_serial_batch = int(os.getenv("VLLM_OMNI_MAX_GRAPH_SERIAL_BATCH", "4"))
        else:
            self.max_serial_batch = int(max_serial_batch)

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

        batch_size = estimator_input.shape[0] // 2
        if (
            not self.enabled
            or torch.cuda.is_current_stream_capturing()
            or estimator_input.device.type != "cuda"
            or batch_size > self.max_serial_batch
        ):
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
    """Clear the cache positions that come from padded frames, in place."""
    if pad_frames <= 0:
        return
    width = cnn_cache.shape[-1]
    if width <= 0:
        return
    zero_from = max(0, width - pad_frames)
    if zero_from < width:
        cnn_cache[..., zero_from:] = 0.0


class WholeEulerExecutionArena:
    """Decoupled memory arena managing static staging buffers for Whole-Euler CFM.

    Shared static buffers (time embeddings, cnn input/output cache, speakers) are
    allocated once per CFM execution lane rather than duplicated across every CUDA Graph
    cache entry. Variable-shape staging buffers (x, mu, cond, att_cache) are reused
    across graph entries sharing matching dimensions.
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

        # Shared static staging buffers (allocated once per lane, shared across shapes)
        self._shared_time_embeddings: dict[int, torch.Tensor] = {}
        self._shared_cnn_in: dict[int, torch.Tensor] = {}
        self._shared_cnn_out: dict[int, torch.Tensor] = {}
        self._shared_speakers: dict[tuple[int, int], torch.Tensor] = {}

        # Shape-keyed variable staging buffers (reused when dimensions match)
        self._x_buffers: dict[tuple[int, int, int], torch.Tensor] = {}
        self._mu_buffers: dict[tuple[int, int, int], torch.Tensor] = {}
        self._cond_buffers: dict[tuple[int, int, int], torch.Tensor] = {}
        self._att_cache_in: dict[tuple[int, int], torch.Tensor] = {}
        self._att_cache_out: dict[tuple[int, int], torch.Tensor] = {}
        self._mask_buffers: dict[tuple[int, int, int], torch.Tensor] = {}

    def get_time_embeddings(self, batch_size: int) -> torch.Tensor:
        t_emb = self._shared_time_embeddings.get(batch_size)
        if t_emb is None:
            t_emb = torch.stack(
                [
                    self.estimator.t_embedder(self.time_steps[s].expand(2 * batch_size)).unsqueeze(1)
                    for s in range(self.n_timesteps)
                ],
                dim=0,
            )
            self._shared_time_embeddings[batch_size] = t_emb
        return t_emb

    def get_cnn_cache_in(self, batch_size: int) -> torch.Tensor:
        buf = self._shared_cnn_in.get(batch_size)
        if buf is None:
            buf = torch.zeros(
                self.n_timesteps,
                self.depth,
                2 * batch_size,
                self.cnn_channels,
                self.cnn_width,
                device=self.device,
                dtype=self.dtype,
            )
            self._shared_cnn_in[batch_size] = buf
        return buf

    def get_cnn_cache_out(self, batch_size: int) -> torch.Tensor:
        buf = self._shared_cnn_out.get(batch_size)
        if buf is None:
            buf = torch.empty(
                self.n_timesteps,
                self.depth,
                2 * batch_size,
                self.cnn_channels,
                self.cnn_width,
                device=self.device,
                dtype=self.dtype,
            )
            self._shared_cnn_out[batch_size] = buf
        return buf

    def get_speakers_staging(self, batch_size: int, spk_dim: int) -> torch.Tensor:
        key = (batch_size, spk_dim)
        buf = self._shared_speakers.get(key)
        if buf is None:
            buf = torch.zeros(2 * batch_size, spk_dim, device=self.device, dtype=self.dtype)
            self._shared_speakers[key] = buf
        return buf

    def get_x_staging(self, batch_size: int, channels: int, mel_width: int) -> torch.Tensor:
        key = (batch_size, channels, mel_width)
        buf = self._x_buffers.get(key)
        if buf is None:
            buf = torch.zeros(batch_size, channels, mel_width, device=self.device, dtype=self.dtype)
            self._x_buffers[key] = buf
        return buf

    def get_mu_staging(self, batch_size: int, channels: int, mel_width: int) -> torch.Tensor:
        key = (batch_size, channels, mel_width)
        buf = self._mu_buffers.get(key)
        if buf is None:
            buf = torch.zeros(2 * batch_size, channels, mel_width, device=self.device, dtype=self.dtype)
            self._mu_buffers[key] = buf
        return buf

    def get_cond_staging(self, batch_size: int, channels: int, mel_width: int) -> torch.Tensor:
        key = (batch_size, channels, mel_width)
        buf = self._cond_buffers.get(key)
        if buf is None:
            buf = torch.zeros(2 * batch_size, channels, mel_width, device=self.device, dtype=self.dtype)
            self._cond_buffers[key] = buf
        return buf

    def get_att_cache_in(self, batch_size: int, offset: int) -> torch.Tensor:
        key = (batch_size, offset)
        buf = self._att_cache_in.get(key)
        if buf is None:
            buf = torch.zeros(
                self.n_timesteps,
                self.depth,
                2 * batch_size,
                self.heads,
                offset,
                self.att_width,
                device=self.device,
                dtype=self.att_cache_dtype,
            )
            self._att_cache_in[key] = buf
        return buf

    def get_att_cache_out(self, batch_size: int, total_len: int) -> torch.Tensor:
        key = (batch_size, total_len)
        buf = self._att_cache_out.get(key)
        if buf is None:
            buf = torch.empty(
                self.n_timesteps,
                self.depth,
                2 * batch_size,
                self.heads,
                total_len,
                self.att_width,
                device=self.device,
                dtype=self.att_cache_dtype,
            )
            self._att_cache_out[key] = buf
        return buf

    def get_mask_staging(self, batch_size: int, mel_width: int, total_len: int) -> torch.Tensor:
        key = (batch_size, mel_width, total_len)
        buf = self._mask_buffers.get(key)
        if buf is None:
            buf = torch.ones(
                2 * batch_size,
                mel_width,
                total_len,
                dtype=torch.bool,
                device=self.device,
            )
            self._mask_buffers[key] = buf
        return buf

    def clear(self) -> None:
        self._shared_time_embeddings.clear()
        self._shared_cnn_in.clear()
        self._shared_cnn_out.clear()
        self._shared_speakers.clear()
        self._x_buffers.clear()
        self._mu_buffers.clear()
        self._cond_buffers.clear()
        self._att_cache_in.clear()
        self._att_cache_out.clear()
        self._mask_buffers.clear()


class WholeEulerCFMGraphWrapper:
    """Per-shape CUDA graph capture/replay for the entire 10-step CFM Euler ODE solver.

    Captures all 10 iterations of:
    (time embedding -> in_proj -> DiT blocks -> final layer -> CFG combination -> Euler step)
    into a single CUDA Graph replay.

    Replaces 10 separate step-level graph replays and 30 Python clones per chunk
    with a single graph replay, reducing CPU launch overhead and eliminating host-device
    synchronization bubbles during high concurrency.
    """

    max_serial_batch: int = 4

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
    ) -> None:
        self.estimator = estimator
        self.n_timesteps = int(n_timesteps)
        self.inference_cfg_rate = float(inference_cfg_rate)
        self.att_cache_dtype = att_cache_dtype
        self.max_graphs = int(max_graphs)
        if max_serial_batch is None:
            self.max_serial_batch = int(os.getenv("VLLM_OMNI_MAX_GRAPH_SERIAL_BATCH", "4"))
        else:
            self.max_serial_batch = int(max_serial_batch)
        if micro_batch_size is not None:
            self.micro_batch_size = int(micro_batch_size)
        else:
            self.micro_batch_size = int(os.getenv("VLLM_OMNI_GRAPH_MICRO_BATCH_SIZE", "4"))
        if max_graph_batch is None:
            if max_serial_batch is not None and max_serial_batch < self.micro_batch_size:
                self.max_graph_batch = int(max_serial_batch)
            else:
                self.max_graph_batch = int(os.getenv("VLLM_OMNI_MAX_GRAPH_BATCH", "16"))
        else:
            self.max_graph_batch = int(max_graph_batch)
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

        timeline = torch.linspace(
            0,
            1,
            self.n_timesteps + 1,
            device=self.device,
            dtype=self.dtype,
        )
        self.timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        self.dt_steps = [float((self.timeline[i + 1] - self.timeline[i]).item()) for i in range(self.n_timesteps)]
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
        if not self._cache:
            return
        torch.accelerator.synchronize(self.device)
        for entry in self._cache.values():
            entry[4].reset()
        self._cache.clear()
        self.arena.clear()
        self._stats["flushes"] += 1
        logger.info("Whole-Euler CFM graph cache flushed; stats=%s", self.stats_snapshot())

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
        static_x: torch.Tensor,
        static_mu_cfg: torch.Tensor,
        static_speakers_cfg: torch.Tensor,
        static_cond_cfg: torch.Tensor,
        static_cnn_cache: torch.Tensor,
        static_att_cache: torch.Tensor,
        static_attn_mask: torch.Tensor | None,
        out_cnn_cache: torch.Tensor,
        out_att_cache: torch.Tensor,
        static_time_embeddings: torch.Tensor,
        *,
        batch_size: int,
        has_cnn_cache: bool,
        has_att_cache: bool,
    ) -> torch.Tensor:
        cur_x = static_x
        depth = len(self.estimator.blocks)
        width = int(static_mu_cfg.shape[2])
        speaker_features = static_speakers_cfg.unsqueeze(-1).expand(-1, -1, width)

        for step in range(self.n_timesteps):
            dt = self.dt_steps[step]
            time_embedding = static_time_embeddings[step]
            x_cfg = torch.cat((cur_x, cur_x), dim=0)
            estimator_input = torch.cat((x_cfg, static_mu_cfg, speaker_features, static_cond_cfg), dim=1)

            step_cnn_out = out_cnn_cache[step]
            step_att_out = out_att_cache[step]
            old_cnn = static_cnn_cache[step] if has_cnn_cache else [None] * depth
            old_att = static_att_cache[step] if has_att_cache else [None] * depth

            estimate = self.estimator.blocks_forward_chunk(
                estimator_input,
                time_embedding,
                static_attn_mask,
                old_cnn,
                old_att,
                step_cnn_out,
                step_att_out,
            )

            cur_x = _fused_euler_step(cur_x, estimate, dt, self.inference_cfg_rate, batch_size)

        return cur_x

    def _capture(
        self,
        key: tuple,
        *,
        x: torch.Tensor,
        mu_cfg: torch.Tensor,
        speakers_cfg: torch.Tensor,
        cond_cfg: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        batch_size: int,
        offset: int,
        has_cnn_cache: bool,
        has_att_cache: bool,
        has_mask: bool,
    ) -> tuple | None:
        try:
            channels = int(x.shape[1])
            mel_width = int(mu_cfg.shape[2])
            spk_dim = int(speakers_cfg.shape[1])

            static_x = self.arena.get_x_staging(batch_size, channels, mel_width)
            static_mu_cfg = self.arena.get_mu_staging(batch_size, channels, mel_width)
            static_speakers_cfg = self.arena.get_speakers_staging(batch_size, spk_dim)
            static_cond_cfg = self.arena.get_cond_staging(batch_size, channels, mel_width)

            static_cnn_cache = self.arena.get_cnn_cache_in(batch_size)
            static_att_cache = self.arena.get_att_cache_in(batch_size, offset if (has_att_cache and offset > 0) else 0)
            static_attn_mask = (
                self.arena.get_mask_staging(batch_size, mel_width, offset + mel_width)
                if (has_mask and attn_mask is not None)
                else None
            )

            out_cnn_cache = self.arena.get_cnn_cache_out(batch_size)
            out_att_cache = self.arena.get_att_cache_out(batch_size, offset + mel_width)
            static_time_embeddings = self.arena.get_time_embeddings(batch_size)

            static_x.copy_(x)
            static_mu_cfg.copy_(mu_cfg)
            static_speakers_cfg.copy_(speakers_cfg)
            static_cond_cfg.copy_(cond_cfg)
            if has_cnn_cache and cnn_cache is not None:
                static_cnn_cache.copy_(cnn_cache)
            else:
                static_cnn_cache.zero_()
            if has_att_cache and att_cache is not None and offset > 0:
                static_att_cache.copy_(att_cache)
            else:
                static_att_cache.zero_()
            if static_attn_mask is not None and attn_mask is not None:
                static_attn_mask.copy_(attn_mask)
        except Exception:
            logger.warning("Failed to allocate static buffers for Whole-Euler graph key: %s", key, exc_info=True)
            self._unsupported.add(key)
            return None

        memory_before = _memory_snapshot(self.device)
        try:
            current_stream = torch.cuda.current_stream(self.device)
            warmup_stream = torch.cuda.Stream(device=self.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream), torch.no_grad():
                for _ in range(3):
                    static_x.copy_(x)
                    _ = self._run_euler_loop(
                        static_x,
                        static_mu_cfg,
                        static_speakers_cfg,
                        static_cond_cfg,
                        static_cnn_cache,
                        static_att_cache,
                        static_attn_mask,
                        out_cnn_cache,
                        out_att_cache,
                        static_time_embeddings,
                        batch_size=batch_size,
                        has_cnn_cache=has_cnn_cache,
                        has_att_cache=has_att_cache,
                    )
            current_stream.wait_stream(warmup_stream)

            static_x.copy_(x)
            graph = CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_final_x = self._run_euler_loop(
                    static_x,
                    static_mu_cfg,
                    static_speakers_cfg,
                    static_cond_cfg,
                    static_cnn_cache,
                    static_att_cache,
                    static_attn_mask,
                    out_cnn_cache,
                    out_att_cache,
                    static_time_embeddings,
                    batch_size=batch_size,
                    has_cnn_cache=has_cnn_cache,
                    has_att_cache=has_att_cache,
                )
        except Exception:
            self._disable("capture failed", key)
            return None

        static_inputs = (
            static_x,
            static_mu_cfg,
            static_speakers_cfg,
            static_cond_cfg,
            static_cnn_cache,
            static_att_cache,
            static_attn_mask,
            static_time_embeddings,
        )
        self._stats["captures"] += 1
        logger.info(
            "Captured Whole-Euler CFM CUDA Graph for shape %s (cache=%d/%d, stats=%s)%s",
            key,
            len(self._cache) + 1,
            self.max_graphs,
            self.stats_snapshot(),
            _format_memory_delta(memory_before, _memory_snapshot(self.device)),
        )
        return (static_inputs, static_final_x, out_cnn_cache, out_att_cache, graph)

    def replay(
        self,
        *,
        x: torch.Tensor,
        mu_cfg: torch.Tensor,
        speakers_cfg: torch.Tensor,
        cond_cfg: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        attn_mask: torch.Tensor | None = None,
        mel_frames: int | None = None,
        pad_frames: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        self._stats["calls"] += 1

        if not self.enabled or torch.cuda.is_current_stream_capturing() or x.device.type != "cuda":
            return None

        batch_size = int(x.shape[0])
        mel_width = int(mu_cfg.shape[2])
        if mel_frames is None:
            mel_frames = mel_width - pad_frames

        if batch_size > self.max_graph_batch:
            return None

        native_batch_sizes = (1, self.micro_batch_size)
        if batch_size not in native_batch_sizes:
            chunk_sizes: list[int] = []
            rem = batch_size
            while rem >= self.micro_batch_size:
                chunk_sizes.append(self.micro_batch_size)
                rem -= self.micro_batch_size
            while rem > 0:
                chunk_sizes.append(1)
                rem -= 1

            if len(chunk_sizes) > self.max_serial_batch:
                return None

            chunk_mels: list[torch.Tensor] = []
            out_cnns: list[torch.Tensor] = []
            out_atts: list[torch.Tensor] = []
            start = 0
            for k in chunk_sizes:
                end = start + k
                sub_x = x[start:end]
                sub_mu = torch.cat((mu_cfg[start:end], mu_cfg[batch_size + start : batch_size + end]), dim=0)
                sub_spk = torch.cat(
                    (speakers_cfg[start:end], speakers_cfg[batch_size + start : batch_size + end]), dim=0
                )
                sub_cond = torch.cat((cond_cfg[start:end], cond_cfg[batch_size + start : batch_size + end]), dim=0)
                sub_cnn = (
                    torch.cat(
                        (cnn_cache[:, :, start:end], cnn_cache[:, :, batch_size + start : batch_size + end]),
                        dim=2,
                    )
                    if cnn_cache is not None
                    else None
                )
                sub_att = (
                    torch.cat(
                        (att_cache[:, :, start:end], att_cache[:, :, batch_size + start : batch_size + end]),
                        dim=2,
                    )
                    if att_cache is not None
                    else None
                )
                sub_mask = (
                    torch.cat(
                        (attn_mask[start:end], attn_mask[batch_size + start : batch_size + end]),
                        dim=0,
                    )
                    if attn_mask is not None
                    else None
                )
                sub_res = self.replay(
                    x=sub_x,
                    mu_cfg=sub_mu,
                    speakers_cfg=sub_spk,
                    cond_cfg=sub_cond,
                    cnn_cache=sub_cnn,
                    att_cache=sub_att,
                    attn_mask=sub_mask,
                    mel_frames=mel_frames,
                    pad_frames=pad_frames,
                )
                if sub_res is None:
                    return None
                chunk_mels.append(sub_res[0])
                out_cnns.append(sub_res[1])
                out_atts.append(sub_res[2])
                start = end

            chunk_mel = torch.cat(chunk_mels, dim=0)
            cond_cnns = torch.cat([c[:, :, 0:k] for c, k in zip(out_cnns, chunk_sizes)], dim=2)
            uncond_cnns = torch.cat([c[:, :, k : 2 * k] for c, k in zip(out_cnns, chunk_sizes)], dim=2)
            out_cnn = torch.cat((cond_cnns, uncond_cnns), dim=2)

            cond_atts = torch.cat([a[:, :, 0:k] for a, k in zip(out_atts, chunk_sizes)], dim=2)
            uncond_atts = torch.cat([a[:, :, k : 2 * k] for a, k in zip(out_atts, chunk_sizes)], dim=2)
            out_att = torch.cat((cond_atts, uncond_atts), dim=2)
            return chunk_mel, out_cnn, out_att
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        has_cnn_cache = cnn_cache is not None
        has_att_cache = att_cache is not None
        has_mask = attn_mask is not None

        key = (
            "whole_euler",
            batch_size,
            mel_width,
            offset,
            has_cnn_cache,
            has_att_cache,
            has_mask,
            str(x.dtype),
            str(x.device),
        )

        if key in self._unsupported:
            return None

        entry = self._cache.get(key)
        if entry is None:
            if len(self._cache) >= self.max_graphs:
                self._flush()
            entry = self._capture(
                key,
                x=x,
                mu_cfg=mu_cfg,
                speakers_cfg=speakers_cfg,
                cond_cfg=cond_cfg,
                cnn_cache=cnn_cache,
                att_cache=att_cache,
                attn_mask=attn_mask,
                batch_size=batch_size,
                offset=offset,
                has_cnn_cache=has_cnn_cache,
                has_att_cache=has_att_cache,
                has_mask=has_mask,
            )
            if entry is None:
                return None
            self._cache[key] = entry
        else:
            self._stats["hits"] += 1

        static_inputs, static_final_x, out_cnn_cache, out_att_cache, graph = entry
        (
            static_x,
            static_mu_cfg,
            static_speakers_cfg,
            static_cond_cfg,
            static_cnn_cache,
            static_att_cache,
            static_attn_mask,
            _static_time_embeddings,
        ) = static_inputs

        static_x.copy_(x)
        static_mu_cfg.copy_(mu_cfg)
        static_speakers_cfg.copy_(speakers_cfg)
        static_cond_cfg.copy_(cond_cfg)
        if has_cnn_cache and cnn_cache is not None:
            static_cnn_cache.copy_(cnn_cache)
        elif not has_cnn_cache:
            static_cnn_cache.zero_()
        if has_att_cache and att_cache is not None and offset > 0:
            static_att_cache.copy_(att_cache)
        if has_mask and attn_mask is not None and static_attn_mask is not None:
            static_attn_mask.copy_(attn_mask)

        graph.replay()

        out_cnn = out_cnn_cache.detach().clone()
        out_att = out_att_cache.detach().clone()
        if pad_frames > 0:
            _zero_padded_cnn_cache(out_cnn, self.estimator, pad_frames)
            out_att[..., mel_frames : mel_frames + pad_frames, :] = 0.0

        return (
            static_final_x[:, :, :mel_frames].detach().clone(),
            out_cnn,
            out_att,
        )
