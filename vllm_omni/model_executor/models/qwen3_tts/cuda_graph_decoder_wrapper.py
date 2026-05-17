# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import os
import time
from collections import Counter
from collections.abc import Callable, Sequence

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _normalize_decode_lengths(lengths: Sequence[int], batch_size: int, max_len: int) -> tuple[int, ...]:
    if len(lengths) != batch_size:
        raise ValueError(f"Expected {batch_size} decode lengths, got {len(lengths)}")

    normalized: list[int] = []
    for length in lengths:
        length_int = int(length)
        if length_int < 0 or length_int > max_len:
            raise ValueError(f"Invalid decode length {length_int}; expected 0 <= length <= {max_len}")
        normalized.append(length_int)
    return tuple(normalized)


def _batched_chunked_decode(
    codes: torch.Tensor,
    lengths: Sequence[int],
    *,
    decode_fn: Callable[[torch.Tensor], torch.Tensor],
    total_upsample: int,
    chunk_size: int = 300,
    left_context_size: int = 25,
    max_batch_size: int = 0,
) -> torch.Tensor:
    """Decode a padded batch by grouping same-round chunks across requests."""
    if codes.dim() < 3:
        raise ValueError(f"Expected codes with shape [B, Q, F], got {tuple(codes.shape)}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if left_context_size < 0:
        raise ValueError(f"left_context_size must be non-negative, got {left_context_size}")
    if max_batch_size < 0:
        raise ValueError(f"max_batch_size must be non-negative, got {max_batch_size}")

    batch_size = int(codes.shape[0])
    max_input_len = int(codes.shape[-1])
    length_values = _normalize_decode_lengths(lengths, batch_size, max_input_len)
    max_decode_len = max(length_values, default=0)
    if max_decode_len == 0:
        return torch.empty((batch_size, 1, 0), dtype=torch.float32, device=codes.device)

    total_upsample = int(total_upsample)
    wav_out: torch.Tensor | None = None
    num_rounds = (max_decode_len + chunk_size - 1) // chunk_size

    for round_index in range(num_rounds):
        start_index = round_index * chunk_size
        grouped_jobs: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
        for req_index, total_len in enumerate(length_values):
            if start_index >= total_len:
                continue
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index
            input_start = start_index - context_size
            input_end = end_index
            chunk_len = input_end - input_start
            grouped_jobs.setdefault(chunk_len, []).append(
                (req_index, input_start, input_end, start_index, end_index, context_size)
            )

        for jobs in grouped_jobs.values():
            job_batches = (
                [jobs]
                if max_batch_size <= 0 or len(jobs) <= max_batch_size
                else [jobs[start : start + max_batch_size] for start in range(0, len(jobs), max_batch_size)]
            )
            for job_batch in job_batches:
                chunk_rows = [
                    codes[req_index, :, input_start:input_end]
                    for req_index, input_start, input_end, _, _, _ in job_batch
                ]
                codes_chunk = torch.stack(
                    chunk_rows,
                    dim=0,
                )
                wav_chunk = decode_fn(codes_chunk)
                if wav_chunk.shape[0] != len(job_batch):
                    raise ValueError(
                        f"Decoder returned batch size {wav_chunk.shape[0]} for input batch size {len(job_batch)}"
                    )
                if wav_out is None:
                    wav_out = torch.empty(
                        (batch_size, *wav_chunk.shape[1:-1], max_decode_len * total_upsample),
                        dtype=wav_chunk.dtype,
                        device=wav_chunk.device,
                    )

                for row, (req_index, _, _, chunk_start, chunk_end, context_size) in enumerate(job_batch):
                    src_start = context_size * total_upsample
                    dst_start = chunk_start * total_upsample
                    dst_end = chunk_end * total_upsample
                    src_end = src_start + (dst_end - dst_start)
                    if src_end > wav_chunk.shape[-1]:
                        raise ValueError(
                            f"Decoder returned too-short chunk output: need {src_end}, got {wav_chunk.shape[-1]}"
                        )
                    wav_out[req_index, ..., dst_start:dst_end].copy_(wav_chunk[row, ..., src_start:src_end])

    if wav_out is None:
        return torch.empty((batch_size, 1, 0), dtype=torch.float32, device=codes.device)
    return wav_out


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(decoder, capture_sizes=[25, 50, 100, 200, 300])
        wrapper.warmup(device)

        # During inference:
        output = wrapper.decode(codes)  # Automatically uses CUDA graph if possible
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_sizes: list[int] | None = None,
        capture_batch_sizes: list[int] | None = None,
        extra_capture_shapes: list[tuple[int, int]] | None = None,
        compile_shapes: list[tuple[int, int]] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
    ):
        self.decoder = decoder
        self._explicit_sizes = capture_sizes is not None
        self.capture_sizes = sorted(capture_sizes) if capture_sizes else []
        self.capture_batch_sizes = sorted(set(capture_batch_sizes or [1]))
        self.extra_capture_shapes = sorted(
            {
                (int(batch_size), int(size))
                for batch_size, size in extra_capture_shapes or []
                if int(batch_size) > 0 and int(size) > 0
            }
        )
        self.compile_shapes = sorted(
            {
                (int(batch_size), int(size))
                for batch_size, size in compile_shapes or []
                if int(batch_size) > 0 and int(size) > 0
            }
        )
        self._bucket_sizes = self.capture_sizes
        self.num_quantizers = num_quantizers
        self.enabled = enabled

        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self._compiled_decoder: Callable[[torch.Tensor], torch.Tensor] | None = None
        self._compiled_graphs: dict[tuple[int, int], CUDAGraph] = {}
        self._compiled_static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self._compiled_static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self._compiled_shapes: set[tuple[int, int]] = set()

        self._warmed_up = False
        self._device = None
        self._stats_enabled = os.environ.get("VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        self._stats_log_every = int(os.environ.get("VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS_LOG_EVERY", "0") or 0)
        self._stats_total = 0
        self._stats_hits = 0
        self._stats_compiled_hits = 0
        self._stats_fallbacks = 0
        self._stats_stream_capture_fallbacks = 0
        self._stats_requests: Counter[tuple[int, int]] = Counter()
        self._stats_hit_shapes: Counter[tuple[int, int, int]] = Counter()
        self._stats_compiled_shapes: Counter[tuple[int, int]] = Counter()
        self._stats_fallback_shapes: Counter[tuple[int, int, int]] = Counter()

    @staticmethod
    def compute_capture_sizes(
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ) -> list[int]:
        """Compute capture sizes from chunking config for high graph hit rate."""
        sizes: set[int] = set()

        # Streaming exact hits
        if codec_chunk_frames > 0:
            sizes.add(codec_chunk_frames)
            if codec_left_context_frames > 0:
                sizes.add(codec_chunk_frames + codec_left_context_frames)

        # Non-streaming chunked decode: full chunk + last-chunk buckets
        non_stream_max = decode_chunk_size + decode_left_context
        sizes.add(non_stream_max)

        # Power-of-2 buckets covering both streaming IC sizes and non-streaming last-chunk sizes
        for p2 in [2, 4, 8, 16, 32, 64, 128, 256]:
            if p2 <= non_stream_max:
                sizes.add(p2)

        return sorted(sizes)

    def _get_padded_size(self, actual_size: int) -> int | None:
        for size in self._bucket_sizes:
            if actual_size <= size:
                return size
        return None

    def _get_capture_shapes(self) -> list[tuple[int, int]]:
        shapes = {(batch_size, size) for batch_size in self.capture_batch_sizes for size in self.capture_sizes}
        shapes.update(self.extra_capture_shapes)
        return sorted(shapes)

    def _record_decode_stats(
        self,
        *,
        hit: bool,
        batch_size: int,
        actual_size: int,
        padded_size: int | None,
        stream_capture: bool = False,
        compiled: bool = False,
    ) -> None:
        if not self._stats_enabled:
            return

        padded_key = int(padded_size) if padded_size is not None else -1
        self._stats_total += 1
        self._stats_requests[(batch_size, actual_size)] += 1
        if hit:
            self._stats_hits += 1
            if compiled:
                self._stats_compiled_hits += 1
                self._stats_compiled_shapes[(batch_size, actual_size)] += 1
            else:
                self._stats_hit_shapes[(batch_size, actual_size, padded_key)] += 1
        else:
            self._stats_fallbacks += 1
            self._stats_fallback_shapes[(batch_size, actual_size, padded_key)] += 1
            if stream_capture:
                self._stats_stream_capture_fallbacks += 1

        if self._stats_log_every > 0 and self._stats_total % self._stats_log_every == 0:
            self.log_decode_stats()

    def log_decode_stats(self) -> None:
        if not self._stats_enabled or self._stats_total == 0:
            return
        hit_rate = 100.0 * self._stats_hits / self._stats_total
        logger.info(
            "Code2Wav CUDA Graph stats: total=%d hits=%d fallbacks=%d "
            "compiled_hits=%d stream_capture_fallbacks=%d hit_rate=%.2f%% "
            "top_requests=%s top_compiled=%s top_hits=%s top_fallbacks=%s",
            self._stats_total,
            self._stats_hits,
            self._stats_fallbacks,
            self._stats_compiled_hits,
            self._stats_stream_capture_fallbacks,
            hit_rate,
            self._stats_requests.most_common(12),
            self._stats_compiled_shapes.most_common(12),
            self._stats_hit_shapes.most_common(12),
            self._stats_fallback_shapes.most_common(12),
        )

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
        codec_chunk_frames: int = 0,
        codec_left_context_frames: int = 0,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ):
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        self._device = device
        self.decoder.eval()

        if not self._explicit_sizes:
            self.capture_sizes = self.compute_capture_sizes(
                codec_chunk_frames=codec_chunk_frames,
                codec_left_context_frames=codec_left_context_frames,
                decode_chunk_size=decode_chunk_size,
                decode_left_context=decode_left_context,
            )

        self.capture_batch_sizes = [bs for bs in self.capture_batch_sizes if bs > 0]
        if not self.capture_batch_sizes:
            self.capture_batch_sizes = [1]

        self._bucket_sizes = sorted(set(self.capture_sizes) | {size for _, size in self.extra_capture_shapes})
        capture_shapes = self._get_capture_shapes()

        logger.info(
            "Starting CUDA Graph warmup for %d shapes: batch_sizes=%s seq_lens=%s extra_shapes=%s",
            len(capture_shapes),
            self.capture_batch_sizes,
            self.capture_sizes,
            self.extra_capture_shapes,
        )
        warmup_start_s = time.perf_counter()
        mem_before = self._get_cuda_memory_stats(device)

        # Warmup runs to ensure CUDA memory is allocated
        for batch_size, size in capture_shapes:
            dummy = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
            with torch.no_grad():
                _ = self.decoder(dummy)

        torch.accelerator.synchronize(device)

        for batch_size, size in capture_shapes:
            try:
                self._capture(batch_size, size, device, dtype)
                logger.info("  Captured CUDA Graph for batch=%d size=%d", batch_size, size)
            except Exception:
                logger.warning("  Failed to capture graph for batch=%d size=%d", batch_size, size, exc_info=True)

        if self.compile_shapes:
            self._warmup_compile_shapes(device, dtype)

        self._warmed_up = True
        warmup_ms = (time.perf_counter() - warmup_start_s) * 1000.0
        mem_after = self._get_cuda_memory_stats(device)
        logger.info(
            "CUDA Graph warmup complete: %d/%d captured in %.1f ms%s",
            len(self.graphs),
            len(capture_shapes),
            warmup_ms,
            self._format_cuda_memory_delta(mem_before, mem_after),
        )

    def _warmup_compile_shapes(self, device: torch.device, dtype: torch.dtype) -> None:
        logger.info("Starting torch.compile + CUDA Graph warmup for decoder shapes: %s", self.compile_shapes)
        compile_start_s = time.perf_counter()
        try:
            self._compiled_decoder = torch.compile(
                self.decoder.forward,
                mode="default",
                fullgraph=False,
                dynamic=False,
            )
        except Exception:
            logger.warning("Failed to create torch.compile decoder wrapper", exc_info=True)
            self._compiled_decoder = None
            return

        assert self._compiled_decoder is not None
        for batch_size, size in self.compile_shapes:
            shape_start_s = time.perf_counter()
            try:
                self._capture_compiled(batch_size, size, device, dtype)
                self._compiled_shapes.add((batch_size, size))
                logger.info(
                    "  torch.compile + CUDA Graph ready for batch=%d size=%d in %.1f ms",
                    batch_size,
                    size,
                    (time.perf_counter() - shape_start_s) * 1000.0,
                )
            except Exception:
                logger.warning(
                    "  Failed to capture torch.compile CUDA Graph for batch=%d size=%d; "
                    "this shape will use CUDA Graph/eager fallback",
                    batch_size,
                    size,
                    exc_info=True,
                )
        logger.info(
            "torch.compile + CUDA Graph warmup complete: %d/%d shapes ready in %.1f ms",
            len(self._compiled_shapes),
            len(self.compile_shapes),
            (time.perf_counter() - compile_start_s) * 1000.0,
        )

    def _capture(self, batch_size: int, size: int, device: torch.device, dtype: torch.dtype):
        key = (batch_size, size)
        static_input = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.decoder(static_input)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder(static_input)

        self.graphs[key] = graph
        self.static_inputs[key] = static_input
        self.static_outputs[key] = static_output

    def _capture_compiled(self, batch_size: int, size: int, device: torch.device, dtype: torch.dtype):
        if self._compiled_decoder is None:
            raise RuntimeError("Compiled decoder is not initialized")

        key = (batch_size, size)
        static_input = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.inference_mode():
            for _ in range(5):
                _ = self._compiled_decoder(static_input)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self._compiled_decoder(static_input)

        self._compiled_graphs[key] = graph
        self._compiled_static_inputs[key] = static_input
        self._compiled_static_outputs[key] = static_output

    @staticmethod
    def _get_cuda_memory_stats(device: torch.device) -> tuple[int, int, int] | None:
        if device.type != "cuda":
            return None
        try:
            return (
                int(torch.cuda.memory_allocated(device)),
                int(torch.cuda.memory_reserved(device)),
                int(torch.cuda.max_memory_reserved(device)),
            )
        except Exception:
            return None

    @staticmethod
    def _format_cuda_memory_delta(
        before: tuple[int, int, int] | None,
        after: tuple[int, int, int] | None,
    ) -> str:
        if before is None or after is None:
            return ""

        def gib(value: int) -> float:
            return value / 1024**3

        alloc_before, reserved_before, max_reserved_before = before
        alloc_after, reserved_after, max_reserved_after = after
        return (
            f" (cuda_mem allocated {gib(alloc_before):.2f}->{gib(alloc_after):.2f} GiB, "
            f"reserved {gib(reserved_before):.2f}->{gib(reserved_after):.2f} GiB, "
            f"max_reserved {gib(max_reserved_before):.2f}->{gib(max_reserved_after):.2f} GiB)"
        )

    def _decode(self, codes: torch.Tensor, *, clone_graph_output: bool) -> torch.Tensor:
        if not self.enabled or not self._warmed_up:
            return self.decoder(codes)

        # Inner CUDA graph replay is illegal while an outer stream capture is
        # active (e.g. vLLM's cudagraph_mode=FULL warmup on Stage 1). Fall back
        # to eager in that case so the outer capture can complete. The guard is
        # a no-op at runtime: is_current_stream_capturing() returns False
        # outside the startup capture window, so normal inference still hits
        # the graph fast path.
        if torch.cuda.is_current_stream_capturing():
            self._record_decode_stats(
                hit=False,
                batch_size=int(codes.shape[0]),
                actual_size=int(codes.shape[-1]),
                padded_size=None,
                stream_capture=True,
            )
            return self.decoder(codes)

        batch_size = int(codes.shape[0])
        actual_size = int(codes.shape[-1])
        padded_size = self._get_padded_size(actual_size)
        compile_key = (batch_size, actual_size)
        if compile_key not in self._compiled_shapes and padded_size is not None:
            compile_key = (batch_size, padded_size)
        if compile_key in self._compiled_shapes:
            compiled_size = compile_key[1]
            self._record_decode_stats(
                hit=True,
                batch_size=batch_size,
                actual_size=actual_size,
                padded_size=compiled_size,
                compiled=True,
            )
            static_input = self._compiled_static_inputs[compile_key]
            if actual_size == compiled_size:
                static_input.copy_(codes)
            else:
                static_input.zero_()
                static_input[:, :, :actual_size] = codes
            self._compiled_graphs[compile_key].replay()
            actual_out_len = actual_size * self.decoder.total_upsample
            output = self._compiled_static_outputs[compile_key][..., :actual_out_len]
            if clone_graph_output:
                return output.clone()
            return output

        graph_key = (batch_size, padded_size) if padded_size is not None else None

        if graph_key is None or graph_key not in self.graphs:
            self._record_decode_stats(
                hit=False,
                batch_size=batch_size,
                actual_size=actual_size,
                padded_size=padded_size,
            )
            return self.decoder(codes)

        self._record_decode_stats(
            hit=True,
            batch_size=batch_size,
            actual_size=actual_size,
            padded_size=padded_size,
        )
        static_input = self.static_inputs[graph_key]
        if actual_size == padded_size:
            static_input.copy_(codes)
        else:
            static_input.zero_()
            static_input[:, :, :actual_size] = codes
        self.graphs[graph_key].replay()

        actual_out_len = actual_size * self.decoder.total_upsample
        output = self.static_outputs[graph_key][..., :actual_out_len]
        if clone_graph_output:
            return output.clone()
        return output

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        return self._decode(codes, clone_graph_output=True)

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample
        wav_out = None

        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            wav_chunk = self._decode(codes_chunk, clone_graph_output=False)

            if wav_out is None:
                wav_out = torch.empty(
                    (*wav_chunk.shape[:-1], total_len * total_upsample),
                    dtype=wav_chunk.dtype,
                    device=wav_chunk.device,
                )
            src_start = context_size * total_upsample
            dst_start = start_index * total_upsample
            dst_end = end_index * total_upsample
            wav_out[..., dst_start:dst_end].copy_(wav_chunk[..., src_start:])
            start_index = end_index

        if wav_out is None:
            return self.decoder(codes)
        return wav_out

    def batched_chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        lengths: Sequence[int],
        chunk_size: int = 300,
        left_context_size: int = 25,
        max_batch_size: int = 0,
    ) -> torch.Tensor:
        return _batched_chunked_decode(
            codes,
            lengths,
            decode_fn=lambda codes_chunk: self._decode(codes_chunk, clone_graph_output=False),
            total_upsample=self.decoder.total_upsample,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            max_batch_size=max_batch_size,
        )
