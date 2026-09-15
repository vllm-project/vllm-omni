# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Common CUDA Graph wrappers for streaming vocoder (HiFT) and CFM flow estimator.

Shared across CosyVoice, StepAudio2, and MiniCPM-o to eliminate kernel launch
overheads in the Code2Wav stage.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _tensor_signature(value: torch.Tensor) -> tuple:
    return tuple(value.shape), str(value.dtype), str(value.device)


_DTYPE_MAP = {str(dtype): dtype for dtype in (torch.float32, torch.float16, torch.bfloat16, torch.float64)}


def _memory_snapshot(device: torch.device) -> tuple[int, int] | None:
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
    tensors = []
    for shape, dtype_str, device_str in key[1:]:
        dtype = _DTYPE_MAP[dtype_str]
        device = torch.device(device_str)
        tensors.append(torch.zeros(shape, dtype=dtype, device=device))
    return tuple(tensors)


class HiFTGraphWrapper:
    """CUDA graph capture/replay wrapper for the HiFT neural vocoder.

    Captures ``_inference_pre_istft`` as a static graph and runs ``_finalize_decode``
    outside the graph for maximum compatibility.
    """

    def __init__(
        self,
        token2wav_or_hift: Any,
        connector_config: dict[str, Any] | None = None,
        capture_batch_sizes: tuple[int, ...] | list[int] = (1, 2, 4),
        *,
        mel_cache_len: int = 8,
        source_cache_len: int = 2048,
        mel_frames: int = 80,
    ) -> None:
        if hasattr(token2wav_or_hift, "hift"):
            hift = token2wav_or_hift.hift
            self.mel_cache_len = int(getattr(token2wav_or_hift, "mel_cache_len", mel_cache_len))
            self.source_cache_len = int(getattr(token2wav_or_hift, "source_cache_len", source_cache_len))
            lookahead_layer = getattr(getattr(token2wav_or_hift, "flow", None), "encoder", None)
            pre_lookahead_len = getattr(getattr(lookahead_layer, "pre_lookahead_layer", None), "pre_lookahead_len", 3)
            self.pre_lookahead_len = int(pre_lookahead_len) if pre_lookahead_len is not None else 3
            self.flow_upsample_rate = int(getattr(getattr(token2wav_or_hift, "flow", None), "token_mel_ratio", 2))
        else:
            hift = token2wav_or_hift
            self.mel_cache_len = int(mel_cache_len)
            self.source_cache_len = int(source_cache_len)
            self.pre_lookahead_len = 3
            self.flow_upsample_rate = 2

        self.decode_fn = getattr(hift, "inference")
        self.graph_fn = getattr(hift, "_inference_pre_istft", None)
        self.finalize_fn = getattr(hift, "_finalize_decode", None)

        if connector_config is not None:
            self.codec_chunk_frames = connector_config.get("codec_chunk_frames", 25)
            self.codec_left_context_frames = connector_config.get("codec_left_context_frames", 0)
        else:
            self.codec_chunk_frames = 25
            self.codec_left_context_frames = 0

        conv_pre = getattr(hift, "conv_pre", None)
        in_channels = getattr(conv_pre, "in_channels", mel_frames) if conv_pre is not None else mel_frames
        self.mel_frames = int(in_channels)
        self.capture_bucket_size, self.capture_source_cache_len = self.derive_capture_bucket_size()
        self.capture_batch_sizes = tuple(capture_batch_sizes)
        self.graph: dict[tuple[int, int, int], torch.cuda.CUDAGraph] = {}
        self.static_speech_inputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_magnitude_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_phase_outputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_cache_source_inputs: dict[tuple[int, int, int], torch.Tensor] = {}
        self.static_cache_source_outputs: dict[tuple[int, int, int], torch.Tensor] = {}

        parameter = next(hift.parameters())
        self.device = parameter.device
        self.dtype = parameter.dtype
        self.max_lazy_graphs = 8
        self.lazy_graph_count = 0

    def derive_capture_bucket_size(self) -> tuple[list[int], list[int]]:
        chunk_mel_frames = (
            self.codec_chunk_frames + self.codec_left_context_frames - self.pre_lookahead_len
        ) * self.flow_upsample_rate

        return [chunk_mel_frames, chunk_mel_frames + self.mel_cache_len], [
            0,
            self.source_cache_len,
        ]

    def capture(self) -> None:
        if self.graph_fn is None or self.device.type != "cuda":
            return
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
    ) -> None:
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

    def replay(self, speech_feat: torch.Tensor, cache_source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if torch.cuda.is_current_stream_capturing():
            return self.decode_fn(speech_feat, cache_source)

        batch_size = speech_feat.shape[0]
        num_frames = speech_feat.shape[2]
        cache_source_len = cache_source.shape[2]
        target_b = next((b for b in sorted(self.capture_batch_sizes) if b >= batch_size), None)

        if target_b is None:
            return self.decode_fn(speech_feat, cache_source)

        key = (target_b, num_frames, cache_source_len)
        if key not in self.graph:
            if self.lazy_graph_count >= self.max_lazy_graphs:
                return self.decode_fn(speech_feat, cache_source)
            try:
                self._capture(*key)
                self.lazy_graph_count += 1
            except Exception:
                return self.decode_fn(speech_feat, cache_source)

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


class CFMGraphWrapper:
    """Per-shape CUDA graph capture/replay for the CFM DiT estimator.

    Captures one blocks_forward_chunk call as the graph target.
    """

    def __init__(
        self,
        graph_fn: Any,
        *,
        max_graphs: int = 32,
    ) -> None:
        self.graph_fn = graph_fn
        self.max_graphs = int(max_graphs)
        self.device = next(graph_fn.__self__.parameters()).device
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

    def stats_snapshot(self) -> dict[str, int]:
        return {**self._stats, "cache_size": len(self._cache)}

    def _call_graph_fn(self, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.graph_fn(args[0], args[1], None, args[2], args[3], args[4], args[5])

    def _eager(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self._stats["eager"] += 1
        with torch.no_grad():
            result = self._call_graph_fn(inputs)
        return result, inputs[4], inputs[5]

    def _flush(self) -> None:
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

    def _capture(self, key: tuple) -> tuple | None:
        try:
            static_inputs = _tensors_from_key(key)
        except KeyError:
            logger.warning("CFM graph key carries an unsupported dtype: %s; using eager", key)
            self._unsupported.add(key)
            return None

        memory_before = _memory_snapshot(self.device)
        try:
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        inputs = (estimator_input, time_emb, cnn_cache, att_cache, cnn_out, att_out)
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
            entry = self._capture(key)
            if entry is None:
                return self._eager(inputs)
            self._cache[key] = entry
        else:
            self._stats["hits"] += 1

        static_inputs, static_output, graph = entry
        for static, current in zip(static_inputs, inputs, strict=True):
            static.copy_(current)
        graph.replay()
        return (
            static_output.detach().clone(),
            static_inputs[4].detach().clone(),
            static_inputs[5].detach().clone(),
        )
