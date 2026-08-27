# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import lru_cache

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class HiFTGraphWrapper:
    def __init__(self, token2wav, connector_config, capture_batch_sizes):
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


def _tensor_signature(value: torch.Tensor) -> tuple:
    return tuple(value.shape), str(value.dtype), str(value.device)


_DTYPE_MAP = {
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "torch.float64": torch.float64,
}


def _tensors_from_key(key: tuple) -> tuple[torch.Tensor, ...]:
    """Rebuild zero tensors from a cache key (shape, dtype, device tuples)."""
    tensors = []
    for shape, dtype_str, device_str in key[1:]:
        dtype = _DTYPE_MAP.get(dtype_str, torch.float32)
        device = torch.device(device_str)
        tensors.append(torch.zeros(shape, dtype=dtype, device=device))
    return tuple(tensors)


class CFMGraphWrapper:
    """Per-shape CUDA graph capture/replay for the CFM DiT estimator.

    Captures one blocks_forward_chunk call (in_proj -> DiT blocks -> final_layer)
    as the graph target. The 10-step Euler loop stays in Python, replaying
    the graph 10 times per decode.

    Uses functools.lru_cache for automatic LRU eviction. Cache misses
    trigger capture; capture failures fall back to eager. Outputs are
    cloned after replay to prevent streaming cache corruption.
    """

    def __init__(self, graph_fn, *, max_graphs: int = 32) -> None:
        self.graph_fn = graph_fn
        self.max_graphs = int(max_graphs)
        self.device = next(graph_fn.__self__.parameters()).device

        @lru_cache(maxsize=self.max_graphs)
        def _capture_graph(key: tuple):
            return self._capture(key)

        self._capture_graph = _capture_graph

    def _call_graph_fn(self, args: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return self.graph_fn(args[0], args[1], None, args[2], args[3], args[4], args[5])

    def _capture(self, key: tuple) -> tuple | None:
        """Capture a CUDA graph for the given key. Returns None on failure."""
        static_inputs = _tensors_from_key(key)

        current_stream = torch.cuda.current_stream(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(current_stream)
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            for _ in range(3):
                self._call_graph_fn(static_inputs)
        current_stream.wait_stream(warmup_stream)

        try:
            graph = CUDAGraph()
            with torch.no_grad(), torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self._call_graph_fn(static_inputs)
        except Exception:
            logger.warning(
                "CFM graph capture failed for shape=%s; using eager",
                key,
                exc_info=True,
            )
            return None

        logger.info(
            "Captured CFM CUDA Graph for shape %s (cache=%d/%d, hits=%d, misses=%d)",
            key,
            self._capture_graph.cache_info().currsize,
            self.max_graphs,
            self._capture_graph.cache_info().hits,
            self._capture_graph.cache_info().misses,
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

        if torch.cuda.is_current_stream_capturing() or estimator_input.device.type != "cuda":
            with torch.no_grad():
                result = self._call_graph_fn(inputs)
            return result, inputs[4], inputs[5]

        key = ("estimator_step",) + tuple(_tensor_signature(v) for v in inputs)
        entry = self._capture_graph(key)

        if entry is None:
            logger.debug(
                "CFM graph eager fallback for shape=%s (hits=%d, misses=%d)",
                key,
                self._capture_graph.cache_info().hits,
                self._capture_graph.cache_info().misses,
            )
            with torch.no_grad():
                result = self._call_graph_fn(inputs)
            return result, inputs[4], inputs[5]

        static_inputs, static_output, graph = entry
        for static, current in zip(static_inputs, inputs, strict=True):
            static.copy_(current)
        graph.replay()
        return (
            static_output.detach().clone(),
            static_inputs[4].detach().clone(),
            static_inputs[5].detach().clone(),
        )
