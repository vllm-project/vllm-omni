# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
"""
CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

This module provides CUDA Graph acceleration for the speech tokenizer decoder,
reducing kernel launch overhead during inference.
"""

import bisect
import os
from collections import Counter

import torch
from torch.cuda import CUDAGraph
from transformers.cache_utils import DynamicCache
from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


class CUDAGraphDecoderWrapper:
    """
    CUDA Graph wrapper for Qwen3TTSTokenizerV2Decoder.

    This wrapper captures the decoder forward pass for fixed input sizes
    and replays them during inference to reduce kernel launch overhead.

    Usage:
        wrapper = CUDAGraphDecoderWrapper(decoder)
        wrapper.warmup(device)

        # During inference:
        output = wrapper.chunked_decode_with_cudagraph(codes, caches)  # Uses CUDA graph if possible
    """

    def __init__(
        self,
        decoder: torch.nn.Module,
        capture_batch_sizes: list[int] | None = None,
        stateless_capture_sizes: list[int] | None = None,
        num_quantizers: int = 8,
        enabled: bool = True,
        async_chunk: bool = True,
        initial_chunk_frames: int = 1,
        codec_chunk_frames: int = 25,
        codec_chunk_ramp: list[int] | None = None,
        decode_chunk_size: int = 300,
        decode_left_context: int = 25,
    ):
        self.decoder = decoder
        self.capture_batch_sizes = sorted(set(capture_batch_sizes or [1]))
        self.num_quantizers = num_quantizers
        self.enabled = enabled
        self.async_chunk = bool(async_chunk)
        self._warmed_up = False

        self.combined_states: dict[tuple[str, int, int], dict] = {}
        self.icl_prefix_states: dict[int, dict] = {}
        self.xvec_prefix_states: dict[int, dict] = {}
        self.stateless_states: dict[tuple[int, int], dict] = {}

        self._device = None
        self.prefix_length = int(getattr(self.decoder.config, "sliding_window", 0) or 0)
        self.codec_chunk_frames = int(codec_chunk_frames)
        self.codec_chunk_ramp = [int(size) for size in codec_chunk_ramp or ()]
        self.initial_chunk_frames = self.codec_chunk_ramp[0] if self.codec_chunk_ramp else int(initial_chunk_frames)
        self.decode_chunk_size = int(decode_chunk_size)
        self.decode_left_context = int(decode_left_context)
        if self.prefix_length <= 2:
            raise ValueError(f"decoder sliding_window must be greater than 2, got {self.prefix_length}")
        if (
            self.initial_chunk_frames <= 0
            or self.codec_chunk_frames <= 0
            or any(size <= 0 for size in self.codec_chunk_ramp)
        ):
            raise ValueError(
                "initial_chunk_frames and codec_chunk_frames must be positive, "
                f"got {self.initial_chunk_frames} and {self.codec_chunk_frames}"
            )
        self._icl_previous_frames_by_target = self._derive_icl_suffix_transitions()
        self._xvec_previous_frames_by_target = self._derive_xvec_suffix_transitions()
        self.icl_capture_sizes = sorted(self._icl_previous_frames_by_target)
        self.stateless_capture_sizes = self._derive_stateless_capture_sizes(stateless_capture_sizes)
        self._stats_enabled = os.environ.get(
            "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS",
            "",
        ).lower() in ("1", "true", "yes", "on")
        self._stats_log_every = int(
            os.environ.get(
                "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS_LOG_EVERY",
                "100",
            )
            or 0
        )
        self._stats_file = os.environ.get(
            "VLLM_OMNI_QWEN3_CODE2WAV_CUDAGRAPH_STATS_FILE",
            "",
        )
        self._stats_total_requests = 0
        self._stats_hit_requests = 0
        self._stats_fallback_requests = 0
        self._stats_replays: Counter[tuple[str, int, int]] = Counter()
        self._stats_fallbacks: Counter[tuple[str, int]] = Counter()

    def _derive_icl_suffix_transitions(self) -> dict[int, int]:
        return self._derive_suffix_transitions()

    def _derive_xvec_suffix_transitions(self) -> dict[int, int]:
        return self._derive_suffix_transitions()

    def _derive_suffix_transitions(self) -> dict[int, int]:
        transitions: dict[int, int] = {}
        previous = self.initial_chunk_frames
        for new_frames in self.codec_chunk_ramp[1:]:
            if previous >= self.prefix_length:
                break
            target = previous + new_frames
            transitions[target] = previous
            previous = target
        while previous < self.prefix_length:
            target = previous + self.codec_chunk_frames
            transitions[target] = previous
            previous = target
        transitions[self.prefix_length + self.codec_chunk_frames] = self.prefix_length
        return transitions

    def _derive_stateless_capture_sizes(self, configured_sizes: list[int] | None) -> list[int]:
        max_frames = self.decode_chunk_size + self.decode_left_context
        sizes = {150, max_frames}
        sizes.update(int(size) for size in configured_sizes or () if int(size) > 0)
        return sorted(sizes)

    def _transitions_for_mode(self, mode: str) -> dict[int, int]:
        return self._xvec_previous_frames_by_target if mode == "xvec" else self._icl_previous_frames_by_target

    def _cached_conv_frames(self, previous_frames: int) -> int:
        return min(previous_frames, self.prefix_length - 2)

    def _record_graph_hit(self, phase: str, batch_size: int, request_count: int) -> None:
        if (
            not getattr(self, "_stats_enabled", False)
            or getattr(self, "_suppress_stats", False)
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        self._stats_total_requests += request_count
        self._stats_hit_requests += request_count
        self._stats_replays[(phase, batch_size, request_count)] += 1
        self._maybe_log_stats()

    def _record_graph_fallback(self, phase: str, request_count: int) -> None:
        if (
            not getattr(self, "_stats_enabled", False)
            or getattr(self, "_suppress_stats", False)
            or torch.cuda.is_current_stream_capturing()
        ):
            return
        self._stats_total_requests += request_count
        self._stats_fallback_requests += request_count
        self._stats_fallbacks[(phase, request_count)] += 1
        self._maybe_log_stats()

    def _maybe_log_stats(self) -> None:
        if self._stats_log_every > 0 and self._stats_total_requests % self._stats_log_every == 0:
            self.log_decode_stats()

    def log_decode_stats(self) -> None:
        if not getattr(self, "_stats_enabled", False) or self._stats_total_requests == 0:
            return
        hit_rate = 100.0 * self._stats_hit_requests / self._stats_total_requests
        message = (
            "Segmented Code2Wav CUDA Graph stats: "
            f"requests={self._stats_total_requests} hits={self._stats_hit_requests} "
            f"fallbacks={self._stats_fallback_requests} hit_rate={hit_rate:.2f}% "
            f"replays={self._stats_replays.most_common(20)} "
            f"fallback_groups={self._stats_fallbacks.most_common(20)}"
        )
        logger.warning("%s", message)
        stats_file = getattr(self, "_stats_file", "")
        if stats_file:
            fd = os.open(stats_file, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                os.write(fd, f"pid={os.getpid()} {message}\n".encode())
            finally:
                os.close(fd)

    def _get_icl_capture_shapes(self) -> list[tuple[int, int]]:
        shapes = {(batch_size, size) for batch_size in self.capture_batch_sizes for size in self.icl_capture_sizes}
        return sorted(shapes)

    def _make_dummy_icl_cache(
        self,
        batch_size: int,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> dict:
        config = self.decoder.config
        dummy_hidden = torch.zeros(
            batch_size,
            config.codebook_dim,
            self.prefix_length,
            device=device,
            dtype=model_dtype,
        )
        dummy_conv = torch.zeros(
            batch_size,
            self.prefix_length,
            config.latent_dim,
            device=device,
            dtype=model_dtype,
        )
        dummy_kv = DynamicCache(config=config)
        assert self.prefix_length == config.sliding_window
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        prefix_kv_shape = (batch_size, config.num_key_value_heads, self.prefix_length, head_dim)

        prefix_keys = torch.zeros(prefix_kv_shape, device=device, dtype=model_dtype)
        prefix_values = torch.zeros_like(prefix_keys)

        for layer_idx in range(config.num_hidden_layers):
            dummy_kv.update(prefix_keys, prefix_values, layer_idx)

        expected_kv_length = self.prefix_length - 1
        if dummy_kv.get_seq_length() != self.prefix_length or any(
            layer.keys.shape[-2] != expected_kv_length for layer in dummy_kv.layers
        ):
            raise RuntimeError(
                "Dummy sliding-window cache does not match the captured prefix "
                f"state: expected logical length {self.prefix_length} and "
                f"physical KV length {expected_kv_length}"
            )

        dummy_prefix_hidden = torch.zeros(
            batch_size,
            self.prefix_length,
            config.latent_dim,
            device=device,
            dtype=model_dtype,
        )
        return {
            "ref_hidden": dummy_hidden,
            "ref_conv": dummy_conv,
            "past_key_values": dummy_kv,
            "prefix_hidden": dummy_prefix_hidden,
        }

    def _make_dummy_xvec_cache(
        self,
        batch_size: int,
        device: torch.device,
        model_dtype: torch.dtype,
    ) -> dict:
        config = self.decoder.config
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        empty_prefix_cache = DynamicCache(config=config)
        empty_prefix_cache.early_initialization(
            batch_size=batch_size,
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=model_dtype,
            device=device,
        )
        return {
            "ref_hidden": torch.zeros(batch_size, config.codebook_dim, 0, device=device, dtype=model_dtype),
            "ref_conv": torch.zeros(batch_size, 0, config.latent_dim, device=device, dtype=model_dtype),
            "past_key_values": empty_prefix_cache,
            "prefix_hidden": torch.zeros(batch_size, 0, config.latent_dim, device=device, dtype=model_dtype),
        }

    def warmup(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.long,
    ):
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return

        self._device = device
        self.decoder.eval()

        self.capture_batch_sizes = [bs for bs in self.capture_batch_sizes if bs > 0]
        if not self.capture_batch_sizes:
            self.capture_batch_sizes = [1]

        if not self.async_chunk:
            capture_shapes = [
                (batch_size, size) for batch_size in self.capture_batch_sizes for size in self.stateless_capture_sizes
            ]
            logger.info(
                "Starting stateless CUDA Graph warmup for %d shapes: batch_sizes=%s seq_lens=%s",
                len(capture_shapes),
                self.capture_batch_sizes,
                self.stateless_capture_sizes,
            )
            for batch_size, size in capture_shapes:
                try:
                    self._capture_stateless(batch_size, size, device, dtype)
                    logger.info("Captured stateless CUDA Graph for batch=%d size=%d", batch_size, size)
                except Exception:
                    logger.warning(
                        "Failed to capture stateless CUDA Graph for batch=%d size=%d",
                        batch_size,
                        size,
                        exc_info=True,
                    )
            self._warmed_up = bool(self.stateless_states)
            return

        if not self.icl_capture_sizes:
            raise ValueError("No reachable segmented capture sizes were configured")
        icl_capture_shapes = self._get_icl_capture_shapes()

        for batch_size in self.capture_batch_sizes:
            try:
                self._capture_icl_prefix(batch_size, device, dtype)
                logger.info("Captured ICL prefix CUDA Graph for batch=%d", batch_size)
            except Exception:
                logger.warning("Failed to capture ICL prefix CUDA Graph for batch=%d", batch_size, exc_info=True)
            try:
                self._capture_xvec_prefix(batch_size, device, dtype)
                logger.info("Captured xvec prefix CUDA Graph for batch=%d", batch_size)
            except Exception:
                logger.warning("Failed to capture xvec prefix CUDA Graph for batch=%d", batch_size, exc_info=True)

        logger.info(
            "Starting CUDA Graph warmup for %d shapes: batch_sizes=%s seq_lens=%s",
            len(icl_capture_shapes),
            self.capture_batch_sizes,
            self.icl_capture_sizes,
        )

        for batch_size, size in icl_capture_shapes:
            try:
                caches = self._make_dummy_icl_cache(batch_size, device, next(self.decoder.parameters()).dtype)
                self._capture_combined_suffix("icl", batch_size, size, caches, device, dtype)
                logger.info("Captured combined suffix CUDA Graph for batch=%d size=%d", batch_size, size)
            except Exception:
                logger.warning(
                    "Failed to capture combined suffix graph for batch=%d size=%d",
                    batch_size,
                    size,
                    exc_info=True,
                )

        model_dtype = next(self.decoder.parameters()).dtype
        for batch_size in self.capture_batch_sizes:
            for size, previous_frames in self._xvec_previous_frames_by_target.items():
                try:
                    caches = self._make_dummy_xvec_cache(batch_size, device, model_dtype)
                    self._capture_combined_suffix("xvec", batch_size, size, caches, device, dtype)
                    logger.info("Captured xvec suffix CUDA Graph for batch=%d size=%d", batch_size, size)
                except Exception:
                    logger.warning(
                        "Failed to capture xvec suffix graph for batch=%d size=%d",
                        batch_size,
                        size,
                        exc_info=True,
                    )

        self._warmed_up = bool(self.icl_prefix_states) or bool(self.xvec_prefix_states) or bool(self.combined_states)

    def _capture_stateless(
        self,
        batch_size: int,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        key = (batch_size, size)
        static_input = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.decoder._forward_exact(static_input)
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder._forward_exact(static_input)

        self.stateless_states[key] = {
            "graph": graph,
            "input": {"codes": static_input},
            "output": static_output,
            "cache": None,
        }

    def _run_combined_suffix(
        self,
        mode: str,
        codes: torch.Tensor,
        old_quantized: torch.Tensor,
        old_conv: torch.Tensor,
        caches: dict,
        target_frames: int,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        previous_frames = self._transitions_for_mode(mode)[target_frames]
        new_frames = target_frames - previous_frames
        rolling = self.decoder._is_suffix_cache_rolling(previous_frames, int(old_quantized.shape[-1]))
        return self.decoder._decode_suffix(
            codes,
            old_quantized,
            old_conv,
            caches,
            self.prefix_length if mode == "icl" else 0,
            new_frames,
            rolling,
            attention_mask=attention_mask,
        )

    def _capture_combined_suffix(
        self,
        mode: str,
        batch_size: int,
        target_frames: int,
        caches: dict,
        device: torch.device,
        code_dtype: torch.dtype,
    ) -> None:
        previous_frames = self._transitions_for_mode(mode)[target_frames]
        new_frames = target_frames - previous_frames
        model_dtype = next(self.decoder.parameters()).dtype
        key = (mode, batch_size, target_frames)
        static_codes = torch.zeros(batch_size, self.num_quantizers, new_frames, dtype=code_dtype, device=device)
        static_quantized = torch.zeros(
            batch_size,
            self.decoder.config.codebook_dim,
            min(previous_frames, self.prefix_length),
            dtype=model_dtype,
            device=device,
        )
        static_conv = torch.zeros(
            batch_size,
            self._cached_conv_frames(previous_frames),
            self.decoder.config.latent_dim,
            dtype=model_dtype,
            device=device,
        )
        prefix_frames = self.prefix_length if mode == "icl" else 0
        static_mask = torch.ones(batch_size, prefix_frames + target_frames, dtype=torch.bool, device=device)

        with torch.no_grad():
            _ = self._run_combined_suffix(
                mode,
                static_codes,
                static_quantized,
                static_conv,
                caches,
                target_frames,
                static_mask,
            )
        torch.accelerator.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                outputs = self._run_combined_suffix(
                    mode,
                    static_codes,
                    static_quantized,
                    static_conv,
                    caches,
                    target_frames,
                    static_mask,
                )

        self.combined_states[key] = {
            "graph": graph,
            "input": {
                "codes": static_codes,
                "quantized": static_quantized,
                "conv": static_conv,
                "mask": static_mask,
            },
            "output": outputs,
            "cache": caches,
        }

    def _capture_icl_prefix(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        size = self.prefix_length + self.initial_chunk_frames
        caches: dict = {}
        static_input = torch.zeros(batch_size, self.num_quantizers, size, dtype=dtype, device=device)
        with torch.no_grad():
            _ = self.decoder._decode_icl_first_chunk(static_input, caches, self.prefix_length)
        torch.accelerator.synchronize(device)

        config = self.decoder.config
        model_dtype = next(self.decoder.parameters()).dtype
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        prefix_cache = DynamicCache(config=config)
        prefix_cache.early_initialization(
            batch_size=batch_size,
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=model_dtype,
            device=device,
        )
        prefix_hidden_mask = torch.ones(batch_size, 1, size, dtype=model_dtype, device=device)
        prefix_attention_mask = torch.ones(batch_size, self.prefix_length, dtype=torch.bool, device=device)
        suffix_attention_mask = torch.ones(batch_size, size, dtype=torch.bool, device=device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder._decode_icl_first_chunk(
                    static_input,
                    caches,
                    self.prefix_length,
                    prefix_cache=prefix_cache,
                    prefix_hidden_mask=prefix_hidden_mask,
                    prefix_attention_mask=prefix_attention_mask,
                    suffix_attention_mask=suffix_attention_mask,
                )

        self.icl_prefix_states[batch_size] = {
            "graph": graph,
            "input": {
                "codes": static_input,
                "hidden_mask": prefix_hidden_mask,
                "prefix_attention_mask": prefix_attention_mask,
                "suffix_attention_mask": suffix_attention_mask,
            },
            "output": static_output,
            "cache": caches,
        }

    def _capture_xvec_prefix(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> None:
        static_input = torch.zeros(
            batch_size,
            self.num_quantizers,
            self.initial_chunk_frames,
            dtype=dtype,
            device=device,
        )
        with torch.no_grad():
            _ = self.decoder._decode_xvec_first_chunk(static_input, {})
        torch.accelerator.synchronize(device)

        config = self.decoder.config
        model_dtype = next(self.decoder.parameters()).dtype
        head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        prefix_cache = DynamicCache(config=config)
        prefix_cache.early_initialization(
            batch_size=batch_size,
            num_heads=config.num_key_value_heads,
            head_dim=head_dim,
            dtype=model_dtype,
            device=device,
        )
        caches: dict = {"past_key_values": prefix_cache}
        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decoder._decode_xvec_first_chunk(static_input, caches)

        self.xvec_prefix_states[batch_size] = {
            "graph": graph,
            "input": {"codes": static_input},
            "output": static_output,
            "cache": caches,
        }

    def _ensure_suffix_buffers(self, caches: dict) -> None:
        if "_suffix_quantized_buffers" in caches:
            return

        suffix_quantized = caches["suffix_quantized"]
        suffix_conv = caches["suffix_conv"]
        quantized_capacity = self.prefix_length
        conv_capacity = self.prefix_length - 2
        quantized_buffers = [
            suffix_quantized.new_zeros((*suffix_quantized.shape[:-1], quantized_capacity)) for _ in range(2)
        ]
        conv_buffers = [
            suffix_conv.new_zeros((suffix_conv.shape[0], conv_capacity, suffix_conv.shape[-1])) for _ in range(2)
        ]
        quantized_valid = min(int(suffix_quantized.shape[-1]), quantized_capacity)
        conv_valid = min(int(suffix_conv.shape[1]), conv_capacity)
        quantized_buffers[0][:, :, :quantized_valid].copy_(suffix_quantized[:, :, -quantized_valid:])
        conv_buffers[0][:, :conv_valid, :].copy_(suffix_conv[:, -conv_valid:, :])
        caches["_suffix_quantized_buffers"] = quantized_buffers
        caches["_suffix_conv_buffers"] = conv_buffers
        caches["_suffix_buffer_index"] = 0
        caches.setdefault("suffix_frames", int(suffix_quantized.shape[-1]))
        caches["suffix_quantized"] = quantized_buffers[0][:, :, :quantized_valid]
        caches["suffix_conv"] = conv_buffers[0][:, :conv_valid, :]

    def _copy_request_caches_to_batch(self, request_caches: list[dict], static_caches: dict) -> None:
        tensor_cache_keys = ("ref_hidden", "ref_conv", "prefix_hidden")
        for row, request_cache in enumerate(request_caches):
            for key in tensor_cache_keys:
                static_caches[key][row : row + 1].copy_(request_cache[key])

            for static_layer, request_layer in zip(
                static_caches["past_key_values"].layers,
                request_cache["past_key_values"].layers,
                strict=True,
            ):
                if request_layer.keys is None or request_layer.values is None:
                    if static_layer.keys.numel() == 0 and static_layer.values.numel() == 0:
                        continue
                    raise RuntimeError("Uninitialized request KV cache cannot populate a non-empty graph cache")
                static_layer.keys[row : row + 1].copy_(request_layer.keys)
                static_layer.values[row : row + 1].copy_(request_layer.values)

    def _select_batch_bucket(self, request_count: int, available: set[int]) -> int | None:
        return next((size for size in self.capture_batch_sizes if size >= request_count and size in available), None)

    @staticmethod
    def _split_for_graph_buckets(request_count: int, available: set[int]) -> list[tuple[int, int]]:
        if request_count <= 0 or not available:
            return []
        largest = max(available)
        ranges: list[tuple[int, int]] = []
        start = 0
        while request_count - start > largest:
            ranges.append((start, start + largest))
            start += largest
        ranges.append((start, request_count))
        return ranges

    def _decode_icl_prefix_batch(
        self,
        codes_list: list[torch.Tensor],
        request_caches: list[dict],
    ) -> list[torch.Tensor] | None:
        available = set(self.icl_prefix_states)
        if torch.cuda.is_current_stream_capturing():
            self._record_graph_fallback("icl_prefix:capture", len(codes_list))
            return None
        if not available:
            self._record_graph_fallback("icl_prefix:no_graph", len(codes_list))
            return None
        if len(codes_list) > max(available, default=0):
            outputs: list[torch.Tensor] = []
            for start, end in self._split_for_graph_buckets(len(codes_list), available):
                chunk_outputs = self._decode_icl_prefix_batch(
                    codes_list[start:end],
                    request_caches[start:end],
                )
                if chunk_outputs is None:
                    return None
                outputs.extend(chunk_outputs)
            return outputs

        batch_size = self._select_batch_bucket(len(codes_list), available)
        if batch_size is None:
            self._record_graph_fallback("icl_prefix:no_graph", len(codes_list))
            return None

        state = self.icl_prefix_states[batch_size]
        static_inputs = state["input"]
        static_input = static_inputs["codes"]
        hidden_mask = static_inputs["hidden_mask"]
        prefix_mask = static_inputs["prefix_attention_mask"]
        suffix_mask = static_inputs["suffix_attention_mask"]
        static_input.zero_()
        hidden_mask.fill_(1)
        prefix_mask.fill_(1)
        suffix_mask.fill_(1)

        prefix_pads: list[int] = []
        for row, (codes, cache) in enumerate(zip(codes_list, request_caches, strict=True)):
            prefix_frames = int(cache["prefix_frames"])
            suffix_frames = int(codes.shape[-1]) - prefix_frames
            if not 0 < prefix_frames <= self.prefix_length or suffix_frames != self.initial_chunk_frames:
                return None
            prefix_pad = self.prefix_length - prefix_frames
            prefix_pads.append(prefix_pad)
            static_input[row, :, prefix_pad : self.prefix_length].copy_(codes[0, :, :prefix_frames])
            static_input[row, :, self.prefix_length :].copy_(codes[0, :, prefix_frames:])
            hidden_mask[row, :, :prefix_pad] = 0
            prefix_mask[row, :prefix_pad] = 0
            suffix_mask[row, :prefix_pad] = 0

        state["graph"].replay()
        self._record_graph_hit("icl_prefix", batch_size, len(codes_list))
        static_caches = state["cache"]
        outputs: list[torch.Tensor] = []
        tensor_keys = ("ref_hidden", "ref_conv", "prefix_hidden")
        for row, (cache, prefix_pad) in enumerate(zip(request_caches, prefix_pads, strict=True)):
            for key in tensor_keys:
                cache[key] = static_caches[key][row : row + 1].clone()
            cache["past_key_values"] = self.decoder._slice_dynamic_cache(static_caches["past_key_values"], row)
            cache["decoder_prefix_frames"] = self.prefix_length
            cache["prefix_pad_frames"] = prefix_pad
            cache["suffix_quantized"] = static_caches["suffix_quantized"][row : row + 1].clone()
            cache["suffix_conv"] = static_caches["suffix_conv"][row : row + 1].clone()
            cache["suffix_frames"] = self.initial_chunk_frames
            outputs.append(
                state["output"][row : row + 1, :, -self.initial_chunk_frames * self.decoder.total_upsample :].clone()
            )
        return outputs

    def _decode_xvec_prefix_batch(
        self,
        codes_list: list[torch.Tensor],
        request_caches: list[dict],
    ) -> list[torch.Tensor] | None:
        available = set(getattr(self, "xvec_prefix_states", {}))
        if not available:
            self._record_graph_fallback("xvec_prefix:no_graph", len(codes_list))
            return None
        if torch.cuda.is_current_stream_capturing():
            self._record_graph_fallback("xvec_prefix:capture", len(codes_list))
            return None
        if any(int(codes.shape[-1]) != self.initial_chunk_frames for codes in codes_list):
            self._record_graph_fallback("xvec_prefix:shape", len(codes_list))
            return None
        if len(codes_list) > max(available, default=0):
            outputs: list[torch.Tensor] = []
            for start, end in self._split_for_graph_buckets(len(codes_list), available):
                chunk_outputs = self._decode_xvec_prefix_batch(
                    codes_list[start:end],
                    request_caches[start:end],
                )
                if chunk_outputs is None:
                    return None
                outputs.extend(chunk_outputs)
            return outputs

        batch_size = self._select_batch_bucket(len(codes_list), available)
        if batch_size is None:
            self._record_graph_fallback("xvec_prefix:no_graph", len(codes_list))
            return None

        state = self.xvec_prefix_states[batch_size]
        static_input = state["input"]["codes"]
        static_input.zero_()
        for row, codes in enumerate(codes_list):
            static_input[row].copy_(codes[0])

        state["graph"].replay()
        self._record_graph_hit("xvec_prefix", batch_size, len(codes_list))
        static_caches = state["cache"]
        outputs: list[torch.Tensor] = []
        tensor_keys = ("ref_hidden", "ref_conv", "prefix_hidden")
        for row, cache in enumerate(request_caches):
            for key in tensor_keys:
                cache[key] = static_caches[key][row : row + 1].clone()
            cache["past_key_values"] = self.decoder._slice_dynamic_cache(static_caches["past_key_values"], row)
            cache["decoder_prefix_frames"] = 0
            cache["suffix_quantized"] = static_caches["suffix_quantized"][row : row + 1].clone()
            cache["suffix_conv"] = static_caches["suffix_conv"][row : row + 1].clone()
            cache["suffix_frames"] = self.initial_chunk_frames
            self._ensure_suffix_buffers(cache)
            outputs.append(state["output"][row : row + 1].clone())
        return outputs

    def _decode_suffix_batch(
        self,
        mode: str,
        target_frames: int,
        codes_list: list[torch.Tensor],
        request_caches: list[dict],
        new_frames_list: list[int],
    ) -> list[torch.Tensor] | None:
        available = {
            batch
            for graph_mode, batch, target in self.combined_states
            if graph_mode == mode and target == target_frames
        }
        if torch.cuda.is_current_stream_capturing():
            self._record_graph_fallback(f"suffix_{target_frames}:capture", len(codes_list))
            return None
        if not available:
            self._record_graph_fallback(f"suffix_{target_frames}:no_graph", len(codes_list))
            return None
        previous_frames = self._transitions_for_mode(mode).get(target_frames)
        if previous_frames is None:
            return None
        expected_new_frames = target_frames - previous_frames
        if any(not 0 < new_frames <= expected_new_frames for new_frames in new_frames_list):
            self._record_graph_fallback(f"suffix_{target_frames}:shape", len(codes_list))
            return None
        if len(codes_list) > max(available, default=0):
            outputs: list[torch.Tensor] = []
            for start, end in self._split_for_graph_buckets(len(codes_list), available):
                chunk_outputs = self._decode_suffix_batch(
                    mode,
                    target_frames,
                    codes_list[start:end],
                    request_caches[start:end],
                    new_frames_list[start:end],
                )
                if chunk_outputs is None:
                    return None
                outputs.extend(chunk_outputs)
            return outputs

        batch_size = self._select_batch_bucket(len(codes_list), available)
        if batch_size is None:
            self._record_graph_fallback(f"suffix_{target_frames}:no_graph", len(codes_list))
            return None
        key = (mode, batch_size, target_frames)
        state = self.combined_states[key]
        static_inputs = state["input"]
        static_codes = static_inputs["codes"]
        static_quantized = static_inputs["quantized"]
        static_conv = static_inputs["conv"]
        static_mask = static_inputs["mask"]
        static_caches = state["cache"]
        static_codes.zero_()
        static_mask.fill_(1)

        for row, (codes, cache, new_frames) in enumerate(zip(codes_list, request_caches, new_frames_list, strict=True)):
            self._ensure_suffix_buffers(cache)
            static_codes[row, :, :new_frames].copy_(codes[0, :, -new_frames:])
            static_quantized[row : row + 1].copy_(cache["suffix_quantized"])
            static_conv[row : row + 1].copy_(cache["suffix_conv"])
            if mode == "icl":
                static_mask[row, : int(cache.get("prefix_pad_frames", 0))] = 0

        self._copy_request_caches_to_batch(request_caches, static_caches)
        state["graph"].replay()
        self._record_graph_hit(f"suffix_{target_frames}", batch_size, len(codes_list))
        graph_output, graph_next_quantized, graph_next_conv = state["output"]
        outputs: list[torch.Tensor] = []
        for row, (cache, new_frames) in enumerate(zip(request_caches, new_frames_list, strict=True)):
            actual_suffix_frames = previous_frames + new_frames
            output = self._trim_replay_output(
                graph_output[row : row + 1],
                actual_suffix_frames,
                target_frames,
            ).clone()
            outputs.append(output)
            if new_frames == expected_new_frames:
                current_index = int(cache["_suffix_buffer_index"])
                next_index = 1 - current_index
                next_quantized = cache["_suffix_quantized_buffers"][next_index]
                next_conv = cache["_suffix_conv_buffers"][next_index]
                quantized_valid = int(graph_next_quantized.shape[-1])
                conv_valid = int(graph_next_conv.shape[1])
                next_quantized[:, :, :quantized_valid].copy_(graph_next_quantized[row : row + 1])
                next_conv[:, :conv_valid, :].copy_(graph_next_conv[row : row + 1])
                cache["_suffix_buffer_index"] = next_index
                cache["suffix_frames"] = target_frames
                cache["suffix_quantized"] = next_quantized[:, :, :quantized_valid]
                cache["suffix_conv"] = next_conv[:, :conv_valid, :]
        return outputs

    def _batched_request_decode(
        self,
        codes_list: list[torch.Tensor],
        request_caches: list[dict],
    ) -> list[torch.Tensor]:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            return [self.decoder._forward_exact(codes) for codes in codes_list]
        previous_suppression = getattr(self, "_suppress_stats", False)
        self._suppress_stats = previous_suppression or (
            bool(request_caches) and all(cache.get("_is_dummy_run", False) for cache in request_caches)
        )
        try:
            return self.decoder.batched_request_decode(
                codes_list,
                request_caches,
                prefix_length=self.prefix_length,
                initial_chunk_frames=self.initial_chunk_frames,
                codec_chunk_frames=self.codec_chunk_frames,
                transitions_by_mode={
                    "icl": self._icl_previous_frames_by_target,
                    "xvec": self._xvec_previous_frames_by_target,
                },
                decode_icl_prefix_batch=self._decode_icl_prefix_batch,
                decode_xvec_prefix_batch=self._decode_xvec_prefix_batch,
                decode_suffix_batch=self._decode_suffix_batch,
                decode_fallback=self._decode_request_fallback,
            )
        finally:
            self._suppress_stats = previous_suppression

    def _decode_request_fallback(self, codes: torch.Tensor, cache: dict) -> torch.Tensor:
        if "suffix_quantized" not in cache:
            if int(cache["prefix_frames"]) == 0:
                return self.decoder._decode_xvec_first_chunk(codes, cache)
            prefix_frames = int(cache["prefix_frames"])
            output = self.decoder._decode_icl_first_chunk(codes, cache, prefix_frames)
            suffix_frames = max(0, int(codes.shape[-1]) - prefix_frames)
            return output[..., -suffix_frames * self.decoder.total_upsample :]
        return self._decode_suffix_eager_fallback(codes, cache)

    def batched_chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        lengths: list[int],
        caches: list[dict] | None = None,
        chunk_size: int = 300,
        left_context_size: int = 25,
        max_batch_size: int = 0,
    ) -> list[torch.Tensor]:
        if not self.async_chunk:
            if caches is not None:
                raise ValueError("caches must be None when async_chunk=False")
            return self._batched_stateless_chunked_decode(codes, lengths, chunk_size, left_context_size, max_batch_size)

        if caches is None:
            raise ValueError("caches are required when async_chunk=True")
        if len(caches) != codes.shape[0] or len(lengths) != codes.shape[0]:
            raise ValueError("codes, lengths, and caches must have the same batch size")
        codes_list = [codes[row : row + 1, :, :length] for row, length in enumerate(lengths)]
        outputs = self._batched_request_decode(codes_list, caches)
        return [output[0] for output in outputs]

    def _batched_stateless_chunked_decode(
        self,
        codes: torch.Tensor,
        lengths: list[int],
        chunk_size: int,
        left_context_size: int,
        max_batch_size: int,
    ) -> list[torch.Tensor]:
        batch_size = int(codes.shape[0])
        if len(lengths) != batch_size:
            raise ValueError("codes and lengths must have the same batch size")
        if any(length < 0 or length > codes.shape[-1] for length in lengths):
            raise ValueError(f"Decode lengths must be within [0, {codes.shape[-1]}], got {lengths}")
        max_length = max(lengths, default=0)
        if max_length == 0:
            return [codes.new_empty((1, 0), dtype=torch.float32) for _ in range(batch_size)]

        total_upsample = self.decoder.total_upsample
        wav_out: list[torch.Tensor] | None = None
        num_rounds = (max_length + chunk_size - 1) // chunk_size
        for round_index in range(num_rounds):
            start = round_index * chunk_size
            groups: dict[int, list[tuple[int, int, int, int, int]]] = {}
            for request_index, request_length in enumerate(lengths):
                if start >= request_length:
                    continue
                end = min(start + chunk_size, request_length)
                context = left_context_size if start - left_context_size > 0 else start
                input_start = start - context
                actual_frames = end - input_start
                bucket_index = bisect.bisect_left(self.stateless_capture_sizes, actual_frames)
                padded_frames = (
                    self.stateless_capture_sizes[bucket_index]
                    if bucket_index < len(self.stateless_capture_sizes)
                    else actual_frames
                )
                groups.setdefault(padded_frames, []).append((request_index, input_start, end, start, context))

            for padded_frames, jobs in groups.items():
                available_batches = {
                    graph_batch for graph_batch, graph_frames in self.stateless_states if graph_frames == padded_frames
                }
                offset = 0
                while offset < len(jobs):
                    remaining = len(jobs) - offset
                    graph_batch = (
                        max(available_batches)
                        if available_batches and remaining > max(available_batches)
                        else self._select_batch_bucket(remaining, available_batches)
                    )
                    if graph_batch is not None:
                        request_count = min(remaining, graph_batch)
                        decode_batch = graph_batch
                    else:
                        request_count = min(remaining, max_batch_size) if max_batch_size > 0 else remaining
                        decode_batch = request_count
                    job_batch = jobs[offset : offset + request_count]
                    codes_chunk = codes.new_zeros((decode_batch, codes.shape[1], padded_frames))
                    for row, (request_index, input_start, input_end, _, _) in enumerate(job_batch):
                        chunk = codes[request_index, :, input_start:input_end]
                        codes_chunk[row, :, : chunk.shape[-1]].copy_(chunk)
                    wav_chunk = self._decode_stateless(codes_chunk, clone_graph_output=False)
                    if wav_out is None:
                        wav_out = [
                            wav_chunk.new_empty((*wav_chunk.shape[1:-1], length * total_upsample)) for length in lengths
                        ]
                    for row, (request_index, _, input_end, chunk_start, context) in enumerate(job_batch):
                        src_start = context * total_upsample
                        dst_start = chunk_start * total_upsample
                        dst_end = input_end * total_upsample
                        src_end = src_start + dst_end - dst_start
                        wav_out[request_index][..., dst_start:dst_end].copy_(wav_chunk[row, ..., src_start:src_end])
                    offset += request_count

        if wav_out is None:
            return [codes.new_empty((1, 0), dtype=torch.float32) for _ in range(batch_size)]
        return wav_out

    def _decode_suffix_eager_fallback(self, codes: torch.Tensor, caches: dict) -> torch.Tensor:
        for key in (
            "_suffix_quantized_buffers",
            "_suffix_conv_buffers",
            "_suffix_buffer_index",
        ):
            caches.pop(key, None)
        decoder_prefix_frames = int(caches.get("decoder_prefix_frames", caches["prefix_frames"]))
        return self.decoder.decode_suffix(codes, caches, decoder_prefix_frames)

    def _decode_stateless(self, codes: torch.Tensor, *, clone_graph_output: bool) -> torch.Tensor:
        if self.async_chunk:
            raise RuntimeError("Stateless decode is unavailable when async_chunk=True")
        if not self.enabled or not self._warmed_up or torch.cuda.is_current_stream_capturing():
            return self.decoder._forward_exact(codes)

        batch_size = int(codes.shape[0])
        actual_frames = int(codes.shape[-1])
        bucket_index = bisect.bisect_left(self.stateless_capture_sizes, actual_frames)
        padded_frames = (
            self.stateless_capture_sizes[bucket_index] if bucket_index < len(self.stateless_capture_sizes) else None
        )
        key = (batch_size, padded_frames) if padded_frames is not None else None
        if key is None or key not in self.stateless_states:
            return self.decoder._forward_exact(codes)

        state = self.stateless_states[key]
        static_input = state["input"]["codes"]
        if actual_frames == padded_frames:
            static_input.copy_(codes)
        else:
            static_input.zero_()
            static_input[:, :, :actual_frames].copy_(codes)
        state["graph"].replay()
        output = self._trim_replay_output(
            state["output"],
            actual_frames,
            padded_frames,
        )
        return output.clone() if clone_graph_output else output

    def _trim_replay_output(self, static_output: torch.Tensor, actual_size: int, padded_size: int) -> torch.Tensor:
        """Trim a graph/compiled replay output to the eager-equivalent length.

        The captured ``static_output`` already reflects the decoder's TRUE output
        length for ``padded_size`` frames, which for causal decoders (e.g.
        Qwen3-Omni Code2Wav) is shorter than ``padded_size * total_upsample`` by a
        fixed amount. Each zero-padded frame beyond ``actual_size`` contributes
        ``total_upsample`` trailing samples that are stale buffer content, so trim
        relative to the captured length instead of the nominal length. This is a
        no-op for decoders whose output equals the nominal length (e.g. Qwen3-TTS).
        """
        drop = (padded_size - actual_size) * self.decoder.total_upsample
        actual_out_len = max(0, static_output.shape[-1] - drop)
        return static_output[..., :actual_out_len]

    def chunked_decode_with_cudagraph(
        self,
        codes: torch.Tensor,
        caches: dict | None = None,
        chunk_size: int = 300,
        left_context_size: int = 25,
    ) -> torch.Tensor:
        if self.async_chunk and caches is None:
            raise ValueError("caches are required when async_chunk=True")
        if not self.async_chunk and caches is not None:
            raise ValueError("caches must be None when async_chunk=False")

        wavs = []
        start_index = 0
        total_len = codes.shape[-1]
        total_upsample = self.decoder.total_upsample
        while start_index < total_len:
            end_index = min(start_index + chunk_size, total_len)
            context_size = left_context_size if start_index - left_context_size > 0 else start_index

            codes_chunk = codes[..., start_index - context_size : end_index]
            if not self.async_chunk:
                wav_chunk = self._decode_stateless(codes_chunk, clone_graph_output=False)
            else:
                wav_chunk = self._batched_request_decode([codes_chunk], [caches])[0]

            # Keep origin/main's concat semantics: Qwen3-Omni can return a chunk
            # that is shorter than the nominal code_len * total_upsample length.
            # Clone each slice because graph outputs are static buffers that later
            # replays may overwrite.
            if self.async_chunk:
                wavs.append(wav_chunk.clone())
            else:
                wavs.append(wav_chunk[..., context_size * total_upsample :].clone())
            start_index = end_index

        if not wavs:
            return self.decoder._forward_exact(codes)
        output = torch.cat(wavs, dim=-1)
        if output.is_cuda and not torch.cuda.is_current_stream_capturing():
            torch.cuda.current_stream(output.device).synchronize()
        return output
