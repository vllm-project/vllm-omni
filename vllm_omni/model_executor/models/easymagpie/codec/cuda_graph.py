# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in CUDA graphs for uniform batches of the existing packed codec.

Capture changes dispatch only: kernels, weights, precision and state pages stay
unchanged. Captures snapshot and restore touched cache pages, and replay copies
its waveform out of graph-owned storage before another graph may use the pool.
"""

from __future__ import annotations

import copy
import os
import threading
from collections import Counter
from dataclasses import dataclass
from typing import Any

import torch
from vllm.forward_context import get_forward_context, set_forward_context

from vllm_omni.model_executor.models.easymagpie.codec.packed import CodecStateLayer, CodecStateMetadata

_ENABLE_ENV = "EASYMAGPIE_VLLM_CODEC_CUDA_GRAPHS"
_LIMIT_ENV = "EASYMAGPIE_VLLM_CODEC_GRAPH_MAX_ENTRIES"
_SHARED_GUARD_ENV = "EASYMAGPIE_VLLM_CODEC_SHARED_METADATA_GUARD"
_TENSOR_FIELDS = (
    "state_indices_tensor_d",
    "state_indices_tensor_p",
    "query_start_loc_p",
    "has_initial_states_p",
)
_SCALAR_FIELDS = (
    "num_prefills",
    "num_prefill_tokens",
    "num_decodes",
    "num_decode_tokens",
    "num_reqs",
    "codec_uniform",
    "codec_prefill_uniform",
    "codec_max_query_len",
)


def codec_graphs_enabled() -> bool:
    return os.environ.get(_ENABLE_ENV, "0").lower() in ("1", "true", "yes", "on")


def uniform_codec_metadata(batch: int, frames: int, device: torch.device) -> CodecStateMetadata:
    """Build a uniform prefill for graph warmup using existing pages 0..B-1."""
    if batch < 1 or frames < 1:
        raise ValueError("codec graph batch and frame counts must be positive")
    return CodecStateMetadata(
        num_prefills=batch,
        num_prefill_tokens=batch * frames,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=batch,
        has_initial_states_p=torch.zeros(batch, dtype=torch.bool, device=device),
        query_start_loc_p=torch.arange(batch + 1, dtype=torch.int32, device=device) * frames,
        state_indices_tensor_p=torch.arange(batch, dtype=torch.int32, device=device),
        state_indices_tensor_d=torch.empty((0, 1), dtype=torch.int32, device=device),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        seq_lens=torch.full((batch,), frames, dtype=torch.int32, device=device),
        codec_uniform=True,
        codec_prefill_uniform=True,
        codec_max_query_len=frames,
    )


@dataclass
class _Entry:
    graph: Any
    codes: torch.Tensor
    metadata: Any
    output: torch.Tensor


class CodecCudaGraphs:
    """Serialize graph replays while preserving vLLM's cache/output ownership."""

    def __init__(self, codec, vllm_config, *, max_entries: int | None = None) -> None:
        self.codec = codec
        self.vllm_config = vllm_config
        self.max_entries = int(os.environ.get(_LIMIT_ENV, "256")) if max_entries is None else int(max_entries)
        if self.max_entries < 1:
            raise ValueError("codec CUDA graph entry limit must be positive")
        self.layers = tuple(module for module in codec.modules() if isinstance(module, CodecStateLayer))
        self.entries: dict[tuple, _Entry] = {}
        self.fallbacks: Counter[str] = Counter()
        self.captures = 0
        self.replays = 0
        self.warmed_shapes: set[tuple[int, int]] = set()
        self.capture_seconds = 0.0
        self._lock = threading.RLock()
        self._pool = None
        self._capture_stream = None
        self._last_event = None
        self._frozen = False
        self.shared_metadata_guard = os.environ.get(_SHARED_GUARD_ENV, "0") == "1"
        self.metadata_same_object_skips = 0
        self.metadata_shared_calls = 0

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "entries": len(self.entries),
            "captures": self.captures,
            "replays": self.replays,
            "fallbacks_by_reason": dict(self.fallbacks),
            "max_entries": self.max_entries,
            "warmed_shapes": [list(shape) for shape in sorted(self.warmed_shapes)],
            "capture_seconds": self.capture_seconds,
            "frozen": self._frozen,
            "shared_metadata_guard": {
                "enabled": self.shared_metadata_guard,
                "same_object_checks_skipped": self.metadata_same_object_skips,
                "calls_with_shared_metadata": self.metadata_shared_calls,
            },
            "graph_memory_policy": (
                "shared-transient-pool; retained per-graph outputs; serialized replay and output clone"
            ),
        }

    def freeze(self) -> None:
        """Prevent runtime captures after explicit startup warmup."""
        self._frozen = True

    @staticmethod
    def _tensor_layout(tensor):
        if tensor is None:
            return None
        return (tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype, tensor.device)

    @staticmethod
    def _tensor_fields(metadata):
        return _TENSOR_FIELDS[1:] if metadata.num_prefills else _TENSOR_FIELDS[:1]

    def _metadata(self, codes, raw):
        if not codes.is_cuda:
            return None, "cpu"
        if codes.dtype != torch.long or codes.ndim != 2 or not codes.is_contiguous():
            return None, "input_layout"
        if torch.cuda.is_current_stream_capturing():
            return None, "outer_capture"
        if not isinstance(raw, dict) or not self.layers:
            return None, "profile_or_missing_metadata"
        selected = [raw.get(layer.prefix) for layer in self.layers]
        if any(meta is None for meta in selected):
            return None, "missing_layer_metadata"
        first = selected[0]
        if not getattr(first, "codec_uniform", False):
            return None, "ragged"
        if bool(first.num_decodes) == bool(first.num_prefills):
            return None, "mixed_or_empty"
        if first.num_decodes and first.num_decode_tokens != first.num_decodes:
            return None, "multiframe_decode"
        if first.num_prefills and (
            not getattr(first, "codec_prefill_uniform", False) or getattr(first, "codec_max_query_len", None) is None
        ):
            return None, "unknown_prefill_shape"
        # vLLM shares the same metadata across identical codec cache groups.
        # Distinct per-layer routing must use eager execution until supported.
        same_object_skips = 0
        for meta in selected:
            # Aliases of the same metadata cannot differ from themselves.
            # Keep every existing check for distinct objects, even if equal.
            if self.shared_metadata_guard and meta is first:
                same_object_skips += 1
                continue
            if any(getattr(meta, name, None) != getattr(first, name, None) for name in _SCALAR_FIELDS):
                return None, "per_layer_shape"
            for name in self._tensor_fields(first):
                lhs, rhs = getattr(meta, name, None), getattr(first, name, None)
                if lhs is not rhs and (
                    lhs is None
                    or rhs is None
                    or lhs.data_ptr() != rhs.data_ptr()
                    or self._tensor_layout(lhs) != self._tensor_layout(rhs)
                ):
                    return None, "per_layer_routing"
        if same_object_skips:
            self.metadata_same_object_skips += same_object_skips
            self.metadata_shared_calls += 1
        if any(not layer.kv_cache[0].is_cuda or not layer.kv_cache[0].numel() for layer in self.layers):
            return None, "unbound_cache"
        if codes.shape[0] != first.num_prefill_tokens + first.num_decode_tokens:
            return None, "token_count"
        return first, None

    def _key(self, codes, metadata):
        return (
            self._tensor_layout(codes),
            tuple(getattr(metadata, name, None) for name in _SCALAR_FIELDS),
            tuple(self._tensor_layout(getattr(metadata, name, None)) for name in self._tensor_fields(metadata)),
            tuple((layer.kv_cache[0].data_ptr(), self._tensor_layout(layer.kv_cache[0])) for layer in self.layers),
            torch.backends.cudnn.allow_tf32,
            torch.backends.cudnn.deterministic,
            torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,
            torch.is_inference_mode_enabled(),
        )

    def _copy_metadata(self, target, source):
        for name in self._tensor_fields(source):
            value = getattr(source, name, None)
            if value is not None:
                getattr(target, name).copy_(value, non_blocking=True)

    def _snapshot(self, metadata):
        indices = metadata.state_indices_tensor_p if metadata.num_prefills else metadata.state_indices_tensor_d
        if indices.ndim == 2:
            indices = indices[:, 0]
        rows = torch.unique(indices[indices >= 0].to(torch.long))
        snapshots = []
        for layer in self.layers:
            cache = layer.kv_cache[0]
            snapshots.append((cache, rows, cache.index_select(0, rows)))
        return snapshots

    @staticmethod
    def _restore(snapshots):
        for cache, rows, original in snapshots:
            cache.index_copy_(0, rows, original)

    def _capture(self, codes, metadata):
        import time

        started = time.perf_counter()
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
            self._capture_stream = torch.cuda.Stream(device=codes.device)
        current = torch.cuda.current_stream(codes.device)
        stream = self._capture_stream
        static_codes = codes.clone()
        static_metadata = copy.copy(metadata)
        for name in self._tensor_fields(metadata):
            value = getattr(metadata, name, None)
            if value is not None:
                setattr(static_metadata, name, value.clone())
        snapshots = self._snapshot(metadata)
        stream.wait_stream(current)
        graph = torch.cuda.CUDAGraph()
        raw = {layer.prefix: static_metadata for layer in self.layers}
        with set_forward_context(raw, self.vllm_config):
            try:
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.codec(static_codes)
                        self._restore(snapshots)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream, pool=self._pool):
                    output = self.codec(static_codes)
                # Warmup advances state. Return all live pages to the exact
                # state they held before this graph was prepared.
            finally:
                with torch.cuda.stream(stream):
                    self._restore(snapshots)
                stream.synchronize()
                current.wait_stream(stream)
        self.captures += 1
        self.capture_seconds += time.perf_counter() - started
        return _Entry(graph, static_codes, static_metadata, output)

    def _order(self, device):
        if self._last_event is not None:
            torch.cuda.current_stream(device).wait_event(self._last_event)

    def _record(self, device):
        if self._last_event is None:
            self._last_event = torch.cuda.Event()
        self._last_event.record(torch.cuda.current_stream(device))

    @torch.no_grad()
    def run(self, codes):
        with self._lock:
            raw = get_forward_context().attn_metadata
            metadata, reason = self._metadata(codes, raw)
            if reason is not None:
                self.fallbacks[reason] += 1
                # Outer capture cannot use this owner's event dependency protocol.
                # It is the caller's responsibility to serialize the enclosing graph.
                if codes.is_cuda and reason != "outer_capture":
                    self._order(codes.device)
                output = self.codec(codes)
                if codes.is_cuda and reason != "outer_capture":
                    self._record(codes.device)
                return output
            self._order(codes.device)
            key = self._key(codes, metadata)
            entry = self.entries.get(key)
            if entry is None:
                if self._frozen or len(self.entries) >= self.max_entries:
                    self.fallbacks["unwarmed_shape" if self._frozen else "entry_limit"] += 1
                    output = self.codec(codes)
                    self._record(codes.device)
                    return output
                entry = self._capture(codes, metadata)
                self.entries[key] = entry
            entry.codes.copy_(codes, non_blocking=True)
            self._copy_metadata(entry.metadata, metadata)
            entry.graph.replay()
            # Request outputs can outlive this call, including asynchronous D2H.
            output = entry.output.clone()
            self._record(codes.device)
            self.replays += 1
            return output

    @torch.no_grad()
    def prewarm(self, batch_sizes, frame_sizes=(2, 6, 8), metadata_template=None, *, freeze=True, descending=False):
        """Capture exact shapes without advancing any live codec state pages.

        Run after cache binding and before service. Graph misses use the unchanged
        eager path after freezing, so no runtime capture contaminates benchmarks.
        """
        del metadata_template  # Uniform metadata is fully specified by this API.
        if not self.layers or any(not layer.kv_cache[0].is_cuda for layer in self.layers):
            raise RuntimeError("codec graph prewarm requires bound CUDA state pages")
        device = self.layers[0].kv_cache[0].device
        batches = sorted(set(int(batch) for batch in batch_sizes), reverse=descending)
        frames_list = sorted(set(int(frames) for frames in frame_sizes), reverse=descending)
        if any(batch < 1 for batch in batches) or any(frames < 1 for frames in frames_list):
            raise ValueError("codec graph warmup shapes must be positive")
        capacity = min(layer.kv_cache[0].shape[0] for layer in self.layers)
        if batches and max(batches) > capacity:
            raise ValueError(f"codec graph warmup exceeds cache capacity {capacity}")
        if len(batches) * len(frames_list) > self.max_entries:
            raise ValueError("codec graph warmup exceeds entry limit")
        with self._lock:
            self._order(device)
            for batch in batches:
                for frames in frames_list:
                    metadata = uniform_codec_metadata(batch, frames, device)
                    codes = torch.zeros(
                        (batch * frames, self.codec.config.num_stacked_codebooks), dtype=torch.long, device=device
                    )
                    key = self._key(codes, metadata)
                    if key not in self.entries:
                        if len(self.entries) >= self.max_entries:
                            raise ValueError("codec graph warmup exceeds remaining entry capacity")
                        self.entries[key] = self._capture(codes, metadata)
                    self.warmed_shapes.add((batch, frames))
            self._record(device)
            if freeze:
                self.freeze()
        return self.status()
