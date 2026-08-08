# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA graph replay for Cosmos3 cached GEN forwards.

This intentionally stays Cosmos3-specific. The cached GEN path has two
requirements that vLLM's generic LLM ``CUDAGraphWrapper`` does not own for us:

* denoising-step tensors must be copied into static CUDA graph buffers;
* ``cached_kv`` / ``cached_freqs_gen`` are stable across denoising steps and
  should only be refreshed when the UND cache changes.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform

from vllm_omni.diffusion.cuda_graph import DiffusionCUDAGraphConfig
from vllm_omni.diffusion.models.cosmos3.utils import (
    _clone_static,
    _copy_inputs,
    _key_for_value,
    _return_output,
)

logger = init_logger(__name__)

_CACHE_STATIC_INPUT_NAMES = frozenset({"cached_kv", "cached_freqs_gen"})


@dataclass
class _CachedGenGraphEntry:
    graph: torch.cuda.CUDAGraph
    static_kwargs: dict[str, Any]
    static_output: Any
    cache_generation: Hashable | None
    hits: int = 0


class Cosmos3CUDAGraphManager:
    """Owns CUDA graphs for ``Cosmos3Transformer.forward_cached_gen``."""

    def __init__(
        self,
        forward_cached_gen: Callable[..., Any],
        config: DiffusionCUDAGraphConfig | None = None,
    ) -> None:
        self.forward_cached_gen = forward_cached_gen
        self.config = config or DiffusionCUDAGraphConfig()
        self._cache: OrderedDict[tuple[Hashable, ...], _CachedGenGraphEntry] = OrderedDict()
        self.last_call_info: dict[str, Any] = {}

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def clear(self) -> None:
        for entry in self._cache.values():
            entry.graph.reset()
        self._cache.clear()
        self.last_call_info = {"mode": "clear", "cache_size": 0}

    def run_cached_gen(
        self,
        *,
        branch: str,
        cache_generation: Hashable | None,
        **kwargs: Any,
    ) -> Any:
        key = _build_cached_gen_key(branch, kwargs)

        entry = self._cache.get(key)
        if entry is None:
            hidden_states = kwargs["hidden_states"]
            entry = self._capture(kwargs, cache_generation, hidden_states.device)
            self._cache[key] = entry
            self._evict_if_needed()
            entry.hits += 1
            self.last_call_info = {
                "mode": "graph",
                "reason": "capture",
                "cache_size": len(self._cache),
                "key_size": len(key),
            }
            # Capture records the graph; replay fills static_output for this first request.
            entry.graph.replay()
            return _return_output(entry.static_output, clone=self.config.clone_outputs)

        self._cache.move_to_end(key)
        self._copy_replay_inputs(entry, kwargs, cache_generation)
        entry.graph.replay()
        entry.hits += 1
        self.last_call_info = {
            "mode": "graph",
            "reason": "hit",
            "cache_size": len(self._cache),
            "key_size": len(key),
            "hits": entry.hits,
        }
        return _return_output(entry.static_output, clone=self.config.clone_outputs)

    def _capture(
        self,
        kwargs: Mapping[str, Any],
        cache_generation: Hashable | None,
        device: torch.device,
    ) -> _CachedGenGraphEntry:
        static_kwargs = {name: _clone_static(value) for name, value in kwargs.items()}

        with torch.no_grad():
            for _ in range(self.config.warmup_steps):
                self.forward_cached_gen(**static_kwargs)
                torch.accelerator.synchronize(device)
                if self.config.clear_cuda_cache_on_capture:
                    torch.accelerator.empty_cache()

        graph = torch.cuda.CUDAGraph()
        graph_pool = current_platform.get_global_graph_pool() if self.config.use_global_graph_pool else None
        with torch.no_grad():
            if graph_pool is None:
                with torch.cuda.graph(graph):
                    static_output = self.forward_cached_gen(**static_kwargs)
            else:
                with torch.cuda.graph(graph, pool=graph_pool):
                    static_output = self.forward_cached_gen(**static_kwargs)

        logger.debug("Captured Cosmos3 cached GEN CUDA graph.")
        return _CachedGenGraphEntry(
            graph=graph,
            static_kwargs=static_kwargs,
            static_output=static_output,
            cache_generation=cache_generation,
        )

    def _copy_replay_inputs(
        self,
        entry: _CachedGenGraphEntry,
        kwargs: Mapping[str, Any],
        cache_generation: Hashable | None,
    ) -> None:
        for name, value in kwargs.items():
            if name in _CACHE_STATIC_INPUT_NAMES:
                continue
            _copy_inputs(entry.static_kwargs[name], value)

        if entry.cache_generation != cache_generation:
            for name in _CACHE_STATIC_INPUT_NAMES:
                if name in kwargs:
                    _copy_inputs(entry.static_kwargs[name], kwargs[name])
            entry.cache_generation = cache_generation

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self.config.max_graphs:
            _, entry = self._cache.popitem(last=False)
            entry.graph.reset()


def _build_cached_gen_key(branch: str, kwargs: Mapping[str, Any]) -> tuple[Hashable, ...]:
    key_parts: list[Hashable] = [("branch", branch)]

    for name in (
        "hidden_states",
        "timestep",
        "video_shape",
        "action_latents",
        "action_domain_ids",
        "action_noisy_mask",
        "sound_latents",
        "noisy_frame_mask",
        "control_latents",
        "cached_kv",
        "cached_freqs_gen",
    ):
        if name in kwargs:
            key_parts.append((name, _key_for_value(kwargs[name])))
    return tuple(key_parts)
