# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CUDA Graph runner for LingBot-VLA 2.0 flow-matching action denoising loop."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from vllm.logger import init_logger

if TYPE_CHECKING:
    from .modeling_lingbot_vla_v2 import LingBotVLAV2ForActionPrediction

logger = init_logger(__name__)

PrefixKV = dict[int, dict[str, torch.Tensor]]


@dataclass
class _GraphEntry:
    graph: torch.cuda.CUDAGraph
    num_steps: int
    static_state: torch.Tensor
    static_noise: torch.Tensor
    static_masks: torch.Tensor
    static_positions: torch.Tensor
    static_kv: PrefixKV
    static_dt: torch.Tensor
    static_time_init: torch.Tensor
    static_out: torch.Tensor


class LingBotActionLoopCUDAGraph:
    """Bounded cache of captured multi-step Euler action denoising graphs.

    Only batch-size-1 requests are captured; arbitrary prefix shapes and step
    counts are keyed independently up to ``max_graphs`` entries with LRU
    eviction and an eager fallback whenever capture is inapplicable or fails.

    Concurrency assumption: ``DiffusionWorker`` executes requests serially
    (``max_num_seqs = 1``, ``forward`` rejects ``num_reqs != 1``). Each graph
    entry owns static input/prefix buffers overwritten on replay and returns
    ``static_out.clone()`` so callers never alias the overwritten output buffer.
    """

    def __init__(
        self,
        model: LingBotVLAV2ForActionPrediction,
        device: torch.device | str = "cuda",
        *,
        max_graphs: int = 8,
    ) -> None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be at least 1")
        self.model = model
        self.device = torch.device(device)
        self.max_graphs = int(max_graphs)
        self.enabled = self.device.type == "cuda"
        self._cache: OrderedDict[tuple, _GraphEntry] = OrderedDict()
        self._pool_handle: tuple[int, int] | None = None
        self._capture_stream: torch.cuda.Stream | None = None

    @staticmethod
    def _kv_signature(kv: PrefixKV) -> tuple[tuple[int, tuple[int, ...], tuple[int, ...], str], ...]:
        return tuple(
            (
                layer_idx,
                tuple(kv[layer_idx]["key_states"].shape),
                tuple(kv[layer_idx]["value_states"].shape),
                str(kv[layer_idx]["key_states"].dtype),
            )
            for layer_idx in sorted(kv)
        )

    @classmethod
    def _make_key(
        cls,
        num_steps: int,
        state: torch.Tensor,
        masks: torch.Tensor,
        positions: torch.Tensor,
        kv: PrefixKV,
        noise: torch.Tensor,
    ) -> tuple:
        return (
            int(num_steps),
            tuple(state.shape),
            str(state.dtype),
            tuple(masks.shape),
            str(masks.dtype),
            tuple(positions.shape),
            str(positions.dtype),
            tuple(noise.shape),
            str(noise.dtype),
            cls._kv_signature(kv),
            state.device.index,
        )

    @staticmethod
    def _tensor_matches(src: torch.Tensor, dst: torch.Tensor) -> bool:
        return src.shape == dst.shape and src.dtype == dst.dtype and src.device == dst.device

    @classmethod
    def _matches(
        cls,
        entry: _GraphEntry,
        num_steps: int,
        state: torch.Tensor,
        masks: torch.Tensor,
        kv: PrefixKV,
        positions: torch.Tensor,
        noise: torch.Tensor,
    ) -> bool:
        if entry.num_steps != num_steps:
            return False
        if not (
            cls._tensor_matches(state, entry.static_state)
            and cls._tensor_matches(masks, entry.static_masks)
            and cls._tensor_matches(positions, entry.static_positions)
            and cls._tensor_matches(noise, entry.static_noise)
        ):
            return False
        if kv.keys() != entry.static_kv.keys():
            return False
        for layer_idx, static_layer in entry.static_kv.items():
            live_layer = kv[layer_idx]
            if not (
                cls._tensor_matches(live_layer["key_states"], static_layer["key_states"])
                and cls._tensor_matches(live_layer["value_states"], static_layer["value_states"])
            ):
                return False
        return True

    def _run_eager_loop(
        self,
        *,
        num_steps: int,
        state: torch.Tensor,
        masks: torch.Tensor,
        kv: PrefixKV,
        positions: torch.Tensor,
        noise: torch.Tensor,
        dt: torch.Tensor,
        time_init: torch.Tensor,
    ) -> torch.Tensor:
        x_t = noise.clone()
        time = time_init.clone()
        cache: dict = {}
        for _ in range(num_steps):
            velocity = self.model.predict_velocity(
                state=state,
                prefix_pad_masks=masks,
                past_key_values=kv,
                x_t=x_t,
                timestep=time.expand(state.shape[0]),
                prefix_position_ids=positions,
                _denoise_cache=cache,
            )
            x_t = x_t + dt * velocity
            time = time + dt
        return x_t

    def _capture_entry(
        self,
        *,
        num_steps: int,
        state: torch.Tensor,
        masks: torch.Tensor,
        kv: PrefixKV,
        positions: torch.Tensor,
        noise: torch.Tensor,
        num_warmup: int = 3,
    ) -> _GraphEntry:
        # Allocate static buffers outside inference_mode so replay-side copy_ works
        # even when sample_actions is called inside @torch.inference_mode().
        with torch.inference_mode(False), torch.no_grad():
            static_state = state.detach().clone(memory_format=torch.contiguous_format)
            static_noise = noise.detach().clone(memory_format=torch.contiguous_format)
            static_masks = masks.detach().clone(memory_format=torch.contiguous_format)
            static_positions = positions.detach().clone(memory_format=torch.contiguous_format)
            static_kv: PrefixKV = {
                layer_idx: {
                    "key_states": kv_dict["key_states"].detach().clone(memory_format=torch.contiguous_format),
                    "value_states": kv_dict["value_states"].detach().clone(memory_format=torch.contiguous_format),
                }
                for layer_idx, kv_dict in kv.items()
            }
            # Allocate scalar schedule constants before capture; torch.tensor(..., device="cuda")
            # performs a non-pinned H2D copy that CUDA graph capture rejects.
            dt = torch.full((), -1.0 / num_steps, device=static_state.device, dtype=static_state.dtype)
            time_init = torch.full((), 1.0, device=static_state.device, dtype=static_state.dtype)

            if self._pool_handle is None:
                self._pool_handle = torch.cuda.graph_pool_handle()
            if self._capture_stream is None:
                self._capture_stream = torch.cuda.Stream(device=self.device)

            caller_stream = torch.cuda.current_stream(self.device)
            self._capture_stream.wait_stream(caller_stream)
            try:
                with torch.cuda.stream(self._capture_stream):
                    for _ in range(num_warmup):
                        self._run_eager_loop(
                            num_steps=num_steps,
                            state=static_state,
                            masks=static_masks,
                            kv=static_kv,
                            positions=static_positions,
                            noise=static_noise,
                            dt=dt,
                            time_init=time_init,
                        )
            finally:
                caller_stream.wait_stream(self._capture_stream)

            graph = torch.cuda.CUDAGraph()
            torch._C._cuda_clearCublasWorkspaces()
            try:
                with torch.cuda.stream(caller_stream):
                    with torch.cuda.graph(graph, pool=self._pool_handle, stream=self._capture_stream):
                        static_out = self._run_eager_loop(
                            num_steps=num_steps,
                            state=static_state,
                            masks=static_masks,
                            kv=static_kv,
                            positions=static_positions,
                            noise=static_noise,
                            dt=dt,
                            time_init=time_init,
                        )
            finally:
                torch._C._cuda_clearCublasWorkspaces()

        return _GraphEntry(
            graph=graph,
            num_steps=num_steps,
            static_state=static_state,
            static_noise=static_noise,
            static_masks=static_masks,
            static_positions=static_positions,
            static_kv=static_kv,
            static_dt=dt,
            static_time_init=time_init,
            static_out=static_out,
        )

    @torch.no_grad()
    def capture(
        self,
        *,
        num_steps: int,
        state: torch.Tensor,
        masks: torch.Tensor,
        kv: PrefixKV,
        positions: torch.Tensor,
        noise: torch.Tensor,
        num_warmup: int = 3,
    ) -> _GraphEntry:
        """Warm up and capture the multi-step denoising loop into the bounded cache."""
        key = self._make_key(num_steps, state, masks, positions, kv, noise)
        entry = self._capture_entry(
            num_steps=num_steps,
            state=state,
            masks=masks,
            kv=kv,
            positions=positions,
            noise=noise,
            num_warmup=num_warmup,
        )
        while len(self._cache) >= self.max_graphs:
            self._cache.popitem(last=False)
        self._cache[key] = entry
        logger.info(
            "Captured LingBot-VLA action denoising CUDA graph: steps=%d prefix_len=%d cached=%d",
            num_steps,
            masks.shape[1],
            len(self._cache),
        )
        return entry

    @torch.no_grad()
    def replay(
        self,
        *,
        num_steps: int,
        state: torch.Tensor,
        masks: torch.Tensor,
        kv: PrefixKV,
        positions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor | None:
        """Replay a matching cached graph, or return ``None`` on a miss/mismatch."""
        key = self._make_key(num_steps, state, masks, positions, kv, noise)
        entry = self._cache.get(key)
        if entry is None or not self._matches(entry, num_steps, state, masks, kv, positions, noise):
            return None
        self._cache.move_to_end(key)
        entry.static_state.copy_(state)
        entry.static_noise.copy_(noise)
        entry.static_masks.copy_(masks)
        entry.static_positions.copy_(positions)
        for layer_idx, kv_dict in kv.items():
            entry.static_kv[layer_idx]["key_states"].copy_(kv_dict["key_states"])
            entry.static_kv[layer_idx]["value_states"].copy_(kv_dict["value_states"])
        entry.graph.replay()
        return entry.static_out.clone()

    @torch.no_grad()
    def run(
        self,
        *,
        num_steps: int,
        state: torch.Tensor,
        masks: torch.Tensor,
        kv: PrefixKV,
        positions: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor | None:
        """Run via captured CUDA graph (capturing on cache miss), or ``None`` for eager fallback."""
        if (
            not self.enabled
            or state.device != self.device
            or state.device.type != "cuda"
            or state.shape[0] != 1
            or torch.cuda.is_current_stream_capturing()
        ):
            return None

        replayed = self.replay(
            num_steps=num_steps,
            state=state,
            masks=masks,
            kv=kv,
            positions=positions,
            noise=noise,
        )
        if replayed is not None:
            return replayed

        try:
            entry = self.capture(
                num_steps=num_steps,
                state=state,
                masks=masks,
                kv=kv,
                positions=positions,
                noise=noise,
            )
        except Exception as exc:  # noqa: BLE001 - fall back cleanly to eager on capture error
            logger.warning(
                "LingBot-VLA action CUDA graph capture failed (steps=%d, prefix_len=%d): %s. Falling back to eager.",
                num_steps,
                masks.shape[1],
                exc,
            )
            return None

        entry.graph.replay()
        return entry.static_out.clone()

    def clear(self) -> int:
        """Drop all captured graphs and static buffers; returns how many graphs were released."""
        count = len(self._cache)
        self._cache.clear()
        self._pool_handle = None
        self._capture_stream = None
        return count


__all__ = ["LingBotActionLoopCUDAGraph"]
