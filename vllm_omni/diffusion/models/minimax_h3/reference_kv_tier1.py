# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-resident reference-KV staging for MiniMax-H3.

MiniMax-H3 uses bidirectional packed self-attention, so visual-condition K/V
changes after the first DiT block as target tokens evolve. This module provides
two explicit experimental caches backed by pinned host memory: Tier1 bulk-stages
every layer before a reuse forward, while Tier2 caches the balanced post-Ulysses
layout and prefetches into a small device ring. Tier2 optionally stores FP8 or
scaled INT8 host values. A separate observer measures reference-KV drift without
substituting cached values.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.profiler import record_function
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

_log = init_logger(__name__)

_ENABLE_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_TIER1"
_TIER2_ENABLE_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_TIER2"
_SKIP_PROJECTION_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_SKIP_PROJECTION"
_OBSERVER_ENABLE_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER"
_REFRESH_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_REFRESH_INTERVAL"
_RING_SIZE_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_RING_SIZE"
_HOST_QUANTIZATION_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_HOST_QUANTIZATION"
_OBSERVER_INTERVALS_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_INTERVALS"
_OBSERVER_ROWS_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_ROWS"
_OBSERVER_HEADS_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_HEADS"
_OBSERVER_OUTPUT_ENV = "VLLM_OMNI_MINIMAX_H3_REF_KV_OBSERVER_OUTPUT"


def _env_enabled(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _refresh_interval_from_env() -> int:
    raw = os.getenv(_REFRESH_ENV, "2").strip()
    try:
        interval = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_REFRESH_ENV} must be an integer, got {raw!r}") from exc
    if interval < 0:
        raise ValueError(f"{_REFRESH_ENV} must be >= 0, got {interval}")
    return interval


def _ring_size_from_env() -> int:
    raw = os.getenv(_RING_SIZE_ENV, "2").strip()
    try:
        ring_size = int(raw)
    except ValueError as exc:
        raise ValueError(f"{_RING_SIZE_ENV} must be an integer, got {raw!r}") from exc
    if not 2 <= ring_size <= 3:
        raise ValueError(f"{_RING_SIZE_ENV} must be 2 or 3, got {ring_size}")
    return ring_size


def _host_quantization_from_env() -> str:
    raw = os.getenv(_HOST_QUANTIZATION_ENV, "none").strip().lower()
    aliases = {"": "none", "off": "none", "false": "none", "0": "none"}
    mode = aliases.get(raw, raw)
    if mode not in {"none", "fp8", "int8"}:
        raise ValueError(f"{_HOST_QUANTIZATION_ENV} must be one of none, fp8, int8, got {raw!r}")
    return mode


def _positive_int_from_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _observer_intervals_from_env() -> tuple[int, ...]:
    raw = os.getenv(_OBSERVER_INTERVALS_ENV, "2,3,4,6,8")
    intervals: set[int] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            interval = int(item)
        except ValueError as exc:
            raise ValueError(f"{_OBSERVER_INTERVALS_ENV} must be comma-separated integers, got {raw!r}") from exc
        if interval <= 1:
            raise ValueError(f"{_OBSERVER_INTERVALS_ENV} values must be > 1, got {interval}")
        intervals.add(interval)
    if not intervals:
        raise ValueError(f"{_OBSERVER_INTERVALS_ENV} must not be empty")
    return tuple(sorted(intervals))


@dataclass
class MiniMaxH3ReferenceKVTier1Stats:
    refresh_steps: int = 0
    h2d_steps: int = 0
    captured_layers: int = 0
    substituted_layers: int = 0
    h2d_bytes: int = 0
    projection_skipped_layers: int = 0
    projection_skipped_rows: int = 0
    projection_total_rows: int = 0


@dataclass
class MiniMaxH3ReferenceKVTier2Stats(MiniMaxH3ReferenceKVTier1Stats):
    prefetched_layers: int = 0
    ready_wait_layers: int = 0
    max_device_staging_bytes: int = 0
    dequantized_layers: int = 0
    quantization_fallbacks: int = 0


class MiniMaxH3ReferenceKVTier1State:
    """Request-local pinned-host K/V pool plus one-forward device staging.

    ``refresh_interval=2`` captures on even denoise steps and reuses on odd
    steps.  ``refresh_interval=0`` captures only step 0, matching a strict
    populate-once prototype; this is intentionally explicit because full-depth
    MiniMax-H3 reference K/V is not invariant across denoise steps.
    """

    tier_name = "Tier1"
    stats: MiniMaxH3ReferenceKVTier1Stats

    def __init__(
        self,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
        refresh_interval: int,
        pin_memory: bool = True,
        skip_reference_projection: bool = False,
        host_quantization: str = "none",
    ) -> None:
        if num_layers <= 0:
            raise ValueError(f"num_layers must be positive, got {num_layers}")
        if global_reference_rows <= 0:
            raise ValueError(f"global_reference_rows must be positive, got {global_reference_rows}")
        if refresh_interval < 0:
            raise ValueError(f"refresh_interval must be >= 0, got {refresh_interval}")
        self.num_layers = int(num_layers)
        self.global_reference_rows = int(global_reference_rows)
        self.device = torch.device(device)
        self.refresh_interval = int(refresh_interval)
        self.pin_memory = bool(pin_memory)
        self.skip_reference_projection = bool(skip_reference_projection)
        if host_quantization not in {"none", "fp8", "int8"}:
            raise ValueError(f"host_quantization must be one of none, fp8, int8, got {host_quantization!r}")
        self.host_quantization = host_quantization
        self.effective_host_quantization = host_quantization
        self.stats = MiniMaxH3ReferenceKVTier1Stats()

        self._host_pool: torch.Tensor | None = None
        self._host_scale_pool: torch.Tensor | None = None
        self._device_pool: torch.Tensor | None = None
        self._local_reference_positions: torch.Tensor | None = None
        self._local_non_reference_positions: torch.Tensor | None = None
        self._local_sequence_rows: int | None = None
        self._global_sequence_rows: int | None = None
        self._global_non_reference_positions: torch.Tensor | None = None
        self._global_reference_positions: torch.Tensor | None = None
        self._compact_rope_table: torch.Tensor | None = None
        self._step: int | None = None
        self._refresh = False
        self._global_refresh = False
        self._inactive = False
        self._layers_seen: set[int] = set()

    @classmethod
    def from_environment(
        cls,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
    ) -> MiniMaxH3ReferenceKVTier1State | None:
        observer_enabled = _env_enabled(_OBSERVER_ENABLE_ENV)
        tier2_enabled = _env_enabled(_TIER2_ENABLE_ENV)
        tier1_enabled = _env_enabled(_ENABLE_ENV)
        if not observer_enabled and not tier1_enabled and not tier2_enabled:
            return None
        tier_name = "Observer" if observer_enabled else "Tier2" if tier2_enabled else "Tier1"
        if global_reference_rows <= 0:
            _log.info(
                "MiniMax-H3 reference-KV %s skipped: request has no visual "
                "condition rows (T2VA has no reference KV to offload)",
                tier_name,
            )
            return None
        if observer_enabled:
            if tier1_enabled or tier2_enabled:
                _log.warning(
                    "MiniMax-H3 reference-KV Observer takes precedence over "
                    "Tier1/Tier2 and will not substitute cached K/V"
                )
            observer_state = MiniMaxH3ReferenceKVObserverState(
                num_layers=num_layers,
                global_reference_rows=global_reference_rows,
                device=device,
                intervals=_observer_intervals_from_env(),
                sample_rows=_positive_int_from_env(_OBSERVER_ROWS_ENV, 32),
                sample_heads=_positive_int_from_env(_OBSERVER_HEADS_ENV, 8),
                output_path=os.getenv(
                    _OBSERVER_OUTPUT_ENV,
                    "/tmp/minimax_h3_reference_kv_drift.jsonl",
                ),
            )
            _log.info(
                "MiniMax-H3 reference-KV Observer enabled: layers=%d, "
                "global_ref_rows=%d, intervals=%s, sample_rows=%d, "
                "sample_heads=%d, device=%s, output=%s",
                num_layers,
                global_reference_rows,
                observer_state.intervals,
                observer_state.sample_rows,
                observer_state.sample_heads,
                device,
                observer_state.output_path,
            )
            return observer_state
        interval = _refresh_interval_from_env()
        skip_projection = _env_enabled(_SKIP_PROJECTION_ENV)
        if interval == 0:
            _log.warning(
                "MiniMax-H3 reference-KV %s strict populate-once mode is "
                "enabled. Full-depth reference K/V drifts in bidirectional "
                "attention, so this mode is lossy; use refresh interval 2 for "
                "the measured quality-oriented prototype.",
                tier_name,
            )
        state: MiniMaxH3ReferenceKVTier1State
        details = ""
        if tier2_enabled:
            tier2_state = MiniMaxH3ReferenceKVTier2State(
                num_layers=num_layers,
                global_reference_rows=global_reference_rows,
                device=device,
                refresh_interval=interval,
                ring_size=_ring_size_from_env(),
                skip_reference_projection=skip_projection,
                host_quantization=_host_quantization_from_env(),
            )
            state = tier2_state
            details = (
                f", ring_size={tier2_state.ring_size}, cache_layout=post_ulysses"
                f", host_quantization={tier2_state.host_quantization}"
            )
        else:
            state = cls(
                num_layers=num_layers,
                global_reference_rows=global_reference_rows,
                device=device,
                refresh_interval=interval,
                skip_reference_projection=skip_projection,
            )
        _log.info(
            "MiniMax-H3 reference-KV %s enabled: layers=%d, global_ref_rows=%d, "
            "refresh_interval=%d, device=%s, skip_reference_projection=%s%s",
            tier_name,
            num_layers,
            global_reference_rows,
            interval,
            device,
            skip_projection,
            details,
        )
        return state

    @property
    def host_pool(self) -> torch.Tensor | None:
        return self._host_pool

    @property
    def device_pool(self) -> torch.Tensor | None:
        return self._device_pool

    @property
    def host_bytes(self) -> int:
        total = 0
        if self._host_pool is not None:
            total += self._host_pool.numel() * self._host_pool.element_size()
        if self._host_scale_pool is not None:
            total += self._host_scale_pool.numel() * self._host_scale_pool.element_size()
        return total

    @property
    def post_parallel_cache(self) -> bool:
        return False

    @property
    def current_step(self) -> int | None:
        return self._step

    def set_global_reference_mask(self, reference_mask: torch.Tensor) -> None:
        """Cache the request-wide target row indices once, before SP sharding."""
        if reference_mask.ndim != 1 or reference_mask.dtype != torch.bool:
            raise ValueError("global reference mask must be a 1-D bool tensor")
        sequence_rows = int(reference_mask.numel())
        if self._global_non_reference_positions is None:
            target_positions = torch.nonzero(~reference_mask, as_tuple=False).view(-1)
            reference_rows = sequence_rows - target_positions.numel()
            if reference_rows != self.global_reference_rows:
                raise ValueError(
                    f"global reference row count mismatch: {reference_rows} != {self.global_reference_rows}"
                )
            self._global_sequence_rows = sequence_rows
            self._global_reference_positions = torch.nonzero(reference_mask, as_tuple=False).view(-1)
            self._global_non_reference_positions = target_positions
        elif sequence_rows != self._global_sequence_rows:
            raise ValueError(
                "global sequence row count changed within one denoise run: "
                f"{sequence_rows} != {self._global_sequence_rows}"
            )

    @property
    def global_non_reference_positions(self) -> torch.Tensor:
        if self._global_non_reference_positions is None:
            raise RuntimeError("global reference mask was not initialized")
        return self._global_non_reference_positions

    @property
    def global_reference_positions(self) -> torch.Tensor:
        if self._global_reference_positions is None:
            raise RuntimeError("global reference mask was not initialized")
        return self._global_reference_positions

    @property
    def compact_rope_table(self) -> torch.Tensor | None:
        return self._compact_rope_table

    def cache_compact_rope_table(self, rope_table: torch.Tensor) -> torch.Tensor:
        if self._compact_rope_table is None:
            self._compact_rope_table = rope_table
        elif (
            self._compact_rope_table.shape != rope_table.shape
            or self._compact_rope_table.dtype != rope_table.dtype
            or self._compact_rope_table.device != rope_table.device
        ):
            raise RuntimeError("compact reference RoPE layout changed within one run")
        return self._compact_rope_table

    @property
    def local_sequence_rows(self) -> int:
        if self._local_sequence_rows is None:
            raise RuntimeError("local reference mask was not initialized")
        return self._local_sequence_rows

    @property
    def should_skip_reference_projection(self) -> bool:
        """Whether this global reuse step may compact reference query rows.

        The decision must be identical on every sequence-parallel rank.  It
        therefore follows the global refresh schedule instead of ``_inactive``;
        only the rank owning reference rows has a host cache, but all ranks must
        shard the same compact target sequence.
        """
        return self.skip_reference_projection and self._step is not None and not self._global_refresh

    @property
    def local_non_reference_positions(self) -> torch.Tensor:
        if self._local_non_reference_positions is None:
            raise RuntimeError("local reference mask was not initialized")
        return self._local_non_reference_positions

    def record_projection_skip(
        self,
        layer_index: int,
        total_rows: int,
        *,
        projected_rows: int | None = None,
    ) -> None:
        if not self.should_skip_reference_projection:
            raise RuntimeError("reference projection skip recorded outside reuse step")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} outside [0, {self.num_layers})")
        if projected_rows is None:
            if self._local_reference_positions is None:
                raise RuntimeError("local reference mask was not initialized")
            projected_rows = total_rows - self._local_reference_positions.numel()
        if not 0 <= projected_rows <= total_rows:
            raise ValueError(f"projected_rows must be within [0, {total_rows}], got {projected_rows}")
        self.stats.projection_skipped_layers += 1
        self.stats.projection_skipped_rows += total_rows - projected_rows
        self.stats.projection_total_rows += total_rows

    def _is_refresh_step(self, step: int) -> bool:
        if self._inactive:
            return False
        if self._host_pool is None:
            return True
        return self.refresh_interval > 0 and step % self.refresh_interval == 0

    def begin_step(self, step: int, *, sigma: float | None = None) -> None:
        del sigma
        if self._step is not None:
            raise RuntimeError(f"reference-KV Tier1 step {self._step} is still active")
        self._step = int(step)
        self._layers_seen.clear()
        self._global_refresh = step == 0 or (self.refresh_interval > 0 and step % self.refresh_interval == 0)
        self._refresh = self._is_refresh_step(step)
        if self._inactive:
            return
        if self._refresh:
            self.stats.refresh_steps += 1
            return
        if self._host_pool is None:
            raise RuntimeError("reference-KV Tier1 reuse requested before populate")
        with record_function("minimax_h3.ref_kv_tier1.h2d_bulk"):
            self._device_pool = self._host_pool.to(
                device=self.device,
                non_blocking=False,
            )
        copied = self.host_bytes
        self.stats.h2d_steps += 1
        self.stats.h2d_bytes += copied
        _log.info(
            "MiniMax-H3 reference-KV Tier1 H2D bulk: step=%d, bytes=%d, layers=%d",
            step,
            copied,
            self.num_layers,
        )

    def set_local_reference_mask(self, reference_mask: torch.Tensor) -> None:
        if reference_mask.dim() != 1 or reference_mask.dtype != torch.bool:
            raise ValueError(
                "reference_mask must be a 1-D bool tensor, got "
                f"shape={tuple(reference_mask.shape)} dtype={reference_mask.dtype}"
            )
        positions = torch.nonzero(reference_mask, as_tuple=False).view(-1)
        if self._local_reference_positions is None:
            self._local_reference_positions = positions
            self._local_non_reference_positions = torch.nonzero(~reference_mask, as_tuple=False).view(-1)
            self._local_sequence_rows = reference_mask.numel()
            self._inactive = positions.numel() == 0
            if self._inactive and self._refresh and self.stats.refresh_steps:
                self.stats.refresh_steps -= 1
                self._refresh = False
            return
        if (
            self._local_sequence_rows != reference_mask.numel()
            or self._local_reference_positions.numel() != positions.numel()
        ):
            raise RuntimeError("MiniMax-H3 reference row layout changed within one denoise request")

    def _allocate_host_pool(self, k: torch.Tensor) -> None:
        if self._local_reference_positions is None:
            raise RuntimeError("local reference mask must be set before KV capture")
        shape = (
            2,
            self.num_layers,
            self._local_reference_positions.numel(),
            *k.shape[1:],
        )
        kwargs = {
            "size": shape,
            "dtype": k.dtype,
            "device": "cpu",
        }
        try:
            self._host_pool = torch.empty(
                **kwargs,
                pin_memory=self.pin_memory,
            )
        except RuntimeError:
            if self.device.type != "cpu" or not self.pin_memory:
                raise
            # CPU-only unit-test environments may not provide a pin allocator.
            self._host_pool = torch.empty(**kwargs)
        _log.info(
            "MiniMax-H3 reference-KV %s host pool populated lazily: shape=%s, pinned=%s, bytes=%d",
            self.tier_name,
            tuple(self._host_pool.shape),
            self._host_pool.is_pinned(),
            self.host_bytes,
        )

    def process_layer(
        self,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._step is None:
            raise RuntimeError("begin_step must be called before processing KV")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} outside [0, {self.num_layers})")
        if layer_index in self._layers_seen:
            raise RuntimeError(f"MiniMax-H3 layer {layer_index} processed twice in step {self._step}")
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} != {v.shape}")
        positions = self._local_reference_positions
        if positions is None:
            raise RuntimeError("local reference mask was not initialized")
        if self._local_sequence_rows != k.shape[0]:
            raise ValueError(f"local reference mask/KV row mismatch: {self._local_sequence_rows} != {k.shape[0]}")
        if k.shape[0] == 0 or positions.numel() == 0:
            self._layers_seen.add(layer_index)
            return k, v

        if self._refresh:
            if self._host_pool is None:
                self._allocate_host_pool(k)
            assert self._host_pool is not None
            expected = tuple(self._host_pool.shape[2:])
            actual = (positions.numel(), *k.shape[1:])
            if actual != expected:
                raise ValueError(f"reference K/V shape changed: expected {expected}, got {actual}")
            with record_function(f"minimax_h3.ref_kv_{self.tier_name.lower()}.populate_host"):
                self._host_pool[0, layer_index].copy_(
                    k.index_select(0, positions),
                    non_blocking=False,
                )
                self._host_pool[1, layer_index].copy_(
                    v.index_select(0, positions),
                    non_blocking=False,
                )
            self.stats.captured_layers += 1
        else:
            if self._device_pool is None:
                raise RuntimeError("bulk H2D staging is missing on a reuse step")
            k.index_copy_(0, positions, self._device_pool[0, layer_index])
            v.index_copy_(0, positions, self._device_pool[1, layer_index])
            self.stats.substituted_layers += 1
        self._layers_seen.add(layer_index)
        return k, v

    def take_cached_reference_layer(
        self,
        layer_index: int,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached local reference K/V without rebuilding full K/V rows."""
        if self._step is None or self._global_refresh:
            raise RuntimeError("cached reference K/V requested outside a reuse step")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} outside [0, {self.num_layers})")
        if layer_index in self._layers_seen:
            raise RuntimeError(f"MiniMax-H3 layer {layer_index} processed twice in step {self._step}")
        if self._inactive:
            self._layers_seen.add(layer_index)
            empty = like.new_empty((0, *like.shape[1:]))
            return empty, empty
        if self._device_pool is None:
            raise RuntimeError("bulk H2D staging is missing on a reuse step")
        self._layers_seen.add(layer_index)
        self.stats.substituted_layers += 1
        return self._device_pool[0, layer_index], self._device_pool[1, layer_index]

    def end_step(self) -> None:
        if self._step is None:
            raise RuntimeError("no active reference-KV Tier1 step")
        if len(self._layers_seen) != self.num_layers:
            raise RuntimeError(
                "MiniMax-H3 reference-KV Tier1 saw "
                f"{len(self._layers_seen)}/{self.num_layers} layers in step {self._step}"
            )
        # The authoritative cache remains on pinned host memory.  Releasing the
        # only device-pool reference returns the full-L staging allocation to the
        # accelerator allocator after every reuse forward.
        self._device_pool = None
        self._step = None
        self._layers_seen.clear()

    def abort_step(self) -> None:
        self._device_pool = None
        self._step = None
        self._layers_seen.clear()

    def close(self) -> None:
        if self._step is not None:
            self.abort_step()
        self._device_pool = None
        _log.info(
            "MiniMax-H3 reference-KV Tier1 summary: host_bytes=%d, "
            "refresh_steps=%d, h2d_steps=%d, captured_layers=%d, "
            "substituted_layers=%d, h2d_bytes=%d, "
            "projection_skipped_layers=%d, projection_skipped_rows=%d, "
            "projection_total_rows=%d",
            self.host_bytes,
            self.stats.refresh_steps,
            self.stats.h2d_steps,
            self.stats.captured_layers,
            self.stats.substituted_layers,
            self.stats.h2d_bytes,
            self.stats.projection_skipped_layers,
            self.stats.projection_skipped_rows,
            self.stats.projection_total_rows,
        )
        self._host_pool = None
        self._host_scale_pool = None
        self._local_reference_positions = None
        self._local_non_reference_positions = None
        self._local_sequence_rows = None
        self._global_non_reference_positions = None
        self._global_reference_positions = None
        self._global_sequence_rows = None
        self._compact_rope_table = None


class MiniMaxH3ReferenceKVTier2State(MiniMaxH3ReferenceKVTier1State):
    """Layerwise asynchronous H2D staging backed by a small device ring.

    Refresh steps capture the balanced K/V head shard after Ulysses All-to-All.
    Reuse steps prefetch one layer at a time from the pinned host pool. Layer
    ``i + 1`` is submitted immediately after layer ``i`` consumes its slot, so
    its H2D can overlap the current layer's attention and MLP work.
    """

    tier_name = "Tier2"
    stats: MiniMaxH3ReferenceKVTier2Stats

    @property
    def post_parallel_cache(self) -> bool:
        return True

    def __init__(
        self,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
        refresh_interval: int,
        ring_size: int = 2,
        pin_memory: bool = True,
        skip_reference_projection: bool = False,
        host_quantization: str = "none",
    ) -> None:
        if not 2 <= ring_size <= 3:
            raise ValueError(f"ring_size must be 2 or 3, got {ring_size}")
        super().__init__(
            num_layers=num_layers,
            global_reference_rows=global_reference_rows,
            device=device,
            refresh_interval=refresh_interval,
            pin_memory=pin_memory,
            skip_reference_projection=skip_reference_projection,
            host_quantization=host_quantization,
        )
        self.ring_size = int(ring_size)
        self.stats = MiniMaxH3ReferenceKVTier2Stats()
        self._device_ring: torch.Tensor | None = None
        self._device_scale_ring: torch.Tensor | None = None
        self._copy_stream: Any | None = None
        self._slot_layers: list[int | None] = [None] * self.ring_size
        self._ready_events: list[Any | None] = [None] * self.ring_size
        self._consumed_events: list[Any | None] = [None] * self.ring_size

    @property
    def device_pool(self) -> torch.Tensor | None:
        return self._device_ring

    @property
    def device_staging_bytes(self) -> int:
        total = 0
        if self._device_ring is not None:
            total += self._device_ring.numel() * self._device_ring.element_size()
        if self._device_scale_ring is not None:
            total += self._device_scale_ring.numel() * self._device_scale_ring.element_size()
        return total

    def set_local_reference_mask(self, reference_mask: torch.Tensor) -> None:
        """Track pre-Ulysses row counts without deactivating zero-row ranks.

        After Ulysses every rank owns the full reference sequence and one head
        shard, even if its pre-Ulysses sequence shard contained no references.
        """
        if self._global_reference_positions is None:
            # Preserve the standalone/legacy pre-Ulysses behavior used without
            # the MiniMax transformer post-parallel hook.
            return super().set_local_reference_mask(reference_mask)
        if reference_mask.dim() != 1 or reference_mask.dtype != torch.bool:
            raise ValueError(
                "reference_mask must be a 1-D bool tensor, got "
                f"shape={tuple(reference_mask.shape)} dtype={reference_mask.dtype}"
            )
        positions = torch.nonzero(reference_mask, as_tuple=False).view(-1)
        if self._local_reference_positions is None:
            self._local_reference_positions = positions
            self._local_non_reference_positions = torch.nonzero(~reference_mask, as_tuple=False).view(-1)
            self._local_sequence_rows = reference_mask.numel()
            self._inactive = False
            return
        if (
            self._local_sequence_rows != reference_mask.numel()
            or self._local_reference_positions.numel() != positions.numel()
        ):
            raise RuntimeError("MiniMax-H3 reference row layout changed within one denoise request")

    def _fall_back_to_int8(self, reason: Exception | str) -> None:
        if self.effective_host_quantization == "int8":
            return
        requested = self.effective_host_quantization
        self.effective_host_quantization = "int8"
        self.stats.quantization_fallbacks += 1
        _log.warning(
            "MiniMax-H3 reference-KV Tier2 %s host quantization is unavailable; falling back to INT8: %s",
            requested,
            reason,
        )

    def _fall_back_to_none(self, reason: Exception | str) -> None:
        if self.effective_host_quantization == "none":
            return
        requested = self.effective_host_quantization
        self.effective_host_quantization = "none"
        self.stats.quantization_fallbacks += 1
        _log.warning(
            "MiniMax-H3 reference-KV Tier2 %s host quantization is unavailable; "
            "falling back to unquantized host storage: %s",
            requested,
            reason,
        )

    def _probe_quantization(self, like: torch.Tensor) -> None:
        if self.effective_host_quantization == "fp8":
            fp8_dtype = getattr(torch, "float8_e4m3fn", None)
            if fp8_dtype is None:
                self._fall_back_to_int8("torch.float8_e4m3fn is unavailable")
            else:
                try:
                    probe = like.reshape(-1)[: min(16, like.numel())].to(fp8_dtype)
                    host_probe = probe.to(device="cpu")
                    device_probe = torch.empty(
                        host_probe.shape,
                        dtype=fp8_dtype,
                        device=self.device,
                    )
                    device_probe.copy_(host_probe, non_blocking=False)
                    device_probe.to(dtype=like.dtype)
                except Exception as exc:
                    self._fall_back_to_int8(exc)

        if self.effective_host_quantization != "int8":
            return
        try:
            sample = like[: min(2, like.shape[0])]
            pair = torch.stack((sample, sample), dim=0)
            absmax = pair.abs().amax(dim=(1, 3)).to(dtype=torch.float32)
            scales = (absmax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
            encoded = (
                torch.round(pair / scales.to(dtype=pair.dtype)[:, None, :, None]).clamp_(-127, 127).to(dtype=torch.int8)
            )
            host_probe = encoded.to(device="cpu")
            device_probe = torch.empty(
                host_probe.shape,
                dtype=torch.int8,
                device=self.device,
            )
            device_probe.copy_(host_probe, non_blocking=False)
            device_probe.to(dtype=like.dtype) * scales.to(dtype=like.dtype)[:, None, :, None]
        except Exception as exc:
            self._fall_back_to_none(exc)

    @staticmethod
    def _empty_host_tensor(
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        pin_memory: bool,
        device_type: str,
    ) -> torch.Tensor:
        kwargs = {"size": shape, "dtype": dtype, "device": "cpu"}
        try:
            return torch.empty(**kwargs, pin_memory=pin_memory)
        except RuntimeError:
            if device_type != "cpu" or not pin_memory:
                raise
            return torch.empty(**kwargs)

    def _allocate_post_ulysses_host_pool(self, reference_k: torch.Tensor) -> None:
        self._probe_quantization(reference_k)
        mode = self.effective_host_quantization
        if mode == "none":
            storage_dtype = reference_k.dtype
        elif mode == "int8":
            storage_dtype = torch.int8
        else:
            storage_dtype = torch.float8_e4m3fn

        shape = (2, self.num_layers, *reference_k.shape)
        self._host_pool = self._empty_host_tensor(
            shape,
            dtype=storage_dtype,
            pin_memory=self.pin_memory,
            device_type=self.device.type,
        )
        if mode == "int8":
            scale_shape = (2, self.num_layers, int(reference_k.shape[-2]))
            self._host_scale_pool = self._empty_host_tensor(
                scale_shape,
                dtype=torch.float32,
                pin_memory=self.pin_memory,
                device_type=self.device.type,
            )
        _log.info(
            "MiniMax-H3 reference-KV Tier2 post-Ulysses host pool populated: "
            "shape=%s, source_dtype=%s, storage_dtype=%s, quantization=%s, "
            "pinned=%s, bytes=%d",
            tuple(self._host_pool.shape),
            reference_k.dtype,
            self._host_pool.dtype,
            mode,
            self._host_pool.is_pinned(),
            self.host_bytes,
        )

    def _encode_reference_pair(
        self,
        reference_k: torch.Tensor,
        reference_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        pair = torch.stack((reference_k, reference_v), dim=0)
        mode = self.effective_host_quantization
        if mode == "none":
            return pair, None
        if mode == "fp8":
            fp8_limit = torch.finfo(torch.float8_e4m3fn).max
            return pair.clamp(-fp8_limit, fp8_limit).to(dtype=torch.float8_e4m3fn), None

        # One symmetric scale per (K/V, head). This keeps scale metadata tiny
        # while preserving independent dynamic ranges across attention heads.
        absmax = pair.abs().amax(dim=(1, 3)).to(dtype=torch.float32)
        scales = (absmax / 127.0).clamp_min(torch.finfo(torch.float32).tiny)
        encoded = (
            torch.round(pair / scales.to(dtype=pair.dtype)[:, None, :, None]).clamp_(-127, 127).to(dtype=torch.int8)
        )
        return encoded, scales

    def _capture_post_ulysses_layer(
        self,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if key.shape != value.shape or key.ndim != 4 or key.shape[0] != 1:
            raise ValueError(
                f"post-Ulysses reference K/V must be matching [1, S, H, D] tensors, got {key.shape} and {value.shape}"
            )
        if self._global_sequence_rows != int(key.shape[1]):
            raise ValueError(f"post-Ulysses sequence length mismatch: {key.shape[1]} != {self._global_sequence_rows}")
        positions = self.global_reference_positions
        reference_k = key[0].index_select(0, positions)
        reference_v = value[0].index_select(0, positions)
        if self._host_pool is None:
            self._allocate_post_ulysses_host_pool(reference_k)
        assert self._host_pool is not None
        expected = tuple(self._host_pool.shape[2:])
        if tuple(reference_k.shape) != expected:
            raise ValueError(
                f"post-Ulysses reference K/V shape changed: expected {expected}, got {tuple(reference_k.shape)}"
            )
        encoded, scales = self._encode_reference_pair(reference_k, reference_v)
        with record_function("minimax_h3.ref_kv_tier2.populate_post_ulysses"):
            self._host_pool[:, layer_index].copy_(encoded, non_blocking=False)
            if scales is not None:
                if self._host_scale_pool is None:
                    raise RuntimeError("INT8 host scale pool is missing")
                self._host_scale_pool[:, layer_index].copy_(scales, non_blocking=False)
        self.stats.captured_layers += 1
        self._layers_seen.add(layer_index)

    def process_post_parallel_layer(
        self,
        layer_index: int,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        compact: bool,
        parallel_strategy: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if parallel_strategy not in {"ulysses", "none"}:
            raise NotImplementedError(
                "post-Ulysses MiniMax-H3 reference KV supports pure Ulysses "
                f"or no sequence parallelism, got {parallel_strategy!r}"
            )
        if self._step is None:
            raise RuntimeError("begin_step must be called before processing KV")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} outside [0, {self.num_layers})")
        if layer_index in self._layers_seen:
            raise RuntimeError(f"MiniMax-H3 layer {layer_index} processed twice in step {self._step}")
        if self._refresh:
            if compact:
                raise RuntimeError("refresh steps must retain reference rows")
            self._capture_post_ulysses_layer(layer_index, key, value)
            return key, value

        cached_k, cached_v = self.take_cached_reference_layer(layer_index, key)
        if cached_k.shape[0] != self.global_reference_rows:
            raise ValueError(
                f"cached post-Ulysses reference row mismatch: {cached_k.shape[0]} != {self.global_reference_rows}"
            )
        if cached_k.shape[1:] != key.shape[2:]:
            raise ValueError(
                f"cached post-Ulysses reference head layout mismatch: {cached_k.shape[1:]} != {key.shape[2:]}"
            )
        if compact:
            return (
                torch.cat((cached_k.unsqueeze(0), key), dim=1),
                torch.cat((cached_v.unsqueeze(0), value), dim=1),
            )

        positions = self.global_reference_positions
        return (
            torch.index_copy(key, 1, positions, cached_k.unsqueeze(0)),
            torch.index_copy(value, 1, positions, cached_v.unsqueeze(0)),
        )

    def _ensure_device_ring(self) -> None:
        if self._device_ring is not None:
            return
        if self._host_pool is None:
            raise RuntimeError("Tier2 ring allocation requested before populate")
        shape = (self.ring_size, 2, *self._host_pool.shape[2:])
        self._device_ring = torch.empty(
            shape,
            dtype=self._host_pool.dtype,
            device=self.device,
        )
        if self._host_scale_pool is not None:
            scale_shape = (
                self.ring_size,
                self._host_scale_pool.shape[0],
                *self._host_scale_pool.shape[2:],
            )
            self._device_scale_ring = torch.empty(
                scale_shape,
                dtype=self._host_scale_pool.dtype,
                device=self.device,
            )
        if self.device.type != "cpu":
            try:
                self._copy_stream = current_omni_platform.Stream()
            except Exception as exc:
                _log.warning(
                    "MiniMax-H3 reference-KV Tier2 could not create an "
                    "asynchronous copy stream; falling back to synchronous "
                    "layerwise H2D: %s",
                    exc,
                )
                self._copy_stream = None
        self.stats.max_device_staging_bytes = max(
            self.stats.max_device_staging_bytes,
            self.device_staging_bytes,
        )
        _log.info(
            "MiniMax-H3 reference-KV Tier2 device ring allocated: shape=%s, bytes=%d, ring_size=%d, quantization=%s",
            tuple(self._device_ring.shape),
            self.device_staging_bytes,
            self.ring_size,
            self.effective_host_quantization,
        )

    def _prefetch_layer(self, layer_index: int) -> None:
        if self._host_pool is None or self._device_ring is None:
            raise RuntimeError("Tier2 prefetch requested before pool allocation")
        slot = layer_index % self.ring_size
        consumed = self._consumed_events[slot]
        host_layer = self._host_pool[:, layer_index]
        device_slot = self._device_ring[slot]
        host_scales = self._host_scale_pool[:, layer_index] if self._host_scale_pool is not None else None
        device_scales = self._device_scale_ring[slot] if self._device_scale_ring is not None else None

        with record_function("minimax_h3.ref_kv_tier2.prefetch"):
            if self._copy_stream is None:
                device_slot.copy_(host_layer, non_blocking=False)
                if host_scales is not None:
                    assert device_scales is not None
                    device_scales.copy_(host_scales, non_blocking=False)
                ready = None
            else:
                try:
                    if consumed is not None:
                        self._copy_stream.wait_event(consumed)
                    ready = current_omni_platform.Event()
                    with current_omni_platform.stream(self._copy_stream):
                        device_slot.copy_(
                            host_layer,
                            non_blocking=host_layer.is_pinned(),
                        )
                        if host_scales is not None:
                            assert device_scales is not None
                            device_scales.copy_(
                                host_scales,
                                non_blocking=host_scales.is_pinned(),
                            )
                        ready.record(self._copy_stream)
                except Exception as exc:
                    _log.warning(
                        "MiniMax-H3 reference-KV Tier2 asynchronous prefetch "
                        "failed at layer %d; falling back to synchronous "
                        "layerwise H2D: %s",
                        layer_index,
                        exc,
                    )
                    try:
                        self._copy_stream.synchronize()
                    except Exception as sync_exc:
                        raise RuntimeError(
                            "Tier2 async prefetch failed and the copy stream could not be synchronized safely"
                        ) from sync_exc
                    self._copy_stream = None
                    device_slot.copy_(host_layer, non_blocking=False)
                    if host_scales is not None:
                        assert device_scales is not None
                        device_scales.copy_(host_scales, non_blocking=False)
                    ready = None

        self._slot_layers[slot] = layer_index
        self._ready_events[slot] = ready
        self.stats.prefetched_layers += 1
        copied = host_layer.numel() * host_layer.element_size()
        if host_scales is not None:
            copied += host_scales.numel() * host_scales.element_size()
        self.stats.h2d_bytes += copied

    def begin_step(self, step: int, *, sigma: float | None = None) -> None:
        del sigma
        if self._step is not None:
            raise RuntimeError(f"reference-KV Tier2 step {self._step} is still active")
        self._step = int(step)
        self._layers_seen.clear()
        self._global_refresh = step == 0 or (self.refresh_interval > 0 and step % self.refresh_interval == 0)
        self._refresh = self._is_refresh_step(step)
        if self._inactive:
            return
        if self._refresh:
            self.stats.refresh_steps += 1
            return
        if self._host_pool is None:
            raise RuntimeError("reference-KV Tier2 reuse requested before populate")

        self._ensure_device_ring()
        self.stats.h2d_steps += 1
        self._prefetch_layer(0)
        _log.info(
            "MiniMax-H3 reference-KV Tier2 layerwise H2D: step=%d, layer_bytes=%d, layers=%d, ring_size=%d",
            step,
            self.host_bytes // self.num_layers,
            self.num_layers,
            self.ring_size,
        )

    def take_cached_reference_layer(
        self,
        layer_index: int,
        like: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._step is None or self._global_refresh:
            raise RuntimeError("cached reference K/V requested outside a reuse step")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} outside [0, {self.num_layers})")
        if layer_index in self._layers_seen:
            raise RuntimeError(f"MiniMax-H3 layer {layer_index} processed twice in step {self._step}")
        if layer_index != len(self._layers_seen):
            raise RuntimeError(
                "Tier2 layerwise prefetch requires sequential layer execution: "
                f"expected {len(self._layers_seen)}, got {layer_index}"
            )
        if self._inactive:
            self._layers_seen.add(layer_index)
            empty = like.new_empty((0, *like.shape[1:]))
            return empty, empty
        if self._device_ring is None:
            raise RuntimeError("Tier2 device ring is missing on a reuse step")

        slot = layer_index % self.ring_size
        if self._slot_layers[slot] != layer_index:
            raise RuntimeError(f"Tier2 slot {slot} contains layer {self._slot_layers[slot]}, expected {layer_index}")
        ready = self._ready_events[slot]
        compute_stream = None
        if ready is not None:
            compute_stream = current_omni_platform.current_stream()
            with record_function("minimax_h3.ref_kv_tier2.wait"):
                compute_stream.wait_event(ready)
            self.stats.ready_wait_layers += 1

        # The ring slot is reused two or three layers later. Clone its
        # reference-only payload before the copy stream advances the ring.
        with record_function("minimax_h3.ref_kv_tier2.take_cached"):
            cached = self._device_ring[slot].clone()
            mode = self.effective_host_quantization
            if mode == "fp8":
                cached = cached.to(dtype=like.dtype)
                self.stats.dequantized_layers += 1
            elif mode == "int8":
                if self._device_scale_ring is None:
                    raise RuntimeError("INT8 device scale ring is missing")
                scales = self._device_scale_ring[slot].clone()
                cached = cached.to(dtype=like.dtype) * scales.to(dtype=like.dtype)[:, None, :, None]
                self.stats.dequantized_layers += 1
        self.stats.substituted_layers += 1
        self._layers_seen.add(layer_index)

        if compute_stream is not None:
            consumed = current_omni_platform.Event()
            consumed.record(compute_stream)
            self._consumed_events[slot] = consumed
        else:
            self._consumed_events[slot] = None

        next_layer = layer_index + 1
        if next_layer < self.num_layers:
            self._prefetch_layer(next_layer)
        return cached[0], cached[1]

    def process_layer(
        self,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._refresh:
            return super().process_layer(layer_index, k, v)
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} != {v.shape}")
        positions = self._local_reference_positions
        if positions is None:
            raise RuntimeError("local reference mask was not initialized")
        if self._local_sequence_rows != k.shape[0]:
            raise ValueError(f"local reference mask/KV row mismatch: {self._local_sequence_rows} != {k.shape[0]}")
        if self._inactive:
            if layer_index in self._layers_seen:
                raise RuntimeError(f"MiniMax-H3 layer {layer_index} processed twice in step {self._step}")
            self._layers_seen.add(layer_index)
            return k, v
        cached_k, cached_v = self.take_cached_reference_layer(layer_index, k)
        if positions.numel():
            with record_function("minimax_h3.ref_kv_tier2.substitute"):
                k.index_copy_(0, positions, cached_k)
                v.index_copy_(0, positions, cached_v)
        return k, v

    def end_step(self) -> None:
        if self._step is None:
            raise RuntimeError("no active reference-KV Tier2 step")
        if len(self._layers_seen) != self.num_layers:
            raise RuntimeError(
                "MiniMax-H3 reference-KV Tier2 saw "
                f"{len(self._layers_seen)}/{self.num_layers} layers in step {self._step}"
            )
        self._step = None
        self._layers_seen.clear()

    def abort_step(self) -> None:
        if self._copy_stream is not None:
            try:
                self._copy_stream.synchronize()
            except Exception:
                _log.debug("Tier2 copy-stream cleanup failed", exc_info=True)
        self._device_ring = None
        self._device_scale_ring = None
        self._copy_stream = None
        self._slot_layers = [None] * self.ring_size
        self._ready_events = [None] * self.ring_size
        self._consumed_events = [None] * self.ring_size
        self._step = None
        self._layers_seen.clear()

    def close(self) -> None:
        if self._step is not None:
            self.abort_step()
        elif self._copy_stream is not None:
            try:
                self._copy_stream.synchronize()
            except Exception:
                _log.debug("Tier2 copy-stream close failed", exc_info=True)
        _log.info(
            "MiniMax-H3 reference-KV Tier2 summary: host_bytes=%d, "
            "device_staging_bytes=%d, refresh_steps=%d, h2d_steps=%d, "
            "captured_layers=%d, substituted_layers=%d, "
            "prefetched_layers=%d, ready_wait_layers=%d, h2d_bytes=%d, "
            "projection_skipped_layers=%d, projection_skipped_rows=%d, "
            "projection_total_rows=%d, inactive=%s, cache_layout=post_ulysses, "
            "host_quantization=%s, effective_host_quantization=%s, "
            "dequantized_layers=%d, quantization_fallbacks=%d",
            self.host_bytes,
            self.device_staging_bytes,
            self.stats.refresh_steps,
            self.stats.h2d_steps,
            self.stats.captured_layers,
            self.stats.substituted_layers,
            self.stats.prefetched_layers,
            self.stats.ready_wait_layers,
            self.stats.h2d_bytes,
            self.stats.projection_skipped_layers,
            self.stats.projection_skipped_rows,
            self.stats.projection_total_rows,
            self._inactive,
            self.host_quantization,
            self.effective_host_quantization,
            self.stats.dequantized_layers,
            self.stats.quantization_fallbacks,
        )
        self._device_ring = None
        self._device_scale_ring = None
        self._copy_stream = None
        self._slot_layers = [None] * self.ring_size
        self._ready_events = [None] * self.ring_size
        self._consumed_events = [None] * self.ring_size
        self._host_pool = None
        self._host_scale_pool = None
        self._local_reference_positions = None
        self._local_non_reference_positions = None
        self._local_sequence_rows = None
        self._global_non_reference_positions = None
        self._global_reference_positions = None
        self._global_sequence_rows = None
        self._compact_rope_table = None


@dataclass
class MiniMaxH3ReferenceKVObserverStats(MiniMaxH3ReferenceKVTier1Stats):
    observed_steps: int = 0
    sampled_layers: int = 0
    metric_records: int = 0
    sampled_d2h_bytes: int = 0


class MiniMaxH3ReferenceKVObserverState(MiniMaxH3ReferenceKVTier1State):
    """Measure reference-KV drift without changing attention inputs.

    The observer samples reference rows and KV heads after QK norm and RoPE,
    compares the current values with hypothetical fixed-interval snapshots, and
    always returns the original K/V tensors unchanged. It is therefore suitable
    for collecting drift on the unmodified MiniMax-H3 denoise trajectory.
    """

    tier_name = "Observer"
    stats: MiniMaxH3ReferenceKVObserverStats

    def __init__(
        self,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
        intervals: tuple[int, ...],
        sample_rows: int,
        sample_heads: int,
        output_path: str,
    ) -> None:
        super().__init__(
            num_layers=num_layers,
            global_reference_rows=global_reference_rows,
            device=device,
            refresh_interval=1,
            pin_memory=False,
        )
        if not intervals or any(interval <= 1 for interval in intervals):
            raise ValueError(f"observer intervals must all be > 1, got {intervals}")
        if sample_rows <= 0 or sample_heads <= 0:
            raise ValueError(
                f"observer sample_rows and sample_heads must both be positive, got {sample_rows} and {sample_heads}"
            )
        self.intervals = tuple(sorted(set(int(value) for value in intervals)))
        self.sample_rows = int(sample_rows)
        self.sample_heads = int(sample_heads)
        self.stats = MiniMaxH3ReferenceKVObserverStats()

        self._run_id = f"{os.getpid()}-{time.time_ns()}"
        self._output_path = self._resolve_output_path(output_path)
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._snapshots: torch.Tensor | None = None
        self._source_steps: list[int | None] = [None] * len(self.intervals)
        self._source_sigmas: list[float | None] = [None] * len(self.intervals)
        self._refresh_indices: set[int] = set()
        self._sampled_positions: torch.Tensor | None = None
        self._sampled_heads: torch.Tensor | None = None
        self._sampled_row_indices: list[int] = []
        self._sampled_head_indices: list[int] = []
        self._pending_records: list[dict[str, Any]] = []
        self._sigma: float | None = None

        self._write_records(
            [
                {
                    "kind": "run_start",
                    "run_id": self._run_id,
                    "pid": os.getpid(),
                    "device": str(self.device),
                    "num_layers": self.num_layers,
                    "global_reference_rows": self.global_reference_rows,
                    "intervals": list(self.intervals),
                    "requested_sample_rows": self.sample_rows,
                    "requested_sample_heads": self.sample_heads,
                    "metric_definition": {
                        "cosine": "per sampled token-head vector over head_dim",
                        "rel_l2": "norm(current-cached)/norm(current)",
                        "normalized_max": "max_abs_error/rms(current)",
                    },
                }
            ]
        )

    def _resolve_output_path(self, raw_path: str) -> Path:
        base = Path(raw_path)
        device_label = f"{self.device.type}{self.device.index}" if self.device.index is not None else self.device.type
        if base.suffix:
            return base.with_name(f"{base.stem}.{device_label}{base.suffix}")
        return base / f"minimax_h3_reference_kv_drift.{device_label}.jsonl"

    @property
    def output_path(self) -> str:
        return str(self._output_path)

    @property
    def host_bytes(self) -> int:
        if self._snapshots is None:
            return 0
        return self._snapshots.numel() * self._snapshots.element_size()

    @property
    def device_pool(self) -> None:
        return None

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        with self._output_path.open("a", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _evenly_spaced_indices(
        total: int,
        limit: int,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        count = min(total, limit)
        if count == total:
            return torch.arange(total, device=device, dtype=torch.long)
        return torch.linspace(0, total - 1, steps=count, device=device).round().to(torch.long).unique(sorted=True)

    def _initialize_layout(self, k: torch.Tensor) -> None:
        positions = self._local_reference_positions
        if positions is None:
            raise RuntimeError("observer reference mask was not initialized")
        row_indices = self._evenly_spaced_indices(
            positions.numel(),
            self.sample_rows,
            device=positions.device,
        )
        head_indices = self._evenly_spaced_indices(
            k.shape[1],
            self.sample_heads,
            device=k.device,
        )
        self._sampled_positions = positions.index_select(0, row_indices)
        self._sampled_heads = head_indices
        self._sampled_row_indices = row_indices.cpu().tolist()
        self._sampled_head_indices = head_indices.cpu().tolist()

        shape = (
            len(self.intervals),
            2,
            self.num_layers,
            row_indices.numel(),
            head_indices.numel(),
            k.shape[-1],
        )
        self._snapshots = torch.empty(shape, dtype=k.dtype, device="cpu")
        self._write_records(
            [
                {
                    "kind": "layout",
                    "run_id": self._run_id,
                    "local_reference_rows": positions.numel(),
                    "sampled_reference_rows": row_indices.numel(),
                    "sampled_kv_heads": head_indices.numel(),
                    "head_dim": k.shape[-1],
                    "sampled_row_indices": self._sampled_row_indices,
                    "sampled_head_indices": self._sampled_head_indices,
                    "snapshot_bytes": self.host_bytes,
                }
            ]
        )
        _log.info(
            "MiniMax-H3 reference-KV Observer layout: local_ref_rows=%d, "
            "sampled_rows=%d, sampled_heads=%d, head_dim=%d, "
            "snapshot_bytes=%d, output=%s",
            positions.numel(),
            row_indices.numel(),
            head_indices.numel(),
            k.shape[-1],
            self.host_bytes,
            self.output_path,
        )

    def _sample_current(self, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        if self._sampled_positions is None or self._sampled_heads is None:
            self._initialize_layout(k)
        assert self._sampled_positions is not None
        assert self._sampled_heads is not None
        sampled_k = k.index_select(0, self._sampled_positions).index_select(1, self._sampled_heads)
        sampled_v = v.index_select(0, self._sampled_positions).index_select(1, self._sampled_heads)
        sampled = (
            torch.stack((sampled_k, sampled_v))
            .detach()
            .to(
                device="cpu",
                non_blocking=False,
            )
        )
        self.stats.sampled_d2h_bytes += sampled.numel() * sampled.element_size()
        return sampled.contiguous()

    @staticmethod
    def _metrics_batch(
        current: torch.Tensor,
        cached: torch.Tensor,
        *,
        eps: float = 1e-12,
    ) -> dict[str, torch.Tensor]:
        """Vectorize CPU metrics over cached versions and K/V."""
        current_vectors = current.float().reshape(1, 2, -1, current.shape[-1])
        cached_vectors = cached.float().reshape(cached.shape[0], 2, -1, cached.shape[-1])
        diff_vectors = current_vectors - cached_vectors

        current_norm = torch.linalg.vector_norm(current_vectors, dim=-1)
        cached_norm = torch.linalg.vector_norm(cached_vectors, dim=-1)
        cosine = (current_vectors * cached_vectors).sum(dim=-1) / (current_norm * cached_norm).clamp_min(eps)
        relative = torch.linalg.vector_norm(diff_vectors, dim=-1) / (current_norm.clamp_min(eps))
        absolute = diff_vectors.abs().flatten(start_dim=2)
        global_relative = torch.linalg.vector_norm(
            diff_vectors.flatten(start_dim=2), dim=-1
        ) / torch.linalg.vector_norm(current_vectors.flatten(start_dim=2), dim=-1).clamp_min(eps)
        rms = current_vectors.square().mean(dim=(-1, -2)).sqrt().clamp_min(eps)

        return {
            "cosine_mean": cosine.mean(dim=-1),
            "cosine_p50": torch.quantile(cosine, 0.50, dim=-1),
            "cosine_p01": torch.quantile(cosine, 0.01, dim=-1),
            "cosine_min": cosine.amin(dim=-1),
            "rel_l2": global_relative,
            "rel_l2_p50": torch.quantile(relative, 0.50, dim=-1),
            "rel_l2_p99": torch.quantile(relative, 0.99, dim=-1),
            "rel_l2_max": relative.amax(dim=-1),
            "abs_error_p99": torch.quantile(absolute, 0.99, dim=-1),
            "abs_error_max": absolute.amax(dim=-1),
            "normalized_max": absolute.amax(dim=-1) / rms,
        }

    def begin_step(self, step: int, *, sigma: float | None = None) -> None:
        if self._step is not None:
            raise RuntimeError(f"reference-KV Observer step {self._step} is still active")
        self._step = int(step)
        self._sigma = None if sigma is None else float(sigma)
        self._layers_seen.clear()
        self._pending_records.clear()
        self._refresh_indices = {
            index
            for index, interval in enumerate(self.intervals)
            if self._source_steps[index] is None or step % interval == 0
        }

    def set_local_reference_mask(self, reference_mask: torch.Tensor) -> None:
        if reference_mask.dim() != 1 or reference_mask.dtype != torch.bool:
            raise ValueError(
                "reference_mask must be a 1-D bool tensor, got "
                f"shape={tuple(reference_mask.shape)} dtype={reference_mask.dtype}"
            )
        positions = torch.nonzero(reference_mask, as_tuple=False).view(-1)
        if self._local_reference_positions is None:
            self._local_reference_positions = positions
            self._local_sequence_rows = reference_mask.numel()
            self._inactive = positions.numel() == 0
            return
        if (
            self._local_sequence_rows != reference_mask.numel()
            or self._local_reference_positions.numel() != positions.numel()
        ):
            raise RuntimeError("MiniMax-H3 observer reference row layout changed within a request")

    def process_layer(
        self,
        layer_index: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._step is None:
            raise RuntimeError("begin_step must be called before observing KV")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} outside [0, {self.num_layers})")
        if layer_index in self._layers_seen:
            raise RuntimeError(f"MiniMax-H3 layer {layer_index} observed twice in step {self._step}")
        if k.shape != v.shape:
            raise ValueError(f"K/V shape mismatch: {k.shape} != {v.shape}")
        positions = self._local_reference_positions
        if positions is None:
            raise RuntimeError("observer local reference mask was not initialized")
        if self._local_sequence_rows != k.shape[0]:
            raise ValueError(f"local reference mask/KV row mismatch: {self._local_sequence_rows} != {k.shape[0]}")
        if self._inactive or positions.numel() == 0:
            self._layers_seen.add(layer_index)
            return k, v

        sampled = self._sample_current(k, v)
        assert self._snapshots is not None
        reuse_indices: list[int] = []
        for interval_index in range(len(self.intervals)):
            if interval_index in self._refresh_indices:
                self._snapshots[interval_index, :, layer_index].copy_(sampled)
            else:
                reuse_indices.append(interval_index)

        if reuse_indices:
            cached = self._snapshots[reuse_indices, :, layer_index]
            metrics = self._metrics_batch(sampled, cached)
            for batch_index, interval_index in enumerate(reuse_indices):
                interval = self.intervals[interval_index]
                source_step = self._source_steps[interval_index]
                if source_step is None:
                    raise RuntimeError("observer reuse requested before snapshot capture")
                age = self._step - source_step
                source_sigma = self._source_sigmas[interval_index]
                sigma_delta = None if self._sigma is None or source_sigma is None else self._sigma - source_sigma
                for kv_index, kv_name in enumerate(("K", "V")):
                    record: dict[str, Any] = {
                        "kind": "metric",
                        "run_id": self._run_id,
                        "step": self._step,
                        "sigma": self._sigma,
                        "layer": layer_index,
                        "interval": interval,
                        "source_step": source_step,
                        "source_sigma": source_sigma,
                        "sigma_delta": sigma_delta,
                        "age": age,
                        "kv": kv_name,
                    }
                    record.update({name: float(values[batch_index, kv_index]) for name, values in metrics.items()})
                    self._pending_records.append(record)
        self.stats.sampled_layers += 1
        self._layers_seen.add(layer_index)
        return k, v

    def end_step(self) -> None:
        if self._step is None:
            raise RuntimeError("no active reference-KV Observer step")
        if len(self._layers_seen) != self.num_layers:
            raise RuntimeError(
                "MiniMax-H3 reference-KV Observer saw "
                f"{len(self._layers_seen)}/{self.num_layers} layers "
                f"in step {self._step}"
            )
        for interval_index in self._refresh_indices:
            self._source_steps[interval_index] = self._step
            self._source_sigmas[interval_index] = self._sigma
        if not self._inactive:
            self.stats.observed_steps += 1
        self._write_records(self._pending_records)
        self.stats.metric_records += len(self._pending_records)
        self._pending_records.clear()
        self._refresh_indices.clear()
        self._step = None
        self._sigma = None
        self._layers_seen.clear()

    def abort_step(self) -> None:
        self._pending_records.clear()
        self._refresh_indices.clear()
        self._step = None
        self._sigma = None
        self._layers_seen.clear()

    def close(self) -> None:
        if self._step is not None:
            self.abort_step()
        self._write_records(
            [
                {
                    "kind": "run_end",
                    "run_id": self._run_id,
                    "observed_steps": self.stats.observed_steps,
                    "sampled_layers": self.stats.sampled_layers,
                    "metric_records": self.stats.metric_records,
                    "sampled_d2h_bytes": self.stats.sampled_d2h_bytes,
                    "snapshot_bytes": self.host_bytes,
                    "source_steps": {
                        str(interval): source
                        for interval, source in zip(self.intervals, self._source_steps, strict=True)
                    },
                    "source_sigmas": {
                        str(interval): sigma
                        for interval, sigma in zip(self.intervals, self._source_sigmas, strict=True)
                    },
                    "inactive": self._inactive,
                }
            ]
        )
        _log.info(
            "MiniMax-H3 reference-KV Observer summary: output=%s, "
            "observed_steps=%d, sampled_layers=%d, metric_records=%d, "
            "sampled_d2h_bytes=%d, snapshot_bytes=%d, inactive=%s",
            self.output_path,
            self.stats.observed_steps,
            self.stats.sampled_layers,
            self.stats.metric_records,
            self.stats.sampled_d2h_bytes,
            self.host_bytes,
            self._inactive,
        )
        self._snapshots = None
        self._sampled_positions = None
        self._sampled_heads = None
        self._local_reference_positions = None
        self._local_sequence_rows = None


__all__ = [
    "MiniMaxH3ReferenceKVTier1State",
    "MiniMaxH3ReferenceKVTier1Stats",
    "MiniMaxH3ReferenceKVTier2State",
    "MiniMaxH3ReferenceKVTier2Stats",
    "MiniMaxH3ReferenceKVObserverState",
    "MiniMaxH3ReferenceKVObserverStats",
]
