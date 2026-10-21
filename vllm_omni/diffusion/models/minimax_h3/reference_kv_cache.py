# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Post-Ulysses reference-KV cache with layerwise host-to-device prefetch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.profiler import record_function
from vllm.logger import init_logger

from vllm_omni.diffusion.kv_offload_config import ReferenceKVConfig
from vllm_omni.platforms import current_omni_platform

_log = init_logger(__name__)


@dataclass
class MiniMaxH3ReferenceKVStats:
    block_cached_layers: int = 0
    refresh_steps: int = 0
    h2d_steps: int = 0
    captured_layers: int = 0
    substituted_layers: int = 0
    h2d_bytes: int = 0
    projection_skipped_layers: int = 0
    projection_skipped_rows: int = 0
    projection_total_rows: int = 0
    prefetched_layers: int = 0
    ready_wait_layers: int = 0
    max_device_staging_bytes: int = 0
    dequantized_layers: int = 0
    quantization_fallbacks: int = 0


class _MiniMaxH3ReferenceKVBase:
    """Request-local refresh schedule and reference-row bookkeeping."""

    stats: MiniMaxH3ReferenceKVStats

    def __init__(
        self,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
        refresh_interval: int,
        pin_memory: bool = True,
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
        if host_quantization not in {"none", "fp8", "int8"}:
            raise ValueError(f"host_quantization must be one of none, fp8, int8, got {host_quantization!r}")
        self.host_quantization = host_quantization
        self.effective_host_quantization = host_quantization
        self.stats = MiniMaxH3ReferenceKVStats()

        self._host_pool: torch.Tensor | None = None
        self._host_scale_pool: torch.Tensor | None = None
        self._local_sequence_rows: int | None = None
        self._global_sequence_rows: int | None = None
        self._global_non_reference_positions: torch.Tensor | None = None
        self._global_reference_positions: torch.Tensor | None = None
        self._compact_rope_table: torch.Tensor | None = None
        self._step: int | None = None
        self._refresh = False
        self._global_refresh = False
        self._layers_seen: set[int] = set()

    @property
    def host_bytes(self) -> int:
        total = 0
        if self._host_pool is not None:
            total += self._host_pool.numel() * self._host_pool.element_size()
        if self._host_scale_pool is not None:
            total += self._host_scale_pool.numel() * self._host_scale_pool.element_size()
        return total

    @property
    def current_step(self) -> int | None:
        return self._step

    @property
    def reference_epoch(self) -> int:
        if self._step is None:
            raise RuntimeError("no active reference-KV step")
        return self._step // self.refresh_interval if self.refresh_interval > 0 else 0

    @property
    def is_reference_refresh_step(self) -> bool:
        return self._step is not None and self._global_refresh

    @property
    def supports_block_cache_reuse(self) -> bool:
        return True

    def mark_block_cache_skipped(self, layer_indices: list[int]) -> None:
        """Account for a Cache-DiT middle span without reading its reference KV."""
        if self._step is None or self._global_refresh or not self.supports_block_cache_reuse:
            raise RuntimeError("block cache skips are only allowed on reference-KV reuse steps")
        next_layer = len(self._layers_seen)
        if layer_indices != list(range(next_layer, next_layer + len(layer_indices))):
            raise RuntimeError("block cache skips must follow the executed layers in order")
        if any(index >= self.num_layers for index in layer_indices):
            raise IndexError("block cache skip exceeds reference-KV layer count")
        self._layers_seen.update(layer_indices)
        self.stats.block_cached_layers += len(layer_indices)

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
            raise RuntimeError("local sequence size was not initialized")
        return self._local_sequence_rows

    def set_local_sequence_rows(self, rows: int) -> None:
        if rows <= 0:
            raise ValueError("local sequence size must be positive")
        if self._local_sequence_rows is None:
            self._local_sequence_rows = rows
        elif self._local_sequence_rows != rows:
            raise RuntimeError("local sequence size changed within one denoise request")

    @property
    def should_skip_reference_projection(self) -> bool:
        """Whether this global reuse step may compact reference query rows.

        Every rank must compact the same target sequence on reuse steps.
        """
        return self._step is not None and not self._global_refresh

    def record_projection_skip(
        self,
        layer_index: int,
        total_rows: int,
        *,
        projected_rows: int,
    ) -> None:
        if not self.should_skip_reference_projection:
            raise RuntimeError("reference projection skip recorded outside reuse step")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError(f"layer_index {layer_index} outside [0, {self.num_layers})")
        if not 0 <= projected_rows <= total_rows:
            raise ValueError(f"projected_rows must be within [0, {total_rows}], got {projected_rows}")
        self.stats.projection_skipped_layers += 1
        self.stats.projection_skipped_rows += total_rows - projected_rows
        self.stats.projection_total_rows += total_rows

    def _is_refresh_step(self, step: int) -> bool:
        if self._host_pool is None:
            return True
        return self.refresh_interval > 0 and step % self.refresh_interval == 0


class MiniMaxH3ReferenceKVTier2State(_MiniMaxH3ReferenceKVBase):
    """Layerwise asynchronous H2D staging backed by a small device ring.

    Refresh steps capture the balanced K/V head shard after Ulysses All-to-All.
    Reuse steps prefetch one layer at a time from the pinned host pool. Layer
    ``i + 1`` is submitted immediately after layer ``i`` consumes its slot, so
    its H2D can overlap the current layer's attention and MLP work.
    """

    tier_name = "Tier2"
    stats: MiniMaxH3ReferenceKVStats

    @classmethod
    def from_config(
        cls,
        *,
        config: ReferenceKVConfig | None,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
    ) -> MiniMaxH3ReferenceKVTier2State | None:
        if config is None:
            return None
        if config.mode != "on":
            raise ValueError(f"unsupported MiniMax-H3 reference-KV mode: {config.mode!r}")
        if global_reference_rows <= 0:
            _log.info("MiniMax-H3 reference-KV skipped: request has no visual condition rows")
            return None
        if config.kv_refresh_interval == 0:
            _log.warning("MiniMax-H3 reference-KV populate-once mode is lossy with bidirectional attention")
        state = cls(
            num_layers=num_layers,
            global_reference_rows=global_reference_rows,
            device=device,
            refresh_interval=config.kv_refresh_interval,
            host_quantization=config.kv_host_quantization,
        )
        _log.info(
            "MiniMax-H3 reference-KV enabled: layers=%d, global_ref_rows=%d, "
            "refresh_interval=%d, device=%s, ring_size=%d, "
            "host_quantization=%s, skip_reference_projection=True, cache_layout=post_ulysses",
            num_layers,
            global_reference_rows,
            config.kv_refresh_interval,
            device,
            state.ring_size,
            state.host_quantization,
        )
        return state

    def __init__(
        self,
        *,
        num_layers: int,
        global_reference_rows: int,
        device: torch.device,
        refresh_interval: int,
        ring_size: int = 2,
        pin_memory: bool = True,
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
            host_quantization=host_quantization,
        )
        self.ring_size = int(ring_size)
        self.stats = MiniMaxH3ReferenceKVStats()
        self._device_ring: torch.Tensor | None = None
        self._device_scale_ring: torch.Tensor | None = None
        self._copy_stream: Any | None = None
        self._slot_layers: list[int | None] = [None] * self.ring_size
        self._ready_events: list[Any | None] = [None] * self.ring_size
        self._consumed_events: list[Any | None] = [None] * self.ring_size

    @property
    def device_staging_bytes(self) -> int:
        total = 0
        if self._device_ring is not None:
            total += self._device_ring.numel() * self._device_ring.element_size()
        if self._device_scale_ring is not None:
            total += self._device_scale_ring.numel() * self._device_scale_ring.element_size()
        return total

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
        if not compact:
            raise RuntimeError("reference-KV reuse must skip reference projection")

        cached_k, cached_v = self.take_cached_reference_layer(layer_index, key)
        if cached_k.shape[0] != self.global_reference_rows:
            raise ValueError(
                f"cached post-Ulysses reference row mismatch: {cached_k.shape[0]} != {self.global_reference_rows}"
            )
        if cached_k.shape[1:] != key.shape[2:]:
            raise ValueError(
                f"cached post-Ulysses reference head layout mismatch: {cached_k.shape[1:]} != {key.shape[2:]}"
            )
        return (
            torch.cat((cached_k.unsqueeze(0), key), dim=1),
            torch.cat((cached_v.unsqueeze(0), value), dim=1),
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

    def mark_block_cache_skipped(self, layer_indices: list[int]) -> None:
        super().mark_block_cache_skipped(layer_indices)
        next_layer = len(self._layers_seen)
        if next_layer < self.num_layers:
            slot = next_layer % self.ring_size
            if self._slot_layers[slot] != next_layer:
                self._prefetch_layer(next_layer)

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
            "captured_layers=%d, substituted_layers=%d, block_cached_layers=%d, "
            "prefetched_layers=%d, ready_wait_layers=%d, h2d_bytes=%d, "
            "projection_skipped_layers=%d, projection_skipped_rows=%d, "
            "projection_total_rows=%d, cache_layout=post_ulysses, "
            "host_quantization=%s, effective_host_quantization=%s, "
            "dequantized_layers=%d, quantization_fallbacks=%d",
            self.host_bytes,
            self.device_staging_bytes,
            self.stats.refresh_steps,
            self.stats.h2d_steps,
            self.stats.captured_layers,
            self.stats.substituted_layers,
            self.stats.block_cached_layers,
            self.stats.prefetched_layers,
            self.stats.ready_wait_layers,
            self.stats.h2d_bytes,
            self.stats.projection_skipped_layers,
            self.stats.projection_skipped_rows,
            self.stats.projection_total_rows,
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
        self._local_sequence_rows = None
        self._global_non_reference_positions = None
        self._global_reference_positions = None
        self._global_sequence_rows = None
        self._compact_rope_table = None
