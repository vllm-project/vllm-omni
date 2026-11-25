# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Prefix-only double-buffered BF16 delta streamer for resident quantized DiTs.

Scheme A (``prefix_bf16_master_mode="stream"``):
- Quantized weights stay resident on the accelerator for zero offload overhead
  on all non-fallback steps (e.g. steps 5..49).
- Only the BF16 delta weights for wrapped ``TimestepBF16FallbackLinearMethod``
  layers live in pageable CPU RAM, optionally sharded across the active DP/SP
  rank group (e.g. 1/4 per rank under USP=4).
- During active fallback steps (e.g. steps 0..4), a fixed 2-slot accelerator
  ring buffer (``2 * max_block_bf16_bytes``) overlaps CPU->device DMA and
  AllGather for block ``i+1`` with compute for block ``i``.
- Uses a dedicated collective communicator (``dist.new_group``) when sharding
  over SP/DP so weight AllGather never contends with compute-stream Ulysses
  AllToAll or Ring collectives.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform
from vllm_omni.quantization.timestep_bf16_fallback import (
    BF16_FALLBACK_MANAGED_ATTR,
    BF16_FALLBACK_WEIGHT_ATTR,
    TimestepBF16Policy,
    current_denoise_step,
    iter_bf16_fallback_layers,
)

from .block_discovery import get_blocks_from_dit
from .module_collector import ModuleDiscovery
from .offload_plan import get_offload_plan

logger = init_logger(__name__)

AllGatherFn = Callable[[torch.Tensor, torch.Tensor], None]
DEFAULT_PINNED_CHUNK_BYTES = 32 * 1024 * 1024  # 32 MiB per chunk slot (64 MiB/rank total)


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class _LayerSlice:
    layer: nn.Module
    offset: int
    numel: int
    shape: tuple[int, ...]


@dataclass
class _BlockPayload:
    block_idx: int
    block: nn.Module
    slices: tuple[_LayerSlice, ...]
    cpu_shard: torch.Tensor
    total_numel: int
    padded_numel: int
    shard_numel: int


class _BlockRingStreamer:
    """Manages one contiguous block sequence (e.g. ``transformer.blocks``)."""

    def __init__(
        self,
        name: str,
        blocks: Sequence[nn.Module],
        policy: TimestepBF16Policy,
        device: torch.device,
        *,
        shard_size: int = 1,
        shard_rank: int = 0,
        comm_group: dist.ProcessGroup | None = None,
        all_gather_fn: AllGatherFn | None = None,
        use_pinned_staging: bool | None = None,
        pinned_chunk_bytes: int = DEFAULT_PINNED_CHUNK_BYTES,
    ) -> None:
        if shard_size < 1 or not (0 <= shard_rank < shard_size):
            raise ValueError(f"Invalid shard topology: shard_rank={shard_rank}, shard_size={shard_size}")
        if shard_size > 1 and comm_group is None and all_gather_fn is None:
            raise ValueError("Multi-rank prefix streaming requires comm_group or all_gather_fn")
        if pinned_chunk_bytes <= 0:
            raise ValueError(f"pinned_chunk_bytes must be positive, got {pinned_chunk_bytes}")

        self.name = name
        self.policy = policy
        self.device = device
        self.shard_size = shard_size
        self.shard_rank = shard_rank
        self.comm_group = comm_group
        self._all_gather_fn = all_gather_fn
        self._is_device_accel = device.type != "cpu"
        self._use_pinned_staging = self._is_device_accel if use_pinned_staging is None else bool(use_pinned_staging)
        self._pinned_chunk_bytes = int(pinned_chunk_bytes)

        self._payloads: list[_BlockPayload] = []
        self._block_to_ring_idx: dict[int, int] = {}
        self._handles: list[Any] = []
        self._managed_layers: list[nn.Module] = []

        self._pack_blocks(blocks)
        if not self._payloads:
            return

        max_padded = max(p.padded_numel for p in self._payloads)
        max_shard = max(p.shard_numel for p in self._payloads)

        # Fixed 2-slot accelerator ring buffer (ping-pong A/B).
        self._gpu_slots: list[torch.Tensor] = [
            torch.empty(max_padded, dtype=torch.bfloat16, device=device) for _ in range(2)
        ]
        self._gpu_shard_slots: list[torch.Tensor] | None = (
            [torch.empty(max_shard, dtype=torch.bfloat16, device=device) for _ in range(2)]
            if self.shard_size > 1
            else None
        )
        # Use bounded ping-pong pinned chunks (e.g. 2 x 32 MiB = 64 MiB/rank)
        # so multi-rank workers share the host pinned pool without starving
        # sibling ranks or IPC video buffers.
        self._pinned_chunk_slots: list[torch.Tensor] | None = None
        self._pinned_chunk_numel: int = 0
        if self._use_pinned_staging:
            base_chunk_numel = min(max_shard, max(1, self._pinned_chunk_bytes // 2))
            candidates = [base_chunk_numel]
            if base_chunk_numel > 8 * 1024 * 1024:
                candidates.append(8 * 1024 * 1024)  # 16 MiB fallback
            last_exc: RuntimeError | None = None
            for cand in candidates:
                try:
                    self._pinned_chunk_slots = [
                        torch.empty(cand, dtype=torch.bfloat16, device="cpu", pin_memory=True) for _ in range(2)
                    ]
                    self._pinned_chunk_numel = cand
                    break
                except RuntimeError as exc:
                    self._pinned_chunk_slots = None
                    last_exc = exc
            if self._pinned_chunk_slots is None and last_exc is not None:
                logger.warning(
                    "Prefix BF16 streamer %s failed to allocate pinned staging (%s); falling back to pageable H2D",
                    self.name,
                    last_exc,
                )

        self._copy_stream = current_omni_platform.Stream() if self._is_device_accel else None
        self._comm_stream = current_omni_platform.Stream() if (self._is_device_accel and self.shard_size > 1) else None

        # Synchronization events protecting shared slots across overlapping streams.
        self._slot_ready: list[Any | None] = [None, None]
        self._pinned_chunk_done: list[Any | None] = [None, None]
        self._slot_compute_done: list[Any | None] = [None, None]
        self._prefetched_ring_idx: list[int | None] = [None, None]

        # Counters for observability and testing.
        self.prefetch_count: int = 0
        self.sync_cold_prefetch_count: int = 0

        self._register_hooks()

        ring_bytes = 2 * max_padded * 2
        shard_gpu_bytes = (2 * max_shard * 2) if self.shard_size > 1 else 0
        cpu_bytes = sum(p.cpu_shard.numel() * 2 for p in self._payloads)
        pinned_bytes = (2 * self._pinned_chunk_numel * 2) if self._pinned_chunk_slots is not None else 0
        logger.info(
            "Enabled PrefixBlockStreamer[%s]: blocks=%d layers=%d shard=%d/%d "
            "device_ring=%.2f MiB device_shard=%.2f MiB host_pageable=%.2f GiB host_pinned=%.2f MiB steps=%s",
            self.name,
            len(self._payloads),
            len(self._managed_layers),
            self.shard_rank,
            self.shard_size,
            ring_bytes / (1024**2),
            shard_gpu_bytes / (1024**2),
            cpu_bytes / (1024**3),
            pinned_bytes / (1024**2),
            sorted(self.policy.steps),
        )

    @property
    def managed_layer_ids(self) -> set[int]:
        return {id(layer) for layer in self._managed_layers}

    def _pack_blocks(self, blocks: Sequence[nn.Module]) -> None:
        for block in blocks:
            entries: list[tuple[nn.Module, torch.Tensor]] = []
            for _, child, wrapper in iter_bf16_fallback_layers(block):
                if wrapper.policy.master_mode != "stream":
                    continue
                weight = getattr(child, BF16_FALLBACK_WEIGHT_ATTR, None)
                if not isinstance(weight, torch.Tensor) or weight.numel() == 0:
                    raise RuntimeError(
                        f"PrefixBlockStreamer[{self.name}] found stream-mode BF16 layer without "
                        "materialized CPU BF16 weight"
                    )
                entries.append((child, weight))
            if not entries:
                continue

            total_numel = sum(w.numel() for _, w in entries)
            shard_numel = _ceil_div(total_numel, self.shard_size)
            padded_numel = shard_numel * self.shard_size

            # Pack only this rank's contiguous slice directly into pageable CPU RAM
            # without retaining all ranks' copies simultaneously.
            shard_begin = self.shard_rank * shard_numel
            shard_end = shard_begin + shard_numel
            cpu_shard = torch.zeros(shard_numel, dtype=torch.bfloat16, device="cpu")

            slices: list[_LayerSlice] = []
            offset = 0
            for child, weight in entries:
                numel = weight.numel()
                shape = tuple(weight.shape)
                slices.append(_LayerSlice(layer=child, offset=offset, numel=numel, shape=shape))

                ov_begin = max(offset, shard_begin)
                ov_end = min(offset + numel, shard_end)
                if ov_begin < ov_end:
                    flat = weight.detach().to(device="cpu", dtype=torch.bfloat16).reshape(-1)
                    cpu_shard[ov_begin - shard_begin : ov_end - shard_begin].copy_(
                        flat[ov_begin - offset : ov_end - offset]
                    )

                # Release the unsharded per-layer CPU tensor immediately.
                setattr(child, BF16_FALLBACK_WEIGHT_ATTR, None)
                setattr(child, BF16_FALLBACK_MANAGED_ATTR, "stream")
                self._managed_layers.append(child)
                offset += numel

            ring_idx = len(self._payloads)
            self._block_to_ring_idx[id(block)] = ring_idx
            self._payloads.append(
                _BlockPayload(
                    block_idx=ring_idx,
                    block=block,
                    slices=tuple(slices),
                    cpu_shard=cpu_shard,
                    total_numel=total_numel,
                    padded_numel=padded_numel,
                    shard_numel=shard_numel,
                )
            )

    def _run_all_gather(self, dst_full: torch.Tensor, src_shard: torch.Tensor) -> None:
        if self._all_gather_fn is not None:
            self._all_gather_fn(dst_full, src_shard)
        else:
            assert self.comm_group is not None
            dist.all_gather_into_tensor(dst_full, src_shard, group=self.comm_group)

    def _copy_h2d_on_copy_stream(self, dst_device: torch.Tensor, src_cpu: torch.Tensor) -> Any:
        """Stream ``src_cpu`` into ``dst_device`` on ``self._copy_stream`` and return completion event."""
        assert self._copy_stream is not None
        if self._pinned_chunk_slots is None:
            evt = current_omni_platform.Event()
            with current_omni_platform.stream(self._copy_stream):
                dst_device.copy_(src_cpu, non_blocking=False)
                evt.record(self._copy_stream)
            return evt

        total = src_cpu.numel()
        chunk_sz = self._pinned_chunk_numel
        last_evt: Any | None = None
        chunk_idx = 0
        for off in range(0, total, chunk_sz):
            end = min(off + chunk_sz, total)
            buf_idx = chunk_idx & 1
            prev_evt = self._pinned_chunk_done[buf_idx]
            if prev_evt is not None:
                prev_evt.synchronize()
                self._pinned_chunk_done[buf_idx] = None
            staged = self._pinned_chunk_slots[buf_idx][: end - off]
            staged.copy_(src_cpu[off:end])
            evt = current_omni_platform.Event()
            with current_omni_platform.stream(self._copy_stream):
                dst_device[off:end].copy_(staged, non_blocking=True)
                evt.record(self._copy_stream)
            self._pinned_chunk_done[buf_idx] = evt
            last_evt = evt
            chunk_idx += 1
        assert last_evt is not None
        return last_evt

    @torch.compiler.disable
    def _launch_prefetch(self, ring_idx: int, slot: int) -> None:
        payload = self._payloads[ring_idx]
        self.prefetch_count += 1

        if not self._is_device_accel:
            if self.shard_size > 1:
                assert self._gpu_shard_slots is not None
                shard_dst = self._gpu_shard_slots[slot][: payload.shard_numel]
                shard_dst.copy_(payload.cpu_shard)
                full_dst = self._gpu_slots[slot][: payload.padded_numel]
                self._run_all_gather(full_dst, shard_dst)
            else:
                self._gpu_slots[slot][: payload.total_numel].copy_(payload.cpu_shard)
            self._prefetched_ring_idx[slot] = ring_idx
            return

        assert self._copy_stream is not None
        if self.shard_size > 1:
            assert self._gpu_shard_slots is not None
            assert self._comm_stream is not None
            # Wait for any previous AllGather reading gpu_shard_slots[slot] before overwriting it.
            prev_ready = self._slot_ready[slot]
            if prev_ready is not None:
                self._copy_stream.wait_event(prev_ready)

            gpu_shard = self._gpu_shard_slots[slot][: payload.shard_numel]
            h2d_evt = self._copy_h2d_on_copy_stream(gpu_shard, payload.cpu_shard)

            # Wait for H2D completion AND for previous block compute reading gpu_slots[slot].
            self._comm_stream.wait_event(h2d_evt)
            prev_compute = self._slot_compute_done[slot]
            if prev_compute is not None:
                self._comm_stream.wait_event(prev_compute)

            ready_evt = current_omni_platform.Event()
            with current_omni_platform.stream(self._comm_stream):
                gpu_full = self._gpu_slots[slot][: payload.padded_numel]
                self._run_all_gather(gpu_full, gpu_shard)
                ready_evt.record(self._comm_stream)
            self._slot_ready[slot] = ready_evt
        else:
            prev_compute = self._slot_compute_done[slot]
            if prev_compute is not None:
                self._copy_stream.wait_event(prev_compute)
            gpu_full = self._gpu_slots[slot][: payload.total_numel]
            ready_evt = self._copy_h2d_on_copy_stream(gpu_full, payload.cpu_shard)
            self._slot_ready[slot] = ready_evt

        self._prefetched_ring_idx[slot] = ring_idx

    def _next_prefetch_index(self, ring_idx: int, step: int) -> int | None:
        num_blocks = len(self._payloads)
        if ring_idx + 1 < num_blocks:
            return ring_idx + 1
        # Only hide block-0 cold start across consecutive active fallback steps.
        if self.policy.is_active(step + 1):
            return 0
        return None

    @torch.compiler.disable
    def _pre_forward(self, ring_idx: int) -> None:
        step = current_denoise_step()
        if not self.policy.is_active(step):
            return
        assert step is not None
        slot = ring_idx & 1
        if self._prefetched_ring_idx[slot] != ring_idx:
            self.sync_cold_prefetch_count += 1
            self._launch_prefetch(ring_idx, slot)

        if self._is_device_accel:
            ready_evt = self._slot_ready[slot]
            if ready_evt is not None:
                current_omni_platform.current_stream().wait_event(ready_evt)

        buf = self._gpu_slots[slot]
        payload = self._payloads[ring_idx]
        for sl in payload.slices:
            setattr(sl.layer, BF16_FALLBACK_WEIGHT_ATTR, buf[sl.offset : sl.offset + sl.numel].view(sl.shape))

        # Overlap prefetch for the next block when it targets the opposite ping-pong slot.
        next_idx = self._next_prefetch_index(ring_idx, step)
        if next_idx is not None:
            next_slot = next_idx & 1
            if next_slot != slot:
                self._launch_prefetch(next_idx, next_slot)

    @torch.compiler.disable
    def _post_forward(self, ring_idx: int) -> None:
        step = current_denoise_step()
        if not self.policy.is_active(step):
            return
        assert step is not None
        slot = ring_idx & 1
        if self._is_device_accel:
            compute_evt = current_omni_platform.Event()
            compute_evt.record(current_omni_platform.current_stream())
            self._slot_compute_done[slot] = compute_evt

        payload = self._payloads[ring_idx]
        for sl in payload.slices:
            setattr(sl.layer, BF16_FALLBACK_WEIGHT_ATTR, None)

        # Handle odd block counts where tail-to-head prefetch reuses the same slot.
        next_idx = self._next_prefetch_index(ring_idx, step)
        if next_idx is not None and (next_idx & 1) == slot:
            self._launch_prefetch(next_idx, slot)

    def _register_hooks(self) -> None:
        for payload in self._payloads:
            idx = payload.block_idx

            def _make_pre(ring_idx: int) -> Callable[..., None]:
                def _pre(_module: nn.Module, _inputs: Any) -> None:
                    self._pre_forward(ring_idx)

                return _pre

            def _make_post(ring_idx: int) -> Callable[..., None]:
                def _post(_module: nn.Module, _inputs: Any, _output: Any) -> None:
                    self._post_forward(ring_idx)

                return _post

            self._handles.append(payload.block.register_forward_pre_hook(_make_pre(idx)))
            self._handles.append(payload.block.register_forward_hook(_make_post(idx)))

    def disable(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        for layer in self._managed_layers:
            setattr(layer, BF16_FALLBACK_WEIGHT_ATTR, None)
            setattr(layer, BF16_FALLBACK_MANAGED_ATTR, None)
        self._gpu_slots.clear()
        if self._gpu_shard_slots is not None:
            self._gpu_shard_slots.clear()
        if self._pinned_chunk_slots is not None:
            self._pinned_chunk_slots.clear()
        self._payloads.clear()


def _resolve_prefix_shard_group() -> tuple[dist.ProcessGroup | None, int, int, bool]:
    """Create a dedicated communicator matching the active DP/SP shard group.

    Returns ``(comm_group, shard_size, shard_rank, owns_group)``.
    Using a dedicated group avoids HCCL/NCCL stream contention with compute-stream
    Ulysses AllToAll or Ring collectives on the SP communicator.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return None, 1, 0, False

    from vllm_omni.diffusion.distributed.parallel_state import (
        get_data_parallel_world_size,
        get_dp_group,
        get_sequence_parallel_world_size,
        get_sp_group,
    )

    dp_world = get_data_parallel_world_size()
    sp_world = get_sequence_parallel_world_size()
    if dp_world > 1:
        coord = get_dp_group()
    elif sp_world > 1:
        coord = get_sp_group()
    else:
        return None, 1, 0, False

    if coord.world_size <= 1:
        return None, 1, 0, False

    global_world = dist.get_world_size()
    my_ranks = list(coord.ranks)
    if len(my_ranks) == global_world:
        comm_group = dist.new_group(ranks=my_ranks)
        return comm_group, coord.world_size, coord.rank_in_group, True

    # Build peer subgroups symmetrically across all ranks to prevent new_group deadlocks.
    all_groups: list[list[int]] = [None] * global_world  # type: ignore[list-item]
    dist.all_gather_object(all_groups, my_ranks)
    unique_groups: list[tuple[int, ...]] = sorted({tuple(g) for g in all_groups})
    selected: dist.ProcessGroup | None = None
    my_tuple = tuple(my_ranks)
    for group_ranks in unique_groups:
        pg = dist.new_group(ranks=list(group_ranks))
        if group_ranks == my_tuple:
            selected = pg
    if selected is None:
        raise RuntimeError(f"Failed to create dedicated prefix AllGather group for ranks={my_ranks}")
    return selected, coord.world_size, coord.rank_in_group, True


def _discover_dit_blocks(pipeline: nn.Module, dit_name: str, dit: nn.Module) -> list[nn.Module]:
    plan = get_offload_plan(pipeline)
    block_attrs = plan.block_attrs.get(dit_name) if plan is not None else None
    _, blocks = get_blocks_from_dit(dit, block_attrs=block_attrs)
    if blocks:
        return blocks
    for fallback_attr in ("blocks", "transformer_blocks", "layers"):
        candidate = getattr(dit, fallback_attr, None)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return list(candidate)
    return []


class PrefixBF16FallbackManager:
    """Pipeline-level owner of ``stream`` / ``resident`` / ``dlo`` BF16 fallback."""

    def __init__(
        self,
        device: torch.device,
        *,
        shard_size: int | None = None,
        shard_rank: int = 0,
        comm_group: dist.ProcessGroup | None = None,
        all_gather_fn: AllGatherFn | None = None,
        use_pinned_staging: bool | None = None,
        pinned_chunk_bytes: int = DEFAULT_PINNED_CHUNK_BYTES,
    ) -> None:
        self.device = device
        self._explicit_topology = shard_size is not None
        self.shard_size = shard_size or 1
        self.shard_rank = shard_rank
        self.comm_group = comm_group
        self._owns_comm_group = False
        self._all_gather_fn = all_gather_fn
        self._use_pinned_staging = use_pinned_staging
        self._pinned_chunk_bytes = pinned_chunk_bytes
        self.streamers: list[_BlockRingStreamer] = []
        self.resident_fallback_layers: list[nn.Module] = []

    def enable(self, pipeline: nn.Module) -> bool:
        """Attach runtime management for all BF16-fallback layers on ``pipeline``."""
        all_layers = list(iter_bf16_fallback_layers(pipeline))
        if not all_layers:
            return False

        stream_layers = [(n, m, w) for n, m, w in all_layers if w.policy.master_mode == "stream"]
        resident_layers = [(n, m, w) for n, m, w in all_layers if w.policy.master_mode == "resident"]
        dlo_layers = [(n, m, w) for n, m, w in all_layers if w.policy.master_mode == "dlo"]

        # 1. Validate DLO-managed layers were actually adopted by DistributedLayerwiseOffloadHook.
        for name, layer, _ in dlo_layers:
            if not getattr(layer, "_layerwise_prefix_managed", False):
                raise RuntimeError(
                    f"Layer {name!r} requested prefix_bf16_master_mode='dlo' but was not claimed by "
                    "DistributedLayerwiseOffloadBackend; enable distributed layerwise offload for this "
                    "component or switch to prefix_bf16_master_mode='stream'"
                )
            setattr(layer, BF16_FALLBACK_MANAGED_ATTR, "dlo")

        # 2. Resident mode: move BF16 fallback weights straight onto the execution device.
        for name, layer, _ in resident_layers:
            weight = getattr(layer, BF16_FALLBACK_WEIGHT_ATTR, None)
            if not isinstance(weight, torch.Tensor) or weight.numel() == 0:
                raise RuntimeError(f"Resident BF16 fallback layer {name!r} lacks a materialized BF16 weight")
            setattr(layer, BF16_FALLBACK_WEIGHT_ATTR, weight.to(device=self.device, dtype=torch.bfloat16))
            setattr(layer, BF16_FALLBACK_MANAGED_ATTR, "resident")
            self.resident_fallback_layers.append(layer)

        # 3. Stream mode: attach per-DiT double-buffered block ring streamers.
        if stream_layers:
            if not self._explicit_topology:
                (
                    self.comm_group,
                    self.shard_size,
                    self.shard_rank,
                    self._owns_comm_group,
                ) = _resolve_prefix_shard_group()

            discovered = ModuleDiscovery.discover(pipeline)
            managed_ids: set[int] = set()
            for dit_name, dit in zip(discovered.dit_names, discovered.dits):
                dit_stream_layers = [
                    (n, m, w) for n, m, w in iter_bf16_fallback_layers(dit) if w.policy.master_mode == "stream"
                ]
                if not dit_stream_layers:
                    continue
                policies = {w.policy for _, _, w in dit_stream_layers}
                if len(policies) != 1:
                    raise ValueError(
                        f"All stream-mode BF16 fallback layers in {dit_name!r} must share one policy; "
                        f"got {len(policies)} distinct policies"
                    )
                policy = next(iter(policies))
                blocks = _discover_dit_blocks(pipeline, dit_name, dit)
                if blocks:
                    streamer = _BlockRingStreamer(
                        name=dit_name,
                        blocks=blocks,
                        policy=policy,
                        device=self.device,
                        shard_size=self.shard_size,
                        shard_rank=self.shard_rank,
                        comm_group=self.comm_group,
                        all_gather_fn=self._all_gather_fn,
                        use_pinned_staging=self._use_pinned_staging,
                        pinned_chunk_bytes=self._pinned_chunk_bytes,
                    )
                    if streamer.managed_layer_ids:
                        self.streamers.append(streamer)
                        managed_ids.update(streamer.managed_layer_ids)

            # Any stream-mode layers outside repeated blocks (e.g., final_layer or token_refiner)
            # are small; promote them to resident on device so they never trigger implicit CPU copies.
            for name, layer, _ in stream_layers:
                if id(layer) in managed_ids:
                    continue
                weight = getattr(layer, BF16_FALLBACK_WEIGHT_ATTR, None)
                if not isinstance(weight, torch.Tensor) or weight.numel() == 0:
                    raise RuntimeError(f"Non-block BF16 fallback layer {name!r} lacks a materialized weight")
                setattr(layer, BF16_FALLBACK_WEIGHT_ATTR, weight.to(device=self.device, dtype=torch.bfloat16))
                setattr(layer, BF16_FALLBACK_MANAGED_ATTR, "resident")
                self.resident_fallback_layers.append(layer)
                logger.info("Promoted non-block BF16 fallback layer %s to device-resident", name)

        return True

    def disable(self) -> None:
        for streamer in self.streamers:
            streamer.disable()
        self.streamers.clear()
        for layer in self.resident_fallback_layers:
            setattr(layer, BF16_FALLBACK_WEIGHT_ATTR, None)
            setattr(layer, BF16_FALLBACK_MANAGED_ATTR, None)
        self.resident_fallback_layers.clear()
        if self._owns_comm_group and self.comm_group is not None and dist.is_initialized():
            try:
                dist.destroy_process_group(self.comm_group)
            except Exception:
                logger.debug("Ignoring error destroying prefix BF16 communicator on shutdown", exc_info=True)
            self.comm_group = None
            self._owns_comm_group = False


def enable_prefix_bf16_fallback(
    pipeline: nn.Module,
    device: torch.device,
) -> PrefixBF16FallbackManager | None:
    """Install generic BF16 fallback streaming/residency management when active."""
    manager = PrefixBF16FallbackManager(device=device)
    if manager.enable(pipeline):
        return manager
    return None


__all__ = [
    "PrefixBF16FallbackManager",
    "enable_prefix_bf16_fallback",
]
