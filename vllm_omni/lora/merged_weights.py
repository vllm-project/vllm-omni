# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pristine dense weights for repeated, storage-preserving LoRA merges."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from vllm.logger import init_logger

logger = init_logger(__name__)

# Provisional parking threshold (RFC #10444 §2.1 H2); frozen only after TP4
# colocated measurements exercise both parking branches.
PARKING_HBM_MULTIPLIER = 2
PARKING_HBM_MARGIN_BYTES = 4 * 1024**3
PINNED_CHUNK_MAX_BYTES = 64 * 1024**2
_ARENA_ALIGNMENT_BYTES = 256


def check_merge_error(error: Exception | None, group: torch.distributed.ProcessGroup | None) -> None:
    """Raise a local or peer failure after a CPU-group readiness vote.

    Args:
        error: This rank's preparation or commit failure, if any.
        group: Participating CPU process group, or None for a local operation.

    Raises:
        Exception: The local error, or RuntimeError when another rank failed.
    """
    failed = torch.tensor(int(error is not None), dtype=torch.int32, device="cpu")
    if group is not None:
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX, group=group)
    if error is not None:
        raise error
    if failed.item():
        raise RuntimeError("Merged LoRA update failed on another worker; previous policy retained")


def _align_up(size: int, alignment: int = _ARENA_ALIGNMENT_BYTES) -> int:
    return ((size + alignment - 1) // alignment) * alignment


def _default_alloc_pinned_bytes(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)


def _default_free_device_memory(device: torch.device) -> int | None:
    if device.type == "cpu":
        return None
    from vllm_omni.platforms import current_omni_platform

    try:
        return int(current_omni_platform.get_free_memory(device))
    except NotImplementedError:
        return None


def _default_use_pinned_host(device: torch.device) -> bool:
    return device.type != "cpu"


def _slice_arena(
    arena: torch.Tensor,
    specs: list[tuple[str, torch.Size, torch.dtype]],
) -> dict[str, torch.Tensor]:
    buffers: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape, dtype in specs:
        nbytes = shape.numel() * dtype.itemsize
        buffers[name] = arena[offset : offset + nbytes].view(dtype).view(shape)
        offset += _align_up(nbytes)
    return buffers


def _allocate_host_buffers(
    specs: list[tuple[str, torch.Size, torch.dtype]],
    *,
    pinned: bool,
    chunk_max_bytes: int,
    alloc_pinned_bytes: Callable[[int], torch.Tensor],
) -> tuple[dict[str, torch.Tensor], str]:
    """Allocate host buffers with arena -> chunked-pinned -> pageable fallback.

    Returns ``(buffers, mode)`` where ``mode`` is one of ``"arena_pinned"``,
    ``"chunked_pinned"``, or ``"pageable"``. Allocation failures on pinned
    memory never abort a sync.
    """
    if not specs:
        return {}, "pageable"
    if not pinned:
        return (
            {name: torch.empty(shape, dtype=dtype, device="cpu") for name, shape, dtype in specs},
            "pageable",
        )

    aligned_sizes = [_align_up(shape.numel() * dtype.itemsize) for _, shape, dtype in specs]
    total_bytes = sum(aligned_sizes)
    try:
        arena = alloc_pinned_bytes(total_bytes)
        return _slice_arena(arena, specs), "arena_pinned"
    except (RuntimeError, MemoryError) as exc:
        logger.warning(
            "Pinned host arena allocation of %d bytes failed (%s); falling back to chunked pinned buffers.",
            total_bytes,
            exc,
        )

    # Chunked pinned fallback.
    chunks: list[tuple[list[tuple[str, torch.Size, torch.dtype]], int]] = []
    current_specs: list[tuple[str, torch.Size, torch.dtype]] = []
    current_bytes = 0
    for spec, aligned in zip(specs, aligned_sizes):
        if current_specs and current_bytes + aligned > chunk_max_bytes:
            chunks.append((current_specs, current_bytes))
            current_specs = []
            current_bytes = 0
        current_specs.append(spec)
        current_bytes += aligned
    if current_specs:
        chunks.append((current_specs, current_bytes))

    buffers: dict[str, torch.Tensor] = {}
    fell_back_pageable = False
    for chunk_specs, chunk_bytes in chunks:
        if not fell_back_pageable:
            try:
                chunk_arena = alloc_pinned_bytes(chunk_bytes)
                buffers.update(_slice_arena(chunk_arena, chunk_specs))
                continue
            except (RuntimeError, MemoryError) as exc:
                logger.warning(
                    "Chunked pinned host allocation of %d bytes failed (%s); falling back to pageable host memory.",
                    chunk_bytes,
                    exc,
                )
                fell_back_pageable = True
        for name, shape, dtype in chunk_specs:
            buffers[name] = torch.empty(shape, dtype=dtype, device="cpu")
    return buffers, ("pageable" if fell_back_pageable else "chunked_pinned")


class MergedLinearWeights:
    """Host-resident pristine base weights for repeated in-place LoRA merges.

    Pristine copies live in host memory for the bank's lifetime so level-1
    sleep never strands them in HBM. Prepared values park on device only when
    free HBM meets the ``2 * M + 4 GiB`` headroom budget; otherwise they park
    on host and peak extra device memory stays bounded by one fp32 merge
    scratch, one fp32 delta scratch, and one cast scratch, reused across syncs.
    """

    def __init__(
        self,
        *,
        parking_hbm_multiplier: int = PARKING_HBM_MULTIPLIER,
        parking_hbm_margin_bytes: int = PARKING_HBM_MARGIN_BYTES,
        pinned_chunk_max_bytes: int = PINNED_CHUNK_MAX_BYTES,
        free_device_memory_fn: Callable[[torch.device], int | None] | None = None,
        alloc_pinned_bytes_fn: Callable[[int], torch.Tensor] | None = None,
        use_pinned_host_fn: Callable[[torch.device], bool] | None = None,
    ) -> None:
        self._weights: dict[str, torch.Tensor] = {}
        self._base: dict[str, torch.Tensor] = {}
        self._fp32_scratch: dict[torch.device, torch.Tensor] = {}
        self._delta_scratch: dict[torch.device, torch.Tensor] = {}
        self._cast_scratch: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._parking_hbm_multiplier = parking_hbm_multiplier
        self._parking_hbm_margin_bytes = parking_hbm_margin_bytes
        self._pinned_chunk_max_bytes = pinned_chunk_max_bytes
        self._free_device_memory_fn = free_device_memory_fn or _default_free_device_memory
        self._alloc_pinned_bytes_fn = alloc_pinned_bytes_fn or _default_alloc_pinned_bytes
        self._use_pinned_host_fn = use_pinned_host_fn or _default_use_pinned_host
        self.last_parking_modes: dict[torch.device, str] = {}
        self.last_host_alloc_mode: str | None = None

    @staticmethod
    def _rollback_snapshot(weight: torch.Tensor) -> torch.Tensor:
        """Keep rollback copies in host memory instead of duplicating HBM."""
        return weight.detach().to(device="cpu", copy=True)

    def _get_fp32_scratch(self, device: torch.device, numel: int) -> torch.Tensor:
        scratch = self._fp32_scratch.get(device)
        if scratch is None or scratch.numel() < numel:
            scratch = torch.empty(numel, dtype=torch.float32, device=device)
            self._fp32_scratch[device] = scratch
        return scratch[:numel]

    def _get_delta_scratch(self, device: torch.device, numel: int) -> torch.Tensor:
        scratch = self._delta_scratch.get(device)
        if scratch is None or scratch.numel() < numel:
            scratch = torch.empty(numel, dtype=torch.float32, device=device)
            self._delta_scratch[device] = scratch
        return scratch[:numel]

    def _get_cast_scratch(self, device: torch.device, dtype: torch.dtype, numel: int) -> torch.Tensor:
        key = (device, dtype)
        scratch = self._cast_scratch.get(key)
        if scratch is None or scratch.numel() < numel:
            scratch = torch.empty(numel, dtype=dtype, device=device)
            self._cast_scratch[key] = scratch
        return scratch[:numel]

    def _should_park_on_device(self, device: torch.device, active_bytes: int) -> bool:
        free_bytes = self._free_device_memory_fn(device)
        if free_bytes is None:
            return True
        required = self._parking_hbm_multiplier * active_bytes + self._parking_hbm_margin_bytes
        return free_bytes >= required

    @torch.no_grad()
    def prepare(self, modules: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
        """Compute a complete policy without changing live weights.

        Args:
            modules: Manager binding wrappers with rank-local fp32 LoRA slots.

        Returns:
            New dense values, including host pristine base values for omitted
            adapter targets.

        Raises:
            ValueError: A weight layout or computed value is unsupported.
        """
        active_targets: list[
            tuple[str, nn.Module, torch.Tensor, tuple[bool, ...], tuple[int, ...], tuple[float, ...]]
        ] = []
        active_bytes_by_device: dict[torch.device, int] = {}
        new_base_specs_by_pinned: dict[bool, list[tuple[str, torch.Size, torch.dtype]]] = {False: [], True: []}

        for name, layer in modules.items():
            active = tuple(layer._diffusion_lora_active_slices)
            if not any(active):
                continue
            weight = layer.base_layer.weight
            if weight.ndim != 2 or weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(f"Merged LoRA requires an unquantized matrix: {name}")
            if name in self._base and weight is not self._weights[name]:
                raise ValueError(f"Base parameter changed for {name}; reset the merged bank before reloading")
            slices = tuple(getattr(layer, "output_slices", (weight.shape[0],)))
            if len(slices) != len(active) or sum(slices) != weight.shape[0]:
                raise ValueError(f"Invalid packed LoRA output slices for {name}")
            scales = tuple(getattr(layer, "_merged_lora_scales", (1.0,) * len(slices)))
            active_targets.append((name, layer, weight, active, slices, scales))
            active_bytes_by_device[weight.device] = active_bytes_by_device.get(weight.device, 0) + weight.nbytes
            if name not in self._base:
                new_base_specs_by_pinned[self._use_pinned_host_fn(weight.device)].append(
                    (name, weight.shape, weight.dtype)
                )

        # Capture pristine base weights on host once per bank lifecycle.
        newly_added: list[str] = []
        for use_pinned, specs in new_base_specs_by_pinned.items():
            if not specs:
                continue
            host_bufs, mode = _allocate_host_buffers(
                specs,
                pinned=use_pinned,
                chunk_max_bytes=self._pinned_chunk_max_bytes,
                alloc_pinned_bytes=self._alloc_pinned_bytes_fn,
            )
            self.last_host_alloc_mode = mode
            for name, _, _ in specs:
                self._base[name] = host_bufs[name]
                newly_added.append(name)

        newly_added_set = set(newly_added)
        try:
            for name, _, weight, _, _, _ in active_targets:
                if name in newly_added_set:
                    self._weights[name] = weight
                    base_buf = self._base[name]
                    base_buf.copy_(weight.detach(), non_blocking=base_buf.is_pinned())

            # Decide device vs host parking per device from free HBM headroom.
            park_on_device: dict[torch.device, bool] = {}
            for device, active_bytes in active_bytes_by_device.items():
                park_on_device[device] = self._should_park_on_device(device, active_bytes)
            self.last_parking_modes = {
                device: ("device" if on_device else "host") for device, on_device in park_on_device.items()
            }

            host_park_specs_by_pinned: dict[bool, list[tuple[str, torch.Size, torch.dtype]]] = {False: [], True: []}
            for name, _, weight, _, _, _ in active_targets:
                if not park_on_device[weight.device]:
                    host_park_specs_by_pinned[self._use_pinned_host_fn(weight.device)].append(
                        (name, weight.shape, weight.dtype)
                    )

            host_parked_buffers: dict[str, torch.Tensor] = {}
            for use_pinned, specs in host_park_specs_by_pinned.items():
                if not specs:
                    continue
                bufs, mode = _allocate_host_buffers(
                    specs,
                    pinned=use_pinned,
                    chunk_max_bytes=self._pinned_chunk_max_bytes,
                    alloc_pinned_bytes=self._alloc_pinned_bytes_fn,
                )
                self.last_host_alloc_mode = mode
                host_parked_buffers.update(bufs)

            prepared: dict[str, torch.Tensor] = dict(self._base)
            finite_checks: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}

            for name, layer, weight, active, slices, scales in active_targets:
                merged = self._get_fp32_scratch(weight.device, weight.numel()).view(weight.shape)
                if name in newly_added_set:
                    # Live weight is still pristine on device; avoid an immediate H2D round-trip.
                    merged.copy_(weight.detach())
                else:
                    base_host = self._base[name]
                    merged.copy_(base_host, non_blocking=base_host.is_pinned())

                offset = 0
                for enabled, rows, a, b, scale in zip(
                    active, slices, layer.lora_a_stacked, layer.lora_b_stacked, scales
                ):
                    if enabled:
                        a_fp32 = a[0, 0].to(device=weight.device, dtype=torch.float32)
                        b_fp32 = b[0, 0].to(device=weight.device, dtype=torch.float32)
                        if (
                            a_fp32.ndim != 2
                            or b_fp32.ndim != 2
                            or b_fp32.shape[1] != a_fp32.shape[0]
                            or b_fp32.shape[0] != rows
                            or a_fp32.shape[1] != weight.shape[1]
                        ):
                            raise ValueError(f"LoRA shape does not match the local dense weight: {name}")
                        # Write B @ A into the reused fp32 delta scratch and
                        # accumulate with alpha=scale (no dense delta*scale
                        # temporary, and bitwise identical to (B @ A) * scale).
                        delta = self._get_delta_scratch(weight.device, rows * weight.shape[1]).view(
                            rows, weight.shape[1]
                        )
                        torch.mm(b_fp32, a_fp32, out=delta)
                        merged[offset : offset + rows].add_(delta, alpha=float(scale))
                    offset += rows

                if park_on_device[weight.device]:
                    value = merged.to(dtype=weight.dtype, copy=True)
                    finite_checks.setdefault(weight.device, []).append((name, torch.isfinite(value).all()))
                    prepared[name] = value
                else:
                    cast_buf = self._get_cast_scratch(weight.device, weight.dtype, weight.numel()).view(weight.shape)
                    cast_buf.copy_(merged)
                    finite_checks.setdefault(weight.device, []).append((name, torch.isfinite(cast_buf).all()))
                    host_value = host_parked_buffers[name]
                    host_value.copy_(cast_buf, non_blocking=host_value.is_pinned())
                    prepared[name] = host_value

            # One device-to-host synchronization per device anchors both the
            # finite checks and any non-blocking D2H copies enqueued above.
            for checks in finite_checks.values():
                if not torch.stack([finite for _, finite in checks]).all().item():
                    failed_name = next(name for name, finite in checks if not finite.item())
                    raise ValueError(f"Merged LoRA produced non-finite weights for {failed_name}")
            return prepared
        except Exception:
            for name in newly_added:
                self._base.pop(name, None)
                self._weights.pop(name, None)
            raise

    @torch.no_grad()
    def apply(self, prepared: dict[str, torch.Tensor], group: torch.distributed.ProcessGroup | None = None) -> None:
        """Install prepared values in place, rolling back a failed copy.

        Rollback snapshots stay on the host over pinned buffers (with
        chunked-pinned and pageable fallbacks) so transient HBM use stays flat.
        Copies run ``non_blocking`` and are anchored by the synchronization
        before the commit vote.

        Args:
            prepared: Values returned by ``prepare`` on this bank.
            group: Optional CPU group coordinating allocation and copy failures.

        Raises:
            RuntimeError: Allocation or copying failed; prior weights stay live.
        """
        previous: dict[str, torch.Tensor] = {}
        error = None
        has_accelerator_weight = any(self._weights[name].device.type != "cpu" for name in prepared)
        try:
            rollback_specs_by_pinned: dict[bool, list[tuple[str, torch.Size, torch.dtype]]] = {False: [], True: []}
            for name in prepared:
                weight = self._weights[name]
                rollback_specs_by_pinned[self._use_pinned_host_fn(weight.device)].append(
                    (name, weight.shape, weight.dtype)
                )
            for use_pinned, specs in rollback_specs_by_pinned.items():
                if not specs:
                    continue
                bufs, _ = _allocate_host_buffers(
                    specs,
                    pinned=use_pinned,
                    chunk_max_bytes=self._pinned_chunk_max_bytes,
                    alloc_pinned_bytes=self._alloc_pinned_bytes_fn,
                )
                for name, _, _ in specs:
                    buf = bufs[name]
                    buf.copy_(self._weights[name].detach(), non_blocking=buf.is_pinned())
                    previous[name] = buf
        except Exception as exc:
            error = exc
        check_merge_error(error, group)
        try:
            for name, value in prepared.items():
                self._weights[name].copy_(value, non_blocking=value.is_pinned())
            if has_accelerator_weight:
                torch.accelerator.synchronize()
        except Exception as exc:
            error = exc
        try:
            check_merge_error(error, group)
        except Exception:
            for name, value in previous.items():
                self._weights[name].copy_(value, non_blocking=value.is_pinned())
            if has_accelerator_weight:
                torch.accelerator.synchronize()
            raise

    @torch.no_grad()
    def close(self) -> None:
        """Restore exact base values in place and release the bank and scratch."""
        has_accelerator_weight = any(weight.device.type != "cpu" for weight in self._weights.values())
        for name, value in self._base.items():
            self._weights[name].copy_(value, non_blocking=value.is_pinned())
        if has_accelerator_weight:
            torch.accelerator.synchronize()
        self._base.clear()
        self._weights.clear()
        self._fp32_scratch.clear()
        self._delta_scratch.clear()
        self._cast_scratch.clear()
        self.last_parking_modes.clear()
