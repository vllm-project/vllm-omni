# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Budgeted static AdaLN residency for ordinary, rank-local layerwise offload."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn
from vllm.logger import init_logger

from vllm_omni.platforms import current_omni_platform

from .tensor_utils import is_dtensor, physical_storage_numel, set_tensor_storage

logger = init_logger(__name__)


@dataclass(frozen=True)
class AdaLNCacheStatus:
    requested: bool = False
    enabled: bool = False
    reason: str = "disabled"
    tensor_count: int = 0
    bytes: int = 0


def select_adaln_tensors(modules: Iterable[nn.Module]) -> list[tuple[str, torch.Tensor]]:
    """Include weights, biases and quantization scales in AdaLN-named modules.

    The attention's BF16/FP16/mix/MXFP8 precision is intentionally not an input:
    cached model tensors retain the dtype and layout produced by the loader.
    """
    selected = []
    seen = set()
    for index, module in enumerate(modules):
        buffers = [
            (name, tensor)
            for name, tensor in module.named_buffers()
            if name.split(".")[-1] in {"weight", "bias", "scale", "offset", "zero_point", "zero_points"}
            or name.endswith(("_scale", "_scales", "_offset", "_offsets", "_zero_point", "_zero_points"))
        ]
        for name, tensor in list(module.named_parameters()) + buffers:
            if id(tensor) in seen:
                continue
            if any("adaln" in part.replace("_", "").casefold() for part in name.split(".")):
                seen.add(id(tensor))
                selected.append((f"{index}.{name}", tensor))
    return selected


class AdaLNWeightCache:
    """Own startup cache copies and restore original storage on disable/failure."""

    def __init__(self, device: torch.device):
        self.device = device
        self.status = AdaLNCacheStatus()
        self._originals: list[tuple[torch.Tensor, torch.Tensor]] = []

    @property
    def resident_ids(self) -> frozenset[int]:
        return frozenset(id(target) for target, _ in self._originals)

    def _fallback(self, reason: str, count: int, size: int) -> frozenset[int]:
        self.status = AdaLNCacheStatus(True, False, reason, count, size)
        logger.warning(
            "AdaLN cache disabled: %s; tensors=%d bytes=%d; using ordinary LO",
            reason,
            count,
            size,
            extra={"cache_scope": "adaln", "tensor_count": count, "cache_bytes": size},
        )
        return frozenset()

    def prepare(
        self,
        modules: Iterable[nn.Module],
        *,
        max_gib: float,
        reserve_gib: float,
        startup_bytes: int = 0,
        resident_ids: frozenset[int] = frozenset(),
    ) -> frozenset[int]:
        if self._originals:
            return self.resident_ids
        if self.device.type == "npu":
            from vllm_omni.platforms.npu import is_a5

            if is_a5(self.device):
                return self._fallback("unsupported_on_a5", 0, 0)
        tensors = [(name, tensor) for name, tensor in select_adaln_tensors(modules) if id(tensor) not in resident_ids]
        if not tensors:
            return self._fallback("no_adaln_tensors", 0, 0)
        if any(is_dtensor(t) or t.is_meta or t.is_quantized or t.numel() == 0 for _, t in tensors):
            return self._fallback("unsupported_or_unmaterialized_tensor", len(tensors), 0)
        size = sum(physical_storage_numel(t) * t.element_size() for _, t in tensors)
        if size > int(max_gib * 2**30):
            return self._fallback("cache_budget_exceeded", len(tensors), size)
        reserve = int(reserve_gib * 2**30) + startup_bytes
        try:
            free = int(current_omni_platform.get_free_memory(self.device))
        except (NotImplementedError, RuntimeError, ValueError, TypeError, OverflowError):
            return self._fallback("memory_query_unavailable", len(tensors), size)
        if free < size + reserve:
            return self._fallback("insufficient_free_memory", len(tensors), size)

        # Allocate/copy everything before changing a model tensor. Quantized
        # plain tensors (INT8, FP8 and E8M0 scales) are copied without conversion.
        prepared: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        fresh = None
        try:
            for _, target in tensors:
                original = target.detach()
                # Prefer the platform's device-copy allocation for dense weights.
                # Preserve explicit strides when that copy normalizes a layout.
                fresh = original.to(device=self.device, copy=True) if original.is_contiguous() else None
                if fresh is None or fresh.stride() != target.stride():
                    fresh = None
                    fresh = torch.empty_strided(target.shape, target.stride(), dtype=target.dtype, device=self.device)
                    fresh.copy_(original)
                prepared.append((target, original, fresh))
            current_omni_platform.synchronize()
            if int(current_omni_platform.get_free_memory(self.device)) < reserve:
                prepared.clear()
                fresh = None
                current_omni_platform.empty_cache()
                return self._fallback("post_allocation_reserve_exceeded", len(tensors), size)
            for target, original, fresh in prepared:
                self._originals.append((target, original))
                set_tensor_storage(target, fresh)
        except (torch.OutOfMemoryError, RuntimeError, NotImplementedError, ValueError, TypeError, OverflowError):
            self.restore()
            prepared.clear()
            fresh = None
            current_omni_platform.synchronize()
            current_omni_platform.empty_cache()
            return self._fallback("cache_allocation_or_copy_failed", len(tensors), size)

        self.status = AdaLNCacheStatus(True, True, "enabled", len(tensors), size)
        logger.info(
            "AdaLN cache enabled: tensors=%d bytes=%d reserve_bytes=%d",
            len(tensors),
            size,
            reserve,
            extra={"cache_scope": "adaln", "tensor_count": len(tensors), "cache_bytes": size, "reserve_bytes": reserve},
        )
        return self.resident_ids

    def restore(self) -> None:
        """Restore original storage; retain ownership if restoration fails."""
        if self._originals:
            current_omni_platform.synchronize()
        while self._originals:
            target, original = self._originals[-1]
            set_tensor_storage(target, original)
            self._originals.pop()
        self.status = AdaLNCacheStatus(self.status.requested, False, "released")
