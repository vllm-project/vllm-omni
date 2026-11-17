# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pristine dense weights for repeated, storage-preserving LoRA merges."""

import torch
from torch import nn


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


class MergedLinearWeights:
    """Retain base weights on their original device, outside the offload pool.

    Binding wrappers are owned by the manager, never installed in the live model.
    All callers must quiesce inference before preparing or applying an update.
    """

    def __init__(self) -> None:
        self._weights: dict[str, torch.Tensor] = {}
        self._base: dict[str, torch.Tensor] = {}

    @staticmethod
    def _rollback_snapshot(weight: torch.Tensor) -> torch.Tensor:
        """Keep rollback copies in host memory instead of duplicating HBM."""
        return weight.detach().to(device="cpu", copy=True)

    @torch.no_grad()
    def prepare(self, modules: dict[str, nn.Module]) -> dict[str, torch.Tensor]:
        """Compute a complete policy without changing live weights.

        Args:
            modules: Manager binding wrappers with rank-local fp32 LoRA slots.

        Returns:
            New dense values, including base values for omitted adapter targets.

        Raises:
            ValueError: A weight layout or computed value is unsupported.
        """
        prepared = dict(self._base)
        finite_checks: dict[torch.device, list[tuple[str, torch.Tensor]]] = {}
        for name, layer in modules.items():
            active = layer._diffusion_lora_active_slices
            if not any(active):
                continue
            weight = layer.base_layer.weight
            if weight.ndim != 2 or weight.dtype not in (torch.float32, torch.float16, torch.bfloat16):
                raise ValueError(f"Merged LoRA requires an unquantized matrix: {name}")
            if name not in self._base:
                self._base[name] = weight.detach().clone()
                self._weights[name] = weight
            elif weight is not self._weights[name]:
                raise ValueError(f"Base parameter changed for {name}; reset the merged bank before reloading")
            merged = self._base[name].float().clone()
            slices = getattr(layer, "output_slices", (weight.shape[0],))
            if len(slices) != len(active) or sum(slices) != weight.shape[0]:
                raise ValueError(f"Invalid packed LoRA output slices for {name}")
            offset = 0
            scales = getattr(layer, "_merged_lora_scales", (1.0,) * len(slices))
            for enabled, rows, a, b, scale in zip(active, slices, layer.lora_a_stacked, layer.lora_b_stacked, scales):
                if enabled:
                    delta = b[0, 0].float() @ a[0, 0].float()
                    if delta.shape != (rows, weight.shape[1]):
                        raise ValueError(f"LoRA shape does not match the local dense weight: {name}")
                    merged[offset : offset + rows].add_(delta * scale)
                offset += rows
            value = merged.to(dtype=weight.dtype)
            finite_checks.setdefault(value.device, []).append((name, torch.isfinite(value).all()))
            prepared[name] = value

        # One device-to-host synchronization per device instead of branching on
        # one CUDA scalar for every target projection.
        for checks in finite_checks.values():
            if not torch.stack([finite for _, finite in checks]).all().item():
                failed_name = next(name for name, finite in checks if not finite.item())
                raise ValueError(f"Merged LoRA produced non-finite weights for {failed_name}")
        return prepared

    @torch.no_grad()
    def apply(self, prepared: dict[str, torch.Tensor], group: torch.distributed.ProcessGroup | None = None) -> None:
        """Install prepared values in place, rolling back a failed copy.

        Rollback snapshots stay on the host to cap transient HBM use. This
        intentionally adds one device-to-host copy per committed update; the
        model-level E2E tests report its update latency alongside peak memory.

        Args:
            prepared: Values returned by ``prepare`` on this bank.
            group: Optional CPU group coordinating allocation and copy failures.

        Raises:
            RuntimeError: Allocation or copying failed; prior weights stay live.
        """
        previous: dict[str, torch.Tensor] = {}
        error = None
        try:
            previous = {name: self._rollback_snapshot(self._weights[name]) for name in prepared}
        except Exception as exc:
            error = exc
        check_merge_error(error, group)
        try:
            for name, value in prepared.items():
                self._weights[name].copy_(value)
            if any(value.device.type != "cpu" for value in prepared.values()):
                torch.accelerator.synchronize()
        except Exception as exc:
            error = exc
        try:
            check_merge_error(error, group)
        except Exception:
            for name, value in previous.items():
                self._weights[name].copy_(value)
            raise

    @torch.no_grad()
    def close(self) -> None:
        """Restore exact base values in place and release the bank."""
        for name, value in self._base.items():
            self._weights[name].copy_(value)
        self._base.clear()
        self._weights.clear()
