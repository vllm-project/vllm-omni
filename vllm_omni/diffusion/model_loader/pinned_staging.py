# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded pinned-memory staging for diffusion checkpoint tensors.

The iterator copies eligible pageable CPU tensors into one fixed-size pinned
slab before yielding them to ``load_weights``.  The slab is reused only when
the consumer requests the next tensor, so opted-in consumers can overlap the
driver's pinned H2D path with no unbounded pool or size-class growth.
"""

from __future__ import annotations

from collections.abc import Generator, Iterator
from dataclasses import dataclass

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

DEFAULT_PINNED_STAGING_CAPACITY_BYTES = 256 << 20
DEFAULT_PINNED_STAGING_MIN_BYTES = 64 << 10


@dataclass
class PinnedStagingState:
    allocated: bool = False


def _alloc_pinned(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def release_pinned_staging_cache() -> None:
    """Release inactive pinned blocks when the installed Torch supports it."""
    empty_host_cache = getattr(torch.accelerator, "empty_host_cache", None)
    if empty_host_cache is None:
        empty_host_cache = getattr(torch._C, "_host_emptyCache", None)
    if empty_host_cache is None:
        logger.debug_once(
            "Torch does not expose host-cache cleanup; the bounded staging slab may remain in the host allocator cache."
        )
        return
    try:
        empty_host_cache()
    except Exception as exc:  # Cache cleanup must never replace the load result.
        logger.warning("Failed to release the pinned host allocator cache: %s", exc)


def _can_stage(tensor: torch.Tensor, *, capacity_bytes: int, min_bytes: int) -> bool:
    nbytes = tensor.numel() * tensor.element_size()
    return (
        type(tensor) is torch.Tensor
        and tensor.device.type == "cpu"
        and tensor.layout is torch.strided
        and tensor.is_contiguous()
        and not tensor.requires_grad
        and min_bytes <= nbytes <= capacity_bytes
    )


def pinned_staging_weights_iterator(
    weights: Iterator[tuple[str, torch.Tensor]],
    *,
    capacity_bytes: int = DEFAULT_PINNED_STAGING_CAPACITY_BYTES,
    min_bytes: int = DEFAULT_PINNED_STAGING_MIN_BYTES,
    state: PinnedStagingState | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield weights in order through one bounded pinned-memory slab.

    The caller must guarantee that every staged tensor is consumed
    synchronously before requesting the next item.  Unsupported and oversized
    tensors pass through by identity.  If the initial pinned allocation fails,
    the untouched input iterator is forwarded without staging.
    """
    if capacity_bytes <= 0:
        raise ValueError("capacity_bytes must be positive")
    if min_bytes < 0:
        raise ValueError("min_bytes must be non-negative")

    source = iter(weights)
    slab: torch.Tensor | None = None
    staging_available = True
    staged_bytes = 0
    staged_tensors = 0
    for name, tensor in source:
        if not staging_available or not _can_stage(
            tensor,
            capacity_bytes=capacity_bytes,
            min_bytes=min_bytes,
        ):
            yield name, tensor
            continue

        if slab is None:
            try:
                slab = _alloc_pinned(capacity_bytes)
                if state is not None:
                    state.allocated = True
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                staging_available = False
                logger.warning_once(
                    "Pinned weight staging unavailable; using pageable weights: %s",
                    exc,
                )
                yield name, tensor
                continue

        nbytes = tensor.numel() * tensor.element_size()
        staged = (
            slab[:nbytes]
            .view(tensor.dtype)
            .as_strided(
                tensor.shape,
                tensor.stride(),
            )
        )
        staged.copy_(tensor)
        staged_bytes += nbytes
        staged_tensors += 1
        yield name, staged

    if staged_tensors:
        logger.info(
            "Pinned weight staging copied %.2f GiB across %d tensors (%.2f GiB fixed capacity).",
            staged_bytes / (1 << 30),
            staged_tensors,
            capacity_bytes / (1 << 30),
        )
