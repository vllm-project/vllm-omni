# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from enum import Enum


class DiffusionKVCacheMode(str, Enum):
    """Migration mode for diffusion KV ownership."""

    DENSE_LEGACY = "dense_legacy"
    # Reserved for the experimental Worker-owned paging design. Production
    # configuration rejects it until that engine is migrated to the common
    # Scheduler-owned path.
    PAGED_WORKER_LOCAL = "paged_worker_local"
    PAGED_SCHEDULER = "paged_scheduler"


def parse_diffusion_kv_cache_mode(value: object) -> DiffusionKVCacheMode:
    """Parse a selectable Diffusion KV ownership mode.

    ``PAGED_WORKER_LOCAL`` remains in the enum to document the migration
    topology, but it is not a valid production selection until that runtime is
    wired. Keep this validation shared by structured and runtime configs so an
    unsupported mode fails before managed-process startup.
    """

    try:
        mode = DiffusionKVCacheMode(value)
    except (TypeError, ValueError) as exc:
        supported = ", ".join(mode.value for mode in DiffusionKVCacheMode)
        raise ValueError(f"Unsupported Diffusion KV diffusion_kv_mode {value!r}; expected one of: {supported}") from exc
    if mode is DiffusionKVCacheMode.PAGED_WORKER_LOCAL:
        raise ValueError(
            "Diffusion KV diffusion_kv_mode 'paged_worker_local' is reserved but not implemented; "
            "use 'dense_legacy' or 'paged_scheduler'"
        )
    return mode


def is_scheduler_paged_kv_mode(mode: DiffusionKVCacheMode) -> bool:
    """Return whether a parsed cache mode uses Scheduler-owned paging."""
    return mode is DiffusionKVCacheMode.PAGED_SCHEDULER
