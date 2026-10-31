# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Isolation selects the existing once-only Host/INT8/asynchronous-ring path."""

from __future__ import annotations

import torch
from vllm.logger import init_logger

from .reference_kv_cache import MiniMaxH3ReferenceKVTier2State

_log = init_logger(__name__)


def create_isolated_reference_cache(
    *, num_layers: int, global_reference_rows: int, device: torch.device
) -> MiniMaxH3ReferenceKVTier2State | None:
    """Reuse Tier2 unchanged; isolate mode owns its fixed request-local policy."""
    if global_reference_rows <= 0:
        return None
    state = MiniMaxH3ReferenceKVTier2State(
        num_layers=num_layers,
        global_reference_rows=global_reference_rows,
        device=device,
        refresh_interval=0,
        ring_size=2,
        host_quantization="int8",
    )
    _log.info(
        "MiniMax-H3 isolation cache enabled: mode=isolate, residency=pinned_host, "
        "once_per_request=True, host_quantization=int8, ring_size=2, "
        "async_prefetch=True, periodic_refresh=disabled, hit_priority=disabled; "
        "--kv-isolate takes precedence over --kv-offload-config",
    )
    return state
