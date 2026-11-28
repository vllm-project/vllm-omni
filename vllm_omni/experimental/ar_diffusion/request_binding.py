# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bind existing service Session identity to one KV compute invocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm_omni.experimental.ar_diffusion.tick_protocol import ARDiffusionTickRequest

if TYPE_CHECKING:
    from vllm_omni.diffusion.request import OmniDiffusionRequest


@dataclass(frozen=True)
class KVRequestBinding:
    owner: str
    persist: bool
    reset: bool
    close: bool
    chunk_base: int


def kv_request_binding(request: OmniDiffusionRequest) -> KVRequestBinding:
    """Resolve one identity for pool access and request cleanup, tick first."""
    extra = request.sampling_params.extra_args or {}
    tick = ARDiffusionTickRequest.from_extra_args(extra)
    if tick is not None:
        return KVRequestBinding(
            tick.session_id,
            not tick.close_session,
            tick.reset,
            tick.close_session,
            tick.chunk_index,
        )
    raw_session = extra.get("session_id")
    explicit = raw_session is not None and str(raw_session) != ""
    close = bool(extra.get("close_session", False))
    owner = str(raw_session) if explicit else str(getattr(request, "request_id", None) or "default")
    chunk_base = int(extra.get("chunk_base", 0) or 0)
    if chunk_base < 0:
        raise ValueError("chunk_base must be non-negative")
    return KVRequestBinding(owner, explicit and not close, bool(extra.get("reset", False)), close, chunk_base)
