# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Read-only detection of the vLLM 0.28 retained-request append contract.

Omni's utilities are declared on its own Core and client subclasses. Importing
this module never adds methods or enum aliases to upstream vLLM classes.
"""

from __future__ import annotations


def scheduler_native_append_unavailable_reason() -> str | None:
    """Report a missing 0.28 contract without installing compatibility shims."""
    try:
        from vllm.v1.core.sched.output import CachedRequestData
        from vllm.v1.core.sched.scheduler import Scheduler
        from vllm.v1.engine.core_client import AsyncMPClient
        from vllm.v1.request import RequestStatus, StreamingUpdate
    except (ImportError, AttributeError) as exc:
        return f"vLLM retained-request append imports are unavailable: {exc}"

    required = {
        "AsyncMPClient.add_request_async": callable(getattr(AsyncMPClient, "add_request_async", None)),
        "AsyncMPClient.call_utility_async": callable(getattr(AsyncMPClient, "call_utility_async", None)),
        "Scheduler._update_request_as_session": callable(getattr(Scheduler, "_update_request_as_session", None)),
        "RequestStatus.WAITING_FOR_STREAMING_REQ": hasattr(RequestStatus, "WAITING_FOR_STREAMING_REQ"),
        "CachedRequestData.new_token_ids": "new_token_ids" in getattr(CachedRequestData, "__dataclass_fields__", {}),
        "StreamingUpdate prompt fields": {"prompt_token_ids", "max_tokens", "arrival_time", "sampling_params"}.issubset(
            getattr(StreamingUpdate, "__dataclass_fields__", {})
        ),
    }
    missing = [name for name, available in required.items() if not available]
    if missing:
        return "vLLM 0.28 retained-request append contract is unavailable: " + ", ".join(missing)
    return None


def scheduler_native_append_available() -> bool:
    return scheduler_native_append_unavailable_reason() is None
