# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Scheduler test helpers shared by AR / chunk-transfer unit tests."""

from __future__ import annotations

from types import MethodType

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler


def bind_omits_transfer_helpers(sched) -> None:
    """Bind the real omit helpers so a MagicMock scheduler does not skip save_async.

    MagicMock is truthy, so an unbound ``_request_omits_chunk_transfer_to_next_stage``
    silently skips the chunk put.
    """
    sched._omits_kv_transfer_cache = {}
    sched._omni_final_stage_flags = MethodType(OmniARScheduler._omni_final_stage_flags, sched)
    sched._request_omits_kv_transfer_to_next_stage = MethodType(
        OmniARScheduler._request_omits_kv_transfer_to_next_stage,
        sched,
    )
    sched._request_omits_chunk_transfer_to_next_stage = MethodType(
        OmniARScheduler._request_omits_chunk_transfer_to_next_stage,
        sched,
    )
