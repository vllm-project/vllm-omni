# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Public contract for the parked-running rows used by admission."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
from vllm_omni.core.sched.omni_scheduler_mixin import OmniSchedulerMixin
from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator
from vllm_omni.core.sched.utils import parked_running_requests
from vllm_omni.distributed.omni_connectors.transfer_adapter.chunk_transfer_adapter import (
    OmniChunkTransferAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _adapter() -> OmniChunkTransferAdapter:
    adapter = OmniChunkTransferAdapter.__new__(OmniChunkTransferAdapter)
    adapter.waiting_for_chunk_running_requests = deque()
    adapter._held_non_active = deque()
    return adapter


def _coordinator() -> OmniSchedulingCoordinator:
    coordinator = OmniSchedulingCoordinator.__new__(OmniSchedulingCoordinator)
    coordinator._waiting_for_chunk_running = deque()
    return coordinator


def test_adapter_helper_returns_both_deques() -> None:
    adapter = _adapter()
    first, second, held = (SimpleNamespace(request_id=name) for name in ("first", "second", "held"))
    adapter.waiting_for_chunk_running_requests.extend([first, second])
    adapter._held_non_active.append(held)

    assert adapter.parked_running_requests() == (first, second, held)


def test_coordinator_helper_returns_held_rows() -> None:
    coordinator = _coordinator()
    held = SimpleNamespace(request_id="held")
    coordinator._waiting_for_chunk_running.append(held)

    assert coordinator.parked_running_requests() == (held,)


def test_shared_helper_prefers_the_public_method() -> None:
    parked = SimpleNamespace(request_id="parked")
    owner = SimpleNamespace(parked_running_requests=lambda: (parked,), _held_non_active=[object()])

    assert parked_running_requests(owner, "_held_non_active") == (parked,)


def test_shared_helper_falls_back_to_private_deques() -> None:
    parked = SimpleNamespace(request_id="parked")
    owner = SimpleNamespace(_held_non_active=[parked])

    assert parked_running_requests(owner, "_held_non_active") == (parked,)


def test_scheduler_counts_a_public_only_adapter() -> None:
    parked = SimpleNamespace(request_id="parked")
    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler.chunk_transfer_adapter = SimpleNamespace(parked_running_requests=lambda: (parked,))
    scheduler.input_coordinator = None

    assert scheduler._parked_running_requests() == (parked,)


def test_mixin_reserved_slots_use_the_public_method() -> None:
    parked = SimpleNamespace(request_id="parked")
    mixin = OmniSchedulerMixin.__new__(OmniSchedulerMixin)
    mixin.requests = {"parked": parked}
    mixin.chunk_transfer_adapter = SimpleNamespace(parked_running_requests=lambda: (parked,))
    mixin.input_coordinator = None

    assert mixin._get_async_chunk_reserved_running_slots() == 1
