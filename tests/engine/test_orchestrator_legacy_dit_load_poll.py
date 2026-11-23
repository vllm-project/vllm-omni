# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Legacy output loop polls DiT load independently of diffusion output.

Review ask (PR #9634): with ``VLLM_OMNI_EVENT_DRIVEN_ORCH=0``, the throttled
DTPS load update must run before the output-only early exits, so a replica
that supplies a load response is still collected when
``poll_diffusion_output()`` returns ``None`` (no ready output, metrics-only
ticks) and the AR scheduler never keeps scheduling on stale load during a
long DiT request.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from vllm_omni.engine.messages import ShutdownRequestMessage

from .test_orchestrator import FakeStageClient, _build_harness

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_DIT_LOAD = (3, 2, ["w-1", "w-2", "w-3"], ["r-1", "r-2"])


class _RecordingArClient(FakeStageClient):
    """AR replica that records utility pushes (``push_dit_load`` target)."""

    def __init__(self) -> None:
        super().__init__(final_output=True, final_output_type="text")
        self.utility_calls: list[tuple[str, dict[str, Any]]] = []

    async def call_utility_async(self, method: str, payload: dict[str, Any]) -> bool:
        self.utility_calls.append((method, dict(payload)))
        return True


class _LoadOnlyDiTClient(FakeStageClient):
    """DiT replica that answers load queries but never produces output."""

    def __init__(self) -> None:
        super().__init__(stage_type="diffusion", final_output=True, final_output_type="image")
        self.diffusion_polls = 0

    async def get_dit_load_async(self) -> tuple[int, int, list[str], list[str]]:
        return _DIT_LOAD

    def get_diffusion_output_nowait(self) -> Any:
        self.diffusion_polls += 1
        return super().get_diffusion_output_nowait()


def test_legacy_loop_polls_dit_load_when_no_diffusion_output(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_OMNI_EVENT_DRIVEN_ORCH", "0")

    ar_client = _RecordingArClient()
    dit_client = _LoadOnlyDiTClient()

    fixture = _build_harness(
        [ar_client, dit_client],
        dit_load_aware_config=[{"ar_stage_id": 0, "dit_stage_id": 1, "threshold": 2}],
    )
    orch = fixture.orchestrator
    assert orch._event_driven_orch is False
    assert orch._dit_load_aware is True

    try:
        deadline = time.monotonic() + 5.0
        while not ar_client.utility_calls:
            assert time.monotonic() < deadline, "timed out waiting for the DiT load push"
            time.sleep(0.01)
    finally:
        fixture.request_sync_q.put_nowait(ShutdownRequestMessage())
        fixture.thread.join(5)

    assert not fixture.thread.is_alive()
    fixture.result_future.result(timeout=0)

    # The replica never produced a diffusion output (early-exit path) ...
    assert dit_client.diffusion_polls > 0
    # ... yet its load response was polled and pushed to the AR stage.
    method, payload = ar_client.utility_calls[0]
    assert method == "omni_update_dit_load"
    assert payload["fresh"] is True
    assert payload["total_waiting"] == 3
    assert payload["num_replicas"] == 1
    snapshot = orch._dit_load_state.snapshot(stage_id=1)
    assert snapshot["total_waiting"] == 3
