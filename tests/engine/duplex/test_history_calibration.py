# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.session.history_calibration import HistoryCalibration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_cache_keeps_working_after_many_evictions_and_never_restarts_a_live_suffix():
    draining: set[str] = set()
    session = SimpleNamespace(active_response_id=None, response_has_draining_request=draining.__contains__)

    async def calibrate(snapshot):
        pytest.fail("Recording audio must not start ASR")

    controller = HistoryCalibration(session, calibrate, max_bytes=16)
    output = {"history_audio_pcm": b"\0" * 8, "sample_rate_hz": 24000}
    try:
        session.active_response_id = "old"
        controller.record("old", output)
        draining.add("old")
        for i in range(40):
            response_id = f"response-{i}"
            session.active_response_id = response_id
            controller.record(response_id, output)
            assert controller.audio[response_id].size == 8
            assert controller.retained_bytes <= 16
        assert "old" in controller.evicted
        controller.record("old", output)
        assert "old" not in controller.audio
        draining.clear()
        controller.record(session.active_response_id, output)
        assert not controller.evicted
        # Deleting an item does not necessarily stop its audio generation.
        controller.discard(session.active_response_id)
        controller.record(session.active_response_id, output)
        assert session.active_response_id not in controller.audio
    finally:
        controller.close()
