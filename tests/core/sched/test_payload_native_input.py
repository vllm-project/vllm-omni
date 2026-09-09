# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

from vllm_omni.core.sched.omni_scheduling_coordinator import OmniSchedulingCoordinator

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_native_generation_schedules_control_slot_without_codec_token_list():
    coordinator = OmniSchedulingCoordinator(stage_id=1, async_chunk=True)
    coordinator.payload_native = True
    request = SimpleNamespace(
        prompt_token_ids=[1, 2, 3], _all_token_ids=[1, 2, 3], _output_token_ids=[9], num_computed_tokens=3
    )
    coordinator.update_request_metadata(
        {"r": request}, {"r": {"payload_ready": True, "left_context_size": 72}}, model_mode="generation"
    )
    assert request.prompt_token_ids == [0]
    assert request._all_token_ids == [0]
    assert request.num_computed_tokens == 0
    assert request._omni_initial_model_buffer == {"meta": {"left_context_size": 72}}
