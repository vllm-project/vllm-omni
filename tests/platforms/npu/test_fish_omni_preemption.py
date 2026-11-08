# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise Drama 3 preemption with native NPU history buffers."""

import pytest
import torch

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from tests.model_executor.models.fish_omni.test_fish_omni_model_state import (
    make_state as make_state,
)
from tests.model_executor.models.fish_omni.test_fish_omni_model_state import (
    test_preempted_request_resumes_with_its_generated_frames as check_resume,
)

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A2"}, num_cards=1)]


def test_npu_history_survives_preemption_and_slot_reuse(make_state):  # noqa: F811 - imported pytest fixture
    if not torch.npu.is_available():
        pytest.skip("NPU is unavailable")

    def make_npu_state():
        state = make_state()
        state.device = torch.device("npu")
        state.buffers.slot_history = state.buffers.slot_history.to(state.device)
        state.buffers.slot_generated = state.buffers.slot_generated.to(state.device)
        assert not state.buffers.slot_history.is_cuda
        return state

    check_resume(make_npu_state)
