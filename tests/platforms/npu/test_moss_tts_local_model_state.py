# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MOSS-TTS Local keeps its slot state and takes Ascend attention metadata on NPU."""

import pytest

pytest.importorskip("vllm_ascend")

from vllm_ascend.worker.v2.model_states.default import AscendModelState

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.moss_tts.local_model_state import MossLocalModelState
from vllm_omni.platforms.npu.models.moss_tts_local_model_state import NPUMossLocalModelState
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


def test_ascend_local_state_keeps_local_slots_and_ascend_attention():
    mro = NPUMossLocalModelState.__mro__
    assert mro.index(MossLocalModelState) < mro.index(OmniModelState) < mro.index(AscendModelState)
    assert NPUMossLocalModelState.uses_ascend_attn_metadata
