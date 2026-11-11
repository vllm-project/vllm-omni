# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""HunyuanImage3 MRv2 must use Ascend attention metadata on NPU."""

import pytest
import torch

pytest.importorskip("vllm_ascend")

from vllm.config import CUDAGraphMode
from vllm_ascend.worker.v2.input_batch import AscendInputBatch
from vllm_ascend.worker.v2.model_states.default import AscendModelState

from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import HunyuanImage3ForConditionalGeneration
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu.models.hunyuan_image3_model_state import NPUHunyuanImage3ModelState
from vllm_omni.worker_v2.model_states import init_omni_model_state
from vllm_omni.worker_v2.model_states.omni_model_state import OmniModelState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("npu", [False, True])
def test_factory_selects_platform_state(monkeypatch, mocker, npu):
    monkeypatch.setattr(current_omni_platform, "is_npu", lambda: npu)
    init = mocker.patch.object(OmniModelState, "__init__", return_value=None)
    model = HunyuanImage3ForConditionalGeneration.__new__(HunyuanImage3ForConditionalGeneration)
    config, cache = object(), object()
    device = torch.device("cpu")
    state = init_omni_model_state(config, model, cache, device)
    assert type(state) is (NPUHunyuanImage3ModelState if npu else OmniModelState)
    init.assert_called_once_with(config, model, cache, device)


def test_ascend_attention_preserves_omni_inputs_and_sampler(mocker):
    state = NPUHunyuanImage3ModelState.__new__(NPUHunyuanImage3ModelState)
    prepare = mocker.patch.object(AscendModelState, "prepare_attn", return_value={})
    batch = AscendInputBatch.__new__(AscendInputBatch)
    args = (batch, CUDAGraphMode.NONE, (), None, mocker.sentinel.attn_groups, None)
    assert state.prepare_attn(*args) == {}
    prepare.assert_called_once_with(*args, for_capture=False, ubatch_idx=0, model_specific_attn_metadata=None)
    assert state.uses_ascend_attn_metadata
    assert NPUHunyuanImage3ModelState.prepare_inputs is OmniModelState.prepare_inputs
    assert NPUHunyuanImage3ModelState.custom_sampler is OmniModelState.custom_sampler
