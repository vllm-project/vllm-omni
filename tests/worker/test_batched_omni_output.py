# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import MossTTSLocalTalkerForGeneration

pytestmark = [
    pytest.mark.core_model,
    *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1),
]


def test_mixed_stop_flags_do_not_synchronize_cuda():
    model = object.__new__(MossTTSLocalTalkerForGeneration)
    torch.nn.Module.__init__(model)
    model.n_vq = 12
    model.audio_pad_token_id = 1024
    valid = torch.arange(12, device="cuda").view(1, 12)
    stopped = torch.full_like(valid, 1024)
    infos = [{"audio_codes": {"current": valid}}, {}, {"audio_codes": {"current": stopped}}]
    hidden = torch.ones((3, 4), device="cuda")
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode("error")
        output = model.make_omni_output(hidden, runtime_additional_information=infos)
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    assert model._batch_should_continue.tolist() == [True, True, False]
    torch.testing.assert_close(output.multimodal_outputs["codes"]["audio"], [valid, valid[:0], stopped])
    assert all("current" not in info.get("audio_codes", {}) for info in infos)
    prefill = model.make_omni_output(hidden[:1], runtime_additional_information=[{}])
    assert model._batch_should_continue is None and prefill.multimodal_outputs == {}
