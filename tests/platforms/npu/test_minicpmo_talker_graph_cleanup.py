# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Real NPUGraph lifecycle regression; the backbone is a small linear probe.

This verifies host cleanup across replay, not Llama attention/KV correctness.
"""

import pytest
import torch
from torch import nn

pytest.importorskip("torch_npu")

from tests.helpers.mark import hardware_marks
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import MiniCPMO45OmniTTSForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.omni, *hardware_marks(res={"npu": "A3"}, num_cards=1)]


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(8, 8, bias=False)
        self.calls = 0

    def forward(self, *, inputs_embeds, **kwargs):
        self.calls += 1
        return self.projection(inputs_embeds)


@pytest.mark.parametrize("rows", [1, 4, 16])
@torch.inference_mode()
def test_cleanup_runs_when_device_graph_skips_talker_forward(rows):
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.tts_model = _Backbone().to("npu")
    talker._deferred_cleanup_ids = set()
    talker._request_audio_states = {"live": {"step": 1}}
    talker._request_condition_states = {}
    talker._decode_codec_ids = {}
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "tts"
    model.talker = talker
    inputs = torch.randn(rows, 8, device="npu")
    for _ in range(3):
        talker(inputs_embeds=inputs)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = talker(inputs_embeds=inputs)
    calls = talker.tts_model.calls
    for step in range(3):
        done = f"done-{step}"
        talker._request_audio_states[done] = {}
        talker._request_condition_states[done] = {}
        talker._decode_codec_ids[done] = inputs
        talker.on_requests_finished([done])
        model.preprocess_batch(req_ids=["live"], model_intermediate_buffer={}, device=inputs.device)
        inputs.normal_()
        expected = talker.tts_model.projection(inputs)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected)
        assert talker.tts_model.calls == calls
        assert talker._request_audio_states == {"live": {"step": 1}}
        assert not talker._request_condition_states
        assert not talker._decode_codec_ids
        assert not talker._deferred_cleanup_ids
