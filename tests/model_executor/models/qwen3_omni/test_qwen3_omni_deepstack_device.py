# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Deepstack buffers must remain usable when the vision tower is skipped."""

import pytest
import torch
from torch import nn
from transformers import Qwen3OmniMoeThinkerConfig
from vllm.config import DeviceConfig, ModelConfig, MultiModalConfig, SchedulerConfig, VllmConfig
from vllm.model_executor.models.utils import StageMissingLayer

import vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker as thinker_module

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _LanguageModel(nn.Module):
    def make_empty_intermediate_tensors(self):
        raise AssertionError("No pipeline-parallel allocation is needed by this test")


class _DecoderLayer(nn.Module):
    def forward(self, positions, hidden_states, residual):
        return hidden_states, residual


class _Norm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states, residual


@pytest.mark.parametrize("vision_limit", [0, 1])
@pytest.mark.parametrize("num_tokens", [4, 8])
def test_deepstack_buffers_use_execution_device(mocker, vision_limit, num_tokens):
    """Exercise the real constructor, tower-skip context and LM addition."""
    config = Qwen3OmniMoeThinkerConfig(
        text_config={"hidden_size": 8},
        vision_config={"out_hidden_size": 8, "deepstack_visual_indexes": [0, 1]},
    )
    multimodal_config = MultiModalConfig(limit_per_prompt={"image": vision_limit, "video": vision_limit, "audio": 1})
    model_config = mocker.Mock(spec=ModelConfig, hf_config=config, multimodal_config=multimodal_config)
    model_config.get_multimodal_config.return_value = multimodal_config
    vllm_config = mocker.Mock(
        spec=VllmConfig,
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=mocker.Mock(spec=SchedulerConfig, max_num_batched_tokens=4),
        quant_config=None,
    )
    vllm_config.with_hf_config.return_value = vllm_config

    # Replace only the large networks. vLLM's real _mark_tower_model context
    # still builds the disabled visual module on meta and removes its weights.
    mocker.patch.object(thinker_module, "Qwen3OmniMoeAudioEncoder", side_effect=lambda *args, **kwargs: nn.Linear(8, 8))
    mocker.patch.object(thinker_module, "Qwen3Omni_VisionTransformer", side_effect=lambda **kwargs: nn.Linear(8, 8))
    mocker.patch.object(thinker_module, "Qwen3MoeLLMForCausalLM", return_value=_LanguageModel())
    thinker = thinker_module.Qwen3OmniMoeThinkerForConditionalGeneration(vllm_config=vllm_config)
    assert isinstance(thinker.visual, StageMissingLayer) is (vision_limit == 0)

    # Resizing must also retain the buffer device rather than the ambient one.
    with torch.device("meta"):
        deepstack = thinker._get_deepstack_input_embeds(num_tokens)

    model = object.__new__(thinker_module.Qwen3MoeLLMModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([_DecoderLayer(), _DecoderLayer()])
    model.start_layer = 0
    model.end_layer = 2
    model.norm = _Norm()
    mocker.patch.object(thinker_module, "get_pp_group", return_value=mocker.Mock(is_first_rank=True, is_last_rank=True))
    inputs = torch.ones(num_tokens, 8)
    # Before the fix, audio-only construction reaches this actual forward
    # with meta buffers and raises the reported cross-device addition error.
    output, _ = thinker_module.Qwen3MoeLLMModel.forward(
        model,
        input_ids=None,
        positions=torch.arange(num_tokens),
        inputs_embeds=inputs,
        deepstack_input_embeds=deepstack,
    )
    torch.testing.assert_close(output, inputs)
    assert len(thinker.deepstack_input_embeds) == 2
    assert thinker.deepstack_input_embeds_num_tokens == 0
    for buffer in thinker.deepstack_input_embeds:
        assert buffer.device == torch.device("cpu")
        assert not buffer.is_meta
        assert buffer.shape == (num_tokens, 8)
        assert buffer.dtype == torch.get_default_dtype()
        assert torch.count_nonzero(buffer).item() == 0
