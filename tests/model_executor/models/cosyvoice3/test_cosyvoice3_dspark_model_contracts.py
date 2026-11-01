# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU model/real-Eagle-mixin contracts; no checkpoint or CUDA execution."""

from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.models.interfaces import EagleModelMixin

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 import CosyVoice3Model
from vllm_omni.model_executor.models.cosyvoice3.dspark import UNIFIED_MASK_TOKEN_ID, converted_config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_capture_layers_match_converted_checkpoint_conventions():
    class Decoder(torch.nn.Module, EagleModelMixin):
        pass

    decoder = Decoder()
    model = CosyVoice3Model.__new__(CosyVoice3Model)
    torch.nn.Module.__init__(model)
    model.model_stage = "cosyvoice3_talker"
    model.talker = SimpleNamespace(llm=SimpleNamespace(model=decoder))
    layers = model.get_eagle3_default_aux_hidden_state_layers()
    config = converted_config(
        dict(
            speculators_model_type="dspark",
            markov_head_type="vanilla",
            mask_token_id=UNIFIED_MASK_TOKEN_ID,
            aux_hidden_state_layer_ids=list(layers),
            transformer_layer_config=dict(hidden_size=896),
            markov_rank=256,
            block_size=8,
        ),
        6500,
    )
    model.set_aux_hidden_state_layers(tuple(config["eagle_aux_hidden_state_layer_ids"]))
    assert decoder.aux_hidden_state_layers == layers
    captured: list[torch.Tensor] = []
    for index in range(24):
        decoder._maybe_add_hidden_state(captured, index, torch.full((1, 2), index), None)
    assert [int(value[0, 0]) for value in captured] == config["eagle_aux_hidden_state_layer_ids"]
    assert config["target_layer_ids"] == [index - 1 for index in layers]
    assert config["dflash_config"]["target_layer_ids"] == config["target_layer_ids"]
    assert model._returns_tuple and model.supports_mrv2_full_graph_aux_outputs


def test_dspark_language_model_is_the_decoder_not_the_speech_head():
    model = CosyVoice3Model.__new__(CosyVoice3Model)
    torch.nn.Module.__init__(model)
    model._dspark_config = object()
    decoder = object()
    model.talker = SimpleNamespace(llm=decoder)
    assert model.get_language_model() is decoder
