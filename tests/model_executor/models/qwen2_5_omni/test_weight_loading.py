# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.qwen2_5_omni.qwen2_5_omni_token2wav import (
    Qwen2_5OmniToken2WavForConditionalGenerationVLLM,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _BufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_dict_calls = 0
        self.register_buffer("persistent", torch.ones(1))
        self.register_buffer("temporary", torch.ones(1), persistent=False)
        self.child = nn.Module()
        self.child.register_buffer("persistent", torch.ones(1))

    def state_dict(self, *args, **kwargs):
        self.state_dict_calls += 1
        return super().state_dict(*args, **kwargs)


def test_token2wav_buffer_discovery_avoids_state_dict():
    model = _BufferModel()

    buffers = Qwen2_5OmniToken2WavForConditionalGenerationVLLM.find_all_registers(model)

    assert set(buffers) == {"persistent", "child.persistent"}
    assert model.state_dict_calls == 0


class TestQwen2_5OmniWeightLoading:
    def test_qkv_weight_mapping(self):
        mock_quant_description = {
            "visual.blocks.0.attn.q.weight": {"quant_type": "FLOAT"},
            "visual.blocks.0.attn.k.weight": {"quant_type": "FLOAT"},
            "visual.blocks.0.attn.v.weight": {"quant_type": "FLOAT"},
        }

        prefix = "visual.blocks.0.attn.qkv"
        packed_modules_mapping = {
            "attn_akv_proj": [
                "attn_q_proj",
                "attn_k_proj",
                "attn_v_proj",
            ],
            "qkv": [
                "q",
                "k",
                "v",
            ],
        }

        try:
            proj_name = prefix.split(".")[-1]
            if proj_name in packed_modules_mapping:
                quant_type = None
                shard_prefixes = [
                    prefix.replace(proj_name, shard_proj_name) for shard_proj_name in packed_modules_mapping[proj_name]
                ]
                for shard_prefix in shard_prefixes:
                    shard_quant_type = mock_quant_description[shard_prefix + ".weight"]

                    if quant_type is None:
                        quant_type = shard_quant_type
                    elif shard_quant_type != quant_type:
                        raise ValueError(
                            f"Not all shards of {prefix} are quantized with same quant type."
                            f"Shard {proj_name} uses {shard_quant_type}, but another shard"
                            f"use {quant_type}. Please check quantization config."
                        )
            else:
                quant_type = mock_quant_description[prefix + ".weight"]
        except KeyError as e:
            pytest.fail(f"KeyError was raised: {e}\n")
