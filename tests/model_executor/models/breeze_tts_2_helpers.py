# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Weight-free Breeze talker and request helpers shared by model tests."""

from dataclasses import dataclass
from unittest.mock import Mock

import torch

from vllm_omni.model_executor.models.breeze_tts_2.first_code_sampler import BreezeFirstCodeSampler
from vllm_omni.model_executor.models.breeze_tts_2.modeling_breeze import BreezeForConditionalGeneration


@dataclass
class TalkerConfig:
    vocab_size: int
    eos_token_id: int


@dataclass
class DepthStub:
    generate_frames: Mock


def _small_talker():
    model = BreezeForConditionalGeneration.__new__(BreezeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._first_code_sampler = BreezeFirstCodeSampler()
    model.config = TalkerConfig(vocab_size=8, eos_token_id=7)
    model.num_codebooks, model.codebook_size, model.hidden_size = 3, 4, 2
    model.lm_head = torch.nn.Linear(2, 8, bias=False)
    with torch.no_grad():
        model.lm_head.weight.zero_()
        model.lm_head.weight[1] = torch.tensor([10.0, 0.0])
        model.lm_head.weight[2] = torch.tensor([9.5, 0.0])
        model.lm_head.weight[3] = torch.tensor([0.0, 10.0])
        model.lm_head.weight[7] = torch.tensor([-10.0, -10.0])
    model.depth_decoder = DepthStub(
        generate_frames=Mock(side_effect=lambda hidden, first, **kwargs: first[:, None].repeat(1, 3))
    )
    return model


def _request_info(request_id):
    return {
        "global_request_id": [request_id],
        "breeze_prompt": {"role": "cond", "guidance_scale": 1.0},
        "breeze_sampling": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.1},
        "breeze_state": {
            "generator": torch.Generator().manual_seed(42),
            "history": torch.empty(1, 0, dtype=torch.long),
            "current": torch.zeros(1, 3, dtype=torch.long),
        },
    }


def _embedding_talker(dtype):
    model = BreezeForConditionalGeneration.__new__(BreezeForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.num_codebooks, model.hidden_size = 16, 8
    model.register_buffer("offsets", torch.arange(model.num_codebooks) * 7)
    model.depth_decoder = torch.nn.Module()
    model.depth_decoder.embed_tokens = torch.nn.Embedding(16 * 7, 8, dtype=dtype)
    with torch.no_grad():
        model.depth_decoder.embed_tokens.weight.copy_(
            torch.randn(16 * 7, 8, generator=torch.Generator().manual_seed(17))
        )
    return model


def _decode_info(request_id, current, seed):
    return {
        "global_request_id": [request_id],
        "breeze_prompt": {"role": "cond", "guidance_scale": 1.0},
        "breeze_sampling": {"temperature": 0.9, "top_k": 4, "top_p": 1.0, "repetition_penalty": 1.1},
        "breeze_state": {
            "current": current,
            "history": torch.tensor([[1, 2, 3]], dtype=torch.long),
            "generator": torch.Generator().manual_seed(seed),
        },
        "_omni_is_prefill": False,
    }
