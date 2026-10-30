# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeSnakeBeta,
)

from vllm_omni.model_executor.models.common.snake_activation import SnakeBeta
from vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_code2wav import Qwen3OmniMoeCode2Wav, use_fused_snake

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _blocks():
    config = Qwen3OmniMoeCode2WavConfig(decoder_dim=32, upsample_rates=[2, 2])
    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(Qwen3OmniMoeCode2WavDecoderBlock(config, i) for i in range(2)).eval()
    with torch.no_grad():
        for module in blocks.modules():
            if isinstance(module, Qwen3OmniMoeSnakeBeta):
                module.alpha.uniform_(-0.5, 0.5)
                module.beta.uniform_(-0.5, 0.5)
    return blocks


def test_decoder_blocks_swap_every_hf_snake_and_keep_weight_names():
    blocks = _blocks()
    hf_count = sum(isinstance(m, Qwen3OmniMoeSnakeBeta) for m in blocks.modules())
    keys = set(blocks.state_dict())

    assert use_fused_snake(blocks) == hf_count > 0
    assert not any(isinstance(m, Qwen3OmniMoeSnakeBeta) for m in blocks.modules())
    assert sum(isinstance(m, SnakeBeta) for m in blocks.modules()) == hf_count
    assert set(blocks.state_dict()) == keys  # checkpoint weights load unchanged


def test_fused_snake_blocks_match_hf_blocks():
    reference = _blocks()
    fused = copy.deepcopy(reference)
    use_fused_snake(fused)
    fused.load_state_dict(reference.state_dict())
    x = torch.randn(2, 32, 20)
    with torch.no_grad():
        expected = x
        actual = x
        for ref_block, fused_block in zip(reference, fused):
            expected = ref_block(expected)
            actual = fused_block(actual)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    ("option", "device", "cuda", "fused"),
    [
        (False, "cuda", True, False),
        (True, "cuda", True, True),
        (True, "cpu", True, False),
        (True, "cuda", False, False),
    ],
)
def test_decoder_block_fusion_requires_cuda_opt_in(monkeypatch, option, device, cuda, fused):
    from vllm_omni.platforms import current_omni_platform

    monkeypatch.setattr(current_omni_platform, "is_cuda", lambda: cuda)
    config = Qwen3OmniMoeCode2WavConfig(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        decoder_dim=32,
    )
    model = Qwen3OmniMoeCode2Wav(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=config, stage_connector_config={"extra": {"codec_fused_snake": option}}
            ),
            device_config=SimpleNamespace(device=device),
        )
    )
    assert any(isinstance(m, Qwen3OmniMoeSnakeBeta) for m in model.decoder.modules()) is (not fused)
