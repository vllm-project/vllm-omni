# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from pathlib import Path

import pytest
import torch

from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Config,
    Qwen3TTSTokenizerV2DecoderConfig,
)
from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
    Qwen3TTSTokenizerV2Decoder,
    Qwen3TTSTokenizerV2Model,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_model(composite: bool):
    decoder_config = dict(
        codebook_size=32,
        hidden_size=16,
        latent_dim=16,
        codebook_dim=16,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        num_hidden_layers=1,
        num_quantizers=2,
        decoder_dim=16,
        upsample_rates=(2, 2),
        upsampling_ratios=(2, 2),
        sliding_window=8,
        max_position_embeddings=128,
    )
    if not composite:
        config = Qwen3TTSTokenizerV2DecoderConfig(**decoder_config)
        config._attn_implementation = "eager"
        return Qwen3TTSTokenizerV2Decoder(config).eval()

    config = Qwen3TTSTokenizerV2Config(
        decoder_config=decoder_config,
        encoder_config=dict(
            hidden_size=16,
            num_filters=4,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=8,
            intermediate_size=32,
            num_hidden_layers=1,
            codebook_size=32,
            codebook_dim=16,
            vector_quantization_hidden_dimension=16,
            num_quantizers=2,
            num_semantic_quantizers=1,
            upsampling_ratios=[2, 2],
            upsample_groups=16,
            frame_rate=6000,
        ),
        encoder_valid_num_quantizers=2,
    )
    config._attn_implementation = "eager"
    return Qwen3TTSTokenizerV2Model(config).eval()


@pytest.mark.parametrize("composite", [False, True], ids=["decoder", "tokenizer"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@torch.inference_mode()
def test_tokenizer_rope_survives_pretrained_reload(tmp_path: Path, composite: bool, dtype: torch.dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = _make_model(composite)
    saved_parameters = {name: parameter.clone() for name, parameter in model.named_parameters()}
    model.save_pretrained(tmp_path)

    # Exercise HF's meta-device load, including nested PreTrainedModel initialization.
    loaded = type(model).from_pretrained(tmp_path, dtype=dtype, device_map="cpu", attn_implementation="eager")
    decoder = loaded.decoder if composite else loaded
    rotary = decoder.pre_transformer.rotary_emb
    expected = 1.0 / (decoder.config.rope_theta ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    for name in ("inv_freq", "original_inv_freq"):
        buffer = getattr(rotary, name)
        assert buffer.device.type == "cpu", f"{name} was left on {buffer.device}"
        torch.testing.assert_close(buffer, expected.to(buffer.dtype), rtol=0, atol=0)
        assert name in rotary._non_persistent_buffers_set
    assert not any("inv_freq" in name for name in loaded.state_dict())

    # Position zero alone would hide an all-zero frequency table.
    positions = torch.tensor([[1, 17]])
    cos, sin = rotary(torch.empty(1, 2, 8, dtype=dtype), positions)
    angles = positions[..., None].float() * expected[None, None, :]
    angles = torch.cat((angles, angles), dim=-1)
    torch.testing.assert_close(cos, angles.cos().to(dtype), rtol=0, atol=0)
    torch.testing.assert_close(sin, angles.sin().to(dtype), rtol=0, atol=0)

    assert next(loaded.parameters()).dtype == dtype
    for name, parameter in loaded.named_parameters():
        torch.testing.assert_close(parameter, saved_parameters[name].to(parameter.dtype), rtol=0, atol=0)
