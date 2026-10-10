# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.model_executor.models.lychee_fd.audio_encoder import (
    LycheeAudioAdaptor,
    LycheeAudioEncoder,
)
from vllm_omni.model_executor.models.lychee_fd.configuration_lychee import LycheeAudioEncoderConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _tiny_config() -> LycheeAudioEncoderConfig:
    return LycheeAudioEncoderConfig(
        n_mels=8,
        n_audio_ctx=32,
        n_audio_state=16,
        n_audio_head=4,
        n_audio_layer=2,
        llm_dim=24,
        kernel_size=3,
        adapter_stride=2,
    )


def test_encoder_and_adaptor_preserve_reference_lengths() -> None:
    config = _tiny_config()
    encoder = LycheeAudioEncoder(config).eval()
    adaptor = LycheeAudioAdaptor(config).eval()
    features = torch.randn(2, config.n_mels, 42)
    lengths = torch.tensor([40, 28], dtype=torch.int32)

    encoded, encoder_lengths = encoder(features, lengths)
    adapted = adaptor(encoded)
    adapted_lengths = adaptor.output_lengths(encoder_lengths)

    assert encoded.shape == (2, 10, config.n_audio_state)
    assert encoder_lengths.tolist() == [10, 7]
    assert adapted.shape == (2, 5, config.llm_dim)
    assert adapted_lengths.tolist() == [5, 4]


def test_encoder_accepts_fp32_features_with_bfloat16_parameters() -> None:
    config = _tiny_config()
    encoder = LycheeAudioEncoder(config).eval().to(torch.bfloat16)
    adaptor = LycheeAudioAdaptor(config).eval().to(torch.bfloat16)
    features = torch.randn(1, config.n_mels, 42, dtype=torch.float32)
    lengths = torch.tensor([40], dtype=torch.int32)

    encoded, encoder_lengths = encoder(features, lengths)
    adapted = adaptor(encoded)

    assert encoded.dtype == torch.float32
    assert adapted.dtype == torch.float32
    assert encoder_lengths.tolist() == [10]


def test_audio_parameter_names_match_checkpoint_roots() -> None:
    config = _tiny_config()
    modules = torch.nn.ModuleDict(
        {
            "encoder": LycheeAudioEncoder(config),
            "adapter": LycheeAudioAdaptor(config),
        }
    )
    names = set(dict(modules.named_parameters()))

    assert "encoder.blocks.0.attn.query.weight" in names
    assert "encoder.blocks.0.mlp.0.weight" in names
    assert "encoder.positional_embedding.weight" in names
    assert "adapter.conv.weight" in names
    assert "adapter.linear2.bias" in names
