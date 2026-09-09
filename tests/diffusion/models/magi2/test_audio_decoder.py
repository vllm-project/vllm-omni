# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.magi2.audio_decoder import (
    convert_stable_audio_decoder_key,
    convert_stable_audio_decoder_state_dict,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@pytest.mark.parametrize(
    ("source", "target"),
    [
        ("pretransform.model.decoder.layers.0.weight_v", "conv1.weight_v"),
        ("pretransform.model.decoder.layers.1.layers.0.alpha", "block.0.snake1.alpha"),
        ("pretransform.model.decoder.layers.3.layers.1.weight_g", "block.2.conv_t1.weight_g"),
        (
            "pretransform.model.decoder.layers.5.layers.4.layers.2.beta",
            "block.4.res_unit3.snake2.beta",
        ),
        (
            "pretransform.model.decoder.layers.2.layers.2.layers.3.weight_v",
            "block.1.res_unit1.conv2.weight_v",
        ),
        ("pretransform.model.decoder.layers.6.alpha", "snake1.alpha"),
        ("pretransform.model.decoder.layers.7.weight_g", "conv2.weight_g"),
    ],
)
def test_stable_audio_decoder_key_conversion(source: str, target: str):
    assert convert_stable_audio_decoder_key(source) == target


def test_stable_audio_decoder_conversion_filters_and_reshapes_snake():
    alpha = torch.arange(4, dtype=torch.float32)
    weight = torch.ones(2, 4, 7)
    state = {
        "pretransform.model.decoder.layers.6.alpha": alpha,
        "pretransform.model.decoder.layers.7.weight_v": weight,
        "model.model.transformer.layers.0.weight": torch.empty(1),
    }
    converted = convert_stable_audio_decoder_state_dict(state)
    assert set(converted) == {"snake1.alpha", "conv2.weight_v"}
    assert converted["snake1.alpha"].shape == (1, 4, 1)
    torch.testing.assert_close(converted["snake1.alpha"].flatten(), alpha)
    assert converted["conv2.weight_v"] is weight


def test_unknown_decoder_key_fails_closed():
    with pytest.raises(ValueError, match="Unsupported Stable Audio decoder key"):
        convert_stable_audio_decoder_key("pretransform.model.decoder.future.weight")
