# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

import pytest
import torch

from vllm_omni.diffusion.models.longcat_video.longcat_video_avatar_transformer import (
    LongCatVideoAvatarTransformer3DModel,
    _read_config,
    replace_linear_with_quantized,
)
from vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar import (
    _avatar_model_allow_patterns,
    _default_at2v_shape,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize(
    ("resolution", "expected_shape"),
    [
        ("480p", (480, 832)),
        ("720p", (736, 1248)),
    ],
)
def test_longcat_video_avatar_at2v_default_shape_uses_resolution_bucket(
    resolution: str,
    expected_shape: tuple[int, int],
):
    assert _default_at2v_shape(resolution) == expected_shape


@pytest.mark.parametrize(
    ("use_int8", "expected_weight_dir", "unexpected_weight_dir"),
    [
        (True, "base_model_int8/*", "base_model/*"),
        (False, "base_model/*", "base_model_int8/*"),
    ],
)
def test_longcat_video_avatar_allow_patterns_download_one_weight_set(
    use_int8: bool,
    expected_weight_dir: str,
    unexpected_weight_dir: str,
):
    allow_patterns = _avatar_model_allow_patterns(use_int8)

    assert expected_weight_dir in allow_patterns
    assert unexpected_weight_dir not in allow_patterns


def test_longcat_video_avatar_read_config_filters_non_constructor_metadata(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "_class_name": "LongCatVideoAvatarTransformer3DModel",
                "architectures": ["LongCatVideoAvatarTransformer3DModel"],
                "_diffusers_version": "0.35.1",
                "model_max_length": 512,
                "hidden_size": 4096,
                "depth": 48,
                "num_heads": 32,
            }
        ),
        encoding="utf-8",
    )

    config = _read_config(config_path)

    assert config == {
        "hidden_size": 4096,
        "depth": 48,
        "num_heads": 32,
    }


def test_longcat_video_avatar_load_weights_supports_int8_buffers():
    model = LongCatVideoAvatarTransformer3DModel(
        hidden_size=4,
        depth=0,
        num_heads=1,
        caption_channels=4,
        intermediate_dim=4,
        output_dim=4,
        audio_channel=4,
        context_tokens=1,
    )
    replace_linear_with_quantized(model)

    buffer_name = "t_embedder.mlp.0.weight_int8"
    buffers = dict(model.named_buffers())
    loaded_weight = torch.ones_like(buffers[buffer_name])

    loaded_params = model.load_weights([(buffer_name, loaded_weight)])

    assert buffer_name in loaded_params
    assert torch.equal(dict(model.named_buffers())[buffer_name], loaded_weight)
