# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.longcat_video.longcat_video_avatar_transformer import (
    LongCatVideoAvatarTransformer3DModel,
    _read_config,
    replace_linear_with_quantized,
)
from vllm_omni.diffusion.models.longcat_video.pipeline_longcat_video_avatar import (
    _avatar_model_allow_patterns,
    _build_multi_speaker_ref_target_masks,
    _default_at2v_shape,
    _infer_asset_root_from_path,
    _prepare_multi_speaker_audio_arrays,
    _resolve_num_segments,
    prepare_longcat_video_avatar_model_for_omni,
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
    assert "whisper-large-v3/*" not in allow_patterns
    assert "whisper-large-v3/model.safetensors" in allow_patterns
    assert "whisper-large-v3/pytorch_model.bin" not in allow_patterns
    assert "whisper-large-v3/flax_model.msgpack" not in allow_patterns
    assert "vocal_separator/*" in allow_patterns


def test_longcat_video_avatar_prepare_model_adds_omni_metadata(tmp_path):
    model_dir = tmp_path / "LongCat-Video-Avatar-1.5"
    base_model_dir = model_dir / "base_model_int8"
    base_model_dir.mkdir(parents=True)
    (base_model_dir / "config.json").write_text('{"model_type": "longcat_avatar"}', encoding="utf-8")
    (model_dir / "model_index.json").write_text("{}", encoding="utf-8")

    prepared = prepare_longcat_video_avatar_model_for_omni(str(model_dir), use_int8=True)

    assert prepared == str(model_dir)
    model_index = json.loads((model_dir / "model_index.json").read_text(encoding="utf-8"))
    assert model_index["_class_name"] == "LongCatVideoAvatarPipeline"
    assert model_index["_diffusers_version"] == "0.38.0"
    transformer_config = json.loads((model_dir / "transformer" / "config.json").read_text(encoding="utf-8"))
    assert transformer_config == {"model_type": "longcat_avatar"}


def test_longcat_video_avatar_infers_repo_root_from_official_asset_path(tmp_path):
    input_json = tmp_path / "assets" / "avatar" / "single_example_1.json"
    input_json.parent.mkdir(parents=True)
    input_json.write_text("{}", encoding="utf-8")

    assert _infer_asset_root_from_path(input_json) == tmp_path


def test_longcat_video_avatar_falls_back_to_json_parent_without_official_layout(tmp_path):
    input_json = tmp_path / "single_example_1.json"
    input_json.write_text("{}", encoding="utf-8")

    assert _infer_asset_root_from_path(input_json) == tmp_path


@pytest.mark.parametrize(
    ("value", "audio_duration", "expected"),
    [
        (None, 26.6, 1),
        ("1", 26.6, 1),
        (3, 26.6, 3),
        ("auto", 3.0, 1),
        ("auto", 23.8, 8),
        ("auto", 26.6, 9),
    ],
)
def test_longcat_video_avatar_resolve_num_segments(
    value,
    audio_duration: float,
    expected: int,
):
    assert (
        _resolve_num_segments(
            value,
            audio_duration=audio_duration,
            num_frames=93,
            num_cond_frames=13,
            save_fps=25,
        )
        == expected
    )


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


def test_longcat_video_avatar_multi_speaker_para_audio_arrays_are_aligned():
    left = np.ones(3, dtype=np.float32)
    right = np.ones(5, dtype=np.float32) * 2

    left_out, right_out = _prepare_multi_speaker_audio_arrays(
        [left, right],
        audio_type="para",
        generate_duration=0.5,
        sample_rate=10,
    )

    assert left_out.tolist() == [1, 1, 1, 0, 0]
    assert right_out.tolist() == [2, 2, 2, 2, 2]


def test_longcat_video_avatar_multi_speaker_add_audio_arrays_are_sequential():
    left = np.ones(3, dtype=np.float32)
    right = np.ones(2, dtype=np.float32) * 2

    left_out, right_out = _prepare_multi_speaker_audio_arrays(
        [left, right],
        audio_type="add",
        generate_duration=0.1,
        sample_rate=10,
    )

    assert left_out.tolist() == [1, 1, 1, 0, 0]
    assert right_out.tolist() == [0, 0, 0, 2, 2]


def test_longcat_video_avatar_multi_speaker_masks_use_explicit_bboxes():
    image = Image.new("RGB", (8, 6))
    masks, use_background = _build_multi_speaker_ref_target_masks(
        image,
        {
            "person1": [1, 1, 4, 3],
            "person2": [2, 4, 5, 7],
        },
    )

    assert not use_background
    assert masks.shape == (3, 6, 8)
    assert masks[0, 1:4, 1:3].sum() == 6
    assert masks[1, 2:5, 4:7].sum() == 9
    assert masks[2, 0, 0] == 1
    assert masks[2, 2, 2] == 0


def test_longcat_video_avatar_multi_speaker_masks_default_to_left_right_split():
    image = Image.new("RGB", (10, 8))
    masks, use_background = _build_multi_speaker_ref_target_masks(image, None)

    assert not use_background
    assert masks.shape == (3, 8, 10)
    assert masks[0, :, :5].sum() > 0
    assert masks[0, :, 5:].sum() == 0
    assert masks[1, :, :5].sum() == 0
    assert masks[1, :, 5:].sum() > 0


def test_longcat_video_avatar_transformer_accepts_multi_speaker_masks():
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
    hidden_states = torch.randn(1, 16, 5, 4, 4)
    timestep = torch.zeros(1, 5)
    encoder_hidden_states = torch.randn(1, 1, 4, 4)
    audio_embs = torch.randn(2, 5, 5, 12, 4)
    ref_target_masks = torch.zeros(3, 4, 4)
    ref_target_masks[0, :, :2] = 1
    ref_target_masks[1, :, 2:] = 1
    ref_target_masks[2] = torch.where(ref_target_masks[0] + ref_target_masks[1] > 0, 0, 1)

    output = model(
        hidden_states=hidden_states,
        timestep=timestep,
        encoder_hidden_states=encoder_hidden_states,
        num_cond_latents=1,
        audio_embs=audio_embs,
        ref_target_masks=ref_target_masks,
    )

    assert output.shape == hidden_states.shape
