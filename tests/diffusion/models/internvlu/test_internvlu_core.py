# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.internvlu.internvlu_transformer import (
    InternVLUTransformer2DModel,
    build_position_ids_3d,
)
from vllm_omni.diffusion.models.internvlu.pipeline_internvlu import (
    _move_conditioning_to_device,
    _validate_generation_config,
    process_diff_norm,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _released_generation_config():
    return {
        "decoder_projector_type": "identity",
        "padding_encoder_hidden_states": True,
        "pad_to_fix_length": False,
        "max_sequence_length": 768,
        "input_hidden_size": 4096,
        "output_hidden_size": 4096,
        "vae_downsample_factor": 8,
        "decoder_config": {
            "patch_size": 2,
            "in_channels": 64,
            "out_channels": 16,
            "num_layers": 20,
            "attention_head_dim": 128,
            "num_attention_heads": 12,
            "joint_attention_dim": 4096,
            "axes_dims_rope": [16, 56, 56],
            "video_position_scale_factor": 1.0,
            "vlm_cond_position_scale_factor": 0.25,
        },
    }


def test_generation_config_rejects_shape_drift():
    config = _released_generation_config()
    _validate_generation_config(config)
    config["decoder_config"]["num_layers"] = 19
    with pytest.raises(ValueError, match="num_layers"):
        _validate_generation_config(config)


def test_generation_config_requires_consistent_conditioning_width():
    """The conditioning width is checkpoint configuration: Stage 1 sizes its
    checks from input_hidden_size, so the three width declarations must agree
    and be a positive integer."""
    config = _released_generation_config()
    config["input_hidden_size"] = 0
    with pytest.raises(ValueError, match="input_hidden_size"):
        _validate_generation_config(config)

    config = _released_generation_config()
    config["output_hidden_size"] = 2048
    with pytest.raises(ValueError, match="output_hidden_size"):
        _validate_generation_config(config)

    config = _released_generation_config()
    config["decoder_config"]["joint_attention_dim"] = 2048
    with pytest.raises(ValueError, match="joint_attention_dim"):
        _validate_generation_config(config)


def test_position_ids_cover_latents_reference_and_vlm_image():
    image_mask = torch.tensor([False, True, True, True, True, False])
    positions = build_position_ids_3d(
        [(1, 2, 2), (1, 2, 2)],
        image_mask,
        torch.tensor([[1, 2, 2]]),
    )
    assert positions.shape == (14, 3)
    assert positions.dtype == torch.float32


def test_position_ids_fail_on_vlm_grid_mismatch():
    with pytest.raises(ValueError, match="does not match its token span"):
        build_position_ids_3d(
            [(1, 2, 2)],
            torch.tensor([False, True, True, False]),
            torch.tensor([[1, 2, 2]]),
        )


def test_process_diff_norm_matches_reference_piecewise_rule():
    norm = torch.tensor([0.0, 0.5, 1.0, 16.0])
    actual = process_diff_norm(norm, k=0.4)
    expected = torch.tensor([1.0, 1.0, 1.0, 16.0**0.4])
    torch.testing.assert_close(actual, expected)


def test_reference_count_respects_released_anyres_bound():
    model = InternVLUTransformer2DModel.__new__(InternVLUTransformer2DModel)
    torch.nn.Module.__init__(model)
    model.patch_size = 2
    model.in_channels = 64
    model.video_position_scale_factor = 1.0

    hidden_states = torch.zeros(3, 16, 8, 8)
    encoder_hidden_states = torch.zeros(3, 1, 4096)
    encoder_attention_mask = torch.ones(3, 1, dtype=torch.bool)
    encoder_image_token_mask = torch.zeros(3, 1, dtype=torch.bool)
    reference = torch.zeros(16, 8, 8)
    reference_latents = [[reference, reference], [], []]
    image_fhw_cond = [torch.empty(0, 3, dtype=torch.long) for _ in range(3)]

    with pytest.raises(ValueError, match="must be less than 2"):
        model._pack_inputs(
            hidden_states,
            encoder_hidden_states,
            encoder_attention_mask,
            encoder_image_token_mask,
            reference_latents,
            image_fhw_cond,
        )


def test_conditioning_is_moved_to_diffusion_device():
    prompt_embeds = torch.zeros(3, 4, 8, dtype=torch.float32)
    attention_mask = torch.ones(3, 4, dtype=torch.bool)
    image_mask = torch.zeros(3, 4, dtype=torch.bool)

    moved_embeds, moved_attention_mask, moved_image_mask = _move_conditioning_to_device(
        prompt_embeds,
        attention_mask,
        image_mask,
        device=torch.device("meta"),
        dtype=torch.bfloat16,
    )

    assert moved_embeds.device.type == "meta"
    assert moved_embeds.dtype == torch.bfloat16
    assert moved_attention_mask.device.type == "meta"
    assert moved_attention_mask.dtype == torch.bool
    assert moved_image_mask.device.type == "meta"
    assert moved_image_mask.dtype == torch.bool


def test_internvlu_metadata_allows_multiple_reference_images():
    metadata = get_diffusion_model_metadata("InternVLUPipeline")
    assert metadata.supports_multimodal_inputs is True
    assert metadata.max_multimodal_image_inputs is None
