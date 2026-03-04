# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from pytest_mock import MockerFixture

from vllm_omni.diffusion.models.hunyuan_image_3 import pipeline_hunyuan_image_3 as hy3_module
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_hunyuan_image3_preprocess_builds_joint_image_info(mocker: MockerFixture):
    config = SimpleNamespace(vae_downsample_factor=(8, 8), patch_size=2)
    mocker.patch.object(hy3_module, "get_config", return_value=config)

    class _DummyVisionProcessor:
        patch_size = 16

        def __call__(self, image, return_tensors="pt"):
            del image, return_tensors
            return {
                "pixel_values": torch.zeros(1, 10, 4),
                "spatial_shapes": torch.tensor([[2, 5]], dtype=torch.long),
                "pixel_attention_mask": torch.ones(1, 10, dtype=torch.long),
            }

    class _DummyImageProcessor:
        def __init__(self, _cfg):
            self.reso_group = SimpleNamespace(
                get_target_size=lambda width, height: (np.int64(width), np.int64(height)),
                get_base_size_and_ratio_index=lambda width, height: (np.int64(1024), np.int64(3)),
            )
            self.vae_processor = lambda image: torch.zeros(3, image.size[1], image.size[0])
            self.vision_encoder_processor = _DummyVisionProcessor()

    mocker.patch.object(hy3_module, "HunyuanImage3ImageProcessor", _DummyImageProcessor)

    preprocess = hy3_module.get_hunyuan_image_3_pre_process_func(SimpleNamespace(model="dummy-model"))
    request = OmniDiffusionRequest(
        prompts=[{"prompt": "edit image", "multi_modal_data": {"image": [Image.new("RGB", (32, 16), "white")]}}],
        sampling_params=OmniDiffusionSamplingParams(),
    )

    request = preprocess(request)
    prompt = request.prompts[0]
    cond_infos = prompt["additional_information"]["batch_cond_image_info"]
    assert len(cond_infos) == 1

    cond_info_payload = cond_infos[0]
    assert isinstance(cond_info_payload, dict)
    assert cond_info_payload["type"] == "joint_image_info"
    assert cond_info_payload["vae_image_info"]["image_tensor"].shape == (3, 16, 32)
    assert cond_info_payload["vae_image_info"]["token_width"] == 2
    assert cond_info_payload["vae_image_info"]["token_height"] == 1
    assert isinstance(cond_info_payload["vae_image_info"]["image_width"], int)
    assert isinstance(cond_info_payload["vae_image_info"]["image_height"], int)
    assert isinstance(cond_info_payload["vae_image_info"]["base_size"], int)
    assert isinstance(cond_info_payload["vae_image_info"]["ratio_index"], int)
    assert tuple(cond_info_payload["vision_encoder_kwargs"]["spatial_shapes"].tolist()) == (2, 5)
    roundtrip_cond_info = hy3_module._joint_image_info_from_payload(cond_info_payload)
    assert isinstance(roundtrip_cond_info, hy3_module.JointImageInfo)
    assert roundtrip_cond_info.vae_image_info.image_tensor.shape == (3, 16, 32)
    assert request.sampling_params.width == 32
    assert request.sampling_params.height == 16


def test_hunyuan_image3_light_projector_is_callable():
    projector = hy3_module.LightProjector(
        {
            "projector_type": "linear",
            "input_dim": 4,
            "n_embed": 8,
        }
    )
    inputs = torch.randn(2, 3, 4)
    outputs = projector(inputs)
    assert outputs.shape == (2, 3, 8)
