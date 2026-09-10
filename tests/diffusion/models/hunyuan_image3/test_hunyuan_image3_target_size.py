# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-size precedence at the HunyuanImage3 AR-to-DiT boundary."""

from types import SimpleNamespace

import pytest
import torch
from PIL import Image

import vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 as pipeline_module
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _ResolutionGroup:
    def get_base_size_and_ratio_index(self, width, height):
        return 1024, 0

    def __getitem__(self, index):
        return SimpleNamespace(width=32, height=32)


class _VisionProcessor:
    patch_size = 16

    def __call__(self, image, **kwargs):
        return {
            "pixel_values": torch.zeros(1, 1, 3),
            "spatial_shapes": torch.tensor([[1, 1]]),
            "pixel_attention_mask": torch.ones(1, 1, dtype=torch.bool),
        }


@pytest.fixture(params=[DiffusionKVCacheMode.DENSE_LEGACY, DiffusionKVCacheMode.PAGED_SCHEDULER])
def preprocess(monkeypatch, request):
    hf_config = SimpleNamespace(vae_downsample_factor=(8, 8), patch_size=2, image_base_size=1024)
    image_processor = SimpleNamespace(
        reso_group=_ResolutionGroup(),
        vision_encoder_processor=_VisionProcessor(),
        vae_processor=lambda image: torch.zeros(1, 3, image.height, image.width),
    )
    monkeypatch.setattr(pipeline_module, "get_config", lambda *_args, **_kwargs: hf_config)
    monkeypatch.setattr(pipeline_module, "HunyuanImage3ImageProcessor", lambda _config: image_processor)
    monkeypatch.setattr(pipeline_module, "TokenizerWrapper", lambda _model: object())
    monkeypatch.setattr(pipeline_module.GenerationConfig, "from_pretrained", lambda _model: object())
    layout_sizes = []

    def prepare_layout(diffusion_request, **kwargs):
        layout_sizes.append((diffusion_request.sampling_params.height, diffusion_request.sampling_params.width))
        return object()

    monkeypatch.setattr(pipeline_module.request_layout_utils, "prepare_hunyuan_layout", prepare_layout)
    monkeypatch.setattr(pipeline_module.request_layout_utils, "build_hunyuan_diffusion_kv_requests", lambda *_args: ())
    process = pipeline_module.get_hunyuan_image_3_pre_process_func(
        SimpleNamespace(model="model", diffusion_kv_mode=request.param)
    )

    def run(prompt, sampling_size, expected_size):
        diffusion_request = OmniDiffusionRequest(
            prompt=prompt,
            sampling_params=OmniDiffusionSamplingParams(height=sampling_size[0], width=sampling_size[1]),
            request_id="target-size",
        )
        assert process(diffusion_request) is diffusion_request
        assert (diffusion_request.sampling_params.height, diffusion_request.sampling_params.width) == expected_size
        if request.param is DiffusionKVCacheMode.PAGED_SCHEDULER:
            assert layout_sizes == [expected_size]
        else:
            assert not layout_sizes
        return diffusion_request

    return run


@pytest.mark.parametrize(
    "bridge_size,sampling_size,expected_size",
    [
        ((832, 1216), (None, None), (832, 1216)),
        ((832, 1216), (512, 768), (512, 768)),
        ((832, 1216), (512, None), (512, 1216)),
        ((832, 1216), (None, 768), (832, 768)),
        ((832, None), (None, None), (832, None)),
        ((None, 1216), (None, None), (None, 1216)),
        ((None, None), (None, None), (None, None)),
    ],
)
def test_t2i_target_size(preprocess, bridge_size, sampling_size, expected_size):
    prompt = {"prompt": "A cat", "height": bridge_size[0], "width": bridge_size[1]}
    preprocess(prompt, sampling_size, expected_size)


def test_plain_text_retains_pipeline_size_default(preprocess):
    result = preprocess("A cat", (None, None), (None, None))
    assert result.prompt["prompt"] == "A cat"


@pytest.mark.parametrize("image_key", ["multi_modal_data", "pil_image"])
@pytest.mark.parametrize(
    "bridge_size,sampling_size,expected_size",
    [
        ((None, None), (None, None), (600, 800)),
        ((832, 1216), (None, None), (832, 1216)),
        ((832, 1216), (512, 768), (512, 768)),
        ((832, None), (None, 768), (832, 768)),
        ((None, 1216), (512, None), (512, 1216)),
    ],
)
def test_it2i_target_size(preprocess, image_key, bridge_size, sampling_size, expected_size):
    image = Image.new("RGB", (800, 600))
    prompt = {"prompt": "Edit the image", "height": bridge_size[0], "width": bridge_size[1]}
    prompt[image_key] = {"image": [image]} if image_key == "multi_modal_data" else image
    result = preprocess(prompt, sampling_size, expected_size)
    assert len(result.prompt["additional_information"]["batch_cond_image_info"]) == 1
