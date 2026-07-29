# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for HunyuanImage3 infer_align_image_size handling."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from vllm_omni.diffusion.models.hunyuan_image3 import pipeline_hunyuan_image3 as hunyuan_pipeline
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3ImageProcessor,
    ImageInfo,
    JointImageInfo,
)
from vllm_omni.diffusion.models.hunyuan_image3.pipeline_hunyuan_image3 import (
    _flag_value_enabled,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
    HunyuanImage3Processor,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _gradient_image(width: int = 8, height: int = 4) -> Image.Image:
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :, 0] = np.arange(width, dtype=np.uint8) * 25
    arr[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None] * 60
    arr[:, :, 2] = 17
    return Image.fromarray(arr, mode="RGB")


def test_flag_parser_handles_string_false():
    assert _flag_value_enabled("false") is False
    assert _flag_value_enabled("true") is True
    assert _flag_value_enabled("0") is False
    assert _flag_value_enabled("yes") is True
    assert _flag_value_enabled(False) is False
    assert _flag_value_enabled(True) is True


class _FixedTargetResolutionGroup:
    def get_target_size(self, width: int, height: int):
        return 4, 4

    def get_base_size_and_ratio_index(self, width: int, height: int):
        return 1024, 0


def _fake_vae_processor(image: Image.Image):
    arr = np.asarray(image, dtype=np.float32)
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return tensor.unsqueeze(0)


def _fake_vit_processor(_image: Image.Image):
    return {
        "pixel_values": torch.zeros((1, 1, 3), dtype=torch.float32),
        "pixel_attention_mask": torch.ones((1, 1), dtype=torch.bool),
        "spatial_shapes": torch.tensor([[1, 1]], dtype=torch.long),
    }


@pytest.mark.parametrize(
    ("infer_align_image_size", "expected_image"),
    [
        (False, lambda src: HunyuanImage3Processor._resize_and_crop(None, src, (4, 4), crop_type="center")),
        (True, lambda src: HunyuanImage3Processor._resize_and_crop(None, src, (4, 4), crop_type="resize")),
    ],
)
def test_ar_process_image_uses_requested_crop_mode_and_preserves_original_size(
    infer_align_image_size: bool,
    expected_image,
):
    src = _gradient_image()
    processor = object.__new__(HunyuanImage3Processor)
    processor.infer_align_image_size = infer_align_image_size
    processor.hf_config = SimpleNamespace(
        vit={"num_channels": 3},
        vae_downsample_factor=(1, 1),
        patch_size=1,
    )
    processor.reso_group = _FixedTargetResolutionGroup()
    processor.vae_processor = _fake_vae_processor
    processor.vision_encoder_processor = _fake_vit_processor

    result = processor.process_image(src)

    expected_tensor = _fake_vae_processor(expected_image(src)).squeeze(0).reshape(-1)
    assert torch.equal(result["vae_pixel_values"], expected_tensor)
    assert result["ori_image_width"].tolist() == [src.width]
    assert result["ori_image_height"].tolist() == [src.height]


def test_resize_and_crop_rejects_unknown_crop_type():
    src = _gradient_image()

    with pytest.raises(ValueError, match="Unsupported crop_type"):
        HunyuanImage3Processor._resize_and_crop(None, src, (4, 4), crop_type="unknown")


class _FixedDiffusionResolutionGroup:
    base_size = 1024

    def __getitem__(self, idx: int):
        assert idx == 0
        return SimpleNamespace(width=4, height=4, ratio=1.0)

    def get_base_size_and_ratio_index(self, width: int, height: int):
        return self.base_size, 0


class _FakeVisionProcessor:
    patch_size = 1

    def __call__(self, image: Image.Image, return_tensors: str = "pt"):
        assert return_tensors == "pt"
        return _fake_vit_processor(image)


class _FakeDiffusionImageProcessor:
    def __init__(self, _hf_config):
        self.reso_group = _FixedDiffusionResolutionGroup()
        self.vae_processor = _fake_vae_processor
        self.vision_encoder_processor = _FakeVisionProcessor()


@pytest.mark.parametrize(
    ("infer_align_image_size", "expected_image"),
    [
        (False, lambda src: HunyuanImage3Processor._resize_and_crop(None, src, (4, 4), crop_type="center")),
        (True, lambda src: HunyuanImage3Processor._resize_and_crop(None, src, (4, 4), crop_type="resize")),
    ],
)
def test_dit_preprocess_uses_requested_crop_mode_and_records_original_size(
    monkeypatch: pytest.MonkeyPatch,
    infer_align_image_size: bool,
    expected_image,
):
    monkeypatch.setattr(
        hunyuan_pipeline,
        "get_config",
        lambda *args, **kwargs: SimpleNamespace(vae_downsample_factor=(1, 1), patch_size=1),
    )
    monkeypatch.setattr(hunyuan_pipeline, "HunyuanImage3ImageProcessor", _FakeDiffusionImageProcessor)

    src = _gradient_image()
    request = OmniDiffusionRequest(
        prompts=[{"prompt": "edit", "multi_modal_data": {"image": src}}],
        sampling_params=OmniDiffusionSamplingParams(extra_args={"infer_align_image_size": infer_align_image_size}),
        request_id="req-1",
    )

    processed = hunyuan_pipeline.get_hunyuan_image_3_pre_process_func(SimpleNamespace(model="fake"))(request)
    payload = processed.prompts[0]["additional_information"]["batch_cond_image_info"][0]
    vae_payload = payload["vae_image_info"]

    expected_tensor = _fake_vae_processor(expected_image(src))
    assert torch.equal(vae_payload["image_tensor"], expected_tensor)
    assert vae_payload["ori_image_width"] == src.width
    assert vae_payload["ori_image_height"] == src.height
    assert processed.sampling_params.width == src.width
    assert processed.sampling_params.height == src.height


class _FakeResolutionGroup:
    base_size = 1024

    def __init__(self) -> None:
        self._ratios = {
            0: 1.0,
            1: 0.75,
            2: 2.0,
        }

    def __getitem__(self, idx: int):
        return SimpleNamespace(ratio=self._ratios[idx])

    def get_base_size_and_ratio_index(self, width: int, height: int):
        ratio = height / width
        if abs(ratio - 0.75) < 0.01:
            return self.base_size, 1
        if abs(ratio - 2.0) < 0.01:
            return self.base_size, 2
        return self.base_size, 0


def _fake_processor():
    processor = object.__new__(HunyuanImage3ImageProcessor)
    processor.reso_group = _FakeResolutionGroup()
    return processor


def _joint_cond(*, ratio_index: int, ori_width: int, ori_height: int) -> JointImageInfo:
    vae = ImageInfo(
        image_type="vae",
        image_width=1024,
        image_height=1024,
        token_width=64,
        token_height=64,
        base_size=1024,
        ratio_index=ratio_index,
        ori_image_width=ori_width,
        ori_image_height=ori_height,
    )
    vit = ImageInfo(
        image_type="siglip2",
        image_width=1024,
        image_height=1024,
        token_width=64,
        token_height=64,
        image_token_length=4096,
        ori_image_width=ori_width,
        ori_image_height=ori_height,
    )
    return JointImageInfo(vae_image_info=vae, vision_image_info=vit)


def test_postprocess_single_matching_bucket_resizes_to_input_ratio_area():
    output = Image.new("RGB", (1024, 1024), color="white")
    cond = _joint_cond(ratio_index=0, ori_width=1200, ori_height=800)

    processed = _fake_processor().postprocess_outputs([output], [[cond]], infer_align_image_size=True)

    assert processed[0].size == (1254, 836)


def test_postprocess_keeps_output_when_disabled_or_bucket_mismatched():
    output = Image.new("RGB", (1024, 1024), color="white")
    cond = _joint_cond(ratio_index=2, ori_width=1200, ori_height=800)

    disabled = _fake_processor().postprocess_outputs([output], [[cond]], infer_align_image_size=False)
    mismatched = _fake_processor().postprocess_outputs([output], [[cond]], infer_align_image_size=True)

    assert disabled[0] is output
    assert mismatched[0] is output


def test_postprocess_multi_image_uses_first_matching_bucket_only():
    output = Image.new("RGB", (1024, 1024), color="white")
    mismatch = _joint_cond(ratio_index=2, ori_width=2048, ori_height=1024)
    match = _joint_cond(ratio_index=0, ori_width=1200, ori_height=800)

    processed = _fake_processor().postprocess_outputs([output], [[mismatch, match]], infer_align_image_size=True)

    assert processed[0].size == (1254, 836)


def test_postprocess_returns_outputs_when_batch_has_no_cond_info():
    outputs = [Image.new("RGB", (1024, 1024), color="white")]

    assert _fake_processor().postprocess_outputs(outputs, None, infer_align_image_size=True) is outputs
