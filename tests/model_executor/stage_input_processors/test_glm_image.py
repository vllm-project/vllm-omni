# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for GLM-Image stage input processor."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.glm_image import (
    _first_source_image,
    _has_source_image,
    _parse_generated_tokens,
    _upsample_token_ids,
    ar2diffusion,
    compute_max_tokens,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# =============================================================================
# Helpers
# =============================================================================


def _source_output(token_ids: list[int], mm_output: dict | None = None):
    """Create a minimal AR output mock."""
    return SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=token_ids)],
        multimodal_output=mm_output,
    )


def _stage_with_outputs(outputs):
    """Create a stage list entry with engine_outputs."""
    return SimpleNamespace(engine_outputs=outputs)


# =============================================================================
# Tests for _has_source_image
# =============================================================================


class TestHasSourceImage:
    def test_none_input(self):
        assert _has_source_image(None) is False

    def test_non_dict_input(self):
        assert _has_source_image("not_a_dict") is False

    def test_empty_dict(self):
        assert _has_source_image({}) is False

    def test_image_key_present(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        assert _has_source_image({"image": img}) is True

    def test_image_key_none(self):
        assert _has_source_image({"image": None}) is False

    def test_img2img_key_present(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        assert _has_source_image({"img2img": img}) is True

    def test_images_key_list(self):
        from PIL import Image

        imgs = [Image.new("RGB", (64, 64))]
        assert _has_source_image({"images": imgs}) is True

    def test_images_key_empty_list(self):
        assert _has_source_image({"images": []}) is False

    def test_images_key_single(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        assert _has_source_image({"images": img}) is True


# =============================================================================
# Tests for _first_source_image
# =============================================================================


class TestFirstSourceImage:
    def test_none_input(self):
        assert _first_source_image(None) is None

    def test_non_dict_input(self):
        assert _first_source_image("not_a_dict") is None

    def test_image_key_single(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        assert _first_source_image({"image": img}) is img

    def test_image_key_list(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        assert _first_source_image({"image": [img]}) is img

    def test_image_key_empty_list(self):
        assert _first_source_image({"image": []}) is None

    def test_img2img_key_single(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        assert _first_source_image({"img2img": img}) is img

    def test_images_key_list(self):
        from PIL import Image

        imgs = [Image.new("RGB", (64, 64))]
        assert _first_source_image({"images": imgs}) is imgs[0]

    def test_images_key_empty_list(self):
        assert _first_source_image({"images": []}) is None

    def test_images_key_single_not_list(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        assert _first_source_image({"images": img}) is img


# =============================================================================
# Tests for compute_max_tokens
# =============================================================================


class TestComputeMaxTokens:
    def test_t2i_1024x1024(self):
        # t2i: small_tokens + large_tokens + 1 (EOS)
        # token_h = 1024/32 = 32, token_w = 1024/32 = 32
        # large = 32*32 = 1024
        # ratio = 1.0, small_h = sqrt(1)*16 = 16, small_w = sqrt(1)*16 = 16, small = 256
        # total = 256 + 1024 + 1 = 1281
        result = compute_max_tokens(1024, 1024, is_i2i=False)
        assert result == 1281

    def test_i2i_1024x1024(self):
        # i2i: large_tokens + 1 (EOS)
        # large = 32*32 = 1024, total = 1025
        result = compute_max_tokens(1024, 1024, is_i2i=True)
        assert result == 1025

    def test_t2i_512x512(self):
        # token_h = 16, token_w = 16, large = 256
        # ratio = 1.0, small_h = 16, small_w = 16, small = 256
        # total = 256 + 256 + 1 = 513
        result = compute_max_tokens(512, 512, is_i2i=False)
        assert result == 513

    def test_i2i_512x512(self):
        # large = 256, total = 257
        result = compute_max_tokens(512, 512, is_i2i=True)
        assert result == 257

    def test_non_square_t2i(self):
        # 1024x512: token_h=32, token_w=16, large=512
        # ratio = 32/16 = 2.0
        # small_h = max(1, int(sqrt(2)*16)) = 22, small_w = max(1, int(sqrt(0.5)*16)) = 11
        # small = 22*11 = 242
        # total = 242 + 512 + 1 = 755
        result = compute_max_tokens(1024, 512, is_i2i=False)
        assert result == 242 + 512 + 1

    def test_custom_factor(self):
        # factor=16, 512x512: token_h=32, token_w=32, large=1024
        # ratio=1.0, small_h=8, small_w=8, small=64
        # total = 64 + 1024 + 1 = 1089
        result = compute_max_tokens(512, 512, factor=16, is_i2i=False)
        assert result == 1089

    def test_i2i_smaller_than_t2i(self):
        t2i = compute_max_tokens(1024, 1024, is_i2i=False)
        i2i = compute_max_tokens(1024, 1024, is_i2i=True)
        assert i2i < t2i


# =============================================================================
# Tests for _upsample_token_ids
# =============================================================================


class TestUpsampleTokenIds:
    def test_2x2_to_4x4(self):
        tokens = torch.tensor([1, 2, 3, 4])
        result = _upsample_token_ids(tokens, 2, 2)
        assert result.shape == (16,)  # 4 * 4 = 16 (2x each dim)

    def test_1x1_to_2x2(self):
        tokens = torch.tensor([7])
        result = _upsample_token_ids(tokens, 1, 1)
        assert result.shape == (4,)  # 2 * 2
        assert (result == 7).all()

    def test_4x4_to_8x8(self):
        tokens = torch.arange(16, dtype=torch.long)
        result = _upsample_token_ids(tokens, 4, 4)
        assert result.shape == (64,)

    def test_preserves_dtype(self):
        tokens = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        result = _upsample_token_ids(tokens, 2, 2)
        assert result.dtype == torch.long


# =============================================================================
# Tests for _parse_generated_tokens
# =============================================================================


class TestParseGeneratedTokens:
    def test_t2i_standard(self):
        # 1024x1024, t2i: small(256) + large(1024) + EOS
        # Generate 256 + 1024 + 1 = 1281 tokens, last is EOS (16385)
        large_tokens = list(range(1024))
        small_tokens = list(range(1000, 1256))
        eos = [16385]
        token_ids = small_tokens + large_tokens + eos

        prior, h, w = _parse_generated_tokens(token_ids, 1024, 1024, is_i2i=False)
        assert h == 1024
        assert w == 1024
        # Prior tokens should be upsampled: 1024 tokens -> 4*1024 = 4096
        assert prior.shape[0] == 1024 * 4

    def test_i2i_standard(self):
        # 1024x1024, i2i: large(1024) + EOS
        large_tokens = list(range(1024))
        eos = [16385]
        token_ids = large_tokens + eos

        prior, h, w = _parse_generated_tokens(token_ids, 1024, 1024, is_i2i=True)
        assert h == 1024
        assert w == 1024
        assert prior.shape[0] == 1024 * 4

    def test_i2i_without_eos(self):
        # i2i without EOS marker
        large_tokens = list(range(1024))
        prior, h, w = _parse_generated_tokens(large_tokens, 1024, 1024, is_i2i=True)
        assert h == 1024
        assert w == 1024

    def test_i2i_too_few_tokens_raises(self):
        with pytest.raises(ValueError, match="i2i token parse failed"):
            _parse_generated_tokens([1, 2, 3], 1024, 1024, is_i2i=True)

    def test_t2i_too_few_tokens_raises(self):
        # Only large tokens, no small preview
        large_tokens = list(range(1024))
        with pytest.raises(ValueError, match="t2i token parse failed"):
            _parse_generated_tokens(large_tokens, 1024, 1024, is_i2i=False)

    def test_i2i_t2i_style_layout_fallback(self):
        # i2i but got t2i-style (small + large) tokens
        small_tokens = list(range(256))
        large_tokens = list(range(1024))
        token_ids = small_tokens + large_tokens

        prior, h, w = _parse_generated_tokens(token_ids, 1024, 1024, is_i2i=True)
        # Should extract the large portion
        assert h == 1024
        assert w == 1024


# =============================================================================
# Tests for ar2diffusion
# =============================================================================


class TestAr2Diffusion:
    def test_basic_t2i(self):
        """Test basic text-to-image pipeline: AR -> Diffusion."""
        # 1024x1024 t2i: small(256) + large(1024) + EOS
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        stage_list = [_stage_with_outputs([_source_output(token_ids)])]

        prompt = {"prompt": "a cat", "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024}}

        result = ar2diffusion(stage_list, [0], prompt=[prompt])
        assert len(result) == 1
        assert result[0]["prompt"] == "a cat"
        assert result[0]["height"] == 1024
        assert result[0]["width"] == 1024
        assert "prior_token_ids" in result[0]["extra"]

    def test_i2i_with_mm_output(self):
        """Test image-to-image with prior_token_image_ids from AR model."""
        token_ids = list(range(1024)) + [16385]
        mm_output = {"prior_token_image_ids": torch.tensor([1, 2, 3])}
        stage_list = [_stage_with_outputs([_source_output(token_ids, mm_output)])]

        from PIL import Image

        img = Image.new("RGB", (64, 64))
        prompt = {
            "prompt": "edit this",
            "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024},
            "multi_modal_data": {"image": img},
        }

        result = ar2diffusion(stage_list, [0], prompt=[prompt])
        assert len(result) == 1
        assert result[0]["extra"]["prior_token_image_ids"] is not None

    def test_i2i_detected_via_modalities(self):
        """Test i2i mode detected via modalities field."""
        token_ids = list(range(1024)) + [16385]
        stage_list = [_stage_with_outputs([_source_output(token_ids)])]

        prompt = {
            "prompt": "edit this",
            "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024},
            "modalities": ["img2img"],
        }

        result = ar2diffusion(stage_list, [0], prompt=[prompt])
        assert len(result) == 1

    def test_empty_engine_input_source_raises(self):
        with pytest.raises(ValueError, match="engine_input_source cannot be empty"):
            ar2diffusion([], [], prompt={})

    def test_invalid_stage_id_raises(self):
        with pytest.raises(IndexError, match="Invalid stage_id"):
            ar2diffusion([_stage_with_outputs(None)], [5], prompt={})

    def test_no_outputs_raises(self):
        with pytest.raises(RuntimeError, match="has no outputs yet"):
            ar2diffusion([SimpleNamespace(engine_outputs=None)], [0], prompt={})

    def test_default_dimensions(self):
        """When no height/width in prompt, defaults to 1024x1024."""
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        stage_list = [_stage_with_outputs([_source_output(token_ids)])]

        prompt = {"prompt": "test"}
        result = ar2diffusion(stage_list, [0], prompt=[prompt])
        assert result[0]["height"] == 1024
        assert result[0]["width"] == 1024

    def test_requires_multimodal_data_with_pil_image(self):
        """Test that pil_image is included when requires_multimodal_data=True."""
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        stage_list = [_stage_with_outputs([_source_output(token_ids)])]

        from PIL import Image

        img = Image.new("RGB", (64, 64))
        prompt = {
            "prompt": "test",
            "multi_modal_data": {"image": img},
        }

        result = ar2diffusion(stage_list, [0], prompt=[prompt], requires_multimodal_data=True)
        assert result[0]["pil_image"] is img

    def test_extra_params_passed_through(self):
        """Test that seed, num_inference_steps, guidance_scale, negative_prompt are passed."""
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        stage_list = [_stage_with_outputs([_source_output(token_ids)])]

        prompt = {
            "prompt": "test",
            "seed": 42,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry",
        }

        result = ar2diffusion(stage_list, [0], prompt=[prompt])
        assert result[0]["seed"] == 42
        assert result[0]["num_inference_steps"] == 50
        assert result[0]["guidance_scale"] == 7.5
        assert result[0]["negative_prompt"] == "blurry"

    def test_batch_requests(self):
        """Test processing multiple requests in a batch."""
        tokens1 = list(range(256)) + list(range(1024)) + [16385]
        tokens2 = list(range(256)) + list(range(1024)) + [16385]
        stage_list = [_stage_with_outputs([_source_output(tokens1), _source_output(tokens2)])]

        prompts = [
            {"prompt": "first", "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024}},
            {"prompt": "second", "mm_processor_kwargs": {"target_h": 512, "target_w": 512}},
        ]

        result = ar2diffusion(stage_list, [0], prompt=prompts)
        assert len(result) == 2
        assert result[0]["prompt"] == "first"
        assert result[1]["prompt"] == "second"
