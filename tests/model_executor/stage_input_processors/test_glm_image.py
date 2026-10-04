# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for GLM-Image stage input processor."""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.stage_input_processors.glm_image import (
    AR_GRID_FACTOR,
    DEFAULT_TARGET_SIZE,
    _first_source_image,
    _has_source_image,
    _parse_generated_tokens,
    _resolve_target_size,
    _snap_to_ar_grid,
    _upsample_token_ids,
    ar2diffusion,
    compute_max_tokens,
    prepare_ar_prompt,
)

# The diffusion stage derives its expected prior-token sequence length by
# dividing each edge by `vae_scale_factor * patch_size`.
DIT_PATCH_PIXELS = 16

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


# =============================================================================
# Helpers
# =============================================================================


def _source_output(token_ids: list[int], mm_output: dict | None = None):
    """Create a minimal AR output mock."""
    return SimpleNamespace(
        outputs=[SimpleNamespace(token_ids=token_ids, cumulative_token_ids=token_ids)],
        multimodal_output=mm_output,
    )


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
        source_outputs = [_source_output(token_ids)]

        prompt = {"prompt": "a cat", "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024}}

        result = ar2diffusion(source_outputs, prompt=prompt)
        assert result["prompt"] == "a cat"
        assert result["height"] == 1024
        assert result["width"] == 1024
        assert "prior_token_ids" in result["extra"]

    def test_i2i_with_mm_output(self):
        """Test image-to-image with prior_token_image_ids from AR model."""
        token_ids = list(range(1024)) + [16385]
        mm_output = {"ids": {"prior_image": torch.tensor([1, 2, 3])}}
        source_outputs = [_source_output(token_ids, mm_output)]

        from PIL import Image

        img = Image.new("RGB", (64, 64))
        prompt = {
            "prompt": "edit this",
            "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024},
            "multi_modal_data": {"image": img},
        }

        result = ar2diffusion(source_outputs, prompt=prompt)
        assert result["extra"]["prior_token_image_ids"] is not None

    def test_i2i_detected_via_modalities(self):
        """Test i2i mode detected via modalities field."""
        token_ids = list(range(1024)) + [16385]
        source_outputs = [_source_output(token_ids)]

        prompt = {
            "prompt": "edit this",
            "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024},
            "modalities": ["img2img"],
        }

        result = ar2diffusion(source_outputs, prompt=prompt)
        assert result["prompt"] == "edit this"

    def test_empty_source_outputs_returns_none(self):
        assert ar2diffusion([], prompt=None) is None

    def test_default_dimensions(self):
        """When no height/width in prompt, defaults to 1024x1024."""
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        source_outputs = [_source_output(token_ids)]

        prompt = {"prompt": "test"}
        result = ar2diffusion(source_outputs, prompt=prompt)
        assert result["height"] == 1024
        assert result["width"] == 1024

    def test_requires_multimodal_data_with_pil_image(self):
        """Test that pil_image is included when requires_multimodal_data=True."""
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        source_outputs = [_source_output(token_ids)]

        from PIL import Image

        img = Image.new("RGB", (64, 64))
        prompt = {
            "prompt": "test",
            "multi_modal_data": {"image": img},
        }

        result = ar2diffusion(source_outputs, prompt=prompt, requires_multimodal_data=True)
        assert result["pil_image"] is img

    def test_extra_params_passed_through(self):
        """Test that seed, num_inference_steps, guidance_scale, negative_prompt are passed."""
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        source_outputs = [_source_output(token_ids)]

        prompt = {
            "prompt": "test",
            "seed": 42,
            "num_inference_steps": 50,
            "guidance_scale": 7.5,
            "negative_prompt": "blurry",
        }

        result = ar2diffusion(source_outputs, prompt=prompt)
        assert result["prompt"] == "test"
        assert result["seed"] == 42
        assert result["num_inference_steps"] == 50
        assert result["guidance_scale"] == 7.5
        assert result["negative_prompt"] == "blurry"

    def test_multiple_source_outputs_uses_first_payload_only(self):
        """Test the GLM bridge keeps a single diffusion payload for one request."""
        tokens1 = list(range(256)) + list(range(1024)) + [16385]
        tokens2 = [1, 2, 3]
        source_outputs = [_source_output(tokens1), _source_output(tokens2)]

        prompt = {"prompt": "first", "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024}}

        result = ar2diffusion(source_outputs, prompt=prompt)
        assert result["prompt"] == "first"
        assert result["height"] == 1024
        assert result["width"] == 1024

    def test_raw_string_prompt_keeps_its_text(self):
        """Offline ``Omni.generate("<text>")`` hands the bridge a bare ``str``.

        The orchestrator stores the *untransformed* prompt, so for an offline
        request ``prompt`` is the string itself. Dropping it would condition the
        diffusion stage on an empty caption.
        """
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        source_outputs = [_source_output(token_ids)]

        result = ar2diffusion(source_outputs, prompt="a red apple on a white table")
        assert result["prompt"] == "a red apple on a white table"

    def test_sampling_params_supply_the_size(self):
        """The diffusion stage's params size the token layout offline.

        ``prompt`` carries no size at all here, so without the fallback this
        would parse a 1024x1024 layout out of 512x512 worth of tokens and fail.
        """
        # 512x512 t2i: small(16x16=256) + large(16x16=256) + EOS
        token_ids = list(range(256)) + list(range(256)) + [16385]
        source_outputs = [_source_output(token_ids)]

        result = ar2diffusion(
            source_outputs,
            prompt="a cat",
            sampling_params=OmniDiffusionSamplingParams(height=512, width=512),
        )
        assert result["height"] == 512
        assert result["width"] == 512
        # 256 large tokens upsampled 2x in each dimension.
        assert len(result["extra"]["prior_token_ids"]) == 1024

    def test_prompt_size_wins_over_sampling_params(self):
        """An explicit request size is never overridden by the stage default."""
        token_ids = list(range(256)) + list(range(1024)) + [16385]
        source_outputs = [_source_output(token_ids)]

        result = ar2diffusion(
            source_outputs,
            prompt={"prompt": "a cat", "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024}},
            sampling_params=OmniDiffusionSamplingParams(height=512, width=512),
        )
        assert result["height"] == 1024
        assert result["width"] == 1024

    def test_sampling_params_parameter_name_is_load_bearing(self):
        """The orchestrator discovers this kwarg by name via ``inspect``.

        ``orchestrator._forward_to_next_stage_unguarded`` only passes the stage
        params when the probe finds a parameter called exactly
        ``sampling_params``, so a rename would silently disable the fallback.
        """
        assert "sampling_params" in inspect.signature(ar2diffusion).parameters


# =============================================================================
# Tests for prepare_ar_prompt (stage-0 prompt transform)
# =============================================================================


class TestPrepareArPrompt:
    def test_string_prompt_gets_target_size_from_sampling_params(self):
        """A raw offline prompt is what used to miss the AR grid scaffold."""
        result = prepare_ar_prompt(
            "a red apple on a white table",
            [SimpleNamespace(), OmniDiffusionSamplingParams(height=512, width=768)],
        )
        assert result["prompt"] == "a red apple on a white table"
        assert result["mm_processor_kwargs"] == {"target_h": 512, "target_w": 768}

    def test_serving_layer_kwargs_take_precedence(self):
        """The served path already resolved the size; leave it alone."""
        result = prepare_ar_prompt(
            {"prompt": "a cat", "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024}},
            [SimpleNamespace(), OmniDiffusionSamplingParams(height=512, width=512)],
        )
        assert result["mm_processor_kwargs"]["target_h"] == 1024
        assert result["mm_processor_kwargs"]["target_w"] == 1024

    def test_top_level_height_width_used_before_sampling_params(self):
        result = prepare_ar_prompt(
            {"prompt": "a cat", "height": 640, "width": 640},
            [SimpleNamespace(), OmniDiffusionSamplingParams(height=512, width=512)],
        )
        assert result["mm_processor_kwargs"] == {"target_h": 640, "target_w": 640}

    def test_unset_sizes_fall_back_to_default(self):
        result = prepare_ar_prompt("a cat", [SimpleNamespace(), OmniDiffusionSamplingParams()])
        assert result["mm_processor_kwargs"] == {
            "target_h": DEFAULT_TARGET_SIZE,
            "target_w": DEFAULT_TARGET_SIZE,
        }

    def test_missing_diffusion_params_fall_back_to_default(self):
        """No single diffusion stage to read a size from — keep the old default."""
        result = prepare_ar_prompt("a cat", [SimpleNamespace()])
        assert result["mm_processor_kwargs"] == {
            "target_h": DEFAULT_TARGET_SIZE,
            "target_w": DEFAULT_TARGET_SIZE,
        }

    def test_other_prompt_fields_are_preserved(self):
        from PIL import Image

        img = Image.new("RGB", (64, 64))
        result = prepare_ar_prompt(
            {"prompt": "edit this", "multi_modal_data": {"image": img}, "negative_prompt": "blurry"},
            [SimpleNamespace(), OmniDiffusionSamplingParams(height=512, width=512)],
        )
        assert result["multi_modal_data"] == {"image": img}
        assert result["negative_prompt"] == "blurry"
        assert result["mm_processor_kwargs"] == {"target_h": 512, "target_w": 512}

    def test_does_not_mutate_the_caller_prompt(self):
        """Downstream stages receive the original prompt; it must stay pristine."""
        prompt = {"prompt": "a cat"}
        prepare_ar_prompt(prompt, [SimpleNamespace(), OmniDiffusionSamplingParams(height=512, width=512)])
        assert prompt == {"prompt": "a cat"}

    def test_embeds_prompt_passes_through_untouched(self):
        """This caller already decided the AR stage's input."""
        prompt = {"prompt_embeds": "embeds"}
        assert prepare_ar_prompt(prompt, [OmniDiffusionSamplingParams(height=512, width=512)]) is prompt

    def test_token_prompt_is_stamped(self):
        """Safe to stamp: the scaffold append is guarded by a suffix check, so a
        caller that built its own scaffold keeps it (``glm_image_ar.py`` ``apply``).
        """
        result = prepare_ar_prompt(
            {"prompt_token_ids": [1, 2, 3]},
            [SimpleNamespace(), OmniDiffusionSamplingParams(height=512, width=512)],
        )
        assert result["mm_processor_kwargs"] == {"target_h": 512, "target_w": 512}

    def test_non_prompt_object_passes_through_unchanged(self):
        sentinel = SimpleNamespace(already_built=True)
        assert prepare_ar_prompt(sentinel, [OmniDiffusionSamplingParams()]) is sentinel

    def test_list_prompt_passes_through_unchanged(self):
        """A list of prompts is not this hook's to rewrite."""
        prompt = ["a cat", "a dog"]
        assert prepare_ar_prompt(prompt, [OmniDiffusionSamplingParams(height=512, width=512)]) is prompt

    def test_transformed_prompt_clears_the_multimodal_routing_gate(self):
        """This gate is the defect: only prompts with ``mm_processor_kwargs``
        reach ``GlmImageMultiModalProcessor``, so before the transform a raw
        offline prompt bypassed it and the AR stage saw plain text.
        """
        from vllm_omni.inputs.preprocess import OmniRenderer

        raw = "a red apple on a white table"
        assert OmniRenderer._routes_no_media_kwargs({"prompt": raw, "prompt_token_ids": [1, 2, 3]}) is False

        transformed = prepare_ar_prompt(raw, [SimpleNamespace(), OmniDiffusionSamplingParams(height=512, width=512)])
        # prompt_token_ids is filled in by upstream tokenization before the
        # renderer sees the prompt.
        transformed["prompt_token_ids"] = [1, 2, 3]
        assert OmniRenderer._routes_no_media_kwargs(transformed) is True

    def test_agrees_with_ar2diffusion_size_resolution(self):
        """The two sites must resolve the same size, per dimension.

        ``prepare_ar_prompt`` picks the grid the AR stage generates and
        ``ar2diffusion`` re-derives it to slice the prior tokens back out. A
        disagreement would slice at the wrong offsets without raising.
        """
        diffusion_params = OmniDiffusionSamplingParams(height=512, width=768)
        prompts = [
            "a cat",
            {"prompt": "a cat"},
            {"prompt": "a cat", "height": 640, "width": 896},
            {"prompt": "a cat", "mm_processor_kwargs": {"target_h": 1024, "target_w": 1024}},
        ]
        for prompt in prompts:
            transformed = prepare_ar_prompt(prompt, [SimpleNamespace(), diffusion_params])
            ar_size = (
                transformed["mm_processor_kwargs"]["target_h"],
                transformed["mm_processor_kwargs"]["target_w"],
            )
            # ar2diffusion sees the *untransformed* prompt plus the stage params.
            bridge_size = _resolve_target_size(
                prompt if isinstance(prompt, dict) else {"prompt": prompt},
                diffusion_params,
            )
            assert ar_size == bridge_size


# =============================================================================
# Tests for the stage-0 registration
# =============================================================================


class TestPipelineRegistration:
    def test_stage0_registers_the_prompt_transform(self):
        """Without this wiring the offline path never gets target_h/target_w."""
        from vllm_omni.model_executor.models.glm_image.pipeline import GLM_IMAGE_PIPELINE

        path = GLM_IMAGE_PIPELINE.stages[0].prompt_transform_func
        assert path is not None
        # Resolved the same way stage_init_utils resolves the hook.
        module_path, fn_name = path.rsplit(".", 1)
        assert getattr(importlib.import_module(module_path), fn_name) is prepare_ar_prompt


# =============================================================================
# Tests for AR-grid snapping (sizes that are not a multiple of 32)
# =============================================================================


class TestSnapToArGrid:
    def test_multiple_of_the_factor_is_untouched(self):
        assert _snap_to_ar_grid(1024) == 1024
        assert _snap_to_ar_grid(1312) == 1312

    def test_floors_onto_the_grid(self):
        # 1328 is a multiple of 16 but not of 32 -- the case that used to reach
        # the DiT as 1328 while the AR had already committed to 1312.
        assert _snap_to_ar_grid(1328) == 1312

    def test_never_returns_zero(self):
        """A sub-cell request still has to name one whole cell."""
        assert _snap_to_ar_grid(1) == AR_GRID_FACTOR
        assert _snap_to_ar_grid(AR_GRID_FACTOR - 1) == AR_GRID_FACTOR

    def test_resolve_target_size_snaps(self):
        height, width = _resolve_target_size({}, OmniDiffusionSamplingParams(height=1328, width=1328))
        assert (height, width) == (1312, 1312)

    def test_resolve_target_size_snaps_serving_kwargs(self):
        height, width = _resolve_target_size({"mm_processor_kwargs": {"target_h": 1328, "target_w": 720}})
        assert (height, width) == (1312, 704)

    def test_prepare_ar_prompt_stamps_the_snapped_size(self):
        prompt = prepare_ar_prompt("a cat", [None, OmniDiffusionSamplingParams(height=1328, width=1328)])
        assert prompt["mm_processor_kwargs"] == {"target_h": 1312, "target_w": 1312}


class TestParseGeneratedTokensReportsEffectiveSize:
    """The size handed onward must describe the grid, not the original request."""

    @staticmethod
    def _t2i_stream(token_h: int, token_w: int, *, eos: bool = True) -> list[int]:
        small = [7] * (16 * 16)  # square target -> 16x16 preview
        large = list(range(token_h * token_w))
        return small + large + ([16385] if eos else [])

    def test_non_multiple_of_factor_reports_floored_size(self):
        # 1328 floors to a 41x41 grid == 1312px.
        prior, h, w = _parse_generated_tokens(self._t2i_stream(41, 41), 1328, 1328)
        assert (h, w) == (1312, 1312)
        assert prior.shape[-1] == 82 * 82

    def test_reported_size_satisfies_the_dit_length_check(self):
        """This equality is exactly what pipeline_glm_image.py asserts."""
        for requested in (1024, 1312, 1328, 1344):
            token_edge = requested // AR_GRID_FACTOR
            prior, h, w = _parse_generated_tokens(self._t2i_stream(token_edge, token_edge), requested, requested)
            expected = (h // DIT_PATCH_PIXELS) * (w // DIT_PATCH_PIXELS)
            assert prior.shape[-1] == expected, requested

    def test_multiple_of_factor_is_unchanged(self):
        _, h, w = _parse_generated_tokens(self._t2i_stream(32, 32), 1024, 1024)
        assert (h, w) == (1024, 1024)

    def test_unterminated_overrun_raises(self):
        """max_tokens truncation must not be sliced into a plausible-looking image."""
        runaway = self._t2i_stream(32, 32, eos=False) + [11] * 2000
        with pytest.raises(ValueError, match="never emitted EOS"):
            _parse_generated_tokens(runaway, 1024, 1024)

    def test_terminated_overrun_raises(self):
        """A stream that stopped cleanly but is too long described a different grid.

        Observed at 768x768, where the AR emits 1120 tokens plus EOS although the
        scaffolded 16x16 preview + 24x24 target only accounts for 832. The extra
        tokens mean the offsets below do not delimit the image, so slicing them
        would hand the diffusion stage a confidently wrong picture.
        """
        overrun = [7] * (16 * 16) + list(range(24 * 24)) + [11] * 288 + [16385]
        with pytest.raises(ValueError, match="different grid"):
            _parse_generated_tokens(overrun, 768, 768)

    def test_exact_length_without_eos_is_still_accepted(self):
        """Some configs strip the stop token; only an overrun is a failure."""
        _, h, w = _parse_generated_tokens(self._t2i_stream(32, 32, eos=False), 1024, 1024)
        assert (h, w) == (1024, 1024)

    def test_i2i_t2i_style_layout_still_accepted_without_eos(self):
        stream = [7] * (16 * 16) + list(range(32 * 32))
        prior, h, w = _parse_generated_tokens(stream, 1024, 1024, is_i2i=True)
        assert (h, w) == (1024, 1024)
        assert prior.shape[-1] == 64 * 64
