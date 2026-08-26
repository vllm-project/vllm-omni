# SPDX-License-Identifier: Apache-2.0
"""Regression tests for HunyuanImage3 image-edit prompt normalization."""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_hunyuan_image3_img2img_prompt_gets_it2i_template():
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
        _maybe_build_hunyuan_image3_it2i_prompt,
    )

    prompt = _maybe_build_hunyuan_image3_it2i_prompt(
        "将背景更改为森林",
        num_images=1,
        mm_kwargs={"modalities": ["img2img"]},
    )

    assert "<|startoftext|>" in prompt
    assert "User: <img>将背景更改为森林" in prompt
    assert prompt.endswith("\n\nAssistant: <think>")


def test_hunyuan_image3_prompt_with_existing_placeholder_is_unchanged():
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
        _maybe_build_hunyuan_image3_it2i_prompt,
    )

    prompt = "User: <img>将背景更改为森林\n\nAssistant: <think>"

    assert (
        _maybe_build_hunyuan_image3_it2i_prompt(
            prompt,
            num_images=1,
            mm_kwargs={"modalities": ["img2img"]},
        )
        == prompt
    )


def test_hunyuan_image3_image_modality_does_not_force_it2i_template():
    from vllm_omni.model_executor.models.hunyuan_image3.hunyuan_image3 import (
        _maybe_build_hunyuan_image3_it2i_prompt,
    )

    assert (
        _maybe_build_hunyuan_image3_it2i_prompt(
            "生成一张森林图片",
            num_images=1,
            mm_kwargs={"modalities": ["image"]},
        )
        == "生成一张森林图片"
    )
