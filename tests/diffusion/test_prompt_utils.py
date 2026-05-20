# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.prompt_utils import (
    extract_batch_prompts,
    has_negative_prompt,
    normalize_omni_diffusion_prompts,
    normalize_prompt_entry,
)
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_none_negative_stripped():
    result = normalize_prompt_entry({"prompt": "a cat", "negative_prompt": None})
    assert result == {"prompt": "a cat"}
    assert "negative_prompt" not in result


def test_empty_negative_stripped():
    result = normalize_prompt_entry({"prompt": "a cat", "negative_prompt": "  "})
    assert result == {"prompt": "a cat"}


def test_valid_negative_kept():
    result = normalize_prompt_entry({"prompt": "a cat", "negative_prompt": "blurry"})
    assert result == {"prompt": "a cat", "negative_prompt": "blurry"}


def test_string_prompt_stripped():
    assert normalize_prompt_entry("  hello  ") == "hello"


def test_preserves_multimodal_keys():
    result = normalize_prompt_entry(
        {
            "prompt": "edit",
            "negative_prompt": None,
            "multi_modal_data": {"image": "placeholder"},
        }
    )
    assert result["multi_modal_data"] == {"image": "placeholder"}
    assert "negative_prompt" not in result


def test_invalid_type_raises():
    with pytest.raises(TypeError, match="str or dict"):
        normalize_prompt_entry(123)


def test_batch_extract_all_none():
    prompts = normalize_omni_diffusion_prompts([{"prompt": "cat", "negative_prompt": None}])
    prompt, negative = extract_batch_prompts(prompts)
    assert prompt == ["cat"]
    assert negative is None


def test_batch_extract_mixed():
    prompt, negative = extract_batch_prompts(
        [
            {"prompt": "a cat", "negative_prompt": "blurry"},
            {"prompt": "a dog"},
            "plain",
        ]
    )
    assert prompt == ["a cat", "a dog", "plain"]
    assert negative == ["blurry", "", ""]


def test_has_negative_prompt():
    assert not has_negative_prompt([{"prompt": "x"}])
    assert has_negative_prompt([{"prompt": "x", "negative_prompt": "blur"}])


def test_request_post_init_strips_none_negative():
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "cat", "negative_prompt": None}],
        sampling_params=OmniDiffusionSamplingParams(guidance_scale=3.5),
    )
    assert req.prompts[0] == {"prompt": "cat"}
    assert req.sampling_params.do_classifier_free_guidance is False


def test_request_post_init_sets_cfg_with_negative():
    req = OmniDiffusionRequest(
        prompts=[{"prompt": "cat", "negative_prompt": "blurry"}],
        sampling_params=OmniDiffusionSamplingParams(guidance_scale=3.5),
    )
    assert req.sampling_params.do_classifier_free_guidance is True
