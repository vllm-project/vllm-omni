# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from vllm import SamplingParams

from vllm_omni.model_executor.models.glm_image.prompt import propagate_target_size

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_target_size_is_per_request():
    shared = SamplingParams(extra_args={"other": 1})
    first = [shared]
    second = [shared]
    prompt = {"prompt": "apple", "mm_processor_kwargs": {"target_h": 768, "target_w": 1536}}

    assert propagate_target_size(prompt, first) is prompt
    assert first[0].extra_args == {"other": 1, "target_h": 768, "target_w": 1536}
    assert second[0].extra_args == {"other": 1}


def test_mixed_target_pair_falls_back_pairwise():
    # A half-specified target pair must not mix dimensions across pairs: the
    # processor resolves 768x512-style input to the 512x512 height/width pair,
    # and the propagated extra_args must follow the same grid.
    params = [SamplingParams()]
    prompt = {"prompt": "apple", "mm_processor_kwargs": {"target_h": 768, "height": 512, "width": 512}}

    propagate_target_size(prompt, params)
    assert params[0].extra_args == {"target_h": 512, "target_w": 512}


def test_missing_sizes_default_to_1024_pair():
    params = [SamplingParams()]
    propagate_target_size({"prompt": "apple"}, params)
    assert params[0].extra_args == {"target_h": 1024, "target_w": 1024}
