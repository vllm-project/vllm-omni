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
