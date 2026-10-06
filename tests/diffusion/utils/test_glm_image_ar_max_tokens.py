# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Tests for GLM-Image AR max_tokens budget helper."""

import pytest

from vllm_omni.diffusion.utils.param_utils import glm_image_ar_max_tokens
from vllm_omni.model_executor.stage_input_processors.glm_image import compute_max_tokens

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class TestGlmImageArMaxTokens:
    def test_matches_stage_processor_helper(self):
        for height, width, is_i2i in (
            (1024, 1024, False),
            (1024, 1024, True),
            (512, 768, False),
            (512, 768, True),
        ):
            assert glm_image_ar_max_tokens(height, width, is_i2i=is_i2i) == compute_max_tokens(
                height, width, is_i2i=is_i2i
            )

    def test_1024_t2i_budget(self):
        # small(256) + large(1024) + EOS
        assert glm_image_ar_max_tokens(1024, 1024, is_i2i=False) == 1281

    def test_invalid_dimensions(self):
        assert glm_image_ar_max_tokens(0, 1024) is None
        assert glm_image_ar_max_tokens(1024, 0) is None
