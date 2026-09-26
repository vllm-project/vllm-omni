# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.diffusion.models.dreamzero.utils import (
    DEFAULT_CFG_SCALE,
    DEFAULT_EMBODIMENT_NAME_TO_ID,
    DEFAULT_NEGATIVE_PROMPT,
    DEFAULT_NUM_INFERENCE_STEPS,
    DEFAULT_SEED,
    DEFAULT_SIGMA_SHIFT,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_dreamzero_default_constants_match_source_baseline():
    assert DEFAULT_NUM_INFERENCE_STEPS == 16
    assert DEFAULT_CFG_SCALE == 5.0
    assert DEFAULT_SIGMA_SHIFT == 5.0
    assert DEFAULT_SEED == 1140
    assert "worst quality" in DEFAULT_NEGATIVE_PROMPT
    assert DEFAULT_EMBODIMENT_NAME_TO_ID["oxe_droid"] == 17
