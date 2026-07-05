# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_moss_tts_delay_alias_points_to_local_module():
    assert _OMNI_MODELS["moss_tts_delay"] == (
        "moss_tts",
        "moss_tts_local",
        "MossTTSForConditionalGeneration",
    )
