# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Registration tests for the native Mage-VL executor."""

import pytest

from vllm_omni.model_executor.models.registry import _OMNI_MODELS

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_mage_vl_architecture_registered() -> None:
    assert _OMNI_MODELS["MageVLForConditionalGeneration"] == (
        "mage_vl",
        "mage_vl",
        "MageVLForConditionalGeneration",
    )
