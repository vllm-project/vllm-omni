# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Qwen3-TTS and VoxCPM2 configs must not call deprecated transformers validators."""

from __future__ import annotations

import logging
import warnings
from logging.handlers import BufferingHandler

import pytest

from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSTalkerCodePredictorConfig
from vllm_omni.transformers_utils.configs.voxcpm2 import VoxCPM2Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_LONGROPE = {
    "type": "longrope",
    "short_factor": [1.0] * 64,
    "long_factor": [1.0] * 64,
    "original_max_position_embeddings": 32768,
}


@pytest.fixture
def transformers_warnings():
    # transformers' library logger does not propagate to the root logger, so caplog misses it.
    handler = BufferingHandler(capacity=1000)
    logger = logging.getLogger("transformers")
    old_level = logger.level
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)
    yield handler.buffer
    logger.removeHandler(handler)
    logger.setLevel(old_level)


@pytest.mark.parametrize(
    ("build", "rope_type"),
    [
        (Qwen3TTSTalkerCodePredictorConfig, "default"),
        (lambda: VoxCPM2Config(rope_scaling=dict(_LONGROPE)), "longrope"),
    ],
    ids=["qwen3_tts_code_predictor", "voxcpm2_longrope"],
)
def test_config_init_does_not_use_deprecated_validators(build, rope_type, transformers_warnings):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        config = build()

    assert config.rope_parameters["rope_type"] == rope_type
    assert not [r.getMessage() for r in transformers_warnings if "deprecated" in r.getMessage()]
