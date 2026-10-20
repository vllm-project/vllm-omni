# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Drama 3 main-token candidates: end of speech and every semantic code but 0."""

import pytest

from vllm_omni.model_executor.models.fish_omni.configuration_fish_omni import FishOmniConfig
from vllm_omni.model_executor.models.fish_omni.fish_omni_model import main_candidate_ids

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_semantic_code_zero_is_never_a_candidate():
    candidates = main_candidate_ids(eos_token_id=5, semantic_begin_id=100, semantic_end_id=103)
    assert candidates == [5, 101, 102, 103]


def test_released_checkpoint_ids():
    config = FishOmniConfig()
    candidates = main_candidate_ids(config.eos_token_id, config.semantic_start_token_id, config.semantic_end_token_id)
    assert config.eos_token_id in candidates
    assert config.semantic_start_token_id not in candidates
    assert candidates[-1] == config.semantic_end_token_id
    assert len(candidates) == config.semantic_end_token_id - config.semantic_start_token_id + 1
