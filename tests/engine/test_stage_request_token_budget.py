# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token-budget clamping when the orchestrator builds a stage request."""

from __future__ import annotations

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.orchestrator import build_engine_core_request_from_tokens

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StageModelConfig:
    """Only the field ``build_engine_core_request_from_tokens`` reads."""

    max_model_len = 4096


class TestStageRequestTokenBudget:
    def test_min_tokens_is_clamped_to_the_remaining_context(self):
        # A min_tokens wider than the leftover window makes ``check_stop``
        # return early forever: the request can neither stop nor be granted
        # more tokens, so it stalls instead of being length-capped.
        params = SamplingParams(max_tokens=4096, min_tokens=50)

        request = build_engine_core_request_from_tokens(
            request_id="req-clamped",
            prompt={"prompt_token_ids": [0] * 4050},
            params=params,
            model_config=_StageModelConfig(),
        )

        assert request.sampling_params.min_tokens == 46
        assert params.min_tokens == 50

    def test_min_tokens_survives_when_the_prompt_leaves_room(self):
        params = SamplingParams(max_tokens=4096, min_tokens=50)

        request = build_engine_core_request_from_tokens(
            request_id="req-untouched",
            prompt={"prompt_token_ids": [0] * 68},
            params=params,
            model_config=_StageModelConfig(),
        )

        assert request.sampling_params.min_tokens == 50
        assert request.sampling_params.max_tokens == 4096

    def test_min_tokens_floors_at_zero_when_the_prompt_fills_the_context(self):
        params = SamplingParams(max_tokens=16, min_tokens=8)

        request = build_engine_core_request_from_tokens(
            request_id="req-full",
            prompt={"prompt_token_ids": [0] * 4096},
            params=params,
            model_config=_StageModelConfig(),
        )

        assert request.sampling_params.min_tokens == 0

    def test_absent_max_tokens_still_defaults_to_the_remaining_context(self):
        params = SamplingParams(min_tokens=0)
        params.max_tokens = None

        request = build_engine_core_request_from_tokens(
            request_id="req-default",
            prompt={"prompt_token_ids": [0] * 96},
            params=params,
            model_config=_StageModelConfig(),
        )

        assert request.sampling_params.max_tokens == 4000
