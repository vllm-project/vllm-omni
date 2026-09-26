# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for _build_add_request_message input validation."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from vllm.sampling_params import SamplingParams

from vllm_omni.engine.async_omni_engine import AsyncOmniEngine, StageRuntimeInfo

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_engine(num_stages: int = 3) -> AsyncOmniEngine:
    """Construct a minimal AsyncOmniEngine without __init__."""
    engine = object.__new__(AsyncOmniEngine)
    engine.num_stages = num_stages
    engine.default_sampling_params_list = [SamplingParams(max_tokens=8)] * num_stages
    engine.stage_metadata = [
        StageRuntimeInfo(final_output=i == num_stages - 1, final_output_type=None, stage_type="llm")
        for i in range(num_stages)
    ]
    engine.input_processor = Mock()
    engine.stage_pools = []
    return engine


class TestFinalStageIdValidation:
    def test_negative_final_stage_id_raises(self):
        engine = _make_engine(num_stages=3)
        with pytest.raises(ValueError, match="final_stage_id=-1 is out of range"):
            engine._build_add_request_message(
                request_id="req-1",
                prompt={"prompt_token_ids": [1]},
                final_stage_id=-1,
            )

    def test_final_stage_id_equals_num_stages_raises(self):
        engine = _make_engine(num_stages=2)
        with pytest.raises(ValueError, match="final_stage_id=2 is out of range"):
            engine._build_add_request_message(
                request_id="req-1",
                prompt={"prompt_token_ids": [1]},
                final_stage_id=2,
            )

    def test_final_stage_id_exceeds_num_stages_raises(self):
        engine = _make_engine(num_stages=2)
        with pytest.raises(ValueError, match="final_stage_id=10 is out of range"):
            engine._build_add_request_message(
                request_id="req-1",
                prompt={"prompt_token_ids": [1]},
                final_stage_id=10,
            )

    def test_valid_final_stage_id_zero_passes(self):
        engine = _make_engine(num_stages=2)
        engine.input_processor.process_inputs.return_value = Mock(
            request_id="req-1",
            prompt_token_ids=[1],
            additional_information=None,
            model_intermediate_buffer=None,
            reasoning_ended=None,
        )
        engine.supported_tasks = ("generate",)
        engine.prompt_transform_func = None
        engine.prompt_expand_func = None
        msg = engine._build_add_request_message(
            request_id="req-1",
            prompt={"prompt_token_ids": [1]},
            final_stage_id=0,
        )
        assert msg.final_stage_id == 0


class TestSamplingParamsListLengthValidation:
    def test_sampling_params_list_too_short_raises(self):
        engine = _make_engine(num_stages=3)
        with pytest.raises(ValueError, match="sampling_params_list has 1 entries"):
            engine._build_add_request_message(
                request_id="req-1",
                prompt={"prompt_token_ids": [1]},
                sampling_params_list=[SamplingParams(max_tokens=8)],
                final_stage_id=2,
            )

    def test_sampling_params_list_exact_length_passes(self):
        engine = _make_engine(num_stages=3)
        engine.input_processor.process_inputs.return_value = Mock(
            request_id="req-1",
            prompt_token_ids=[1],
            additional_information=None,
            model_intermediate_buffer=None,
            reasoning_ended=None,
        )
        engine.supported_tasks = ("generate",)
        engine.prompt_transform_func = None
        engine.prompt_expand_func = None
        spl = [SamplingParams(max_tokens=8)] * 3
        msg = engine._build_add_request_message(
            request_id="req-1",
            prompt={"prompt_token_ids": [1]},
            sampling_params_list=spl,
            final_stage_id=2,
        )
        assert len(msg.sampling_params_list) == 3

    def test_empty_sampling_params_list_raises(self):
        engine = _make_engine(num_stages=2)
        engine.default_sampling_params_list = []
        with pytest.raises(ValueError, match="Missing sampling params"):
            engine._build_add_request_message(
                request_id="req-1",
                prompt={"prompt_token_ids": [1]},
                final_stage_id=0,
            )
