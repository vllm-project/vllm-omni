# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm_omni.model_executor.stage_input_processors.stage_bypass import (
    OMNI_SKIP_STAGES_KEY,
    make_mock_text_stage_output,
    should_skip_stage,
    should_skip_stage_from_info,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_should_skip_stage_reads_additional_information_flag():
    assert should_skip_stage({}, stage_id=0) is False
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0]}}, stage_id=0) is True
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0]}}, stage_id=1) is False
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: [0, 1]}}, stage_id=1) is True
    assert should_skip_stage({"additional_information": {OMNI_SKIP_STAGES_KEY: []}}, stage_id=0) is False


def test_should_skip_stage_from_info_matches_prompt_helper():
    info = {OMNI_SKIP_STAGES_KEY: [0]}
    prompt = {"additional_information": info}
    assert should_skip_stage_from_info(info, 0) is True
    assert should_skip_stage(prompt, 0) is True

    mock = make_mock_text_stage_output("req-skip")
    assert mock.request_id == "req-skip"
    assert mock.outputs[0].cumulative_text == ""
