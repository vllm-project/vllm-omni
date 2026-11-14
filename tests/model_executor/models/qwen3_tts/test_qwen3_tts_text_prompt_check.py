# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest

from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_FUSED_PIPELINE, QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
    check_text_prompt,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _prompt(**info):
    return {"prompt_token_ids": [0] * 8, "additional_information": {"task_type": ["CustomVoice"], **info}}


@pytest.mark.parametrize("text", [[""], ["   \t\n  "], "  "])
def test_blank_text_is_rejected(text):
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        check_text_prompt(_prompt(text=text), [])
    # One blank prompt in a list rejects the call.
    with pytest.raises(ValueError, match="empty or whitespace-only"):
        check_text_prompt([_prompt(text=["Hello."]), _prompt(text=text)], [])


@pytest.mark.parametrize(
    "prompt",
    [
        _prompt(text=["Hello."]),
        _prompt(text=["  Hello.  "]),
        # Precomputed text ids carry the text; the raw string is not used.
        _prompt(text=[""], **{PRECOMPUTED_TEXT_IDS_KEY: [[1, 2, 3]]}),
        # No text key at all is left to the Talker's own check.
        _prompt(),
        # The streaming end-of-input update carries no additional_information.
        {"prompt_token_ids": [0]},
        "plain string prompt",
    ],
)
def test_other_prompts_pass_through_unchanged(prompt):
    assert check_text_prompt(prompt, []) is prompt


@pytest.mark.parametrize("pipeline", [QWEN3_TTS_PIPELINE, QWEN3_TTS_FUSED_PIPELINE])
def test_talker_stage_runs_the_check(pipeline):
    stage0 = next(stage for stage in pipeline.stages if stage.stage_id == 0)
    assert stage0.prompt_transform_func == (
        "vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder.check_text_prompt"
    )
