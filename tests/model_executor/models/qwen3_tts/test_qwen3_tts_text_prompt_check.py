# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_FUSED_PIPELINE, QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
    check_text_prompt,
    has_tts_text_conditioning,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NO_TEXT = object()
_ERROR = "first item is non-blank"


def _prompt(text=_NO_TEXT, task_type="CustomVoice", **info):
    if text is not _NO_TEXT:
        info["text"] = text
    return {"prompt_token_ids": [0] * 8, "additional_information": {"task_type": [task_type], **info}}


_REJECTED = {
    "empty": _prompt([""]),
    "whitespace": _prompt(["   \t\n  "]),
    "blank-str": _prompt("  "),
    "empty-list": _prompt([]),
    "none": _prompt([None]),
    "not-a-list": _prompt("Hello."),
    "missing": _prompt(),
    "base-missing": _prompt(task_type="Base"),
    "none-text-ids": _prompt([""], **{PRECOMPUTED_TEXT_IDS_KEY: None}),
    "blank-trailing-hidden": _prompt([""], hidden_states={"trailing_text": None}),
    "plain-string": "Hello.",
    "text-prompt-dict": {"prompt": "Hello."},
}
_ACCEPTED = {
    "text": _prompt(["Hello."]),
    "padded-text": _prompt(["  Hello.  "]),
    "first-item-only": _prompt(["Hello.", ""]),
    # Precomputed text ids carry the text; the raw string is not used.
    "text-ids": _prompt([""], **{PRECOMPUTED_TEXT_IDS_KEY: [[1, 2, 3]]}),
    "trailing-hidden": _prompt([""], hidden_states={"trailing_text": torch.zeros(1)}),
    # AsyncOmni closes a streaming input with a token prompt that has no additional_information.
    "stream-close": {"prompt_token_ids": [0]},
}


@pytest.mark.parametrize("prompt", list(_REJECTED.values()), ids=list(_REJECTED))
def test_prompt_without_usable_text_is_rejected(prompt):
    with pytest.raises(ValueError, match=_ERROR):
        check_text_prompt(prompt, [])
    # One bad prompt in a list rejects the call.
    with pytest.raises(ValueError, match=_ERROR):
        check_text_prompt([_prompt(["Hello."]), prompt], [])


@pytest.mark.parametrize("prompt", list(_ACCEPTED.values()), ids=list(_ACCEPTED))
def test_prompt_with_usable_text_passes_through_unchanged(prompt):
    assert check_text_prompt(prompt, []) is prompt


_ACCEPTED_INFOS = {name: p["additional_information"] for name, p in _ACCEPTED.items() if "additional_information" in p}


@pytest.mark.parametrize("info", list(_ACCEPTED_INFOS.values()), ids=list(_ACCEPTED_INFOS))
def test_accepted_prompts_still_satisfy_the_talker_after_ipc(info):
    # The Talker checks the deserialized payload, so anything accepted here must pass there too.
    info = deserialize_additional_information(serialize_additional_information(info))
    assert has_tts_text_conditioning(info, info.get("hidden_states"))


def test_talker_check_keeps_accepting_blank_text():
    # Only the stage-0 check rejects blank text; the engine-side check is unchanged.
    assert has_tts_text_conditioning({"text": ["  "]})
    assert not has_tts_text_conditioning({"text": ["  "]}, allow_blank_text=False)
    assert not has_tts_text_conditioning({"text": []})


@pytest.mark.parametrize("pipeline", [QWEN3_TTS_PIPELINE, QWEN3_TTS_FUSED_PIPELINE])
def test_talker_stage_runs_the_check(pipeline):
    stage0 = next(stage for stage in pipeline.stages if stage.stage_id == 0)
    assert stage0.prompt_transform_func == (
        "vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder.check_text_prompt"
    )
