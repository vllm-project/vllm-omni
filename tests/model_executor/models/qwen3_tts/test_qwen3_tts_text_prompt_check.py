# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.core.sched.output import OmniNewRequestData
from vllm_omni.engine.serialization import serialize_additional_information
from vllm_omni.model_executor.models.qwen3_tts.pipeline import QWEN3_TTS_FUSED_PIPELINE, QWEN3_TTS_PIPELINE
from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import (
    PRECOMPUTED_TEXT_IDS_KEY,
    check_text_prompt,
    check_text_update,
    has_tts_text_conditioning,
)
from vllm_omni.worker.gpu_model_runner import OmniGPUModelRunner
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_NO_TEXT = object()
_ERROR = "first item is non-blank"


def _prompt(text=_NO_TEXT, task_type="CustomVoice", buffer=None, **info):
    if text is not _NO_TEXT:
        info["text"] = text
    prompt = {"prompt_token_ids": [0] * 8, "additional_information": {"task_type": [task_type], **info}}
    if buffer is not None:
        prompt["model_intermediate_buffer"] = buffer
    return prompt


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
    # Ids that coerce_token_ids discards; the Talker would tokenize the blank text instead.
    "empty-text-ids": _prompt(["   "], **{PRECOMPUTED_TEXT_IDS_KEY: []}),
    "nested-empty-text-ids": _prompt(["   "], **{PRECOMPUTED_TEXT_IDS_KEY: [[]]}),
    "empty-tensor-text-ids": _prompt(["   "], **{PRECOMPUTED_TEXT_IDS_KEY: torch.zeros(0, dtype=torch.long)}),
    "blank-trailing-hidden": _prompt([""], hidden_states={"trailing_text": None}),
    # The Talker writes trailing_text itself at first prefill; a client copy carries no text.
    "trailing-hidden": _prompt([""], hidden_states={"trailing_text": torch.zeros(1)}),
    # Model Runner V2 lays model_intermediate_buffer over additional_information.
    "buffer-blank-text": _prompt(["Hello."], buffer={"text": [""]}),
    "buffer-only-blank-text": _prompt(buffer={"text": ["  "]}),
    "buffer-none-text-ids": _prompt([""], buffer={PRECOMPUTED_TEXT_IDS_KEY: None}, **{PRECOMPUTED_TEXT_IDS_KEY: [[1]]}),
    "plain-string": "Hello.",
    "text-prompt-dict": {"prompt": "Hello."},
    "token-only": {"prompt_token_ids": [0]},
    # The engine stamps a request id into every dict prompt before this check runs.
    "request-id-only": {"prompt_token_ids": [0], "additional_information": {"global_request_id": ["r"]}},
    "non-dict-info": {"prompt_token_ids": [0], "additional_information": "Hello."},
}
_ACCEPTED = {
    "text": _prompt(["Hello."]),
    "padded-text": _prompt(["  Hello.  "]),
    "first-item-only": _prompt(["Hello.", ""]),
    # Precomputed text ids carry the text; the raw string is not used.
    "text-ids": _prompt([""], **{PRECOMPUTED_TEXT_IDS_KEY: [[1, 2, 3]]}),
    "tensor-text-ids": _prompt([""], **{PRECOMPUTED_TEXT_IDS_KEY: torch.tensor([[1, 2, 3]])}),
    "buffer-text": _prompt(buffer={"text": ["Hello."]}),
    "buffer-text-ids": _prompt(["Hello."], buffer={PRECOMPUTED_TEXT_IDS_KEY: [[1, 2, 3]]}),
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


def _update(info=None, buffer=None):
    prompt = {"prompt_token_ids": [0] * 4, "additional_information": {"global_request_id": ["r"], **(info or {})}}
    if buffer is not None:
        prompt["model_intermediate_buffer"] = buffer
    return prompt


_REPLACE = {"meta": {"replace_streaming_prompt": True}}
_REJECTED_UPDATES = {
    # A plain stage-0 update installs its model_intermediate_buffer on the session.
    "buffer-blank-text": _update(buffer={"text": [""]}),
    "buffer-none-text-ids": _update(buffer={PRECOMPUTED_TEXT_IDS_KEY: None}),
    # A replacing update installs both payloads, so it needs a first chunk's text.
    "replace-blank-text": _update({"text": [""], **_REPLACE}),
    "buffer-replace-blank-text": _update({"text": ["  "]}, buffer=_REPLACE),
}
_ACCEPTED_UPDATES = {
    # AsyncOmni closes every stream with TokensPrompt([0]); the engine stamps the request id.
    "stream-close": {"prompt_token_ids": [0], "additional_information": {"global_request_id": ["r"]}},
    "text-less": _update(),
    # A plain update keeps the first chunk's additional_information.
    "plain-blank-text": _update({"text": ["  "]}),
    "buffer-text": _update(buffer={"text": ["Hello."]}),
    "replace-text": _update({"text": ["Hello."], **_REPLACE}),
}


@pytest.mark.parametrize("prompt", list(_REJECTED_UPDATES.values()), ids=list(_REJECTED_UPDATES))
def test_streaming_update_that_blanks_the_text_is_rejected(prompt):
    with pytest.raises(ValueError, match=_ERROR):
        check_text_update(prompt, [])


@pytest.mark.parametrize("prompt", list(_ACCEPTED_UPDATES.values()), ids=list(_ACCEPTED_UPDATES))
def test_streaming_update_that_keeps_the_text_passes_through_unchanged(prompt):
    assert check_text_update(prompt, []) is prompt


def _mrv2_talker_info(new_req):
    buffer = OmniIntermediateBuffer(1)
    buffer.add_request(0, new_req)
    return buffer.buffers[0]


def _mrv1_talker_info(new_req):
    runner = object.__new__(OmniGPUModelRunner)
    runner.model = SimpleNamespace()
    runner.requests = {new_req.req_id: SimpleNamespace()}
    runner.model_intermediate_buffer = {}
    scheduler_output = SimpleNamespace(scheduled_new_reqs=[new_req], scheduled_cached_reqs=SimpleNamespace())
    OmniGPUModelRunner._update_additional_information(runner, scheduler_output)
    return runner.model_intermediate_buffer[new_req.req_id]


@pytest.mark.parametrize("talker_info", [_mrv2_talker_info, _mrv1_talker_info], ids=["mrv2", "mrv1"])
@pytest.mark.parametrize("prompt", list(_ACCEPTED.values()), ids=list(_ACCEPTED))
def test_accepted_prompts_still_satisfy_the_talker_after_ipc(prompt, talker_info):
    # Each runner merges the payloads its own way; the Talker must see usable text in both.
    new_req = OmniNewRequestData(
        req_id="r",
        prompt_token_ids=prompt["prompt_token_ids"],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        additional_information=serialize_additional_information(prompt["additional_information"]),
        model_intermediate_buffer=prompt.get("model_intermediate_buffer"),
    )
    info = talker_info(new_req)
    assert has_tts_text_conditioning(info, info.get("hidden_states"), allow_blank_text=False)


def test_talker_check_keeps_accepting_blank_text():
    # Only the stage-0 check rejects blank text; the engine-side check is unchanged.
    assert has_tts_text_conditioning({"text": ["  "]})
    assert not has_tts_text_conditioning({"text": ["  "]}, allow_blank_text=False)
    assert not has_tts_text_conditioning({"text": []})
    # Likewise, the Talker keeps accepting ids it would discard and falls back to the text.
    assert has_tts_text_conditioning({"text": ["  "], PRECOMPUTED_TEXT_IDS_KEY: [[]]})


@pytest.mark.parametrize("pipeline", [QWEN3_TTS_PIPELINE, QWEN3_TTS_FUSED_PIPELINE])
def test_talker_stage_runs_the_check(pipeline):
    stage0 = next(stage for stage in pipeline.stages if stage.stage_id == 0)
    assert stage0.prompt_transform_func == (
        "vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder.check_text_prompt"
    )
