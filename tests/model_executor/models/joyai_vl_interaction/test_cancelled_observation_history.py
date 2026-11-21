# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""History boundaries after cancelled JoyAI visual replies."""

import struct
from copy import deepcopy

import pytest

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import Processor, frame, open_joyai
from vllm_omni.engine.duplex.commands import AppendAudio, CancelResponse, Commit, CreateItem, DeleteItem
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import USER_QUERY_HEADER
from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import asr_to_joyai

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("speech_followup", [False, True], ids=["camera", "speech"])
@pytest.mark.parametrize("delete_response", [False, True], ids=["retained", "deleted"])
@pytest.mark.parametrize("image_limit", [1, 2], ids=["one-image", "two-images"])
async def test_cancelled_observation_preserves_empty_assistant_boundary(
    mocker, speech_followup, delete_response, image_limit
):
    h = await open_joyai(asr=True)
    h.runner.plugin.max_images = image_limit
    template = mocker.spy(h.runner.plugin.processor, "apply_chat_template")
    mocker.patch(
        "vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction.cached_processor_from_config",
        return_value=h.runner.plugin.processor,
    )
    try:
        await h.run(
            CreateItem(
                item={
                    "id": "counting-instruction",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Say the color. If green, count to fifty."}],
                }
            )
        )
        await h.run(AppendAudio(video_frames=(frame("green"),)))
        await h.run(Commit(final=True, create_response=True))
        request_id = h.port.submissions[-1].context.request_id
        response_item_id = f"item_{h.session.active_response_id}"
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Green. One, two, three.", finished=True),
            stage_id=1,
            segment_finished=True,
        )
        await h.run(CancelResponse())
        assert h.session.history_item_ids[response_item_id]["content"] == ""
        if delete_response:
            await h.run(DeleteItem(item_id=response_item_id))
        query = "What color is the newest image?"
        if not speech_followup:
            await h.run(
                CreateItem(
                    item={
                        "id": "latest-color-query",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": query}],
                    }
                )
            )
        await h.run(
            AppendAudio(
                audio=struct.pack("<h", 1000) * 156800 if speech_followup else b"",
                format="pcm16",
                sample_rate_hz=16000,
                video_frames=(frame("yellow"),),
            )
        )
        await h.run(Commit(final=True, create_response=True))
        followup = h.port.submissions[-1]
        assert followup.context.request_id != request_id
        assert followup.context.stage_id == (0 if speech_followup else 1)
        if speech_followup:
            prompt = asr_to_joyai(
                [tts_output(followup.context.request_id, text=query, finished=True)], dict(followup.prompt)
            )[0]
        else:
            prompt = followup.prompt
        messages = template.call_args.args[0]
        assistants = [message for message in messages if message["role"] == "assistant"]
        keep_boundary = not delete_response and image_limit > 1
        assert assistants == ([{"role": "assistant", "content": ""}] if keep_boundary else [])
        image_messages = [
            index
            for index, message in enumerate(messages)
            if isinstance(message["content"], list) and any(part["type"] == "image" for part in message["content"])
        ]
        assert len(image_messages) == image_limit
        assert sum(part["type"] == "image" for part in messages[-1]["content"]) == 1
        if keep_boundary:
            assert messages[-2] == {"role": "assistant", "content": ""}
            assert image_messages == [len(messages) - 3, len(messages) - 1]
        if image_limit > 1:
            assert query not in str(messages[image_messages[0]]["content"])
        assert query in str(messages[-1]["content"])
        assert "Green. One" not in prompt["prompt"]
        assert [image.getpixel((0, 0)) for image in prompt["multi_modal_data"]["image"]] == (
            [(0, 128, 0), (255, 255, 0)] if image_limit > 1 else [(255, 255, 0)]
        )
    finally:
        await h.manager.shutdown()


def test_asr_context_budget_evicts_frames_with_their_turns_without_mutating_handoff(mocker):
    processor = Processor()
    mocker.patch(
        "vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction.cached_processor_from_config",
        return_value=processor,
    )
    template = mocker.spy(processor, "apply_chat_template")
    query = "What color is the newest image?"
    current = [{"type": "text", "text": "<1.0 seconds>"}, {"type": "image"}]
    expected = [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": "Keep answers short."},
        {"role": "user", "content": [*current, {"type": "text", "text": f"{USER_QUERY_HEADER}\n{query}"}]},
    ]
    minimum = processor.apply_chat_template(expected)
    prompt = {
        "additional_information": {
            "joyai_prompt_context": {
                "system": "policy",
                "instruction": "Keep answers short.",
                "history": [{"role": "assistant", "content": "", "_item_key": "cancelled"}],
                "working_messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": "<0.0 seconds>"}, {"type": "image"}],
                        "_response_item_id": "cancelled",
                    },
                    {"role": "user", "content": current, "_current": True},
                ],
            },
            "joyai_prompt_budget": len(processor.tokenizer(text=[minimum])["input_ids"][0]) + 255,
            "joyai_image_token_counts": [256, 256],
            "deferred_multi_modal_data": {"image": ["green", "yellow"]},
        }
    }
    original = deepcopy(prompt)
    (result,) = asr_to_joyai([tts_output("asr", text=query, finished=True)], prompt)
    assert template.call_args.args[0] == expected
    assert result["multi_modal_data"] == {"image": ["yellow"]}
    assert prompt == original


@pytest.mark.parametrize(
    "working_messages",
    [
        None,
        [],
        [None],
        [{"role": "user", "content": [], "_current": False}],
        [{"role": "assistant", "content": "", "_current": True}],
        [{"role": "user", "content": [], "_current": True, "_query_key": []}],
        [{"role": "user", "content": [], "_current": True, "_history_anchors": [None]}],
        [{"role": "user", "content": [{"type": "text", "text": None}], "_current": True}],
        [{"role": "user", "content": [{"type": "image"}], "_current": True}],
    ],
)
def test_asr_rejects_malformed_visual_associations(working_messages):
    prompt = {
        "additional_information": {
            "joyai_prompt_context": {"system": "policy", "history": [], "working_messages": working_messages},
            "joyai_prompt_budget": 500,
            "joyai_image_token_counts": [],
        }
    }
    with pytest.raises(OmniClientError, match="working_messages"):
        asr_to_joyai([tts_output("asr", text="Look.", finished=True)], prompt)
