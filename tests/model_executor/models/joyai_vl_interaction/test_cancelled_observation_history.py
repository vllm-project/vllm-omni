# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""History boundaries after cancelled JoyAI visual replies."""

import struct
from copy import deepcopy

import pytest

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import summary_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import Processor, frame, open_joyai
from tests.model_executor.models.joyai_vl_interaction.test_visual_memory_pipeline import submissions
from vllm_omni.engine.duplex.commands import AppendAudio, CancelResponse, Commit, CreateItem, DeleteItem
from vllm_omni.engine.duplex.plugin import duplex_text_history
from vllm_omni.errors import OmniClientError
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import USER_QUERY_HEADER
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import WorkingObservation
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin import JoyAIDuplexPlugin
from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import asr_to_joyai

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("speech_followup", [False, True], ids=["camera", "speech"])
@pytest.mark.parametrize("delete_response", [False, True], ids=["retained", "deleted"])
@pytest.mark.parametrize("image_limit", [1, 2], ids=["one-image", "two-images"])
@pytest.mark.parametrize("finished", [False, True], ids=["partial", "complete"])
async def test_cancelled_observation_preserves_response_action_boundary(
    mocker, speech_followup, delete_response, image_limit, finished
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
            tts_output(request_id, text="</response> Green. One, two, three.", finished=finished),
            stage_id=1,
            segment_finished=finished,
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
        assert assistants == ([{"role": "assistant", "content": "</response>"}] if keep_boundary else [])
        image_messages = [
            index
            for index, message in enumerate(messages)
            if isinstance(message["content"], list) and any(part["type"] == "image" for part in message["content"])
        ]
        assert len(image_messages) == image_limit
        assert sum(part["type"] == "image" for part in messages[-1]["content"]) == 1
        if keep_boundary:
            assert messages[-2] == {"role": "assistant", "content": "</response>"}
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


@pytest.mark.asyncio
@pytest.mark.parametrize("summarized", [False, True], ids=["pending", "summarized"])
async def test_audio_only_followup_retains_latest_frame_until_summarized(mocker, summarized):
    h = await open_joyai(asr=True)
    plugin = h.runner.plugin
    plugin.max_images = 1
    template = mocker.spy(plugin.processor, "apply_chat_template")
    mocker.patch(
        "vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction.cached_processor_from_config",
        return_value=plugin.processor,
    )
    try:
        await h.run(AppendAudio(video_frames=(frame("green"),)))
        await h.run(Commit(final=True, create_response=True))
        request_id = submissions(h, auxiliary=False)[-1].context.request_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Green.", finished=True),
            stage_id=1,
            segment_finished=True,
        )
        await h.run(CancelResponse())
        state = h.runner.model_state
        assert state.visual_memory.pending[-1].id <= state.closed_frame_id
        timestamp = f"<{state.visual_memory.pending[-1].time_s:.1f} seconds>"
        if summarized:
            request = await plugin.prepare_auxiliary_request(
                history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
            )
            assert request is not None
            assert plugin.apply_auxiliary_result(
                request=request, output=summary_output("summary", "A green image."), state=state
            )
            assert not state.visual_memory.pending
        await h.run(AppendAudio(audio=struct.pack("<h", 1000) * 156800, format="pcm16", sample_rate_hz=16000))
        await h.run(Commit(final=True, create_response=True))
        followup = submissions(h, auxiliary=False)[-1]
        assert followup.context.stage_id == 0
        query = "What color was the last image?"
        (prompt,) = asr_to_joyai(
            [tts_output(followup.context.request_id, text=query, finished=True)], dict(followup.prompt)
        )
        messages = template.call_args.args[0]
        if summarized:
            assert not prompt.get("multi_modal_data", {}).get("image")
            assert "A green image." in prompt["prompt"]
        else:
            assert [image.getpixel((0, 0)) for image in prompt["multi_modal_data"]["image"]] == [(0, 128, 0)]
            assert messages[-3:] == [
                {"role": "user", "content": [{"type": "text", "text": timestamp}, {"type": "image"}]},
                {"role": "assistant", "content": "</response>"},
                {"role": "user", "content": [{"type": "text", "text": f"{USER_QUERY_HEADER}\n{query}"}]},
            ]
        assert "</response> Green." not in prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.parametrize("reply", ["The image was green.", ""], ids=["played", "unheard"])
def test_recent_history_boundary_uses_observation_refs_only_for_committed_replies(reply):
    conversation = [
        {"role": "user", "content": "What color is the image?"},
        {"role": "assistant", "content": reply},
        *[{"role": role, "content": f"{role} {index}"} for index in range(7) for role in ("user", "assistant")],
        {"role": "user", "content": "Latest question"},
    ]
    item_ids = [f"item-{index}" for index in range(len(conversation))]
    history, _, _ = JoyAIDuplexPlugin._bounded_text_history(
        {
            "conversation": conversation,
            "conversation_item_ids": item_ids,
            "joyai_observations": (WorkingObservation("old", (0,), "", response_item_id=item_ids[1]),),
        },
        {},
        response_refs=set(),
    )
    if reply:
        assert history[0] == {"role": "assistant", "content": f"</response> {reply}", "_item_key": item_ids[1]}
        assert len(history) == 16
    else:
        assert all(item["_item_key"] != item_ids[1] for item in history)


@pytest.mark.parametrize("image_text", [None, USER_QUERY_HEADER], ids=["null", "query-header"])
def test_asr_query_filter_preserves_image_parts_with_text_metadata(mocker, image_text):
    processor = Processor()
    template = mocker.spy(processor, "apply_chat_template")
    mocker.patch(
        "vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction.cached_processor_from_config",
        return_value=processor,
    )
    image_part = {"type": "image", "text": image_text}
    prompt = {
        "additional_information": {
            "joyai_prompt_context": {
                "system": "policy",
                "history": [],
                "working_messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"{USER_QUERY_HEADER}\nOld question"}, image_part],
                        "_query_key": "deleted-question",
                    },
                    {"role": "user", "content": [], "_current": True},
                ],
            },
            "joyai_prompt_budget": 16384,
            "joyai_image_token_counts": [256],
            "deferred_multi_modal_data": {"image": ["green"]},
        }
    }
    (result,) = asr_to_joyai([tts_output("asr", text="New question", finished=True)], prompt)
    assert template.call_args.args[0] == [
        {"role": "system", "content": "policy"},
        {"role": "user", "content": [image_part]},
        {"role": "user", "content": [{"type": "text", "text": f"{USER_QUERY_HEADER}\nNew question"}]},
    ]
    assert result["multi_modal_data"] == {"image": ["green"]}


@pytest.mark.parametrize(
    ("old_text", "retain_text"),
    [
        ("<0.0 seconds>", False),
        ("The event started at <2 seconds>", True),
        ("The event started at <2.0 seconds>", True),
    ],
    ids=["timestamp", "caption-integer", "caption-decimal"],
)
def test_asr_context_budget_evicts_frames_with_their_turns_without_mutating_handoff(mocker, old_text, retain_text):
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
        {"role": "user", "content": [{"type": "text", "text": f"{USER_QUERY_HEADER}\n{query}"}, *current]},
    ]
    if retain_text:
        expected.insert(2, {"role": "user", "content": [{"type": "text", "text": old_text}]})
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
                        "content": [{"type": "text", "text": old_text}, {"type": "image"}],
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
