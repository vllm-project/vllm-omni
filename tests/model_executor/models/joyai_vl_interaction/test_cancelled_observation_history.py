# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""History boundaries after cancelled JoyAI visual replies."""

import struct

import pytest

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import frame, open_joyai
from vllm_omni.engine.duplex.commands import AppendAudio, CancelResponse, Commit, CreateItem, DeleteItem
from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import asr_to_joyai

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("speech_followup", [False, True], ids=["camera", "speech"])
@pytest.mark.parametrize("delete_response", [False, True], ids=["retained", "deleted"])
async def test_cancelled_observation_preserves_empty_assistant_boundary(mocker, speech_followup, delete_response):
    h = await open_joyai(asr=True)
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
        assert assistants == ([] if delete_response else [{"role": "assistant", "content": ""}])
        if not delete_response:
            assert messages[-2] == {"role": "assistant", "content": ""}
        assert query in str(messages[-1]["content"])
        assert "Green. One" not in prompt["prompt"]
        assert prompt["multi_modal_data"]["image"][-1].getpixel((0, 0)) == (255, 255, 0)
    finally:
        await h.manager.shutdown()
