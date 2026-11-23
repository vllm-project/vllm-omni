# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Keep a spoken query attached to the camera observation it accompanied."""

import struct

import pytest

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import frame, open_joyai
from vllm_omni.engine.duplex.commands import AckPlayback, AppendAudio, CancelResponse, Commit, CreateItem, DeleteItem
from vllm_omni.engine.duplex.contracts import DuplexOutputContext, DuplexRequestIdentity
from vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction import asr_to_joyai

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("followup_kind", ["camera", "text", "speech"])
@pytest.mark.parametrize("cancelled", [False, True], ids=["played", "cancelled"])
@pytest.mark.parametrize("deleted", [False, True], ids=["retained", "deleted"])
async def test_asr_history_keeps_query_in_its_visual_turn(mocker, followup_kind, cancelled, deleted):
    h = await open_joyai(asr=True)
    processor = h.runner.plugin.processor
    template = mocker.spy(processor, "apply_chat_template")
    mocker.patch(
        "vllm_omni.model_executor.stage_input_processors.joyai_vl_interaction.cached_processor_from_config",
        return_value=processor,
    )
    transcript = "For every new image, say the color of only the newest image."
    new_query = "Describe the newest image."
    try:
        await h.run(AppendAudio(audio=struct.pack("<h", 1000) * 16000, video_frames=(frame("red"),)))
        await h.run(Commit(final=True, create_response=True))
        submission = h.port.submissions[-1]
        request_id = submission.context.request_id
        response_id = h.session.active_response_id
        transcript_item_id = submission.prompt["additional_information"]["realtime_item_id"]
        h.runner.on_stage_output(
            0,
            tts_output(request_id, text=transcript, finished=True),
            None,
            request_id=request_id,
            context=DuplexOutputContext(
                identity=DuplexRequestIdentity(session_id=h.session.session_id, fence=submission.context.fence),
                final_stage_id=3,
                segment_finished=True,
            ),
            input_transcript=transcript,
            input_item_id=transcript_item_id,
        )
        await h.settle()
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Red", finished=True), stage_id=1, segment_finished=True
        )
        if cancelled:
            await h.run(CancelResponse())
        else:
            await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=3, segment_finished=True)
            await h.run(AckPlayback(response_id=response_id, played_ms=1000))
        if deleted:
            await h.run(DeleteItem(item_id=transcript_item_id))
        if followup_kind == "text":
            await h.run(
                CreateItem(
                    item={
                        "id": "new-query",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": new_query}],
                    }
                )
            )
        await h.run(
            AppendAudio(
                audio=struct.pack("<h", 1000) * 16000 if followup_kind == "speech" else b"",
                video_frames=(frame("blue"),),
            )
        )
        await h.run(Commit(final=True, create_response=True))
        followup = h.port.submissions[-1]
        if followup_kind == "speech":
            asr_to_joyai(
                [tts_output(followup.context.request_id, text=new_query, finished=True)],
                dict(followup.prompt),
            )
        messages = template.call_args.args[0]
        visual_turns = [
            message
            for message in messages
            if isinstance(message["content"], list) and any(part["type"] == "image" for part in message["content"])
        ]
        assert len(visual_turns) == 2
        if followup_kind != "camera":
            assert new_query in visual_turns[-1]["content"][0]["text"]
            assert new_query not in str(visual_turns[0]["content"])
        # A standalone historical user message loses the original query/frame association.
        if deleted:
            assert all(transcript not in str(message["content"]) for message in messages)
            return
        assert transcript in visual_turns[0]["content"][0]["text"]
        assert transcript not in str(visual_turns[1]["content"])
        assert sum(transcript in str(message["content"]) for message in messages) == 1
    finally:
        await h.manager.shutdown()
