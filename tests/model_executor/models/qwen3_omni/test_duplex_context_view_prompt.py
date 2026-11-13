# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""L1 reaches final Qwen submissions through the real session model channel."""

import base64
import io

import numpy as np
import pytest
from PIL import Image

from tests.engine.duplex.test_session_runner import (
    Harness,
    append_audio,
    close_harness,
    open_harness,
    tts_output,
)
from tests.model_executor.models.qwen3_omni.test_duplex_plugin import _publish_call
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex.commands import Commit, CreateItem, CreateResponse
from vllm_omni.model_executor.models.qwen3_omni.duplex.plugin import Qwen3OmniDuplexPlugin
from vllm_omni.protocol.realtime.items import normalize_conversation_item

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


class _PromptFormatter:
    def apply_chat_template(self, messages: list[dict[str, object]], **kwargs: object) -> str:
        return repr(messages)


def _image():
    with io.BytesIO() as stream:
        Image.new("RGB", (2, 2), color="blue").save(stream, format="PNG")
        return {
            "type": "image_url",
            "image_url": {"url": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")},
        }


async def _open(keep: int | None) -> Harness:
    plugin = Qwen3OmniDuplexPlugin(lambda *args: "")
    # Avoid checkpoint loading; image/PCM decoding and plan construction stay real.
    plugin.processor = _PromptFormatter()
    return await open_harness(
        plugin=plugin,
        auto_response=False,
        stage_count=3,
        modalities=("text",),
        runtime_config=DuplexSessionRuntimeConfig(context_keep_media_turns=keep),
    )


async def _submit(h: Harness, *, value: float = 0.25) -> None:
    before = len(h.port.submissions)
    await h.run(append_audio(value=value))
    events = await h.run(Commit(final=True, create_response=True))
    assert len(h.port.submissions) == before + 1, [event.type for event in events]


@pytest.mark.parametrize("keep", [None, 1, 0])
async def test_old_mixed_image_is_removed_from_final_submission(keep):
    h = await _open(keep)
    try:
        h.session.register_history_item(
            "old-user",
            {"role": "user", "content": [{"type": "text", "text": "old question"}, _image()]},
        )
        h.session.register_history_item("old-reply", {"role": "assistant", "content": "old answer"})
        before = h.session.history_snapshot()

        await _submit(h)

        prompt = h.port.submissions[-1].prompt
        assert len(prompt["multi_modal_data"].get("image", [])) == (1 if keep is None else 0)
        assert "old question" in prompt["prompt"] and "old answer" in prompt["prompt"]
        assert h.session.history_snapshot().messages[:2] == before.messages
        audios = prompt["multi_modal_data"]["audio"]
        assert len(audios) == 1
        np.testing.assert_allclose(audios[0][0], 0.25)
    finally:
        await close_harness(h)


async def test_transcribed_retained_audio_is_removed_from_final_submission():
    h = await _open(1)
    try:
        await _submit(h, value=0.125)
        item_id = h.session.history_snapshot().item_ids[0]
        assert isinstance(item_id, str)
        assert h.session.update_input_transcript(
            item_id, "old spoken question", audio_part=h.session.input_audio_part(item_id)
        )
        request_id = h.port.submissions[-1].context.request_id
        await h.deliver_and_settle(tts_output(request_id, samples=0, text="old answer", finished=True), stage_id=0)
        await h.deliver_and_settle(tts_output(request_id, samples=0, finished=True), stage_id=2)
        before = h.session.history_snapshot()

        await _submit(h)

        prompt = h.port.submissions[-1].prompt
        audios = prompt["multi_modal_data"]["audio"]
        assert len(audios) == 1
        np.testing.assert_allclose(audios[0][0], 0.25)
        assert "old spoken question" in prompt["prompt"] and "old answer" in prompt["prompt"]
        assert h.session.history_snapshot().messages[:2] == before.messages
    finally:
        await close_harness(h)


async def test_keep_zero_preserves_current_image_and_combined_audio():
    h = await _open(0)
    try:
        h.session.register_history_item("current-image", {"role": "user", "content": [_image()]})
        for value in (0.125, 0.25):
            await h.run(append_audio(value=value))
            await h.run(Commit(final=True, create_response=False))
            item_id = h.session.history_snapshot().item_ids[-1]
            assert isinstance(item_id, str)
            assert h.session.update_input_transcript(
                item_id, "current question", audio_part=h.session.input_audio_part(item_id)
            )

        await h.run(CreateResponse())

        assert len(h.port.submissions) == 1
        prompt = h.port.submissions[-1].prompt
        assert len(prompt["multi_modal_data"].get("image", [])) == 1
        audios = prompt["multi_modal_data"]["audio"]
        assert len(audios) == 1
        np.testing.assert_allclose(audios[0][0], np.repeat(np.array([0.125, 0.25], dtype=np.float32), 16000))
    finally:
        await close_harness(h)


async def test_keep_zero_preserves_current_image_during_tool_followup():
    h = await _open(0)
    try:
        h.session.register_history_item(
            "current-question", {"role": "user", "content": [{"type": "text", "text": "look at this"}, _image()]}
        )
        await _publish_call(h, "call_view", name="describe", arguments="{}")
        await h.run(
            CreateItem(
                item=normalize_conversation_item(
                    {"type": "function_call_output", "call_id": "call_view", "output": "tool result"}
                )
            )
        )

        await h.run(CreateResponse())

        assert len(h.port.submissions) == 1
        prompt = h.port.submissions[-1].prompt
        assert "tool result" in prompt["prompt"] and "describe" in prompt["prompt"]
        assert len(prompt["multi_modal_data"].get("image", [])) == 1
    finally:
        await close_harness(h)


@pytest.mark.parametrize("keep", [None, 0])
async def test_current_image_survives_system_item_before_response(keep):
    h = await _open(keep)
    try:
        await h.run(
            CreateItem(
                item=normalize_conversation_item(
                    {
                        "type": "message",
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "describe this picture"},
                            {"type": "input_image", "image_url": _image()["image_url"]["url"]},
                        ],
                    }
                )
            )
        )
        await h.run(
            CreateItem(
                item=normalize_conversation_item(
                    {
                        "type": "message",
                        "role": "system",
                        "content": [{"type": "input_text", "text": "answer briefly"}],
                    }
                )
            )
        )

        await h.run(CreateResponse())

        assert len(h.port.submissions) == 1
        prompt = h.port.submissions[-1].prompt
        assert "describe this picture" in prompt["prompt"]
        assert "answer briefly" in prompt["prompt"]
        assert len(prompt["multi_modal_data"].get("image", [])) == 1
    finally:
        await close_harness(h)
