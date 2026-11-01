# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Visual payload and lifecycle regressions; mocked text is not a quality eval."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from PIL import Image
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_memory import summary_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import Processor, frame, open_joyai
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    AppendAudio,
    CancelInput,
    Commit,
    CreateResponse,
    DeleteItem,
    Heartbeat,
)
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError, duplex_text_history
from vllm_omni.engine.duplex.realtime_commands import translate_realtime_command
from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import USER_QUERY_HEADER
from vllm_omni.experimental.fullduplex.joyvl.memory.memory import MidTermSummary, WorkingChunk
from vllm_omni.experimental.fullduplex.joyvl.serving.config import InteractionConfig
from vllm_omni.experimental.fullduplex.joyvl.serving.session import InteractionSession
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.data_plane import WorkingObservation
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.memory import VisualSummary
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin import JoyAIDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def submissions(h, *, auxiliary):
    return [item for item in h.port.submissions if ("auxiliary-" in item.context.request_id) == auxiliary]


async def settle(h):
    # Auxiliary requests deliberately wait for test-controlled completions.
    return await h.settle(idle_s=0.01, timeout_s=0.15)


async def command(h, value):
    h.submit(value)
    return await settle(h)


async def wait_auxiliary(h, count=1):
    async def wait():
        while len(submissions(h, auxiliary=True)) < count:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait(), timeout=2)
    return submissions(h, auxiliary=True)[count - 1]


async def silence(h, submission):
    assert h.deliver(
        tts_output(submission.context.request_id, text="</silence>", finished=True),
        stage_id=submission.context.stage_id,
        segment_finished=True,
    )
    await settle(h)


async def camera(h, color="red"):
    before = len(submissions(h, auxiliary=False))
    await command(h, AppendAudio(video_frames=(frame(color),)))
    await command(h, Commit(final=True, create_response=True))
    observations = submissions(h, auxiliary=False)
    assert len(observations) == before + 1
    submission = observations[-1]
    await silence(h, submission)
    return submission


async def finish_summary(h, submission, text):
    assert h.deliver(
        summary_output(submission.context.request_id, text),
        stage_id=submission.context.stage_id,
        segment_finished=True,
    )
    return await settle(h)


def user_item(item_id, content):
    return translate_realtime_command(
        {
            "type": "conversation.item.create",
            "item": {"id": item_id, "type": "message", "role": "user", "content": content},
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_id,stage_count", [(0, 3), (1, 4)])
async def test_silent_frames_submit_timestamped_images_to_selected_private_stage(stage_id, stage_count):
    h = await open_joyai(modalities=("text",), asr=stage_id == 1)
    try:
        assert h.port.stage_count == stage_count
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 4
        for color in ("red", "blue", "green", "yellow"):
            await camera(h, color)
        auxiliary = await wait_auxiliary(h)
        context = auxiliary.context
        assert context.stage_id == context.final_stage_id == stage_id
        assert not auxiliary.resumable and not auxiliary.already_submitted
        assert context.request_id not in h.runner.plugin.data_plane.requests
        images = auxiliary.prompt["multi_modal_data"]["image"]
        assert len(images) == 4 and all(isinstance(image, Image.Image) for image in images)
        assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255), (0, 128, 0), (255, 255, 0)]
        assert all(image.tobytes() == entry.image.tobytes() for image, entry in zip(images, memory.pending))
        for entry in memory.pending:
            assert f"<{entry.time_s:.1f} seconds>" in auxiliary.prompt["prompt"]
        events = await finish_summary(h, auxiliary, "A red object preceded blue, green, and yellow objects.")
        assert not memory.pending and len(memory.mid_term) == 1
        assert not any(event.to_realtime()["type"].startswith("response.") for event in events)
        assert len(h.port.submissions) == 5, "Private summaries must not create TTS-stage requests"
        assert ([context.request_id], True) in h.port.cleanups
        assert context.request_id not in h.session.resource_request_ids()
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_camera_only_recovers_after_visual_retry_budget_is_exhausted(mocker):
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 2
        memory.max_pending = 3
        prepare = h.runner.plugin.prepare_auxiliary_request

        async def short_delays(**kwargs):
            request = await prepare(**kwargs)
            if request is None:
                return None
            assert request.retry_cooldown_s == 60.0
            return replace(request, retry_delays_s=(0,))

        mocker.patch.object(h.runner.plugin, "prepare_auxiliary_request", side_effect=short_delays)
        await camera(h, "red")
        await camera(h, "blue")
        for attempt in (1, 2):
            auxiliary = await wait_auxiliary(h, attempt)
            events = await finish_summary(h, auxiliary, "")
            assert not any(event.to_realtime()["type"] == "error" for event in events)
        timer = h.runner.memory._retry_timer
        assert timer is not None
        await camera(h, "green")
        assert len(memory.pending) == 3
        assert h.runner.memory._retry_timer is timer
        # No text/audio input or source change is needed to recover.
        timer.cancel()
        h.runner._post_memory_event(
            "auxiliary_retry",
            {"source": h.runner.memory._source, "operation": h.runner.memory._generation},
        )
        retry = await wait_auxiliary(h, 3)
        await finish_summary(h, retry, "A red object was followed by a blue object.")
        assert len(memory.pending) == 1
        assert len(memory.mid_term) == 1
        assert h.runner.memory._failures == 0
        await camera(h, "yellow")
        assert len(submissions(h, auxiliary=False)) == 4
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_new_camera_tail_survives_mid_summary_and_long_term_rollover():
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 2
        memory.mid_group = 2
        await camera(h, "red")
        await camera(h, "blue")
        first = await wait_auxiliary(h)
        first_ids = [entry.id for entry in memory.pending]
        await camera(h, "green")
        tail = memory.pending[-1]
        assert len(submissions(h, auxiliary=False)[-1].prompt["multi_modal_data"]["image"]) == 1
        assert len(memory.pending) == 3
        assert ([first.context.request_id], True) not in h.port.cleanups
        await finish_summary(h, first, "A red cup was followed by a blue cup.")
        assert memory.pending == [tail]
        assert (memory.mid_term[0].first_id, memory.mid_term[0].last_id) == tuple(first_ids)
        await camera(h, "yellow")
        second = await wait_auxiliary(h, 2)
        await finish_summary(h, second, "A green cup was followed by a yellow cup.")
        long_request = await wait_auxiliary(h, 3)
        assert long_request.prompt["multi_modal_data"] == {}
        assert "red cup" in long_request.prompt["prompt"] and "green cup" in long_request.prompt["prompt"]
        assert not memory.pending and len(memory.mid_term) == 2
        await camera(h, "purple")
        await camera(h, "orange")
        assert h.runner.model_state.closed_frame_id == 5
        assert h.runner.model_state.visual_source.kind == "long"
        await camera(h, "white")
        assert len(submissions(h, auxiliary=False)[-1].prompt["multi_modal_data"]["image"]) == 1
        await finish_summary(h, long_request, "Cups appeared in red, blue, green, and yellow order.")
        assert not memory.mid_term and len(memory.long_term) == 1
        assert (memory.long_term[0].first_id, memory.long_term[0].last_id) == (0, 3)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("closure", ["count", "capacity"])
async def test_closed_source_survives_cancel_failure_and_retry_without_reappearing(closure):
    h = await open_joyai(modalities=("text",))
    try:
        plugin, state = h.runner.plugin, h.runner.model_state
        memory = state.visual_memory
        memory.chunk_frames = 2 if closure == "count" else 100
        await camera(h, "red")
        await camera(h, "blue")
        if closure == "capacity":
            memory.max_encoded_bytes = 2 * memory.encoded_bytes
            await command(h, Heartbeat())
        await wait_auxiliary(h)
        source = state.visual_source
        assert source.covered_ids == (0, 1)
        assert source.closure_reason == closure
        original = tuple(memory.pending)
        await command(h, CancelInput())
        assert state.closed_frame_id == 1 and tuple(memory.pending) == original
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert request.model_source == source
        assert not plugin.apply_auxiliary_result(
            request=request, output=summary_output("failed", "", "length"), state=state
        )
        assert state.closed_frame_id == 1 and tuple(memory.pending) == original

        # A text-only observation must not rehydrate the legacy last-four cache.
        await command(h, user_item("q1", [{"type": "input_text", "text": "What do you see?"}]))
        await command(h, CreateResponse())
        observation = submissions(h, auxiliary=False)[-1]
        assert observation.prompt["multi_modal_data"] == {}
        await silence(h, observation)
        await camera(h, "green")
        assert len(submissions(h, auxiliary=False)[-1].prompt["multi_modal_data"]["image"]) == 1
        assert state.closed_frame_id == 1  # Closed backlog cannot close a smaller live tail.
        assert [frame.id for frame in memory.pending] == [0, 1, 2]
        assert plugin.apply_auxiliary_result(
            request=request, output=summary_output("retry", "Red followed by blue."), state=state
        )
        assert [frame.id for frame in memory.pending] == [2]
        assert not plugin.apply_auxiliary_result(
            request=request, output=summary_output("stale", "A stale result."), state=state
        )
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["inflight", "mid", "long"])
async def test_deleting_conversation_image_revokes_visual_memory_without_new_append(phase):
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 1
        memory.mid_group = 2
        await command(
            h, user_item("picture", [{"type": "input_image", "image_url": "data:image/png;base64," + frame()}])
        )
        await command(h, CreateResponse())
        await silence(h, submissions(h, auxiliary=False)[-1])
        first = await wait_auxiliary(h)
        if phase != "inflight":
            await finish_summary(h, first, "A red cup is on the table.")
            assert memory.mid_term
        if phase == "long":
            await camera(h, "blue")
            await finish_summary(h, await wait_auxiliary(h, 2), "A blue cup replaced it.")
            await finish_summary(h, await wait_auxiliary(h, 3), "The red cup was replaced by a blue cup.")
            assert memory.long_term
        generation = memory.generation
        observation_count = len(submissions(h, auxiliary=False))
        await command(h, DeleteItem(item_id="picture"))
        await command(h, Heartbeat())
        if phase == "inflight":
            h.deliver(
                summary_output(first.context.request_id, "Stale red cup memory."),
                stage_id=first.context.stage_id,
                segment_finished=True,
            )
            await settle(h)
        assert memory.generation > generation
        assert not memory.pending and not memory.mid_term and not memory.long_term
        assert h.runner.model_state.closed_frame_id == -1
        assert h.runner.model_state.visual_source is None
        assert len(submissions(h, auxiliary=False)) == observation_count
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_delayed_question_keeps_early_unsummarized_image_beyond_four_frames():
    h = await open_joyai(modalities=("text",))
    try:
        h.runner.model_state.visual_memory.chunk_frames = 100
        for color in ("red", "blue", "blue", "blue", "blue"):
            await camera(h, color)
        assert not submissions(h, auxiliary=True)
        await command(h, user_item("question", [{"type": "input_text", "text": "What was the first color?"}]))
        await command(h, CreateResponse())
        prompt = submissions(h, auxiliary=False)[-1].prompt
        assert "What was the first color?" in prompt["prompt"]
        images = prompt["multi_modal_data"]["image"]
        assert len(images) == 5
        assert images[0].getpixel((0, 0)) == (255, 0, 0)
        assert all(image.getpixel((0, 0)) == (0, 0, 255) for image in images[1:])
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", ["count", "bytes"])
async def test_full_visual_memory_rejects_frame_but_next_text_request_still_runs(limit):
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.visual_memory
        memory.chunk_frames = 100
        if limit == "count":
            memory.max_pending = 2
        else:
            memory.max_pixel_bytes = 2 * 32 * 32 * 3
        await camera(h, "red")
        await camera(h, "blue")
        before = tuple(memory.pending)
        count = len(submissions(h, auxiliary=False))
        await command(h, AppendAudio(video_frames=(frame("green"),)))
        events = await command(h, Commit(final=True, create_response=True))
        assert any(event.to_realtime()["type"] == "error" for event in events)
        assert tuple(memory.pending) == before
        assert len(submissions(h, auxiliary=False)) == count
        assert not h.runner.closing
        await command(h, user_item("question", [{"type": "input_text", "text": "Describe the earlier objects."}]))
        await command(h, CreateResponse())
        assert len(submissions(h, auxiliary=False)) == count + 1
        assert "Describe the earlier objects." in submissions(h, auxiliary=False)[-1].prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_context_capacity_closes_short_prefix_without_discarding_tail():
    h = await open_joyai(modalities=("text",))
    try:
        plugin = h.runner.plugin
        state = h.runner.model_state
        memory = state.visual_memory
        for color in ("red", "blue", "green", "yellow"):
            await camera(h, color)
        memory.chunk_frames = 4
        source = memory.frame_source()
        full = plugin._prepare_visual_summary(source, tuple(memory.pending), memory.mid_max_tokens)
        assert full is not None
        text_tokens = len(plugin.processor.tokenizer(text=[full.prompt["prompt"]])["input_ids"][0])
        visual_tokens = sum(entry.token_count for entry in memory.pending) - 4
        plugin.max_model_len = text_tokens + visual_tokens + memory.mid_max_tokens - 1
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert request is not None
        covered = len(request.model_source.covered_ids)
        assert 0 < covered < 4
        assert request.model_source.closure_reason == "context"
        assert len(request.prompt["multi_modal_data"]["image"]) == covered
        tail = memory.pending[covered:]
        assert plugin.apply_auxiliary_result(
            request=request, output=summary_output("test", "The early colors."), state=state
        )
        assert memory.pending == tail
        assert memory.mid_term[0].last_id == covered - 1
        assert memory.mid_term[0].closure_reason == "context"
        # The budget-split source stays closed to interaction, but remains
        # eligible for consolidation even below the normal four-frame limit.
        config = plugin.prepare_prompt_config({"conversation": list(h.session.history)}, state=state, payload={})
        assert config["joyai_pending_frames"] == ()
        remainder = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert remainder.model_source.covered_ids == tuple(frame.id for frame in tail)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_image_count_limit_closes_short_prefix_with_image_limit_reason():
    h = await open_joyai(modalities=("text",))
    try:
        plugin = h.runner.plugin
        state = h.runner.model_state
        memory = state.visual_memory
        for color in ("red", "blue", "green", "yellow"):
            await camera(h, color)
        memory.chunk_frames = 4
        plugin.max_images = 2
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert request is not None
        source = request.model_source
        assert source.covered_ids == (0, 1)
        assert source.closure_reason == "image_limit"
        images = request.prompt["multi_modal_data"]["image"]
        assert [image.getpixel((0, 0)) for image in images] == [(255, 0, 0), (0, 0, 255)]
        tail = memory.pending[2:]
        assert plugin.apply_auxiliary_result(
            request=request, output=summary_output("test", "Red then blue."), state=state
        )
        assert memory.pending == tail
        assert memory.mid_term[0].closure_reason == "image_limit"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_spoken", [False, True])
async def test_working_prompt_matches_experimental_session_order_and_memory_role(mocker, first_spoken):
    """Compare real experimental prompts after both silent and spoken actions."""
    backend = SimpleNamespace(generate=AsyncMock(return_value=("</response> Red.", None)))
    reference = InteractionSession("reference", InteractionConfig(persona="silent"), backend)
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    plugin.max_images = 999
    state = plugin.create_session_state()
    memory_text = "Earlier a green card showed code K731."
    reference._policy.brain.memory.mid_term_summaries.append(MidTermSummary(0, "0.0 seconds-0.0 seconds", memory_text))
    state.visual_memory.mid_term.append(VisualSummary(0, 0, 0.0, 0.0, memory_text, 40, chunk_index=0))
    state.visual_memory.next_id = 1
    state.visual_memory.next_chunk_index = 1
    first_query = "What color is it?" if first_spoken else ""
    current_query = "What changed since your last answer?"
    try:
        await reference.step(["data:image/png;base64," + frame()], query=first_query, t=1.0)
        await reference.step(["data:image/png;base64," + frame("blue")], query=current_query, t=2.0)
        expected = backend.generate.call_args.args[0]
        for message in expected:
            if isinstance(message["content"], list):
                message["content"] = [
                    {"type": "image"} if part["type"] == "image_url" else part for part in message["content"]
                ]
        old = state.visual_memory.append(
            source_id="old", image=Image.new("RGB", (32, 32), "red"), token_count=256, time_s=1.0
        )
        state.observations.append(
            WorkingObservation(
                "old",
                (old.id,),
                first_query,
                silent=not first_spoken,
                query_item_id="q1" if first_spoken else None,
                history_anchor_item_ids=("q1",) if first_spoken else (),
                response_item_id="a1" if first_spoken else None,
            )
        )
        conversation = []
        item_ids = []
        if first_spoken:
            conversation.extend([{"role": "user", "content": first_query}, {"role": "assistant", "content": "Red."}])
            item_ids.extend(["q1", "a1"])
        conversation.append({"role": "user", "content": current_query})
        item_ids.append("q2")
        config = plugin.prepare_prompt_config(
            {"conversation": conversation, "conversation_item_ids": item_ids}, state=state, payload={}
        )
        config.update(
            joyai_new_images=[Image.new("RGB", (32, 32), "blue")],
            joyai_new_image_tokens=[256],
            joyai_new_frame_time=2.0,
        )
        template = mocker.spy(plugin.processor, "apply_chat_template")
        plan = plugin.plan_append(
            request_id="current",
            fence=DuplexFence("s"),
            session_config=config,
            runtime_config=plugin.runtime_config_for_update(DuplexSessionConfig(), {}),
            seq=2,
            turn_seq=2,
            payload={},
            final=True,
            sampling_params=SamplingParams(max_tokens=256),
        )
        assert template.call_args.args[0] == expected
        assert memory_text not in expected[0]["content"]
        assert expected[2] == {"role": "assistant", "content": "</response> Red." if first_spoken else "</silence>"}
        assert [image.getpixel((0, 0)) for image in plan.prompt["multi_modal_data"]["image"]] == [
            (255, 0, 0),
            (0, 0, 255),
        ]
    finally:
        await reference.aclose()


@pytest.mark.parametrize("with_new_frame,query_index", [(False, 0), (True, 0), (True, 2)])
def test_repeated_fresh_query_survives_visual_history_deduplication(mocker, with_new_frame, query_index):
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    plugin.max_images = 999
    state = plugin.create_session_state()
    query = "What is on the card?"
    old = state.visual_memory.append(source_id="old", image=Image.new("RGB", (32, 32)), token_count=256, time_s=1.0)
    item_ids = ("q1", "a1", "q2")
    state.observations.append(
        WorkingObservation(
            "old",
            (old.id,),
            query,
            query_index=query_index,
            query_item_id=item_ids[query_index],
            history_anchor_item_ids=item_ids[: query_index + 1],
            response_item_id="a1" if query_index == 0 else None,
        )
    )
    config = plugin.prepare_prompt_config(
        {
            "conversation": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": "K731"},
                {"role": "user", "content": query},
            ],
            "conversation_item_ids": item_ids,
        },
        state=state,
        payload={},
    )
    if with_new_frame:
        config.update(joyai_new_images=[Image.new("RGB", (32, 32), "blue")], joyai_new_image_tokens=[256])
    template = mocker.spy(plugin.processor, "apply_chat_template")
    plugin.plan_append(
        request_id="repeat",
        fence=DuplexFence("s"),
        session_config=config,
        runtime_config=plugin.runtime_config_for_update(DuplexSessionConfig(), {}),
        seq=2,
        turn_seq=2,
        payload={},
        final=True,
        sampling_params=SamplingParams(max_tokens=256),
    )
    messages = template.call_args.args[0]
    assert messages[-1]["role"] == "user"
    if query_index == 0:
        assert any(query in part.get("text", "") for part in messages[-1]["content"])
    else:
        # A camera update after the repeated query was already observed keeps
        # that unanswered query in its original visual position.
        assert any(query in str(message["content"]) for message in messages[1:])
    assert sum(query in str(message["content"]) for message in messages) == 2
    assert all(set(message) == {"role", "content"} for message in messages)


def test_visual_ledger_does_not_restore_deleted_user_instruction():
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.max_model_len = 16384
    plugin.max_images = 999
    state = plugin.create_session_state()
    old = state.visual_memory.append(source_id="old", image=Image.new("RGB", (32, 32)), token_count=256, time_s=1.0)
    state.observations.append(WorkingObservation("old", (old.id,), "Deleted instruction", silent=True))
    config = plugin.prepare_prompt_config(
        {"conversation": [{"role": "user", "content": "Current instruction"}]}, state=state, payload={}
    )
    messages, images, _ = plugin._working_messages(config, {"initial_user_text": "Current instruction"})
    assert "Deleted instruction" not in str(messages)
    assert "Current instruction" in str(messages)
    assert "</silence>" in str(messages)
    assert len(images) == 1


@pytest.mark.asyncio
async def test_observation_ledger_keeps_early_frames_and_retires_only_accepted_source():
    h = await open_joyai(modalities=("text",))
    try:
        for index in range(6):
            await camera(h, "red" if index == 0 else "blue")
        state = h.runner.model_state
        assert len(state.observations) == 6
        assert all(observation.silent for observation in state.observations)
        prompt = submissions(h, auxiliary=False)[-1].prompt
        assert len(prompt["multi_modal_data"]["image"]) == 6
        assert prompt["multi_modal_data"]["image"][0].getpixel((0, 0)) == (255, 0, 0)
        state.visual_memory.chunk_frames = 4
        request = await h.runner.plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        before = tuple(state.observations)
        assert not h.runner.plugin.apply_auxiliary_result(
            request=request, output=summary_output("private", "", "length"), state=state
        )
        assert tuple(state.observations) == before
        assert h.runner.plugin.apply_auxiliary_result(
            request=request, output=summary_output("private", "A red object preceded blue objects."), state=state
        )
        assert [item.frame_ids for item in state.observations] == [(4,), (5,)]
        assert not h.runner.plugin.apply_auxiliary_result(
            request=request, output=summary_output("private", "Stale replacement"), state=state
        )
        assert [item.frame_ids for item in state.observations] == [(4,), (5,)]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_real_runner_keeps_text_only_qa_between_visual_observations(mocker):
    h = await open_joyai(modalities=("text",))
    template = mocker.spy(h.runner.plugin.processor, "apply_chat_template")
    try:
        await command(h, user_item("q1", [{"type": "input_text", "text": "What color?"}]))
        await command(h, AppendAudio(video_frames=(frame("red"),)))
        await command(h, Commit(final=True, create_response=True))
        first = submissions(h, auxiliary=False)[-1]
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(first.context.request_id, text="</response> Red.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        state = h.runner.model_state
        assert state.observations[0].response_item_id == f"item_{response_id}"
        await command(h, user_item("q2", [{"type": "input_text", "text": "What was before your answer?"}]))
        await command(h, CreateResponse())
        second = submissions(h, auxiliary=False)[-1]
        await h.deliver_and_settle(
            tts_output(second.context.request_id, text="</response> The red card.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        assert len(state.observations) == 1, "Pure text Q&A stays in shared Session history."
        await camera(h, "blue")
        messages = template.call_args.args[0]
        assert [message["role"] for message in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
        assert "What color?" in str(messages[1]["content"])
        assert {"type": "image"} in messages[1]["content"]
        assert messages[2]["content"] == "</response> Red."
        assert "What was before your answer?" in str(messages[3]["content"])
        assert messages[4]["content"] == "</response> The red card."
        assert {"type": "image"} in messages[5]["content"]
        assert "What was before your answer?" not in str(messages[5]["content"])
        # Deleting the latest anchor falls back to the prior surviving item,
        # keeping the question before the blue image rather than after it.
        second_response_id = next(
            item_id
            for item_id, message in h.session.history_item_ids.items()
            if message.get("content") == "The red card."
        )
        await command(h, DeleteItem(item_id=second_response_id))
        await camera(h, "green")
        messages = template.call_args.args[0]
        question_pos = next(
            i for i, message in enumerate(messages) if "What was before your answer?" in str(message["content"])
        )
        image_positions = [
            i
            for i, message in enumerate(messages)
            if isinstance(message["content"], list) and {"type": "image"} in message["content"]
        ]
        assert image_positions[0] < question_pos < image_positions[1]
        assert "The red card." not in str(messages)
        state.visual_memory.chunk_frames = 3
        request = await h.runner.plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert h.runner.plugin.apply_auxiliary_result(
            request=request, output=summary_output("private", "The card changed red, blue, then green."), state=state
        )
        assert not state.observations
        await camera(h, "yellow")
        messages = template.call_args.args[0]
        assert str(messages).count("Red.") == 1
        assert "[Q&A History]" in str(messages)
        assert str(messages).count("What color?") == 1
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_initial_user_text_survives_its_spoken_answer_and_later_camera(mocker):
    h = await open_joyai(modalities=("text",))
    template = mocker.spy(h.runner.plugin.processor, "apply_chat_template")
    try:
        h.session.replace_runtime_config({**h.session.runtime_config, "initial_user_text": "Watch the card color."})
        await command(h, AppendAudio(video_frames=(frame(),)))
        await command(h, Commit(final=True, create_response=True))
        request_id = submissions(h, auxiliary=False)[-1].context.request_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Red.", finished=True), stage_id=0, segment_finished=True
        )
        assert "Watch the card color." not in str(h.session.history)
        await camera(h, "blue")
        messages = template.call_args.args[0]
        assert "Watch the card color." in str(messages[1]["content"])
        assert {"type": "image"} in messages[1]["content"]
        assert messages[2] == {"role": "assistant", "content": "</response> Red."}
        assert "Watch the card color." not in str(messages[3:])
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("retire_before_ack", [False, True])
async def test_visual_response_reference_uses_only_current_playback_committed_text(mocker, retire_before_ack):
    h = await open_joyai()
    plugin = h.runner.plugin
    template = mocker.spy(plugin.processor, "apply_chat_template")
    try:
        await command(h, user_item("q1", [{"type": "input_text", "text": "Describe the card."}]))
        await command(h, AppendAudio(video_frames=(frame(),)))
        await command(h, Commit(final=True, create_response=True))
        request_id = submissions(h, auxiliary=False)[-1].context.request_id
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Red card.", finished=True), stage_id=0, segment_finished=True
        )
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
        assert h.runner.model_state.observations[0].response_item_id == f"item_{response_id}"

        def render():
            conversation, item_ids = h.session.history_snapshot()
            config = plugin.prepare_prompt_config(
                {"conversation": conversation, "conversation_item_ids": item_ids},
                state=h.runner.model_state,
                payload={},
            )
            config.update(joyai_new_images=[Image.new("RGB", (32, 32), "blue")], joyai_new_image_tokens=[256])
            plugin.plan_append(
                request_id="probe",
                fence=h.session.fence,
                session_config=config,
                runtime_config=h.session.runtime_config,
                seq=2,
                turn_seq=2,
                payload={},
                final=True,
                sampling_params=SamplingParams(max_tokens=256),
            )
            return template.call_args.args[0]

        async def retire():
            state = h.runner.model_state
            state.visual_memory.chunk_frames = 1
            request = await plugin.prepare_auxiliary_request(
                history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
            )
            assert plugin.apply_auxiliary_result(
                request=request, output=summary_output("private", "A red object."), state=state
            )

        if retire_before_ack:
            await retire()
        assert "Red card." not in str(render())
        assert "[Q&A History]" not in str(render())
        # Materialization changes the history shape; the response is still
        # attached to the same visual source through its stable item identity.
        await command(h, AckPlayback(response_id=response_id, played_ms=1000))
        messages = render()
        if not retire_before_ack:
            assert messages[2] == {"role": "assistant", "content": "</response> Red card."}
            assert "Red card." not in str(messages[3:])
            await retire()
        archived = render()
        assert "[Q&A History]" in str(archived)
        assert str(archived).count("Red card.") == 1
        assert h.session.truncate_history_item(f"item_{response_id}", audio_end_ms=0, hard=True)
        assert "Red card." not in str(render())
        await command(h, DeleteItem(item_id=f"item_{response_id}"))
        assert "Red card." not in str(render())
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("text_summary", ["", "T" * 2000], ids=["visual", "text-and-visual"])
async def test_omitted_visual_summaries_rebudget_answered_open_chunk(text_summary):
    h = await open_joyai(modalities=("text",))
    try:
        query = "Tell me whenever the card color changes."
        await command(
            h,
            user_item(
                "query",
                [
                    {"type": "input_text", "text": query},
                    {"type": "input_image", "image_url": "data:image/png;base64," + frame()},
                ],
            ),
        )
        await command(h, CreateResponse())
        request_id = submissions(h, auxiliary=False)[-1].context.request_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> I will monitor the color.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        plugin, state = h.runner.plugin, h.runner.model_state
        conversation, item_ids = h.session.history_snapshot()
        config = plugin.prepare_prompt_config(
            {"conversation": conversation, "conversation_item_ids": item_ids},
            state=state,
            payload={"video_frames": [frame("blue")]},
        )
        config["joyai_new_images"] = plugin._decode_images(config["joyai_new_frames"])
        config["joyai_new_image_tokens"] = plugin._image_token_counts(config["joyai_new_images"])
        assert state.closed_frame_id < state.observations[0].frame_ids[0]

        def plan(prepared):
            return plugin.plan_append(
                request_id="next-observation",
                fence=DuplexFence("s"),
                session_config=prepared,
                runtime_config=h.session.runtime_config,
                seq=2,
                turn_seq=2,
                payload={},
                final=True,
                sampling_params=SamplingParams(max_tokens=256),
            ).prompt

        expected = plan(config)
        inputs = plugin.processor(text=[expected["prompt"]], images=expected["multi_modal_data"]["image"])
        plugin.max_model_len = len(inputs["input_ids"][0]) + 256
        summaries = [VisualSummary(i, i, 0.0, 0.0, "S" * 2000, 2000, chunk_index=i) for i in range(2)]
        actual = plan({**config, "joyai_summary": text_summary, "joyai_visual_summaries": summaries})
        assert actual["prompt"] == expected["prompt"]
        assert actual["prompt"].count(query) == 1
        assert "I will monitor the color." in actual["prompt"]
        assert len(actual["multi_modal_data"]["image"]) == 2
        assert len(summaries) == 2  # Budget omission leaves accepted memory available.
    finally:
        await h.manager.shutdown()


@pytest.mark.parametrize("answer_count,covered_messages", [(1, 0), (20, 0), (1, 2)])
def test_answered_instruction_survives_history_and_context_trimming(mocker, answer_count, covered_messages):
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 16384
    plugin.max_images = 999
    query = "Tell me whenever the card color changes."
    config = {
        "joyai_summary_messages": covered_messages,
        "conversation": [
            {"role": "user", "content": query},
            *[{"role": "assistant", "content": "Red."} for _ in range(answer_count)],
        ],
    }
    kwargs = dict(
        request_id="budget",
        fence=DuplexFence("s"),
        runtime_config=plugin.runtime_config_for_update(DuplexSessionConfig(), {}),
        seq=1,
        turn_seq=1,
        payload={},
        final=True,
        sampling_params=SamplingParams(max_tokens=256),
    )
    template = mocker.spy(plugin.processor, "apply_chat_template")
    initial = plugin.plan_append(session_config=config, **kwargs).prompt
    assert initial["prompt"].count(query) == 1
    messages = [message for message in template.call_args.args[0] if message["role"] != "assistant"]
    expected = plugin.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    plugin.max_model_len = len(plugin.processor.tokenizer(text=[expected])["input_ids"][0]) + 256
    actual = plugin.plan_append(session_config=config, **kwargs).prompt
    assert actual["prompt"].count(query) == 1
    assert "Red." not in actual["prompt"]
    plugin.max_model_len -= 1
    with pytest.raises(DuplexRuntimeConfigError, match="context budget"):
        plugin.plan_append(session_config=config, **kwargs)


def test_visual_summary_rebudget_tokenizations_scale_with_history_plus_summaries(mocker):
    plugin = JoyAIDuplexPlugin(lambda *args: "")
    plugin.processor = Processor()
    plugin.max_model_len = 65536
    plugin.max_images = 100
    config = {
        "conversation": [
            {"role": role, "content": f"{role} turn {index}: " + "text " * 10}
            for index in range(8)
            for role in ("user", "assistant")
        ],
        "joyai_new_images": [Image.new("RGB", (32, 32))] * 100,
        "joyai_new_image_tokens": [256] * 100,
    }
    kwargs = dict(
        request_id="budget",
        fence=DuplexFence("s"),
        runtime_config=plugin.runtime_config_for_update(DuplexSessionConfig(), {}),
        seq=1,
        turn_seq=1,
        payload={},
        final=True,
        sampling_params=SamplingParams(max_tokens=256),
    )
    expected = plugin.plan_append(session_config=config, **kwargs).prompt
    inputs = plugin.processor(text=[expected["prompt"]], images=expected["multi_modal_data"]["image"])
    plugin.max_model_len = len(inputs["input_ids"][0]) + 256
    summaries = [VisualSummary(i, i, 0.0, 0.0, "S" * 2000, 2000, chunk_index=i) for i in range(15)]
    tokenizer = mocker.spy(plugin.processor, "tokenizer")
    actual = plugin.plan_append(session_config={**config, "joyai_visual_summaries": summaries}, **kwargs).prompt
    assert actual["prompt"] == expected["prompt"]
    assert len(actual["multi_modal_data"]["image"]) == 100
    assert len(summaries) == 15
    # Trim history once, omit summaries against the protected instruction,
    # then rebudget recent history once against the remaining space.
    assert tokenizer.call_count <= 2 * len(config["conversation"]) + len(summaries) + 3


@pytest.mark.asyncio
@pytest.mark.parametrize("changed_query", [False, True])
@pytest.mark.parametrize("summary_ready", [False, True])
async def test_boundary_matches_experimental_archived_qa_and_standing_query(mocker, changed_query, summary_ready):
    backend = SimpleNamespace(
        generate=AsyncMock(side_effect=[("</response> Red.", None), ("</response> Blue.", None), ("</silence>", None)])
    )
    reference = InteractionSession("reference", InteractionConfig(persona="silent"), backend)
    h = await open_joyai(modalities=("text",))
    plugin = h.runner.plugin
    clock = [0.0]
    h.runner.model_state.visual_memory.start_time = 0.0
    mocker.patch(
        "vllm_omni.model_executor.models.joyai_vl_interaction.duplex.plugin.monotonic", side_effect=lambda: clock[0]
    )
    template = mocker.spy(plugin.processor, "apply_chat_template")
    query = "Tell me whenever the card color changes."
    summary = "A red card changed to blue."
    try:
        # Match the standing persona; the harness otherwise adds "Watch the pot."
        h.session.replace_runtime_config({**h.session.runtime_config, "instructions": None})
        await command(h, user_item("q1", [{"type": "input_text", "text": query}]))
        for index, color in enumerate(("red", "blue")):
            clock[0] = float(index)
            await reference.step(
                ["data:image/png;base64," + frame(color)], query=query if index == 0 else None, t=float(index)
            )
            await command(h, AppendAudio(video_frames=(frame(color),)))
            await command(h, Commit(final=True, create_response=True))
            request_id = submissions(h, auxiliary=False)[-1].context.request_id
            await h.deliver_and_settle(
                tts_output(request_id, text=f"</response> {color.title()}.", finished=True),
                stage_id=0,
                segment_finished=True,
            )
        # Both close the interactive chunk before consolidation completes;
        # native still owns the closed source until its summary is accepted.
        reference._policy.close_chunk()
        reference._policy.take_working_frames()
        reference.chunk = WorkingChunk()
        state = h.runner.model_state
        state.visual_memory.chunk_frames = 2
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        if summary_ready:
            reference._policy.brain.memory.mid_term_summaries.append(
                MidTermSummary(1, "0.0 seconds-1.0 seconds", summary)
            )
            assert plugin.apply_auxiliary_result(
                request=request, output=summary_output("private", summary), state=state
            )
        else:
            assert len(state.visual_memory.pending) == 2
        next_query = "What was the first color?" if changed_query else None
        if next_query:
            await command(h, user_item("q2", [{"type": "input_text", "text": next_query}]))
        clock[0] = 2.0
        await reference.step(["data:image/png;base64," + frame("green")], query=next_query, t=2.0)
        await command(h, AppendAudio(video_frames=(frame("green"),)))
        await command(h, Commit(final=True, create_response=True))
        expected = backend.generate.call_args.args[0]
        for message in expected:
            if isinstance(message["content"], list):
                message["content"] = [
                    {"type": "image"} if part["type"] == "image_url" else part for part in message["content"]
                ]
        assert template.call_args.args[0] == expected
        assert "[Q&A History]" in str(expected)
        assert "[Response@0.0 seconds] Red." in str(expected)
        assert "[Response@1.0 seconds] Blue." in str(expected)
        assert (USER_QUERY_HEADER + "\n" + query in expected[1]["content"][0]["text"]) is not changed_query
    finally:
        await reference.aclose()
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_partial_consolidation_keeps_closed_query_and_answer_in_prefix(mocker):
    h = await open_joyai(modalities=("text",))
    plugin = h.runner.plugin
    template = mocker.spy(plugin.processor, "apply_chat_template")
    query = "Describe all three colors."
    try:
        await command(
            h,
            user_item(
                "q1",
                [
                    {"type": "input_text", "text": query},
                    *[
                        {"type": "input_image", "image_url": "data:image/png;base64," + frame(color)}
                        for color in ("red", "blue", "green")
                    ],
                ],
            ),
        )
        await command(h, CreateResponse())
        request_id = submissions(h, auxiliary=False)[-1].context.request_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Red, blue, green.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        state = h.runner.model_state
        assert state.observations[0].frame_ids == (0, 1, 2)
        state.visual_memory.chunk_frames = 1
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert state.closed_frame_id == 2
        # Closing a multi-image observation is idempotent even when summary
        # requests consume smaller prefixes than that accepted observation.
        plugin._close_working_chunk(state)
        assert state.closed_frame_id == 2
        assert plugin.apply_auxiliary_result(
            request=request, output=summary_output("private", "A red card."), state=state
        )
        state.visual_memory.chunk_frames = 100
        assert state.observations[0].frame_ids == (1, 2)
        await command(h, AppendAudio(video_frames=(frame("yellow"),)))
        await command(h, Commit(final=True, create_response=True))
        messages = template.call_args.args[0]
        assert "[Q&A History]" in str(messages)
        assert str(messages).count("Red, blue, green.") == 1
        assert str(messages).count("'type': 'image'") == 1
        assert [frame.id for frame in state.visual_memory.pending] == [1, 2, 3]
    finally:
        await h.manager.shutdown()
