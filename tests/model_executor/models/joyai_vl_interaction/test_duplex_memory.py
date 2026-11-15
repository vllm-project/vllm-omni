# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import struct
from dataclasses import replace

import pytest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams

from tests.engine.duplex.test_session_runner import tts_output
from tests.model_executor.models.joyai_vl_interaction.test_duplex_plugin import frame, observe, open_joyai
from vllm_omni.engine.duplex.commands import (
    AckPlayback,
    AppendAudio,
    CancelInput,
    ClearInput,
    Commit,
    CreateItem,
    DeleteItem,
    Heartbeat,
    UpdateSession,
)
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.engine.duplex.messages import CloseDuplexSessionMessage
from vllm_omni.engine.duplex.plugin import duplex_history_digest, duplex_text_history
from vllm_omni.engine.duplex.session import memory_runtime
from vllm_omni.model_executor.models.joyai_vl_interaction.duplex.memory import VisualSummary
from vllm_omni.model_executor.stage_input_processors import joyai_vl_interaction as joyai_bridge

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def committed_turns(session, count=6):
    for index in range(count):
        for role, text in (("user", f"Remember cabinet {index}: K731"), ("assistant", f"Cabinet {index} recorded.")):
            message = {"role": role, "content": text}
            session.append_history_message(message)
            session.register_history_item(f"{role}-{index}", message)


async def start_summary(h):
    h.submit(Heartbeat())

    async def wait_for_submission():
        while not h.port.submissions:
            await asyncio.sleep(0.005)

    await asyncio.wait_for(wait_for_submission(), timeout=2)
    return h.port.submissions[-1]


def summary_output(request_id, text="Cabinet 0 has code K731.", finish_reason="stop"):
    return RequestOutput(
        request_id=request_id,
        prompt="",
        prompt_token_ids=[],
        prompt_logprobs=None,
        finished=True,
        outputs=[
            CompletionOutput(
                index=0,
                text=text,
                token_ids=[],
                cumulative_logprob=0.0,
                logprobs=None,
                finish_reason=finish_reason,
                stop_reason=None,
            )
        ],
    )


def admitted_large_exchange(h):
    plugin = h.runner.plugin
    question = "Remember the cabinet code K731. " * 450
    answer = "Recorded the cabinet code K731."
    kwargs = dict(
        request_id="long-question",
        fence=h.session.fence,
        session_config={"conversation": [{"role": "user", "content": question}]},
        runtime_config=h.session.runtime_config,
        seq=1,
        turn_seq=1,
        payload={},
        final=True,
        sampling_params=SamplingParams(max_tokens=len(answer)),
    )
    plan = plugin.plan_append(**kwargs)
    plugin.max_model_len = len(plugin.processor.tokenizer(text=[plan.prompt["prompt"]])["input_ids"][0]) + len(answer)
    plugin._prepare_memory_budget(plugin.processor, max_model_len=plugin.max_model_len)
    assert plugin.plan_append(**kwargs).prompt == plan.prompt
    return (("user", question), ("assistant", answer)) + tuple(
        (role, f"Recent {index} {role}") for index in range(4) for role in ("user", "assistant")
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_summary", [False, True])
async def test_oversized_first_exchange_summarizes_in_full_before_advancing_coverage(mocker, existing_summary):
    h = await open_joyai(modalities=("text",))
    try:
        history = admitted_large_exchange(h)
        covered = 2 if existing_summary else 0
        previous_summary = "Earlier cabinet fact." if existing_summary else ""
        if existing_summary:
            history = (("user", "An earlier question."), ("assistant", "An earlier answer."), *history)
        memory = h.runner.model_state.memory
        memory.text = previous_summary
        memory.covered_messages = covered
        memory.covered_digest = duplex_history_digest(history[:covered])
        for role, text in history:
            h.session.append_history_message({"role": role, "content": text})
        requests = []
        plugin = h.runner.plugin
        original = plugin.prepare_auxiliary_request

        async def capture(**kwargs):
            request = await original(**kwargs)
            requests.append(request)
            return request

        mocker.patch.object(plugin, "prepare_auxiliary_request", side_effect=capture)
        submission = await start_summary(h)
        transcript = "\n".join(
            f"[{index + 1}] {role}: {text}"
            for index, (role, text) in enumerate(history[covered : covered + 2], start=covered)
        )
        offset = 0
        for part in range(4):
            request = requests[-1]
            source = request.model_source
            assert request.history_length == covered + 2
            assert source is not None
            assert offset < source.end_offset <= len(transcript)
            assert (source.previous.offset if source.previous else 0) == offset
            # The fixture's chat template renders messages with repr().
            assert transcript[offset : source.end_offset].replace("\n", "\\n") in request.prompt["prompt"]
            assert (
                len(plugin.processor.tokenizer(text=[request.prompt["prompt"]])["input_ids"][0]) + 512
                <= plugin.max_model_len
            )
            if part:
                assert "Partial cabinet fact." in request.prompt["prompt"]
            else:
                assert previous_summary in request.prompt["prompt"]
                # A newer tail must neither invalidate nor restart this Q&A.
                h.session.append_history_message({"role": "user", "content": "A newer observation."})
            h.deliver(
                summary_output(submission.context.request_id, text="Partial cabinet fact."),
                stage_id=0,
                segment_finished=True,
            )
            await h.settle()
            offset = source.end_offset
            if source.complete:
                assert offset == len(transcript)
                assert memory.covered_messages == covered + 2
                assert memory.text == "Partial cabinet fact."
                assert memory.partial is None
                break
            assert memory.covered_messages == covered and memory.text == previous_summary
            assert memory.partial.offset == offset
            assert not plugin.auxiliary_source_matches(
                request=request, state=h.runner.model_state, history=h.session.history
            )
            config = plugin.prepare_prompt_config(
                {"conversation": list(h.session.history)}, state=h.runner.model_state, payload={}
            )
            assert config["joyai_summary"] == previous_summary

            async def next_part():
                while len(h.port.submissions) <= part + 1:
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(next_part(), timeout=2)
            submission = h.port.submissions[-1]
        else:
            pytest.fail("bounded excerpts did not finish the first exchange")

        completed = len(h.port.submissions)
        h.session.append_history_message({"role": "assistant", "content": "The newer observation is recorded."})
        h.submit(Heartbeat())

        async def next_exchange():
            while len(h.port.submissions) <= completed:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(next_exchange(), timeout=2)
        request = requests[-1]
        assert request.model_source.complete and request.model_source.previous is None
        assert request.history_length == covered + 4
        assert "Partial cabinet fact." in request.prompt["prompt"]
        assert "Recent 0 user" in request.prompt["prompt"]
        assert "Recent 0 assistant" in request.prompt["prompt"]
        h.deliver(
            summary_output(h.port.submissions[-1].context.request_id, text="Cabinet fact and later exchange."),
            stage_id=0,
            segment_finished=True,
        )
        await h.settle()
        assert memory.covered_messages == covered + 4
        assert memory.text == "Cabinet fact and later exchange."
        assert memory.partial is None
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["edit", "delete"])
async def test_partial_summary_retries_failed_excerpt_and_rejects_changed_source(mutation):
    h = await open_joyai(modalities=("text",))
    try:
        history = admitted_large_exchange(h)
        plugin, state = h.runner.plugin, h.runner.model_state

        async def prepare():
            return await plugin.prepare_auxiliary_request(
                history=history, runtime_config=h.session.runtime_config, state=state
            )

        first = await prepare()
        assert first is not None
        assert plugin.apply_auxiliary_result(request=first, output=summary_output("first"), state=state)
        progress = state.memory.partial
        assert progress is not None and state.memory.covered_messages == 0
        second = await prepare()
        assert not plugin.apply_auxiliary_result(
            request=second, output=summary_output("second", finish_reason="length"), state=state
        )
        assert state.memory.partial == progress
        retry = await prepare()
        assert retry.prompt != second.prompt
        assert retry.model_source.previous == progress
        assert progress.offset < retry.model_source.end_offset < second.model_source.end_offset
        assert state.memory.excerpt_limit is not None
        history = (("user", "Edited question"), *history[1:]) if mutation == "edit" else history[1:]
        conversation = tuple({"role": role, "content": text} for role, text in history)
        assert not plugin.auxiliary_source_matches(request=retry, state=state, history=conversation)
        plugin.prepare_prompt_config({"conversation": list(conversation)}, state=state, payload={})
        assert state.memory.partial is None
        assert state.memory.excerpt_limit is None
        assert state.memory.covered_messages == 0 and state.memory.text == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["length", "empty"])
async def test_rejected_second_fold_retries_less_evidence_then_completes(failure):
    h = await open_joyai(modalities=("text",))
    try:
        history = admitted_large_exchange(h)
        plugin, state = h.runner.plugin, h.runner.model_state

        async def prepare():
            return await plugin.prepare_auxiliary_request(
                history=history, runtime_config=h.session.runtime_config, state=state
            )

        first = await prepare()
        assert plugin.apply_auxiliary_result(request=first, output=summary_output("first"), state=state)
        progress = state.memory.partial
        assert progress is not None
        second = await prepare()
        assert not plugin.apply_auxiliary_result(
            request=second,
            output=summary_output(
                "second",
                text="" if failure == "empty" else "partial",
                finish_reason="length" if failure == "length" else "stop",
            ),
            state=state,
        )
        limit = state.memory.excerpt_limit
        assert limit is not None and state.memory.partial == progress
        history = (*history, ("user", "A newer observation must not reset the rejected prefix's budget."))
        retry = await prepare()
        assert retry.model_source.previous == progress
        assert progress.offset < retry.model_source.end_offset < second.model_source.end_offset
        assert retry.prompt != second.prompt
        for part in range(20):
            source = retry.model_source
            assert plugin.apply_auxiliary_result(
                request=retry, output=summary_output(f"fold-{part}", "Retained cabinet code K731."), state=state
            )
            if source.complete:
                break
            assert state.memory.covered_messages == 0 and state.memory.text == ""
            assert state.memory.partial.offset == source.end_offset
            assert state.memory.excerpt_limit == limit
            retry = await prepare()
            assert retry.model_source.end_offset - retry.model_source.previous.offset <= limit.max_chars
        else:
            pytest.fail("smaller excerpts did not finish the rejected exchange")
        assert state.memory.covered_messages == 2
        assert state.memory.text == "Retained cabinet code K731."
        assert state.memory.partial is None and state.memory.excerpt_limit is None
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_tiny_rejected_excerpt_preserves_labels_and_obeys_token_budget():
    h = await open_joyai(modalities=("text",))
    try:
        plugin, state = h.runner.plugin, h.runner.model_state
        history = (("user", "K"), ("assistant", "A")) + tuple(
            (role, f"Recent {index} {role}") for index in range(4) for role in ("user", "assistant")
        )

        async def prepare():
            return await plugin.prepare_auxiliary_request(
                history=history, runtime_config=h.session.runtime_config, state=state
            )

        request = await prepare()
        for _ in range(2):
            assert not plugin.apply_auxiliary_result(
                request=request, output=summary_output("rejected", finish_reason="length"), state=state
            )
            request = await prepare()
            assert request is not None
        assert state.memory.excerpt_limit.max_chars < len("[1] user: ")
        assert request.model_source.end_offset == len("[1] user: K")
        assert not request.model_source.complete
        assert "[1] user: K" in request.prompt["prompt"]

        original_budget = plugin.max_model_len
        minimum_budget = (
            len(plugin.processor.tokenizer(text=[request.prompt["prompt"]])["input_ids"][0])
            + request.sampling_params.max_tokens
        )
        plugin.max_model_len = minimum_budget - 1
        assert await prepare() is None
        plugin.max_model_len = minimum_budget
        retry = await prepare()
        assert retry is not None and retry.prompt == request.prompt
        assert plugin.apply_auxiliary_result(request=retry, output=summary_output("first", "K"), state=state)
        assert state.memory.partial.offset == len("[1] user: K")
        assert state.memory.covered_messages == 0

        plugin.max_model_len = original_budget
        final = await prepare()
        assert final is not None and final.model_source.complete
        assert "[2] assistant: A" in final.prompt["prompt"]
        assert plugin.apply_auxiliary_result(request=final, output=summary_output("final", "K: A"), state=state)
        assert state.memory.covered_messages == 2
        assert state.memory.partial is None and state.memory.excerpt_limit is None
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_rejected_whole_prefix_retries_as_smaller_excerpts():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        history = duplex_text_history(h.session.history)
        plugin, state = h.runner.plugin, h.runner.model_state
        request = await plugin.prepare_auxiliary_request(
            history=history, runtime_config=h.session.runtime_config, state=state
        )
        assert request.model_source.complete and request.model_source.previous is None
        assert not plugin.apply_auxiliary_result(
            request=request, output=summary_output("full", finish_reason="length"), state=state
        )
        retry = await plugin.prepare_auxiliary_request(
            history=history, runtime_config=h.session.runtime_config, state=state
        )
        assert retry.history_digest == request.history_digest
        assert retry.history_length == request.history_length
        assert retry.prompt != request.prompt and not retry.model_source.complete
        assert retry.model_source.end_offset <= request.model_source.end_offset // 2
        assert state.memory.partial is None and state.memory.covered_messages == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["length", "empty"])
async def test_runner_retries_rejected_fold_after_committed_history_progress(mocker, failure):
    h = await open_joyai(modalities=("text",))
    try:
        history = admitted_large_exchange(h)
        for role, text in history:
            h.session.append_history_message({"role": role, "content": text})
        plugin, memory = h.runner.plugin, h.runner.model_state.memory
        requests = []
        original = plugin.prepare_auxiliary_request

        async def capture(**kwargs):
            request = await original(**kwargs)
            if request is not None:
                requests.append(request)
            return request

        async def submission_at(index):
            async def wait_for_submission():
                while len(h.port.submissions) <= index:
                    await asyncio.sleep(0.005)

            await asyncio.wait_for(wait_for_submission(), timeout=2)
            return h.port.submissions[index]

        mocker.patch.object(plugin, "prepare_auxiliary_request", side_effect=capture)
        first = await start_summary(h)
        h.deliver(summary_output(first.context.request_id), stage_id=0, segment_finished=True)
        second = await submission_at(1)
        progress = memory.partial
        assert progress is not None and memory.covered_messages == 0
        rejected = requests[1]
        assert rejected.model_source.previous == progress
        h.deliver(
            summary_output(
                second.context.request_id,
                text="" if failure == "empty" else "partial",
                finish_reason="length" if failure == "length" else "stop",
            ),
            stage_id=0,
            segment_finished=True,
        )
        await h.run(Heartbeat())
        assert len(h.port.submissions) == 2
        assert memory.partial == progress and memory.covered_messages == 0
        assert memory.excerpt_limit is not None

        h.submit(
            CreateItem(
                item={
                    "id": "new-committed-context",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "The newer observation is recorded."}],
                }
            )
        )
        retry = await submission_at(2)
        adapted = requests[2]
        assert adapted.history_digest == rejected.history_digest
        assert adapted.history_length == rejected.history_length == 2
        assert adapted.model_source.previous == progress
        assert progress.offset < adapted.model_source.end_offset < rejected.model_source.end_offset
        assert len(retry.prompt["prompt"]) < len(second.prompt["prompt"])
        assert memory.partial == progress and memory.covered_messages == 0
        assert "auxiliary-" in retry.context.request_id
        assert retry.context.request_id not in plugin.data_plane.requests
        assert h.session.active_response_id is None

        for index in range(2, 22):
            request = requests[index]
            h.deliver(
                summary_output(retry.context.request_id, text="Retained cabinet code K731."),
                stage_id=0,
                segment_finished=True,
            )
            if request.model_source.complete:
                await h.settle()
                break
            retry = await submission_at(index + 1)
            assert memory.covered_messages == 0
            assert memory.partial.offset == request.model_source.end_offset
        else:
            pytest.fail("runner did not finish the adapted summary prefix")
        assert memory.covered_messages == 2
        assert memory.covered_digest == duplex_history_digest(duplex_text_history(h.session.history)[:2])
        assert memory.text == "Retained cabinet code K731."
        assert memory.partial is None and memory.excerpt_limit is None
        assert not any(event.to_realtime()["type"].startswith("response.") for event in h.events)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("question", ["cabinet-code " * 800, "K731" * 2200])
async def test_summary_excerpt_boundaries_and_continued_speaker_labels(question):
    h = await open_joyai(modalities=("text",))
    try:
        plugin, state = h.runner.plugin, h.runner.model_state
        plugin.max_model_len = 4096
        history = (("user", question), ("assistant", "Recorded.")) + tuple(
            (role, f"Recent {index} {role}") for index in range(4) for role in ("user", "assistant")
        )
        first = await plugin.prepare_auxiliary_request(
            history=history, runtime_config=h.session.runtime_config, state=state
        )
        source = first.model_source
        assert source is not None and not source.complete
        assert "[1] user: " in first.prompt["prompt"]
        transcript = f"[1] user: {question}\n[2] assistant: Recorded."
        if " " in question:
            assert transcript[source.end_offset].isspace()
        else:
            assert transcript[source.end_offset] in "K731"
        assert plugin.apply_auxiliary_result(request=first, output=summary_output("first"), state=state)
        second = await plugin.prepare_auxiliary_request(
            history=history, runtime_config=h.session.runtime_config, state=state
        )
        assert second.model_source.previous.offset == source.end_offset
        assert "[1] user: (continued) " in second.prompt["prompt"]
        assert (
            len(plugin.processor.tokenizer(text=[second.prompt["prompt"]])["input_ids"][0]) + 512
            <= plugin.max_model_len
        )
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("stage_id,stage_count", [(0, 3), (1, 4)])
async def test_summary_uses_private_same_stage_request_and_preserves_new_tail(stage_id, stage_count):
    h = await open_joyai(modalities=("text",), asr=stage_id == 1)
    try:
        assert h.port.stage_count == stage_count
        committed_turns(h.session)
        initial_seq = h.session.input_seq
        submission = await start_summary(h)
        context = submission.context
        assert context.stage_id == context.final_stage_id == stage_id
        assert not submission.resumable and not submission.already_submitted
        assert "auxiliary-" in context.request_id
        assert context.request_id not in h.runner.plugin.data_plane.requests
        # New observations must not starve consolidation of an unchanged prefix.
        h.session.append_history_message({"role": "user", "content": "Look at the window now."})
        h.deliver(summary_output(context.request_id), stage_id=stage_id, segment_finished=True)
        events = await h.settle()
        assert h.runner.model_state.memory.text == "Cabinet 0 has code K731."
        assert h.runner.model_state.memory.covered_messages == 4
        assert h.session.history[-1]["content"] == "Look at the window now."
        assert h.session.active_response_id is None
        assert h.session.input_seq == initial_seq
        assert not any(event.to_realtime()["type"].startswith("response.") for event in events)
        assert h.runner.model_state.last_response_text == ""
        assert ([context.request_id], True) in h.port.cleanups
        assert context.request_id not in h.session.resource_request_ids()
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["edit", "delete", "epoch", "instructions"])
async def test_summary_rejects_changed_source_or_context(mutation):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        if mutation == "edit":
            h.session.register_history_item("assistant-0", {"role": "assistant", "content": "Only the played prefix."})
        elif mutation == "delete":
            h.session.delete_history_item("user-0")
        elif mutation == "instructions":
            revision = h.session.config_revision
            await h.run(UpdateSession(patch={"instructions": "New instructions"}))
            assert h.session.config_revision == revision + 1
        else:
            h.session.epoch += 1
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.text == ""
        assert not any(event.to_realtime()["type"].startswith("response.") for event in h.events)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["error", "timeout", "empty", "truncated"])
async def test_failed_summary_keeps_memory_and_does_not_retry_same_prefix(failure):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        memory = h.runner.model_state.memory
        memory.text = "Earlier cabinet fact."
        memory.covered_messages = 2
        memory.covered_digest = duplex_history_digest(duplex_text_history(h.session.history)[:2])
        if failure == "timeout":
            original = h.runner.plugin.prepare_auxiliary_request

            async def short_deadline(**kwargs):
                request = await original(**kwargs)
                return replace(request, timeout_s=0.02) if request is not None else None

            h.runner.plugin.prepare_auxiliary_request = short_deadline
        submission = await start_summary(h)
        request_id = submission.context.request_id
        if failure == "error":
            h.runner.on_stage_failure(0, RuntimeError("synthetic failure"), request_id=request_id)
        elif failure != "timeout":
            h.deliver(
                summary_output(
                    request_id,
                    text="" if failure == "empty" else "partial",
                    finish_reason="length" if failure == "truncated" else "stop",
                ),
                stage_id=0,
                segment_finished=True,
            )
        await h.settle()
        assert memory.text == "Earlier cabinet fact."
        await h.run(Heartbeat())
        assert len(h.port.submissions) == 1
        assert not any(event.to_realtime()["type"].startswith("response.") for event in h.events)
        assert ([request_id], True) in h.port.cleanups
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_failed_full_summary_chunk_retries_after_committed_progress(mocker):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=8)
        generation = h.session.config_generation
        submit = mocker.spy(h.port, "submit")
        h.port.fail_submit = RuntimeError("transient submission failure")
        await h.run(Heartbeat())
        assert submit.call_count == 1
        first_prompt = submit.call_args.args[0].prompt
        assert "Cabinet 3 recorded." in first_prompt["prompt"]
        assert "Cabinet 4 recorded." not in first_prompt["prompt"]

        for index in range(8, 11):
            await h.run(Heartbeat())
            assert submit.call_count == index - 7
            assert h.session.resource_request_ids(submitted=True) == []
            for role, text in (("user", f"New cabinet {index}."), ("assistant", f"Noted cabinet {index}.")):
                h.session.append_history_message({"role": role, "content": text})
            if index == 10:
                h.port.fail_submit = None
            await h.run(Heartbeat())
            assert submit.call_count == index - 6
            assert submit.call_args.args[0].prompt == first_prompt

        assert len(h.port.submissions) == 1
        retry = h.port.submissions[0]
        assert h.session.config_generation == generation
        h.deliver(summary_output(retry.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.covered_messages == 8
        assert h.runner.model_state.memory.text == "Cabinet 0 has code K731."
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_camera_only_commits_do_not_displace_eight_recent_text_messages():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=4)
        for _ in range(8):
            h.session.commit_audio_input()
        history = duplex_text_history(h.session.history)
        assert len(history) == 16
        assert sum(bool(text) for _, text in history) == 8
        await h.run(Heartbeat())
        assert h.port.submissions == []
        assert h.runner.model_state.memory.covered_messages == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_summary", [False, True])
@pytest.mark.parametrize("later_turns", [6, 10, 20])
async def test_summary_advances_past_silent_camera_turns(existing_summary, later_turns):
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.memory
        if existing_summary:
            h.session.append_history_message({"role": "user", "content": "Remember the old cabinet."})
            h.session.append_history_message({"role": "assistant", "content": "Recorded the old cabinet."})
            memory.text = "The old cabinet is important."
            memory.covered_messages = 2
            memory.covered_digest = duplex_history_digest(duplex_text_history(h.session.history))
        covered = memory.covered_messages
        for _ in range(8):
            request_id = await observe(h)
            h.deliver(tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True)
            await h.settle()
        history = duplex_text_history(h.session.history)
        assert history[covered:] == (("user", ""),) * 8
        committed_turns(h.session, count=later_turns)
        history = duplex_text_history(h.session.history)
        submissions = len(h.port.submissions)
        await h.run(Heartbeat())
        assert len(h.port.submissions) == submissions + 1
        submission = h.port.submissions[-1]
        assert "auxiliary-" in submission.context.request_id
        expected_text = min(8, later_turns * 2 - 8)
        expected_end = covered + 8 + expected_text
        assert f"Cabinet {expected_text // 2 - 1} recorded." in submission.prompt["prompt"]
        assert f"Remember cabinet {expected_text // 2}:" not in submission.prompt["prompt"]
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert memory.covered_messages == expected_end
        assert memory.covered_digest == duplex_history_digest(history[:expected_end])
        assert sum(bool(text) for _, text in history[expected_end:]) >= 8
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_text_cutoff_preserves_ledger_indices_and_empty_prefix_fingerprints():
    h = await open_joyai(modalities=("text",))
    try:
        for index in range(6):
            for role, text in (("user", f"Question {index}."), ("assistant", f"Answer {index}.")):
                h.session.append_history_message({"role": role, "content": text})
            committed = h.session.commit_audio_input()
            h.session.register_history_item(f"camera-{index}", committed.message)
        for _ in range(8):
            h.session.commit_audio_input()
        history = duplex_text_history(h.session.history)
        submission = await start_summary(h)
        assert "Answer 1." in submission.prompt["prompt"]
        assert "Question 2." not in submission.prompt["prompt"]
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        memory = h.runner.model_state.memory
        assert memory.covered_messages == 5
        assert memory.covered_digest == duplex_history_digest(history[:5])
        assert sum(role in {"user", "assistant"} and bool(text) for role, text in history[5:]) == 8
        assert len(h.port.submissions) == 1
        h.session.register_history_item("camera-0", {"role": "user", "content": "Corrected camera transcript."})
        config = h.runner.plugin.prepare_prompt_config(
            {"conversation": list(h.session.history)}, state=h.runner.model_state, payload={}
        )
        assert config["joyai_summary"] == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("existing_summary", [False, True])
async def test_summary_catches_up_past_text_bearing_silent_turns(existing_summary):
    h = await open_joyai(modalities=("text",))
    try:
        memory = h.runner.model_state.memory
        if existing_summary:
            committed_turns(h.session, count=1)
            memory.text = "The cabinet code is K731."
            memory.covered_messages = 2
            memory.covered_digest = duplex_history_digest(duplex_text_history(h.session.history))
        covered = memory.covered_messages
        for index in range(8):
            await h.run(
                CreateItem(
                    item={
                        "id": f"silent-{index}",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": f"Remember silent fact {index}."}],
                    }
                )
            )
            request_id = await observe(h)
            await h.deliver_and_settle(
                tts_output(request_id, text="</silence>", finished=True), stage_id=0, segment_finished=True
            )
        history = duplex_text_history(h.session.history)
        assert [text for role, text in history[covered:] if text] == [
            f"Remember silent fact {index}." for index in range(8)
        ]
        committed_turns(h.session, count=10)
        history = duplex_text_history(h.session.history)
        await h.run(Heartbeat())
        for _ in range(8):
            if memory.covered_messages >= len(history) - 8:
                break
            submission = h.port.submissions[-1]
            assert "auxiliary-" in submission.context.request_id
            previous = memory.covered_messages
            h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
            await h.settle()
            assert memory.covered_messages > previous
        assert memory.covered_messages == len(history) - 8
        assert memory.covered_digest == duplex_history_digest(history[:-8])
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("shrink_to_budget", [False, True])
@pytest.mark.parametrize("empty_observations", [0, 2])
async def test_summary_boundary_keeps_question_with_answer(shrink_to_budget, empty_observations):
    h = await open_joyai(modalities=("text",))
    try:
        plugin = h.runner.plugin
        silent = tuple(("user", f"Silent fact {index}.") for index in range(6 if shrink_to_budget else 7))
        tail = tuple((role, f"Recent {index} {role}.") for index in range(4) for role in ("user", "assistant"))
        question, answer = "What is the cabinet code?", "The cabinet code is CODE_K731."
        if shrink_to_budget:
            # Fit the silent prefix and question alone, but not its answer.
            # Shrinking the chunk must back off over the whole pair.
            prefix_request = await plugin.prepare_auxiliary_request(
                history=silent + (("user", question),) + tail,
                runtime_config=h.session.runtime_config,
                state=h.runner.model_state,
            )
            plugin.max_model_len = (
                len(plugin.processor.tokenizer(text=[prefix_request.prompt["prompt"]])["input_ids"][0])
                + prefix_request.sampling_params.max_tokens
            )
        history = (
            silent + (("user", question),) + (("user", ""),) * empty_observations + (("assistant", answer),) + tail
        )
        request = await plugin.prepare_auxiliary_request(
            history=history, runtime_config=h.session.runtime_config, state=h.runner.model_state
        )
        assert request is not None
        assert request.history_length == len(silent)
        assert question not in request.prompt["prompt"] and answer not in request.prompt["prompt"]
        config = {
            "conversation": [{"role": role, "content": text} for role, text in history],
            "joyai_summary_messages": request.history_length,
            "joyai_summary": "The older silent facts.",
        }
        # The next interactive prompt must retain both halves of the pair.
        plugin.max_model_len = 16384
        plan = plugin.plan_append(
            request_id="boundary-check",
            fence=h.session.fence,
            session_config=config,
            runtime_config=h.session.runtime_config,
            seq=1,
            turn_seq=1,
            payload={},
            final=True,
            sampling_params=SamplingParams(max_tokens=256),
        )
        assert question in plan.prompt["prompt"] and answer in plan.prompt["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("preempt", [False, True])
async def test_close_during_auxiliary_cleanup_preserves_cleanup_ownership(monkeypatch, preempt):
    h = await open_joyai(modalities=("text",))
    entered, release = asyncio.Event(), asyncio.Event()
    original_cleanup = h.port.cleanup
    closing = None
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        request_id = submission.context.request_id

        async def blocked_cleanup(request_ids, *, abort=False):
            entered.set()
            await release.wait()
            await original_cleanup(request_ids, abort=abort)

        monkeypatch.setattr(h.port, "cleanup", blocked_cleanup)
        if preempt:
            h.runner.memory.preempt_for_submission()
        else:
            h.deliver(summary_output(request_id), stage_id=0, segment_finished=True)
        await asyncio.wait_for(entered.wait(), 2)
        closing = asyncio.create_task(h.runner.close("client close"))
        await asyncio.sleep(0.02)
        # Repeated close and runner shutdown may both cancel the owned task.
        h.runner.memory.close()
        release.set()
        await asyncio.wait_for(closing, 2)
        assert ([request_id], True) in h.port.cleanups
        assert request_id not in h.session.resource_request_ids()
        assert request_id not in h.manager._request_index
        assert h.runner.memory._request_id is None
        assert h.runner.memory._output is None
    finally:
        release.set()
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        monkeypatch.setattr(h.port, "cleanup", original_cleanup)
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("tight_budget", [False, True])
async def test_asr_uses_bounded_memory_and_preserves_initial_instruction(monkeypatch, tight_budget):
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        instruction = "Always warn me when the kettle boils."
        h.session.replace_runtime_config({**h.session.runtime_config, "initial_user_text": instruction})
        committed_turns(h.session, count=60)
        memory = h.runner.model_state.memory
        memory.text = "The cabinet contains a blue mug."
        memory.covered_messages = 80
        memory.covered_digest = duplex_history_digest(duplex_text_history(h.session.history)[:80])
        h.runner.model_state.visual_memory.mid_term.append(
            VisualSummary(0, 7, 0.0, 7.0, "The kettle is on the stove.", 20, chunk_index=0)
        )
        monkeypatch.setattr(joyai_bridge, "cached_processor_from_config", lambda config: h.runner.plugin.processor)
        await h.run(AppendAudio(audio=struct.pack("<hh", 1000, 2000), format="pcm16", sample_rate_hz=16000))
        await h.run(Commit(final=True, create_response=True))
        submission = h.port.submissions[-1]
        assert submission.context.stage_id == 0
        # The stage port materializes the immutable submission mapping before
        # passing its raw prompt to the stage input processor.
        raw_prompt = dict(submission.prompt)
        info = raw_prompt["additional_information"]
        transcript = "Is the water boiling now?"
        if tight_budget:
            # Fit the standing instruction and actual transcript, with no room
            # for old history. This uses the same tokenizer as the handoff.
            minimum = h.runner.plugin.processor.apply_chat_template(
                [
                    {"role": "system", "content": info["joyai_prompt_context"]["system"]},
                    {"role": "user", "content": instruction},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": f"{joyai_bridge.USER_QUERY_HEADER}\n{transcript}"}],
                    },
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            info["joyai_prompt_budget"] = len(h.runner.plugin.processor.tokenizer(text=[minimum])["input_ids"][0]) + 32
        (result,) = joyai_bridge.asr_to_joyai([summary_output("asr", text=transcript)], raw_prompt)
        prompt = result["prompt"]
        assert instruction in prompt
        assert transcript in prompt
        assert len(h.runner.plugin.processor.tokenizer(text=[prompt])["input_ids"][0]) <= info["joyai_prompt_budget"]
        assert len(info["joyai_prompt_context"]["history"]) <= 16
        if not tight_budget:
            assert "The cabinet contains a blue mug." in prompt
            assert "The kettle is on the stove." in prompt
            assert "Remember cabinet 0:" not in prompt
        assert raw_prompt["additional_information"]["tts_speaker"] == "Vivian"
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_oldest_visual", [False, True])
async def test_asr_evicts_memory_blocks_before_rebudgeting_history(monkeypatch, drop_oldest_visual):
    from vllm_omni.experimental.fullduplex.joyvl.decision.prompts import VIDEO_HISTORY_HEADER

    h = await open_joyai(asr=True, modalities=("text",))
    try:
        instruction = "Always warn me when the kettle boils."
        h.session.replace_runtime_config({**h.session.runtime_config, "initial_user_text": instruction})
        committed_turns(h.session, count=12)
        memory = h.runner.model_state.memory
        memory.text = "Older text summary. " * 400
        memory.covered_messages = 8
        memory.covered_digest = duplex_history_digest(duplex_text_history(h.session.history)[:8])
        visuals = [
            VisualSummary(
                0, 7, 0.0, 7.0, "The cabinet is closed. " * (200 if drop_oldest_visual else 1), 20, chunk_index=0
            ),
            VisualSummary(8, 15, 8.0, 15.0, "The kettle is on the stove.", 20, chunk_index=1),
        ]
        h.runner.model_state.visual_memory.mid_term.extend(visuals)
        monkeypatch.setattr(joyai_bridge, "cached_processor_from_config", lambda config: h.runner.plugin.processor)
        await h.run(AppendAudio(audio=struct.pack("<hh", 1000, 2000), format="pcm16", sample_rate_hz=16000))
        await h.run(Commit(final=True, create_response=True))
        raw_prompt = dict(h.port.submissions[-1].prompt)
        info = raw_prompt["additional_information"]
        context = info["joyai_prompt_context"]
        transcript = "Is the water boiling now?"
        retained_memory = VIDEO_HISTORY_HEADER + "\n\n".join(
            f"<{entry.start_s:.1f} seconds-{entry.end_s:.1f} seconds>\n{entry.text}"
            for entry in visuals[int(drop_oldest_visual) :]
        )
        # Fit remaining visual memory plus the latest Q&A, leaving no space
        # for the text summary (or the oversized oldest visual block).
        expected = h.runner.plugin.processor.apply_chat_template(
            [
                {"role": "system", "content": context["system"]},
                {"role": "user", "content": retained_memory},
                {"role": "user", "content": instruction},
                *context["history"][-2:],
                {
                    "role": "user",
                    "content": [{"type": "text", "text": f"{joyai_bridge.USER_QUERY_HEADER}\n{transcript}"}],
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        info["joyai_prompt_budget"] = len(h.runner.plugin.processor.tokenizer(text=[expected])["input_ids"][0])
        (result,) = joyai_bridge.asr_to_joyai([summary_output("asr", text=transcript)], raw_prompt)
        prompt = result["prompt"]
        assert instruction in prompt and transcript in prompt
        assert "Older text summary." not in prompt
        assert "The kettle is on the stove." in prompt
        assert ("The cabinet is closed." in prompt) is not drop_oldest_visual
        assert "Remember cabinet 11:" in prompt and "Cabinet 11 recorded." in prompt
        assert len(h.runner.plugin.processor.tokenizer(text=[prompt])["input_ids"][0]) <= info["joyai_prompt_budget"]
        # Eviction changes only this prompt, not the stored model memory.
        assert memory.text.startswith("Older text summary.")
        assert h.runner.model_state.visual_memory.mid_term == visuals
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("pending_text", [False, True])
async def test_clear_committed_audio_resumes_memory_and_preserves_text_intent(pending_text):
    h = await open_joyai(asr=True, modalities=("text",))
    try:
        await h.run(AppendAudio(audio=struct.pack("<hh", 1000, 2000), format="pcm16", sample_rate_hz=16000))
        await h.run(Commit(final=True, create_response=False))
        assert h.runner.model_state.committed_audio_payload is not None
        assert h.runner.memory._input_pending
        committed_turns(h.session)
        if pending_text:
            await h.run(
                CreateItem(
                    item={
                        "id": "pending-question",
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "Where is the kettle?"}],
                    }
                )
            )
        await h.run(ClearInput())
        assert h.runner.model_state.committed_audio_payload is None
        assert not h.runner.memory._input_pending
        if pending_text:
            await h.run(Heartbeat())
            assert h.runner.memory._pending_user_items == {"pending-question"}
            assert h.port.submissions == []
            await h.run(DeleteItem(item_id="pending-question"))
        submission = await start_summary(h)
        assert "auxiliary-" in submission.context.request_id
        assert submission.context.stage_id == 1
        h.deliver(summary_output(submission.context.request_id), stage_id=1, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.covered_messages > 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_close_cancels_auxiliary_and_late_result_cannot_write():
    h = await open_joyai(modalities=("text",))
    committed_turns(h.session)
    submission = await start_summary(h)
    task = h.runner.memory._task
    await h.manager.shutdown()
    assert task.done()
    assert ([submission.context.request_id], True) in h.port.cleanups
    assert h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
    assert h.runner.model_state.memory.text == ""


@pytest.mark.asyncio
async def test_auxiliary_failure_does_not_finish_concurrent_response():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        response_id = h.session.begin_response()
        h.runner.on_stage_failure(0, RuntimeError("synthetic failure"), request_id=submission.context.request_id)
        events = await h.settle()
        assert h.session.active_response_id == response_id
        assert not any(event.to_realtime()["type"] in {"response.done", "error"} for event in events)
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_summary_excludes_unplayed_speech_and_rejects_late_full_ack():
    h = await open_joyai()
    try:
        h.session.append_history_message({"role": "user", "content": "Observe the cabinet."})
        request_id = await observe(h)
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(request_id, text="</response> Unheard secret is Z984.", finished=True),
            stage_id=0,
            segment_finished=True,
        )
        await h.deliver_and_settle(tts_output(request_id, finished=True), stage_id=2, segment_finished=True)
        committed_turns(h.session)
        # The normal observation is already recorded in the stage-port harness.
        h.port.submissions.clear()
        submission = await start_summary(h)
        assert "Unheard secret" not in submission.prompt["prompt"]
        h.submit(AckPlayback(response_id=response_id, played_ms=1000))
        await h.settle(timeout_s=0.1)
        assert any("Unheard secret" in str(message.get("content")) for message in h.session.history)
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.text == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_failed_auxiliary_cleanup_retains_one_resource_and_blocks_more_jobs():
    h = await open_joyai(modalities=("text",))
    original_cleanup = h.port.cleanup
    try:
        committed_turns(h.session)
        submission = await start_summary(h)

        async def fail_cleanup(*args, **kwargs):
            raise RuntimeError("synthetic cleanup failure")

        h.port.cleanup = fail_cleanup
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert submission.context.request_id in h.session.resource_request_ids(submitted=True)
        committed_turns(h.session, count=2)
        await h.run(Heartbeat())
        assert len(h.port.submissions) == 1
    finally:
        h.port.cleanup = original_cleanup
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_close", [False, True])
async def test_input_cancel_retains_auxiliary_after_failed_aborts_until_close_retry(fail_close):
    h = await open_joyai(modalities=("text",))
    original_cleanup = h.port.cleanup
    failures_remaining = 2

    async def fail_cleanup(request_ids: list[str], *, abort: bool = False) -> None:
        nonlocal failures_remaining
        await original_cleanup(request_ids, abort=abort)
        if failures_remaining:
            failures_remaining -= 1
            raise RuntimeError("synthetic abort failure")

    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        request_id = submission.context.request_id
        cancelled_request_ids = h.session.resource_request_ids()
        assert h.session.resource_request_ids(submitted=True) == [request_id]
        old_epoch = h.session.epoch
        task = h.runner.memory._task
        h.port.cleanup = fail_cleanup

        events = await h.run(CancelInput())
        assert any(event.to_realtime()["type"] == "input_audio_buffer.cleared" for event in events)
        assert h.session.epoch == old_epoch + 1
        assert task is not None and task.done()
        # Open reserves the normal stage-0 request without submitting it. The
        # cancellation batch includes it; the auxiliary's retry contains only
        # the submitted summary request.
        assert h.port.cleanups == [(cancelled_request_ids, True), ([request_id], True)]
        assert failures_remaining == 0
        assert h.session.resource_request_ids(submitted=True) == [request_id]
        await h.run(Heartbeat())
        assert len(h.port.submissions) == 1

        failures_remaining = int(fail_close)
        await h.manager.close(CloseDuplexSessionMessage(control_id="close-memory", session_id=h.session.session_id))
        assert [abort for ids, abort in h.port.cleanups if request_id in ids] == [True] * 3
        if fail_close:
            assert h.manager._closing[h.session.session_id].submitted_request_ids == (request_id,)
            assert await h.manager.reap_expired() == 1
            assert [abort for ids, abort in h.port.cleanups if request_id in ids] == [True] * 4
        assert h.session.session_id not in h.manager._closing
    finally:
        h.port.cleanup = original_cleanup
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_input_cancel_cleans_auxiliary_only_once_when_abort_succeeds():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        request_id = submission.context.request_id
        cancelled_request_ids = h.session.resource_request_ids()
        await h.run(CancelInput())
        assert h.port.cleanups == [(cancelled_request_ids, True)]
        assert request_id not in h.session.resource_request_ids()
        assert h.runner.model_state.memory.text == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_memory_is_bounded_and_instructions_are_outside_summary():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=20)
        plugin, state = h.runner.plugin, h.runner.model_state
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        assert request.history_length == 8
        assert request.sampling_params.max_tokens == 512
        assert "Watch the pot." not in request.prompt["prompt"]
        plugin.apply_auxiliary_result(request=request, output=summary_output("private"), state=state)
        prepared = plugin.prepare_prompt_config({"conversation": list(h.session.history)}, state=state, payload={})
        assert prepared["joyai_summary"] == "Cabinet 0 has code K731."
        h.session.delete_history_item("user-0")
        prepared = plugin.prepare_prompt_config({"conversation": list(h.session.history)}, state=state, payload={})
        assert prepared["joyai_summary"] == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("image_count", [0, 4])
@pytest.mark.parametrize("dropped_pairs", [0, 1])
async def test_dropped_summary_rebudgets_recent_history(image_count, dropped_pairs):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        plugin, state = h.runner.plugin, h.runner.model_state
        request = await plugin.prepare_auxiliary_request(
            history=duplex_text_history(h.session.history), runtime_config=h.session.runtime_config, state=state
        )
        plugin.apply_auxiliary_result(request=request, output=summary_output("private", "S" * 479), state=state)
        assert state.memory.covered_messages == 4
        saved_memory = replace(state.memory)
        # Keep the removable tail cheaper than the accepted summary under the
        # fake character tokenizer, including each historical query's wrapper.
        conversation = [dict(message) for message in h.session.history[:8]]
        for index, message in enumerate(conversation[4:]):
            message["content"] = f"tail-{index}"
        conversation.append({"role": "user", "content": "Q"})
        prepared = plugin.prepare_prompt_config(
            {"conversation": conversation}, state=state, payload={"video_frames": [frame()] * image_count}
        )
        kwargs = dict(
            request_id="budget-test",
            fence=DuplexFence("s"),
            runtime_config=h.session.runtime_config,
            seq=1,
            turn_seq=1,
            payload={},
            final=True,
            sampling_params=SamplingParams(max_tokens=256),
        )

        def plan(config):
            return plugin.plan_append(session_config=config, **kwargs).prompt

        def token_count(prompt):
            inputs = plugin.processor(text=[prompt["prompt"]], images=prompt["multi_modal_data"].get("image"))
            return len(inputs["input_ids"][0])

        assert saved_memory.text in plan(prepared)["prompt"]  # Short queries still use the accepted summary.
        short_prompt = plan(
            {
                **prepared,
                "joyai_summary": "",
                "joyai_summary_messages": saved_memory.covered_messages + 2 * dropped_pairs,
            }
        )
        budget = plugin.max_model_len - 256
        # Size the query to fit the chosen recent tail exactly, so template
        # formatting changes cannot silently change which pairs should fit.
        conversation[-1]["content"] = "Q" * (budget - token_count(short_prompt) + 1)
        expected = plan({**prepared, "joyai_summary": ""})
        actual = plan(prepared)
        assert actual["prompt"] == expected["prompt"]
        assert actual["prompt"].count("tail-") == 4 - 2 * dropped_pairs
        assert actual["prompt"].count(conversation[-1]["content"]) == 1
        assert "Remember cabinet" not in actual["prompt"]  # Covered history stays covered.
        assert saved_memory.text not in actual["prompt"]
        assert "Watch the pot." in actual["prompt"]
        assert len(actual["multi_modal_data"].get("image", [])) == image_count
        assert token_count(actual) <= budget
        assert state.memory == saved_memory
        conversation[-1]["content"] = "Q"
        assert saved_memory.text in plan(prepared)["prompt"]
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("part_transcript", [True, False])
async def test_spoken_instruction_is_summarized_once_and_correction_invalidates(part_transcript):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        part = {"type": "audio_url", "audio_url": {"url": "native-duplex:input-audio"}}
        if part_transcript:
            part["transcript"] = "Watch the red cabinet."
        spoken = {"role": "user", "content": [part], "transcript": "Watch the red cabinet."}
        h.session.register_history_item("user-0", spoken)
        submission = await start_summary(h)
        assert submission.prompt["prompt"].count("Watch the red cabinet.") == 1
        assert "native-duplex:input-audio" not in submission.prompt["prompt"]
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.text
        corrected = {"role": "user", "content": [{**part, "transcript": "Watch the blue cabinet."}]}
        h.session.register_history_item("user-0", corrected)
        config = h.runner.plugin.prepare_prompt_config(
            {"conversation": list(h.session.history)}, state=h.runner.model_state, payload={}
        )
        assert config["joyai_summary"] == ""
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_imported_history_catches_up_without_omitting_unsummarized_middle():
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=12)
        first = await start_summary(h)
        assert "Cabinet 3 recorded." in first.prompt["prompt"]
        h.deliver(summary_output(first.context.request_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert len(h.port.submissions) == 2
        second = h.port.submissions[-1]
        assert "Cabinet 0 has code K731." in second.prompt["prompt"]
        assert "Cabinet 7 recorded." in second.prompt["prompt"]
        config = h.runner.plugin.prepare_prompt_config(
            {"conversation": list(h.session.history)}, state=h.runner.model_state, payload={}
        )
        prompt = h.runner.plugin.plan_append(
            request_id="normal",
            fence=DuplexFence("s"),
            session_config=config,
            runtime_config=h.session.runtime_config,
            seq=1,
            turn_seq=1,
            payload={},
            final=True,
            sampling_params=SamplingParams(max_tokens=64),
        ).prompt["prompt"]
        assert "Cabinet 0 has code K731." in prompt
        assert "Cabinet 4 recorded." in prompt
        assert "Cabinet 11 recorded." in prompt
        h.deliver(
            summary_output(second.context.request_id, "Cabinets 0 through 7 recorded."),
            stage_id=0,
            segment_finished=True,
        )
        await h.settle()
        assert h.runner.model_state.memory.covered_messages == 16
        assert len(h.port.submissions) == 2
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_unchanged_short_history_does_not_schedule_repeated_preparation(mocker):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=2)
        prepare = mocker.spy(h.runner.plugin, "prepare_auxiliary_request")
        snapshot = mocker.spy(memory_runtime, "duplex_text_history")
        await h.run(Heartbeat())
        first_count = prepare.call_count
        assert first_count == 1
        snapshots = snapshot.call_count
        await h.run(Heartbeat())
        assert prepare.call_count == first_count
        assert snapshot.call_count == snapshots
        assert h.port.submissions == []
        # Editing an existing message must invalidate the fast path even when
        # the number of messages is unchanged.
        before_edit = prepare.call_count
        h.session.register_history_item("user-0", {"role": "user", "content": "Edited instruction"})
        await h.run(Heartbeat())
        assert prepare.call_count == before_edit + 1
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_interactive_turn_preempts_summary_with_one_stage_slot(mocker):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session)
        summary = await start_summary(h)
        summary_id = summary.context.request_id
        released = asyncio.Event()
        original_submit, original_cleanup = h.port.submit, h.port.cleanup

        async def single_slot_submit(submission):
            if "auxiliary-" not in submission.context.request_id:
                await released.wait()
            return await original_submit(submission)

        async def cleanup(request_ids, *, abort=False):
            await original_cleanup(request_ids, abort=abort)
            if summary_id in request_ids:
                released.set()

        mocker.patch.object(h.port, "submit", side_effect=single_slot_submit)
        mocker.patch.object(h.port, "cleanup", side_effect=cleanup)
        h.submit(
            CreateItem(
                item={
                    "id": "interactive-query",
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "What do you see now?"}],
                }
            )
        )
        h.submit(AppendAudio(video_frames=(frame(),)))
        h.submit(Commit(final=True, create_response=True))

        async def wait_for_interaction():
            while len(h.port.submissions) == 1:
                await asyncio.sleep(0.005)

        await asyncio.wait_for(wait_for_interaction(), timeout=2)
        assert released.is_set()
        assert ([summary_id], True) in h.port.cleanups
        assert summary_id not in h.session.resource_request_ids()
        assert "auxiliary-" not in h.port.submissions[-1].context.request_id
        h.deliver(summary_output(summary_id), stage_id=0, segment_finished=True)
        await h.settle()
        assert h.runner.model_state.memory.text == ""
        foreground = h.port.submissions[-1]
        h.deliver(
            tts_output(foreground.context.request_id, finished=True, text="</silence>"),
            stage_id=0,
            segment_finished=True,
        )
        await h.settle(timeout_s=0.2)
        assert "auxiliary-" in h.port.submissions[-1].context.request_id
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_failed_summary_keeps_interactive_history_bounded(mocker):
    h = await open_joyai(modalities=("text",))
    try:
        committed_turns(h.session, count=30)
        h.port.fail_submit = RuntimeError("summary unavailable")
        await h.run(Heartbeat())
        original_history = h.session.history
        plugin = h.runner.plugin
        render = mocker.spy(plugin.processor, "apply_chat_template")
        config = plugin.prepare_prompt_config(
            {"conversation": list(original_history)}, state=h.runner.model_state, payload={}
        )
        plan = plugin.plan_append(
            request_id="normal",
            fence=DuplexFence("s"),
            session_config=config,
            runtime_config=h.session.runtime_config,
            seq=1,
            turn_seq=1,
            payload={},
            final=True,
            sampling_params=SamplingParams(max_tokens=64),
        )
        assert len(render.call_args.args[0]) <= 18  # system + 16 history messages + observation
        assert "Cabinet 29 recorded." in plan.prompt["prompt"]
        assert "Cabinet 0 recorded." not in plan.prompt["prompt"]
        assert h.session.history == original_history
        assert h.runner.model_state.memory.covered_messages == 0
    finally:
        await h.manager.shutdown()


@pytest.mark.asyncio
async def test_queued_completion_keeps_identity_when_earlier_command_appends_tail(monkeypatch):
    h = await open_joyai(modalities=("text",))
    release = asyncio.Event()
    try:
        committed_turns(h.session)
        submission = await start_summary(h)
        task = h.runner.memory._task
        entered = asyncio.Event()
        original = h.runner._on_command

        async def hold_command(command):
            entered.set()
            await release.wait()
            await original(command)

        monkeypatch.setattr(h.runner, "_on_command", hold_command)
        h.deliver(summary_output(submission.context.request_id), stage_id=0, segment_finished=True)
        h.submit(Heartbeat())
        await asyncio.wait_for(entered.wait(), timeout=2)
        await asyncio.wait_for(task, timeout=2)
        h.session.append_history_message({"role": "user", "content": "New observation."})
        release.set()
        await h.settle()
        assert h.runner.model_state.memory.text == "Cabinet 0 has code K731."
    finally:
        release.set()
        await h.manager.shutdown()
