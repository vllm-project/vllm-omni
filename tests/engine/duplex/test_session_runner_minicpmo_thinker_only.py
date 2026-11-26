# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The session runner driven by the MiniCPM-o plugin on the Thinker-only pipeline.

Same harness as ``test_session_runner.py``, with a single text-final stage.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerFast

from tests.engine.duplex.test_playback_finalization import _assert_released
from tests.engine.duplex.test_session_runner import (
    _fake_encode_audio,
    append_audio,
    close_harness,
    commands,
    open_harness,
    stage_stats,
    types,
)
from vllm_omni.clients.duplex import EventCollector
from vllm_omni.engine.duplex.events import DuplexEvent
from vllm_omni.engine.duplex.session.engine_session import RESPONSE_REQUEST_MEASUREMENT_ORIGIN
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin, _tokenizer_key
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_THINKER_ONLY_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

LISTEN, SPEAK, CHUNK_EOS, TURN_EOS = 7, 8, 5, 9
SPECIAL_TOKEN_IDS = {
    "listen_token_id": LISTEN,
    "speak_token_id": SPEAK,
    "chunk_eos_token_id": CHUNK_EOS,
    "turn_eos_token_id": TURN_EOS,
}
A, B = 100, 101


def _thinker_only_plugin() -> MiniCPMO45DuplexPlugin:
    tokens = {index: f"unused_{index}" for index in range(B + 1)}
    tokens.update({LISTEN: "<|listen|>", SPEAK: "<|speak|>", CHUNK_EOS: "<|chunk_eos|>", TURN_EOS: "<|turn_eos|>"})
    tokens.update({A: "嗯，", B: "好的"})
    backend = Tokenizer(models.WordLevel({token: index for index, token in tokens.items()}, unk_token="unused_0"))
    backend.decoder = decoders.Fuse()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="unused_0",
        additional_special_tokens=[tokens[index] for index in (LISTEN, SPEAK, CHUNK_EOS, TURN_EOS)],
    )
    plugin = MiniCPMO45DuplexPlugin(_fake_encode_audio)
    plugin.configure_pipeline(MINICPMO_4_5_THINKER_ONLY_PIPELINE)
    plugin._tokenizers[_tokenizer_key(None)] = tokenizer
    return plugin


def _stage0(request_id: str, token_ids: list[int], text: str, *, stop: bool) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        finished=False,
        outputs=[
            SimpleNamespace(
                text=text,
                token_ids=list(token_ids),
                finish_reason="stop" if stop else None,
                stop_reason=token_ids[-1] if stop else None,
                multimodal_output={},
            )
        ],
        multimodal_output={
            "latent": np.ones((1, 16), dtype=np.float32),
            "special_token_ids": dict(SPECIAL_TOKEN_IDS),
        },
    )


@pytest.mark.asyncio
async def test_turn_runs_from_listen_to_response_done() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        assert h.session.capabilities.supported_output_modalities == {"text"}
        await h.run(append_audio())
        request_id = h.stage0_request_id()

        async def deliver(token_ids: list[int], text: str, *, stop: bool):
            return await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=stop), stage_id=0, segment_finished=stop
            )

        listen = await deliver([LISTEN], "", stop=True)
        assert "response.listen" in types(listen)
        assert "response.created" not in types(listen)

        speaking = await deliver([LISTEN, SPEAK, A], "嗯，", stop=False)
        assert "response.created" in types(speaking)
        response_id = h.session.active_response_id
        assert response_id is not None

        submissions = len(h.port.submissions)
        unit_end = await deliver([LISTEN, SPEAK, A, CHUNK_EOS], "嗯，", stop=True)
        assert "response.done" not in types(unit_end)
        assert len(h.port.submissions) == submissions + 1, "the unit end did not continue generation"
        assert h.session.active_response_id == response_id

        final = await deliver([LISTEN, SPEAK, A, CHUNK_EOS, SPEAK, B, TURN_EOS, LISTEN], "嗯，好的", stop=True)
        done = [event for event in final if event.type == "response.done"]
        assert len(done) == 1
        assert h.session.active_response_id is None
        assert h.session.history[-1] == {"role": "assistant", "content": "嗯，好的"}

        events = listen + speaking + unit_end + final
        deltas = "".join(e.delta for e in events if e.type == "response.output_text.delta")
        assert deltas == "嗯，好的"
        assert not any(e.type.startswith("response.output_audio") for e in events)
        assert not any(e.type == "error" for e in events), types(events)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("complete_second_turn", [False, True])
async def test_coalesced_turns_keep_separate_response_text_and_history(complete_second_turn: bool) -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        token_ids = [SPEAK, A, TURN_EOS, LISTEN, SPEAK, B]
        if complete_second_turn:
            token_ids += [TURN_EOS, LISTEN]
        events = await h.deliver_and_settle(
            _stage0(request_id, token_ids, "嗯，好的", stop=complete_second_turn),
            stage_id=0,
            segment_finished=complete_second_turn,
        )
        if not complete_second_turn:
            assert h.session.history[-1] == {"role": "assistant", "content": "嗯，"}
            repeated = await h.deliver_and_settle(_stage0(request_id, token_ids, "嗯，好的", stop=False), stage_id=0)
            assert not any(event.type == "response.output_text.delta" for event in repeated)
            events += repeated
            events += await h.deliver_and_settle(
                _stage0(request_id, [*token_ids, TURN_EOS, LISTEN], "嗯，好的", stop=True),
                stage_id=0,
                segment_finished=True,
            )

        created = [event for event in events if event.type == "response.created"]
        done = [event for event in events if event.type == "response.done"]
        assert len(created) == len(done) == 2, types(events)
        assert [event.response_id for event in created] == [event.response_id for event in done]
        assert all(event.response["status"] == "completed" for event in done)
        assert [
            "".join(
                event.delta
                for event in events
                if event.type == "response.output_text.delta" and event.response_id == response.response_id
            )
            for response in created
        ] == ["嗯，", "好的"]
        assert h.session.history[-2:] == (
            {"role": "assistant", "content": "嗯，"},
            {"role": "assistant", "content": "好的"},
        )
        assert h.session.active_response_id is None
        assert not any(event.type == "error" or event.type.startswith("response.output_audio") for event in events)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_length_stop_fails_the_response_without_committing() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        speaking = await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A], "嗯，", stop=False),
            stage_id=0,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=2),
        )
        assert "response.created" in types(speaking)
        history_before = tuple(h.session.history)

        capped = _stage0(request_id, [SPEAK, A, B, TURN_EOS], "嗯，好的", stop=True)
        capped.outputs[0].finish_reason = "length"
        capped.outputs[0].stop_reason = None
        final = await h.deliver_and_settle(
            capped,
            stage_id=0,
            segment_finished=True,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=2),
        )

        done = [event for event in final if event.type == "response.done"]
        assert len(done) == 1
        assert done[0].response["status"] == "failed"
        stage0 = done[0].to_realtime()["response"]["metadata"]["vllm_omni"]["stage_metrics"]["0"]
        assert stage0["num_tokens_out"] == 4
        assert stage0["stage_gen_time_ms"] == pytest.approx(240.0)
        assert any(e.type == "error" and e.error["code"] == "runtime_data_plane_unterminated_unit" for e in final)
        assert h.session.active_response_id is None
        assert tuple(h.session.history) == history_before, "a length-capped reply was committed"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_unterminated_reply_releases_zero_audio_state_without_committing() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(_stage0(request_id, [SPEAK, A], "嗯，", stop=False), stage_id=0)
        response_id = h.session.active_response_id
        assert response_id is not None
        capped = _stage0(request_id, [SPEAK, A, B], "嗯，好的", stop=False)
        capped.outputs[0].finish_reason = "length"
        final = await h.deliver_and_settle(capped, stage_id=0, segment_finished=True)

        [done] = [event for event in final if event.type == "response.done"]
        assert done.response["status"] == "failed"
        assert done.response["status_details"]["reason"] == "runtime_data_plane_unterminated_unit"
        assert h.session.active_response_id is None
        assert not any(message.get("role") == "assistant" for message in h.session.history)

        _assert_released(h.session, response_id)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_conversation_none_releases_zero_audio_state_without_committing() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1, auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=False))
        await h.run(commands.CreateResponse(options={"conversation": "none"}))
        request_id = h.stage0_request_id()
        response_id = h.session.active_response_id
        assert response_id is not None
        final = await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A, TURN_EOS, LISTEN], "嗯，", stop=True),
            stage_id=0,
            segment_finished=True,
        )

        [done] = [event for event in final if event.type == "response.done"]
        assert done.response["status"] == "completed"
        assert h.session.active_response_id is None
        assert not any(message.get("role") == "assistant" for message in h.session.history)

        _assert_released(h.session, response_id)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("interrupt", [commands.CancelResponse(), commands.BargeIn()], ids=["cancel", "barge-in"])
@pytest.mark.parametrize("conversation", ["auto", "none"])
async def test_interrupted_reply_releases_zero_audio_state(interrupt, conversation: str) -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1, auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=False))
        await h.run(commands.CreateResponse(options={"conversation": conversation}))
        request_id = h.stage0_request_id()
        response_id = h.session.active_response_id
        assert response_id is not None
        await h.deliver_and_settle(_stage0(request_id, [SPEAK, A], "嗯，", stop=False), stage_id=0)
        final = await h.run(interrupt)

        [done] = [event for event in final if event.type == "response.done"]
        assert done.response["status"] == "cancelled"
        assert h.session.active_response_id is None
        assert any(message.get("role") == "assistant" for message in h.session.history) == (conversation == "auto")

        _assert_released(h.session, response_id)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stage_failure_releases_zero_audio_state_without_committing() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(_stage0(request_id, [SPEAK, A], "嗯，", stop=False), stage_id=0)
        response_id = h.session.active_response_id
        assert response_id is not None
        h.runner.on_stage_failure(0, RuntimeError("input stream failed"), request_id=request_id)
        final = await h.settle()

        [done] = [event for event in final if event.type == "response.done"]
        assert done.response["status"] == "failed"
        assert h.session.active_response_id is None
        assert not any(message.get("role") == "assistant" for message in h.session.history)

        _assert_released(h.session, response_id)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_explicit_response_continues_past_a_unit_end() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1, auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.stage0_request_id()

        async def deliver(token_ids: list[int], text: str, *, stop: bool):
            return await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=stop), stage_id=0, segment_finished=stop
            )

        speaking = await deliver([SPEAK, A], "嗯，", stop=False)
        response_id = h.session.active_response_id
        assert response_id is not None, types(speaking)

        submissions = len(h.port.submissions)
        unit_end = await deliver([SPEAK, A, CHUNK_EOS], "嗯，", stop=True)
        assert "response.done" not in types(unit_end)
        assert len(h.port.submissions) == submissions + 1, "the explicit response stalled after its first unit"
        assert h.session.active_response_id == response_id

        final = await deliver([SPEAK, A, CHUNK_EOS, SPEAK, B, TURN_EOS, LISTEN], "嗯，好的", stop=True)
        done = [event for event in final if event.type == "response.done"]
        assert len(done) == 1
        assert done[0].response_id == response_id
        assert h.session.active_response_id is None
        events = speaking + unit_end + final
        assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "嗯，好的"
        assert not any(e.type == "error" for e in events), types(events)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["commit", "response.create"])
async def test_explicit_response_first_text_carries_server_ttft_to_the_client(entry: str) -> None:
    clock = {"now": 1000.0}
    h = await open_harness(
        plugin=_thinker_only_plugin(), stage_count=1, auto_response=False, clock=lambda: clock["now"]
    )
    try:
        await h.run(append_audio())
        events = await h.run(commands.Commit(create_response=entry == "commit"))
        if entry == "response.create":
            events += await h.run(commands.CreateResponse())
        response_id = h.session.active_response_id
        assert response_id is not None, types(events)
        assert "response.created" in types(events)

        clock["now"] = 1001.3
        request_id = h.stage0_request_id()
        speaking = await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A], "嗯，", stop=False),
            stage_id=0,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=3),
        )
        assert "response.created" not in types(speaking)
        assert "response.speak" not in types(speaking)
        [delta] = [event for event in speaking if event.type == "response.output_text.delta"]
        runtime_metadata = delta.to_realtime()["metadata"]["vllm_omni"]
        metrics = runtime_metadata["response_request_metrics"]
        assert metrics["source"] == "server_monotonic_request_start"
        assert metrics["measurement_origin"] == dict(RESPONSE_REQUEST_MEASUREMENT_ORIGIN)
        assert metrics["ttft_ms"] == pytest.approx(1300.0)

        collector = EventCollector()
        # Client receive timing deliberately differs from the server's clock.
        for event in events:
            collector.add(event.to_realtime(), received_at_s=10.0)
        for event in speaking:
            collector.add(event.to_realtime(), received_at_s=12.0)
        final = await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A, TURN_EOS, LISTEN], "嗯，", stop=True),
            stage_id=0,
            segment_finished=True,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=2),
        )
        [done] = [event for event in final if event.type == "response.done"]
        stage0 = done.to_realtime()["response"]["metadata"]["vllm_omni"]["stage_metrics"]["0"]
        assert stage0["num_tokens_out"] == 5
        for event in final:
            collector.add(event.to_realtime(), received_at_s=12.1)
        timing = collector.timing_summary(after_s=9.0, response_id=response_id)
        assert timing["stage0_tokens"]["output_token_count"] == 5
        assert timing["stages"]["0"]["output_token_count"] == 5
        assert h.session._ended_response_stage_metrics == {}
        assert timing["request_metrics"]["source"] == "server_request_start_and_client_receive"
        assert timing["request_metrics"]["ttft_ms"] == pytest.approx(1300.0)
        assert timing["request_metrics"]["measurement_origin"]["ttft"] == RESPONSE_REQUEST_MEASUREMENT_ORIGIN["ttft"]
        assert "ttfp_ms" not in timing["request_metrics"]
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_explicit_response_ends_at_its_continuation_limit() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1, auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.stage0_request_id()
        token_ids: list[int] = []
        text = ""
        events: list[DuplexEvent] = []
        for _ in range(h.runner.model._RESPONSE_MAX_CONTINUATION_UNITS + 2):
            if h.session.active_response_id is None and events:
                break
            token_ids += [SPEAK, A, CHUNK_EOS]
            text += "嗯，"
            events += await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=True), stage_id=0, segment_finished=True
            )
        done = [event for event in events if event.type == "response.done"]
        assert len(done) == 1, types(events)
        assert done[0].response["status"] == "incomplete"
        assert done[0].response["status_details"] == {"type": "incomplete", "reason": "continuation_limit"}
        assert h.session.active_response_id is None
        assert not any(m.get("role") == "assistant" for m in h.session.history), "a cut-off reply was committed"
        response_id = done[0].response_id
        _assert_released(h.session, response_id)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_explicit_response_continues_past_an_initial_listen() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1, auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.stage0_request_id()
        response_id = h.session.active_response_id
        assert response_id is not None

        async def deliver(token_ids: list[int], text: str, *, stop: bool):
            return await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=stop), stage_id=0, segment_finished=stop
            )

        submissions = len(h.port.submissions)
        listen = await deliver([LISTEN], "", stop=True)
        assert "response.done" not in types(listen)
        assert len(h.port.submissions) == submissions + 1, "the listen unit did not continue the response"
        assert h.session.active_response_id == response_id
        assert h.session.active_request_id == request_id

        final = await deliver([LISTEN, SPEAK, A, TURN_EOS, LISTEN], "嗯，", stop=True)
        done = [event for event in final if event.type == "response.done"]
        assert len(done) == 1
        assert done[0].response_id == response_id
        assert done[0].response["status"] == "completed"
        assert h.session.active_response_id is None
        events = listen + final
        assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "嗯，"
        assert not any(e.type == "error" for e in events), types(events)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_auto_response_budget_ends_incomplete() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        speaking = await h.deliver_and_settle(_stage0(request_id, [SPEAK, A], "嗯，", stop=False), stage_id=0)
        response_id = h.session.active_response_id
        assert response_id is not None, types(speaking)
        model = h.runner.model
        h.runner.model_state.continuation_owner_id = f"response:{response_id}"
        h.runner.model_state.continuation_units = model._AUTO_RESPONSE_MAX_CONTINUATION_UNITS

        final = await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A, LISTEN], "嗯，", stop=True), stage_id=0, segment_finished=True
        )
        done = [event for event in final if event.type == "response.done"]
        assert len(done) == 1, types(final)
        assert done[0].response_id == response_id
        assert done[0].response["status"] == "incomplete"
        assert done[0].response["status_details"] == {"type": "incomplete", "reason": "continuation_limit"}
        assert h.session.active_response_id is None
        assert not any(m.get("role") == "assistant" for m in h.session.history)
        _assert_released(h.session, response_id)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_response_done_carries_stage0_metrics() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()

        async def deliver(token_ids: list[int], text: str, *, stop: bool, tokens_out: int):
            return await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=stop),
                stage_id=0,
                segment_finished=stop,
                metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=tokens_out, itls_ms=[20.0]),
            )

        await deliver([LISTEN, SPEAK, A], "嗯，", stop=False, tokens_out=3)
        await deliver([LISTEN, SPEAK, A, CHUNK_EOS], "嗯，", stop=True, tokens_out=1)
        final = await deliver(
            [LISTEN, SPEAK, A, CHUNK_EOS, SPEAK, B, TURN_EOS, LISTEN], "嗯，好的", stop=True, tokens_out=4
        )

        [done] = [event for event in final if event.type == "response.done"]
        metadata = done.to_realtime()["response"]["metadata"]
        stage0 = metadata["vllm_omni"]["stage_metrics"]["0"]
        assert stage0["num_tokens_out"] == 8, "every snapshot's tokens count once"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_empty_units_before_a_reply_do_not_add_to_its_metrics() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()

        async def deliver(token_ids: list[int], text: str, *, tokens_out: int):
            metrics = stage_stats(stage_id=0, request_id=request_id, num_tokens_out=tokens_out, itls_ms=[20.0])
            return await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=True), stage_id=0, segment_finished=True, metrics=metrics
            )

        idle = [CHUNK_EOS] * 20
        for count in range(1, len(idle) + 1):
            events = await deliver(idle[:count], "", tokens_out=1)
            assert "response.created" not in types(events)
        final = await deliver([*idle, SPEAK, A, TURN_EOS, CHUNK_EOS], "嗯，", tokens_out=4)

        [done] = [event for event in final if event.type == "response.done"]
        stage0 = done.to_realtime()["response"]["metadata"]["vllm_omni"]["stage_metrics"]["0"]
        assert stage0["num_tokens_out"] == 4, "units with no open response are not charged to the next one"
        assert stage0["vllm_itls_ms"] == [20.0]
        deltas = [event.to_realtime() for event in final if event.type == "response.output_text.delta"]
        assert deltas
        assert not any("stage_metrics" in (delta.get("metadata", {}).get("vllm_omni") or {}) for delta in deltas)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_listen_unit_metrics_count_toward_the_response() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1, auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.stage0_request_id()

        async def deliver(token_ids: list[int], text: str, *, stop: bool, tokens_out: int):
            return await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=stop),
                stage_id=0,
                segment_finished=stop,
                metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=tokens_out),
            )

        # stage_stats reports 120 ms of generation per snapshot.
        await deliver([LISTEN], "", stop=True, tokens_out=1)
        await deliver([LISTEN, SPEAK, A], "嗯，", stop=False, tokens_out=3)
        final = await deliver([LISTEN, SPEAK, A, TURN_EOS, LISTEN], "嗯，", stop=True, tokens_out=4)

        [done] = [event for event in final if event.type == "response.done"]
        stage0 = done.to_realtime()["response"]["metadata"]["vllm_omni"]["stage_metrics"]["0"]
        assert stage0["num_tokens_out"] == 8
        assert stage0["stage_gen_time_ms"] == pytest.approx(360.0)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancelled_response_keeps_stage0_metrics() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        speaking = await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A], "嗯，", stop=False),
            stage_id=0,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=3),
        )
        response_id = h.session.active_response_id
        assert response_id is not None, types(speaking)

        cancelled = await h.run(commands.CancelResponse())
        [done] = [event for event in cancelled if event.type == "response.done"]
        assert done.response_id == response_id
        assert done.response["status"] == "cancelled"
        stage0 = done.to_realtime()["response"]["metadata"]["vllm_omni"]["stage_metrics"]["0"]
        assert stage0["num_tokens_out"] == 3
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_stage_failure_keeps_stage0_metrics() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A], "嗯，", stop=False),
            stage_id=0,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=3),
        )
        assert h.session.active_response_id is not None

        h.runner.on_stage_failure(0, RuntimeError("boom"))
        events = await h.settle()
        [done] = [event for event in events if event.type == "response.done"]
        assert done.response["status_details"]["reason"] == "runtime_data_plane_stream_failed"
        stage0 = done.to_realtime()["response"]["metadata"]["vllm_omni"]["stage_metrics"]["0"]
        assert stage0["num_tokens_out"] == 3
    finally:
        await close_harness(h)


def _stage0_tokens_out(event) -> object:
    return event.to_realtime()["response"]["metadata"]["vllm_omni"]["stage_metrics"]["0"]["num_tokens_out"]


@pytest.mark.asyncio
async def test_each_response_reports_only_its_own_stage0_metrics() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()

        async def deliver(token_ids: list[int], text: str, *, stop: bool, tokens_out: int):
            return await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=stop),
                stage_id=0,
                segment_finished=stop,
                metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=tokens_out),
            )

        first_turn = [SPEAK, A, TURN_EOS, LISTEN]
        await deliver([SPEAK, A], "嗯，", stop=False, tokens_out=3)
        first = await deliver(first_turn, "嗯，", stop=True, tokens_out=1)
        [first_done] = [event for event in first if event.type == "response.done"]
        assert _stage0_tokens_out(first_done) == 4
        assert h.session._ended_response_stage_metrics == {}, "emitted metrics stay cached"

        await deliver([*first_turn, SPEAK, B], "嗯，好的", stop=False, tokens_out=2)
        second = await deliver([*first_turn, SPEAK, B, TURN_EOS, LISTEN], "嗯，好的", stop=True, tokens_out=1)
        [second_done] = [event for event in second if event.type == "response.done"]
        assert second_done.response_id != first_done.response_id
        assert _stage0_tokens_out(second_done) == 3, "the first response's metrics leaked into the second"
        assert h.session._ended_response_stage_metrics == {}
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_output_backpressure_failure_keeps_stage0_metrics() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A], "嗯，", stop=False),
            stage_id=0,
            metrics=stage_stats(stage_id=0, request_id=request_id, num_tokens_out=3),
        )
        response_id = h.session.active_response_id
        assert response_id is not None

        recovery = h.runner.fail_output("output buffer full", response_id=response_id, terminal=None)
        [done] = [event for event in recovery if event.type == "response.done"]
        assert done.response["status_details"] == {"type": "failed", "reason": "output_backpressure"}
        assert _stage0_tokens_out(done) == 3
        assert h.session._ended_response_stage_metrics == {}
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_response", [False, True])
async def test_an_empty_queued_snapshot_does_not_read_later_tokens(auto_response: bool) -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1, auto_response=auto_response)
    try:
        await h.run(append_audio())
        if not auto_response:
            await h.run(commands.Commit(create_response=True))
        request_id = h.stage0_request_id()
        live: list[int] = []
        first = _stage0(request_id, [SPEAK], "", stop=False)
        first.outputs[0].token_ids = live
        first.outputs[0].cumulative_token_ids = []
        h.deliver(first, stage_id=0)
        live.extend([SPEAK, A, B, TURN_EOS, LISTEN])
        final = _stage0(request_id, live, "嗯，好的", stop=True)
        final.outputs[0].cumulative_token_ids = list(live)
        h.deliver(final, stage_id=0, segment_finished=True)
        events = await h.settle()

        assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "嗯，好的"
        done = [e for e in events if e.type == "response.done"]
        assert len(done) == 1, types(events)
        assert done[0].response["status"] == "completed"
        assert h.session.active_response_id is None
    finally:
        await close_harness(h)
