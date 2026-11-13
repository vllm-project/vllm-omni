# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The session runner driven by the MiniCPM-o plugin on the Thinker-only pipeline.

Same harness as ``test_session_runner.py``, with a single text-final stage.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tests.engine.duplex.test_session_runner import (
    _fake_encode_audio,
    append_audio,
    close_harness,
    commands,
    open_harness,
    stage_stats,
    types,
)
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import MiniCPMO45DuplexPlugin
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
    plugin = MiniCPMO45DuplexPlugin(_fake_encode_audio)
    plugin.configure_pipeline(MINICPMO_4_5_THINKER_ONLY_PIPELINE)
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
async def test_explicit_response_ends_at_its_continuation_limit() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1, auto_response=False)
    try:
        await h.run(append_audio())
        await h.run(commands.Commit(create_response=True))
        request_id = h.stage0_request_id()
        token_ids: list[int] = []
        text = ""
        events = []
        for _ in range(h.runner.model._RESPONSE_MAX_CONTINUATION_UNITS + 2):
            if h.session.active_response_id is None and events:
                break
            token_ids += [SPEAK, A, CHUNK_EOS]
            text += "嗯"
            events += await h.deliver_and_settle(
                _stage0(request_id, token_ids, text, stop=True), stage_id=0, segment_finished=True
            )
        done = [event for event in events if event.type == "response.done"]
        assert len(done) == 1, types(events)
        assert done[0].response["status"] == "incomplete"
        assert done[0].response["status_details"] == {"type": "incomplete", "reason": "continuation_limit"}
        assert h.session.active_response_id is None
        assert not any(m.get("role") == "assistant" for m in h.session.history), "a cut-off reply was committed"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_missing_control_tokens_close_the_session() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()

        def bare(token_ids: list[int], text: str) -> SimpleNamespace:
            output = _stage0(request_id, token_ids, text, stop=False)
            output.multimodal_output = {"latent": np.ones((1, 16), dtype=np.float32)}
            return output

        first = await h.deliver_and_settle(bare([SPEAK, A], "嗯"), stage_id=0)
        assert [e.error["code"] for e in first if e.type == "error"] == ["runtime_data_plane_missing_control_tokens"]
        assert "response.created" not in types(first)
        assert not any(e.type == "response.output_text.delta" for e in first)
        assert "session.closed" in types(first)

        second = await h.deliver_and_settle(bare([SPEAK, A, B], "嗯这"), stage_id=0)
        assert "response.created" not in types(second)
        assert not any(e.type == "response.output_text.delta" for e in second)
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

        final = await deliver([LISTEN, SPEAK, A, TURN_EOS, LISTEN], "好", stop=True)
        done = [event for event in final if event.type == "response.done"]
        assert len(done) == 1
        assert done[0].response_id == response_id
        assert done[0].response["status"] == "completed"
        assert h.session.active_response_id is None
        events = listen + final
        assert "".join(e.delta for e in events if e.type == "response.output_text.delta") == "好"
        assert not any(e.type == "error" for e in events), types(events)
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_auto_response_budget_ends_incomplete() -> None:
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio())
        request_id = h.stage0_request_id()
        speaking = await h.deliver_and_settle(_stage0(request_id, [SPEAK, A], "嗯", stop=False), stage_id=0)
        response_id = h.session.active_response_id
        assert response_id is not None, types(speaking)
        model = h.runner.model
        h.runner.model_state.continuation_owner_id = f"response:{response_id}"
        h.runner.model_state.continuation_units = model._AUTO_RESPONSE_MAX_CONTINUATION_UNITS

        final = await h.deliver_and_settle(
            _stage0(request_id, [SPEAK, A, LISTEN], "嗯", stop=True), stage_id=0, segment_finished=True
        )
        done = [event for event in final if event.type == "response.done"]
        assert len(done) == 1, types(final)
        assert done[0].response_id == response_id
        assert done[0].response["status"] == "incomplete"
        assert done[0].response["status_details"] == {"type": "incomplete", "reason": "continuation_limit"}
        assert h.session.active_response_id is None
        assert not any(m.get("role") == "assistant" for m in h.session.history)
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
        await deliver([LISTEN, SPEAK, A], "好", stop=False, tokens_out=3)
        final = await deliver([LISTEN, SPEAK, A, TURN_EOS, LISTEN], "好", stop=True, tokens_out=4)

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
