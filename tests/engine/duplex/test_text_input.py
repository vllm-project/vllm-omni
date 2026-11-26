# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native text appends share ordering, admission, and cancellation with media."""

import asyncio
import base64
import json
from dataclasses import replace
from types import SimpleNamespace

import msgspec
import pytest
import torch

from tests.engine.duplex.test_minicpmo_window_plugin import _stub_tokenizer
from tests.engine.duplex.test_session_runner import (
    append_audio,
    close_harness,
    listen_output,
    open_harness,
    stage_stats,
    tts_output,
    types,
)
from tests.engine.duplex.test_session_runner_minicpmo_thinker_only import (
    CHUNK_EOS,
    SPEAK,
    TURN_EOS,
    A,
    B,
    _stage0,
    _thinker_only_plugin,
)
from tests.model_executor.models.minicpmo_4_5.helpers import stage0_vision_runtime
from tests.model_executor.models.minicpmo_4_5.test_native_duplex_sampling import (
    LISTEN,
    TOKEN_IDS,
    VOCAB,
    _policy_model_with_state,
    _policy_row,
)
from vllm_omni.config.stage_config import DuplexSessionRuntimeConfig
from vllm_omni.engine.duplex import commands
from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import plugin as plugin_module
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import _MiniCPMO45Stage0SessionState
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO_4_5_PIPELINE

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def text_tokenizer(monkeypatch):
    _stub_tokenizer(monkeypatch)


@pytest.mark.asyncio
@pytest.mark.parametrize("thinker_only", [False, True])
async def test_text_append_uses_custom_tokenizer_path_and_revision_after_serialization(monkeypatch, thinker_only):
    plugin = _thinker_only_plugin()
    tokenizer = plugin._tokenizers.pop(plugin_module._tokenizer_key(None))
    if not thinker_only:
        plugin.configure_pipeline(MINICPMO_4_5_PIPELINE)
    loaded = []

    def load_tokenizer(config):
        loaded.append(plugin_module._tokenizer_ref(config))
        return tokenizer

    monkeypatch.setattr(plugin_module, "_load_tokenizer", load_tokenizer)
    model_config = SimpleNamespace(model="model-path", tokenizer="tokenizer-path", tokenizer_revision="revision-a")
    runtime = await plugin.prepare_runtime_config(DuplexSessionConfig(modalities=["text"]), model_config=model_config)
    runtime = msgspec.msgpack.decode(msgspec.msgpack.encode(json.loads(json.dumps(runtime))))
    assert runtime["duplex_tokenizer_key"] == json.dumps(("tokenizer-path", "revision-a"))
    assert loaded == [("tokenizer-path", "revision-a")]
    plan = await plugin.prepare_append_plan(
        request_id="text-append",
        fence=DuplexFence("text-append"),
        session_config={},
        runtime_config=runtime,
        seq=1,
        turn_seq=1,
        payload={"type": "text", "text": "嗯，"},
        final=False,
        sampling_params=None,
    )
    assert plan.prompt["model_intermediate_buffer"]["duplex"]["payload"]["text_token_ids"] == tokenizer.encode(
        "嗯，", add_special_tokens=False
    )


def _text_output(request_id: str, *, empty: bool = False):
    output = listen_output(request_id)
    output.finished = False
    output.outputs[0].token_ids = [151718] if empty else [42]
    output.outputs[0].stop_reason = 151718 if empty else None
    output.multimodal_output.update(duplex_text_input=torch.tensor(True))
    output.multimodal_output["meta.chunk_eos_token_id"] = 151718
    return output


@pytest.mark.asyncio
async def test_text_fragments_prefill_in_order_with_audio_without_flushing_partial_pcm():
    h = await open_harness()
    try:
        assert "text" in h.session.capabilities.as_dict()["optional_input_modalities"]
        assert "append_text" in h.session.capabilities.as_dict()["input_modes"]
        await h.run(append_audio(samples=8000))
        pcm_bytes = h.session.pending_input_bytes
        assert pcm_bytes == 32000
        h.submit(commands.AppendText(text="记住", force_listen=True))
        h.submit(commands.AppendText(text="蓝色", force_listen=True))
        h.submit(append_audio(samples=24000))
        events = await h.settle()
        assert "error" not in types(events)
        inputs = [s.prompt["model_intermediate_buffer"]["duplex"] for s in h.port.submissions]
        assert [p["seq"] for p in inputs] == [1, 2, 3, 4]
        assert [p["payload"].get("text") for p in inputs] == ["记住", "蓝色", None, None]
        assert inputs[0]["payload"]["text_token_ids"] == [ord(c) for c in "记住"]
        assert inputs[1]["payload"]["force_listen"] is True
        assert len(h.port.submissions[1].prompt["prompt_token_ids"]) == 2 + 1 + len("蓝色")
        # Text units did not start the mel window: this is still the first
        # 1035 ms audio unit, despite its append sequence being 3.
        assert len(h.port.submissions[2].prompt["prompt_token_ids"]) == 2 + 12 - 1
        assert h.runner.model_state.audio_buffer._audio_window_started
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == 0
        await h.run(commands.ClearInput())
        assert h.session.pending_input_bytes == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("buffered_samples", [0, 8000])
async def test_unforced_text_continues_after_an_empty_model_unit(buffered_samples):
    h = await open_harness()
    try:
        await h.run(commands.AppendText(text="Remember blue whale.", force_listen=True))
        assert h.session.active_response_id is None
        if buffered_samples:
            await h.run(append_audio(samples=buffered_samples))
        assert h.session.pending_input_bytes == buffered_samples * 4
        events = await h.run(commands.AppendText(text="Say it now."))
        assert "response.created" in types(events)
        response_id = h.session.active_response_id
        empty = listen_output(h.stage0_request_id())
        empty.finished = False
        empty.outputs[0].token_ids = [151718]
        listen_id = empty.multimodal_output["meta.listen_token_id"]
        empty.outputs[0].cumulative_token_ids = [listen_id, listen_id, 151718]
        empty.outputs[0].stop_reason = 151718
        empty.multimodal_output["meta.chunk_eos_token_id"] = 151718
        empty.multimodal_output["duplex_text_input"] = True
        events = await h.deliver_and_settle(
            empty,
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[151718],
            segment_output_metadata=empty.multimodal_output,
            model_turn_id=0,
        )
        assert "error" not in types(events)
        assert len(h.port.submissions) == 3
        assert h.runner.model_state.continuation_units == 1
        assert h.session.active_response_id == response_id
        assert (
            h.session.pending_input_bytes
            == h.runner.model_state.audio_buffer.pending_byte_count
            == buffered_samples * 4
        )

        # A subsequent speech unit also continues without consuming the partial
        # microphone input, and the model's terminal closes the same response.
        await h.deliver_and_settle(
            tts_output(h.stage0_request_id(), text="blue whale", tts_is_last_chunk=True, finished=True)
        )
        assert len(h.port.submissions) == 4
        events = await h.deliver_and_settle(
            tts_output(h.stage0_request_id(), text="blue whale", tts_is_last_chunk=True, turn_end=True, finished=True)
        )
        done = [event for event in events if event.type == "response.done"]
        assert len(done) == 1
        assert done[0].response_id == response_id
        assert done[0].status == "completed"
        assert h.session.pending_input_bytes == buffered_samples * 4
        if buffered_samples:
            tail = append_audio(samples=16000 - buffered_samples, value=0.1)
            await h.run(tail)
            payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
            assert base64.b64decode(payload["audio"]) == append_audio(samples=buffered_samples).audio + tail.audio
            assert h.session.pending_input_bytes == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("first_unit", [[CHUNK_EOS], [SPEAK, A, CHUNK_EOS]], ids=["empty", "text"])
async def test_thinker_only_unforced_text_continues_past_partial_pcm(first_unit):
    h = await open_harness(plugin=_thinker_only_plugin(), stage_count=1)
    try:
        await h.run(append_audio(samples=8000))
        events = await h.run(commands.AppendText(text="Say it now."))
        assert "response.created" in types(events)
        response_id = h.session.active_response_id
        request_id = h.stage0_request_id()
        submitted = len(h.port.submissions)

        async def deliver(token_ids, text, *, tokens_out):
            output = _stage0(request_id, token_ids, text, stop=True)
            output.multimodal_output["duplex_text_input"] = True
            metrics = stage_stats(stage_id=0, request_id=request_id, num_tokens_out=tokens_out, itls_ms=[20.0])
            return await h.deliver_and_settle(
                output, stage_id=0, segment_finished=True, model_turn_id=0, metrics=metrics
            )

        events = await deliver(first_unit, "嗯，" if A in first_unit else "", tokens_out=1)
        assert "error" not in types(events)
        # The unit continues without more client audio; the partial PCM stays buffered.
        assert len(h.port.submissions) == submitted + 1
        assert h.session.active_response_id == response_id
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == 8000 * 4

        events = await deliver(
            [*first_unit, SPEAK, B, TURN_EOS, CHUNK_EOS], "嗯，好的" if A in first_unit else "好的", tokens_out=4
        )
        done = [event for event in events if event.type == "response.done"]
        assert len(done) == 1, types(events)
        assert done[0].response_id == response_id
        assert done[0].status == "completed"
        # Each snapshot counts once, whether its unit was empty or carried text.
        stage0 = done[0].to_realtime()["response"]["metadata"]["vllm_omni"]["stage_metrics"]["0"]
        assert stage0["num_tokens_out"] == 5
        assert stage0["stage_gen_time_ms"] == pytest.approx(240.0)
        assert stage0["vllm_itls_ms"] == [20.0, 20.0]
        assert h.session.pending_input_bytes == 8000 * 4
        # The buffered PCM is intact: completing its unit submits it unchanged, ahead of the new tail.
        tail = append_audio(samples=8000, value=0.1)
        await h.run(tail)
        payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
        assert base64.b64decode(payload["audio"]) == append_audio(samples=8000).audio + tail.audio
        assert h.session.pending_input_bytes == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_text_continuation_rearms_after_queued_audio_stays_partial(monkeypatch):
    h = await open_harness()
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        await h.run(append_audio(samples=4000))
        await h.run(commands.AppendText(text="Answer now."))
        await h.deliver_and_settle(_text_output(h.stage0_request_id()), stage_id=0, model_turn_id=0)
        response_id = h.session.active_response_id
        original = h.runner.control.run_turn_detection

        async def blocked_turn_detection(event):
            entered.set()
            await release.wait()
            return await original(event)

        monkeypatch.setattr(h.runner.control, "run_turn_detection", blocked_turn_detection)
        h.submit(append_audio(samples=4000))
        h.submit(append_audio(samples=4000))
        await asyncio.wait_for(entered.wait(), timeout=2)
        await h.runner.model.maybe_continue_response(expected_epoch=h.session.epoch, expected_response_id=response_id)
        assert len(h.port.submissions) == 1
        assert h.runner._deferred_silence is not None
        # A rearm that races another queued append must retain the retry even
        # though the buffer already contains part of a microphone unit.
        await h.runner._rearm_deferred_silence()
        assert h.runner._deferred_silence is not None
        release.set()
        events = await h.settle()
        assert "error" not in types(events)
        assert len(h.port.submissions) == 2
        assert h.runner.model_state.continuation_units == 1
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == 12000 * 4
        payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
        assert set(base64.b64decode(payload["audio"])) == {0}
    finally:
        release.set()
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_samples", [0, 8000])
async def test_text_drains_complete_buffered_audio_and_finishes_without_more_input(partial_samples):
    h = await open_harness()
    try:
        audio = replace(append_audio(samples=48000 + partial_samples), hints={"force_listen": True})
        await h.run(audio)
        await h.run(commands.AppendText(text="Answer now."))
        inputs = [s.prompt["model_intermediate_buffer"]["duplex"]["payload"] for s in h.port.submissions]
        assert [p["type"] for p in inputs] == ["audio", "audio", "audio", "text"]
        assert b"".join(base64.b64decode(p["audio"]) for p in inputs[:-1]) == audio.audio[: 48000 * 4]
        assert all(p["force_listen"] is True for p in inputs[:-1])
        assert h.session.pending_input_bytes == partial_samples * 4
        response_id = h.session.active_response_id
        await h.deliver_and_settle(
            _text_output(h.stage0_request_id(), empty=True), stage_id=0, segment_finished=True, model_turn_id=0
        )
        assert len(h.port.submissions) == 5
        assert h.runner.model_state.continuation_units == 1
        events = await h.deliver_and_settle(
            tts_output(h.stage0_request_id(), tts_is_last_chunk=True, turn_end=True, finished=True)
        )
        done = [e for e in events if e.type == "response.done"]
        assert len(done) == 1 and done[0].response_id == response_id
        assert done[0].status == "completed"
        assert h.session.pending_input_bytes == partial_samples * 4
        if partial_samples:
            tail = append_audio(samples=16000 - partial_samples, value=0.1)
            await h.run(tail)
            payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
            assert base64.b64decode(payload["audio"]) == audio.audio[48000 * 4 :] + tail.audio
            assert h.session.pending_input_bytes == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("partial_samples", [0, 8000])
@pytest.mark.parametrize("text_confirmed_before_audio", [False, True])
async def test_text_reply_drains_later_complete_audio_without_more_client_input(
    partial_samples, text_confirmed_before_audio
):
    h = await open_harness()
    try:
        await h.run(commands.AppendText(text="Answer now."))
        response_id = h.session.active_response_id
        if text_confirmed_before_audio:
            await h.deliver_and_settle(_text_output(h.stage0_request_id()), stage_id=0, model_turn_id=0)
        audio = replace(append_audio(samples=32000 + partial_samples), hints={"force_listen": True})
        await h.run(audio)
        if not text_confirmed_before_audio:
            await h.deliver_and_settle(_text_output(h.stage0_request_id()), stage_id=0, model_turn_id=0)
        inputs = [s.prompt["model_intermediate_buffer"]["duplex"]["payload"] for s in h.port.submissions]
        assert [p["type"] for p in inputs] == ["text", "audio", "audio"]
        assert b"".join(base64.b64decode(p["audio"]) for p in inputs[1:]) == audio.audio[: 32000 * 4]
        assert all(p["force_listen"] is True for p in inputs[1:])
        assert h.session.pending_input_bytes == partial_samples * 4
        # No client input follows until this reply has completed.
        await h.deliver_and_settle(
            tts_output(h.stage0_request_id(), text="Answer", tts_is_last_chunk=True, finished=True)
        )
        assert len(h.port.submissions) == 4
        assert h.runner.model_state.continuation_units == 1
        await h.deliver_and_settle(
            tts_output(h.stage0_request_id(), tts_is_last_chunk=True, turn_end=True, finished=True)
        )
        done = [e for e in h.events if e.type == "response.done"]
        assert len(done) == 1 and done[0].response_id == response_id
        assert done[0].status == "completed"
        assert (
            h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == partial_samples * 4
        )
        if partial_samples:
            tail = append_audio(samples=16000 - partial_samples, value=0.1)
            await h.run(tail)
            payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
            assert base64.b64decode(payload["audio"]) == audio.audio[32000 * 4 :] + tail.audio
            assert h.session.pending_input_bytes == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_text_continuation_exception_does_not_apply_to_a_later_audio_response():
    h = await open_harness()
    try:
        await h.run(append_audio(samples=8000))
        await h.run(commands.AppendText(text="Answer now."))
        await h.deliver_and_settle(_text_output(h.stage0_request_id()), stage_id=0, model_turn_id=0)
        text_response_id = h.session.active_response_id
        await h.deliver_and_settle(
            tts_output(h.stage0_request_id(), tts_is_last_chunk=True, turn_end=True, finished=True)
        )
        assert h.session.active_response_id is None
        # One new audio unit is submitted, with the original half-unit's size
        # left buffered. It must block the audio-only response as before.
        await h.run(append_audio())
        await h.deliver_and_settle(
            tts_output(
                h.stage0_request_id(),
                samples=48000,  # Audio snapshots accumulate across model turns.
                turn_id=h.session.turn_id,
                tts_is_last_chunk=True,
                finished=True,
            )
        )
        assert h.session.active_response_id not in (None, text_response_id)
        assert len(h.port.submissions) == 2
        assert h.runner.model_state.continuation_units == 0
        assert h.session.pending_input_bytes == 8000 * 4
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_text_reply_continues_when_appended_while_previous_talker_drains():
    h = await open_harness()
    try:
        await h.run(append_audio())
        await h.deliver_and_settle(tts_output(h.stage0_request_id(), text="First reply"))
        old_response_id = h.session.active_response_id
        await h.run(append_audio(samples=8000))
        await h.run(commands.AppendText(text="Second question."))
        # Thinker has already handed off turn 0's EOS, but its Talker output
        # still owns the active protocol response when this text is admitted.
        assert h.runner._real_input_waiting()
        assert h.deliver(_text_output(h.stage0_request_id()), stage_id=0, model_turn_id=1) is False
        await h.settle()
        assert h.session.active_response_id == old_response_id
        assert h.runner._real_input_waiting()
        await h.deliver_and_settle(
            tts_output(
                h.stage0_request_id(),
                samples=48000,
                text="First reply done",
                turn_id=0,
                tts_is_last_chunk=True,
                turn_end=True,
                finished=True,
            )
        )
        await h.deliver_and_settle(
            tts_output(
                h.stage0_request_id(),
                samples=72000,
                text="Second reply",
                turn_id=1,
                tts_is_last_chunk=True,
                finished=True,
            )
        )
        assert h.session.active_response_id not in (None, old_response_id)
        assert len(h.port.submissions) == 3
        assert h.runner.model_state.continuation_units == 1
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == 32000
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("previous_talker_finished", [False, True])
@pytest.mark.parametrize("talker_boundary", [None, 151718, 151719])
async def test_empty_text_unit_opens_its_own_response_during_talker_overlap(previous_talker_finished, talker_boundary):
    h = await open_harness()
    try:
        await h.run(append_audio())
        await h.deliver_and_settle(tts_output(h.stage0_request_id(), text="First reply"))
        old_response_id = h.session.active_response_id
        await h.run(append_audio(samples=8000))
        await h.run(commands.AppendText(text="Second question."))
        empty = _text_output(h.stage0_request_id(), empty=True)
        if talker_boundary is not None:
            # SPEAK + a chunk terminator has no TTS condition rows, but it
            # still takes the normal handoff and closes an empty TTS segment.
            empty.outputs[0].token_ids = [151704, talker_boundary]
            empty.outputs[0].stop_reason = talker_boundary
            empty.multimodal_output["meta.speak_token_id"] = 151704
            empty.multimodal_output["meta.chunk_tts_eos_token_id"] = 151719
            assert h.deliver(empty, stage_id=0, segment_finished=True, model_turn_id=1) is False
            await h.settle()
            assert h.session.active_response_id == old_response_id
            empty = tts_output(
                h.stage0_request_id(), samples=0, text="", turn_id=1, tts_is_last_chunk=True, finished=True
            )
        if previous_talker_finished:
            await h.deliver_and_settle(
                tts_output(h.stage0_request_id(), tts_is_last_chunk=True, turn_end=True, finished=True)
            )
        events = await h.deliver_and_settle(
            empty,
            stage_id=0 if talker_boundary is None else 1,
            segment_finished=True,
            model_turn_id=1,
        )
        assert "error" not in types(events)
        if not previous_talker_finished:
            assert h.session.active_response_id == old_response_id
            assert "response.done" not in types(events)
            assert "response.created" not in types(events)
            assert len(h.port.submissions) == 2
            events = await h.deliver_and_settle(
                tts_output(
                    h.stage0_request_id(),
                    samples=48000,
                    text="First reply done",
                    turn_id=0,
                    tts_is_last_chunk=True,
                    turn_end=True,
                    finished=True,
                )
            )
            audio = [e for e in events if e.type == "response.output_audio.delta"]
            assert len(audio) == 1 and audio[0].response_id == old_response_id
            assert audio[0].delta == "wav-24000"
            done = [e for e in h.events if e.type == "response.done" and e.response_id == old_response_id]
            assert len(done) == 1 and done[0].status == "completed"
            assert events.index(audio[0]) < events.index(done[0])
            assert events.index(done[0]) < types(events).index("response.created")
        assert "response.created" in types(events)
        assert h.session.active_response_id not in (None, old_response_id)
        assert h.session.active_response_turn_id == 1
        assert len(h.port.submissions) == 3
        assert h.runner.model_state.continuation_units == 1
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == 32000
        new_response_id = h.session.active_response_id
        events = await h.deliver_and_settle(
            tts_output(
                h.stage0_request_id(),
                samples=72000,
                text="Second reply",
                turn_id=1,
                tts_is_last_chunk=True,
                turn_end=True,
                finished=True,
            )
        )
        done = [e for e in events if e.type == "response.done"]
        assert len(done) == 1 and done[0].response_id == new_response_id
        assert done[0].status == "completed"
        assert len([e for e in h.events if e.type == "response.done" and e.response_id == new_response_id]) == 1
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == 32000
        # The reply completed without another client event. Later microphone
        # input must still consume the original half-unit without alteration.
        tail = append_audio(samples=8000, value=0.1)
        await h.run(tail)
        payload = h.port.submissions[-1].prompt["model_intermediate_buffer"]["duplex"]["payload"]
        assert base64.b64decode(payload["audio"]) == append_audio(samples=8000).audio + tail.audio
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("via_talker", [False, True])
async def test_cancel_discards_empty_text_waiting_for_previous_tail(via_talker):
    h = await open_harness()
    try:
        await h.run(append_audio())
        await h.deliver_and_settle(tts_output(h.stage0_request_id(), text="First reply"))
        await h.run(commands.AppendText(text="Second question."))
        old_request_id = h.stage0_request_id()
        if via_talker:
            await h.deliver_and_settle(_text_output(old_request_id), stage_id=0, model_turn_id=1)
            empty = tts_output(old_request_id, samples=0, text="", turn_id=1, tts_is_last_chunk=True, finished=True)
        else:
            empty = _text_output(old_request_id, empty=True)
        await h.deliver_and_settle(empty, stage_id=1 if via_talker else 0, segment_finished=True, model_turn_id=1)
        assert h.runner.model._deferred_text_listen is not None
        await h.run(commands.BargeIn())
        assert h.session.epoch == 1
        events = await h.deliver_and_settle(
            tts_output(old_request_id, samples=48000, turn_end=True, finished=True), epoch=0
        )
        assert "response.created" not in types(events)
        assert h.session.active_response_id is None
        assert len(h.port.submissions) == 2
        # Fresh output in the new epoch must not replay the deferred old listen.
        await h.run(append_audio())
        events = await h.deliver_and_settle(
            tts_output(
                h.stage0_request_id(),
                epoch=1,
                turn_id=h.session.turn_id,
                tts_is_last_chunk=True,
                turn_end=True,
                finished=True,
            )
        )
        assert types(events).count("response.created") == 1
        assert types(events).count("response.done") == 1
        assert h.session.active_response_id is None
        assert len(h.port.submissions) == 3
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("text", ["\ud800", "hello\udfff"])
async def test_unencodable_text_is_refused_without_breaking_dispatch(text):
    h = await open_harness(runtime_config=DuplexSessionRuntimeConfig(collect_stats=True))
    try:
        command = commands.command_from_realtime(
            {"type": "input.text.append", "text": text, "event_id": "invalid-utf8"}
        )
        events = await h.run(command)
        assert types(events) == ["error"]
        assert events[0].code == "bad_event"
        assert events[0].related_event_id == "invalid-utf8"
        assert h.session.pending_input_bytes == 0
        assert not h.port.submissions
        events = await h.run(commands.AppendText(text="你好", force_listen=True))
        assert "error" not in types(events)
        assert len(h.port.submissions) == 1
        assert h.session.pending_input_bytes == 0
        events = await h.run(commands.GetSessionStats(event_id="still-dispatching"))
        assert types(events) == ["session.stats"]
        assert events[0].request_event_id == "still-dispatching"
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_listen_without_unforced_text_preserves_the_draining_response():
    h = await open_harness()
    try:
        await h.run(append_audio())
        await h.deliver_and_settle(tts_output(h.stage0_request_id()))
        response_id = h.session.active_response_id
        await h.run(append_audio(samples=8000))
        # A forced text fragment does not release a reply, even when Thinker's
        # bridge is already ahead of the response whose audio is still draining.
        await h.run(commands.AppendText(text="More context.", force_listen=True))
        output = listen_output(h.stage0_request_id())
        output.multimodal_output["duplex_text_input"] = torch.tensor(False)
        events = await h.deliver_and_settle(output, stage_id=0, segment_finished=True, model_turn_id=1)
        assert h.session.active_response_id == response_id
        assert "response.done" not in types(events)
        assert h.runner._real_input_waiting()
        assert h.runner.model_state.continuation_units == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_cancelled_text_output_cannot_grant_continuation_in_a_new_epoch():
    h = await open_harness()
    try:
        await h.run(commands.AppendText(text="First question."))
        old_output = _text_output(h.stage0_request_id())
        await h.deliver_and_settle(old_output, stage_id=0, model_turn_id=0)
        await h.run(commands.BargeIn())
        assert h.session.epoch == 1
        await h.run(append_audio(samples=24000))
        await h.deliver_and_settle(old_output, stage_id=0, model_turn_id=0, epoch=0)
        await h.deliver_and_settle(tts_output(h.stage0_request_id(), epoch=1, tts_is_last_chunk=True, finished=True))
        assert h.session.active_response_id is not None
        assert len(h.port.submissions) == 2
        assert h.runner.model_state.continuation_units == 0
        assert h.session.pending_input_bytes == 32000
        assert h.runner._real_input_waiting()
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("force_listen", [True, False])
async def test_text_release_opens_response_after_listening(force_listen):
    h = await open_harness()
    try:
        await h.run(commands.AppendText(text="Answer now.", force_listen=force_listen))
        payload = h.port.submissions[0].prompt["model_intermediate_buffer"]["duplex"]["payload"]
        state = _MiniCPMO45Stage0SessionState(session_id="s0")
        state.current_turn_ended = True
        runtime = stage0_vision_runtime()
        kwargs = dict(
            input_text_token_ids=payload["text_token_ids"], force_listen=payload["force_listen"], epoch=0, seq=1
        )
        result = runtime._stage_prefill_embeddings_only(state, None, **kwargs)
        assert result["text_input"] is not force_listen
        model = _policy_model_with_state(state)
        logits = torch.zeros(1, VOCAB)
        model.prepare_duplex_sampling(logits, None, rows=(_policy_row(payload),))
        sampled = model._finalize_minicpmo45_native_duplex_sample(0, LISTEN, TOKEN_IDS)
        assert sampled == (LISTEN if force_listen else TOKEN_IDS["tts_bos_token_id"])
        assert state.current_turn_ended is force_listen
        if not force_listen:
            model._record_minicpmo45_duplex_terminator(0, 20, TOKEN_IDS)
            model._record_minicpmo45_duplex_terminator(0, TOKEN_IDS["turn_eos_token_id"], TOKEN_IDS)
            # Replayed prefill and later decode steps cannot reopen a reply
            # that has already ended on this same input fragment.
            runtime._stage_prefill_embeddings_only(state, None, **kwargs)
            model.prepare_duplex_sampling(logits, None, rows=(_policy_row(payload),))
            assert state.current_turn_ended
    finally:
        await close_harness(h)


@pytest.mark.asyncio
@pytest.mark.parametrize("finish", ["submit", "cancel", "close", "failure"])
async def test_queued_text_bytes_are_bounded_and_released(finish):
    h = await open_harness(runtime_config=DuplexSessionRuntimeConfig(max_pending_input_bytes_per_session=12))
    h.port.submit_gate = asyncio.Event()
    try:
        h.submit(commands.AppendText(text="你好", force_listen=True))
        await asyncio.wait_for(h.port.submit_started.wait(), timeout=2)
        h.submit(commands.AppendText(text="世界", force_listen=True))
        assert h.session.pending_input_bytes == 12  # UTF-8, not character count
        h.submit(commands.AppendText(text="!", event_id="overflow"))
        events = await h.settle(timeout_s=0.1)
        assert any(
            e.type == "error" and e.code == "input_backpressure" and e.related_event_id == "overflow" for e in events
        )
        if finish == "cancel":
            await h.run(commands.CancelInput())
        elif finish == "close":
            await close_harness(h)
        elif finish == "failure":
            h.port.fail_submit = RuntimeError("stage unavailable")
        h.port.submit_gate.set()
        await h.settle()
        assert h.session.pending_input_bytes == 0
        if finish == "submit":
            assert [s.prompt["model_intermediate_buffer"]["duplex"]["payload"]["text"] for s in h.port.submissions] == [
                "你好",
                "世界",
            ]
    finally:
        h.port.submit_gate.set()
        await close_harness(h)


@pytest.mark.asyncio
async def test_manual_response_mode_rejects_text_without_leaking_admission():
    h = await open_harness(auto_response=False)
    try:
        events = await h.run(commands.AppendText(text="hello", event_id="manual"))
        assert types(events) == ["error"]
        assert events[0].related_event_id == "manual"
        assert "auto_response" in events[0].message
        assert h.session.pending_input_bytes == 0
        assert not h.port.submissions
    finally:
        await close_harness(h)
