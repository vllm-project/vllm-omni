# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from tools.lychee_session_lifecycle_probe import SYSTEM_PREFIX, _open
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.lychee_fd.duplex.codec import LycheeCodecStreams
from vllm_omni.model_executor.models.lychee_fd.duplex.data_plane import LycheeDataPlaneContext, LycheeDataPlaneSession
from vllm_omni.model_executor.models.lychee_fd.duplex.history import LycheeSessionHistory
from vllm_omni.model_executor.models.lychee_fd.duplex.plugin import LycheeDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]
REQ = "duplex-s.cHJvYmU.e.0.r.stage0"


def snapshot(ticks, *, speech=None, controls=None, epoch=0):
    count = len(ticks)
    return {
        "lychee_tick": torch.tensor(ticks),
        "lychee_text_token_ids": torch.tensor([158358] * count),
        "lychee_speech_token_ids": torch.tensor(speech or [158359] * count),
        "lychee_control_token_ids": torch.tensor(controls or [158357] * count),
        "lychee_execution_epoch": torch.tensor([epoch] * count),
    }


def result(payload, *, finished=False):
    completion = SimpleNamespace(multimodal_output=payload, finish_reason=None)
    return {"data_plane_outputs": [SimpleNamespace(request_id=REQ, finished=finished, outputs=[completion])]}


def context(epoch=0):
    return LycheeDataPlaneContext(
        epoch=epoch,
        turn_id=0,
        active_response_turn_id=0,
        active_response_id="public-response",
        auto_responds=True,
        response_format="pcm16",
        speed=None,
        modalities=("audio",),
    )


def test_initial_prefill_and_rebuild_keep_exact_channel_masks_and_frontier():
    history = LycheeSessionHistory(list(SYSTEM_PREFIX))
    assert history.text == SYSTEM_PREFIX + [158358]
    assert history.speech == [None] * 10 + [158359]
    assert history.control == [None] * 10 + [158357]
    assert history.ticks == [-1] * 10 + [0]
    history.record_outputs(snapshot(list(range(1, 10))))
    history.record_outputs(snapshot(list(range(1, 10))))
    saved = history.snapshot(execution_epoch=1)
    assert saved["logical_ticks"] == [-1] * 10 + list(range(10))
    assert len(saved["text_input_ids"]) == 20
    assert saved["execution_epoch"] == 1
    with pytest.raises(ValueError, match="history gap"):
        history.record_outputs(snapshot([11]))


def test_append_budget_clones_defaults_and_rebuild_replays_committed_evidence():
    plugin = LycheeDuplexPlugin(lambda *args: None)
    params = SamplingParams(max_tokens=10, min_tokens=10, ignore_eos=True)
    payload = {
        "audio": "pcmfixture",
        "format": "pcm_f32le",
        "sample_rate_hz": 16000,
        "lychee_audio_ledger": {"consumable_tick_start": 0, "consumable_tick_end": 10},
    }
    common = dict(
        session_config={},
        runtime_config={"lychee_system_token_ids": SYSTEM_PREFIX},
        turn_seq=1,
        payload=payload,
        final=False,
        sampling_params=params,
    )
    first = plugin.plan_append(request_id=REQ, fence=DuplexFence("probe"), seq=1, **common)
    assert first.sampling_params.max_tokens == 9
    assert params.max_tokens == 10
    plugin.histories["probe"].record_outputs(snapshot([1, 2, 3, 4, 5]))
    second = plugin.plan_append(request_id="new-binding", fence=DuplexFence("probe", epoch=1), seq=1, **common)
    assert second.sampling_params.max_tokens == 14
    history = second.prompt["model_intermediate_buffer"]["duplex"]["lychee_history"]
    assert history["logical_ticks"][-1] == 5
    assert [window["start_tick"] for window in history["audio_windows"]] == [0, 10]
    assert history["execution_epoch"] == 1


def test_codec_delta_is_once_and_final_is_response_scoped():
    streams = LycheeCodecStreams()
    assert streams.consume(REQ, snapshot([9], controls=[158352]), session_epoch=0) is None
    start = snapshot([10, 11], speech=[151693, 151696])
    packet = streams.consume(REQ, start, session_epoch=0)
    assert packet["codec_ids"] == [0]
    assert packet["response_number"] == 1
    assert streams.consume(REQ, start, session_epoch=0) is None
    final = streams.consume(REQ, snapshot([12], speech=[151694]), session_epoch=0)
    assert final["empty"] is True and final["final"] is True
    assert final["response_id"] == packet["response_id"]
    assert streams.consume(REQ, snapshot([12], speech=[151694]), session_epoch=0) is None
    streams.close_request(REQ)
    assert streams.states == {}


def test_codec_rejects_tokenizer_only_speech_ids():
    streams = LycheeCodecStreams()
    with pytest.raises(ValueError, match="codebook"):
        streams.consume(REQ, snapshot([9], speech=[158300], controls=[158352]), session_epoch=0)


def test_waveform_final_dedup_late_epoch_and_steady_response_state():
    encoded = []

    def encode(audio: torch.Tensor, *_args: object) -> str:
        encoded.append(audio.clone())
        return "pcm"

    plane = LycheeDataPlaneSession(encode)
    for number in range(1, 301):
        owner = {
            "response_id": f"model-{number}",
            "response_number": number,
            "execution_epoch": 0,
            "session_epoch": 0,
            "chunk_seq": 1,
            "final": True,
        }
        chunk = result({"audio": torch.ones(240), "sr": torch.tensor(24000), "lychee_t2w": owner})
        (event,) = tuple(plane.project(chunk, context=context()))
        assert event["end_of_turn"] is True and event["audio_duration_ms"] == 10
        assert tuple(plane.project(chunk, context=context())) == ()
    assert len(encoded) == 300
    assert plane._audio_seq == {}
    assert len(plane._audio_completed) == 1
    owner.update(response_id="old-epoch", response_number=301)
    assert tuple(plane.project(result({"audio": torch.ones(2), "lychee_t2w": owner}), context=context(epoch=1))) == ()
    plane.close_session("probe", active_request_id=REQ)
    assert not plane._audio_seq and not plane._audio_completed


def test_error_projection_rebinds_real_session_and_retains_history():
    async def exercise():
        harness = await _open()
        try:
            session = harness.runner.session
            request_id = harness.port.ensured[0].request_id
            session.bind_request(request_id)
            history = LycheeSessionHistory(SYSTEM_PREFIX)
            history.record_outputs(snapshot([1, 2, 3]))
            harness.runner.plugin.histories[session.session_id] = history
            await harness.runner.model._send_one_model_output_event(
                {
                    "data_plane_request_id": request_id,
                    "error_code": "lychee_request_aborted",
                    "error": "merge fault",
                    "retryable": True,
                    "recover_binding": True,
                },
                expected_epoch=0,
            )
            assert session.epoch == 1
            assert session.active_request_id is None
            assert harness.port.cleanups[-1] == ([request_id], True)
            assert harness.runner.plugin.histories[session.session_id].frontier_tick == 3
            assert harness.runner.plugin.data_plane.is_terminal(request_id)
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_cumulative_snapshot_drains_every_response_without_a_later_output():
    streams = LycheeCodecStreams()
    payload = snapshot(
        [9, 10, 11, 19, 20],
        speech=[151693, 151697, 151694, 151693, 151698],
        controls=[158352, 158357, 158357, 158352, 158357],
    )
    packets = streams.consume_all(REQ, payload, session_epoch=0)
    assert [(p["response_number"], p["codec_ids"], p["final"]) for p in packets] == [(1, [1], True), (2, [2], False)]
    assert streams.states[REQ].last_tick == 20
    assert streams.consume_all(REQ, payload, session_epoch=0) == []


def test_backchannel_to_speak_closes_distinct_waveform_owners():
    streams = LycheeCodecStreams()
    payload = snapshot(
        [9, 10, 19, 20, 21],
        speech=[151693, 151696, 151693, 151697, 151694],
        controls=[158362, 158357, 158352, 158357, 158357],
    )
    first, second = streams.consume_all(REQ, payload, session_epoch=0)
    assert first["codec_ids"] == [0] and first["final"] is True
    assert first["tick"] == 10 and first["chunk_seq"] == 0
    assert second["codec_ids"] == [1] and second["final"] is True
    assert first["response_id"] != second["response_id"]
    assert second["tick"] == 21 and second["response_number"] == 2


def test_delayed_pcm_final_keeps_actual_public_responses_and_text_separate():
    async def exercise():
        harness = await _open()
        try:
            session = harness.runner.session
            request_id = harness.port.ensured[0].request_id
            session.bind_request(request_id)
            plane = harness.runner.plugin.data_plane
            plane._decode_text = lambda ids: "".join({101: "first", 102: "second"}[i] for i in ids)
            plane._encode_audio = lambda *args: "pcm"

            def output(payload):
                completion = SimpleNamespace(multimodal_output=payload, finish_reason=None)
                return {"data_plane_outputs": [SimpleNamespace(request_id=request_id, outputs=[completion])]}

            async def deliver(payload):
                for event in plane.project(output(payload), context=context()):
                    await harness.runner.model._send_one_model_output_event(event, expected_epoch=0)

            first = snapshot([9, 10], speech=[151693, 151694], controls=[158352, 158357])
            first["lychee_text_token_ids"] = torch.tensor([101, 158358])
            await deliver(first)
            response1 = session.active_response_id
            assert session.assistant_transcript() == "first"
            second = snapshot([19, 20], speech=[151693, 151694], controls=[158352, 158357])
            second["lychee_text_token_ids"] = torch.tensor([102, 158358])
            await deliver(second)
            assert session.active_response_id == response1
            assert session.assistant_transcript() == "first"

            def pcm(number):
                return {
                    "model_outputs": torch.ones(240),
                    "sr": torch.tensor(24000),
                    **{
                        f"lychee_t2w.{k}": torch.tensor(v)
                        for k, v in {
                            "response_number": number,
                            "session_epoch": 0,
                            "execution_epoch": 0,
                            "chunk_seq": 0,
                            "tick": number * 10,
                            "final": True,
                            "discarded": False,
                        }.items()
                    },
                }

            # Even if transport delivers the later final first, it stays queued.
            await deliver(pcm(2))
            assert session.active_response_id == response1
            await deliver(pcm(1))
            assert session.active_response_id is None
            events = await harness.settle()
            created = [e.response_id for e in events if e.type == "response.created"]
            done = [e.response_id for e in events if e.type == "response.done"]
            deltas = [
                (e.response_id, getattr(e, "text", None))
                for e in events
                if e.type == "response.output_audio_transcript.delta" and getattr(e, "text", None)
            ]
            assert len(created) == 2 and created[0] != created[1]
            assert done == created
            assert deltas == [(created[0], "first"), (created[1], "second")]
            assert plane._pending_events == {} and plane._audio_seq == {}
            assert plane._published_completed[request_id] == 2
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_audio_only_session_close_releases_ordering_and_queued_response_owners():
    plane = LycheeDataPlaneSession(lambda *args: "pcm")

    def pcm(number):
        return result(
            {
                "audio": torch.ones(240),
                "lychee_t2w": {
                    "response_number": number,
                    "session_epoch": 0,
                    "execution_epoch": 0,
                    "chunk_seq": 0,
                    "final": True,
                },
            }
        )

    assert tuple(plane.project(pcm(2), context=context())) == ()
    assert plane._pending_events
    plane.close_session("probe")
    assert not plane._pending_events and not plane._audio_seq and not plane._audio_completed
    assert not plane._published_completed and not plane.codec_streams.states


@pytest.mark.parametrize("cancel_input", [False, True])
def test_explicit_input_cancel_discards_only_uncomputed_audio_evidence(cancel_input):
    from vllm_omni.engine.duplex.commands import CancelInput, CancelResponse

    async def exercise():
        harness = await _open()
        try:
            session = harness.runner.session
            request_id = harness.port.ensured[0].request_id
            session.bind_request(request_id)
            history = LycheeSessionHistory(SYSTEM_PREFIX)
            for seq in (1, 2):
                history.append_audio(
                    {
                        "audio": "original-pcm",
                        "lychee_audio_ledger": {
                            "consumable_tick_start": (seq - 1) * 10,
                            "consumable_tick_end": seq * 10,
                        },
                    },
                    epoch=0,
                    seq=seq,
                )
            history.record_outputs(snapshot([1, 2, 3, 4, 5]))
            history.request_ids.add(request_id)
            harness.runner.plugin.histories[session.session_id] = history
            if not cancel_input:
                session.begin_response()
            await harness.run(CancelInput() if cancel_input else CancelResponse())
            assert history.frontier_tick == 5 and history.force_listen_at_frontier
            if cancel_input:
                assert [w["discard_after_tick"] for w in history.audio_windows] == [4, 9]
            else:
                assert all("discard_after_tick" not in w for w in history.audio_windows)
            assert all(w["payload"]["audio"] == "original-pcm" for w in history.audio_windows)
            assert session.epoch == 1
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())


@pytest.mark.parametrize("nested", [False, True])
def test_materialized_waveform_rows_reconstruct_all_chunks_and_owners(nested):
    seen = []

    def encode(audio: torch.Tensor, *_args: object) -> str:
        seen.append(audio.tolist())
        return "pcm"

    plane = LycheeDataPlaneSession(encode)
    payload = {
        "model_outputs": torch.tensor([0.1, 0.2, 0.3]),
        "sr": torch.tensor([24000, 24000]),
        **{
            f"lychee_t2w.{name}": torch.tensor(values)
            for name, values in {
                "session_epoch": [0, 0],
                "execution_epoch": [0, 0],
                "response_number": [1, 1],
                "chunk_seq": [0, 1],
                "tick": [11, 12],
                "final": [False, True],
                "discarded": [False, False],
                "num_samples": [2, 1],
            }.items()
        },
    }
    if nested:
        payload["chunk"] = {key: payload.pop(key) for key in list(payload) if key.startswith("lychee_t2w.")}
    events = tuple(plane.project(result(payload), context=context()))
    assert len(events) == 2 and [e["end_of_turn"] for e in events] == [False, True]
    assert seen[0] == pytest.approx([0.1, 0.2]) and seen[1] == pytest.approx([0.3])
    assert tuple(plane.project(result(payload), context=context())) == ()
    (payload["chunk"] if nested else payload)["lychee_t2w.num_samples"] = torch.tensor([1, 1])
    with pytest.raises(ValueError, match="lengths disagree"):
        tuple(plane.project(result(payload), context=context()))


def test_first_submit_failure_retries_full_history_before_binding():
    from tools.lychee_session_lifecycle_probe import _append

    async def exercise():
        harness = await _open()
        original_submit = harness.port.submit
        calls = 0

        async def fail_once(submission):
            nonlocal calls
            calls += 1
            if calls == 1:
                harness.port.submissions.append(submission)
                raise RuntimeError("injected first submit failure")
            return await original_submit(submission)

        harness.port.submit = fail_once
        try:
            await harness.run(_append(6400))
            session = harness.runner.session
            history = harness.runner.plugin.histories[session.session_id]
            request_id = harness.port.ensured[0].request_id
            assert request_id not in history.request_ids
            assert not session.stage_request_submitted(0, request_id)
            assert harness.runner.model_state.audio_buffer.pending_byte_count == 6400 * 4
            history.force_listen_at_frontier = True
            await harness.run(_append(3200))
            retry = harness.port.submissions[-1]
            saved = retry.prompt["model_intermediate_buffer"]["duplex"]["lychee_history"]
            assert retry.prompt["prompt_token_ids"] == SYSTEM_PREFIX + [158358]
            assert retry.context.stage_sampling_params.max_tokens == 9
            assert retry.already_submitted is False
            assert saved["force_listen_at_frontier"] is True
            assert len(history.audio_windows) == 1
            assert request_id in history.request_ids
            assert history.force_listen_at_frontier is False
            await harness.run(_append(3200))
            appended = harness.port.submissions[-1]
            assert appended.prompt["prompt_token_ids"] == [158358]
            assert appended.context.stage_sampling_params.max_tokens == 10
            assert appended.already_submitted is True
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())


def test_raw_worker_error_reaches_actual_session_mailbox_and_rebuilds_once():
    async def exercise():
        harness = await _open()
        try:
            session = harness.runner.session
            request_id = harness.port.ensured[0].request_id
            session.bind_request(request_id)
            history = LycheeSessionHistory(SYSTEM_PREFIX)
            history.record_outputs(snapshot([1, 2, 3]))
            harness.runner.plugin.histories[session.session_id] = history
            history.request_ids.add(request_id)
            error = "Lychee sampling transaction aborted; rebuild required; execution_epoch=1: injected fault"
            harness.runner.on_stage_request_error(0, error, request_id=request_id, expected_epoch=0)
            events = await harness.settle()
            assert session.epoch == 1
            assert any(event.type == "error" for event in events)
            assert any(event.type == "session.updated" for event in events)
            assert history.frontier_tick == 3 and history.force_listen_at_frontier
            assert harness.port.cleanups[-1] == ([request_id], True)
            count = len(harness.port.cleanups)
            harness.runner.on_stage_request_error(0, error, request_id=request_id, expected_epoch=0)
            assert await harness.settle() == []
            assert len(harness.port.cleanups) == count and session.epoch == 1
        finally:
            await harness.manager.shutdown()

    asyncio.run(exercise())
