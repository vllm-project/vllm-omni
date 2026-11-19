# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Native text appends share ordering, admission, and cancellation with media."""

import asyncio

import pytest
import torch

from tests.engine.duplex.test_minicpmo_window_plugin import _stub_tokenizer
from tests.engine.duplex.test_session_runner import append_audio, close_harness, listen_output, open_harness, types
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
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import _MiniCPMO45Stage0SessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture(autouse=True)
def text_tokenizer(monkeypatch):
    _stub_tokenizer(monkeypatch)


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
        assert [p["seq"] for p in inputs] == [1, 2, 3]
        assert [p["payload"].get("text") for p in inputs] == ["记住", "蓝色", None]
        assert inputs[0]["payload"]["text_token_ids"] == [ord(c) for c in "记住"]
        assert inputs[1]["payload"]["force_listen"] is True
        assert len(h.port.submissions[1].prompt["prompt_token_ids"]) == 2 + 1 + len("蓝色")
        # Text units did not start the mel window: this is still the first
        # 1035 ms audio unit, despite its append sequence being 3.
        assert len(h.port.submissions[2].prompt["prompt_token_ids"]) == 2 + 12 - 1
        assert h.runner.model_state.audio_buffer._audio_window_started
        assert h.session.pending_input_bytes == h.runner.model_state.audio_buffer.pending_byte_count == 64000
        await h.run(commands.ClearInput())
        assert h.session.pending_input_bytes == 0
    finally:
        await close_harness(h)


@pytest.mark.asyncio
async def test_unforced_text_continues_after_an_empty_model_unit():
    h = await open_harness()
    try:
        await h.run(commands.AppendText(text="Remember blue whale.", force_listen=True))
        assert h.session.active_response_id is None
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
        events = await h.deliver_and_settle(
            empty,
            stage_id=0,
            segment_finished=True,
            segment_token_ids=[151718],
            segment_output_metadata=empty.multimodal_output,
        )
        assert "error" not in types(events)
        assert len(h.port.submissions) == 3
        assert h.runner.model_state.continuation_units == 1
        assert h.session.active_response_id == response_id
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
        kwargs = dict(text_token_ids=payload["text_token_ids"], force_listen=payload["force_listen"], epoch=0, seq=1)
        runtime._stage_prefill_embeddings_only(state, None, **kwargs)
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
