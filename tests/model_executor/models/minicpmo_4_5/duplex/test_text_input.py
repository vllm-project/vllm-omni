# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, StreamingUpdate

from tests.model_executor.models.minicpmo_4_5.helpers import stage0_vision_runtime
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import build_duplex_data_plane_prompt
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import APPEND_TEXT_TOKEN_IDS_KEY, INTERRUPTION_KEY
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import _MiniCPMO45Stage0SessionState
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_plan import prepare_stage0_window
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
    _duplex_row_outputs,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("mode", ["basic", "context"])
def test_interrupted_unit_and_followup_keep_scheduler_and_worker_window_lengths_in_sync(mode):
    runtime = stage0_vision_runtime()
    state = _MiniCPMO45Stage0SessionState(session_id="window-interrupt", window_enabled=True)
    state.context_token_ids = state.context_prefix_token_ids = [200, 201]
    state.context_embeds = state.context_prefix_embeds = [runtime._embed_token(i) for i in [200, 201]]
    runtime._stage_prefill_embeddings_only(state, None, input_text_token_ids=[21, 22], epoch=0, seq=1)
    request = Request(
        request_id="resident",
        prompt_token_ids=[0] * 5,
        sampling_params=SamplingParams(max_tokens=8),
        pooling_params=None,
        arrival_time=0,
        block_hasher=None,
    )
    request.append_output_token_ids([51, 52, runtime.chunk_eos_token_id])
    request.num_computed_tokens = 7
    cfg = {
        "duplex_first_append_context_tokens": 2,
        "duplex_window_prefix_tokens": 2,
        "duplex_window_previous_marker_token_ids": [70, 71],
        "duplex_window_special_token_ids": list(runtime._special_token_ids().values()),
        "duplex_window_config": {
            "sliding_window_mode": mode,
            "basic_window_high_tokens": 14,
            "basic_window_low_tokens": 10,
            "context_max_units": 1,
        },
    }
    for seq, interrupted, text_ids, generated, terminator in (
        (2, True, [runtime.turn_eos_token_id], [51, 52], runtime.chunk_eos_token_id),
        (3, False, [23], [], runtime.listen_token_id),
    ):
        payload = {
            "type": "text",
            "text_token_ids": text_ids,
            "force_listen": interrupted,
            INTERRUPTION_KEY: interrupted,
        }
        update = StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[0] * 4,
            max_tokens=8,
            arrival_time=seq,
            sampling_params=SamplingParams(max_tokens=8),
        )
        update.model_intermediate_buffer = {
            "duplex": {
                "data_plane": True,
                "seq": seq,
                "payload": payload,
                "runtime_config": cfg,
            }
        }
        replaced = prepare_stage0_window(request, update, segment_output_ids=generated, completed_terminator=terminator)
        plan = update.model_intermediate_buffer["duplex"]["stage0_window"]
        result = runtime._stage_prefill_embeddings_only(
            state,
            None,
            input_text_token_ids=text_ids,
            interruption=interrupted,
            force_listen=interrupted,
            epoch=0,
            seq=seq,
            stage0_window=plan,
        )
        assert [u["length"] for u in request._minicpmo45_window_units] == [len(u.token_ids) for u in state.window_units]
        if seq == 2:
            assert not replaced
            assert request._minicpmo45_window_open_start == 10
            assert result["num_input_tokens"] == 4
            # Apply the same prompt extension as the production scheduler; the
            # forced LISTEN is the next uncomputed sample, not part of the KV.
            scheduler = Scheduler.__new__(Scheduler)
            scheduler.hash_block_size = 16
            scheduler.log_stats = False
            scheduler._update_request_as_session(request, update)
            request.append_output_token_ids(runtime.listen_token_id)
            request.num_computed_tokens = request.num_prompt_tokens
        else:
            assert replaced
            assert result["num_input_tokens"] == len(update.prompt_token_ids)
            assert state.pending_window_unit.token_ids == [runtime.unit_token_id, 23]


@pytest.mark.parametrize(
    "window,terminator",
    [(False, "normal"), (True, "normal"), (False, None), (False, "chunk_eos_token_id"), (True, "chunk_eos_token_id")],
)
def test_text_units_match_exact_scheduler_budget_and_preserve_audio_state(window, terminator):
    runtime = stage0_vision_runtime()
    state = _MiniCPMO45Stage0SessionState(session_id="text", window_enabled=window)
    state.context_token_ids = [200, 201]
    state.context_embeds = [runtime._embed_token(i) for i in state.context_token_ids]
    state.audio_buffer = np.array([0.125, 0.25], dtype=np.float32)
    for seq, text_ids in enumerate(([21, 22, 23], [24, 25]), start=1):
        interruption = terminator != "normal" and seq == 2
        if interruption:
            text_ids = [runtime.turn_eos_token_id]
            state.pending_terminator_token = getattr(runtime, terminator) if terminator else None
            state.generated_tokens = [51, 52]
            state.current_turn_ended = False
            state.pending_speech_context = state.pending_speech_response_open = True
        payload = {
            "type": "text",
            "text_token_ids": text_ids,
            "force_listen": interruption,
            INTERRUPTION_KEY: interruption,
        }
        audio = runtime._decode_audio_payload(payload)
        assert audio.size == 0
        kwargs = MiniCPMO45OmniForConditionalGeneration._minicpmo45_duplex_prefill_kwargs(
            {"epoch": 0, "seq": seq}, payload
        )
        if interruption:
            kwargs["stage0_window"] = {"completed_token_ids": [51, 52]}
        result = runtime._stage_prefill_embeddings_only(state, audio, **kwargs)
        prompt = build_duplex_data_plane_prompt(
            request_id="text",
            fence=DuplexFence("text"),
            session_config={},
            runtime_config={"duplex_first_append_context_tokens": 2},
            seq=seq,
            turn_seq=seq,
            payload=payload,
            final=False,
        )
        closure = ([runtime.turn_eos_token_id] if interruption else []) + [
            runtime.listen_token_id,
            runtime.unit_end_token_id,
        ]
        expected = ([200, 201] if seq == 1 else closure) + [runtime.unit_token_id]
        if not interruption:
            expected += text_ids
        assert result["success"] and result["input_token_ids"] == expected
        assert result["num_input_tokens"] == len(prompt["prompt_token_ids"]) == len(expected)
        assert _duplex_row_outputs([{"duplex": result}, {"duplex": {}}])[INTERRUPTION_KEY] == [interruption, False]
        torch.testing.assert_close(
            result["inputs_embeds"], torch.cat([runtime._embed_token(i).reshape(1, -1) for i in expected])
        )
        assert state.pending_speech_context is not interruption
        if interruption:
            assert state.generated_tokens == [51, 52] and state.current_turn_ended
            assert not state.pending_speech_response_open
        assert state.audio_encoder_chunk_idx == 0
        np.testing.assert_array_equal(state.audio_buffer, [0.125, 0.25])
        replay = runtime._stage_prefill_embeddings_only(state, audio, **kwargs)
        assert replay["inputs_embeds"] is result["inputs_embeds"]
        assert state.audio_chunk_idx == seq
        state.pending_terminator_token = runtime.listen_token_id
    if window:
        generated = [51, 52] if interruption else []
        assert len(state.window_units) == 1
        assert state.window_units[0].token_ids == [runtime.unit_token_id, 21, 22, 23, *generated, *closure]
        assert state.pending_window_unit.token_ids == [runtime.unit_token_id, *([] if interruption else [24, 25])]


@pytest.mark.parametrize(("input_ids", "result_ids"), [([21, 22], []), ([], [200, 201]), ([21, 22], [200, 201])])
def test_prefill_keeps_client_text_and_server_result_tokens_in_order(input_ids, result_ids):
    runtime = stage0_vision_runtime()
    state = _MiniCPMO45Stage0SessionState(session_id="mixed-text", window_enabled=True)
    payload = {"type": "text", "text_token_ids": input_ids} if input_ids else {"type": "audio"}
    kwargs = MiniCPMO45OmniForConditionalGeneration._minicpmo45_duplex_prefill_kwargs(
        {"epoch": 0, "seq": 1, "runtime_config": {APPEND_TEXT_TOKEN_IDS_KEY: result_ids}}, payload
    )
    audio = np.zeros(0 if input_ids else 4, dtype=np.float32)

    output = runtime._stage_prefill_embeddings_only(state, audio, **kwargs)

    expected = [runtime.unit_token_id, *(input_ids or [runtime._audio_embedding_placeholder_token_id()]), *result_ids]
    assert output["success"]
    assert output["input_token_ids"] == expected
    assert output["num_input_tokens"] == len(expected)
    assert state.pending_window_unit.token_ids == expected
    assert output["text_input"] is bool(input_ids)
    assert state.audio_encoder_chunk_idx == int(not input_ids)
