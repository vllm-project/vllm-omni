# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch

from tests.model_executor.models.minicpmo_4_5.helpers import stage0_vision_runtime
from vllm_omni.engine.duplex.contracts import DuplexFence
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.plugin import build_duplex_data_plane_prompt
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import APPEND_TEXT_TOKEN_IDS_KEY
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.stage0 import _MiniCPMO45Stage0SessionState
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import MiniCPMO45OmniForConditionalGeneration

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("window", [False, True])
def test_text_units_match_exact_scheduler_budget_and_preserve_audio_state(window):
    runtime = stage0_vision_runtime()
    state = _MiniCPMO45Stage0SessionState(session_id="text", window_enabled=window)
    state.context_token_ids = [200, 201]
    state.context_embeds = [runtime._embed_token(i) for i in state.context_token_ids]
    state.audio_buffer = np.array([0.125, 0.25], dtype=np.float32)
    for seq, text_ids in enumerate(([21, 22, 23], [24, 25]), start=1):
        payload = {"type": "text", "text_token_ids": text_ids}
        audio = runtime._decode_audio_payload(payload)
        assert audio.size == 0
        result = runtime._stage_prefill_embeddings_only(state, audio, input_text_token_ids=text_ids, epoch=0, seq=seq)
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
        expected = [200, 201] if seq == 1 else [runtime.listen_token_id, runtime.unit_end_token_id]
        expected += [runtime.unit_token_id, *text_ids]
        assert result["success"]
        assert result["input_token_ids"] == expected
        assert result["num_input_tokens"] == len(prompt["prompt_token_ids"])
        torch.testing.assert_close(
            result["inputs_embeds"], torch.cat([runtime._embed_token(i).reshape(1, -1) for i in expected])
        )
        assert state.pending_speech_context
        assert state.audio_encoder_chunk_idx == 0
        np.testing.assert_array_equal(state.audio_buffer, [0.125, 0.25])
        replay = runtime._stage_prefill_embeddings_only(state, audio, input_text_token_ids=text_ids, epoch=0, seq=seq)
        assert replay["inputs_embeds"] is result["inputs_embeds"]
        assert state.audio_chunk_idx == seq
        state.pending_terminator_token = runtime.listen_token_id
    if window:
        assert state.window_units[0].token_ids == [
            runtime.unit_token_id,
            21,
            22,
            23,
            runtime.listen_token_id,
            runtime.unit_end_token_id,
        ]
        assert state.pending_window_unit.token_ids == [runtime.unit_token_id, 24, 25]


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
