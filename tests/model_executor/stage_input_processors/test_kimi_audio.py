# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for Kimi-Audio request admission and stage transfer."""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.kimi_audio.audio_processing import prepare_kimi_audio_inputs
from vllm_omni.model_executor.models.kimi_audio.prompt import KimiAudioPromptBuilder, KimiAudioSpecialTokens
from vllm_omni.model_executor.stage_input_processors.kimi_audio import (
    kimi_audio_to_decoder,
    kimi_audio_to_decoder_async_chunk,
    prepare_kimi_audio_request,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def prompt_builder():
    return KimiAudioPromptBuilder(
        encode_text=lambda text: list(text.encode("utf-8")),
        special_tokens=KimiAudioSpecialTokens(
            msg_end=0,
            media_begin=1,
            media_end=2,
            kimia_text_blank=3,
            kimia_text_eos=4,
            kimia_user_msg_start=5,
            kimia_assistant_msg_start=6,
            kimia_speech_ct_id=7,
            kimia_speech_ctd_id=8,
        ),
        audio_token_offset=256,
        audio_vocab_size=128,
        audio_delay=2,
        continuous_feature_size=4,
    )


@pytest.fixture
def audio_request(prompt_builder):
    prompt = prepare_kimi_audio_inputs(
        [{"role": "user", "message_type": "text", "content": "Hi"}], prompt_builder, output_type="both"
    )
    params = SamplingParams(include_stop_str_in_output=True, stop_token_ids=[prompt_builder.tokens.msg_end], seed=42)
    return prepare_kimi_audio_request(prompt, [params]), params


@pytest.mark.parametrize("codes", [[7, 8], []])
def test_complete_audio_transfer_removes_controls_and_offset(prompt_builder, audio_request, codes):
    prompt, params = audio_request
    tokens = [prompt_builder.audio_token_offset + code for code in codes] + [prompt_builder.tokens.media_end]
    source = SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(finish_reason="stop", multimodal_output={"codes": {"audio": torch.tensor(tokens)}})],
    )
    converted = kimi_audio_to_decoder([source], prompt)[0]
    assert converted["prompt_token_ids"] == (codes or [0])
    assert converted["model_intermediate_buffer"]["codes"]["audio"] == codes
    assert converted["model_intermediate_buffer"]["meta"] == {"finished": True, "audio_seed": params.seed}


@pytest.mark.parametrize("code_count", [30, 35, 60])
def test_stream_transfer_emits_each_code_once_and_flushes_final_chunk(prompt_builder, audio_request, code_count):
    prompt, params = audio_request
    transfer = SimpleNamespace(
        request_payload={},
        code_prompt_token_ids=defaultdict(list),
        record_send_failure=lambda request_id, reason: pytest.fail(reason),
    )
    request = SimpleNamespace(
        request_id="internal-request",
        external_req_id="request",
        model_intermediate_buffer=prompt["model_intermediate_buffer"],
        sampling_params=params,
        is_finished=lambda: False,
    )
    chunks = []
    for code in range(code_count):
        chunk = kimi_audio_to_decoder_async_chunk(
            transfer, {"codes": {"audio": torch.tensor([prompt_builder.audio_token_offset + code])}}, request
        )
        if chunk is not None:
            assert not chunk.meta.stream_finished.item()
            assert chunk.codes.audio.shape[0] == 30
            chunks.append(chunk)
    final = kimi_audio_to_decoder_async_chunk(
        transfer,
        {"codes": {"audio": torch.tensor([prompt_builder.tokens.media_end])}},
        request,
        is_finished=True,
    )
    assert final is not None and final.meta.stream_finished.item()
    chunks.append(final)
    assert torch.cat([chunk.codes.audio for chunk in chunks]).flatten().tolist() == list(range(code_count))
    assert [chunk.meta.chunk_seq for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.meta.audio_seed == params.seed for chunk in chunks)
    assert transfer.code_prompt_token_ids[request.external_req_id] == []
