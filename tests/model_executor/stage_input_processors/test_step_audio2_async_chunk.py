# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.models.step_audio2.step_audio2_constants import (
    DEFAULT_STREAM_CONFIG,
    DEFAULT_TOKEN_CONFIG,
)
from vllm_omni.model_executor.stage_input_processors.step_audio2 import (
    thinker2token2wav_async_chunk,
)


def _req(external_req_id: str, *, prompt_token_ids: list[int], all_token_ids: list[int], finished: bool):
    return SimpleNamespace(
        external_req_id=external_req_id,
        prompt_token_ids=prompt_token_ids,
        all_token_ids=all_token_ids,
        is_finished=lambda: finished,
    )


def test_step_audio2_async_chunk_uses_decode_only_tokens_not_prompt_history():
    audio_start = DEFAULT_TOKEN_CONFIG.audio_start
    audio_eos = DEFAULT_TOKEN_CONFIG.audio_eos
    transfer_manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list))

    prompt = [11, 12, audio_start + 5]  # historical prompt audio token
    generated = [99, audio_start + 1, audio_start + 2, audio_start + audio_eos]
    request = _req(
        "rid-decode-only",
        prompt_token_ids=prompt,
        all_token_ids=prompt + generated,
        finished=True,
    )

    payload = thinker2token2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload is not None
    assert payload["code_predictor_codes"] == [1, 2]
    assert payload["left_context_size"] == 1
    assert payload["finished"].item() is True


def test_step_audio2_async_chunk_returns_none_when_not_enough_tokens():
    audio_start = DEFAULT_TOKEN_CONFIG.audio_start
    required = DEFAULT_STREAM_CONFIG.chunk_size + DEFAULT_STREAM_CONFIG.pre_lookahead_len
    transfer_manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list))

    generated = [audio_start + i for i in range(required - 1)]
    request = _req(
        "rid-not-ready",
        prompt_token_ids=[1, 2, 3],
        all_token_ids=[1, 2, 3] + generated,
        finished=False,
    )

    payload = thinker2token2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload is None


def test_step_audio2_async_chunk_emits_non_last_chunk_and_advances_consumed_by_chunk_size():
    audio_start = DEFAULT_TOKEN_CONFIG.audio_start
    chunk_size = DEFAULT_STREAM_CONFIG.chunk_size
    required = chunk_size + DEFAULT_STREAM_CONFIG.pre_lookahead_len
    transfer_manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list))

    generated = [audio_start + i for i in range(required + 10)]
    request = _req(
        "rid-ready",
        prompt_token_ids=[7, 8],
        all_token_ids=[7, 8] + generated,
        finished=False,
    )

    payload = thinker2token2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload is not None
    assert payload["left_context_size"] == 0
    assert payload["finished"].item() is False
    assert payload["code_predictor_codes"] == list(range(required))
    assert transfer_manager.code_prompt_token_ids["rid-ready"] == list(range(chunk_size))


def test_step_audio2_async_chunk_emits_eof_when_finished_with_no_remaining_audio():
    transfer_manager = SimpleNamespace(code_prompt_token_ids=defaultdict(list))
    transfer_manager.code_prompt_token_ids["rid-eof"] = [1, 2, 3]

    request = _req(
        "rid-eof",
        prompt_token_ids=[10, 11],
        all_token_ids=[
            10,
            11,
            DEFAULT_TOKEN_CONFIG.audio_start + 1,
            DEFAULT_TOKEN_CONFIG.audio_start + 2,
            DEFAULT_TOKEN_CONFIG.audio_start + 3,
        ],
        finished=True,
    )

    payload = thinker2token2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload == {
        "code_predictor_codes": [],
        "left_context_size": 1,
        "finished": torch.tensor(True, dtype=torch.bool),
    }
