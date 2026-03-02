# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import torch

from vllm_omni.model_executor.stage_input_processors.qwen3_tts import talker2code2wav_async_chunk


def _req(external_req_id: str, *, finished: bool):
    return SimpleNamespace(
        external_req_id=external_req_id,
        is_finished=lambda: finished,
    )


def test_talker2code2wav_async_chunk_does_not_emit_empty_chunk_when_not_finished():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 25}}),
    )
    request = _req("rid-empty", finished=False)

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output={"audio_codes": torch.zeros((0,))},
        request=request,
    )

    assert payload is None


def test_talker2code2wav_async_chunk_flushes_tail_when_finished_without_pooler_output():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 25}}),
    )
    request_id = "rid-tail"
    transfer_manager.code_prompt_token_ids[request_id] = [[1, 2, 3, 4] for _ in range(24)]
    request = _req(request_id, finished=True)

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,  # e.g. EOS step with no audio_codes
        request=request,
    )

    assert payload is not None
    assert payload["finished"].item() is True
    # ctx_frames header + flat codes
    assert len(payload["code_predictor_codes"]) == 1 + 4 * 24


def test_talker2code2wav_async_chunk_emits_eof_marker_when_finished_with_no_frames():
    transfer_manager = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 25}}),
    )
    request = _req("rid-eof", finished=True)

    payload = talker2code2wav_async_chunk(
        transfer_manager=transfer_manager,
        pooling_output=None,
        request=request,
    )

    assert payload == {
        "code_predictor_codes": [],
        "finished": torch.tensor(True, dtype=torch.bool),
    }
