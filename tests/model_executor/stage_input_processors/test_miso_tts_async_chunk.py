# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.miso_tts import (
    _MISO_NUM_CODEBOOKS,
    talker2mimi,
    talker2mimi_async_chunk,
    talker2mimi_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_Q = _MISO_NUM_CODEBOOKS
_FRAME = list(range(_Q))


def _req(rid: str, *, finished: bool) -> SimpleNamespace:
    return SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: finished,
        additional_information=None,
    )


def _tm(*, chunk_frames: int = 25, left_context: int = 25, initial_chunk_frames: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": chunk_frames,
                    "codec_left_context_frames": left_context,
                    "initial_codec_chunk_frames": initial_chunk_frames,
                }
            }
        ),
    )


def test_async_chunk_empty_returns_none() -> None:
    tm = _tm()
    assert (
        talker2mimi_async_chunk(
            transfer_manager=tm,
            pooling_output={"codes": {"audio": torch.zeros(0)}},
            request=_req("r", finished=False),
        )
        is None
    )


def test_async_chunk_eof_when_finished_with_no_frames() -> None:
    tm = _tm()
    payload = talker2mimi_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req("r", finished=True),
        is_finished=True,
    )
    assert payload is not None
    assert payload.codes.audio.tolist() == []
    assert payload.meta.finished.item() is True


def test_async_chunk_waits_until_chunk_size() -> None:
    tm = _tm(chunk_frames=25)
    rid = "req-1"
    tm.code_prompt_token_ids[rid] = [_FRAME[:] for _ in range(24)]
    assert (
        talker2mimi_async_chunk(
            transfer_manager=tm,
            pooling_output=None,
            request=_req(rid, finished=False),
        )
        is None
    )


def test_async_chunk_emits_codebook_major_tensor() -> None:
    tm = _tm(chunk_frames=2, left_context=0)
    rid = "req-1"
    f0 = [i for i in range(_Q)]
    f1 = [i + 100 for i in range(_Q)]
    tm.code_prompt_token_ids[rid] = [f0, f1]
    payload = talker2mimi_async_chunk(
        transfer_manager=tm,
        pooling_output=None,
        request=_req(rid, finished=True),
        is_finished=True,
    )
    assert payload is not None
    audio = payload.codes.audio
    assert len(audio) == _Q * 2
    # Layout: for each codebook q, all frames f — window[f][q]
    assert audio[0].item() == f0[0]
    assert audio[1].item() == f1[0]
    assert audio[2].item() == f0[1]
    assert audio[3].item() == f1[1]
    assert audio[_Q * 2 - 2].item() == f0[_Q - 1]
    assert audio[_Q * 2 - 1].item() == f1[_Q - 1]
    assert payload.meta.finished.item() is True
    assert tm.code_prompt_token_ids[rid] == []


def test_async_chunk_appends_frame_from_pooling_output() -> None:
    tm = _tm(chunk_frames=1, left_context=0)
    rid = "req-1"
    frame = torch.tensor(_FRAME, dtype=torch.long)
    payload = talker2mimi_async_chunk(
        transfer_manager=tm,
        pooling_output={"codes": {"audio": [frame]}},
        request=_req(rid, finished=True),
        is_finished=True,
    )
    assert payload is not None
    assert len(payload.codes.audio) == _Q
    assert tm.code_prompt_token_ids[rid] == []


def test_full_payload_flushes_accumulated_frames_on_finish() -> None:
    tm = _tm()
    rid = "req-2"
    tm.code_prompt_token_ids[rid] = [_FRAME[:], [x + 1 for x in _FRAME]]
    payload = talker2mimi_full_payload(
        transfer_manager=tm,
        pooling_output=None,
        request=_req(rid, finished=True),
        is_finished=True,
    )
    assert payload is not None
    assert len(payload.codes.audio) == _Q * 2
    assert tm.code_prompt_token_ids[rid] == []


def test_talker2mimi_non_async_codebook_major_flat() -> None:
    frames = torch.tensor(
        [[1] * _Q, [2] * _Q],
        dtype=torch.long,
    )
    talker_output = SimpleNamespace(
        finished=True,
        outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": frames}})],
    )
    prompts = talker2mimi([talker_output])
    assert len(prompts) == 1
    flat = prompts[0]["prompt_token_ids"]
    assert flat[0] == 1
    assert flat[1] == 2
    assert flat[2] == 1
    assert flat[3] == 2


def test_talker2mimi_skips_unfinished_and_zero_frames() -> None:
    zero = torch.zeros(_Q, dtype=torch.long)
    valid = torch.ones(_Q, dtype=torch.long)
    outs = [
        SimpleNamespace(
            finished=False,
            outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": valid}})],
        ),
        SimpleNamespace(
            finished=True,
            outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": zero}})],
        ),
        SimpleNamespace(
            finished=True,
            outputs=[SimpleNamespace(multimodal_output={"codes": {"audio": valid}})],
        ),
    ]
    prompts = talker2mimi(outs)
    assert len(prompts) == 1
    assert len(prompts[0]["prompt_token_ids"]) == _Q
