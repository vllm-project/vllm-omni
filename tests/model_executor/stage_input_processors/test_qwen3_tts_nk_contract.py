# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Text-token contract for Qwen3-TTS rho instrumentation (#6496).

Code2Wav computes rho = A_k / n_k from ``request_text_tokens`` in the payload
meta. These tests pin the producer side: the request's precomputed assistant
text ids (``PRECOMPUTED_TEXT_IDS_KEY``) yield their count, and the count rides
the payload meta.

Scope: the producer reports a count only for requests that already carry
precomputed ids; a request carrying only the raw text string reports None and
rho is logged without n_k. The count is the request's *running* text-token
count -- one request ID covers every segment a resumable request is fed, and
the ids handed down for the segment being released cover the text released so
far -- so Code2Wav derives each segment's n_k from the increase between
consecutive segment boundaries. These tests pin the count as reported; the
per-segment derivation is pinned in
``tests/model_executor/models/qwen3_tts/test_qwen3_tts_code2wav.py``.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.engine import AdditionalInformationEntry, AdditionalInformationPayload
from vllm_omni.model_executor.stage_input_processors.qwen3_tts import (
    _precomputed_assistant_text_tokens,
    talker2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _Req:
    def __init__(self, additional_information):
        self.additional_information = additional_information


def test_text_tokens_from_precomputed_ids():
    """The tts_pass_token_ids path exposes text as precomputed ids: count = len."""
    payload = AdditionalInformationPayload(
        entries={"_qwen3_tts_text_ids": AdditionalInformationEntry(list_data=[[1, 2, 3, 4, 5]])}
    )
    assert _precomputed_assistant_text_tokens(_Req(payload)) == 5


def test_text_tokens_none_when_only_raw_text():
    """Without precomputed ids (raw text string), the producer reports None."""
    payload = AdditionalInformationPayload(entries={"text": AdditionalInformationEntry(list_data=["hello world"])})
    assert _precomputed_assistant_text_tokens(_Req(payload)) is None


def test_text_tokens_from_tensor_serialized_ids():
    """The ids may cross the process boundary as raw tensor bytes plus shape."""
    payload = AdditionalInformationPayload(
        entries={
            "_qwen3_tts_text_ids": AdditionalInformationEntry(
                tensor_data=b"\x00" * 40,
                tensor_shape=[5],
                tensor_dtype="int64",
            )
        }
    )
    assert _precomputed_assistant_text_tokens(_Req(payload)) == 5


def test_text_tokens_are_the_running_count_of_the_request():
    """A resumable request reports the count its payload currently covers.

    The producer reads the ids the payload carries, so segment 2 of a request
    whose text has grown to 16 tokens reports 16 even though only 6 of them
    belong to segment 2. Code2Wav, not the producer, subtracts the count its
    previous boundary already accounted for.
    """
    first_segment = _Req(
        AdditionalInformationPayload(
            entries={"_qwen3_tts_text_ids": AdditionalInformationEntry(list_data=[list(range(10))])}
        )
    )
    second_segment = _Req(
        AdditionalInformationPayload(
            entries={"_qwen3_tts_text_ids": AdditionalInformationEntry(list_data=[list(range(16))])}
        )
    )
    assert _precomputed_assistant_text_tokens(first_segment) == 10
    assert _precomputed_assistant_text_tokens(second_segment) == 16


def _tm_with_frames(rid, n_frames, max_num_seqs=8):
    extra = {
        "codec_chunk_frames": 25,
        "codec_left_context_frames": 25,
        "initial_codec_chunk_frames": 0,
    }
    tm = SimpleNamespace(
        code_prompt_token_ids=defaultdict(list),
        scheduler_max_num_seqs=max_num_seqs,
        put_req_chunk=defaultdict(int),
        ramp_chunk_count=defaultdict(int),
        request_payload={},
        connector=SimpleNamespace(config={"extra": extra}),
    )
    frame = [1, 2, 3, 4]
    tm.code_prompt_token_ids[rid] = [frame[:] for _ in range(n_frames)]
    return tm


def test_async_chunk_omits_request_text_tokens_without_precomputed_ids():
    """Without precomputed text ids the producer leaves the field unset (rho n/a)."""
    rid = "nk-rid-2"
    tm = _tm_with_frames(rid, n_frames=2)
    request = SimpleNamespace(
        external_req_id=rid,
        is_finished=lambda: False,
        additional_information=SimpleNamespace(entries={}),
    )
    payload = talker2code2wav_async_chunk(
        transfer_manager=tm,
        multimodal_output={"codes": {"audio": torch.zeros((0,))}},
        request=request,
        is_finished=False,
    )
    assert payload is not None
    assert payload.meta.request_text_tokens is None
