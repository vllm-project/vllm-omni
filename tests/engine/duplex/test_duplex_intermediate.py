# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.engine.duplex.intermediate import (
    get_stream_request_key,
    get_tts_handoff,
    pack_tts_hidden,
    set_tts_handoff,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_get_stream_request_key_requires_stable_identifier():
    with pytest.raises(ValueError, match="stable request id"):
        get_stream_request_key({"ids": {"tts": [1, 2, 3]}})


def test_get_stream_request_key_accepts_global_request_id():
    assert get_stream_request_key({"global_request_id": ["duplex-sid-stage0"]}) == "duplex-sid-stage0"


def test_packed_tts_hidden_survives_the_engine_request_wire():
    # bfloat16 rows widen to the float32 values the list transport produced.
    hidden = torch.randn(3, 8).to(torch.bfloat16)
    buffer: dict[str, object] = {}
    set_tts_handoff(buffer, [7, 8, 9], pack_tts_hidden(hidden))
    request = OmniEngineCoreRequest(
        request_id="r",
        prompt_token_ids=[0, 0, 0, 0],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        model_intermediate_buffer=buffer,
    )

    decoded = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(request))
    token_ids, rows = get_tts_handoff(decoded.model_intermediate_buffer)

    assert token_ids == [7, 8, 9]
    assert rows.dtype == torch.float32
    assert torch.equal(rows, torch.tensor(hidden.tolist(), dtype=torch.float32))


def test_get_tts_handoff_keeps_list_and_tensor_payloads():
    rows = torch.ones(1, 2)
    assert get_tts_handoff({"hidden_states": {"tts": rows}})[1] is rows
    assert get_tts_handoff({"tts_hidden_states": [[1.0, 2.0]]})[1] == [[1.0, 2.0]]


def test_packed_tts_hidden_requires_bytes_and_shape():
    with pytest.raises(ValueError, match="bytes data and a shape"):
        get_tts_handoff({"hidden_states": {"tts": {"packed_float32": [1.0], "shape": [1, 1]}}})
