# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.model_executor.models.lfm3_omni.code2wav import Lfm3OmniCode2Wav

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass
class DecodeOutput:
    audio_values: torch.Tensor


def make_model():
    model = Lfm3OmniCode2Wav.__new__(Lfm3OmniCode2Wav)
    torch.nn.Module.__init__(model)
    model.model_config = Mock(async_chunk=True)
    model.codec = Mock()
    model.codec.from_quantizer.weight = torch.empty(1)
    model.codec.config.n_q = 12
    model.codec.decode.return_value = DecodeOutput(torch.zeros(1, 16))
    model._cache_class = Mock(side_effect=lambda config: object())
    model._streams = {}
    return model


def chunk(request_id, finished=False, frames=4):
    return {
        "request_id": request_id,
        "codes": {"audio": torch.zeros(12, frames, dtype=torch.long)},
        # The connector consumes meta.finished; the codec must use the
        # explicitly preserved stream marker after transport.
        "meta": {"stream_finished": torch.tensor(finished)},
    }


def test_stream_cache_survives_chunks_and_finalizes_empty_tail():
    model = make_model()
    model(runtime_additional_information=[chunk("a"), chunk("b")])
    a, b = model._streams["a"], model._streams["b"]
    assert a is not b
    model(runtime_additional_information=[chunk("a")])
    assert model.codec.decode.call_args.kwargs["streaming_cache"] is a
    assert not model.codec.decode.call_args.kwargs["finalize"]
    model(runtime_additional_information=[chunk("a", finished=True, frames=0)])
    assert model.codec.decode.call_args.kwargs["streaming_cache"] is a
    assert model.codec.decode.call_args.kwargs["finalize"]
    assert set(model._streams) == {"b"}
    model.on_requests_finished(["b", "b"])
    assert not model._streams


def test_codec_rejects_missing_payload_or_request_identity():
    model = make_model()
    with pytest.raises(ValueError, match="codes.audio"):
        model(runtime_additional_information=[{"request_id": "a"}])
    payload = chunk("a")
    del payload["request_id"]
    with pytest.raises(ValueError, match="request id"):
        model(runtime_additional_information=[payload])


def test_transport_coalescing_preserves_reference_decode_chunks():
    model = make_model()
    model(runtime_additional_information=[chunk("a", finished=True, frames=10)])
    calls = model.codec.decode.call_args_list
    assert [call.args[0].shape[-1] for call in calls] == [4, 4, 2]
    assert [call.kwargs["finalize"] for call in calls] == [False, False, True]
    assert len({id(call.kwargs["streaming_cache"]) for call in calls}) == 1
    assert not model._streams
