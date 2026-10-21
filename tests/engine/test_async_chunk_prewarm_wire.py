# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Wire format of the async-chunk prewarm payload: the flat prewarm keys must
survive serialize -> msgpack -> deserialize, the float32 reference bit for bit
and the sample rate as a Python int."""

from types import SimpleNamespace

import msgspec
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.data_entry_keys import ASYNC_CHUNK_PREWARM_NS
from vllm_omni.engine import AdditionalInformationPayload, OmniEngineCoreRequest
from vllm_omni.engine.orchestrator import _attach_async_chunk_prewarm_payload, build_engine_core_request_from_tokens
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import code2wav_prewarm_payload

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_REF_KEY = f"{ASYNC_CHUNK_PREWARM_NS}.ref_audio"
_SR_KEY = f"{ASYNC_CHUNK_PREWARM_NS}.ref_audio_sr"
_WIRE_KEYS = {"global_request_id", _REF_KEY, _SR_KEY}


def _reference() -> torch.Tensor:
    # Signed zeros, subnormals, float32 extremes and 1/3 change bits under a lossy path.
    special = [-0.0, 0.0, 1e-45, -1e-45, 1.1754944e-38, 3.4028235e38, -3.4028235e38, 1 / 3]
    body = torch.randn(4096, generator=torch.Generator().manual_seed(1234), dtype=torch.float32)
    return torch.cat([torch.tensor(special, dtype=torch.float32), body])


def _info(reference: torch.Tensor) -> dict[str, object]:
    return {"global_request_id": ["req-wire"], _REF_KEY: reference, _SR_KEY: 24000}


def _msgpack_round_trip(info: dict) -> object:
    wire = msgspec.msgpack.encode(serialize_additional_information(info))
    return msgspec.msgpack.decode(wire, type=AdditionalInformationPayload)


def _engine_core_round_trip(info: dict) -> object:
    request = build_engine_core_request_from_tokens(
        request_id="req-wire",
        prompt={"prompt_token_ids": [0, 0, 0], "additional_information": info},
        params=SamplingParams(max_tokens=4),
    )
    received = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(request))
    assert isinstance(received, OmniEngineCoreRequest)
    return received.additional_information


def _assert_received(payload: object, reference: torch.Tensor, sample_rate: int) -> None:
    assert isinstance(payload, AdditionalInformationPayload)
    assert set(payload.entries) == _WIRE_KEYS
    decoded = deserialize_additional_information(payload)
    assert decoded["global_request_id"] == ["req-wire"]
    prewarm = decoded[ASYNC_CHUNK_PREWARM_NS]
    assert set(prewarm) == {"ref_audio", "ref_audio_sr"}
    ref = prewarm["ref_audio"]
    assert ref.dtype == torch.float32 and ref.shape == reference.shape
    # Integer views compare the bits, so -0.0 and 0.0 differ here.
    assert torch.equal(ref.contiguous().view(torch.int32), reference.contiguous().view(torch.int32))
    assert type(prewarm["ref_audio_sr"]) is int and prewarm["ref_audio_sr"] == sample_rate


def test_serialize_keeps_each_value_under_its_flat_key() -> None:
    reference = _reference()
    entries = serialize_additional_information(_info(reference)).entries
    assert set(entries) == _WIRE_KEYS
    assert entries[_REF_KEY].tensor_dtype == "float32"
    assert entries[_REF_KEY].tensor_shape == [reference.numel()]
    assert entries[_REF_KEY].tensor_data == reference.numpy().tobytes()
    assert entries[_SR_KEY].tensor_data is None and entries[_SR_KEY].scalar_data == 24000


@pytest.mark.parametrize("round_trip", [_msgpack_round_trip, _engine_core_round_trip], ids=["msgpack", "engine_core"])
def test_round_trip_is_bit_exact(round_trip) -> None:
    reference = _reference()
    _assert_received(round_trip(_info(reference)), reference, 24000)


def test_attached_code2wav_payload_survives_engine_core_transport() -> None:
    waveform = torch.randn(2, 960, generator=torch.Generator().manual_seed(7), dtype=torch.float64)
    base_input = {"additional_information": {"global_request_id": ["req-wire"]}}
    stage_client = SimpleNamespace(async_chunk_prewarm_payload_func=code2wav_prewarm_payload)
    prompt = {"multi_modal_data": {"audio": (waveform, 16000)}}

    _attach_async_chunk_prewarm_payload(base_input, prompt, stage_client, "req-wire", 2)

    received = _engine_core_round_trip(base_input["additional_information"])
    _assert_received(received, waveform.to(torch.float32).mean(dim=0), 16000)
