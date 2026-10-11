# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import base64
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.serial_utils import MsgpackDecoder, MsgpackEncoder

from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import _normalize_reference
from vllm_omni.model_executor.models.minicpmo_4_5.reference_audio import (
    decode_reference_audio,
    encode_reference_audio,
)
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    _native_runtime_ref_audio_payload,
    tts2code2wav_async_chunk,
    tts2code2wav_full_payload,
)
from vllm_omni.worker_v2.model_states.intermediate_buffer import OmniIntermediateBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reference_survives_engine_ipc_buffer_and_codec_bridge() -> None:
    waveform = torch.linspace(-1, 1, 96256)
    payload = encode_reference_audio(waveform)
    request = OmniEngineCoreRequest(
        request_id="ref-request",
        prompt_token_ids=[0],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        model_intermediate_buffer={"codes": {"ref": payload}, "meta": {"ref_audio_sr": 16000}},
    )
    wire = MsgpackDecoder(OmniEngineCoreRequest).decode(MsgpackEncoder().encode(request))
    assert isinstance(wire.model_intermediate_buffer["codes"]["ref"]["data"], bytes)
    buffer = OmniIntermediateBuffer(1)
    buffer.add_request(
        0,
        SimpleNamespace(
            req_id=request.request_id, mm_features=[], model_intermediate_buffer=wire.model_intermediate_buffer
        ),
    )
    stored = buffer.buffers[0]
    torch.testing.assert_close(decode_reference_audio(stored["codes"]["ref"]), waveform, rtol=0, atol=0)
    wire.model_intermediate_buffer = stored
    for processor, kwargs in ((tts2code2wav_full_payload, {}), (tts2code2wav_async_chunk, {"is_finished": True})):
        codec = processor(SimpleNamespace(), torch.tensor([10, 20]), wire, **kwargs)
        torch.testing.assert_close(codec.codes.ref, waveform, rtol=0, atol=0)
    compact, compact_sr = _normalize_reference(payload, 16000)
    legacy, legacy_sr = _normalize_reference(waveform.tolist(), 16000)
    assert compact_sr == legacy_sr
    torch.testing.assert_close(compact, legacy, rtol=0, atol=0)
    buffer.remove_request(0)
    buffer.add_request(0, SimpleNamespace(req_id="replacement", mm_features=[]))
    assert "codes" not in buffer.buffers[0]
    assert request.request_id not in buffer.req_id_to_index


def test_reference_snapshot_owns_input_and_decoded_samples() -> None:
    waveform = torch.arange(12, dtype=torch.float32)[::2]
    expected = waveform.clone()
    payload = encode_reference_audio(waveform)
    waveform.zero_()
    decoded = decode_reference_audio(payload)
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)
    decoded.zero_()
    torch.testing.assert_close(decode_reference_audio(payload), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("ref_audio_data", base64.b64encode(torch.tensor([0.5]).numpy().tobytes()).decode()),
        ("ref_audio_sample_rate_hz", 22050),
        ("session_id", "next-session"),
        ("epoch", 1),
        ("request_id", "next-request"),
    ],
)
def test_context_reference_cache_invalidates_at_reference_and_lifecycle_boundaries(monkeypatch, field, value) -> None:
    from vllm_omni.model_executor.models.minicpmo_4_5.duplex import input as duplex_input

    calls = []
    original = duplex_input.decode_native_ref_audio_from_config

    def decode(config):
        calls.append(config)
        return original(config)

    monkeypatch.setattr(duplex_input, "decode_native_ref_audio_from_config", decode)
    waveform = torch.tensor([0.1, -0.2, 0.3])
    metadata = {
        "session_id": "session",
        "epoch": 0,
        "runtime_config": {
            "ref_audio_data": base64.b64encode(waveform.numpy().tobytes()).decode(),
            "ref_audio_format": "pcm_f32le",
            "ref_audio_sample_rate_hz": 16000,
        },
    }
    context = SimpleNamespace(bridge_states={})
    first = _native_runtime_ref_audio_payload(metadata, context, request_id="request")
    second = _native_runtime_ref_audio_payload(metadata, context, request_id="request")
    assert len(calls) == 1
    assert first == second and first[0] is not second[0]
    assert first[0]["data"] is second[0]["data"]
    first[0]["format"] = "caller-mutated"
    unchanged = _native_runtime_ref_audio_payload(metadata, context, request_id="request")
    assert unchanged[0]["format"] == "pcm_f32le" and len(calls) == 1
    torch.testing.assert_close(decode_reference_audio(unchanged[0]), waveform, rtol=0, atol=0)
    request_id = "request"
    if field == "request_id":
        request_id = value
    elif field in {"session_id", "epoch"}:
        metadata[field] = value
    else:
        runtime_config = metadata["runtime_config"]
        assert isinstance(runtime_config, dict)
        runtime_config[field] = value
    changed = _native_runtime_ref_audio_payload(metadata, context, request_id=request_id)
    assert len(calls) == 2
    expected = torch.tensor([0.5]) if field == "ref_audio_data" else waveform
    torch.testing.assert_close(decode_reference_audio(changed[0]), expected, rtol=0, atol=0)
    assert changed[1] == (22050 if field == "ref_audio_sample_rate_hz" else 16000)


def test_reference_removal_clears_context_cache_and_other_context_is_independent() -> None:
    metadata = {"runtime_config": {"ref_audio_data": base64.b64encode(b"\x00" * 4).decode()}}
    context = SimpleNamespace(bridge_states={})
    other = SimpleNamespace(bridge_states={})
    assert _native_runtime_ref_audio_payload(metadata, context, request_id="r") is not None
    assert other.bridge_states == {}
    other_metadata = {
        "runtime_config": {"ref_audio_data": base64.b64encode(torch.tensor([0.5]).numpy().tobytes()).decode()}
    }
    other_payload = _native_runtime_ref_audio_payload(other_metadata, other, request_id="r")
    torch.testing.assert_close(decode_reference_audio(other_payload[0]), torch.tensor([0.5]), rtol=0, atol=0)
    first_payload = _native_runtime_ref_audio_payload(metadata, context, request_id="r")
    torch.testing.assert_close(decode_reference_audio(first_payload[0]), torch.tensor([0.0]), rtol=0, atol=0)
    metadata["runtime_config"].pop("ref_audio_data")
    assert _native_runtime_ref_audio_payload(metadata, context, request_id="r") is None
    assert context.bridge_states == {}
    assert "minicpmo45_reference_audio" in other.bridge_states


@pytest.mark.parametrize(
    "value",
    [{"format": "bad", "data": b""}, {"format": "pcm_f32le", "data": b"x"}, {"format": "pcm_f32le", "data": [0.0]}],
)
def test_malformed_reference_payload_is_rejected(value) -> None:
    with pytest.raises(ValueError):
        decode_reference_audio(value)


def test_legacy_references_and_generic_tensor_lists_remain_supported() -> None:
    expected = torch.tensor([0.25, 0.5])
    torch.testing.assert_close(decode_reference_audio(expected.tolist()), expected, rtol=0, atol=0)
    torch.testing.assert_close(decode_reference_audio(expected), expected, rtol=0, atol=0)
    buffer = OmniIntermediateBuffer(1)
    buffer.add_request(
        0,
        SimpleNamespace(
            req_id="r", mm_features=[], model_intermediate_buffer={"hidden_states": {"layers": [expected, 7]}}
        ),
    )
    layers = buffer.buffers[0]["hidden_states"]["layers"]
    assert isinstance(layers[0], torch.Tensor) and layers[1] == 7
    torch.testing.assert_close(layers[0], expected, rtol=0, atol=0)
