# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Host-side transport of the MiniCPM-o 4.5 stage handoffs.

The bridges must hand the Talker and Code2Wav bit-identical values while
keeping per-handoff work off the orchestrator thread.
"""

import base64
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.v1.request import RequestStatus

from vllm_omni.engine.duplex.intermediate import get_tts_handoff, pack_transport_tensor, unpack_transport_tensor
from vllm_omni.model_executor.models.minicpmo_4_5.duplex import input as duplex_input
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    _extract_native_runtime_ref_audio,
    llm2tts,
    tts2code2wav_async_chunk,
    tts2code2wav_full_payload,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_SPECIAL = {
    "tts_bos_token_id": 9301,
    "tts_eos_token_id": 9302,
    "listen_token_id": 9303,
    "speak_token_id": 9304,
    "chunk_eos_token_id": 9308,
    "chunk_tts_eos_token_id": 9309,
    "turn_eos_token_id": 9310,
}


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.contiguous().view(torch.int32)


def _native_source(output_ids: list[int], request_id: str = "req-1"):
    prompt_ids = [101, 102]
    row_ids = [*prompt_ids, *output_ids]
    latent = torch.randn(len(row_ids), 8)
    completion = SimpleNamespace(
        token_ids=output_ids,
        text="hello",
        multimodal_output={
            "latent": latent,
            "duplex_prompt_token_ids": prompt_ids,
            "latent_input_ids": torch.tensor(row_ids).reshape(-1, 1),
            "latent_positions": torch.arange(len(row_ids)).reshape(-1, 1),
            "meta": dict(_SPECIAL),
        },
    )
    return SimpleNamespace(request_id=request_id, prompt_token_ids=prompt_ids, outputs=[completion]), latent


def _ref_config(samples: np.ndarray, sample_rate: int = 16000) -> dict:
    return {
        "ref_audio_data": base64.b64encode(samples.astype("<f4").tobytes()).decode("ascii"),
        "ref_audio_format": "pcm_f32le",
        "ref_audio_sample_rate_hz": sample_rate,
    }


def _context(runtime_config: dict) -> SimpleNamespace:
    return SimpleNamespace(bridge_states={"duplex": {"epoch": 3, "model_turn_id": 7, "runtime_config": runtime_config}})


@pytest.fixture()
def decode_calls(monkeypatch):
    calls = []
    real_decode = duplex_input.decode_native_ref_audio_from_config

    def counting_decode(session_config):
        calls.append(session_config)
        return real_decode(session_config)

    monkeypatch.setattr(duplex_input, "decode_native_ref_audio_from_config", counting_decode)
    return calls


def test_native_duplex_handoff_is_packed_bit_exact() -> None:
    source, latent = _native_source([9304, 21, 22, 9308])

    info = llm2tts([source], prompt=[{}], _streaming_context=_context({}))[0]["model_intermediate_buffer"]

    packed = info["hidden_states"]["tts"]
    assert isinstance(packed["data"], bytes)
    token_ids, hidden = get_tts_handoff(info)
    assert token_ids == [21, 22]
    assert hidden.dtype == torch.float32
    # Same values the legacy tolist()/as_tensor transport produced.
    assert torch.equal(_bits(hidden), _bits(torch.as_tensor(latent[3:5].tolist(), dtype=torch.float32)))
    assert info["meta"]["next_stage_prompt_len"] == 3


def test_reference_voice_is_decoded_once_per_session_config(decode_calls) -> None:
    samples = np.linspace(-1.0, 1.0, 1600, dtype=np.float32)
    context = _context(_ref_config(samples))

    refs = []
    output_ids: list[int] = []
    for unit in range(3):
        # Each unit extends the session's cumulative Stage-0 output.
        output_ids = [*output_ids, 9304, 21 + unit, 40 + unit, 9308]
        source, _ = _native_source(output_ids)
        info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
        assert info["ids"]["tts"] == [21 + unit, 40 + unit]
        refs.append(unpack_transport_tensor(info["codes"]["ref"]))
        assert info["meta"]["ref_audio_sr"] == 16000

    assert len(decode_calls) == 1
    for ref in refs:
        assert torch.equal(_bits(ref), _bits(torch.from_numpy(samples)))

    # A new voice (or rate) is a new key: decode again, never serve stale audio.
    reversed_samples = samples[::-1].copy()
    context.bridge_states["duplex"]["runtime_config"] = _ref_config(reversed_samples, sample_rate=24000)
    source, _ = _native_source([*output_ids, 9304, 30, 31, 9308])
    info = llm2tts([source], prompt=[{}], _streaming_context=context)[0]["model_intermediate_buffer"]
    assert len(decode_calls) == 2
    assert torch.equal(_bits(unpack_transport_tensor(info["codes"]["ref"])), _bits(torch.from_numpy(reversed_samples)))
    assert info["meta"]["ref_audio_sr"] == 24000


def test_listen_only_unit_skips_reference_voice_decode(decode_calls) -> None:
    samples = np.ones(160, dtype=np.float32)
    source, _ = _native_source([9303])

    assert llm2tts([source], prompt=[{}], _streaming_context=_context(_ref_config(samples))) == []
    assert decode_calls == []


def test_reference_voice_without_bridge_state_decodes_every_call(decode_calls) -> None:
    metadata = {"runtime_config": _ref_config(np.full(160, 0.5, dtype=np.float32))}

    first = _extract_native_runtime_ref_audio(metadata, None)
    second = _extract_native_runtime_ref_audio(metadata, SimpleNamespace(bridge_states=None))

    assert len(decode_calls) == 2
    assert torch.equal(first[0], second[0]) and first[1] == second[1] == 16000


def test_reference_voice_path_is_still_rejected_on_every_call(decode_calls) -> None:
    config = {**_ref_config(np.ones(16, dtype=np.float32)), "ref_audio_path": "/tmp/voice.wav"}
    context = _context(config)

    for _ in range(2):
        with pytest.raises(ValueError, match="ref_audio_path"):
            _extract_native_runtime_ref_audio({"runtime_config": config}, context)
    assert len(decode_calls) == 2


def _manager():
    return SimpleNamespace(
        connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25, "codec_left_context_frames": 3}}),
        code_prompt_token_ids=defaultdict(list),
        request_payload={},
        put_req_chunk=defaultdict(int),
    )


def _talker_request(ref: object):
    request = SimpleNamespace(external_req_id="req", request_id="req", status=RequestStatus.RUNNING)
    request.is_finished = lambda: RequestStatus.is_finished(request.status)
    request.model_intermediate_buffer = {"codes": {"ref": ref}, "meta": {"ref_audio_sr": 16000}}
    return request


@pytest.mark.parametrize("packed", [True, False])
def test_code2wav_first_chunk_reads_packed_or_list_reference(packed) -> None:
    ref = torch.randn(64)
    request = _talker_request(pack_transport_tensor(ref) if packed else ref.tolist())
    delta = {"codes": {"audio": torch.arange(7, dtype=torch.long).reshape(-1, 1)}, "meta": {}}

    payload = tts2code2wav_async_chunk(_manager(), delta, request, True)

    assert payload is not None
    assert torch.equal(_bits(payload.codes.ref), _bits(ref))
    assert payload.meta.ref_audio_sr == 16000


@pytest.mark.parametrize("packed", [True, False])
def test_code2wav_full_payload_reads_packed_or_list_reference(packed) -> None:
    ref = torch.randn(64)
    request = _talker_request(pack_transport_tensor(ref) if packed else ref.tolist())

    payload = tts2code2wav_full_payload(
        _manager(),
        {"codes.audio": torch.arange(7, dtype=torch.long).reshape(-1, 1)},
        request,
    )

    assert torch.equal(_bits(payload.codes.ref), _bits(ref))
