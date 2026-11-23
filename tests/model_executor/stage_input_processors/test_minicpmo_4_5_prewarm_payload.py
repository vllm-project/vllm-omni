# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Tests for the MiniCPM-o 4.5 Code2Wav async-chunk prewarm payload.

Code2Wav keys its reference caches on the waveform bytes and the sample rate, so
the prewarm reference must match, bit for bit, the one chunk 0 carries later
(``llm2tts`` -> ``codes.ref`` list -> ``tts2code2wav_async_chunk``).
"""

from types import SimpleNamespace
from typing import Any

import msgspec
import numpy as np
import pytest
import torch

from vllm_omni.data_entry_keys import ASYNC_CHUNK_PREWARM_NS
from vllm_omni.engine import AdditionalInformationPayload
from vllm_omni.engine.serialization import deserialize_additional_information, serialize_additional_information
from vllm_omni.model_executor.models.minicpmo_4_5.pipeline import MINICPMO45_REFERENCE_AUDIO_KEY as REF_KEY
from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    _CODE2WAV_PREWARM_MAX_REF_SECONDS,
    code2wav_prewarm_payload,
    llm2tts,
    tts2code2wav_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _wave(*shape: int, seed: int = 0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed), dtype=dtype)


def _mm(audio: Any) -> dict[str, Any]:
    return {"multi_modal_data": {"audio": audio}}


_A, _B = _wave(640, seed=1), _wave(320, seed=2)
_NP = np.random.default_rng(3).standard_normal(1200).astype(np.float32)
_STEREO64 = _wave(2, 800, seed=4, dtype=torch.float64)
_STEREO_LAST = _wave(640, 2, seed=5)
_STRIDED = _wave(1280, seed=6)[::2]

# id -> (prompt, expected reference, expected sample rate)
_CASES = {
    "serving-key-numpy-float-sr": ({REF_KEY: (_NP, 24000.0)}, torch.as_tensor(_NP), 24000),
    "mm-wins-over-serving-key": ({**_mm((_A, 16000)), REF_KEY: (_B, 24000)}, _A, 16000),
    "first-mm-audio-item": (_mm([{"array": _A.tolist(), "sampling_rate": 22050}, (_B, 8000)]), _A, 22050),
    "attribute-prompt": (SimpleNamespace(multi_modal_data={"audio": (_A, 16000)}), _A, 16000),
    "stereo-float64-channels-first": (_mm((_STEREO64, 16000)), _STEREO64.float().mean(dim=0), 16000),
    "stereo-channels-last": (_mm((_STEREO_LAST, 16000)), _STEREO_LAST.mean(dim=-1), 16000),
    "strided": (_mm((_STRIDED, 16000)), _STRIDED, 16000),
}
parametrize_cases = pytest.mark.parametrize("case", list(_CASES.values()), ids=list(_CASES))


def _chunk0_reference(prompt: Any) -> tuple[torch.Tensor, int]:
    """Run the real ``llm2tts`` -> ``tts2code2wav_async_chunk`` path and return chunk 0's reference."""
    completion = SimpleNamespace(
        token_ids=[11, 12, 9002],
        text="hello",
        multimodal_output={
            "latent": torch.arange(20, dtype=torch.float32).reshape(5, 4),
            "meta": {"tts_bos_token_id": 9001, "tts_eos_token_id": 9002},
        },
    )
    thinker_output = SimpleNamespace(request_id="req-1", prompt_token_ids=[101, 9001], outputs=[completion])
    info = llm2tts([thinker_output], prompt=[prompt])[0]["model_intermediate_buffer"]
    manager = SimpleNamespace(connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25}}))
    request = SimpleNamespace(external_req_id="req-1", request_id="req-1", model_intermediate_buffer=info)
    delta = {"codes": {"audio": torch.arange(25, dtype=torch.long).reshape(-1, 1)}}
    payload = tts2code2wav_async_chunk(manager, delta, request, False)
    assert payload is not None and payload.meta.chunk_seq == 0
    return payload.codes.ref, payload.meta.ref_audio_sr


@parametrize_cases
def test_payload_is_float32_contiguous_mono_reference(case: tuple[Any, torch.Tensor, int]) -> None:
    prompt, expected, sample_rate = case

    payload = code2wav_prewarm_payload(prompt)

    assert payload is not None and set(payload) == {"ref_audio", "ref_audio_sr"}
    ref = payload["ref_audio"]
    assert ref.dtype == torch.float32 and ref.ndim == 1 and ref.is_contiguous()
    assert torch.equal(ref, expected)
    assert type(payload["ref_audio_sr"]) is int and payload["ref_audio_sr"] == sample_rate


@parametrize_cases
def test_payload_matches_chunk0_reference_before_and_after_transport(case: tuple[Any, torch.Tensor, int]) -> None:
    payload = code2wav_prewarm_payload(case[0])
    chunk0_ref, chunk0_sr = _chunk0_reference(case[0])

    # Ship the payload the way the orchestrator does and decode it on the stage side.
    wire = msgspec.msgpack.encode(
        serialize_additional_information({f"{ASYNC_CHUNK_PREWARM_NS}.{k}": v for k, v in payload.items()})
    )
    received = deserialize_additional_information(msgspec.msgpack.decode(wire, type=AdditionalInformationPayload))
    received = received[ASYNC_CHUNK_PREWARM_NS]

    for ref in (payload["ref_audio"], received["ref_audio"]):
        assert ref.dtype == chunk0_ref.dtype == torch.float32
        assert ref.numpy().tobytes() == chunk0_ref.numpy().tobytes()
    assert type(received["ref_audio_sr"]) is int
    assert payload["ref_audio_sr"] == received["ref_audio_sr"] == chunk0_sr


@pytest.mark.parametrize(
    "prompt",
    [
        pytest.param(None, id="none"),
        pytest.param("plain text prompt", id="text"),
        pytest.param({"prompt_token_ids": [1, 2, 3]}, id="tokens-only"),
        pytest.param({"multi_modal_data": {"image": [object()]}}, id="image-only"),
        pytest.param(_mm([]), id="empty-audio-list"),
        pytest.param(_mm((torch.zeros(0), 16000)), id="empty-waveform"),
        pytest.param(_mm((torch.ones(8), 0)), id="zero-sample-rate"),
        pytest.param(_mm({"array": [0.1, 0.2]}), id="missing-sample-rate"),
        pytest.param(_mm((object(), 16000)), id="unconvertible-samples"),
    ],
)
def test_no_usable_reference_returns_none_without_raising(prompt: Any) -> None:
    assert code2wav_prewarm_payload(prompt) is None


def test_reference_longer_than_cap_is_left_to_chunk0() -> None:
    cap = _CODE2WAV_PREWARM_MAX_REF_SECONDS * 100
    assert code2wav_prewarm_payload(_mm((_wave(cap), 100))) is not None
    assert code2wav_prewarm_payload(_mm((_wave(cap + 1), 100))) is None
