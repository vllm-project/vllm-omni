# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Pre-computed reference codes (``ref_audio_codes``) for Fish Audio Drama 3 (fish_omni)."""

import asyncio
import types

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.tts_adapters.base import conditioning_cache_salt

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _codes(frames=7, seed=0, codebooks=10):
    """Valid codec codes: a 4096-entry semantic codebook, then 1024-entry residual ones."""
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 4096 if row == 0 else 1024, size=frames).tolist() for row in range(codebooks)]


def _request(**kw):
    body = {"input": "Get the trust fund to the bank early.", "ref_text": "We asked over twenty different people."}
    body.update(kw)
    return OpenAICreateSpeechRequest(**body)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 for c in text]


class _Server:
    _tts_executor = None
    uploaded_speakers: dict = {}

    def __init__(self):
        self.resolved = 0

    def _apply_uploaded_speaker(self, request):
        return None

    def _validate_ref_audio_format(self, ref_audio):
        return None

    async def _resolve_ref_audio(self, ref_audio):
        self.resolved += 1
        return [0.0] * 44100, 44100, "audio-key"


def _adapter(max_model_len=4096):
    from vllm_omni.entrypoints.openai.tts_adapters.fish_omni import FishOmniAdapter

    hf_config = types.SimpleNamespace(audio_decoder_config=types.SimpleNamespace(num_codebooks=10, vocab_size=4096))
    engine = types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_config=hf_config, max_model_len=max_model_len, model="m")
    )
    adapter = FishOmniAdapter(types.SimpleNamespace(server=_Server(), engine_client=engine))
    adapter._tokenizer = _Tokenizer()
    return adapter


def _build(adapter, request):
    return asyncio.run(adapter.build(request, [], has_inline_ref_audio=False))


def test_protocol_accepts_a_codes_grid_and_rejects_malformed_ones():
    assert _request(ref_audio_codes=_codes()).ref_audio_codes == _codes()
    malformed: list[object] = [[], [[]], [[1, 2], [3]], [[1.5, 2]], [[True, 2]], [["1", "2"]], [1, 2, 3]]
    for value in malformed:
        with pytest.raises(ValidationError):
            _request(ref_audio_codes=value)
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _request(ref_audio_codes=_codes(), ref_audio="data:audio/wav;base64,AAAA")
    with pytest.raises(ValidationError, match="mutually exclusive"):
        _request(ref_audio_codes=_codes(), speaker_embedding=[0.1, 0.2])
    with pytest.raises(ValidationError, match="ref_text"):
        _request(ref_audio_codes=_codes(), ref_text=" ")


def test_adapter_checks_codes_against_the_codec():
    adapter = _adapter()
    assert adapter.validate(_request(ref_audio_codes=_codes())) is None
    assert "codebooks=10" in adapter.validate(_request(ref_audio_codes=_codes(codebooks=9)))
    # [frames, codebooks] is not guessed back into shape.
    assert "codebooks=10" in adapter.validate(_request(ref_audio_codes=np.array(_codes(frames=10)).T[:7].tolist()))
    semantic = _codes()
    semantic[0][2] = 4095
    assert adapter.validate(_request(ref_audio_codes=semantic)) is None
    semantic[0][2] = 4096
    assert "row 0 (semantic codebook) values must be in [0, 4095]" in adapter.validate(
        _request(ref_audio_codes=semantic)
    )
    # Residual codebooks have 1024 entries, not 4096.
    for value in (1024, 4095):
        residual = _codes()
        residual[3][2] = value
        assert "row 3 (residual codebook) values must be in [0, 1023]" in adapter.validate(
            _request(ref_audio_codes=residual)
        )
    # Negative values and integers past int64 are a client error before anything converts them.
    for value in (-1, 2**31, 2**63, -(2**63) - 1):
        bad = _codes()
        bad[4][1] = value
        assert "row 4 (residual codebook)" in adapter.validate(_request(ref_audio_codes=bad))
    assert "uploaded" in adapter.validate(_request(ref_audio_codes=_codes(), voice="alice"))


def test_codes_request_reserves_the_reference_rows_without_audio():
    codes = _codes(frames=9)
    adapter = _adapter()
    prepared = _build(adapter, _request(ref_audio_codes=codes))
    assert adapter.ctx.server.resolved == 0
    info = prepared.prompt["additional_information"]
    assert torch.equal(info["ref_codes"], torch.tensor(codes).T)
    assert "ref_audio_wav" not in info
    from vllm_omni.model_executor.models.fish_omni.prompt import prompt_length

    expected = prompt_length(_Tokenizer(), info["text"], [info["ref_text"]], [9])
    assert len(prepared.prompt["prompt_token_ids"]) == expected


def test_cache_salt_follows_the_codes():
    adapter = _adapter()
    one = _build(adapter, _request(ref_audio_codes=_codes(seed=1))).prompt["cache_salt"]
    again = _build(adapter, _request(ref_audio_codes=_codes(seed=1))).prompt["cache_salt"]
    other = _build(adapter, _request(ref_audio_codes=_codes(seed=2))).prompt["cache_salt"]
    assert one == again != other
    plain = _request()
    assert conditioning_cache_salt(plain, {}) == conditioning_cache_salt(plain, {"ref_codes_cache_key": None})


def test_prompt_beyond_the_model_limit_is_rejected():
    with pytest.raises(ValueError, match="limit"):
        _build(_adapter(max_model_len=64), _request(ref_audio_codes=_codes(frames=80)))


def test_adapters_without_codes_support_reject_them():
    from vllm_omni.entrypoints.openai.tts_adapters.base import validate_tts_request

    request = _request(ref_audio_codes=_codes())
    plain = types.SimpleNamespace(supports_reference_codes=False, name="qwen3_tts", validate=lambda r: None)
    assert "not supported by qwen3_tts" in validate_tts_request(plain, request)
    assert validate_tts_request(plain, _request()) is None
    assert validate_tts_request(_adapter(), request) is None
