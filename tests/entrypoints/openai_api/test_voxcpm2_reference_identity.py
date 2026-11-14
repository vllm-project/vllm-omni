# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Explicit VoxCPM2 reference audio must not inherit a named-voice identity."""

import asyncio

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext, TTSCapabilities
from vllm_omni.entrypoints.openai.tts_adapters.voxcpm2 import VoxCPM2Adapter
from vllm_omni.model_executor.models.voxcpm2.voxcpm2_talker import (
    VoxCPM2TalkerForConditionalGeneration,
    _RequestState,
)
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def adapter(mocker):
    server = mocker.Mock(spec=OmniOpenAIServingSpeech)
    server.uploaded_speakers = {"alice": {"embedding_source": "audio"}}
    server._voice_created_at.return_value = 7
    server._load_uploaded_audio.return_value = (np.zeros(24, dtype=np.float32), 16000)
    server._resolve_ref_audio_array = mocker.AsyncMock(
        return_value=(np.ones(24, dtype=np.float32), 16000, "inline-reference")
    )
    engine = mocker.Mock()
    engine.model_config.hf_config = mocker.Mock(
        audio_vae_config={"sample_rate": 16000, "encoder_rates": [2]}, patch_size=2
    )
    result = VoxCPM2Adapter(SpeechServingContext(server=server, engine_client=engine))
    result._tokenizer = mocker.Mock(bos_token_id=1)
    result._tokenizer.encode.return_value = [1, 2, 3]
    result.capabilities = TTSCapabilities(
        precomputed_speakers={"preset": {"mode": "reference", "ref_audio_feat_len": 6}}
    )
    return result


@pytest.mark.parametrize("voice", ["alice", "preset", None])
@pytest.mark.parametrize("ref_text", [None, "Reference transcript"])
def test_inline_reference_excludes_registered_identity(adapter, voice, ref_text):
    request = OpenAICreateSpeechRequest(
        input="Hello", voice=voice, ref_audio="data:audio/wav;base64,BBBB", ref_text=ref_text
    )

    prepared = asyncio.run(adapter.build(request, [], has_inline_ref_audio=True))

    info = prepared.prompt["additional_information"]
    audio_key = "reference_audio" if ref_text is None else "prompt_audio"
    assert audio_key in info
    assert "voice_profile" not in info
    assert "voice_name" not in info
    assert "voice_created_at" not in info
    assert "voice_created_at" not in prepared.tts_params
    adapter.ctx.server._load_uploaded_audio.assert_not_called()
    adapter.ctx.server._voice_created_at.assert_not_called()
    adapter.ctx.server._resolve_ref_audio_array.assert_awaited_once_with(request.ref_audio)


@pytest.mark.parametrize("voice", ["alice", "preset"])
def test_registered_reference_keeps_cache_identity(adapter, voice):
    request = OpenAICreateSpeechRequest(input="Hello", voice=voice)

    prepared = asyncio.run(adapter.build(request, [], has_inline_ref_audio=False))

    info = prepared.prompt["additional_information"]
    assert info["voice_name"] == voice
    assert info["voice_created_at"] == 7
    assert prepared.tts_params["voice_created_at"] == 7
    adapter.ctx.server._resolve_ref_audio_array.assert_not_awaited()
    if voice == "alice":
        adapter.ctx.server._load_uploaded_audio.assert_called_once_with(voice)
        assert "reference_audio" in info
    else:
        adapter.ctx.server._load_uploaded_audio.assert_not_called()
        assert info["voice_profile"] == adapter.capabilities.precomputed_speakers[voice]


@pytest.mark.parametrize("voice", ["alice", "preset"])
@pytest.mark.parametrize("inline", [False, True])
def test_native_preprocess_uses_the_matching_reference_cache(adapter, voice, inline, mocker):
    request = OpenAICreateSpeechRequest(
        input="Hello", voice=voice, ref_audio="data:audio/wav;base64,BBBB" if inline else None
    )
    prepared = asyncio.run(adapter.build(request, [], has_inline_ref_audio=inline))
    model = object.__new__(VoxCPM2TalkerForConditionalGeneration)
    torch.nn.Module.__init__(model)
    state = _RequestState(request_id="reference-identity")
    model.config = mocker.Mock(bos_token_id=1)
    mocker.patch.object(model, "_get_multichar_zh_split", return_value={})
    mocker.patch.object(model, "_get_or_create_state", return_value=state)
    mocker.patch.object(model, "_discard_pending_stop_mask")
    model._speaker_cache = SpeakerEmbeddingCache(max_bytes=1024)
    key = model._speaker_cache.make_cache_key(voice, "voxcpm2", 7)
    model._speaker_cache.put(key, {"ref_audio_feat": torch.tensor([1.0])})
    encode = mocker.patch.object(
        model, "_build_prompt_cache", return_value={"mode": "reference", "ref_audio_feat": torch.tensor([9.0])}
    )

    class BeforeNeuralPrefillError(Exception):
        pass

    mocker.patch.object(model, "_build_prefill_inputs", side_effect=BeforeNeuralPrefillError)
    with pytest.raises(BeforeNeuralPrefillError):
        model.preprocess(
            torch.tensor([2, 3]),
            None,
            request_id="reference-identity",
            additional_information=prepared.prompt["additional_information"],
        )

    assert encode.call_count == int(inline)
    torch.testing.assert_close(state.prompt_cache["ref_audio_feat"], torch.tensor([9.0 if inline else 1.0]))
    torch.testing.assert_close(model._speaker_cache.get(key)["ref_audio_feat"], torch.tensor([1.0]))
