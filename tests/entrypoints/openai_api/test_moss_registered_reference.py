# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext, conditioning_cache_salt
from vllm_omni.entrypoints.openai.tts_adapters.moss_tts import MossTTSAdapter
from vllm_omni.model_executor.models.moss_tts.reference_encoder import MossReferenceEncoder
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def setup(tmp_path):
    server = OmniOpenAIServingSpeech.__new__(OmniOpenAIServingSpeech)
    server.uploaded_speakers_dir = tmp_path
    server.uploaded_speakers = {
        "alice": dict(
            embedding_source="audio", created_at=123, file_path=str(tmp_path / "old.safetensors"), ref_text="Reference."
        )
    }
    server._speaker_cache = SpeakerEmbeddingCache()
    server._ref_audio_data_url_cache = {}
    server._ref_audio_resolve_cache = {}
    server._voice_created_at = Mock(side_effect=[123, 124])
    adapter = MossTTSAdapter(SpeechServingContext(server=server))
    adapter._moss_variant = "local"
    encoder = MossReferenceEncoder(None, variant="local", n_vq=12, sr_target=24000, speaker_cache=server._speaker_cache)
    adapter._get_moss_ref_encoder = lambda: encoder

    class Processor:
        def build_user_message(self, **kwargs):
            return kwargs

        def build_assistant_message(self, **kwargs):
            return kwargs

        def __call__(self, *, conversations, mode):
            message = conversations[0][-1]
            codes = message.get("audio_codes_list", message.get("reference"))[0]
            text = torch.zeros((len(codes), 1), dtype=torch.int64)
            return {"input_ids": [torch.cat([text, codes], dim=1)]}

    adapter._get_moss_processor = lambda: Processor()
    return server, adapter, encoder


@pytest.mark.asyncio
async def test_hot_registered_voice_skips_uri_and_resolve(setup):
    server, adapter, encoder = setup
    codes = torch.arange(36, dtype=torch.int32).reshape(3, 12)
    server._speaker_cache.put(encoder._make_cache_key("alice", 123), {"codes": codes})
    for name in [
        "_get_uploaded_audio_data",
        "_validate_ref_audio_format",
        "_load_registered_reference",
        "_resolve_ref_audio_array",
    ]:
        setattr(server, name, Mock(side_effect=AssertionError(name)))
    request = OpenAICreateSpeechRequest(input="Hello", voice="alice")
    assert adapter.validate(request) is None
    result = await adapter.build(request, [], False)
    assert torch.equal(result.tts_params["codes"]["ref"], codes)
    assert request.ref_audio is None
    assert not server._ref_audio_data_url_cache
    server._voice_created_at.assert_not_called()


@pytest.mark.asyncio
async def test_reupload_during_encoding_keeps_codes_and_salt_generation(setup):
    server, adapter, encoder = setup
    entered, release = asyncio.Event(), asyncio.Event()
    server._load_registered_reference = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))

    async def submit(waveform, sr):
        entered.set()
        await release.wait()
        return torch.full((3, 12), 7, dtype=torch.int64)

    encoder._batcher.submit = submit
    request = OpenAICreateSpeechRequest(input="Hello", voice="alice")
    assert adapter.validate(request) is None
    task = asyncio.create_task(adapter.build(request, [], False))
    await entered.wait()
    server.uploaded_speakers["alice"] = dict(server.uploaded_speakers["alice"], created_at=124, ref_text="New.")
    release.set()
    result = await task
    assert torch.equal(result.tts_params["codes"]["ref"], torch.full((3, 12), 7))
    assert result.tts_params["voice_created_at"] == [123]
    assert request.ref_text == "Reference."
    assert result.prompt["cache_salt"] == conditioning_cache_salt(
        request, result.tts_params, registered_voice=("alice", 123)
    )
    assert server._speaker_cache.get(encoder._make_cache_key("alice", 124)) is None
    server._voice_created_at.assert_not_called()


def test_inline_override_is_not_bound(setup):
    server, adapter, _ = setup
    request = OpenAICreateSpeechRequest(input="Hello", voice="alice", ref_audio="file:///reference.wav")
    assert adapter.validate(request) is None
    assert request._registered_voice_reference is None


def test_private_snapshot_cannot_be_supplied_as_json(setup):
    request = OpenAICreateSpeechRequest.model_validate(
        dict(input="Hello", voice="alice", _registered_voice_reference={"name": "alice", "created_at": 999})
    )
    assert request._registered_voice_reference is None


@pytest.mark.asyncio
@pytest.mark.parametrize("channels", [1, 2])
async def test_loaded_waveform_matches_legacy_data_uri(setup, channels):
    server, adapter, _ = setup
    samples = np.random.default_rng(1).uniform(-1.1, 1.1, (24000, channels)).astype(np.float32).squeeze()
    info = server.uploaded_speakers["alice"]
    save_file(
        {"audio": torch.from_numpy(samples)}, info["file_path"], metadata={"created_at": "123", "sample_rate": "24000"}
    )
    request = OpenAICreateSpeechRequest(input="Hello", voice="alice")
    assert adapter.validate(request) is None
    waveform, sr = server._load_registered_reference(request._registered_voice_reference)
    from vllm.multimodal.media import MediaConnector

    legacy, legacy_sr = await MediaConnector().fetch_audio_async(server._get_uploaded_audio_data("alice"))
    expected, _, _, _ = server._finalize_fetched_ref_audio(legacy, legacy_sr)
    np.testing.assert_array_equal(waveform, expected)
    assert sr == legacy_sr
    # A deleted captured generation must not fall back to the new upload.
    from pathlib import Path

    Path(info["file_path"]).unlink()
    with pytest.raises((FileNotFoundError, OSError)):
        server._load_registered_reference(request._registered_voice_reference)


@pytest.mark.asyncio
async def test_registered_cold_and_hot_salt_match(setup):
    server, adapter, encoder = setup
    server._load_registered_reference = Mock(return_value=(np.zeros(24000, dtype=np.float32), 24000))

    async def submit(waveform, sr):
        return torch.full((3, 12), 7, dtype=torch.int64)

    encoder._batcher.submit = submit
    results = []
    for _ in range(2):
        request = OpenAICreateSpeechRequest(input="Hello", voice="alice", ref_text="Override.")
        assert adapter.validate(request) is None
        assert request.ref_text == "Override."
        results.append(await adapter.build(request, [], False))
    assert results[0].prompt["cache_salt"] == results[1].prompt["cache_salt"]
    server._load_registered_reference.assert_called_once()


@pytest.mark.parametrize("generation", [0, -1])
def test_invalid_registered_generation_is_rejected(setup, generation):
    server, adapter, _ = setup
    server.uploaded_speakers["alice"]["created_at"] = generation
    request = OpenAICreateSpeechRequest(input="Hello", voice="alice")
    assert "no valid generation" in adapter.validate(request)
    assert request._registered_voice_reference is None


@pytest.mark.asyncio
async def test_new_request_uses_reuploaded_generation(setup):
    server, adapter, encoder = setup
    results = []
    for generation in [123, 124]:
        server.uploaded_speakers["alice"]["created_at"] = generation
        codes = torch.full((3, 12), generation, dtype=torch.int32)
        server._speaker_cache.put(encoder._make_cache_key("alice", generation), {"codes": codes})
        request = OpenAICreateSpeechRequest(input="Hello", voice="alice")
        assert adapter.validate(request) is None
        result = await adapter.build(request, [], False)
        assert torch.equal(result.tts_params["codes"]["ref"], codes)
        results.append(result)
    assert results[0].prompt["cache_salt"] != results[1].prompt["cache_salt"]
    server.uploaded_speakers.clear()
    assert "requires 'ref_audio'" in adapter.validate(OpenAICreateSpeechRequest(input="Hello", voice="alice"))


def test_cold_load_rejects_mismatched_generation(setup):
    server, adapter, _ = setup
    info = server.uploaded_speakers["alice"]
    save_file({"audio": torch.zeros(24000)}, info["file_path"], metadata={"created_at": "124", "sample_rate": "24000"})
    request = OpenAICreateSpeechRequest(input="Hello", voice="alice")
    assert adapter.validate(request) is None
    with pytest.raises(ValueError, match="generation changed"):
        server._load_registered_reference(request._registered_voice_reference)
