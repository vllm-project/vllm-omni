# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio

import numpy as np
import pytest
import torch

from vllm_omni.config.speech_cache import SpeechCacheConfig
from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.tts_adapters.base import SpeechServingContext
from vllm_omni.entrypoints.openai.tts_adapters.cosyvoice3 import CosyVoice3Adapter
from vllm_omni.model_executor.models.cosyvoice3.reference_encoder import (
    REFERENCE_CONDITIONING_KEY,
    CosyReferenceArtifacts,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def server(mocker):
    obj = mocker.Mock(spec=OmniOpenAIServingSpeech)
    obj._tts_stage = mocker.Mock()
    obj._tts_stage.runtime_config.additional_config = {"shared_reference_encoder": True}
    obj.model_config = mocker.Mock()
    obj.model_config.model = "cosy-model"
    obj.model_config.hf_config.to_dict.return_value = {"model_type": "cosyvoice3"}
    obj.speech_cache_config = SpeechCacheConfig()
    obj.api_server_shared_dir = None
    obj.uploaded_speakers = {"alice": {}}
    obj._voice_created_at.return_value = 7
    obj._resolve_ref_audio_array = mocker.AsyncMock(return_value=([0.1, 0.2, 0.3], 16000, "key"))
    return obj


def test_shared_prompt_contains_all_artifacts_and_skips_local_prefetch(server, mocker):
    adapter = CosyVoice3Adapter(SpeechServingContext(server=server))
    artifact = CosyReferenceArtifacts(
        torch.ones(1, 4, 80),
        torch.ones(1, 2, dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.ones(1, 192),
    )
    pool = mocker.Mock()
    pool.get = mocker.AsyncMock(return_value=artifact)
    adapter._start_shared_reference_encoder = mocker.AsyncMock(return_value=pool)
    adapter._prefetch_reference = mocker.AsyncMock(side_effect=AssertionError("local encoding"))
    request = OpenAICreateSpeechRequest(input="target", voice="ALICE", ref_audio="clip.wav", ref_text="reference")
    try:
        prompt = asyncio.run(adapter._build_prompt(request))
        kwargs = prompt["mm_processor_kwargs"]
        assert kwargs["voice_name"] == "alice"
        assert kwargs["voice_created_at"] == 7
        assert set(kwargs[REFERENCE_CONDITIONING_KEY]) == {
            "speech_feat",
            "speech_token",
            "speech_token_len",
            "embedding",
        }
        assert kwargs["prompt_text"].endswith("<|endofprompt|>reference")
        assert prompt["prompt"] == "target"
        assert pool.get.await_args is not None
        wave, rate = pool.get.await_args.args
        np.testing.assert_array_equal(wave, np.array([0.1, 0.2, 0.3], dtype=np.float32))
        assert rate == 16000
        adapter._prefetch_reference.assert_not_awaited()
        inline = asyncio.run(adapter._build_prompt(request, has_inline_ref_audio=True))
        assert "voice_name" not in inline["mm_processor_kwargs"]
        server._resolve_ref_audio_array.return_value = ([0.9, 0.8, 0.7], 16000, "replacement")
        server._voice_created_at.return_value = 8
        replaced = asyncio.run(adapter._build_prompt(request))
        assert replaced["mm_processor_kwargs"]["voice_created_at"] == 8
        assert pool.get.await_args is not None
        np.testing.assert_array_equal(pool.get.await_args.args[0], np.array([0.9, 0.8, 0.7], dtype=np.float32))
    finally:
        adapter.shutdown()


def test_disabled_path_retains_existing_prefetch(server, mocker):
    server._tts_stage.runtime_config.additional_config = {}
    adapter = CosyVoice3Adapter(SpeechServingContext(server=server))
    adapter._prefetch_reference = mocker.AsyncMock(return_value=(np.ones(3, dtype=np.float32), 24000))
    request = OpenAICreateSpeechRequest(input="target", ref_audio="clip.wav", ref_text="reference")
    try:
        prompt = asyncio.run(adapter._build_prompt(request))
        adapter._prefetch_reference.assert_awaited_once()
        assert prompt["mm_processor_kwargs"]["sample_rate"] == 24000
        assert REFERENCE_CONDITIONING_KEY not in prompt["mm_processor_kwargs"]
    finally:
        adapter.shutdown()


def test_shared_startup_failure_propagates_without_local_fallback(server, monkeypatch, mocker):
    from vllm_omni.utils import shared_reference_encoder as shared

    pool = mocker.Mock()
    pool.start = mocker.AsyncMock(side_effect=RuntimeError("encoder load failed"))
    constructor = mocker.Mock(return_value=pool)
    monkeypatch.setattr(shared, "ReferenceEncoderPool", constructor)
    adapter = CosyVoice3Adapter(SpeechServingContext(server=server))
    try:
        with pytest.raises(RuntimeError, match="encoder load failed"):
            asyncio.run(adapter.warmup())
        spec, _ = constructor.call_args.args
        assert spec.workers == 1 and spec.cache_bytes == 0
        assert spec.kwargs["cache_bytes"] == server.speech_cache_config.speaker_max_bytes
        assert adapter._reference_executor is None
    finally:
        adapter.shutdown()
    adapter.shutdown()
    pool.close.assert_called_once()
    with pytest.raises(RuntimeError, match="closed"):
        asyncio.run(adapter._start_shared_reference_encoder())


def test_concurrent_startup_constructs_only_one_pool(server, monkeypatch, mocker):
    from vllm_omni.utils import shared_reference_encoder as shared

    pool = mocker.Mock()

    async def ready():
        await asyncio.sleep(0)

    pool.start = mocker.AsyncMock(side_effect=ready)
    constructor = mocker.Mock(return_value=pool)
    monkeypatch.setattr(shared, "ReferenceEncoderPool", constructor)
    adapter = CosyVoice3Adapter(SpeechServingContext(server=server))

    async def run():
        first, second = await asyncio.gather(
            adapter._start_shared_reference_encoder(), adapter._start_shared_reference_encoder()
        )
        assert first is second is pool

    try:
        asyncio.run(run())
        constructor.assert_called_once()
    finally:
        adapter.shutdown()


def test_long_shared_socket_path_is_rejected_not_silently_duplicated(server):
    server.api_server_shared_dir = "/tmp/" + "x" * 110
    adapter = CosyVoice3Adapter(SpeechServingContext(server=server))
    try:
        with pytest.raises(ValueError, match="too long"):
            asyncio.run(adapter.warmup())
        assert adapter._shared_reference_pool is None
    finally:
        adapter.shutdown()


@pytest.mark.parametrize("setting", ["true", 1, None])
def test_shared_reference_setting_requires_boolean(server, setting):
    server._tts_stage.runtime_config.additional_config["shared_reference_encoder"] = setting
    with pytest.raises(ValueError, match="boolean"):
        CosyVoice3Adapter(SpeechServingContext(server=server))
