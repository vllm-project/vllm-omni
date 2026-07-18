"""Tests for uploaded voice cache warmup endpoint."""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import (
    OmniOpenAIServingSpeech,
    SpeakerCacheUnsupportedError,
    SpeakerNotFoundError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class FakeSpeakerCache:
    def __init__(self):
        self.entries: dict[tuple[str, str, int], dict[str, Any]] = {}

    @staticmethod
    def make_cache_key(speaker_name: str, model_type: str, created_at: int = 0) -> tuple[str, str, int]:
        return (model_type, speaker_name.lower(), int(created_at))

    def get(self, key: tuple[str, str, int]) -> dict[str, Any] | None:
        return self.entries.get(key)

    def put(self, key: tuple[str, str, int], artifacts: dict[str, Any]) -> None:
        self.entries[key] = artifacts

    def clear(self, speaker_name: str | None = None) -> int:
        if speaker_name is None:
            removed = len(self.entries)
            self.entries.clear()
            return removed
        before = len(self.entries)
        normalized = speaker_name.lower()
        self.entries = {key: value for key, value in self.entries.items() if key[1] != normalized}
        return before - len(self.entries)


def _make_server(mocker, *, model_stage: str = "qwen3_tts") -> OmniOpenAIServingSpeech:
    engine_client = mocker.MagicMock()
    engine_client.errored = False
    engine_client.tts_max_instructions_length = None
    engine_client.default_sampling_params_list = [{}]
    engine_client.model_config.hf_config.custom_voice_dir = None
    engine_client.model_config.hf_config.talker_config.hidden_size = 1024

    stage = mocker.MagicMock()
    stage.engine_args.model_stage = model_stage
    stage.stage_id = 0
    stage.tts_args = {}
    engine_client.stage_configs = [stage]
    engine_client.collective_rpc = AsyncMock()

    models = mocker.MagicMock()
    models.is_base_model.return_value = True
    server = OmniOpenAIServingSpeech(
        engine_client=engine_client,
        models=models,
        request_logger=mocker.MagicMock(),
    )
    server._speaker_cache = FakeSpeakerCache()
    return server


def _audio_speaker_info(*, ref_text: str | None = None) -> dict[str, Any]:
    info: dict[str, Any] = {
        "name": "voice_a",
        "voice_name_lower": "voice_a",
        "consent": "consent",
        "file_path": "/tmp/voice_a.safetensors",
        "created_at": int(time.time()),
        "mime_type": "audio/wav",
        "sample_rate": 16000,
        "embedding_source": "audio",
    }
    if ref_text is not None:
        info["ref_text"] = ref_text
    return info


def _direct_speaker_info() -> dict[str, Any]:
    return {
        "name": "voice_a",
        "voice_name_lower": "voice_a",
        "consent": "consent",
        "file_path": "/tmp/voice_a.safetensors",
        "created_at": int(time.time()),
        "mime_type": "application/x-safetensors",
        "embedding_source": "direct",
        "embedding_dim": 1024,
    }


class TestVoiceCacheWarmup:
    @pytest.fixture
    def server(self, mocker):
        server = _make_server(mocker)
        yield server
        server.shutdown()

    @pytest.mark.asyncio
    async def test_missing_voice_returns_not_found(self, server):
        with pytest.raises(SpeakerNotFoundError):
            await server.create_voice_cache("missing")

    @pytest.mark.asyncio
    async def test_direct_embedding_voice_is_rejected(self, server):
        server.uploaded_speakers = {"voice_a": _direct_speaker_info()}
        with pytest.raises(SpeakerCacheUnsupportedError):
            await server.create_voice_cache("voice_a")

    @pytest.mark.asyncio
    async def test_non_qwen3_model_is_rejected(self, mocker):
        server = _make_server(mocker, model_stage="audio_generation")
        server.uploaded_speakers = {"voice_a": _audio_speaker_info()}
        with pytest.raises(SpeakerCacheUnsupportedError):
            await server.create_voice_cache("voice_a")

    @pytest.mark.asyncio
    async def test_existing_cache_returns_idempotent_ready(self, server):
        speaker_info = _audio_speaker_info(ref_text="hello")
        server.uploaded_speakers = {"voice_a": speaker_info}
        key = server._qwen3_speaker_cache_key("voice_a", speaker_info)
        server._speaker_cache.put(key, {"ref_code": None, "ref_spk_embedding": object(), "icl_mode": True})

        result = await server.create_voice_cache("voice_a")

        assert result["cache_status"] == "ready"
        assert "already exists" in result["message"]
        server.engine_client.collective_rpc.assert_not_called()

    @pytest.mark.asyncio
    async def test_existing_profile_returns_idempotent_ready(self, server):
        speaker_info = _audio_speaker_info(ref_text="hello")
        server.uploaded_speakers = {"voice_a": speaker_info}
        server.precomputed_speakers = {
            "voice_a": {
                "name": "voice_a",
                "voice_name_lower": "voice_a",
                "model_type": "qwen3_tts",
                "mode": "icl",
                "created_at": speaker_info["created_at"],
            }
        }

        result = await server.create_voice_cache("voice_a")

        assert result["cache_status"] == "ready"
        assert "already exists" in result["message"]
        server.engine_client.collective_rpc.assert_not_called()

    @pytest.mark.asyncio
    async def test_success_warms_shared_speaker_cache(self, server, mocker):
        speaker_info = _audio_speaker_info(ref_text="hello")
        server.uploaded_speakers = {"voice_a": speaker_info}
        mocker.patch.object(server, "_load_uploaded_audio", return_value=(np.zeros(16000, dtype=np.float32), 16000))
        save_profile = mocker.patch.object(
            server,
            "_save_qwen3_uploaded_speaker_profile",
            return_value={
                "name": "voice_a",
                "voice_name_lower": "voice_a",
                "model_type": "qwen3_tts",
                "mode": "icl",
                "created_at": speaker_info["created_at"],
            },
        )
        server.engine_client.collective_rpc.return_value = [
            [
                {
                    "ref_spk_embedding": [0.1] * 1024,
                    "ref_code": [[1, 2], [3, 4]],
                    "x_vector_only_mode": False,
                    "icl_mode": True,
                    "ref_text": "hello",
                }
            ]
        ]

        result = await server.create_voice_cache("voice_a")

        assert result == {"voice": "voice_a", "cache_status": "ready"}
        key = server._qwen3_speaker_cache_key("voice_a", speaker_info)
        cached = server._speaker_cache.get(key)
        assert cached is not None
        assert cached["icl_mode"] is True
        assert cached["ref_code"].shape == (2, 2)
        assert server.precomputed_speakers["voice_a"]["created_at"] == speaker_info["created_at"]
        save_profile.assert_called_once()
        server.engine_client.collective_rpc.assert_awaited_once()

    def test_build_tts_params_uses_warmed_cache_without_ref_audio(self, server, mocker):
        speaker_info = _audio_speaker_info(ref_text="hello")
        server.uploaded_speakers = {"voice_a": speaker_info}
        key = server._qwen3_speaker_cache_key("voice_a", speaker_info)
        server._speaker_cache.put(
            key,
            {
                "ref_code": torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.long),
                "ref_spk_embedding": object(),
                "icl_mode": True,
            },
        )
        get_audio_data = mocker.patch.object(server, "_get_uploaded_audio_data")

        params = server._build_tts_params(
            OpenAICreateSpeechRequest(
                input="test",
                voice="voice_a",
                response_format="wav",
            )
        )

        assert params["task_type"] == ["Base"]
        assert params["speaker"] == ["voice_a"]
        assert params["ref_text"] == ["hello"]
        assert params["x_vector_only_mode"] == [False]
        assert params["ref_code_length"] == [3]
        assert "ref_audio" not in params
        get_audio_data.assert_not_called()

    def test_build_tts_params_uses_persisted_profile_without_api_cache(self, server, mocker):
        speaker_info = _audio_speaker_info(ref_text="hello")
        server.uploaded_speakers = {"voice_a": speaker_info}
        server.precomputed_speakers = {
            "voice_a": {
                "name": "voice_a",
                "voice_name_lower": "voice_a",
                "model_type": "qwen3_tts",
                "mode": "icl",
                "created_at": speaker_info["created_at"],
                "ref_text": "hello",
                "ref_code_length": 4,
            }
        }
        get_audio_data = mocker.patch.object(server, "_get_uploaded_audio_data")

        params = server._build_tts_params(
            OpenAICreateSpeechRequest(
                input="test",
                voice="voice_a",
                response_format="wav",
            )
        )

        assert params["task_type"] == ["Base"]
        assert params["speaker"] == ["voice_a"]
        assert params["ref_text"] == ["hello"]
        assert params["x_vector_only_mode"] == [False]
        assert params["ref_code_length"] == [4]
        assert "ref_audio" not in params
        get_audio_data.assert_not_called()

    def test_build_tts_params_cached_uploaded_voice_no_request_override(self, server):
        speaker_info = _audio_speaker_info(ref_text="hello")
        server.uploaded_speakers = {"voice_a": speaker_info}
        key = server._qwen3_speaker_cache_key("voice_a", speaker_info)
        server._speaker_cache.put(
            key,
            {
                "ref_code": torch.tensor([[1, 2]], dtype=torch.long),
                "ref_spk_embedding": object(),
                "icl_mode": True,
            },
        )

        params = server._build_tts_params(
            OpenAICreateSpeechRequest(
                input="test",
                voice="voice_a",
                response_format="wav",
                ref_text="override",
                x_vector_only_mode=True,
            )
        )

        assert params["ref_text"] == ["hello"]
        assert params["x_vector_only_mode"] == [False]

    def test_build_tts_params_falls_back_to_raw_audio_without_cache(self, server, mocker):
        server.uploaded_speakers = {"voice_a": _audio_speaker_info(ref_text="hello")}
        get_audio_data = mocker.patch.object(
            server, "_get_uploaded_audio_data", return_value="data:audio/wav;base64,AA=="
        )

        params = server._build_tts_params(
            OpenAICreateSpeechRequest(
                input="test",
                voice="voice_a",
                response_format="wav",
            )
        )

        assert params["ref_audio"] == ["data:audio/wav;base64,AA=="]
        get_audio_data.assert_called_once_with("voice_a")
