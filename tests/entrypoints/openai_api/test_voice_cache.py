"""Tests for voice cache generation endpoint (issue #1760)."""

import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
from pytest_mock import MockerFixture

from vllm_omni.entrypoints.openai.serving_speech import (
    OmniOpenAIServingSpeech,
    SpeakerCacheUnsupportedError,
    SpeakerNotFoundError,
)
from vllm_omni.utils.voice_cache import VoiceCacheManager

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_server(mocker: MockerFixture, *, model_stage: str = "qwen3_tts") -> OmniOpenAIServingSpeech:
    """Helper to create an OmniOpenAIServingSpeech with a mocked Qwen3-TTS stage."""
    mock_engine_client = mocker.MagicMock()
    mock_engine_client.errored = False
    mock_engine_client.tts_max_instructions_length = None

    mock_stage = mocker.MagicMock()
    mock_stage.engine_args.model_stage = model_stage
    mock_stage.stage_id = 0
    mock_stage.tts_args = {}
    mock_engine_client.stage_configs = [mock_stage]

    # Provide collective_rpc as AsyncMock
    mock_engine_client.collective_rpc = AsyncMock()

    mock_models = mocker.MagicMock()
    mock_models.is_base_model.return_value = True

    server = OmniOpenAIServingSpeech(
        engine_client=mock_engine_client,
        models=mock_models,
        request_logger=mocker.MagicMock(),
    )
    return server


def _audio_speaker_info(
    file_path: str = "/tmp/voice_samples/test_consent_123.wav",
    ref_text: str | None = None,
    cache_status: str = "pending",
    **extra: Any,
) -> dict[str, Any]:
    """Build a typical audio-uploaded speaker_info dict."""
    info: dict[str, Any] = {
        "name": "test_voice",
        "consent": "consent_123",
        "file_path": file_path,
        "created_at": int(time.time()),
        "mime_type": "audio/wav",
        "embedding_source": "audio",
        "cache_status": cache_status,
        "cache_file": None,
        "cache_generated_at": None,
    }
    if ref_text is not None:
        info["ref_text"] = ref_text
    info.update(extra)
    return info


def _direct_speaker_info(**extra: Any) -> dict[str, Any]:
    """Build a typical direct-embedding speaker_info dict."""
    info: dict[str, Any] = {
        "name": "emb_voice",
        "consent": "consent_emb",
        "file_path": "/tmp/voice_samples/emb_voice.safetensors",
        "created_at": int(time.time()),
        "mime_type": "application/x-safetensors",
        "embedding_source": "direct",
        "cache_status": "ready",
        "cache_file": "/tmp/voice_samples/emb_voice.safetensors",
        "cache_generated_at": int(time.time()),
        "embedding_dim": 1024,
    }
    info.update(extra)
    return info


def _cached_prompt_payload(
    *,
    ref_code: list[list[int]] | None = None,
    x_vector_only_mode: bool = True,
    icl_mode: bool = False,
    ref_text: str | None = None,
) -> dict[str, Any]:
    return {
        "ref_spk_embedding": [0.1] * 1024,
        "ref_code": ref_code,
        "x_vector_only_mode": x_vector_only_mode,
        "icl_mode": icl_mode,
        "ref_text": ref_text,
    }


# ── Cache generation tests ──


class TestCreateVoiceCache:
    @pytest.fixture
    def server(self, mocker: MockerFixture):
        return _make_server(mocker)

    @pytest.mark.asyncio
    async def test_create_cache_not_found(self, server):
        """Voice entry does not exist -> SpeakerNotFoundError."""
        server.uploaded_speakers = {}
        with pytest.raises(SpeakerNotFoundError, match="not found"):
            await server.create_voice_cache("nonexistent")

    @pytest.mark.asyncio
    async def test_create_cache_direct_rejected(self, server):
        """Direct embedding voice -> SpeakerCacheUnsupportedError."""
        server.uploaded_speakers = {"emb_voice": _direct_speaker_info()}
        with pytest.raises(SpeakerCacheUnsupportedError, match="pre-computed embedding"):
            await server.create_voice_cache("emb_voice")

    @pytest.mark.asyncio
    async def test_create_cache_non_qwen3_rejected(self, mocker: MockerFixture):
        """Non Qwen3-TTS model -> SpeakerCacheUnsupportedError."""
        server = _make_server(mocker, model_stage="audio_generation")
        server.uploaded_speakers = {"v": _audio_speaker_info()}
        with pytest.raises(SpeakerCacheUnsupportedError, match="Qwen3-TTS"):
            await server.create_voice_cache("v")

    @pytest.mark.asyncio
    async def test_create_cache_no_collective_rpc(self, mocker: MockerFixture):
        """Engine without collective_rpc -> SpeakerCacheUnsupportedError."""
        server = _make_server(mocker)
        del server.engine_client.collective_rpc
        server.uploaded_speakers = {"v": _audio_speaker_info()}
        with pytest.raises(SpeakerCacheUnsupportedError, match="multi-stage"):
            await server.create_voice_cache("v")

    @pytest.mark.asyncio
    async def test_create_cache_idempotent(self, server):
        """ready + valid cache -> returns already exists."""
        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=_cached_prompt_payload())
        result = await server.create_voice_cache("v")
        assert result["cache_status"] == "ready"
        assert "already exists" in result["message"]

    @pytest.mark.asyncio
    async def test_create_cache_ready_corrupted_rebuilds(self, server):
        """ready but cache file corrupted -> rebuild (will fail at audio read in this test)."""
        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=None)
        # Will fail at file read since file doesn't exist
        with pytest.raises(ValueError, match="missing from disk"):
            await server.create_voice_cache("v")

    @pytest.mark.asyncio
    async def test_create_cache_processing_active(self, server):
        """processing + not timed out -> returns in progress."""
        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="processing", cache_generated_at=time.time())}
        result = await server.create_voice_cache("v")
        assert result["cache_status"] == "processing"

    @pytest.mark.asyncio
    async def test_create_cache_processing_timeout(self, server):
        """processing + timed out -> allows rebuild (fails at file read)."""
        server.uploaded_speakers = {
            "v": _audio_speaker_info(cache_status="processing", cache_generated_at=time.time() - 600)
        }
        with pytest.raises(ValueError, match="missing from disk"):
            await server.create_voice_cache("v")

    @pytest.mark.asyncio
    async def test_create_cache_force_bypasses_processing(self, server):
        """force=True + processing -> allows rebuild (fails at file read)."""
        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="processing", cache_generated_at=time.time())}
        with pytest.raises(ValueError, match="missing from disk"):
            await server.create_voice_cache("v", force=True)

    @pytest.mark.asyncio
    async def test_create_cache_processing_bad_timestamp(self, server):
        """Invalid cache_generated_at type -> treated as stale, allows rebuild."""
        server.uploaded_speakers = {
            "v": _audio_speaker_info(cache_status="processing", cache_generated_at="not_a_number")
        }
        with pytest.raises(ValueError, match="missing from disk"):
            await server.create_voice_cache("v")

    @pytest.mark.asyncio
    async def test_create_cache_audio_missing(self, server):
        """Voice exists but audio file is gone -> ValueError (not 404)."""
        server.uploaded_speakers = {"v": _audio_speaker_info()}
        with pytest.raises(ValueError, match="missing from disk"):
            await server.create_voice_cache("v")

    @pytest.mark.asyncio
    async def test_create_cache_metadata_no_filepath(self, server):
        """file_path field missing from metadata -> ValueError."""
        info = _audio_speaker_info()
        del info["file_path"]
        server.uploaded_speakers = {"v": info}
        with pytest.raises(ValueError, match="no file_path"):
            await server.create_voice_cache("v")

    @pytest.mark.asyncio
    async def test_create_cache_success_e2e(self, server):
        """Full success path: audio read + RPC + save -> ready."""
        server.uploaded_speakers = {"v": _audio_speaker_info()}
        server._speaker_cache_manager.save_speaker_cache = MagicMock(return_value=True)

        rpc_payload = {
            "ref_spk_embedding": [0.1] * 1024,
            "ref_code": None,
            "x_vector_only_mode": True,
            "icl_mode": False,
            "ref_text": None,
        }

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("soundfile.read", return_value=([0.0] * 100, 16000)),
        ):
            server.engine_client.collective_rpc.return_value = [[rpc_payload]]
            result = await server.create_voice_cache("v")

        assert result["cache_status"] == "ready"
        server._speaker_cache_manager.save_speaker_cache.assert_called_once()


# ── Failure rollback tests ──


class TestCacheFailureRollback:
    @pytest.fixture
    def server(self, mocker: MockerFixture):
        return _make_server(mocker)

    @pytest.mark.asyncio
    async def test_failure_rollback_sf_read(self, server):
        """sf.read failure (pre-save) -> cache_status restored to previous, not failed."""
        server.uploaded_speakers = {"v": _audio_speaker_info()}

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("soundfile.read", side_effect=RuntimeError("corrupt audio")),
        ):
            with pytest.raises(RuntimeError, match="corrupt audio"):
                await server.create_voice_cache("v")

        # Previous status was "pending" -> should be restored, not "failed"
        assert server.uploaded_speakers["v"]["cache_status"] == "pending"

    @pytest.mark.asyncio
    async def test_failure_rollback_ready_force_presave(self, server):
        """force=true on ready voice, pre-save failure -> restores to ready."""
        server.uploaded_speakers = {
            "v": _audio_speaker_info(cache_status="ready"),
        }
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=_cached_prompt_payload())

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("soundfile.read", side_effect=RuntimeError("read fail")),
        ):
            with pytest.raises(RuntimeError, match="read fail"):
                await server.create_voice_cache("v", force=True)

        # Old cache untouched (save never attempted), status restored to "ready"
        assert server.uploaded_speakers["v"]["cache_status"] == "ready"

    @pytest.mark.asyncio
    async def test_failure_rollback_rpc(self, server):
        """RPC failure (pre-save) -> cache_status restored to previous."""
        server.uploaded_speakers = {"v": _audio_speaker_info()}

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("soundfile.read", return_value=([0.0] * 100, 16000)),
        ):
            server.engine_client.collective_rpc.side_effect = RuntimeError("RPC failed")
            with pytest.raises(RuntimeError, match="RPC failed"):
                await server.create_voice_cache("v")

        # RPC failure is pre-save -> restore to "pending"
        assert server.uploaded_speakers["v"]["cache_status"] == "pending"

    @pytest.mark.asyncio
    async def test_failure_rollback_save(self, server):
        """_save_voice_cache failure -> cache_status rolled back to failed."""
        server.uploaded_speakers = {"v": _audio_speaker_info()}

        rpc_payload = {
            "ref_spk_embedding": [0.1] * 1024,
            "ref_code": None,
            "x_vector_only_mode": True,
            "icl_mode": False,
            "ref_text": None,
        }

        with (
            patch("pathlib.Path.is_file", return_value=True),
            patch("soundfile.read", return_value=([0.0] * 100, 16000)),
        ):
            server.engine_client.collective_rpc.return_value = [[rpc_payload]]
            server._speaker_cache_manager.save_speaker_cache = MagicMock(return_value=False)
            with pytest.raises(ValueError, match="Failed to save"):
                await server.create_voice_cache("v")

        assert server.uploaded_speakers["v"]["cache_status"] == "failed"


# ── Audio prompt cache tests ──


class TestAudioPromptCache:
    @pytest.fixture
    def server(self, mocker: MockerFixture):
        return _make_server(mocker)

    def test_load_cached_speaker_prompt_memoizes_in_memory(self, server, tmp_path):
        """Audio cached prompt should hit disk once, then reuse in-memory payload."""
        from safetensors.torch import save_file

        cache_file = tmp_path / "voice.safetensors"
        save_file(
            {
                "__len__": torch.tensor(1, dtype=torch.int64),
                "item_0_ref_spk_embedding": torch.randn(1024),
                "item_0_has_ref_code": torch.tensor(1, dtype=torch.int8),
                "item_0_ref_code": torch.randint(0, 10, (2, 4), dtype=torch.int64),
                "item_0_x_vector_only_mode": torch.tensor(0, dtype=torch.int8),
                "item_0_icl_mode": torch.tensor(1, dtype=torch.int8),
            },
            str(cache_file),
            metadata={"item_0_ref_text": "cached transcript"},
        )

        info = _audio_speaker_info(
            file_path=str(tmp_path / "voice.wav"),
            cache_status="ready",
            cache_file=str(cache_file),
        )
        server.uploaded_speakers = {"v": info}
        server.supported_speakers = {"v"}
        server.uploaded_speakers_dir = tmp_path
        server._speaker_cache_manager = VoiceCacheManager(tmp_path)

        payload1 = server._speaker_cache_manager.load_cached_speaker_prompt("v", info)
        assert payload1 is not None
        assert payload1["ref_text"] == "cached transcript"

        cache_file.unlink()
        payload2 = server._speaker_cache_manager.load_cached_speaker_prompt("v", info)
        assert payload2 == payload1

    def test_save_speaker_cache_uses_atomic_replace_and_populates_memory(self, server, tmp_path):
        """Saving cache should write via temp file + replace and warm the in-memory cache."""
        raw_audio = tmp_path / "voice.wav"
        raw_audio.write_bytes(b"fake-wav")

        server.uploaded_speakers_dir = tmp_path
        server._speaker_cache_manager = VoiceCacheManager(tmp_path)
        speaker_info = _audio_speaker_info(file_path=str(raw_audio))
        payload = _cached_prompt_payload(
            ref_code=[[1, 2], [3, 4]],
            x_vector_only_mode=False,
            icl_mode=True,
            ref_text="hello transcript",
        )

        with patch("vllm_omni.entrypoints.openai.serving_speech.os.replace", wraps=os.replace) as replace_mock:
            ok = server._speaker_cache_manager.save_speaker_cache("v", speaker_info, raw_audio, payload)

        assert ok is True
        assert replace_mock.call_count == 1
        assert Path(speaker_info["cache_file"]).is_file()
        assert server._speaker_cache_manager._speaker_prompt_cache["v"] == payload

    def test_delete_voice_invalidates_speaker_prompt_cache(self, server, tmp_path):
        """Deleting a voice should clear any memoized audio prompt entry."""
        raw_audio = tmp_path / "voice.wav"
        cache_file = tmp_path / "voice.safetensors"
        raw_audio.write_bytes(b"fake-wav")
        cache_file.write_bytes(b"fake-cache")

        info = _audio_speaker_info(
            file_path=str(raw_audio),
            cache_status="ready",
            cache_file=str(cache_file),
        )
        server.uploaded_speakers = {"v": info}
        server.supported_speakers = {"v"}
        server._speaker_cache_manager._speaker_prompt_cache["v"] = _cached_prompt_payload()
        server._speaker_cache_manager._speaker_locks["v"] = asyncio.Lock()

        assert asyncio.run(server.delete_voice("v")) is True
        assert "v" not in server._speaker_cache_manager._speaker_prompt_cache
        assert "v" not in server._speaker_cache_manager._speaker_locks


# ── RPC payload extraction tests ──


class TestExtractRpcPayload:
    def test_empty_results(self):
        with pytest.raises(ValueError, match="Empty RPC response"):
            OmniOpenAIServingSpeech._extract_rpc_payload([])

    def test_stage_error_dict(self):
        with pytest.raises(ValueError, match="Stage RPC failed"):
            OmniOpenAIServingSpeech._extract_rpc_payload([{"supported": False, "error": "bad stage"}])

    def test_todo_dict(self):
        with pytest.raises(ValueError, match="not supported"):
            OmniOpenAIServingSpeech._extract_rpc_payload([{"todo": True, "reason": "not impl"}])

    def test_empty_worker_list(self):
        with pytest.raises(ValueError, match="empty worker"):
            OmniOpenAIServingSpeech._extract_rpc_payload([[]])

    def test_invalid_payload_type(self):
        with pytest.raises(ValueError, match="Invalid RPC payload"):
            OmniOpenAIServingSpeech._extract_rpc_payload([["not_a_dict"]])

    def test_missing_key(self):
        with pytest.raises(ValueError, match="Invalid RPC payload"):
            OmniOpenAIServingSpeech._extract_rpc_payload([[{"some_other_key": 1}]])

    def test_valid_multi_rank(self):
        payload = {"ref_spk_embedding": [0.1], "ref_code": None}
        result = OmniOpenAIServingSpeech._extract_rpc_payload([[payload, payload]])
        assert result == payload

    def test_valid_single_result(self):
        payload = {"ref_spk_embedding": [0.1], "ref_code": None}
        result = OmniOpenAIServingSpeech._extract_rpc_payload([payload])
        assert result == payload


# ── Validation tests ──


class TestValidationWithCache:
    @pytest.fixture
    def server(self, mocker: MockerFixture):
        return _make_server(mocker)

    def test_validate_cached_voice_no_audio_ok(self, server):
        """Uploaded voice with ready cache + missing audio file -> validation passes."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server.supported_speakers = {"v"}
        req = OpenAICreateSpeechRequest(input="Hello", voice="v", task_type="Base")
        # validation should pass (no error about missing audio)
        result = server._validate_qwen_tts_request(req)
        assert result is None

    def test_validate_no_cache_no_audio_fail(self, server):
        """No cache + missing audio -> validation fails with guidance."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="pending")}
        server.supported_speakers = {"v"}
        req = OpenAICreateSpeechRequest(input="Hello", voice="v", task_type="Base")
        result = server._validate_qwen_tts_request(req)
        assert result is not None
        assert "not found on disk" in result

    def test_validate_direct_embedding_no_audio_ok(self, server):
        """Direct embedding voice (always ready) + no audio file -> validation passes."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"emb": _direct_speaker_info()}
        server.supported_speakers = {"emb"}
        req = OpenAICreateSpeechRequest(input="Hello", voice="emb", task_type="Base")
        result = server._validate_qwen_tts_request(req)
        assert result is None


# ── TTS consumption path tests ──


class TestBuildTtsParamsWithCache:
    @pytest.fixture
    def server(self, mocker: MockerFixture):
        return _make_server(mocker)

    def test_tts_uses_cached_prompt(self, server):
        """When cache is ready + valid, params should use voice_clone_prompt, not ref_audio."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready", ref_text="hello transcript")}
        server.supported_speakers = {"v"}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(
            return_value=_cached_prompt_payload(
                ref_code=[[1, 2], [3, 4]],
                x_vector_only_mode=False,
                icl_mode=True,
                ref_text="hello transcript",
            )
        )

        req = OpenAICreateSpeechRequest(input="Hello", voice="v")
        params = server._build_tts_params(req)

        assert "voice_clone_prompt" in params
        assert "ref_audio" not in params
        assert params["task_type"] == ["Base"]
        assert params["x_vector_only_mode"] == [False]
        # Verify all values are plain Python types (no tensors)
        vcp = params["voice_clone_prompt"][0]
        assert isinstance(vcp["ref_spk_embedding"], list)
        assert isinstance(vcp["ref_code"], list)

    def test_tts_cached_no_override_ref_text(self, server):
        """Request-level ref_text should NOT override cached ref_text."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server.supported_speakers = {"v"}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=_cached_prompt_payload())

        req = OpenAICreateSpeechRequest(input="Hello", voice="v", ref_text="should_be_ignored")
        params = server._build_tts_params(req)

        # ref_text from request should NOT appear (uploaded voice resolved)
        assert "ref_text" not in params

    def test_tts_cached_no_override_xvec_mode(self, server):
        """Request-level x_vector_only_mode should NOT override cached mode."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server.supported_speakers = {"v"}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=_cached_prompt_payload())

        req = OpenAICreateSpeechRequest(input="Hello", voice="v", x_vector_only_mode=False)
        params = server._build_tts_params(req)

        # Should use cached value (True), not request value (False)
        assert params["x_vector_only_mode"] == [True]

    def test_tts_fallback_corrupted_cache_with_audio(self, server):
        """Cache corrupted but raw audio exists -> fallback to raw audio."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server.supported_speakers = {"v"}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=None)

        with (
            patch.object(server, "_get_uploaded_audio_data", return_value="data:audio/wav;base64,abc"),
            patch("pathlib.Path.is_file", return_value=True),
        ):
            req = OpenAICreateSpeechRequest(input="Hello", voice="v")
            params = server._build_tts_params(req)
            assert "ref_audio" in params
            assert params["task_type"] == ["Base"]

    def test_tts_fallback_corrupted_cache_no_audio(self, server):
        """Cache corrupted AND raw audio missing -> explicit error."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server.supported_speakers = {"v"}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=None)

        req = OpenAICreateSpeechRequest(input="Hello", voice="v")
        with pytest.raises(ValueError, match="corrupted.*also missing"):
            server._build_tts_params(req)

    def test_tts_no_cache_fallback(self, server):
        """No cache (pending) -> fallback to raw audio."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="pending")}
        server.supported_speakers = {"v"}

        with patch.object(server, "_get_uploaded_audio_data", return_value="data:audio/wav;base64,abc"):
            req = OpenAICreateSpeechRequest(input="Hello", voice="v")
            params = server._build_tts_params(req)
            assert "ref_audio" in params

    def test_tts_xvec_only_no_ref_text(self, server):
        """Audio voice with no ref_text -> xvec-only cache."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server.supported_speakers = {"v"}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=_cached_prompt_payload())

        req = OpenAICreateSpeechRequest(input="Hello", voice="v")
        params = server._build_tts_params(req)

        assert params["x_vector_only_mode"] == [True]
        assert "ref_text" not in params

    def test_tts_direct_embedding_voice(self, server, tmp_path):
        """Direct embedding voice -> uses voice_clone_prompt with embedding."""
        from safetensors.torch import save_file

        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        # Create a real safetensors file in tmp_path
        emb_file = tmp_path / "emb_voice.safetensors"
        save_file({"speaker_embedding": torch.randn(1024)}, str(emb_file))

        info = _direct_speaker_info(cache_file=str(emb_file))
        info["file_path"] = str(emb_file)
        server.uploaded_speakers = {"emb_voice": info}
        server.supported_speakers = {"emb_voice"}
        # Point uploaded_speakers_dir to tmp_path so path validation passes
        server.uploaded_speakers_dir = tmp_path
        server._speaker_cache_manager = VoiceCacheManager(tmp_path)

        req = OpenAICreateSpeechRequest(input="Hello", voice="emb_voice")
        params = server._build_tts_params(req)

        assert "voice_clone_prompt" in params
        assert params["task_type"] == ["Base"]
        assert params["x_vector_only_mode"] == [True]
        vcp = params["voice_clone_prompt"][0]
        assert isinstance(vcp["ref_spk_embedding"], list)
        assert len(vcp["ref_spk_embedding"]) == 1024


# ── Direct-embedding error branch tests ──


class TestDirectEmbeddingErrorBranches:
    @pytest.fixture
    def server(self, mocker: MockerFixture):
        return _make_server(mocker)

    def test_direct_embedding_no_cache_file(self, server):
        """cache_file missing from metadata -> ValueError with guidance."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        info = _direct_speaker_info()
        info["cache_file"] = None
        server.uploaded_speakers = {"emb": info}
        server.supported_speakers = {"emb"}

        req = OpenAICreateSpeechRequest(input="Hello", voice="emb")
        with pytest.raises(ValueError, match="no cache_file"):
            server._build_tts_params(req)

    def test_direct_embedding_empty_cache_file(self, server):
        """cache_file is empty string -> ValueError."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        info = _direct_speaker_info()
        info["cache_file"] = ""
        server.uploaded_speakers = {"emb": info}
        server.supported_speakers = {"emb"}

        req = OpenAICreateSpeechRequest(input="Hello", voice="emb")
        with pytest.raises(ValueError, match="no cache_file"):
            server._build_tts_params(req)

    def test_direct_embedding_path_traversal(self, server):
        """cache_file outside voice_samples_dir -> ValueError."""
        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        info = _direct_speaker_info()
        info["cache_file"] = "/etc/passwd"
        server.uploaded_speakers = {"emb": info}
        server.supported_speakers = {"emb"}

        req = OpenAICreateSpeechRequest(input="Hello", voice="emb")
        with pytest.raises(ValueError, match="Illegal cache path"):
            server._build_tts_params(req)

    def test_direct_embedding_missing_key(self, server, tmp_path):
        """safetensors file has no speaker_embedding key -> ValueError."""
        from safetensors.torch import save_file

        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        emb_file = tmp_path / "bad.safetensors"
        save_file({"wrong_key": torch.randn(1024)}, str(emb_file))

        info = _direct_speaker_info(cache_file=str(emb_file))
        server.uploaded_speakers = {"emb": info}
        server.supported_speakers = {"emb"}
        server.uploaded_speakers_dir = tmp_path
        server._speaker_cache_manager = VoiceCacheManager(tmp_path)

        req = OpenAICreateSpeechRequest(input="Hello", voice="emb")
        with pytest.raises(ValueError, match="no speaker_embedding key"):
            server._build_tts_params(req)

    def test_direct_embedding_wrong_shape(self, server, tmp_path):
        """speaker_embedding is 2D -> ValueError."""
        from safetensors.torch import save_file

        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        emb_file = tmp_path / "bad_shape.safetensors"
        save_file({"speaker_embedding": torch.randn(2, 1024)}, str(emb_file))

        info = _direct_speaker_info(cache_file=str(emb_file))
        server.uploaded_speakers = {"emb": info}
        server.supported_speakers = {"emb"}
        server.uploaded_speakers_dir = tmp_path
        server._speaker_cache_manager = VoiceCacheManager(tmp_path)

        req = OpenAICreateSpeechRequest(input="Hello", voice="emb")
        with pytest.raises(ValueError, match="unexpected shape"):
            server._build_tts_params(req)

    def test_voice_clone_prompt_no_tensors(self, server, tmp_path):
        """Verify voice_clone_prompt values are all plain Python types."""
        from safetensors.torch import save_file

        from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest

        emb_file = tmp_path / "ok.safetensors"
        save_file({"speaker_embedding": torch.randn(1024)}, str(emb_file))

        info = _direct_speaker_info(cache_file=str(emb_file))
        server.uploaded_speakers = {"emb": info}
        server.supported_speakers = {"emb"}
        server.uploaded_speakers_dir = tmp_path
        server._speaker_cache_manager = VoiceCacheManager(tmp_path)

        req = OpenAICreateSpeechRequest(input="Hello", voice="emb")
        params = server._build_tts_params(req)
        vcp = params["voice_clone_prompt"][0]

        # No torch.Tensor anywhere in the dict
        for key, val in vcp.items():
            assert not isinstance(val, torch.Tensor), f"voice_clone_prompt[{key!r}] is a Tensor"


# ── Handler contract tests (via lightweight FastAPI app, same pattern as test_serving_speech.py) ──


class TestVoiceCacheHandlerContract:
    """Test handler HTTP contract: status codes and response shapes."""

    @pytest.fixture
    def app_and_server(self, mocker: MockerFixture):
        """Create a FastAPI app with the cache route wired up."""
        from fastapi import FastAPI, Query
        from fastapi.testclient import TestClient

        server = _make_server(mocker)

        async def cache_route(
            name: str,
            force: bool = Query(False),
        ):
            try:
                result = await server.create_voice_cache(name, force=force)
                return result
            except SpeakerNotFoundError as e:
                from fastapi.responses import JSONResponse

                return JSONResponse(content={"success": False, "error": str(e)}, status_code=404)
            except SpeakerCacheUnsupportedError as e:
                from fastapi.responses import JSONResponse

                return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)
            except ValueError as e:
                from fastapi.responses import JSONResponse

                return JSONResponse(content={"success": False, "error": str(e)}, status_code=400)

        app = FastAPI()
        app.add_api_route("/v1/audio/voices/{name}/cache", cache_route, methods=["POST"])
        client = TestClient(app)
        return client, server

    def test_route_not_found_returns_404(self, app_and_server):
        client, server = app_and_server
        server.uploaded_speakers = {}
        resp = client.post("/v1/audio/voices/nonexistent/cache")
        assert resp.status_code == 404
        assert "not found" in resp.json()["error"]

    def test_route_direct_embedding_returns_400(self, app_and_server):
        client, server = app_and_server
        server.uploaded_speakers = {"emb": _direct_speaker_info()}
        resp = client.post("/v1/audio/voices/emb/cache")
        assert resp.status_code == 400
        assert "pre-computed embedding" in resp.json()["error"]

    def test_route_audio_missing_returns_400(self, app_and_server):
        client, server = app_and_server
        server.uploaded_speakers = {"v": _audio_speaker_info()}
        resp = client.post("/v1/audio/voices/v/cache")
        assert resp.status_code == 400
        assert "missing from disk" in resp.json()["error"]

    def test_route_ready_returns_200(self, app_and_server):
        client, server = app_and_server
        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=_cached_prompt_payload())
        resp = client.post("/v1/audio/voices/v/cache")
        assert resp.status_code == 200
        assert resp.json()["cache_status"] == "ready"

    def test_route_processing_returns_200(self, app_and_server):
        client, server = app_and_server
        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="processing", cache_generated_at=time.time())}
        resp = client.post("/v1/audio/voices/v/cache")
        assert resp.status_code == 200
        assert resp.json()["cache_status"] == "processing"

    def test_route_force_param(self, app_and_server):
        """force=true should bypass ready guard (fails at file read here)."""
        client, server = app_and_server
        server.uploaded_speakers = {"v": _audio_speaker_info(cache_status="ready")}
        server._speaker_cache_manager.load_cached_speaker_prompt = MagicMock(return_value=_cached_prompt_payload())
        resp = client.post("/v1/audio/voices/v/cache?force=true")
        # Should try to rebuild (and fail at file read), returning 400
        assert resp.status_code == 400
        assert "missing from disk" in resp.json()["error"]
