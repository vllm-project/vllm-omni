# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Qwen3-Omni's model-owned duplex plugin contract."""

from __future__ import annotations

import struct

import pybase64 as base64
import pytest

from vllm_omni.engine.duplex.config import DuplexSessionConfig
from vllm_omni.engine.duplex.plugin import load_duplex_plugin
from vllm_omni.model_executor.models.qwen3_omni.duplex.chat_fallback_plugin import (
    Qwen3OmniChatFallbackPlugin,
)
from vllm_omni.model_executor.models.qwen3_omni.pipeline import (
    QWEN3_OMNI_CHAT_FALLBACK_PIPELINE,
    QWEN3_OMNI_PIPELINE,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _encode_audio(audio: object, sample_rate_hz: int, fmt: str, speed: float | None) -> str | None:
    del audio, sample_rate_hz, fmt, speed
    return "encoded-audio"


def _pcm(*samples: float) -> str:
    return base64.b64encode(struct.pack(f"<{len(samples)}f", *samples)).decode("ascii")


def test_qwen3_plugin_loads_through_the_unified_contract() -> None:
    plugin = load_duplex_plugin(
        "vllm_omni.model_executor.models.qwen3_omni.duplex.chat_fallback_plugin.Qwen3OmniChatFallbackPlugin",
        _encode_audio,
    )

    capabilities = plugin.capabilities(max_sessions=1)
    assert capabilities.supports_chat_fallback is True
    assert capabilities.supports_input_append is False
    assert capabilities.supports_chat_completions is True
    assert capabilities.input_modes == ["turn_commit_only"]
    assert plugin.default_auto_response is True


def test_qwen3_fallback_policy_adds_interruption_note_until_request_starts() -> None:
    plugin = Qwen3OmniChatFallbackPlugin(_encode_audio)
    state = plugin.create_session_state()
    state.last_turn_interrupted = True

    messages = plugin.fallback_policy_messages(state)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "system"
    assert messages[1]["content"]
    plugin.on_fallback_started(state)
    assert state.last_turn_interrupted is False


def test_qwen3_barge_in_marks_the_next_fallback_as_interrupted() -> None:
    plugin = Qwen3OmniChatFallbackPlugin(_encode_audio)
    state = plugin.create_session_state()

    plugin.on_barge_in(state)

    assert state.last_turn_interrupted is True


def test_qwen3_native_data_plane_fails_closed() -> None:
    plugin = Qwen3OmniChatFallbackPlugin(_encode_audio)

    with pytest.raises(RuntimeError, match="chat fallback"):
        list(plugin.data_plane.project({}, context=None))


def test_qwen3_buffer_returns_one_committed_pcm_turn_and_clears() -> None:
    plugin = Qwen3OmniChatFallbackPlugin(_encode_audio)
    buffer = plugin.create_session_state().audio_buffer
    payload = {
        "type": "audio",
        "format": "pcm_f32le",
        "sample_rate_hz": 16_000,
        "is_speech": True,
    }

    assert (
        buffer.prepare_append(
            {**payload, "audio": _pcm(0.1)}, operation_id="one", chunk_period_ms=1000, allow_emit=True
        )
        is None
    )
    assert (
        buffer.prepare_append(
            {**payload, "audio": _pcm(0.2)}, operation_id="two", chunk_period_ms=1000, allow_emit=True
        )
        is None
    )
    reservation = buffer.prepare_commit(operation_id="commit", chunk_period_ms=1000)

    assert reservation.payload is not None
    assert reservation.payload["format"] == "pcm_f32le"
    assert reservation.payload["sample_rate_hz"] == 16_000
    assert base64.b64decode(reservation.payload["audio"]) == struct.pack("<2f", 0.1, 0.2)
    reservation.commit()
    assert not buffer.has_pending()
    assert not buffer.has_reserved()

    buffer.clear()
    assert buffer.pending_byte_count == 0


def test_qwen3_pipeline_uses_the_unified_plugin_binding() -> None:
    assert QWEN3_OMNI_PIPELINE.duplex_plugin == (
        "vllm_omni.model_executor.models.qwen3_omni.duplex.plugin.Qwen3OmniDuplexPlugin"
    )


def test_qwen3_chat_fallback_is_an_explicit_pipeline_variant() -> None:
    assert QWEN3_OMNI_CHAT_FALLBACK_PIPELINE.duplex_plugin == (
        "vllm_omni.model_executor.models.qwen3_omni.duplex.chat_fallback_plugin.Qwen3OmniChatFallbackPlugin"
    )
    assert QWEN3_OMNI_CHAT_FALLBACK_PIPELINE.stages == QWEN3_OMNI_PIPELINE.stages
    assert QWEN3_OMNI_CHAT_FALLBACK_PIPELINE.default_session_mode == "duplex"


def test_qwen3_chat_fallback_deploy_disables_chunking_and_limits_admission() -> None:
    from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config

    deploy = load_deploy_config(_DEPLOY_DIR / "qwen3_omni_chat_fallback.yaml")

    assert deploy.pipeline == QWEN3_OMNI_CHAT_FALLBACK_PIPELINE.model_type
    assert deploy.session_mode == "duplex"
    assert deploy.async_chunk is False
    assert deploy.duplex_session.max_sessions == 1
    assert len(deploy.stages) == 3


def test_qwen3_streaming_overlay_keeps_fallback_pipeline_and_admission() -> None:
    from vllm_omni.config.stage_config import _DEPLOY_DIR, load_deploy_config

    deploy = load_deploy_config(_DEPLOY_DIR / "qwen3_omni_chat_fallback_streaming.yaml")

    assert deploy.pipeline == QWEN3_OMNI_CHAT_FALLBACK_PIPELINE.model_type
    assert deploy.session_mode == "duplex"
    assert deploy.async_chunk is True
    assert deploy.duplex_session.max_sessions == 1
    assert len(deploy.stages) == 3


def test_qwen3_plugin_preserves_session_config_shape() -> None:
    plugin = Qwen3OmniChatFallbackPlugin(_encode_audio)

    plugin.validate_client_extra_body({})
    assert (
        plugin.runtime_config_for_update(
            DuplexSessionConfig(),
            {},
        )
        == {}
    )
