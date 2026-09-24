# tests/e2e/features/fullduplex/test_qwen3omni_serving_adapter.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Serving runtime adapter tests for Qwen3-Omni duplex."""

import asyncio

import pytest

from vllm_omni.entrypoints.duplex.runtime_adapter import (
    ServingRuntimeConfigError,
    load_serving_runtime_adapter,
)
from vllm_omni.experimental.fullduplex.qwen3omni.serving_adapter import (
    Qwen3OmniServingRuntimeAdapter,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_PATH = "vllm_omni.experimental.fullduplex.qwen3omni.serving_adapter.Qwen3OmniServingRuntimeAdapter"


def _encode(samples, sample_rate, fmt, speed=None):
    return "audio-b64"


def test_adapter_id_and_capabilities():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    assert adapter.adapter_id == "qwen3omni"
    assert adapter.supports_runtime_control is False
    caps = adapter.capabilities(max_sessions=1)
    assert caps.supports_model_native_turn_policy is False
    assert caps.supports_client_commit is True
    assert caps.supports_barge_in is True
    assert caps.supports_realtime_endpoint is True


def test_load_serving_runtime_adapter_validates():
    adapter = load_serving_runtime_adapter(_PATH, _encode)
    assert adapter.adapter_id == "qwen3omni"
    adapter.create_session_state()
    assert "s1" not in adapter.session_states
    state = adapter.session_state("s1")
    assert state is not None
    adapter.remove_session_state("s1")
    assert "s1" not in adapter.session_states


def test_is_enabled_and_private_keys():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    assert adapter.is_enabled({"session_mode": "duplex"}) is True
    assert adapter.is_enabled({"session_mode": "turn"}) is True
    assert adapter.is_enabled(object()) is True
    assert "auto_commit_silence_ms" in adapter.private_runtime_config_keys


def test_adapter_auto_respond_hook():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    state = adapter.session_state("s1")

    assert callable(adapter.auto_respond_on_commit)
    assert adapter.auto_respond_on_commit("s1", state) is True


def test_data_plane_context():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    ctx = adapter.data_plane_context(
        epoch=0,
        turn_id=1,
        active_response_turn_id=None,
        active_response_id=None,
        auto_responds=True,
        response_format="pcm16",
        speed=None,
        modalities=("audio", "text"),
    )
    assert ctx.auto_responds is True


def test_validate_client_extra_body_rejects_private_keys():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    with pytest.raises(ServingRuntimeConfigError):
        adapter.validate_client_extra_body({"auto_commit_silence_ms": 300})
    adapter.validate_client_extra_body({"instructions": "hi"})  # no raise


class _FakeSessionConfig:
    def __init__(self, extra_body=None):
        self.extra_body = extra_body or {}


def test_enabled_and_runtime_config_via_session_config_shape():
    adapter = Qwen3OmniServingRuntimeAdapter(_encode)
    assert adapter.is_enabled(_FakeSessionConfig({"session_mode": "duplex"})) is True
    assert adapter.is_enabled(_FakeSessionConfig({})) is True
    runtime = asyncio.run(
        adapter.prepare_runtime_config(
            _FakeSessionConfig({"auto_commit_silence_ms": 500}),
            model_config=None,
        )
    )
    assert runtime == {"auto_commit_silence_ms": 500}
    merged = adapter.runtime_config_for_update(
        _FakeSessionConfig({"auto_commit_silence_ms": 700}),
        {"auto_commit_silence_ms": 500},
    )
    assert merged == {"auto_commit_silence_ms": 700}
