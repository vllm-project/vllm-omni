# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the default TTS KV prefix-cache salt."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest
from vllm_omni.entrypoints.openai.serving_speech import (
    OmniOpenAIServingSpeech,
    _ensure_cache_salt,
)
from vllm_omni.entrypoints.openai.tts_adapters.base import (
    ARTTSAdapter,
    PreparedRequest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _ar_adapter():
    return MagicMock(spec=ARTTSAdapter)


def _request(**overrides):
    fields = {"input": "Hello from the test.", "voice": "vivian"}
    fields.update(overrides)
    return OpenAICreateSpeechRequest(**fields)


def test_default_salt_applies_to_unsalted_ar_prompt() -> None:
    adapter = _ar_adapter()
    prompt: dict = {}
    _ensure_cache_salt(adapter, _request(), prompt, {})
    assert isinstance(prompt.get("cache_salt"), str) and prompt["cache_salt"]


def test_default_salt_stable_and_sensitive() -> None:
    adapter = _ar_adapter()
    first: dict = {}
    _ensure_cache_salt(adapter, _request(), first, {})
    repeat: dict = {}
    _ensure_cache_salt(adapter, _request(), repeat, {})
    assert repeat["cache_salt"] == first["cache_salt"]

    other_voice: dict = {}
    _ensure_cache_salt(adapter, _request(voice="marcus"), other_voice, {})
    assert other_voice["cache_salt"] != first["cache_salt"]

    other_text: dict = {}
    _ensure_cache_salt(adapter, _request(input="Something else entirely."), other_text, {})
    assert other_text["cache_salt"] != first["cache_salt"]


def test_explicit_salt_wins() -> None:
    adapter = _ar_adapter()
    prompt = {"cache_salt": "adapter-provided"}
    _ensure_cache_salt(adapter, _request(), prompt, {})
    assert prompt["cache_salt"] == "adapter-provided"


def test_client_salt_beats_derived_default() -> None:
    adapter = _ar_adapter()
    prompt: dict = {}
    _ensure_cache_salt(adapter, _request(cache_salt="tenant-7"), prompt, {})
    assert prompt["cache_salt"] == "tenant-7"


def test_explicit_salt_beats_client_salt() -> None:
    adapter = _ar_adapter()
    prompt = {"cache_salt": "adapter-provided"}
    _ensure_cache_salt(adapter, _request(cache_salt="tenant-7"), prompt, {})
    assert prompt["cache_salt"] == "adapter-provided"


def test_invalid_client_salt_rejected() -> None:
    with pytest.raises(Exception):
        _request(cache_salt="x" * 200)


def test_non_ar_backend_skipped() -> None:
    adapter = object()
    prompt: dict = {}
    _ensure_cache_salt(adapter, _request(), prompt, {})
    assert "cache_salt" not in prompt


class _UnsaltedAdapter(ARTTSAdapter):
    """AR adapter that builds a prompt without a salt, like the unsalted family."""

    def __init__(self) -> None:
        self.seen_prompt: dict = {}

    def validate(self, request):
        return None

    async def build(self, request, sampling_params_list, has_inline_ref_audio):
        self.seen_prompt = {"prompt_token_ids": [1, 2, 3]}
        return PreparedRequest(prompt=self.seen_prompt, tts_params={})

    def apply_sampling_overrides(self, sampling_params_list, request, prompt=None, request_id=None):
        return sampling_params_list


async def test_prepare_speech_generation_salts_unsalted_adapter_prompt() -> None:
    """The serving flow wires the default salt in: prove it on the prompt dict
    the fake adapter built, as observed after _prepare returns."""
    server, fake = _make_server()

    await server._prepare_speech_generation(_request())

    assert isinstance(fake.seen_prompt.get("cache_salt"), str) and fake.seen_prompt["cache_salt"]


async def test_prepare_speech_generation_forwards_client_salt() -> None:
    """Same wiring proof for the client tier: a forwarded salt must reach the
    built prompt unchanged instead of the derived default."""
    server, fake = _make_server()

    await server._prepare_speech_generation(_request(cache_salt="tenant-7"))

    assert fake.seen_prompt.get("cache_salt") == "tenant-7"


def _make_server():
    engine = MagicMock()
    engine.errored = False
    engine.default_sampling_params_list = [{}]
    server = OmniOpenAIServingSpeech(
        engine_client=engine,
        models=MagicMock(),
        request_logger=MagicMock(),
    )
    server.model_config = SimpleNamespace(async_chunk=True)
    server._tts_model_type = "unsalted-test"
    fake = _UnsaltedAdapter()
    server._get_tts_adapter = lambda: fake  # noqa: E731
    return server, fake
