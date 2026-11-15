# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the TTS KV prefix-cache salt tiers."""

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
    # Real subclass: MagicMock(spec=...) passes spec checks but not
    # isinstance, which _ensure_cache_salt gates on.
    return _UnsaltedAdapter()


def _request(**overrides):
    fields = {"input": "Hello from the test.", "voice": "vivian"}
    fields.update(overrides)
    return OpenAICreateSpeechRequest(**fields)


def _derived(**overrides) -> str:
    prompt: dict = {}
    _ensure_cache_salt(_ar_adapter(), _request(**overrides), prompt, {})
    return prompt["cache_salt"]


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
    """With no caller salt, an adapter-computed salt is used as-is."""
    adapter = _ar_adapter()
    prompt = {"cache_salt": "adapter-provided"}
    _ensure_cache_salt(adapter, _request(), prompt, {})
    assert prompt["cache_salt"] == "adapter-provided"


def test_client_salt_composes_with_derived_identity() -> None:
    adapter = _ar_adapter()
    prompt: dict = {}
    _ensure_cache_salt(adapter, _request(cache_salt="tenant-7"), prompt, {})
    salt = prompt["cache_salt"]

    assert salt != "tenant-7"
    assert salt != _derived()
    assert salt.endswith(f":{_derived()}")
    # Tenant axis: same voice, different callers do not share.
    other_tenant: dict = {}
    _ensure_cache_salt(adapter, _request(cache_salt="tenant-8"), other_tenant, {})
    assert other_tenant["cache_salt"] != salt
    # Conditioning axis: same caller, different voices do not share either.
    other_voice: dict = {}
    _ensure_cache_salt(adapter, _request(cache_salt="tenant-7", voice="marcus"), other_voice, {})
    assert other_voice["cache_salt"] != salt


def test_explicit_salt_composes_with_client_salt() -> None:
    """An adapter-computed salt is the base identity composed with a caller
    salt, not a value that drops the caller's namespace entirely."""
    adapter = _ar_adapter()
    prompt = {"cache_salt": "adapter-provided"}
    _ensure_cache_salt(adapter, _request(cache_salt="tenant-7"), prompt, {})
    assert prompt["cache_salt"] == "tenant-7:adapter-provided"


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
    built prompt composed with the derived identity, not replace it."""
    server, fake = _make_server()

    await server._prepare_speech_generation(_request(cache_salt="tenant-7"))

    salt = fake.seen_prompt.get("cache_salt")
    assert salt is not None and salt.startswith("tenant-7:")
    assert len(salt) > len("tenant-7:")


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


def _make_adapterless_server():
    """No TTS adapter resolves (empty stage_configs skips the talker guard)."""
    engine = MagicMock()
    engine.errored = False
    engine.default_sampling_params_list = [{}]
    server = OmniOpenAIServingSpeech(
        engine_client=engine,
        models=MagicMock(),
        request_logger=MagicMock(),
    )
    server.model_config = SimpleNamespace(async_chunk=True)
    server._get_tts_adapter = lambda: None  # noqa: E731
    return server, engine


async def test_prepare_speech_generation_without_adapter_forwards_client_salt() -> None:
    """No adapter identity to compose with; forward the caller salt as-is."""
    server, engine = _make_adapterless_server()

    await server._prepare_speech_generation(_request(cache_salt="tenant-7"))

    _, kwargs = engine.generate.call_args
    assert kwargs["prompt"]["cache_salt"] == "tenant-7"


async def test_prepare_speech_generation_without_adapter_skips_salt_when_absent() -> None:
    server, engine = _make_adapterless_server()

    await server._prepare_speech_generation(_request())

    _, kwargs = engine.generate.call_args
    assert "cache_salt" not in kwargs["prompt"]
