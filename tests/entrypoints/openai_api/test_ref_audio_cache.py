# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import asyncio
import hashlib
from collections import OrderedDict

import numpy as np
import pytest

from vllm_omni.config.speech_cache import SpeechCacheConfig
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def cache():
    server = object.__new__(OmniOpenAIServingSpeech)
    server._diffusion_mode = False
    server.model_config = None
    server._ref_audio_resolve_cache = OrderedDict()
    server._ref_audio_resolve_cache_bytes = 0
    config = SpeechCacheConfig()
    server._ref_audio_resolve_cache_max_entries = config.resolve_max_entries
    server._ref_audio_resolve_cache_max_bytes = config.resolve_max_bytes
    server._ref_audio_model_artifact_ready = set()
    server._request_ref_audio_artifact_keys = {}
    return server


def _put(cache, key, samples, artifact=None):
    cache._put_resolved_ref_audio(key, samples, 24000, artifact or key)


def test_compact_storage_and_default_capacity(cache):
    assert cache._ref_audio_resolve_cache_max_entries == 2048
    assert cache._ref_audio_resolve_cache_max_bytes == 4 * 1024**3
    _put(cache, "a", [0.0, 0.25, -0.5, 1.0])
    value = cache._ref_audio_resolve_cache["a"]
    assert value[0].dtype == np.float32
    assert value[2] == cache._ref_audio_resolve_cache_bytes == 16


def test_cached_float32_roundtrip_and_mutation_isolation(cache):
    source = "data:audio/wav;base64,already-cached"
    key = hashlib.sha1(source.encode()).hexdigest()
    values = np.random.default_rng(42).standard_normal(48000).astype(np.float32).tolist()
    expected = values.copy()
    _put(cache, key, values)
    values[0] = 99.0
    first, sample_rate, _ = asyncio.run(cache._resolve_ref_audio(source))
    assert first == expected
    first[0] = 100.0
    second, _, _ = asyncio.run(cache._resolve_ref_audio(source))
    assert sample_rate == 24000
    assert second == expected


def test_entry_eviction_invalidates_artifact_readiness(cache):
    cache._ref_audio_resolve_cache_max_entries = 2
    _put(cache, "a", [0.0])
    cache._ref_audio_model_artifact_ready.add(("a", False))
    _put(cache, "b", [0.0])
    _put(cache, "c", [0.0])
    assert list(cache._ref_audio_resolve_cache) == ["b", "c"]
    assert ("a", False) not in cache._ref_audio_model_artifact_ready


def test_byte_budget_evicts_and_rejects_oversized_entries(cache):
    cache._ref_audio_resolve_cache_max_bytes = 16
    _put(cache, "a", [0.0] * 3)
    _put(cache, "b", [0.0] * 3)
    assert list(cache._ref_audio_resolve_cache) == ["b"]
    assert cache._ref_audio_resolve_cache_bytes == 12
    _put(cache, "too-large", [0.0] * 5)
    assert "too-large" not in cache._ref_audio_resolve_cache


def test_replacement_accounts_bytes_and_preserves_referenced_artifact(cache):
    _put(cache, "a", [0.0] * 3, "same")
    _put(cache, "alias", [0.0] * 2, "same")
    cache._ref_audio_model_artifact_ready.add(("same", False))
    _put(cache, "a", [1.0] * 4, "new")
    assert cache._ref_audio_resolve_cache_bytes == 24
    assert ("same", False) in cache._ref_audio_model_artifact_ready


def test_pool_accepts_666_references(cache):
    for index in range(666):
        _put(cache, str(index), [0.0] * 8)
    assert len(cache._ref_audio_resolve_cache) == 666
    assert cache._ref_audio_resolve_cache_bytes == 666 * 8 * 4
