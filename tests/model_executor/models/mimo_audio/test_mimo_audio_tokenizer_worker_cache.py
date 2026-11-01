# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for the MiMo-Audio tokenizer-worker cache key.

The worker cache key used to keep only the device *type*, so an explicit
``MIMO_AUDIO_TOKENIZER_DEVICE=cuda:1`` silently loaded the tokenizer on the
process's current CUDA device and shared one cache entry with ``cuda:0``.
The index must survive both the cache key and the worker constructor.

CPU-only and weight-free: the worker class is replaced by a stub so no
tokenizer weights are loaded.
"""

from __future__ import annotations

import functools

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TOK_PATH = "/models/mimo-audio-tokenizer"


@functools.lru_cache(maxsize=1)
def _code2wav_module():
    """Defer the import (pulls vLLM model_executor) until first use."""
    from vllm_omni.model_executor.models.mimo_audio import mimo_audio_code2wav

    return mimo_audio_code2wav


class _StubWorker:
    def __init__(self, device_str: str, config_path: str | None, audio_tokenizer_path: str):
        self.device = device_str
        self.config_path = config_path
        self.audio_tokenizer_path = audio_tokenizer_path


@pytest.fixture
def isolated_worker_cache(monkeypatch):
    """Empty worker cache with the real worker replaced by a stub."""
    mod = _code2wav_module()
    monkeypatch.setattr(mod, "MiMoAudioTokenizerWorker", _StubWorker)
    monkeypatch.setattr(mod, "_TOKENIZER_WORKER_CACHE", {})
    return mod


@pytest.mark.parametrize("device", [torch.device("cuda:1"), "cuda:1"])
def test_cache_key_keeps_cuda_index(device):
    """Both torch.device and string spellings retain the index."""
    mod = _code2wav_module()
    key = mod._normalize_tokenizer_worker_cache_key(device, None, _TOK_PATH)
    assert key[0] == "cuda:1"


def test_cache_key_cpu_unchanged():
    mod = _code2wav_module()
    key = mod._normalize_tokenizer_worker_cache_key(torch.device("cpu"), None, _TOK_PATH)
    assert key[0] == "cpu"


def test_cache_key_bare_cuda_resolves_to_current_device(monkeypatch):
    """A bare "cuda" shares the entry with its equivalent indexed spelling."""
    mod = _code2wav_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 3)

    bare = mod._normalize_tokenizer_worker_cache_key(torch.device("cuda"), None, _TOK_PATH)
    indexed = mod._normalize_tokenizer_worker_cache_key(torch.device("cuda:3"), None, _TOK_PATH)
    assert bare[0] == "cuda:3"
    assert bare == indexed


def test_worker_constructed_on_requested_index(isolated_worker_cache):
    mod = isolated_worker_cache
    worker = mod.get_tokenizer_worker(
        device=torch.device("cuda:1"),
        config_path=None,
        audio_tokenizer_path=_TOK_PATH,
    )
    assert worker.device == "cuda:1"


def test_distinct_cuda_indices_get_distinct_workers(isolated_worker_cache):
    mod = isolated_worker_cache
    first = mod.get_tokenizer_worker(
        device=torch.device("cuda:0"),
        config_path=None,
        audio_tokenizer_path=_TOK_PATH,
    )
    second = mod.get_tokenizer_worker(
        device=torch.device("cuda:1"),
        config_path=None,
        audio_tokenizer_path=_TOK_PATH,
    )
    assert first is not second
    assert (first.device, second.device) == ("cuda:0", "cuda:1")


def test_same_device_reuses_cached_worker(isolated_worker_cache):
    mod = isolated_worker_cache
    first = mod.get_tokenizer_worker(
        device=torch.device("cuda:1"),
        config_path=None,
        audio_tokenizer_path=_TOK_PATH,
    )
    second = mod.get_tokenizer_worker(
        device="cuda:1",
        config_path=None,
        audio_tokenizer_path=_TOK_PATH,
    )
    assert first is second
