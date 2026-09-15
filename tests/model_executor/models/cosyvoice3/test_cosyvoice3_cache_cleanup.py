# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for CosyVoice3 process-wide cache cleanup.

CosyVoice3 keeps several process-wide caches in the main process (keyed by
``model_dir`` and never evicted). ``clear_process_runtime_caches`` releases them
on engine shutdown to bound memory growth; these tests verify it empties every
cache, is resilient to a failing release, and is idempotent.
"""

from __future__ import annotations

import threading

import pytest

import vllm_omni.model_executor.models.cosyvoice3.cosyvoice3 as c3
import vllm_omni.model_executor.models.cosyvoice3.speaker_embedding_trt as trt
from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache, get_speaker_cache

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _StubCampplus:
    """Stand-in for CampplusTRT that records close() without building an engine."""

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _isolate_caches(fresh_speaker_cache):
    """Snapshot and restore the process-wide caches around each test.

    ``fresh_speaker_cache`` resets the speaker-cache singleton; here we also
    save/restore the cosyvoice3 module globals and the class-level s3 model so a
    test never leaks cache state into the rest of the suite.
    """
    saved_runtime = dict(c3._RUNTIME_COMPONENTS_CACHE)
    saved_campplus = dict(trt._CAMPPLUS_CACHE)
    saved_s3 = c3.CosyVoice3MultiModalProcessor._s3_model

    c3._RUNTIME_COMPONENTS_CACHE.clear()
    trt._CAMPPLUS_CACHE.clear()
    c3.CosyVoice3MultiModalProcessor._s3_model = None
    try:
        yield
    finally:
        c3._RUNTIME_COMPONENTS_CACHE.clear()
        c3._RUNTIME_COMPONENTS_CACHE.update(saved_runtime)
        trt._CAMPPLUS_CACHE.clear()
        trt._CAMPPLUS_CACHE.update(saved_campplus)
        c3.CosyVoice3MultiModalProcessor._s3_model = saved_s3


def _populate_all_caches() -> _StubCampplus:
    """Fill every process-wide cache and return the stub campplus instance."""
    stub = _StubCampplus()
    c3.CosyVoice3MultiModalProcessor._s3_model = ("model", "s3module", "cuda")
    c3._RUNTIME_COMPONENTS_CACHE["/some/model_dir"] = {
        "tokenizer": object(),
        "campplus_trt": stub,
    }
    trt._CAMPPLUS_CACHE[("/some/campplus.onnx", "cuda")] = stub
    get_speaker_cache().put(
        SpeakerEmbeddingCache.make_cache_key("alice", model_type="cosyvoice3"),
        {"emb": object()},
    )
    return stub


class TestClearProcessRuntimeCaches:
    def test_clears_all_four_caches(self):
        stub = _populate_all_caches()
        speaker_key = SpeakerEmbeddingCache.make_cache_key("alice", model_type="cosyvoice3")
        # Sanity: everything is populated before the call.
        assert c3.CosyVoice3MultiModalProcessor._s3_model is not None
        assert c3._RUNTIME_COMPONENTS_CACHE
        assert trt._CAMPPLUS_CACHE
        assert get_speaker_cache().get(speaker_key) is not None

        c3.clear_process_runtime_caches()

        assert c3.CosyVoice3MultiModalProcessor._s3_model is None
        assert c3._RUNTIME_COMPONENTS_CACHE == {}
        assert trt._CAMPPLUS_CACHE == {}
        assert get_speaker_cache().get(speaker_key) is None
        # The campplus engine was released via close().
        assert stub.closed is True

    def test_idempotent_on_empty_caches(self):
        # Caches start empty (autouse fixture). Calling twice must not raise.
        c3.clear_process_runtime_caches()
        c3.clear_process_runtime_caches()
        assert c3.CosyVoice3MultiModalProcessor._s3_model is None
        assert c3._RUNTIME_COMPONENTS_CACHE == {}
        assert trt._CAMPPLUS_CACHE == {}

    def test_speaker_cache_clear_failure_is_swallowed(self, monkeypatch):
        """A failing speaker-cache clear must not abort the shutdown path."""
        _populate_all_caches()

        class _Boom:
            def clear(self, *a, **k):
                raise RuntimeError("speaker cache boom")

        monkeypatch.setattr(c3, "get_speaker_cache", lambda: _Boom())

        # Must not raise, and the caches cleared before the speaker step are
        # still released.
        c3.clear_process_runtime_caches()
        assert c3.CosyVoice3MultiModalProcessor._s3_model is None
        assert c3._RUNTIME_COMPONENTS_CACHE == {}
        assert trt._CAMPPLUS_CACHE == {}

    def test_release_s3_model_classmethod(self):
        c3.CosyVoice3MultiModalProcessor._s3_model = ("m", "s3", "cuda")
        c3.CosyVoice3MultiModalProcessor._release_s3_model()
        assert c3.CosyVoice3MultiModalProcessor._s3_model is None


class TestClearCampplusTrtCache:
    def test_returns_count_and_closes_each(self):
        a, b = _StubCampplus(), _StubCampplus()
        trt._CAMPPLUS_CACHE[("a", "cuda")] = a
        trt._CAMPPLUS_CACHE[("b", "cuda")] = b

        n = trt.clear_campplus_trt_cache()

        assert n == 2
        assert a.closed and b.closed
        assert trt._CAMPPLUS_CACHE == {}

    def test_failing_close_is_swallowed_and_cache_still_cleared(self):
        class _BadClose:
            def close(self):
                raise RuntimeError("close boom")

        good = _StubCampplus()
        trt._CAMPPLUS_CACHE[("bad", "cuda")] = _BadClose()
        trt._CAMPPLUS_CACHE[("good", "cuda")] = good

        # Must not raise; the dict is emptied and the well-behaved engine closed.
        n = trt.clear_campplus_trt_cache()
        assert n == 2
        assert trt._CAMPPLUS_CACHE == {}
        assert good.closed is True

    def test_empty_cache_returns_zero(self):
        assert trt.clear_campplus_trt_cache() == 0


class TestCampplusTrtClose:
    def test_close_drops_context_and_engine(self):
        # Build a bare CampplusTRT without deserializing a real engine.
        inst = trt.CampplusTRT.__new__(trt.CampplusTRT)
        inst._lock = threading.Lock()
        inst.context = object()
        inst.engine = object()

        inst.close()

        assert inst.context is None
        assert inst.engine is None
