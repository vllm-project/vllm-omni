# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Regression tests for single-flight reference-audio encoding in MOSS-TTS.

Guards the duplicate-encode class from RFC #4676 ("Add single-flight for the
same uncached reference audio so concurrent requests trigger only one real
encoding job"). The speaker cache is written only *after* an encode finishes,
so a burst of voice-clone requests arriving on one cold reference — the common
shape: one speaker, many utterances — used to run N multi-second CPU codec
passes before the first one populated the cache.

The invariants under test are the ones that make single-flight safe to put in
front of a shared cache: exactly one real encode per cold reference, distinct
references never collapsed, failures never cached (the next request retries),
one caller giving up never taking the others down, and every caller getting an
independent tensor rather than a view of the cached one.

CPU-only and weight-free: the MOSS processor and the audio resolver are both
injected, so nothing here touches the codec, the talker, or torchaudio.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


torch = pytest.importorskip("torch")

SR = 24000
N_VQ = 4


@pytest.fixture(autouse=True)
def _clear_inflight():
    """Keep the module-global in-flight table from leaking between tests."""
    from vllm_omni.model_executor.models.moss_tts import reference_encoder

    reference_encoder._INFLIGHT.clear()
    yield
    reference_encoder._INFLIGHT.clear()


class _FakeProcessor:
    """Counts real encodes and returns a code tensor derived from the waveform."""

    def __init__(self):
        self.calls = 0
        self.fail_with: Exception | None = None

    def encode_audios_from_wav(self, wavs, sampling_rate, n_vq):
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        # First waveform sample doubles as a fingerprint, so a test can prove
        # which reference a returned tensor actually came from.
        marker = float(wavs[0].reshape(-1)[0])
        return [torch.full((n_vq, 3), marker)]


class _Gate:
    """An async ``resolve_ref_audio`` the test can hold open and count."""

    def __init__(self):
        self.event = asyncio.Event()
        self.calls = 0
        self.fail_with: Exception | None = None
        self.cache_key: str | None = None
        self.waveform_marker: float | None = None

    async def __call__(self, ref_str: str) -> tuple[list, int, str]:
        self.calls += 1
        await self.event.wait()
        if self.fail_with is not None:
            raise self.fail_with
        # Encode the reference identity into the waveform itself unless a test
        # deliberately models two locators for one resolved content snapshot.
        marker = self.waveform_marker if self.waveform_marker is not None else float(len(ref_str))
        return [marker] * 8, SR, self.cache_key or f"key:{ref_str}"


def _make_cache():
    from vllm_omni.utils.speaker_cache import SpeakerEmbeddingCache

    return SpeakerEmbeddingCache()


async def _encode(ref_str, *, processor, resolve, cache, voice_name=None):
    from vllm_omni.model_executor.models.moss_tts.reference_encoder import encode_reference_codes

    return await encode_reference_codes(
        ref_str,
        processor=processor,
        resolve_ref_audio=resolve,
        speaker_cache=cache,
        variant="local",
        n_vq=N_VQ,
        sr_target=SR,
        voice_name=voice_name,
    )


async def _let_callers_reach_the_join():
    """Yield enough for every gathered caller to park on the shared task."""
    for _ in range(3):
        await asyncio.sleep(0)


async def test_concurrent_cold_requests_run_one_encode():
    """Eight concurrent requests for one cold reference → one codec pass."""
    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()

    pending = asyncio.gather(*[_encode("ref-a", processor=processor, resolve=resolve, cache=cache) for _ in range(8)])
    await _let_callers_reach_the_join()
    resolve.event.set()
    results = await pending

    assert processor.calls == 1, "reference audio was encoded more than once"
    # Main derives anonymous identity from the resolver result, so every
    # caller resolves before joining the one encode keyed by that identity.
    assert resolve.calls == 8
    assert len(results) == 8
    for codes in results:
        assert torch.equal(codes, results[0])


async def test_distinct_references_are_not_collapsed():
    """Single-flight keys on the cache key, so two references stay independent."""
    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()

    pending = asyncio.gather(
        _encode("ref-a", processor=processor, resolve=resolve, cache=cache),
        _encode("ref-bb", processor=processor, resolve=resolve, cache=cache),
    )
    await _let_callers_reach_the_join()
    resolve.event.set()
    first, second = await pending

    assert processor.calls == 2
    # The fake encodes len(ref_str) into every element, proving each caller got
    # codes built from its own reference rather than the other one's.
    assert first.flatten()[0].item() == float(len("ref-a"))
    assert second.flatten()[0].item() == float(len("ref-bb"))


async def test_different_locators_with_same_resolver_key_are_collapsed():
    """Anonymous single-flight identity comes from the resolver, not ref_str."""
    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()
    resolve.cache_key = "same-content-snapshot"
    resolve.waveform_marker = 7.0

    pending = asyncio.gather(
        _encode("first-locator", processor=processor, resolve=resolve, cache=cache),
        _encode("second-locator", processor=processor, resolve=resolve, cache=cache),
    )
    await _let_callers_reach_the_join()
    resolve.event.set()
    first, second = await pending

    assert resolve.calls == 2
    assert processor.calls == 1
    assert torch.equal(first, second)


async def test_warm_cache_skips_the_encode():
    """Once stored, later requests hit the cache and never re-encode."""
    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()
    resolve.event.set()

    first = await _encode("ref-a", processor=processor, resolve=resolve, cache=cache)
    second = await _encode("ref-a", processor=processor, resolve=resolve, cache=cache)

    assert processor.calls == 1
    assert torch.equal(first, second)


async def test_anonymous_resolver_key_change_reencodes():
    """A changed content snapshot must not reuse the prior anonymous entry."""
    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()
    resolve.event.set()

    resolve.cache_key = "snapshot-a"
    resolve.waveform_marker = 1.0
    first = await _encode("same-locator", processor=processor, resolve=resolve, cache=cache)
    resolve.cache_key = "snapshot-b"
    resolve.waveform_marker = 2.0
    second = await _encode("same-locator", processor=processor, resolve=resolve, cache=cache)

    assert processor.calls == 2
    assert first.flatten()[0].item() == 1.0
    assert second.flatten()[0].item() == 2.0


async def test_named_warm_cache_skips_resolve():
    """A named-voice hit keeps main's pre-resolve fast path."""
    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()
    resolve.event.set()

    first = await _encode("ref-a", processor=processor, resolve=resolve, cache=cache, voice_name="alice")
    second = await _encode("ref-a", processor=processor, resolve=resolve, cache=cache, voice_name="alice")

    assert resolve.calls == 1
    assert processor.calls == 1
    assert torch.equal(first, second)


async def test_failure_is_not_cached_and_next_request_retries():
    """A failed encode propagates to every waiter and leaves no poisoned slot."""
    from vllm_omni.model_executor.models.moss_tts import reference_encoder

    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()
    resolve.event.set()
    processor.fail_with = RuntimeError("codec exploded")

    pending = asyncio.gather(
        *[_encode("ref-a", processor=processor, resolve=resolve, cache=cache, voice_name="alice") for _ in range(3)],
        return_exceptions=True,
    )
    await _let_callers_reach_the_join()
    results = await pending

    assert all(isinstance(r, RuntimeError) for r in results), results
    assert not reference_encoder._INFLIGHT, "failed encode left a poisoned in-flight entry"
    assert processor.calls == 1

    # The retry must run a fresh encode rather than inherit the failure.
    processor.fail_with = None
    codes = await _encode("ref-a", processor=processor, resolve=resolve, cache=cache, voice_name="alice")
    assert processor.calls == 2
    assert codes.shape == (N_VQ, 3)


async def test_cancelled_caller_does_not_abort_the_shared_encode():
    """One client disconnecting must not fail the others sharing its encode."""
    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()

    quitter = asyncio.ensure_future(
        _encode("ref-a", processor=processor, resolve=resolve, cache=cache, voice_name="alice")
    )
    stayer = asyncio.ensure_future(
        _encode("ref-a", processor=processor, resolve=resolve, cache=cache, voice_name="alice")
    )
    await _let_callers_reach_the_join()

    quitter.cancel()
    resolve.event.set()
    codes = await stayer

    assert quitter.cancelled()
    assert processor.calls == 1
    assert codes.shape == (N_VQ, 3)


async def test_every_caller_gets_an_independent_tensor():
    """Callers must not share storage with each other or with the cache entry."""
    processor, resolve, cache = _FakeProcessor(), _Gate(), _make_cache()

    pending = asyncio.gather(*[_encode("ref-a", processor=processor, resolve=resolve, cache=cache) for _ in range(3)])
    await _let_callers_reach_the_join()
    resolve.event.set()
    first, second, third = await pending

    first.fill_(-1.0)
    assert not torch.equal(first, second), "callers of one in-flight encode share storage"
    assert torch.equal(second, third)

    # The cache entry itself must survive a caller mutating its own copy.
    fourth = await _encode("ref-a", processor=processor, resolve=resolve, cache=cache)
    assert processor.calls == 1
    assert torch.equal(fourth, second)
