# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract tests for :func:`align`'s batching orchestration.

``align`` is shared by two consumers with different failure expectations:
``serving_speech_stream`` (streaming TTS word timestamps, one call per
sentence, catches ``ForcedAlignerLoadError`` to report once) and
``serving_transcription`` (one call per request). Batching coalesces callers
into a single ``encode``, so the properties that matter are that results get
routed back to the right caller and that one bad job cannot poison its batch.

The GPU half is stubbed; these run on CPU with no weights.
"""

import asyncio

import pytest

from vllm_omni.utils import forced_aligner as fa

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _cfg() -> fa.ForcedAlignerConfig:
    return fa.ForcedAlignerConfig(model="stub-aligner", pooling_task="token_classify")


@pytest.fixture(autouse=True)
def _clean_state():
    """Batching state is module-global and loop-bound; reset around each test."""
    fa._reset_for_tests()
    yield
    fa._reset_for_tests()


@pytest.fixture
def stub_batches(monkeypatch):
    """Replace the sync GPU call, keep the real async orchestration.

    Returns the list of observed batch sizes so tests can assert on how jobs
    were grouped.
    """

    def _apply(*, delay: float = 0.02, fail_texts: frozenset[str] = frozenset()):
        sizes: list[int] = []
        # Non-None so align() takes the already-loaded fast path.
        monkeypatch.setattr(fa, "_llm", object())

        def fake_batch(jobs, config):
            sizes.append(len(jobs))
            # Stand in for GPU time so later arrivals queue behind this batch.
            import time

            time.sleep(delay)
            return [None if j.text in fail_texts else [fa.WordTimestamp(j.text, 0, 100)] for j in jobs]

        monkeypatch.setattr(fa, "_align_batch_sync", fake_batch)
        return sizes

    return _apply


async def test_single_caller_gets_its_own_result(stub_batches):
    stub_batches()
    got = await fa.align(audio=b"\x00\x00", text="hello", sample_rate=16000, config=_cfg())
    assert got == [fa.WordTimestamp("hello", 0, 100)]


async def test_concurrent_callers_each_get_their_own_result(stub_batches):
    """The property batching must not break: no cross-talk between callers.

    Results come back as one list per batch, so an off-by-one in the
    job/result zip would hand caller A caller B's timestamps -- silently, and
    with plausible-looking output.
    """
    stub_batches()
    texts = [f"utterance-{i}" for i in range(24)]
    results = await asyncio.gather(
        *(fa.align(audio=b"\x00\x00", text=t, sample_rate=16000, config=_cfg()) for t in texts)
    )
    assert [r[0].word for r in results] == texts


async def test_batches_actually_form_under_concurrency(stub_batches):
    sizes = stub_batches(delay=0.05)
    await asyncio.gather(
        *(fa.align(audio=b"\x00\x00", text=f"t{i}", sample_rate=16000, config=_cfg()) for i in range(32))
    )
    assert max(sizes) > 1, f"no batching occurred; batch sizes were {sizes}"
    assert sum(sizes) == 32


async def test_enqueue_is_not_blocked_by_an_in_flight_encode(stub_batches):
    """Regression test for the bug that made batching a no-op.

    align() took ``_encode_lock`` for its (idempotent) load check while the
    worker holds that same lock for the whole encode. Callers therefore
    blocked *before* reaching the queue, so the queue was empty at every drain
    and batch size was pinned at 1.

    Asserting on batch sizes alone does not catch this: in a burst, callers
    blocked on the lock all release together once the worker frees it and then
    enqueue as a group, which still looks like batching. The invariant that
    actually distinguishes the two is narrower -- enqueue must not depend on
    the encode lock at all.

    Queue depth is the wrong observable -- the worker dequeues greedily and
    then blocks on the lock, so it reads empty either way. Observe the handoff
    instead: reaching the queue at all requires running
    ``_ensure_batch_worker``, which a caller stuck on the lock never gets to.
    """
    stub_batches(delay=0.01)
    assert fa._batch_queue is None

    async with fa._encode_lock:  # stand in for an encode in progress
        pending = asyncio.create_task(fa.align(audio=b"\x00\x00", text="mid-batch", sample_rate=16000, config=_cfg()))
        await asyncio.sleep(0.05)
        assert fa._batch_queue is not None, "caller never reached the queue while _encode_lock was held"

    assert await pending == [fa.WordTimestamp("mid-batch", 0, 100)]


async def test_idle_caller_is_not_delayed_waiting_for_a_batch(stub_batches):
    """No linger: a lone request must not wait for stragglers.

    A fixed linger window was tried and measurably cost throughput, so the
    worker drains what is queued and never waits. One caller should see only
    the encode time.
    """
    stub_batches(delay=0.02)
    start = asyncio.get_running_loop().time()
    await fa.align(audio=b"\x00\x00", text="solo", sample_rate=16000, config=_cfg())
    assert asyncio.get_running_loop().time() - start < 0.5


async def test_one_failing_job_does_not_poison_its_batch(stub_batches):
    """Per-item failure is None for that item only -- the single-shot contract."""
    stub_batches(delay=0.05, fail_texts=frozenset({"bad"}))
    texts = ["ok-1", "bad", "ok-2"]
    results = await asyncio.gather(
        *(fa.align(audio=b"\x00\x00", text=t, sample_rate=16000, config=_cfg()) for t in texts)
    )
    assert results[0] == [fa.WordTimestamp("ok-1", 0, 100)]
    assert results[1] is None
    assert results[2] == [fa.WordTimestamp("ok-2", 0, 100)]


async def test_batch_wide_failure_resolves_every_caller_as_none(monkeypatch):
    """A crash inside the batch must not leave callers awaiting forever.

    TTS awaits align() inside a WebSocket turn; an unresolved future would
    hang the session rather than degrade to `timestamps: null`.
    """
    monkeypatch.setattr(fa, "_llm", object())

    def boom(jobs, config):
        raise RuntimeError("encode exploded")

    monkeypatch.setattr(fa, "_align_batch_sync", boom)
    results = await asyncio.gather(
        *(fa.align(audio=b"\x00\x00", text=f"t{i}", sample_rate=16000, config=_cfg()) for i in range(4))
    )
    assert results == [None, None, None, None]


async def test_load_failure_raises_rather_than_returning_none(monkeypatch):
    """``serving_speech_stream`` catches ForcedAlignerLoadError specifically to
    report the reason once, instead of every sentence silently losing
    timestamps. Keep it distinguishable from a per-request failure."""

    def bad_load(config):
        raise RuntimeError("no such model")

    monkeypatch.setattr(fa, "_ensure_loaded", bad_load)
    with pytest.raises(fa.ForcedAlignerLoadError):
        await fa.align(audio=b"\x00\x00", text="hello", sample_rate=16000, config=_cfg())


async def test_worker_is_rebuilt_for_a_new_event_loop(stub_batches):
    """The worker is a module global. A task from a previous loop is not
    ``done()``, merely dead, so a stale queue would silently swallow jobs and
    hang every future. Simulate a second loop by dropping the recorded one."""
    stub_batches()
    await fa.align(audio=b"\x00\x00", text="first", sample_rate=16000, config=_cfg())
    first_queue = fa._batch_queue

    fa._batch_loop = None  # pretend the worker belongs to a different loop
    got = await fa.align(audio=b"\x00\x00", text="second", sample_rate=16000, config=_cfg())

    assert got == [fa.WordTimestamp("second", 0, 100)]
    assert fa._batch_queue is not first_queue


async def test_batch_size_is_capped(stub_batches, monkeypatch):
    """A burst must not build an unbounded batch: memory scales with it, and
    the head of the queue would wait on everything behind it."""
    monkeypatch.setattr(fa, "_ALIGNER_MAX_BATCH", 4)
    sizes = stub_batches(delay=0.05)
    await asyncio.gather(
        *(fa.align(audio=b"\x00\x00", text=f"t{i}", sample_rate=16000, config=_cfg()) for i in range(20))
    )
    assert max(sizes) <= 4
