# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Selectable TaoMate-H3 KV retention: selection, ledger order, and cleanup.

These tests drive the real segmented KV pool, so every assertion is about
storage that exists. They cover the retention policy itself (which committed
segments survive), the rule that the model ledger only advances after the
engine transaction succeeded, and teardown returning free blocks before the
next admission.

Viewer departure is not a concept of this repository's video endpoint: a
disconnect aborts the generating request. What is measurable here is that a
release targets exactly one session, which is covered directly.
"""

from __future__ import annotations

import pytest
import torch

from vllm_omni.diffusion.models.taomate_h3.kv_cache import (
    MAX_RETAINED_SEGMENTS,
    RETENTION_POLICIES,
    TaoMateKVHistory,
    TaoMateRetention,
    resolve_retention,
)
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec, ARDiffusionSegmentedKVSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

# Four phases per request, as the pipeline schedules them (34/34/34/17 frames).
NUM_PHASES = 4
_BLOCK = 16
_HEADS = 2
_HEAD_DIM = 4


def _segmented_pool(*, num_layers: int = 2, capacity: int = 80) -> ARDiffusionKVCache:
    """A real segmented pool: block math, allocation and release all run."""
    return ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=_BLOCK, window_chunks=1, gpu_memory_fraction=0.01),
        num_layers=num_layers,
        num_kv_heads=_HEADS,
        head_size=_HEAD_DIM,
        dtype=torch.bfloat16,
        block_size=_BLOCK,
        max_model_len=100000,
        available_bytes=128 * 1024**2,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        device=torch.device("cpu"),
        segmented=ARDiffusionSegmentedKVSpec(
            block_size=_BLOCK,
            max_retained_tokens=capacity,
            max_retained_segments=MAX_RETAINED_SEGMENTS,
            max_pending_tokens=capacity,
            max_pending_segments=2,
            initial_staging_tokens=capacity * 2 + 8,
        ),
    )


def _open_session(pool: ARDiffusionKVCache, session_id: str, *, num_layers: int = 2):
    """Bind one session's branch adapters and return its segmented state."""
    state = ARDiffusionKVState(pool, session_id, {"main": pool.begin_request(session_id)}, num_layers=num_layers)
    return state, state.segmented("main")


def _free(pool: ARDiffusionKVCache) -> int:
    return pool.manager.block_pool.get_num_free_blocks()


def _blocks_for(tokens: int) -> int:
    return (tokens + _BLOCK - 1) // _BLOCK


def _commit_phase(cache, ledger, *, audio: int, video: int, layers: int = 2) -> torch.Tensor:
    """Run one phase's clean commit through the real transaction."""
    text = 3
    cache.prepare_layer_views(media_tokens=audio + video, condition_tokens=text)
    media = torch.randn(audio + video, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)
    condition = torch.randn(text, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)
    ledger.begin_commit(audio_tokens=audio, video_tokens=video)
    for layer in range(layers):
        cache.assemble(layer, condition, condition, media, media)
        cache.stage(layer)
    ledger.commit()
    return media


def _expected_rows(phases: list[tuple[int, int]], retention: TaoMateRetention) -> int:
    """Rows a persisted history holds under ``retention`` after ``phases``.

    Mirrors ``TaoMateKVHistory._retained``: the sink is the first phase's video
    segment, included once even while it is still inside the recent window.
    """
    audio_rows = [audio for audio, _ in phases]
    video_rows = [video for _, video in phases]
    window = retention.recent_phases
    recent_audio = sum(audio_rows[-window:]) if window > 0 else 0
    recent_video = sum(video_rows[-window:]) if window > 0 else 0
    sink_video = video_rows[0] if retention.keep_sink and video_rows else 0
    # A recent phase that is also the sink contributes its video rows once.
    sink_already_recent = retention.keep_sink and window > 0 and len(video_rows) <= window
    return recent_audio + recent_video + (0 if sink_already_recent else sink_video)


# ----------------------------------------------------------------------------
# policy resolution and bounds


def test_default_policy_preserves_the_previous_behaviour() -> None:
    assert resolve_retention(None) == TaoMateRetention()
    assert resolve_retention(None) == RETENTION_POLICIES["sink_recent"]
    assert resolve_retention("recent_only") == TaoMateRetention(keep_sink=False, recent_phases=2)


def test_unknown_policy_is_rejected_rather_than_silently_defaulted() -> None:
    """A typo must not silently change which KV the model still attends to."""
    with pytest.raises(ValueError, match="Unknown TaoMate retention policy"):
        resolve_retention("sink-recent-typo")


@pytest.mark.parametrize("keep_sink", [True, False])
def test_policy_over_the_segment_budget_fails_at_construction(keep_sink) -> None:
    """The pipeline reserves five segments; a wider policy must not load."""
    with pytest.raises(ValueError, match="above the declared budget"):
        TaoMateRetention(keep_sink=keep_sink, recent_phases=3)


def test_negative_recent_phases_is_rejected() -> None:
    with pytest.raises(ValueError, match="recent_phases must be non-negative"):
        TaoMateRetention(recent_phases=-1)


# ----------------------------------------------------------------------------
# eviction and audio reset, checked against the pool itself


@pytest.mark.parametrize("policy_name", sorted(RETENTION_POLICIES))
def test_retention_keeps_exactly_the_policys_segments_over_thirteen_windows(policy_name) -> None:
    retention = RETENTION_POLICIES[policy_name]
    pool = _segmented_pool()
    baseline = _free(pool)
    state, cache = _open_session(pool, "test")
    ledger = TaoMateKVHistory(lambda: cache, retention)
    phases: list[tuple[int, int]] = []
    try:
        for window in range(13):
            for phase in range(NUM_PHASES):
                if not phases:
                    audio, video = 3 + (window + phase) % 3, 12
                else:
                    audio = 3 + (window + phase) % 3
                    video = 10 if phase == 0 else (10, 10, 5)[phase - 1]
                _commit_phase(cache, ledger, audio=audio, video=video)
                phases.append((audio, video))
                # Retained rows must equal the policy's selection exactly: one
                # row more or less is a wrong history.
                assert cache.history_tokens == _expected_rows(phases, retention)

            handles = ledger._retained(ledger._recent, ledger._sink)
            assert len(handles) == retention.recent_phases * 2 + int(retention.keep_sink)
            # Eviction returns every segment the policy dropped.
            assert baseline - _free(pool) == sum(_blocks_for(handle.num_tokens) for handle in handles)

        assert ledger.committed_phases == 13 * NUM_PHASES
        handles = ledger._retained(ledger._recent, ledger._sink)
        assert handles[-1] is ledger._recent[-1][1]
        if retention.keep_sink:
            assert handles[0] is ledger._sink
    finally:
        state.close()
    # Teardown restores every free block.
    assert _free(pool) == baseline


def test_audio_reset_drops_only_audio_and_keeps_video_history() -> None:
    retention = RETENTION_POLICIES["sink_recent"]
    pool = _segmented_pool()
    state, cache = _open_session(pool, "test")
    ledger = TaoMateKVHistory(lambda: cache, retention)
    phases: list[tuple[int, int]] = []
    try:
        for _ in range(2 * NUM_PHASES):
            audio, video = 4, 10
            _commit_phase(cache, ledger, audio=audio, video=video)
            phases.append((audio, video))

        before = cache.history_tokens
        dropped = ledger.drop_audio_history()

        # Only the recent window's audio is released.
        expected_dropped = sum(audio for audio, _ in phases[-retention.recent_phases :])
        assert dropped == expected_dropped
        assert cache.history_tokens == before - expected_dropped
        assert all(audio is None for audio, _ in ledger._recent)
        assert ledger._sink is not None
        # Video rows are untouched: the sink plus the recent videos.
        expected_video = phases[0][1] + sum(video for _, video in phases[-retention.recent_phases :])
        assert cache.history_tokens == expected_video
    finally:
        state.close()


def test_audio_reset_is_refused_during_a_commit() -> None:
    pool = _segmented_pool()
    state, cache = _open_session(pool, "test")
    ledger = TaoMateKVHistory(lambda: cache)
    try:
        _commit_phase(cache, ledger, audio=4, video=10)
        cache.prepare_layer_views(media_tokens=14, condition_tokens=1)
        ledger.begin_commit(audio_tokens=4, video_tokens=10)
        with pytest.raises(RuntimeError, match="Cannot reset audio during a clean commit"):
            ledger.drop_audio_history()
        ledger.rollback()
    finally:
        state.close()


def test_audio_reset_on_an_empty_session_is_a_noop() -> None:
    pool = _segmented_pool()
    state, cache = _open_session(pool, "test")
    ledger = TaoMateKVHistory(lambda: cache)
    try:
        assert ledger.drop_audio_history() == 0
        assert ledger.committed_phases == 0 and cache.history_tokens == 0
    finally:
        state.close()


# ----------------------------------------------------------------------------
# transaction order: a failed transaction never advances the ledger


@pytest.mark.parametrize("policy_name", sorted(RETENTION_POLICIES))
def test_failed_commit_rolls_back_without_advancing_the_ledger(policy_name) -> None:
    pool = _segmented_pool()
    baseline = _free(pool)
    state, cache = _open_session(pool, "test")
    ledger = TaoMateKVHistory(lambda: cache, RETENTION_POLICIES[policy_name])
    try:
        cache.prepare_layer_views(media_tokens=5, condition_tokens=1)
        ledger.begin_commit(audio_tokens=2, video_tokens=3)
        rows = torch.ones(5, _HEADS, _HEAD_DIM, dtype=torch.bfloat16)
        # Stage one of the two layers, so ``commit`` must refuse.
        cache.assemble(0, rows[:1], rows[:1], rows, rows)
        cache.stage(0)
        with pytest.raises(RuntimeError):
            ledger.commit()
        assert ledger.committed_phases == 0
        assert cache.history_tokens == 0
        ledger.rollback()
        assert _free(pool) == baseline

        # The same phase commits normally afterwards.
        _commit_phase(cache, ledger, audio=2, video=3)
        assert ledger.committed_phases == 1 and cache.history_tokens == 5
    finally:
        state.close()
        assert _free(pool) == baseline


def test_failed_retain_leaves_the_ledger_describing_stored_rows() -> None:
    """A failing audio reset must not report audio as dropped."""
    pool = _segmented_pool()
    state, cache = _open_session(pool, "test")
    ledger = TaoMateKVHistory(lambda: cache)
    try:
        _commit_phase(cache, ledger, audio=4, video=10)
        before_rows = cache.history_tokens
        recent_before = list(ledger._recent)

        def refuse(handles):
            raise RuntimeError("retain refused")

        original = cache.retain
        cache.retain = refuse
        try:
            with pytest.raises(RuntimeError, match="retain refused"):
                ledger.drop_audio_history()
        finally:
            cache.retain = original

        # The engine change did not land, so the ledger must still name the
        # audio segments that are still stored.
        assert ledger._recent == recent_before
        assert cache.history_tokens == before_rows
    finally:
        state.close()


# ----------------------------------------------------------------------------
# release failures: no ledger that names reclaimed pages


def test_release_failure_never_leaves_the_ledger_naming_reclaimed_pages() -> None:
    pool = _segmented_pool()
    state, cache = _open_session(pool, "test")
    ledger = TaoMateKVHistory(lambda: cache)
    released: list[str] = []
    try:
        for _ in range(4):
            _commit_phase(cache, ledger, audio=4, video=10)

        original = pool.release_history

        def half_failing(adapter):
            # Return the first segment to the pool, then refuse the second, so
            # the ledger is straddling released and unreleased segments.
            if released:
                raise RuntimeError("release refused")
            released.append(adapter.request_id)
            return original(adapter)

        pool.release_history = half_failing
        try:
            # A narrower retention forces a publish that must release segments.
            with pytest.raises(RuntimeError, match="release refused"):
                cache.retain(list(ledger._retained(ledger._recent[-1:], None)))
        finally:
            pool.release_history = original

        assert released, "the test must have released at least one segment first"
        # The ledger must never name a handle whose allocation is gone: that is
        # exactly what would make the next gather read another request's rows.
        assert all(handle in cache._allocations for handle in cache.handles)
        # The session refuses further attention instead of reading stale pages.
        with pytest.raises(RuntimeError, match="failed release"):
            cache.prepare_layer_views(media_tokens=5, condition_tokens=1)
    finally:
        # Cleanup still reclaims every remaining allocation.
        state.close()
    assert _free(pool) == _free(_segmented_pool())


# ----------------------------------------------------------------------------
# release scope: one session's departure leaves another's history intact


def test_releasing_one_session_preserves_another_sessions_history() -> None:
    """This is the in-repository form of "one viewer leaves" (task 11/12/14
    own viewer separation and the broadcast; here a release names one session).
    """
    pool = _segmented_pool()
    baseline = _free(pool)
    kept_state, kept_cache = _open_session(pool, "kept")
    kept_ledger = TaoMateKVHistory(lambda: kept_cache)
    left_state, left_cache = _open_session(pool, "left")
    left_ledger = TaoMateKVHistory(lambda: left_cache)
    try:
        for _ in range(NUM_PHASES):
            _commit_phase(kept_cache, kept_ledger, audio=4, video=10)
            _commit_phase(left_cache, left_ledger, audio=4, video=10)

        kept_rows = kept_cache.history_tokens
        kept_handles = kept_cache.handles
        kept_blocks = list(pool.block_table(kept_state.adapter("main")))
        free_with_both = _free(pool)

        # The departing session is released exactly as a close/abort would.
        left_state.close()

        assert kept_cache.history_tokens == kept_rows
        assert kept_cache.handles == kept_handles
        assert pool.block_table(kept_state.adapter("main")) == kept_blocks
        assert _free(pool) > free_with_both
    finally:
        kept_state.close()
    # And the room the departed session gave back is usable again.
    assert _free(pool) == baseline
    reopened, reopened_cache = _open_session(pool, "reopened")
    try:
        _commit_phase(reopened_cache, TaoMateKVHistory(lambda: reopened_cache), audio=4, video=10)
        assert reopened_cache.history_tokens == 14
    finally:
        reopened.close()
    assert _free(pool) == baseline
