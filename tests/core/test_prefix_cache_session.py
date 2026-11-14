# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Unit tests for vllm_omni/core/prefix_cache/session_store.

CPU-only, no vLLM runtime required (same namespace-parent shim as
test_prefix_cache.py: bypasses vllm_omni/__init__ which imports vllm).
"""

import logging
import random
import sys
import threading
import time
from pathlib import Path

import pytest
import torch

try:  # pragma: no cover - shim only matters on vllm-less dev machines
    import vllm  # noqa: F401
except ModuleNotFoundError:
    _root = Path(__file__).resolve().parents[2]
    for _pkg in ("vllm_omni", "vllm_omni.core", "vllm_omni.utils"):
        if _pkg not in sys.modules:
            _m = __import__("types").ModuleType(_pkg)
            _m.__path__ = [str(_root / _pkg.replace(".", "/"))]
            sys.modules[_pkg] = _m
    _vllm = __import__("types").ModuleType("vllm")
    _vllm_logger = __import__("types").ModuleType("vllm.logger")
    _vllm_logger.init_logger = logging.getLogger
    _vllm.logger = _vllm_logger
    sys.modules["vllm"] = _vllm
    sys.modules["vllm.logger"] = _vllm_logger

from vllm_omni.core.prefix_cache.session_store import SessionPrefixStore

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _rows(start: int, n: int, width: int = 4) -> torch.Tensor:
    return torch.arange(start, start + n, dtype=torch.float32).unsqueeze(1).expand(n, width).clone()


def test_bind_unbind_session_of():
    store = SessionPrefixStore()
    assert store.session_of("req-1") is None
    store.bind("req-1", "sess-A")
    assert store.session_of("req-1") == "sess-A"
    store.unbind("req-1")
    assert store.session_of("req-1") is None


def test_append_fetch_multi_segment_prefix():
    store = SessionPrefixStore()
    assert store.append("sess-A", "hidden", _rows(0, 3))
    assert store.append("sess-A", "hidden", _rows(3, 2))
    assert store.stored_tokens("sess-A", "hidden") == 5
    assert store.covers("sess-A", "hidden", 5)
    fetched = store.fetch("sess-A", "hidden", 4)
    assert fetched is not None
    assert torch.equal(fetched, _rows(0, 4))
    # Partial fetch: first 3 rows only.
    assert torch.equal(store.fetch("sess-A", "hidden", 3), _rows(0, 3))


def test_fetch_returns_none_when_not_covered():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 2))
    assert not store.covers("sess-A", "hidden", 3)
    assert store.fetch("sess-A", "hidden", 3) is None
    assert store.fetch("sess-A", "hidden", 0) is None
    assert store.fetch("sess-A", "other-key", 1) is None
    assert store.fetch("sess-B", "hidden", 1) is None


def test_snapshot_range_returns_window_and_none_when_uncovered():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 8))
    store.append("sess-A", "hidden", _rows(8, 8))
    # A middle window spanning both segments.
    parts = store.snapshot_range("sess-A", "hidden", 4, 12)
    assert parts is not None
    assert torch.equal(torch.cat(parts, dim=0), _rows(4, 8))
    # Full prefix and a prefix window are windows too.
    assert store.snapshot_range("sess-A", "hidden", 0, 16) is not None
    assert torch.equal(torch.cat(store.snapshot_range("sess-A", "hidden", 0, 4), dim=0), _rows(0, 4))
    # Past the end, and an empty window, are not covered.
    assert store.snapshot_range("sess-A", "hidden", 12, 20) is None
    assert store.snapshot_range("sess-A", "hidden", 8, 8) is None
    assert store.snapshot_range("sess-A", "hidden", 0, 0) is None


def test_sessions_are_isolated():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 3))
    store.append("sess-B", "hidden", _rows(100, 3))
    # Invalidating A must not touch B's rows.
    assert store.invalidate("sess-A") == 1
    assert store.stored_tokens("sess-A", "hidden") == 0
    assert store.fetch("sess-A", "hidden", 1) is None
    assert torch.equal(store.fetch("sess-B", "hidden", 3), _rows(100, 3))
    assert store.session_ids() == ["sess-B"]


def test_invalidate_resets_stream_for_rebuild():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 3))
    store.invalidate("sess-A")
    # Fresh stream after invalidation: new rows start at position 0.
    assert store.append("sess-A", "hidden", _rows(50, 2))
    assert torch.equal(store.fetch("sess-A", "hidden", 2), _rows(50, 2))


def test_shape_mismatch_raises():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 2, width=4))
    with pytest.raises(ValueError, match="shape mismatch"):
        store.append("sess-A", "hidden", _rows(0, 2, width=8))


def test_byte_budget_refuses_without_partial_apply():
    # 3 tokens x 4 floats x 4 bytes = 48 bytes; budget fits one append only.
    store = SessionPrefixStore(max_bytes=48)
    assert store.append("sess-A", "hidden", _rows(0, 3))
    assert not store.append("sess-A", "hidden", _rows(3, 3))
    assert store.refused_appends == 1
    assert store.stored_tokens("sess-A", "hidden") == 3
    assert store.stored_bytes() == 48


def test_drop_unbinds_requests_and_segments():
    store = SessionPrefixStore()
    store.bind("req-1", "sess-A")
    store.bind("req-2", "sess-A")
    store.append("sess-A", "hidden", _rows(0, 2))
    assert store.drop("sess-A") == 1
    assert store.session_of("req-1") is None
    assert store.session_of("req-2") is None
    assert store.fetch("sess-A", "hidden", 1) is None


def test_fetch_returns_copy_not_alias():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 2))
    fetched = store.fetch("sess-A", "hidden", 2)
    fetched[0, 0] = -999.0
    assert torch.equal(store.fetch("sess-A", "hidden", 2), _rows(0, 2))


def test_append_if_length_is_atomic():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 3))
    assert store.append_if_length("sess-A", "hidden", 3, _rows(3, 2))
    assert store.stored_tokens("sess-A", "hidden") == 5
    # Stale expectation: nothing is stored, stream unchanged.
    assert not store.append_if_length("sess-A", "hidden", 3, _rows(99, 2))
    assert store.stored_tokens("sess-A", "hidden") == 5
    assert torch.equal(store.fetch("sess-A", "hidden", 5), _rows(0, 5))


def test_append_beyond_overlap_appends_only_tail():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 8))
    # Batch [4, 12): only the not-yet-stored tail [8, 12) appends.
    assert store.append_beyond("sess-A", "hidden", 4, _rows(4, 8))
    assert store.stored_tokens("sess-A", "hidden") == 12
    assert torch.equal(store.fetch("sess-A", "hidden", 12), _rows(0, 12))


def test_append_beyond_fully_covered_is_noop():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 8))
    assert store.append_beyond("sess-A", "hidden", 4, _rows(4, 4))
    assert store.stored_tokens("sess-A", "hidden") == 8
    assert torch.equal(store.fetch("sess-A", "hidden", 8), _rows(0, 8))


def test_append_beyond_gap_is_refused():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 4))
    # Batch [8, 12) starts ahead of the stream: the missing middle
    # cannot be filled from this batch, so nothing is stored.
    assert not store.append_beyond("sess-A", "hidden", 8, _rows(8, 4))
    assert store.stored_tokens("sess-A", "hidden") == 4
    assert torch.equal(store.fetch("sess-A", "hidden", 4), _rows(0, 4))
    with pytest.raises(ValueError, match="batch_start >= 0"):
        store.append_beyond("sess-A", "hidden", -1, _rows(0, 2))


def test_append_beyond_exact_continuation_appends_in_full():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 4))
    assert store.append_beyond("sess-A", "hidden", 4, _rows(4, 4))
    assert store.stored_tokens("sess-A", "hidden") == 8
    assert torch.equal(store.fetch("sess-A", "hidden", 8), _rows(0, 8))


# Fork-detection tests (direction 1): block-aligned batches, as a real
# pool hit is always block aligned. A batch overlapping stored rows must
# carry the same content there; a mismatch means the session forked, so
# the whole session is invalidated instead of serving stale rows later.
def test_append_beyond_noop_mismatch_invalidates_session():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 32))
    store.append("sess-A", "other", _rows(0, 32))
    # Branch B rewrites positions [0, 16) with different content; the
    # batch is fully covered, so without a content check it no-ops.
    assert not store.append_beyond("sess-A", "hidden", 0, _rows(100, 16))
    # Whole session is gone, all keys, not just the mismatched one.
    assert store.stored_tokens("sess-A", "hidden") == 0
    assert store.stored_tokens("sess-A", "other") == 0
    assert store.fetch("sess-A", "hidden", 1) is None
    # Rebuild from scratch works (the seeding path appends at 0).
    assert store.append("sess-A", "hidden", _rows(100, 16))
    assert torch.equal(store.fetch("sess-A", "hidden", 16), _rows(100, 16))


def test_append_beyond_overlap_mismatch_invalidates_session():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 32))
    store.append("sess-A", "other", _rows(0, 32))
    # Batch [16, 48): overlapping prefix is fork content, tail is new.
    batch = torch.cat([_rows(100, 16), _rows(32, 16)], dim=0)
    assert not store.append_beyond("sess-A", "hidden", 16, batch)
    assert store.stored_tokens("sess-A", "hidden") == 0
    assert store.stored_tokens("sess-A", "other") == 0
    # No misaligned tail was stored: the stream is empty, not
    # old-prefix + new-tail.
    assert store.fetch("sess-A", "hidden", 1) is None


def test_append_beyond_overlap_mismatch_spans_segments():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 16))
    store.append("sess-A", "hidden", _rows(16, 16))
    # Batch [8, 40): overlapping [8, 32) spans both stored segments.
    batch = torch.cat([_rows(100, 24), _rows(32, 8)], dim=0)
    assert not store.append_beyond("sess-A", "hidden", 8, batch)
    assert store.stored_tokens("sess-A", "hidden") == 0


def test_append_beyond_overlap_match_within_tolerance_appends_tail():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 32))
    # Same prefix up to numeric noise: only the tail appends.
    batch = torch.cat([_rows(16, 16) + 0.001, _rows(32, 16)], dim=0)
    assert store.append_beyond("sess-A", "hidden", 16, batch)
    assert store.stored_tokens("sess-A", "hidden") == 48
    assert torch.equal(store.fetch("sess-A", "hidden", 48), _rows(0, 48))


def test_append_beyond_noop_match_within_tolerance_is_noop():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 32))
    assert store.append_beyond("sess-A", "hidden", 0, _rows(0, 16) + 0.001)
    assert store.stored_tokens("sess-A", "hidden") == 32
    assert torch.equal(store.fetch("sess-A", "hidden", 32), _rows(0, 32))


def test_append_beyond_beyond_tolerance_mismatch_invalidates():
    store = SessionPrefixStore()
    store.append("sess-A", "hidden", _rows(0, 32))
    # Perturbation 1.0 exceeds atol + rtol*|value| at row 0 (allowed 0.01).
    assert not store.append_beyond("sess-A", "hidden", 0, _rows(0, 16) + 1.0)
    assert store.stored_tokens("sess-A", "hidden") == 0


def test_concurrent_append_invalidate_fetch_never_serves_misaligned_rows():
    """Writer/invalidator/reader race: a fetch is always None or the
    exact [0, upto) prefix of the current stream.

    The writer derives row values from the stream position it read, so
    any misaligned append (rows landing at the wrong positions after a
    racing invalidate) would surface here as a fetch whose values do not
    start at 0. ``append_if_length`` retries on a stale read, which is
    what keeps the invariant.
    """
    store = SessionPrefixStore()
    errors: list[Exception] = []

    def writer():
        try:
            for _ in range(300):
                pos = store.stored_tokens("S", "hidden")
                rows = torch.arange(pos, pos + 2, dtype=torch.float32).unsqueeze(1).expand(2, 4).clone()
                if store.append_if_length("S", "hidden", pos, rows):
                    time.sleep(0.0005)
        except Exception as e:  # noqa: BLE001 - collected for the assertion below
            errors.append(e)

    def invalidator():
        try:
            for _ in range(60):
                time.sleep(0.001)
                store.invalidate("S")
        except Exception as e:  # noqa: BLE001 - collected for the assertion below
            errors.append(e)

    def reader():
        try:
            for _ in range(300):
                upto = random.randint(1, 64)
                got = store.fetch("S", "hidden", upto)
                if got is not None:
                    expected = torch.arange(0, upto, dtype=torch.float32).unsqueeze(1).expand(upto, 4).clone()
                    assert torch.equal(got, expected), f"misaligned fetch upto={upto}"
                time.sleep(0.0005)
        except Exception as e:  # noqa: BLE001 - collected for the assertion below
            errors.append(e)

    threads = [threading.Thread(target=writer), threading.Thread(target=invalidator)] + [
        threading.Thread(target=reader) for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors
    assert all(not t.is_alive() for t in threads)


def test_terminate_resumable_drops_session():
    store = SessionPrefixStore()
    store.bind("req-1", "sess-A", resumable=True)
    store.append("sess-A", "hidden", _rows(0, 3))
    assert store.terminate("req-1") == "sess-A"
    assert store.session_of("req-1") is None
    assert store.stored_tokens("sess-A", "hidden") == 0
    assert "sess-A" not in store.session_ids()


def test_terminate_ephemeral_only_unbinds():
    store = SessionPrefixStore()
    store.bind("req-1", "sess-A")
    store.append("sess-A", "hidden", _rows(0, 3))
    assert store.terminate("req-1") is None
    assert store.session_of("req-1") is None
    assert store.stored_tokens("sess-A", "hidden") == 3


def test_terminate_unknown_request_is_noop():
    store = SessionPrefixStore()
    assert store.terminate("no-such-req") is None


def test_session_epoch_record_and_read():
    """6b: the store records the history generation the manager hands it."""
    store = SessionPrefixStore()
    assert store.epoch_of("S") is None
    store.set_epoch("S", 2)
    assert store.epoch_of("S") == 2
    store.set_epoch("S", 5)
    assert store.epoch_of("S") == 5
    # Epoch 0 is a valid generation (the fence default), not "absent".
    store.set_epoch("S", 0)
    assert store.epoch_of("S") == 0


def test_session_epoch_survives_invalidate_but_not_drop():
    """6b: invalidate keeps the generation (same session, fresh stream);
    only drop forgets it."""
    store = SessionPrefixStore()
    store.append("S", "hidden", _rows(0, 3))
    store.set_epoch("S", 1)
    store.invalidate("S")
    assert store.stored_tokens("S", "hidden") == 0
    assert store.epoch_of("S") == 1
    store.append("S", "hidden", _rows(0, 2))
    store.drop("S")
    assert store.epoch_of("S") is None


def test_compare_and_set_epoch_invalidates_atomically():
    """Epoch advancement drops the old rows and records the new generation
    in one lock hold (no window for an in-flight append)."""
    store = SessionPrefixStore()
    store.append("S", "hidden", _rows(0, 3))
    # First sight records without touching rows.
    assert store.compare_and_set_epoch("S", 0) is None
    assert store.stored_tokens("S", "hidden") == 3
    # Same generation keeps rows.
    assert store.compare_and_set_epoch("S", 0) is None
    assert store.stored_tokens("S", "hidden") == 3
    # Advancing drops the rows and reports the previous generation.
    assert store.compare_and_set_epoch("S", 1) == 0
    assert store.stored_tokens("S", "hidden") == 0
    assert store.epoch_of("S") == 1


def test_append_refuses_rows_from_a_superseded_epoch():
    """The epoch is re-checked under the store lock with the length/position,
    so an old materialize cannot seed a stream a new STARTED already reset."""
    store = SessionPrefixStore()
    store.compare_and_set_epoch("S", 0)
    store.append("S", "hidden", _rows(0, 3))
    assert store.compare_and_set_epoch("S", 1) == 0
    assert not store.append_if_length("S", "hidden", 0, _rows(0, 2), expected_epoch=0)
    assert store.stored_tokens("S", "hidden") == 0
    assert not store.append_beyond("S", "hidden", 0, _rows(0, 2), expected_epoch=0)
    assert store.stored_tokens("S", "hidden") == 0
    # A current-generation append still lands.
    assert store.append_beyond("S", "hidden", 0, _rows(0, 2), expected_epoch=1)
    assert store.stored_tokens("S", "hidden") == 2


def test_drop_idle_sessions_reaps_only_idle():
    """7: the reaper drops sessions idle past the TTL and keeps live ones."""
    store = SessionPrefixStore()
    store.append("S-old", "hidden", _rows(0, 3))
    store.append("S-live", "hidden", _rows(0, 2))
    store.set_epoch("S-old", 1)
    store.bind("r-old", "S-old")
    store._last_touch["S-old"] = 99.0
    store._last_touch["S-live"] = 690.0
    dropped = store.drop_idle_sessions(now=700.0, ttl_s=600.0)
    assert dropped == ["S-old"]
    assert store.stored_tokens("S-old", "hidden") == 0
    assert store.stored_tokens("S-live", "hidden") == 2
    assert store.epoch_of("S-old") is None
    assert store.session_of("r-old") is None
    assert "S-old" not in store._last_touch


def test_serve_refreshes_idle_touch():
    """7: serving a session's rows marks it live, so the reap skips it."""
    store = SessionPrefixStore()
    store.append("S", "hidden", _rows(0, 3))
    store._last_touch["S"] = 100.0
    assert store.fetch("S", "hidden", 3) is not None
    touched = store._last_touch["S"]
    assert touched > 100.0
    assert store.drop_idle_sessions(now=touched + 100.0, ttl_s=600.0) == []
    assert store.stored_tokens("S", "hidden") == 3


def test_refused_append_warns_once_per_session(caplog):
    """7: a budget-refused append warns once per session (later refusals
    only debug), and warns again after the session is invalidated."""
    store = SessionPrefixStore(max_bytes=32)
    assert store.append("S", "hidden", _rows(0, 2))
    with caplog.at_level(logging.DEBUG, logger="vllm_omni.core.prefix_cache.session_store"):
        assert not store.append("S", "hidden", _rows(2, 1))
        assert not store.append("S", "hidden", _rows(2, 1))
    assert store.refused_appends == 2
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    debugs = [r for r in caplog.records if r.levelname == "DEBUG"]
    assert len(warnings) == 1
    assert "S" in warnings[0].getMessage()
    assert len(debugs) == 1
    # After invalidate the session may warn again.
    store.invalidate("S")
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="vllm_omni.core.prefix_cache.session_store"):
        assert not store.append("S", "hidden", _rows(0, 3))
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1


def test_reserve_pending_shares_the_byte_budget():
    """Queued (cloned) rows count against max_bytes until they land."""
    store = SessionPrefixStore(max_bytes=32)
    rows = _rows(0, 2)  # 2 * 4 float32 = 32 bytes
    assert store.reserve_pending("S", rows.numel() * rows.element_size())
    assert store.pending_bytes() == 32
    # The budget is full: neither another reservation nor an append fits.
    assert not store.reserve_pending("S", 1)
    assert not store.append("S", "hidden", _rows(0, 1))
    store.release_pending(32)
    assert store.pending_bytes() == 0
    assert store.append("S", "hidden", rows)


def test_append_beyond_with_reserved_bytes_is_atomic(monkeypatch):
    """The reserved release and the append share one lock hold.

    This test discriminates atomicity: a racing ``release_pending``
    (another session appending the moment the budget is freed) makes
    the split sequence fail, while the single ``append_beyond`` call
    never exposes that window — it releases the reservation without
    calling ``release_pending`` at all, so the competitor never runs.
    A single-threaded test without the racer would pass for the split
    version too.
    """
    rows = _rows(0, 2)  # 32 bytes
    reserved = rows.numel() * rows.element_size()

    def racing_release(store):
        real_release = SessionPrefixStore.release_pending

        def _release(nbytes):
            real_release(store, nbytes)
            # Another session takes the freed budget before the append.
            store.append("T", "hidden", _rows(0, 1))

        return _release

    # Split sequence: release, then append. The competitor lands first
    # (16 bytes) and the 32-byte append no longer fits.
    store = SessionPrefixStore(max_bytes=32)
    assert store.reserve_pending("S", reserved)
    assert not store.append("S", "hidden", _rows(0, 1))
    monkeypatch.setattr(store, "release_pending", racing_release(store))
    store.release_pending(reserved)
    assert store.stored_tokens("T", "hidden") == 1
    assert not store.append_beyond("S", "hidden", 0, rows)
    assert store.stored_tokens("S", "hidden") == 0

    # Single call: the reservation is released and the rows append under
    # one lock hold, so the competitor never gets a turn.
    store = SessionPrefixStore(max_bytes=32)
    assert store.reserve_pending("S", reserved)
    assert not store.append("S", "hidden", _rows(0, 1))
    monkeypatch.setattr(store, "release_pending", racing_release(store))
    assert store.append_beyond("S", "hidden", 0, rows, reserved_bytes=reserved)
    assert store.stored_tokens("T", "hidden") == 0
    assert store.pending_bytes() == 0
    assert store.stored_tokens("S", "hidden") == 2


def test_drop_does_not_orphan_segments_from_an_interleaved_append(monkeypatch):
    """drop() must invalidate and forget under one lock hold.

    On the old two-phase drop, an append that landed after ``invalidate``
    released the lock stored segments and touched the session, then
    ``_drop_locked`` removed the touch record — leaving segments that
    ``drop_idle_sessions`` never scans but that still count against the
    budget. Interleaving an append from the public ``invalidate`` call
    reproduces that window; the atomic drop never takes it."""
    store = SessionPrefixStore()
    store.append("S", "hidden", _rows(0, 3))
    store.bind("r", "S")
    real_invalidate = SessionPrefixStore.invalidate

    def racing_invalidate(session_id):
        dropped = real_invalidate(store, session_id)
        # The old drop() released the lock here; simulate another thread's
        # append landing in the window before _drop_locked runs.
        store.append(session_id, "hidden", _rows(0, 1))
        return dropped

    monkeypatch.setattr(store, "invalidate", racing_invalidate)
    store.drop("S")
    assert [e for e in store._segments if e[0] == "S"] == []
    assert store.stored_bytes() == 0
    assert "S" not in store._last_touch
