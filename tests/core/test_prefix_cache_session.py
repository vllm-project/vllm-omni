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
