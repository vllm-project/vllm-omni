# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in per-request latency marks for engine-path attribution.

Disabled unless ``VLLM_OMNI_LATENCY_TRACE_DIR`` is set. When enabled, every
process that handles a request appends ``(request_id, mark, t_ns)`` rows to
``<dir>/trace-<pid>.tsv``. ``t_ns`` is ``time.perf_counter_ns()``, which is
``CLOCK_MONOTONIC`` on Linux and therefore comparable across the API server,
the diffusion engine thread and worker processes on one host. Consecutive
marks of one request telescope, so the per-segment times add up exactly to the
end-to-end time of that same request.

Marks are buffered in memory and written by a background thread, so no file
I/O happens on the request path. With the variable unset, :func:`mark` is a
no-op and nothing else in this module runs.
"""

from __future__ import annotations

import atexit
import contextvars
import os
import threading
import time
from collections import deque

_TRACE_DIR = os.environ.get("VLLM_OMNI_LATENCY_TRACE_DIR") or None
ENABLED = _TRACE_DIR is not None
_FLUSH_INTERVAL_S = 0.25

now_ns = time.perf_counter_ns

_events: deque[tuple[str, str, int]] = deque()
_flush_lock = threading.Lock()
_flusher: threading.Thread | None = None
_flusher_pid: int | None = None

# Request id of the request the current asyncio task / thread is serving, for
# layers that timestamp before or after the id is in scope (websocket I/O).
_current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "vllm_omni_latency_trace_request_id", default=None
)
_keyed_request_ids: dict[str, str] = {}
_pending_marks: contextvars.ContextVar[tuple[tuple[str, int], ...]] = contextvars.ContextVar(
    "vllm_omni_latency_trace_pending", default=()
)


def flush() -> None:
    """Write buffered marks to this process's trace file."""
    if not ENABLED:
        return
    with _flush_lock:
        if not _events:
            return
        rows = []
        while True:
            try:
                rows.append(_events.popleft())
            except IndexError:
                break
        path = os.path.join(_TRACE_DIR, f"trace-{os.getpid()}.tsv")  # type: ignore[arg-type]
        with open(path, "a", encoding="utf-8") as handle:
            handle.write("".join(f"{rid}\t{name}\t{t_ns}\n" for rid, name, t_ns in rows))


def _flush_loop() -> None:
    while True:
        time.sleep(_FLUSH_INTERVAL_S)
        try:
            flush()
        except Exception:  # noqa: BLE001 - tracing must never break serving
            pass


def _start_flusher() -> None:
    global _flusher, _flusher_pid
    pid = os.getpid()
    if _flusher_pid == pid:
        return
    _flusher_pid = pid
    os.makedirs(_TRACE_DIR, exist_ok=True)  # type: ignore[arg-type]
    _flusher = threading.Thread(target=_flush_loop, name="latency-trace-flush", daemon=True)
    _flusher.start()
    atexit.register(flush)


def _after_fork_in_child() -> None:
    # A forked child inherits the parent's unflushed rows but not its thread.
    global _flusher_pid
    _events.clear()
    _flusher_pid = None
    _start_flusher()


if ENABLED:
    _start_flusher()
    os.register_at_fork(after_in_child=_after_fork_in_child)

    def mark(request_id: str | None, name: str, t_ns: int | None = None) -> None:
        """Record ``name`` for ``request_id`` now, or at an earlier ``t_ns``."""
        if request_id is None:
            return
        _events.append((request_id, name, time.perf_counter_ns() if t_ns is None else t_ns))

    def defer(name: str, t_ns: int | None = None) -> None:
        """Remember a mark taken before the request id exists; see :func:`bind`."""
        _pending_marks.set((*_pending_marks.get(), (name, time.perf_counter_ns() if t_ns is None else t_ns)))

    def bind(request_id: str) -> None:
        """Attach this task to ``request_id`` and emit its deferred marks."""
        _current_request_id.set(request_id)
        for name, t_ns in _pending_marks.get():
            _events.append((request_id, name, t_ns))
        _pending_marks.set(())

    def mark_current(name: str) -> None:
        """Record ``name`` for the request bound to the current task."""
        request_id = _current_request_id.get()
        if request_id is not None:
            _events.append((request_id, name, time.perf_counter_ns()))

    def unbind() -> None:
        _current_request_id.set(None)
        _pending_marks.set(())

    def current() -> str | None:
        return _current_request_id.get()

    def remember(key: str, request_id: str | None) -> None:
        """Associate an opaque handoff key (e.g. an async output id) with a request."""
        if request_id is not None:
            _keyed_request_ids[key] = request_id

    def recall(key: str) -> str | None:
        return _keyed_request_ids.pop(key, None)

else:

    def mark(request_id: str | None, name: str, t_ns: int | None = None) -> None:
        return None

    def defer(name: str, t_ns: int | None = None) -> None:
        return None

    def bind(request_id: str) -> None:
        return None

    def mark_current(name: str) -> None:
        return None

    def unbind() -> None:
        return None

    def current() -> str | None:
        return None

    def remember(key: str, request_id: str | None) -> None:
        return None

    def recall(key: str) -> str | None:
        return None
