# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One reference-audio encoder shared by the API processes of a server.

Each API process otherwise loads its own copy of the audio tokenizer on the
GPU and encodes only the clips it receives. When the processes share a
directory (a multi-API server's own by default, or
``VLLM_OMNI_MOSS_REF_CODES_SHARED_DIR``), the first one to take the lock in it
hosts the encoder and serves the others over a Unix socket in that directory.
Requests from all processes are merged into batches, and only the host keeps
the tokenizer and its CUDA graphs on the GPU.

Messages carry raw arrays and a JSON header; nothing is unpickled. A client
that cannot reach the host encodes locally instead.
"""

from __future__ import annotations

import collections
import fcntl
import json
import os
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from multiprocessing.connection import Client, Connection, Listener

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_LOCK_NAME = "ref-encoder.lock"
_SOCKET_NAME = "ref-encoder.sock"
# A Unix socket path must fit in sun_path with its terminator: 108 bytes on
# Linux, 104 on macOS.
_MAX_SOCKET_PATH_BYTES = 103
# Clips one worker merges into an encode, across all API processes. Several
# workers encode concurrently on their own streams: one serial queue of large
# batches makes a burst of new references wait behind each other. A free
# worker takes what is queued without waiting for more: API processes already
# gather their clips before sending them.
_MAX_BATCH = 8
# A client gives up on the host after this long and encodes locally.
_CONNECT_TIMEOUT_S = 120.0
_REQUEST_TIMEOUT_S = 120.0
# First-time compilation can outlast a normal encode request. Readiness has a
# separate deadline and must finish before the API starts accepting requests.
_STARTUP_TIMEOUT_S = 1200.0

EncodeFn = Callable[[list[torch.Tensor]], list]

# Open for the life of the process: closing it would release the host lock.
_host_lock_handle = None


class SharedReferenceEncoderStartupError(RuntimeError):
    """The shared encoder cannot safely accept requests yet."""


def socket_path_fits(shared_dir: str) -> bool:
    """Whether the host's Unix socket in ``shared_dir`` has a valid address."""
    return len(os.fsencode(os.path.join(shared_dir, _SOCKET_NAME))) <= _MAX_SOCKET_PATH_BYTES


def elect_host(shared_dir: str) -> bool:
    """True in the one process that holds the encoder lock for ``shared_dir``.

    The lock is held for the life of the process, so a restarted server
    elects a new host even if the old socket file is still present.
    """
    global _host_lock_handle
    path = os.path.join(shared_dir, _LOCK_NAME)
    handle = open(path, "a+")  # noqa: SIM115 - kept open to hold the lock
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    _host_lock_handle = handle
    return True


def _send(conn: Connection, header: dict, arrays: list[np.ndarray]) -> None:
    conn.send_bytes(json.dumps(header).encode())
    for array in arrays:
        conn.send_bytes(np.ascontiguousarray(array).tobytes())


def _recv(conn: Connection) -> tuple[dict, list[bytes]]:
    header = json.loads(conn.recv_bytes().decode())
    return header, [conn.recv_bytes() for _ in range(int(header["count"]))]


class _Job:
    __slots__ = ("wav", "done", "result", "queued_at")

    def __init__(self, wav: torch.Tensor) -> None:
        self.wav = wav
        self.done = threading.Event()
        self.result: torch.Tensor | BaseException | None = None
        self.queued_at = time.monotonic()


class SharedReferenceEncoderHost:
    """Serves reference encodes for every API process sharing ``shared_dir``.

    ``workers`` pairs each worker's encode function with an optional warmup
    (for example, capturing that worker's CUDA graphs).
    """

    def __init__(
        self,
        shared_dir: str,
        workers: list[tuple[EncodeFn, Callable[[], object] | None]],
        *,
        stream_priority: int = 0,
        batch_window_ms: float = 0.0,
    ) -> None:
        self._stream_priority = stream_priority
        self._batch_window_s = max(0.0, float(batch_window_ms)) / 1000
        self._pending: collections.deque[_Job] = collections.deque()
        # Includes jobs already taken by a worker, so shutdown can release
        # every caller without waiting for an in-flight GPU operation.
        self._outstanding: set[_Job] = set()
        self._ready = threading.Condition()
        self._path = os.path.join(shared_dir, _SOCKET_NAME)
        if os.path.exists(self._path):
            os.unlink(self._path)  # left by a previous host; we hold the lock
        self._listener = Listener(self._path, family="AF_UNIX")
        os.chmod(self._path, 0o600)
        self._closed = False
        self._started = threading.Event()
        self._startup_error: Exception | None = None
        threading.Thread(
            target=self._start_workers, args=(workers,), name="moss-ref-encoder-start", daemon=True
        ).start()
        threading.Thread(target=self._accept_loop, name="moss-ref-encoder-accept", daemon=True).start()
        logger.info("MOSS shared reference encoder serving at %s with %d workers", self._path, len(workers))

    def encode(self, wavs: list[torch.Tensor]) -> list:
        """Encode on behalf of the host process itself."""
        self.wait_until_ready()
        jobs = [_Job(wav) for wav in wavs]
        with self._ready:
            if self._closed:
                raise RuntimeError("shared reference encoder is closed")
            self._outstanding.update(jobs)
            self._pending.extend(jobs)
            self._ready.notify(len(jobs))
        for job in jobs:
            job.done.wait()
        return [job.result for job in jobs]

    def close(self) -> None:
        with self._ready:
            self._closed = True
            self._started.set()
            for job in self._outstanding:
                job.result = RuntimeError("shared reference encoder closed before encoding completed")
                job.done.set()
            self._outstanding.clear()
            self._pending.clear()
            self._ready.notify_all()
        try:
            self._listener.close()
        except OSError:
            pass

    def wait_until_ready(self) -> None:
        if not self._started.wait(_STARTUP_TIMEOUT_S):
            raise SharedReferenceEncoderStartupError("shared reference encoder warmup timed out")
        if self._closed or self._startup_error is not None:
            raise SharedReferenceEncoderStartupError("shared reference encoder startup failed") from self._startup_error

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), name="moss-ref-encoder-conn", daemon=True).start()

    def _serve(self, conn: Connection) -> None:
        with conn:
            while True:
                try:
                    header, payloads = _recv(conn)
                except (EOFError, OSError):
                    return
                if header.get("op") == "ready":
                    try:
                        self.wait_until_ready()
                        error = None
                    except SharedReferenceEncoderStartupError as exc:
                        error = str(exc)
                    try:
                        _send(conn, {"count": 0, "ready": error is None, "error": error}, [])
                    except OSError:
                        return
                    continue
                wavs = [
                    torch.from_numpy(np.frombuffer(data, dtype=np.float32).reshape(shape).copy())
                    for data, shape in zip(payloads, header["shapes"])
                ]
                results = self.encode(wavs)
                errors = [None if isinstance(r, torch.Tensor) else repr(r) for r in results]
                codes = [r.to(torch.int32).numpy() for r in results if isinstance(r, torch.Tensor)]
                reply = {"count": len(codes), "shapes": [list(c.shape) for c in codes], "errors": errors}
                try:
                    _send(conn, reply, codes)
                except OSError:
                    return

    def _start_workers(self, workers: list[tuple[EncodeFn, Callable[[], object] | None]]) -> None:
        # Every warmup (graph capture) finishes before any worker encodes: CUDA
        # work in another thread would invalidate a capture in progress.
        try:
            for _, warmup in workers:
                if warmup is not None:
                    warmup()
            if self._closed:
                return
            for index, (encode_fn, _) in enumerate(workers):
                threading.Thread(
                    target=self._encode_loop, args=(encode_fn,), name=f"moss-ref-encoder-{index}", daemon=True
                ).start()
            logger.info("MOSS shared reference encoder ready: %d workers", len(workers))
        except Exception as exc:  # noqa: BLE001 - wake readiness waiters with the failure
            self._startup_error = exc
            logger.exception("MOSS shared reference encoder startup failed")
        finally:
            self._started.set()

    def _encode_loop(self, encode_fn: EncodeFn) -> None:
        stream = torch.cuda.Stream(priority=self._stream_priority) if torch.cuda.is_available() else None
        with torch.cuda.stream(stream) if stream is not None else nullcontext():
            self._serve_jobs(encode_fn)

    def _serve_jobs(self, encode_fn: EncodeFn) -> None:
        while True:
            with self._ready:
                while True:
                    if self._closed:
                        return
                    if not self._pending:
                        self._ready.wait()
                        continue
                    remaining = self._pending[0].queued_at + self._batch_window_s - time.monotonic()
                    if remaining > 0 and len(self._pending) < _MAX_BATCH:
                        # Keep clips in the common queue while collecting;
                        # workers must not each reserve a singleton first.
                        self._ready.wait(remaining)
                        continue
                    break
                jobs = [self._pending.popleft() for _ in range(min(_MAX_BATCH, len(self._pending)))]
            try:
                results = encode_fn([job.wav for job in jobs])
                if len(results) != len(jobs):
                    raise RuntimeError(
                        f"shared reference encoder returned {len(results)} results for {len(jobs)} clips"
                    )
            except Exception as error:  # noqa: BLE001 - every waiter gets the failure
                results = [error] * len(jobs)
            with self._ready:
                for job, result in zip(jobs, results, strict=True):
                    # close() may already have completed this caller with an
                    # error. A late GPU result must not replace that outcome.
                    if not job.done.is_set():
                        job.result = result
                        job.done.set()
                    self._outstanding.discard(job)


class SharedReferenceEncoderClient:
    """Sends reference clips to the host process of ``shared_dir``."""

    def __init__(self, shared_dir: str) -> None:
        self._path = os.path.join(shared_dir, _SOCKET_NAME)
        # One connection per request in flight; idle ones are reused.
        self._idle: list[Connection] = []
        self._lock = threading.Lock()
        self._connected = False

    def _connect(self) -> Connection:
        # The host may still be starting until the first connection succeeds;
        # later reconnects are quick.
        deadline = time.monotonic() + (1.0 if self._connected else _CONNECT_TIMEOUT_S)
        while True:
            try:
                conn = Client(self._path, family="AF_UNIX")
                self._connected = True
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.2)
        try:
            _send(conn, {"op": "ready", "count": 0}, [])
            if not conn.poll(_STARTUP_TIMEOUT_S):
                raise SharedReferenceEncoderStartupError("shared reference encoder readiness timed out")
            header, _ = _recv(conn)
            if not header.get("ready"):
                raise SharedReferenceEncoderStartupError(header.get("error", "shared reference encoder not ready"))
            return conn
        except BaseException:
            conn.close()
            raise

    def wait_until_ready(self) -> None:
        """Startup handshake, without encoding or populating the reference cache."""
        try:
            conn = self._connect()
        except Exception as exc:
            raise SharedReferenceEncoderStartupError("shared reference encoder is not ready") from exc
        with self._lock:
            self._idle.append(conn)

    def encode(self, wavs: list[torch.Tensor]) -> list:
        """Codes (or an exception) per clip; raises when the host is unreachable."""
        with self._lock:
            conn = self._idle.pop() if self._idle else None
        try:
            if conn is None:
                conn = self._connect()
            arrays = [w.detach().to(torch.float32).cpu().numpy() for w in wavs]
            _send(conn, {"count": len(arrays), "shapes": [list(a.shape) for a in arrays]}, arrays)
            if not conn.poll(_REQUEST_TIMEOUT_S):
                raise TimeoutError(f"no reply from the shared reference encoder in {_REQUEST_TIMEOUT_S}s")
            header, payloads = _recv(conn)
        except BaseException:
            if conn is not None:
                conn.close()
            raise
        with self._lock:
            self._idle.append(conn)
        codes = iter(
            torch.from_numpy(np.frombuffer(data, dtype=np.int32).reshape(shape).copy()).long()
            for data, shape in zip(payloads, header["shapes"])
        )
        return [next(codes) if error is None else RuntimeError(error) for error in header["errors"]]
