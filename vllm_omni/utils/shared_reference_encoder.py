# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference-audio encoders shared by a server's API processes.

Each API process otherwise loads its own copy of a model's reference encoder on
the GPU and encodes only the clips it receives. When the processes share a
directory (a multi-API server's own), the first one to take the lock in it
hosts the encoder workers and serves the others over a Unix socket in that
directory. Requests from all processes are merged into batches, and only the
host's workers keep the encoder on the GPU.

A worker is an encode function, for example one running in this process on its
own CUDA stream, or a ``ProcessEncoderWorker`` that runs the model's encoder in
a spawned process: encoders whose cost is host-side kernel launches encode in
parallel only in separate processes. A model describes how its clips and
artifacts cross process boundaries with a ``WireFormat``. Messages carry raw
arrays and a JSON header; nothing is unpickled. A client that cannot reach the
host encodes locally instead.
"""

from __future__ import annotations

import asyncio
import collections
import fcntl
import importlib
import json
import multiprocessing
import os
import signal
import threading
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from multiprocessing.connection import Client, Connection, Listener
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

from vllm_omni.utils.reference_encoding import ReferenceEncodingService, content_key

logger = init_logger(__name__)

_LOCK_NAME = "ref-encoder.lock"
_SOCKET_NAME = "ref-encoder.sock"
# A Unix socket path must fit in sun_path with its terminator: 108 bytes on
# Linux, 104 on macOS.
_MAX_SOCKET_PATH_BYTES = 103
# Clips one worker merges into an encode, across all API processes. Several
# workers encode concurrently: one serial queue of large batches makes a burst
# of new references wait behind each other. A free worker takes what is queued
# without waiting for more: API processes already gather their clips before
# sending them.
_MAX_BATCH = 8
# A client gives up on the host after this long and encodes locally.
_CONNECT_TIMEOUT_S = 120.0
_REQUEST_TIMEOUT_S = 120.0
# First-time compilation can outlast a normal encode request. Readiness has a
# separate deadline and must finish before the API starts accepting requests.
_STARTUP_TIMEOUT_S = 1200.0

EncodeFn = Callable[[list], list]

# Open for the life of the process, one per shared directory: closing one
# would release that host lock.
_host_lock_handles: dict[str, Any] = {}


@dataclass(frozen=True)
class WireFormat:
    """How a model's clips and artifacts cross process boundaries as arrays.

    A clip travels as one array and an optional sample rate; an artifact as a
    list of arrays.
    """

    clip_to_array: Callable[[Any], tuple[np.ndarray, int | None]]
    clip_from_array: Callable[[np.ndarray, int | None], Any]
    artifact_to_arrays: Callable[[Any], list[np.ndarray]]
    artifact_from_arrays: Callable[[list[np.ndarray]], Any]


# A waveform tensor in, a codes tensor out (MOSS-TTS).
CODES_WIRE = WireFormat(
    clip_to_array=lambda wav: (wav.detach().to(torch.float32).cpu().numpy(), None),
    clip_from_array=lambda array, _sr: torch.from_numpy(array.copy()),
    artifact_to_arrays=lambda codes: [codes.to(torch.int32).numpy()],
    artifact_from_arrays=lambda arrays: torch.from_numpy(arrays[0].copy()).long(),
)


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
    path = os.path.join(shared_dir, _LOCK_NAME)
    handle = open(path, "a+")  # noqa: SIM115 - kept open to hold the lock
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    _host_lock_handles[path] = handle
    return True


def _send(conn: Connection, header: dict, arrays: list[np.ndarray]) -> None:
    conn.send_bytes(json.dumps(header).encode())
    for array in arrays:
        conn.send_bytes(np.ascontiguousarray(array).tobytes())


def _recv(conn: Connection) -> tuple[dict, list[bytes]]:
    header = json.loads(conn.recv_bytes().decode())
    return header, [conn.recv_bytes() for _ in range(int(header["count"]))]


def _send_clips(conn: Connection, clips: list, wire: WireFormat, header: dict | None = None) -> None:
    pairs = [wire.clip_to_array(clip) for clip in clips]
    arrays = [np.ascontiguousarray(array, dtype=np.float32) for array, _ in pairs]
    meta = {"count": len(arrays), "shapes": [list(a.shape) for a in arrays], "rates": [sr for _, sr in pairs]}
    _send(conn, {**(header or {}), **meta}, arrays)


def _clips_from(header: dict, payloads: list[bytes], wire: WireFormat) -> list:
    rates = header.get("rates") or [None] * len(payloads)
    return [
        wire.clip_from_array(np.frombuffer(data, dtype=np.float32).reshape(shape), sr)
        for data, shape, sr in zip(payloads, header["shapes"], rates)
    ]


def _send_results(conn: Connection, results: list, wire: WireFormat) -> None:
    metas: list[dict] = []
    arrays: list[np.ndarray] = []
    for result in results:
        if isinstance(result, BaseException):
            metas.append({"error": repr(result)})
            continue
        parts = [np.ascontiguousarray(a) for a in wire.artifact_to_arrays(result)]
        metas.append({"error": None, "arrays": [{"shape": list(a.shape), "dtype": a.dtype.str} for a in parts]})
        arrays.extend(parts)
    _send(conn, {"count": len(arrays), "results": metas}, arrays)


def _results_from(header: dict, payloads: list[bytes], wire: WireFormat) -> list:
    data = iter(payloads)
    results: list = []
    for meta in header["results"]:
        if meta["error"] is not None:
            results.append(RuntimeError(meta["error"]))
            continue
        parts = [
            np.frombuffer(next(data), dtype=np.dtype(spec["dtype"])).reshape(spec["shape"]) for spec in meta["arrays"]
        ]
        results.append(wire.artifact_from_arrays(parts))
    return results


class _Job:
    __slots__ = ("clip", "done", "result", "queued_at")

    def __init__(self, clip: Any) -> None:
        self.clip = clip
        self.done = threading.Event()
        self.result: Any = None
        self.queued_at = time.monotonic()


class SharedReferenceEncoderHost:
    """Serves reference encodes for this process and, with ``shared_dir``, every API process sharing it.

    ``workers`` pairs each worker's encode function with an optional warmup
    (for example, capturing that worker's CUDA graphs, or starting its
    process). ``worker_streams`` gives each in-process worker its own CUDA
    stream; process workers need none.
    """

    def __init__(
        self,
        shared_dir: str | None,
        workers: list[tuple[EncodeFn, Callable[[], object] | None]],
        *,
        wire: WireFormat = CODES_WIRE,
        stream_priority: int = 0,
        batch_window_ms: float = 0.0,
        worker_streams: bool = True,
        name: str = "reference",
    ) -> None:
        self._wire = wire
        self._name = name
        self._stream_priority = stream_priority
        self._worker_streams = worker_streams
        self._batch_window_s = max(0.0, float(batch_window_ms)) / 1000
        self._pending: collections.deque[_Job] = collections.deque()
        # Includes jobs already taken by a worker, so shutdown can release
        # every caller without waiting for an in-flight GPU operation.
        self._outstanding: set[_Job] = set()
        self._ready = threading.Condition()
        self._closed = False
        self._started = threading.Event()
        self._startup_error: Exception | None = None
        self._listener: Listener | None = None
        if shared_dir is not None:
            self._path = os.path.join(shared_dir, _SOCKET_NAME)
            if os.path.exists(self._path):
                os.unlink(self._path)  # left by a previous host; we hold the lock
            self._listener = Listener(self._path, family="AF_UNIX")
            os.chmod(self._path, 0o600)
            threading.Thread(target=self._accept_loop, name=f"{name}-encoder-accept", daemon=True).start()
        threading.Thread(target=self._start_workers, args=(workers,), name=f"{name}-encoder-start", daemon=True).start()
        logger.info(
            "Shared %s encoder %s with %d workers",
            name,
            f"serving at {self._path}" if shared_dir else "local",
            len(workers),
        )

    def encode(self, clips: list) -> list:
        """Encode on behalf of the host process itself."""
        self.wait_until_ready()
        jobs = [_Job(clip) for clip in clips]
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
        if self._listener is not None:
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
        assert self._listener is not None
        while not self._closed:
            try:
                conn = self._listener.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), name=f"{self._name}-encoder-conn", daemon=True).start()

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
                results = self.encode(_clips_from(header, payloads, self._wire))
                try:
                    _send_results(conn, results, self._wire)
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
                    target=self._encode_loop, args=(encode_fn,), name=f"{self._name}-encoder-{index}", daemon=True
                ).start()
            logger.info("Shared %s encoder ready: %d workers", self._name, len(workers))
        except Exception as exc:  # noqa: BLE001 - wake readiness waiters with the failure
            self._startup_error = exc
            logger.exception("Shared %s encoder startup failed", self._name)
        finally:
            self._started.set()

    def _encode_loop(self, encode_fn: EncodeFn) -> None:
        stream = None
        if self._worker_streams and torch.cuda.is_available():
            stream = torch.cuda.Stream(priority=self._stream_priority)
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
                results = list(encode_fn([job.clip for job in jobs]))
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

    def __init__(self, shared_dir: str, *, wire: WireFormat = CODES_WIRE) -> None:
        self._path = os.path.join(shared_dir, _SOCKET_NAME)
        self._wire = wire
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

    def encode(self, clips: list) -> list:
        """An artifact (or an exception) per clip; raises when the host is unreachable."""
        with self._lock:
            conn = self._idle.pop() if self._idle else None
        try:
            if conn is None:
                conn = self._connect()
            _send_clips(conn, clips, self._wire)
            if not conn.poll(_REQUEST_TIMEOUT_S):
                raise TimeoutError(f"no reply from the shared reference encoder in {_REQUEST_TIMEOUT_S}s")
            header, payloads = _recv(conn)
        except BaseException:
            if conn is not None:
                conn.close()
            raise
        with self._lock:
            self._idle.append(conn)
        return _results_from(header, payloads, self._wire)


def _resolve(path: str) -> Any:
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def _process_worker_main(factory: str, kwargs: dict, wire_path: str, conn: Connection) -> None:
    """Spawned worker: build the encoder, then answer encode requests until the pipe closes."""
    # The parent owns shutdown: it closes the pipe (or the worker dies with it
    # as a daemon). An interrupt sent to the whole process group must not
    # end a worker mid-reply.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    wire: WireFormat = _resolve(wire_path)
    try:
        encoder = _resolve(factory)(**kwargs)
    except Exception as exc:  # noqa: BLE001 - reported to the parent as a startup failure
        _send(conn, {"count": 0, "ready": False, "error": repr(exc)}, [])
        return
    _send(conn, {"count": 0, "ready": True}, [])
    while True:
        try:
            header, payloads = _recv(conn)
        except (EOFError, OSError):
            return
        clips = _clips_from(header, payloads, wire)
        try:
            results = list(encoder.encode_batch(clips))
        except Exception as exc:  # noqa: BLE001 - every clip of the batch gets the failure
            results = [exc] * len(clips)
        _send_results(conn, results, wire)


class ProcessEncoderWorker:
    """An encode function served by a spawned process that builds the encoder itself.

    ``factory`` (``"module:callable"``) is called with ``kwargs`` in the child
    and must return an object with ``encode_batch(clips)`` returning one
    artifact or exception per clip. ``wire`` (``"module:attribute"``) names the
    ``WireFormat`` both sides use. Calls block the calling thread without
    holding the GIL while the child encodes.
    """

    def __init__(self, factory: str, kwargs: dict, *, wire: str, name: str = "reference-encoder") -> None:
        self._wire: WireFormat = _resolve(wire)
        context = multiprocessing.get_context("spawn")
        self._conn, child = context.Pipe()
        self._process = context.Process(
            target=_process_worker_main, args=(factory, kwargs, wire, child), name=name, daemon=True
        )
        self._process.start()
        child.close()
        self._lock = threading.Lock()

    def wait_until_ready(self) -> None:
        try:
            if not self._conn.poll(_STARTUP_TIMEOUT_S):
                raise SharedReferenceEncoderStartupError("reference encoder process did not start in time")
            header, _ = _recv(self._conn)
        except (EOFError, OSError) as exc:
            self._process.join(timeout=5)
            raise SharedReferenceEncoderStartupError(
                f"reference encoder process exited during startup (exit code {self._process.exitcode})"
            ) from exc
        if not header.get("ready"):
            raise SharedReferenceEncoderStartupError(f"reference encoder process failed: {header.get('error')}")

    def __call__(self, clips: list) -> list:
        with self._lock:
            _send_clips(self._conn, clips, self._wire)
            if not self._conn.poll(_REQUEST_TIMEOUT_S):
                raise TimeoutError(f"no reply from the reference encoder process in {_REQUEST_TIMEOUT_S}s")
            header, payloads = _recv(self._conn)
        return _results_from(header, payloads, self._wire)

    def close(self) -> None:
        self._conn.close()
        self._process.join(timeout=5)
        if self._process.is_alive():
            self._process.kill()


@dataclass(frozen=True)
class ReferenceEncoderSpec:
    """A model's reference encoder, as the API server runs it for voice cloning.

    ``factory`` (``"module:callable"``) builds the encoder from ``kwargs``; its
    ``encode_batch(clips)`` returns or yields one artifact or exception per
    clip, where a clip is ``(waveform, sample_rate)``. ``wire``
    (``"module:attribute"``) names the model's ``WireFormat``. ``workers``
    encoder processes run on the host; 0 builds the encoder in the host API
    process itself. Artifacts are cached under ``namespace`` (the model's
    identity) within ``cache_bytes``, sized by ``nbytes``.
    """

    name: str
    factory: str
    kwargs: dict[str, Any]
    wire: str
    namespace: str
    nbytes: Callable[[Any], int]
    workers: int = 1
    max_batch: int = _MAX_BATCH
    cache_bytes: int = 256 * 1024 * 1024


def _in_process_worker(spec: ReferenceEncoderSpec) -> tuple[EncodeFn, Callable[[], object]]:
    encoder: list[Any] = []

    def build() -> None:
        encoder.append(_resolve(spec.factory)(**spec.kwargs))

    def encode(clips: list) -> list:
        return list(encoder[0].encode_batch(clips))

    return encode, build


class ReferenceEncoderPool:
    """A model's reference encoder shared by a server's API processes, behind a content-keyed cache.

    With a server directory, the API process that wins the election for
    ``<server dir>/<spec.name>`` hosts the encoder workers and the others send
    it their clips; otherwise this process hosts them. Construction starts the
    workers; ``start`` waits until they are ready.
    """

    def __init__(self, spec: ReferenceEncoderSpec, server_shared_dir: str | None) -> None:
        self.spec = spec
        shared_dir = os.path.join(server_shared_dir, spec.name) if server_shared_dir else None
        if shared_dir is not None and not socket_path_fits(shared_dir):
            logger.warning(
                "%s is too long a path for a Unix socket; each API process hosts its own encoder", shared_dir
            )
            shared_dir = None
        if shared_dir is not None:
            os.makedirs(shared_dir, exist_ok=True)
        wire: WireFormat = _resolve(spec.wire)
        self.backend: SharedReferenceEncoderHost | SharedReferenceEncoderClient
        self._processes: list[ProcessEncoderWorker] = []
        if shared_dir is not None and not elect_host(shared_dir):
            self.backend = SharedReferenceEncoderClient(shared_dir, wire=wire)
            role = "client of the shared host"
        elif spec.workers <= 0:
            self.backend = SharedReferenceEncoderHost(shared_dir, [_in_process_worker(spec)], wire=wire, name=spec.name)
            role = "host, encoding in this process"
        else:
            self._processes = [
                ProcessEncoderWorker(spec.factory, spec.kwargs, wire=spec.wire, name=f"{spec.name}-ref-{index}")
                for index in range(spec.workers)
            ]
            self.backend = SharedReferenceEncoderHost(
                shared_dir,
                [(process, process.wait_until_ready) for process in self._processes],
                wire=wire,
                worker_streams=False,
                name=spec.name,
            )
            role = f"host of {spec.workers} encoder processes"
        logger.info("%s reference encoder: %s", spec.name, role)
        self._service: ReferenceEncodingService | None = None

    async def start(self) -> None:
        """Wait until the encoder can serve; raises ``SharedReferenceEncoderStartupError`` if it cannot."""
        if self._service is not None:
            return
        await asyncio.to_thread(self.backend.wait_until_ready)
        if self._service is None:
            self._service = ReferenceEncodingService(
                self.backend.encode,
                nbytes=self.spec.nbytes,
                max_bytes=self.spec.cache_bytes,
                max_batch=self.spec.max_batch,
                max_inflight=max(1, self.spec.workers),
                name=self.spec.name,
            )

    async def get(self, waveform: np.ndarray, sample_rate: int) -> Any:
        """The artifact of one clip: cached, or encoded once for all concurrent requests for it."""
        await self.start()
        assert self._service is not None
        wav = np.ascontiguousarray(waveform, dtype=np.float32).reshape(-1)
        return await self._service.get(content_key(self.spec.namespace, wav, sample_rate), (wav, int(sample_rate)))

    def close(self) -> None:
        """Release waiting callers and stop the encoder processes (they also end with this process)."""
        if isinstance(self.backend, SharedReferenceEncoderHost):
            self.backend.close()
        for process in self._processes:
            process.close()
