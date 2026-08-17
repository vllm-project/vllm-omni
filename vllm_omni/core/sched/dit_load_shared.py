"""Cross-PROCESS DiT-stage load state (shared memory).

The Orchestrator (main process) collects each DiT replica's queue depth, but
the AR scheduler (``DTPSScheduler``) that consumes it runs in a separate AR
subprocess. A plain :class:`DitLoadState` cannot cross a spawn boundary — it
pickles into an independent copy — so this class holds the *aggregate* DiT-load
snapshot in a ``multiprocessing.shared_memory`` buffer. Only the buffer **name**
rides in the pickled ``omni_dtps_config``; the AR subprocess reattaches by name.

Buffer layout (variable-length, little-endian)::

    fixed header (32 bytes, struct '<8i'):
        [seq, min_waiting, max_waiting, total_waiting,
         total_running, num_replicas, fresh, blob_len]
    ID blob (blob_len bytes):
        <I n_w>  then n_w x (<H len> + len utf-8 id bytes)   # waiting ids
        <I n_r>  then n_r x (<H len> + len utf-8 id bytes)   # running ids

``seq`` is a seqlock: the writer increments to ODD before writing the
header+blob and to EVEN after, so a reader that re-reads ``seq`` can detect a
torn read (odd, or changed) and retry. The ID blob lets the AR scheduler
de-duplicate its blind in-flight set; blob size scales with the actual number
of queued ids, not buffer capacity.
"""

from __future__ import annotations

import struct
from typing import NamedTuple

from vllm.logger import init_logger

from vllm_omni.core.sched.dit_load_state import DitLoadSnapshot

logger = init_logger(__name__)

_HEADER = struct.Struct("<8i")
_HEADER_SIZE = _HEADER.size
_COUNT = struct.Struct("<I")
_LEN = struct.Struct("<H")

_BUF_SIZE = 16384
_MAX_BLOB = _BUF_SIZE - _HEADER_SIZE
_OFF_SEQ = 0

_SEQLOCK_RETRIES = 8

_MAX_FIELD = 10_000_000
_MAX_REPLICAS = 1_000
_MAX_ID_BYTES = 65535

_DIT_LOAD_SHM_NAME_KEY = "_dit_load_shm_name"

_IDX_SEQ = 0
_IDX_MIN_W = 1
_IDX_MAX_W = 2
_IDX_TOT_W = 3
_IDX_TOT_R = 4
_IDX_N_REPS = 5
_IDX_FRESH = 6
_IDX_BLOB_LEN = 7


class DitLoadSharedRead(NamedTuple):
    snapshot: DitLoadSnapshot
    seq: int


class DitLoadSharedState:
    """Cross-process aggregate DiT-load snapshot backed by a shared-memory seg.

    :meth:`create` allocates a NEW segment (main process / writer);
    :meth:`attach` reattaches to an EXISTING segment by name (AR subprocess /
    reader).
    """

    def __init__(self, shm: object, *, owns: bool) -> None:
        self._shm = shm
        self._owns = owns
        self._name = shm.name  # type: ignore[attr-defined]
        self._write_seq = 0

    @classmethod
    def create(cls) -> DitLoadSharedState:
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(size=_BUF_SIZE, create=True)
        try:
            shm.buf[:] = bytes(_BUF_SIZE)
        except (AttributeError, TypeError, ValueError):
            shm._mmap[:] = bytes(_BUF_SIZE)  # type: ignore[attr-defined]
        logger.info(
            "[OmniDTPS] DiT-load shared memory created: name=%s size=%d",
            shm.name,
            _BUF_SIZE,
        )
        return cls(shm, owns=True)

    @classmethod
    def attach(cls, name: str) -> DitLoadSharedState:
        from multiprocessing import shared_memory

        shm = shared_memory.SharedMemory(name=name, create=False)
        logger.info(
            "[OmniDTPS] DiT-load shared memory attached (reader): name=%s",
            name,
        )
        return cls(shm, owns=False)

    @property
    def name(self) -> str:
        return self._name

    @property
    def owns_segment(self) -> bool:
        return self._owns

    def write(self, snapshot: DitLoadSnapshot) -> None:
        self._write_seq += 1
        seq_odd = self._write_seq
        self._pack(seq_odd, snapshot)
        self._write_seq += 1
        seq_even = self._write_seq
        self._pack(seq_even, snapshot)

    def _pack(self, seq: int, snapshot: DitLoadSnapshot) -> None:
        fresh = 1 if snapshot.get("fresh") else 0
        waiting_ids = sorted(snapshot.get("waiting_ids") or frozenset())
        running_ids = sorted(snapshot.get("running_ids") or frozenset())
        blob = self._encode_blob(waiting_ids, running_ids)
        header = _HEADER.pack(
            int(seq) & 0x7FFFFFFF,
            int(snapshot.get("min_waiting", 0) or 0),
            int(snapshot.get("max_waiting", 0) or 0),
            int(snapshot.get("total_waiting", 0) or 0),
            int(snapshot.get("total_running", 0) or 0),
            int(snapshot.get("num_replicas", 0) or 0),
            fresh,
            len(blob),
        )
        packed = header + blob
        self._shm.buf[: len(packed)] = packed  # type: ignore[attr-defined]

    @staticmethod
    def _encode_blob(waiting_ids: list[str], running_ids: list[str]) -> bytes:
        DitLoadSharedState._truncate_inplace(waiting_ids, running_ids)
        parts: list[bytes] = [_COUNT.pack(len(waiting_ids))]
        for rid in waiting_ids:
            DitLoadSharedState._append_id(parts, rid)
        parts.append(_COUNT.pack(len(running_ids)))
        for rid in running_ids:
            DitLoadSharedState._append_id(parts, rid)
        return b"".join(parts)

    @staticmethod
    def _append_id(parts: list[bytes], rid: str) -> None:
        b = rid.encode("utf-8")
        if len(b) > _MAX_ID_BYTES:
            b = b[:_MAX_ID_BYTES]
        parts.append(_LEN.pack(len(b)))
        parts.append(b)

    @staticmethod
    def _truncate_inplace(waiting_ids: list[str], running_ids: list[str]) -> None:
        # Drop trailing ids (waiting first, then running) until the blob fits
        # _MAX_BLOB; counts are written from the truncated lists, so the blob
        # stays self-consistent.
        while True:
            size = _COUNT.size * 2
            for rid in waiting_ids:
                size += _LEN.size + min(len(rid.encode("utf-8")), _MAX_ID_BYTES)
            for rid in running_ids:
                size += _LEN.size + min(len(rid.encode("utf-8")), _MAX_ID_BYTES)
            if size <= _MAX_BLOB:
                return
            if waiting_ids:
                dropped = waiting_ids.pop()
                logger.warning(
                    "[OmniDTPS] DiT-load ID blob full (%d > %d); dropping waiting id %r",
                    size,
                    _MAX_BLOB,
                    dropped,
                )
            elif running_ids:
                dropped = running_ids.pop()
                logger.warning(
                    "[OmniDTPS] DiT-load ID blob full (%d > %d); dropping running id %r",
                    size,
                    _MAX_BLOB,
                    dropped,
                )
            else:
                return

    def snapshot(self) -> DitLoadSharedRead:
        for _ in range(_SEQLOCK_RETRIES):
            seq1 = self._read_seq()
            if seq1 % 2 == 1:
                continue
            read = self._read_payload(seq1)
            if read is not None:
                return read
        return self._empty_read()

    def _read_payload(self, seq1: int) -> DitLoadSharedRead | None:
        buf = self._shm.buf  # type: ignore[attr-defined]
        header = _HEADER.unpack(bytes(buf[:_HEADER_SIZE]))
        min_w = header[_IDX_MIN_W]
        max_w = header[_IDX_MAX_W]
        tot_w = header[_IDX_TOT_W]
        tot_r = header[_IDX_TOT_R]
        n_reps = header[_IDX_N_REPS]
        fresh = header[_IDX_FRESH]
        blob_len = header[_IDX_BLOB_LEN]
        if not (
            0 <= min_w <= _MAX_FIELD
            and 0 <= max_w <= _MAX_FIELD
            and 0 <= tot_w <= _MAX_FIELD
            and 0 <= tot_r <= _MAX_FIELD
            and 0 <= n_reps <= _MAX_REPLICAS
            and fresh in (0, 1)
            and 0 <= blob_len <= _MAX_BLOB
        ):
            return None
        blob = bytes(buf[_HEADER_SIZE : _HEADER_SIZE + blob_len])
        seq2 = self._read_seq()
        if not (seq1 == seq2 and seq2 % 2 == 0):
            return None
        waiting_ids, running_ids = self._decode_blob(blob)
        if waiting_ids is None or running_ids is None:
            return None
        snap: DitLoadSnapshot = {
            "min_waiting": min_w,
            "max_waiting": max_w,
            "total_waiting": tot_w,
            "total_running": tot_r,
            "num_replicas": n_reps,
            "fresh": bool(fresh),
            "waiting_ids": waiting_ids,
            "running_ids": running_ids,
        }
        return DitLoadSharedRead(snap, seq2)

    @staticmethod
    def _decode_blob(blob: bytes) -> tuple[frozenset[str] | None, frozenset[str] | None]:
        try:
            pos = 0
            n_w = _COUNT.unpack_from(blob, pos)[0]
            pos += _COUNT.size
            waiting_ids: list[str] = []
            for _ in range(n_w):
                n = _LEN.unpack_from(blob, pos)[0]
                pos += _LEN.size
                waiting_ids.append(blob[pos : pos + n].decode("utf-8"))
                pos += n
            n_r = _COUNT.unpack_from(blob, pos)[0]
            pos += _COUNT.size
            running_ids: list[str] = []
            for _ in range(n_r):
                n = _LEN.unpack_from(blob, pos)[0]
                pos += _LEN.size
                running_ids.append(blob[pos : pos + n].decode("utf-8"))
                pos += n
            if pos != len(blob):
                return None, None
            return frozenset(waiting_ids), frozenset(running_ids)
        except (struct.error, UnicodeDecodeError, IndexError):
            return None, None

    def _read_seq(self) -> int:
        buf = self._shm.buf  # type: ignore[attr-defined]
        return struct.unpack_from("<i", buf, _OFF_SEQ)[0]

    @staticmethod
    def _empty_read() -> DitLoadSharedRead:
        return DitLoadSharedRead(
            {
                "min_waiting": 0,
                "max_waiting": 0,
                "total_waiting": 0,
                "total_running": 0,
                "num_replicas": 0,
                "fresh": False,
                "waiting_ids": frozenset(),
                "running_ids": frozenset(),
            },
            -1,
        )

    def close(self) -> None:
        shm = self._shm
        if shm is None:
            return
        try:
            shm.close()  # type: ignore[attr-defined]
        except Exception:
            logger.debug(
                "[OmniDTPS] shared memory close failed for name=%s",
                self._name,
                exc_info=True,
            )
        self._shm = None

    def unlink(self) -> None:
        if not self._owns:
            return
        shm = self._shm
        try:
            from multiprocessing import shared_memory

            if shm is not None:
                shm.unlink()  # type: ignore[attr-defined]
            else:
                tmp = shared_memory.SharedMemory(name=self._name, create=False)
                try:
                    tmp.unlink()
                finally:
                    tmp.close()
        except Exception:
            logger.debug(
                "[OmniDTPS] shared memory unlink failed for name=%s (segment may already be gone)",
                self._name,
                exc_info=True,
            )
