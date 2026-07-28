from __future__ import annotations

import base64
import binascii
from typing import Any

import numpy as np


class MiniCPMO45PcmAppendReservation:
    __slots__ = (
        "_active",
        "_force_listen",
        "_is_speech",
        "_owner",
        "_raw",
        "_sample_rate_hz",
        "_turn_had_speech",
        "_video_frames",
        "operation_id",
        "payload",
    )

    def __init__(
        self,
        *,
        owner: MiniCPMO45PcmAppendBuffer,
        operation_id: str,
        payload: dict[str, object] | None,
        raw: bytes,
        sample_rate_hz: int,
        force_listen: bool,
        is_speech: bool,
        turn_had_speech: bool = False,
        video_frames: list[str] | None = None,
    ) -> None:
        self._owner = owner
        self.operation_id = operation_id
        self.payload = payload
        self._raw = raw
        self._sample_rate_hz = sample_rate_hz
        self._force_listen = force_listen
        self._is_speech = is_speech
        self._turn_had_speech = turn_had_speech
        self._video_frames = list(video_frames or [])
        self._active = True

    @property
    def active(self) -> bool:
        return self._active

    @property
    def byte_count(self) -> int:
        return len(self._raw)

    def commit(self) -> None:
        self._owner._commit_reservation(self)

    def rollback(self) -> None:
        self._owner._rollback_reservation(self)


def validate_native_ref_audio_config(session_config: dict[str, Any]) -> None:
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    if any(key in session_config for key in ("ref_audio_path", "tts_ref_audio_path")) or any(
        key in extra_body for key in ("ref_audio_path", "tts_ref_audio_path")
    ):
        raise ValueError("native duplex ref_audio_path is not accepted; resolve ref_audio in serving first")


def decode_native_ref_audio_from_config(session_config: dict[str, Any]) -> np.ndarray | None:
    validate_native_ref_audio_config(session_config)
    extra_body = session_config.get("extra_body")
    if not isinstance(extra_body, dict):
        extra_body = {}
    audio_data = extra_body.get("ref_audio_data")
    if audio_data is None:
        return None
    if not isinstance(audio_data, str):
        raise TypeError("native duplex ref_audio_data must be base64 pcm_f32le")
    fmt = extra_body.get("ref_audio_format") or "pcm_f32le"
    if fmt != "pcm_f32le":
        raise ValueError(f"unsupported native duplex ref_audio_format: {fmt!r}")
    try:
        raw = base64.b64decode(audio_data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid native duplex ref_audio_data") from exc
    if len(raw) % 4:
        raise ValueError("invalid native duplex ref_audio_data length")
    return np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)


class MiniCPMO45PcmAppendBuffer:
    """Accumulates short native-duplex PCM chunks into model-sized appends."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._sample_rate_hz: int | None = None
        self._force_listen = False
        self._is_speech = False
        self._turn_had_speech = False
        self._reservation_seq = 0
        self._reservations: list[MiniCPMO45PcmAppendReservation] = []
        # Omni duplex: queued camera frames (base64 JPEG), consumed FIFO at
        # one frame per emitted model unit alongside the unit's audio.
        self._frame_queue: list[str] = []

    def clear(self) -> None:
        for reservation in self._reservations:
            reservation._active = False
        self._reservations.clear()
        self._buffer.clear()
        self._frame_queue.clear()
        self._sample_rate_hz = None
        self._force_listen = False
        self._is_speech = False
        self._turn_had_speech = False

    def clear_force_listen(self) -> None:
        self._force_listen = False

    def has_pending(self) -> bool:
        return bool(self._buffer)

    def has_reserved(self) -> bool:
        return bool(self._reservations)

    @property
    def pending_byte_count(self) -> int:
        return len(self._buffer)

    def _reserve_passthrough(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
    ) -> MiniCPMO45PcmAppendReservation:
        sample_rate_hz = payload.get("sample_rate_hz")
        if isinstance(payload, dict) and "video_frames" in payload:
            # Frames align to whole audio units; a passthrough payload has no
            # unit framing for Stage0 to interleave against.
            payload = {key: value for key, value in payload.items() if key != "video_frames"}
        reservation = MiniCPMO45PcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=payload,
            raw=b"",
            sample_rate_hz=sample_rate_hz if isinstance(sample_rate_hz, int) else 0,
            force_listen=bool(payload.get("force_listen", False)),
            is_speech=bool(payload.get("is_speech", False)),
            turn_had_speech=bool(payload.get("is_speech", False)),
        )
        self._reservations.append(reservation)
        return reservation

    def prepare_append(
        self,
        payload: dict[str, object],
        *,
        operation_id: str,
        chunk_period_ms: int,
        flush: bool = False,
        allow_emit: bool = True,
    ) -> MiniCPMO45PcmAppendReservation | None:
        fmt = payload.get("format")
        sample_rate_hz = payload.get("sample_rate_hz")
        audio = payload.get("audio")
        if fmt != "pcm_f32le" or not isinstance(sample_rate_hz, int) or not isinstance(audio, str):
            return self._reserve_passthrough(payload, operation_id=operation_id)
        try:
            raw = base64.b64decode(audio, validate=True)
        except (binascii.Error, ValueError):
            return self._reserve_passthrough(payload, operation_id=operation_id)
        if len(raw) % 4 != 0:
            return self._reserve_passthrough(payload, operation_id=operation_id)

        if self._sample_rate_hz is not None and self._sample_rate_hz != sample_rate_hz:
            raise ValueError("MiniCPM-o native duplex audio append sample_rate_hz changed within a session")
        self._sample_rate_hz = sample_rate_hz
        self._buffer.extend(raw)
        frames_in = payload.get("video_frames")
        if isinstance(frames_in, list):
            self._frame_queue.extend(frame for frame in frames_in if isinstance(frame, str) and frame)
        self._turn_had_speech = self._turn_had_speech or bool(payload.get("is_speech", False))
        self._force_listen = self._force_listen or bool(payload.get("force_listen", False))
        self._is_speech = self._is_speech or bool(payload.get("is_speech", False))
        if not allow_emit:
            return None

        min_samples = max(1, int(sample_rate_hz * max(1, int(chunk_period_ms)) / 1000))
        buffered_samples = len(self._buffer) // 4
        if not flush and buffered_samples < min_samples:
            return None

        # Emit whole model chunks only: the engine reserves scheduler slots
        # from the payload size and the worker consumes whole chunks per
        # append, so partial chunks would turn into pad embeddings inside the
        # model KV. On flush, the tail is zero-padded (silence) up to the
        # chunk boundary instead.
        if flush:
            emit_samples = buffered_samples
            remainder = emit_samples % min_samples
            pad_samples = (min_samples - remainder) if remainder else 0
        else:
            emit_samples = min_samples
            pad_samples = 0
        emit_bytes = emit_samples * 4
        reserved_raw = bytes(self._buffer[:emit_bytes])
        emit_raw = reserved_raw + b"\x00" * (pad_samples * 4)
        del self._buffer[:emit_bytes]

        out = dict(payload)
        out.pop("force_speak", None)
        out.pop("video_frames", None)
        out["audio"] = base64.b64encode(emit_raw).decode("ascii")
        out["sample_rate_hz"] = sample_rate_hz
        # Omni duplex: attach at most one queued camera frame per emitted
        # model unit (official cadence: one frame per 1 s chunk). The engine
        # budgets 66 scheduler slots per attached frame from this payload.
        # Official omni cadence is one frame per ~1 s chunk, and the first
        # append consumes extra samples (1035 ms first window), so per-unit
        # attachment could outrun the units Stage0 actually builds. Attach at
        # most ONE frame per emitted payload; the rest stay queued.
        attached_frames: list[str] = []
        if emit_samples + pad_samples >= min_samples and self._frame_queue:
            attached_frames = [self._frame_queue.pop(0)]
            out["video_frames"] = attached_frames
        out["force_listen"] = self._force_listen
        out["is_speech"] = self._is_speech
        if not self._buffer:
            self._force_listen = False
            self._is_speech = False
        reservation = MiniCPMO45PcmAppendReservation(
            owner=self,
            operation_id=operation_id,
            payload=out,
            raw=reserved_raw,
            sample_rate_hz=sample_rate_hz,
            force_listen=bool(out.get("force_listen", False)),
            is_speech=bool(out.get("is_speech", False)),
            turn_had_speech=self._turn_had_speech,
            video_frames=attached_frames,
        )
        self._reservations.append(reservation)
        return reservation

    def prepare_commit(
        self,
        *,
        operation_id: str,
        chunk_period_ms: int,
    ) -> MiniCPMO45PcmAppendReservation:
        """Reserve the terminal payload without invalidating prior appends."""
        had_speech = self._turn_had_speech
        reservation: MiniCPMO45PcmAppendReservation | None = None
        if had_speech and self._buffer:
            payload: dict[str, object] = {
                "type": "audio",
                "audio": "",
                "format": "pcm_f32le",
                "sample_rate_hz": self._sample_rate_hz or 16000,
                "force_listen": self._force_listen,
                "is_speech": self._is_speech,
            }
            reservation = self.prepare_append(
                payload,
                operation_id=operation_id,
                chunk_period_ms=chunk_period_ms,
                flush=True,
            )
            assert reservation is not None
            assert reservation.payload is not None
            reservation.payload["final"] = True
        if reservation is None:
            reservation = MiniCPMO45PcmAppendReservation(
                owner=self,
                operation_id=operation_id,
                payload=None,
                raw=b"",
                sample_rate_hz=self._sample_rate_hz or 0,
                force_listen=self._force_listen,
                is_speech=self._is_speech,
                turn_had_speech=had_speech,
            )
            self._reservations.append(reservation)

        # Open the next physical input generation without touching already
        # reserved appends. A rollback restores this generation's metadata.
        self._sample_rate_hz = None
        self._force_listen = False
        self._is_speech = False
        self._turn_had_speech = False
        return reservation

    def append(
        self,
        payload: dict[str, object],
        *,
        chunk_period_ms: int,
        flush: bool = False,
        allow_emit: bool = True,
    ) -> dict[str, object] | None:
        self._reservation_seq += 1
        reservation = self.prepare_append(
            payload,
            operation_id=f"immediate-{self._reservation_seq}",
            chunk_period_ms=chunk_period_ms,
            flush=flush,
            allow_emit=allow_emit,
        )
        if reservation is None:
            return None
        reservation.commit()
        assert reservation.payload is not None
        return reservation.payload

    def _commit_reservation(self, reservation: MiniCPMO45PcmAppendReservation) -> None:
        if not reservation._active:
            return
        if not self._reservations or self._reservations[0] is not reservation:
            raise RuntimeError("PCM append reservations must commit in wire order")
        self._reservations.pop(0)
        reservation._active = False

    def _rollback_reservation(self, reservation: MiniCPMO45PcmAppendReservation) -> None:
        if not reservation._active:
            return
        try:
            index = self._reservations.index(reservation)
        except ValueError:
            reservation._active = False
            return
        rolled_back = self._reservations[index:]
        restored = b"".join(item._raw for item in rolled_back)
        self._buffer[:0] = restored
        restored_frames = [frame for item in rolled_back for frame in item._video_frames]
        if restored_frames:
            self._frame_queue[:0] = restored_frames
        self._sample_rate_hz = self._sample_rate_hz or reservation._sample_rate_hz
        self._force_listen = self._force_listen or any(item._force_listen for item in rolled_back)
        self._is_speech = self._is_speech or any(item._is_speech for item in rolled_back)
        self._turn_had_speech = self._turn_had_speech or any(item._turn_had_speech for item in rolled_back)
        for item in rolled_back:
            item._active = False
        del self._reservations[index:]

    def flush(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        if not self._buffer:
            return None
        payload: dict[str, object] = {
            "type": "audio",
            "audio": "",
            "format": "pcm_f32le",
            "sample_rate_hz": self._sample_rate_hz or 16000,
            "force_listen": self._force_listen,
            "is_speech": self._is_speech,
        }
        return self.append(payload, chunk_period_ms=chunk_period_ms, flush=True)

    def commit(self, *, chunk_period_ms: int) -> dict[str, object] | None:
        """Commit the real residual PCM and reset per-client-input state.

        Complete model units are emitted by :meth:`append` as soon as they are
        available.  A commit at that exact boundary therefore has no payload;
        synthesizing another unit would add a model decision that does not
        exist in the official continuous streaming loop.
        """
        self._reservation_seq += 1
        reservation = self.prepare_commit(
            operation_id=f"immediate-commit-{self._reservation_seq}",
            chunk_period_ms=chunk_period_ms,
        )
        reservation.commit()
        return reservation.payload


__all__ = [
    "MiniCPMO45PcmAppendReservation",
    "MiniCPMO45PcmAppendBuffer",
    "decode_native_ref_audio_from_config",
    "validate_native_ref_audio_config",
]
