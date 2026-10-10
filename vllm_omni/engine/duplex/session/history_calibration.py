# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded output-audio retention and asynchronous heard-text calibration.

The runner owns this object on its event loop. A model callback receives an
immutable snapshot and returns an original-text offset; only the engine session
may apply that offset to the authoritative conversation history.
"""

from __future__ import annotations

import asyncio
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm_omni.engine.duplex.session.engine_session import DuplexEngineSession

logger = init_logger(__name__)


@dataclass(frozen=True)
class HeardTextSnapshot:
    """Original reply and confirmed playback cutoff in milliseconds.

    Audio is mono float32 PCM from the reply's start. A policy that does not
    require audio receives empty bytes and a zero sample rate instead.
    """

    response_id: str
    text: str
    pcm_f32le: bytes
    sample_rate_hz: int
    played_ms: int


@dataclass(frozen=True)
class HistoryCalibrationPolicy:
    """Model callback and its audio-retention requirement.

    The callback returns an exclusive original-text character offset (zero
    means an empty prefix), or None to retain the existing history policy.
    It must not mutate session history; the framework validates its result.
    """

    calibrate: Callable[[HeardTextSnapshot], Awaitable[int | None]]
    requires_audio: bool = True


@dataclass
class _Audio:
    chunks: list[bytes] = field(default_factory=list)
    size: int = 0
    rate: int = 0
    valid: bool = True


class HistoryCalibration:
    """Own per-session output recordings, calibration tasks and writeback fences."""

    def __init__(
        self,
        session: DuplexEngineSession,
        calibrate: Callable[[HeardTextSnapshot], Awaitable[int | None]],
        *,
        max_bytes: int = 8 * 1024 * 1024,
        timeout_s: float = 1.0,
        requires_audio: bool = True,
    ) -> None:
        if max_bytes <= 0 or timeout_s <= 0:
            raise ValueError("Calibration limits must be positive")
        self.session = session
        self.calibrate = calibrate
        self.max_bytes = max_bytes
        self.timeout_s = timeout_s
        self.requires_audio = requires_audio
        self.audio: OrderedDict[str, _Audio] = OrderedDict()
        self.tasks: dict[str, asyncio.Task[None]] = {}
        self.versions: dict[str, tuple[str, int, int]] = {}
        self.closed = False
        self.evicted: set[str] = set()

    @property
    def retained_bytes(self) -> int:
        """Return retained PCM bytes, excluding temporary callback copies."""
        return sum(item.size for item in self.audio.values())

    def _is_generating(self, response_id: str) -> bool:
        return response_id == self.session.active_response_id or self.session.response_has_draining_request(response_id)

    def record(self, response_id: str, output: dict) -> None:
        """Record projected output without restarting a discarded live reply."""
        if self.closed:
            return
        # Only live outputs can restart an evicted recording with a suffix.
        # ModelChannel rejects late output after the response stops generating.
        self.evicted = {rid for rid in self.evicted if self._is_generating(rid)}
        if response_id in self.evicted:
            return
        item = self.audio.setdefault(response_id, _Audio())
        was_valid = item.valid
        reason = "audio_limit"
        if output.get("history_audio_supported") is False:
            item.valid = False
            reason = "unsupported_audio"
        pcm = output.get("history_audio_pcm")
        if self.requires_audio and isinstance(pcm, bytes) and pcm:
            rate = output.get("sample_rate_hz")
            if not isinstance(rate, int) or rate <= 0 or len(pcm) % 4 or (item.rate and rate != item.rate):
                item.valid = False
                reason = "invalid_audio"
            elif item.valid:
                item.rate = rate
                item.chunks.append(pcm)
                item.size += len(pcm)
        if not item.valid or item.size > self.max_bytes:
            if was_valid:
                logger.info("Duplex history cache response=%s reason=%s bytes=%d", response_id, reason, item.size)
            item.valid = False
            item.chunks.clear()
            item.size = 0
        # Never retain a suffix and pretend it starts at zero. Evict whole replies.
        while self.retained_bytes > self.max_bytes or len(self.audio) > 4:
            evicted_id = next(iter(self.audio))
            logger.info("Duplex history cache response=%s reason=evicted", evicted_id)
            self.discard(evicted_id)

    def discard(self, response_id: str) -> None:
        """Release a reply's recording and revoke any pending calibration."""
        if self._is_generating(response_id):
            self.evicted.add(response_id)
        self.audio.pop(response_id, None)
        self.versions.pop(response_id, None)
        task = self.tasks.pop(response_id, None)
        if task is not None:
            task.cancel()

    def request_ready(self) -> None:
        """Start only ended replies with a partial, client-confirmed playback cursor."""
        if self.closed or self.session.config.playback_commit_policy != "ack_only":
            return
        for response_id, audio in list(self.audio.items()):
            if self._is_generating(response_id):
                continue
            text = self.session.assistant_transcript(response_id)
            if not text:
                self.discard(response_id)
                continue
            playback = self.session.playback_for_response(response_id)
            cutoff = self.session.history_audio_cutoff(response_id)
            if playback.audio_complete and cutoff >= max(playback.generated_ms, playback.sent_ms):
                self.discard(response_id)
                continue
            if not audio.valid or (self.requires_audio and not audio.size) or cutoff <= 0:
                continue
            version = (text, cutoff, audio.size)
            if self.versions.get(response_id) == version:
                continue
            task = self.tasks.get(response_id)
            if task is not None:
                task.cancel()
            self.versions[response_id] = version
            snapshot = HeardTextSnapshot(
                response_id,
                text,
                b"".join(audio.chunks),
                audio.rate,
                cutoff,
            )
            self.tasks[response_id] = asyncio.create_task(self._run(snapshot, version))

    def release_confirmed(self) -> None:
        """Free fully acknowledged replies without running ASR on frequent ACKs."""
        for response_id in list(self.audio):
            if self._is_generating(response_id):
                continue
            if not self.session.assistant_transcript(response_id):
                self.discard(response_id)

    async def _run(self, snapshot: HeardTextSnapshot, version: tuple[str, int, int]) -> None:
        started = time.monotonic()
        status = "refused"
        retained_chars = 0
        try:
            char_end = await asyncio.wait_for(self.calibrate(snapshot), timeout=self.timeout_s)
            if (
                not self.closed
                and self.tasks.get(snapshot.response_id) is asyncio.current_task()
                and self.versions.get(snapshot.response_id) == version
                and isinstance(char_end, int)
                and not isinstance(char_end, bool)
            ):
                applied = self.session.apply_heard_text(
                    snapshot.response_id,
                    text=snapshot.text,
                    audio_end_ms=snapshot.played_ms,
                    char_end=char_end,
                )
                status = "applied" if applied else "stale_or_invalid"
                retained_chars = char_end if applied else 0
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        except Exception as exc:
            # No text/audio in logs. A failed or timed-out task keeps the old policy.
            logger.warning("Duplex history calibration failed: %s", type(exc).__name__)
            status = type(exc).__name__
        finally:
            logger.info(
                "Duplex history calibration response=%s status=%s duration_ms=%.1f played_ms=%d retained_chars=%d",
                snapshot.response_id,
                status,
                (time.monotonic() - started) * 1000,
                snapshot.played_ms,
                retained_chars,
            )
            if self.tasks.get(snapshot.response_id) is asyncio.current_task():
                self.tasks.pop(snapshot.response_id, None)

    async def before_prompt(self) -> None:
        """Wait within one deadline for eligible work, including replacements."""
        started = time.monotonic()
        deadline = started + self.timeout_s
        self.request_ready()
        pending = len(self.tasks)
        status = "ready"
        try:
            while self.tasks:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    # Keep versions to avoid retrying the same failed evidence.
                    # Removing ownership rejects late cancellation results.
                    status = "deadline"
                    tasks = tuple(self.tasks.values())
                    self.tasks.clear()
                    for task in tasks:
                        task.cancel()
                    return
                # A stricter truncate can replace a task while this wait yields.
                # Recheck tasks without extending this prompt's budget.
                # asyncio.wait leaves calibration alive if the prompt is cancelled.
                await asyncio.wait(tuple(self.tasks.values()), timeout=remaining)
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            logger.info(
                "Duplex history prompt wait session=%s status=%s wait_ms=%.1f "
                "pending_at_start=%d retained_bytes=%d cached_replies=%d",
                self.session.session_id,
                status,
                (time.monotonic() - started) * 1000,
                pending,
                self.retained_bytes,
                len(self.audio),
            )

    def close(self) -> None:
        """Release recordings and prevent further calibration or writeback."""
        self.closed = True
        for response_id in list(self.audio):
            self.discard(response_id)
        self.evicted.clear()
